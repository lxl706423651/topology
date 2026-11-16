"""有限窗口研究数据：复用旧有限因子，另存原始观测来源及240分钟简单均值ATR。

仅以当前/历史合法分钟线确定指标来源；不读取测试预测、交易或收益。旧274因子不重建。
新增缓存的代码/数据/列序指纹与旧Wilder ATR缓存隔离，供重训及同契约回测使用。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_backtest.annual_panel import load_annual_panel, load_symbol_bars, write_json
from quant_backtest.annual_portfolio import MarketStore
from quant_backtest.annual_training import select_signal_tradable_panel
from quant_backtest.learned_exit_models import calculate_base_matrices, build_exit_samples
from quant_backtest.multiscale_training import build_context_panel
from quant_backtest.inventory_t_data import load_signal_eligibility, apply_signal_eligibility


RAW_LOOKBACK = 240
FINITE_SINGLE = {"oc_ret", "hl_range", "upper_shadow", "lower_shadow", "close_pos_in_bar", "vwap_gap",
                 "minute_sin", "minute_cos", "stoch_k_14", "stoch_d_14"}
FINITE_FAMILIES = (
    "momentum", "reversal", "ret_mean", "ret_std", "ret_skew", "weighted_momentum",
    "downside_vol", "parkinson_vol", "garman_klass_vol", "rogers_satchell_vol",
    "drawdown_from_high", "rolling_sharpe", "amihud_illiq", "sma_gap", "price_rank",
    "dist_high", "dist_low", "breakout_pos", "volume_ratio", "amount_ratio",
    "volume_change", "amount_change", "volume_z", "amount_z", "money_flow_imbalance",
    "ret_volume_corr", "ret_amount_corr", "range_mean", "range_std", "vwap_gap_mean", "vol_change",
)


def create_finite_factor_names() -> set[str]:
    """返回当前原因子中有明确有限量价窗口的白名单；最大为vol_change_120读取241个原始观测。

    不保留EMA/RSI/Wilder ATR/ADX/MACD、累计OBV及使用全局序号的趋势估计。
    后者排除是保守数值稳定性处理，不声称趋势本身具有无限量价记忆。
    """
    names = set(FINITE_SINGLE)
    names.update(f"{family}_{n}" for family in FINITE_FAMILIES for n in [5, 10, 20, 30, 60, 120])
    names.update(f"ma_cross_{a}_{b}" for a, b in [(3, 10), (5, 20), (10, 60), (20, 120)])
    names.update(f"boll_{kind}_{n}" for kind in ["pctb", "width", "upper_gap", "lower_gap"] for n in [10, 20, 60])
    names.update(f"{kind}_{n}" for kind in ["roc", "cci", "willr", "mfi"] for n in [6, 12, 14, 24, 48])
    return names


def build_symbol_support(raw: pd.DataFrame, clock: pd.DatetimeIndex) -> pd.DataFrame:
    """对单股已清理raw生成clock行的有限ATR、有限因子起点及实际前收来源，不填未来行情。

    raw需唯一递增datetime和合法high/low/close；240个真实波幅另需一个前收，最早i-240。
    不足241根时起点记首条历史而ATR缺失；当前无观测时原始起点为NaT，不能猜成当天。
    """
    bars = raw.set_index("datetime")
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("原始分钟观测必须唯一且递增")
    if not clock.is_unique or not clock.is_monotonic_increasing:
        raise ValueError("共同采样时钟必须唯一且递增")
    previous = bars.close.shift(1)
    tr = pd.concat([bars.high - bars.low, (bars.high - previous).abs(), (bars.low - previous).abs()], axis=1).max(axis=1)
    tr.loc[previous.isna()] = np.nan
    atr = tr.rolling(RAW_LOOKBACK, min_periods=RAW_LOOKBACK).mean()
    positions = bars.index.get_indexer(clock)
    valid = positions >= 0
    starts = np.full(len(clock), np.datetime64("NaT"), dtype="datetime64[ns]")
    starts[valid] = bars.index.to_numpy()[np.maximum(0, positions[valid] - RAW_LOOKBACK)]
    dates = pd.DatetimeIndex(sorted(clock.normalize().unique()))
    observed = pd.Series(bars.index.to_numpy(), index=bars.index.normalize()).groupby(level=0).last()
    previous_source = observed.reindex(dates).ffill().shift(1).reindex(clock.normalize()).to_numpy()
    frame = pd.DataFrame({"datetime": clock, "finite_atr": atr.reindex(clock).to_numpy(),
                          "raw_source_start": starts, "previous_close_source": previous_source})
    return frame


def build_support_file(task: tuple) -> str:
    """task=(symbol,cfg,clock,cache_dir)只读合法原分钟线，原子保存单股支持pkl，可断点复用，不算274因子。"""
    symbol, cfg, clock, directory = task
    path = Path(directory) / f"{symbol}.pkl"
    if path.exists():
        return str(path)
    raw, quality = load_symbol_bars(Path(cfg["data_root"]), symbol, cfg["start_date"], cfg["end_date"])
    frame = build_symbol_support(raw, clock)
    frame["symbol"] = symbol
    temporary = path.with_suffix(".tmp.pkl")
    pd.to_pickle({"support": frame, "raw_quality": quality}, temporary)
    temporary.replace(path)
    return str(path)


def propagate_feature_sources(panel: pd.DataFrame, support: pd.DataFrame, clock: pd.DatetimeIndex) -> pd.DataFrame:
    """为完整旧截面panel附保守来源起点；support给逐股观测/前收起点，clock给60交易日固定格点回看。

    须在交易资格过滤前调用，原缓存截面秩使用的所有股票都传播来源；未知贡献者让整时点失败关闭。
    同时覆盖后续环境变量的前收贡献、最长60日动量和上一自然周选股，不把来源列作为模型特征。
    """
    keys = ["datetime", "symbol"]
    quotes = support[keys + ["raw_source_start", "previous_close_source"]]
    out = panel.merge(quotes, on=keys, how="left", validate="one_to_one")
    indices = clock.get_indexer(out.datetime)
    if (indices < 0).any():
        raise ValueError("面板出现共同日历外时刻")
    counts = pd.Series(clock.normalize()).value_counts()
    if not (counts == 11).all():
        raise ValueError("有限来源暂仅支持每日11格的24分钟公共时钟")
    calendar_source = clock.to_numpy()[np.maximum(0, indices - 60 * 11)]
    week_start = out.datetime.dt.normalize() - pd.to_timedelta(out.datetime.dt.dayofweek + 7, unit="D")
    own = np.minimum(out.raw_source_start.to_numpy(), calendar_source)
    own = np.minimum(own, week_start.to_numpy())
    own = np.minimum(own, out.previous_close_source.to_numpy())
    starts = pd.Series(own, index=out.index)
    unknown = starts.isna().groupby(out.datetime).transform("any")
    out["feature_source_start"] = starts.groupby(out.datetime).transform("min").mask(unknown)
    out = out.drop(columns=["raw_source_start", "previous_close_source"])
    if (out.feature_source_start > out.datetime).any():
        raise ValueError("特征起点不能晚于信号")
    return out


def select_bounded_rows(frame: pd.DataFrame, times: pd.DatetimeIndex, heldout_year: int, embargo_days: int) -> pd.DataFrame:
    """按frame.feature_source_start严格剔除量价来源与heldout_year相交行，另保留times的embargo_days隔离。

    来源区间包括原截面所有贡献者；只过滤本折内部拟合/校准/验证，不修改外层行情或来源文件。
    """
    from scripts.run_multiscale_year_validation import select_outer_feature_rows
    outer = select_outer_feature_rows(frame, times, heldout_year, embargo_days)
    if "feature_source_start" not in outer:
        raise ValueError("有限窗口重训必须提供实际来源起点")
    start, end = pd.Timestamp(f"{heldout_year}-01-01"), pd.Timestamp(f"{heldout_year + 1}-01-01")
    overlap = (outer.feature_source_start < end) & (outer.datetime >= start)
    known = outer.feature_source_start.notna()
    return outer.loc[known & ~overlap].copy()


def select_virtual_entries(panel: pd.DataFrame, stride: int) -> pd.DataFrame:
    """按panel的datetime/symbol稳定键及正整数stride抽虚拟入口，其他年份增删行不得平移抽样。"""
    if not isinstance(stride, int) or stride < 1 or panel.duplicated(["datetime", "symbol"]).any():
        raise ValueError("虚拟入口抽样步长须正整数，时间股票键须唯一")
    hashed = pd.util.hash_pandas_object(panel[["datetime", "symbol"]], index=False).to_numpy()
    return panel.loc[hashed % stride == 0].copy()


def load_bounded_data(cfg: dict) -> tuple:
    """按cfg复用年度原缓存，只扩展来源/有限ATR支持pkl、有限面板/独立持仓pkl；返回重训统一六项契约。

    bounded_cache_dir为独立空间；股票/源码/基础缓存/信号资格/费用均入指纹。旧ATR/退出缓存不混用。
    合资格入口不看未来成交；未来列仅是标签，日期来源只依过去观测及完整原截面。
    """
    panel, raw, membership, manifest = load_annual_panel(cfg)
    del membership
    clock = pd.DatetimeIndex(sorted(raw.datetime.unique()))
    source = Path(manifest["cache_path"])
    kept = sorted(c for c in panel if c.startswith("cs_rank_") and c.removeprefix("cs_rank_") in create_finite_factor_names())
    if not kept:
        raise ValueError("没有白名单有限因子")
    excluded = sorted(c for c in panel if c.startswith("cs_rank_") and c not in kept)
    panel = panel.drop(columns=excluded)
    specs = {"source": str(source), "manifest": manifest, "finite_columns": kept,
             "source_files": [(n, (source / n).stat().st_size, (source / n).stat().st_mtime_ns) for n in ["panel.pkl", "market.pkl"]],
             "code": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                      [str(Path(__file__)), str(Path(__file__).with_name("multiscale_training.py")),
                       str(Path(__file__).with_name("learned_exit_models.py")), str(Path(__file__).with_name("inventory_t_data.py")),
                       str(Path(__file__).with_name("annual_training.py"))]},
             "cfg": {k: cfg[k] for k in ["entry_horizons", "minimum_signal_amount", "filter_signal_untradable",
                      "exit_ages", "exit_entry_stride", "label_clip", "slippage_rate", "commission_rate", "transfer_rate",
                      "stamp_rate_from_20230828", "stamp_rate_before_20230828", "max_bar_amount_fraction", "lot_size"]}}
    key = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cfg["bounded_cache_dir"]) / key
    directory.mkdir(parents=True, exist_ok=True)
    support_dir = directory / "symbol_support"
    support_dir.mkdir(exist_ok=True)
    support_path = directory / "support.pkl"
    if support_path.exists():
        print(f"复用有限来源/ATR支持：{support_path}", flush=True)
        support = pd.read_pickle(support_path)
    else:
        symbols = sorted(raw.symbol.unique())
        tasks = [(s, cfg, clock, str(support_dir)) for s in symbols]
        paths = []
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            for count, path in enumerate(pool.map(build_support_file, tasks), 1):
                paths.append(path)
                if count % 100 == 0 or count == len(tasks):
                    print(f"有限来源/ATR支持 {count}/{len(tasks)}；未重建原因子", flush=True)
        support = pd.concat([pd.read_pickle(p)["support"] for p in paths], ignore_index=True)
        temporary = support_path.with_suffix(".tmp.pkl")
        support.to_pickle(temporary)
        temporary.replace(support_path)
    indexed = support.set_index(["datetime", "symbol"])
    keys = pd.MultiIndex.from_frame(raw[["datetime", "symbol"]])
    if not indexed.index.is_unique or (indexed.index.get_indexer(keys) < 0).any():
        raise ValueError("有限ATR支持缺少行情键或含重复")
    raw["atr"] = indexed.finite_atr.reindex(keys).to_numpy()
    market = MarketStore(raw)
    bundle = directory / "bounded_panel.pkl"
    exit_path = directory / "bounded_exit.pkl"
    if bundle.exists() and exit_path.exists():
        print(f"复用有限特征与持仓缓存：{bundle}", flush=True)
        out = pd.read_pickle(bundle)
        data = pd.read_pickle(exit_path)
    else:
        panel = propagate_feature_sources(panel, support, clock)
        panel = select_signal_tradable_panel(panel, raw, cfg)
        flags, _ = load_signal_eligibility(manifest, cfg)
        panel = apply_signal_eligibility(panel, flags, cfg)
        amounts = raw.set_index(["datetime", "symbol"]).amount
        panel = panel.loc[amounts.reindex(pd.MultiIndex.from_frame(panel[["datetime", "symbol"]])).to_numpy() >= cfg["minimum_signal_amount"]].copy()
        panel["cross_section_count"] = panel.groupby("datetime").symbol.transform("size")
        matrices = calculate_base_matrices(market)
        out = build_context_panel(panel, market, matrices, cfg)
        # 稳定键抽样不能因留出年增减若干入口行而改变其他年份的虚拟持仓训练样本。
        sampled = select_virtual_entries(out, cfg["exit_entry_stride"])
        data = {"matrices": matrices, "samples": build_exit_samples(sampled, market, matrices, {**cfg, "exit_entry_stride": 1})}
        for path, payload in [(bundle, out), (exit_path, data)]:
            temporary = path.with_suffix(".tmp.pkl")
            pd.to_pickle(payload, temporary)
            temporary.replace(path)
        print(f"已保存有限特征面板{len(out):,}行/持仓样本{len(data['samples']):,}行", flush=True)
    write_json(directory / "metadata.json", {**specs, "excluded_columns": excluded,
        "atr_definition": "最近240个原始有效分钟真实波幅的简单算术均值，另需1条前收，不是Wilder递推",
        "known_limitations": "非历史ST、复权、撮合代理及未来年份训练仍保留；来源区间是保守包络，不是逐个因子最短窗口"})
    cache = {"bounded_cache_path": str(bundle), "exit_cache_path": str(exit_path), "key": key,
             "kept_base_factors": len(kept), "excluded_base_factors": excluded, "rows": len(out),
             "support_cache": str(support_path)}
    return out, market, data["matrices"], data["samples"], manifest, cache
