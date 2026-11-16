"""多尺度开仓训练数据：复用分钟因子、持仓状态与交易资格缓存，不改原始数据。

新增过去1/3/5/10/20/60交易日价格状态、当期股票池环境和3/5日未来标签。
过去特征与未来标签分列保存，后者仅在完整处于拟合窗口时使用；不根据测试预测选规则。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_backtest.annual_training import select_training_features
from quant_backtest.inventory_t_data import load_signal_eligibility, apply_signal_eligibility


CONTEXT_NAMES = {
    **{f"cs_rank_context_momentum_{n}d": f"过去{n}交易日同刻收益在当时合资格周池内的百分位" for n in [1, 3, 5, 10, 20, 60]},
    **{f"cs_rank_context_volatility_{n}d": f"过去{n}交易日采样收益波动率的截面百分位" for n in [5, 20]},
    **{f"cs_rank_context_drawdown_{n}d": f"距过去{n}交易日采样最高收盘价回撤的截面百分位" for n in [5, 20, 60]},
    "cs_rank_context_daily_return": "当前价相对前日收盘收益的截面百分位",
    "cs_rank_context_amount": "当前分钟成交额的截面百分位",
    "cs_rank_context_volume_ratio": "当前分钟量相对过去11个采样点均量的截面百分位",
    "context_pool_return": "当前合资格周池相对前收盘平均收益",
    "context_pool_breadth": "当前合资格周池上涨股票比例",
    "context_pool_momentum5": "当前合资格周池过去5交易日平均收益",
    "context_pool_dispersion": "当前合资格周池相对前收盘收益截面标准差",
    "context_session_fraction": "当前交易时段位置，午休不计时",
}
MACRO_COLUMNS = [c for c in CONTEXT_NAMES if not c.startswith("cs_rank_")]


def calculate_multiday_labels(panel: pd.DataFrame, market, horizon: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """对panel键生成horizon交易日同刻下一分钟收益和结束时间；market仅用于标签，不决定推断行是否存在。"""
    idx = market.times.get_indexer(panel.datetime)
    assets = np.array([market.asset_index[s] for s in panel.symbol])
    days = pd.DatetimeIndex(sorted(market.days.unique()))
    future_day = market.day_index[idx] + horizon
    covered = future_day < len(days)
    future = np.full(len(panel), -1, dtype=int)
    future[covered] = market.times.get_indexer(days[future_day[covered]] + (market.times[idx[covered]] - market.days[idx[covered]]))
    covered &= future >= 0
    output = np.full(len(panel), np.nan)
    end = np.full(len(panel), np.datetime64("NaT"), dtype="datetime64[ns]")
    pos = np.flatnonzero(covered)
    buy = market.values["execution_close"][idx[pos], assets[pos]]
    sell = market.values["execution_close"][future[pos], assets[pos]]
    valid = (buy > 0) & np.isfinite(buy) & (sell > 0) & np.isfinite(sell)
    valid &= (market.values["execution_volume"][idx[pos], assets[pos]] > 0) & (market.values["execution_volume"][future[pos], assets[pos]] > 0)
    output[pos[valid]] = sell[valid] / buy[valid] - 1
    end[pos] = (market.times[future[pos]] + pd.Timedelta(minutes=1)).to_numpy()
    return output, pd.DatetimeIndex(end)


def build_context_panel(panel: pd.DataFrame, market, matrices: dict, cfg: dict) -> pd.DataFrame:
    """为panel已过滤的当时周池添加因果多尺度特征，market/matrices提供过去行情；cfg给标签期限。"""
    out = panel.copy()
    indices = market.times.get_indexer(out.datetime)
    assets = np.array([market.asset_index[s] for s in out.symbol])
    close = pd.DataFrame(market.values["close"])
    raw = {}
    for days in [1, 3, 5, 10, 20, 60]:
        raw[f"momentum_{days}d"] = close.pct_change(11 * days, fill_method=None).to_numpy()[indices, assets]
    returns = close.pct_change(fill_method=None)
    for days in [5, 20]:
        raw[f"volatility_{days}d"] = returns.rolling(11 * days, min_periods=11 * days // 2).std().to_numpy()[indices, assets]
    for days in [5, 20, 60]:
        raw[f"drawdown_{days}d"] = (close / close.rolling(11 * days, min_periods=11 * days // 2).max() - 1).to_numpy()[indices, assets]
    raw["daily_return"] = matrices["daily_return"][indices, assets]
    raw["amount"] = market.values["amount"][indices, assets]
    raw["volume_ratio"] = matrices["volume_ratio"][indices, assets]
    for name, values in raw.items():
        value = pd.Series(np.where(np.isfinite(values), values, np.nan), index=out.index)
        out[f"cs_rank_context_{name}"] = value.groupby(out.datetime).rank(pct=True).astype("float32")
    daily = pd.Series(raw["daily_return"], index=out.index)
    out["context_pool_return"] = daily.groupby(out.datetime).transform("mean").astype("float32")
    breadth = (daily > 0).astype(float).where(daily.notna())
    out["context_pool_breadth"] = breadth.groupby(out.datetime).transform("mean").astype("float32")
    out["context_pool_dispersion"] = daily.groupby(out.datetime).transform("std").astype("float32")
    out["context_pool_momentum5"] = pd.Series(raw["momentum_5d"], index=out.index).groupby(out.datetime).transform("mean").astype("float32")
    out["context_session_fraction"] = matrices["session_fraction"][indices, assets]
    for horizon in cfg["entry_horizons"]:
        out[f"future_return_{horizon}d"], out[f"future_end_{horizon}d"] = calculate_multiday_labels(out, market, horizon)
    return out


def load_multiscale_panel(panel: pd.DataFrame, market, matrices: dict, manifest: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """复用资格及cfg.multiscale_cache_dir多日pkl；先按当前非一字线/成交额筛候选，不按未来成交删推断行。"""
    flags, eligibility = load_signal_eligibility(manifest, cfg)
    eligible = apply_signal_eligibility(panel, flags, cfg)
    indices = market.times.get_indexer(eligible.datetime)
    assets = np.array([market.asset_index[s] for s in eligible.symbol])
    eligible = eligible.loc[market.values["amount"][indices, assets] >= cfg["minimum_signal_amount"]].copy()
    eligible["cross_section_count"] = eligible.groupby("datetime").symbol.transform("size")
    source = Path(manifest["cache_path"])
    specs = {"source": str(source), "file_stats": [(n, (source / n).stat().st_size, (source / n).stat().st_mtime_ns) for n in ["panel.pkl", "market.pkl"]],
             "eligibility": eligibility["fingerprint"], "minimum_signal_amount": cfg["minimum_signal_amount"],
             "entry_horizons": cfg["entry_horizons"], "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "base_feature_code": hashlib.sha256(Path(__file__).with_name("learned_exit_models.py").read_bytes()).hexdigest(),
             "keys_sha256": hashlib.sha256(pd.util.hash_pandas_object(eligible[["datetime", "symbol"]], index=False).to_numpy().tobytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cfg["multiscale_cache_dir"]) / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "multiscale_panel.pkl"
    if path.exists():
        print(f"复用多尺度面板：{path}", flush=True)
        out = pd.read_pickle(path)
    else:
        print(f"只扩展多尺度特征，不重建原因子：{path}，{len(eligible):,}行", flush=True)
        out = build_context_panel(eligible, market, matrices, cfg)
        temporary = path.with_suffix(".tmp.pkl")
        out.to_pickle(temporary)
        temporary.replace(path)
        (directory / "metadata.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, {"key": key, "cache_path": str(path), "rows": len(out), "definitions": CONTEXT_NAMES}


def select_context_features(train: pd.DataFrame, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    """在train拟合片段选最多cfg.max_features列，先保留有覆盖/变化的5个环境变量，余下名额筛截面因子。"""
    macro = [c for c in MACRO_COLUMNS if train[c].notna().mean() >= cfg["feature_min_coverage"] and train[c].std() > 1e-6]
    features, manifest = select_training_features(train, {**cfg, "max_features": cfg["max_features"] - len(macro)})
    manifest.loc[manifest.feature.isin(CONTEXT_NAMES), "说明"] = manifest.loc[manifest.feature.isin(CONTEXT_NAMES), "feature"].map(CONTEXT_NAMES)
    extras = [{"feature": c, "说明": CONTEXT_NAMES[c], "coverage": train[c].notna().mean(), "selected": c in macro} for c in MACRO_COLUMNS]
    return features + macro, pd.concat([manifest, pd.DataFrame(extras)], ignore_index=True)


def select_exit_samples(samples: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """从samples保留虚拟入场时属于panel合资格周池的批次；不重建持仓矩阵，不查看实际测试持仓。"""
    keys = pd.MultiIndex.from_frame(panel[["datetime", "symbol"]])
    entries = pd.MultiIndex.from_arrays([samples.entry_datetime - pd.Timedelta(minutes=1), samples.symbol])
    return samples.loc[keys.get_indexer(entries) >= 0].copy()
