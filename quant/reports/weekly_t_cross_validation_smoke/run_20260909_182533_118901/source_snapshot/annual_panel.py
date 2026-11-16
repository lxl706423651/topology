"""年度研究的数据缓存：输入只读分钟行情，输出共同时钟的特征和执行行情 pkl。

依赖配置中的数据目录、日期、股票名称快照和股票池设置。缓存与旧的逐股票
行号采样面板隔离；原始行情不修改。名称快照并非历史点时 ST 状态，必须披露。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_backtest.factor_library import build_factor_library, calculate_atr
from scripts.run_all_a_method_backtests import is_mainboard_symbol
from scripts.run_dynamic_high_vol_method_backtests import (
    build_weekly_membership, calculate_symbol_daily_volatility,
)

SCHEMA_VERSION = 1


def write_json(path: Path, data: object) -> None:
    """把 data 写入 path，中文不转义；日期和路径转换为字符串。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def create_sampling_clock(dates: pd.DatetimeIndex, step: int) -> pd.DatetimeIndex:
    """为 dates 中每个交易日生成 step 分钟共同时钟及15:00估值点；不跨午休计数。"""
    if step < 2 or 240 % step:
        raise ValueError("采样步长须大于1且整除240")
    offsets = np.r_[np.arange(571, 691), np.arange(781, 901)][::step]
    offsets = np.unique(np.r_[offsets, 900])
    values = dates.normalize().to_numpy()[:, None] + offsets[None, :] * np.timedelta64(1, "m")
    return pd.DatetimeIndex(values.ravel()).sort_values()


def load_symbol_bars(root: Path, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """读取 root 内 symbol 的 start/end 分钟行情，去重并剔除非法OHLCV；返回质量计数。"""
    parts = [pd.read_csv(p) for p in sorted(root.glob(f"[0-9][0-9][0-9][0-9]/{symbol}.csv"))]
    frame = pd.concat(parts, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.loc[frame.datetime.between(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), inclusive="left")]
    duplicates = int(frame.datetime.duplicated().sum())
    frame = frame.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    cols = ["open", "high", "low", "close", "volume", "amount"]
    frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(frame[cols]).all(axis=1)
    valid &= (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= (frame[["volume", "amount"]] >= 0).all(axis=1)
    valid &= frame.high + 1e-6 >= frame[["open", "close"]].max(axis=1)
    valid &= frame.low - 1e-6 <= frame[["open", "close"]].min(axis=1)
    invalid = int((~valid).sum())
    return frame.loc[valid].reset_index(drop=True), {"symbol": symbol, "duplicates": duplicates, "invalid_bars": invalid}


def build_symbol_cache(task: tuple) -> str:
    """由 task=(股票,配置,交易日,入池日,缓存目录) 生成单股票特征/行情文件，可断点复用。"""
    symbol, cfg, dates, member_dates, directory = task
    path = Path(directory) / f"{symbol}.pkl"
    if path.exists():
        return str(path)
    raw, audit = load_symbol_bars(Path(cfg["data_root"]), symbol, cfg["start_date"], cfg["end_date"])
    if raw.empty:
        raise ValueError(f"{symbol} 没有合法行情")
    clock = create_sampling_clock(pd.DatetimeIndex(dates), cfg["sample_minutes"])
    bars = raw.set_index("datetime")
    atr = calculate_atr(bars.high, bars.low, bars.close, 240)
    market = bars.reindex(clock)[["close", "volume", "amount"]].copy()
    market["atr"] = atr.reindex(clock)
    # 本点信号只看本点之前的区间最高价，包含中间未采样的分钟线。
    buckets = clock.searchsorted(bars.index, side="left")
    highs = bars.high.groupby(buckets).max()
    market["interval_high"] = highs.reindex(range(len(clock))).to_numpy()
    execution = clock + pd.Timedelta(minutes=1)
    for col in ["close", "high", "low", "volume", "amount"]:
        market[f"execution_{col}"] = bars[col].reindex(execution).to_numpy()
    daily_close = bars.close.groupby(bars.index.normalize()).last().reindex(dates)
    previous_close = daily_close.ffill().shift(1)
    market["previous_close"] = previous_close.reindex(clock.normalize()).to_numpy()
    market["symbol"] = symbol
    market.index.name = "datetime"
    eligible = clock.normalize().isin(pd.to_datetime(member_dates)) & (clock.hour != 15)
    eligible &= market.close.notna() & (market.volume > 0)
    signal_times = clock[eligible]
    factors = build_factor_library(raw)
    factors.index = pd.DatetimeIndex(raw.datetime)
    panel = factors.reindex(signal_times).astype("float32")
    panel["symbol"] = symbol
    panel.index.name = "datetime"
    next_days = pd.Series(pd.DatetimeIndex(dates)[1:], index=pd.DatetimeIndex(dates)[:-1])
    label_dates = pd.to_datetime(next_days.reindex(signal_times.normalize()).to_numpy())
    label_times = label_dates + (signal_times - signal_times.normalize()) + pd.Timedelta(minutes=1)
    entry = bars.close.reindex(signal_times + pd.Timedelta(minutes=1)).to_numpy()
    end_price = bars.close.reindex(label_times).to_numpy()
    panel["label_end"] = label_times
    panel["forward_return"] = (end_price / entry - 1).astype("float32")
    # 标签缺失只影响训练，不能因为未来缺失就从当时的推断股票池中删除。
    payload = {"panel": panel.reset_index(), "market": market.reset_index(), "audit": audit}
    temporary = path.with_suffix(".tmp")
    pd.to_pickle(payload, temporary)
    temporary.replace(path)
    return str(path)


def create_data_manifest(cfg: dict) -> tuple[list[str], dict]:
    """根据 cfg 的所有年份文件并集和显式名称快照生成候选及缓存指纹，不要求未来存续。"""
    root = Path(cfg["data_root"])
    paths = sorted(root.glob("[0-9][0-9][0-9][0-9]/*.csv"))
    symbols = sorted({p.stem for p in paths if is_mainboard_symbol(p.stem)})
    names = pd.read_csv(cfg["name_snapshot"], dtype={"code": str})
    excluded = set(names.loc[names.name.str.contains("ST|退", case=False, na=False), "code"].str.zfill(6))
    symbols = [s for s in symbols if s.split(".")[-1] not in excluded]
    if cfg["max_symbols"]:
        symbols = symbols[:cfg["max_symbols"]]
    symbol_set = set(symbols)
    files = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.stem in symbol_set]
    library = Path(__file__).with_name("factor_library.py")
    manifest = {"schema": SCHEMA_VERSION, "files": files, "symbols": symbols,
                "date_range": [cfg["start_date"], cfg["end_date"]],
                "universe_size": cfg["universe_size"], "min_week_days": cfg["min_week_days"],
                "sample_minutes": cfg["sample_minutes"], "name_snapshot_sha256": hashlib.sha256(Path(cfg["name_snapshot"]).read_bytes()).hexdigest(),
                "factor_code_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "panel_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "st_limitation": "只用2026年名称快照排除ST/退；历史点时状态缺失，存在前视筛选及幸存者偏差"}
    return symbols, manifest


def load_annual_panel(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """按 cfg 构建或加载共同时钟pkl；返回截面秩特征、执行行情、日股票池和来源元数据。"""
    symbols, manifest = create_data_manifest(cfg)
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(cfg["cache_dir"]) / key
    cache.mkdir(parents=True, exist_ok=True)
    manifest["cache_path"] = str(cache)
    write_json(cache / "metadata.json", manifest)
    if (cache / "panel.pkl").exists() and (cache / "market.pkl").exists():
        print(f"复用年度面板缓存：{cache}", flush=True)
        return pd.read_pickle(cache / "panel.pkl"), pd.read_pickle(cache / "market.pkl"), pd.read_pickle(cache / "membership.pkl"), manifest
    daily_path = cache / "daily.pkl"
    if daily_path.exists():
        daily = pd.read_pickle(daily_path)
    else:
        tasks = [(s, Path(cfg["data_root"]), pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])) for s in symbols]
        frames = []
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            for index, frame in enumerate(pool.map(calculate_symbol_daily_volatility, tasks), 1):
                frames.append(frame)
                if index % 250 == 0 or index == len(tasks):
                    print(f"日波动率缓存 {index}/{len(tasks)}", flush=True)
        daily = pd.concat(frames, ignore_index=True)
        daily.to_pickle(daily_path)
    dates = pd.DatetimeIndex(sorted(daily.trade_date.unique()))
    membership = build_weekly_membership(daily, dates.date, cfg["universe_size"], cfg["min_week_days"])
    membership.to_pickle(cache / "membership.pkl")
    symbol_dates = membership.groupby("symbol").trade_date.unique()
    tasks = [(s, cfg, dates, d, str(cache)) for s, d in symbol_dates.items()]
    print(f"共同时钟特征：{len(tasks)}只曾入池股票，{len(dates)}个交易日", flush=True)
    files = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        for index, path in enumerate(pool.map(build_symbol_cache, tasks), 1):
            files.append(path)
            if index % 100 == 0 or index == len(tasks):
                print(f"单股票特征缓存 {index}/{len(tasks)}", flush=True)
    panels, markets, audits = [], [], []
    for path in files:
        result = pd.read_pickle(path)
        panels.append(result["panel"])
        markets.append(result["market"])
        audits.append(result["audit"])
    panel = pd.concat(panels, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    factors = [c for c in panel if c not in {"datetime", "symbol", "label_end", "forward_return"}]
    ranked = panel.groupby("datetime", sort=False)[factors].rank(pct=True).astype("float32")
    ranked.columns = [f"cs_rank_{c}" for c in factors]
    panel = pd.concat([panel[["datetime", "symbol", "label_end", "forward_return"]], ranked], axis=1)
    panel["target_rank"] = panel.groupby("datetime").forward_return.rank(pct=True).astype("float32")
    panel["cross_section_count"] = panel.groupby("datetime").symbol.transform("size")
    market = pd.concat(markets, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    pd.DataFrame(audits).to_csv(cache / "data_quality.csv", index=False)
    for name, value in [("panel", panel), ("market", market)]:
        temporary = cache / f"{name}.tmp"
        value.to_pickle(temporary)
        temporary.replace(cache / f"{name}.pkl")
    return panel, market, membership, manifest
