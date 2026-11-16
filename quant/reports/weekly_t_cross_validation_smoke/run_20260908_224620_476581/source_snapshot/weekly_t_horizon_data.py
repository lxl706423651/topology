"""独立构建短周期做T标签和冻结选股年度内的真实恢复报价。

标签从信号下一交易分钟的close起算20/40/60个交易分钟，跳午休、不跨日。
未来价格只进入标签/执行字段，缺失分钟不填充，也不据此删除当前信号。
恢复报价覆盖冻结选股股票的整个被选年度，仅在实际到达该分钟后可供执行；
不得把未来能否恢复作为首腿条件。本模块不改既有周度、日内或来源追踪缓存。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from .weekly_t_data import (
    DEFAULT_SIGNAL_TIMES, MAINBOARD_PREFIXES, load_weekly_symbol_bars, write_weekly_cache,
)


SCHEMA_VERSION = 1
DEFAULT_HORIZONS = [20, 40, 60]
TRADING_MINUTES = tuple(range(571, 691)) + tuple(range(781, 901))
MINUTE_POSITIONS = {minute: index for index, minute in enumerate(TRADING_MINUTES)}
LABEL_COLUMNS = ["datetime", "symbol", "horizon_minutes", "execution_time", "execution_price",
                 "recovery_time", "recovery_price", "label_start", "label_end", "target_return"]
QUOTE_COLUMNS = ["datetime", "symbol", "open", "high", "low", "close", "volume", "amount"]
SELECTION_COLUMNS = ["symbol", "trade_week", "entry_date", "exit_date"]


def advance_trading_minutes(timestamp: object, steps: int) -> pd.Timestamp:
    """从精确交易分钟向后移动steps步，跳午休；超过当日15:00返回NaT。

09:36加20步等于09:56；11:01加60步等于13:31。输入必须是无时区的
09:31–11:30/13:01–15:00整分钟，steps必须为非负整数；NaT原样传递。
"""
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("交易分钟步数须为非负整数")
    value = pd.Timestamp(timestamp)
    if pd.isna(value):
        return pd.NaT
    minute = value.hour * 60 + value.minute
    if value.tzinfo is not None or value.second or value.microsecond or value.nanosecond or minute not in MINUTE_POSITIONS:
        raise ValueError("起点必须是无时区的精确交易分钟")
    position = MINUTE_POSITIONS[minute] + int(steps)
    return (value.normalize() + pd.Timedelta(minutes=TRADING_MINUTES[position])
            if position < len(TRADING_MINUTES) else pd.NaT)


def next_trading_minute(timestamp: object) -> pd.Timestamp:
    """返回同日下一交易分钟，11:30后为13:01，15:00后为NaT。"""
    return advance_trading_minutes(timestamp, 1)


def validate_horizon_config(cfg: dict) -> tuple[list[str], list[int], list[int]]:
    """校验固定信号和周期，返回信号、周期及恢复后五分钟的去重时钟。"""
    signals = list(cfg.get("signal_times", DEFAULT_SIGNAL_TIMES))
    horizons = list(cfg.get("horizons", DEFAULT_HORIZONS))
    if not signals or len(signals) != len(set(signals)):
        raise ValueError("signal_times须为非空且不重复的分钟列表")
    if not horizons or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                           or value <= 0 or value > 240 for value in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons须为1至240范围内不重复的正整数")
    signals = sorted(signals)
    horizons = sorted(int(value) for value in horizons)
    planned = set()
    for clock in signals:
        if not isinstance(clock, str) or len(clock) != 5:
            raise ValueError("signal_times须使用HH:MM格式")
        signal = pd.Timestamp("2000-01-03 " + clock)
        execution = next_trading_minute(signal)
        if pd.isna(execution):
            raise ValueError("信号必须在当日仍有下一交易分钟")
        for horizon in horizons:
            recovery = advance_trading_minutes(execution, horizon)
            for retry in range(6):
                candidate = advance_trading_minutes(recovery, retry)
                if pd.notna(candidate) and candidate.hour * 60 + candidate.minute < 895:
                    planned.add(candidate.hour * 60 + candidate.minute)
    return signals, horizons, sorted(planned)


def load_frozen_selections(cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """读取冻结各折内/外层选股文件，核对同股票周交易日期，记录逐文件SHA。

默认从frozen_selection_report_dir/year_*读取两个固定文件名；测试或显式复用
也可传frozen_selection_files列表。恢复报价按trade_week年份的股票并集覆盖
全年，不依据选股收益、实际恢复成功或未来退市情况筛选。
"""
    project_root = Path(__file__).resolve().parents[1]
    if cfg.get("frozen_selection_files"):
        paths = [Path(value) for value in cfg["frozen_selection_files"]]
        paths = [path if path.is_absolute() else project_root / path for path in paths]
    else:
        if not cfg.get("frozen_selection_report_dir"):
            raise ValueError("必须提供冻结选股报告目录或文件列表")
        root = Path(cfg["frozen_selection_report_dir"])
        if not root.is_absolute():
            root = project_root / root
        paths = [path for name in ("validation_weekly_selections.csv", "weekly_selections.csv")
                 for path in root.glob("year_*/" + name)]
    paths = sorted(set(path.resolve() for path in paths))
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("没有完整可读的冻结选股文件")
    parts, sources = [], []
    for path in paths:
        frame = pd.read_csv(path, usecols=SELECTION_COLUMNS)
        if frame[SELECTION_COLUMNS].isna().any().any():
            raise ValueError("冻结选股键与交易日期不能缺失")
        for column in SELECTION_COLUMNS[1:]:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
            if not frame[column].eq(frame[column].dt.normalize()).all():
                raise ValueError("冻结选股日期必须是自然日零点")
        if (frame.entry_date.gt(frame.exit_date).any()
                or frame.entry_date.lt(frame.trade_week).any()
                or frame.exit_date.ge(frame.trade_week + pd.Timedelta(days=7)).any()
                or frame.trade_week.dt.weekday.ne(0).any()):
            raise ValueError("冻结选股开平仓日期必须属于trade_week自然周")
        parts.append(frame)
        stat = path.stat()
        sources.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(frame)})
    selections = pd.concat(parts, ignore_index=True).drop_duplicates()
    if selections.duplicated(["symbol", "trade_week"]).any():
        raise ValueError("冻结文件中同股票周存在冲突交易日期")
    return selections.sort_values(["symbol", "trade_week"]).reset_index(drop=True), sources


def build_symbol_horizon_data(raw: pd.DataFrame, symbol: str, selected_years: list[int],
                              cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从单股合法raw提取标签和年度固定时钟报价，禁止填补未来缺价或修改raw。

每个真实存在的当前信号生成所有周期行，即使下一成交分钟缺失、恢复分钟
缺失或计划超过当日也保留行。增量报价只删去会与已有尾盘六分钟缓存重复的
14:55–15:00，不删除零量或一字线，且不依赖当前信号或标签是否可用。
"""
    signals, horizons, quote_minutes = validate_horizon_config(cfg)
    timestamp = pd.to_datetime(raw.datetime, errors="raise")
    exact = timestamp.dt.second.eq(0) & timestamp.dt.microsecond.eq(0) & timestamp.dt.nanosecond.eq(0)
    frame = raw.loc[exact].copy()
    if frame.datetime.duplicated().any():
        raise ValueError("单股raw必须已经去重")
    indexed = frame.set_index("datetime").sort_index()
    signal_minutes = {int(clock[:2]) * 60 + int(clock[3:]) for clock in signals}
    minute = frame.datetime.dt.hour * 60 + frame.datetime.dt.minute
    observed = pd.DatetimeIndex(frame.loc[minute.isin(signal_minutes), "datetime"]).sort_values()
    # 时钟表仅有8×3行，跨日映射用向量化日期偏移，避免对六千万样本逐行运算。
    offsets = {}
    for clock in signals:
        current = pd.Timestamp("2000-01-03 " + clock)
        execution = next_trading_minute(current)
        offsets[current.hour * 60 + current.minute] = (
            execution - current.normalize(),
            [advance_trading_minutes(execution, value) - current.normalize() for value in horizons])
    horizon_count = len(horizons)
    labels = pd.DataFrame({"datetime": observed.repeat(horizon_count), "symbol": symbol,
                           "horizon_minutes": np.tile(np.array(horizons, dtype=np.int16), len(observed))})
    observed_minutes = observed.hour * 60 + observed.minute
    execution_delta = pd.to_timedelta([offsets[value][0] for value in observed_minutes])
    recovery_delta = pd.to_timedelta([offsets[value][1][index]
                                     for value in observed_minutes for index in range(horizon_count)])
    executions = observed.normalize().repeat(horizon_count) + execution_delta.repeat(horizon_count)
    recoveries = observed.normalize().repeat(horizon_count) + recovery_delta
    labels["execution_time"] = executions
    labels["execution_price"] = indexed.close.reindex(executions).to_numpy(dtype=float)
    labels["recovery_time"] = recoveries
    labels["recovery_price"] = indexed.close.reindex(recoveries).to_numpy(dtype=float)
    labels["label_start"] = executions
    labels["label_end"] = recoveries
    labels["target_return"] = labels.recovery_price / labels.execution_price - 1.0
    quote_mask = frame.datetime.dt.year.isin(selected_years) & minute.isin(quote_minutes)
    quotes = frame.loc[quote_mask].copy()
    quotes["symbol"] = symbol
    return (labels[LABEL_COLUMNS].reset_index(drop=True),
            quotes[QUOTE_COLUMNS].sort_values("datetime").reset_index(drop=True))


def build_horizon_symbol_cache(task: tuple) -> str:
    """并行执行单股读取，原子写分股标签、报价和质量统计，允许中断后复用。"""
    symbol, paths, start, end, selected_years, effective_cfg, directory = task
    destination = Path(directory) / f"{symbol}.pkl"
    if destination.exists():
        return str(destination)
    raw, audit = load_weekly_symbol_bars(paths, start, end)
    labels, quotes = build_symbol_horizon_data(raw, symbol, selected_years, effective_cfg)
    audit.update({"symbol": symbol, "label_rows": len(labels), "quote_rows": len(quotes),
                  "signal_rows": len(labels) // len(effective_cfg["horizons"]),
                  "finite_label_rows": int(np.isfinite(labels.target_return).sum()),
                  "missing_execution_rows": int(labels.execution_price.isna().sum()),
                  "missing_recovery_rows": int(labels.recovery_price.isna().sum()),
                  "beyond_session_rows": int(labels.recovery_time.isna().sum()),
                  "quote_zero_volume_rows": int(quotes.volume.eq(0).sum()),
                  "quote_zero_amount_rows": int(quotes.amount.eq(0).sum())})
    write_weekly_cache({"labels": labels, "quotes": quotes, "audit": audit}, destination)
    return str(destination)


def load_horizon_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """建立或复用短周期全主板标签和冻结选股年度报价，返回labels、quotes、manifest。

cfg读取data_root/start_date/end_date/workers/max_symbols/signal_times/horizons、
frozen_selection_report_dir（或显式文件列表）及horizon_cache_dir。指纹覆盖本模块、
共享CSV读取模块、原始文件stat、有效数据配置和全部冻结选股文件SHA；workers
仅控制资源不改变数据。quotes为增量报价，需要调用方合并既有尾盘六分钟缓存。
"""
    signals, horizons, quote_minutes = validate_horizon_config(cfg)
    project_root = Path(__file__).resolve().parents[1]
    root = Path(cfg["data_root"])
    root = (root if root.is_absolute() else project_root / root).resolve()
    start, end = str(cfg.get("start_date", "2023-01-01")), str(cfg.get("end_date", "2026-04-30"))
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start_date不能晚于end_date")
    maximum, workers = int(cfg.get("max_symbols", 0)), int(cfg.get("workers", 4))
    if maximum < 0 or workers < 1:
        raise ValueError("max_symbols须非负，workers须至少为1")
    selections, selection_sources = load_frozen_selections(cfg)
    paths = sorted(path for path in root.glob("[0-9][0-9][0-9][0-9]/*.csv")
                   if path.stem.startswith(MAINBOARD_PREFIXES)
                   and pd.Timestamp(start).year <= int(path.parent.name) <= pd.Timestamp(end).year)
    symbols = sorted({path.stem for path in paths})
    if maximum:
        symbols = symbols[:maximum]
    allowed = set(symbols)
    paths = [path for path in paths if path.stem in allowed]
    if not paths:
        raise ValueError("指定目录及日期范围没有沪深主板CSV")
    source_stats = [[str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]
    selected_years = selections.assign(year=selections.trade_week.dt.year).groupby("symbol").year.unique().to_dict()
    fingerprint = {"schema": SCHEMA_VERSION, "data_root": str(root), "files": source_stats,
                   "symbols": symbols, "start_date": start, "end_date": end,
                   "signal_times": signals, "horizons": horizons, "recovery_retry_steps": 5,
                   "incremental_quote_minutes": quote_minutes,
                   "frozen_selection_sources": selection_sources,
                   "quote_scope": "股票在冻结选股并集中出现的每个trade_week年度的全部真实日期",
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "reader_code_sha256": hashlib.sha256(Path(__file__).with_name("weekly_t_data.py").read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    cache_root = Path(cfg.get("horizon_cache_dir") or project_root / "reports/weekly_t_cross_validation/horizon_cache")
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    cache = cache_root / f"horizon_{key}"
    cache.mkdir(parents=True, exist_ok=True)
    labels_path, quotes_path, metadata_path = cache / "labels.pkl", cache / "recovery_quotes.pkl", cache / "metadata.json"
    if labels_path.exists() and quotes_path.exists() and metadata_path.exists():
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != key:
            raise ValueError("短周期缓存指纹与元数据不一致")
        labels, quotes = pd.read_pickle(labels_path), pd.read_pickle(quotes_path)
        if (list(labels) != LABEL_COLUMNS or list(quotes) != QUOTE_COLUMNS
                or len(labels) != manifest["label_rows"] or len(quotes) != manifest["quote_rows"]):
            raise ValueError("短周期缓存列或行数与元数据不一致")
        print(f"复用短周期缓存：{len(labels):,}标签/{len(quotes):,}增量报价；{cache}", flush=True)
        return labels, quotes, {**manifest, "cache_hit": True}
    files_by_symbol = {symbol: [] for symbol in symbols}
    for path in paths:
        files_by_symbol[path.stem].append(str(path))
    effective_cfg = {"signal_times": signals, "horizons": horizons}
    tasks = [(symbol, files_by_symbol[symbol], start, end,
              [int(year) for year in selected_years.get(symbol, [])], effective_cfg, str(cache)) for symbol in symbols]
    print(f"构建短周期缓存：{len(symbols)}股/{workers}进程；{horizons}交易分钟/{len(quote_minutes)}增量报价时钟", flush=True)
    fragments = []
    if workers == 1:
        for index, path in enumerate(map(build_horizon_symbol_cache, tasks), 1):
            fragments.append(path)
            if index % 250 == 0 or index == len(tasks):
                print(f"短周期分股缓存 {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for index, path in enumerate(pool.map(build_horizon_symbol_cache, tasks), 1):
                fragments.append(path)
                if index % 250 == 0 or index == len(tasks):
                    print(f"短周期分股缓存 {index}/{len(tasks)}", flush=True)
    label_parts, quote_parts, audits = [], [], []
    for path in fragments:
        payload = pd.read_pickle(path)
        if list(payload["labels"]) != LABEL_COLUMNS or list(payload["quotes"]) != QUOTE_COLUMNS:
            raise ValueError("短周期分股缓存列结构不匹配")
        label_parts.append(payload["labels"])
        quote_parts.append(payload["quotes"])
        audits.append(payload["audit"])
    # 分股字典序及各股时间序已固定，避免再次全局排序六千万标签的内存峰值。
    labels = pd.concat(label_parts, ignore_index=True)
    quotes = pd.concat(quote_parts, ignore_index=True)
    del label_parts, quote_parts
    quality = pd.DataFrame(audits)
    year_symbol_count = selections.assign(year=selections.trade_week.dt.year).groupby("year").symbol.nunique()
    manifest = {**fingerprint, "fingerprint": key, "cache_path": str(cache),
                "labels_path": str(labels_path), "quotes_path": str(quotes_path), "cache_hit": False,
                "source_symbol_count": len(symbols), "label_rows": len(labels), "quote_rows": len(quotes),
                "signal_rows": int(quality.signal_rows.sum()), "finite_label_rows": int(quality.finite_label_rows.sum()),
                "missing_execution_rows": int(quality.missing_execution_rows.sum()),
                "missing_recovery_rows": int(quality.missing_recovery_rows.sum()),
                "beyond_session_rows": int(quality.beyond_session_rows.sum()),
                "quote_zero_volume_rows": int(quality.quote_zero_volume_rows.sum()),
                "quote_zero_amount_rows": int(quality.quote_zero_amount_rows.sum()),
                "source_duplicates": int(quality.duplicates.sum()), "source_invalid_bars": int(quality.invalid_bars.sum()),
                "frozen_selected_stockweeks": len(selections),
                "frozen_selected_symbols_by_year": {str(year): int(count) for year, count in year_symbol_count.items()},
                "label_columns": LABEL_COLUMNS, "quote_columns": QUOTE_COLUMNS,
                "row_order": "symbol,datetime,horizon_minutes（报价无horizon列）",
                "horizon_anchor": "execution_time + horizon个交易分钟；execution_time=当前信号下一交易分钟",
                "missing_data_rule": "保留所有真实当前信号；未来缺分钟/跨日只使标签NaN，绝不前后填价",
                "feature_contract": "本缓存没有f_*特征；execution/recovery/target只作标签或到时执行，禁止首腿未来过滤",
                "tail_quote_contract": "增量报价不含14:55–15:00；执行前须合并既有六分钟恢复缓存",
                "scope_limit": "股票未曾被该年度冻结选股选中的年度不补早时钟报价；不从未来年度选股扩充该年股票"}
    write_weekly_cache(labels, labels_path)
    write_weekly_cache(quotes, quotes_path)
    quality.to_csv(cache / "data_quality.csv", index=False)
    write_weekly_cache(manifest, metadata_path)
    print(f"短周期缓存完成：{len(labels):,}标签/{len(quotes):,}增量报价；{cache}", flush=True)
    return labels, quotes, manifest
