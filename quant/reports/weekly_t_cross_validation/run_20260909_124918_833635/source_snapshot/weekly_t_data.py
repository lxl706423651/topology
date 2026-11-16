"""全主板周预测与日内做T数据：只读分钟CSV，生成独立、可复用的分股及汇总缓存。

全部合法分钟参与日/周聚合；周模型特征只使用该周及此前数据，T特征仅使用
已完成信号分钟及历史。未来成交、恢复价格和下周表现仅作为执行字段/标签。
不使用当前名称或ST快照筛除历史股票；不修改原始数据或旧年度面板。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_SIGNAL_TIMES = ["09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00"]
MAINBOARD_PREFIXES = ("SH.600", "SH.601", "SH.603", "SH.605", "SZ.000", "SZ.001", "SZ.002", "SZ.003")


def write_weekly_cache(value: object, path: Path) -> None:
    """将DataFrame/字典原子写入path；JSON用于可读元数据，其余扩展名使用pkl。"""
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if path.suffix == ".json":
            temporary_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        else:
            pd.to_pickle(value, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def calculate_week_start(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """把日期映射到所属自然周周一的Timestamp，避免使用字符串或Period作训练键。"""
    index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    return index - pd.to_timedelta(index.weekday, unit="D")


def load_weekly_symbol_bars(paths: list[str], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """读取单股paths中的分钟CSV，按日期限制、去重、OHLCV关系过滤并返回质量计数。

    仅保留09:31–11:30、13:01–15:00的收盘标签分钟；开盘成交价格取09:31bar的
    open。缺失分钟不补齐成交价格，非法价格不进入特征或执行数据。
    """
    columns = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    parts = [pd.read_csv(path, usecols=columns) for path in paths]
    raw = pd.concat(parts, ignore_index=True)
    raw["datetime"] = pd.to_datetime(raw.datetime, errors="raise")
    raw = raw.loc[raw.datetime.between(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), inclusive="left")]
    duplicates = int(raw.datetime.duplicated().sum())
    raw = raw.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    raw[columns[1:]] = raw[columns[1:]].apply(pd.to_numeric, errors="coerce")
    numeric = raw[columns[1:]]
    valid = np.isfinite(numeric).all(axis=1)
    valid &= (raw[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= (raw[["volume", "amount"]] >= 0).all(axis=1)
    valid &= raw.high + 1e-6 >= raw[["open", "close"]].max(axis=1)
    valid &= raw.low - 1e-6 <= raw[["open", "close"]].min(axis=1)
    valid &= raw.high >= raw.low
    minute = raw.datetime.dt.hour * 60 + raw.datetime.dt.minute
    in_session = minute.between(571, 690) | minute.between(781, 900)
    audit = {"duplicates": duplicates, "invalid_bars": int((~valid).sum()),
             "outside_session": int((~in_session).sum()), "raw_rows_in_range": len(raw)}
    return raw.loc[valid & in_session].reset_index(drop=True), audit


def build_symbol_weekly_features(raw: pd.DataFrame, symbol: str,
                                 signal_times: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """由单股raw全部分钟生成未接下周标签的周表、日内采样表及精确日执行表。

    周成交额集中度、半方差等包含全部分钟；日内累计统计只到当前时刻。
    返回值中的f_*是唯一可供模型使用的字段，execution/recovery/target不属于特征。
    """
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    frame = raw.copy()
    frame["date"] = frame.datetime.dt.normalize()
    groups = frame.groupby("date", sort=True)
    minute_return = groups.close.pct_change(fill_method=None)
    frame["return_square"] = minute_return.pow(2)
    frame["up_square"] = minute_return.clip(lower=0).pow(2)
    frame["down_square"] = minute_return.clip(upper=0).pow(2)
    frame["up_bar"] = (minute_return > 0).astype(float).where(minute_return.notna())
    frame["amount_square"] = frame.amount.pow(2)
    daily = frame.groupby("date", sort=True).agg(
        observed_open=("open", "first"), observed_close=("close", "last"),
        high=("high", "max"), low=("low", "min"), amount=("amount", "sum"),
        volume=("volume", "sum"), bar_count=("datetime", "size"),
        first_datetime=("datetime", "min"), last_datetime=("datetime", "max"),
        return_square=("return_square", "sum"), up_square=("up_square", "sum"),
        down_square=("down_square", "sum"), return_count=("return_square", "count"),
        up_bar_count=("up_bar", "sum"), amount_square=("amount_square", "sum"),
        maximum_minute_amount=("amount", "max"))
    dates = pd.DatetimeIndex(daily.index)
    indexed = frame.set_index("datetime")
    for prefix, minute, price_column in (("open", 571, "open"), ("close", 900, "close")):
        timestamps = dates + pd.Timedelta(minutes=minute)
        exact = indexed.reindex(timestamps)
        daily[prefix] = exact[price_column].to_numpy()
        daily[prefix + "_datetime"] = timestamps.where(exact.close.notna().to_numpy())
        for column in ("amount", "high", "low", "volume"):
            daily[f"{prefix}_{column}"] = exact[column].to_numpy()
    early_close = indexed.close.reindex(dates + pd.Timedelta(hours=10)).to_numpy()
    late_start = indexed.close.reindex(dates + pd.Timedelta(hours=14, minutes=30)).to_numpy()
    daily["previous_close"] = daily.close.shift(1)
    daily["daily_return"] = daily.close / daily.open - 1
    daily["daily_amplitude"] = (daily.high - daily.low) / daily.open
    daily["early_return"] = early_close / daily.open - 1
    daily["late_return"] = daily.close / late_start - 1
    daily["trend_strength"] = (daily.close - daily.open).abs() / (daily.high - daily.low).replace(0, np.nan)
    daily["reversal_day"] = ((daily.early_return * daily.late_return) < 0).astype(float).where(
        daily.early_return.notna() & daily.late_return.notna())
    daily["history_days"] = np.arange(1, len(daily) + 1)
    daily["date"] = dates
    daily["trade_week"] = calculate_week_start(dates)
    daily["symbol"] = symbol
    week = daily.groupby("trade_week", sort=True).agg(
        observed_first_open=("observed_open", "first"), observed_last_close=("observed_close", "last"),
        observed_last_datetime=("last_datetime", "max"), f_week_high=("high", "max"),
        f_week_low=("low", "min"), f_week_amount=("amount", "sum"), f_week_volume=("volume", "sum"),
        f_observed_days=("date", "size"), f_week_bar_count=("bar_count", "sum"),
        f_history_days=("history_days", "max"), f_daily_amplitude_mean=("daily_amplitude", "mean"),
        f_daily_amplitude_std=("daily_amplitude", "std"), f_daily_return_mean=("daily_return", "mean"),
        f_daily_return_std=("daily_return", "std"), f_early_return_mean=("early_return", "mean"),
        f_late_return_mean=("late_return", "mean"), f_daily_trend_strength=("trend_strength", "mean"),
        f_intraday_reversal_fraction=("reversal_day", "mean"), return_square=("return_square", "sum"),
        up_square=("up_square", "sum"), down_square=("down_square", "sum"),
        return_count=("return_count", "sum"), up_bar_count=("up_bar_count", "sum"),
        amount_square=("amount_square", "sum"), maximum_minute_amount=("maximum_minute_amount", "max"))
    week["f_week_days"] = week.f_observed_days
    week["f_last_close"] = week.observed_last_close
    week["f_week_return"] = week.observed_last_close / week.observed_first_open - 1
    week["f_week_range"] = (week.f_week_high - week.f_week_low) / week.observed_first_open
    counts = week.return_count.replace(0, np.nan)
    week["f_minute_return_volatility"] = np.sqrt(week.return_square / counts)
    week["f_upside_semivariance"] = week.up_square / counts
    week["f_downside_semivariance"] = week.down_square / counts
    week["f_up_minute_fraction"] = week.up_bar_count / counts
    week["f_amount_concentration"] = week.amount_square / week.f_week_amount.pow(2).replace(0, np.nan)
    week["f_largest_minute_amount_share"] = week.maximum_minute_amount / week.f_week_amount.replace(0, np.nan)
    week["f_amount_ratio_prior4w"] = week.f_week_amount / week.f_week_amount.shift(1).rolling(4, min_periods=1).mean().replace(0, np.nan)
    week["f_return_prior_week"] = week.f_week_return.shift(1)
    week["f_return_mean_prior4w"] = week.f_week_return.shift(1).rolling(4, min_periods=1).mean()
    week["f_return_std_prior4w"] = week.f_week_return.shift(1).rolling(4, min_periods=2).std()
    week["f_amplitude_mean_prior4w"] = week.f_daily_amplitude_mean.shift(1).rolling(4, min_periods=1).mean()
    week["feature_week"] = pd.DatetimeIndex(week.index)
    week["signal_datetime"] = week.observed_last_datetime
    week["history_days"] = week.f_history_days.astype(int)
    week["symbol"] = symbol

    # 先在全部分钟累积，再取固定信号点；08:00或未来14:55价格不参加当前信号。
    frame["price_volume"] = frame.close * frame.volume
    grouped = frame.groupby("date", sort=False)
    cumulative_volume = grouped.volume.cumsum()
    cumulative_amount = grouped.amount.cumsum()
    cumulative_vwap = grouped.price_volume.cumsum() / cumulative_volume.replace(0, np.nan)
    cumulative_high = grouped.high.cummax()
    cumulative_low = grouped.low.cummin()
    day_open = frame.date.map(daily.open)
    frame["f_open_return"] = frame.close / day_open - 1
    frame["f_vwap_deviation"] = frame.close / cumulative_vwap - 1
    frame["f_range_position"] = (frame.close - cumulative_low) / (cumulative_high - cumulative_low).replace(0, np.nan)
    frame["f_known_amplitude"] = (cumulative_high - cumulative_low) / day_open
    frame["f_realized_volatility"] = np.sqrt(grouped.return_square.cumsum())
    frame["f_return_5m"] = frame.close / grouped.close.shift(5) - 1
    frame["f_return_15m"] = frame.close / grouped.close.shift(15) - 1
    frame["f_cumulative_amount_log"] = np.log1p(cumulative_amount)
    frame["f_cumulative_volume_log"] = np.log1p(cumulative_volume)
    frame["f_current_volume_ratio"] = frame.volume / grouped.volume.transform(lambda values: values.rolling(20, min_periods=5).mean()).replace(0, np.nan)
    frame["f_amount_vs_previous_day"] = cumulative_amount / frame.date.map(daily.amount.shift(1)).replace(0, np.nan)
    frame["f_previous_day_return"] = frame.date.map(daily.daily_return.shift(1))
    frame["f_previous_day_amplitude"] = frame.date.map(daily.daily_amplitude.shift(1))
    frame["f_history_days"] = frame.date.map(daily.history_days - 1)
    minute = frame.datetime.dt.hour * 60 + frame.datetime.dt.minute
    frame["f_session_fraction"] = (minute - 570 - np.where(minute >= 781, 90, 0)) / 240
    offsets = [pd.Timedelta(hours=int(value[:2]), minutes=int(value[3:])) for value in signal_times]
    signal_clock = pd.DatetimeIndex(np.concatenate([(dates + offset).to_numpy() for offset in offsets])).sort_values()
    feature_columns = [column for column in frame if column.startswith("f_")]
    selected = frame.set_index("datetime").reindex(signal_clock)
    selected = selected.loc[selected.close.notna()].copy()
    intraday = selected[["date", *feature_columns]].copy()
    intraday["datetime"] = selected.index
    intraday["symbol"] = symbol
    intraday["trade_week"] = calculate_week_start(selected.index)
    intraday["previous_close"] = selected.date.map(daily.previous_close)
    intraday["signal_price"] = selected.close
    intraday["signal_high"] = selected.high
    intraday["signal_low"] = selected.low
    intraday["signal_amount"] = selected.amount
    intraday["signal_volume"] = selected.volume
    intraday["signal_bar_tradable"] = (selected.high > selected.low) & (selected.volume > 0)
    intraday["history_days"] = intraday.f_history_days
    for target, source_column in (("f_previous_week_return", "f_week_return"),
                                  ("f_previous_week_amplitude", "f_daily_amplitude_mean"),
                                  ("f_previous_week_amount", "f_week_amount")):
        intraday[target] = intraday.trade_week.map(week[source_column].shift(1))
    for prefix, timestamps in (("execution", selected.index + pd.Timedelta(minutes=1)),
                                ("recovery", selected.index.normalize() + pd.Timedelta(hours=14, minutes=55))):
        exact = indexed.reindex(timestamps)
        intraday[prefix + "_time"] = timestamps
        intraday[prefix + "_price"] = exact.close.to_numpy()
        for column in ("high", "low", "amount", "volume"):
            intraday[prefix + "_" + column] = exact[column].to_numpy()
    intraday["label_start"] = intraday.execution_time
    intraday["label_end"] = intraday.recovery_time
    intraday["target_return"] = intraday.recovery_price / intraday.execution_price - 1
    week = week[["symbol", "feature_week", "signal_datetime", "observed_last_datetime", "history_days",
                 *[column for column in week if column.startswith("f_")]]].reset_index(drop=True)
    for table in (week, intraday):
        columns = [column for column in table if column.startswith("f_")]
        table[columns] = table[columns].replace([np.inf, -np.inf], np.nan).astype("float32")
    return week, intraday.reset_index(drop=True), daily.reset_index(drop=True)


def build_weekly_symbol_cache(task: tuple) -> str:
    """执行单股缓存任务(symbol,文件列表,日期,时钟,输出目录)，已有完整pkl直接复用。"""
    symbol, paths, start, end, signal_times, directory = task
    path = Path(directory) / f"{symbol}.pkl"
    if path.exists():
        return str(path)
    raw, audit = load_weekly_symbol_bars(paths, start, end)
    weekly, intraday, daily = build_symbol_weekly_features(raw, symbol, signal_times)
    write_weekly_cache({"weekly": weekly, "intraday": intraday, "daily": daily,
                        "audit": {"symbol": symbol, **audit}}, path)
    return str(path)


def attach_weekly_calendar_labels(weekly: pd.DataFrame, intraday: pd.DataFrame,
                                  daily: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """以全市场交易日连接相邻有交易周，补充标签和当前截面状态，不补未来成交价。

    历史节假日短周仍完整，无交易自然周跳过。最后观测周需
    known_complete_end_date确认完整；否则该周训练/回测标签设为缺失。不存在未来
    周的最终feature_week保留推断行，trade_week暂指下一自然周，实际开闭日期未知。
    """
    days = pd.DatetimeIndex(sorted(daily.date.unique()))
    calendar = pd.DataFrame({"date": days, "week": calculate_week_start(days)})
    week_dates = calendar.groupby("week").agg(entry_date=("date", "min"), exit_date=("date", "max"), market_days=("date", "size"))
    weeks = pd.DatetimeIndex(week_dates.index)
    complete = pd.Series(True, index=weeks)
    complete.iloc[-1] = bool(cfg.get("known_complete_end_date", False))
    next_week = pd.Series(weeks[1:], index=weeks[:-1])
    result = weekly.copy()
    result["trade_week"] = result.feature_week.map(next_week)
    unavailable = result.trade_week.isna()
    result.loc[unavailable, "trade_week"] = result.loc[unavailable, "feature_week"] + pd.Timedelta(days=7)
    result["future_week_observed"] = ~unavailable
    result["entry_date"] = result.trade_week.map(week_dates.entry_date)
    result["exit_date"] = result.trade_week.map(week_dates.exit_date)
    result["label_start"] = result.entry_date + pd.Timedelta(hours=9, minutes=31)
    result["label_end"] = result.exit_date + pd.Timedelta(hours=15)
    result["signal_datetime"] = result.feature_week.map(week_dates.exit_date) + pd.Timedelta(hours=15)
    result["feature_week_complete"] = result.feature_week.map(complete).eq(True)
    result["trade_week_complete"] = result.trade_week.map(complete).eq(True)
    result["f_week_days"] = result.feature_week.map(week_dates.market_days).astype("float32")
    result["f_avg_daily_amount"] = (result.f_week_amount / result.f_week_days).astype("float32")
    indexed = daily.set_index(["date", "symbol"])
    entry_keys = pd.MultiIndex.from_arrays([result.entry_date, result.symbol], names=["date", "symbol"])
    exit_keys = pd.MultiIndex.from_arrays([result.exit_date, result.symbol], names=["date", "symbol"])
    result["entry_price"] = indexed.open.reindex(entry_keys).to_numpy()
    result["exit_price"] = indexed.close.reindex(exit_keys).to_numpy()
    result["target_return"] = result.exit_price / result.entry_price - 1
    amplitude = daily.groupby(["trade_week", "symbol"]).daily_amplitude.agg(["mean", "count"])
    target_keys = pd.MultiIndex.from_arrays([result.trade_week, result.symbol], names=["trade_week", "symbol"])
    target_amplitude = amplitude.reindex(target_keys)
    result["target_amplitude"] = target_amplitude["mean"].to_numpy()
    expected_days = result.trade_week.map(week_dates.market_days)
    complete_amplitude = target_amplitude["count"].to_numpy() == expected_days.to_numpy()
    result.loc[~complete_amplitude, "target_amplitude"] = np.nan
    result.loc[~result.trade_week_complete, ["target_return", "target_amplitude"]] = np.nan
    result["label_available"] = np.isfinite(result.target_return) & np.isfinite(result.target_amplitude) & result.feature_week_complete
    intraday = intraday.copy()
    intraday["trade_week_complete"] = intraday.trade_week.map(complete).eq(True)
    intraday.loc[~intraday.trade_week_complete, "target_return"] = np.nan
    intraday["f_market_return"] = intraday.groupby("datetime").f_open_return.transform("mean").astype("float32")
    intraday["f_cs_return_rank"] = intraday.groupby("datetime").f_open_return.rank(pct=True).astype("float32")
    intraday["cross_section_count"] = intraday.groupby("datetime").symbol.transform("size")
    daily = daily.copy()
    daily["trade_week_complete"] = daily.trade_week.map(complete).eq(True)
    calendar_metadata = {"trading_days": len(days), "market_weeks": len(weeks),
                         "first_date": str(days.min().date()), "last_date": str(days.max().date()),
                         "last_week": str(weeks[-1].date()), "last_week_complete": bool(complete.iloc[-1]),
                         "known_complete_end_date": bool(cfg.get("known_complete_end_date", False)),
                         "calendar_rule": "全市场有行情日决定历史交易周；无交易自然周跳过；最后周完整性由已核验节假日配置确认"}
    return result, intraday, daily, calendar_metadata


def load_weekly_t_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """按cfg构建/复用全主板周特征、日内T样本、日执行表与可核验来源manifest。

    cfg包含data_root/start_date/end_date/cache_dir/workers/max_symbols/signal_times和
    known_complete_end_date。不读取名称快照；min_history_days/min_avg_daily_amount/
    min_price的筛选由调用者按f_history_days/f_week_amount/f_week_days/f_last_close进行。
    首次生成分股pkl及汇总三表，后续同指纹加载；workers不改变缓存内容。
    """
    project_root = Path(__file__).resolve().parents[1]
    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = project_root / root
    start, end = str(cfg.get("start_date", "2023-01-01")), str(cfg.get("end_date", "2026-04-30"))
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start_date不能晚于end_date")
    signal_times = list(cfg.get("signal_times", DEFAULT_SIGNAL_TIMES))
    if not signal_times or len(set(signal_times)) != len(signal_times):
        raise ValueError("signal_times须非空且不重复")
    for value in signal_times:
        timestamp = pd.Timestamp(f"2000-01-01 {value}")
        minute = timestamp.hour * 60 + timestamp.minute
        if not (571 <= minute < 690 or 781 <= minute < 894):
            raise ValueError("T信号须在09:31–11:29或13:01–14:53，下一分钟成交且早于14:55恢复")
    paths = sorted(path for path in root.glob("[0-9][0-9][0-9][0-9]/*.csv")
                   if path.stem.startswith(MAINBOARD_PREFIXES) and pd.Timestamp(start).year <= int(path.parent.name) <= pd.Timestamp(end).year)
    symbols = sorted({path.stem for path in paths})
    max_symbols = int(cfg.get("max_symbols", 0))
    if max_symbols < 0:
        raise ValueError("max_symbols不能为负数")
    if max_symbols:
        symbols = symbols[:max_symbols]
    allowed = set(symbols)
    paths = [path for path in paths if path.stem in allowed]
    if not paths:
        raise ValueError("指定日期范围内没有沪深主板分钟CSV")
    stats = [[str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]
    fingerprint = {"schema": SCHEMA_VERSION, "data_root": str(root.resolve()), "files": stats, "symbols": symbols,
                   "start_date": start, "end_date": end, "signal_times": signal_times,
                   "known_complete_end_date": bool(cfg.get("known_complete_end_date", False)),
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "universe_rule": "所有年份主板代码文件并集，不使用当前名称/ST或未来存续过滤",
                   "price_adjustment": "原始CSV价格，未新增复权、分红或公司行动处理"}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    cache_root = Path(cfg.get("cache_dir", project_root / "reports/weekly_t_cross_validation/cache"))
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    cache = cache_root / f"weekly_t_{key}"
    cache.mkdir(parents=True, exist_ok=True)
    completed = [cache / f"{name}.pkl" for name in ("weekly", "intraday", "daily")]
    metadata_path = cache / "metadata.json"
    if all(path.exists() for path in completed) and metadata_path.exists():
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
        tables = [pd.read_pickle(path) for path in completed]
        for name, table in zip(("weekly", "intraday", "daily"), tables):
            if len(table) != manifest[name + "_rows"]:
                raise ValueError(f"{name}缓存行数与元数据不一致")
        print(f"复用周/T缓存：{cache}；周{len(tables[0]):,}行、日内{len(tables[1]):,}行", flush=True)
        return *tables, {**manifest, "cache_hit": True}
    files_by_symbol = {symbol: [] for symbol in symbols}
    for path in paths:
        files_by_symbol[path.stem].append(str(path))
    tasks = [(symbol, files_by_symbol[symbol], start, end, signal_times, str(cache)) for symbol in symbols]
    workers = max(1, int(cfg.get("workers", 4)))
    print(f"构建全主板周/T缓存：{len(symbols)}股，{workers}进程；每股全部分钟参与聚合", flush=True)
    results = []
    if workers == 1:
        iterator = map(build_weekly_symbol_cache, tasks)
        for index, path in enumerate(iterator, 1):
            results.append(path)
            if index % 100 == 0 or index == len(tasks):
                print(f"单股周/T缓存 {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, path in enumerate(pool.map(build_weekly_symbol_cache, tasks), 1):
                results.append(path)
                if index % 100 == 0 or index == len(tasks):
                    print(f"单股周/T缓存 {index}/{len(tasks)}", flush=True)
    chunks = {name: [] for name in ("weekly", "intraday", "daily")}
    audits = []
    for path in results:
        payload = pd.read_pickle(path)
        audits.append(payload["audit"])
        for name in chunks:
            if len(payload[name]) or (len(payload["daily"]) and not chunks[name]):
                chunks[name].append(payload[name])
    if not chunks["daily"]:
        raise ValueError("日期范围内没有合法分钟行情")
    weekly = pd.concat(chunks["weekly"], ignore_index=True)
    intraday = pd.concat(chunks["intraday"], ignore_index=True)
    daily = pd.concat(chunks["daily"], ignore_index=True)
    del chunks
    weekly, intraday, daily, calendar = attach_weekly_calendar_labels(weekly, intraday, daily, cfg)
    weekly = weekly.sort_values(["feature_week", "symbol"]).reset_index(drop=True)
    intraday = intraday.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    daily = daily.sort_values(["date", "symbol"]).reset_index(drop=True)
    manifest = {**fingerprint, "fingerprint": key, "cache_path": str(cache), "cache_hit": False,
                "weekly_rows": len(weekly), "intraday_rows": len(intraday), "daily_rows": len(daily),
                "symbol_count": daily.symbol.nunique(), "calendar": calendar,
                "weekly_features": [column for column in weekly if column.startswith("f_")],
                "intraday_features": [column for column in intraday if column.startswith("f_")],
                "inference_policy": "未来执行/恢复价格和标签缺失不删除当前样本；f_*不包含未来价格",
                "weekly_label": "下一有交易周首日09:31bar open至末日15:00bar close收益；日振幅=(high-low)/dayopen的周内均值",
                "intraday_label": "同日14:55bar close / 信号下一分钟bar close - 1"}
    for name, table in (("weekly", weekly), ("intraday", intraday), ("daily", daily)):
        write_weekly_cache(table, cache / f"{name}.pkl")
    pd.DataFrame(audits).to_csv(cache / "data_quality.csv", index=False)
    write_weekly_cache(manifest, metadata_path)
    print(f"周/T缓存完成：周{len(weekly):,}行、日内{len(intraday):,}行、日{len(daily):,}行；{cache}", flush=True)
    return weekly, intraday, daily, manifest
