"""固定底仓股票的完整真实分钟覆盖；独立内容寻址缓存，不改旧因子或恢复缓存。

只在因果底仓计划已经冻结之后取其全部评价日行情，不按T盈利、成交成功或退出
时刻筛选行。缺分钟保留缺失；全文件SHA、日历、股票和读取代码绑定缓存身份。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .balanced_t_data import record_balanced_input
from .weekly_t_quote_store import (DenseRecoveryBarStore, QUOTE_COLUMNS,
    build_dense_recovery_store, calculate_file_sha256, validate_session_times)


def encode_full_minute_identity(value: dict) -> bytes:
    """对仅含普通JSON类型的请求身份进行稳定编码。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def validate_full_minute_prices(frame: pd.DataFrame) -> None:
    """检查真实有效OHLC的关系；实际NaN和零成交保留供撮合拒绝，不填未来值。"""
    if frame.duplicated(["datetime", "symbol"]).any():
        raise ValueError("完整分钟存在重复时间/股票，不能任意选择一条")
    for key in QUOTE_COLUMNS:
        values = pd.to_numeric(frame[key], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"完整分钟{key}有负数或无穷值")
    prices = frame[["open", "high", "low", "close"]]
    finite = np.isfinite(prices).all(axis=1)
    known = prices.loc[finite]
    tolerance = 1e-10
    if (known.high.lt(known[["open", "close", "low"]].max(axis=1) - tolerance).any()
            or known.low.gt(known[["open", "close", "high"]].min(axis=1) + tolerance).any()):
        raise ValueError("完整分钟OHLC上下界不一致")
    positive_flow = frame.volume.gt(0) | frame.amount.gt(0)
    if (finite & positive_flow & prices.le(0).any(axis=1)).any():
        raise ValueError("真实有成交的分钟价格必须为正")


def read_full_minute_inputs(root: Path, names: list[str], calendar: pd.DatetimeIndex,
                            evaluation_year: int) -> tuple[pd.DataFrame, list[dict]]:
    """规范化原始年度CSV的声明日历切片，返回实际行及每股缺失统计。"""
    pieces, per_symbol = [], []
    for symbol in names:
        path = root / str(evaluation_year) / f"{symbol}.csv"
        frame = pd.read_csv(path, usecols=["datetime", *QUOTE_COLUMNS])
        frame["datetime"] = pd.to_datetime(frame.datetime, format="mixed", errors="raise")
        if frame.datetime.isna().any() or frame.datetime.dt.tz is not None:
            raise ValueError("原始完整分钟时间不合法")
        frame = frame.loc[frame.datetime.dt.normalize().isin(calendar)].copy()
        # 与已审计weekly_t_data的收盘标签分钟合同相同：只取声明会话。
        # 原始13:00等会话外记录单独审计，不平移到13:01，也不成为成交机会。
        minute = frame.datetime.dt.hour * 60 + frame.datetime.dt.minute
        in_session = minute.between(571, 690) | minute.between(781, 900)
        outside = frame.loc[~in_session, "datetime"]
        auctions = minute.eq(570)
        auction_rows = int(auctions.sum())
        frame = frame.loc[in_session].copy()
        validate_session_times(pd.DatetimeIndex(frame.datetime))
        frame["symbol"] = symbol
        for key in QUOTE_COLUMNS:
            frame[key] = pd.to_numeric(frame[key], errors="raise")
        validate_full_minute_prices(frame)
        per_symbol.append({"symbol": symbol, "present_rows": len(frame),
            "expected_market_minute_slots": len(calendar) * 240,
            "missing_rows": len(calendar) * 240 - len(frame), "excluded_auction_rows": auction_rows,
            "excluded_outside_session_rows": len(outside),
            "excluded_clock_counts": {str(key): int(value) for key, value in outside.dt.strftime("%H:%M:%S").value_counts().items()},
            "excluded_timestamps": outside.astype(str).tolist()})
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        {"datetime": pd.Series(dtype="datetime64[ns]"), "symbol": pd.Series(dtype=object),
         **{key: pd.Series(dtype=float) for key in QUOTE_COLUMNS}})
    return frame, per_symbol


def validate_full_minute_cache_values(store: DenseRecoveryBarStore, frame: pd.DataFrame) -> None:
    """逐实际键/六列核缓存与当前完整raw内容；不能只相信可自洽重写的manifest。

利用只读稠密数组做向量核对，不调用未来查价、不生成信号、不填补缺行。
    """
    if store.row_count != len(frame) or int(np.count_nonzero(store._present)) != len(frame):
        raise ValueError("完整分钟缓存实际present行数与原始切片不同")
    if not len(frame):
        return
    times = pd.Index(store._times).get_indexer(frame.datetime.to_numpy(dtype="datetime64[ns]").view("int64"))
    symbols = pd.Index(sorted(store.symbols)).get_indexer(frame.symbol)
    if (times < 0).any() or (symbols < 0).any() or not store._present[times, symbols].all():
        raise ValueError("完整分钟缓存遗漏原始实际键")
    for index, key in enumerate(QUOTE_COLUMNS):
        expected = frame[key].to_numpy(dtype="float64")
        actual = store._quotes[times, symbols, index]
        if not np.array_equal(expected, actual, equal_nan=True):
            raise ValueError(f"完整分钟缓存{key}数值与已绑定原始切片不同")


def audit_signal_execution_coverage(intraday: pd.DataFrame, store: DenseRecoveryBarStore) -> dict:
    """核原信号表已知的下一分钟执行元数据与新raw store一致；不据未来值删信号。

原执行时刻/字段缺失仍保留原缺失，不能借新缓存改写入场资格。数值对比容许原
CSV解析与持久化产生的1e-10绝对误差，不容许缺实际行或不一致的有效价量。
    """
    mapping = {"execution_price": "close", "execution_high": "high", "execution_low": "low",
               "execution_volume": "volume", "execution_amount": "amount"}
    required = {"symbol", "execution_time", *mapping}
    if not required.issubset(intraday):
        raise ValueError("原信号表缺少执行元数据，无法核对新完整报价")
    times = pd.to_datetime(intraday.execution_time)
    current = intraday.loc[times.notna()].copy()
    if current.empty:
        return {"input_rows": len(intraday), "verified_execution_rows": 0,
                "missing_original_execution_times_retained": int(times.isna().sum()), "no_signal_rows_filtered": True}
    time_indices = pd.Index(store._times).get_indexer(pd.to_datetime(current.execution_time).to_numpy(dtype="datetime64[ns]").view("int64"))
    symbol_indices = pd.Index(sorted(store.symbols)).get_indexer(current.symbol)
    valid_indices = (time_indices >= 0) & (symbol_indices >= 0)
    present = np.zeros(len(current), dtype=bool)
    present[valid_indices] = store._present[time_indices[valid_indices], symbol_indices[valid_indices]].astype(bool)
    verified = np.zeros(len(current), dtype=bool)
    for source, target in mapping.items():
        expected = pd.to_numeric(current[source], errors="raise").to_numpy(dtype=float)
        known = np.isfinite(expected)
        if (known & ~present).any():
            raise ValueError(f"原{source}有报价但新完整分钟缺对应实际行")
        actual = np.full(len(current), np.nan)
        actual[present] = store._quotes[time_indices[present], symbol_indices[present], QUOTE_COLUMNS.index(target)]
        if not np.isclose(expected[known], actual[known], rtol=0, atol=1e-10, equal_nan=False).all():
            raise ValueError(f"原{source}与新完整分钟当前对应报价不一致")
        verified |= known
    return {"input_rows": len(intraday), "verified_execution_rows": int(verified.sum()),
            "missing_original_execution_times_retained": int(times.isna().sum()),
            "original_execution_rows_without_known_values": int((~verified).sum()),
            "all_declared_execution_values_match_raw": True, "no_signal_rows_filtered": True}


def load_full_minute_store(cfg: dict, evaluation_year: int, symbols: set | list,
                           days: pd.DatetimeIndex, inputs: dict,
                           audit_directory: Path) -> tuple[DenseRecoveryBarStore, dict]:
    """构建或核验复用所持股票的完整分钟mmap，返回只读store和来源/缺失审计。

cfg读取data_root/full_minute_cache_dir，价格默认CSV解析器与旧行情协议一致。
inputs写入完整raw及生成缓存/请求清单身份。缺失原文件报错，缺失分钟不补齐。
实际交易日来自调用方完整市场日历，不能由已持股票的报价行数反推。
    """
    calendar = pd.DatetimeIndex(days).normalize().unique().sort_values()
    if calendar.empty or calendar.hasnans or calendar.tz is not None or not (calendar.year == evaluation_year).all():
        raise ValueError("完整分钟必须有本年非空、无时区市场日历")
    names = sorted(set(str(value) for value in symbols))
    if any(not value.strip() or value == "__MARKET_CALENDAR__" for value in names):
        raise ValueError("完整分钟股票代码非法")
    root = Path(cfg["data_root"]).resolve()
    raw_sources = {}
    for symbol in names:
        path = root / str(evaluation_year) / f"{symbol}.csv"
        if not path.is_file():
            raise ValueError(f"所持股票原始全年分钟文件缺失：{path}")
        record_balanced_input(path, inputs)
        raw_sources[str(path)] = dict(inputs[str(path)])
    identity = {"schema_version": 1, "year": int(evaluation_year), "symbols": names,
        "market_days": [str(value.date()) for value in calendar], "raw_sources": raw_sources,
        "reader_sha256": calculate_file_sha256(Path(__file__)), "pandas_version": pd.__version__,
        "csv_float_parser": "pandas_default_same_as_original_market_cache",
        "session": "09:31-11:30/13:01-15:00; present raw rows only; no interpolation"}
    fingerprint = hashlib.sha256(encode_full_minute_identity(identity)).hexdigest()
    cache_root = Path(cfg["full_minute_cache_dir"]).resolve() / str(evaluation_year)
    manifest_path = cache_root / f"full_minutes_{fingerprint}.json"
    audit_directory.mkdir(parents=True, exist_ok=True)
    frame, per_symbol = read_full_minute_inputs(root, names, calendar, evaluation_year)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity or manifest.get("request_fingerprint") != fingerprint:
            raise ValueError("完整分钟缓存请求身份不一致，不覆盖旧缓存")
        metadata_path = Path(manifest["dense_cache_path"]) / "metadata.json"
        if calculate_file_sha256(metadata_path) != manifest["dense_metadata_sha256"]:
            raise ValueError("完整分钟缓存元数据与请求清单不一致")
        store = DenseRecoveryBarStore(manifest["dense_cache_path"], verify_integrity=True)
        audit = {**manifest["coverage_audit"], "cache_hit": True}
    else:
        store = build_dense_recovery_store(frame, cache_root / "quotes")
        audit = {"cache_hit": False, "symbols": names, "market_days": len(calendar),
            "expected_market_minute_slots": len(names) * len(calendar) * 240,
            "present_rows": len(frame), "missing_rows": len(names) * len(calendar) * 240 - len(frame),
            "per_symbol": per_symbol, "observation_and_execution_share_same_coverage": True,
            "missing_rows_not_filled": True, "whole_holding_period_kept": True}
        # 构建期间原文件必须仍相同；新增缓存不会修改raw。
        for name, record in raw_sources.items():
            if calculate_file_sha256(Path(name)) != record["sha256"]:
                raise ValueError(f"构建完整分钟期间原始文件改变：{name}")
        manifest = {"identity": identity, "request_fingerprint": fingerprint,
            "dense_cache_path": str(store.cache_path),
            "dense_metadata_sha256": calculate_file_sha256(store.cache_path / "metadata.json"),
            "coverage_audit": audit}
        cache_root.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".full_minutes_", dir=cache_root)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(encode_full_minute_identity(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            # 每次运行串行准备上下文；已有清单绝不因损坏而自动覆盖。
            if manifest_path.exists():
                raise ValueError("完整分钟请求清单在构建期间被并发创建")
            os.link(temporary, manifest_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    validate_full_minute_cache_values(store, frame)
    if (audit["symbols"] != names or audit["market_days"] != len(calendar)
            or audit["expected_market_minute_slots"] != len(names) * len(calendar) * 240
            or audit["present_rows"] != len(frame)
            or audit["missing_rows"] != len(names) * len(calendar) * 240 - len(frame)
            or audit["per_symbol"] != per_symbol):
        raise ValueError("完整分钟缓存覆盖审计与实际原始切片不同")
    for name, record in raw_sources.items():
        if calculate_file_sha256(Path(name)) != record["sha256"]:
            raise ValueError(f"读取完整分钟期间原始文件改变：{name}")
    record_balanced_input(manifest_path, inputs)
    record_balanced_input(store.cache_path / "metadata.json", inputs)
    if not store.symbols.issubset(set(names)) or store.row_count != audit["present_rows"]:
        raise ValueError("完整分钟缓存股票/行数与请求审计不同")
    audit = {**audit, "manifest_path": str(manifest_path), "request_fingerprint": fingerprint,
             "dense_cache_path": str(store.cache_path), "raw_sources": raw_sources}
    (audit_directory / "full_minute_coverage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return store, audit
