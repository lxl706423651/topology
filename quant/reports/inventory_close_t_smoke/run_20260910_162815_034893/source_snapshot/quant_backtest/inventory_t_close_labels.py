"""从原盘中缓存分离同日14:55参考标签，再左联已冻结永久底仓核心机会。

本模块只处理数据接口，不训练、生成交易规则或读取文件。全部当前键均保留，
旧完整周资格不适用于永久底仓；未来端点缺价仅标记训练标签不可用。有限参考
价格不证明存在成交量、可恢复或实际组合利润。原输入、因子与来源不原地修改。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inventory_t import calculate_next_trading_minute


SIGNAL_TIMES = ("09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00")
CLOSE_CLOCK = "14:55"
IDENTITY = {"recovery_key": "close", "recovery_kind": "fixed_clock", "recovery_clock": CLOSE_CLOCK}
KEYS = ["datetime", "symbol"]
PRICE_COLUMNS = ["execution_price", "recovery_price"]
TIME_COLUMNS = ["execution_time", "recovery_time", "label_start", "label_end"]
LABEL_COLUMNS = [*KEYS, "execution_time", "execution_price", "recovery_time", "recovery_price",
                 "label_start", "label_end", "target_return", "label_available", *IDENTITY]
FUTURE_PREFIXES = ("execution_", "recovery_", "target_", "label_")


def validate_fixed_close_config(cfg: dict) -> None:
    """只支持原8个信号和统一14:55钟点；旧整数horizons配置不参与本标签身份。"""
    configured = cfg.get("signal_times")
    if not isinstance(configured, (list, tuple)) or tuple(configured) != SIGNAL_TIMES:
        raise ValueError("固定尾盘标签须沿用原8个信号时钟及其顺序")
    if cfg.get("fixed_recovery_time", CLOSE_CLOCK) != CLOSE_CLOCK:
        raise ValueError("固定尾盘参考端点必须为同日14:55，不是15:00收盘")


def validate_current_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """验证唯一当前键和原会话信号；返回标准时间键，不改其他当前列及其dtype。"""
    if frame.columns.duplicated().any() or not set(KEYS).issubset(frame):
        raise ValueError("输入须有唯一列名及datetime/symbol当前键")
    if frame[KEYS].isna().any().any():
        raise ValueError("当前键不得缺失")
    symbols = pd.unique(frame.symbol)
    if not all(isinstance(value, str) and value.strip() == value and value for value in symbols):
        raise ValueError("symbol须为非空且无首尾空白的字符串")
    keys = frame[KEYS].copy()
    times = pd.to_datetime(keys.datetime, errors="raise", format="mixed")
    if not pd.api.types.is_datetime64_any_dtype(times) or times.dt.tz is not None:
        raise ValueError("当前时刻必须是本地无时区分钟时间")
    if (times.isna().any() or times.dt.second.ne(0).any() or times.dt.microsecond.ne(0).any()
            or times.dt.nanosecond.ne(0).any() or not times.dt.strftime("%H:%M").isin(SIGNAL_TIMES).all()):
        raise ValueError("当前时刻须为原8个合法信号的精确分钟")
    keys["datetime"] = times
    if keys.duplicated(KEYS).any():
        raise ValueError("datetime/symbol当前键重复，不能按未来报价挑一条")
    return keys


def calculate_fixed_close_times(times: pd.Series) -> tuple[pd.Series, pd.Series]:
    """从当前时钟计算下一合法成交分钟和同日14:55，不查询任何未来报价。"""
    mapping = {stamp: calculate_next_trading_minute(stamp) for stamp in times.unique()}
    first = pd.to_datetime(times.map(mapping))
    end = times.dt.normalize() + pd.Timedelta(hours=14, minutes=55)
    if (first.isna().any() or not first.dt.normalize().eq(times.dt.normalize()).all()
            or not first.gt(times).all() or not end.gt(first).all()):
        raise ValueError("首腿必须在下一合法交易分钟，且严格早于同日14:55")
    return first, end


def read_reference_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """参考价格只允许正有限数或NaN；零量不在此处作为未来资格过滤。"""
    if not set(PRICE_COLUMNS).issubset(frame):
        raise ValueError("缺少原下一分钟/14:55参考价格")
    prices = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors="raise").astype("float64")
    if np.isinf(prices).any().any() or (prices.notna() & prices.le(0)).any().any():
        raise ValueError("参考价格须为正有限数或缺失NaN，不得把错误值静默当缺价")
    return prices


def validate_reference_times(frame: pd.DataFrame, first: pd.Series, end: pd.Series,
                             *, require_all: bool) -> None:
    """原缓存缺价格仍有确定时钟；错误/缺失时间身份不能被新目标静默改写。"""
    required = set(TIME_COLUMNS if require_all else ("execution_time", "recovery_time"))
    if not required.issubset(frame):
        raise ValueError("缺少原精确端点时间")
    expected = {"execution_time": first, "label_start": first, "recovery_time": end, "label_end": end}
    for name, wanted in expected.items():
        if name in frame:
            actual = pd.to_datetime(frame[name], errors="raise", format="mixed")
            if not pd.api.types.is_datetime64_any_dtype(actual) or actual.dt.tz is not None or not actual.eq(wanted).all():
                raise ValueError(f"{name}与下一合法分钟/同日14:55身份不一致")
    for name, value in IDENTITY.items():
        if name in frame and not frame[name].eq(value).all():
            raise ValueError(f"{name}不是固定尾盘标签身份")
    if "horizon_minutes" in frame:
        raise ValueError("固定钟点标签不得伪装或混入整数horizon_minutes")


def extract_fixed_close_labels(intraday: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """从已加载原缓存取标签；保持全部当前唯一键，忽略旧target及完整周资格。

    必需原execution_time/price与recovery_time/price；若含label_start/end则一并
    验证。source/当前f_*不进入标签表。cfg只消费原signal_times和14:55固定时钟。
    """
    validate_fixed_close_config(cfg)
    keys = validate_current_keys(intraday)
    first, end = calculate_fixed_close_times(keys.datetime)
    validate_reference_times(intraday, first, end, require_all=False)
    prices = read_reference_prices(intraday)
    result = keys.copy()
    result["execution_time"], result["recovery_time"] = first, end
    result["execution_price"], result["recovery_price"] = prices.execution_price, prices.recovery_price
    result["label_start"], result["label_end"] = first, end
    result["target_return"] = prices.recovery_price / prices.execution_price - 1
    result["label_available"] = prices.notna().all(axis=1)
    for name, value in IDENTITY.items():
        result[name] = value
    result = result[LABEL_COLUMNS]
    if len(result) != len(intraday):
        raise AssertionError("提取标签不得改变当前机会行数")
    available = result.label_available
    audit = {"rows": len(result), "all_current_keys_preserved": True, **IDENTITY,
             "target_definition": "same_day_14_55_reference_price/next_trading_minute_reference_price-1",
             "fixed_integer_horizon_used": False, "current_features_in_label_table": False,
             "label_available_rows": int(available.sum()), "label_unavailable_rows": int((~available).sum()),
             "missing_execution_price_rows": int(prices.execution_price.isna().sum()),
             "missing_recovery_price_rows": int(prices.recovery_price.isna().sum()),
             "legacy_week_eligibility_ignored": "trade_week_complete" in intraday,
             "legacy_target_recomputed": "target_return" in intraday,
             "future_volume_used_for_opportunity_selection": False,
             "label_is_executable_fill_guarantee": False, "target_is_portfolio_execution_net_pnl": False}
    if "trade_week_complete" in intraday:
        audit["incomplete_legacy_week_rows_retained"] = int(intraday.trade_week_complete.eq(False).sum())
    if "target_return" in intraday:
        legacy = pd.to_numeric(intraday.target_return, errors="coerce")
        audit["legacy_missing_target_with_available_prices"] = int((legacy.isna() & available).sum())
    for name in ("execution_volume", "execution_amount", "recovery_volume", "recovery_amount"):
        audit[f"finite_reference_labels_with_zero_{name}"] = (int((available & pd.to_numeric(
            intraday[name], errors="raise").eq(0)).sum()) if name in intraday else None)
    return result, audit


def attach_fixed_close_labels(core: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """冻结纯当前core后才左联标签；原index/行序/因子/source不变，缺整条标签也保留。

    本函数不能用于推断资格。core须已剥离未来字段及旧完整周资格；缺未来价格
    产生NaN目标，确定端点时间仍从当前键生成，供后续独立训练purge审计。
    """
    validate_fixed_close_config(cfg)
    forbidden = [name for name in core if name.startswith(FUTURE_PREFIXES)
                 or name in ("trade_week_complete", "horizon_minutes")]
    if forbidden:
        raise ValueError(f"标签左联前core须已剥离未来字段/旧期限资格：{forbidden}")
    keys = validate_current_keys(core)
    label_keys = validate_current_keys(labels)
    if set(labels) != set(LABEL_COLUMNS):
        raise ValueError("labels必须使用独立固定尾盘标签schema，不混入旧因子/资格或期限")
    label_first, label_end = calculate_fixed_close_times(label_keys.datetime)
    validate_reference_times(labels, label_first, label_end, require_all=True)
    prices = read_reference_prices(labels)
    expected = prices.recovery_price / prices.execution_price - 1
    actual = pd.to_numeric(labels.target_return, errors="raise").astype("float64")
    if (not np.array_equal(expected.to_numpy(), actual.to_numpy(), equal_nan=True)
            or not labels.label_available.eq(prices.notna().all(axis=1)).all()):
        raise ValueError("固定尾盘标签数值/可用标记与原精确价格不一致")
    source = prices.copy()
    source.index = pd.MultiIndex.from_frame(label_keys)
    aligned = source.reindex(pd.MultiIndex.from_frame(keys))
    result = core.copy()
    first, end = calculate_fixed_close_times(keys.datetime)
    # to_numpy保证原index可非连续/重复；不能让pandas按位置索引意外错配来源列。
    result["execution_time"], result["recovery_time"] = first.to_numpy(), end.to_numpy()
    result["execution_price"], result["recovery_price"] = aligned.execution_price.to_numpy(), aligned.recovery_price.to_numpy()
    result["label_start"], result["label_end"] = first.to_numpy(), end.to_numpy()
    result["target_return"] = result.recovery_price / result.execution_price - 1
    result["label_available"] = result[PRICE_COLUMNS].notna().all(axis=1)
    for name, value in IDENTITY.items():
        result[name] = value
    if len(result) != len(core):
        raise AssertionError("标签左联改变了已冻结核心机会行数")
    return result
