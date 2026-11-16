"""以当前已知时钟及周参考价过滤核心 OOF 做 T 训练域，不判断未来成交。

输入为 v6 已扩展真实来源的分钟队列、同股周资格表及配置；返回原样行的副本、
逐行过滤审计和可持久化元数据。无文件写入，不拟合模型，不改特征、标签或源表。
静态初始资金槽位只模拟计划整手数；动态权益、容量、现金及 T+1 仍由引擎判断。
"""

from __future__ import annotations

import hashlib
from numbers import Real
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .weekly_t_short_horizon import planned_recovery_times


DOMAIN_NAME = "known_slot_and_recovery_deadline"
MINUTE_COLUMNS = ("datetime", "symbol", "trade_week", "feature_source_start")
WEEKLY_COLUMNS = ("trade_week", "symbol", "f_last_close", "signal_datetime", "feature_source_start")


def validate_training_domain_config(cfg: dict, horizon: int) -> dict:
    """检查过滤所消费的配置，保留引擎整手公式，禁止默默截断整数或改变时钟。"""
    if not isinstance(cfg, dict) or cfg.get("t_training_domain") != DOMAIN_NAME:
        raise ValueError(f"t_training_domain 必须为 {DOMAIN_NAME}")
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 240:
        raise ValueError("horizon 必须为 1 至 240 的整数交易分钟")
    resolved = {"t_training_domain": DOMAIN_NAME, "horizon_minutes": int(horizon)}
    for key in ("initial_cash", "base_fraction", "short_t_fraction", "slippage_rate"):
        value = cfg.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{key} 必须为有限数值")
        resolved[key] = float(value)
    if resolved["initial_cash"] <= 0 or not 0 < resolved["base_fraction"] <= 1 or not 0 < resolved["short_t_fraction"] <= 1:
        raise ValueError("initial_cash 必须为正，base_fraction/short_t_fraction 必须处于 (0, 1]")
    if resolved["slippage_rate"] < 0:
        raise ValueError("slippage_rate 必须非负")
    for key in ("top_n", "lot_size"):
        value = cfg.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} 必须为正整数")
        resolved[key] = int(value)
    if resolved["lot_size"] < 100 or resolved["lot_size"] % 100:
        raise ValueError("lot_size 必须为 100 的正整数倍")
    if "weekly_top_n" in cfg and cfg["weekly_top_n"] != resolved["top_n"]:
        raise ValueError("weekly_top_n 必须与 top_n 相同")
    clock = cfg.get("latest_planned_recovery_time")
    if not isinstance(clock, str) or re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", clock) is None:
        raise ValueError("latest_planned_recovery_time 必须使用 HH:MM 格式")
    hour, minute = map(int, clock.split(":"))
    clock_minutes = hour * 60 + minute
    if not (571 <= clock_minutes <= 690 or 781 <= clock_minutes <= 900):
        raise ValueError("latest_planned_recovery_time 必须是合法交易分钟")
    resolved["latest_planned_recovery_time"] = clock
    return resolved


def prepare_training_domain_inputs(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> pd.DataFrame:
    """仅复制确实消费的当前列；规范无时区时间并拒绝非法股周键，不访问未来列。"""
    if not isinstance(frame, pd.DataFrame) or not set(required).issubset(frame.columns):
        raise ValueError(f"{name} 缺少必需列 {required}")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} 列名重复")
    result = frame.loc[:, list(required)].copy(deep=True).reset_index(drop=True)
    if not result.symbol.map(lambda value: isinstance(value, str) and bool(value.strip()) and value == value.strip()).all():
        raise ValueError(f"{name}.symbol 必须为非空且无首尾空格的字符串")
    for column in set(required) - {"symbol", "f_last_close"}:
        try:
            values = pd.to_datetime(result[column], errors="raise")
            index = pd.DatetimeIndex(values)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}.{column} 不是合法日期时间") from exc
        if index.tz is not None or index.hasnans:
            raise ValueError(f"{name}.{column} 必须非缺失且无时区")
        result[column] = values
    if not (result.trade_week.eq(result.trade_week.dt.normalize()) & result.trade_week.dt.dayofweek.eq(0)).all():
        raise ValueError(f"{name}.trade_week 必须为周一零点")
    keys = ["datetime", "symbol"] if "datetime" in required else ["trade_week", "symbol"]
    if result.duplicated(keys).any():
        raise ValueError(f"{name} 的 {keys} 重复")
    return result


def calculate_training_domain_hash(frame: pd.DataFrame) -> str:
    """摘要仅包含传入的当前输入/审计列及原索引，不读取未来标签或执行行情。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def filter_core_t_training_domain(frame: pd.DataFrame, weekly_universe: pd.DataFrame,
                                  horizon: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按已知计划时钟和初始槽位整手过滤每个周期的核心 OOF 队列，返回副本/审计/元数据。

frame 须含 datetime/symbol/trade_week/feature_source_start；周表须另含 f_last_close、
signal_datetime。已有分钟最早来源必须已经覆盖对应周表来源，本函数不修改该来源。
只按下一交易分钟加 horizon 算同日恢复期限；不读已有 planned_recovery_time、
execution/recovery/label 列及其有效性。过滤后仍须执行原严格核心年标签与来源隔离。
审计与全部输入行一一对应且保留原索引；保留行的所有原字段和 attrs 完全不变。
"""
    resolved = validate_training_domain_config(cfg, horizon)
    minutes = prepare_training_domain_inputs(frame, MINUTE_COLUMNS, "frame")
    weeks = prepare_training_domain_inputs(weekly_universe, WEEKLY_COLUMNS, "weekly_universe")
    try:
        prices = pd.to_numeric(weeks.f_last_close, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("weekly_universe.f_last_close 必须为有限正价格") from exc
    if weeks.f_last_close.map(lambda value: isinstance(value, (bool, np.bool_))).any() or not (np.isfinite(prices) & prices.gt(0)).all():
        raise ValueError("weekly_universe.f_last_close 必须为有限正价格")
    weeks["f_last_close"] = prices.astype(np.float64)
    if not (weeks.feature_source_start.le(weeks.signal_datetime) & weeks.signal_datetime.lt(weeks.trade_week)).all():
        raise ValueError("周价格来源必须早于当周，且不得晚于周信号")
    if not (minutes.datetime.ge(minutes.trade_week) & minutes.datetime.lt(minutes.trade_week + pd.Timedelta(days=7))
            & minutes.feature_source_start.le(minutes.datetime)).all():
        raise ValueError("分钟不属于对应股周，或其真实来源晚于当前信号")
    weeks = weeks.rename(columns={"f_last_close": "universe_previous_close",
                                  "feature_source_start": "weekly_reference_source_start",
                                  "signal_datetime": "weekly_reference_signal_datetime"})
    joined = minutes.merge(weeks, on=["trade_week", "symbol"], how="left", sort=False,
                           validate="many_to_one", indicator="_weekly_join")
    if not joined._weekly_join.eq("both").all():
        raise ValueError("核心分钟缺少唯一对应的周参考价格/来源")
    if not (joined.feature_source_start.le(joined.weekly_reference_source_start)
            & joined.weekly_reference_signal_datetime.lt(joined.datetime)).all():
        raise ValueError("原分钟 feature_source_start 尚未覆盖周参考来源，或周信号并非当时已知")
    joined = joined.drop(columns="_weekly_join")
    joined.index = frame.index.copy()
    known_inputs_sha = calculate_training_domain_hash(joined)
    recovery = planned_recovery_times(joined.datetime, resolved["horizon_minutes"])
    same_day = recovery.notna() & recovery.dt.normalize().eq(joined.datetime.dt.normalize())
    hour, minute = map(int, resolved["latest_planned_recovery_time"].split(":"))
    deadline = joined.datetime.dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    clock_ok = same_day & recovery.le(deadline)
    slot_budget = resolved["initial_cash"] * resolved["base_fraction"] / resolved["top_n"]
    lot = resolved["lot_size"]
    # 与 weekly_t_engine 的 requested 计划量一致：分母只有滑点，佣金在后续现金检查。
    lots = slot_budget / (joined.universe_previous_close * (1 + resolved["slippage_rate"])) / lot
    if not np.isfinite(lots).all() or (lots >= np.iinfo(np.int64).max // lot).any():
        raise ValueError("静态计划股数超出有限 int64 范围")
    base_shares = np.floor(lots).astype(np.int64) * lot
    t_shares = (np.floor(base_shares * resolved["short_t_fraction"] / lot).astype(np.int64) * lot)
    slot_ok = t_shares.ge(lot)
    keep = clock_ok & slot_ok
    reasons = np.full(len(joined), "keep", dtype=object)
    reasons[~clock_ok.to_numpy()] = "recovery_after_deadline"
    reasons[~same_day.to_numpy()] = "no_same_day_planned_recovery"
    below = ~slot_ok.to_numpy()
    reasons[below & clock_ok.to_numpy()] = "static_t_below_one_lot"
    both = below & ~clock_ok.to_numpy()
    reasons[both] = np.char.add(reasons[both].astype(str), ";static_t_below_one_lot")
    audit = joined.copy(deep=True)
    audit["horizon_minutes"] = resolved["horizon_minutes"]
    audit["planned_recovery_time"] = recovery
    audit["planned_recovery_deadline"] = deadline
    audit["same_day_planned_recovery"] = same_day
    audit["recovery_deadline_eligible"] = clock_ok
    audit["static_slot_budget"] = slot_budget
    audit["static_base_shares"] = base_shares
    audit["static_t_shares"] = t_shares
    audit["static_slot_eligible"] = slot_ok
    audit["keep"] = keep
    audit["drop_reason"] = reasons
    result = frame.iloc[np.flatnonzero(keep.to_numpy())].copy(deep=True)
    source_files = ("weekly_t_train_domain.py", "weekly_t_short_horizon.py", "weekly_t_horizon_data.py", "weekly_t_engine.py")
    metadata = {
        "schema_version": 1, "t_training_domain": DOMAIN_NAME, "config": resolved,
        "input_rows": len(frame), "kept_rows": len(result), "dropped_rows": int((~keep).sum()),
        "drop_reason_counts": {str(key): int(value) for key, value in audit.drop_reason.value_counts().sort_index().items()},
        "failed_condition_counts": {"recovery_deadline": int((~clock_ok).sum()), "static_t_below_one_lot": int((~slot_ok).sum()),
                                    "both": int((~clock_ok & ~slot_ok).sum())},
        "known_inputs_sha256": known_inputs_sha, "audit_sha256": calculate_training_domain_hash(audit),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_files},
        "formula": "slot=initial_cash*base_fraction/top_n; base=int(slot/(f_last_close*(1+slippage_rate))/lot_size)*lot_size; t=int(base*short_t_fraction/lot_size)*lot_size",
        "scope": "仅核心 OOF 训练域；当前周参考价和确定性同日时钟；不依据未来价格或标签有效性",
        "limitation": "静态初始资金计划代理，不计动态权益、实际底仓、现金、容量、限价、债务或T+1；不保证实际成交。仍须原严格核心年标签/来源过滤。",
        "feature_source_unchanged": True, "future_columns_consumed": [],
    }
    return result, audit, metadata
