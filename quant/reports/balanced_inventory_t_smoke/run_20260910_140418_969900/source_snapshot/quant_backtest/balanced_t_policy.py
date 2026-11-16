"""等金额固定底仓做T的透明择时对照，只使用已完成信号分钟。

按配置预先声明VWAP反转、确认反转、顺势规则及恢复期限/旧股比例；不读取未来
端点、不复用旧T模型概率、不把VWAP偏离称为预期利润。候选只由内部实际账户
选择；本模块不读写市场缓存、模型或报告。
"""
from __future__ import annotations

from itertools import product
from numbers import Real

import numpy as np
import pandas as pd

from .weekly_t_features import select_recoverable_signal_sides
from .weekly_t_short_horizon import planned_recovery_times


VERSION = "balanced_inventory_t_v12"
CURRENT_FEATURES = ["f_vwap_deviation", "f_return_5m", "f_return_15m", "f_open_return"]
DECISION_COLUMNS = ["datetime", "symbol", "action", "fraction", "priority",
                    "planned_recovery_time", "candidate_index", "signal_strength", "regime"]


def build_balanced_candidates(cfg: dict) -> list[dict]:
    """以YAML列表的稳定顺序声明全部候选；参数在任何内部或外层评价前保存。"""
    families = cfg.get("signal_families", ["vwap_reversal", "confirmed_reversal", "momentum"])
    levels = cfg.get("entry_deviations", [.003, .006, .01])
    horizons = cfg.get("recovery_horizons", [20, 60, "close"])
    fractions = cfg.get("t_trade_fractions", [.5, 1.])
    allowed = {"vwap_reversal", "confirmed_reversal", "momentum"}
    for name, values in [("signal_families", families), ("entry_deviations", levels),
                         ("recovery_horizons", horizons), ("t_trade_fractions", fractions)]:
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name}须为非空、无重复列表")
    if not set(families).issubset(allowed):
        raise ValueError("不支持的当前信号规则")
    for value in levels:
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("entry_deviations须为(0,1)内有限数")
    for value in fractions:
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("t_trade_fractions须为(0,1]内有限数")
    if any(isinstance(h, bool) or h not in (20, 40, 60, "close") for h in horizons):
        raise ValueError("恢复期限只支持20/40/60交易分钟及close")
    candidates = [{"candidate_index": 0, "enabled": False, "family": "hold",
                   "entry_deviation": 0., "recovery_horizon": "close", "fraction": 0.}]
    for family, level, horizon, fraction in product(families, levels, horizons, fractions):
        candidates.append({"candidate_index": len(candidates), "enabled": True,
                           "family": family, "entry_deviation": float(level),
                           "recovery_horizon": horizon, "fraction": float(fraction)})
    return candidates


def create_empty_balanced_decisions() -> pd.DataFrame:
    """返回可交给引擎的有类型空表；hold不会伪造成交或盈利。"""
    dates = {"datetime", "planned_recovery_time"}
    strings = {"symbol", "regime"}
    integers = {"action", "candidate_index"}
    return pd.DataFrame({key: pd.Series(dtype="datetime64[ns]" if key in dates else
                                       "object" if key in strings else "int64" if key in integers else "float64")
                         for key in DECISION_COLUMNS})


def generate_balanced_decisions(intraday: pd.DataFrame, rule: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """用当刻VWAP偏离、5/15分钟和开盘方向生成首腿，保留当天后续重试机会。

执行时间由引擎确定为下一交易分钟；恢复时间只由当前时钟及已冻结期限确定。
同刻有限现金按偏离绝对值降序、代码升序分配，不按未来收益排队。交易股数及
日内重复次数只由引擎的日初旧股余额决定，当前规则不根据未来成交缺失删样本。
    """
    candidates = build_balanced_candidates(cfg)
    index = rule.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(candidates):
        raise ValueError("candidate_index不在预先声明列表")
    if rule != candidates[int(index)]:
        raise ValueError("规则与预先声明配置不一致")
    if intraday.columns.duplicated().any() or not {"datetime", "symbol"}.issubset(intraday):
        raise ValueError("输入须有唯一datetime/symbol列")
    frame = intraday.copy()
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
    if frame.datetime.isna().any() or frame.datetime.dt.tz is not None or frame.symbol.isna().any():
        raise ValueError("信号键不能缺失，时间为中国市场本地无时区分钟")
    if frame.duplicated(["datetime", "symbol"]).any():
        raise ValueError("同一datetime/symbol不能重复")
    audit = {"candidate_index": int(index), "input_rows": len(frame), "enabled": rule["enabled"],
             "uses_future_endpoints": False, "signal_strength_is_expected_return": False}
    if not rule["enabled"]:
        return create_empty_balanced_decisions(), {**audit, "request_rows": 0}
    required = {"f_vwap_deviation", "signal_amount", "signal_volume"}
    if rule["family"] == "confirmed_reversal":
        required.add("f_return_5m")
    elif rule["family"] == "momentum":
        required.update(CURRENT_FEATURES)
    if not required.issubset(frame):
        raise ValueError(f"当前规则缺少字段{sorted(required.difference(frame))}")
    values = {key: pd.to_numeric(frame[key], errors="raise") for key in required}
    valid = pd.Series(True, index=frame.index)
    for value in values.values():
        valid &= np.isfinite(value)
    valid &= values["signal_amount"].gt(0) & values["signal_volume"].gt(0)
    deviation = values["f_vwap_deviation"]
    sign = np.sign(deviation)
    action = -sign if rule["family"] != "momentum" else sign
    strength = deviation.abs()
    current = valid & strength.ge(rule["entry_deviation"])
    audit["finite_liquid_current_rows"] = int(valid.sum())
    audit["deviation_qualified_rows"] = int(current.sum())
    if rule["family"] == "confirmed_reversal":
        current &= (values["f_return_5m"] * action).gt(0)
    elif rule["family"] == "momentum":
        for key in ("f_return_5m", "f_return_15m", "f_open_return"):
            current &= (values[key] * action).gt(0)
    audit["direction_qualified_rows"] = int(current.sum())
    current &= select_recoverable_signal_sides(frame, action, cfg)
    audit["current_exit_side_qualified_rows"] = int(current.sum())
    day = frame.datetime.dt.normalize()
    close_offset = pd.to_timedelta(str(cfg.get("fixed_recovery_time", "14:55")) + ":00")
    planned = (day + close_offset if rule["recovery_horizon"] == "close" else
               planned_recovery_times(frame.datetime, int(rule["recovery_horizon"])))
    last_entry = pd.to_timedelta(str(cfg.get("last_entry_time", "14:30")) + ":00")
    current &= frame.datetime.le(day + last_entry)
    current &= planned.notna() & planned.dt.normalize().eq(day) & planned.le(day + close_offset)
    current &= planned.gt(frame.datetime + pd.Timedelta(minutes=1))
    chosen = frame.loc[current, ["datetime", "symbol"]].copy()
    chosen["action"] = action.loc[current].astype("int64")
    chosen["fraction"] = rule["fraction"]
    chosen["priority"] = strength.loc[current]
    chosen["planned_recovery_time"] = planned.loc[current]
    chosen["candidate_index"] = int(index)
    chosen["signal_strength"] = strength.loc[current]
    chosen["regime"] = rule["family"]
    chosen = chosen[DECISION_COLUMNS].sort_values(["datetime", "priority", "symbol"],
                     ascending=[True, False, True], kind="stable").reset_index(drop=True)
    audit["request_rows"] = len(chosen)
    audit["request_stock_days"] = len(chosen.assign(date=chosen.datetime.dt.normalize())[["date", "symbol"]].drop_duplicates())
    return chosen, audit


def select_balanced_candidate(table: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """按内部真实净额选执行合格主动规则；亏损不伪装为hold获利，频率不压过净额。

hold用于共同底仓对照。无完整T或有T1/库存/恢复/底仓偏差的候选不能获选；若
全部不合格则明确hold并给失败状态。没有100组、CI或正收益的自动停用开关。
    """
    failures = ["inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
                "t1_violation_count", "base_plan_mismatch_count"]
    required = ["candidate_index", "incremental_net_pnl", "t_pair_count", *failures]
    if table.empty or not set(required).issubset(table):
        raise ValueError("候选表缺少实际收益和完整执行审计字段")
    result = table.copy()
    if result.candidate_index.duplicated().any() or result.candidate_index.eq(0).sum() != 1:
        raise ValueError("候选编号须唯一且包含一个hold对照")
    for key in required:
        result[key] = pd.to_numeric(result[key], errors="raise")
        if not np.isfinite(result[key]).all():
            raise ValueError(f"候选指标{key}含非有限值")
    counts = ["candidate_index", "t_pair_count", *failures]
    if any((result[key].lt(0) | result[key].mod(1).ne(0)).any() for key in counts):
        raise ValueError("计数指标必须为非负整数")
    hold = result.loc[result.candidate_index.eq(0)].iloc[0]
    if any(hold[key] != 0 for key in ["incremental_net_pnl", "t_pair_count", *failures]):
        raise ValueError("hold对照不得有T增量、T成交或执行偏差")
    eligible = result.candidate_index.gt(0) & result.t_pair_count.gt(0) & result[failures].eq(0).all(axis=1)
    result["execution_eligible"] = eligible
    candidates = result.loc[eligible].sort_values(["incremental_net_pnl", "candidate_index"], ascending=[False, True], kind="stable")
    selected = hold.to_dict() if candidates.empty else candidates.iloc[0].to_dict()
    selected["selection_reason"] = ("no_execution_valid_active_candidate" if candidates.empty else
                                    "maximum_inner_net_pnl_among_execution_valid_active_candidates")
    result["selected"] = result.candidate_index.eq(selected["candidate_index"])
    return selected, result
