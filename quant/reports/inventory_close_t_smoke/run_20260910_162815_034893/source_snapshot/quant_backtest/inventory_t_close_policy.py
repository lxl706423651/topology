"""永久底仓固定14:55目标择时，只消费单个模型的当前预测。

37个事前候选沿用原费用、份额与退出模式；不跨预测期限取最大值，不按未来
参考价格或可成交性删信号。下一分钟首腿、库存和动态恢复由独立执行器处理。
"""
from __future__ import annotations

from itertools import product
from numbers import Real

import numpy as np
import pandas as pd

from .balanced_t_policy import create_empty_balanced_decisions
from .inventory_t_close_labels import (IDENTITY, calculate_fixed_close_times,
                                       validate_current_keys, validate_fixed_close_config)
from .inventory_t_close_models import PROFILES, VERSION
from .weekly_t_features import select_recoverable_signal_sides
from .weekly_t_short_horizon import predicted_pair_edges


EXIT_MODES = ("fixed", "profit_buffer", "profit_buffer_stop")
MARGINS = (0., .001, .003)
FRACTIONS = (.5, 1.)


def build_inventory_close_candidates(cfg: dict) -> list[dict]:
    """固定两模型、三净边际、两份额、三退出，加hold；身份不能混入旧期限。"""
    validate_fixed_close_config(cfg)
    if (cfg.get("forecast_target") != "same_day_1455" or cfg.get("recovery_key") != "close"
            or "horizons" in cfg or "t_label_horizon_minutes" in cfg):
        raise ValueError("本策略仅接受same_day_1455/close，不接受整数预测期限")
    if cfg.get("t_recheck_exit_side_at_execution") is not True:
        raise ValueError("固定尾盘研究必须保留v15成交时反向限价复查")
    if cfg.get("last_entry_time") != "14:30":
        raise ValueError("沿用原首腿信号最晚14:30，不按评价结果修改")
    for key, expected in (("learning_model_profiles", PROFILES), ("learning_min_net_edges", MARGINS),
                          ("t_trade_fractions", FRACTIONS), ("dynamic_exit_modes", EXIT_MODES)):
        value = cfg.get(key)
        if not isinstance(value, list) or tuple(value) != expected or any(isinstance(v, bool) for v in value):
            raise ValueError(f"{key}必须保持本轮预声明网格{expected}")
    if cfg.get("dynamic_take_net_return") != .003 or cfg.get("dynamic_stop_net_return") != .01:
        raise ValueError("沿用预声明动态退出净额阈值")
    for key in ("commission_rate", "transfer_rate", "slippage_rate", "stamp_rate_before_20230828",
                "stamp_rate_from_20230828", "min_commission"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key}须为非负有限费率")
    if cfg["min_commission"] != 0:
        raise ValueError("净边际换算要求min_commission=0，不能静默低估最低佣金")
    rules = [{"candidate_index": 0, "enabled": False, "profile": "hold", "min_net_edge": 0.,
              "fraction": 0., "exit_mode": "fixed"}]
    for profile, margin, fraction, mode in product(PROFILES, MARGINS, FRACTIONS, EXIT_MODES):
        rules.append({"candidate_index": len(rules), "enabled": True, "profile": profile,
                      "min_net_edge": margin, "fraction": fraction, "exit_mode": mode})
    return rules


def create_inventory_close_rule_config(cfg: dict, rule: dict) -> dict:
    """只给独立账户设置候选退出模式，保留原配置对象和费用、股数合同。"""
    rules = build_inventory_close_candidates(cfg)
    index = rule.get("candidate_index")
    if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
            or not 0 <= index < len(rules) or rule != rules[index]):
        raise ValueError("规则不在本轮事前37候选中")
    return {**cfg, "dynamic_exit_mode": rule["exit_mode"]}


def generate_inventory_close_decisions(predictions: pd.DataFrame, rule: dict,
                                       cfg: dict) -> tuple[pd.DataFrame, dict]:
    """按单一14:55预测选双腿扣费后较优方向；当前同刻按净边际/代码排序。

固定端点净边际不等于实际成交利润。额外边际0仍要求预测净额严格正；当前
恢复方向、当前量能与原信号期限沿旧协议，未来execution/recovery/label无输入权。
    """
    create_inventory_close_rule_config(cfg, rule)
    audit = {"research_version": VERSION, "candidate_index": rule["candidate_index"],
             "model_profile": rule["profile"], **IDENTITY, "forecast_target": "same_day_1455",
             "uses_future_execution_or_recovery": False, "cross_horizon_argmax": False,
             "probability_gate": False, "feature_support_gate": False,
             "prediction_is_execution_profit": False}
    if not rule["enabled"]:
        empty = create_empty_balanced_decisions()
        for key in (*IDENTITY, "model_profile"):
            empty[key] = pd.Series(dtype="object")
        for key in ("prediction", "predicted_net_edge"):
            empty[key] = pd.Series(dtype="float64")
        return empty, {**audit, "request_rows": 0, "current_rows": 0}
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("每个profile只接受一张固定尾盘当前预测表，不接受多期限映射")
    required = {"datetime", "symbol", "prediction", "model_profile", "signal_amount", "signal_volume", *IDENTITY}
    if predictions.columns.duplicated().any() or not required.issubset(predictions):
        raise ValueError("当前预测缺必需字段或列名重复")
    if "horizon_minutes" in predictions:
        raise ValueError("固定尾盘预测不得混入horizon_minutes身份")
    keys = validate_current_keys(predictions)
    if not predictions.model_profile.eq(rule["profile"]).all():
        raise ValueError("当前模型profile与候选不一致")
    for key, expected in IDENTITY.items():
        if not predictions[key].eq(expected).all():
            raise ValueError(f"当前预测{key}不是固定14:55目标身份")
    frame = predictions.copy()
    frame["datetime"] = keys.datetime
    prediction = pd.to_numeric(frame.prediction, errors="raise")
    if not np.isfinite(prediction).all() or not prediction.gt(-1).all():
        raise ValueError("模型参考价格比必须有限且大于零")
    frame["prediction"] = prediction
    buy, sell = predicted_pair_edges(frame, cfg)
    action = pd.Series(np.where(buy.ge(sell), 1, -1), index=frame.index)
    edge = pd.Series(np.maximum(buy, sell), index=frame.index)
    first, deadline = calculate_fixed_close_times(frame.datetime)
    valid = edge.gt(0) & edge.ge(rule["min_net_edge"])
    after_edge = int(valid.sum())
    valid &= select_recoverable_signal_sides(frame, action, cfg)
    after_side = int(valid.sum())
    amount = pd.to_numeric(frame.signal_amount, errors="raise")
    volume = pd.to_numeric(frame.signal_volume, errors="raise")
    valid &= np.isfinite(amount) & np.isfinite(volume) & amount.gt(0) & volume.gt(0)
    after_volume = int(valid.sum())
    valid &= frame.datetime.le(frame.datetime.dt.normalize() + pd.Timedelta(hours=14, minutes=30))
    valid &= deadline.gt(first)
    chosen = frame.loc[valid, ["datetime", "symbol"]].copy()
    chosen["action"], chosen["priority"] = action.loc[valid], edge.loc[valid]
    chosen["planned_recovery_time"], chosen["prediction"] = deadline.loc[valid], prediction.loc[valid]
    chosen["fraction"], chosen["candidate_index"] = rule["fraction"], rule["candidate_index"]
    chosen["signal_strength"], chosen["predicted_net_edge"] = chosen.priority, chosen.priority
    chosen["regime"], chosen["model_profile"] = "learned_permanent_fixed_close_inventory", rule["profile"]
    for key, value in IDENTITY.items():
        chosen[key] = value
    chosen = chosen.sort_values(["datetime", "priority", "symbol"],
        ascending=[True, False, True], kind="stable").reset_index(drop=True)
    return chosen, {**audit, "current_rows": len(frame), "positive_net_predictions": after_edge,
                    "after_current_exit_side": after_side, "after_current_volume": after_volume,
                    "request_rows": len(chosen), "stable_tie_order": "net_desc_buy_first_then_shared_cash_symbol_asc"}
