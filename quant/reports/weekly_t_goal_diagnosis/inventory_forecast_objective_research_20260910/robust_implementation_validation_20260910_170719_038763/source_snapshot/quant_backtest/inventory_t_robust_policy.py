"""永久底仓Huber择时：从原20/40/60分钟当前分数中选择扣费换算边际最高方向。

预声明37个候选；模型拟合和实际库存/资金/恢复撮合均由其他模块负责。本模块
不按未来成交或恢复成功删信号。Huber位置分数不是已校准条件均值或可成交
净额；成本换算和排序是明确研究规则，原费用/容量/旧股与恢复约束保持。
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .balanced_t_policy import create_empty_balanced_decisions
from .inventory_t_robust_models import HORIZONS, PROFILES, VERSION, SOURCE_PROFILE_MAP
from .weekly_t_features import select_recoverable_signal_sides
from .weekly_t_short_horizon import planned_recovery_times, predicted_pair_edges


EXIT_MODES = ("fixed", "profit_buffer", "profit_buffer_stop")
MARGINS = (0., .001, .003)
FRACTIONS = (.5, 1.)
SIGNAL_TIMES = ("09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00")


def build_inventory_robust_candidates(cfg: dict) -> list[dict]:
    """固定2模型×3净边际×2旧股比例×3退出，另有真实无T基准。"""
    for key, expected in (("learning_model_profiles", PROFILES), ("learning_min_net_edges", MARGINS),
                          ("t_trade_fractions", FRACTIONS), ("dynamic_exit_modes", EXIT_MODES),
                          ("horizons", HORIZONS)):
        value = cfg.get(key)
        if not isinstance(value, list) or tuple(value) != expected or any(isinstance(v, bool) for v in value):
            raise ValueError(f"{key}必须保持本轮预声明网格{expected}")
    if (cfg.get("training_loss") != "huber" or isinstance(cfg.get("t_huber_alpha"), bool)
            or cfg.get("t_huber_alpha") != .01
            or cfg.get("forecast_target") != "fixed_20_40_60_reference_return"
            or cfg.get("t_recheck_exit_side_at_execution") is not True):
        raise ValueError("本轮须固定Huber alpha=0.01及原v15成交时复核")
    if (tuple(cfg.get("signal_times", ())) != SIGNAL_TIMES or cfg.get("last_entry_time") != "14:30"
            or cfg.get("fixed_recovery_time") != "14:55"):
        raise ValueError("保持原8个信号和14:30/14:55时钟合同")
    for key in ("commission_rate", "transfer_rate", "slippage_rate", "stamp_rate_before_20230828",
                "stamp_rate_from_20230828", "min_commission"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key}须为非负有限原费率")
    if cfg["min_commission"] != 0:
        raise ValueError("原成本换算要求min_commission=0")
    if cfg.get("dynamic_take_net_return") != .003 or cfg.get("dynamic_stop_net_return") != .01:
        raise ValueError("沿用v13预声明退出净额阈值，不随评价结果改变")
    rules = [{"candidate_index": 0, "enabled": False, "profile": "hold", "min_net_edge": 0.,
              "fraction": 0., "exit_mode": "fixed"}]
    for profile, margin, fraction, mode in product(PROFILES, MARGINS, FRACTIONS, EXIT_MODES):
        rules.append({"candidate_index": len(rules), "enabled": True, "profile": profile,
                      "min_net_edge": margin, "fraction": fraction, "exit_mode": mode})
    return rules


def create_inventory_robust_rule_config(cfg: dict, rule: dict) -> dict:
    """给独立账户传递所选退出模式，不改变共同费用/底仓/5%目标。"""
    rules = build_inventory_robust_candidates(cfg)
    index = rule.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(rules) or rule != rules[index]:
        raise ValueError("规则不在本轮事前候选中")
    return {**cfg, "dynamic_exit_mode": rule["exit_mode"]}


def generate_inventory_robust_decisions(predictions: dict[int, pd.DataFrame], rule: dict,
                                          cfg: dict) -> tuple[pd.DataFrame, dict]:
    """保留当刻Huber分数换算最高正净边际的合法期限/方向，下一分钟由引擎尝试。

所有期限覆盖相同当前键。并列净边际时优先较短期限、再先买；共享现金同刻按
分数换算边际降序、股票代码升序。排序不使用事后盈亏或真实成交数量。
    """
    create_inventory_robust_rule_config(cfg, rule)
    audit = {"research_version": VERSION, "candidate_index": rule["candidate_index"], "model_profile": rule["profile"],
             "uses_future_execution_or_recovery": False, "probability_gate": False,
             "feature_support_gate": False, "horizon_funnel": [], "prediction_is_execution_profit": False,
             "score_is_calibrated_conditional_mean": False, "forecast_loss": "huber", "forecast_alpha": .01,
             "source_l2_profile": SOURCE_PROFILE_MAP.get(rule["profile"], "hold")}
    if not rule["enabled"]:
        return create_empty_balanced_decisions(), {**audit, "request_rows": 0}
    if set(predictions) != set(HORIZONS):
        raise ValueError("必须提供20/40/60分钟模型的全部当前预测")
    alternatives, expected_keys = [], None
    for horizon in HORIZONS:
        frame = predictions[horizon].copy()
        required = {"datetime", "symbol", "prediction", "raw_prediction", "return_score", "model_profile", "horizon_minutes",
                    "model_version", "forecast_loss", "forecast_alpha", "signal_amount", "signal_volume"}
        if frame.columns.duplicated().any() or not required.issubset(frame):
            raise ValueError("当前预测缺必需字段或列名重复")
        frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
        if (frame[["datetime", "symbol"]].isna().any().any() or frame.datetime.dt.tz is not None
                or frame.duplicated(["datetime", "symbol"]).any()
                or not frame.model_profile.eq(rule["profile"]).all()
                or not frame.horizon_minutes.eq(horizon).all()
                or not frame.model_version.eq(VERSION).all() or not frame.forecast_loss.eq("huber").all()
                or not frame.forecast_alpha.eq(.01).all()
                or not frame.datetime.dt.strftime("%H:%M").isin(SIGNAL_TIMES).all()
                or frame.datetime.dt.second.ne(0).any() or frame.datetime.dt.microsecond.ne(0).any()
                or frame.datetime.dt.nanosecond.ne(0).any()):
            raise ValueError("当前键、模型或期限身份不一致")
        keys = frame[["datetime", "symbol"]].sort_values(["datetime", "symbol"]).reset_index(drop=True)
        if expected_keys is not None and not keys.equals(expected_keys):
            raise ValueError("各期限当前键必须完全相同，不能按未来标签有效性取交集")
        expected_keys = keys
        prediction = pd.to_numeric(frame.prediction, errors="raise")
        if not np.isfinite(prediction).all() or not prediction.gt(-1).all():
            raise ValueError("模型参考价格比必须有限且大于零")
        if (not np.array_equal(prediction.to_numpy(), pd.to_numeric(frame.return_score, errors="raise").to_numpy())
                or not np.array_equal(prediction.to_numpy(), pd.to_numeric(frame.raw_prediction, errors="raise").to_numpy())):
            raise ValueError("原Huber分数与用于成本换算的预测值不同")
        buy, sell = predicted_pair_edges(frame, cfg)
        action = pd.Series(np.where(buy.ge(sell), 1, -1), index=frame.index)
        edge = pd.Series(np.maximum(buy, sell), index=frame.index)
        deadline = planned_recovery_times(frame.datetime, horizon)
        day = frame.datetime.dt.normalize()
        valid = edge.gt(0) & edge.ge(rule["min_net_edge"])
        before_side = int(valid.sum())
        valid &= select_recoverable_signal_sides(frame, action, cfg)
        amount = pd.to_numeric(frame.signal_amount, errors="raise")
        volume = pd.to_numeric(frame.signal_volume, errors="raise")
        valid &= np.isfinite(amount) & np.isfinite(volume) & amount.gt(0) & volume.gt(0)
        valid &= frame.datetime.le(day + pd.to_timedelta(cfg["last_entry_time"]+":00"))
        valid &= deadline.notna() & deadline.dt.normalize().eq(day)
        valid &= deadline.le(day + pd.to_timedelta(cfg["fixed_recovery_time"]+":00"))
        valid &= deadline.gt(frame.datetime+pd.Timedelta(minutes=1))
        chosen = frame.loc[valid, ["datetime", "symbol"]].copy()
        chosen["action"], chosen["priority"] = action.loc[valid], edge.loc[valid]
        chosen["planned_recovery_time"], chosen["prediction"] = deadline.loc[valid], prediction.loc[valid]
        chosen["horizon_minutes"] = horizon
        alternatives.append(chosen)
        audit["horizon_funnel"].append({"horizon_minutes": horizon, "current_rows": len(frame),
            "positive_net_predictions": before_side, "current_eligible_alternatives": len(chosen)})
    combined = pd.concat(alternatives, ignore_index=True)
    combined = combined.sort_values(["datetime", "symbol", "priority", "horizon_minutes", "action"],
        ascending=[True, True, False, True, False], kind="stable").drop_duplicates(["datetime", "symbol"])
    combined["fraction"], combined["candidate_index"] = rule["fraction"], rule["candidate_index"]
    combined["signal_strength"], combined["predicted_net_edge"] = combined.priority, combined.priority
    combined["regime"], combined["model_profile"] = "learned_permanent_inventory", rule["profile"]
    combined["return_score"] = combined.prediction
    combined["model_version"], combined["forecast_loss"], combined["forecast_alpha"] = VERSION, "huber", .01
    combined = combined.sort_values(["datetime", "priority", "symbol"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    return combined, {**audit, "request_rows": len(combined), "alternatives": sum(len(x) for x in alternatives),
                      "stable_tie_order": "net_desc_horizon_asc_buy_first_then_shared_cash_symbol_asc"}
