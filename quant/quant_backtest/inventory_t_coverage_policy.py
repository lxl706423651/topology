"""覆盖率消融的身份包装；原L2多期限排序、费用、当前资格与37数字规则不变。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inventory_t_coverage_models import VERSION, PROFILES, HORIZONS, SOURCE_PROFILE_MAP
from .inventory_t_learning_policy import (build_inventory_learning_candidates,
    create_inventory_learning_rule_config, generate_inventory_learning_decisions)

SIGNAL_TIMES = ("09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00")


def build_inventory_coverage_candidates(cfg: dict) -> list[dict]:
    """仅接受单一覆盖率消融和原费用/时钟/guard，规则内容复用原L2纯函数。"""
    if (cfg.get("training_loss") != "regression_l2" or cfg.get("feature_min_coverage") != .70
            or cfg.get("source_feature_min_coverage") != .80
            or cfg.get("forecast_target") != "fixed_20_40_60_reference_return"
            or cfg.get("t_recheck_exit_side_at_execution") is not True):
        raise ValueError("本轮仅允许L2覆盖率0.80→0.70及原成交时复核")
    if (tuple(cfg.get("signal_times", ())) != SIGNAL_TIMES or cfg.get("last_entry_time") != "14:30"
            or cfg.get("fixed_recovery_time") != "14:55"):
        raise ValueError("须保持原八个信号时钟和日内截止")
    for key in ("commission_rate", "transfer_rate", "slippage_rate", "stamp_rate_before_20230828",
                "stamp_rate_from_20230828", "min_commission"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key}须为原非负有限费率")
    if cfg["min_commission"] != 0:
        raise ValueError("原价格边际换算要求min_commission=0")
    return build_inventory_learning_candidates(cfg)


def create_inventory_coverage_rule_config(cfg: dict, rule: dict) -> dict:
    """校验版本合同后使用原L2规则配置，不修改费用、底仓和退出参数。"""
    build_inventory_coverage_candidates(cfg)
    return create_inventory_learning_rule_config(cfg, rule)


def generate_inventory_coverage_decisions(forecasts: dict[int, pd.DataFrame], rule: dict,
                                          cfg: dict) -> tuple[pd.DataFrame, dict]:
    """强核当前评分身份后调用原多期限纯策略；未来字段不影响资格或去重。"""
    create_inventory_coverage_rule_config(cfg, rule)
    if rule["enabled"]:
        if set(forecasts) != set(HORIZONS):
            raise ValueError("须提供三个原期限的全部当前预测")
        for horizon, frame in forecasts.items():
            required = {"datetime", "prediction", "raw_prediction", "return_score", "model_version",
                        "forecast_loss", "feature_min_coverage"}
            if frame.columns.duplicated().any() or not required.issubset(frame) or "forecast_alpha" in frame:
                raise ValueError("当前预测缺v18身份或混入Huber阈值")
            times = pd.to_datetime(frame.datetime, errors="raise")
            if (times.isna().any() or times.dt.tz is not None or not times.dt.strftime("%H:%M").isin(SIGNAL_TIMES).all()
                    or not times.eq(times.dt.floor("min")).all()):
                raise ValueError("信号必须是原八个完整分钟时钟")
            if (not frame.model_version.eq(VERSION).all() or not frame.forecast_loss.eq("regression_l2").all()
                    or not frame.feature_min_coverage.eq(.70).all()):
                raise ValueError("当前模型版本、损失或覆盖率身份不一致")
            values = frame[["prediction", "raw_prediction", "return_score"]].apply(pd.to_numeric, errors="raise")
            if (not np.isfinite(values).all().all() or not values.prediction.equals(values.raw_prediction)
                    or not values.prediction.equals(values.return_score)):
                raise ValueError("三个原始收益评分字段须逐值一致且有限")
    decisions, audit = generate_inventory_learning_decisions(forecasts, rule, cfg)
    decisions = decisions.copy()
    decisions["model_version"], decisions["forecast_loss"], decisions["feature_min_coverage"] = VERSION, "regression_l2", .70
    return decisions, {**audit, "research_version": VERSION, "forecast_loss": "regression_l2",
        "feature_min_coverage": .70, "source_feature_min_coverage": .80,
        "score_is_calibrated_conditional_mean": False, "same_numeric_l2_policy": True}
