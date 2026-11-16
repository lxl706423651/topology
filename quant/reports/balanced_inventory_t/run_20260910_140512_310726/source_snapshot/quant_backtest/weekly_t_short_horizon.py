"""冻结周度选股上的短周期库存T规则；只以当前预测和核心训练支持域发出请求。

从quant项目调用。未来恢复价仅构成训练标签，计划恢复时间由交易时钟计算。
输入面板不原地修改；配置、模型及候选记录由运行脚本保存在独立报告目录。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .strategy import BaseStrategy
from .weekly_t_features import select_recoverable_signal_sides
from .weekly_t_horizon_data import advance_trading_minutes, next_trading_minute


def planned_recovery_times(datetimes: pd.Series, horizon: int) -> pd.Series:
    """按下一成交分钟加固定交易分钟算计划时间，跳过午休且不跨日。"""
    times = pd.to_datetime(datetimes, errors="raise")
    mapping = {timestamp: advance_trading_minutes(next_trading_minute(timestamp), horizon)
               for timestamp in times.unique()}
    return pd.to_datetime(times.map(mapping))


def attach_horizon_labels(intraday: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """把指定周期标签对齐到全部当前信号；未来缺价仍保留行，不改变当前f_*。"""
    selected = labels.loc[labels.horizon_minutes.eq(horizon)]
    keys = ["datetime", "symbol"]
    fields = ["recovery_time", "recovery_price", "label_start", "label_end", "target_return"]
    if selected.duplicated(keys).any():
        raise ValueError("短周期标签存在重复信号键")
    result = intraday.drop(columns=[c for c in fields + ["label_available"] if c in intraday]).merge(
        selected[keys + fields], on=keys, how="left", validate="one_to_one", sort=False)
    result["label_available"] = np.isfinite(result.target_return)
    result["horizon_minutes"] = int(horizon)
    # 原端点的高低量额仍属14:55，不能与新周期价格混配；实际执行须读取当前quote store。
    for column in ("recovery_open", "recovery_high", "recovery_low", "recovery_volume", "recovery_amount"):
        if column in result:
            result[column] = np.nan
    # 单独从时钟生成，禁止通过未来标签是否存在决定是否请求首腿。
    result["planned_recovery_time"] = planned_recovery_times(result.datetime, horizon)
    return result


def fit_feature_support(training: pd.DataFrame, cfg: dict) -> dict:
    """仅用严格隔离后的核心训练行拟合固定输入分位支持域，不看内部/外层验证。"""
    columns = list(cfg["t_support_features"])
    lower, upper = map(float, cfg["t_support_quantiles"])
    if not columns or not 0 <= lower < upper <= 1:
        raise ValueError("输入支持域配置为空或分位数非法")
    bounds = {}
    for column in columns:
        if not column.startswith("f_") or column not in training:
            raise ValueError(f"支持域缺少因果输入：{column}")
        values = pd.to_numeric(training[column], errors="coerce")
        finite = values.loc[np.isfinite(values)]
        if finite.empty:
            raise ValueError(f"支持域训练列无有限观测：{column}")
        bounds[column] = {"lower": float(finite.quantile(lower)), "upper": float(finite.quantile(upper)),
                          "finite_rows": len(finite)}
    signature = hashlib.sha256(pd.util.hash_pandas_object(
        training[["datetime", "symbol", *columns]], index=False).to_numpy().tobytes()).hexdigest()
    return {"bounds": bounds, "quantiles": [lower, upper], "training_rows": len(training),
            "training_inputs_sha256": signature, "scope": "仅核心训练，不从推断年份重估边界"}


def feature_support_mask(frame: pd.DataFrame, support: dict) -> pd.Series:
    """按冻结训练分位界检查当前f_*；缺失或域外输入拒绝发起新T。"""
    allowed = pd.Series(True, index=frame.index)
    for column, bounds in support["bounds"].items():
        values = pd.to_numeric(frame[column], errors="coerce")
        allowed &= np.isfinite(values) & values.between(bounds["lower"], bounds["upper"])
    return allowed


def predicted_pair_edges(predictions: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """由预测价格比计算每元首腿参考名义额的双向净利，包含双边费率及当日印花税。

零最低佣金时该换算精确；成交容量、资金和整手由引擎在真实到达分钟约束。
非零最低佣金需按实际股数估计，故当前规则拒绝该不支持配置，避免低估费用。
"""
    if float(cfg.get("min_commission", 0)) != 0:
        raise ValueError("短周期预测边际当前要求min_commission=0；固定最低佣金需另行按股数建模")
    fee = float(cfg["commission_rate"]) + float(cfg["transfer_rate"])
    slip = float(cfg["slippage_rate"])
    stamp = pd.Series(np.where(predictions.datetime < pd.Timestamp("2023-08-28"),
                              cfg["stamp_rate_before_20230828"], cfg["stamp_rate_from_20230828"]),
                      index=predictions.index)
    ratio = 1.0 + predictions.prediction
    buy = ratio * (1 - slip) * (1 - fee - stamp) - (1 + slip) * (1 + fee)
    sell = (1 - slip) * (1 - fee - stamp) - ratio * (1 + slip) * (1 + fee)
    return buy, sell


class ShortHorizonTStrategy(BaseStrategy):
    """保留周底仓，以预期净利决定先买/先卖，在固定短周期恢复昨日股数。"""

    name = "weekly_short_horizon_t"

    def __init__(self, rule: dict, cfg: dict, support: dict):
        """保存已冻结规则、成本和训练支持域，不触及未来成交行情。"""
        self.rule, self.cfg, self.support = dict(rule), dict(cfg), support

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """返回与输入同索引的目标底仓、T方向、净边际及固定恢复时间。"""
        result = pd.DataFrame(index=data.index)
        result["target_position"] = 1.0
        result["t_action"] = 0
        result["fraction"] = float(self.cfg["short_t_fraction"])
        result["planned_recovery_time"] = planned_recovery_times(data.datetime, int(self.rule["horizon_minutes"]))
        buy, sell = predicted_pair_edges(data, self.cfg)
        result["priority"] = np.maximum(buy, sell)
        if self.rule.get("enabled", True):
            valid = feature_support_mask(data, self.support)
            valid &= result.planned_recovery_time.notna()
            valid &= result.planned_recovery_time.dt.strftime("%H:%M").le(self.cfg["latest_planned_recovery_time"])
            valid &= np.isfinite(data.prediction) & data.prediction.gt(-1)
            valid &= result.priority.ge(float(self.rule["net_margin"]))
            result.loc[valid & buy.ge(sell), "t_action"] = 1
            result.loc[valid & sell.gt(buy), "t_action"] = -1
            result.loc[~select_recoverable_signal_sides(data, result.t_action, self.cfg), "t_action"] = 0
        return result


def create_short_decisions(predictions: pd.DataFrame, rule: dict, cfg: dict, support: dict) -> pd.DataFrame:
    """所有当前门控通过后，每股每日只请求最早一次，保留同刻预测强度排序。"""
    signals = ShortHorizonTStrategy(rule, cfg, support).generate_signals(predictions)
    decisions = predictions[["datetime", "symbol"]].copy()
    decisions["action"] = signals.t_action
    for key in ("fraction", "priority", "planned_recovery_time"):
        decisions[key] = signals[key]
    decisions = decisions.loc[decisions.action.ne(0)].sort_values(["datetime", "symbol"], kind="stable")
    decisions["date"] = decisions.datetime.dt.normalize()
    return decisions.drop_duplicates(["date", "symbol"], keep="first").reset_index(drop=True)


def candidate_support_statistics(result: dict, cfg: dict) -> dict:
    """从真实成交和逐周权益增量度量样本支持；恢复片段不重复计为独立pair。"""
    trips = result["round_trips"]
    events = result["trade_events"]
    first = events.loc[events.event_type.eq("t")].sort_values("datetime", kind="stable").drop_duplicates("pair_id")
    days = pd.to_datetime(first.datetime).dt.normalize()
    weeks = days.dt.to_period("W-SUN").dt.start_time
    daily_counts, weekly_counts = days.value_counts(), weeks.value_counts()
    curve = result["equity_curve"]
    week = pd.to_datetime(curve.datetime).dt.to_period("W-SUN").dt.start_time
    added = (curve.equity - curve.baseline_equity).groupby(week).last()
    increments = added.diff().fillna(added).to_numpy()
    rng = np.random.default_rng(int(cfg["random_state"]))
    samples = rng.choice(increments, size=(int(cfg["bootstrap_samples"]), len(increments)), replace=True).sum(axis=1)
    daily_pnl = pd.Series(dtype=float)
    weekly_pnl = pd.Series(dtype=float)
    if len(trips):
        first_days = pd.to_datetime(trips.first_datetime).dt.normalize()
        daily_pnl = trips.net_pnl.groupby(first_days).sum()
        weekly_pnl = trips.net_pnl.groupby(first_days.dt.to_period("W-SUN").dt.start_time).sum()
    return {"active_days": int(len(daily_counts)), "active_weeks": int(len(weekly_counts)),
            "active_symbols": int(first.symbol.nunique()), "initiated_pairs": int(len(first)),
            "max_day_pair_fraction": float(daily_counts.max() / len(first)) if len(first) else 0.,
            "max_week_pair_fraction": float(weekly_counts.max() / len(first)) if len(first) else 0.,
            "max_day_abs_pnl_fraction": float(daily_pnl.abs().max() / daily_pnl.abs().sum()) if daily_pnl.abs().sum() else 0.,
            "max_week_abs_pnl_fraction": float(weekly_pnl.abs().max() / weekly_pnl.abs().sum()) if weekly_pnl.abs().sum() else 0.,
            "weekly_increment_ci_low_rmb": float(np.quantile(samples, .025)),
            "weekly_increment_ci_high_rmb": float(np.quantile(samples, .975))}


def candidate_passes_support(evaluation: dict, cfg: dict) -> tuple[bool, list[str]]:
    """按事前硬门槛判断内部候选，失败原因完整返回，不因样本少而放宽约束。"""
    required = cfg["candidate_support_requirements"]
    metrics, support = evaluation["metrics"], evaluation["support"]
    reasons = [key for key in ("inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
                               "base_plan_mismatch_count", "t1_violation_count") if metrics.get(key, 0)]
    checked = {"incremental_net_pnl": evaluation["incremental_net_pnl"],
               "t_pair_count": metrics["t_pair_count"], **support}
    reasons.extend("nonfinite_" + key for key, value in checked.items() if not np.isfinite(value))
    if evaluation["incremental_net_pnl"] <= 0:
        reasons.append("nonpositive_net_pnl")
    if support["weekly_increment_ci_low_rmb"] <= 0:
        reasons.append("nonpositive_weekly_bootstrap_lower")
    if metrics["t_pair_count"] < required["min_complete_pairs"]:
        reasons.append("insufficient_complete_pairs")
    for key in ("active_days", "active_weeks", "active_symbols"):
        if support[key] < required["min_" + key]:
            reasons.append("insufficient_" + key)
    for key in ("max_day_pair_fraction", "max_week_pair_fraction", "max_day_abs_pnl_fraction", "max_week_abs_pnl_fraction"):
        if support[key] > required[key] or support[key] < 0:
            reasons.append(key)
    return not reasons, reasons
