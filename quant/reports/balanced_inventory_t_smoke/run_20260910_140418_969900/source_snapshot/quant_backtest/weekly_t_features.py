"""周度做T的补充会话特征与恢复风险门控，只读取当前及历史已知行情。

输入为weekly_t_data的盘中表；输出增加f_*的复制表或同索引资格掩码。
未来execution/recovery/target列均不参与计算，不改原始缓存。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_weekly_execution_features(weekly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """加入已完成第N周收盘竞价流动性；缺当周任何收盘报价/成交额则最小容量记0。

    只按feature_week连接过去周，不读取第N+1周执行数据。用于事先限制周末退出
    过于脆弱的候选，不能保证未来停牌或未来收盘容量。
    """
    required = {"date", "symbol", "close", "close_amount"}
    if not required.issubset(daily) or not {"feature_week", "symbol", "f_week_days"}.issubset(weekly):
        raise ValueError("周度执行特征缺少日收盘量价或特征周日历字段")
    dates = pd.to_datetime(daily.date).dt.normalize()
    values = daily[["symbol", "close", "close_amount"]].copy()
    values["feature_week"] = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    valid = np.isfinite(values.close) & (values.close > 0) & np.isfinite(values.close_amount) & (values.close_amount > 0)
    values["known_amount"] = values.close_amount.where(valid, 0.)
    values["valid_close"] = valid.astype(int)
    aggregates = values.groupby(["feature_week", "symbol"]).agg(
        minimum=("known_amount", "min"), mean=("known_amount", "mean"), valid_days=("valid_close", "sum"))
    keys = pd.MultiIndex.from_frame(weekly[["feature_week", "symbol"]])
    aligned = aggregates.reindex(keys)
    result = weekly.copy()
    complete = aligned.valid_days.to_numpy() == result.f_week_days.to_numpy()
    result["f_prior_close_auction_amount_min"] = np.where(complete, aligned.minimum.to_numpy(), 0.).astype("float32")
    result["f_prior_close_auction_amount_mean"] = aligned["mean"].to_numpy(dtype="float32")
    result["f_missing_prior_close_days"] = (result.f_week_days.to_numpy() - aligned.valid_days.fillna(0).to_numpy()).astype("float32")
    return result


def add_session_context_features(intraday: pd.DataFrame) -> pd.DataFrame:
    """加入隔夜跳空、前收涨跌幅及波动标准化特征，市场状态只按相同datetime汇总。

    dayopen由当前close和截至当前已知的f_open_return恒等还原；previous_close是
    先前交易日观测，历史振幅下限0.3%仅是固定数值稳定常量，无未来标定。
    """
    required = {"datetime", "symbol", "signal_price", "previous_close", "f_open_return",
                "f_vwap_deviation", "f_previous_day_amplitude"}
    if not required.issubset(intraday):
        raise ValueError(f"会话特征缺列：{sorted(required.difference(intraday.columns))}")
    if intraday.duplicated(["datetime", "symbol"]).any():
        raise ValueError("会话特征要求datetime/symbol唯一")
    result = intraday.copy()
    previous = pd.to_numeric(result.previous_close, errors="coerce").where(lambda x: x > 0)
    signal = pd.to_numeric(result.signal_price, errors="coerce").where(lambda x: x > 0)
    dayopen = signal / (1 + result.f_open_return)
    scale = result.f_previous_day_amplitude.clip(lower=.003)
    additions = {
        "f_overnight_gap": dayopen / previous - 1,
        "f_day_return": signal / previous - 1,
        "f_signal_price_log": np.log(signal),
        "f_open_return_by_previous_amplitude": result.f_open_return / scale,
        "f_vwap_deviation_by_previous_amplitude": result.f_vwap_deviation / scale,
    }
    for key, value in additions.items():
        result[key] = value.replace([np.inf, -np.inf], np.nan).astype("float32")
    result["f_market_day_return"] = result.groupby("datetime").f_day_return.transform("mean").astype("float32")
    result["f_relative_day_return"] = (result.f_day_return - result.f_market_day_return).astype("float32")
    result["f_day_return_by_previous_amplitude"] = (result.f_day_return / scale).astype("float32")
    return result


def select_recoverable_signal_sides(predictions: pd.DataFrame, actions: pd.Series, cfg: dict) -> pd.Series:
    """仅当信号时刻的恢复方向已锁在代理涨跌停时拒绝发起T，不预测未来能否恢复。

    买先最后需卖旧股，当前锁跌停时不发起；卖先最后需买回，当前锁涨停时不发起。
    普通平价有量分钟不是锁板；5%只在价格精确匹配分币候选限价时判断。
    """
    if not cfg.get("t_avoid_locked_exit_side", False):
        return pd.Series(True, index=predictions.index)
    if not actions.index.equals(predictions.index):
        raise ValueError("动作与预测行的索引必须完全对齐")
    required = {"signal_price", "signal_high", "signal_low", "previous_close"}
    if not required.issubset(predictions):
        raise ValueError(f"当前恢复风险门控缺列：{sorted(required.difference(predictions.columns))}")
    p = predictions.signal_price.astype(float)
    previous = predictions.previous_close.astype(float)
    valid = (np.isfinite(p) & np.isfinite(previous) & (p > 0) & (previous > 0)
             & np.isfinite(predictions.signal_high) & np.isfinite(predictions.signal_low)
             & (predictions.signal_high >= p) & (predictions.signal_low <= p))
    flat = predictions.signal_high.eq(predictions.signal_low)
    signed = pd.Series(np.where(actions > 0, -1., 1.), index=predictions.index)
    move = (p / previous - 1) * signed
    candidate = np.floor(previous * (1 + signed * cfg["flat_st_limit_rate"]) * 100 + .5 + 1e-9) / 100
    locked = (move >= cfg["nonflat_limit_proxy"]) | (flat & ((p - candidate).abs() <= 1e-6))
    return valid & ~locked
