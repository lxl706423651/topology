"""原XGBoost/CTA双执行口径：旧lot函数直接对照，约束版只调整撮合和可知信息时刻。

两者共用冻结分数/门槛、前5、10%总资产及原Wilder ATR240的2/4倍退出。
旧式卖出同分钟看价成交只能作为有偏诊断；不在收益出来后选择执行口径。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_backtest.annual_portfolio import MarketStore, run_lot_portfolio, calculate_daily_metrics
from scripts.run_all_a_method_backtests import FeeConfig
from scripts.run_weekly_high_vol_lot_exit_backtests import MethodSpec, LotExitRule, run_lot_backtest


class OriginalAtrExitPolicy:
    """对market每个独立批次按原2/4公式判断；只读取当前信号high/close/ATR及入场信号ATR。"""

    def __init__(self, market: MarketStore):
        """market必须包含按年初始化的Wilder ATR及信号分钟high。"""
        self.market = market
        self.highs = {}

    def __call__(self, index: int, positions: list) -> dict:
        """index是当前已完成信号分钟，positions为存续批次；返回触发原因并记录原动态止损线。"""
        result = {}
        for lot in positions:
            asset = lot.asset
            high = self.market.values["high"][index, asset]
            previous_high = self.highs.get(lot.lot_id, lot.buy_price)
            self.highs[lot.lot_id] = max(previous_high, high) if np.isfinite(high) else previous_high
            entry_atr = self.market.values["atr"][lot.entry_index, asset]
            current_atr = self.market.values["atr"][index, asset]
            stop = lot.buy_price - 2 * entry_atr
            if np.isfinite(current_atr) and current_atr > 0:
                stop = max(stop, self.highs[lot.lot_id] - 4 * current_atr)
            # 原公式随ATR变化可放松；不偷换为20/40倍或追加3/5日到期。
            lot.stop_price = stop
            close = self.market.values["close"][index, asset]
            if np.isfinite(close) and close <= stop:
                result[lot.lot_id] = "原ATR240初始2倍/跟踪4倍退出（信号后成交）"
        return result


class CachedIndicatorStore:
    """给旧lot函数提供精确成交分钟指标，避免重复读取全量原CSV；有偏同刻信息只用于旧式口径。"""

    def __init__(self, market: MarketStore):
        """market含下一分钟close/high/ATR矩阵，不对成交报价向前填充。"""
        self.market = market

    def get_row(self, symbol: str, dt: pd.Timestamp):
        """返回symbol在dt精确分钟的旧接口Series，无价格则None。"""
        index = self.market.time_index.get(dt - pd.Timedelta(minutes=1))
        asset = self.market.asset_index.get(symbol)
        if index is None or asset is None:
            return None
        price = self.market.values["execution_close"][index, asset]
        if not np.isfinite(price) or price <= 0:
            return None
        return pd.Series({"close": price, "high": self.market.values["execution_high"][index, asset],
                          "atr_240": self.market.values["execution_atr"][index, asset]})


def convert_legacy_result(native: tuple, scores: pd.DataFrame, market: MarketStore,
                          start: pd.Timestamp, end: pd.Timestamp, cfg: dict) -> dict:
    """将native旧函数成交转换为标准CSV，并用market补全部日终估值；scores给买入信号时间。

    不改变任何旧函数买卖，保留native原指标供追溯；标准日频指标不可与旧不规则bar年化混用。
    旧买入预算是成交时卖出后权益，signal_equity列在此口径明确保存该旧预算基数。
    """
    old_curve, _, old_events, _, old_open, old_metrics = native
    columns = ["datetime", "signal_datetime", "symbol", "lot_id", "side", "shares", "price", "notional", "fee", "reason", "signal_equity"]
    events = []
    if len(old_events):
        for row in old_events.to_dict("records"):
            dt = pd.Timestamp(row["交易时间(execution_datetime)"])
            buy = row["方向(side)"] == "买入"
            events.append({"datetime": dt, "signal_datetime": dt - pd.Timedelta(minutes=1) if buy else dt,
                "symbol": row["股票代码(symbol)"], "lot_id": row["lot_id"], "side": "buy" if buy else "sell",
                "shares": int(row["股数(shares)"]), "price": row["价格(price)"], "notional": row["成交金额(traded_value)"],
                "fee": row["交易成本(cost)"], "reason": row["触发原因(trigger_reason)"], "signal_equity": np.nan})
    event_frame = pd.DataFrame(events, columns=columns)
    for col in ["datetime", "signal_datetime"]:
        event_frame[col] = pd.to_datetime(event_frame[col])
    buy_fees = event_frame.loc[event_frame.side == "buy"].groupby("datetime").fee.sum()
    if len(event_frame):
        base = old_curve.equity.copy()
        base.index = pd.to_datetime(base.index) + pd.Timedelta(minutes=1)
        base = base + buy_fees.reindex(base.index, fill_value=0)
        event_frame["signal_equity"] = event_frame.datetime.map(base)
    grouped = {dt: frame for dt, frame in event_frame.groupby("datetime", sort=False)}
    positions, trips, curves, holdings = {}, [], [], []
    cash = float(cfg["initial_cash"])
    ids = np.flatnonzero((market.times >= start) & (market.times < end))
    for index in ids:
        dt = market.times[index]
        mark = market.marks[index]
        execution = dt + pd.Timedelta(minutes=1) if dt.hour != 15 else dt
        rows = grouped.get(execution)
        if rows is not None:
            for row in rows.to_dict("records"):
                if row["side"] == "buy":
                    cash -= row["notional"] + row["fee"]
                    positions[row["lot_id"]] = row
                else:
                    buy = positions.pop(row["lot_id"])
                    if buy["shares"] != row["shares"]:
                        raise ValueError("旧式口径出现未声明的部分卖出")
                    cash += row["notional"] - row["fee"]
                    trips.append({"lot_id": row["lot_id"], "symbol": row["symbol"], "buy_datetime": buy["datetime"],
                        "sell_datetime": row["datetime"], "buy_signal_datetime": buy["signal_datetime"], "sell_signal_datetime": row["signal_datetime"],
                        "buy_price": buy["price"], "sell_price": row["price"], "shares": row["shares"],
                        "buy_fee": buy["fee"], "sell_fee": row["fee"],
                        "net_pnl": row["notional"] - buy["notional"] - buy["fee"] - row["fee"],
                        "holding_days": (row["datetime"] - buy["datetime"]).total_seconds() / 86400,
                        "buy_reason": buy["reason"], "sell_reason": row["reason"]})
        if dt.hour != 15:
            current = market.values["execution_close"][index]
            mark = np.where(np.isfinite(current) & (current > 0), current, mark)
        value = sum(p["shares"] * mark[market.asset_index[p["symbol"]]] for p in positions.values())
        equity = cash + value
        curves.append({"datetime": execution, "equity": equity, "cash": cash, "exposure": value / equity,
                       "open_lots": len(positions), "holding_count": len({p["symbol"] for p in positions.values()})})
        if dt.hour == 15:
            for lot_id, p in positions.items():
                holdings.append({"datetime": dt, "lot_id": lot_id, "symbol": p["symbol"], "shares": p["shares"],
                                 "mark_price": mark[market.asset_index[p["symbol"]]], "stop_price": np.nan})
    trip_cols = ["lot_id", "symbol", "buy_datetime", "sell_datetime", "buy_signal_datetime", "sell_signal_datetime", "buy_price", "sell_price",
                 "shares", "buy_fee", "sell_fee", "net_pnl", "holding_days", "buy_reason", "sell_reason"]
    opened = [{"lot_id": k, "symbol": p["symbol"], "buy_datetime": p["datetime"], "shares": p["shares"],
        "buy_price": p["price"], "buy_fee": p["fee"], "mark_price": market.marks[ids[-1], market.asset_index[p["symbol"]]],
        "stop_price": np.nan, "pending_reason": ""} for k, p in positions.items()]
    open_cols = ["lot_id", "symbol", "buy_datetime", "shares", "buy_price", "buy_fee", "mark_price", "stop_price", "pending_reason"]
    curve, trip_frame, open_frame = pd.DataFrame(curves), pd.DataFrame(trips, columns=trip_cols), pd.DataFrame(opened, columns=open_cols)
    for col in ["buy_datetime", "sell_datetime", "buy_signal_datetime", "sell_signal_datetime"]:
        trip_frame[col] = pd.to_datetime(trip_frame[col])
    # 原函数期末股数必须与转换后的事件库存完全一致，日终估值口径允许不同。
    old_shares = {} if old_open.empty else dict(zip(old_open.lot_id, old_open.shares))
    if old_shares != {k: p["shares"] for k, p in positions.items()}:
        raise ValueError("旧函数与标准转换的期末股数不一致")
    return {"curve": curve, "events": event_frame, "trips": trip_frame, "open": open_frame,
            "holdings": pd.DataFrame(holdings, columns=["datetime", "lot_id", "symbol", "shares", "mark_price", "stop_price"]),
            "metrics": calculate_daily_metrics(curve, event_frame, trip_frame, cfg["initial_cash"], set(positions)),
            "blocked": {}, "legacy_native_metrics": old_metrics}


def run_original_portfolio(scores: pd.DataFrame, market: MarketStore, start: pd.Timestamp, end: pd.Timestamp,
                           method: str, threshold: float, mode: str, cfg: dict) -> dict:
    """对scores的method原打分按threshold运行[start,end)，mode分旧式/约束；cfg给资金与费用。

    不根据结果切换两口径。旧式直接调用既有函数；约束版沿用经过检查的共同账户引擎。
    """
    column = "prediction" if method == "method1" else "cta_trend_breakout_score"
    frame = scores.loc[(scores.datetime >= start) & (scores.datetime < end)].copy()
    if mode == "constrained":
        # 原公式由回调控制；禁用引擎额外的单向收紧吊顶，不换成近期研究的20/40或期限退出。
        rule = {"initial": 2., "trailing": 1e9, "max_days": 0, "threshold": threshold}
        return run_lot_portfolio(frame, market, start, end, column, rule, cfg, OriginalAtrExitPolicy(market))
    if mode != "legacy":
        raise ValueError("未知执行口径")
    indices = market.times.get_indexer(frame.datetime)
    assets = frame.symbol.map(market.asset_index).to_numpy(dtype=int)
    frame["entry_datetime"] = frame.datetime + pd.Timedelta(minutes=1)
    frame["entry_price"] = market.values["execution_close"][indices, assets]
    # 旧模型的费用固定卖出万五，约束版使用日期对应印花税；两者在报告中明确区分。
    config = SimpleNamespace(initial_cash=cfg["initial_cash"], lot_size=cfg["lot_size"],
        min_assets_per_rebalance=cfg["min_cross_section"], evaluation_step_bars=cfg["sample_minutes"],
        fee=FeeConfig(cfg["commission_rate"], cfg["stamp_rate_from_20230828"], cfg["transfer_rate"], cfg["min_commission"], True))
    spec = MethodSpec(method, "原XGBoost" if method == "method1" else "原XGBoost加CTA", column, threshold)
    rule = LotExitRule("atr_initial2_trail4_atr240", 240, initial_atr_multiple=2., trailing_atr_multiple=4.)
    native = run_lot_backtest(frame, spec, rule, config, CachedIndicatorStore(market), cfg["entry_top_n"], cfg["entry_equity_fraction"])
    return convert_legacy_result(native, frame, market, start, end, cfg)
