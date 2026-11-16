"""年度研究的独立买入批次引擎：仅做多、T+1、下一分钟成交、费用和容量约束。

输入为共同时钟行情与事先训练的模型分数，不接受未来标签。输出净值、订单事件、
分批平仓和未平仓记录。不连接券商，不强行按年度末市价平仓。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PositionLot:
    """独立买入批次；记录股票列号、数量、价格、成本、时间、最高价及止损状态。"""

    lot_id: int
    asset: int
    shares: int
    buy_time: pd.Timestamp
    buy_price: float
    buy_fee: float
    buy_signal: pd.Timestamp
    buy_score: float
    high_watermark: float
    stop_price: float
    entry_index: int
    pending_reason: str = ""


class MarketStore:
    """将 market 共同时钟行情转为矩阵，供多轮策略复用；只向前填充估值，不填充成交。"""

    def __init__(self, market: pd.DataFrame):
        """market 需包含 annual_panel 输出的时间、股票、当前行情及下一分钟执行行情。"""
        self.times = pd.DatetimeIndex(sorted(market.datetime.unique()))
        self.symbols = sorted(market.symbol.unique())
        self.asset_index = {s: i for i, s in enumerate(self.symbols)}
        self.time_index = {t: i for i, t in enumerate(self.times)}
        indexed = market.set_index(["datetime", "symbol"])
        self.values = {}
        for col in market.columns.difference(["datetime", "symbol"]):
            matrix = indexed[col].unstack("symbol").reindex(index=self.times, columns=self.symbols)
            self.values[col] = matrix.to_numpy(dtype="float64")
        self.marks = pd.DataFrame(self.values["close"]).ffill().to_numpy()
        self.days = self.times.normalize()
        self.day_index = pd.factorize(self.days)[0]


def calculate_fee(value: float, side: str, timestamp: pd.Timestamp, cfg: dict) -> float:
    """按 cfg 和 timestamp 的历史费率计算 value 成交额的 side 买/卖费用，免5可配置。"""
    fee = max(value * cfg["commission_rate"], cfg["min_commission"]) + value * cfg["transfer_rate"]
    if side == "sell":
        key = "stamp_rate_from_20230828" if timestamp >= pd.Timestamp("2023-08-28") else "stamp_rate_before_20230828"
        fee += value * cfg[key]
    return float(fee)


def calculate_daily_metrics(curve: pd.DataFrame, events: pd.DataFrame, trips: pd.DataFrame, initial: float,
                            open_lot_ids: set[int] | None = None) -> dict:
    """从curve日终权益计算指标；events为交易，trips为平仓片段，initial为初始现金。

    open_lot_ids列出尚未完全退出的批次，计算批次胜率时排除这些未完成交易。
    """
    daily = curve.groupby(curve.datetime.dt.normalize(), sort=True).last()
    equity = daily.equity.astype(float)
    returns = equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.iloc[0] / initial - 1
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    total = float(equity.iloc[-1] / initial - 1)
    # 纳入初始现金，避免漏掉首笔费用和首日亏损。
    peaks = np.maximum.accumulate(np.r_[initial, equity.to_numpy()])[1:]
    mdd = float((equity.to_numpy() / peaks - 1).min())
    calendar_days = max((equity.index[-1] - equity.index[0]).days + 1, 1)
    annual = (1 + total) ** (365.25 / calendar_days) - 1 if total > -1 else -1.0
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
    buys = events.loc[events.side == "buy"] if len(events) else events
    sells = events.loc[events.side == "sell"] if len(events) else events
    completed = trips.loc[~trips.lot_id.isin(open_lot_ids or set())] if len(trips) else trips
    return {"total_return": total, "annual_return": annual, "annual_volatility": std * np.sqrt(252),
            "sharpe": float(returns.mean() / std * np.sqrt(252)) if std else 0.0,
            "sortino": float(returns.mean() / downside * np.sqrt(252)) if downside else 0.0,
            "max_drawdown": mdd, "calmar": annual / abs(mdd) if mdd else 0.0,
            "buy_count": len(buys), "sell_count": len(sells), "closed_lot_count": int(completed.lot_id.nunique()) if len(completed) else 0,
            "trade_lots": float(events.shares.sum() / 100) if len(events) else 0.0,
            "win_rate": float((completed.groupby("lot_id").net_pnl.sum() > 0).mean()) if len(completed) else 0.0,
            "mean_holding_days": float(completed.holding_days.mean()) if len(completed) else 0.0,
            "turnover": float(events.notional.sum() / equity.mean()) if len(events) else 0.0,
            "fees": float(events.fee.sum()) if len(events) else 0.0,
            "mean_exposure": float(curve.exposure.mean()), "short_exposure": 0,
            "trading_days": len(daily), "start": str(equity.index[0].date()), "end": str(equity.index[-1].date()),
            "open_lots": int(curve.open_lots.iloc[-1]), "signal_lag_bars": 1}


def run_lot_portfolio(scores: pd.DataFrame, market: MarketStore, start: pd.Timestamp,
                      end: pd.Timestamp, score_col: str, rule: dict, cfg: dict, exit_policy=None) -> dict:
    """在[start,end)对 scores 的 score_col 按 rule 回测；market/cfg 提供行情及成本资金约束。

    所有决策只看信号时间；成交按下一分钟收盘加滑点，缺失/零量取消或延后。
    跌出股票池不触发平仓；各批次止损只能收紧。期末保留持仓按市值计价。
    exit_policy可选：接收当前时钟下标和批次列表，返回lot_id到中文退出原因的映射；
    回调只能读取已完成行情，不允许访问未来执行价或标签。未提供时保持原ATR行为。
    """
    times = market.times
    positions: list[PositionLot] = []
    cash = float(cfg["initial_cash"])
    events, trips, records, holdings = [], [], [], []
    next_id = 1
    score_groups = {dt: g for dt, g in scores.groupby("datetime", sort=False)}
    indices = np.flatnonzero((times >= start) & (times < end))
    if len(indices) == 0:
        raise ValueError("回测区间没有共同时钟行情")
    blocked = {"missing_or_zero_volume": 0, "price_limit": 0, "capacity": 0, "t1": 0}
    for index in indices:
        timestamp = times[index]
        day = timestamp.normalize()
        marks = market.marks[index]
        signal_equity = cash + sum(p.shares * marks[p.asset] for p in positions)
        current = market.values["close"][index]
        atr = market.values["atr"][index]
        high = market.values["interval_high"][index]
        exit_reasons = {}
        model_reasons = exit_policy(index, positions) if exit_policy is not None else {}
        for lot in positions:
            asset = lot.asset
            # 首个监控区间可能包含买入成交前的分钟内高点，保守只采纳当前收盘。
            observed_high = current[asset] if index == lot.entry_index + 1 else high[asset]
            if np.isfinite(observed_high):
                lot.high_watermark = max(lot.high_watermark, observed_high)
            if np.isfinite(atr[asset]):
                lot.stop_price = max(lot.stop_price, lot.high_watermark - rule["trailing"] * atr[asset])
            reason = lot.pending_reason
            reason = reason or model_reasons.get(lot.lot_id, "")
            if np.isfinite(current[asset]) and current[asset] <= lot.stop_price:
                reason = "触及独立批次ATR移动止损"
            held_days = market.day_index[index] - market.day_index[lot.entry_index]
            if rule["max_days"] and held_days >= rule["max_days"]:
                reason = reason or "达到最大持仓交易日保护"
            if reason:
                lot.pending_reason = reason
                if day > lot.buy_time.normalize():
                    exit_reasons[lot.lot_id] = reason
                else:
                    blocked["t1"] += 1
        execution_time = timestamp + pd.Timedelta(minutes=1)
        executable = timestamp.hour != 15 and execution_time < end
        execution_close = market.values["execution_close"][index]
        execution_volume = market.values["execution_volume"][index]
        previous_close = market.values["previous_close"][index]
        capacity = np.nan_to_num(market.values["execution_amount"][index], nan=0.0) * cfg["max_bar_amount_fraction"]
        used_capacity: dict[int, float] = {}

        def calculate_fill(asset: int, side: str, requested: int, budget: float) -> tuple[int, float]:
            """对 asset 的 side 请求按 requested 股/budget 元、成交额和涨跌停代理限制模拟成交。"""
            price = execution_close[asset]
            if not executable or not np.isfinite(price) or execution_volume[asset] <= 0:
                blocked["missing_or_zero_volume"] += 1
                return 0, 0.0
            # 原始文件缺少真实涨跌停价；对主板使用前收盘±9.5%的保守阻成交代理。
            ratio = price / previous_close[asset]
            if (side == "buy" and ratio >= 1.095) or (side == "sell" and ratio <= 0.905):
                blocked["price_limit"] += 1
                return 0, 0.0
            price *= 1 + cfg["slippage_rate"] * (1 if side == "buy" else -1)
            available = max(0.0, capacity[asset] - used_capacity.get(asset, 0.0))
            shares = min(requested, int(min(budget, available) / price / cfg["lot_size"]) * cfg["lot_size"])
            if shares < requested:
                blocked["capacity"] += 1
            return shares, float(price)

        if executable:
            for lot in positions.copy():
                if lot.lot_id not in exit_reasons:
                    continue
                shares, price = calculate_fill(lot.asset, "sell", lot.shares, float("inf"))
                if not shares:
                    continue
                value = shares * price
                fee = calculate_fee(value, "sell", execution_time, cfg)
                allocated_buy_fee = lot.buy_fee * shares / lot.shares
                cash += value - fee
                used_capacity[lot.asset] = used_capacity.get(lot.asset, 0) + value
                events.append({"datetime": execution_time, "signal_datetime": timestamp, "symbol": market.symbols[lot.asset],
                               "lot_id": lot.lot_id, "side": "sell", "shares": shares, "price": price, "notional": value,
                               "fee": fee, "reason": exit_reasons[lot.lot_id], "signal_equity": signal_equity})
                trips.append({"lot_id": lot.lot_id, "symbol": market.symbols[lot.asset], "buy_datetime": lot.buy_time,
                              "sell_datetime": execution_time, "buy_signal_datetime": lot.buy_signal, "sell_signal_datetime": timestamp,
                              "buy_price": lot.buy_price, "sell_price": price, "shares": shares,
                              "buy_fee": allocated_buy_fee, "sell_fee": fee,
                              "net_pnl": shares * (price - lot.buy_price) - allocated_buy_fee - fee,
                              "holding_days": (execution_time - lot.buy_time).total_seconds() / 86400,
                              "buy_reason": "当期前5且通过训练期校准门控", "sell_reason": exit_reasons[lot.lot_id]})
                lot.shares -= shares
                lot.buy_fee -= allocated_buy_fee
                if lot.shares == 0:
                    positions.remove(lot)
            group = score_groups.get(timestamp)
            if group is not None and len(group) >= cfg["min_cross_section"]:
                selected = group.sort_values([score_col, "symbol"], ascending=[False, True]).head(cfg["entry_top_n"])
                for row in selected.itertuples(index=False):
                    score = getattr(row, score_col)
                    if not np.isfinite(score) or score < rule["threshold"]:
                        continue
                    asset = market.asset_index[row.symbol]
                    if any(p.asset == asset and p.buy_time.normalize() >= day for p in positions):
                        continue
                    if not np.isfinite(atr[asset]) or atr[asset] <= 0 or not np.isfinite(current[asset]):
                        continue
                    budget = signal_equity * cfg["entry_equity_fraction"]
                    requested = int(budget / current[asset] / cfg["lot_size"]) * cfg["lot_size"]
                    affordable = max(0.0, cash - cfg["min_commission"]) / (1 + cfg["commission_rate"] + cfg["transfer_rate"])
                    shares, price = calculate_fill(asset, "buy", requested, min(budget, affordable))
                    if not shares:
                        continue
                    value = shares * price
                    fee = calculate_fee(value, "buy", execution_time, cfg)
                    if value + fee > cash + 1e-7:
                        raise AssertionError("买入计划超过现金")
                    cash -= value + fee
                    used_capacity[asset] = used_capacity.get(asset, 0) + value
                    stop = price - rule["initial"] * atr[asset]
                    positions.append(PositionLot(next_id, asset, shares, execution_time, price, fee, timestamp, score, price, stop, index))
                    events.append({"datetime": execution_time, "signal_datetime": timestamp, "symbol": row.symbol,
                                   "lot_id": next_id, "side": "buy", "shares": shares, "price": price, "notional": value,
                                   "fee": fee, "reason": "当期前5且通过训练期校准门控", "signal_equity": signal_equity})
                    next_id += 1
        after_marks = np.where(np.isfinite(execution_close), execution_close, marks) if executable else marks
        held_value = sum(p.shares * after_marks[p.asset] for p in positions)
        equity = cash + held_value
        if not np.isfinite(equity) or cash < -1e-6 or equity <= 0:
            raise AssertionError("现金或权益账本异常")
        record_time = execution_time if executable else timestamp
        records.append({"datetime": record_time, "equity": equity, "cash": cash, "exposure": held_value / equity,
                        "open_lots": len(positions), "holding_count": len({p.asset for p in positions})})
        if timestamp.hour == 15:
            for lot in positions:
                holdings.append({"datetime": timestamp, "lot_id": lot.lot_id, "symbol": market.symbols[lot.asset],
                                 "shares": lot.shares, "mark_price": marks[lot.asset], "stop_price": lot.stop_price})
    curve = pd.DataFrame(records)
    event_columns = ["datetime", "signal_datetime", "symbol", "lot_id", "side", "shares", "price", "notional", "fee", "reason", "signal_equity"]
    trip_columns = ["lot_id", "symbol", "buy_datetime", "sell_datetime", "buy_signal_datetime", "sell_signal_datetime", "buy_price", "sell_price", "shares", "buy_fee", "sell_fee", "net_pnl", "holding_days", "buy_reason", "sell_reason"]
    events_frame = pd.DataFrame(events, columns=event_columns)
    trips_frame = pd.DataFrame(trips, columns=trip_columns)
    open_frame = pd.DataFrame([{"lot_id": p.lot_id, "symbol": market.symbols[p.asset], "buy_datetime": p.buy_time,
                                "shares": p.shares, "buy_price": p.buy_price, "buy_fee": p.buy_fee,
                                "mark_price": market.marks[indices[-1], p.asset], "stop_price": p.stop_price,
                                "pending_reason": p.pending_reason} for p in positions],
                              columns=["lot_id", "symbol", "buy_datetime", "shares", "buy_price", "buy_fee", "mark_price", "stop_price", "pending_reason"])
    return {"curve": curve, "events": events_frame, "trips": trips_frame, "open": open_frame,
            "holdings": pd.DataFrame(holdings, columns=["datetime", "lot_id", "symbol", "shares", "mark_price", "stop_price"]),
            "metrics": calculate_daily_metrics(curve, events_frame, trips_frame, cfg["initial_cash"], {p.lot_id for p in positions}), "blocked": blocked}
