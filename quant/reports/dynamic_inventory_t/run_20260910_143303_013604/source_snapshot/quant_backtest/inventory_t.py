"""固定底仓的日内做 T 引擎；输入一分钟行情及预定底仓买单，返回回测表，不写文件。

信号只读取已完成分钟，下一交易分钟收盘成交。买入不增加当日可卖股数，底仓
期末不清仓。缺失、零量、一字线或现金不足时不会虚构恢复交易，偏差如实跨日。
行情按中国股票收盘时间标签处理，午休 11:30 的下一分钟为 13:01；不支持复权
换算和公司行动，真实涨跌停价及订单队列不可得时，一字线仅为保守阻成交代理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .annual_portfolio import calculate_daily_metrics, calculate_fee
from .strategy import BaseStrategy


DEFAULT_CONFIG = {
    "initial_cash": 1_000_000.0, "commission_rate": 0.0008,
    "min_commission": 0.0, "transfer_rate": 0.00001,
    "stamp_rate_before_20230828": 0.001, "stamp_rate_from_20230828": 0.0005,
    "slippage_rate": 0.0002, "lot_size": 100, "max_bar_amount_fraction": 0.01,
    "last_entry_time": "14:30", "flatten_signal_time": "14:54",
    "max_cycles_per_day": 2, "repurchase_buffer": 0.12,
}

EVENT_COLUMNS = [
    "datetime", "signal_time", "signal_datetime", "symbol", "lot_id", "pair_id",
    "side", "reason", "shares", "price", "reference_price", "notional", "fee",
    "slippage_cost", "cash_after", "sellable_before", "sellable_after", "event_type",
]
TRIP_COLUMNS = [
    "lot_id", "pair_id", "symbol", "direction", "first_datetime", "second_datetime",
    "buy_datetime", "sell_datetime", "buy_signal_datetime", "sell_signal_datetime",
    "buy_price", "sell_price", "shares", "buy_fee", "sell_fee", "gross_pnl", "net_pnl",
    "slippage_cost", "holding_days", "buy_reason", "sell_reason", "recovered_same_day",
]
AUDIT_COLUMNS = [
    "date", "symbol", "opening_shares", "opening_sellable", "base_manual_adds", "closing_shares",
    "target_shares", "inventory_difference", "inventory_preserved", "day_sold", "closing_sellable",
    "t1_violation_count", "cycles_started", "recovery_failure", "recovery_blocked_attempts",
    "baseline_target_shares", "baseline_inventory_difference", "stale_price",
]
HOLDING_COLUMNS = [
    "symbol", "shares", "mark_price", "market_value", "mark_datetime", "baseline_target_shares",
    "inventory_difference", "active_pair_id", "pending_direction", "pending_shares", "repurchase_reserved_cash",
]
ATTEMPT_COLUMNS = ["datetime", "signal_time", "symbol", "side", "shares", "reverse", "reason"]


@dataclass
class InventoryTStrategy(BaseStrategy):
    """A 股固定底仓策略：日内偏离累计成交量加权收盘价时建议做 T，不做空。

    rule 控制 entry_deviation、direction 和 enabled。target_position 始终为 1，
    t_action 的 +1/-1 分别表示先买后卖/先卖后买，并非空头目标仓位。
    输入仅限单一股票，输出与原索引对齐；累计 close*volume 不依赖成交量单位。
    """

    rule: dict = field(default_factory=dict)
    name: str = "inventory_t"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """按已完成单股 datetime/close/volume 分钟生成因果 VWAP 偏离信号。"""
        frame = data.copy()
        dates = pd.to_datetime(frame["datetime"])
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("做 T 信号要求单股 datetime 严格递增且无重复")
        if "symbol" in frame and frame.symbol.nunique() > 1:
            raise ValueError("InventoryTStrategy 每次仅接受一只股票")
        close = pd.to_numeric(frame.close, errors="raise")
        volume = pd.to_numeric(frame.volume, errors="raise")
        valid = close.notna() & (close > 0) & volume.notna() & (volume > 0)
        weighted = (close * volume).where(valid, 0).groupby(dates.dt.normalize()).cumsum()
        cumulative_volume = volume.where(valid, 0).groupby(dates.dt.normalize()).cumsum()
        vwap = weighted / cumulative_volume.replace(0, np.nan)
        deviation = close / vwap - 1.0
        action = pd.Series(0, index=data.index, dtype="int8")
        threshold = float(self.rule.get("entry_deviation", 0.01))
        if threshold <= 0:
            raise ValueError("entry_deviation 必须为正数")
        direction = self.rule.get("direction", "both")
        if direction not in {"both", "buy_first", "sell_first"}:
            raise ValueError("direction 仅接受 both/buy_first/sell_first")
        if self.rule.get("enabled", True):
            if direction in {"both", "buy_first"}:
                action.loc[valid & (deviation <= -threshold)] = 1
            if direction in {"both", "sell_first"}:
                action.loc[valid & (deviation >= threshold)] = -1
        return pd.DataFrame({"target_position": 1.0, "t_action": action,
                             "intraday_vwap": vwap, "vwap_deviation": deviation}, index=data.index)


def calculate_next_trading_minute(timestamp: pd.Timestamp) -> pd.Timestamp | None:
    """返回收盘标签的一分钟执行时钟；午休跳至 13:01，15:00 后不跨日下单。"""
    minute = timestamp.hour * 60 + timestamp.minute
    if minute == 11 * 60 + 30:
        return timestamp.normalize() + pd.Timedelta(hours=13, minutes=1)
    if minute >= 15 * 60 or 11 * 60 + 30 < minute < 13 * 60:
        return None
    return timestamp + pd.Timedelta(minutes=1)


def validate_inventory_inputs(bars: dict[str, pd.DataFrame], entries: pd.DataFrame,
                              dates: pd.DatetimeIndex, cfg: dict, rule: dict) -> tuple[dict, pd.DataFrame, pd.DatetimeIndex]:
    """规范化并校验输入；行情重复时间显式保留最后一行，缺失价格保留但禁止成交。"""
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    if not len(days):
        raise ValueError("做 T 回测 dates 不能为空")
    if cfg["initial_cash"] <= 0 or cfg["lot_size"] < 1:
        raise ValueError("initial_cash 与 lot_size 必须为正")
    if int(cfg["lot_size"]) != cfg["lot_size"] or cfg["lot_size"] % 100:
        raise ValueError("A 股 lot_size 必须为 100 的正整数倍")
    if not 0 < cfg["max_bar_amount_fraction"] <= 1:
        raise ValueError("max_bar_amount_fraction 必须处于 (0, 1]")
    for key in ("commission_rate", "min_commission", "transfer_rate", "slippage_rate",
                "stamp_rate_before_20230828", "stamp_rate_from_20230828", "repurchase_buffer"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} 必须为有限非负数")
    if not 0 < rule.get("trade_fraction", 0.25) <= 1 or rule.get("take_profit", 0.002) < 0:
        raise ValueError("trade_fraction 必须处于 (0,1]，take_profit 必须非负")
    prepared = {}
    for symbol, original in bars.items():
        frame = original.copy()
        required = {"datetime", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{symbol} 缺少行情列 {required.difference(frame.columns)}")
        frame["datetime"] = pd.to_datetime(frame.datetime)
        frame = frame.loc[frame.datetime.dt.normalize().isin(days)].sort_values("datetime")
        frame = frame.drop_duplicates("datetime", keep="last").reset_index(drop=True)
        for column in ["open", "high", "low", "close", "volume", "amount"]:
            if column not in frame:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if np.isinf(frame[column]).any() or (frame[column].dropna() < 0).any():
                raise ValueError(f"{symbol} 的 {column} 有负数或无穷值")
        prices = frame[["open", "high", "low", "close"]]
        if (prices == 0).any().any():
            raise ValueError(f"{symbol} 存在零价格；缺失行情应使用 NaN 或缺行")
        if ((frame.high < prices[["open", "close"]].max(axis=1)) |
                (frame.low > prices[["open", "close"]].min(axis=1)) |
                (frame.high < frame.low)).any():
            raise ValueError(f"{symbol} OHLC 关系不合法")
        frame["t_action"] = InventoryTStrategy(rule).generate_signals(frame).t_action if len(frame) else 0
        prepared[str(symbol)] = frame
    scheduled = entries.copy()
    if scheduled.empty:
        scheduled = pd.DataFrame(columns=["datetime", "symbol", "shares", "price", "reason"])
    if not {"datetime", "symbol", "shares", "price"}.issubset(scheduled.columns):
        raise ValueError("entries 必须包含 datetime/symbol/shares/price")
    scheduled["datetime"] = pd.to_datetime(scheduled.datetime)
    scheduled["symbol"] = scheduled.symbol.astype(str)
    scheduled = scheduled.sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
    if "reason" not in scheduled:
        scheduled["reason"] = "预定底仓开仓"
    if not scheduled.datetime.dt.normalize().isin(days).all():
        raise ValueError("entries 含 dates 之外的买单")
    if not set(scheduled.symbol).issubset(prepared):
        raise ValueError("entries 存在未提供行情的股票")
    if ((scheduled.shares <= 0) | (scheduled.shares % cfg["lot_size"] != 0)).any():
        raise ValueError("底仓买单股数必须为正整手")
    if not np.isfinite(scheduled.price.astype(float)).all() or (scheduled.price <= 0).any():
        raise ValueError("底仓买单价格必须为有限正数")
    return prepared, scheduled, days


def run_inventory_t(bars: dict[str, pd.DataFrame], entries: pd.DataFrame,
                    dates: pd.DatetimeIndex, cfg: dict, rule: dict) -> dict:
    """按 cfg 成本/现金约束和 rule 做 T，返回日净值、交易、配对、库存审计及指标。

    entries 的 datetime 是预定底仓实际成交时间，price 是未加滑点的行情价格；可
    包含人为加仓，均计入当日目标库存但次日才可卖。不足以执行既定底仓时抛错。
    amount 优先用于容量；无 amount 时使用 close*volume，成交量若为手则更保守。
    每日总卖出不超过日初实际库存；先买后卖的卖单同样消耗此前持有的股票。
    回补买单保留按首腿参考价加 repurchase_buffer 的现金，仍不保证极端行情成交。
    """
    cfg = {**DEFAULT_CONFIG, **cfg}
    rule = {"enabled": True, "entry_deviation": 0.01, "take_profit": 0.002,
            "trade_fraction": 0.25, "direction": "both", **rule}
    prepared, scheduled, days = validate_inventory_inputs(bars, entries, dates, cfg, rule)
    symbols = sorted(prepared)
    grouped = {symbol: {day: frame for day, frame in data.groupby(data.datetime.dt.normalize())}
               for symbol, data in prepared.items()}
    scheduled_days = {day: frame for day, frame in scheduled.groupby(scheduled.datetime.dt.normalize())}
    cash = float(cfg["initial_cash"])
    baseline_cash = cash
    positions = dict.fromkeys(symbols, 0)
    baseline_shares = dict.fromkeys(symbols, 0)
    marks, mark_times, active = {}, {}, {}
    events, trips, audits, records, attempts = [], [], [], [], []
    next_pair_id, next_base_id = 1, -1
    t_cashflow = 0.0
    completed_profit = 0.0
    lot_size = int(cfg["lot_size"])
    last_entry = pd.Timestamp(f"2000-01-01 {cfg['last_entry_time']}").time()
    flatten_time = pd.Timestamp(f"2000-01-01 {cfg['flatten_signal_time']}").time()
    if not last_entry < flatten_time < pd.Timestamp("2000-01-01 15:00").time():
        raise ValueError("必须满足 last_entry_time < flatten_signal_time < 15:00")
    if int(cfg["max_cycles_per_day"]) != cfg["max_cycles_per_day"] or cfg["max_cycles_per_day"] < 0:
        raise ValueError("max_cycles_per_day 必须为非负整数")

    def calculate_reserved_cash(except_symbol: str | None = None) -> float:
        """返回其他未恢复先卖后买组合的回补现金保留额，不为当日新买股放大卖出额度。"""
        return sum(pair["reserve"] for symbol, pair in active.items() if symbol != except_symbol)

    def calculate_execution(row: dict | None, side: str, quantity: int,
                            timestamp: pd.Timestamp, used_amount: float) -> tuple[int, float, str]:
        """只根据当前执行分钟判断可成交股数；不检查未来分钟是否能完成反向交易。"""
        if row is None or not all(np.isfinite(row[key]) for key in ("close", "high", "low", "volume")):
            return 0, 0.0, "missing_bar"
        if row["volume"] <= 0:
            return 0, 0.0, "zero_volume"
        if row["high"] == row["low"]:
            return 0, 0.0, "locked_bar"
        # 真实限价字段存在时额外执行单边约束；缺失时仅采用一字线保守代理。
        bound = row.get("limit_up" if side == "buy" else "limit_down", np.nan)
        if np.isfinite(bound) and ((side == "buy" and row["close"] >= bound) or
                                   (side == "sell" and row["close"] <= bound)):
            return 0, 0.0, "price_limit"
        reference = float(row["close"])
        price = reference * (1 + cfg["slippage_rate"] * (1 if side == "buy" else -1))
        amount = row.get("amount", reference * row["volume"])
        if not np.isfinite(amount) or amount <= 0:
            return 0, price, "missing_amount"
        remaining = max(0.0, amount * cfg["max_bar_amount_fraction"] - used_amount)
        capacity = int(remaining / reference / lot_size) * lot_size
        return min(quantity, capacity), price, "capacity" if capacity < quantity else ""

    for day in days:
        opening = positions.copy()
        sellable = opening.copy()
        additions = dict.fromkeys(symbols, 0)
        sold = dict.fromkeys(symbols, 0)
        cycles = dict.fromkeys(symbols, 0)
        t1_violations = dict.fromkeys(symbols, 0)
        recovery_attempts = dict.fromkeys(symbols, 0)
        pending = {}
        day_rows = {}
        for symbol in symbols:
            frame = grouped[symbol].get(day)
            if frame is not None:
                for row in frame.to_dict("records"):
                    day_rows.setdefault(row["datetime"], {})[symbol] = row
        entry_rows = {}
        if day in scheduled_days:
            for row in scheduled_days[day].to_dict("records"):
                entry_rows.setdefault(row["datetime"], []).append(row)
        # 确定性交易时钟使整分钟缺失不会被错误顺延成可成交的下一分钟。
        clock = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=11, minutes=30), freq="min")
        clock = clock.append(pd.date_range(day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=15), freq="min"))
        if set(entry_rows).difference(clock):
            raise ValueError("底仓成交时间不属于正常 A 股分钟交易时段")

        def record_fill(symbol: str, side: str, quantity: int, price: float, reference: float,
                        timestamp: pd.Timestamp, signal_time: pd.Timestamp, reason: str,
                        pair_id: int, event_type: str) -> dict:
            """登记实际成交现金和库存；任何卖单都必须消耗日初结转可卖股数。"""
            nonlocal cash, t_cashflow
            before = sellable[symbol]
            notional = quantity * price
            fee = calculate_fee(notional, side, timestamp, cfg)
            if side == "buy":
                if notional + fee > cash + 1e-7:
                    raise AssertionError("买入成交超过账户现金")
                positions[symbol] += quantity
                cash -= notional + fee
                flow = -notional - fee
            else:
                if quantity > sellable[symbol] or quantity > positions[symbol]:
                    t1_violations[symbol] += 1
                    raise AssertionError("卖单超过日初剩余可卖股数")
                positions[symbol] -= quantity
                sellable[symbol] -= quantity
                sold[symbol] += quantity
                cash += notional - fee
                flow = notional - fee
            if event_type == "t":
                t_cashflow += flow
            event = {"datetime": timestamp, "signal_time": signal_time, "signal_datetime": signal_time,
                     "symbol": symbol, "lot_id": pair_id, "pair_id": pair_id if event_type == "t" else 0,
                     "side": side, "reason": reason, "shares": quantity, "price": price,
                     "reference_price": reference, "notional": notional, "fee": fee,
                     "slippage_cost": quantity * abs(price - reference), "cash_after": cash,
                     "sellable_before": before, "sellable_after": sellable[symbol], "event_type": event_type}
            events.append(event)
            return event

        for timestamp in clock:
            current_rows = day_rows.get(timestamp, {})
            used = dict.fromkeys(symbols, 0.0)
            for symbol, row in current_rows.items():
                if np.isfinite(row["close"]) and row["close"] > 0:
                    marks[symbol], mark_times[symbol] = float(row["close"]), timestamp
            # 既定底仓买单优先；价格及容量必须与输入的真实分钟对应。
            for entry in entry_rows.get(timestamp, []):
                symbol, quantity = entry["symbol"], int(entry["shares"])
                row = current_rows.get(symbol)
                filled, price, blocked = calculate_execution(row, "buy", quantity, timestamp, used[symbol])
                if filled != quantity or row is None or not np.isclose(entry["price"], row["close"], rtol=1e-7, atol=1e-7):
                    raise ValueError(f"底仓计划无法按原样成交: {symbol} {timestamp} {blocked or 'price_mismatch'}")
                fee = calculate_fee(quantity * price, "buy", timestamp, cfg)
                if quantity * price + fee > baseline_cash + 1e-7:
                    raise ValueError(f"底仓计划超过纯持有基准现金，不能使用做 T 利润融资: {symbol} {timestamp}")
                if quantity * price + fee > cash - calculate_reserved_cash() + 1e-7:
                    raise ValueError(f"底仓计划现金不足: {symbol} {timestamp}")
                default_signal = timestamp - pd.Timedelta(minutes=1)
                if timestamp.time() == pd.Timestamp("2000-01-01 13:01").time():
                    default_signal = day + pd.Timedelta(hours=11, minutes=30)
                signal_time = entry.get("signal_time", entry.get("signal_datetime", default_signal))
                signal_time = pd.Timestamp(signal_time) if pd.notna(signal_time) else default_signal
                if signal_time >= timestamp:
                    raise ValueError("底仓 signal_time 必须早于实际成交时间")
                record_fill(symbol, "buy", quantity, price, row["close"], timestamp,
                            signal_time, entry["reason"], next_base_id, "baseline")
                next_base_id -= 1
                baseline_cash -= quantity * price + fee
                baseline_shares[symbol] += quantity
                additions[symbol] += quantity
                used[symbol] += quantity * row["close"]

            for symbol in symbols:
                request = pending.pop(symbol, None)
                if request is None or request["execution_time"] != timestamp:
                    if request is not None:
                        pending[symbol] = request
                    continue
                row = current_rows.get(symbol)
                is_reverse = request["reverse"]
                pair = active.get(symbol)
                side = request["side"]
                quantity = pair["shares"] if is_reverse else request["shares"]
                filled, price, blocked = calculate_execution(row, side, quantity, timestamp, used[symbol])
                if side == "sell":
                    filled = min(filled, sellable[symbol])
                if is_reverse:
                    # 反向腿整笔恢复，容量不足则继续挂起；避免伪造部分恢复后的配对利润。
                    if filled < quantity:
                        filled = 0
                        blocked = blocked or "sellable_budget"
                    if side == "buy" and filled:
                        cost = quantity * price + calculate_fee(quantity * price, side, timestamp, cfg)
                        if cost > cash - calculate_reserved_cash(symbol) + 1e-7:
                            filled, blocked = 0, "cash"
                else:
                    available_cash = cash - calculate_reserved_cash()
                    while filled > 0:
                        value = filled * price
                        fee = calculate_fee(value, side, timestamp, cfg)
                        if side == "buy":
                            affordable = value + fee <= available_cash + 1e-7
                        else:
                            reserve_price = row["close"] * (1 + cfg["repurchase_buffer"]) * (1 + cfg["slippage_rate"])
                            reserve = filled * reserve_price + calculate_fee(filled * reserve_price, "buy", timestamp, cfg)
                            affordable = reserve <= available_cash + value - fee + 1e-7
                        if affordable:
                            break
                        filled -= lot_size
                    if filled == 0:
                        blocked = blocked or "cash_or_repurchase_reserve"
                if filled == 0:
                    attempts.append({"datetime": timestamp, "signal_time": request["signal_time"], "symbol": symbol,
                                     "side": side, "shares": quantity, "reverse": is_reverse, "reason": blocked})
                    if is_reverse:
                        pair["must_recover"] = True
                        recovery_attempts[symbol] += 1
                    continue
                if is_reverse:
                    event = record_fill(symbol, side, filled, price, row["close"], timestamp,
                                        request["signal_time"], request["reason"], pair["pair_id"], "t")
                    first = pair["first_event"]
                    buy, sell = (first, event) if first["side"] == "buy" else (event, first)
                    gross = filled * (sell["price"] - buy["price"])
                    profit = gross - buy["fee"] - sell["fee"]
                    completed_profit += profit
                    trips.append({"lot_id": pair["pair_id"], "pair_id": pair["pair_id"], "symbol": symbol,
                                  "direction": pair["direction"], "first_datetime": first["datetime"], "second_datetime": timestamp,
                                  "buy_datetime": buy["datetime"], "sell_datetime": sell["datetime"],
                                  "buy_signal_datetime": buy["signal_time"], "sell_signal_datetime": sell["signal_time"],
                                  "buy_price": buy["price"], "sell_price": sell["price"], "shares": filled,
                                  "buy_fee": buy["fee"], "sell_fee": sell["fee"], "gross_pnl": gross, "net_pnl": profit,
                                  "slippage_cost": buy["slippage_cost"] + sell["slippage_cost"],
                                  "holding_days": (timestamp - first["datetime"]).total_seconds() / 86400,
                                  "buy_reason": buy["reason"], "sell_reason": sell["reason"],
                                  "recovered_same_day": first["datetime"].normalize() == day})
                    del active[symbol]
                else:
                    direction = "buy_first" if side == "buy" else "sell_first"
                    event = record_fill(symbol, side, filled, price, row["close"], timestamp,
                                        request["signal_time"], "VWAP 下偏先买" if side == "buy" else "VWAP 上偏先卖",
                                        next_pair_id, "t")
                    reserve = 0.0
                    if side == "sell":
                        reserve_price = row["close"] * (1 + cfg["repurchase_buffer"]) * (1 + cfg["slippage_rate"])
                        reserve = filled * reserve_price + calculate_fee(filled * reserve_price, "buy", timestamp, cfg)
                    active[symbol] = {"pair_id": next_pair_id, "direction": direction, "shares": filled,
                                      "first_event": event, "reserve": reserve, "must_recover": False}
                    next_pair_id += 1
                    cycles[symbol] += 1
                used[symbol] += filled * row["close"]

            execution_time = calculate_next_trading_minute(timestamp)
            if execution_time is None:
                continue
            for symbol in symbols:
                pair = active.get(symbol)
                row = current_rows.get(symbol)
                if pair is not None:
                    side = "sell" if pair["direction"] == "buy_first" else "buy"
                    first = pair["first_event"]
                    must_recover = pair["must_recover"] or first["datetime"].normalize() < day or timestamp.time() >= flatten_time
                    hit_target = False
                    if row is not None and np.isfinite(row["close"]):
                        reverse_price = row["close"] * (1 + cfg["slippage_rate"] * (1 if side == "buy" else -1))
                        fee = calculate_fee(pair["shares"] * reverse_price, side, timestamp, cfg)
                        gross = pair["shares"] * (reverse_price - first["price"]) * (1 if side == "sell" else -1)
                        hit_target = (gross - first["fee"] - fee) / first["notional"] >= rule["take_profit"]
                    if must_recover or hit_target:
                        pending[symbol] = {"execution_time": execution_time, "signal_time": timestamp, "side": side,
                                           "reverse": True, "reason": "收盘/失败续单恢复股数" if must_recover else "价差覆盖成本后回转"}
                elif (rule["enabled"] and row is not None and execution_time.time() <= last_entry
                      and cycles[symbol] < cfg["max_cycles_per_day"] and sellable[symbol] >= lot_size):
                    action = row["t_action"]
                    quantity = min(int(opening[symbol] * rule["trade_fraction"] / lot_size) * lot_size, sellable[symbol])
                    if action and quantity >= lot_size:
                        pending[symbol] = {"execution_time": execution_time, "signal_time": timestamp,
                                           "side": "buy" if action > 0 else "sell", "reverse": False, "shares": quantity}

        held_value = sum(quantity * marks[symbol] for symbol, quantity in positions.items() if quantity)
        baseline_value = sum(quantity * marks[symbol] for symbol, quantity in baseline_shares.items() if quantity)
        equity = cash + held_value
        baseline_equity = baseline_cash + baseline_value
        end_timestamp = day + pd.Timedelta(hours=15)
        stale = sum(quantity > 0 and mark_times.get(symbol) != end_timestamp for symbol, quantity in positions.items())
        records.append({"datetime": end_timestamp, "equity": equity, "cash": cash, "holdings_value": held_value,
                        "exposure": held_value / equity if equity else 0.0,
                        "open_lots": sum(quantity > 0 for quantity in positions.values()),
                        "holding_count": sum(quantity > 0 for quantity in positions.values()),
                        "stale_price_count": stale, "baseline_equity": baseline_equity,
                        "t_added_equity": equity - baseline_equity, "t_realized_pnl": completed_profit,
                        "pending_pair_count": len(active), "reserved_cash": calculate_reserved_cash()})
        for symbol in symbols:
            if not (opening[symbol] or positions[symbol] or additions[symbol] or symbol in active):
                continue
            target = opening[symbol] + additions[symbol]
            mismatch = positions[symbol] - target
            audits.append({"date": day, "symbol": symbol, "opening_shares": opening[symbol],
                           "opening_sellable": opening[symbol], "base_manual_adds": additions[symbol],
                           "closing_shares": positions[symbol], "target_shares": target,
                           "inventory_difference": mismatch, "inventory_preserved": mismatch == 0,
                           "day_sold": sold[symbol], "closing_sellable": sellable[symbol],
                           "t1_violation_count": t1_violations[symbol], "cycles_started": cycles[symbol],
                           "recovery_failure": symbol in active, "recovery_blocked_attempts": recovery_attempts[symbol],
                           "baseline_target_shares": baseline_shares[symbol],
                           "baseline_inventory_difference": positions[symbol] - baseline_shares[symbol],
                           "stale_price": mark_times.get(symbol) != end_timestamp})
        if cash < -1e-6 or any(quantity < 0 for quantity in positions.values()):
            raise AssertionError("做 T 引擎产生负现金或空头库存")

    curve = pd.DataFrame(records)
    curve["return"] = curve.equity.pct_change(fill_method=None)
    curve.loc[0, "return"] = curve.equity.iloc[0] / cfg["initial_cash"] - 1
    curve["net_value"] = curve.equity / cfg["initial_cash"]
    curve["cumulative_return"] = curve.net_value - 1
    curve["drawdown"] = curve.equity / np.maximum.accumulate(np.r_[cfg["initial_cash"], curve.equity])[1:] - 1
    event_frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    trip_frame = pd.DataFrame(trips, columns=TRIP_COLUMNS)
    audit_frame = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    holdings = []
    for symbol in symbols:
        if positions[symbol] or symbol in active:
            pair = active.get(symbol, {})
            holdings.append({"symbol": symbol, "shares": positions[symbol], "mark_price": marks.get(symbol, np.nan),
                             "market_value": positions[symbol] * marks.get(symbol, 0),
                             "mark_datetime": mark_times.get(symbol), "baseline_target_shares": baseline_shares[symbol],
                             "inventory_difference": positions[symbol] - baseline_shares[symbol],
                             "active_pair_id": pair.get("pair_id", 0), "pending_direction": pair.get("direction", ""),
                             "pending_shares": pair.get("shares", 0), "repurchase_reserved_cash": pair.get("reserve", 0)})
    metrics = calculate_daily_metrics(curve, event_frame, trip_frame, cfg["initial_cash"])
    metrics.update({"strategy": "inventory_t" if rule["enabled"] else "buy_and_hold",
                    "total_t_net_profit": float(trip_frame.net_pnl.sum()) if len(trip_frame) else 0.0,
                    "total_t_marked_pnl": float(curve.t_added_equity.iloc[-1]),
                    "t_cashflow": t_cashflow, "t_pair_count": len(trip_frame),
                    "t_fee": float(event_frame.loc[event_frame.event_type == "t", "fee"].sum()),
                    "inventory_mismatch_days": int(audit_frame.loc[~audit_frame.inventory_preserved, "date"].nunique()) if len(audit_frame) else 0,
                    "inventory_mismatch_rows": int((~audit_frame.inventory_preserved).sum()) if len(audit_frame) else 0,
                    "recovery_failure_days": int(audit_frame.loc[audit_frame.recovery_failure, "date"].nunique()) if len(audit_frame) else 0,
                    "pending_pairs": len(active), "t1_violation_count": int(audit_frame.t1_violation_count.sum()) if len(audit_frame) else 0,
                    "stale_valuation_days": int((curve.stale_price_count > 0).sum()), "blocked_order_count": len(attempts),
                    "baseline_total_return": float(curve.baseline_equity.iloc[-1] / cfg["initial_cash"] - 1),
                    "no_terminal_liquidation": True})
    return {"equity_curve": curve, "trade_events": event_frame, "round_trips": trip_frame,
            "open_positions": pd.DataFrame(holdings, columns=HOLDING_COLUMNS), "daily_inventory_audit": audit_frame,
            "order_attempts": pd.DataFrame(attempts, columns=ATTEMPT_COLUMNS), "metrics": metrics}
