"""周度等权底仓加日内做 T：输入冻结选股、日行情与稀疏分钟执行样本，返回审计表。

开仓用真实首分钟 open，在 09:30 模拟成交；周度退出为事先指定日期 15:00 的
收盘订单。首末分钟量价只用于成交模拟。普通集合竞价平价线可成交；真实限价
优先，缺失时用前收盘及方向性的保守限价代理。T 首腿下一分钟执行，14:55 原
数量恢复，失败再尝试实际收盘、次日开盘/收盘，绝不使用未来可恢复性筛首腿。
仅做多、无杠杆、100 股整手；所有卖出扣减昨日库存，末日禁止新 T 以保留清仓
额度。纯持有账户独立记账并决定以后各周基础买单，T 盈亏不放大底仓计划。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .annual_portfolio import calculate_daily_metrics, calculate_fee
from .inventory_t import DEFAULT_CONFIG, calculate_next_trading_minute


EVENT_COLUMNS = ["datetime", "signal_datetime", "signal_time", "symbol", "plan_id", "lot_id", "pair_id",
                 "side", "shares", "planned_shares", "price", "reference_price", "notional", "fee", "slippage_cost", "reason",
                 "event_type", "cash_after", "sellable_before", "sellable_after", "account"]
TRIP_COLUMNS = ["lot_id", "pair_id", "symbol", "direction", "shares", "first_datetime", "second_datetime",
                "buy_datetime", "sell_datetime", "buy_price", "sell_price", "buy_fee", "sell_fee",
                "gross_pnl", "net_pnl", "slippage_cost", "holding_days", "recovered_same_day", "reason"]
AUDIT_COLUMNS = ["date", "symbol", "opening_shares", "opening_sellable", "weekly_buys", "weekly_sells",
                 "weekly_buy_target", "weekly_sell_target", "target_shares", "closing_shares",
                 "inventory_difference", "inventory_preserved", "day_sold", "closing_sellable",
                 "baseline_closing_shares", "baseline_inventory_difference", "t1_violation_count",
                 "recovery_failure", "weekly_liquidation_failure", "cycles_started", "stale_price"]
ATTEMPT_COLUMNS = ["datetime", "signal_datetime", "symbol", "plan_id", "side", "requested_shares",
                   "filled_shares", "event_type", "reason", "account"]
POSITION_COLUMNS = ["plan_id", "symbol", "trade_week", "entry_date", "exit_date", "shares",
                    "baseline_shares", "mark_price", "market_value", "mark_datetime", "inventory_difference",
                    "overdue_weekly_exit", "pending_pair_id", "pending_direction", "pending_shares"]


def prepare_weekly_inputs(selections: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame,
                          decisions: pd.DataFrame, cfg: dict) -> tuple:
    """校验并复制冻结输入；日期不齐保留缺失，拒绝重复键及不合法的 T 延迟时间。"""
    if cfg["initial_cash"] <= 0 or not 0 < cfg["base_fraction"] <= 1 or int(cfg["top_n"]) < 1:
        raise ValueError("initial_cash、base_fraction、top_n 配置非法")
    if cfg["lot_size"] < 100 or cfg["lot_size"] % 100 or int(cfg["lot_size"]) != cfg["lot_size"]:
        raise ValueError("lot_size 必须为 100 的正整数倍")
    if not 0 < cfg["max_bar_amount_fraction"] <= 1:
        raise ValueError("max_bar_amount_fraction 必须处于 (0, 1]")
    for key in ["commission_rate", "min_commission", "transfer_rate", "stamp_rate_before_20230828",
                "stamp_rate_from_20230828", "slippage_rate", "repurchase_buffer"]:
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} 必须有限且非负")
    selected = selections.copy()
    required = {"trade_week", "entry_date", "exit_date", "symbol", "rank"}
    if not required.issubset(selected):
        raise ValueError(f"selections 缺列 {required.difference(selected.columns)}")
    for key in ["trade_week", "entry_date", "exit_date"]:
        selected[key] = pd.to_datetime(selected[key]).dt.normalize()
    selected["symbol"] = selected.symbol.astype(str)
    selected = selected.sort_values(["entry_date", "rank", "symbol"], kind="stable").reset_index(drop=True)
    if selected.duplicated(["trade_week", "symbol"]).any():
        raise ValueError("同一周同一股票不能重复入选")
    if (selected.exit_date <= selected.entry_date).any():
        raise ValueError("每周须至少两个已知交易日，才能执行 T+1 周度清仓")
    if (selected.groupby("trade_week").size() > cfg["top_n"]).any():
        raise ValueError("某周 selections 股票数超过 cfg.top_n")
    selected["plan_id"] = np.arange(1, len(selected) + 1)
    prices = daily.copy()
    required = {"date", "symbol", "open", "close", "open_amount", "open_high", "open_low",
                "close_amount", "close_high", "close_low"}
    if not required.issubset(prices):
        raise ValueError(f"daily 缺列 {required.difference(prices.columns)}")
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    prices["symbol"] = prices.symbol.astype(str)
    prices = prices.sort_values(["date", "symbol"]).reset_index(drop=True)
    if prices.duplicated(["date", "symbol"]).any() or prices.empty:
        raise ValueError("daily 必须非空且 date/symbol 唯一")
    for key in required.difference({"date", "symbol"}):
        prices[key] = pd.to_numeric(prices[key], errors="raise")
        if np.isinf(prices[key]).any() or (prices[key].dropna() < 0).any():
            raise ValueError(f"daily.{key} 存在负数或无穷值")
    dates = pd.DatetimeIndex(prices.date.unique()).sort_values()
    if not selected.entry_date.isin(dates).all() or not selected.exit_date.isin(dates).all():
        raise ValueError("selections 的开平仓日期须属于提供的行情日历")
    if decisions.empty:
        return selected, prices, pd.DataFrame(columns=["date"]), dates
    signals, samples = decisions.copy(), intraday.copy()
    if not {"datetime", "symbol", "action", "fraction"}.issubset(signals):
        raise ValueError("decisions 须包含 datetime/symbol/action/fraction")
    if not {"datetime", "symbol", "execution_time", "execution_price", "execution_high", "execution_low", "execution_amount"}.issubset(samples):
        raise ValueError("intraday 缺少信号及下一分钟成交字段")
    for frame in (signals, samples):
        frame["datetime"] = pd.to_datetime(frame.datetime)
        frame["symbol"] = frame.symbol.astype(str)
        if frame.duplicated(["datetime", "symbol"]).any():
            raise ValueError("intraday/decisions 的 datetime/symbol 必须唯一")
    if not signals.action.isin([-1, 0, 1]).all() or not signals.fraction.between(0, 1).all():
        raise ValueError("action 须为 -1/0/+1，fraction 须处于 [0,1]")
    samples = samples.drop(columns=["action", "fraction"], errors="ignore")
    requests = signals[["datetime", "symbol", "action", "fraction"]].merge(
        samples, on=["datetime", "symbol"], how="left", validate="one_to_one")
    requests["date"] = requests.datetime.dt.normalize()
    requests["execution_time"] = pd.to_datetime(requests.execution_time)
    for row in requests.loc[requests.execution_time.notna()].itertuples(index=False):
        if row.execution_time != calculate_next_trading_minute(row.datetime):
            raise ValueError("T 首腿必须按信号后的下一交易分钟执行")
    return selected, prices, requests.sort_values(["execution_time", "symbol"]), dates


def run_weekly_portfolio(selections: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame,
                         decisions: pd.DataFrame, cfg: dict) -> dict:
    """回测周度冻结选股与训练冻结的做 T 决策，返回标准结果及 baseline_events。

    每个基础买单预算固定为基准权益×base_fraction/top_n，不把缺股/容量空缺分给
    其他股票。baseline 独立决定基础订单，实际账户不能使用额外 T 利润扩大订单；
    若实际现金/容量/T+1 导致基础买卖偏离，则记录 base_plan_mismatches。T 配对利润
    单列，t_added_equity 还包括未恢复库存及后续基础计划偏离的真实市值影响。
    """
    cfg = {**DEFAULT_CONFIG, "base_fraction": 0.95, "top_n": 100, "slippage_rate": 0.0005,
           "max_bar_amount_fraction": 0.05, "flat_st_limit_rate": 0.05, "nonflat_limit_proxy": 0.098, **cfg}
    selected, prices, requests, dates = prepare_weekly_inputs(selections, daily, intraday, decisions, cfg)
    plan_meta = {int(row.plan_id): row._asdict() for row in selected.itertuples(index=False)}
    plan_ids = list(plan_meta)
    symbols = sorted(set(prices.symbol).union(selected.symbol))
    plans_by_symbol = {symbol: [plan_id for plan_id in plan_ids if plan_meta[plan_id]["symbol"] == symbol] for symbol in symbols}
    daily_groups = {date: frame.set_index("symbol").to_dict("index") for date, frame in prices.groupby("date")}
    request_groups = {date: frame.to_dict("records") for date, frame in requests.groupby("date")}
    entry_groups = {date: frame.plan_id.astype(int).tolist() for date, frame in selected.groupby("entry_date")}
    actual = {"name": "actual", "cash": float(cfg["initial_cash"]), "qty": dict.fromkeys(plan_ids, 0), "events": []}
    baseline = {"name": "baseline", "cash": float(cfg["initial_cash"]), "qty": dict.fromkeys(plan_ids, 0), "events": []}
    marks, mark_times, previous_marks, active = {}, {}, {}, {}
    attempts, trips, audits, curves, mismatches, entry_plans = [], [], [], [], [], []
    lot, next_pair = int(cfg["lot_size"]), 1
    flat_fills = 0
    limit_unknown_fills = 0
    t_cashflow = 0.0
    last_entry_time = pd.Timestamp(f"2000-01-01 {cfg['last_entry_time']}").time()

    def aggregate_quantities(book: dict) -> dict:
        """汇总独立周度计划的真实股份；保持计划归属以处理跨周滞留批次。"""
        return {symbol: sum(book["qty"][pid] for pid in ids) for symbol, ids in plans_by_symbol.items()}

    def calculate_reserves(exclude: str | None = None) -> float:
        """保留其他先卖组合的回购资金，避免用卖出所得无限开新组合。"""
        return sum(pair["reserve"] for symbol, pair in active.items() if symbol != exclude)

    def create_quote(row: dict | None, prefix: str, symbol: str) -> dict:
        """从当前真实开盘/收盘/分钟执行数据提取成交信息，缺字段保留缺失。"""
        row = row or {}
        price_key = prefix if prefix in {"open", "close"} else f"{prefix}_price"
        previous = row.get("previous_close", row.get("prev_close", previous_marks.get(symbol, np.nan)))
        if not np.isfinite(previous):
            previous = previous_marks.get(symbol, np.nan)
        return {"price": row.get(price_key, np.nan), "high": row.get(f"{prefix}_high", np.nan),
                "low": row.get(f"{prefix}_low", np.nan), "amount": row.get(f"{prefix}_amount", np.nan),
                "volume": row.get(f"{prefix}_volume", np.nan), "previous_close": previous,
                "limit_up": row.get(f"{prefix}_limit_up", row.get("limit_up", np.nan)),
                "limit_down": row.get(f"{prefix}_limit_down", row.get("limit_down", np.nan))}

    def calculate_quote_fill(quote: dict, side: str, quantity: int, used: float) -> tuple[int, float, str]:
        """只用正在成交分钟的数据模拟价格/容量/方向性限价；普通平价集合竞价允许成交。"""
        if not all(np.isfinite(quote[key]) for key in ["price", "high", "low", "amount"]):
            return 0, 0.0, "missing_quote"
        raw = float(quote["price"])
        if raw <= 0 or quote["amount"] <= 0 or (np.isfinite(quote["volume"]) and quote["volume"] <= 0):
            return 0, 0.0, "zero_price_or_volume"
        if quote["high"] < quote["low"] or raw < quote["low"] - 1e-6 or raw > quote["high"] + 1e-6:
            return 0, 0.0, "invalid_ohlc"
        limit = quote["limit_up"] if side == "buy" else quote["limit_down"]
        direction = 1 if side == "buy" else -1
        if np.isfinite(limit) and limit > 0:
            if (raw - limit) * direction >= -1e-8:
                return 0, 0.0, "price_limit"
        elif np.isfinite(quote["previous_close"]) and quote["previous_close"] > 0:
            # 集合竞价经常 high==low；只在价位恰好匹配按分四舍五入的5%候选限价
            # 时保守拦截，不能把6%/7%/8%的正常竞价都误作ST涨跌停。
            candidate_st = np.floor(quote["previous_close"] * (1 + direction * cfg["flat_st_limit_rate"]) * 100 + 0.5 + 1e-8) / 100
            near_st_limit = quote["high"] == quote["low"] and abs(raw - candidate_st) <= 1e-6
            if (raw / quote["previous_close"] - 1) * direction >= cfg["nonflat_limit_proxy"] or near_st_limit:
                return 0, 0.0, "conservative_price_limit_proxy"
        capacity = int(max(quote["amount"] * cfg["max_bar_amount_fraction"] - used, 0.0) / raw / lot) * lot
        price = raw * (1 + cfg["slippage_rate"] * direction)
        return min(quantity, capacity), price, "capacity" if capacity < quantity else ""

    def calculate_cash_quantity(quantity: int, price: float, cash: float, timestamp: pd.Timestamp) -> int:
        """在非负现金及双边费用约束下缩小整手买单，不引入杠杆。"""
        if price <= 0 or cash <= 0:
            return 0
        quantity = min(quantity, int(cash / price / lot) * lot)
        while quantity and quantity * price + calculate_fee(quantity * price, "buy", timestamp, cfg) > cash + 1e-7:
            quantity -= lot
        return quantity

    def record_attempt(book: dict, symbol: str, pid, side: str, requested: int, filled: int,
                       timestamp: pd.Timestamp, signal: pd.Timestamp, kind: str, reason: str) -> None:
        """记录拒绝或部分成交，保留真实时刻；不以之后能否成交反向修改历史。"""
        attempts.append({"datetime": timestamp, "signal_datetime": signal, "symbol": symbol, "plan_id": pid,
                         "side": side, "requested_shares": requested, "filled_shares": filled,
                         "event_type": kind, "reason": reason, "account": book["name"]})

    for date in dates:
        day_rows = daily_groups[date]
        actual["sellable"], baseline["sellable"] = actual["qty"].copy(), baseline["qty"].copy()
        opening = aggregate_quantities(actual)
        # 开盘前预算只用昨日已知估值；今日开盘价仅在下方作为成交结果使用。
        baseline_entry_equity = baseline["cash"] + sum(quantity * marks.get(plan_meta[pid]["symbol"], 0.0)
                                                       for pid, quantity in baseline["qty"].items())
        weekly_buys, weekly_sells = dict.fromkeys(symbols, 0), dict.fromkeys(symbols, 0)
        base_buys, base_sells = dict.fromkeys(symbols, 0), dict.fromkeys(symbols, 0)
        day_sold, cycles = dict.fromkeys(symbols, 0), dict.fromkeys(symbols, 0)
        actual["used"], baseline["used"] = {}, {}
        for symbol, row in day_rows.items():
            if np.isfinite(row["open"]) and row["open"] > 0:
                marks[symbol], mark_times[symbol] = float(row["open"]), date + pd.Timedelta(hours=9, minutes=30)

        def fill_book(book: dict, symbol: str, allocations: list[tuple[int, int]], side: str,
                      price: float, quote: dict, timestamp: pd.Timestamp, signal: pd.Timestamp,
                      kind: str, reason: str, pair_id: int = 0) -> dict:
            """登记真实数量、昨日可卖预算和现金；T 买回分配到原周度计划但当日不可再卖。"""
            nonlocal t_cashflow, flat_fills, limit_unknown_fills
            quantity = sum(amount for _, amount in allocations)
            before = sum(book["sellable"][pid] for pid in plans_by_symbol[symbol])
            value = quantity * price
            fee = calculate_fee(value, side, timestamp, cfg)
            if side == "buy":
                if value + fee > book["cash"] + 1e-7:
                    raise AssertionError("买单超过实际现金")
                book["cash"] -= value + fee
                for pid, amount in allocations:
                    book["qty"][pid] += amount
                flow = -value - fee
            else:
                for pid, amount in allocations:
                    if amount > book["sellable"][pid] or amount > book["qty"][pid]:
                        raise AssertionError("卖出超过昨日可卖股份")
                    book["qty"][pid] -= amount
                    book["sellable"][pid] -= amount
                book["cash"] += value - fee
                flow = value - fee
            if kind == "weekly":
                ledger = (weekly_buys if side == "buy" else weekly_sells) if book is actual else (base_buys if side == "buy" else base_sells)
                ledger[symbol] += quantity
            if book is actual:
                if side == "sell":
                    day_sold[symbol] += quantity
                if kind == "t":
                    t_cashflow += flow
                flat_fills += int(quote["high"] == quote["low"])
                limit_unknown_fills += int(not np.isfinite(quote["previous_close"]) and
                                           not np.isfinite(quote["limit_up"] if side == "buy" else quote["limit_down"]))
            book["used"][symbol] = book["used"].get(symbol, 0.0) + quantity * quote["price"]
            pid_value = "|".join(str(pid) for pid, _ in allocations)
            event = {"datetime": timestamp, "signal_datetime": signal, "signal_time": signal,
                     "symbol": symbol, "plan_id": pid_value, "lot_id": pair_id if kind == "t" else f"weekly_{pid_value}",
                     "pair_id": pair_id, "side": side, "shares": quantity, "planned_shares": quantity, "price": price,
                     "reference_price": quote["price"], "notional": value, "fee": fee,
                     "slippage_cost": quantity * abs(price - quote["price"]), "reason": reason,
                     "event_type": kind, "cash_after": book["cash"], "sellable_before": before,
                     "sellable_after": sum(book["sellable"][pid] for pid in plans_by_symbol[symbol]), "account": book["name"]}
            book["events"].append(event)
            return event

        def allocate_old_shares(symbol: str, quantity: int) -> list[tuple[int, int]]:
            """按周度计划建立 T 股份归属，允许跨计划但总数不超过日初库存预算。"""
            allocations = []
            for pid in plans_by_symbol[symbol]:
                amount = min(actual["sellable"][pid], quantity)
                if amount:
                    allocations.append((pid, amount))
                    quantity -= amount
                if not quantity:
                    break
            return allocations

        def recover_pair(symbol: str, timestamp: pd.Timestamp, prefix: str) -> None:
            """用当前可见的 14:55/收盘/次日开盘行情尝试整笔恢复，不部分伪配对。"""
            pair = active[symbol]
            row = pair["recovery_row"] if prefix == "recovery" else day_rows.get(symbol)
            quote = create_quote(row, prefix, symbol)
            side = "buy" if pair["direction"] == "sell_first" else "sell"
            quantity = pair["shares"]
            filled, price, reason = calculate_quote_fill(quote, side, quantity, actual["used"].get(symbol, 0.0))
            if side == "sell" and any(actual["sellable"][pid] < amount for pid, amount in pair["allocations"]):
                filled, reason = 0, "t1_budget"
            if side == "buy" and filled:
                if quantity * price + calculate_fee(quantity * price, "buy", timestamp, cfg) > actual["cash"] - calculate_reserves(symbol) + 1e-7:
                    filled, reason = 0, "cash"
            if filled != quantity:
                record_attempt(actual, symbol, "|".join(str(pid) for pid, _ in pair["allocations"]), side,
                               quantity, 0, timestamp, timestamp - pd.Timedelta(minutes=1), "t_recovery", reason or "capacity")
                return
            second = fill_book(actual, symbol, pair["allocations"], side, price, quote, timestamp,
                               timestamp - pd.Timedelta(minutes=1), "t", f"{prefix}原数量恢复", pair["pair_id"])
            first = pair["first"]
            buy, sell = (first, second) if first["side"] == "buy" else (second, first)
            gross = quantity * (sell["price"] - buy["price"])
            trips.append({"lot_id": pair["pair_id"], "pair_id": pair["pair_id"], "symbol": symbol,
                          "direction": pair["direction"], "shares": quantity, "first_datetime": first["datetime"],
                          "second_datetime": timestamp, "buy_datetime": buy["datetime"], "sell_datetime": sell["datetime"],
                          "buy_price": buy["price"], "sell_price": sell["price"], "buy_fee": buy["fee"], "sell_fee": sell["fee"],
                          "gross_pnl": gross, "net_pnl": gross - buy["fee"] - sell["fee"],
                          "slippage_cost": buy["slippage_cost"] + sell["slippage_cost"],
                          "holding_days": (timestamp - first["datetime"]).total_seconds() / 86400,
                          "recovered_same_day": timestamp.normalize() == first["datetime"].normalize(), "reason": f"{prefix}恢复"})
            del active[symbol]

        def liquidate_weekly(timestamp: pd.Timestamp, prefix: str, include_today: bool) -> None:
            """执行预定周末退出或跨日未完成退出；两账户独立受容量及 T+1 约束。"""
            for pid in plan_ids:
                meta = plan_meta[pid]
                due = meta["exit_date"] <= date if include_today else meta["exit_date"] < date
                if not due or not (actual["qty"][pid] or baseline["qty"][pid]):
                    continue
                symbol = meta["symbol"]
                quote = create_quote(day_rows.get(symbol), prefix, symbol)
                fills = {}
                for book in [baseline, actual]:
                    requested = book["qty"][pid]
                    filled, price, reason = calculate_quote_fill(quote, "sell", requested, book["used"].get(symbol, 0.0))
                    if filled > book["sellable"][pid]:
                        filled = book["sellable"][pid]
                        reason = "t1_budget"
                    if filled:
                        fill_book(book, symbol, [(pid, filled)], "sell", price, quote, timestamp,
                                  meta["exit_date"] - pd.Timedelta(days=1), "weekly", "预定周末退出" if date == meta["exit_date"] else "跨日补周度退出")
                    if filled < requested:
                        record_attempt(book, symbol, pid, "sell", requested, filled, timestamp,
                                       meta["exit_date"] - pd.Timedelta(days=1), "weekly", reason or "capacity")
                    fills[book["name"]] = filled
                if fills["actual"] != fills["baseline"]:
                    mismatches.append({"date": date, "datetime": timestamp, "plan_id": pid, "symbol": symbol,
                                       "side": "sell", "baseline_shares": fills["baseline"], "actual_shares": fills["actual"]})

        opening_time = date + pd.Timedelta(hours=9, minutes=30)
        for symbol in list(active):
            recover_pair(symbol, opening_time, "open")
        liquidate_weekly(opening_time, "open", False)
        # 周度预算只取基准权益，分母始终为配置 top_n，不重新分配容量留下的现金。
        entry_budget = baseline_entry_equity * cfg["base_fraction"] / cfg["top_n"]
        for pid in entry_groups.get(date, []):
            meta, symbol = plan_meta[pid], plan_meta[pid]["symbol"]
            quote = create_quote(day_rows.get(symbol), "open", symbol)
            reference = meta.get("universe_previous_close", np.nan)
            if not np.isfinite(reference):
                reference = quote["previous_close"]
            requested = int(entry_budget / (reference * (1 + cfg["slippage_rate"])) / lot) * lot if np.isfinite(reference) and reference > 0 else 0
            filled, price, reason = calculate_quote_fill(quote, "buy", requested, baseline["used"].get(symbol, 0.0))
            planned = calculate_cash_quantity(filled, price, baseline["cash"], opening_time)
            if planned:
                event = fill_book(baseline, symbol, [(pid, planned)], "buy", price, quote, opening_time,
                                  meta["trade_week"] - pd.Timedelta(days=1), "weekly", "周初固定等权底仓")
                event["planned_shares"] = requested
            if planned < requested or not requested:
                record_attempt(baseline, symbol, pid, "buy", requested, planned, opening_time,
                               meta["trade_week"] - pd.Timedelta(days=1), "weekly", reason or ("cash" if requested else "below_one_lot_or_missing_previous_price"))
            actual_filled, actual_price, actual_reason = calculate_quote_fill(quote, "buy", planned, actual["used"].get(symbol, 0.0))
            if actual_filled and actual_filled * actual_price + calculate_fee(actual_filled * actual_price, "buy", opening_time, cfg) > actual["cash"] - calculate_reserves() + 1e-7:
                actual_filled, actual_reason = 0, "cash_or_t_repurchase_reserve"
            if actual_filled:
                event = fill_book(actual, symbol, [(pid, actual_filled)], "buy", actual_price, quote, opening_time,
                                  meta["trade_week"] - pd.Timedelta(days=1), "weekly", "复制基准固定底仓买单")
                event["planned_shares"] = planned
            if actual_filled < planned:
                record_attempt(actual, symbol, pid, "buy", planned, actual_filled, opening_time,
                               meta["trade_week"] - pd.Timedelta(days=1), "weekly", actual_reason or "capacity")
            if actual_filled != planned:
                mismatches.append({"date": date, "datetime": opening_time, "plan_id": pid, "symbol": symbol,
                                   "side": "buy", "baseline_shares": planned, "actual_shares": actual_filled})
            entry_plans.append({"datetime": opening_time, "plan_id": pid, "symbol": symbol,
                                "known_reference_price": reference, "baseline_entry_equity": baseline_entry_equity,
                                "entry_budget": entry_budget, "planned_shares": requested,
                                "baseline_filled_shares": planned, "actual_filled_shares": actual_filled})

        day_requests = request_groups.get(date, [])
        execution_groups = {}
        for request in day_requests:
            if pd.isna(request["execution_time"]):
                if request["action"]:
                    record_attempt(actual, request["symbol"], "", "buy" if request["action"] > 0 else "sell", 0, 0,
                                   request["datetime"] + pd.Timedelta(minutes=1), request["datetime"], "t_entry", "missing_execution_minute")
                continue
            execution_groups.setdefault(request["execution_time"], []).append(request)
        for timestamp, group in sorted(execution_groups.items()):
            actual["used"] = {}
            for row in group:
                symbol, side = row["symbol"], "buy" if row["action"] > 0 else "sell"
                if not row["action"] or symbol not in plans_by_symbol:
                    continue
                ids = plans_by_symbol[symbol]
                old = sum(actual["sellable"][pid] for pid in ids)
                blocked = ""
                if symbol in active:
                    blocked = "pending_t_recovery"
                elif any(actual["qty"][pid] and plan_meta[pid]["exit_date"] <= date for pid in ids):
                    blocked = "weekly_exit_day_t1_reserve"
                elif cycles[symbol]:
                    blocked = "daily_one_pair_limit"
                elif timestamp.time() > last_entry_time:
                    blocked = "last_entry_time"
                elif old < lot or opening[symbol] < lot:
                    blocked = "no_opening_t1_inventory"
                quantity = min(int(opening[symbol] * row["fraction"] / lot) * lot, old)
                if blocked or quantity < lot:
                    record_attempt(actual, symbol, "", side, quantity, 0, timestamp, row["datetime"], "t_entry", blocked or "below_one_lot")
                    continue
                requested_quantity = quantity
                quote = create_quote(row, "execution", symbol)
                quantity, price, reason = calculate_quote_fill(quote, side, quantity, actual["used"].get(symbol, 0.0))
                available_cash = actual["cash"] - calculate_reserves()
                if side == "buy":
                    quantity = calculate_cash_quantity(quantity, price, available_cash, timestamp)
                else:
                    while quantity:
                        reserve_price = quote["price"] * (1 + cfg["repurchase_buffer"]) * (1 + cfg["slippage_rate"])
                        reserve = quantity * reserve_price + calculate_fee(quantity * reserve_price, "buy", timestamp, cfg)
                        proceeds = quantity * price - calculate_fee(quantity * price, "sell", timestamp, cfg)
                        if reserve <= available_cash + proceeds + 1e-7:
                            break
                        quantity -= lot
                if quantity < lot:
                    record_attempt(actual, symbol, "", side, requested_quantity, 0, timestamp, row["datetime"], "t_entry", reason or "cash_or_repurchase_reserve")
                    continue
                allocations = allocate_old_shares(symbol, quantity)
                event = fill_book(actual, symbol, allocations, side, price, quote, timestamp, row["datetime"],
                                  "t", "训练规则先买" if side == "buy" else "训练规则先卖", next_pair)
                reserve = 0.0
                if side == "sell":
                    reserve_price = quote["price"] * (1 + cfg["repurchase_buffer"]) * (1 + cfg["slippage_rate"])
                    reserve = quantity * reserve_price + calculate_fee(quantity * reserve_price, "buy", timestamp, cfg)
                active[symbol] = {"pair_id": next_pair, "shares": quantity, "allocations": allocations,
                                  "direction": "buy_first" if side == "buy" else "sell_first", "first": event,
                                  "reserve": reserve, "recovery_row": row}
                next_pair += 1
                cycles[symbol] += 1
        actual["used"] = {}
        for symbol in list(active):
            if active[symbol]["first"]["datetime"].normalize() == date:
                # recovery_time 的有效性仅在到达恢复时钟后检查，不参与首腿准入。
                provided_time = pd.to_datetime(active[symbol]["recovery_row"].get("recovery_time", pd.NaT))
                recovery_time = date + pd.Timedelta(hours=14, minutes=55)
                if pd.notna(provided_time) and provided_time != recovery_time:
                    active[symbol]["recovery_row"] = {}
                recover_pair(symbol, recovery_time, "recovery")

        close_time = date + pd.Timedelta(hours=15)
        actual["used"], baseline["used"] = {}, {}
        for symbol, row in day_rows.items():
            if np.isfinite(row["close"]) and row["close"] > 0:
                marks[symbol], mark_times[symbol] = float(row["close"]), close_time
        for symbol in list(active):
            recover_pair(symbol, close_time, "close")
        liquidate_weekly(close_time, "close", True)
        actual_total, base_total = aggregate_quantities(actual), aggregate_quantities(baseline)
        held_value = sum(quantity * marks.get(symbol, 0.0) for symbol, quantity in actual_total.items())
        baseline_equity = baseline["cash"] + sum(quantity * marks.get(symbol, 0.0) for symbol, quantity in base_total.items())
        equity = actual["cash"] + held_value
        curves.append({"datetime": close_time, "equity": equity, "cash": actual["cash"], "holdings_value": held_value,
                       "baseline_equity": baseline_equity, "baseline_cash": baseline["cash"], "t_added_equity": equity - baseline_equity,
                       "t_realized_pnl": sum(trip["net_pnl"] for trip in trips), "pending_pair_count": len(active),
                       "holding_count": sum(quantity > 0 for quantity in actual_total.values()),
                       "open_lots": sum(quantity > 0 for quantity in actual["qty"].values()),
                       "exposure": held_value / equity if equity else 0.0,
                       "stale_price_count": sum(quantity > 0 and mark_times.get(symbol) != close_time for symbol, quantity in actual_total.items())})
        for symbol, ids in plans_by_symbol.items():
            if not (opening[symbol] or actual_total[symbol] or base_total[symbol] or weekly_buys[symbol] or weekly_sells[symbol] or symbol in active):
                continue
            target = opening[symbol] + weekly_buys[symbol] - weekly_sells[symbol]
            difference = actual_total[symbol] - target
            overdue = any((actual["qty"][pid] or baseline["qty"][pid]) and plan_meta[pid]["exit_date"] <= date for pid in ids)
            if symbol in active:
                overdue |= any(plan_meta[pid]["exit_date"] <= date for pid, _ in active[symbol]["allocations"])
            audits.append({"date": date, "symbol": symbol, "opening_shares": opening[symbol], "opening_sellable": opening[symbol],
                           "weekly_buys": weekly_buys[symbol], "weekly_sells": weekly_sells[symbol],
                           "weekly_buy_target": base_buys[symbol], "weekly_sell_target": base_sells[symbol],
                           "target_shares": target, "closing_shares": actual_total[symbol], "inventory_difference": difference,
                           "inventory_preserved": difference == 0, "day_sold": day_sold[symbol],
                           "closing_sellable": sum(actual["sellable"][pid] for pid in ids), "baseline_closing_shares": base_total[symbol],
                           "baseline_inventory_difference": actual_total[symbol] - base_total[symbol], "t1_violation_count": 0,
                           "recovery_failure": symbol in active, "weekly_liquidation_failure": overdue,
                           "cycles_started": cycles[symbol], "stale_price": mark_times.get(symbol) != close_time})
        if actual["cash"] < -1e-6 or baseline["cash"] < -1e-6 or any(quantity < 0 for quantity in actual["qty"].values()):
            raise AssertionError("账户产生负现金或空头库存")
        if any(day_sold[symbol] > opening[symbol] for symbol in symbols):
            raise AssertionError("当日累计卖出超过日初库存")
        for symbol, row in day_rows.items():
            if np.isfinite(row["close"]) and row["close"] > 0:
                previous_marks[symbol] = float(row["close"])

    curve = pd.DataFrame(curves)
    curve["return"] = curve.equity.pct_change(fill_method=None)
    curve.loc[0, "return"] = curve.equity.iloc[0] / cfg["initial_cash"] - 1
    curve["net_value"] = curve.equity / cfg["initial_cash"]
    curve["cumulative_return"] = curve.net_value - 1
    curve["drawdown"] = curve.equity / np.maximum.accumulate(np.r_[cfg["initial_cash"], curve.equity])[1:] - 1
    event_frame = pd.DataFrame(actual["events"], columns=EVENT_COLUMNS)
    base_frame = pd.DataFrame(baseline["events"], columns=EVENT_COLUMNS)
    trip_frame = pd.DataFrame(trips, columns=TRIP_COLUMNS)
    audit_frame = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    holdings = []
    for pid, meta in plan_meta.items():
        symbol = meta["symbol"]
        pair = active.get(symbol, {})
        pending = next((amount for plan, amount in pair.get("allocations", []) if plan == pid), 0)
        if actual["qty"][pid] or baseline["qty"][pid] or pending:
            holdings.append({"plan_id": pid, "symbol": symbol, "trade_week": meta["trade_week"],
                             "entry_date": meta["entry_date"], "exit_date": meta["exit_date"],
                             "shares": actual["qty"][pid], "baseline_shares": baseline["qty"][pid],
                             "mark_price": marks.get(symbol, np.nan), "market_value": actual["qty"][pid] * marks.get(symbol, 0.0),
                             "mark_datetime": mark_times.get(symbol), "inventory_difference": actual["qty"][pid] - baseline["qty"][pid],
                             "overdue_weekly_exit": meta["exit_date"] <= dates[-1], "pending_pair_id": pair.get("pair_id", 0),
                             "pending_direction": pair.get("direction", ""), "pending_shares": pending})
    metrics = calculate_daily_metrics(curve, event_frame, trip_frame, cfg["initial_cash"])
    metrics.update({"strategy": "weekly_inventory_t", "total_t_net_profit": float(trip_frame.net_pnl.sum()) if len(trip_frame) else 0.0,
                    "total_t_marked_pnl": float(curve.t_added_equity.iloc[-1]), "t_cashflow": t_cashflow,
                    "t_pair_count": len(trip_frame), "pending_pairs": len(active),
                    "t_fee": float(event_frame.loc[event_frame.event_type == "t", "fee"].sum()),
                    "baseline_total_return": float(curve.baseline_equity.iloc[-1] / cfg["initial_cash"] - 1),
                    "base_plan_mismatch_count": len(mismatches), "same_base_plan": not mismatches,
                    "inventory_mismatch_days": int(audit_frame.loc[~audit_frame.inventory_preserved, "date"].nunique()) if len(audit_frame) else 0,
                    "recovery_failure_days": int(audit_frame.loc[audit_frame.recovery_failure, "date"].nunique()) if len(audit_frame) else 0,
                    "weekly_liquidation_failure_days": int(audit_frame.loc[audit_frame.weekly_liquidation_failure, "date"].nunique()) if len(audit_frame) else 0,
                    "weekly_exit_failure_days": int(audit_frame.loc[audit_frame.weekly_liquidation_failure, "date"].nunique()) if len(audit_frame) else 0,
                    "weekly_unclosed_positions": sum(actual["qty"][pid] > 0 and plan_meta[pid]["exit_date"] <= dates[-1] for pid in plan_ids),
                    "baseline_weekly_unclosed_positions": sum(baseline["qty"][pid] > 0 and plan_meta[pid]["exit_date"] <= dates[-1] for pid in plan_ids),
                    "t1_violation_count": 0, "flat_bar_fill_count": flat_fills, "limit_guard_unknown_fill_count": limit_unknown_fills,
                    "stale_valuation_days": int((curve.stale_price_count > 0).sum()),
                    "weekly_buy_count": int(((event_frame.event_type == "weekly") & (event_frame.side == "buy")).sum()),
                    "weekly_sell_count": int(((event_frame.event_type == "weekly") & (event_frame.side == "sell")).sum()),
                    "blocked_order_count": len(attempts), "short_exposure": 0})
    return {"equity_curve": curve, "trade_events": event_frame, "round_trips": trip_frame,
            "open_positions": pd.DataFrame(holdings, columns=POSITION_COLUMNS), "daily_inventory_audit": audit_frame,
            "order_attempts": pd.DataFrame(attempts, columns=ATTEMPT_COLUMNS), "metrics": metrics,
            "baseline_events": base_frame, "base_plan_mismatches": pd.DataFrame(mismatches),
            "weekly_entry_plans": pd.DataFrame(entry_plans)}
