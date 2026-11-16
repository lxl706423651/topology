"""同刻首腿反方向限价复核；复制冻结动态账户核，旧引擎与缓存保持原样。

事前开启t_recheck_exit_side_at_execution后，仅在首腿本方向已有可填数量时，
用同一到达分钟quote检查反方向限价，命中则整笔取消。该条件属于到达报价的
成交限制，不是上一信号分钟已知的新因子；不能回看后续恢复价量或成功状态。
复核不模拟反向成交、不消耗容量/现金/旧股。默认关闭时原全部账户字段精确
复现，另增first_leg_exit_side_audit。底仓仍由原plan_dynamic_entries冻结。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .annual_portfolio import calculate_daily_metrics, calculate_fee
from .inventory_t import calculate_next_trading_minute
from .weekly_t_engine import (RecoveryBarStore, prepare_recovery_bars, read_fixed_recovery_minutes)
from .balanced_t_engine import (BALANCED_DEFAULTS, validate_balanced_config, prepare_base_requests,
                                prepare_balanced_inputs, plan_balanced_quantity,
                                EVENT_COLUMNS as BASE_EVENT_COLUMNS, TRIP_COLUMNS as BASE_TRIP_COLUMNS,
                                AUDIT_COLUMNS, POSITION_COLUMNS as BASE_POSITION_COLUMNS,
                                ATTEMPT_COLUMNS, AMOUNT_COLUMNS)

EXIT_EVENT_COLUMNS = ["original_deadline", "exit_signal_time", "exit_earliest_time", "exit_reason"]
EVENT_COLUMNS = [*BASE_EVENT_COLUMNS, *EXIT_EVENT_COLUMNS]
TRIP_COLUMNS = [*BASE_TRIP_COLUMNS, *EXIT_EVENT_COLUMNS]
POSITION_COLUMNS = [*BASE_POSITION_COLUMNS, *EXIT_EVENT_COLUMNS, "exit_state"]
EXIT_DECISION_COLUMNS = ["pair_id", "symbol", "direction", "exit_mode", "first_signal_time", "first_execution_time",
                         "original_deadline", "exit_signal_time", "exit_earliest_time", "exit_reason",
                         "observed_price", "estimated_net_pnl", "estimated_net_return", "normalization_notional",
                         "remaining_shares", "realized_fragment_net_pnl"]
SESSION_MINUTES = tuple(range(571, 691)) + tuple(range(781, 901))
DYNAMIC_DEFAULTS = {"dynamic_exit_mode": "fixed", "dynamic_take_net_return": .003,
                    "dynamic_stop_net_return": .01}
FIRST_LEG_EXIT_SIDE_AUDIT_COLUMNS = [
    "datetime", "signal_datetime", "symbol", "side", "opposite_side", "requested_shares",
    "own_fillable_shares", "guard_enabled", "opposite_checked", "cancelled", "reason",
    "opposite_quote_reason", "reference_price", "high", "low", "amount", "volume",
    "previous_close", "limit_up", "limit_down", "minute_amount_used_before", "cash_before",
    "old_sellable_before"]


def validate_dynamic_config(cfg: dict) -> None:
    """模式和阈值事前声明，拒绝非法配置，不从回测结果推断或优化。"""
    if cfg["dynamic_exit_mode"] not in {"fixed", "profit_buffer", "profit_buffer_stop"}:
        raise ValueError("dynamic_exit_mode只支持fixed/profit_buffer/profit_buffer_stop")
    for key in ("dynamic_take_net_return", "dynamic_stop_net_return"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key}须为事前冻结的有限正数")


def run_guarded_portfolio(selections: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame,
                         decisions: pd.DataFrame, cfg: dict,
                         recovery_bars: pd.DataFrame | RecoveryBarStore | None = None,
                           manual_entries: pd.DataFrame | None = None) -> dict:
    """固定底仓加动态恢复；返回原标准表以及每组唯一的退出意图审计。

    自动每股只首次成功入场，实际滑点后名义额须在配置区间。显式manual_entries
    为用户基础买单，金额区间例外仍受现金、容量和T+1约束。日内T仅消耗日初旧股；
    max_cycles_per_day=0表示不额外限次数，单股同时最多一个待恢复组。
    基准独立决定底仓订单，实际不能以T利润扩大底仓。冻结计划可由
    dynamic_t_engine.plan_dynamic_entries生成后重放；不重新规划或扩大基础股数。
    所有模式遍历相同240会话分钟，恢复意图锁存后逐分钟尝试；recovery_bars
    缺当前真实行时不补未来价。共同市场估值不依赖实际T是否触发或成交。
    """
    cfg = {**BALANCED_DEFAULTS, **DYNAMIC_DEFAULTS, "t_recheck_exit_side_at_execution": False, **cfg}
    if not isinstance(cfg["t_recheck_exit_side_at_execution"], (bool, np.bool_)):
        raise ValueError("t_recheck_exit_side_at_execution须为显式布尔值")
    validate_balanced_config(cfg)
    validate_dynamic_config(cfg)
    selected_input = prepare_base_requests(selections, manual_entries, cfg)
    selected, prices, requests, dates = prepare_balanced_inputs(selected_input, daily, intraday, decisions, cfg)
    fixed_recovery_minutes = read_fixed_recovery_minutes(cfg)
    recovery_lookup = prepare_recovery_bars(recovery_bars, dates)
    # pandas命名元组会改写下划线字段，基础计划须保留_manual等精确字段名。
    plan_meta = {int(row["plan_id"]): row for row in selected.to_dict("records")}
    plan_ids = list(plan_meta)
    symbols = sorted(set(prices.symbol).union(selected.symbol))
    plans_by_symbol = {symbol: [plan_id for plan_id in plan_ids if plan_meta[plan_id]["symbol"] == symbol] for symbol in symbols}
    daily_groups = {date: frame.set_index("symbol").to_dict("index") for date, frame in prices.groupby("date")}
    request_groups = {date: frame.to_dict("records") for date, frame in requests.groupby("date")}
    entry_groups = {time: frame.plan_id.astype(int).tolist() for time, frame in selected.groupby("_entry_time")}
    actual = {"name": "actual", "cash": float(cfg["initial_cash"]), "qty": dict.fromkeys(plan_ids, 0), "events": []}
    baseline = {"name": "baseline", "cash": float(cfg["initial_cash"]), "qty": dict.fromkeys(plan_ids, 0), "events": []}
    marks, mark_times, previous_marks, active = {}, {}, {}, {}
    attempts, trips, audits, curves, mismatches, entry_plans, amount_audits = [], [], [], [], [], [], []
    exit_decisions = []
    first_leg_exit_side_audits = []
    base_opened = set()
    marked_symbols = set()
    base_cash_floor = cfg["initial_cash"] * (1 - cfg["base_fraction"])
    lot, next_pair = int(cfg["lot_size"]), 1
    flat_fills = 0
    limit_unknown_fills = 0
    t_cashflow = 0.0
    peak_pending_t_notional = 0.0
    pending_fraction = cfg.get("max_pending_t_notional_fraction")
    pending_notional_limit = None if pending_fraction is None else float(cfg["initial_cash"] * pending_fraction)
    completed_pair_ids = set()
    last_entry_time = pd.Timestamp(f"2000-01-01 {cfg['last_entry_time']}").time()

    def aggregate_quantities(book: dict) -> dict:
        """汇总基础入场计划的真实股份，保留人工加仓和待恢复股份的批次归属。"""
        return {symbol: sum(book["qty"][pid] for pid in ids) for symbol, ids in plans_by_symbol.items()}

    def calculate_reserves(exclude: str | None = None) -> float:
        """保留其他先卖组合的回购资金，避免用卖出所得无限开新组合。"""
        return sum(pair["reserve"] for symbol, pair in active.items() if symbol != exclude)

    def calculate_pending_t_notional() -> float:
        """以首腿当刻已知参考价计量剩余债务；不随后续价格或预测恢复状态变化。"""
        return sum(pair["shares"] * pair["first"]["reference_price"] for pair in active.values())

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
        baseline_opening = aggregate_quantities(baseline)
        manual_adds = dict.fromkeys(symbols, 0)
        # 开盘前预算只用昨日已知估值；今日开盘价仅在下方作为成交结果使用。
        baseline_entry_equity = baseline["cash"] + sum(quantity * marks.get(plan_meta[pid]["symbol"], 0.0)
                                                       for pid, quantity in baseline["qty"].items())
        weekly_buys, weekly_sells = dict.fromkeys(symbols, 0), dict.fromkeys(symbols, 0)
        base_buys, base_sells = dict.fromkeys(symbols, 0), dict.fromkeys(symbols, 0)
        day_sold, cycles = dict.fromkeys(symbols, 0), dict.fromkeys(symbols, 0)
        actual["used"], baseline["used"] = {}, {}
        # 按当前时钟重置，仅缓存正在读取的真实quote；不缓存或探测未来能否成交。
        current_quote_cache = {}
        for symbol, row in day_rows.items():
            if np.isfinite(row["open"]) and row["open"] > 0:
                marks[symbol], mark_times[symbol] = float(row["open"]), date + pd.Timedelta(hours=9, minutes=30)

        def fill_book(book: dict, symbol: str, allocations: list[tuple[int, int]], side: str,
                      price: float, quote: dict, timestamp: pd.Timestamp, signal: pd.Timestamp,
                      kind: str, reason: str, pair_id: int = 0) -> dict:
            """登记真实数量、旧股预算和现金；T买回分配到原基础批次，当日不可再卖。"""
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
                if side == "buy":
                    # 固定底仓不卖；只需为曾成功持有的股票读取共同逐分钟价格。
                    # 全市场候选规划阶段不能每分钟扫描数千从未持有的股票。
                    marked_symbols.add(symbol)
            if book is actual:
                if side == "sell":
                    day_sold[symbol] += quantity
                if kind == "t":
                    t_cashflow += flow
                flat_fills += int(quote["high"] == quote["low"])
                limit_unknown_fills += int(not np.isfinite(quote["previous_close"]) and
                                           not np.isfinite(quote["limit_up"] if side == "buy" else quote["limit_down"]))
            book["used"][symbol] = book["used"].get(symbol, 0.0) + quantity * quote["price"]
            # 配对账户共同的基础/人工订单保留缺端点时的参考价兜底。T成交本身
            # 不额外改变共同mark，统一由下方同一完整行情流更新，避免不同模式
            # 的成交次数改变无T基准在缺收盘日可见的最后市场价。
            if kind == "weekly" and (symbol not in mark_times or timestamp >= mark_times[symbol]):
                marks[symbol], mark_times[symbol] = float(quote["price"]), timestamp
            pid_value = "|".join(str(pid) for pid, _ in allocations)
            event = {"datetime": timestamp, "signal_datetime": signal, "signal_time": signal,
                     "symbol": symbol, "plan_id": pid_value, "lot_id": pair_id if kind == "t" else f"weekly_{pid_value}",
                     "pair_id": pair_id, "side": side, "shares": quantity, "planned_shares": quantity, "price": price,
                     "reference_price": quote["price"], "notional": value, "fee": fee,
                     "slippage_cost": quantity * abs(price - quote["price"]), "reason": reason,
                     "event_type": kind, "base_order_type": "", "cash_after": book["cash"], "sellable_before": before,
                     "sellable_after": sum(book["sellable"][pid] for pid in plans_by_symbol[symbol]), "account": book["name"],
                     "fragment_id": 0, "signal_priority": 0.0, "planned_recovery_time": pd.NaT,
                     "original_deadline": pd.NaT, "exit_signal_time": pd.NaT,
                     "exit_earliest_time": pd.NaT, "exit_reason": ""}
            book["events"].append(event)
            return event

        def allocate_old_shares(symbol: str, quantity: int) -> list[tuple[int, int]]:
            """按基础批次建立T股份归属，允许跨批次但总数不超过日初旧股预算。"""
            allocations = []
            for pid in plans_by_symbol[symbol]:
                amount = min(actual["sellable"][pid], quantity)
                if amount:
                    allocations.append((pid, amount))
                    quantity -= amount
                if not quantity:
                    break
            return allocations

        def create_current_recovery_quote(symbol: str, timestamp: pd.Timestamp) -> dict:
            """到达恢复时钟才读取该分钟；缺行保持不可成交，不向后找可用价格。"""
            if symbol in current_quote_cache:
                return current_quote_cache[symbol]
            row = recovery_lookup.get_quote(timestamp, symbol) if recovery_lookup is not None else None
            if row is None:
                quote = create_quote({}, "close", symbol)
                # 无store的明确兼容分支只允许当刻实际日终端点；不能拿首腿样本
                # 中未来recovery_price/amount填补盘中动态观察。
                if recovery_lookup is None and timestamp.time() == pd.Timestamp("15:00").time():
                    quote = create_quote(day_rows.get(symbol), "close", symbol)
                current_quote_cache[symbol] = quote
                return quote
            current = dict(day_rows.get(symbol, {}))
            for key in ["high", "low", "amount", "volume"]:
                current[f"close_{key}"] = row.get(key, np.nan)
            current["close"] = row.get("close", np.nan)
            for key in ["previous_close", "prev_close", "limit_up", "limit_down"]:
                if key in row:
                    current[key] = row[key]
            quote = create_quote(current, "close", symbol)
            current_quote_cache[symbol] = quote
            return quote

        def observe_and_latch_exit(symbol: str, timestamp: pd.Timestamp) -> None:
            """当前完成分钟估计全组含费净率；锁存后不因回撤、限价或缺量撤销。

            首腿费仅按剩余股数比例归属；已完成片段净额计入全组预计结果，
            归一化金额始终为原首腿参考价乘原股数。未来价格/分片次数未知。
            """
            pair = active[symbol]
            if pair["exit_state"] != "monitoring":
                return
            earliest = calculate_next_trading_minute(timestamp)
            if earliest is None:
                return
            quote = create_current_recovery_quote(symbol, timestamp)
            raw = quote["price"]
            observed = (all(np.isfinite(quote[key]) for key in ("price", "high", "low", "amount", "volume"))
                        and raw > 0 and quote["amount"] > 0 and quote["volume"] > 0
                        and quote["low"] - 1e-6 <= raw <= quote["high"] + 1e-6)
            estimated, net_return = np.nan, np.nan
            if observed:
                first, quantity = pair["first"], pair["shares"]
                buy_back = pair["direction"] == "sell_first"
                reverse_price = raw * (1 + cfg["slippage_rate"] * (1 if buy_back else -1))
                gross = quantity * ((first["price"] - reverse_price) if buy_back else (reverse_price - first["price"]))
                fee = calculate_fee(quantity * reverse_price, "buy" if buy_back else "sell", timestamp, cfg)
                estimated = (pair["realized_fragment_net_pnl"] + gross - fee
                             - first["fee"] * quantity / pair["original_shares"])
                net_return = estimated / pair["normalization_notional"]
            # 原期限是首腿之前已声明的执行时刻；在它的前一交易分钟发出时间
            # 意图，即便当前bar缺失。13:01期限对应11:30信号，不伪造午休bar。
            reason = ""
            if earliest >= pair["original_deadline"]:
                reason = "hard_deadline"
            elif cfg["dynamic_exit_mode"] != "fixed" and observed:
                if net_return >= cfg["dynamic_take_net_return"]:
                    reason = "profit_buffer"
                elif cfg["dynamic_exit_mode"] == "profit_buffer_stop" and net_return <= -cfg["dynamic_stop_net_return"]:
                    reason = "stop_loss"
            if not reason:
                return
            if not pair["first"]["signal_datetime"] < timestamp < earliest:
                raise AssertionError("退出信号必须晚于首腿信号，且早于任何恢复成交")
            pair.update(exit_state="exit_latched", exit_signal_time=timestamp,
                        exit_earliest_time=earliest, exit_reason=reason)
            exit_decisions.append({"pair_id": pair["pair_id"], "symbol": symbol, "direction": pair["direction"],
                                   "exit_mode": cfg["dynamic_exit_mode"],
                                   "first_signal_time": pair["first"]["signal_datetime"],
                                   "first_execution_time": pair["first"]["datetime"],
                                   "original_deadline": pair["original_deadline"], "exit_signal_time": timestamp,
                                   "exit_earliest_time": earliest, "exit_reason": reason,
                                   "observed_price": raw if observed else np.nan,
                                   "estimated_net_pnl": estimated, "estimated_net_return": net_return,
                                   "normalization_notional": pair["normalization_notional"],
                                   "remaining_shares": pair["shares"],
                                   "realized_fragment_net_pnl": pair["realized_fragment_net_pnl"]})

        def recover_pair(symbol: str, timestamp: pd.Timestamp, prefix: str,
                         quote_override: dict | None = None) -> None:
            """按当前价量、现金及旧库存恢复一段，剩余债务和回购准备金继续保留。

            每段只承担首腿原始总费用/滑点的股数比例，反向腿费用按本段真实成交
            独立计算。每分钟容量与同刻人工加仓及新T首腿共享，不虚构全量恢复。
            """
            pair = active[symbol]
            if (pair["exit_state"] == "monitoring" or timestamp < pair["exit_earliest_time"]
                    or timestamp <= pair["exit_signal_time"]):
                raise AssertionError("尚未锁存退出，或试图同bar恢复")
            row = pair["recovery_row"] if prefix == "recovery" else day_rows.get(symbol)
            quote = create_quote(row, prefix, symbol) if quote_override is None else quote_override
            side = "buy" if pair["direction"] == "sell_first" else "sell"
            quantity = pair["shares"]
            filled, price, reason = calculate_quote_fill(quote, side, quantity, actual["used"].get(symbol, 0.0))
            if side == "sell":
                available = sum(min(actual["sellable"][pid], amount) for pid, amount in pair["allocations"])
                if filled > available:
                    filled, reason = available, "t1_budget"
            if side == "buy" and filled:
                affordable = calculate_cash_quantity(filled, price, actual["cash"] - calculate_reserves(symbol), timestamp)
                if affordable < filled:
                    filled, reason = affordable, "cash"
            if filled != quantity:
                record_attempt(actual, symbol, "|".join(str(pid) for pid, _ in pair["allocations"]), side,
                               quantity, filled, timestamp, pair["exit_signal_time"], "t_recovery", reason or "capacity")
            if not filled:
                return
            allocations, remaining_allocations, to_fill = [], [], filled
            for pid, amount in pair["allocations"]:
                available = min(amount, actual["sellable"][pid]) if side == "sell" else amount
                allocated = min(available, to_fill)
                if allocated:
                    allocations.append((pid, allocated))
                    to_fill -= allocated
                if amount > allocated:
                    remaining_allocations.append((pid, amount - allocated))
            if to_fill:
                raise AssertionError("恢复分配未覆盖实际成交股数")
            second = fill_book(actual, symbol, allocations, side, price, quote, timestamp,
                               pair["exit_signal_time"], "t", f"{prefix}分段恢复", pair["pair_id"])
            second["planned_shares"] = quantity
            second["fragment_id"] = len(trips) + 1
            second["signal_priority"] = pair["first"]["signal_priority"]
            second["planned_recovery_time"] = pair["planned_recovery_time"]
            for key in EXIT_EVENT_COLUMNS:
                second[key] = pair[key]
            first = pair["first"]
            fraction = filled / pair["original_shares"]
            allocated_first = {**first, "shares": filled, "notional": first["price"] * filled,
                               "fee": first["fee"] * fraction, "slippage_cost": first["slippage_cost"] * fraction}
            buy, sell = (allocated_first, second) if first["side"] == "buy" else (second, allocated_first)
            gross = filled * (sell["price"] - buy["price"])
            fragment_net = gross - buy["fee"] - sell["fee"]
            pair["realized_fragment_net_pnl"] += fragment_net
            pair["shares"] -= filled
            pair["allocations"] = remaining_allocations
            pair["reserve"] *= pair["shares"] / quantity
            if pair["direction"] == "sell_first" and pair["shares"]:
                # 本金按股数同比释放；最低佣金不能被切成小于真实下一单的费用。
                reserved_value = pair["shares"] * pair["reserve_price"]
                pair["reserve"] = reserved_value + calculate_fee(reserved_value, "buy", timestamp, cfg)
            if pair["shares"] != sum(amount for _, amount in remaining_allocations):
                raise AssertionError("恢复剩余股数与原计划债务不一致")
            trips.append({"lot_id": pair["pair_id"], "pair_id": pair["pair_id"], "symbol": symbol,
                          "direction": pair["direction"], "shares": filled, "first_datetime": first["datetime"],
                          "second_datetime": timestamp, "buy_datetime": buy["datetime"], "sell_datetime": sell["datetime"],
                          "buy_price": buy["price"], "sell_price": sell["price"], "buy_fee": buy["fee"], "sell_fee": sell["fee"],
                          "gross_pnl": gross, "net_pnl": fragment_net,
                          "slippage_cost": buy["slippage_cost"] + sell["slippage_cost"],
                          "holding_days": (timestamp - first["datetime"]).total_seconds() / 86400,
                          "recovered_same_day": timestamp.normalize() == first["datetime"].normalize(), "reason": f"{prefix}恢复",
                          "fragment_id": second["fragment_id"], "original_pair_shares": pair["original_shares"],
                          "remaining_shares_after": pair["shares"], "pair_fully_recovered": not pair["shares"],
                          "planned_recovery_time": pair["planned_recovery_time"],
                          **{key: pair[key] for key in EXIT_EVENT_COLUMNS}})
            pair["exit_state"] = "partially_recovered" if pair["shares"] else "completed"
            if not pair["shares"]:
                completed_pair_ids.add(pair["pair_id"])
                del active[symbol]

        def execute_base_entries(timestamp: pd.Timestamp, pids: list[int]) -> None:
            """仅按已知请求和当前真实成交条件买入底仓；两个账户使用同一基础计划。"""
            for pid in pids:
                meta, symbol = plan_meta[pid], plan_meta[pid]["symbol"]
                manual = bool(meta["_manual"])
                if not manual and symbol in base_opened:
                    continue
                if timestamp.time() == pd.Timestamp("09:30").time():
                    quote = create_quote(day_rows.get(symbol), "open", symbol)
                elif timestamp.time() == pd.Timestamp("15:00").time():
                    quote = create_quote(day_rows.get(symbol), "close", symbol)
                else:
                    quote = create_current_recovery_quote(symbol, timestamp)
                reference = meta.get("universe_previous_close", np.nan)
                if not np.isfinite(reference):
                    reference = quote["previous_close"]
                requested = int(meta["_manual_shares"]) if manual else plan_balanced_quantity(reference, cfg)
                frozen = pd.notna(meta.get("frozen_base_shares", np.nan))
                if frozen:
                    requested = int(meta["frozen_base_shares"])
                filled, price, reason = calculate_quote_fill(quote, "buy", requested, baseline["used"].get(symbol, 0.0))
                if not manual and price > 0:
                    filled = min(filled, int((cfg["max_position_notional"] + 1e-7) / price / lot) * lot)
                planned = calculate_cash_quantity(filled, price, baseline["cash"] - base_cash_floor, timestamp)
                if planned < filled:
                    reason = "base_cash_reserve"
                if not manual and planned and planned * price < cfg["min_position_notional"] - 1e-7:
                    planned, reason = 0, "below_min_position_notional"
                if frozen:
                    if planned != requested or not np.isclose(quote["price"], meta["frozen_base_reference_price"], rtol=0, atol=1e-10):
                        raise ValueError("冻结基础入场量价与当前输入不一致")
                    expected_fee = calculate_fee(planned * price, "buy", timestamp, cfg)
                    if not np.isclose(expected_fee, meta["frozen_base_fee"], rtol=0, atol=1e-7):
                        raise ValueError("冻结基础入场费用发生变化")
                signal = pd.Timestamp(meta["_signal_time"])
                order_type = "manual" if manual else "automatic"
                base_fee = calculate_fee(planned * price, "buy", timestamp, cfg) if planned else 0.0
                if planned:
                    event = fill_book(baseline, symbol, [(pid, planned)], "buy", price, quote, timestamp,
                                      signal, "weekly", "人工基础加仓" if manual else "均衡金额首次底仓")
                    event["planned_shares"], event["base_order_type"] = requested, order_type
                    base_opened.add(symbol)
                if planned < requested or not requested:
                    record_attempt(baseline, symbol, pid, "buy", requested, planned, timestamp, signal, "weekly",
                                   reason or ("capacity" if requested else "no_lot_in_amount_band_or_missing_previous_price"))
                actual_filled, actual_price, actual_reason = calculate_quote_fill(quote, "buy", planned, actual["used"].get(symbol, 0.0))
                if actual_filled and actual_filled * actual_price + calculate_fee(actual_filled * actual_price, "buy", timestamp, cfg) > actual["cash"] - base_cash_floor - calculate_reserves() + 1e-7:
                    actual_filled, actual_reason = 0, "base_cash_or_t_repurchase_reserve"
                if not manual and actual_filled and actual_filled * actual_price < cfg["min_position_notional"] - 1e-7:
                    actual_filled, actual_reason = 0, "below_min_position_notional"
                actual_fee = calculate_fee(actual_filled * actual_price, "buy", timestamp, cfg) if actual_filled else 0.0
                if actual_filled:
                    event = fill_book(actual, symbol, [(pid, actual_filled)], "buy", actual_price, quote, timestamp,
                                      signal, "weekly", "人工基础加仓" if manual else "复制基准固定金额底仓")
                    event["planned_shares"], event["base_order_type"] = planned, order_type
                    if manual:
                        manual_adds[symbol] += actual_filled
                if actual_filled < planned:
                    record_attempt(actual, symbol, pid, "buy", planned, actual_filled, timestamp, signal, "weekly", actual_reason or "capacity")
                if actual_filled != planned:
                    mismatches.append({"date": date, "datetime": timestamp, "plan_id": pid, "symbol": symbol,
                                       "side": "buy", "baseline_shares": planned, "actual_shares": actual_filled})
                entry_plans.append({"datetime": timestamp, "plan_id": pid, "symbol": symbol,
                                    "known_reference_price": reference, "baseline_entry_equity": baseline_entry_equity,
                                    "entry_budget": cfg["target_position_notional"], "planned_shares": requested,
                                    "baseline_filled_shares": planned, "actual_filled_shares": actual_filled})
                amount_audits.append({"datetime": timestamp, "plan_id": pid, "symbol": symbol, "order_type": order_type,
                                     "known_reference_price": reference, "planned_shares": requested,
                                     "baseline_shares": planned, "actual_shares": actual_filled,
                                     "baseline_notional": planned * price, "actual_notional": actual_filled * actual_price,
                                     "baseline_fee": base_fee, "actual_fee": actual_fee,
                                     "min_notional": cfg["min_position_notional"], "max_notional": cfg["max_position_notional"],
                                     "target_notional": cfg["target_position_notional"],
                                     "baseline_in_band": bool(planned and cfg["min_position_notional"]-1e-7 <= planned*price <= cfg["max_position_notional"]+1e-7),
                                     "actual_in_band": bool(actual_filled and cfg["min_position_notional"]-1e-7 <= actual_filled*actual_price <= cfg["max_position_notional"]+1e-7),
                                     "manual_band_exception": manual, "reason": actual_reason if actual_filled < planned else reason})

        opening_time = date + pd.Timedelta(hours=9, minutes=30)
        for symbol in list(active):
            recover_pair(symbol, opening_time, "open")
        execute_base_entries(opening_time, entry_groups.get(opening_time, []))

        day_requests = request_groups.get(date, [])
        execution_groups = {}
        for request in day_requests:
            if pd.isna(request["execution_time"]):
                if request["action"]:
                    record_attempt(actual, request["symbol"], "", "buy" if request["action"] > 0 else "sell", 0, 0,
                                   request["datetime"] + pd.Timedelta(minutes=1), request["datetime"], "t_entry", "missing_execution_minute")
                continue
            execution_groups.setdefault(request["execution_time"], []).append(request)
        def execute_first_requests(timestamp: pd.Timestamp, group: list[dict]) -> None:
            """只处理当前分钟首腿；调用方已先恢复到期债务，不能在此重置分钟容量。"""
            nonlocal next_pair, peak_pending_t_notional
            for row in group:
                symbol, side = row["symbol"], "buy" if row["action"] > 0 else "sell"
                if not row["action"] or symbol not in plans_by_symbol:
                    continue
                ids = plans_by_symbol[symbol]
                old = sum(actual["sellable"][pid] for pid in ids)
                blocked = ""
                if symbol in active:
                    blocked = "pending_t_recovery"
                elif cfg["max_cycles_per_day"] and cycles[symbol] >= cfg["max_cycles_per_day"]:
                    blocked = "daily_pair_limit"
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
                if quantity:
                    opposite_side = "sell" if side == "buy" else "buy"
                    enabled = bool(cfg["t_recheck_exit_side_at_execution"])
                    opposite_reason = "not_checked"
                    if enabled:
                        # 原纯函数先检查quote及方向限价。零数量仅询问限价，不能
                        # 以虚拟反向整笔容量筛首腿，更不能调用fill_book/recover_pair。
                        _, _, opposite_reason = calculate_quote_fill(quote, opposite_side, 0, 0.0)
                        if opposite_reason not in ("", "price_limit", "conservative_price_limit_proxy"):
                            raise AssertionError("本方向已可填的同一quote出现非限价反向拒因")
                    cancelled = enabled and opposite_reason in ("price_limit", "conservative_price_limit_proxy")
                    cancel_reason = "execution_exit_side_" + opposite_reason if cancelled else ""
                    first_leg_exit_side_audits.append({
                        "datetime": timestamp, "signal_datetime": row["datetime"], "symbol": symbol,
                        "side": side, "opposite_side": opposite_side, "requested_shares": requested_quantity,
                        "own_fillable_shares": quantity, "guard_enabled": enabled,
                        "opposite_checked": enabled, "cancelled": cancelled, "reason": cancel_reason,
                        "opposite_quote_reason": opposite_reason, "reference_price": quote["price"],
                        **{key: quote[key] for key in ("high", "low", "amount", "volume", "previous_close", "limit_up", "limit_down")},
                        "minute_amount_used_before": actual["used"].get(symbol, 0.0),
                        "cash_before": actual["cash"], "old_sellable_before": old})
                    if cancelled:
                        record_attempt(actual, symbol, "", side, requested_quantity, 0, timestamp,
                                       row["datetime"], "t_entry", cancel_reason)
                        continue
                notional_clipped = False
                if quantity and pending_notional_limit is not None:
                    available_notional = max(pending_notional_limit - calculate_pending_t_notional(), 0.0)
                    capped_quantity = int((available_notional + 1e-7) / quote["price"] / lot) * lot
                    if capped_quantity < quantity:
                        quantity = capped_quantity
                        reason = "pending_t_notional_limit"
                        notional_clipped = True
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
                event["signal_priority"] = float(row["priority"])
                event["planned_recovery_time"] = row["planned_recovery_time"]
                event["original_deadline"] = row["planned_recovery_time"]
                if notional_clipped:
                    event["planned_shares"] = requested_quantity
                    record_attempt(actual, symbol, "", side, requested_quantity, quantity, timestamp,
                                   row["datetime"], "t_entry", "pending_t_notional_limit")
                reserve = 0.0
                reserve_price = 0.0
                if side == "sell":
                    reserve_price = quote["price"] * (1 + cfg["repurchase_buffer"]) * (1 + cfg["slippage_rate"])
                    reserve = quantity * reserve_price + calculate_fee(quantity * reserve_price, "buy", timestamp, cfg)
                active[symbol] = {"pair_id": next_pair, "shares": quantity, "original_shares": quantity, "allocations": allocations,
                                  "direction": "buy_first" if side == "buy" else "sell_first", "first": event,
                                  "reserve": reserve, "reserve_price": reserve_price, "recovery_row": row,
                                  "planned_recovery_time": row["planned_recovery_time"],
                                  "original_deadline": row["planned_recovery_time"],
                                  "exit_state": "monitoring", "exit_signal_time": pd.NaT,
                                  "exit_earliest_time": pd.NaT, "exit_reason": "",
                                  "normalization_notional": quantity * event["reference_price"],
                                  "realized_fragment_net_pnl": 0.0}
                peak_pending_t_notional = max(peak_pending_t_notional, calculate_pending_t_notional())
                next_pair += 1
                cycles[symbol] += 1

        close_time = date + pd.Timedelta(hours=15)
        # 所有模式使用相同完整会话时钟，包括真实缺行；不让未来可成交性
        # 或其他股票的决策决定某组是否有重试机会。
        timeline = {date + pd.Timedelta(minutes=minute) for minute in SESSION_MINUTES}
        timeline.update(execution_groups)
        timeline.update(time for time in entry_groups if time.normalize() == date and time > opening_time)
        for timestamp in sorted(timeline):
            # 09:30开盘成交使用09:31首根原始分钟bar的open和amount；完整会话
            # 新增09:31收盘恢复时，不能再使用该同一bar的第二份成交容量。
            if timestamp != date + pd.Timedelta(minutes=571):
                actual["used"], baseline["used"] = {}, {}
            current_quote_cache = {}
            for symbol in list(active):
                pair = active[symbol]
                if pair["exit_state"] == "monitoring" or timestamp < pair["exit_earliest_time"]:
                    continue
                recover_pair(symbol, timestamp, "minute_recovery", create_current_recovery_quote(symbol, timestamp))
            execute_base_entries(timestamp, entry_groups.get(timestamp, []))
            execute_first_requests(timestamp, execution_groups.get(timestamp, []))
            # 共同价格流按确定的持仓宇宙更新，而不是仅在T成交时更新。使用
            # 当前未滑点价；有效日收盘在循环后覆盖。无T和所有模式末价一致。
            for symbol in sorted(marked_symbols):
                quote = create_current_recovery_quote(symbol, timestamp)
                if np.isfinite(quote["price"]) and quote["price"] > 0:
                    marks[symbol], mark_times[symbol] = float(quote["price"]), timestamp
            # 观察是本分钟最后一个阶段；产生的意图不能回到上方成交阶段。
            for symbol in list(active):
                observe_and_latch_exit(symbol, timestamp)

        # 每日按真实收盘价估值；固定底仓没有周退出或期末卖出分支。
        baseline["used"] = {}
        for symbol, row in day_rows.items():
            if np.isfinite(row["close"]) and row["close"] > 0:
                marks[symbol], mark_times[symbol] = float(row["close"]), close_time
        actual_total, base_total = aggregate_quantities(actual), aggregate_quantities(baseline)
        held_value = sum(quantity * marks.get(symbol, 0.0) for symbol, quantity in actual_total.items())
        baseline_equity = baseline["cash"] + sum(quantity * marks.get(symbol, 0.0) for symbol, quantity in base_total.items())
        equity = actual["cash"] + held_value
        curves.append({"datetime": close_time, "equity": equity, "cash": actual["cash"], "holdings_value": held_value,
                       "baseline_equity": baseline_equity, "baseline_cash": baseline["cash"], "t_added_equity": equity - baseline_equity,
                       "t_realized_pnl": sum(trip["net_pnl"] for trip in trips), "pending_pair_count": len(active),
                       "completed_pair_count": len(completed_pair_ids), "fragment_count": len(trips),
                       "pending_t_notional": calculate_pending_t_notional(),
                       "repurchase_reserved_cash": calculate_reserves(),
                       "holding_count": sum(quantity > 0 for quantity in actual_total.values()),
                       "open_lots": sum(quantity > 0 for quantity in actual["qty"].values()),
                       "exposure": held_value / equity if equity else 0.0,
                       "stale_price_count": sum(quantity > 0 and mark_times.get(symbol) != close_time for symbol, quantity in actual_total.items())})
        for symbol, ids in plans_by_symbol.items():
            if not (opening[symbol] or actual_total[symbol] or base_total[symbol] or weekly_buys[symbol] or weekly_sells[symbol] or symbol in active):
                continue
            target = opening[symbol] + weekly_buys[symbol] - weekly_sells[symbol]
            difference = actual_total[symbol] - target
            overdue = False
            audits.append({"date": date, "symbol": symbol, "opening_shares": opening[symbol], "opening_sellable": opening[symbol],
                           "weekly_buys": weekly_buys[symbol], "weekly_sells": weekly_sells[symbol],
                           "weekly_buy_target": base_buys[symbol], "weekly_sell_target": base_sells[symbol],
                           "target_shares": target, "closing_shares": actual_total[symbol], "inventory_difference": difference,
                           "inventory_preserved": difference == 0, "day_sold": day_sold[symbol],
                           "closing_sellable": sum(actual["sellable"][pid] for pid in ids), "baseline_closing_shares": base_total[symbol],
                           "baseline_inventory_difference": actual_total[symbol] - base_total[symbol], "t1_violation_count": 0,
                           "recovery_failure": symbol in active, "weekly_liquidation_failure": overdue,
                           "cycles_started": cycles[symbol], "stale_price": mark_times.get(symbol) != close_time,
                           "baseline_opening_shares": baseline_opening[symbol], "base_buys": weekly_buys[symbol] - manual_adds[symbol],
                           "manual_buys": manual_adds[symbol]})
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
    if len(trip_frame):
        trip_frame["pair_fully_recovered"] = trip_frame.pair_id.isin(completed_pair_ids)
    audit_frame = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    holdings = []
    for pid, meta in plan_meta.items():
        symbol = meta["symbol"]
        pair = active.get(symbol, {})
        pending = next((amount for plan, amount in pair.get("allocations", []) if plan == pid), 0)
        if actual["qty"][pid] or baseline["qty"][pid] or pending:
            holdings.append({"plan_id": pid, "symbol": symbol, "trade_week": meta["trade_week"],
                             "entry_date": meta["entry_date"], "exit_date": pd.NaT, "source_exit_date": meta["exit_date"],
                             "shares": actual["qty"][pid], "baseline_shares": baseline["qty"][pid],
                             "mark_price": marks.get(symbol, np.nan), "market_value": actual["qty"][pid] * marks.get(symbol, 0.0),
                             "mark_datetime": mark_times.get(symbol), "inventory_difference": actual["qty"][pid] - baseline["qty"][pid],
                             "overdue_weekly_exit": False, "pending_pair_id": pair.get("pair_id", 0),
                             "pending_direction": pair.get("direction", ""), "pending_shares": pending,
                             **{key: pair.get(key, "" if key == "exit_reason" else pd.NaT) for key in EXIT_EVENT_COLUMNS},
                             "exit_state": pair.get("exit_state", "")})
    metrics = calculate_daily_metrics(curve, event_frame, trip_frame, cfg["initial_cash"])
    completed_trips = trip_frame.loc[trip_frame.pair_id.isin(completed_pair_ids)]
    completed_profits = completed_trips.groupby("pair_id").net_pnl.sum() if len(completed_trips) else pd.Series(dtype=float)
    metrics.update({"strategy": "dynamic_fixed_inventory_t", "total_t_net_profit": float(trip_frame.net_pnl.sum()) if len(trip_frame) else 0.0,
                    "total_t_marked_pnl": float(curve.t_added_equity.iloc[-1]), "t_cashflow": t_cashflow,
                    "t_pair_count": len(completed_pair_ids), "completed_pair_count": len(completed_pair_ids),
                    "fragment_count": len(trip_frame), "t_fragment_count": len(trip_frame),
                    "started_pair_count": next_pair - 1, "pending_pairs": len(active),
                    "max_pending_t_notional_fraction": pending_fraction,
                    "pending_t_notional_limit": pending_notional_limit,
                    "peak_pending_t_notional": peak_pending_t_notional,
                    "final_pending_t_notional": calculate_pending_t_notional(),
                    "closed_lot_count": len(completed_pair_ids), "completed_pair_net_profit": float(completed_profits.sum()),
                    "win_rate": float((completed_profits > 0).mean()) if len(completed_profits) else 0.0,
                    "recovery_execution_mode": "full_session_latched_minute_partial",
                    "intraday_allocation_order": "priority_desc_symbol_asc",
                    "same_minute_event_order": "due_recovery_then_base_buy_then_t_first_leg_then_current_observation",
                    "scheduled_recovery_enabled": bool(fixed_recovery_minutes) or bool(requests.get("explicit_recovery_schedule", pd.Series(dtype=bool)).any()),
                    "scheduled_recovery_times": [f"{minute // 60:02d}:{minute % 60:02d}" for minute in fixed_recovery_minutes],
                    "t_fee": float(event_frame.loc[event_frame.event_type == "t", "fee"].sum()),
                    "baseline_total_return": float(curve.baseline_equity.iloc[-1] / cfg["initial_cash"] - 1),
                    "base_plan_mismatch_count": len(mismatches), "same_base_plan": not mismatches,
                    "inventory_mismatch_days": int(audit_frame.loc[~audit_frame.inventory_preserved, "date"].nunique()) if len(audit_frame) else 0,
                    "recovery_failure_days": int(audit_frame.loc[audit_frame.recovery_failure, "date"].nunique()) if len(audit_frame) else 0,
                    "weekly_liquidation_failure_days": int(audit_frame.loc[audit_frame.weekly_liquidation_failure, "date"].nunique()) if len(audit_frame) else 0,
                    "weekly_exit_failure_days": int(audit_frame.loc[audit_frame.weekly_liquidation_failure, "date"].nunique()) if len(audit_frame) else 0,
                    "weekly_unclosed_positions": 0,
                    "baseline_weekly_unclosed_positions": 0,
                    "t1_violation_count": 0, "flat_bar_fill_count": flat_fills, "limit_guard_unknown_fill_count": limit_unknown_fills,
                    "stale_valuation_days": int((curve.stale_price_count > 0).sum()),
                    "weekly_buy_count": int(((event_frame.event_type == "weekly") & (event_frame.side == "buy")).sum()),
                    "weekly_sell_count": int(((event_frame.event_type == "weekly") & (event_frame.side == "sell")).sum()),
                    "blocked_order_count": len(attempts), "short_exposure": 0})
    t_events = event_frame.loc[event_frame.event_type.eq("t")]
    first_events = t_events.loc[t_events.fragment_id.eq(0)]
    metrics.update({"no_terminal_liquidation": True, "base_cash_floor": base_cash_floor,
                    "target_position_notional": cfg["target_position_notional"],
                    "min_position_notional": cfg["min_position_notional"], "max_position_notional": cfg["max_position_notional"],
                    "max_cycles_per_day": cfg["max_cycles_per_day"],
                    "base_buy_count": int(event_frame.base_order_type.eq("automatic").sum()),
                    "manual_buy_count": int(event_frame.base_order_type.eq("manual").sum()),
                    "total_t_traded_lots": float(t_events.shares.sum() / lot),
                    "completed_t_pair_lots": float(first_events.loc[first_events.pair_id.isin(completed_pair_ids), "shares"].sum() / lot),
                    "started_t_pair_lots": float(first_events.shares.sum() / lot),
                    "final_base_position_count": sum(quantity > 0 for quantity in aggregate_quantities(baseline).values()),
                    "automatic_amount_band_violation_count": sum(row["actual_shares"] > 0 and not row["actual_in_band"] and not row["manual_band_exception"] for row in amount_audits)})
    metrics.update({"dynamic_exit_mode": cfg["dynamic_exit_mode"],
                    "dynamic_take_net_return": cfg["dynamic_take_net_return"],
                    "dynamic_stop_net_return": cfg["dynamic_stop_net_return"],
                    "exit_decision_count": len(exit_decisions),
                    "profit_exit_decision_count": sum(row["exit_reason"] == "profit_buffer" for row in exit_decisions),
                    "stop_exit_decision_count": sum(row["exit_reason"] == "stop_loss" for row in exit_decisions),
                    "deadline_exit_decision_count": sum(row["exit_reason"] == "hard_deadline" for row in exit_decisions),
                    "common_mark_mode": "all_held_symbols_current_completed_quote_then_valid_daily_close",
                    "opening_0930_and_0931_share_raw_bar_capacity": True,
                    "recovery_session_minute_count": len(SESSION_MINUTES),
                    "recovery_store_provided": recovery_lookup is not None})
    return {"equity_curve": curve, "trade_events": event_frame, "round_trips": trip_frame,
            "open_positions": pd.DataFrame(holdings, columns=POSITION_COLUMNS), "daily_inventory_audit": audit_frame,
            "order_attempts": pd.DataFrame(attempts, columns=ATTEMPT_COLUMNS), "metrics": metrics,
            "baseline_events": base_frame, "base_plan_mismatches": pd.DataFrame(mismatches),
            "weekly_entry_plans": pd.DataFrame(entry_plans),
            "position_amount_audit": pd.DataFrame(amount_audits, columns=AMOUNT_COLUMNS),
            "market_dates": dates.copy(),
            "exit_decisions": pd.DataFrame(exit_decisions, columns=EXIT_DECISION_COLUMNS),
            "first_leg_exit_side_audit": pd.DataFrame(first_leg_exit_side_audits,
                                                      columns=FIRST_LEG_EXIT_SIDE_AUDIT_COLUMNS)}
