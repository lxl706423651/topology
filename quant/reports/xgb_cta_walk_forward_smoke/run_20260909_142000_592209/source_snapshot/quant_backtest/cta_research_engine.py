"""三板块独立批次CTA撮合：下一分钟成交、T+1、板块数量限制和共享现金。

输入为只读分钟缓存及事先冻结的信号，输出实际成交/分批配对/未成交原因/净值。
Numba只加速确定性账本；信号不读取未来，撮合时才读取该分钟报价与成交容量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .annual_portfolio import calculate_daily_metrics

EXIT_NAMES = {1: "触及初始ATR止损", 2: "触及ATR跟踪止盈保护线", 3: "价格跌破日EMA20趋势失效",
              4: "历史ST状态变更，禁止加仓并申请退出", 5: "达到最大持仓交易日", 0: "CTA趋势条件及模型净收益门槛通过"}
BLOCK_NAMES = {1: "行情/状态缺失或无成交", 2: "涨跌停价格禁止交易", 3: "成交容量或板块最小数量不足",
               4: "T+1尚不可卖", 5: "现金或单股/总仓位上限", 6: "当日已买入或刚平仓，不重复开仓",
               7: "下一分钟价格超过委托保护范围", 8: "尾盘不再新开仓"}
EVENT_COLUMNS = ["time_index", "signal_index", "lot_id", "asset", "side_code", "shares", "price", "notional", "fee",
                 "reason_code", "signal_equity", "exit_profile", "prediction", "reference_price", "cash_after", "post_trade_equity"]
TRIP_COLUMNS = ["lot_id", "asset", "buy_index", "sell_index", "buy_signal_index", "sell_signal_index", "buy_price",
                "sell_price", "shares", "buy_fee", "sell_fee", "net_pnl", "reason_code", "exit_profile"]


@njit(cache=True)
def calculate_quantity(requested, budget, price, capacity, board, buying):
    """以requested股、budget元和capacity元限制数量；board=2按科创200起步/1股递增。"""
    if price <= 0 or not np.isfinite(price) or budget <= 0 or capacity <= 0:
        return 0
    increment = 1 if board == 2 else 100
    maximum = 100000 if board == 2 else (300000 if board == 1 else 1000000)
    shares = min(requested, maximum, int(min(budget, capacity) / price + 1e-9) // increment * increment)
    if board == 2 and shares < 200 and (buying or requested >= 200 or shares != requested):
        return 0
    return max(0, shares)


@njit(cache=True)
def calculate_fill(quote, flags, side, requested, budget, used, board, parameters, reference):
    """在当前quote/flags下撮合side请求；parameters给费用/容量，reference为上一完成分钟委托参考价。"""
    close, amount, upper, lower = quote[0], quote[3], quote[6], quote[7]
    allowed = flags[0] if side == 1 else flags[1]
    if not allowed or not np.isfinite(close) or close <= 0 or not np.isfinite(amount) or amount <= 0:
        return 0, 0.0, 1
    if not np.isfinite(upper) or not np.isfinite(lower) or close >= upper - 1e-8 or close <= lower + 1e-8:
        return 0, 0.0, 2
    price = close * (1 + parameters[6] * side)
    price = (np.ceil(price * 100 - 1e-8) if side == 1 else np.floor(price * 100 + 1e-8)) / 100
    if price >= upper - 1e-8 or price <= lower + 1e-8:
        return 0, 0.0, 2
    # 无逐笔盘口，用前一分钟参考价的2%保护范围保守近似可成交限价单。
    if np.isfinite(reference) and reference > 0:
        if (side == 1 and price > reference * 1.02) or (side == -1 and price < reference * 0.98):
            return 0, 0.0, 7
    capacity = max(0.0, amount * parameters[7] - used)
    quantity = calculate_quantity(requested, budget, price, capacity, board, side == 1)
    return quantity, price, 0 if quantity > 0 else 3


@njit(cache=True)
def calculate_trade_fee(notional, selling, stamp, parameters):
    """按parameters双边佣金/最低费/过户及stamp卖出印花税计费。"""
    return max(notional * parameters[4], parameters[8]) + notional * parameters[5] + (notional * stamp if selling else 0.0)


@njit(cache=True)
def update_exit(quote, is_st, high_watermark, stop, buy_price, entry_atr, elapsed, profile):
    """依据当前已完成quote和profile更新独立止损；先检查旧线，避免假设分钟内高低价先后。"""
    close, high, low, atr, ema = quote[0], quote[1], quote[2], quote[4], quote[5]
    reason = 0
    if np.isfinite(low) and low <= stop:
        reason = 2 if stop > buy_price - profile[0] * entry_atr + 1e-8 else 1
    if np.isfinite(high):
        high_watermark = max(high_watermark, high)
    if np.isfinite(atr) and atr > 0:
        stop = max(stop, high_watermark - profile[1] * atr)
    if reason == 0 and np.isfinite(close) and close <= stop:
        reason = 2
    if reason == 0 and is_st:
        reason = 4
    if reason == 0 and profile[3] and np.isfinite(close) and np.isfinite(ema) and close < ema:
        reason = 3
    if reason == 0 and elapsed >= int(profile[2]) * 240:
        reason = 5
    return high_watermark, stop, reason


@njit(cache=True)
def simulate_path_labels(quotes, flags, boards, signal_indices, assets, profiles, parameters, stamps):
    """在signal_indices/assets上计算各profiles的独立10%预算交易净收益与真实结束时点。

    只用于监督标签。无法入场记0；入场后超过max_days+10交易日仍无法清仓记NaN，
    不把最后价格当成可成交退出；后续推断不按标签是否可用过滤。
    """
    values = np.full((len(signal_indices), len(profiles)), np.nan)
    ends = np.full((len(signal_indices), len(profiles)), -1, dtype=np.int64)
    for row in range(len(signal_indices)):
        signal, asset = signal_indices[row], assets[row]
        entry = signal + 1
        if entry >= len(quotes) or entry // 240 != signal // 240 or signal % 240 >= 236:
            continue
        current = quotes[signal, asset]
        if not np.isfinite(current[0]) or not np.isfinite(current[4]) or current[4] <= 0:
            continue
        budget = parameters[0] * parameters[1]
        increment = 1 if boards[asset] == 2 else 100
        request = int(budget / current[0]) // increment * increment
        qty, price, reason = calculate_fill(quotes[entry, asset], flags[entry, asset], 1, request, budget, 0,
                                           boards[asset], parameters, current[0])
        if qty == 0:
            values[row, :] = 0.0
            ends[row, :] = entry
            continue
        entry_cost = qty * price + calculate_trade_fee(qty * price, False, stamps[entry], parameters)
        for p in range(len(profiles)):
            profile = profiles[p]
            remaining, proceeds, highest = qty, 0.0, price
            stop = price - profile[0] * current[4]
            pending = 0
            limit = min(len(quotes), entry + (int(profile[2]) + 10) * 240)
            for t in range(entry + 1, limit):
                if pending and t // 240 > entry // 240 and t % 240 < 237:
                    reference = quotes[t - 1, asset, 0]
                    amount, exit_price, blocked = calculate_fill(quotes[t, asset], flags[t, asset], -1, remaining,
                        np.inf, 0, boards[asset], parameters, reference)
                    if amount:
                        notional = amount * exit_price
                        proceeds += notional - calculate_trade_fee(notional, True, stamps[t], parameters)
                        remaining -= amount
                        if remaining == 0:
                            values[row, p] = proceeds / entry_cost - 1
                            ends[row, p] = t
                            break
                highest, stop, triggered = update_exit(quotes[t, asset], flags[t, asset, 2], highest, stop,
                                                       price, current[4], t - entry, profile)
                if not pending and triggered:
                    pending = triggered
    return values, ends


@njit(cache=True)
def simulate_account(quotes, flags, boards, candidates, predictions, profile_ids, profiles, parameters, stamps, start, end):
    """在[start,end)分钟执行冻结candidates；最多4096独立批次，不跨账户借用现金。

    每分钟先处理此前退出/买入请求，后更新保护线和生成新买入预算。返回数值账本
    供Python映射中文；各批次保存入场时退出方案，年度换模型不改变旧批次的规则。
    """
    capacity = max(10000, (end - start) * 10)
    events = np.empty((capacity, 16), dtype=np.float64)
    trips = np.empty((capacity, 14), dtype=np.float64)
    blocks = np.empty((capacity, 7), dtype=np.float64)
    curve = np.empty((end - start, 6), dtype=np.float64)
    # 股数为0代表空槽；列为股票/股数/买时/价/费/信号/ATR/高点/止损/待退出/触发时/规则/分数/批次号。
    lots = np.zeros((4096, 14), dtype=np.float64)
    last_block = np.full(4096, -1, dtype=np.int64)
    marks = np.full(len(boards), np.nan)
    last_buy_day = np.full(len(boards), -1, dtype=np.int64)
    last_sell_day = np.full(len(boards), -1, dtype=np.int64)
    used = np.zeros(len(boards))
    ec, tc, bc, next_id = 0, 0, 0, 1
    cash = parameters[0]
    signal_equity = cash
    active_slots = 0
    for t in range(start, end):
        day = t // 240
        used[:] = 0
        held = 0.0
        for k in range(active_slots):
            if lots[k, 1] > 0:
                asset = int(lots[k, 0])
                observed = quotes[t, asset, 0]
                if np.isfinite(observed) and observed > 0:
                    marks[asset] = observed
                held += lots[k, 1] * marks[asset]
        equity = cash + held
        for k in range(active_slots):
            lot = lots[k]
            if lot[1] <= 0 or not lot[9] or t <= lot[10]:
                continue
            asset, buy_t = int(lot[0]), int(lot[2])
            failure = 0
            if day <= buy_t // 240:
                failure = 4
            elif t % 240 >= 237:
                continue
            else:
                quantity, price, failure = calculate_fill(quotes[t, asset], flags[t, asset], -1, int(lot[1]),
                    np.inf, used[asset], boards[asset], parameters, quotes[t - 1, asset, 0])
                if quantity:
                    value = quantity * price
                    fee = calculate_trade_fee(value, True, stamps[t], parameters)
                    allocated = lot[4] * quantity / lot[1]
                    cash += value - fee
                    used[asset] += value
                    held -= quantity * marks[asset]
                    equity = cash + held
                    events[ec] = np.array([t, t - 1, lot[13], asset, -1, quantity, price, value, fee,
                        lot[9], signal_equity, lot[11], lot[12], quotes[t, asset, 0], cash, equity])
                    trips[tc] = np.array([lot[13], asset, buy_t, t, lot[5], lot[10], lot[3], price,
                        quantity, allocated, fee, quantity * (price - lot[3]) - allocated - fee, lot[9], lot[11]])
                    ec += 1
                    tc += 1
                    lot[1] -= quantity
                    lot[4] -= allocated
                    last_sell_day[asset] = day
            if failure:
                previous = last_block[k]
                if previous >= 0 and int(blocks[previous, 0]) // 240 == day and blocks[previous, 4] == failure:
                    blocks[previous, 6] += 1
                else:
                    blocks[bc] = np.array([t, lot[10], asset, -1, failure, lot[13], 1])
                    last_block[k] = bc
                    bc += 1
        # 买单仅在信号后紧接的下一分钟尝试一次，不能隔夜按旧分数追单。
        if t > start and t % 240 > 0 and t % 240 < 237:
            signal = t - 1
            for rank in range(candidates.shape[1]):
                asset = candidates[signal, rank]
                if asset < 0:
                    continue
                failure = 0
                if last_buy_day[asset] >= day or last_sell_day[asset] >= day:
                    failure = 6
                else:
                    profile_id = profile_ids[signal, rank]
                    current = quotes[signal, asset]
                    observed = quotes[t, asset, 0]
                    if np.isfinite(observed) and observed > 0:
                        marks[asset] = observed
                    budget = signal_equity * parameters[1]
                    symbol_value = 0.0
                    for k in range(active_slots):
                        if lots[k, 1] > 0 and int(lots[k, 0]) == asset:
                            symbol_value += lots[k, 1] * marks[asset]
                    allowed_budget = min(budget, cash / (1 + parameters[4] + parameters[5]),
                        parameters[2] * equity - symbol_value, parameters[3] * equity - held)
                    if allowed_budget <= 0 or not np.isfinite(current[4]) or current[4] <= 0:
                        failure = 5
                    else:
                        step = 1 if boards[asset] == 2 else 100
                        requested = int(budget / current[0]) // step * step
                        quantity, price, failure = calculate_fill(quotes[t, asset], flags[t, asset], 1, requested,
                            allowed_budget, used[asset], boards[asset], parameters, current[0])
                        while quantity > 0:
                            value = quantity * price
                            fee = calculate_trade_fee(value, False, stamps[t], parameters)
                            after_equity = equity - quantity * (price - marks[asset]) - fee
                            if (cash >= value + fee and symbol_value + quantity * marks[asset] <= parameters[2] * after_equity + 1e-8
                                    and held + quantity * marks[asset] <= parameters[3] * after_equity + 1e-8):
                                break
                            quantity -= step
                        if boards[asset] == 2 and quantity < 200:
                            quantity = 0
                        if quantity <= 0:
                            failure = failure if failure else 5
                        else:
                            slot = -1
                            for k in range(active_slots):
                                if lots[k, 1] == 0:
                                    slot = k
                                    break
                            if slot < 0:
                                slot = active_slots
                                active_slots += 1
                            if active_slots > len(lots):
                                raise ValueError("独立持仓批次超出容量，拒绝丢弃持仓")
                            cash -= value + fee
                            held += quantity * marks[asset]
                            equity = cash + held
                            used[asset] += value
                            score = predictions[signal, rank]
                            stop = price - profiles[profile_id, 0] * current[4]
                            lots[slot] = np.array([asset, quantity, t, price, fee, signal, current[4], price,
                                                  stop, 0, -1, profile_id, score, next_id])
                            last_block[slot] = -1
                            events[ec] = np.array([t, signal, next_id, asset, 1, quantity, price, value, fee,
                                0, signal_equity, profile_id, score, quotes[t, asset, 0], cash, equity])
                            ec += 1
                            next_id += 1
                            last_buy_day[asset] = day
                if failure:
                    blocks[bc] = np.array([t, signal, asset, 1, failure, -1, 1])
                    bc += 1
        count = 0
        for k in range(active_slots):
            lot = lots[k]
            if lot[1] <= 0:
                continue
            count += 1
            asset = int(lot[0])
            if t > lot[2]:
                highest, stop, reason = update_exit(quotes[t, asset], flags[t, asset, 2], lot[7], lot[8],
                    lot[3], lot[6], t - int(lot[2]), profiles[int(lot[11])])
                lot[7], lot[8] = highest, stop
                if reason and not lot[9]:
                    lot[9], lot[10] = reason, t
        if cash < -1e-5 or not np.isfinite(equity) or equity <= 0:
            raise ValueError("现金或权益账本异常")
        curve[t - start] = np.array([t, equity, cash, held / equity, count, held])
        signal_equity = equity
        if ec >= capacity - 4096 or bc >= capacity - 4096:
            raise ValueError("交易/拒单记录容量不足，拒绝静默截断")
    return curve, events[:ec], trips[:tc], blocks[:bc], lots[:active_slots], marks


def create_parameters(cfg: dict, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把cfg费用/仓位/退出参数与times日期费率转换为只读数值数组。"""
    parameters = np.array([cfg[k] for k in ["initial_cash", "entry_equity_fraction", "max_symbol_exposure",
        "max_total_exposure", "commission_rate", "transfer_rate", "slippage_rate", "max_bar_amount_fraction", "min_commission"]])
    profiles = np.array([[p["initial_atr"], p["trailing_atr"], p["max_days"], float(p["trend_exit"])] for p in cfg["exit_profiles"]])
    stamps = np.where(times >= pd.Timestamp("2023-08-28"), cfg["stamp_rate_from_20230828"], cfg["stamp_rate_before_20230828"])
    return parameters, profiles, stamps


def run_portfolio(signals: pd.DataFrame, market, start: pd.Timestamp, end: pd.Timestamp, cfg: dict) -> dict:
    """把冻结signals放入market的[start,end)共享账户；cfg提供资金/费用，不接收标签列。"""
    required = ["datetime", "symbol", "prediction", "exit_profile"]
    if not set(required).issubset(signals):
        raise ValueError(f"信号缺少{required}")
    signals = signals[required].sort_values(["datetime", "prediction", "symbol"], ascending=[True, False, True]).copy()
    shape = (len(market.times), cfg["entry_top_n"])
    candidates = np.full(shape, -1, dtype=np.int64)
    predictions = np.zeros(shape)
    profile_ids = np.zeros(shape, dtype=np.int64)
    ranks = signals.groupby("datetime").cumcount().to_numpy()
    valid = ranks < cfg["entry_top_n"]
    signals = signals.loc[valid]
    ranks = ranks[valid]
    time_indices = market.times.get_indexer(signals.datetime)
    if (time_indices < 0).any() or signals.duplicated(["datetime", "symbol"]).any():
        raise ValueError("信号时钟或股票键非法")
    candidates[time_indices, ranks] = [market.symbol_index[s] for s in signals.symbol]
    predictions[time_indices, ranks] = signals.prediction
    profile_ids[time_indices, ranks] = signals.exit_profile.astype(int)
    parameters, profiles, stamps = create_parameters(cfg, market.times)
    begin, finish = market.times.searchsorted(start), market.times.searchsorted(end)
    if finish <= begin:
        raise ValueError("回测区间没有行情")
    arrays = simulate_account(market.quotes, market.flags, market.boards, candidates, predictions, profile_ids,
                              profiles, parameters, stamps, begin, finish)
    curve, event_array, trip_array, block_array, lots, marks = arrays
    result = pd.DataFrame(curve, columns=["time_index", "equity", "cash", "exposure", "open_lots", "holding_value"])
    result["datetime"] = market.times[result.time_index.astype(int)]
    # 日末估值保留全部现金日，分钟路径只供撮合和审计，不按分钟频率虚增夏普。
    events = pd.DataFrame(event_array, columns=EVENT_COLUMNS)
    for name in ["time_index", "signal_index", "lot_id", "asset", "shares", "exit_profile"]:
        events[name] = events[name].astype(int)
    events["datetime"] = market.times[events.time_index]
    events["signal_datetime"] = market.times[events.signal_index]
    events["symbol"] = events.asset.map(dict(enumerate(market.symbols)))
    events["side"] = events.side_code.map({1: "buy", -1: "sell"})
    events["reason"] = events.reason_code.map(EXIT_NAMES)
    trips = pd.DataFrame(trip_array, columns=TRIP_COLUMNS)
    for name in ["lot_id", "asset", "buy_index", "sell_index", "buy_signal_index", "sell_signal_index", "shares"]:
        trips[name] = trips[name].astype(int)
    for key, idx in [("buy_datetime", "buy_index"), ("sell_datetime", "sell_index"),
                     ("buy_signal_datetime", "buy_signal_index"), ("sell_signal_datetime", "sell_signal_index")]:
        trips[key] = market.times[trips[idx]]
    trips["symbol"] = trips.asset.map(dict(enumerate(market.symbols)))
    trips["holding_days"] = (trips.sell_datetime - trips.buy_datetime).dt.total_seconds() / 86400
    trips["buy_reason"] = EXIT_NAMES[0]
    trips["sell_reason"] = trips.reason_code.map(EXIT_NAMES)
    opened = []
    for lot in lots:
        if lot[1] > 0:
            asset = int(lot[0])
            opened.append({"lot_id": int(lot[13]), "symbol": market.symbols[asset], "shares": int(lot[1]),
                "buy_datetime": market.times[int(lot[2])], "buy_price": lot[3], "remaining_buy_fee": lot[4],
                "mark_price": marks[asset], "market_value": lot[1] * marks[asset], "stop_price": lot[8],
                "pending_reason": EXIT_NAMES.get(int(lot[9]), "") if lot[9] else "", "exit_profile": int(lot[11])})
    open_frame = pd.DataFrame(opened, columns=["lot_id", "symbol", "shares", "buy_datetime", "buy_price", "remaining_buy_fee",
                                             "mark_price", "market_value", "stop_price", "pending_reason", "exit_profile"])
    blocked = pd.DataFrame(block_array, columns=["time_index", "signal_index", "asset", "side_code", "reason_code", "lot_id", "attempt_count"])
    blocked["datetime"] = market.times[blocked.time_index.astype(int)]
    blocked["symbol"] = blocked.asset.map(dict(enumerate(market.symbols)))
    blocked["reason"] = blocked.reason_code.map(BLOCK_NAMES)
    metrics = calculate_daily_metrics(result, events, trips, cfg["initial_cash"], set(open_frame.lot_id))
    # 科创板可以逐股交易，手数只是股数/100的统计单位，不再暗示全部成交为整手。
    metrics["signal_count"] = len(signals)
    metrics["star_buy_count"] = int((events.side.eq("buy") & events.symbol.str.startswith("SH.688")).sum())
    metrics["chinext_buy_count"] = int((events.side.eq("buy") & events.symbol.str.startswith(("SZ.300", "SZ.301"))).sum())
    return {"curve": result, "events": events, "trips": trips, "open": open_frame, "blocked_events": blocked,
            "blocked": blocked.groupby("reason").attempt_count.sum().to_dict(), "metrics": metrics, "initial_cash": cfg["initial_cash"]}
