"""金额均衡的固定底仓做T；保留旧引擎/缓存身份，以真实分片恢复遵守T+1。

旧周排名仅作为首次买入候选，绝不触发基础卖出；期末持股按真实最后价格估值。
每股金额限制针对自动建仓成交，不是持有期间随价格变化的市值保证。
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .annual_portfolio import calculate_daily_metrics, calculate_fee
from .inventory_t import DEFAULT_CONFIG, calculate_next_trading_minute
from .weekly_t_engine import (RecoveryBarStore, prepare_recovery_bars, read_fixed_recovery_minutes,
                             is_recovery_minute, create_planned_recovery_clock,
                             EVENT_COLUMNS as OLD_EVENT_COLUMNS, TRIP_COLUMNS, AUDIT_COLUMNS as OLD_AUDIT_COLUMNS,
                             POSITION_COLUMNS as OLD_POSITION_COLUMNS, ATTEMPT_COLUMNS)

EVENT_COLUMNS = [*OLD_EVENT_COLUMNS, "base_order_type"]
AUDIT_COLUMNS = [*OLD_AUDIT_COLUMNS, "baseline_opening_shares", "base_buys", "manual_buys"]
POSITION_COLUMNS = [*OLD_POSITION_COLUMNS, "source_exit_date"]
AMOUNT_COLUMNS = ["datetime", "plan_id", "symbol", "order_type", "known_reference_price", "planned_shares",
                  "baseline_shares", "actual_shares", "baseline_notional", "actual_notional", "baseline_fee", "actual_fee",
                  "min_notional", "max_notional", "target_notional", "baseline_in_band", "actual_in_band",
                  "manual_band_exception", "reason"]
BALANCED_DEFAULTS = {**DEFAULT_CONFIG, "base_fraction": .85, "target_position_notional": 15000.,
                     "min_position_notional": 10000., "max_position_notional": 20000., "max_cycles_per_day": 0,
                     "slippage_rate": .0005, "max_bar_amount_fraction": .05,
                     "flat_st_limit_rate": .05, "nonflat_limit_proxy": .098}
BASE_IDENTITY_KEYS = ("initial_cash", "base_fraction", "target_position_notional", "min_position_notional",
                      "max_position_notional", "lot_size", "commission_rate", "min_commission", "transfer_rate",
                      "stamp_rate_before_20230828", "stamp_rate_from_20230828", "slippage_rate",
                      "max_bar_amount_fraction", "flat_st_limit_rate", "nonflat_limit_proxy")

def base_config_identity(cfg: dict) -> str:
    """冻结基础入场经济参数，候选不能悄悄改变金额或费用。"""
    full = {**BALANCED_DEFAULTS, **cfg}
    return hashlib.sha256(json.dumps({key: full[key] for key in BASE_IDENTITY_KEYS}, sort_keys=True).encode()).hexdigest()

def validate_balanced_config(cfg: dict) -> None:
    """拒绝不合法金额/次数；0次配置表示仅受真实旧股预算限制。"""
    low, target, high = [cfg[key] for key in ("min_position_notional", "target_position_notional", "max_position_notional")]
    if not all(np.isfinite(value) for value in (low, target, high)) or not 0 < low <= target <= high:
        raise ValueError("必须满足0<最小建仓名义额<=目标<=最大")
    value = cfg["max_cycles_per_day"]
    if not np.isfinite(value) or int(value) != value or value < 0:
        raise ValueError("max_cycles_per_day必须为非负整数，0不额外限次数")

def plan_balanced_quantity(reference: float, cfg: dict) -> int:
    """按信号时已知前收计划最接近目标的合法整手，绝不读取成交后价格。"""
    if not np.isfinite(reference) or reference <= 0:
        return 0
    value = reference * (1 + cfg["slippage_rate"]) * cfg["lot_size"]
    low = max(1, int(np.ceil((cfg["min_position_notional"] - 1e-7) / value)))
    high = int(np.floor((cfg["max_position_notional"] + 1e-7) / value))
    if high < low:
        return 0
    # 最近整手，半手恰好相等时取较少股数。
    near = int(np.floor(cfg["target_position_notional"] / value + .5 - 1e-12))
    return max(low, min(high, near)) * int(cfg["lot_size"])

def prepare_base_requests(selections: pd.DataFrame, manual_entries: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """复制候选并把人工买单变成独立基础计划；禁止暗含卖单或未来信号。"""
    selected = selections.copy()
    required = {"trade_week", "entry_date", "exit_date", "symbol", "rank"}
    if not required.issubset(selected):
        raise ValueError("selections缺少冻结候选日期/股票/排名")
    if "frozen_base_config_sha256" in selected and not selected.frozen_base_config_sha256.eq(base_config_identity(cfg)).all():
        raise ValueError("冻结基础入场配置身份不一致")
    if "frozen_base_config_sha256" in selected and (manual_entries is not None or cfg.get("manual_buys_csv")):
        raise ValueError("冻结基础计划已包含人工请求，重放不得再次注入manual_entries/manual_buys_csv")
    for column in ("trade_week", "entry_date", "exit_date"):
        selected[column] = pd.to_datetime(selected[column], format="mixed")
    if "_manual" not in selected:
        selected["_manual"] = False
    selected["_entry_time"] = pd.to_datetime(selected.get("_entry_time", selected.entry_date + pd.Timedelta(hours=9, minutes=30)), format="mixed")
    source_signal = pd.to_datetime(selected.get("signal_datetime", selected.trade_week - pd.Timedelta(days=1)), format="mixed")
    source_signal = source_signal.fillna(selected.trade_week - pd.Timedelta(days=1))
    selected["_signal_time"] = pd.to_datetime(selected.get("_signal_time", source_signal), format="mixed")
    selected["_manual_shares"] = selected.get("_manual_shares", 0)
    if cfg.get("manual_buys_csv"):
        if manual_entries is not None:
            raise ValueError("manual_entries与manual_buys_csv不能同时提供")
        manual_entries = pd.read_csv(Path(cfg["manual_buys_csv"]))
    if manual_entries is not None and len(manual_entries):
        manual = manual_entries.copy()
        if not {"datetime", "signal_datetime", "symbol", "shares"}.issubset(manual):
            raise ValueError("manual_entries须包含datetime/signal_datetime/symbol/shares")
        manual["datetime"] = pd.to_datetime(manual.datetime, format="mixed")
        manual["signal_datetime"] = pd.to_datetime(manual.signal_datetime, format="mixed")
        if (manual.datetime.isna().any() or manual.signal_datetime.isna().any()
                or not manual.datetime.gt(manual.signal_datetime).all()
                or not (manual.shares.gt(0) & manual.shares.mod(cfg["lot_size"]).eq(0)).all()
                or manual.duplicated(["datetime", "symbol"]).any()):
            raise ValueError("人工加仓时间/整手/重复键不合法")
        for timestamp in manual.datetime:
            if timestamp.time() != pd.Timestamp("09:30").time() and not is_recovery_minute(timestamp):
                raise ValueError("人工加仓仅支持09:30或真实会话内整分钟")
        manual["trade_week"] = manual.datetime.dt.normalize() - pd.to_timedelta(manual.datetime.dt.weekday, unit="D")
        manual["entry_date"], manual["exit_date"] = manual.datetime.dt.normalize(), manual.datetime.dt.normalize()
        manual["rank"], manual["_manual"] = -1, True
        manual["_entry_time"], manual["_signal_time"], manual["_manual_shares"] = manual.datetime, manual.signal_datetime, manual.shares
        selected = pd.concat([selected, manual], ignore_index=True)
    if (selected._entry_time.isna().any() or selected._signal_time.isna().any()
            or not selected._signal_time.lt(selected._entry_time).all()
            or not selected._entry_time.dt.normalize().eq(selected.entry_date.dt.normalize()).all()
            or not selected._manual.isin([True, False]).all()):
        raise ValueError("基础入场信号必须早于成交，成交日期与entry_date一致，人工标记须为布尔")
    if "frozen_base_shares" in selected:
        required_frozen = {"frozen_base_reference_price", "frozen_base_fee", "frozen_base_notional", "frozen_base_config_sha256"}
        if not required_frozen.issubset(selected):
            raise ValueError("冻结基础计划缺少量价费用/配置身份字段")
        shares = selected.frozen_base_shares
        if not (shares.gt(0) & shares.mod(cfg["lot_size"]).eq(0)).all():
            raise ValueError("冻结基础股数必须为正整手")
    return selected

def prepare_balanced_inputs(selections: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame,
                          decisions: pd.DataFrame, cfg: dict) -> tuple:
    """校验并复制冻结输入；日期不齐保留缺失，拒绝重复键及不合法的 T 延迟时间。"""
    if cfg["initial_cash"] <= 0 or not 0 < cfg["base_fraction"] <= 1:
        raise ValueError("initial_cash、base_fraction 配置非法")
    if cfg["lot_size"] < 100 or cfg["lot_size"] % 100 or int(cfg["lot_size"]) != cfg["lot_size"]:
        raise ValueError("lot_size 必须为 100 的正整数倍")
    if not 0 < cfg["max_bar_amount_fraction"] <= 1:
        raise ValueError("max_bar_amount_fraction 必须处于 (0, 1]")
    pending_fraction = cfg.get("max_pending_t_notional_fraction")
    if pending_fraction is not None and (not np.isfinite(pending_fraction) or pending_fraction < 0):
        raise ValueError("max_pending_t_notional_fraction 必须缺省或有限且非负")
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
    if selected.loc[~selected._manual].duplicated(["trade_week", "symbol"]).any():
        raise ValueError("同一周同一股票不能重复入选")
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
    if not selected.entry_date.isin(dates).all():
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
    signals["priority"] = pd.to_numeric(signals.get("priority", pd.Series(0.0, index=signals.index)), errors="raise").fillna(0.0)
    if not np.isfinite(signals.priority).all():
        raise ValueError("decisions.priority 必须为有限数或缺失；缺失按0处理")
    planned = pd.to_datetime(signals.get("planned_recovery_time", pd.Series(pd.NaT, index=signals.index)))
    if "planned_recovery_time" in signals and (signals.action.ne(0) & planned.isna()).any():
        raise ValueError("显式planned_recovery_time不得缺失；已知无法在同日完成的horizon不能改成默认尾盘")
    signals["explicit_recovery_schedule"] = planned.notna()
    signals["planned_recovery_time"] = planned.fillna(signals.datetime.dt.normalize() + pd.Timedelta(hours=14, minutes=55))
    # 计划是信号时已知的输入契约，不可依赖下一分钟行情是否存在才校验。
    # 否则缺sample的次日计划仍会进入本日timeline，造成执行时序越日。
    for row in signals.loc[signals.action.ne(0)].itertuples(index=False):
        expected_execution = calculate_next_trading_minute(row.datetime)
        if (expected_execution is None or not is_recovery_minute(row.planned_recovery_time)
                or row.planned_recovery_time.normalize() != row.datetime.normalize()
                or row.planned_recovery_time <= expected_execution):
            raise ValueError("planned_recovery_time必须同日、会话内，且严格晚于首腿执行")
    samples = samples.drop(columns=["action", "fraction", "priority", "planned_recovery_time", "explicit_recovery_schedule"], errors="ignore")
    requests = signals[["datetime", "symbol", "action", "fraction", "priority", "planned_recovery_time", "explicit_recovery_schedule"]].merge(
        samples, on=["datetime", "symbol"], how="left", validate="one_to_one")
    requests["date"] = requests.datetime.dt.normalize()
    requests["execution_time"] = pd.to_datetime(requests.execution_time)
    for row in requests.loc[requests.execution_time.notna()].itertuples(index=False):
        if row.execution_time != calculate_next_trading_minute(row.datetime):
            raise ValueError("T 首腿必须按信号后的下一交易分钟执行")
    return selected, prices, requests.sort_values(["execution_time", "priority", "symbol"],
                                                   ascending=[True, False, True], kind="stable"), dates


def run_balanced_portfolio(selections: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame,
                         decisions: pd.DataFrame, cfg: dict,
                         recovery_bars: pd.DataFrame | RecoveryBarStore | None = None,
                           manual_entries: pd.DataFrame | None = None) -> dict:
    """冻结入场候选的均衡金额底仓，不主动平仓；返回旧标准表与金额/手数审计。

    自动每股只首次成功入场，实际滑点后名义额须在配置区间。显式manual_entries
    为用户基础买单，金额区间例外仍受现金、容量和T+1约束。日内T仅消耗日初旧股；
    max_cycles_per_day=0表示不额外限次数，单股同时最多一个待恢复组。
    基准独立决定底仓订单，实际不能以T利润扩大底仓。冻结计划可由
    plan_balanced_entries生成后重放；输入失败、真实恢复失败均不得隐藏。
    """
    cfg = {**BALANCED_DEFAULTS, **cfg}
    validate_balanced_config(cfg)
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
    base_opened = set()
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
            if book is actual:
                if side == "sell":
                    day_sold[symbol] += quantity
                if kind == "t":
                    t_cashflow += flow
                flat_fills += int(quote["high"] == quote["low"])
                limit_unknown_fills += int(not np.isfinite(quote["previous_close"]) and
                                           not np.isfinite(quote["limit_up"] if side == "buy" else quote["limit_down"]))
            book["used"][symbol] = book["used"].get(symbol, 0.0) + quantity * quote["price"]
            # 人工盘中首次买入可能缺当日开/收盘端点；真实成交参考价也是已知估值。
            # 使用未加滑点的市场价，之后更晚的真实收盘仍覆盖，缺收盘时保留时点。
            if symbol not in mark_times or timestamp >= mark_times[symbol]:
                marks[symbol], mark_times[symbol] = float(quote["price"]), timestamp
            pid_value = "|".join(str(pid) for pid, _ in allocations)
            event = {"datetime": timestamp, "signal_datetime": signal, "signal_time": signal,
                     "symbol": symbol, "plan_id": pid_value, "lot_id": pair_id if kind == "t" else f"weekly_{pid_value}",
                     "pair_id": pair_id, "side": side, "shares": quantity, "planned_shares": quantity, "price": price,
                     "reference_price": quote["price"], "notional": value, "fee": fee,
                     "slippage_cost": quantity * abs(price - quote["price"]), "reason": reason,
                     "event_type": kind, "base_order_type": "", "cash_after": book["cash"], "sellable_before": before,
                     "sellable_after": sum(book["sellable"][pid] for pid in plans_by_symbol[symbol]), "account": book["name"],
                     "fragment_id": 0, "signal_priority": 0.0, "planned_recovery_time": pd.NaT}
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
            row = recovery_lookup.get_quote(timestamp, symbol) if recovery_lookup is not None else None
            if row is None:
                return create_quote({}, "close", symbol)
            current = dict(day_rows.get(symbol, {}))
            for key in ["high", "low", "amount", "volume"]:
                current[f"close_{key}"] = row.get(key, np.nan)
            current["close"] = row.get("close", np.nan)
            for key in ["previous_close", "prev_close", "limit_up", "limit_down"]:
                if key in row:
                    current[key] = row[key]
            return create_quote(current, "close", symbol)

        def recover_pair(symbol: str, timestamp: pd.Timestamp, prefix: str,
                         quote_override: dict | None = None) -> None:
            """按当前价量、现金及旧库存恢复一段，剩余债务和回购准备金继续保留。

            每段只承担首腿原始总费用/滑点的股数比例，反向腿费用按本段真实成交
            独立计算。每分钟容量与同刻人工加仓及新T首腿共享，不虚构全量恢复。
            """
            pair = active[symbol]
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
                               quantity, filled, timestamp, timestamp - pd.Timedelta(minutes=1), "t_recovery", reason or "capacity")
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
                               timestamp - pd.Timedelta(minutes=1), "t", f"{prefix}分段恢复", pair["pair_id"])
            second["planned_shares"] = quantity
            second["fragment_id"] = len(trips) + 1
            second["signal_priority"] = pair["first"]["signal_priority"]
            second["planned_recovery_time"] = pair["planned_recovery_time"]
            first = pair["first"]
            fraction = filled / pair["original_shares"]
            allocated_first = {**first, "shares": filled, "notional": first["price"] * filled,
                               "fee": first["fee"] * fraction, "slippage_cost": first["slippage_cost"] * fraction}
            buy, sell = (allocated_first, second) if first["side"] == "buy" else (second, allocated_first)
            gross = filled * (sell["price"] - buy["price"])
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
                          "gross_pnl": gross, "net_pnl": gross - buy["fee"] - sell["fee"],
                          "slippage_cost": buy["slippage_cost"] + sell["slippage_cost"],
                          "holding_days": (timestamp - first["datetime"]).total_seconds() / 86400,
                          "recovered_same_day": timestamp.normalize() == first["datetime"].normalize(), "reason": f"{prefix}恢复",
                          "fragment_id": second["fragment_id"], "original_pair_shares": pair["original_shares"],
                          "remaining_shares_after": pair["shares"], "pair_fully_recovered": not pair["shares"],
                          "planned_recovery_time": pair["planned_recovery_time"]})
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
                                  "retry_clock": create_planned_recovery_clock(row["planned_recovery_time"])
                                      if row["explicit_recovery_schedule"] else ()}
                peak_pending_t_notional = max(peak_pending_t_notional, calculate_pending_t_notional())
                next_pair += 1
                cycles[symbol] += 1

        close_time = date + pd.Timedelta(hours=15)
        tail_minutes = range(895, 901) if recovery_lookup is not None else (895, 900)
        global_recovery_clock = {date + pd.Timedelta(minutes=minute)
                                 for minute in (*fixed_recovery_minutes, *tail_minutes)}
        timeline = set(execution_groups) | global_recovery_clock
        timeline.update(time for time in entry_groups if time.normalize() == date and time > opening_time)
        # 这里只读取决策中已声明的计划，不查报价是否存在。每组自动重试只属于自身，
        # 其他组不会因未来另一份决策增加时钟而获得未经配置的额外恢复机会。
        for row in day_requests:
            if row["action"] and row["explicit_recovery_schedule"]:
                timeline.update(create_planned_recovery_clock(row["planned_recovery_time"]))
        for timestamp in sorted(timeline):
            actual["used"], baseline["used"] = {}, {}
            for symbol in list(active):
                pair = active[symbol]
                if timestamp < pair["planned_recovery_time"]:
                    continue
                if timestamp not in global_recovery_clock and timestamp not in pair["retry_clock"]:
                    continue
                if recovery_lookup is not None:
                    recover_pair(symbol, timestamp, "minute_recovery", create_current_recovery_quote(symbol, timestamp))
                elif timestamp == close_time:
                    recover_pair(symbol, timestamp, "close")
                elif pair["first"]["datetime"].normalize() == date:
                    provided_time = pd.to_datetime(pair["recovery_row"].get("recovery_time", pd.NaT))
                    matched = provided_time == timestamp or (pd.isna(provided_time) and timestamp == date + pd.Timedelta(minutes=895))
                    quote = create_quote(pair["recovery_row"] if matched else {}, "recovery", symbol)
                    recover_pair(symbol, timestamp, "recovery", quote)
            execute_base_entries(timestamp, entry_groups.get(timestamp, []))
            execute_first_requests(timestamp, execution_groups.get(timestamp, []))

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
                             "pending_direction": pair.get("direction", ""), "pending_shares": pending})
    metrics = calculate_daily_metrics(curve, event_frame, trip_frame, cfg["initial_cash"])
    completed_trips = trip_frame.loc[trip_frame.pair_id.isin(completed_pair_ids)]
    completed_profits = completed_trips.groupby("pair_id").net_pnl.sum() if len(completed_trips) else pd.Series(dtype=float)
    metrics.update({"strategy": "balanced_fixed_inventory_t", "total_t_net_profit": float(trip_frame.net_pnl.sum()) if len(trip_frame) else 0.0,
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
                    "recovery_execution_mode": "minute_partial" if recovery_lookup is not None else "endpoint_partial",
                    "intraday_allocation_order": "priority_desc_symbol_asc",
                    "same_minute_event_order": "due_recovery_then_base_buy_then_t_first_leg",
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
    return {"equity_curve": curve, "trade_events": event_frame, "round_trips": trip_frame,
            "open_positions": pd.DataFrame(holdings, columns=POSITION_COLUMNS), "daily_inventory_audit": audit_frame,
            "order_attempts": pd.DataFrame(attempts, columns=ATTEMPT_COLUMNS), "metrics": metrics,
            "baseline_events": base_frame, "base_plan_mismatches": pd.DataFrame(mismatches),
            "weekly_entry_plans": pd.DataFrame(entry_plans),
            "position_amount_audit": pd.DataFrame(amount_audits, columns=AMOUNT_COLUMNS),
            "market_dates": dates.copy()}


def plan_balanced_entries(selections: pd.DataFrame, daily: pd.DataFrame, cfg: dict,
                          manual_entries: pd.DataFrame | None = None,
                          recovery_bars: pd.DataFrame | RecoveryBarStore | None = None) -> dict:
    """因果运行一次无T账户，冻结成功底仓以供所有候选复用，不筛未来T收益。

    失败的入场请求保存在baseline_result中。返回的frozen_selections只含成功计划，
    可据此加载这些股票整个研究日历的行情；所有原失败请求仍在规划报告中保留。
    已冻结人工请求随计划一起重放，不应再次传入manual_entries/manual_buys_csv。
    """
    full_cfg = {**BALANCED_DEFAULTS, **cfg}
    result = run_balanced_portfolio(selections, daily, pd.DataFrame(), pd.DataFrame(),
                                   full_cfg, recovery_bars=recovery_bars, manual_entries=manual_entries)
    requests = prepare_base_requests(selections, manual_entries, full_cfg)
    prepared, _, _, _ = prepare_balanced_inputs(requests, daily, pd.DataFrame(), pd.DataFrame(), full_cfg)
    events = result["baseline_events"]
    columns = ["plan_id", "shares", "reference_price", "fee", "notional"]
    fills = events[columns].rename(columns={"shares": "frozen_base_shares", "reference_price": "frozen_base_reference_price",
                                           "fee": "frozen_base_fee", "notional": "frozen_base_notional"})
    fills["plan_id"] = pd.to_numeric(fills.plan_id).astype(int)
    frozen = prepared.merge(fills, on="plan_id", how="inner", validate="one_to_one")
    frozen["frozen_base_config_sha256"] = base_config_identity(full_cfg)
    return {"frozen_selections": frozen, "baseline_result": result,
            "base_entries": events.copy(), "planning_attempts": result["order_attempts"].copy(),
            "base_config_sha256": base_config_identity(full_cfg),
            "replay_cfg": {**cfg, "manual_buys_csv": None}, "market_dates": result["market_dates"]}
