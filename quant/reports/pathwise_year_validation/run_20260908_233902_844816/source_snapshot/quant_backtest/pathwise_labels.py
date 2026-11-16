"""与独立批次引擎对齐的交易结果标签，仅供训练，不决定推断时股票是否入选。

输入有界缓存和只读执行行情；在独立初始账户中仅尝试一次10%总资产买入，
遵守下一分钟、T+1、整手、ATR、期限、费用和部分成交。Numba仅加速批量标签，
原annual_portfolio引擎作为逐例参照。没有成交的意图收益为0，窗口末未退出者按市值记账。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import copy

import numba
import numpy as np
import pandas as pd

from quant_backtest.simple_signals import FEATURE_NAMES, build_simple_panel
from quant_backtest.annual_portfolio import run_lot_portfolio
from quant_backtest.exit_alignment import HorizonExitPolicy
from quant_backtest.annual_panel import write_json

PATH_FEATURES = {**FEATURE_NAMES,
    "signal_atr_ratio": "当前简单分钟ATR除以当前价，表示局部风险幅度",
    "signal_amount_log": "当前分钟成交额加1后取自然对数，表示可见流动性规模",
    "signal_volume_ratio": "当前分钟量相对过去11采样点均量的倍数"}
QUOTE_COLUMNS = ["close", "atr", "interval_high", "execution_close",
                 "execution_volume", "execution_amount", "previous_close"]


@numba.njit(cache=True)
def calculate_path_fill(quote, requested, budget, buy, slippage, capacity_fraction, lot_size):
    """quote为单时钟8列量价；requested股/budget元是上限，buy给方向，其余参数给滑点、容量和整手。"""
    price, volume, amount, previous = quote[3], quote[4], quote[5], quote[6]
    if not (np.isfinite(price) and price > 0 and np.isfinite(volume) and volume > 0
            and np.isfinite(amount) and amount > 0 and np.isfinite(previous) and previous > 0):
        return 0, 0.0
    ratio = price / previous
    if (buy and ratio >= 1.095) or (not buy and ratio <= .905):
        return 0, 0.0
    price *= 1 + slippage if buy else 1 - slippage
    size = min(requested, int(min(budget, amount * capacity_fraction) / price / lot_size) * lot_size)
    return size, price


@numba.njit(cache=True)
def simulate_path_trade(quotes, start, end, day_index, offsets, executable, stamp,
                        horizon, initial_atr, trailing_atr, parameters):
    """quotes为一股[时钟,8量价列]；start/end闭区间、日序号/钟偏移/可执行标记/stamp给日历税费。

    horizon为交易日期限，initial_atr/trailing_atr为分钟ATR倍数。
    parameters依次为初始现金、入场比例、佣金、过户费、滑点、容量比例、整手股数。
    返回账户净收益、实际最后读取索引、初次股数、剩余股数、卖出事件数和入场名义金额。
    """
    initial, fraction, commission, transfer, slippage, capacity, lot_size = parameters
    first = quotes[start]
    if (not executable[start] or not np.isfinite(first[0]) or first[0] <= 0
            or not np.isfinite(first[1]) or first[1] <= 0):
        return 0.0, start, 0, 0, 0, 0.0
    budget = initial * fraction
    requested = int(budget / first[0] / lot_size) * int(lot_size)
    shares, buy_price = calculate_path_fill(first, requested,
        min(budget, initial / (1 + commission + transfer)), True, slippage, capacity, lot_size)
    if shares == 0:
        return 0.0, start, 0, 0, 0, 0.0
    notional = shares * buy_price
    cash = initial - notional - notional * commission - notional * transfer
    entered = shares
    high = buy_price
    stop = buy_price - initial_atr * first[1]
    pending, sell_count = False, 0
    for index in range(start + 1, end + 1):
        quote = quotes[index]
        observed_high = quote[0] if index == start + 1 else quote[2]
        if np.isfinite(observed_high):
            high = max(high, observed_high)
        if np.isfinite(quote[1]):
            stop = max(stop, high - trailing_atr * quote[1])
        age = day_index[index] - day_index[start]
        due = executable[index] and (age > horizon or (age == horizon and offsets[index] >= offsets[start]))
        if due or (np.isfinite(quote[0]) and quote[0] <= stop):
            pending = True
        if pending and day_index[index] > day_index[start] and executable[index]:
            sold, price = calculate_path_fill(quote, shares, np.inf, False, slippage, capacity, lot_size)
            if sold:
                value = sold * price
                cash += value - value * commission - value * transfer - value * stamp[index]
                shares -= sold
                sell_count += 1
                if shares == 0:
                    return (cash - initial) / initial, index, entered, 0, sell_count, notional
    mark = quotes[end, 3] if executable[end] and np.isfinite(quotes[end, 3]) else quotes[end, 7]
    return (cash + shares * mark - initial) / initial, end, entered, shares, sell_count, notional


@numba.njit(cache=True, parallel=True)
def calculate_path_targets(quotes, entries, assets, endpoints, day_index, offsets, executable,
                           stamp, horizon, initial_atr, trailing_atr, parameters):
    """对entries/assets和对应endpoints独立模拟；其余参数与simulate_path_trade一致，不共享虚拟现金或容量。"""
    output = np.empty((len(entries), 6), dtype=np.float64)
    for n in numba.prange(len(entries)):
        result = simulate_path_trade(quotes[assets[n]], entries[n], endpoints[n], day_index, offsets,
            executable, stamp, horizon, initial_atr, trailing_atr, parameters)
        output[n, 0] = result[0]
        output[n, 1] = result[1]
        output[n, 2] = result[2]
        output[n, 3] = result[3]
        output[n, 4] = result[4]
        output[n, 5] = result[5]
    return output


def build_path_panel(panel: pd.DataFrame, market, matrices: dict) -> pd.DataFrame:
    """用panel/market/matrices现有有限特征生成29列视图，不重新构造原因子，不放入未来执行价。"""
    out = build_simple_panel(panel, market, matrices)
    idx = market.times.get_indexer(out.datetime)
    assets = np.array([market.asset_index[s] for s in out.symbol])
    for name in ["atr_ratio", "amount_log", "volume_ratio"]:
        out[f"signal_{name}"] = matrices[name][idx, assets]
    return out


def select_daily_entries(frame: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """从frame按股票/自然日稳定哈希预定一个times信号钟，缺该钟不补选，标签与行序不影响抽样。"""
    if frame.duplicated(["datetime", "symbol"]).any():
        raise ValueError("路径标签入口键重复")
    active = times[times.hour != 15]
    offsets = np.array(sorted(set(active - active.normalize())), dtype="timedelta64[ns]")
    keys = pd.DataFrame({"day": frame.datetime.dt.normalize(), "symbol": frame.symbol.astype(str)})
    slots = pd.util.hash_pandas_object(keys, index=False).to_numpy() % len(offsets)
    mask = (frame.datetime - frame.datetime.dt.normalize()).to_numpy() == offsets[slots]
    return frame.loc[mask].sort_values(["symbol", "datetime"]).copy()


def create_path_arrays(market, cfg: dict) -> tuple:
    """从market生成按股票连续的执行量价副本及cfg成本向量；所有未来字段只允许标签模拟器读取。"""
    quotes = np.stack([market.values[c] for c in QUOTE_COLUMNS] + [market.marks], axis=2).transpose(1, 0, 2).copy()
    offsets = (market.times - market.days).asi8
    executable = np.asarray(market.times.hour != 15)
    stamp = np.where(market.times >= pd.Timestamp("2023-08-28"),
                     cfg["stamp_rate_from_20230828"], cfg["stamp_rate_before_20230828"])
    parameters = np.array([cfg[k] for k in ["initial_cash", "entry_equity_fraction", "commission_rate",
                         "transfer_rate", "slippage_rate", "max_bar_amount_fraction", "lot_size"]])
    return quotes, offsets, executable, stamp, parameters


def load_path_labels(frame: pd.DataFrame, market, bounded_cache: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """按frame入口/market行情及bounded_cache身份构造或复用cfg路径标签pkl；原因子和行情不修改。"""
    keys = ["initial_cash", "entry_equity_fraction", "commission_rate", "transfer_rate", "slippage_rate",
            "max_bar_amount_fraction", "lot_size", "stamp_rate_from_20230828", "stamp_rate_before_20230828",
            "path_horizons", "path_exit_grace_days", "path_initial_atr", "path_trailing_atr", "min_cross_section"]
    if cfg["min_commission"] != 0:
        raise ValueError("路径标签当前只支持免5")
    root = Path(__file__).resolve().parents[1]
    specs = {"bounded_key": bounded_cache["key"], "cfg": {k: cfg[k] for k in keys},
        "code": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
            ["quant_backtest/pathwise_labels.py", "quant_backtest/annual_portfolio.py", "quant_backtest/exit_alignment.py"]}}
    key = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cfg["path_cache_dir"]) / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trade_path_labels.pkl"
    if path.exists():
        labels = pd.read_pickle(path)
        print(f"复用执行路径标签：{path}", flush=True)
    else:
        eligible = frame.loc[frame.cross_section_count >= cfg["min_cross_section"]]
        entries = select_daily_entries(eligible[["datetime", "symbol"]], market.times)
        labels = entries[["datetime", "symbol"]].reset_index(drop=True)
        rows = market.times.get_indexer(labels.datetime)
        assets = np.array([market.asset_index[s] for s in labels.symbol])
        days = pd.DatetimeIndex(sorted(market.days.unique()))
        quotes, offsets, executable, stamp, parameters = create_path_arrays(market, cfg)
        numba.set_num_threads(cfg["workers"])
        for horizon in cfg["path_horizons"]:
            terminal_days = market.day_index[rows] + horizon + cfg["path_exit_grace_days"]
            covered = terminal_days < len(days)
            endpoint = market.times.get_indexer(days[terminal_days[covered]] + pd.to_timedelta(offsets[rows[covered]], unit="ns"))
            if (endpoint < 0).any():
                raise ValueError("路径标签终点没有共同钟点")
            output = calculate_path_targets(quotes, rows[covered], assets[covered], endpoint, market.day_index,
                offsets, executable, stamp, horizon, cfg["path_initial_atr"], cfg["path_trailing_atr"], parameters)
            prefix = f"path_h{horizon}_"
            labels[prefix + "net_return"] = np.nan
            labels[prefix + "window_end"] = pd.NaT
            labels[prefix + "observed_end"] = pd.NaT
            labels.loc[covered, prefix + "net_return"] = output[:, 0]
            labels.loc[covered, prefix + "window_end"] = market.times[endpoint] + pd.Timedelta(minutes=1)
            observed = output[:, 1].astype(int)
            labels.loc[covered, prefix + "observed_end"] = market.times[observed] + pd.Timedelta(minutes=1)
            for j, suffix in enumerate(["entered_shares", "remaining_shares", "sell_events", "entry_notional"], start=2):
                labels[prefix + suffix] = np.nan
                labels.loc[covered, prefix + suffix] = output[:, j]
            print(f"{horizon}日ATR/到期路径：{covered.sum():,}条；有买入{int((output[:, 2] > 0).sum()):,}条", flush=True)
        temporary = path.with_suffix(".tmp.pkl")
        labels.to_pickle(temporary)
        temporary.replace(path)
    metadata = {"key": key, "path": str(path), "rows": len(labels), "specification": specs,
        "target": "独立初始账户一次10%预算入场，实际ATR/期限退出与部分成交后的净权益变动比例；未成交0，残余按窗口末市值",
        "boundary": "即使提前平仓，训练也按H+宽限交易日的固定潜在标签终点隔离，不依好坏结果选择跨段样本"}
    write_json(directory / "metadata.json", metadata)
    return labels, metadata


def validate_path_case(market, cfg: dict, arrays: tuple, asset: int, start: int, end: int, horizon: int) -> dict:
    """对market某asset/start/end/horizon单次意图，比较arrays加速结果与原引擎；cfg只把截面门槛改为1供独立标签参照。"""
    quotes, offsets, executable, stamp, parameters = arrays
    fast = simulate_path_trade(quotes[asset], start, end, market.day_index, offsets, executable, stamp,
        horizon, cfg["path_initial_atr"], cfg["path_trailing_atr"], parameters)
    narrow = copy.copy(market)
    symbol = market.symbols[asset]
    narrow.symbols, narrow.asset_index = [symbol], {symbol: 0}
    narrow.values = {k: v[:, asset:asset + 1] for k, v in market.values.items()}
    narrow.marks = market.marks[:, asset:asset + 1]
    scores = pd.DataFrame({"datetime": [market.times[start]], "symbol": [symbol], "prediction": [1.]})
    rule = {"threshold": 0., "initial": cfg["path_initial_atr"], "trailing": cfg["path_trailing_atr"],
            "max_days": 0, "rule_id": "path_label_reference", "selection": "仅供路径标签等价审计"}
    result = run_lot_portfolio(scores, narrow, market.times[start], market.times[end] + pd.Timedelta(minutes=2),
        "prediction", rule, {**cfg, "min_cross_section": 1}, HorizonExitPolicy(narrow, horizon, "matched_horizon"))
    expected = (result["curve"].equity.iloc[-1] - cfg["initial_cash"]) / cfg["initial_cash"]
    buys = result["events"].loc[result["events"].side == "buy"]
    entry_shares = int(buys.shares.sum())
    remaining = int(result["open"].shares.sum())
    sales = int((result["events"].side == "sell").sum())
    if not np.isclose(fast[0], expected, atol=1e-10, rtol=1e-8):
        raise AssertionError(f"路径与原引擎权益不一致：{fast[0]} != {expected}")
    if [fast[2], fast[3], fast[4]] != [entry_shares, remaining, sales]:
        raise AssertionError("路径与原引擎股数或部分卖出事件不一致")
    return {"symbol": symbol, "start": str(market.times[start]), "end": str(market.times[end]),
            "horizon": horizon, "net_return": float(fast[0]), "entered_shares": entry_shares,
            "remaining_shares": remaining, "sell_events": sales, "error": float(fast[0] - expected)}
