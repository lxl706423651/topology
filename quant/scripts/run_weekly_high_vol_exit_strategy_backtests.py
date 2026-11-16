"""回测周度高波动 top300 股票池下的持仓级平仓策略。

输入为已有动态高波动周度特征缓存、method1 XGBoost 模型、method2 CTA
打分函数和本地 A 股分钟行情。脚本固定使用“每周用上一周高波动 top300”
股票池，不重新训练模型；先在验证集选择 topN 和持仓级退出规则，再把选中
规则搬到测试集。输出中文报告、收益曲线、逐笔交易和平仓复盘。原始行情数
据和既有模型不会被修改。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.factor_library import calculate_atr
from quant_backtest.metrics import performance_metrics
import scripts.run_all_a_method_backtests as all_a
import scripts.run_multi_factor_cta_backtests as cta
import scripts.run_xgboost_long_only_backtest as xb


OUTPUT_DIR = PROJECT_ROOT / "reports" / "weekly_high_vol_exit_strategy_backtests"
WEEKLY_UNIVERSE_ID = "weekly_previous_week_high_vol_top300"
SOURCE_NOTES = [
    {
        "name": "Kaminski and Lo stop-loss framework",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338",
        "note": "止损规则在动量环境中可能改善风险收益，但随机游走环境下可能降低期望收益。",
    },
    {
        "name": "TA-Lib ATR documentation",
        "url": "https://ta-lib.github.io/ta-lib-python/func_groups/volatility_indicators.html",
        "note": "ATR 是波动率指标，可用 high、low、close 计算，用于波动率自适应止损。",
    },
    {
        "name": "Chandelier Exit description",
        "url": "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit",
        "note": "Chandelier Exit 使用最高价减去若干倍 ATR 作为多头移动止损。",
    },
    {
        "name": "Fidelity exit strategies",
        "url": "https://www.fidelity.com/learning-center/trading-investing/trading/exit-strategies",
        "note": "时间退出用于定义最大持仓暴露时间，并提示到期后至少重新评估仓位。",
    },
]


@dataclass(frozen=True)
class ExitRule:
    """持仓级平仓规则。

    rule_id: 稳定规则名。
    stop_loss_pct: 从买入价回撤达到该比例时止损。
    take_profit_pct: 从买入价上涨达到该比例时止盈。
    trailing_stop_pct: 从持仓后最高价回撤达到该比例时移动止损。
    atr_period: ATR 计算窗口，0 表示不启用 ATR。
    atr_multiple: Chandelier Exit 使用的 ATR 倍数。
    max_holding_periods: 最大持仓采样周期数，每个周期为 24 根分钟 bar。
    score_exit_offset: 当前分数低于入场阈值减去该偏移时退出，None 表示不用。
    """

    rule_id: str
    stop_loss_pct: float | None
    take_profit_pct: float | None
    trailing_stop_pct: float | None
    atr_period: int
    atr_multiple: float | None
    max_holding_periods: int | None
    score_exit_offset: float | None


@dataclass(frozen=True)
class MethodSpec:
    """method1/method2 的固定入场分数字段和阈值。"""

    method_id: str
    strategy_name: str
    score_col: str
    entry_threshold: float
    description: str


@dataclass
class PositionState:
    """记录单只持仓的入场和移动止损状态。"""

    symbol: str
    entry_price: float
    entry_period_index: int
    high_watermark: float
    entry_score: float


class MinutePriceStore:
    """按需读取本地分钟行情，为已持仓股票提供动态池外的卖出价格。"""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.frames: dict[str, pd.DataFrame] = {}

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """读取并缓存单只股票的全部本地分钟 OHLCV。"""

        if symbol in self.frames:
            return self.frames[symbol]
        parts: list[pd.DataFrame] = []
        for year_dir in sorted(path for path in self.data_root.iterdir() if path.is_dir() and path.name.isdigit()):
            path = year_dir / f"{symbol}.csv"
            if not path.exists():
                continue
            part = pd.read_csv(path, usecols=lambda col: col in {"datetime", "open", "high", "low", "close", "volume", "amount"})
            parts.append(part)
        if not parts:
            frame = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
        else:
            frame = pd.concat(parts, ignore_index=True)
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last").set_index("datetime")
        self.frames[symbol] = frame
        return frame

    def get_close(self, symbol: str, dt: pd.Timestamp) -> float | None:
        """读取指定时间的收盘价；没有精确分钟 bar 时返回 None。"""

        frame = self.load_symbol(symbol)
        if frame.empty or dt not in frame.index:
            return None
        price = float(frame.at[dt, "close"])
        if not np.isfinite(price) or price <= 0:
            return None
        return price

    def get_atr(self, symbol: str, dt: pd.Timestamp, period: int) -> float | None:
        """用当前时间及以前分钟 bar 计算 ATR。"""

        if period <= 0:
            return None
        frame = self.load_symbol(symbol)
        if frame.empty:
            return None
        history = frame.loc[:dt].tail(max(period * 3, period + 2))
        if len(history) < period + 1:
            return None
        atr = calculate_atr(history["high"], history["low"], history["close"], period).iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            return None
        return float(atr)


def parse_args() -> argparse.Namespace:
    """解析平仓策略研究参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=xb.DATA_ROOT)
    parser.add_argument("--weekly-panel-cache", type=Path, default=None)
    parser.add_argument("--method1-run-dir", type=Path, default=all_a.METHOD1_RUN_DIR)
    parser.add_argument("--method2-run-dir", type=Path, default=all_a.METHOD2_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--top-n-candidates", default="2,3,5")
    parser.add_argument("--max-position-weight", type=float, default=0.10)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--max-validation-periods", type=int, default=0)
    parser.add_argument("--max-test-periods", type=int, default=0)
    parser.add_argument("--broker-commission-rate", type=float, default=0.0008)
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--min-commission", type=float, default=0.0)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    """解析逗号分隔整数列表。"""

    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item < 1 for item in out):
        raise ValueError("topN 候选必须为正整数。")
    return out


def find_weekly_panel_cache(cache_path: Path | None) -> Path:
    """定位周度高波动 top300 特征缓存。"""

    if cache_path is not None:
        if not cache_path.exists():
            raise FileNotFoundError(f"指定 weekly panel cache 不存在：{cache_path}")
        return cache_path
    metadata_paths = sorted((PROJECT_ROOT / "reports" / "dynamic_high_vol_method_backtests" / "panel_cache").glob("*.metadata.json"))
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("universe_id") == WEEKLY_UNIVERSE_ID:
            found = Path(str(metadata["cache_path"]))
            if found.exists():
                return found
    raise FileNotFoundError("未找到 weekly_previous_week_high_vol_top300 特征缓存，请先运行动态高波动股票池脚本。")


def create_fee_config(args: argparse.Namespace) -> all_a.FeeConfig:
    """生成 A 股费用配置。"""

    return all_a.FeeConfig(
        broker_commission_rate=float(args.broker_commission_rate),
        stamp_duty_rate=float(args.stamp_duty_rate),
        transfer_fee_rate=float(args.transfer_fee_rate),
        min_commission=float(args.min_commission),
        commission_includes_exchange_fees=True,
    )


def create_backtest_config(args: argparse.Namespace) -> all_a.AllAMethodConfig:
    """生成复用交易成本和 T+1 函数所需配置。"""

    return all_a.AllAMethodConfig(
        data_root=args.data_root,
        feature_manifest=xb.MODEL_FEATURE_MANIFEST,
        method1_run_dir=args.method1_run_dir,
        method2_run_dir=args.method2_run_dir,
        output_dir=args.output_dir,
        panel_cache_dir=args.output_dir / "panel_cache",
        validation_start=pd.Timestamp(args.validation_start),
        split_date=pd.Timestamp(args.split_date),
        end_date=pd.Timestamp(args.end_date),
        forward_horizon_bars=int(args.horizon),
        evaluation_step_bars=int(args.eval_step),
        initial_cash=float(args.initial_cash),
        min_assets_per_rebalance=int(args.min_assets),
        lot_size=int(args.lot_size),
        max_symbols=0,
        workers=1,
        rebuild_panel=False,
        no_panel_cache=False,
        use_akshare_st_filter=True,
        fee=create_fee_config(args),
    )


def create_exit_rules() -> list[ExitRule]:
    """固定候选平仓规则，所有候选只在验证集选择。"""

    return [
        ExitRule("time_1d_stop_6_take_10", 0.06, 0.10, None, 0, None, 10, None),
        ExitRule("time_2d_stop_6_take_12", 0.06, 0.12, None, 0, None, 20, None),
        ExitRule("trail_5_stop_5_take_12_2d", 0.05, 0.12, 0.05, 0, None, 20, None),
        ExitRule("trail_8_stop_6_take_15_3d", 0.06, 0.15, 0.08, 0, None, 30, None),
        ExitRule("atr3_stop_6_take_15_3d", 0.06, 0.15, None, 120, 3.0, 30, None),
        ExitRule("score_decay_stop_6_take_12_2d", 0.06, 0.12, None, 0, None, 20, 0.03),
    ]


def create_method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    """读取 method1/method2 固定阈值并生成方法定义。"""

    method1_rule = all_a.load_method_rule(args.method1_run_dir)
    method2_rule = all_a.load_cta_rule(args.method2_run_dir, "cta_trend_breakout")
    if method1_rule.confidence_threshold is None or method2_rule.confidence_threshold is None:
        raise ValueError("method1/method2 必须包含固定 confidence_threshold。")
    return [
        MethodSpec(
            "method1_xgboost_exit_managed",
            "method1 XGBoost 周度高波动持仓级平仓",
            "prediction",
            float(method1_rule.confidence_threshold),
            "买入按 XGBoost 预测分 topN，卖出由持仓级退出规则决定。",
        ),
        MethodSpec(
            "method2_cta_exit_managed",
            "method2 CTA 周度高波动持仓级平仓",
            "cta_trend_breakout_score",
            float(method2_rule.confidence_threshold),
            "买入按 CTA 包装分 topN，卖出由持仓级退出规则决定。",
        ),
    ]


def load_scored_weekly_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    """读取周度动态股票池缓存并生成 method1/method2 分数。"""

    cache_path = find_weekly_panel_cache(args.weekly_panel_cache)
    print(f"正在读取周度高波动特征缓存：{cache_path}", flush=True)
    panel = pd.read_pickle(cache_path)
    model = joblib.load(args.method1_run_dir / "xgboost_model.joblib")
    feature_cols = (args.method1_run_dir / "feature_list.txt").read_text(encoding="utf-8").splitlines()
    missing = [col for col in feature_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"周度高波动面板缺少模型特征：{missing[:10]}")
    panel = panel.copy()
    panel["prediction"] = model.predict(panel.loc[:, feature_cols]).astype("float32")
    panel = cta.add_cta_factor_scores(panel, cta.SCORE_BLEND_STRENGTH)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["entry_datetime"] = pd.to_datetime(panel["entry_datetime"])
    panel = panel.loc[panel["datetime"] <= pd.Timestamp(args.end_date) + pd.Timedelta(days=1)]
    return panel.sort_values(["datetime", "symbol"]).reset_index(drop=True), cache_path


def select_entry_candidates(
    tradable: pd.DataFrame,
    method: MethodSpec,
    top_n: int,
    min_assets: int,
) -> pd.DataFrame:
    """按分数阈值和 topN 选择新开仓候选。"""

    if len(tradable) < min_assets:
        return pd.DataFrame()
    work = tradable.dropna(subset=[method.score_col, "entry_price"]).copy()
    work = work.loc[work[method.score_col] >= method.entry_threshold]
    if work.empty:
        return work
    return work.sort_values(method.score_col, ascending=False).head(top_n)


def calculate_buy_shares(price: float, equity: float, max_position_weight: float, lot_size: int) -> int:
    """按单票最大资产占比和整数手计算买入股数。"""

    if not np.isfinite(price) or price <= 0 or equity <= 0:
        return 0
    allocation = equity * max_position_weight
    return int(math.floor(allocation / (price * lot_size)) * lot_size)


def reduce_buy_plan_to_cash(
    buy_plan: dict[str, int],
    entry_prices: dict[str, float],
    cash: float,
    config: all_a.AllAMethodConfig,
) -> dict[str, int]:
    """在现金约束下缩减买入计划。"""

    plan = {symbol: shares for symbol, shares in buy_plan.items() if shares > 0}
    while plan:
        required = sum(
            plan[symbol] * entry_prices[symbol] + all_a.calculate_trade_cost("买入", plan[symbol] * entry_prices[symbol], config.fee)
            for symbol in plan
        )
        if required <= cash + 1e-6:
            return plan
        largest = max(plan, key=lambda symbol: plan[symbol] * entry_prices[symbol])
        reduced = plan[largest] - config.lot_size
        if reduced > 0:
            plan[largest] = reduced
        else:
            del plan[largest]
    return {}


def summarize_position_lots(position_lots: dict[str, list[xb.PositionLot]]) -> dict[str, int]:
    """返回当前每只股票持仓股数。"""

    return {symbol: sum(lot.shares for lot in lots if lot.shares > 0) for symbol, lots in position_lots.items() if sum(lot.shares for lot in lots if lot.shares > 0) > 0}


def value_positions(position_lots: dict[str, list[xb.PositionLot]], prices: dict[str, float]) -> float:
    """按给定价格估算持仓市值。"""

    total = 0.0
    for symbol, shares in summarize_position_lots(position_lots).items():
        price = prices.get(symbol)
        if price is not None and np.isfinite(price) and price > 0:
            total += shares * float(price)
    return float(total)


def get_period_prices(
    tradable: pd.DataFrame,
    execution_time: pd.Timestamp,
    held_symbols: Sequence[str],
    price_store: MinutePriceStore,
    last_prices: dict[str, float],
) -> dict[str, float]:
    """取得当前周期用于估值和交易的价格。"""

    prices = dict(zip(tradable["symbol"].astype(str), tradable["entry_price"].astype(float), strict=False))
    for symbol in held_symbols:
        if symbol in prices and np.isfinite(prices[symbol]) and prices[symbol] > 0:
            continue
        price = price_store.get_close(symbol, execution_time)
        if price is not None:
            prices[symbol] = price
        elif symbol in last_prices:
            prices[symbol] = last_prices[symbol]
    return prices


def choose_exit_reason(
    state: PositionState,
    current_price: float,
    current_score: float | None,
    method: MethodSpec,
    exit_rule: ExitRule,
    period_index: int,
    execution_time: pd.Timestamp,
    price_store: MinutePriceStore,
) -> str | None:
    """判断单只持仓是否触发持仓级平仓规则。"""

    if not np.isfinite(current_price) or current_price <= 0:
        return None
    state.high_watermark = max(state.high_watermark, float(current_price))
    position_return = current_price / state.entry_price - 1.0
    drawdown_from_high = current_price / state.high_watermark - 1.0 if state.high_watermark > 0 else 0.0
    if exit_rule.stop_loss_pct is not None and position_return <= -exit_rule.stop_loss_pct:
        return f"持仓级止损：收益率 {position_return:.2%} <= -{exit_rule.stop_loss_pct:.2%}"
    if exit_rule.take_profit_pct is not None and position_return >= exit_rule.take_profit_pct:
        return f"持仓级止盈：收益率 {position_return:.2%} >= {exit_rule.take_profit_pct:.2%}"
    if exit_rule.trailing_stop_pct is not None and drawdown_from_high <= -exit_rule.trailing_stop_pct:
        return f"持仓级移动止损：较持仓后最高价回撤 {drawdown_from_high:.2%}"
    if exit_rule.atr_period > 0 and exit_rule.atr_multiple is not None:
        atr = price_store.get_atr(state.symbol, execution_time, exit_rule.atr_period)
        if atr is not None:
            stop_price = state.high_watermark - exit_rule.atr_multiple * atr
            if current_price <= stop_price:
                return f"ATR移动止损：现价 {current_price:.4f} <= 最高价-ATR倍数 {stop_price:.4f}"
    if exit_rule.score_exit_offset is not None and current_score is not None:
        score_floor = method.entry_threshold - exit_rule.score_exit_offset
        if current_score <= score_floor:
            return f"模型分数衰减退出：当前分数 {current_score:.6f} <= {score_floor:.6f}"
    if exit_rule.max_holding_periods is not None and period_index - state.entry_period_index >= exit_rule.max_holding_periods:
        return f"最大持仓时间退出：已持有 {period_index - state.entry_period_index} 个采样周期"
    return None


def run_exit_managed_backtest(
    predictions: pd.DataFrame,
    method: MethodSpec,
    exit_rule: ExitRule,
    top_n: int,
    config: all_a.AllAMethodConfig,
    price_store: MinutePriceStore,
    max_position_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """执行不按目标股票换出的持仓级退出回测。"""

    rows: list[dict[str, object]] = []
    holdings: list[pd.DataFrame] = []
    trade_events: list[dict[str, object]] = []
    round_trips: list[dict[str, object]] = []
    position_lots: dict[str, list[xb.PositionLot]] = {}
    position_states: dict[str, PositionState] = {}
    last_prices: dict[str, float] = {}
    cash = float(config.initial_cash)
    previous_equity = float(config.initial_cash)
    grouped = list(predictions.groupby("datetime", sort=True))

    for period_index, (dt, group) in enumerate(grouped):
        tradable = group.dropna(subset=["entry_datetime", "entry_price"]).copy()
        if tradable.empty:
            continue
        tradable["entry_datetime"] = pd.to_datetime(tradable["entry_datetime"])
        execution_time = pd.Timestamp(tradable["entry_datetime"].min())
        held_before = summarize_position_lots(position_lots)
        current_prices = get_period_prices(tradable, execution_time, list(held_before), price_store, last_prices)
        pre_trade_equity = cash + value_positions(position_lots, current_prices)
        if pre_trade_equity <= 0:
            break

        buy_value = sell_value = cost_value = turnover_value = 0.0
        score_by_symbol = dict(zip(tradable["symbol"].astype(str), tradable[method.score_col].astype(float), strict=False))

        for symbol, shares in list(held_before.items()):
            price = current_prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            state = position_states.get(symbol)
            if state is None:
                continue
            reason = choose_exit_reason(
                state,
                float(price),
                score_by_symbol.get(symbol),
                method,
                exit_rule,
                period_index,
                execution_time,
                price_store,
            )
            if reason is None:
                continue
            sold, value, cost, events, trips = all_a.sell_position_lots(
                position_lots,
                symbol,
                shares,
                execution_time,
                float(price),
                reason + "，且持仓已满足 T+1",
                config,
            )
            if sold > 0:
                sell_value += value
                cost_value += cost
                cash += value - cost
                trade_events.extend(events)
                round_trips.extend(trips)
            if summarize_position_lots(position_lots).get(symbol, 0) <= 0:
                position_states.pop(symbol, None)

        current_shares = summarize_position_lots(position_lots)
        slots = max(top_n - len(current_shares), 0)
        if slots > 0:
            candidates = select_entry_candidates(tradable, method, top_n, config.min_assets_per_rebalance)
            if candidates.empty:
                candidates = pd.DataFrame()
            else:
                candidates = candidates.loc[~candidates["symbol"].astype(str).isin(current_shares)].head(slots)
            raw_buy_plan: dict[str, int] = {}
            entry_prices: dict[str, float] = {}
            for row in candidates.itertuples(index=False):
                symbol = str(row.symbol)
                price = float(row.entry_price)
                shares = calculate_buy_shares(price, pre_trade_equity, max_position_weight, config.lot_size)
                if shares > 0:
                    raw_buy_plan[symbol] = shares
                    entry_prices[symbol] = price
            buy_plan = reduce_buy_plan_to_cash(raw_buy_plan, entry_prices, cash, config)
            for row in candidates.itertuples(index=False):
                symbol = str(row.symbol)
                quantity = buy_plan.get(symbol, 0)
                if quantity <= 0:
                    continue
                price = float(row.entry_price)
                buy_time = pd.Timestamp(row.entry_datetime)
                value = quantity * price
                cost = all_a.calculate_trade_cost("买入", value, config.fee)
                if value + cost > cash + 1e-6:
                    continue
                score = float(getattr(row, method.score_col))
                reason = (
                    f"买入：{method.score_col} 进入周度高波动 top{top_n} 入场候选，"
                    f"单票目标仓位 {max_position_weight:.0%}"
                )
                lot = xb.PositionLot(
                    symbol=symbol,
                    shares=int(quantity),
                    buy_datetime=buy_time,
                    buy_date=buy_time.date(),
                    buy_price=price,
                    buy_cost=cost,
                    signal_datetime=pd.Timestamp(dt),
                    prediction=score,
                    reason=reason,
                )
                position_lots.setdefault(symbol, []).append(lot)
                position_states[symbol] = PositionState(symbol, price, period_index, price, score)
                cash -= value + cost
                buy_value += value
                cost_value += cost
                current_prices[symbol] = price
                trade_events.append(
                    {
                        "交易时间(execution_datetime)": buy_time,
                        "股票代码(symbol)": symbol,
                        "方向(side)": "买入",
                        "股数(shares)": int(quantity),
                        "手数(lots)": quantity / config.lot_size,
                        "价格(price)": price,
                        "成交金额(traded_value)": value,
                        "交易成本(cost)": cost,
                        "触发原因(trigger_reason)": reason,
                    }
                )

        current_shares = summarize_position_lots(position_lots)
        position_value = value_positions(position_lots, current_prices)
        equity = cash + position_value
        turnover_value = buy_value + sell_value
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "return": equity / previous_equity - 1.0 if previous_equity > 0 else 0.0,
                "cost_return": cost_value / previous_equity if previous_equity > 0 else 0.0,
                "turnover": turnover_value / previous_equity if previous_equity > 0 else 0.0,
                "buy_turnover": buy_value / previous_equity if previous_equity > 0 else 0.0,
                "sell_turnover": sell_value / previous_equity if previous_equity > 0 else 0.0,
                "holding_count": len(current_shares),
                "gross_exposure": position_value / equity if equity > 0 else 0.0,
                "cash_weight": cash / equity if equity > 0 else 0.0,
                "short_exposure": 0.0,
                "equity": equity,
            }
        )
        previous_equity = equity

        if current_shares:
            held_rows = []
            for symbol, shares in current_shares.items():
                price = current_prices.get(symbol, last_prices.get(symbol, np.nan))
                state = position_states.get(symbol)
                held_rows.append(
                    {
                        "datetime": pd.Timestamp(dt),
                        "symbol": symbol,
                        "score": score_by_symbol.get(symbol, np.nan),
                        "shares": int(shares),
                        "price": price,
                        "market_value": shares * price if np.isfinite(price) else np.nan,
                        "entry_price": state.entry_price if state else np.nan,
                        "high_watermark": state.high_watermark if state else np.nan,
                        "exit_rule": exit_rule.rule_id,
                        "top_n": top_n,
                    }
                )
            holdings.append(pd.DataFrame(held_rows))
        for symbol, price in current_prices.items():
            if np.isfinite(price) and price > 0:
                last_prices[symbol] = float(price)

    curve = pd.DataFrame(rows).sort_values("datetime")
    if curve.empty:
        return curve, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    curve["cumulative_return"] = curve["equity"] / config.initial_cash - 1.0
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    curve = curve.set_index("datetime")
    annualization = 252 * 240 / config.evaluation_step_bars
    metrics: dict[str, Any] = performance_metrics(curve["equity"], curve["return"], annualization=annualization)
    metrics.update(
        {
            "method_id": method.method_id,
            "strategy_name": method.strategy_name,
            "score_col": method.score_col,
            "entry_threshold": method.entry_threshold,
            "exit_rule": exit_rule.rule_id,
            "top_n": int(top_n),
            "max_position_weight": float(max_position_weight),
            "avg_turnover": float(curve["turnover"].mean()),
            "total_turnover": float(curve["turnover"].sum()),
            "avg_buy_turnover": float(curve["buy_turnover"].mean()),
            "avg_sell_turnover": float(curve["sell_turnover"].mean()),
            "avg_holding_count": float(curve["holding_count"].mean()),
            "avg_gross_exposure": float(curve["gross_exposure"].mean()),
            "avg_cash_weight": float(curve["cash_weight"].mean()),
            "short_exposure": 0.0,
            "signal_lag_bars": 1,
            "rebalance_horizon_bars": int(config.forward_horizon_bars),
            "evaluation_step_bars": int(config.evaluation_step_bars),
            "lot_size": int(config.lot_size),
            "trade_count": int(len([event for event in trade_events if event.get("方向(side)") in {"买入", "卖出"}])),
            "buy_trade_count": int(len([event for event in trade_events if event.get("方向(side)") == "买入"])),
            "sell_trade_count": int(len([event for event in trade_events if event.get("方向(side)") == "卖出"])),
            "t_plus_1_sell_rule": "enabled",
        }
    )
    holdings_frame = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    trade_events_frame = pd.DataFrame(trade_events)
    round_trips_frame = pd.DataFrame(round_trips)
    if not round_trips_frame.empty and "单笔收益率(return)" in round_trips_frame.columns:
        metrics["round_trip_count"] = int(len(round_trips_frame))
        metrics["round_trip_win_rate"] = float((round_trips_frame["单笔收益率(return)"] > 0).mean())
        metrics["avg_holding_days"] = float(round_trips_frame["持有天数(holding_days)"].mean())
    else:
        metrics["round_trip_count"] = 0
        metrics["round_trip_win_rate"] = 0.0
        metrics["avg_holding_days"] = 0.0
    if not trade_events_frame.empty and "手数(lots)" in trade_events_frame.columns:
        metrics["total_trade_lots"] = float(trade_events_frame["手数(lots)"].sum())
    else:
        metrics["total_trade_lots"] = 0.0
    open_rows = []
    for symbol, lots in position_lots.items():
        state = position_states.get(symbol)
        for lot in lots:
            if lot.shares <= 0:
                continue
            open_rows.append(
                {
                    "股票代码(symbol)": symbol,
                    "买入时间(buy_datetime)": lot.buy_datetime,
                    "股数(shares)": lot.shares,
                    "手数(lots)": lot.shares / config.lot_size,
                    "买入价格(buy_price)": lot.buy_price,
                    "买入原因(buy_reason)": lot.reason,
                    "买入信号时间(signal_datetime)": lot.signal_datetime,
                    "买入预测分(prediction)": lot.prediction,
                    "持仓后最高价(high_watermark)": state.high_watermark if state else np.nan,
                }
            )
    open_positions_frame = pd.DataFrame(open_rows)
    return curve, holdings_frame, trade_events_frame, round_trips_frame, open_positions_frame, metrics


def save_strategy_outputs(
    strategy_dir: Path,
    validation_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]],
    test_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]],
) -> None:
    """保存选中策略的验证和测试产物。"""

    strategy_dir.mkdir(parents=True, exist_ok=True)
    validation_curve, _, _, _, _, validation_metrics = validation_outputs
    curve, holdings, trades, round_trips, open_positions, metrics = test_outputs
    validation_curve.to_csv(strategy_dir / "validation_equity_curve.csv", encoding="utf-8-sig")
    pd.Series(validation_metrics).to_csv(strategy_dir / "validation_metrics.csv", header=["value"])
    curve.to_csv(strategy_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    holdings.to_csv(strategy_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(strategy_dir / "test_trade_events.csv", index=False, encoding="utf-8-sig")
    round_trips.to_csv(strategy_dir / "test_round_trips.csv", index=False, encoding="utf-8-sig")
    open_positions.to_csv(strategy_dir / "test_open_positions.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(strategy_dir / "test_metrics.csv", header=["value"])
    xb.plot_test_return_curve(strategy_dir, curve, metrics)
    xb.write_output_guides(strategy_dir)


def select_best_validation(candidates: pd.DataFrame, method_id: str) -> pd.Series:
    """按验证 Sharpe、回撤、交易数选择单个方法的规则。"""

    subset = candidates.loc[candidates["method_id"] == method_id].copy()
    if subset.empty:
        raise RuntimeError(f"{method_id} 没有验证候选结果。")
    subset["validation_sharpe_sort"] = subset["validation_sharpe"].fillna(-np.inf)
    subset["validation_max_drawdown_sort"] = subset["validation_max_drawdown"].fillna(-1.0)
    subset = subset.sort_values(
        ["validation_sharpe_sort", "validation_max_drawdown_sort", "validation_buy_trade_count"],
        ascending=[False, False, False],
    )
    return subset.iloc[0]


def write_output_guide(run_dir: Path) -> None:
    """写出输出文件说明。"""

    text = """# 输出文件说明

- `final_report.md`：周度高波动 top300 持仓级平仓策略主报告。
- `resolved_config.json`：解析后的路径、日期、费用、topN 和仓位参数。
- `exit_research_sources.md`：平仓策略资料来源。
- `validation_candidate_metrics.csv`：所有 topN/平仓规则候选的验证集指标。
- `selected_strategy_summary.csv`：method1/method2 由验证集选中的规则及测试指标。
- `<method_id>/`：选中策略的验证曲线、测试收益曲线、持仓、逐笔交易、平仓复盘和指标。
"""
    (run_dir / "输出文件说明.md").write_text(text, encoding="utf-8")


def write_source_notes(run_dir: Path) -> None:
    """写出联网调研资料来源。"""

    lines = ["# 平仓策略资料来源", ""]
    for row in SOURCE_NOTES:
        lines.append(f"- [{row['name']}]({row['url']})：{row['note']}")
    lines.append("")
    (run_dir / "exit_research_sources.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    run_dir: Path,
    validation_metrics: pd.DataFrame,
    selected_summary: pd.DataFrame,
    config: all_a.AllAMethodConfig,
    panel_cache: Path,
) -> None:
    """写出中文最终报告。"""

    lines = [
        "# 周度高波动 top300 持仓级平仓策略报告",
        "",
        "## 实验边界",
        "",
        "- 股票池固定为 `weekly_previous_week_high_vol_top300`：本周使用上一交易周高波动 top300。",
        "- 不重新训练 method1 XGBoost 模型，不重新选择模型特征。",
        "- 不再因为持仓股票掉出当前 topN 目标组合就卖出；买入后只由持仓级退出规则平仓。",
        "- topN、退出规则只按验证集选择，测试集只用于最终验收。",
        f"- 单票买入目标仓位上限：`{selected_summary['max_position_weight'].iloc[0]:.2%}`。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`。",
        f"- 测试集：`datetime >= {config.split_date.date()}`。",
        f"- 特征缓存：`{panel_cache}`。",
        "",
        "## 候选平仓规则",
        "",
        "- 固定止损/止盈：收益达到预设亏损或盈利比例时退出。",
        "- 移动止损：价格从持仓后最高价回撤到阈值时退出。",
        "- ATR/Chandelier Exit：用最高价减去若干倍 ATR 作为多头移动止损。",
        "- 最大持仓时间：持仓超过固定采样周期后退出。",
        "- 模型分数衰减：当前持仓自身分数低于入场阈值减去缓冲时退出。",
        "",
        "## 验证集候选结果前十",
        "",
        "| 方法 | topN | 平仓规则 | 验证 Sharpe | 验证最大回撤 | 验证买入次数 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    top_validation = validation_metrics.sort_values("validation_sharpe", ascending=False).head(10)
    for row in top_validation.itertuples(index=False):
        lines.append(
            f"| `{row.method_id}` | {int(row.top_n)} | `{row.exit_rule}` | "
            f"{float(row.validation_sharpe):.6f} | {float(row.validation_max_drawdown):.6f} | "
            f"{int(row.validation_buy_trade_count)} |"
        )
    lines.extend(
        [
            "",
            "## 选中规则与测试结果",
            "",
            "| 方法 | topN | 平仓规则 | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 已平仓胜率 | 平均持仓天数 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected_summary.sort_values("method_id").itertuples(index=False):
        lines.append(
            f"| `{row.method_id}` | {int(row.top_n)} | `{row.exit_rule}` | "
            f"{float(row.test_sharpe):.6f} | {float(row.test_annual_return):.6f} | "
            f"{float(row.test_max_drawdown):.6f} | {int(row.test_buy_trade_count)} | "
            f"{float(row.test_round_trip_win_rate):.6f} | {float(row.test_avg_holding_days):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "- 当前仍未完整模拟涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。",
            "- 单票 10% 是买入时的目标仓位上限；价格上涨后可能短暂超过 10%，本脚本不做强制市值再平衡。",
            "- 这些结果不能和旧 top1 满仓策略直接比较收益率；新策略按用户要求降低了单票集中度和总仓位。",
            "- 测试集历史已经在本项目中被查看过，后续实盘前仍应使用未来新增数据或 walk-forward 验证。",
        ]
    )
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_weekly_high_vol_exit_strategy_backtests(args: argparse.Namespace) -> Path:
    """执行验证选择和测试回测。"""

    config = create_backtest_config(args)
    panel, panel_cache = load_scored_weekly_panel(args)
    validation = panel.loc[(panel["datetime"] >= config.validation_start) & (panel["datetime"] < config.split_date)].copy()
    test = panel.loc[panel["datetime"] >= config.split_date].copy()
    if args.max_validation_periods:
        dates = validation["datetime"].drop_duplicates().sort_values().head(args.max_validation_periods)
        validation = validation.loc[validation["datetime"].isin(dates)].copy()
    if args.max_test_periods:
        dates = test["datetime"].drop_duplicates().sort_values().head(args.max_test_periods)
        test = test.loc[test["datetime"].isin(dates)].copy()

    top_n_candidates = parse_int_list(args.top_n_candidates)
    exit_rules = create_exit_rules()
    methods = create_method_specs(args)
    run_dir = args.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, Any]] = []
    validation_outputs: dict[tuple[str, int, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for method in methods:
        for top_n in top_n_candidates:
            for exit_rule in exit_rules:
                print(f"验证候选 {method.method_id} top{top_n} {exit_rule.rule_id}", flush=True)
                price_store = MinutePriceStore(config.data_root)
                outputs = run_exit_managed_backtest(
                    validation,
                    method,
                    exit_rule,
                    top_n,
                    config,
                    price_store,
                    float(args.max_position_weight),
                )
                metrics = outputs[-1]
                validation_outputs[(method.method_id, top_n, exit_rule.rule_id)] = outputs
                validation_rows.append(
                    {
                        "method_id": method.method_id,
                        "strategy_name": method.strategy_name,
                        "top_n": top_n,
                        "exit_rule": exit_rule.rule_id,
                        "validation_sharpe": metrics.get("sharpe", 0.0),
                        "validation_total_return": metrics.get("total_return", 0.0),
                        "validation_annual_return": metrics.get("annual_return", 0.0),
                        "validation_max_drawdown": metrics.get("max_drawdown", 0.0),
                        "validation_trade_count": metrics.get("trade_count", 0),
                        "validation_buy_trade_count": metrics.get("buy_trade_count", 0),
                        "validation_round_trip_win_rate": metrics.get("round_trip_win_rate", 0.0),
                        "validation_avg_holding_days": metrics.get("avg_holding_days", 0.0),
                    }
                )

    validation_metrics = pd.DataFrame(validation_rows)
    validation_metrics.to_csv(run_dir / "validation_candidate_metrics.csv", index=False, encoding="utf-8-sig")

    selected_rows: list[dict[str, Any]] = []
    for method in methods:
        selected = select_best_validation(validation_metrics, method.method_id)
        top_n = int(selected["top_n"])
        exit_rule = next(rule for rule in exit_rules if rule.rule_id == str(selected["exit_rule"]))
        print(f"测试选中 {method.method_id} top{top_n} {exit_rule.rule_id}", flush=True)
        price_store = MinutePriceStore(config.data_root)
        test_outputs = run_exit_managed_backtest(
            test,
            method,
            exit_rule,
            top_n,
            config,
            price_store,
            float(args.max_position_weight),
        )
        validation_key = (method.method_id, top_n, exit_rule.rule_id)
        strategy_dir = run_dir / method.method_id
        save_strategy_outputs(strategy_dir, validation_outputs[validation_key], test_outputs)
        test_metrics = test_outputs[-1]
        selected_rows.append(
            {
                "method_id": method.method_id,
                "strategy_name": method.strategy_name,
                "top_n": top_n,
                "exit_rule": exit_rule.rule_id,
                "max_position_weight": float(args.max_position_weight),
                "validation_sharpe": selected["validation_sharpe"],
                "validation_max_drawdown": selected["validation_max_drawdown"],
                "validation_buy_trade_count": selected["validation_buy_trade_count"],
                "test_sharpe": test_metrics.get("sharpe", 0.0),
                "test_total_return": test_metrics.get("total_return", 0.0),
                "test_annual_return": test_metrics.get("annual_return", 0.0),
                "test_annual_volatility": test_metrics.get("annual_volatility", 0.0),
                "test_max_drawdown": test_metrics.get("max_drawdown", 0.0),
                "test_trade_count": test_metrics.get("trade_count", 0),
                "test_buy_trade_count": test_metrics.get("buy_trade_count", 0),
                "test_round_trip_win_rate": test_metrics.get("round_trip_win_rate", 0.0),
                "test_avg_holding_days": test_metrics.get("avg_holding_days", 0.0),
                "test_avg_gross_exposure": test_metrics.get("avg_gross_exposure", 0.0),
            }
        )

    selected_summary = pd.DataFrame(selected_rows)
    selected_summary.to_csv(run_dir / "selected_strategy_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "data_root": str(config.data_root),
                "weekly_panel_cache": str(panel_cache),
                "method1_run_dir": str(args.method1_run_dir),
                "method2_run_dir": str(args.method2_run_dir),
                "validation_start": str(config.validation_start),
                "split_date": str(config.split_date),
                "end_date": str(config.end_date),
                "top_n_candidates": top_n_candidates,
                "max_position_weight": float(args.max_position_weight),
                "fee": asdict(config.fee),
                "exit_rules": [asdict(rule) for rule in exit_rules],
                "sources": SOURCE_NOTES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_source_notes(run_dir)
    write_output_guide(run_dir)
    write_final_report(run_dir, validation_metrics, selected_summary, config, panel_cache)
    print(f"周度高波动持仓级平仓策略回测已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """命令行入口。"""

    run_weekly_high_vol_exit_strategy_backtests(parse_args())


if __name__ == "__main__":
    main()
