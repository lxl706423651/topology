"""回测周度高波动 top5 入场和独立 lot 平仓策略。

输入为已有周度高波动 top300 特征缓存、method1 XGBoost 模型、method2 CTA
打分函数和本地 A 股分钟行情。脚本固定每个采样点最多选择 top5，且每个候选
必须超过各自 method 的入场门控阈值；每次开仓按当前总资产的 10% 买入。
同一股票可以在后续交易日继续加仓，每次开仓形成独立 lot，按各自买入价、
ATR、最高价和技术指标独立平仓。候选退出规则先在策略训练集筛选，再在验证
集确认，最终只把选中规则搬到测试集。原始行情和既有模型不会被修改。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.factor_library import calculate_atr, calculate_rsi, talib
from quant_backtest.metrics import performance_metrics
import scripts.run_all_a_method_backtests as all_a
import scripts.run_multi_factor_cta_backtests as cta
import scripts.run_xgboost_long_only_backtest as xb


OUTPUT_DIR = PROJECT_ROOT / "reports" / "weekly_high_vol_lot_exit_backtests"
WEEKLY_UNIVERSE_ID = "weekly_previous_week_high_vol_top300"
CHINESE_FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
SOURCE_NOTES = [
    {
        "name": "Investopedia Trailing Stop",
        "url": "https://www.investopedia.com/terms/t/trailingstop.asp",
        "note": "移动止损会随着价格向有利方向移动，用于锁定利润并限制反转损失。",
    },
    {
        "name": "Fidelity Exit Strategies",
        "url": "https://www.fidelity.com/learning-center/trading-investing/trading/exit-strategies",
        "note": "常见退出方式包括止损、跟踪止损、利润目标和到期重新评估。",
    },
    {
        "name": "StockCharts Chandelier Exit",
        "url": "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit",
        "note": "Chandelier Exit 使用最高价减去 ATR 倍数作为多头动态止损。",
    },
    {
        "name": "Investopedia Volatility Stops",
        "url": "https://www.investopedia.com/articles/trading/09/volatility-stops.asp",
        "note": "波动率止损强调用 ATR 等工具把止损放在市场噪声之外。",
    },
    {
        "name": "Investopedia Stop Placement",
        "url": "https://www.investopedia.com/articles/trading/06/stopplacement.asp",
        "note": "常见止损方法包括硬止损、ATR 止损、多日高低点和指标止损。",
    },
]


def configure_plot_fonts() -> None:
    """配置 Matplotlib 中文字体，避免收益图标题和图例出现方框。"""

    try:
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as plt
    except Exception:
        return
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback"]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class LotExitRule:
    """独立 lot 平仓规则。

    rule_id: 稳定规则名。
    atr_period: ATR 使用的分钟 bar 窗口。
    chandelier_multiple: 最高价减 ATR 倍数的移动止损，None 表示不用。
    initial_atr_multiple: 买入价减入场 ATR 倍数的初始止损，None 表示不用。
    trailing_atr_multiple: 最高价减当前 ATR 倍数的移动止损，None 表示不用。
    ema_period: EMA 跌破退出窗口，0 表示不用。
    use_macd_exit: 是否启用 MACD 空头交叉退出。
    use_psar_exit: 是否启用 Parabolic SAR 跌破退出。
    rsi_period: RSI 窗口，0 表示不用。
    rsi_overbought: lot 持仓期间 RSI 曾达到该值后，才允许 RSI 回落退出。
    rsi_exit_level: RSI 从过热区回落到该值以下时退出。
    score_exit_offset: 当前分数低于入场阈值减去该偏移时退出，None 表示不用。
    max_holding_periods: 最大持仓采样周期数，None 表示不用。
    """

    rule_id: str
    atr_period: int
    chandelier_multiple: float | None = None
    initial_atr_multiple: float | None = None
    trailing_atr_multiple: float | None = None
    ema_period: int = 0
    use_macd_exit: bool = False
    use_psar_exit: bool = False
    rsi_period: int = 0
    rsi_overbought: float | None = None
    rsi_exit_level: float | None = None
    score_exit_offset: float | None = None
    max_holding_periods: int | None = None


@dataclass(frozen=True)
class MethodSpec:
    """method1/method2 入场分数定义。"""

    method_id: str
    strategy_name: str
    score_col: str
    entry_threshold: float


@dataclass
class ManagedLot:
    """单次开仓形成的独立 lot。"""

    lot_id: int
    symbol: str
    shares: int
    buy_datetime: pd.Timestamp
    buy_date: Any
    buy_price: float
    buy_cost: float
    signal_datetime: pd.Timestamp
    entry_score: float
    reason: str
    entry_period_index: int
    high_watermark: float
    entry_atr: float | None
    seen_rsi_overbought: bool = False


class IndicatorPriceStore:
    """按需读取本地分钟行情，并缓存退出规则需要的技术指标。"""

    def __init__(self, data_root: Path, rules: list[LotExitRule]) -> None:
        self.data_root = data_root
        self.rules = rules
        self.frames: dict[str, pd.DataFrame] = {}

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """读取单只股票分钟行情并计算所有候选规则需要的指标。"""

        if symbol in self.frames:
            return self.frames[symbol]
        parts: list[pd.DataFrame] = []
        for year_dir in sorted(path for path in self.data_root.iterdir() if path.is_dir() and path.name.isdigit()):
            path = year_dir / f"{symbol}.csv"
            if not path.exists():
                continue
            part = pd.read_csv(
                path,
                usecols=lambda col: col in {"datetime", "open", "high", "low", "close", "volume", "amount"},
            )
            parts.append(part)
        if not parts:
            frame = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
            self.frames[symbol] = frame
            return frame
        frame = pd.concat(parts, ignore_index=True)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last").set_index("datetime")
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        for period in sorted({rule.atr_period for rule in self.rules if rule.atr_period > 0}):
            frame[f"atr_{period}"] = calculate_atr(high, low, close, period)
        for period in sorted({rule.ema_period for rule in self.rules if rule.ema_period > 0}):
            frame[f"ema_{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()
        for period in sorted({rule.rsi_period for rule in self.rules if rule.rsi_period > 0}):
            frame[f"rsi_{period}"] = calculate_rsi(close, period)
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        frame["macd"] = ema12 - ema26
        frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
        frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
        if talib is not None:
            frame["psar"] = talib.SAR(high.to_numpy(float), low.to_numpy(float), acceleration=0.02, maximum=0.2)
        else:
            frame["psar"] = np.nan
        self.frames[symbol] = frame
        return frame

    def get_row(self, symbol: str, dt: pd.Timestamp) -> pd.Series | None:
        """取得指定执行时间的行情和指标；没有精确 bar 时返回 None。"""

        frame = self.load_symbol(symbol)
        if frame.empty or dt not in frame.index:
            return None
        return frame.loc[dt]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=xb.DATA_ROOT)
    parser.add_argument("--weekly-panel-cache", type=Path, default=None)
    parser.add_argument("--method1-run-dir", type=Path, default=all_a.METHOD1_RUN_DIR)
    parser.add_argument("--method2-run-dir", type=Path, default=all_a.METHOD2_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--strategy-train-start", default="2024-11-01")
    parser.add_argument("--strategy-validation-start", default="2025-03-01")
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--buy-weight", type=float, default=0.10)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--train-shortlist-size", type=int, default=5)
    parser.add_argument("--max-train-periods", type=int, default=0)
    parser.add_argument("--max-validation-periods", type=int, default=0)
    parser.add_argument("--max-test-periods", type=int, default=0)
    parser.add_argument("--broker-commission-rate", type=float, default=0.0008)
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--min-commission", type=float, default=0.0)
    return parser.parse_args()


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
    raise FileNotFoundError("未找到 weekly_previous_week_high_vol_top300 特征缓存。")


def create_exit_rules() -> list[LotExitRule]:
    """固定动态退出候选集合。"""

    return [
        LotExitRule("chandelier_atr120_3p0", atr_period=120, chandelier_multiple=3.0),
        LotExitRule("chandelier_atr240_3p0", atr_period=240, chandelier_multiple=3.0),
        LotExitRule("atr_initial2_trail3_atr120", atr_period=120, initial_atr_multiple=2.0, trailing_atr_multiple=3.0),
        LotExitRule("atr_initial2_trail4_atr240", atr_period=240, initial_atr_multiple=2.0, trailing_atr_multiple=4.0),
        LotExitRule("ema96_break_chandelier3", atr_period=120, chandelier_multiple=3.0, ema_period=96),
        LotExitRule("macd_bearish_chandelier3", atr_period=120, chandelier_multiple=3.0, use_macd_exit=True),
        LotExitRule("psar_chandelier3", atr_period=120, chandelier_multiple=3.0, use_psar_exit=True),
        LotExitRule(
            "rsi120_reversal_chandelier3",
            atr_period=120,
            chandelier_multiple=3.0,
            rsi_period=120,
            rsi_overbought=70.0,
            rsi_exit_level=55.0,
        ),
        LotExitRule("score_decay_chandelier3", atr_period=120, chandelier_multiple=3.0, score_exit_offset=0.03),
        LotExitRule("chandelier3_max3d", atr_period=120, chandelier_multiple=3.0, max_holding_periods=30),
    ]


def create_config(args: argparse.Namespace) -> all_a.AllAMethodConfig:
    """生成回测配置。"""

    fee = all_a.FeeConfig(
        broker_commission_rate=float(args.broker_commission_rate),
        stamp_duty_rate=float(args.stamp_duty_rate),
        transfer_fee_rate=float(args.transfer_fee_rate),
        min_commission=float(args.min_commission),
        commission_includes_exchange_fees=True,
    )
    return all_a.AllAMethodConfig(
        data_root=args.data_root,
        feature_manifest=xb.MODEL_FEATURE_MANIFEST,
        method1_run_dir=args.method1_run_dir,
        method2_run_dir=args.method2_run_dir,
        output_dir=args.output_dir,
        panel_cache_dir=args.output_dir / "panel_cache",
        validation_start=pd.Timestamp(args.strategy_train_start),
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
        fee=fee,
    )


def create_method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    """读取 method1/method2 固定入场阈值。"""

    method1_rule = all_a.load_method_rule(args.method1_run_dir)
    method2_rule = all_a.load_cta_rule(args.method2_run_dir, "cta_trend_breakout")
    if method1_rule.confidence_threshold is None or method2_rule.confidence_threshold is None:
        raise ValueError("method1/method2 必须有 confidence_threshold。")
    return [
        MethodSpec("method1_xgboost_lot_exit", "method1 XGBoost lot独立平仓", "prediction", float(method1_rule.confidence_threshold)),
        MethodSpec("method2_cta_lot_exit", "method2 CTA lot独立平仓", "cta_trend_breakout_score", float(method2_rule.confidence_threshold)),
    ]


def load_scored_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    """读取周度高波动面板并生成 method1/method2 分数。"""

    cache_path = find_weekly_panel_cache(args.weekly_panel_cache)
    print(f"读取周度高波动特征缓存：{cache_path}", flush=True)
    panel = pd.read_pickle(cache_path)
    model = joblib.load(args.method1_run_dir / "xgboost_model.joblib")
    feature_cols = (args.method1_run_dir / "feature_list.txt").read_text(encoding="utf-8").splitlines()
    missing = [col for col in feature_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"特征缓存缺少模型特征：{missing[:10]}")
    panel = panel.copy()
    panel["prediction"] = model.predict(panel.loc[:, feature_cols]).astype("float32")
    panel = cta.add_cta_factor_scores(panel, cta.SCORE_BLEND_STRENGTH)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["entry_datetime"] = pd.to_datetime(panel["entry_datetime"])
    panel = panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    return panel, cache_path


def limit_periods(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """调试时限制采样周期数量。"""

    if limit <= 0 or frame.empty:
        return frame
    dates = frame["datetime"].drop_duplicates().sort_values().head(limit)
    return frame.loc[frame["datetime"].isin(dates)].copy()


def calculate_equity(cash: float, lots: list[ManagedLot], prices: dict[str, float]) -> float:
    """按当前价格估算账户权益。"""

    value = 0.0
    for lot in lots:
        price = prices.get(lot.symbol)
        if lot.shares > 0 and price is not None and np.isfinite(price) and price > 0:
            value += lot.shares * float(price)
    return float(cash + value)


def sell_lot(
    lot: ManagedLot,
    sell_datetime: pd.Timestamp,
    sell_price: float,
    reason: str,
    config: all_a.AllAMethodConfig,
) -> tuple[float, float, dict[str, object], dict[str, object]]:
    """整笔卖出一个满足 T+1 的独立 lot。"""

    value = lot.shares * sell_price
    cost = all_a.calculate_trade_cost("卖出", value, config.fee)
    gross_pnl = (sell_price - lot.buy_price) * lot.shares
    net_pnl = gross_pnl - lot.buy_cost - cost
    denominator = lot.buy_price * lot.shares + lot.buy_cost
    event = {
        "交易时间(execution_datetime)": sell_datetime,
        "股票代码(symbol)": lot.symbol,
        "lot_id": lot.lot_id,
        "方向(side)": "卖出",
        "股数(shares)": lot.shares,
        "手数(lots)": lot.shares / config.lot_size,
        "价格(price)": sell_price,
        "成交金额(traded_value)": value,
        "交易成本(cost)": cost,
        "触发原因(trigger_reason)": reason,
    }
    trip = {
        "股票代码(symbol)": lot.symbol,
        "lot_id": lot.lot_id,
        "买入时间(buy_datetime)": lot.buy_datetime,
        "卖出时间(sell_datetime)": sell_datetime,
        "持有天数(holding_days)": (pd.Timestamp(sell_datetime).date() - lot.buy_date).days,
        "买入价格(buy_price)": lot.buy_price,
        "卖出价格(sell_price)": sell_price,
        "股数(shares)": lot.shares,
        "手数(lots)": lot.shares / config.lot_size,
        "买入原因(buy_reason)": lot.reason,
        "卖出原因(sell_reason)": reason,
        "买入信号时间(signal_datetime)": lot.signal_datetime,
        "买入预测分(prediction)": lot.entry_score,
        "毛收益(gross_pnl)": gross_pnl,
        "交易成本(cost)": lot.buy_cost + cost,
        "净收益(net_pnl)": net_pnl,
        "单笔收益率(return)": net_pnl / denominator if denominator else 0.0,
    }
    return value, cost, event, trip


def get_rule_atr(row: pd.Series, rule: LotExitRule) -> float | None:
    """读取当前 ATR。"""

    if rule.atr_period <= 0:
        return None
    value = row.get(f"atr_{rule.atr_period}", np.nan)
    if not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def choose_exit_reason(
    lot: ManagedLot,
    row: pd.Series,
    current_score: float | None,
    method: MethodSpec,
    rule: LotExitRule,
    period_index: int,
) -> str | None:
    """根据独立 lot 状态和当前指标判断是否退出。"""

    close = float(row["close"])
    if not np.isfinite(close) or close <= 0:
        return None
    high = float(row.get("high", close))
    lot.high_watermark = max(lot.high_watermark, high if np.isfinite(high) and high > 0 else close)
    atr = get_rule_atr(row, rule)
    stops: list[tuple[str, float]] = []
    if rule.chandelier_multiple is not None and atr is not None:
        stops.append(("Chandelier/ATR吊灯止损", lot.high_watermark - rule.chandelier_multiple * atr))
    if rule.initial_atr_multiple is not None and lot.entry_atr is not None:
        stops.append(("入场ATR初始止损", lot.buy_price - rule.initial_atr_multiple * lot.entry_atr))
    if rule.trailing_atr_multiple is not None and atr is not None:
        stops.append(("ATR移动止损", lot.high_watermark - rule.trailing_atr_multiple * atr))
    if stops:
        stop_name, stop_price = max(stops, key=lambda item: item[1])
        if close <= stop_price:
            return f"{stop_name}：现价 {close:.4f} <= 止损价 {stop_price:.4f}"
    if rule.ema_period > 0:
        ema = row.get(f"ema_{rule.ema_period}", np.nan)
        if np.isfinite(ema) and close < float(ema):
            return f"EMA跌破退出：现价 {close:.4f} < EMA{rule.ema_period} {float(ema):.4f}"
    if rule.use_macd_exit:
        macd = row.get("macd", np.nan)
        signal = row.get("macd_signal", np.nan)
        hist = row.get("macd_hist", np.nan)
        if np.isfinite(macd) and np.isfinite(signal) and np.isfinite(hist) and macd < signal and hist < 0:
            return "MACD空头信号退出：MACD低于Signal且柱体为负"
    if rule.use_psar_exit:
        psar = row.get("psar", np.nan)
        if np.isfinite(psar) and close < float(psar):
            return f"PSAR跌破退出：现价 {close:.4f} < SAR {float(psar):.4f}"
    if rule.rsi_period > 0:
        rsi = row.get(f"rsi_{rule.rsi_period}", np.nan)
        if np.isfinite(rsi):
            if rule.rsi_overbought is not None and float(rsi) >= rule.rsi_overbought:
                lot.seen_rsi_overbought = True
            if lot.seen_rsi_overbought and rule.rsi_exit_level is not None and float(rsi) <= rule.rsi_exit_level:
                return f"RSI过热回落退出：RSI {float(rsi):.2f} <= {rule.rsi_exit_level:.2f}"
    if rule.score_exit_offset is not None and current_score is not None and np.isfinite(current_score):
        score_floor = method.entry_threshold - rule.score_exit_offset
        if current_score <= score_floor:
            return f"模型分数衰减退出：当前分数 {current_score:.6f} <= {score_floor:.6f}"
    if rule.max_holding_periods is not None and period_index - lot.entry_period_index >= rule.max_holding_periods:
        return f"最大持仓周期退出：已持有 {period_index - lot.entry_period_index} 个采样周期"
    return None


def select_top_entries(group: pd.DataFrame, method: MethodSpec, top_n: int, min_assets: int) -> pd.DataFrame:
    """选择超过门控阈值的 topN 入场候选。"""

    if len(group) < min_assets:
        return pd.DataFrame()
    work = group.dropna(subset=[method.score_col, "entry_price"]).copy()
    work = work.loc[work[method.score_col] >= method.entry_threshold]
    return work.sort_values(method.score_col, ascending=False).head(top_n)


def can_add_symbol(active_lots: list[ManagedLot], symbol: str, execution_date: Any) -> bool:
    """判断同一股票是否允许在新交易日加仓。"""

    lots = [lot for lot in active_lots if lot.symbol == symbol and lot.shares > 0]
    if not lots:
        return True
    latest_buy_date = max(lot.buy_date for lot in lots)
    return latest_buy_date < execution_date


def run_lot_backtest(
    predictions: pd.DataFrame,
    method: MethodSpec,
    rule: LotExitRule,
    config: all_a.AllAMethodConfig,
    price_store: IndicatorPriceStore,
    top_n: int,
    buy_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """执行 lot 独立平仓回测。"""

    active_lots: list[ManagedLot] = []
    rows: list[dict[str, object]] = []
    holdings: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    trips: list[dict[str, object]] = []
    cash = float(config.initial_cash)
    previous_equity = float(config.initial_cash)
    last_prices: dict[str, float] = {}
    next_lot_id = 1
    grouped = list(predictions.groupby("datetime", sort=True))
    for period_index, (signal_dt, group) in enumerate(grouped):
        tradable = group.dropna(subset=["entry_datetime", "entry_price"]).copy()
        if tradable.empty:
            continue
        tradable["entry_datetime"] = pd.to_datetime(tradable["entry_datetime"])
        execution_time = pd.Timestamp(tradable["entry_datetime"].min())
        execution_date = execution_time.date()
        symbol_scores = dict(zip(tradable["symbol"].astype(str), tradable[method.score_col].astype(float), strict=False))
        current_prices = dict(zip(tradable["symbol"].astype(str), tradable["entry_price"].astype(float), strict=False))
        for lot in active_lots:
            row = price_store.get_row(lot.symbol, execution_time)
            if row is not None:
                price = float(row["close"])
                if np.isfinite(price) and price > 0:
                    current_prices[lot.symbol] = price
                    last_prices[lot.symbol] = price
            elif lot.symbol in last_prices:
                current_prices[lot.symbol] = last_prices[lot.symbol]
        pre_trade_equity = calculate_equity(cash, active_lots, current_prices)
        if pre_trade_equity <= 0:
            break

        buy_value = sell_value = cost_value = 0.0
        remaining_lots: list[ManagedLot] = []
        for lot in active_lots:
            row = price_store.get_row(lot.symbol, execution_time)
            if row is None:
                remaining_lots.append(lot)
                continue
            price = float(row["close"])
            reason = choose_exit_reason(lot, row, symbol_scores.get(lot.symbol), method, rule, period_index)
            if reason is None or lot.buy_date >= execution_date:
                remaining_lots.append(lot)
                continue
            value, cost, event, trip = sell_lot(
                lot,
                execution_time,
                price,
                reason + "，且该lot已满足T+1",
                config,
            )
            cash += value - cost
            sell_value += value
            cost_value += cost
            events.append(event)
            trips.append(trip)
        active_lots = remaining_lots

        equity_after_sells = calculate_equity(cash, active_lots, current_prices)
        entries = select_top_entries(tradable, method, top_n, config.min_assets_per_rebalance)
        if not entries.empty:
            for row in entries.itertuples(index=False):
                symbol = str(row.symbol)
                buy_time = pd.Timestamp(row.entry_datetime)
                if not can_add_symbol(active_lots, symbol, buy_time.date()):
                    continue
                price = float(row.entry_price)
                if not np.isfinite(price) or price <= 0:
                    continue
                target_value = equity_after_sells * buy_weight
                shares = int(math.floor(target_value / (price * config.lot_size)) * config.lot_size)
                if shares <= 0:
                    continue
                trade_value = shares * price
                buy_cost = all_a.calculate_trade_cost("买入", trade_value, config.fee)
                while shares > 0 and trade_value + buy_cost > cash + 1e-6:
                    shares -= config.lot_size
                    trade_value = shares * price
                    buy_cost = all_a.calculate_trade_cost("买入", trade_value, config.fee)
                if shares <= 0:
                    continue
                ind_row = price_store.get_row(symbol, buy_time)
                entry_atr = get_rule_atr(ind_row, rule) if ind_row is not None else None
                score = float(getattr(row, method.score_col))
                reason = f"买入：{method.score_col} 超过门控并进入周度高波动top{top_n}，本lot按总资产{buy_weight:.0%}开仓"
                lot = ManagedLot(
                    lot_id=next_lot_id,
                    symbol=symbol,
                    shares=int(shares),
                    buy_datetime=buy_time,
                    buy_date=buy_time.date(),
                    buy_price=price,
                    buy_cost=buy_cost,
                    signal_datetime=pd.Timestamp(signal_dt),
                    entry_score=score,
                    reason=reason,
                    entry_period_index=period_index,
                    high_watermark=price,
                    entry_atr=entry_atr,
                )
                next_lot_id += 1
                active_lots.append(lot)
                cash -= trade_value + buy_cost
                buy_value += trade_value
                cost_value += buy_cost
                current_prices[symbol] = price
                last_prices[symbol] = price
                events.append(
                    {
                        "交易时间(execution_datetime)": buy_time,
                        "股票代码(symbol)": symbol,
                        "lot_id": lot.lot_id,
                        "方向(side)": "买入",
                        "股数(shares)": int(shares),
                        "手数(lots)": shares / config.lot_size,
                        "价格(price)": price,
                        "成交金额(traded_value)": trade_value,
                        "交易成本(cost)": buy_cost,
                        "触发原因(trigger_reason)": reason,
                    }
                )

        equity = calculate_equity(cash, active_lots, current_prices)
        position_value = equity - cash
        turnover = buy_value + sell_value
        rows.append(
            {
                "datetime": pd.Timestamp(signal_dt),
                "return": equity / previous_equity - 1.0 if previous_equity > 0 else 0.0,
                "cost_return": cost_value / previous_equity if previous_equity > 0 else 0.0,
                "turnover": turnover / previous_equity if previous_equity > 0 else 0.0,
                "buy_turnover": buy_value / previous_equity if previous_equity > 0 else 0.0,
                "sell_turnover": sell_value / previous_equity if previous_equity > 0 else 0.0,
                "holding_count": len({lot.symbol for lot in active_lots}),
                "open_lot_count": len(active_lots),
                "gross_exposure": position_value / equity if equity > 0 else 0.0,
                "cash_weight": cash / equity if equity > 0 else 0.0,
                "short_exposure": 0.0,
                "equity": equity,
            }
        )
        previous_equity = equity
        for lot in active_lots:
            price = current_prices.get(lot.symbol, np.nan)
            holdings.append(
                {
                    "datetime": pd.Timestamp(signal_dt),
                    "symbol": lot.symbol,
                    "lot_id": lot.lot_id,
                    "shares": lot.shares,
                    "price": price,
                    "market_value": lot.shares * price if np.isfinite(price) else np.nan,
                    "buy_datetime": lot.buy_datetime,
                    "buy_price": lot.buy_price,
                    "high_watermark": lot.high_watermark,
                    "entry_score": lot.entry_score,
                    "current_score": symbol_scores.get(lot.symbol, np.nan),
                    "exit_rule": rule.rule_id,
                }
            )

    curve = pd.DataFrame(rows).sort_values("datetime")
    if curve.empty:
        return curve, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    curve["cumulative_return"] = curve["equity"] / config.initial_cash - 1.0
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    curve = curve.set_index("datetime")
    annualization = 252 * 240 / config.evaluation_step_bars
    metrics: dict[str, Any] = performance_metrics(curve["equity"], curve["return"], annualization=annualization)
    trades = pd.DataFrame(events)
    round_trips = pd.DataFrame(trips)
    holdings_frame = pd.DataFrame(holdings)
    open_positions = pd.DataFrame([asdict(lot) for lot in active_lots])
    metrics.update(
        {
            "method_id": method.method_id,
            "strategy_name": method.strategy_name,
            "rule": f"{method.method_id}_{rule.rule_id}",
            "score_col": method.score_col,
            "entry_threshold": method.entry_threshold,
            "exit_rule": rule.rule_id,
            "top_n": int(top_n),
            "buy_weight": float(buy_weight),
            "avg_turnover": float(curve["turnover"].mean()),
            "total_turnover": float(curve["turnover"].sum()),
            "avg_buy_turnover": float(curve["buy_turnover"].mean()),
            "avg_sell_turnover": float(curve["sell_turnover"].mean()),
            "avg_holding_count": float(curve["holding_count"].mean()),
            "max_holding_count": int(curve["holding_count"].max()),
            "avg_open_lot_count": float(curve["open_lot_count"].mean()),
            "max_open_lot_count": int(curve["open_lot_count"].max()),
            "avg_gross_exposure": float(curve["gross_exposure"].mean()),
            "max_gross_exposure": float(curve["gross_exposure"].max()),
            "avg_cash_weight": float(curve["cash_weight"].mean()),
            "short_exposure": 0.0,
            "signal_lag_bars": 1,
            "lot_size": int(config.lot_size),
            "trade_count": int(len(trades)),
            "buy_trade_count": int((trades["方向(side)"] == "买入").sum()) if not trades.empty else 0,
            "sell_trade_count": int((trades["方向(side)"] == "卖出").sum()) if not trades.empty else 0,
        }
    )
    if not round_trips.empty:
        metrics["round_trip_count"] = int(len(round_trips))
        metrics["round_trip_win_rate"] = float((round_trips["单笔收益率(return)"] > 0).mean())
        metrics["avg_holding_days"] = float(round_trips["持有天数(holding_days)"].mean())
    else:
        metrics["round_trip_count"] = 0
        metrics["round_trip_win_rate"] = 0.0
        metrics["avg_holding_days"] = 0.0
    metrics["total_trade_lots"] = float(trades["手数(lots)"].sum()) if not trades.empty else 0.0
    return curve, holdings_frame, trades, round_trips, open_positions, metrics


def summarize_metrics(prefix: str, metrics: dict[str, Any], method: MethodSpec, rule: LotExitRule) -> dict[str, Any]:
    """把指标字典压平成候选表一行。"""

    return {
        "method_id": method.method_id,
        "strategy_name": method.strategy_name,
        "exit_rule": rule.rule_id,
        f"{prefix}_sharpe": metrics.get("sharpe", 0.0),
        f"{prefix}_total_return": metrics.get("total_return", 0.0),
        f"{prefix}_annual_return": metrics.get("annual_return", 0.0),
        f"{prefix}_max_drawdown": metrics.get("max_drawdown", 0.0),
        f"{prefix}_trade_count": metrics.get("trade_count", 0),
        f"{prefix}_buy_trade_count": metrics.get("buy_trade_count", 0),
        f"{prefix}_round_trip_win_rate": metrics.get("round_trip_win_rate", 0.0),
        f"{prefix}_avg_holding_days": metrics.get("avg_holding_days", 0.0),
        f"{prefix}_avg_gross_exposure": metrics.get("avg_gross_exposure", 0.0),
        f"{prefix}_max_gross_exposure": metrics.get("max_gross_exposure", 0.0),
    }


def save_outputs(
    strategy_dir: Path,
    train_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]],
    validation_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]],
    test_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]],
) -> None:
    """保存选中规则的三段产物。"""

    strategy_dir.mkdir(parents=True, exist_ok=True)
    train_curve, _, _, _, _, train_metrics = train_outputs
    validation_curve, _, _, _, _, validation_metrics = validation_outputs
    curve, holdings, trades, trips, open_positions, metrics = test_outputs
    train_curve.to_csv(strategy_dir / "train_equity_curve.csv", encoding="utf-8-sig")
    validation_curve.to_csv(strategy_dir / "validation_equity_curve.csv", encoding="utf-8-sig")
    pd.Series(train_metrics).to_csv(strategy_dir / "train_metrics.csv", header=["value"])
    pd.Series(validation_metrics).to_csv(strategy_dir / "validation_metrics.csv", header=["value"])
    curve.to_csv(strategy_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    holdings.to_csv(strategy_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(strategy_dir / "test_trade_events.csv", index=False, encoding="utf-8-sig")
    trips.to_csv(strategy_dir / "test_round_trips.csv", index=False, encoding="utf-8-sig")
    open_positions.to_csv(strategy_dir / "test_open_positions.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(strategy_dir / "test_metrics.csv", header=["value"])
    configure_plot_fonts()
    xb.plot_test_return_curve(strategy_dir, curve, metrics)
    xb.write_output_guides(strategy_dir)


def write_output_guide(run_dir: Path) -> None:
    """写出输出文件说明。"""

    text = """# 输出文件说明

- `final_report.md`：周度高波动 top5 独立 lot 平仓主报告。
- `resolved_config.json`：解析后的路径、日期、费用、top5、单笔买入比例和候选规则。
- `exit_research_sources.md`：动态止盈止损资料来源。
- `train_candidate_metrics.csv`：所有候选退出规则在策略训练集上的结果。
- `validation_candidate_metrics.csv`：训练集 shortlist 在验证集上的结果。
- `selected_strategy_summary.csv`：验证后选中规则的训练、验证、测试指标。
- `<method_id>/`：选中规则的三段曲线、测试持仓、逐笔交易、lot 平仓复盘和指标。
"""
    (run_dir / "输出文件说明.md").write_text(text, encoding="utf-8")


def write_source_notes(run_dir: Path) -> None:
    """写出资料来源。"""

    lines = ["# 动态止盈止损资料来源", ""]
    for row in SOURCE_NOTES:
        lines.append(f"- [{row['name']}]({row['url']})：{row['note']}")
    lines.append("")
    (run_dir / "exit_research_sources.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    run_dir: Path,
    selected: pd.DataFrame,
    train_metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    args: argparse.Namespace,
    panel_cache: Path,
) -> None:
    """写出中文最终报告。"""

    lines = [
        "# 周度高波动 top5 独立 lot 平仓回测报告",
        "",
        "## 实验边界",
        "",
        "- 股票池固定为每周用上一周高波动 top300。",
        "- 每个采样点最多买入 top5，且每只股票分数必须超过 method 固定入场门控阈值。",
        "- 每次开仓按当前总资产的 10% 计算目标买入金额；现金不足时按现金约束向下缩减。",
        "- 同一股票后续交易日仍在 top5 且有现金时可以继续加仓；每次加仓形成独立 lot。",
        "- 每个 lot 按自己的买入价、入场 ATR、持仓后最高价和技术指标独立平仓。",
        "- 候选退出规则先在策略训练集筛选，再在验证集确认，测试集只做最终验收。",
        f"- 策略训练集：`{args.strategy_train_start} <= datetime < {args.strategy_validation_start}`。",
        f"- 策略验证集：`{args.strategy_validation_start} <= datetime < {args.split_date}`。",
        f"- 测试集：`datetime >= {args.split_date}`。",
        f"- 周度高波动特征缓存：`{panel_cache}`。",
        "",
        "## 候选退出规则",
        "",
        "- Chandelier/ATR 吊灯止损：最高价减 ATR 倍数。",
        "- 入场 ATR 初始止损 + ATR 移动止损。",
        "- EMA 跌破退出 + Chandelier 保护。",
        "- MACD 空头信号退出 + Chandelier 保护。",
        "- Parabolic SAR 退出 + Chandelier 保护。",
        "- RSI 过热后回落退出 + Chandelier 保护。",
        "- 模型分数衰减退出 + Chandelier 保护。",
        "- Chandelier 保护 + 最大持仓周期约束。",
        "",
        "## 训练集候选前十",
        "",
        "| 方法 | 退出规则 | 训练 Sharpe | 训练最大回撤 | 训练买入次数 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in train_metrics.sort_values("train_sharpe", ascending=False).head(10).itertuples(index=False):
        lines.append(
            f"| `{row.method_id}` | `{row.exit_rule}` | {float(row.train_sharpe):.6f} | "
            f"{float(row.train_max_drawdown):.6f} | {int(row.train_buy_trade_count)} |"
        )
    lines.extend(
        [
            "",
            "## 验证集 shortlist 结果",
            "",
            "| 方法 | 退出规则 | 验证 Sharpe | 验证最大回撤 | 验证买入次数 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in validation_metrics.sort_values("validation_sharpe", ascending=False).head(10).itertuples(index=False):
        lines.append(
            f"| `{row.method_id}` | `{row.exit_rule}` | {float(row.validation_sharpe):.6f} | "
            f"{float(row.validation_max_drawdown):.6f} | {int(row.validation_buy_trade_count)} |"
        )
    lines.extend(
        [
            "",
            "## 选中规则与测试结果",
            "",
            "| 方法 | 退出规则 | 训练 Sharpe | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 胜率 | 平均持仓天数 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected.sort_values("method_id").itertuples(index=False):
        lines.append(
            f"| `{row.method_id}` | `{row.exit_rule}` | {float(row.train_sharpe):.6f} | "
            f"{float(row.validation_sharpe):.6f} | {float(row.test_sharpe):.6f} | "
            f"{float(row.test_annual_return):.6f} | {float(row.test_max_drawdown):.6f} | "
            f"{int(row.test_buy_trade_count)} | {float(row.test_round_trip_win_rate):.6f} | "
            f"{float(row.test_avg_holding_days):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "- 当前仍未完整模拟涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。",
            "- 买入金额按总资产 10% 计算，但现金不足时不能加杠杆，只能减少或跳过买入。",
            "- 本项目历史测试集已经被查看过，新的高 Sharpe 仍需要未来样本或 walk-forward 验证。",
        ]
    )
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_validation_rule(train_metrics: pd.DataFrame, validation_metrics: pd.DataFrame, method_id: str, shortlist_size: int) -> pd.Series:
    """用训练集 shortlist，再按验证集选择最终规则。"""

    train_subset = train_metrics.loc[train_metrics["method_id"] == method_id].copy()
    shortlist = (
        train_subset.sort_values(["train_sharpe", "train_max_drawdown", "train_buy_trade_count"], ascending=[False, False, False])
        .head(shortlist_size)["exit_rule"]
        .tolist()
    )
    validation_subset = validation_metrics.loc[
        (validation_metrics["method_id"] == method_id) & (validation_metrics["exit_rule"].isin(shortlist))
    ].copy()
    if validation_subset.empty:
        raise RuntimeError(f"{method_id} 没有 validation shortlist 结果。")
    return validation_subset.sort_values(
        ["validation_sharpe", "validation_max_drawdown", "validation_buy_trade_count"],
        ascending=[False, False, False],
    ).iloc[0]


def run_weekly_high_vol_lot_exit_backtests(args: argparse.Namespace) -> Path:
    """执行训练筛选、验证确认和测试验收。"""

    rules = create_exit_rules()
    config = create_config(args)
    methods = create_method_specs(args)
    panel, panel_cache = load_scored_panel(args)
    train_start = pd.Timestamp(args.strategy_train_start)
    validation_start = pd.Timestamp(args.strategy_validation_start)
    split_date = pd.Timestamp(args.split_date)
    train = panel.loc[(panel["datetime"] >= train_start) & (panel["datetime"] < validation_start)].copy()
    validation = panel.loc[(panel["datetime"] >= validation_start) & (panel["datetime"] < split_date)].copy()
    test = panel.loc[panel["datetime"] >= split_date].copy()
    train = limit_periods(train, args.max_train_periods)
    validation = limit_periods(validation, args.max_validation_periods)
    test = limit_periods(test, args.max_test_periods)

    run_dir = args.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    outputs_by_key: dict[tuple[str, str, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}

    for method in methods:
        for rule in rules:
            print(f"训练候选 {method.method_id} {rule.rule_id}", flush=True)
            store = IndicatorPriceStore(config.data_root, rules)
            outputs = run_lot_backtest(train, method, rule, config, store, args.top_n, args.buy_weight)
            outputs_by_key[(method.method_id, rule.rule_id, "train")] = outputs
            train_rows.append(summarize_metrics("train", outputs[-1], method, rule))

    train_metrics = pd.DataFrame(train_rows)
    train_metrics.to_csv(run_dir / "train_candidate_metrics.csv", index=False, encoding="utf-8-sig")

    for method in methods:
        shortlist = (
            train_metrics.loc[train_metrics["method_id"] == method.method_id]
            .sort_values(["train_sharpe", "train_max_drawdown", "train_buy_trade_count"], ascending=[False, False, False])
            .head(args.train_shortlist_size)["exit_rule"]
            .tolist()
        )
        for rule in [rule for rule in rules if rule.rule_id in shortlist]:
            print(f"验证候选 {method.method_id} {rule.rule_id}", flush=True)
            store = IndicatorPriceStore(config.data_root, rules)
            outputs = run_lot_backtest(validation, method, rule, config, store, args.top_n, args.buy_weight)
            outputs_by_key[(method.method_id, rule.rule_id, "validation")] = outputs
            validation_rows.append(summarize_metrics("validation", outputs[-1], method, rule))

    validation_metrics = pd.DataFrame(validation_rows)
    validation_metrics.to_csv(run_dir / "validation_candidate_metrics.csv", index=False, encoding="utf-8-sig")

    selected_rows: list[dict[str, Any]] = []
    for method in methods:
        selected_validation = select_validation_rule(train_metrics, validation_metrics, method.method_id, args.train_shortlist_size)
        rule = next(rule for rule in rules if rule.rule_id == str(selected_validation["exit_rule"]))
        print(f"测试选中 {method.method_id} {rule.rule_id}", flush=True)
        store = IndicatorPriceStore(config.data_root, rules)
        test_outputs = run_lot_backtest(test, method, rule, config, store, args.top_n, args.buy_weight)
        train_row = train_metrics.loc[(train_metrics["method_id"] == method.method_id) & (train_metrics["exit_rule"] == rule.rule_id)].iloc[0]
        validation_outputs = outputs_by_key[(method.method_id, rule.rule_id, "validation")]
        train_outputs = outputs_by_key[(method.method_id, rule.rule_id, "train")]
        save_outputs(run_dir / method.method_id, train_outputs, validation_outputs, test_outputs)
        test_metrics = test_outputs[-1]
        selected_rows.append(
            {
                "method_id": method.method_id,
                "strategy_name": method.strategy_name,
                "exit_rule": rule.rule_id,
                "top_n": int(args.top_n),
                "buy_weight": float(args.buy_weight),
                "train_sharpe": train_row["train_sharpe"],
                "train_max_drawdown": train_row["train_max_drawdown"],
                "train_buy_trade_count": train_row["train_buy_trade_count"],
                "validation_sharpe": selected_validation["validation_sharpe"],
                "validation_max_drawdown": selected_validation["validation_max_drawdown"],
                "validation_buy_trade_count": selected_validation["validation_buy_trade_count"],
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
                "test_max_gross_exposure": test_metrics.get("max_gross_exposure", 0.0),
            }
        )

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(run_dir / "selected_strategy_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "data_root": str(config.data_root),
                "weekly_panel_cache": str(panel_cache),
                "strategy_train_start": str(args.strategy_train_start),
                "strategy_validation_start": str(args.strategy_validation_start),
                "split_date": str(args.split_date),
                "end_date": str(args.end_date),
                "top_n": int(args.top_n),
                "buy_weight": float(args.buy_weight),
                "train_shortlist_size": int(args.train_shortlist_size),
                "fee": asdict(config.fee),
                "exit_rules": [asdict(rule) for rule in rules],
                "sources": SOURCE_NOTES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_source_notes(run_dir)
    write_output_guide(run_dir)
    write_final_report(run_dir, selected, train_metrics, validation_metrics, args, panel_cache)
    print(f"周度高波动 top5 独立 lot 平仓回测已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """命令行入口。"""

    run_weekly_high_vol_lot_exit_backtests(parse_args())


if __name__ == "__main__":
    main()
