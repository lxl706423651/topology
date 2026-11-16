"""Analyze and mine A-share long-only factors on OHLCV data.

Required inputs are configured near the top of this file: analysis date range,
stock/index symbols, and optional custom factor definitions. Generated outputs
are CSV/JSON reports under ``reports/factor_analysis`` by default. The script
only reads market data and writes reports; it does not mutate raw data, train
models, or place trades. Run it from the project root.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.data import OHLCV_ALIASES, standardize_ohlcv
from quant_backtest.factor_library import (
    build_factor_library as build_standard_factor_library,
)
from quant_backtest.factor_library import (
    calculate_forward_return as calculate_signal_forward_return,
)
from quant_backtest.factor_library import has_talib

try:  # pragma: no cover - optional dependency
    import talib
except Exception:  # pragma: no cover
    talib = None


# ============================================================================
# 1. 回测/分析时间范围：这里改成你想评估的 A 股分钟或日线区间。
# ============================================================================
BACKTEST_START = "2023-01-03"
BACKTEST_END = "2023-06-30"

# ============================================================================
# 2. 回测股票或者股指：空列表表示使用数据文件里的全部 symbol。
#    示例：["SH.600519", "SZ.000001"] 或 ["000300.SH"]。
# ============================================================================
TARGET_SYMBOLS: list[str] = []
UNIVERSE_FILE: Path | None = PROJECT_ROOT / "config" / "hs300_current_constituents.csv"

# ============================================================================
# 3. 自定义因子定义：函数按单个 symbol 的时间序列调用，必须只用当前和
#    历史已完成 K 线，返回与输入 index 对齐的 DataFrame。
# ============================================================================
def define_custom_factors(symbol_data: pd.DataFrame) -> pd.DataFrame:
    """Return user-defined factor columns for one symbol.

    Parameters
    ----------
    symbol_data:
        One symbol's OHLCV bars sorted by datetime. Required columns are
        datetime, symbol, open, high, low, close, volume, and optional amount.

    Returns
    -------
    pd.DataFrame
        Factor columns aligned to ``symbol_data.index``. Keep labels or future
        returns out of this function to avoid lookahead bias.
    """

    close = symbol_data["close"].astype(float)
    volume = symbol_data["volume"].astype(float).fillna(0)
    factors = pd.DataFrame(index=symbol_data.index)
    factors["custom_momentum_20"] = close / close.shift(20) - 1
    factors["custom_volume_pressure_20"] = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    return factors


# Default data points at 2023 extracted minute CSV files and the current CSI
# 300 universe file. Change DATA_PATH/UNIVERSE_FILE here or override via CLI.
DATA_PATH = PROJECT_ROOT / "ashare_minute_data_2023_2026_04" / "extracted" / "2023"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "factor_analysis"

FORWARD_HORIZON_BARS = 24
TOP_QUANTILE = 0.2
TOP_N_FACTORS = 10
MIN_ASSETS_PER_TIME = 3
MIN_PERIODS_PER_FACTOR = 30
MIN_COVERAGE = 0.35
MIN_DIRECTIONAL_RANK_IC = 0.02
MIN_EXCESS_WIN_RATE = 0.52
MAX_SELECTED_FACTOR_CORRELATION = 0.80
EVALUATION_STEP_BARS = FORWARD_HORIZON_BARS
MAX_FILES_FROM_DIRECTORY = 80
CSV_ENCODING: str | None = None

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


@dataclass(frozen=True)
class FactorConfig:
    """Resolved factor-analysis configuration.

    Parameters map to the top-level constants and CLI overrides. ``symbols`` is
    empty when all symbols in the input data should be used.
    """

    data_path: Path
    output_dir: Path
    start: pd.Timestamp
    end: pd.Timestamp
    symbols: tuple[str, ...]
    universe_file: Path | None
    forward_horizon_bars: int
    top_quantile: float
    top_n_factors: int
    min_assets_per_time: int
    min_periods_per_factor: int
    min_coverage: float
    min_directional_rank_ic: float
    min_excess_win_rate: float
    max_selected_factor_correlation: float
    evaluation_step_bars: int
    max_files_from_directory: int


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the top-of-file configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start", default=BACKTEST_START)
    parser.add_argument("--end", default=BACKTEST_END)
    parser.add_argument("--symbols", default=",".join(TARGET_SYMBOLS), help="Comma-separated symbols. Empty means all.")
    parser.add_argument("--universe-file", type=Path, default=UNIVERSE_FILE, help="CSV with a symbol column.")
    parser.add_argument("--horizon", type=int, default=FORWARD_HORIZON_BARS)
    parser.add_argument("--top-quantile", type=float, default=TOP_QUANTILE)
    parser.add_argument("--top-n", type=int, default=TOP_N_FACTORS)
    parser.add_argument("--eval-step", type=int, default=EVALUATION_STEP_BARS)
    parser.add_argument("--max-files", type=int, default=MAX_FILES_FROM_DIRECTORY)
    parser.add_argument("--min-assets", type=int, default=MIN_ASSETS_PER_TIME)
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> FactorConfig:
    """Create a validated config from constants and CLI arguments."""

    symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.universe_file is not None:
        universe = pd.read_csv(args.universe_file)
        if "symbol" not in universe.columns:
            raise ValueError(f"Universe file must contain a symbol column: {args.universe_file}")
        symbols = tuple(universe["symbol"].astype(str).drop_duplicates().tolist())
    if not 0 < args.top_quantile <= 1:
        raise ValueError("--top-quantile must be in (0, 1].")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.eval_step < 1:
        raise ValueError("--eval-step must be positive.")
    return FactorConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        symbols=symbols,
        universe_file=args.universe_file,
        forward_horizon_bars=args.horizon,
        top_quantile=args.top_quantile,
        top_n_factors=args.top_n,
        min_assets_per_time=args.min_assets,
        min_periods_per_factor=MIN_PERIODS_PER_FACTOR,
        min_coverage=MIN_COVERAGE,
        min_directional_rank_ic=MIN_DIRECTIONAL_RANK_IC,
        min_excess_win_rate=MIN_EXCESS_WIN_RATE,
        max_selected_factor_correlation=MAX_SELECTED_FACTOR_CORRELATION,
        evaluation_step_bars=args.eval_step,
        max_files_from_directory=args.max_files,
    )


def has_any_column(columns: set[str], target: str) -> bool:
    """Return whether input columns contain a target or one of its aliases."""

    candidates = {target, *OHLCV_ALIASES.get(target, [])}
    lower_columns = {col.lower() for col in columns}
    return any(candidate in columns or candidate.lower() in lower_columns for candidate in candidates)


def infer_symbol_from_path(path: Path) -> str:
    """Infer an A-share-style symbol from a single-symbol file name."""

    return path.stem


def read_market_file(path: Path, encoding: str | None = CSV_ENCODING) -> pd.DataFrame:
    """Read one CSV, pickle, or parquet OHLCV file and standardize columns."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path, encoding=encoding)
    elif suffix in {".pkl", ".pickle"}:
        raw = pd.read_pickle(path)
    elif suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported data file type: {path.suffix}")
    if not has_any_column(set(map(str, raw.columns)), "symbol"):
        raw["symbol"] = infer_symbol_from_path(path)
    return standardize_ohlcv(raw)


def select_data_files(path: Path, symbols: tuple[str, ...], max_files: int) -> list[Path]:
    """Return market-data files to load from a file or directory path."""

    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Data path does not exist: {path}")
    files = sorted(
        candidate
        for suffix in ("*.csv", "*.pkl", "*.pickle", "*.parquet")
        for candidate in path.rglob(suffix)
    )
    if symbols:
        symbol_set = set(symbols)
        matched = [file for file in files if file.stem in symbol_set]
        if matched:
            return matched
    return files[:max_files] if max_files else files


def load_market_data(config: FactorConfig) -> pd.DataFrame:
    """Load, validate, and filter OHLCV data without mutating raw files."""

    files = select_data_files(config.data_path, config.symbols, config.max_files_from_directory)
    if not files:
        raise FileNotFoundError(f"No readable market-data files found under {config.data_path}")
    parts = [read_market_file(path) for path in files]
    frame = pd.concat(parts, ignore_index=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if config.symbols:
        frame = frame.loc[frame["symbol"].isin(config.symbols)].copy()
    frame = frame.loc[pd.to_datetime(frame["datetime"]) <= config.end].copy()
    frame = frame.sort_values(["symbol", "datetime"]).drop_duplicates(["symbol", "datetime"], keep="last")
    validate_ohlcv(frame)
    return frame.reset_index(drop=True)


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Validate essential A-share OHLCV integrity before factor analysis."""

    if frame.empty:
        raise ValueError("No rows remain after data filtering.")
    bad_price = (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bool(bad_price.any()):
        raise ValueError(f"Found {int(bad_price.sum())} rows with non-positive prices.")
    invalid_ohlc = (frame["high"] < frame[["open", "close"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close"]].min(axis=1)
    )
    if bool(invalid_ohlc.any()):
        raise ValueError(f"Found {int(invalid_ohlc.sum())} rows with invalid OHLC relation.")
    if bool((frame["volume"] < 0).any()):
        raise ValueError("Found rows with negative volume.")


def safe_divide(numerator: pd.Series, denominator: pd.Series | np.ndarray | float) -> pd.Series:
    """Divide two aligned arrays and treat zero denominators as missing."""

    denom = pd.Series(denominator, index=numerator.index) if not isinstance(denominator, pd.Series) else denominator
    return numerator / denom.replace(0, np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate a rolling z-score using current and historical values."""

    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return safe_divide(series - mean, std)


def calculate_rsi(close: pd.Series, window: int) -> pd.Series:
    """Calculate RSI with TA-Lib when available, otherwise use Wilder-style EWM."""

    if talib is not None:
        return pd.Series(talib.RSI(close.to_numpy(dtype=float), timeperiod=window), index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate ATR with TA-Lib when available, otherwise use true-range mean."""

    if talib is not None:
        values = talib.ATR(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), timeperiod=window)
        return pd.Series(values, index=close.index)
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate CCI with TA-Lib when available, otherwise use mean deviation."""

    if talib is not None:
        values = talib.CCI(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), timeperiod=window)
        return pd.Series(values, index=close.index)
    typical = (high + low + close) / 3
    mean = typical.rolling(window, min_periods=window).mean()
    mad = (typical - mean).abs().rolling(window, min_periods=window).mean()
    return safe_divide(typical - mean, 0.015 * mad)


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Calculate MFI with TA-Lib when available, otherwise use money-flow sums."""

    if talib is not None:
        values = talib.MFI(
            high.to_numpy(float),
            low.to_numpy(float),
            close.to_numpy(float),
            volume.to_numpy(float),
            timeperiod=window,
        )
        return pd.Series(values, index=close.index)
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    direction = np.sign(typical.diff().fillna(0))
    positive = raw_flow.where(direction > 0, 0).rolling(window, min_periods=window).sum()
    negative = raw_flow.where(direction < 0, 0).rolling(window, min_periods=window).sum().abs()
    ratio = positive / negative.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate ADX with TA-Lib when available, otherwise use a compact fallback."""

    if talib is not None:
        values = talib.ADX(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), timeperiod=window)
        return pd.Series(values, index=close.index)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = calculate_atr(high, low, close, window)
    plus_di = 100 * plus_dm.rolling(window, min_periods=window).sum() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window, min_periods=window).sum() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window, min_periods=window).mean()


def calculate_willr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate Williams %R with TA-Lib when available, otherwise use rolling highs/lows."""

    if talib is not None:
        values = talib.WILLR(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), timeperiod=window)
        return pd.Series(values, index=close.index)
    highest = high.rolling(window, min_periods=window).max()
    lowest = low.rolling(window, min_periods=window).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def calculate_factor_library(symbol_data: pd.DataFrame) -> pd.DataFrame:
    """Build built-in technical/statistical factors for one sorted symbol."""

    frame = symbol_data.sort_values("datetime").copy()
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float).fillna(0)
    amount = frame["amount"].astype(float) if "amount" in frame.columns else close * volume * 100

    factors: dict[str, pd.Series | np.ndarray] = {}
    ret_1 = close.pct_change()
    vwap_proxy = safe_divide(amount, volume * 100).fillna((high + low + close) / 3)

    factors["oc_ret"] = close / open_.replace(0, np.nan) - 1
    factors["hl_range"] = safe_divide(high - low, close)
    factors["upper_shadow"] = safe_divide(high - pd.concat([open_, close], axis=1).max(axis=1), close)
    factors["lower_shadow"] = safe_divide(pd.concat([open_, close], axis=1).min(axis=1) - low, close)
    factors["close_pos_in_bar"] = safe_divide(close - low, high - low)
    factors["vwap_gap"] = close / vwap_proxy.replace(0, np.nan) - 1

    for window in (5, 10, 20, 30, 60, 120):
        rolling_ret = ret_1.rolling(window, min_periods=window)
        rolling_close = close.rolling(window, min_periods=window)
        rolling_volume = volume.rolling(window, min_periods=window)
        rolling_amount = amount.rolling(window, min_periods=window)
        sma = rolling_close.mean()
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        high_max = high.rolling(window, min_periods=window).max()
        low_min = low.rolling(window, min_periods=window).min()

        factors[f"momentum_{window}"] = close / close.shift(window) - 1
        factors[f"reversal_{window}"] = -(close / close.shift(window) - 1)
        factors[f"ret_mean_{window}"] = rolling_ret.mean()
        factors[f"ret_std_{window}"] = rolling_ret.std(ddof=0)
        factors[f"ret_skew_{window}"] = rolling_ret.skew()
        factors[f"sma_gap_{window}"] = close / sma.replace(0, np.nan) - 1
        factors[f"ema_gap_{window}"] = close / ema.replace(0, np.nan) - 1
        factors[f"price_rank_{window}"] = close.rolling(window, min_periods=window).rank(pct=True)
        factors[f"dist_high_{window}"] = close / high_max.replace(0, np.nan) - 1
        factors[f"dist_low_{window}"] = close / low_min.replace(0, np.nan) - 1
        factors[f"volume_ratio_{window}"] = volume / rolling_volume.mean().replace(0, np.nan)
        factors[f"amount_ratio_{window}"] = amount / rolling_amount.mean().replace(0, np.nan)
        factors[f"volume_z_{window}"] = rolling_zscore(volume, window)
        factors[f"amount_z_{window}"] = rolling_zscore(amount, window)
        factors[f"range_mean_{window}"] = factors["hl_range"].rolling(window, min_periods=window).mean()
        factors[f"range_std_{window}"] = factors["hl_range"].rolling(window, min_periods=window).std(ddof=0)
        factors[f"vwap_gap_mean_{window}"] = factors["vwap_gap"].rolling(window, min_periods=window).mean()

    for short, long in ((3, 10), (5, 20), (10, 60), (20, 120)):
        short_ma = close.rolling(short, min_periods=short).mean()
        long_ma = close.rolling(long, min_periods=long).mean()
        factors[f"ma_cross_{short}_{long}"] = short_ma / long_ma.replace(0, np.nan) - 1

    for window in (10, 20, 60):
        middle = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=0)
        upper = middle + 2 * std
        lower = middle - 2 * std
        factors[f"boll_pctb_{window}"] = (close - lower) / (upper - lower).replace(0, np.nan)
        factors[f"boll_width_{window}"] = (upper - lower) / middle.replace(0, np.nan)
        factors[f"boll_upper_gap_{window}"] = close / upper.replace(0, np.nan) - 1
        factors[f"boll_lower_gap_{window}"] = close / lower.replace(0, np.nan) - 1

    for window in (6, 12, 14, 24, 48):
        factors[f"rsi_{window}"] = calculate_rsi(close, window)
        factors[f"roc_{window}"] = close / close.shift(window) - 1
        factors[f"atr_ratio_{window}"] = calculate_atr(high, low, close, window) / close.replace(0, np.nan)
        factors[f"cci_{window}"] = calculate_cci(high, low, close, window)
        factors[f"willr_{window}"] = calculate_willr(high, low, close, window)
        factors[f"mfi_{window}"] = calculate_mfi(high, low, close, volume, window)

    for window in (14, 24, 48):
        factors[f"adx_{window}"] = calculate_adx(high, low, close, window)

    for fast, slow, signal in ((6, 13, 5), (12, 26, 9), (24, 52, 18)):
        if talib is not None:
            dif, dea, hist = talib.MACD(close.to_numpy(float), fastperiod=fast, slowperiod=slow, signalperiod=signal)
            factors[f"macd_dif_{fast}_{slow}_{signal}"] = pd.Series(dif, index=frame.index) / close.replace(0, np.nan)
            factors[f"macd_dea_{fast}_{slow}_{signal}"] = pd.Series(dea, index=frame.index) / close.replace(0, np.nan)
            factors[f"macd_hist_{fast}_{slow}_{signal}"] = pd.Series(hist, index=frame.index) / close.replace(0, np.nan)
        else:
            dif = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - close.ewm(
                span=slow,
                adjust=False,
                min_periods=slow,
            ).mean()
            dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
            hist = dif - dea
            factors[f"macd_dif_{fast}_{slow}_{signal}"] = dif / close.replace(0, np.nan)
            factors[f"macd_dea_{fast}_{slow}_{signal}"] = dea / close.replace(0, np.nan)
            factors[f"macd_hist_{fast}_{slow}_{signal}"] = hist / close.replace(0, np.nan)

    low_14 = low.rolling(14, min_periods=14).min()
    high_14 = high.rolling(14, min_periods=14).max()
    factors["stoch_k_14"] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
    factors["stoch_d_14"] = factors["stoch_k_14"].rolling(3, min_periods=3).mean()
    if talib is not None:
        obv = pd.Series(talib.OBV(close.to_numpy(float), volume.to_numpy(float)), index=frame.index)
    else:
        obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    factors["obv_z_60"] = rolling_zscore(obv, 60)
    factors["obv_z_120"] = rolling_zscore(obv, 120)

    factors["minute_sin"] = np.sin(2 * math.pi * (pd.to_datetime(frame["datetime"]).dt.hour * 60 + pd.to_datetime(frame["datetime"]).dt.minute) / (24 * 60))
    factors["minute_cos"] = np.cos(2 * math.pi * (pd.to_datetime(frame["datetime"]).dt.hour * 60 + pd.to_datetime(frame["datetime"]).dt.minute) / (24 * 60))
    factor_frame = pd.DataFrame(factors, index=frame.index)
    return factor_frame.replace([np.inf, -np.inf], np.nan)


def calculate_forward_return(symbol_data: pd.DataFrame, horizon: int) -> pd.Series:
    """Return signal-lagged future return for one symbol.

    The factor is observed on bar t. The simulated long-only evaluation enters
    at the next bar close and exits after ``horizon`` bars, avoiding same-bar
    lookahead.
    """

    close = symbol_data["close"].astype(float)
    entry = close.shift(-1)
    exit_ = close.shift(-(horizon + 1))
    return exit_ / entry.replace(0, np.nan) - 1


def build_factor_frame(config: FactorConfig, market: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create one analysis frame with factors and signal-lagged labels."""

    parts: list[pd.DataFrame] = []
    keep_times: set[pd.Timestamp] | None = None
    if config.evaluation_step_bars > 1:
        times = pd.Index(sorted(market["datetime"].drop_duplicates()))
        keep_times = set(pd.Timestamp(value) for value in times[:: config.evaluation_step_bars])
    for _, group in market.groupby("symbol", sort=False):
        group = group.sort_values("datetime").copy()
        built_in = build_standard_factor_library(group)
        custom = define_custom_factors(group.copy())
        custom = custom.add_prefix("user_") if not custom.empty else custom
        factors = pd.concat([built_in, custom], axis=1)
        part = group[["datetime", "symbol", "open", "high", "low", "close", "volume"]].copy()
        if "amount" in group.columns:
            part["amount"] = group["amount"]
        part["forward_return"] = calculate_signal_forward_return(group, config.forward_horizon_bars)
        part = pd.concat([part, factors], axis=1)
        if keep_times is not None:
            part = part.loc[part["datetime"].isin(keep_times)].copy()
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.loc[(frame["datetime"] >= config.start) & (frame["datetime"] <= config.end)].copy()
    id_cols = {"datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "forward_return"}
    factor_cols = [col for col in frame.select_dtypes(include=[np.number]).columns if col not in id_cols]
    return frame.sort_values(["datetime", "symbol"]).reset_index(drop=True), factor_cols


def calculate_rank_ic(data: pd.DataFrame, factor_col: str, min_assets: int) -> pd.Series:
    """Calculate cross-sectional Spearman Rank IC by timestamp."""

    work = data[["datetime", factor_col, "forward_return"]].dropna().copy()
    if work.empty:
        return pd.Series(dtype=float)
    counts = work.groupby("datetime")[factor_col].transform("count")
    work = work.loc[counts >= min_assets].copy()
    if work.empty:
        return pd.Series(dtype=float)
    work["x"] = work.groupby("datetime")[factor_col].rank(method="average")
    work["y"] = work.groupby("datetime")["forward_return"].rank(method="average")
    work["xy"] = work["x"] * work["y"]
    work["x2"] = work["x"] * work["x"]
    work["y2"] = work["y"] * work["y"]
    stats = work.groupby("datetime").agg(
        n=("x", "count"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxy=("xy", "sum"),
        sx2=("x2", "sum"),
        sy2=("y2", "sum"),
        ux=("x", "nunique"),
        uy=("y", "nunique"),
    )
    numerator = stats["n"] * stats["sxy"] - stats["sx"] * stats["sy"]
    denom_x = stats["n"] * stats["sx2"] - stats["sx"] * stats["sx"]
    denom_y = stats["n"] * stats["sy2"] - stats["sy"] * stats["sy"]
    denominator = np.sqrt(denom_x * denom_y)
    corr = numerator / denominator.replace(0, np.nan)
    corr = corr.loc[(stats["ux"] >= 2) & (stats["uy"] >= 2)]
    return corr.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def evaluate_long_only_returns(
    data: pd.DataFrame,
    factor_col: str,
    direction: int,
    top_quantile: float,
    min_assets: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate a factor as a long-only top-quantile selector."""

    rows: list[dict[str, object]] = []
    previous_symbols: set[str] | None = None
    turnover_values: list[float] = []
    score_col = "__score__"
    working = data[["datetime", "symbol", "forward_return", factor_col]].dropna().copy()
    if working.empty:
        return pd.DataFrame(), {
            "long_mean_return": 0.0,
            "universe_mean_return": 0.0,
            "excess_mean_return": 0.0,
            "long_win_rate": 0.0,
            "excess_win_rate": 0.0,
            "long_sharpe_like": 0.0,
            "avg_turnover": 0.0,
            "periods": 0,
        }
    counts = working.groupby("datetime")["symbol"].transform("count")
    working = working.loc[counts >= min_assets].copy()
    if working.empty:
        return pd.DataFrame(), {
            "long_mean_return": 0.0,
            "universe_mean_return": 0.0,
            "excess_mean_return": 0.0,
            "long_win_rate": 0.0,
            "excess_win_rate": 0.0,
            "long_sharpe_like": 0.0,
            "avg_turnover": 0.0,
            "periods": 0,
        }
    working[score_col] = direction * working[factor_col].astype(float)
    working["rank_pct"] = working.groupby("datetime")[score_col].rank(method="first", pct=True)
    selected = working.loc[working["rank_pct"] > (1 - top_quantile)].copy()
    if selected.empty:
        selected = working.loc[working["rank_pct"] == 1].copy()

    universe_return = working.groupby("datetime")["forward_return"].mean()
    long_return = selected.groupby("datetime")["forward_return"].mean()
    selected_count = selected.groupby("datetime")["symbol"].size()
    symbol_strings = selected.groupby("datetime")["symbol"].agg(lambda values: ",".join(sorted(map(str, values))))

    for dt, symbols_text in symbol_strings.items():
        selected_symbols = set(symbols_text.split(",")) if symbols_text else set()
        if previous_symbols is not None:
            union = selected_symbols | previous_symbols
            turnover_values.append(1 - len(selected_symbols & previous_symbols) / len(union) if union else 0.0)
        previous_symbols = selected_symbols

    common_times = long_return.index.intersection(universe_return.index)
    for dt in common_times:
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "factor": factor_col if direction > 0 else f"neg_{factor_col}",
                "selected_count": int(selected_count.loc[dt]),
                "long_return": float(long_return.loc[dt]),
                "universe_return": float(universe_return.loc[dt]),
                "excess_return": float(long_return.loc[dt] - universe_return.loc[dt]),
                "selected_symbols": str(symbol_strings.loc[dt]),
            }
        )

    returns = pd.DataFrame(rows)
    if returns.empty:
        return returns, {
            "long_mean_return": 0.0,
            "universe_mean_return": 0.0,
            "excess_mean_return": 0.0,
            "long_win_rate": 0.0,
            "excess_win_rate": 0.0,
            "long_sharpe_like": 0.0,
            "avg_turnover": 0.0,
            "periods": 0,
        }
    long_ret = returns["long_return"].astype(float)
    excess = returns["excess_return"].astype(float)
    long_std = float(long_ret.std(ddof=0))
    metrics = {
        "long_mean_return": float(long_ret.mean()),
        "universe_mean_return": float(returns["universe_return"].astype(float).mean()),
        "excess_mean_return": float(excess.mean()),
        "long_win_rate": float((long_ret > 0).mean()),
        "excess_win_rate": float((excess > 0).mean()),
        "long_sharpe_like": float(long_ret.mean() / long_std * math.sqrt(len(long_ret))) if long_std else 0.0,
        "avg_turnover": float(np.mean(turnover_values)) if turnover_values else 0.0,
        "periods": int(len(returns)),
    }
    return returns, metrics


def evaluate_factor(config: FactorConfig, frame: pd.DataFrame, factor_col: str) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one raw factor and choose the better long-only direction."""

    data = frame[["datetime", "symbol", "forward_return", factor_col]].replace([np.inf, -np.inf], np.nan)
    finite_factor = data[factor_col].notna()
    finite_target = data["forward_return"].notna()
    coverage = float((finite_factor & finite_target).mean()) if len(data) else 0.0
    ic = calculate_rank_ic(data, factor_col, config.min_assets_per_time)
    ic_mean = float(ic.mean()) if len(ic) else 0.0
    ic_std = float(ic.std(ddof=0)) if len(ic) else 0.0
    ic_ir = float(ic_mean / ic_std * math.sqrt(len(ic))) if ic_std else 0.0

    long_high, high_metrics = evaluate_long_only_returns(
        data,
        factor_col,
        direction=1,
        top_quantile=config.top_quantile,
        min_assets=config.min_assets_per_time,
    )
    long_low, low_metrics = evaluate_long_only_returns(
        data,
        factor_col,
        direction=-1,
        top_quantile=config.top_quantile,
        min_assets=config.min_assets_per_time,
    )
    high_score = high_metrics["excess_mean_return"] * max(high_metrics["excess_win_rate"], 0.01)
    low_score = low_metrics["excess_mean_return"] * max(low_metrics["excess_win_rate"], 0.01)
    if low_score > high_score:
        direction = -1
        selected_name = f"neg_{factor_col}"
        selected_metrics = low_metrics
        returns = long_low
    else:
        direction = 1
        selected_name = factor_col
        selected_metrics = high_metrics
        returns = long_high

    periods = int(selected_metrics["periods"])
    directional_ic_mean = ic_mean * direction
    feasible = (
        periods >= config.min_periods_per_factor
        and coverage >= config.min_coverage
        and selected_metrics["excess_mean_return"] > 0
        and selected_metrics["long_mean_return"] > 0
        and selected_metrics["excess_win_rate"] >= config.min_excess_win_rate
        and directional_ic_mean >= config.min_directional_rank_ic
    )
    combined_score = (
        directional_ic_mean * max(periods, 1) ** 0.5
        + selected_metrics["excess_mean_return"] * 10_000
        + selected_metrics["long_sharpe_like"]
    )
    result: dict[str, object] = {
        "factor": factor_col,
        "selected_factor": selected_name,
        "direction": direction,
        "coverage": coverage,
        "rank_ic_mean": ic_mean,
        "directional_rank_ic_mean": directional_ic_mean,
        "rank_ic_std": ic_std,
        "rank_ic_ir_like": ic_ir,
        "rank_ic_periods": int(len(ic)),
        "feasible_long_only": bool(feasible),
        "combined_score": float(combined_score),
        **selected_metrics,
    }
    return result, returns


def select_low_correlation_factors(config: FactorConfig, summary: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select effective factors while limiting pairwise factor correlation."""

    effective = summary.loc[summary["feasible_long_only"]].copy()
    effective = effective.sort_values(
        ["combined_score", "excess_mean_return", "directional_rank_ic_mean"],
        ascending=False,
    )
    selected_rows: list[pd.Series] = []
    selected_series: list[pd.Series] = []
    for _, row in effective.iterrows():
        raw_factor = str(row["factor"])
        direction = int(row["direction"])
        candidate = (direction * frame[raw_factor]).replace([np.inf, -np.inf], np.nan)
        max_corr = 0.0
        for existing in selected_series:
            pair = pd.concat([candidate, existing], axis=1).dropna()
            if len(pair) < config.min_periods_per_factor:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
            if pd.notna(corr):
                max_corr = max(max_corr, abs(float(corr)))
        if max_corr <= config.max_selected_factor_correlation:
            out = row.copy()
            out["max_abs_corr_to_selected"] = max_corr
            out["selection_note"] = "selected_with_correlation_limit"
            selected_rows.append(out)
            selected_series.append(candidate)
        if len(selected_rows) >= config.top_n_factors:
            break

    if len(selected_rows) < config.top_n_factors:
        already = {str(row["selected_factor"]) for row in selected_rows}
        for _, row in effective.iterrows():
            if str(row["selected_factor"]) in already:
                continue
            out = row.copy()
            out["max_abs_corr_to_selected"] = np.nan
            out["selection_note"] = "selected_to_fill_top_n_after_correlation_filter"
            selected_rows.append(out)
            already.add(str(row["selected_factor"]))
            if len(selected_rows) >= config.top_n_factors:
                break

    selected = pd.DataFrame(selected_rows)
    return effective, selected


def mine_factors(config: FactorConfig, frame: pd.DataFrame, factor_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all candidate factors and return top long-only candidates."""

    summaries: list[dict[str, object]] = []
    selected_returns: list[pd.DataFrame] = []
    for idx, factor_col in enumerate(factor_cols, start=1):
        result, returns = evaluate_factor(config, frame, factor_col)
        summaries.append(result)
        if not returns.empty:
            selected_returns.append(returns)
        if idx % 25 == 0 or idx == len(factor_cols):
            print(f"evaluated {idx}/{len(factor_cols)} factors")

    summary = pd.DataFrame(summaries)
    ordered_summary = summary.sort_values("combined_score", ascending=False)
    effective, selected = select_low_correlation_factors(config, ordered_summary, frame)
    if selected.empty:
        selected = ordered_summary.head(config.top_n_factors).copy()
        selected["max_abs_corr_to_selected"] = np.nan
        selected["selection_note"] = "fallback_no_effective_factor_met_criteria"
    returns = pd.concat(selected_returns, ignore_index=True) if selected_returns else pd.DataFrame()
    if not returns.empty:
        returns = returns.loc[returns["factor"].isin(set(selected["selected_factor"].astype(str)))].copy()
    return ordered_summary, effective, selected, returns


def save_outputs(
    config: FactorConfig,
    market: pd.DataFrame,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    effective: pd.DataFrame,
    selected: pd.DataFrame,
    returns: pd.DataFrame,
) -> Path:
    """Write factor-analysis reports and resolved run context."""

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(run_dir / "factor_summary.csv", index=False, encoding="utf-8-sig")
    effective.to_csv(run_dir / "effective_factors.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(run_dir / "selected_effective_factors_top10.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(run_dir / "selected_factors_top10.csv", index=False, encoding="utf-8-sig")
    if not returns.empty:
        returns.to_csv(run_dir / "long_only_factor_returns.csv", index=False, encoding="utf-8-sig")
    context = {
        "data_path": str(config.data_path),
        "universe_file": str(config.universe_file) if config.universe_file is not None else None,
        "start": str(config.start),
        "end": str(config.end),
        "symbols": list(config.symbols) if config.symbols else "all",
        "requested_symbol_count": len(config.symbols),
        "missing_requested_symbols": sorted(set(config.symbols) - set(frame["symbol"].astype(str).unique())),
        "loaded_rows": int(len(market)),
        "analysis_rows": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "factor_count": int(len(summary)),
        "top_n_factors": int(config.top_n_factors),
        "forward_horizon_bars": int(config.forward_horizon_bars),
        "evaluation_step_bars": int(config.evaluation_step_bars),
        "top_quantile": float(config.top_quantile),
        "talib_available": bool(has_talib()),
        "effective_factor_criteria": {
            "min_periods_per_factor": int(config.min_periods_per_factor),
            "min_coverage": float(config.min_coverage),
            "min_directional_rank_ic": float(config.min_directional_rank_ic),
            "min_excess_win_rate": float(config.min_excess_win_rate),
            "must_have_positive_excess_mean_return": True,
            "must_have_positive_long_mean_return": True,
            "max_selected_factor_correlation": float(config.max_selected_factor_correlation),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method_notes": [
            "A-share long-only evaluation: select the top quantile by directional factor score; no short leg is constructed.",
            "Forward returns enter at the next bar close and exit after the configured horizon to avoid same-bar lookahead.",
            "Rank IC is cross-sectional Spearman correlation between factor value and future return at each timestamp.",
        ],
    }
    (run_dir / "run_context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    """Run factor mining and stop after reporting the configured top-N factors."""

    config = create_config(parse_args())
    market = load_market_data(config)
    frame, factor_cols = build_factor_frame(config, market)
    if frame.empty:
        raise ValueError("No analysis rows remain after applying date range.")
    if not factor_cols:
        raise ValueError("No numeric factor columns were generated.")
    print(
        f"loaded rows={len(market)} analysis_rows={len(frame)} "
        f"symbols={frame['symbol'].nunique()} factors={len(factor_cols)} talib={has_talib()}"
    )
    summary, effective, selected, returns = mine_factors(config, frame, factor_cols)
    run_dir = save_outputs(config, market, frame, summary, effective, selected, returns)
    selected_view = selected[
        [
            "selected_factor",
            "directional_rank_ic_mean",
            "excess_mean_return",
            "excess_win_rate",
            "long_mean_return",
            "long_win_rate",
            "periods",
            "max_abs_corr_to_selected",
            "feasible_long_only",
        ]
    ]
    print(selected_view.to_string(index=False))
    print(f"Saved factor-analysis report to {run_dir}")


if __name__ == "__main__":
    main()
