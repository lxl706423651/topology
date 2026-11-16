"""Reusable A-share factor construction library.

The functions in this module build technical, price-volume, volatility, and
intraday factors from OHLCV bars. Every factor uses only current and historical
completed bars. Forward returns are provided by a separate helper for analysis
labels and must not be used as model features.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import talib
except Exception:  # pragma: no cover
    talib = None


def has_talib() -> bool:
    """Return whether TA-Lib is available in the current Python environment."""

    return talib is not None


def safe_divide(numerator: pd.Series, denominator: pd.Series | np.ndarray | float) -> pd.Series:
    """Divide two aligned arrays and treat zero denominators as missing."""

    denom = pd.Series(denominator, index=numerator.index) if not isinstance(denominator, pd.Series) else denominator
    return numerator / denom.replace(0, np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate a rolling z-score using current and historical values."""

    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return safe_divide(series - mean, std)


def rolling_weighted_mean(series: pd.Series, window: int) -> pd.Series:
    """Calculate a linearly weighted rolling mean using only historical values."""

    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    return series.rolling(window, min_periods=window).apply(lambda values: float(np.dot(values, weights)), raw=True)


def rolling_trend_stats(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Estimate rolling log-price trend slope and R-squared up to each bar."""

    y = series.astype(float)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    mean_x = x.rolling(window, min_periods=window).mean()
    mean_y = y.rolling(window, min_periods=window).mean()
    cov_xy = (x * y).rolling(window, min_periods=window).mean() - mean_x * mean_y
    var_x = (x * x).rolling(window, min_periods=window).mean() - mean_x * mean_x
    var_y = (y * y).rolling(window, min_periods=window).mean() - mean_y * mean_y
    slope = safe_divide(cov_xy, var_x)
    r_squared = safe_divide(cov_xy * cov_xy, var_x * var_y).clip(lower=0, upper=1)
    return slope, r_squared


def rolling_sqrt_mean_nonnegative(series: pd.Series, window: int) -> pd.Series:
    """Return sqrt of a rolling mean after clipping negative estimator noise."""

    return np.sqrt(series.clip(lower=0).rolling(window, min_periods=window).mean())


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


def calculate_past_return(symbol_data: pd.DataFrame, window: int) -> pd.Series:
    """Return past close-to-close return ending at the current completed bar."""

    close = symbol_data["close"].astype(float)
    return close / close.shift(window).replace(0, np.nan) - 1


def build_factor_library(symbol_data: pd.DataFrame) -> pd.DataFrame:
    """Build a broad, reusable factor library for one sorted symbol.

    Parameters
    ----------
    symbol_data:
        One symbol's OHLCV bars sorted by datetime. Required columns are
        open, high, low, close, volume, and optional amount.

    Returns
    -------
    pd.DataFrame
        Factor columns aligned to ``symbol_data.index``. Families include
        momentum/reversal, volatility, trend, price rank, volume/amount,
        Bollinger, RSI, ROC, ATR, CCI, Williams %R, MFI, ADX, MACD, stochastic,
        OBV, Amihud liquidity, range-based volatility, price-volume interaction,
        and intraday time features. Every feature uses only information available
        at or before the current completed bar; use a one-bar execution lag when
        trading from the generated values.
    """

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
    log_open = np.log(open_.where(open_ > 0))
    log_high = np.log(high.where(high > 0))
    log_low = np.log(low.where(low > 0))
    log_close = np.log(close.where(close > 0))
    log_hl = log_high - log_low
    log_co = log_close - log_open
    parkinson_bar = log_hl.pow(2) / (4 * math.log(2))
    garman_klass_bar = 0.5 * log_hl.pow(2) - (2 * math.log(2) - 1) * log_co.pow(2)
    rogers_satchell_bar = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (
        log_low - log_open
    )
    amihud_bar = safe_divide(ret_1.abs(), amount.replace(0, np.nan)) * 100_000_000
    signed_amount = np.sign(ret_1.fillna(0)) * amount

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
        ret_std = rolling_ret.std(ddof=0)
        downside = ret_1.where(ret_1 < 0, 0.0)
        trend_slope, trend_r2 = rolling_trend_stats(log_close, window)

        factors[f"momentum_{window}"] = close / close.shift(window) - 1
        factors[f"reversal_{window}"] = -(close / close.shift(window) - 1)
        factors[f"ret_mean_{window}"] = rolling_ret.mean()
        factors[f"ret_std_{window}"] = ret_std
        factors[f"ret_skew_{window}"] = rolling_ret.skew()
        factors[f"weighted_momentum_{window}"] = rolling_weighted_mean(ret_1, window)
        factors[f"downside_vol_{window}"] = downside.rolling(window, min_periods=window).std(ddof=0)
        factors[f"parkinson_vol_{window}"] = rolling_sqrt_mean_nonnegative(parkinson_bar, window)
        factors[f"garman_klass_vol_{window}"] = rolling_sqrt_mean_nonnegative(garman_klass_bar, window)
        factors[f"rogers_satchell_vol_{window}"] = rolling_sqrt_mean_nonnegative(rogers_satchell_bar, window)
        factors[f"drawdown_from_high_{window}"] = close / high_max.replace(0, np.nan) - 1
        factors[f"rolling_sharpe_{window}"] = safe_divide(rolling_ret.mean(), ret_std)
        factors[f"amihud_illiq_{window}"] = amihud_bar.rolling(window, min_periods=window).mean()
        factors[f"sma_gap_{window}"] = close / sma.replace(0, np.nan) - 1
        factors[f"ema_gap_{window}"] = close / ema.replace(0, np.nan) - 1
        factors[f"price_rank_{window}"] = close.rolling(window, min_periods=window).rank(pct=True)
        factors[f"dist_high_{window}"] = close / high_max.replace(0, np.nan) - 1
        factors[f"dist_low_{window}"] = close / low_min.replace(0, np.nan) - 1
        factors[f"breakout_pos_{window}"] = safe_divide(close - low_min, high_max - low_min)
        factors[f"volume_ratio_{window}"] = volume / rolling_volume.mean().replace(0, np.nan)
        factors[f"amount_ratio_{window}"] = amount / rolling_amount.mean().replace(0, np.nan)
        factors[f"volume_change_{window}"] = volume / volume.shift(window).replace(0, np.nan) - 1
        factors[f"amount_change_{window}"] = amount / amount.shift(window).replace(0, np.nan) - 1
        factors[f"volume_z_{window}"] = rolling_zscore(volume, window)
        factors[f"amount_z_{window}"] = rolling_zscore(amount, window)
        factors[f"money_flow_imbalance_{window}"] = safe_divide(
            signed_amount.rolling(window, min_periods=window).sum(),
            amount.abs().rolling(window, min_periods=window).sum(),
        )
        factors[f"ret_volume_corr_{window}"] = ret_1.rolling(window, min_periods=window).corr(volume.pct_change())
        factors[f"ret_amount_corr_{window}"] = ret_1.rolling(window, min_periods=window).corr(amount.pct_change())
        factors[f"range_mean_{window}"] = factors["hl_range"].rolling(window, min_periods=window).mean()
        factors[f"range_std_{window}"] = factors["hl_range"].rolling(window, min_periods=window).std(ddof=0)
        factors[f"vwap_gap_mean_{window}"] = factors["vwap_gap"].rolling(window, min_periods=window).mean()
        factors[f"vol_change_{window}"] = ret_std / ret_std.shift(window).replace(0, np.nan) - 1
        factors[f"trend_slope_{window}"] = trend_slope
        factors[f"trend_r2_{window}"] = trend_r2

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

    dt = pd.to_datetime(frame["datetime"])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    factors["minute_sin"] = np.sin(2 * math.pi * minute_of_day / (24 * 60))
    factors["minute_cos"] = np.cos(2 * math.pi * minute_of_day / (24 * 60))

    return pd.DataFrame(factors, index=frame.index).replace([np.inf, -np.inf], np.nan)


def build_cross_sectional_factor_library(
    factor_frame: pd.DataFrame,
    factor_cols: Sequence[str] | None = None,
    *,
    include_rank: bool = True,
    include_zscore: bool = True,
    include_winsorized_zscore: bool = False,
    winsorize_sigma: float = 3.0,
) -> pd.DataFrame:
    """Build same-timestamp cross-sectional rank/z-score factors.

    Parameters
    ----------
    factor_frame:
        Multi-symbol frame with ``datetime`` and numeric factor columns. Each
        cross-sectional transform groups only by the same ``datetime`` and never
        reads future dates.
    factor_cols:
        Candidate factor columns. When omitted, all numeric non-OHLCV columns
        except labels are used.

    Returns
    -------
    pd.DataFrame
        Cross-sectional features aligned to ``factor_frame.index``. These are
        safe for t+1 trading when the raw same-day features are known at t.
    """

    if "datetime" not in factor_frame.columns:
        raise ValueError("factor_frame must contain a datetime column for cross-sectional transforms.")
    excluded = {"open", "high", "low", "close", "volume", "amount"}
    excluded.update(col for col in factor_frame.columns if col.startswith(("future_return", "forward_return", "label")))
    if factor_cols is None:
        factor_cols = [
            col
            for col in factor_frame.select_dtypes(include=[np.number]).columns
            if col not in excluded
        ]
    numeric = factor_frame.loc[:, list(factor_cols)].replace([np.inf, -np.inf], np.nan)
    grouped = numeric.groupby(factor_frame["datetime"], sort=False)
    outputs: list[pd.DataFrame] = []
    if include_rank:
        rank = grouped.rank(method="average", pct=True)
        rank.columns = [f"cs_rank_{col}" for col in rank.columns]
        outputs.append(rank)
    if include_zscore or include_winsorized_zscore:
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        if include_zscore:
            zscore = (numeric - mean) / std
            zscore.columns = [f"cs_zscore_{col}" for col in zscore.columns]
            outputs.append(zscore)
        if include_winsorized_zscore:
            lower = mean - winsorize_sigma * std
            upper = mean + winsorize_sigma * std
            clipped = numeric.clip(lower=lower, upper=upper, axis=1)
            winsorized = (clipped - mean) / std
            winsorized.columns = [f"cs_winz_{col}" for col in winsorized.columns]
            outputs.append(winsorized)
    if not outputs:
        return pd.DataFrame(index=factor_frame.index)
    return pd.concat(outputs, axis=1).replace([np.inf, -np.inf], np.nan)


def validate_factor_frame(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    label_cols: Sequence[str] = (),
    require_sorted: bool = True,
) -> pd.DataFrame:
    """Run structural leakage and data-integrity checks for feature columns."""

    checks: list[dict[str, object]] = []
    columns = set(frame.columns)
    forbidden_patterns = ("future_return", "forward_return", "target", "label")
    forbidden_features = [
        col for col in feature_cols if col in label_cols or any(pattern in col.lower() for pattern in forbidden_patterns)
    ]
    checks.append(
        {
            "check": "feature_names_exclude_labels",
            "passed": not forbidden_features,
            "detail": ",".join(forbidden_features[:20]),
        }
    )
    missing_features = [col for col in feature_cols if col not in columns]
    checks.append(
        {
            "check": "feature_columns_exist",
            "passed": not missing_features,
            "detail": ",".join(missing_features[:20]),
        }
    )
    duplicate_keys = False
    if {"symbol", "datetime"}.issubset(columns):
        duplicate_keys = bool(frame.duplicated(["symbol", "datetime"]).any())
    checks.append({"check": "no_duplicate_symbol_datetime", "passed": not duplicate_keys, "detail": ""})
    unsupported_inf = False
    existing_features = [col for col in feature_cols if col in columns]
    if existing_features:
        unsupported_inf = bool(np.isinf(frame[existing_features].select_dtypes(include=[np.number])).any().any())
    checks.append({"check": "no_infinite_feature_values", "passed": not unsupported_inf, "detail": ""})
    sorted_ok = True
    if require_sorted and {"symbol", "datetime"}.issubset(columns):
        sorted_index = frame.sort_values(["symbol", "datetime"]).index
        sorted_ok = bool(frame.index.equals(sorted_index))
    checks.append({"check": "sorted_by_symbol_datetime", "passed": sorted_ok, "detail": ""})
    return pd.DataFrame(checks)


def validate_no_future_leakage(
    symbol_data: pd.DataFrame,
    build_function: Callable[[pd.DataFrame], pd.DataFrame] = build_factor_library,
    *,
    cutoff_fraction: float = 0.6,
) -> dict[str, object]:
    """Perturb future OHLCV rows and verify earlier features do not change.

    This dynamic check catches common accidental future-window usage in a
    per-symbol feature builder. It mutates rows strictly after a cutoff and then
    compares generated feature values up to the cutoff.
    """

    frame = symbol_data.sort_values("datetime").reset_index(drop=True)
    if len(frame) < 20:
        return {"check": "future_perturbation", "passed": False, "detail": "not_enough_rows"}
    cutoff = max(5, min(len(frame) - 6, int(len(frame) * cutoff_fraction)))
    base = build_function(frame).iloc[: cutoff + 1].copy()
    mutated = frame.copy()
    future_index = mutated.index > cutoff
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in mutated.columns:
            mutated[column] = mutated[column].astype(float)
    for column in ["open", "high", "low", "close"]:
        if column in mutated.columns:
            mutated.loc[future_index, column] = mutated.loc[future_index, column] * 1.123
    for column in ["volume", "amount"]:
        if column in mutated.columns:
            mutated.loc[future_index, column] = mutated.loc[future_index, column] * 0.777
    changed = build_function(mutated).iloc[: cutoff + 1].reindex(columns=base.columns)
    diff = (base - changed).abs()
    bad_cols = diff.columns[(diff > 1e-10).any(axis=0)].tolist()
    return {
        "check": "future_perturbation",
        "passed": not bad_cols,
        "detail": ",".join(bad_cols[:20]),
        "cutoff_row": int(cutoff),
        "checked_feature_count": int(len(base.columns)),
    }


def infer_factor_metadata(factor_names: Sequence[str]) -> pd.DataFrame:
    """Infer factor categories, windows, source inspiration, and alignment notes."""

    rows: list[dict[str, object]] = []
    for name in factor_names:
        rows.append(_infer_one_factor_metadata(str(name)))
    return pd.DataFrame(rows)


def _infer_one_factor_metadata(name: str) -> dict[str, object]:
    """Infer metadata for one factor name from naming conventions."""

    base_name = re.sub(r"^cs_(rank|zscore|winz)_", "", name)
    numbers = re.findall(r"\d+", base_name)
    window = ",".join(numbers) if numbers else ""
    category = "technical_indicator"
    source = "internal_ohlcv_engineering"
    uses_external = False
    if name.startswith("cs_rank_"):
        category = "cross_sectional_rank"
    elif name.startswith(("cs_zscore_", "cs_winz_")):
        category = "cross_sectional_zscore"
    elif base_name.startswith(("momentum", "weighted_momentum", "roc")):
        category = "momentum"
        source = "Jegadeesh_Titman_1993_momentum"
        uses_external = True
    elif base_name.startswith(("reversal",)):
        category = "reversal"
    elif base_name.startswith(("ret_std", "downside_vol", "parkinson_vol", "garman_klass_vol", "rogers_satchell_vol", "range_", "atr_ratio", "vol_change", "drawdown", "boll_width")):
        category = "volatility_risk"
        source = "range_and_low_volatility_literature"
        uses_external = True
    elif base_name.startswith(("volume", "amount", "amihud", "money_flow", "obv", "mfi")):
        category = "volume_liquidity"
        source = "Amihud_2002_or_price_volume_liquidity"
        uses_external = base_name.startswith("amihud")
    elif base_name.startswith(("ret_volume_corr", "ret_amount_corr", "vwap", "oc_ret", "hl_range", "upper_shadow", "lower_shadow", "close_pos")):
        category = "price_volume_or_bar_shape"
    elif base_name.startswith(("sma_gap", "ema_gap", "ma_cross", "trend_", "breakout", "price_rank", "dist_high", "dist_low")):
        category = "trend_cta"
    elif base_name.startswith(("rsi", "cci", "willr", "adx", "macd", "stoch")):
        category = "technical_indicator"
    elif base_name.startswith("ret_skew"):
        category = "higher_moment"
    elif base_name.startswith("minute_"):
        category = "intraday_time"
    return {
        "factor": name,
        "base_factor": base_name,
        "category": category,
        "calculation_window": window,
        "requires_shift_for_trading": True,
        "minimum_execution_lag_bars": 1,
        "uses_external_inspiration": uses_external,
        "source_or_inspiration": source,
        "available_fields": "datetime,open,high,low,close,volume,amount",
        "alignment_note": "Uses current and historical completed bars only; use at earliest on the next bar/day.",
    }
