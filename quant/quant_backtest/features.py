"""Feature engineering for minute-level OHLCV stock data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import talib
except Exception:  # pragma: no cover
    talib = None


DEFAULT_WINDOWS = (5, 10, 20, 30, 60, 120)
DEFAULT_LABEL_HORIZONS = (1, 5, 24)


def _safe_divide(numerator: pd.Series, denominator: pd.Series | np.ndarray | float) -> pd.Series:
    denom = pd.Series(denominator, index=numerator.index) if not isinstance(denominator, pd.Series) else denominator
    return numerator / denom.replace(0, np.nan)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return _safe_divide(series - mean, std)


def _rank01(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).rank(pct=True)


def _talib_or_nan(name: str, index: pd.Index, *args: np.ndarray, **kwargs) -> pd.Series:
    if talib is None:
        return pd.Series(np.nan, index=index)
    func = getattr(talib, name)
    values = func(*args, **kwargs)
    if isinstance(values, tuple):
        raise TypeError(f"{name} returns multiple arrays.")
    return pd.Series(values, index=index)


def build_minute_features(
    data: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    label_horizons: tuple[int, ...] = DEFAULT_LABEL_HORIZONS,
    include_labels: bool = True,
) -> pd.DataFrame:
    """Build 100-200 representative features for one symbol.

    The function only uses current and historical bars for features. Forward
    returns are created separately as labels.
    """

    frame = data.sort_values("datetime").copy()
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float).fillna(0)
    amount = frame["amount"].astype(float) if "amount" in frame.columns else close * volume * 100

    out = frame[["datetime", "symbol", "open", "high", "low", "close", "volume"]].copy()
    if "amount" in frame.columns:
        out["amount"] = amount

    log_close = np.log(close.replace(0, np.nan))
    ret_1 = close.pct_change()
    hl_range = _safe_divide(high - low, close)
    body = _safe_divide(close - open_, open_)
    upper_shadow = _safe_divide(high - pd.concat([open_, close], axis=1).max(axis=1), close)
    lower_shadow = _safe_divide(pd.concat([open_, close], axis=1).min(axis=1) - low, close)
    close_pos = _safe_divide(close - low, high - low)
    vwap_proxy = _safe_divide(amount, volume * 100).fillna((high + low + close) / 3)

    out["ret_1"] = ret_1
    out["logret_1"] = log_close.diff()
    out["hl_range"] = hl_range
    out["oc_ret"] = body
    out["upper_shadow"] = upper_shadow
    out["lower_shadow"] = lower_shadow
    out["close_pos_in_bar"] = close_pos
    out["vwap_gap"] = close / vwap_proxy.replace(0, np.nan) - 1
    out["amount_per_volume"] = vwap_proxy

    for window in windows:
        roll_ret = ret_1.rolling(window, min_periods=window)
        roll_close = close.rolling(window, min_periods=window)
        roll_volume = volume.rolling(window, min_periods=window)
        roll_amount = amount.rolling(window, min_periods=window)
        sma = roll_close.mean()
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        high_max = high.rolling(window, min_periods=window).max()
        low_min = low.rolling(window, min_periods=window).min()
        vol_mean = roll_volume.mean()
        amount_mean = roll_amount.mean()

        out[f"ret_{window}"] = close.pct_change(window)
        out[f"logret_{window}"] = log_close.diff(window)
        out[f"ret_mean_{window}"] = roll_ret.mean()
        out[f"ret_std_{window}"] = roll_ret.std(ddof=0)
        out[f"ret_skew_{window}"] = roll_ret.skew()
        out[f"ret_min_{window}"] = roll_ret.min()
        out[f"ret_max_{window}"] = roll_ret.max()
        out[f"sma_gap_{window}"] = close / sma - 1
        out[f"ema_gap_{window}"] = close / ema - 1
        out[f"price_rank_{window}"] = _rank01(close, window)
        out[f"dist_high_{window}"] = close / high_max.replace(0, np.nan) - 1
        out[f"dist_low_{window}"] = close / low_min.replace(0, np.nan) - 1
        out[f"range_mean_{window}"] = hl_range.rolling(window, min_periods=window).mean()
        out[f"range_std_{window}"] = hl_range.rolling(window, min_periods=window).std(ddof=0)
        out[f"volume_ratio_{window}"] = volume / vol_mean.replace(0, np.nan)
        out[f"volume_z_{window}"] = _zscore(volume, window)
        out[f"amount_ratio_{window}"] = amount / amount_mean.replace(0, np.nan)
        out[f"amount_z_{window}"] = _zscore(amount, window)
        out[f"vwap_gap_mean_{window}"] = out["vwap_gap"].rolling(window, min_periods=window).mean()

    for short, long in ((3, 10), (5, 20), (10, 60), (20, 120), (60, 240)):
        short_ma = close.rolling(short, min_periods=short).mean()
        long_ma = close.rolling(long, min_periods=long).mean()
        out[f"ma_cross_{short}_{long}"] = short_ma / long_ma.replace(0, np.nan) - 1

    for window in (10, 20, 60):
        middle = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=0)
        upper = middle + 2 * std
        lower = middle - 2 * std
        out[f"boll_width_{window}"] = (upper - lower) / middle.replace(0, np.nan)
        out[f"boll_pctb_{window}"] = (close - lower) / (upper - lower).replace(0, np.nan)
        out[f"boll_upper_gap_{window}"] = close / upper.replace(0, np.nan) - 1
        out[f"boll_lower_gap_{window}"] = close / lower.replace(0, np.nan) - 1

    high_np = high.to_numpy(dtype=float)
    low_np = low.to_numpy(dtype=float)
    close_np = close.to_numpy(dtype=float)
    volume_np = volume.to_numpy(dtype=float)
    index = frame.index

    for window in (6, 12, 14, 24, 48):
        if talib is not None:
            out[f"rsi_{window}"] = pd.Series(talib.RSI(close_np, timeperiod=window), index=index)
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
            out[f"rsi_{window}"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        out[f"roc_{window}"] = close.pct_change(window)
        out[f"atr_ratio_{window}"] = _talib_or_nan("ATR", index, high_np, low_np, close_np, timeperiod=window) / close
        out[f"cci_{window}"] = _talib_or_nan("CCI", index, high_np, low_np, close_np, timeperiod=window)
        out[f"willr_{window}"] = _talib_or_nan("WILLR", index, high_np, low_np, close_np, timeperiod=window)
        out[f"mfi_{window}"] = _talib_or_nan("MFI", index, high_np, low_np, close_np, volume_np, timeperiod=window)

    for window in (14, 24, 48):
        out[f"adx_{window}"] = _talib_or_nan("ADX", index, high_np, low_np, close_np, timeperiod=window)
        out[f"adxr_{window}"] = _talib_or_nan("ADXR", index, high_np, low_np, close_np, timeperiod=window)

    for fast, slow, signal in ((6, 13, 5), (12, 26, 9), (24, 52, 18)):
        if talib is not None:
            dif, dea, hist = talib.MACD(close_np, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            out[f"macd_dif_{fast}_{slow}_{signal}"] = dif / close.replace(0, np.nan)
            out[f"macd_dea_{fast}_{slow}_{signal}"] = dea / close.replace(0, np.nan)
            out[f"macd_hist_{fast}_{slow}_{signal}"] = hist / close.replace(0, np.nan)
        else:
            dif = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - close.ewm(
                span=slow,
                adjust=False,
                min_periods=slow,
            ).mean()
            dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
            hist = dif - dea
            out[f"macd_dif_{fast}_{slow}_{signal}"] = dif / close.replace(0, np.nan)
            out[f"macd_dea_{fast}_{slow}_{signal}"] = dea / close.replace(0, np.nan)
            out[f"macd_hist_{fast}_{slow}_{signal}"] = hist / close.replace(0, np.nan)

    if talib is not None:
        slowk, slowd = talib.STOCH(high_np, low_np, close_np)
        out["stoch_slowk"] = slowk
        out["stoch_slowd"] = slowd
        fastk, fastd = talib.STOCHF(high_np, low_np, close_np)
        out["stoch_fastk"] = fastk
        out["stoch_fastd"] = fastd
        out["obv"] = talib.OBV(close_np, volume_np)
    else:
        out["stoch_slowk"] = 100 * (close - low.rolling(14).min()) / (
            high.rolling(14).max() - low.rolling(14).min()
        ).replace(0, np.nan)
        out["stoch_slowd"] = out["stoch_slowk"].rolling(3, min_periods=3).mean()
        out["stoch_fastk"] = out["stoch_slowk"]
        out["stoch_fastd"] = out["stoch_slowd"]
        out["obv"] = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    out["obv_z_60"] = _zscore(out["obv"], 60)
    out["obv_z_240"] = _zscore(out["obv"], 240)

    dt = pd.to_datetime(out["datetime"])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    out["minute_sin"] = np.sin(2 * math.pi * minute_of_day / (24 * 60))
    out["minute_cos"] = np.cos(2 * math.pi * minute_of_day / (24 * 60))
    out["weekday_sin"] = np.sin(2 * math.pi * dt.dt.weekday / 5)
    out["weekday_cos"] = np.cos(2 * math.pi * dt.dt.weekday / 5)
    out["is_morning"] = (dt.dt.hour < 12).astype("int8")

    if include_labels:
        for horizon in label_horizons:
            out[f"future_return_{horizon}"] = close.shift(-horizon) / close.replace(0, np.nan) - 1
        if 24 in label_horizons:
            out["Return_24"] = out["future_return_24"]

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in numeric_cols if c not in {"volume"}]
    for col in feature_cols:
        if out[col].dtype == "float64":
            out[col] = out[col].astype("float32")
    out["volume"] = out["volume"].astype("float32")
    if "amount" in out.columns:
        out["amount"] = out["amount"].astype("float32")
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding identifiers, prices and labels."""

    excluded = {
        "datetime",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "Return_24",
        "target_rank",
    }
    excluded.update(c for c in frame.columns if c.startswith("future_return_"))
    return [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
