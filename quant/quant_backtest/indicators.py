"""Technical indicators with optional TA-Lib acceleration.

The framework does not require TA-Lib. If the package is installed, callers can
still import and use it directly; these functions keep the backtest runnable on
machines where TA-Lib wheels are unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - depends on local optional package
    import talib as _talib
except Exception:  # pragma: no cover - TA-Lib is optional
    _talib = None


def talib_available() -> bool:
    return _talib is not None


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(series, window)
    width = rolling_std(series, window) * num_std
    upper = middle + width
    lower = middle - width
    return upper, middle, lower


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if _talib is not None:
        values = _talib.RSI(series.astype(float).to_numpy(), timeperiod=window)
        return pd.Series(values, index=series.index, name=f"rsi_{window}")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50).rename(f"rsi_{window}")


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if _talib is not None:
        dif, dea, hist = _talib.MACD(
            series.astype(float).to_numpy(),
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal,
        )
        return (
            pd.Series(dif, index=series.index, name="macd_dif"),
            pd.Series(dea, index=series.index, name="macd_dea"),
            pd.Series(hist, index=series.index, name="macd_hist"),
        )

    dif = ema(series, fast) - ema(series, slow)
    dea = ema(dif, signal)
    hist = dif - dea
    return dif.rename("macd_dif"), dea.rename("macd_dea"), hist.rename("macd_hist")


def true_range(data: pd.DataFrame) -> pd.Series:
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    if _talib is not None:
        values = _talib.ATR(
            data["high"].astype(float).to_numpy(),
            data["low"].astype(float).to_numpy(),
            data["close"].astype(float).to_numpy(),
            timeperiod=window,
        )
        return pd.Series(values, index=data.index, name=f"atr_{window}")
    return true_range(data).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = sma(series, window)
    std = rolling_std(series, window)
    return ((series - mean) / std.replace(0, np.nan)).rename(f"zscore_{window}")
