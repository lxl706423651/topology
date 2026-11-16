"""Notebook-style Bollinger strategy example."""

from __future__ import annotations

import pandas as pd

from quant_backtest.indicators import bollinger_bands, sma
from quant_backtest.strategy import BaseStrategy


class BollingerBreakoutStrategy(BaseStrategy):
    name = "bollinger_breakout"

    def __init__(self, window: int = 20, num_std: float = 2.0, trend_window: int = 60) -> None:
        self.window = window
        self.num_std = num_std
        self.trend_window = trend_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        upper, middle, lower = bollinger_bands(close, self.window, self.num_std)
        trend = sma(close, self.trend_window)

        open_signal = ((close.shift(1) <= upper.shift(1)) & (close > upper) & (close > trend)).astype(float)
        open_signal -= ((close.shift(1) >= lower.shift(1)) & (close < lower) & (close < trend)).astype(float)
        close_signal = (((close - middle) * (close.shift(1) - middle.shift(1))) <= 0).astype(float)

        return pd.DataFrame(
            {
                "open_signal": open_signal.fillna(0),
                "close_signal": close_signal.fillna(0),
                "boll_upper": upper,
                "boll_middle": middle,
                "boll_lower": lower,
            },
            index=data.index,
        )
