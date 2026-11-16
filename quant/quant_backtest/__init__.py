"""Reusable backtesting framework extracted from the research notebooks."""

from .engine import BacktestConfig, BacktestEngine, BacktestResult
from .strategy import BaseStrategy, FunctionSignalStrategy, ModelSignalStrategy

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BaseStrategy",
    "FunctionSignalStrategy",
    "ModelSignalStrategy",
]
