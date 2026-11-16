"""Edit this file to run your own strategy.

Usage:
    python run_strategy.py
    python run_strategy.py --data your_ohlcv.csv --output reports/my_run

Your CSV can contain either standard OHLCV columns or notebook-style columns
such as ``underlying_symbol`` and ``datetime``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_backtest import BacktestConfig, BacktestEngine
from quant_backtest.data import load_ohlcv
from quant_backtest.reporting import print_metrics, save_report
from strategies.example_bollinger_strategy import BollingerBreakoutStrategy


def build_strategy():
    """Replace this function with your own strategy or model strategy."""

    return BollingerBreakoutStrategy(window=20, num_std=2.0, trend_window=60)


def demo_data(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dt = pd.date_range("2022-01-01", periods=rows, freq="D")
    noise = rng.normal(0, 0.012, rows)
    close = 100 * np.exp(np.cumsum(noise + 0.0002))
    open_ = close * (1 + rng.normal(0, 0.002, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, rows))
    volume = rng.integers(1000, 5000, rows)
    return pd.DataFrame(
        {
            "datetime": dt,
            "symbol": "DEMO",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="", help="CSV/PKL/Parquet path. Empty means run built-in demo data.")
    parser.add_argument("--output", type=str, default="reports/latest", help="Output directory.")
    parser.add_argument("--return-mode", choices=["percent", "points"], default="percent")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--commission-rate", type=float, default=0.0002)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--contract-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_ohlcv(args.data) if args.data else demo_data()
    strategy = build_strategy()
    config = BacktestConfig(
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        contract_multiplier=args.contract_multiplier,
        return_mode=args.return_mode,
    )
    result = BacktestEngine(config).run(data, strategy)
    save_report(result, Path(args.output))
    print_metrics(result.metrics)
    print(f"Report saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
