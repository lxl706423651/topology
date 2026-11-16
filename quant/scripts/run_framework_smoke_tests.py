"""Run the backtest framework on cleaned real minute-data slices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATA_DIR = PROJECT_ROOT / "ashare_minute_data_2023_2026_04"

from quant_backtest import BacktestConfig, BacktestEngine
from quant_backtest.data import load_ohlcv
from quant_backtest.reporting import print_metrics, save_report
from strategies.example_bollinger_strategy import BollingerBreakoutStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "real_data_smoke_bollinger")
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=200_000)
    return parser.parse_args()


def choose_input(data_dir: Path) -> Path:
    files = sorted((data_dir / "processed" / "pkl").rglob("*.pkl"))
    if not files:
        raise FileNotFoundError("No processed pkl found. Run preprocess_stock_minute_data.py first.")
    return files[0]


def main() -> None:
    args = parse_args()
    input_path = args.input or choose_input(args.data_dir)
    data = load_ohlcv(input_path)
    data["symbol"] = data["symbol"].astype(str)
    symbols = data["symbol"].drop_duplicates().head(args.max_symbols).tolist()
    data = data.loc[data["symbol"].isin(symbols)].sort_values(["symbol", "datetime"])
    if args.max_rows and len(data) > args.max_rows:
        data = data.iloc[: args.max_rows].copy()

    strategy = BollingerBreakoutStrategy(window=20, num_std=2.0, trend_window=60)
    config = BacktestConfig(
        initial_cash=1_000_000,
        commission_rate=0.0002,
        slippage_bps=1.0,
        return_mode="percent",
        signal_lag=1,
        portfolio_weighting="equal",
    )
    result = BacktestEngine(config).run(data, strategy)
    args.output.mkdir(parents=True, exist_ok=True)
    save_report(result, args.output)
    pd.DataFrame(
        [
            {
                "input": str(input_path),
                "rows": len(data),
                "symbols": len(symbols),
                "start": str(data["datetime"].min()),
                "end": str(data["datetime"].max()),
            }
        ]
    ).to_csv(args.output / "run_context.csv", index=False, encoding="utf-8-sig")
    print_metrics(result.metrics)
    print(f"Saved smoke-test report to {args.output}")


if __name__ == "__main__":
    main()
