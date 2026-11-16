"""Report helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .engine import BacktestResult


def print_metrics(metrics: dict) -> None:
    preferred = [
        "strategy",
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "trade_count",
        "trade_win_rate",
        "profit_factor",
        "avg_holding_bars",
        "avg_abs_position",
        "avg_turnover",
        "total_turnover",
    ]
    for key in preferred:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


def save_report(result: BacktestResult, output_dir: str | Path) -> None:
    path = Path(output_dir)
    result.save(path)
    plot_equity(result.equity, path / "equity_curve.png")


def plot_equity(equity: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    equity["equity"].plot(ax=ax, linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
