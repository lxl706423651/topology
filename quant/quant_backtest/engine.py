"""Vectorized backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .data import standardize_ohlcv
from .metrics import performance_metrics
from .strategy import BaseStrategy


ReturnMode = Literal["percent", "points"]


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0002
    slippage_bps: float = 1.0
    contract_multiplier: float = 1.0
    fixed_cost_per_trade: float = 0.0
    max_position: float = 1.0
    return_mode: ReturnMode = "percent"
    annualization: float | None = None
    risk_free_rate: float = 0.0
    signal_lag: int = 1
    portfolio_weighting: Literal["equal", "sum"] = "equal"


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    bars: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(path / "equity_curve.csv", index=True)
        self.bars.to_csv(path / "bars_with_signals.csv", index=False)
        self.trades.to_csv(path / "trades.csv", index=False)
        pd.Series(self.metrics).to_csv(path / "metrics.csv", header=["value"])


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        frame = standardize_ohlcv(data)
        bar_parts = []
        trade_parts = []
        for symbol, group in frame.groupby("symbol", sort=False):
            bars = self._run_symbol(group.reset_index(drop=True), strategy)
            bars["symbol"] = symbol
            bar_parts.append(bars)
            trades = self._extract_trades(bars, symbol)
            if not trades.empty:
                trade_parts.append(trades)

        bars_all = pd.concat(bar_parts, ignore_index=True)
        trades_all = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
        equity = self._build_portfolio_equity(bars_all)
        metrics = performance_metrics(
            equity["equity"],
            equity["return"],
            trades_all,
            annualization=self.config.annualization,
            risk_free_rate=self.config.risk_free_rate,
        )
        metrics.update(self._portfolio_diagnostics(bars_all))
        metrics["strategy"] = getattr(strategy, "name", strategy.__class__.__name__)
        metrics["return_mode"] = self.config.return_mode
        return BacktestResult(equity=equity, bars=bars_all, trades=trades_all, metrics=metrics)

    def _run_symbol(self, data: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
        bars = data.copy()
        signals = strategy.generate_signals(bars.copy())
        signals = signals.reindex(bars.index).fillna(0)
        bars = pd.concat([bars, signals], axis=1)

        if "target_position" in bars:
            desired = pd.to_numeric(bars["target_position"], errors="coerce").fillna(0)
            raw_position = desired.clip(-self.config.max_position, self.config.max_position)
        elif {"open_signal", "close_signal"}.issubset(bars.columns):
            raw_position = self._signals_to_position(bars["open_signal"], bars["close_signal"])
        else:
            raise ValueError("Strategy must return target_position or open_signal/close_signal.")

        position = raw_position.shift(self.config.signal_lag).fillna(0)
        position = position.clip(-self.config.max_position, self.config.max_position)
        bars["raw_position"] = raw_position
        bars["position"] = position
        bars["turnover"] = (position - position.shift(1).fillna(0)).abs()

        close = bars["close"].astype(float)
        if self.config.return_mode == "points":
            pnl = position.shift(1).fillna(0) * close.diff().fillna(0) * self.config.contract_multiplier
            cost = self._point_cost(bars)
            bars["pnl"] = pnl - cost
            bars["strategy_return"] = bars["pnl"] / self.config.initial_cash
        else:
            underlying_return = close.pct_change().fillna(0)
            cost_return = bars["turnover"] * (self.config.commission_rate + self.config.slippage_bps / 10_000)
            fixed_return = bars["turnover"] * self.config.fixed_cost_per_trade / self.config.initial_cash
            bars["pnl"] = (position.shift(1).fillna(0) * underlying_return - cost_return - fixed_return) * self.config.initial_cash
            bars["strategy_return"] = bars["pnl"] / self.config.initial_cash
        return bars

    def _signals_to_position(self, open_signal: pd.Series, close_signal: pd.Series) -> pd.Series:
        open_signal = pd.to_numeric(open_signal, errors="coerce").fillna(0).clip(
            -self.config.max_position,
            self.config.max_position,
        )
        close_signal = pd.to_numeric(close_signal, errors="coerce").fillna(0)
        status = open_signal.replace(0, np.nan)
        status[(close_signal == 1) & (open_signal == 0)] = 0
        status = status.ffill().fillna(0).astype(float)
        status[open_signal == status.shift(1)] = status.shift(1)
        return status.clip(-self.config.max_position, self.config.max_position)

    def _point_cost(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        notional = close * self.config.contract_multiplier
        variable = bars["turnover"] * notional * (self.config.commission_rate + self.config.slippage_bps / 10_000)
        fixed = bars["turnover"] * self.config.fixed_cost_per_trade
        return variable + fixed

    def _build_portfolio_equity(self, bars: pd.DataFrame) -> pd.DataFrame:
        by_time = bars.pivot_table(index="datetime", columns="symbol", values="strategy_return", aggfunc="sum").fillna(0)
        if self.config.return_mode == "points" or self.config.portfolio_weighting == "sum":
            returns = by_time.sum(axis=1)
        else:
            positions = bars.pivot_table(index="datetime", columns="symbol", values="position", aggfunc="last").fillna(0)
            active_count = positions.abs().gt(0).sum(axis=1).replace(0, 1)
            returns = by_time.sum(axis=1) / active_count
        equity = self.config.initial_cash * (1 + returns).cumprod()
        return pd.DataFrame({"return": returns, "equity": equity})

    def _portfolio_diagnostics(self, bars: pd.DataFrame) -> dict[str, float]:
        position = bars["position"].astype(float)
        turnover = bars["turnover"].astype(float)
        return {
            "avg_abs_position": float(position.abs().mean()) if len(position) else 0.0,
            "long_bar_rate": float((position > 0).mean()) if len(position) else 0.0,
            "short_bar_rate": float((position < 0).mean()) if len(position) else 0.0,
            "flat_bar_rate": float((position == 0).mean()) if len(position) else 0.0,
            "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
            "total_turnover": float(turnover.sum()) if len(turnover) else 0.0,
        }

    def _extract_trades(self, bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
        rows = []
        position = bars["position"].astype(float).to_numpy()
        close = bars["close"].astype(float).to_numpy()
        datetimes = pd.to_datetime(bars["datetime"]).to_numpy()
        current_side = 0.0
        entry_price = 0.0
        entry_idx = -1

        for idx, side in enumerate(position):
            if side == current_side:
                continue
            if current_side != 0:
                pnl = self._trade_pnl(current_side, entry_price, close[idx])
                rows.append(
                    {
                        "symbol": symbol,
                        "entry_time": pd.Timestamp(datetimes[entry_idx]),
                        "exit_time": pd.Timestamp(datetimes[idx]),
                        "side": "long" if current_side > 0 else "short",
                        "entry_price": entry_price,
                        "exit_price": close[idx],
                        "pnl": pnl,
                        "return": pnl / self.config.initial_cash,
                        "holding_bars": idx - entry_idx,
                    }
                )
            if side != 0:
                current_side = side
                entry_price = close[idx]
                entry_idx = idx
            else:
                current_side = 0.0
                entry_price = 0.0
                entry_idx = -1

        if current_side != 0 and entry_idx >= 0:
            idx = len(bars) - 1
            pnl = self._trade_pnl(current_side, entry_price, close[idx])
            rows.append(
                {
                    "symbol": symbol,
                    "entry_time": pd.Timestamp(datetimes[entry_idx]),
                    "exit_time": pd.Timestamp(datetimes[idx]),
                    "side": "long" if current_side > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": close[idx],
                    "pnl": pnl,
                    "return": pnl / self.config.initial_cash,
                    "holding_bars": idx - entry_idx,
                }
            )
        return pd.DataFrame(rows)

    def _trade_pnl(self, side: float, entry_price: float, exit_price: float) -> float:
        if self.config.return_mode == "points":
            return float(side * (exit_price - entry_price) * self.config.contract_multiplier)
        return float(side * (exit_price / entry_price - 1) * self.config.initial_cash)
