"""Performance and trade metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def infer_annualization(index: pd.Index, default: int = 252) -> float:
    dt_index = pd.DatetimeIndex(index)
    if len(dt_index) < 3:
        return float(default)
    seconds = dt_index.to_series().diff().dropna().dt.total_seconds()
    seconds = seconds[seconds > 0]
    if seconds.empty:
        return float(default)
    median_seconds = float(seconds.median())
    day = 24 * 60 * 60
    if median_seconds >= day * 0.8:
        return 252.0
    return min((252 * day) / median_seconds, 252 * 24 * 60)


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if equity.empty:
        return 0.0, None, None
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    end = drawdown.idxmin()
    start = equity.loc[:end].idxmax()
    return float(drawdown.loc[end]), start, end


def drawdown_duration(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    running_max = equity.cummax()
    underwater = equity < running_max
    max_duration = duration = 0
    for flag in underwater:
        duration = duration + 1 if flag else 0
        max_duration = max(max_duration, duration)
    return int(max_duration)


def performance_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    annualization: float | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | str | None]:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    equity = equity.dropna()
    if equity.empty:
        return {}

    annualization = float(annualization or infer_annualization(equity.index))
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = max(len(returns) / annualization, 1 / annualization)
    annual_return = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1.0

    excess = returns - risk_free_rate / annualization
    volatility = float(returns.std(ddof=0) * math.sqrt(annualization))
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * math.sqrt(annualization)) if len(downside) else 0.0
    sharpe = float(excess.mean() / returns.std(ddof=0) * math.sqrt(annualization)) if returns.std(ddof=0) else 0.0
    sortino = float(excess.mean() / downside.std(ddof=0) * math.sqrt(annualization)) if len(downside) and downside.std(ddof=0) else 0.0
    mdd, mdd_start, mdd_end = max_drawdown(equity)
    calmar = float(annual_return / abs(mdd)) if mdd else 0.0

    var_95 = float(returns.quantile(0.05)) if len(returns) else 0.0
    cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) else 0.0

    out: dict[str, float | int | str | None] = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_drawdown_start": str(mdd_start) if mdd_start is not None else None,
        "max_drawdown_end": str(mdd_end) if mdd_end is not None else None,
        "max_drawdown_duration_bars": drawdown_duration(equity),
        "calmar": calmar,
        "win_bar_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skew": float(returns.skew()) if len(returns) > 2 else 0.0,
        "kurtosis": float(returns.kurt()) if len(returns) > 3 else 0.0,
        "annualization": annualization,
    }

    if trades is not None and not trades.empty:
        pnl = trades["pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        out.update(
            {
                "trade_count": int(len(trades)),
                "trade_win_rate": float((pnl > 0).mean()),
                "avg_trade_pnl": float(pnl.mean()),
                "median_trade_pnl": float(pnl.median()),
                "best_trade_pnl": float(pnl.max()),
                "worst_trade_pnl": float(pnl.min()),
                "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() else np.inf,
                "avg_holding_bars": float(trades["holding_bars"].mean()) if "holding_bars" in trades else 0.0,
            }
        )
    else:
        out.update(
            {
                "trade_count": 0,
                "trade_win_rate": 0.0,
                "avg_trade_pnl": 0.0,
                "profit_factor": 0.0,
                "avg_holding_bars": 0.0,
            }
        )
    return out
