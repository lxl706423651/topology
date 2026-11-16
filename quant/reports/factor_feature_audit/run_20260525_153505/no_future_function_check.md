# No-Future-Function Check

Status: PASS

Training window: 2023-01-03 00:00:00 to 2025-06-30 23:59:59.999999999.
Label: signal-lagged forward return with entry at next bar close and exit after 24 bars.
Feature timestamps whose label exit would pass train_end are excluded from training metrics.

Checks performed:

- Feature names exclude `future_return`, `forward_return`, `target`, and `label` columns.
- All selected feature columns exist in the training panel.
- No duplicate `(symbol, datetime)` rows in the training panel.
- No infinite feature values after factor generation.
- Training panel sorted by `symbol, datetime` for validation.
- Label windows remain inside the training period.
- Dynamic future perturbation check: for three sample stable symbols, OHLCV rows strictly after a cutoff were changed and all generated factor values up to the cutoff remained unchanged.

Alignment policy:

- Per-symbol rolling, lag, pct-change, trend, volatility, liquidity, and technical features use the current and historical completed bars only.
- Cross-sectional rank/z-score features group only by the same `datetime` and do not use future dates.
- Any generated feature based on bar t is marked as usable no earlier than t+1 trading/execution.
- Feature selection uses only training-window RankIC, missing ratio, monthly stability, and training-set correlation filtering. Test-set IC, Sharpe, or backtest results are not used.
