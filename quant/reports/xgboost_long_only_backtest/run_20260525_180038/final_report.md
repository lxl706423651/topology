# XGBoost Long-Only A-Share Backtest Report

## Data And Split

- Train: `datetime < 2025-05-01`. Inner validation starts at `2024-11-01` and remains inside train.
- Test: `datetime >= 2025-05-01`.
- Symbols: 257 stable HS300 symbols from `/data/lxl/seed-emulator/topology/quant/reports/stable_hs300_factor_correlation/run_20260518_223759/stable_hs300_symbols.csv`.
- Raw fields used: `datetime`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `amount`.
- Requested factor count note: current authoritative model feature manifest contains 156 features, sourced from 156 base factors.

## No-Leakage Controls

- Feature generation uses `quant_backtest.factor_library`; per-symbol rolling/lag features use only current and historical completed bars.
- Cross-sectional ranks are computed only within the same timestamp.
- Label is signal-lagged forward return: feature at t enters at t+1 close and exits after 24 bars.
- Inner training excludes labels whose exit crosses validation start; final training excludes labels whose exit crosses split date.
- XGBoost model, feature medians if needed, and portfolio rule selection are fit/selected only on train/internal validation.
- Test data is used once for transform, predict, and final evaluation.

## Model And Strategy

- Model: `XGBRegressor`, fixed parameters saved in `model_metadata.json`.
- Training target: same-timestamp cross-sectional rank of signal-lagged forward return.
- Strategy: long-only equal-weight `top_n_10p0` by prediction score.
- Shorting: disabled. Leverage: disabled. Gross exposure target: 1.0 when enough tradable names exist.
- Transaction cost: commission/turnover cost `0.0001` (万1). Slippage, stamp duty, T+1 sell limits, price-limit blocking, ST, delisting, and point-in-time constituents are not fully modeled and are listed as residual risks.

## Validation Rule Selection

The portfolio rule was selected on the internal validation slice only.

Top validation rule table is saved to `validation_rule_metrics.csv`.

## Test Metrics

- Sharpe: 1.042768
- Annual return: 0.135547
- Annual volatility: 0.133163
- Max drawdown: -0.104788
- Calmar: 1.293529
- Win bar rate: 0.501860
- Average turnover: 1.490947
- Total turnover: 3606.600000
- Average holdings: 10.00

## Sharpe Target

Sharpe >= 3 achieved: `False`.

## Validation Diagnostics

- Train mean Spearman IC: 0.081678
- Validation mean Spearman IC: 0.057144
- Selected validation rule: `top_n_10p0`.
- Best validation Sharpe: -1.317890
