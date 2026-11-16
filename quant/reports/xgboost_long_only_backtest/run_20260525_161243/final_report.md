# XGBoost Long-Only A-Share Backtest Report

## Data And Split

- Train: `datetime < 2025-05-01`. Inner validation starts at `2024-11-01` and remains inside train.
- Test: `datetime >= 2025-05-01`.
- Symbols: 8 stable HS300 symbols from `/data/lxl/seed-emulator/topology/quant/reports/stable_hs300_factor_correlation/run_20260518_223759/stable_hs300_symbols.csv`.
- Raw fields used: `datetime`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `amount`.
- Requested factor count note: current authoritative model feature manifest contains 154 features, sourced from 154 base factors.

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
- Strategy: long-only equal-weight `top_n_5p0` by prediction score.
- Shorting: disabled. Leverage: disabled. Gross exposure target: 1.0 when enough tradable names exist.
- Transaction cost: commission/turnover cost `0.0001` (万1). Slippage, stamp duty, T+1 sell limits, price-limit blocking, ST, delisting, and point-in-time constituents are not fully modeled and are listed as residual risks.

## Validation Rule Selection

The portfolio rule was selected on the internal validation slice only.

Top validation rule table is saved to `validation_rule_metrics.csv`.

## Test Metrics

- Sharpe: -0.276144
- Annual return: -0.054758
- Annual volatility: 0.152218
- Max drawdown: -0.173377
- Calmar: -0.315833
- Win bar rate: 0.489458
- Average turnover: 0.624969
- Total turnover: 1511.800000
- Average holdings: 5.00

## Sharpe Target

Sharpe >= 3 achieved: `False`.

## Validation Diagnostics

- Train mean Spearman IC: 0.371909
- Validation mean Spearman IC: 0.037963
- Selected validation rule: `top_n_5p0`.
- Best validation Sharpe: -0.337581
