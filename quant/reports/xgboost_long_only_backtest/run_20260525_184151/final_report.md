# XGBoost Long-Only A-Share Backtest Report

## Data And Split

- Train: `datetime < 2025-05-01`. Inner validation starts at `2024-11-01` and remains inside train.
- Test: `datetime >= 2025-05-01`.
- Symbols: 8 stable HS300 symbols from `/data/lxl/seed-emulator/topology/quant/reports/stable_hs300_factor_correlation/run_20260518_223759/stable_hs300_symbols.csv`.
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
- Strategy: long-only lot-sized `top_n_3p0_hold_3_exposure_1p0` by prediction score.
- Shorting: disabled. Leverage: disabled. Target gross exposure: `1.0` when enough tradable names exist.
- Holding rule: rebalance every `3` sampled periods, where each sampled period is `24` bars.
- Lot sizing: buys are rounded down to `100`-share board lots from `1000000.0` initial cash.
- Transaction cost: all-in tax/fee cost `0.0001` (万1) on traded notional. Slippage is intentionally set to zero. T+1 sell limits, price-limit blocking, ST, delisting, and point-in-time constituents are not fully modeled and are listed as residual risks.

## Validation Rule Selection

The portfolio rule was selected on the internal validation slice only.

Top validation rule table is saved to `validation_rule_metrics.csv`.

## Test Metrics

- Sharpe: 0.816816
- Annual return: 0.131335
- Annual volatility: 0.172815
- Gross total return before cost: 0.236780
- Net total return after cost: 0.125753
- Total cost return: 0.091339
- Max drawdown: -0.158825
- Calmar: 0.826912
- Win bar rate: 0.480364
- Average turnover: 0.377591
- Total turnover: 913.393197
- Average gross exposure: 0.996941
- Average cash weight: 0.003021
- Trade count: 3370
- Rebalance count: 805
- Average holdings: 3.00

## Sharpe Target

Sharpe >= 2 achieved: `False`.

## Validation Diagnostics

- Train mean Spearman IC: 0.374994
- Validation mean Spearman IC: 0.020838
- Selected validation rule: `top_n_3p0_hold_3_exposure_1p0`.
- Best validation Sharpe: -0.082666
