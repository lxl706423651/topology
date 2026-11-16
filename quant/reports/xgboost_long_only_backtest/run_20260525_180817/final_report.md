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
- Gross total return before cost: 0.624792
- Net total return after cost: 0.129776
- Total cost return: 0.360660
- Max drawdown: -0.104788
- Calmar: 1.293529
- Win bar rate: 0.501860
- Average turnover: 1.490947
- Total turnover: 3606.600000
- Trade count: 36066
- Rebalance count: 2419
- Average holdings: 10.00

## Sharpe Target

Sharpe >= 3 achieved: `False`.

## Validation Diagnostics

- Train mean Spearman IC: 0.081678
- Validation mean Spearman IC: 0.057144
- Selected validation rule: `top_n_10p0`.
- Best validation Sharpe: -1.317890

## System Structure Reviewed

- Data loading: raw minute OHLCV CSV files under `ashare_minute_data_2023_2026_04/extracted`, read per symbol and sorted by `datetime`.
- Factor construction: `quant_backtest/factor_library.py`, reused by `scripts/build_factor_features.py`, `scripts/audit_factor_feature_library.py`, and this XGBoost run.
- Existing metrics engine: `quant_backtest/metrics.py` is used for annual return, volatility, Sharpe, drawdown, Calmar, win-bar rate, and tail-risk metrics.
- Portfolio layer: this script implements the cross-sectional top-N portfolio loop because the existing single-symbol `BacktestEngine` does not directly express a same-timestamp rank-top-N portfolio across all symbols.
- Outputs: model, predictions, holdings, equity curve, monthly/yearly returns, validation rule table, feature importance, metadata, and this report are saved in this run directory.

## 147-Factor And Final Feature Audit

- Previous stable-HS300 analysis found 147 effective mean factors in `reports/stable_hs300_factor_correlation/run_20260518_223759/factor_mean_metrics.csv`.
- Strict training-only audit for this experiment used `2023-01-03` through `2025-04-30`, reviewed 274 base factors, retained 79 of the previous 147 effective factors, selected 59 new base factors, and produced 156 model features.
- The user requested reuse of the current effective factor set. The current strict no-test artifact contains 156 features, not 157. This report uses the authoritative 156-feature manifest at `reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv`.

## Reproduction Commands

Factor audit command:

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_factor_feature_library.py --workers 4 --train-start 2023-01-03 --train-end 2025-04-30 --min-assets 20 --min-rank-ic-periods 120
```

Train, predict, and backtest command:

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgboost_long_only_backtest.py --feature-manifest reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv --workers 4 --split-date 2025-05-01 --validation-start 2024-11-01 --commission-rate 0.0001 --top-n-candidates 10,20,30,50 --top-quantile-candidates 0.05,0.10,0.20 --min-assets 30
```

Validation commands:

```bash
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile scripts/run_xgboost_long_only_backtest.py scripts/audit_factor_feature_library.py scripts/build_factor_features.py quant_backtest/factor_library.py
python3 -m py_compile scripts/run_xgboost_long_only_backtest.py scripts/audit_factor_feature_library.py scripts/build_factor_features.py quant_backtest/factor_library.py
```

## Final Conclusion

- Strict out-of-sample Sharpe did not reach 3. The achieved test Sharpe is 1.042768.
- No test-set training, feature selection, threshold choice, or strategy selection was used for this reported run.
- The main practical gap is execution realism: slippage, stamp duty, T+1 constraints, lot size, price-limit blocking, ST/delisting filters, and point-in-time historical index constituents are not fully modeled with the current fields.
