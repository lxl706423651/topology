# 全 A 小市值状态过滤策略回测报告

## 研究边界

- 股票池来自本地全 A 小市值代理面板。
- 状态阈值只从验证集估计；候选和组合规则只用验证集选择；测试集只做最终验收。
- 策略是 A 股现货 long-only，不做空，不加杠杆，执行 signal_lag=1、T+1、100 股整数手和交易成本。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`。
- 测试集：`datetime >= 2025-05-01`。

## 验证集状态阈值

- `market_return_120_q40`：-0.00170853
- `market_return_120_q50`：0.00083424
- `market_return_120_q60`：0.00355116
- `breadth_24_q45`：0.33333333
- `breadth_120_q50`：0.44444444
- `market_volatility_120_q60`：0.00234617
- `market_volatility_120_q70`：0.00254260
- `market_liquidity_q40`：391554.18750000

## 验证集候选

| 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 |
|---|---:|---:|---:|---|
| `small_reversal_low_vol` | 4.675771 | 0.593703 | -0.030654 | `top_quantile_0p005_hold_1_exposure_0p5` |
| `small_risk_on` | 3.722268 | 1.003529 | -0.107707 | `top_n_20p0_hold_1_exposure_1p0` |
| `small_momentum_upside` | 3.285619 | 0.529852 | -0.087582 | `top_n_100p0_hold_1_exposure_1p0` |
| `tradable_small_defensive` | 1.277079 | 0.167934 | -0.108217 | `top_n_100p0_hold_1_exposure_0p5` |
| `small_liquidity_supported` | -0.089598 | -0.020285 | -0.179753 | `top_quantile_0p005_hold_1_exposure_0p5` |

## 选中策略

- 候选：`small_reversal_low_vol`。
- 规则：`top_quantile_0p005_hold_1_exposure_0p5`。
- 说明：小市值代理叠加短期反转和低波动，并要求市场反转/低波环境。

## 测试集验收

- Sharpe：1.653745
- 年化收益：0.132299
- 年化波动率：0.077026
- 最大回撤：-0.079178
- 成本后总收益：0.269394
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`
