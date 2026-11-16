# 全 A 小市值开盘反转策略回测报告

## 研究边界

- 信号时间：`09:55`。
- 入场价格使用下一分钟价格，退出价格使用下一交易日同一入场时点价格。
- 状态阈值只从验证集估计；候选和组合规则只用验证集选择；测试集只做最终验收。
- 策略是 A 股现货 long-only，不做空，不加杠杆，执行 signal_lag=1、T+1、100 股整数手和交易成本。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`。
- 测试集：`datetime >= 2025-05-01`。

## 验证集状态阈值

- `open_market_return_q40`：-0.00011335
- `open_market_return_q50`：0.00151426
- `open_market_return_q60`：0.00290128
- `open_breadth_q45`：0.44336811
- `open_breadth_q55`：0.50720030
- `open_volatility_q70`：0.00258232
- `market_liquidity_q40`：529931.18750000

## 验证集候选

| 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 |
|---|---:|---:|---:|---|
| `open_tradable_reversal` | 4.140020 | 0.235435 | -0.003610 | `top_n_100p0_hold_1_exposure_1p0_spreadq_0p6` |
| `open_reversal_stable` | 2.763332 | 0.337105 | -0.034441 | `top_n_100p0_hold_1_exposure_1p0` |
| `open_momentum_risk_on` | 2.225181 | 0.105077 | -0.011980 | `top_n_100p0_hold_1_exposure_1p0_confq_0p8` |
| `open_selloff_reversal` | 1.870886 | 0.038009 | -0.000600 | `top_n_20p0_hold_1_exposure_0p5_confq_0p85_spreadq_0p6` |
| `open_reversal_small` | 1.866672 | 0.155041 | -0.036545 | `top_n_100p0_hold_1_exposure_0p3` |
| `open_rebound_reversal` | 1.802262 | 0.011398 | -0.000651 | `top_quantile_0p005_hold_1_exposure_0p3_spreadq_0p6` |

## 选中策略

- 候选：`open_tradable_reversal`。
- 规则：`top_n_100p0_hold_1_exposure_1p0_spreadq_0p6`。
- 说明：开盘反转小市值策略加入价格健康度，避免极端低价股票。

## 测试集验收

- Sharpe：0.468346
- 年化收益：0.025081
- 年化波动率：0.116699
- 最大回撤：-0.125865
- 成本后总收益：0.023973
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`
