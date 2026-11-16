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
| `open_tradable_reversal` | 3.439208 | 0.244660 | -0.019706 | `top_n_20p0_hold_1_exposure_0p5` |
| `open_reversal_stable` | 2.763332 | 0.337105 | -0.034441 | `top_n_100p0_hold_1_exposure_1p0` |
| `open_momentum_risk_on` | 2.179107 | 0.464803 | -0.085197 | `top_n_100p0_hold_1_exposure_1p0` |
| `open_reversal_small` | 1.704992 | 0.287770 | -0.077016 | `top_n_100p0_hold_1_exposure_0p5` |
| `open_selloff_reversal` | 0.497578 | 0.064020 | -0.073839 | `top_n_100p0_hold_1_exposure_0p5` |
| `open_rebound_reversal` | 0.237677 | 0.003913 | -0.008144 | `top_quantile_0p005_hold_1_exposure_0p5` |

## 选中策略

- 候选：`open_tradable_reversal`。
- 规则：`top_n_20p0_hold_1_exposure_0p5`。
- 说明：开盘反转小市值策略加入价格健康度，避免极端低价股票。

## 测试集验收

- Sharpe：1.720757
- 年化收益：0.140241
- 年化波动率：0.085022
- 最大回撤：-0.093118
- 成本后总收益：0.133728
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`
