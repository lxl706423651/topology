# 资料驱动策略回测报告

## 研究边界

- 当前实现包含两个策略家族：风格中性 long-only 和小市值代理 long-only。
- 风格中性不是股票多空市场中性；当前框架没有融券、股指期货或 ETF 对冲腿。
- 小市值策略使用低价格、低历史成交额和高非流动性作为代理；本地数据没有历史总市值/流通市值。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`。
- 测试集：`datetime >= 2025-05-01`，只用于最终验收。

## 验证集候选排名

| 策略家族 | 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 |
|---|---|---:|---:|---:|---|
| `neutral` | `neutral_residual_basic` | 1.327268 | 0.473500 | -0.198010 | `top_n_1p0_hold_1_exposure_1p0` |
| `neutral` | `neutral_residual_style` | 0.509375 | 0.111325 | -0.134646 | `top_n_1p0_hold_1_exposure_1p0` |
| `neutral` | `neutral_residual_defensive` | -1.910711 | -0.379153 | -0.265036 | `top_n_2p0_hold_1_exposure_1p0` |
| `small_cap` | `small_cap_alpha_50` | 2.095576 | 0.252109 | -0.050139 | `top_n_2p0_hold_1_exposure_1p0_confq_0p9` |
| `small_cap` | `small_cap_quality` | 1.812439 | 0.314648 | -0.056957 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9` |
| `small_cap` | `small_cap_alpha_30` | 1.789380 | 0.262085 | -0.063077 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9` |
| `small_cap` | `small_cap_proxy` | 1.641635 | 0.325013 | -0.130814 | `top_n_2p0_hold_1_exposure_1p0_spreadq_0p7` |

## 每个策略家族的选中结果

- `neutral` 选中 `neutral_residual_basic`，验证 Sharpe 1.327268，规则 `top_n_1p0_hold_1_exposure_1p0`。
- `small_cap` 选中 `small_cap_alpha_50`，验证 Sharpe 2.095576，规则 `top_n_2p0_hold_1_exposure_1p0_confq_0p9`。

## 最终测试集验收

### neutral

- 候选：`neutral_residual_basic`
- Sharpe：0.508005
- 年化收益：0.101132
- 年化波动率：0.259266
- 最大回撤：-0.161026
- 成本后总收益：0.096889
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`

### small_cap

- 候选：`small_cap_alpha_50`
- Sharpe：-0.101036
- 年化收益：-0.015159
- 年化波动率：0.100901
- 最大回撤：-0.119450
- 成本后总收益：-0.014556
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`

## 产物说明

- `strategy_candidate_summary.csv`：所有候选在验证集上的最优规则和指标。
- `strategy_selection_summary.csv`：每个策略家族验证集选中的候选。
- `neutral/`、`small_cap/`：各策略家族最终测试回测产物，含收益曲线和逐笔交易。
- `资料依据.md`：论文和数据接口资料，以及本地实现的限制说明。
