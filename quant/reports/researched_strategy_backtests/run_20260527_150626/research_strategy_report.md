# 资料驱动策略回测报告

## 研究边界

- 当前实现包含两个策略家族：风格中性 long-only 和小市值代理 long-only。
- 风格中性不是股票多空市场中性；当前框架没有融券、股指期货或 ETF 对冲腿。
- 小市值策略优先使用日频换手率反推的历史流通市值；未提供日频市值缓存时退化为低价格、低历史成交额和高非流动性代理。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`。
- 测试集：`datetime >= 2025-05-01`，只用于最终验收。

## 验证集候选排名

| 策略家族 | 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 |
|---|---|---:|---:|---:|---|
| `neutral` | `neutral_residual_defensive` | 0.264866 | 0.022061 | -0.258759 | `top_n_3p0_hold_1_exposure_1p0` |
| `neutral` | `neutral_residual_basic` | 0.172294 | 0.016114 | -0.275219 | `top_n_2p0_hold_1_exposure_1p0` |
| `neutral` | `neutral_residual_style` | 0.034708 | -0.058932 | -0.256570 | `top_n_2p0_hold_1_exposure_1p0` |
| `small_cap` | `small_cap_quality` | 0.586301 | 0.113673 | -0.113570 | `top_n_1p0_hold_1_exposure_1p0` |
| `small_cap` | `small_cap_real_alpha_30` | 0.253581 | 0.032517 | -0.151886 | `top_n_3p0_hold_1_exposure_1p0_spreadq_0p7` |
| `small_cap` | `small_cap_alpha_30` | 0.123042 | -0.029955 | -0.274759 | `top_n_1p0_hold_1_exposure_1p0` |
| `small_cap` | `small_cap_proxy` | -0.047791 | -0.042851 | -0.195314 | `top_n_1p0_hold_1_exposure_1p0_spreadq_0p7` |
| `small_cap` | `small_cap_alpha_50` | -0.575038 | -0.146863 | -0.274511 | `top_n_1p0_hold_1_exposure_1p0_spreadq_0p7` |
| `small_cap` | `small_cap_real_alpha_50` | -0.642069 | -0.236385 | -0.294993 | `top_n_1p0_hold_1_exposure_1p0_spreadq_0p7` |
| `small_cap` | `small_cap_real_size` | -0.695876 | -0.248543 | -0.263549 | `top_n_1p0_hold_1_exposure_1p0` |

## 每个策略家族的选中结果

- `neutral` 选中 `neutral_residual_defensive`，验证 Sharpe 0.264866，规则 `top_n_3p0_hold_1_exposure_1p0`。
- `small_cap` 选中 `small_cap_quality`，验证 Sharpe 0.586301，规则 `top_n_1p0_hold_1_exposure_1p0`。

## 最终测试集验收

### neutral

- 候选：`neutral_residual_defensive`
- Sharpe：0.704099
- 年化收益：0.180532
- 年化波动率：0.310379
- 最大回撤：-0.377644
- 成本后总收益：0.172705
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`

### small_cap

- 候选：`small_cap_quality`
- Sharpe：-1.025807
- 年化收益：-0.179369
- 年化波动率：0.170297
- 最大回撤：-0.227702
- 成本后总收益：-0.172841
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`

## 产物说明

- `strategy_candidate_summary.csv`：所有候选在验证集上的最优规则和指标。
- `strategy_selection_summary.csv`：每个策略家族验证集选中的候选。
- `neutral/`、`small_cap/`：各策略家族最终测试回测产物，含收益曲线和逐笔交易。
- `资料依据.md`：论文和数据接口资料，以及本地实现的限制说明。
