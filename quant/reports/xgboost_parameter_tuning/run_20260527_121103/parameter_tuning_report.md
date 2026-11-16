# XGBoost 参数与策略门控调优报告

## 调优边界

- 核心训练集：`datetime < 2024-11-01`，且标签窗口不跨验证集。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`，且标签窗口不跨测试集。
- 测试集：`datetime >= 2025-05-01`，只在模型参数和策略规则选定后评估一次。
- 特征筛选、XGBoost 参数选择、top-N、置信度门控和分差门控都不使用测试集。

## 参数重要性说明

- `learning_rate`：每棵树的步长，和 n_estimators 共同决定拟合速度；低步长通常更稳但需要更多树。 本轮该参数最佳候选 `lower_lr_more_trees`，验证 Sharpe 1.256299。
- `n_estimators`：树的数量，控制 boosting 轮数；太少欠拟合，太多在噪声分钟数据上可能过拟合。 本轮该参数最佳候选 `lower_lr_more_trees`，验证 Sharpe 1.256299。

## 验证集候选排名

| 排名 | 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 | 改动参数 |
|---:|---|---:|---:|---:|---|---|
| 1 | `baseline_current` | 1.705018 | 0.405071 | -0.094090 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9` | `{}` |
| 2 | `lower_lr_more_trees` | 1.256299 | 0.290840 | -0.133191 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7` | `{"learning_rate": 0.025, "n_estimators": 360}` |

## 选中组合

- 选中模型参数候选：`baseline_current`。
- 候选含义：当前基准参数，用作所有调优方向的对照。
- 参数覆盖：`{}`。
- 选中策略规则：`top_n_1p0_hold_1_exposure_1p0_confq_0p9`。
- 置信度门控阈值：`0.6130301237106324`，来自验证集分位数 `0.9`。
- top1-top2 分差门控阈值：`None`，来自验证集分位数 `None`。

## 最终测试集验收

- Sharpe：-0.820602
- 年化收益：-0.113430
- 年化波动率：0.135505
- 最大回撤：-0.147430
- 成本后总收益：-0.109142
- 平均多头暴露：0.287739
- short_exposure：0.000000

## 产物说明

- `model_parameter_candidates.csv`：每个模型候选在验证集上最优策略规则和指标。
- `parameter_sensitivity.csv`：按参数维度汇总的验证集敏感性。
- `validation_rules_<candidate>.csv`：单个候选模型的全部验证集策略规则表现。
- 最终模型、预测、交易记录、收益曲线和报告复用主回测脚本的输出格式。
