# XGBoost 参数与策略门控调优报告

## 调优边界

- 核心训练集：`datetime < 2024-11-01`，且标签窗口不跨验证集。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`，且标签窗口不跨测试集。
- 测试集：`datetime >= 2025-05-01`，只在模型参数和策略规则选定后评估一次。
- 特征筛选、XGBoost 参数选择、top-N、置信度门控和分差门控都不使用测试集。

## 参数重要性说明

- `min_child_weight`：叶子节点最小样本权重，越大越保守；对分钟级噪声和小样本局部规律很关键。 本轮该参数最佳候选 `deeper_more_regularized`，验证 Sharpe 4.503279。
- `max_depth`：单棵树深度，控制非线性复杂度；越深越能捕捉交互，也越容易过拟合。 本轮该参数最佳候选 `deeper_more_regularized`，验证 Sharpe 4.503279。
- `reg_lambda`：L2 正则，平滑叶子权重，降低极端拟合。 本轮该参数最佳候选 `deeper_more_regularized`，验证 Sharpe 4.503279。
- `n_estimators`：树的数量，控制 boosting 轮数；太少欠拟合，太多在噪声分钟数据上可能过拟合。 本轮该参数最佳候选 `combined_expressive`，验证 Sharpe 4.148167。
- `learning_rate`：每棵树的步长，和 n_estimators 共同决定拟合速度；低步长通常更稳但需要更多树。 本轮该参数最佳候选 `combined_expressive`，验证 Sharpe 4.148167。
- `reg_alpha`：L1 正则，鼓励稀疏和更强约束，可压制弱噪声特征。 本轮该参数最佳候选 `combined_expressive`，验证 Sharpe 4.148167。
- `colsample_bytree`：列采样比例，降低高维因子共线性风险。 本轮该参数最佳候选 `combined_conservative`，验证 Sharpe 3.546800。
- `subsample`：行采样比例，降低过拟合并增加树之间差异。 本轮该参数最佳候选 `higher_sampling`，验证 Sharpe 2.787650。

## 验证集候选排名

| 排名 | 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 | 改动参数 |
|---:|---|---:|---:|---:|---|---|
| 1 | `deeper_more_regularized` | 4.503279 | 5.827162 | -0.130523 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9` | `{"max_depth": 5, "min_child_weight": 40, "reg_lambda": 8.0}` |
| 2 | `combined_expressive` | 4.148167 | 3.496843 | -0.104571 | `top_n_1p0_hold_1_exposure_1p0_confq_0p92_spreadq_0p6` | `{"learning_rate": 0.03, "max_depth": 5, "min_child_weight": 20, "n_estimators": 300, "reg_alpha": 0.1, "reg_lambda": 5.0}` |
| 3 | `lower_min_child_weight` | 3.655257 | 3.348639 | -0.083744 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p8` | `{"min_child_weight": 10}` |
| 4 | `higher_lr_fewer_trees` | 3.616328 | 3.393844 | -0.218314 | `top_n_1p0_hold_1_exposure_1p0_confq_0p92` | `{"learning_rate": 0.06, "n_estimators": 160}` |
| 5 | `combined_conservative` | 3.546800 | 2.634288 | -0.139521 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p6` | `{"colsample_bytree": 0.75, "learning_rate": 0.025, "max_depth": 3, "min_child_weight": 30, "n_estimators": 360, "reg_alpha": 0.3, "reg_lambda": 8.0}` |
| 6 | `lower_lr_more_trees` | 3.472054 | 2.289322 | -0.065091 | `top_n_1p0_hold_1_exposure_1p0_confq_0p92_spreadq_0p8` | `{"learning_rate": 0.025, "n_estimators": 360}` |
| 7 | `baseline_current` | 2.983009 | 2.084481 | -0.108705 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p8` | `{}` |
| 8 | `stronger_regularization` | 2.798060 | 1.631669 | -0.136959 | `top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7` | `{"reg_alpha": 0.5, "reg_lambda": 10.0}` |

## 选中组合

- 选中模型参数候选：`deeper_more_regularized`。
- 候选含义：提高树深以捕捉非线性，同时加大叶子约束和 L2 正则防止过拟合。
- 参数覆盖：`{"max_depth": 5, "min_child_weight": 40, "reg_lambda": 8.0}`。
- 选中策略规则：`top_n_1p0_hold_1_exposure_1p0_confq_0p9`。
- 置信度门控阈值：`0.5603431224822998`，来自验证集分位数 `0.9`。
- top1-top2 分差门控阈值：`None`，来自验证集分位数 `None`。

## 最终测试集验收

- Sharpe：2.800323
- 年化收益：2.295213
- 年化波动率：0.463586
- 最大回撤：-0.298598
- 成本后总收益：2.141427
- 平均多头暴露：0.156053
- short_exposure：0.000000
- Sharpe 目标 `3` 是否达成：`False`。
- 如果最终测试未达目标，也不能根据测试集结果回头修改参数；下一步只能使用新的未来数据、重新定义训练期内部 walk-forward 验证，或在明确标注已触碰测试集的前提下另开研究实验。

## 产物说明

- `model_parameter_candidates.csv`：每个模型候选在验证集上最优策略规则和指标。
- `parameter_sensitivity.csv`：按参数维度汇总的验证集敏感性。
- `validation_rules_<candidate>.csv`：单个候选模型的全部验证集策略规则表现。
- 最终模型、预测、交易记录、收益曲线和报告复用主回测脚本的输出格式。
