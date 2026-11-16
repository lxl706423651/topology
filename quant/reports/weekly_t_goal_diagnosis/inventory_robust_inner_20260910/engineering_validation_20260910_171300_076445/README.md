# Huber 短期限内部诊断工具

本目录只分析明确已经完成的 v17 内部折。使用原 core 模型元数据与内部评价年当前预测、保存账户、原 CSV 精确端点；不读取任何工程烟测或正式外层账户/预测，不拟合、校准、运行账户或修改参数。

运行必须等主进程负责人确认该折已冻结；同时要求该折 `validation_rule_metrics.csv`、`selected_rule.json`、`robust_vs_source_candidates.csv` 全部存在。

```bash
/home/lxl/anaconda3/envs/quant/bin/python reports/weekly_t_goal_diagnosis/inventory_robust_inner_20260910/diagnose.py --report-dir REPORT --stage --years 2023
/home/lxl/anaconda3/envs/quant/bin/python reports/weekly_t_goal_diagnosis/inventory_robust_inner_20260910/diagnose.py --report-dir REPORT --final
```

- `prediction_quality.csv`：两种 Huber 模型、原 20/40/60 交易分钟期限、方向、8 个时钟以及固定评分换算净差箱。分列全部当前键与基准日初旧股，保留缺端点、无效尾部时钟计数。只有可用端点子集用于价差误差描述，原决策不因缺未来数据删行。
- `argmax_premium.csv`：同当前键原合法期限最大评分与合法期限平均值的比较。缺任一备选端点单列；不根据未来比较结果选择期限。
- `candidate_overview.csv`、`candidate_source_comparison.csv`：完整 37 账户，按 Huber→L2 显式模型映射核原数字参数与含费底仓成本。两轮 raw 目标相同而损失、评分及实际信号不同。候选金额不可相加当组合。
- `candidate_pairs.csv`、`candidate_pair_quality.csv`：全部真实首腿，完整组、部分恢复、尚未恢复分开；分别记录标称原固定端点价差、已实现恢复片段、完整组实际净額。
- `candidate_failures.csv`、`candidate_blocked_reasons.csv`：原库存失败行和拒单原因，保留不利结果。
- `margin_candidate_summary.csv`：原 0/10/30 bp 门槛和全部模式的频次/费用/净额分布，只描述不选新阈值。
- `model_training_summary.csv`：原 core 拟合行数、特征数、矩阵/标签 SHA 与模型已保存的样本内残差描述。这里不重复完整模型审计。

价差使用原双腿佣金、过户费、当日印花税和滑点，以首腿未滑点参考金额归一化；最低佣金必须仍为零。成本降低率独立核为完整组实际净额除原始含费底仓成本，含未配对仓位的账户净增量另列。Huber 分数不是已校准的条件均值或真实可成交净利润。

诊断是反复历史研究，不是全新测试。不同日期/周仅是不同值计数，不证明统计独立。输出每次新建目录，阶段结果以脚本、输入及派生结果的完整 SHA 复用；最终主进程成功后只合并已有阶段并核 56 份运行源与原快照，不重读原分钟计算。

纯测试命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python reports/weekly_t_goal_diagnosis/inventory_robust_inner_20260910/test_diagnose.py
```
