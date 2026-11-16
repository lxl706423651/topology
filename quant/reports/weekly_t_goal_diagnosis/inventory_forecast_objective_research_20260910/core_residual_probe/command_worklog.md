# Core residual probe command / worklog

## 2026-09-10 16:51 — 固定 1% 原 L2 核心残差核查

- 授权范围：原 v14 四折各六个模型，只读已审计 training_inputs.pkl、training_labels.csv、return_model.txt、model_metadata.json 及其已通过审计 SHA；只在本目录写统计。
- 主运行：exit 0，19.964944 秒；24 模型、96 模型文件和原完整审计文件始末 SHA 一致。18 次唯一原生推断、6 次严格身份缓存复用。
- 训练与策略：没有调用 fit、改变标签、重新跑账户、读取内层/外层预测或账户，没有改动原 v14/v16 源码或报告。
- 纯算术自检：严格大于 1% 边界、平方误差份额、绝对梯度份额、零分母、非有限值拒绝、输入不变均通过；只使用固定 alpha=.01。
- 两套 Python 语法编译均 exit 0。

```bash
/home/lxl/anaconda3/envs/quant/bin/python reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/core_residual_probe/probe.py --self-test
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/core_residual_probe/probe.py
python -m py_compile reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/core_residual_probe/probe.py
/home/lxl/anaconda3/envs/quant/bin/python reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/core_residual_probe/probe.py --report-dir reports/inventory_learning_t/run_20260910_151232_841321 --audit-file reports/inventory_learning_t/run_20260910_151232_841321/independent_learning_audit/run_20260910_153539_071695/verification.json --audit-sha256 6575f7b09f0a543534839c4789bcddce3c99f7345f4b652df54284cf4de47f68 --output-dir reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/core_residual_probe --prediction-threads 4 > reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/core_residual_probe/probe.log 2>&1
```

- 前置只读 schema 检查：原完整审计 passed/complete/all_read_inputs_unchanged；实际目标 CSV round_trip 读回后内容 SHA 与元数据相同；PKL 保留原 feature dtype 及列顺序。
- 后置归档：按每期限/深度的三个唯一模型写 unique_model_ranges.csv 与 interpretation.md；没有求和候选收益或合并复用折为独立样本。
- 局限：原生预测对的是原保存 float64 target_return；这是已拟合 L2 的样本内残差，不是 Huber 拟合结果、置信区间、外推收益或年度 5% 目标证明。
