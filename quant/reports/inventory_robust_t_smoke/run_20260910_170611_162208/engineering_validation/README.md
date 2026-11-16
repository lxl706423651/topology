# v17真实烟测工程验收

42个不同实现用例、双Python编译及357输入来源预检通过。真实烟测主进程exit0，392输入、56源码、48新模型产物始末一致；独立复核已完整通过（最终进程exit0）；验收证据见下方。原图已人工查看，明确工程用途、零净增量与不可评价胜率。

内部36主动均亏损；所选33完成1组4配对手，净−51.933459元、成本降低−0.030147%。工程外层2026-01-05至01-23为0请求/0成交/0净额，所有库存约束为0失败，不能视为改善。

零成交原因在当前预测即可复算：h20/h40各1440行均未达到原30bp净边际；h60只有2026-01-14 14:00的SZ.000665达到31.153347bp，但下一分钟再推进60交易分钟已跨当日收盘，原合法期限返回NaT，因此不发请求。该股当前限价方向与量正常。没有用未来恢复价格作这项判断，不因零成交调整门槛或读取工程外层择参。

原模型/决策/入口32项证据位于`reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/robust_implementation_validation_20260910_170719_038763/`，报告器10项位于同父目录`robust_reporting_validation_20260910_165855/`。原/tmp日志保留。

实际命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_inventory_robust_t.py --source-report reports/inventory_guarded_t_smoke/run_20260910_154550_255425 --candidate-workers 8 --output-dir reports/inventory_robust_t_smoke
python3 -m py_compile scripts/run_inventory_robust_t.py quant_backtest/inventory_t_robust_models.py quant_backtest/inventory_t_robust_policy.py quant_backtest/inventory_t_robust_reporting.py
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile scripts/run_inventory_robust_t.py quant_backtest/inventory_t_robust_models.py quant_backtest/inventory_t_robust_policy.py quant_backtest/inventory_t_robust_reporting.py
/home/lxl/anaconda3/envs/quant/bin/python /tmp/archive_inventory_robust_smoke.py
```

独立审计补充：

- [完整验收报告](../independent_robust_audit/run_20260910_172441_003843/acceptance.md)：6个Huber模型、48个模型产物、37个内部候选及1个工程外层，含基准共42个账户。核3,800条成交、1,267个首次退出意图与1,291次guard复查；24次取消是跨候选计数，不能当单一组合交易。
- 56运行源、392原登记输入、1,250项独审读取文件及11个审计脚本依赖始末不变。18项新审计反例及8项原guard反例通过；不与原42项实现测试重复混算。
- 两次未完成审计分别因审计合同错误主动终止、分流字段重名异常结束。它们的原目录、日志、脚本及SHA完整保留；只修独立审计器，没有重跑或改写生产模型和账户。
- [尝试与修复记录](audit_completion_20260910_173012_041809/attempt_history.md)；成功verification SHA：`74f78f6012caa5765cb03a1525222c29caa94da41f3080dd9ef14001b7297f26`。
- 原archive_smoke.py记录的是审计完成前的历史命令，本次状态更新由complete_independent_audit.py记录。工程0请求已按完整当前预测独立复算，收益/频率目标仍未实现。
