# v17真实烟测工程验收

42个不同实现用例、双Python编译及357输入来源预检通过。真实烟测主进程exit0，392输入、56源码、48新模型产物始末一致；独立复核尚待完成。原图已人工查看，明确工程用途、零净增量与不可评价胜率。

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
