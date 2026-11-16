# v16 固定尾盘目标报告器验证

新增 `quant_backtest/inventory_t_close_reporting.py`，接口 `write_close_inventory_report(directory, cfg, annual_results, validation_table=None)`。`research_version` 必须为 `inventory_close_t_v16`，目标为 `forecast_target=same_day_1455`、`recovery_key=close`、`fixed_recovery_time=14:55`，不能传旧整数期限。必须明确 `source_execution_report_dir`（v15执行）和 `source_learning_report_dir`（v14完整核心队列）并开启成交时反向限价复查。

报告直接将真实 v16 配置传给公共 balanced 核算，独立生成 v16 图文，不伪装旧版本配置或复用旧报告器的版本入口。`inventory_close_summary.json` 明确每折新拟合两个 close 模型、`frozen_t_models_reused=false`，模型标签为下一分钟至同日14:55参考价收益，既非固定整数长度，也非组合真实成交利润。不再跨20/40/60逐行取最大预测；方向和内部候选选择仍保留。37候选数量与原经济参数不变。

原底仓含费成本、人工加仓成本、零T股票、双腿费用滑点、完整配对手/双腿交易手、T+1及股数失败口径均直接使用公共核算。复查请求、取消、通过但现金等后续拒绝、实际首腿、首次退出锁存和恢复片段分别计数。动态退出可早于14:55，受阻可能跨日；不能把参考标签或取消金额冒称实际盈利。

8 项测试通过（3.967秒），Quant Python 与系统 Python 编译通过。测试实际生成规范CSV/JSON/中文图文，核图标题为v16/14:55、成本分母包含未做T股及人工加仓、5%与未量化频率状态分别列示、恢复失败不隐藏、hold烟测不称成功、已有年度守卫/退出CSV不改写。另以真实guard执行器运行一个两日合成账户：首腿后14:55恢复，实际费用为正、净额为负、库存差为0，报告计数与真实结果一致。没有读真实行情或重复真实回测，没有拟合模型。

这只是报告工程验证，不是新模型盈利证据。51份既有v15冻结运行源码与烟测快照保持不变。完整命令、结果、日志与新源码副本见 `verification.json`、`unit_tests.log`、`source_snapshot/`；`manifest.json` 绑定归档。
