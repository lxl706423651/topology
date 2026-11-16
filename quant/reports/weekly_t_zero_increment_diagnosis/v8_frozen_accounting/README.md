# v8 四年T净增量为零的收益通路诊断

未发现收益函数被常量0覆盖。四个外层规则的 enabled 都为 false，实际 T 请求、T 成交、配对均为空，实际周底仓订单逐笔等于基准；因此净增量精确为0。这个结果只证明没有执行T，不能据此推断“所有做T规则收益都为0”。本诊断只读已有文件、独立核算已存账目，未重新训练或回测。

## 四年实际执行数量

|外层年份|enabled|预测行|非零预测行|T决策行|全部实际事件|T事件|T配对片段|T增量(元)|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|2023|False|102304|102304|0|8711|0|0|0.000000|
|2024|False|111200|111200|0|9843|0|0|0.000000|
|2025|False|110288|110288|0|10120|0|0|0.000000|
|2026|False|35960|35960|0|3169|0|0|0.000000|

最大预测绝对值分别为0.006400554、0.006368240、0.007673699、0.006002019，模型预测并不是常量0。

## 从选参到净收益的代码位置

- [run_weekly_t_residual.py:220](/data/lxl/seed-emulator/topology/quant/scripts/run_weekly_t_residual.py:220)：无候选同时通过内部验收时，明确把最终规则 enabled 设置为 false，保留失败原因。不是先成交再把利润清零。
- [weekly_t_short_horizon.py:110](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_short_horizon.py:110)：T方向先置0，只有 enabled 为真才写入买/卖信号；[134行](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_short_horizon.py:134)随后仅保留非零方向，关闭规则自然得到空订单。
- [weekly_t_engine.py:370](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_engine.py:370)：买入实际扣本金和费用、卖出增加本金并扣费用；394行累计真实T现金流。
- [weekly_t_engine.py:480](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_engine.py:480)：每个恢复片段按股数分摊首腿费用，499行计算 net_pnl = 股数×(卖价−买价)−买费−卖费，未对正负利润截断。滑点已在执行价格里。
- [weekly_t_engine.py:700](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_engine.py:700)：两账户分别计算现金+持仓市值，704行直接相减形成T增量；[766行](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_engine.py:766)只在配对表为空时将配对总利默认记0。
- [weekly_t_validation.py:122](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_validation.py:122)内部评价与[run_weekly_t_cross_validation.py:306](/data/lxl/seed-emulator/topology/quant/scripts/run_weekly_t_cross_validation.py:306)正式保存均直接用期末实际净值减基准净值，没有覆盖为固定0。

## 已保存内部候选确实存在正、负利润

108个内部候选中，27个增量为正、64个为负、17个为0，共91个非零。例如外层折2023/内层2026的候选14（40分钟、l2_depth6、门槛0.003）完成16组：账户增量127.346135364845元，配对净利润127.346135365004元，T现金流127.346135365005元；基准计划偏差0、期末pending0。它因日内库存/恢复失败、周bootstrap下界及样本不足未通过验收。

完整配对且底仓计划无偏差的60个候选，账户增量与配对利润的最大差为7.1793e-10元；所有期末无pending候选的T现金流与配对利润最大差为5.3433e-12元。这些是对已存内部指标的交叉核对。

当底仓计划发生偏差时，账户增量还包含周底仓后续成交差异及其估值，不能强制等于纯T配对利润。例如外层折2026候选21记录账户增量824.609391元、配对利润610.996590元，且有6笔base_plan_mismatch。没有完整内层底仓事件就不能把差额认定为误算或纯T收益。

重要证据限制：当前[weekly_t_validation.py:112](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_validation.py:112)仅将失败/部分恢复attempts与库存异常保存到validation_candidate_audits；完整内层events、round_trips、curve没有落盘。因此不能只凭该目录声称已独立逐笔复算候选14的全部利润。本次未重新执行候选，保留这个限制。

## 已存完整真实执行的独立非零复算

为直接检查逐笔收益函数，读取此前固定四股、2024-01-08当周、10:00信号、20分钟恢复的真实行情诊断。该诊断执行引擎SHA与v8完全相同，费用、滑点、金额参与率相同；股票与两个方向在执行前固定，不是从未来盈利中选出的策略。此处只作为收益代码与现金记账证据。

|既存独立账户|组数/片段|独立配对净利(元)|独立T现金流(元)|实际−基准净值(元)|最大配对与账户差(元)|
|---|---:|---:|---:|---:|---:|
|sell_first|12/27|-1730.362697510005|-1730.362697509992|-1730.362697509932|7.276e-11|
|buy_first|12/27|253.142354010005|253.142354010018|253.142354010022|1.737e-11|

具体正片段：2024-01-09 SH.600004，10:01买入参考成交价9.16458、10:21卖出9.245375，恢复5,000股；毛利润403.975元，分摊首腿费37.116549元、卖出费60.55720625元，净利润306.30124475元。对应pair原始5,900股，其余900股在后续分钟恢复，因此首腿费用按该5,000股占比计入。

本次从真实事件重新计算全部佣金/过户费/印花税、执行价中的滑点、每笔cash_after、每片段净利、每组现金流，并先验证两个账户周底仓订单完全相同；所有检查通过。结果同时有正、负净利，反证收益计算是固定0。它不证明正式残差模型能盈利，也不改变本轮全部关闭T的正式结果。

## 产物

`outer_rule_decision_event_counts.csv`记录四年真实数量；`internal_candidate_accounting_identity.csv`保存108候选汇总恒等式；`example_inner_candidate_014_*`原样引用其已存失败证据；`existing_real_*_independent_fragments.csv`和`*_pairs.csv`保存54个片段、24组现金独立复算。`summary.json`和`archive_manifest.json`绑定来源、代码及产物SHA。
