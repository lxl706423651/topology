# v7 已知训练域与OOF模型独立审计

请求范围完成：True；全部通过：True。
只读模型、数据和选股来源；不读取外层收益、不训练或重新选参。

|周selector|训练年|目标年|样本SHA|来源SHA|OOF排名及选择|
|---|---|---:|---|---|---|
|year_2023/oof_weekly_cohort/selector_2024|[2025]|2024|True|True|True|
|year_2023/oof_weekly_cohort/selector_2025|[2024]|2025|True|True|True|
|year_2024/oof_weekly_cohort/selector_2023|[2025]|2023|True|True|True|
|year_2024/oof_weekly_cohort/selector_2025|[2023]|2025|True|True|True|
|year_2025/oof_weekly_cohort/selector_2023|[2024]|2023|True|True|True|
|year_2025/oof_weekly_cohort/selector_2024|[2023]|2024|True|True|True|
|year_2026/oof_weekly_cohort/selector_2023|[2024]|2023|True|True|True|
|year_2026/oof_weekly_cohort/selector_2024|[2023]|2024|True|True|True|

|T模型|训练行|样本SHA|来源SHA|隔离重叠|原生权重一致|
|---|---:|---|---|---:|---|
|year_2023/inner_t_h20_l2_depth4|9696|True|True|0|True|
|year_2023/inner_t_h20_l2_depth6|9696|True|True|0|True|
|year_2023/inner_t_h20_huber_depth4|9696|True|True|0|True|
|year_2023/inner_t_h40_l2_depth4|8484|True|True|0|True|
|year_2023/inner_t_h40_l2_depth6|8484|True|True|0|True|
|year_2023/inner_t_h40_huber_depth4|8484|True|True|0|True|
|year_2023/inner_t_h60_l2_depth4|7272|True|True|0|True|
|year_2023/inner_t_h60_l2_depth6|7272|True|True|0|True|
|year_2023/inner_t_h60_huber_depth4|7272|True|True|0|True|
|year_2024/inner_t_h20_l2_depth4|8976|True|True|0|True|
|year_2024/inner_t_h20_l2_depth6|8976|True|True|0|True|
|year_2024/inner_t_h20_huber_depth4|8976|True|True|0|True|
|year_2024/inner_t_h40_l2_depth4|7854|True|True|0|True|
|year_2024/inner_t_h40_l2_depth6|7854|True|True|0|True|
|year_2024/inner_t_h40_huber_depth4|7854|True|True|0|True|
|year_2024/inner_t_h60_l2_depth4|6732|True|True|0|True|
|year_2024/inner_t_h60_l2_depth6|6732|True|True|0|True|
|year_2024/inner_t_h60_huber_depth4|6732|True|True|0|True|
|year_2025/inner_t_h20_l2_depth4|8880|True|True|0|True|
|year_2025/inner_t_h20_l2_depth6|8880|True|True|0|True|
|year_2025/inner_t_h20_huber_depth4|8880|True|True|0|True|
|year_2025/inner_t_h40_l2_depth4|7770|True|True|0|True|
|year_2025/inner_t_h40_l2_depth6|7770|True|True|0|True|
|year_2025/inner_t_h40_huber_depth4|7770|True|True|0|True|
|year_2025/inner_t_h60_l2_depth4|6660|True|True|0|True|
|year_2025/inner_t_h60_l2_depth6|6660|True|True|0|True|
|year_2025/inner_t_h60_huber_depth4|6660|True|True|0|True|
|year_2026/inner_t_h20_l2_depth4|8880|True|True|0|True|
|year_2026/inner_t_h20_l2_depth6|8880|True|True|0|True|
|year_2026/inner_t_h20_huber_depth4|8880|True|True|0|True|
|year_2026/inner_t_h40_l2_depth4|7770|True|True|0|True|
|year_2026/inner_t_h40_l2_depth6|7770|True|True|0|True|
|year_2026/inner_t_h40_huber_depth4|7770|True|True|0|True|
|year_2026/inner_t_h60_l2_depth4|6660|True|True|0|True|
|year_2026/inner_t_h60_l2_depth6|6660|True|True|0|True|
|year_2026/inner_t_h60_huber_depth4|6660|True|True|0|True|

OOF目标年只由另一个核心年训练的selector生成。周预测原始四周来源与盘中原来源取最早值，参数训练年与权重SHA另行逐行核对。
较早年份使用较晚核心年训练的参数仍属于回顾性交叉拟合；不能称为当时可部署或全新未接触测试。
小样本内外层上下文保留v4原全池冻结排名，仅按股票过滤，绝不在小池中重排该排名。

审计脚本SHA：c1654c17256cd0dea326a8c71350b14b1ebbad0d42b2e852218d2ad5b4a0814f

|训练域|OOF输入行|保留行|期限排除|整手不足|输入及审计SHA|
|---|---:|---:|---:|---:|---|
|year_2023/h20|12544|10208|0|2336|True|
|year_2023/h40|12544|8932|1568|2336|True|
|year_2023/h60|12544|7656|3136|2336|True|
|year_2024/h20|11808|9648|0|2160|True|
|year_2024/h40|11808|8442|1476|2160|True|
|year_2024/h60|11808|7236|2952|2160|True|
|year_2025/h20|11184|8960|0|2224|True|
|year_2025/h40|11184|7840|1398|2224|True|
|year_2025/h60|11184|6720|2796|2224|True|
|year_2026/h20|11184|8960|0|2224|True|
|year_2026/h40|11184|7840|1398|2224|True|
|year_2026/h60|11184|6720|2796|2224|True|

训练域只根据既有OOF周前收、固定初始槽位预算、滑点、两层整手和确定性同日恢复期限重建；过滤后才连接未来标签并执行严格来源purge。静态整手资格不能保证真实现金/容量或成交。

当前共享annual_portfolio.py整文件SHA与快照不同（current_full_file_match=false）。仅允许已声明精确SHA的run_lot_portfolio并发修复；除该未引用函数外整个模块AST及实际消费费用/指标函数文本均独立核对一致，其他16份源码仍精确匹配。原17份快照未改。
