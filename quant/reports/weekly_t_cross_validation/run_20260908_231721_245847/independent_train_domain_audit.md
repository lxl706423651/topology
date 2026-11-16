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
|year_2023/inner_t_h20_l2_depth4|170376|True|True|0|True|
|year_2023/inner_t_h20_l2_depth6|170376|True|True|0|True|
|year_2023/inner_t_h20_huber_depth4|170376|True|True|0|True|
|year_2023/inner_t_h40_l2_depth4|149079|True|True|0|True|
|year_2023/inner_t_h40_l2_depth6|149079|True|True|0|True|
|year_2023/inner_t_h40_huber_depth4|149079|True|True|0|True|
|year_2023/inner_t_h60_l2_depth4|127782|True|True|0|True|
|year_2023/inner_t_h60_l2_depth6|127782|True|True|0|True|
|year_2023/inner_t_h60_huber_depth4|127782|True|True|0|True|
|year_2024/inner_t_h20_l2_depth4|150568|True|True|0|True|
|year_2024/inner_t_h20_l2_depth6|150568|True|True|0|True|
|year_2024/inner_t_h20_huber_depth4|150568|True|True|0|True|
|year_2024/inner_t_h40_l2_depth4|131747|True|True|0|True|
|year_2024/inner_t_h40_l2_depth6|131747|True|True|0|True|
|year_2024/inner_t_h40_huber_depth4|131747|True|True|0|True|
|year_2024/inner_t_h60_l2_depth4|112926|True|True|0|True|
|year_2024/inner_t_h60_l2_depth6|112926|True|True|0|True|
|year_2024/inner_t_h60_huber_depth4|112926|True|True|0|True|
|year_2025/inner_t_h20_l2_depth4|174936|True|True|0|True|
|year_2025/inner_t_h20_l2_depth6|174936|True|True|0|True|
|year_2025/inner_t_h20_huber_depth4|174936|True|True|0|True|
|year_2025/inner_t_h40_l2_depth4|153069|True|True|0|True|
|year_2025/inner_t_h40_l2_depth6|153069|True|True|0|True|
|year_2025/inner_t_h40_huber_depth4|153069|True|True|0|True|
|year_2025/inner_t_h60_l2_depth4|131202|True|True|0|True|
|year_2025/inner_t_h60_l2_depth6|131202|True|True|0|True|
|year_2025/inner_t_h60_huber_depth4|131202|True|True|0|True|
|year_2026/inner_t_h20_l2_depth4|174936|True|True|0|True|
|year_2026/inner_t_h20_l2_depth6|174936|True|True|0|True|
|year_2026/inner_t_h20_huber_depth4|174936|True|True|0|True|
|year_2026/inner_t_h40_l2_depth4|153069|True|True|0|True|
|year_2026/inner_t_h40_l2_depth6|153069|True|True|0|True|
|year_2026/inner_t_h40_huber_depth4|153069|True|True|0|True|
|year_2026/inner_t_h60_l2_depth4|131202|True|True|0|True|
|year_2026/inner_t_h60_l2_depth6|131202|True|True|0|True|
|year_2026/inner_t_h60_huber_depth4|131202|True|True|0|True|

OOF目标年只由另一个核心年训练的selector生成。周预测原始四周来源与盘中原来源取最早值，参数训练年与权重SHA另行逐行核对。
较早年份使用较晚核心年训练的参数仍属于回顾性交叉拟合；不能称为当时可部署或全新未接触测试。
小样本内外层上下文保留v4原全池冻结排名，仅按股票过滤，绝不在小池中重排该排名。

审计脚本SHA：c1654c17256cd0dea326a8c71350b14b1ebbad0d42b2e852218d2ad5b4a0814f

|训练域|OOF输入行|保留行|期限排除|整手不足|输入及审计SHA|
|---|---:|---:|---:|---:|---|
|year_2023/h20|221496|181600|0|39896|True|
|year_2023/h40|221496|158900|27687|39896|True|
|year_2023/h60|221496|136200|55374|39896|True|
|year_2024/h20|212632|163576|0|49056|True|
|year_2024/h40|212632|143129|26579|49056|True|
|year_2024/h60|212632|122682|53158|49056|True|
|year_2025/h20|213568|176440|0|37128|True|
|year_2025/h40|213568|154385|26696|37128|True|
|year_2025/h60|213568|132330|53392|37128|True|
|year_2026/h20|213568|176440|0|37128|True|
|year_2026/h40|213568|154385|26696|37128|True|
|year_2026/h60|213568|132330|53392|37128|True|

训练域只根据既有OOF周前收、固定初始槽位预算、滑点、两层整手和确定性同日恢复期限重建；过滤后才连接未来标签并执行严格来源purge。静态整手资格不能保证真实现金/容量或成交。

当前共享annual_portfolio.py整文件SHA与快照不同（current_full_file_match=false）。仅允许已声明精确SHA的run_lot_portfolio并发修复；除该未引用函数外整个模块AST及实际消费费用/指标函数文本均独立核对一致，其他16份源码仍精确匹配。原17份快照未改。
