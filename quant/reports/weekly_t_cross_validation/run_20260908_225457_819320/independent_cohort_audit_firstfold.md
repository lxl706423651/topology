# v6 OOF队列与模型独立审计

请求范围完成：True；全部通过：True。
只读模型、数据和选股来源；不读取外层收益、不训练或重新选参。

|周selector|训练年|目标年|样本SHA|来源SHA|OOF排名及选择|
|---|---|---:|---|---|---|
|year_2023/oof_weekly_cohort/selector_2024|[2025]|2024|True|True|True|
|year_2023/oof_weekly_cohort/selector_2025|[2024]|2025|True|True|True|

|T模型|训练行|样本SHA|来源SHA|隔离重叠|原生权重一致|
|---|---:|---|---|---:|---|
|year_2023/inner_t_h20_l2_depth4|207096|True|True|0|True|
|year_2023/inner_t_h20_l2_depth6|207096|True|True|0|True|
|year_2023/inner_t_h20_huber_depth4|207096|True|True|0|True|
|year_2023/inner_t_h40_l2_depth4|207096|True|True|0|True|
|year_2023/inner_t_h40_l2_depth6|207096|True|True|0|True|
|year_2023/inner_t_h40_huber_depth4|207096|True|True|0|True|
|year_2023/inner_t_h60_l2_depth4|181209|True|True|0|True|
|year_2023/inner_t_h60_l2_depth6|181209|True|True|0|True|
|year_2023/inner_t_h60_huber_depth4|181209|True|True|0|True|

OOF目标年只由另一个核心年训练的selector生成。周预测原始四周来源与盘中原来源取最早值，参数训练年与权重SHA另行逐行核对。
较早年份使用较晚核心年训练的参数仍属于回顾性交叉拟合；不能称为当时可部署或全新未接触测试。
小样本内外层上下文保留v4原全池冻结排名，仅按股票过滤，绝不在小池中重排该排名。

审计脚本SHA：1e4233632c07618977a1b3d6c5b7e7f18ed53f2a83fb087969f532bf9e938948
