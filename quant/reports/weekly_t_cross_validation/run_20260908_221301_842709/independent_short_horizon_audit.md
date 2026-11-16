# 短周期T独立模型与输入审计

审计完成：True；全部检查通过：True。
范围：完整冻结年度范围；本次不读取外层收益、不训练或重新选参。

|模型|训练行|拟合行|样本SHA|来源SHA|隔离重叠|原生预测一致|
|---|---:|---:|---|---|---:|---|
|year_2023/inner_t_h20_l2_depth4|6,226,552|1,500,000|True|True|0|True|
|year_2023/inner_t_h20_l2_depth6|6,226,552|1,500,000|True|True|0|True|
|year_2023/inner_t_h20_huber_depth4|6,226,552|1,500,000|True|True|0|True|
|year_2024/inner_t_h20_l2_depth4|5,613,272|1,500,000|True|True|0|True|
|year_2024/inner_t_h20_l2_depth6|5,613,272|1,500,000|True|True|0|True|
|year_2024/inner_t_h20_huber_depth4|5,613,272|1,500,000|True|True|0|True|
|year_2025/inner_t_h20_l2_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2025/inner_t_h20_l2_depth6|4,890,528|1,500,000|True|True|0|True|
|year_2025/inner_t_h20_huber_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2026/inner_t_h20_l2_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2026/inner_t_h20_l2_depth6|4,890,528|1,500,000|True|True|0|True|
|year_2026/inner_t_h20_huber_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2023/inner_t_h40_l2_depth4|6,226,552|1,500,000|True|True|0|True|
|year_2023/inner_t_h40_l2_depth6|6,226,552|1,500,000|True|True|0|True|
|year_2023/inner_t_h40_huber_depth4|6,226,552|1,500,000|True|True|0|True|
|year_2024/inner_t_h40_l2_depth4|5,613,272|1,500,000|True|True|0|True|
|year_2024/inner_t_h40_l2_depth6|5,613,272|1,500,000|True|True|0|True|
|year_2024/inner_t_h40_huber_depth4|5,613,272|1,500,000|True|True|0|True|
|year_2025/inner_t_h40_l2_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2025/inner_t_h40_l2_depth6|4,890,528|1,500,000|True|True|0|True|
|year_2025/inner_t_h40_huber_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2026/inner_t_h40_l2_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2026/inner_t_h40_l2_depth6|4,890,528|1,500,000|True|True|0|True|
|year_2026/inner_t_h40_huber_depth4|4,890,528|1,500,000|True|True|0|True|
|year_2023/inner_t_h60_l2_depth4|5,448,233|1,500,000|True|True|0|True|
|year_2023/inner_t_h60_l2_depth6|5,448,233|1,500,000|True|True|0|True|
|year_2023/inner_t_h60_huber_depth4|5,448,233|1,500,000|True|True|0|True|
|year_2024/inner_t_h60_l2_depth4|4,911,613|1,500,000|True|True|0|True|
|year_2024/inner_t_h60_l2_depth6|4,911,613|1,500,000|True|True|0|True|
|year_2024/inner_t_h60_huber_depth4|4,911,613|1,500,000|True|True|0|True|
|year_2025/inner_t_h60_l2_depth4|4,279,212|1,500,000|True|True|0|True|
|year_2025/inner_t_h60_l2_depth6|4,279,212|1,500,000|True|True|0|True|
|year_2025/inner_t_h60_huber_depth4|4,279,212|1,500,000|True|True|0|True|
|year_2026/inner_t_h60_l2_depth4|4,279,212|1,500,000|True|True|0|True|
|year_2026/inner_t_h60_l2_depth6|4,279,212|1,500,000|True|True|0|True|
|year_2026/inner_t_h60_huber_depth4|4,279,212|1,500,000|True|True|0|True|

已核对36个核心模型，12份核心训练支持域，4份外层模型复制。
折×周期范围12，实际独特训练范围9。

strict raw price feature boundary passed仅表示实际量价来源至标签末端没有进入排除年份及其前后隔离区间。
此前外层历史结果已被查看，本轮仍为历史复测，不能据此称为全新独立测试或保证未来收益。
累计history_days仅可用于事前股票资格，本审计要求它未被模型选作特征。

审计脚本SHA256：7c0626949916ba285bba20d066640edc0edc91391348aa45ba2616ad5db6ab51
