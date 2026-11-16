# 724列候选特征说明

## 组成

724列由原有285列、274列未作截面排名的分钟原因子和165列日线统计组成。
这是候选总数，不是最终入选列数。每折只在实际训练行筛选覆盖率、相关性及冗余，
保留的列序随该模型model_metadata.json保存，不能拿另一模型的列序推断。

## 英文名称解读

| 名称或模式 | 中文含义与公式 |
|---|---|
| cs_rank_后缀 | 同一信号时点、合格周池内的因子截面排名，后缀对应原因子 |
| ctx_raw_后缀 | 原始分钟因子的数值，不是排名；具体后缀公式在quant_backtest/factor_library.py |
| ctx_daily_open/high/low/close_lagK | 截至上一完成日往前K个日记录的开/高/低/收，除以上一完成日收盘，再减1 |
| ctx_daily_vwap_lagK | 往前K日成交额/成交股数得到的成交均价，除以上一完成日收盘，再减1 |
| ctx_daily_volume_lagK | 往前K日成交股数，除以上一完成日成交股数 |
| ctx_daily_momentum_W | 上一完成日收盘/往前W日收盘−1，表示W日动量 |
| ctx_daily_mean_gap_W | 上一完成日收盘/W日平均收盘−1，表示均线偏离 |
| ctx_daily_volatility_W | 过去W个已完成日日收益率的样本标准差 |
| ctx_daily_atr_W | 过去W日真实波幅简单均值/上一完成日收盘 |
| ctx_daily_breakout_W | 上一完成日收盘/过去W日最高价−1；是相对位置，不是未来突破结果 |
| ctx_daily_position_W | （上一完成日收盘−W日最低价）/（W日最高价−W日最低价） |
| ctx_daily_volume_ratio_W | 上一完成日成交股数/W日平均成交股数 |
| ctx_daily_up_fraction_W | 过去W日正收益日的比例，首个未知收益日不当成下跌日填充 |
| ctx_daily_illiquidity_W | 过去W日平均绝对收益/成交额×一亿，表示每亿元成交对应的价格波动强度 |
| ctx_momentum5/20 | 当前已完成分钟价格相对过去5/20日日收盘的涨幅 |
| ctx_ema5_gap / ctx_ema20_gap | 当前价格相对上一完成日5/20日指数均线的偏离 |
| ctx_breakout10 | 当前价格相对过去已完成10日最高价的偏离 |
| ctx_atr_ratio | 上一完成日ATR14/当前价格 |
| ctx_daily_return | 当前价格/当日已知参考前收盘−1 |
| ctx_board | 板块编码：主板0、创业板1、科创板2 |
| ctx_pool_return / ctx_pool_breadth / ctx_pool_momentum20 | 当时合格池的平均当日涨幅、上涨比例及平均20日动量；准确列名以缓存定义表为准 |

K取0至19；K=0也仅指上一已完成日，绝不指尚未收盘的当日。
W取5、10、20、30、60个日记录。停牌/缺失数据原样保留，不用未来日填补。
窗口不足、零分母等无意义值为缺失，由XGBoost原生处理，不在全数据上拟合填充器。
完整列清单见缓存feature_definitions.csv，实际入选状态见各模型model_feature_manifest.csv。

## 目标不是特征

net_return（净收益路径）、label_end（标签结束时间）、actual_target（实际监督目标）
只在独立训练记录和标签PKL中，绝不能作为模型输入。
return目标是净收益裁剪至[-0.5,0.5]后乘100；probability目标是净收益是否大于0；
downside目标同样使用净收益尺度，学习25%分位数。概率与百分比收益的门槛不可混用。
