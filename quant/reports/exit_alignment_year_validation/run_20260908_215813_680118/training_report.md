# 退出期限训练内部研究

已冻结四折。没有计算本轮外层预测或收益，也没有重训模型。

退出候选比较原持仓模型、预测期限同刻退出、到期后启用持仓模型和最早T+1对照；全部保留ATR20/40保护。

## 2023留出折的内部验证

| 方法代码（method1基础、method2趋势叠加）   | 规则代码                                       |   内部平均夏普 |   完整平仓批次 |   正收益片段 |
|:-------------------------------------------|:-----------------------------------------------|---------------:|---------------:|-------------:|
| method1                                    | horizon5_shallow_minimum_horizon_negative_q0.8 |        1.93773 |            325 |            2 |
| method1                                    | horizon5_shallow_matched_horizon_q0.8          |        1.25878 |            370 |            2 |
| method1                                    | horizon5_shallow_minimum_horizon_zero_q0.8     |        1.3199  |            373 |            2 |
| method2                                    | horizon5_shallow_matched_horizon_q0.8          |        1.40402 |            367 |            2 |
| method2                                    | horizon5_shallow_minimum_horizon_negative_q0.8 |        2.08588 |            327 |            2 |
| method2                                    | horizon5_shallow_minimum_horizon_zero_q0.8     |        1.48072 |            362 |            2 |

- method1：内部验证平均正夏普、交易数及正收益片段数达标，horizon5_shallow_minimum_horizon_negative_q0.8。
- method2：内部验证平均正夏普、交易数及正收益片段数达标，horizon5_shallow_minimum_horizon_negative_q0.8。
## 2024留出折的内部验证

| 方法代码（method1基础、method2趋势叠加）   | 规则代码                                       |   内部平均夏普 |   完整平仓批次 |   正收益片段 |
|:-------------------------------------------|:-----------------------------------------------|---------------:|---------------:|-------------:|
| method1                                    | horizon3_shallow_model_control_zero_q0.8       |       0.696001 |            222 |            1 |
| method1                                    | horizon5_shallow_model_control_negative_q0.9   |      -2.18476  |            117 |            1 |
| method1                                    | horizon3_shallow_model_control_negative_q0.8   |       0.679742 |            169 |            1 |
| method2                                    | horizon3_shallow_model_control_zero_q0.8       |       0.790185 |            214 |            1 |
| method2                                    | horizon3_shallow_minimum_horizon_negative_q0.9 |       1.43907  |             97 |            2 |
| method2                                    | horizon3_shallow_model_control_negative_q0.8   |       0.737855 |            167 |            1 |

- method1：内部验证不达标，按事前协议保持现金，非漏跑测试，cash_validation_failed。
- method2：内部验证平均正夏普、交易数及正收益片段数达标，horizon3_shallow_minimum_horizon_negative_q0.9。
## 2025留出折的内部验证

| 方法代码（method1基础、method2趋势叠加）   | 规则代码                                      |   内部平均夏普 |   完整平仓批次 |   正收益片段 |
|:-------------------------------------------|:----------------------------------------------|---------------:|---------------:|-------------:|
| method1                                    | horizon3_deeper_matched_horizon_q0.8          |       -2.41859 |            242 |            0 |
| method1                                    | horizon3_deeper_minimum_horizon_negative_q0.8 |       -2.50278 |            236 |            0 |
| method1                                    | horizon3_deeper_minimum_horizon_zero_q0.8     |       -2.57613 |            235 |            0 |
| method2                                    | horizon3_deeper_matched_horizon_q0.8          |       -2.24751 |            238 |            0 |
| method2                                    | horizon3_deeper_minimum_horizon_negative_q0.8 |       -2.33409 |            231 |            0 |
| method2                                    | horizon3_deeper_minimum_horizon_zero_q0.8     |       -2.40566 |            235 |            0 |

- method1：内部验证不达标，按事前协议保持现金，非漏跑测试，cash_validation_failed。
- method2：内部验证不达标，按事前协议保持现金，非漏跑测试，cash_validation_failed。
## 2026留出折的内部验证

| 方法代码（method1基础、method2趋势叠加）   | 规则代码                                     |   内部平均夏普 |   完整平仓批次 |   正收益片段 |
|:-------------------------------------------|:---------------------------------------------|---------------:|---------------:|-------------:|
| method1                                    | horizon3_deeper_matched_horizon_q0.8         |      -0.234837 |            384 |            2 |
| method1                                    | horizon3_shallow_model_control_negative_q0.5 |       0.810799 |            777 |            3 |
| method1                                    | horizon3_shallow_model_control_negative_q0.8 |       0.810799 |            777 |            3 |
| method2                                    | horizon3_deeper_matched_horizon_q0.8         |      -0.263048 |            381 |            2 |
| method2                                    | horizon3_shallow_model_control_negative_q0.5 |       1.21142  |            783 |            2 |
| method2                                    | horizon3_shallow_model_control_negative_q0.8 |       1.21142  |            783 |            2 |

- method1：内部验证平均正夏普、交易数及正收益片段数达标，horizon3_shallow_model_control_negative_q0.5。
- method2：内部验证平均正夏普、交易数及正收益片段数达标，horizon3_shallow_model_control_negative_q0.5。

## 代码说明

model_control：原模型退出；matched_horizon：预测期限同刻退出；minimum_horizon：到期后启用模型；next_day_control：最早T+1对照；zero/negative：继续收益门槛为0/-0.10%；horizon3/5：预测3/5交易日；shallow/deeper：浅树/深树；q：校准分位。

## 限制

- 本轮没有重训或选用父实验外层胜者：全部开仓档位与持仓模型原样复用，只在内部校准/验证选择退出。
- 当前信号另要求非一字线且成交额至少200000元；这是当前可见资格，不保证下一分钟可成交。
- 多日开仓目标为3或5交易日净收益，持仓模型仍预测继续一天增益；比较最早满足T+1的时间退出对照与动态退出，不是固定价位止盈。
- 增加的多日因子由过去共同时钟收盘计算；60交易日窗口不是60分钟。当期环境仅统计当时合资格周池，不使用未来入池股票。
- 留出年后60个交易日不进入拟合、校准或验证，避免长窗口特征借用留出年价格；因此2025留出折的2026核心拟合可能为空，但2026仍参与其他折拟合及本折后段选参。
- future_trained记录核心模型是否拟合未来年份；future_selection记录是否使用未来年份校准/验证。2025折即使没有未来核心拟合，仍因未来选参属于回顾研究。
- 60日隔离覆盖新增有界窗口；旧分钟因子含EMA/RSI等递推状态，长期缺行情股票的跨段记忆未逐一重置，不能声称全部特征与留出年行情严格独立。
- 每个训练年按交易日分为60%拟合、20%校准、20%验证，分段起点隔离7自然日；标签和虚拟批次入场不得跨越拟合边界。模型冻结后不再重拟合，避免分数阈值漂移。
- 持仓模型预测继续一天相对立即卖出的收益，不是最优停止证明；虚拟批次不模拟入场资金和止损存活，存在训练与真实持仓分布差异。
- 任一年收益无波动则四年平均夏普未定义且目标不达标；不能删除空仓/亏损年后声称平均大于3。全部年份都有有效夏普时按四年等权算术均值评价。
- 历史数据已被此前研究查看；本轮是预先固定流程的历史稳健性检查，不是全新未见样本。
- 原始行情无完整历史ST状态、退市总表、复权因子和公司行动。使用2026年名称快照排除ST/退，仍有前视筛选及幸存者偏差；原始价格的分红拆股也未完整记账。
- 交易只在信号之后的下一分钟收盘模拟成交。使用双边万八佣金、免5、日期对应印花税、双边十万分之一过户费和双边万五滑点。
- 每次成交额最多占下一分钟成交额5%，零量/缺失不成交；主板前收盘±9.5%是保守阻成交代理，不等于完整涨跌停、集合竞价及排队模型。停牌持仓用最后可见价格估值，无法反映退市损失。
- 止损在24分钟采样点和日终检查，不是逐分钟监控；T+1或无法卖出时保存待平仓状态，不保证以止损价成交。入场后的首个监控区间保守只用当前收盘更新最高价，避免引入买入前的分钟内高点。
- 各年从相同现金独立启动，年末持仓按市值保留，不虚构强制成交。跨年拼接只是独立年度收益的统计连接，不是连续持仓实盘仿真。
- 方法二沿用方法一的XGBoost核心，仅增加CTA叠加，不能假定两者为独立收益来源。
- 期限按实际交易日及各批次买入信号时刻计算；ATR可提前退出，缺失行情、T+1及容量可延迟成交。固定预测期限不保证固定实际持仓时长。
- 各对照的拟议开仓分数和门槛相同，但退出改变现金和存活持仓，因此实际买单可以不同；这是组合政策比较，不是强行固定每笔成交的反事实。


[信号期限与交易成本的研究依据](https://www.nber.org/papers/w15205)并不证明本A股退出规则有效。