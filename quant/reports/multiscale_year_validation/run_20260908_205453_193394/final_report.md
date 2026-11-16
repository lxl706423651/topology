# 多尺度模型训练内部研究

> 状态更正：本轮未充分隔离新增60交易日特征的回看边界，不能用于外层验收或实盘模型推荐。下表仅保留最初内部研究记录，不是可靠的独立测试成绩。未启动本轮外层测试；修正后在新目录重训，详见[研究状态说明](研究状态说明.md)。

已冻结四折，尚未评价外层测试，不能报告测试夏普。

多尺度3/5日净收益模型；留一年回顾验证，前三折未来训练；非实盘业绩。

## 2023留出折的内部验证

| method   | rule_id                                              |   mean_sharpe |   closed_lots |   positive_segments |
|:---------|:-----------------------------------------------------|--------------:|--------------:|--------------------:|
| method1  | horizon5_shallow_continuation_negative_atr20_40_q0.9 |    1.25772    |           205 |                   2 |
| method1  | horizon3_shallow_continuation_negative_atr20_40_q0.8 |    0.138385   |           619 |                   2 |
| method1  | horizon5_shallow_continuation0_atr20_40_q0.9         |    0.0412965  |           207 |                   2 |
| method2  | horizon5_shallow_continuation_negative_atr20_40_q0.9 |    1.0857     |           209 |                   2 |
| method2  | horizon3_shallow_continuation_negative_atr20_40_q0.8 |    0.786944   |           631 |                   2 |
| method2  | horizon5_shallow_continuation0_atr20_40_q0.9         |    0.00166474 |           212 |                   2 |

- method1（基础模型）：内部验证平均正夏普、交易数及正收益片段数达标，规则horizon5_shallow_continuation_negative_atr20_40_q0.9。
- method2（趋势叠加模型）：内部验证平均正夏普、交易数及正收益片段数达标，规则horizon5_shallow_continuation_negative_atr20_40_q0.9。
## 2024留出折的内部验证

| method   | rule_id                                              |   mean_sharpe |   closed_lots |   positive_segments |
|:---------|:-----------------------------------------------------|--------------:|--------------:|--------------------:|
| method1  | horizon3_deeper_continuation0_atr30_60_q0.9          |     -0.598199 |           122 |                   1 |
| method1  | horizon3_deeper_continuation0_atr20_40_q0.9          |     -0.814723 |           130 |                   1 |
| method1  | horizon5_shallow_continuation0_atr30_60_q0.8         |     -3.79682  |           210 |                   0 |
| method2  | horizon3_deeper_continuation0_atr30_60_q0.9          |     -0.70683  |           125 |                   1 |
| method2  | horizon3_deeper_continuation0_atr20_40_q0.9          |     -0.771417 |           135 |                   1 |
| method2  | horizon5_shallow_continuation_negative_atr20_40_q0.8 |     -2.46708  |           187 |                   0 |

- method1（基础模型）：内部验证不达标，按事前协议保持现金，非漏跑测试，规则cash_validation_failed。
- method2（趋势叠加模型）：内部验证不达标，按事前协议保持现金，非漏跑测试，规则cash_validation_failed。
## 2025留出折的内部验证

| method   | rule_id                                     |   mean_sharpe |   closed_lots |   positive_segments |
|:---------|:--------------------------------------------|--------------:|--------------:|--------------------:|
| method1  | horizon5_deeper_next_day_control_q0.9       |      -1.63798 |            87 |                   2 |
| method1  | horizon5_deeper_continuation0_atr20_40_q0.9 |      -4.40182 |            87 |                   1 |
| method1  | horizon5_deeper_continuation0_atr30_60_q0.9 |      -4.40182 |            87 |                   1 |
| method2  | horizon5_deeper_next_day_control_q0.9       |      -1.63798 |            87 |                   2 |
| method2  | horizon5_deeper_continuation0_atr20_40_q0.9 |      -4.40182 |            87 |                   1 |
| method2  | horizon5_deeper_continuation0_atr30_60_q0.9 |      -4.40182 |            87 |                   1 |

- method1（基础模型）：内部验证不达标，按事前协议保持现金，非漏跑测试，规则cash_validation_failed。
- method2（趋势叠加模型）：内部验证不达标，按事前协议保持现金，非漏跑测试，规则cash_validation_failed。
## 2026留出折的内部验证

| method   | rule_id                                              |   mean_sharpe |   closed_lots |   positive_segments |
|:---------|:-----------------------------------------------------|--------------:|--------------:|--------------------:|
| method1  | horizon3_shallow_continuation_negative_atr20_40_q0.5 |      0.810799 |           777 |                   3 |
| method1  | horizon3_shallow_continuation_negative_atr20_40_q0.8 |      0.810799 |           777 |                   3 |
| method1  | horizon5_shallow_continuation_negative_atr20_40_q0.5 |     -0.358697 |           711 |                   0 |
| method2  | horizon3_shallow_continuation_negative_atr20_40_q0.5 |      1.21142  |           783 |                   2 |
| method2  | horizon3_shallow_continuation_negative_atr20_40_q0.8 |      1.21142  |           783 |                   2 |
| method2  | horizon3_shallow_continuation_negative_atr20_40_q0.9 |      1.60136  |           603 |                   3 |

- method1（基础模型）：内部验证平均正夏普、交易数及正收益片段数达标，规则horizon3_shallow_continuation_negative_atr20_40_q0.5。
- method2（趋势叠加模型）：内部验证平均正夏普、交易数及正收益片段数达标，规则horizon3_shallow_continuation_negative_atr20_40_q0.9。

## 限制

- 当前信号另要求非一字线且成交额至少200000元；这是当前可见资格，不保证下一分钟可成交。
- 多日开仓目标为3或5交易日净收益，持仓模型仍预测继续一天增益；比较最早满足T+1的时间退出对照与动态退出，不是固定价位止盈。
- 增加的多日因子由过去共同时钟收盘计算；60交易日窗口不是60分钟。当期环境仅统计当时合资格周池，不使用未来入池股票。
- 留一年回顾检验；2023至2025含未来年份训练，不是因果向前测试；2026仅1至4月。
- 每轮外层测试年没有进入该轮拟合/校准/验证，但前三轮的训练数据在测试年之后；年份之间角色轮换不是四个完全独立实验。
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

## 研究依据

[Gu、Kelly与Xiu的机器学习资产定价研究](https://www.nber.org/papers/w25398)支持考察动量、流动性与波动状态，但不证明本策略在A股有效或能达到夏普3。
