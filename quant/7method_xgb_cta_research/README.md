# 扩展特征XGBoost CTA研究

本轮已完成训练及回测，**2026净收益2.32%、夏普0.971，未达到2**。
结果、交易规则及年度口径见[results.md](results.md)，不能用2024样本内高收益替代测试。

## 架构与边界

XGBoost是梯度提升树模型；这里的CTA指股票趋势/突破/回调择时，不是期货多空CTA。
盈利概率模型预测一笔按既定退出规则完成的交易是否扣费盈利；下行分位数模型预测较保守的收益位置，
25%分位数不是25%的胜率，也不等于得到保证的最大亏损。

特征构建、模型训练、开仓信号、独立批次退出、账户撮合和报告分层。
复用第六方法已审计的执行模块，但模型、特征缓存与报告独立，不覆盖既有策略。
本轮最终模型只用2023—2025年；2026年截至4月30日为冻结后的单次历史测试。
历史2026已经被其他研究看到，因此不是从未曝光的独立测试。
测试目标为扣费日收益夏普大于2，不承诺能实现；禁止看到2026结果再调参。

## 文件角色

| 文件 | 角色 |
|---|---|
| README.md | 本架构、调用流程及限制说明 |
| results.md | 本轮真实结果、精确选中规则、年度评价区别及未达标结论 |
| research.md | 研究来源及本地改编说明 |
| feature_guide.md | 724列候选特征的名称解读、公式、滞后和单位 |
| ../config/xgb_cta_expanded.yaml | 用户研究输入，继承第六方法执行合同 |
| ../scripts/run_xgb_cta_expanded.py | 唯一可执行入口，支持缓存准备与小样本测试 |
| ../quant_backtest/cta_expanded_features.py | 特征构造库，原因子与上一完成日日线统计 |
| ../quant_backtest/cta_expanded_training.py | 三类模型比较、冻结、年度回测和报告编排 |
| ../tests/test_cta_expanded.py | 未来扰动、时间隔离及接口单元测试 |
| ../scripts/audit_xgb_cta_expanded.py | 独立重算实际训练行/特征/目标/权重、日夏普及历史状态资格 |
| ../scripts/diagnose_xgb_cta_expanded.py | 预定的固定信号双倍滑点、20日块抽样诊断，不反馈选参 |
| ../reports/xgb_cta_expanded_cache/ | 生成的PKL缓存、特征定义和输入指纹，非用户输入 |
| ../reports/xgb_cta_expanded/run_*/ | 每次新增的候选模型、完整交易账本、曲线及审计 |

## 调用顺序

在quant根目录使用quant环境：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgb_cta_expanded.py --prepare-only
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgb_cta_expanded.py --prepare-only
/home/lxl/anaconda3/envs/quant/bin/python -m pytest -q tests/test_cta_expanded.py tests/test_cta_research.py
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgb_cta_expanded.py --smoke
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgb_cta_expanded.py
```

1. 准备阶段读取旧原因子及历史日线，生成原因子值、已有截面秩、20期价量序列及多尺度日统计；第二次检查复用同一个PKL。
2. 标签阶段把报价截断于2026年之前，模拟T+1、下一分钟执行及独立退出后的净收益；每个训练折还删除未在该折截止前完成的标签。
3. 内部比较固定为2023第三季度、2024全年、2025全年，分别只用各区间之前训练。比较净收益回归、盈利概率、25%下行分位数，以及两种模型复杂度/两套退出/三类CTA/两种频率/两档门槛。
4. 候选评分为内部平均夏普减0.25倍段间标准差，累计完整批次至少20才合格；不足时显示未晋级诊断，不把全现金当成功。预注册有限候选全部保留，不在测试集上无止境寻找达标规则。
5. 唯一方案用2023—2025全量重训；年度过去训练的辅助模型也全部先冻结。2026不用于早停、筛选或分数校准。
6. 同一最终模型回放2023、2024、2025，明确为样本内；单次测试2026。额外向前年份使用此前已完成内部折选规则；2023向前仅第四季度，不能冒充全年。
7. 生成年度重置现金报告与跨年不强制平仓的连续账户，复核费用/T+1/仓位及原生模型读取。根目录收益图和test文件固定代表2026，不挑最好年份替代。

正式结果出来后以实际run目录替换下列路径，仅运行审计和预先约定的固定诊断：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/audit_xgb_cta_expanded.py --report-dir reports/xgb_cta_expanded/run_实际轮次
/home/lxl/anaconda3/envs/quant/bin/python scripts/diagnose_xgb_cta_expanded.py --report-dir reports/xgb_cta_expanded/run_实际轮次
```

诊断新增fixed_diagnostics子目录，双倍滑点账户不重训、不改原信号；重复调用拒绝覆盖。
夏普20日块抽样区间不是DSR或独立性证明。审计输出独立带时间戳JSON，不改已冻结模型。

## 交易合同

每周按上一完整周的高波动前300股票更新候选池；只在有历史正常状态的数据子集研究。
历史ST禁止新买；已有批次在持仓期间变成ST则触发风险退出，在正常、可卖且非涨跌停状态下减仓，
并非把ST继续纳入开仓池。未知或非正常交易状态不能撮合买卖，也不能用当前名称倒推历史资格。
趋势条件要求20日动量及相对过去EMA20为正；trend再要求5日动量、EMA5缺口为正；
breakout要求突破过去10日高点；pullback要求EMA5缺口0至2%、5日动量−5%至5%。
模型预测还必须超过冻结门槛。每个信号时点最多前5，按分数降序；每批预算是决策时总资产10%，
单股20%、总仓80%，无融资无卖空。股票尚有独立批次时可在下一交易日加仓，同股当日不重复开仓。

每批有独立成本、高水位和退出参数。初始保护线为入场价减初始ATR倍数；
随后保护线只上移至已知高水位减吊顶ATR倍数；可选跌破过去EMA20退出和最大交易分钟期限。
不设固定百分比止盈；不因不再属于目标股票而退出。先按旧保护线检查分钟低点，再更新高点并检查收盘，避免假设分钟高低点先后顺序。
T+1未可卖、停牌或涨跌停时退出请求等待，不能保证止损价格成交。未实现用持仓状态训练的学习型退出。

佣金双边万八、免五；过户费万分之0.1；卖出印花税在2023-08-28前千一、之后万五；
滑点万五。沿用既有研究费用假设，不声称与任何具体券商合同完全一致。
每分钟成交额上限5%；信号滞后一个分钟bar，未模拟真实盘口排队。

## YAML字段

以下均必填，默认值为随附YAML；消费入口是run_xgb_cta_expanded.py，特征字段传给特征库、其他字段传给训练库。

基础配置的`fit_deployment_model`（第六方法全量候选开关）、`test_years`（旧向前折列表）、
`reuse_factor_cache`（旧缓存重建入口）不由本入口消费。第七方法不生成含2026拟合的部署模型，
年度边界由本轮protocol.json固定；不能仅凭合并配置中保留的旧字段推断实际训练范围。

| 字段 | 默认值 | 作用 |
|---|---|---|
| base_config | config/xgb_cta_local_history.yaml | 继承费用、资金、周池和数据合同；其余字段解释见6method_xgb_cta/configuration.md |
| source_cache | 已审计976c434161dbad09923d | 原因子及执行行情来源，不使用旧预测 |
| expanded_cache / output_dir | reports/xgb_cta_expanded_cache / reports/xgb_cta_expanded | 新缓存与研究输出根目录 |
| daily_lags / daily_windows | 20 / [5,10,20,30,60] | 已完成日价量序列长度及滚动窗口 |
| max_features / feature_max_correlation | 512 / 0.95 | 仅训练期筛选的数量上限与冗余上限，不要求凑足数量 |
| objectives | return,probability,downside | 净收益、盈利概率、25%下行分位数 |
| entry_minutes_candidates | [24,120] | 开仓检查频率，退出每分钟检查 |
| entry_styles | trend,breakout,pullback | 趋势、突破、回调条件 |
| net_return_gates / probability_gates | [0,0.003] / [0.50,0.55] | 回归/分位数与概率各自门槛 |
| sharpe_target | 2 | 2026实际扣费日收益夏普数值目标 |
| validation_min_closed_lots | 20 | 内部合计已完整平仓批次下限 |
| workers | 12 | 特征线程与模型线程数量 |
| model_profiles | shallow / regularized | 各候选name名称、n_estimators树数、max_depth深度、min_child_weight最小叶权重、reg_lambda正则系数 |
| exit_profiles | 两套ATR退出 | name名称、initial_atr初始倍数、trailing_atr吊顶倍数、max_days交易分钟折算天数、trend_exit是否启用趋势失效退出 |

## 必须保留的限制

本地历史状态仅覆盖755只，749主板、创业板/科创板各3只，历史曾入周池722只；不能称为全市场。
原因子及报价未完整处理分红送转、红利税和特殊交易状态，禁止直接将研究模型接实盘。
绝对预测门槛跨全量重训存在分布漂移，概率未校准；下行分位数也不保证每笔风险覆盖。
多候选内部选择仍有过拟合风险；年度样本内高收益不是泛化证据。
研究输出含每笔成交与批次配对记录，未平仓不算完整胜负；标准图注明策略名、夏普、收益、年化、波动、回撤、手数、胜率。
