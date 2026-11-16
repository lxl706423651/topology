# 做 T 择时能力改进说明（v11，已完成回测）

本轮研究把目标从提高交易次数改为识别能够覆盖费用的价格变动，并区分反转和趋势条件。**正式回测已完成：86组T合计净亏573.72元，每组平均净亏6.67元，本轮没有通过择时能力改善验证。**旧 v10 每组平均净亏5.85元；交易减少带来的总亏损收窄，不能当作更准确的择时。旧结论保留，见[做T频率改进说明](做T频率改进说明.md)。

本轮仍使用同一折原冻结周度名单、70%收益排名＋30%振幅排名、共同底仓、佣金与滑点以及 T+1 执行规则。新增训练的是 T 的原始价格收益与扣费事件模型；不使用旧残差 T 权重，不改变周度选股名单，也不按外层结果回选模型或规则。

## 实际结果与交易质量

正式目录为 [run_20260909_182945_388303](../reports/weekly_t_cross_validation/run_20260909_182945_388303/final_report.md)，另见[精确年度摘要](../reports/weekly_t_cross_validation/run_20260909_182945_388303/annual_summary.csv)、[年度曲线](../reports/weekly_t_cross_validation/run_20260909_182945_388303/test_return_curve.png)和[独立交易质量诊断](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/report.md)。以下为原保存账户结果；完整模型及原始行情独立审计正在进行，尚不提前标为通过。

| 留出年 | 原底仓收益 | 加入v11收益 | T净增量（元） | 完整组/恢复片段 | 净胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | −11.6656% | −11.6640% | +16.10 | 4/4 | 75.00% |
| 2024 | −16.4992% | −16.5436% | −444.94 | 59/59 | 28.81% |
| 2025 | +31.4697% | +31.4589% | −107.85 | 18/19 | 33.33% |
| 2026截至4月30日 | +8.2277% | +8.2240% | −37.03 | 5/5 | 20.00% |

四折均启用且有实际成交，没有持有折。每年独立100万元，净增量合计−573.7200510555413元，不是跨年复利；仅1/4年度增量为正。周增量bootstrap 95%区间为[−1178.95,−76.89]元，也没有达到正增益参考。2023的75%净胜率只有4组，不能据此声称稳定。

| 完整配对口径 | v10频率方案 | v11择时方案 |
| --- | ---: | ---: |
| 完整组/恢复片段 | 1760/1829 | 86/87 |
| 扣费净胜率 | 26.88%（473胜） | 31.40%（27胜） |
| 平均每组参考价毛差 | +0.20元 | −1.52元 |
| 平均每组费用＋滑点 | 6.05元 | 5.15元 |
| 平均每组净收益 | −5.85元 | −6.67元 |
| 完整组参考毛差合计 | +352.00元 | −131.00元 |
| T滑点/费用合计 | 3337.41/7305.97元 | 141.07/301.65元 |
| 账户T净增量合计 | −10291.38元 | −573.72元 |

两方案四个年度的共同底仓净值逐日完全相同。v11的完整组净额为参考毛差−131.00元−滑点141.0735元−费用301.646551055元=−573.720051054985元，与账户差仅有浮点误差。**净胜率略高，却没有更好的每组价差或净收益；这次减亏主要伴随交易从1760组降为86组。**不同版本交易集合和规模不同，上述对比是描述性结果，不是对同一交易机会随机分组后的因果证明。

按真实首腿连接原保存决策后，20/40/60分钟分别成交8/40/38组，净额−148.97/−208.08/−216.67元，20分钟8组均净亏。买先64组净−389.42元，卖先22组净−184.30元。完整[方向、恢复周期、信号时刻分组](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/grouped_pair_quality.csv)保留；不能看到这些结果后取消某个时段或周期再冒称外层验证。

已成交86组的预测净边际等权均值为+65.08bp，而实际相对各组首腿参考名义额的净收益等权均值为−42.95bp，说明这些被选择机会的预期价格优势没有兑现。概率0.6—0.7区间有22组、仅2组净赢，合计−330.45元；0.7—0.8区间只有10组、4组净赢、净额+23.24元，不能概括为“预测越高必然越差”。详见[预测边际与实际结果分箱](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/prediction_realization.csv)。这些概率原本未经校准，而且固定端点标签与分批/延迟恢复后的实际组净额语义不同；此处只是已成交样本的描述，不能把差值命名为严格概率校准误差或实盘胜率承诺。

内部选中的规则全部为反转条件，四折内部仅完成2/3/3/28组，净额分别+13.16/+33.33/+7.66/+51.73元；前三折的训练内部盈利证据尤其稀疏。这个事实与外层价差未兑现共同限制结论，不能据外层亏损重新提高或降低开启门槛。完整冻结规则保存在各年 `selected_rule.json`。

保存账本显示本轮T库存差异、恢复失败、期末待恢复、T+1违规、共同底仓计划差异均为0，86组全部当日完成。但原周度底仓退出受阻仍有171个年度估值日/307股日；2025年末3笔及2026评估期末（4月30日）4笔持仓仍保留，其中陈旧估值9750元。T恢复成功不等于周度退出无风险。交易质量诊断仅核对保存事件、决策、预测和配对的关系；独立原始行情审计的最终状态将另行补充。

## 研究依据与适用范围

下表区分论文的实际结论与本项目提出的可检验假设。美国市场、日频或理想扩散模型的结果，不能直接移植为 A 股20分钟交易的收益或胜率。以下均为论文、作者网站或大学资料页；其中 Gao 等及中国市场论文这里只读取机构摘要，不声称已核查全文所有样本和成交细节。

| 一手来源及日期 | 来源实际支持的结论 | 本项目可检验的假设与不能外推之处 |
| --- | --- | --- |
| Nagel，*Evaporating Liquidity*，RFS 2012年；作者稿2011年12月。[作者全文](https://bpb-us-w2.wpmucdn.com/voices.uchicago.edu/dist/f/575/files/2020/07/LiqSupply.pdf)、[作者研究页](https://voices.uchicago.edu/stefannagel/research/) | 短期反转可以反映提供流动性和承担库存风险的补偿，且随市场状态变化。过去价格变化也可能包含信息，不能都视作暂时偏离。 | 用当时可见的市场收益、截面分散度、成交额进度与个股相对收益区分状态；OHLCV不能直接识别知情订单，也不能把美国日频风险补偿当作A股日内必然回归。 |
| Heston、Korajczyk、Sadka，*Intraday Patterns in the Cross-section of Stock Returns*，JF 2010年8月；稿件2010年5月26日。[论文全文](https://www.bauer.uh.edu/departments/finance/documents/Heston_Korajczyk_Sadka_paper_UH.pdf)、[大学研究页](https://www.kellogg.northwestern.edu/academics-research/research/detail/2010/intraday-patterns-in-the-cross-section-of-stock-returns/) | 研究区分买卖价跳动、短暂深度冲击引起的短期反转及跨日同一时段的延续现象；成本影响可获利性。 | 比较20/40/60交易分钟和信号时刻，避免只用一种恢复期限。分钟OHLCV缺少bid/ask和队列，不能证明观察到的反转是可成交价差。 |
| Gao、Han、Li、Zhou，*Market intraday momentum*，JFE 2018年8月，DOI 10.1016/j.jfineco.2018.05.009。[大学论文页](https://profiles.wustl.edu/en/publications/market-intraday-momentum/) | 美国ETF样本中首半小时收益对末半小时收益具有预测关系，在高波动、高成交量及部分宏观状态更强。 | 增加当前趋势方向、市场共同变动和量能信息，而非对所有上涨都卖出。论文的开收盘窗口不能直接套用本项目的午休、14:30恢复期限或A股个股。 |
| de Groot、Huij、Zhou，*Another Look at Trading Costs and Short-Term Reversal Profits*，JBF 2012年；接受稿2011年7月，DOI 10.1016/j.jbankfin.2011.07.015。[作者机构全文](https://repub.eur.nl/pub/25718/AnotherLook_2011.pdf) | 较高交易成本可吞噬反转毛收益；股票流动性和控制换手会显著影响净结果。论文也讨论等待退出以减少换手。 | 直接预测扣费事件，并先要求预测价格变动覆盖成本。保持原费用，不能把流动性筛选自动当成更低佣金；论文的多日退出不能绕过每日库存恢复合同。 |
| Kitapbayev、Leung，*Mean Reversion Trading with Sequential Deadlines and Transaction Costs*，2017年7月11日首发，2018年1月稿。[原论文](https://arxiv.org/pdf/1707.03498)、[版本记录](https://arxiv.org/abs/1707.03498) | 在给定随机过程、成本和截止期假设下，入场、等待及退出应共同考虑；并非所有偏离都值得交易。 | 允许不入场，并比较当时仍可恢复的固定期限。本轮没有实现论文的连续最优边界，也没有动态止盈止损；理想模型不处理真实A股涨跌停排队。 |
| Chu、Gu、Zhou，*Intraday momentum and reversal in China's stock markets*，Finance Research Letters 30，大学库标注2019年12月31日，DOI 10.1016/j.frl.2019.04.002。[大学原始论文资料页](https://engagedscholarship.csuohio.edu/bus_facpub/288/) | 摘要同时报告日内动量与反转，并指出交易成本阻止套利、可解释现象持续。 | 中国市场也不能统一假设“涨了就回落”；同时检验反转与趋势条件及净成本。仅据摘要不能断言本文具体20分钟参数有效，或低成本即可稳定盈利。 |

这些来源支持提出研究问题，不替代本地真实扣费回测。下述阈值、特征公式和训练方案是事前固定的工程假设，不是论文给出的A股交易参数。

## 训练与策略流程

1. 复用同一折的原周度选股和核心年互为样本外的周度前100队列。先在完整当前股票截面计算新特征，再取已冻结名单，避免把选股子集误当全市场。
2. 每个20/40/60交易分钟周期分别执行原静态训练域过滤、未来标签接入和严格年度来源隔离。训练域只看已知周末参考价算出的整手库存，以及由当前信号推出的恢复时钟；不代表真实动态现金或容量保证，推断不因此删掉高价股票。
3. 只在核心两年拟合两个深度配置的模型。每个周期同时拟合原始收益回归和扣费盈利三分类，共2配置×3周期，每组合两个输出头。最新训练年用于比较实际内部账户净收益，T模型不在全部三年重新拟合。
4. 每折固定比较24个主动候选及一个真实持有候选。每个主动候选对应2配置×3状态条件×2额外净边际×2概率门槛，周期由每个当前信号在20/40/60中选择，**不是再将24乘3**。
5. 内部候选必须实际完成至少一组T、实际净增量为正，且库存偏差、恢复失败、期末待恢复、T+1违规、共同底仓计划差异五项均为0；合格者按内部净增量降序、候选编号升序选择。若没有合格主动候选，明确选择持有。没有100组、日均频率、置信区间或集中度的开启门槛。
6. 全部目标年度规则和模型身份冻结后，统一评价外层年度；不能看到某年收益后修改其方向、模型、概率阈值或是否开启。

核心训练样本从最早原始特征来源到标签结束的整个区间，均不得与排除年度及前后7日相交。周度上下文追踪前四个真实观测周，盘中特征还包含前日及同刻市场来源；停牌间隔不能粗略替换成28个自然日。累计 `f_history_days` 只作已知资格，不进入模型。OOF周模型训练年份的参数来源也独立记录，不能借其他外层折的模型。

2023、2024、2025留一年训练包含后续年份，是回顾性迁移检验；2026折训练截止于2025。但这些历史年份已经参与多轮研究，本轮不是重新获得一份从未看过的测试集。模型隔离和真实核账通过，也不能消除多次研究选择偏差。

## 预测什么、如何决定方向和恢复时间

回归目标为下一交易分钟执行参考价至预定恢复参考价的原始收益 `r=P1/P0−1`，上涨为正。三分类目标为：`0=两方向均不盈利`、`1=买先标称扣费净收益>0`、`2=卖先标称扣费净收益>0`。

令 `s` 为单边滑点，`b=佣金+过户费`，`t` 为卖出日适用印花税，以下是相对首腿未滑点参考价格的净额：

```text
买先： (1+r) × (1−s) × (1−b−t) − (1+s) × (1+b)
卖先： (1−s) × (1−b−t) − (1+r) × (1+s) × (1+b)
```

实际成交仍逐笔计算费用和滑点。这些标签是假设固定端点可成交的经济结果，没有学习涨跌停排队、部分恢复或现金不足的成交概率。`prob_buy_profit`、`prob_sell_profit` 未经概率校准，不能把0.5门槛解释为真实实盘胜率至少50%。分类没有类重加权或重采样；缺少未来标签的推断行保留。

每个信号的原始收益预测先按上述公式转成两个方向的净边际，再要求额外至少5或10bp，以及对应盈利事件概率至少0.4或0.5。常规2023-08-28后、参考价不变时，原费用加滑点已约31.2bp；额外5/10bp是在覆盖这些成本之后，不是总成本只有5/10bp。

三个状态候选如下，数值均是本轮事前假设：

| 状态 | 当前已知条件 |
| --- | --- |
| `learned` | 由两个模型输出选择，不另要求显式反转或趋势形态；仍有共同风险、支持域、成本和概率条件。 |
| `reversal` | 买先要求低于VWAP且最近5分钟已转正，卖先相反；趋势效率≤0.65、VWAP波动标准化偏离绝对值≥0.75。 |
| `momentum` | 买先要求高于VWAP、5/15分钟和开盘以来收益同为正，卖先相反；趋势效率≥0.55。 |

全部状态要求当时可见当日收益绝对值≤3%、已知日内振幅≤6%，以及原训练支持域和恢复方向未锁定条件。支持域仅由各周期严格核心训练行拟合。相同信号和股票多个方向/周期同时合格时，取预测净边际最大者，平手取更短周期，再按稳定方向顺序；后续时刻仍可产生请求，原引擎每天每股最多一次成功首腿。

动态选择的是**入场时就确定的固定恢复期限**：信号后下一交易分钟执行，再数20/40/60个交易分钟，跨午休跳过非交易时间，不能跨日，预定恢复不得晚于14:30。它不是依据入场后的浮盈动态止盈止损。实际恢复仍按原真实报价时钟、部分容量和尾盘六分钟重试；失败会保留到跨日真实恢复，不补虚构成交。

## 特征、配置与文件

新增11列来自当时已知的5/15分钟收益、当日累计成交额、日内进度、VWAP、已实现波动及前日振幅：同期市场短收益、上涨广度、当日收益分散度、个股相对市场收益、成交额进度、趋势效率、20分钟波动尺度、标准化VWAP偏离和短期反转确认。这些量只能作为状态代理，不能辨认真实新闻或机构订单。

| 文件 | 用途 |
| --- | --- |
| [weekly_t_cross_validation_v11.yaml](../config/weekly_t_cross_validation_v11.yaml) | 事前候选、训练、费用、数据与恢复缓存配置。 |
| [run_weekly_t_timing.py](../scripts/run_weekly_t_timing.py) | 只读复用周选择和行情，训练、内部账户选择、全折冻结、外层评价。 |
| [weekly_t_timing_data.py](../quant_backtest/weekly_t_timing_data.py) | 核对原周OOF队列与数据身份；读取固定周期标签，不复用旧T权重。 |
| [weekly_t_timing_models.py](../quant_backtest/weekly_t_timing_models.py) | `fit_timing_model`、`predict_timing_model`，原始回归、三分类和支持域。 |
| [weekly_t_timing.py](../quant_backtest/weekly_t_timing.py) | 当前特征、25候选、方向/恢复周期选择和训练内部选择。 |
| [weekly_t_timing_reporting.py](../quant_backtest/weekly_t_timing_reporting.py) | `write_timing_report(output_dir,cfg)` 从保存CSV生成年度对比、费用、执行失败、持仓和PNG；不重跑账户。 |

| 配置 | 本轮默认及用途 |
| --- | --- |
| `research_version` | `weekly_t_v11_cost_aware_timing`，独立版本和输出身份。 |
| `t_target_mode` | `raw_return_and_cost_profit_class`，原始收益回归＋扣费事件分类；不再采用零市场先验残差目标。 |
| `candidate_activation_policy` | `positive_net_execution_valid`，内部实际正净利及五项T执行要求；否则显式持有。 |
| `source_training_report_dir` | 原v8报告用于复用已审计周OOF选择及行情身份，名字不表示复用旧T模型。 |
| `frozen_selection_report_dir` | 本折v4周模型与内部/外层周名单；不改70/30评分和前100。 |
| `timing_horizons` | `[20,40,60]`，当前信号可选择的预定恢复期限。 |
| `timing_regimes` | `learned/reversal/momentum`，固定三种当前状态条件。 |
| `timing_min_net_edges` | `[0.0005,0.001]`，成本以后额外5/10bp预测净边际。 |
| `timing_min_profit_probabilities` | `[0.4,0.5]`，未校准扣费事件输出门槛。 |
| `timing_max_abs_day_return`、`timing_max_known_amplitude` | 0.03、0.06，仅看信号时已完成行情。 |
| `t_model_profiles` | `l2_depth4/l2_depth6`；各自三个周期，均重新拟合。 |
| `dense_recovery_cache_paths` | 四年原只读mmap报价路径，包含真实0与缺失，不重建217m行报价。 |
| `short_t_fraction`、`max_pending_t_notional_fraction` | 沿用25%昨日股数整手T、10%初始本金未恢复参考名义额上限。 |
| `latest_planned_recovery_time` | 沿用14:30，不能据外层结果延后期限。 |
| `t_training_cohort`、`t_training_domain`、`strict_feature_provenance`、`excluded_model_features` | 沿用同折核心OOF周队列、已知整手/期限训练域、严格原始来源隔离，模型排除累计历史天数。 |
| `t_support_quantiles`、`t_support_features` | 沿用核心训练的0.001/0.999支持域和五个当前风险字段，不能用外层分布重估。 |
| `outer_t_refit` | `false`，内部选定后外层使用同一核心模型权重。 |
| `short_t_net_margins`、`t_huber_alpha` | YAML保留旧共享配置身份；本轮候选由 `timing_min_net_edges` 与两个L2配置决定，不再产生旧27候选或Huber候选。 |

其他基础字段和原执行语义见[短周期做T验证说明](短周期做T验证说明.md)、[选股队列说明](选股队列做T验证说明.md)、[执行条件匹配说明](执行条件匹配做T验证说明.md)。佣金万八、最低佣金0、双边过户费十万一、单边滑点万五、按历史日期的卖出印花税、每分钟成交额5%上限均保持。真实报价、整手、现金、T+1及涨跌停代理仍决定能否成交；选中100只并不保证能实际买入100只。

模型保存原生回归与三分类权重、joblib、模型元数据、特征筛选、`support.json`、`training_targets.csv` 和 `native_reload_audit.json`。目标审计表包含原始收益、两方向净标签、盈利类别、实际拟合标记与来源/标签起止；不把未来字段加入 `f_*`。

## 运行与核查

从仓库根目录执行。完整命令会新训练核心T模型并运行内部候选和外层账户；不会修改旧行情、旧选择或旧报告。

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/run_weekly_t_timing.py --config config/weekly_t_cross_validation_v11.yaml
```

先做单年小范围工程烟测；该命令只验证接线，不能据其收益选参或声称有效：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/run_weekly_t_timing.py --config config/weekly_t_cross_validation_v11.yaml --test-years 2026 --max-symbols 20 --smoke-weeks 3 --output-dir reports/weekly_t_cross_validation_smoke
```

报告器读取 `cross_validation_results.csv` 和每年 `selected_rule.json`、`test_metrics.csv`、`test_equity_curve.csv`、`test_trade_events.csv`、`test_round_trips.csv`、`test_open_positions.csv`、`daily_inventory_audit.csv`、`blocked_orders.csv` 及共同底仓记录。各年原文件保持不变，根目录生成 `annual_summary.csv`、`年度.csv`、标准合并账本、`final_report.md`、`test_return_curve.png`、`timing_reporting_status.json`。合并文件保留 `test_year`，不得将四个独立账户当作跨年连续实盘。

报告明确区分 hold、启用未成交、启用已成交，列出请求/首腿/完整组/恢复片段、费用、滑点、跨日恢复、库存差异、待恢复、T+1、底仓差异、周退出失败和陈旧估值。`net_pnl` 已使用含滑点成交价且扣费用，不能重复扣滑点；尚未恢复时，配对已实现净额与账户估值差分别保留。

完整正式命令已运行并保存上述目录。只读交易质量诊断的复现命令如下；它不会重跑账户或修改规则，默认对照固定v10报告，并拒绝覆盖已经归档的诊断目录：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/diagnose_weekly_t_timing_quality.py --report-dir reports/weekly_t_cross_validation/run_20260909_182945_388303 --reference-frequency-dir reports/weekly_t_cross_validation/run_20260909_135906_004933
```

诊断源码为 [diagnose_weekly_t_timing_quality.py](../scripts/diagnose_weekly_t_timing_quality.py)，归档包含[协议](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/protocol.json)、[完整组明细](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/actual_pairs.csv)、[对照表](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/comparison_v10.csv)、[输入/输出SHA与统计](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/diagnosis.json)及脚本副本。原始运行源、原年度CSV与旧v10报告均未改。

已完成的工程证据：模型11项单测与4股、88行真实核心小样本检查通过，原生/joblib预测及删除未来字段后的推断一致；这些不证明收益。报告器双Python编译、既有hold/active账本纯渲染、同日缺收盘的陈旧估值、矛盾hold拒绝检查通过，未重跑旧账户。交易质量脚本另用真实v10格式、分批完整组、hold及新周期预测字段完成检查，再执行本轮只读诊断。正式模型及原始行情独立审计尚待最终归档，不将代码测试通过写成盈利目标达成。

2026仅评价到4月30日。末日未恢复或原周末底仓未能卖出时继续按真实记录估值和披露，不为使“每日股数不变”看似成立而补成交。当前数据仍缺完整历史精确涨跌停、盘口队列和公司行动校准；静态恢复风险筛选不能保证当日必能买回/卖回。后续验证应以未参与本轮研究的新数据检验净收益与执行稳定性。
