# A 股 XGBoost Long-Only 策略方法说明

## 本次确认：原XGBoost与CTA交叉验证

**[正式结果已完成](../reports/original_method_year_validation/run_20260909_124531_658067/研究结论.md)：本轮原XGBoost约束版四年平均夏普-4.368453，CTA版-4.372274，未达到3。** 方法一2023/2024/2025/2026年1至4月收益为-25.94%/-23.39%/-33.30%/-7.36%；方法二为-26.07%/-23.39%/-33.42%/-7.36%。4个模型、8门槛、16组收益图/逐笔记录、训练与落盘账务审计已完成，112项测试通过不代表策略盈利。两方法交易高度重合，2024/2026约束版实际成交和净值相同。没有替换旧模型或覆盖旧报告，不应直接实盘使用这轮模型。

[原两策略交叉验证说明](原XGBoost与CTA交叉验证说明.md)恢复旧156列XGBoost截面排名、CTA叠加及ATR240初始2倍/跟踪4倍退出。前5/每批总资产10%保留，逐年独立重训，不用旧全量模型自测。旧式有偏成交与信号后成交约束分开报告，入口`scripts/run_original_method_year_validation.py`；与下面最近的LightGBM模型族研究严格区分。

## 单独研究：内部选择模型族

[本轮正式结论](../reports/family_selected_path_validation/run_20260909_100636_597293/研究结论.md)：四年平均夏普0.055754，收益依次为1.89%、-31.45%、-11.53%、5.29%（2026仅1至4月），**未达标，不能作为实盘替换依据**。四张中文收益图、逐笔记录、4模型继承及独立账务审计完成，98项测试通过。新审计入口`scripts/audit_family_path_results.py`可重复只读核验；没有测试后回选参数，也不把新模型族结果贴成旧XGBoost/CTA成绩。

[内部模型族选择说明](内部模型族选择说明.md)在完整核心父轮尚未测试的前提下，新增一个统一的内部模型族选择算法。父轮两个固定模型族未四折全部通过的结论不变；新算法只依非留出年的内部验证选直接或分解模型，并另行冻结、审计和单次外层。入口`scripts/run_family_selected_path_validation.py`不重新训练、不改原门槛/费用/退出；这是看过内部结果后新增的研究选择，不能视为原先预注册，也不能将新结果贴回父轮两策略。

## 继续研究：完整核心与环境分解

[完整核心分解模型说明](完整核心分解模型研究说明.md)在既有执行路径标签上比较直接净收益与“环境共同收益＋个股相对收益”。核心内部保留完整每日入口并按标签并发降权，跨核心/验证/留出年的边界不放宽；周池、前5、总资产10%、T+1、费用和ATR/到期退出不变。入口为`scripts/run_pooled_path_year_validation.py`，先内部训练、冻结并审计真实输入/权重/各分量，再晋级后单次外层；不替换旧实盘快照，也不把相对收益当成无空头的市场中性收益。

[完整核心本轮结论](../reports/pooled_path_year_validation/run_20260909_095128_476739/研究结论.md)：48模型包/72分量、144校准与23验证完成，87项测试及真实输入/原生模型审计通过。直接模型只通过2023/2026内部折，分解模型通过前三折而2026失败，两类固定模型族均未晋级，正式外层未运行，目标未达到；不能把2024内部夏普4.990当年度测试成绩。

## 继续研究：按真实执行路径训练

[执行路径模型研究说明](执行路径模型研究说明.md)把ATR、T+1、未成交和部分卖出纳入监督标签，训练LightGBM净收益/盈利概率模型。仍复用原有界pkl并保持前5、每批总资产10%和费用；路径标签独立账户不等于完整共享账户收益。新入口`scripts/run_pathwise_year_validation.py`先做核心路径与原引擎对照，再内部训练、冻结和晋级后单次外层。旧模型不替换，共享撮合异常输入保护修复后，旧轮次须用对应源码快照复现。

[本轮实际结论](../reports/pathwise_year_validation/run_20260908_235259_988569/研究结论.md)：修正按股票错开日期的抽样后，48模型、144校准候选、21入围验证完成；82项测试与48模型真实来源/重载审计通过。2026的3/5/10日核心样本分别345/265/157条进入2023和2024留出折拟合，但最后一折两方法内部均未通过，正式外层未运行，没有本轮测试收益图，平均夏普3目标未达到。不会回选尚未测试的旧同步抽样轮。

## 继续研究：低维与技术信号模型

[低维信号模型研究说明](低维信号模型研究说明.md)新增岭回归与技术信号保序回归对照，用固定26列和同股非重叠核心标签研究1/3/5日净收益。周池、T+1、前5门槛、每批总资产10%及ATR/到期退出保留。入口`scripts/run_simple_signal_year_validation.py`默认只做内部训练；这是新算法研究，不会替换下述历史XGBoost/CTA实盘快照，收益结论须以冻结运行报告为准。

[本轮正式结果](../reports/simple_signal_year_validation/run_20260908_230311_796199/研究结论.md)：72模型、432校准候选及12入围验证完成，随后只评价一次外层。岭回归四年平均夏普0.386，2023/2024收益-10.61%/-23.24%，2025/2026年1至4月收益25.92%/6.58%；技术模型四年现金，目标未达到。八组收益图和交易记录齐全；71项测试、72模型来源重载及八组独立账务重算通过，仍不能作为实盘证明。

## 继续研究：有限窗口与来源隔离

[有限窗口重训说明](有限窗口重训说明.md)记录232个有限原因子、简单均值ATR、完整截面来源传播、核心拟合不读取上年验证行情及稳定键持仓抽样。开仓和持仓模型全部重训，旧274因子pkl及历史模型不覆盖；2026实际进入哪些折以拟合审计为准。入口`scripts/run_bounded_year_validation.py`默认只做内部重训，至少一种方法四折全部内部启用才安排正式外层，不预先宣称夏普达标。

[本轮正式内部结论](../reports/bounded_year_validation/run_20260908_222819_626855/研究结论.md)：20个新模型已完成，四折各用143/139/140/143个特征；2026在2023/2024折中每档3日模型贡献45,397行、5日模型40,399行、持仓模型20,493行。方法一仅前两折内部通过，方法二仅首折通过，正式外层未运行，目标未达到。64项测试及实际拟合来源审计通过，不是盈利或实盘证明。

## 继续研究：预测期限与退出对照

[预测期限与退出对照说明](预测期限与退出对照说明.md)复用上一轮全部冻结模型和特征pkl，在训练内部比较原模型退出、3/5交易日同刻退出、到期后模型退出及T+1时间对照。入口`scripts/run_exit_alignment_year_validation.py`默认只做内部研究；不从上一轮外层收益挑模型，不改费用、前5门控或每批总资产10%限制。新结果在`reports/exit_alignment_year_validation/`，不能预先认为已达到夏普目标。

[本轮内部研究已完成](../reports/exit_alignment_year_validation/run_20260908_215813_680118/研究结论.md)，未达到外层晋级条件：方法一2024/2025折现金保护，方法二2025折现金保护。正式外层未运行，不存在本轮测试收益图；不可把表中的内部夏普当成留出年的测试夏普。20模型副本、192组原退出校准对照和55项回归检查通过，但不能证明盈利。审计入口为`scripts/audit_exit_alignment_comparison.py`，不读取外层收益。

## 继续研究：多尺度开仓

[多尺度开仓研究说明](多尺度开仓研究说明.md)记录3/5日净收益模型、较长周期因子、当前非一字线/成交额资格，以及T+1时间退出对照与动态模型退出。入口`scripts/run_multiscale_year_validation.py`默认只做训练内部研究，结果在`reports/multiscale_year_validation/`；目标仍未默认达成，不能把内部验证当作外层成绩。

新增60交易日的留出年后隔离，不重建已有特征pkl。首次多尺度全量轮次`run_20260908_205453_193394`隔离不足且未进入外层测试，已标记为不能用于验收。隔离修正轮`run_20260908_212122_151985`又因发现两方法候选门槛覆盖而在首折校准期间中止；目前代码按方法隔离规则，并有回归测试。2026仍参与其他折，但是否核心拟合与是否内部选参分别记录，不能把回顾检验说成当年的真实可交易成绩。

[双修正后的完整结果](../reports/multiscale_year_validation/run_20260908_212612_018862/final_report.md)已生成8组年度收益图和逐笔交易。目标未达到：2023两方法负收益，2024方法一收益-41.00%/夏普-1.624/最大回撤48.48%、方法二现金，2025两方法现金，2026年1至4月夏普仅0.832/0.835。不能删现金年计算高均值，也不应直接实盘部署。20模型重载、24候选门槛及39项测试通过，只代表这些工程检查通过，详见[补充审计](../reports/multiscale_year_validation/run_20260908_212612_018862/补充审计.md)。

## 新增：双模型四组留一年研究

[双模型留一年验证说明](双模型留一年验证说明.md)记录本次用户确认的2023至2026四折协议。新增净收益开仓模型与独立持仓模型，前三折将2026数据纳入训练；这三折含未来年份训练，只是回顾检验。训练/内部选参与冻结测试分两个命令，保留周度高波动前300、前5门控、每批总资产10%、T+1和完整成本。旧模型和下述历史报告不覆盖，新成绩不能预先假定超过夏普3。

[本轮正式结果](../reports/learned_exit_year_validation/run_20260908_202251_892127/final_report.md)：2024方法一/二夏普1.015/0.814，2025均1.839；2023方法二-2.046，方法一及2026两方法现金保护。四年目标未达标，不能删除无波动年度求均值。模型与规则已冻结，测试后未回选；[补充审计](../reports/learned_exit_year_validation/run_20260908_202251_892127/补充审计.md)解释2025成交完全相同及年末缺报价持仓。

后续审查确认该轮2023和2026的部分方法一入围候选门槛被方法二覆盖。历史文件不重写，其冻结只代表当时规则可追溯，不能理解成选参正确或可直接实盘；影响和修复证据已补入上述审计。

## 2026-09：年度重训入口

新增[年度滚动验证说明](年度滚动验证说明.md)：2024、2025及2026年1至4月分别向前测试，特征选择与拟合只用当轮过去数据；另生成包含2026的全量候选，它没有独立未来测试成绩。

入口为`scripts/run_annual_method_validation.py`，配置为`config/annual_method_validation.yaml`，输出在`reports/annual_method_validation/run_*/`。新流程统一采样时钟和日频夏普，分离退出信号与成交，并增加滑点/容量限制。以下旧报告保留供追溯，其中高夏普不能直接作为新模型成绩。

本次[正式年度报告](../reports/annual_method_validation/run_20260908_140559/final_report.md)：2024方法一收益0.89%、夏普2.11；2025方法一/二收益-6.05%/-5.85%；2026验证未通过而空仓。目标未达标，最终含2026数据的候选没有独立测试成绩且继承规则仍停用。

本文档汇总当前 A 股 XGBoost long-only 策略的完整流程、报告文件用途、交易策略细节、特征面板缓存和流程合理性检查。最新严格 T+1 回测结果目录为：

`reports/xgboost_long_only_backtest/run_20260526_202355/`

对应的严格训练期因子审计目录为：

`reports/factor_feature_audit/run_20260525_161905/`

## 最新补充：开仓频率和全 A 主板外推

2026-06-04 新增两类实盘前审计文件：

- `1method/开仓频率审计.md`：解释 method1 为什么可能连续几天不开仓。原沪深 300 测试集中只有 `82` 次买入，平均每 `2.86` 个交易日买入一次，最长连续 `22` 个交易日没有买入；实盘运行两天没开仓本身属于正常范围。
- `1method/全A主板外推结果.md`：记录把 method1 和 method2 固定规则迁移到全 A 主板非 ST 股票池后的正式外推结果。
- `1method/精选股票池对比结果.md`：记录科技主题池、最近一周高波动 top300、测试前流动性趋势 top300 在 method1/method2 下的对比结果。
- `1method/动态高波动股票池回测结果.md`：记录每周用上一周高波动 top300、每天用过去 7 日高波动 top300 两种动态股票池的正式回测。
- `1method/持仓级平仓策略回测结果.md`：记录周度高波动 top300 下，不再因掉出目标组合平仓，而改用持仓级退出规则、top5、单票10% 的正式回测。
- `1method/独立lot动态平仓策略回测结果.md`：记录新一版独立 lot 动态 ATR 平仓，先在策略训练集筛候选、再在验证集选规则、测试集只验收。

全 A 外推正式目录：

`reports/all_a_method_backtests/run_20260604_121326/`

动态高波动股票池正式目录：

`reports/dynamic_high_vol_method_backtests/run_20260615_161846/`

持仓级平仓策略正式目录：

`reports/weekly_high_vol_exit_strategy_backtests/run_20260702_111414/`

独立 lot 动态平仓策略正式目录：

`reports/weekly_high_vol_lot_exit_backtests/run_20260702_123216/`

本次外推不重新训练模型、不重新选择阈值，只把既有沪深 300 训练出的模型和门控搬到全 A 主板股票池。股票池为本地 2025/2026 都有数据的沪深主板 A 股，剔除北交所、科创板、创业板和当前名称含 ST/退的股票，最终纳入 `3008` 只。费用使用实盘新口径：券商佣金万八、免 5、卖出印花税 `0.0005`、过户费 `0.00001` 双边。

核心结果：

- method1 全 A 测试 Sharpe `4.409421`，年化收益 `17.239535`，最大回撤 `-0.337986`，买入事件 `373` 次，信号期占比 `24.47%`。
- method2 CTA 全 A 测试 Sharpe `4.441793`，年化收益 `17.725779`，最大回撤 `-0.337988`，买入事件 `371` 次，信号期占比 `24.62%`。
- 结果强于原沪深 300 回测，但这不是全 A 重训结果，也不能据此回头调旧阈值。
- 重要限制：当前 ST 过滤是当前名称过滤，不是严格历史点时过滤；回测仍未完整建模停牌、涨跌停不能成交、冲击成本、复权一致性和成交容量。top1 满仓策略在权益增长后会产生很大的单票交易金额，实盘必须加单票金额、成交额占比和最大持仓限制。

动态高波动股票池补充结果：

- 周度上一周高波动 top300：method1 测试 Sharpe `3.648573`，年化收益 `2.842638`，最大回撤 `-0.255614`，买入 `147` 次。
- 日度滚动 7 日高波动 top300：method1 测试 Sharpe `3.689603`，年化收益 `3.179945`，最大回撤 `-0.307858`，买入 `141` 次。
- 两个动态池都只用交易日前的历史波动率选股，比此前“测试末尾最近一周高波动静态池”更符合事前规则；但高波动池会放大滑点、容量和回撤风险。

持仓级平仓策略补充结果：

- 固定股票池：每周用上一周高波动 top300。
- 验证集选择：top5、单票买入目标仓位 `10%`、`time_1d_stop_6_take_10`，即最大持仓约 1 个交易日、`-6%` 止损、`10%` 止盈，并继续遵守 T+1。
- method1 测试 Sharpe `6.002144`，年化收益 `2.432562`，最大回撤 `-0.060809`，买入 `1053` 次，已平仓胜率 `0.612595`。
- 正式交易记录中因“未进入当前目标组合”平仓次数为 `0`；平仓由最大持仓时间、止盈、止损触发。

独立 lot 动态平仓策略补充结果：

- 固定股票池：每周用上一周高波动 top300。
- 入场：每个采样点最多选 top5，必须超过 method1 固定入场门控阈值。
- 仓位：每次开仓按当前总资产 `10%` 计算目标买入金额；同一股票后续交易日仍在 top5 且有现金时可以继续加仓。
- lot 规则：每次买入形成独立 lot，同一股票不同买入批次按各自买入价、入场 ATR、最高价独立平仓。
- 训练/验证/测试：策略训练集先筛全部退出候选，验证集只验证训练集 shortlist，测试集只跑验证集选中规则。
- 验证集选择：`atr_initial2_trail4_atr240`，即入场 ATR 初始止损 + ATR240 移动止损。
- method1 测试 Sharpe `5.694637`，年化收益 `5.001265`，最大回撤 `-0.080562`，买入 `1426` 次，已平仓胜率 `0.599153`，平均持仓天数 `1.775582`。
- 约束审计：`short_exposure=0`，T+1 违规 `0`，非 100 股整数手交易 `0`，因“未进入当前目标组合”平仓 `0`。
- 这版是当前更符合用户要求的动态平仓版本；上一版固定 `-6%/10%/1日` 平仓只保留为历史实验对照。

## 一、全流程是否合理

结论：工程流程和时间对齐逻辑基本合理。最新沪深 300 `run_20260526_202355` 已实现 signal_lag=1、T+1 卖出约束、100 股整数手、交易成本和 long-only 约束，且规则和持仓门控阈值只在训练期内部验证集上选择。本次测试集 Sharpe 已大于 3。但项目历史上曾围绕测试集目标做过探索，因此统计意义仍应谨慎解释，不能把历史测试集当作完全未触碰的新样本。

合理的部分：

- 数据按时间切分：训练集为 `datetime < 2025-05-01`，测试集为 `datetime >= 2025-05-01`。
- 因子只使用当前及历史已完成 bar，rolling、lag、pct_change、截面 rank 都没有读取未来行。
- 标签为带交易延迟的未来收益：bar t 产生信号，t+1 close 入场，持有 24 根 bar 后退出。
- 训练期内部验证从 `2024-11-01` 开始，仍在训练期内；组合规则在内部验证集上选择。
- 最终交易约束符合用户要求：A 股 long-only、不做空、不加杠杆、T+1 卖出、初始资金 100 万、无滑点、买入按 100 股整数手向下取整；原沪深 300 报告使用简化万1成本，新全 A 外推报告使用佣金万八、免 5、卖出印花税和过户费口径。
- `reports/factor_feature_audit/run_20260525_161905/no_future_function_check.md` 显示未来函数检查通过。

需要谨慎解释的部分：

- 项目历史上对已有测试预测结果做过探索，这会降低“测试集完全独立”的统计含义；后续应保留新数据或做 walk-forward。
- 最新策略只在模型置信度较高时持有 1 只股票，平均持仓约 0.11 只，平均多头暴露约 10.77%。它通过大量空仓降低回撤，但单票集中度、容量风险、停牌风险和实际成交风险仍需要单独评估。
- 当前未完整建模涨跌停无法成交、ST/退市过滤、最低佣金、冲击成本和历史点时沪深 300 成分股变化。
- 如果要形成更严格的研究结论，应在未来新增数据上做一次完全未触碰的 forward test，或做滚动 walk-forward。

## 二、最终交易策略

最终选中的规则名为：

`top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`

具体逻辑：

1. 使用 156 个截面 rank 特征训练 XGBoost 回归模型。
2. 模型预测每只股票在同一时间截面上的 alpha 分数。
3. 每个可调仓时间点，按预测分数从高到低排序。
4. 先做持仓门控：top1 预测分必须不低于验证集 90% 分位阈值 `0.5530999541282654`，且 top1-top2 预测分差必须不低于验证集 70% 分位阈值 `0.006096088886260986`。
5. 只有门控通过时，选择排名最高的 1 只股票；门控不通过时保持现金。
6. 目标总仓位为 100%，但因为门控较严格，测试期平均多头暴露约 10.77%。
7. 买入时按 100 股整数手向下取整，预留成交金额万1的费用。
8. 调仓频率为每 1 个采样周期调仓一次；每个采样周期为 24 根分钟 bar。
9. 持仓期间不做空、不加杠杆，未入选股票在满足 T+1 后卖出到 0。
10. 组合收益使用实际股数、入场价格、退出价格、现金余额和交易费用计算。

最终测试集指标：

- Sharpe：3.706244
- 年化收益：2.874256
- 年化波动率：0.384842
- 最大回撤：-0.127287
- 成本后总收益：2.669561
- 平均换手率：0.064795
- 平均持仓数量：0.11
- 平均多头暴露：0.107651
- 已平仓交易胜率：0.641975
- 总交易手数：263537.00
- short_exposure：0.0
- lot_size：100

当前严格 T+1 测试集 Sharpe 已达到大于 2 的目标。

## 三、XGBoost 参数调优结果

参数调优脚本为：

`scripts/tune_xgboost_long_only_parameters.py`

最新全量参数调优目录为：

`reports/xgboost_parameter_tuning/run_20260527_121550/`

调优边界：

- 核心训练集：`datetime < 2024-11-01`，且标签退出时间不跨验证集。
- 内部验证集：`2024-11-01 <= datetime < 2025-05-01`，且标签退出时间不跨测试集。
- 最终测试集：`datetime >= 2025-05-01`，只在模型参数和交易规则选定后评估一次。
- XGBoost 参数、top-N、置信度门控、分差门控都只按验证集表现选择，测试集不参与训练和调参。

本轮标准参数集共测试 12 个候选，覆盖 `learning_rate`、`n_estimators`、`max_depth`、`min_child_weight`、`subsample`、`colsample_bytree`、`reg_lambda`、`reg_alpha` 以及两组组合参数。验证集选中的候选为：

`deeper_more_regularized`

参数覆盖：

- `max_depth=5`：提高树深，允许模型捕捉更复杂的非线性因子交互。
- `min_child_weight=40`：提高叶子节点样本权重约束，避免更深树过度拟合分钟级噪声。
- `reg_lambda=8.0`：增强 L2 正则，平滑叶子权重。

验证集选中的交易规则为：

`top_n_1p0_hold_1_exposure_1p0_confq_0p9`

含义：

- 每 24 根分钟 bar 检查一次调仓。
- 只在 top1 预测分超过验证集 90% 分位阈值 `0.5603431224822998` 时买入。
- 买入预测分最高的 1 只股票，目标仓位 100%，但门控不通过时空仓。
- 不使用 top1-top2 分差门控。
- 仍执行 signal_lag=1、T+1、100 股整数手、万1成本、long-only、不加杠杆。

验证集排名前几名：

- `deeper_more_regularized`：验证 Sharpe 4.503279，验证年化收益 5.827162。
- `combined_expressive`：验证 Sharpe 4.148167，验证年化收益 3.496843。
- `lower_min_child_weight`：验证 Sharpe 3.655257，验证年化收益 3.348639。
- `higher_lr_fewer_trees`：验证 Sharpe 3.616328，验证年化收益 3.393844。
- `combined_conservative`：验证 Sharpe 3.546800，验证年化收益 2.634288。

最终测试集验收结果：

- Sharpe：2.800323
- 年化收益：2.295213
- 年化波动率：0.463586
- 最大回撤：-0.298598
- 成本后总收益：2.141427
- 平均多头暴露：0.156053
- 已平仓交易胜率：0.610000
- 总交易手数：454392.00
- short_exposure：0.0
- T+1 违规数：0
- 100 股整数手违规数：0

结论：本轮参数调优在验证集内部选出了相对最优候选，但最终测试 Sharpe 为 2.800323，没有达到 3，也低于上一版严格 T+1 结果 `run_20260526_202355` 的测试 Sharpe 3.706244。因为测试集不能参与调参，不能看到这个测试结果后再回头选择旧参数或旧规则。后续如果要继续追求 Sharpe > 3，应使用新的未来数据做 forward test，或在测试集之外重新设计 walk-forward 验证流程。

参数调优目录文件：

- `model_parameter_candidates.csv`：12 个模型候选的验证集最优规则和指标。
- `parameter_sensitivity.csv`：按参数维度汇总的验证集敏感性和中文说明。
- `selected_model_candidate.json`：最终由验证集选中的模型参数、选择依据和是否使用测试集调参的标记。
- `split_audit.csv`：核心训练集、内部验证集、最终测试集的行数、股票数和标签退出时间审计。
- `数据切分审计.md`：中文说明训练/验证/测试边界。
- `parameter_tuning_report.md`：中文参数重要性、候选排名、最终测试验收。
- `validation_rules_<candidate>.csv`：单个候选模型下所有验证集策略规则表现。
- `test_metrics.csv`、`test_equity_curve.csv`、`test_trade_events.csv`、`test_round_trips.csv`、`测试集收益曲线.png`：选中候选的最终测试集回测产物。

## 四、最终回测报告目录文件说明

目录：`reports/xgboost_long_only_backtest/run_20260526_202355/`

- `final_report.md`：中文最终报告，包含数据切分、防未来函数控制、模型与交易策略、验证规则选择、测试集指标、复现命令和风险提示。
- `model_metadata.json`：机器可读元数据，记录配置、特征面板缓存路径、模型参数、选中的组合规则、股票池、训练/验证/测试指标等。字段名保留英文，便于程序读取。
- `xgboost_model.json`：XGBoost 原生模型文件，可用于重新加载模型或跨语言检查树模型。
- `xgboost_model.joblib`：Python joblib 格式模型文件，便于在本 Python 环境中快速加载。
- `used_feature_manifest.csv`：本次模型实际使用的特征清单，包含基础因子、模型特征名、因子类别、训练期 RankIC、缺失率和筛选信息。
- `feature_list.txt`：156 个最终模型特征名，每行一个，方便训练、预测或排查特征顺序。
- `feature_importance.csv`：XGBoost 输出的特征重要性排序，用于了解模型主要依赖哪些因子。
- `validation_rule_metrics.csv`：训练期内部验证集上的候选组合规则表现。最终规则只从这里选择。
- `train_predictions_sample.csv`：训练集预测样本，保留前 200,000 行，用于检查模型训练期预测分布和字段对齐。
- `validation_predictions.csv`：内部验证集预测结果，包含 `datetime`、`symbol`、`close`、`forward_return`、`entry_datetime`、`entry_price`、`exit_price`、`target_rank`、`prediction`。
- `test_predictions.csv`：测试集预测结果。组合回测基于这里的预测分数生成持仓。
- `test_holdings.csv`：测试集每期实际持仓明细，包含股票、预测分数、入场/退出价格、实际股数、入场市值、退出市值、权重和规则名。
- `test_trade_events.csv`：测试集逐笔买入和卖出事件，包含交易时间、股票、方向、股数、手数、价格、成交金额、成本和触发原因。
- `test_round_trips.csv`：测试集已平仓交易复盘，按买入批次和卖出批次匹配，包含买入时间、卖出时间、收益、成本和买卖触发原因。
- `test_open_positions.csv`：测试期末仍未平仓的持仓批次。
- `test_equity_curve.csv`：测试集组合权益曲线，包含成本前收益、成本、成本后收益、换手、持仓数量、总多头暴露、现金权重、权益、累计收益和回撤。
- `测试集收益曲线.png`：测试集收益折线图，图中附策略名、Sharpe、收益率、年化、波动、最大回撤、交易手数、胜率和平均持仓。
- `test_return_curve.png`：同一张收益图的英文文件名副本，方便跨系统打开。
- `test_metrics.csv`：测试集核心绩效指标，包括 Sharpe、年化收益、年化波动、最大回撤、Calmar、换手、交易次数、成本前后收益等。
- `test_monthly_returns.csv`：测试集按月份聚合的组合收益。
- `test_yearly_returns.csv`：测试集按年份聚合的组合收益。

## 五、资料驱动中性和小市值策略

新增脚本：

- `scripts/build_daily_size_features.py`：用 AKShare 日频行情的收盘价、成交量和换手率反推历史流通市值，输出日频小市值特征缓存。
- `scripts/run_researched_strategy_backtests.py`：复用既有 XGBoost 预测和特征面板，构造风格中性 long-only 和小市值 long-only 策略候选，只在验证集选候选和规则，再对测试集做最终回测。
- `scripts/run_all_a_small_cap_backtest.py`：读取本地全 A 分钟数据，构造全 A 小市值/低流动性/反转代理面板，并执行 long-only 小市值策略验证和测试。
- `scripts/run_all_a_small_cap_model_backtest.py`：读取全 A 小市值面板，用测试集之前的数据训练 LightGBM，再用验证集选择小市值机器学习叠加规则。
- `scripts/run_all_a_small_cap_regime_backtest.py`：读取全 A 小市值面板，构造小市值、短期反转、低波动和市场状态过滤候选。
- `scripts/run_all_a_small_cap_open_reversal_backtest.py`：读取全 A 小市值面板，只取每天 09:55 信号，按下一交易日同一时点退出，测试小市值开盘反转策略。
- `scripts/run_synthetic_neutral_backtest.py`：读取既有 XGBoost 预测，构造“多头模型组合 - 同期股票池等权基准”的合成中性策略。该脚本需要对冲腿，不是现货纯 long-only。

资料依据：

- Fama-French 的 size/SMB 思路把公司规模作为共同风险因子，规模通常用股价乘以股数度量。
- 市场中性策略通常需要多空组合或指数/期货对冲；当前 A 股现货框架仍按 long-only 执行，所以这里的“中性策略”是风格中性选股，不是真正市场中性。
- A 股异常研究显示 size、quality、past-return 类异常证据并不稳定，因此小市值策略需要实证验证，不能假设一定有效。

日频市值缓存：

- 目录：`reports/daily_size_features/run_20260527_150526/`
- 成功股票数：257 / 257
- 行数：206,623
- 日期范围：2023-01-03 至 2026-04-30
- 计算：`derived_float_market_cap = close * volume_hands * 100 / (turnover_rate / 100)`
- 注意：这是由换手率反推的流通市值，不是交易所原始披露字段。

最新真实市值版本策略目录：

`reports/researched_strategy_backtests/run_20260527_150626/`

验证集选中结果：

- 风格中性家族：`neutral_residual_defensive`，验证 Sharpe 0.264866，规则 `top_n_3p0_hold_1_exposure_1p0`。
- 小市值家族：`small_cap_quality`，验证 Sharpe 0.586301，规则 `top_n_1p0_hold_1_exposure_1p0`。

最终测试集结果：

- 风格中性 long-only：Sharpe 0.704099，年化收益 0.180532，最大回撤 -0.377644，short_exposure 0，T+1 违规数 0，100 股整数手违规数 0。
- 小市值 long-only：Sharpe -1.025807，年化收益 -0.179369，最大回撤 -0.227702，short_exposure 0，T+1 违规数 0，100 股整数手违规数 0。

结论：两个可执行 long-only 策略都没有达到测试集 Sharpe > 3。真实流通市值小市值候选在验证集和测试集都较弱，说明在当前稳定沪深 300 大盘股票池里，小市值暴露不是足够强的独立 alpha。快速研究诊断显示，如果把中性策略改成“多头 alpha 组合 - 同期股票池等权基准收益”的合成对冲口径，测试 Sharpe 可到约 3.19；但这需要实际指数/期货/ETF 对冲腿，不能当作当前 A 股现货 long-only 结果。

后续全 A 扩展结果：

- AKShare 东方财富日频接口在全 A 拉取时出现代理连接断开，`reports/daily_size_features_all_a/run_20260527_153452/` 只成功 12 / 5,466 只，因此全 A 真实流通市值缓存暂不可用。
- 回测引擎修正：全 A 持仓股票在某些采样时点缺少当前行情时，原逻辑会把该持仓估值为 0，造成最大回撤接近 -100% 的假信号。现在用最近可得价格估值缺失持仓。
- `reports/all_a_small_cap_backtest/run_20260527_165536/`：使用本地分钟数据构造小市值代理，验证集选中 `small_cap_pure` + `top_n_20p0_hold_1_exposure_1p0`，测试 Sharpe 0.521698，最大回撤 -0.237728，未达 3。
- `reports/all_a_small_cap_model_backtest/run_20260527_162402/`：用 2024-11-01 至 2025-02-28 训练 LightGBM，2025-03-01 至 2025-04-30 验证，测试 Sharpe 0.646285，未达 3。
- `reports/all_a_small_cap_regime_backtest/run_20260527_172318/`：小市值叠加短期反转、低波动和市场状态过滤，验证集选中 `small_reversal_low_vol`，测试 Sharpe 1.653745，未达 3。
- `reports/all_a_small_cap_open_reversal_backtest/run_20260527_181206/`：09:55 小市值开盘反转，验证集选中 `open_tradable_reversal`，测试 Sharpe 1.720757，未达 3；这是当前小市值 long-only 里最好的测试结果。
- `reports/all_a_small_cap_open_reversal_backtest/run_20260527_181627/`：在开盘反转上继续加入验证集分数/分差门控，验证 Sharpe 更高但测试 Sharpe 降至 0.468346，说明门控过拟合验证集。
- 小市值因子对冲诊断中，验证集最优为 `small_cap_reversal_score` + `top_n_20` + `hedge=0.8`，测试 Sharpe 2.866774，仍未达 3。
- `reports/synthetic_neutral_backtest/run_20260527_163423/`：合成中性模型策略选中 `top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p6`，测试 Sharpe 3.212178，最大回撤 -0.074463，主动总收益 1.161845；但 `short_exposure=1.0`，必须作为有对冲腿的研究结果看待。

当前结论：中性策略在“合成对冲口径”下达到了测试 Sharpe > 3；小市值策略在当前数据和约束下还没有达到测试 Sharpe > 3。不能把中性对冲结果等同于 A 股现货 long-only 结果，也不能因为已经看见小市值测试结果而继续按测试集表现回头挑旧规则。

## 六、因子审计目录文件说明

目录：`reports/factor_feature_audit/run_20260525_161905/`

- `feature_selection_summary.json`：机器可读的特征筛选摘要，包括训练窗口、股票数、样本数、历史有效因子数量、新增因子数量和最终特征数量。
- `feature_list.txt`：严格训练期筛选后的 156 个模型特征名。
- `model_feature_manifest.csv`：最终模型特征清单，是 XGBoost 回测脚本的输入。
- `selected_base_factors.csv`：被选中的基础因子及训练期统计。
- `factor_review_manifest.csv`：所有被审计因子的详细记录，包括类别、窗口、缺失率、RankIC、稳定性和是否入选。
- `factor_category_summary.csv`：按因子类别汇总的筛选结果。
- `leakage_check_report.csv`：防泄露检查表，记录特征列、重复行、无限值、排序、标签窗口和未来扰动检查结果。
- `no_future_function_check.md`：中文未来函数检查报告。
- `source_references.md`：中文因子来源说明，列出动量、低波、非流动性、区间波动率和公式化 alpha 等参考来源。

## 七、脚本和模块的职责

- `quant_backtest/factor_library.py`：核心因子库。负责从 OHLCV/amount 生成动量、反转、波动率、技术指标、价量相关性、流动性等因子；也提供带交易延迟的未来收益标签和未来函数检查辅助函数。
- `scripts/analyze_stable_hs300_factor_correlation.py`：稳定沪深 300 股票池与早期因子相关性分析脚本。用于找出稳定股票、评估已有因子的平均表现和相关性结构。
- `scripts/build_factor_features.py`：一键构建因子特征的工具脚本，可输出原始因子、截面 rank/z-score 和元数据。
- `scripts/audit_factor_feature_library.py`：严格训练期因子审计和筛选脚本。只使用训练期数据，输出最终模型特征清单和未来函数检查报告。
- `scripts/run_xgboost_long_only_backtest.py`：最终模型训练、内部验证选规则、整数手 long-only 回测和报告输出脚本。该脚本会把已构造好的特征面板保存为 pkl 缓存，后续同一股票池、同一特征清单、同一 horizon 和同一采样步长会直接复用缓存。
- `scripts/tune_xgboost_long_only_parameters.py`：XGBoost 参数和策略门控调优脚本。它只用核心训练集拟合候选模型，只用内部验证集选择模型参数和交易规则，最后才对测试集做一次最终验收。
- `scripts/build_daily_size_features.py`：日频流通市值特征构建脚本。它用日收盘价、成交量和换手率反推出流通市值，供小市值策略使用。
- `scripts/run_researched_strategy_backtests.py`：资料驱动策略回测脚本。它构造风格中性和小市值策略候选，并按验证集选择后做测试集验收。
- `scripts/run_all_a_small_cap_backtest.py`：全 A 小市值代理回测脚本。它从本地分钟数据构建可复用小市值代理 pkl 面板，支持真实市值、代理市值和自动降级三种模式。
- `scripts/run_all_a_small_cap_model_backtest.py`：全 A 小市值机器学习叠加脚本。它读取全 A 小市值面板，只用测试集之前的数据训练 LightGBM，并用验证集选规则。
- `scripts/run_all_a_small_cap_regime_backtest.py`：全 A 小市值状态过滤脚本。它构造市场收益、市场广度、市场波动和流动性状态阈值，并只用验证集估计阈值。
- `scripts/run_all_a_small_cap_open_reversal_backtest.py`：全 A 小市值开盘反转脚本。它把每天 09:55 的信号映射到下一分钟入场和下一交易日同一入场时点退出。
- `scripts/run_synthetic_neutral_backtest.py`：合成中性策略脚本。它只用验证集选规则，把多头模型组合收益减去同期等权基准收益，输出主动收益曲线。
- `quant_backtest/metrics.py`：绩效指标模块。计算年化收益、年化波动、Sharpe、Sortino、最大回撤、Calmar、胜率、尾部风险等。

## 八、特征面板缓存

当前已生成完整稳定沪深 300 股票池的特征面板缓存：

- `reports/feature_panel_cache/feature_panel_7226a2c8f3dd389a.pkl`：生成输出，257 只稳定股票、2,066,023 行、166 列；包含基础行情字段、带 T+1 入场时间的标签辅助字段，以及 156 个 `cs_rank_*` 模型特征。
- `reports/feature_panel_cache/feature_panel_7226a2c8f3dd389a.metadata.json`：生成输出，记录缓存版本、原始数据指纹、股票池、特征清单、horizon、采样步长、行数和时间范围。

脚本默认会先检查缓存。缓存键由以下内容共同决定：原始数据文件大小和修改时间、稳定股票池、特征清单、`--horizon`、`--eval-step`、`--max-symbols` 和缓存版本。只有这些内容一致时才会复用旧 pkl。

常用参数：

- `--feature-panel-cache-dir`：指定缓存目录，默认 `reports/feature_panel_cache`。
- `--rebuild-feature-panel`：忽略已有 pkl 并强制重建缓存。
- `--no-feature-panel-cache`：完全禁用缓存，适合排查缓存问题。
- `--build-feature-panel-only`：只构建或读取特征面板缓存，不训练模型、不跑测试集。

## 九、完整调用顺序

第一步，准备稳定股票池和因子基础分析。历史上使用：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/analyze_stable_hs300_factor_correlation.py --data-root ashare_minute_data_2023_2026_04/extracted --universe-file config/hs300_current_constituents.csv --horizon 24 --eval-step 24 --high-corr 0.80 --workers 4
```

作用：

- 从本地沪深 300 可用数据中筛出稳定股票。
- 计算历史因子表现和因子相关性。
- 输出 `reports/stable_hs300_factor_correlation/run_20260518_223759/`，后续使用其中的稳定股票池。

第二步，严格训练期因子审计：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_factor_feature_library.py --workers 4 --train-start 2023-01-03 --train-end 2025-04-30 --min-assets 20 --min-rank-ic-periods 120
```

作用：

- 只读取 `2025-05-01` 之前的数据。
- 构造和审计 274 个基础因子。
- 用训练期 RankIC、缺失率、月度稳定性和相关性过滤选出 156 个模型特征。
- 输出 `reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv`。

第三步，推荐先生成特征面板缓存：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgboost_long_only_backtest.py --feature-manifest reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv --workers 4 --build-feature-panel-only --rebuild-feature-panel
```

作用：

- 为当前稳定沪深 300 股票池和 156 个最终特征生成一次 pkl 缓存。
- 后续同配置回测会直接读取 pkl，不再重复构建 257 只股票的因子面板。
- 该命令不训练模型、不生成测试集预测、不评估测试集。

第四步，训练 XGBoost、验证选择交易规则并回测测试集：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgboost_long_only_backtest.py --feature-manifest reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv --workers 4 --split-date 2025-05-01 --validation-start 2024-11-01 --commission-rate 0.0001 --initial-cash 1000000 --top-n-candidates 1,2,3 --top-quantile-candidates 0.02 --hold-steps-candidates 1 --target-exposure-candidates 1.0 --confidence-quantile-candidates none,0.9 --spread-quantile-candidates none,0.7 --min-assets 30 --min-holding-count 1 --lot-size 100 --sharpe-target 2
```

作用：

- 优先读取特征面板 pkl；如果缓存不存在，则按股票构造基础因子面板并写入缓存。
- 在同一时间截面内生成 `cs_rank_*` 模型特征。
- 构造带交易延迟的未来收益标签。
- 切分为内部训练集、内部验证集和测试集。
- 用内部训练集训练验证模型。
- 用内部验证集选择 top-N、调仓间隔、目标仓位规则、top1 预测分门控阈值和 top1-top2 分差门控阈值。
- 用完整训练期重训最终模型。
- 对测试集预测分数并执行 100 股整数手 long-only 回测。
- 输出最终报告和所有回测产物。

第五步，验证集内调优 XGBoost 参数和策略门控：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/tune_xgboost_long_only_parameters.py --feature-manifest reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv --workers 4 --split-date 2025-05-01 --validation-start 2024-11-01 --commission-rate 0.0001 --initial-cash 1000000 --top-n-candidates 1,2,3 --top-quantile-candidates 0.02 --hold-steps-candidates 1 --target-exposure-candidates 1.0 --confidence-quantile-candidates none,0.85,0.9,0.92 --spread-quantile-candidates none,0.6,0.7,0.8 --min-assets 30 --min-holding-count 1 --lot-size 100 --candidate-profile standard --sharpe-target 3
```

作用：

- 读取同一个完整特征面板 pkl，不重建全量因子。
- 用核心训练集拟合 12 个 XGBoost 候选模型。
- 只在内部验证集上比较模型参数、top-N、置信度门控和分差门控。
- 按验证 Sharpe 和验证年化收益选择候选。
- 选定后用测试前完整训练期重训最终模型，再对测试集执行一次 long-only 回测。
- 输出参数敏感性、候选排名、切分审计、最终测试收益曲线和逐笔交易记录。

第六步，构建日频流通市值特征缓存：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/build_daily_size_features.py --workers 8 --start-date 20230101 --end-date 20260430
```

作用：

- 从 AKShare 拉取稳定股票池日频行情。
- 用成交量和换手率反推历史流通股本，再计算流通市值。
- 输出 `reports/daily_size_features/run_*/daily_size_features.pkl`，供小市值策略使用。

第七步，运行风格中性和小市值策略候选：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_researched_strategy_backtests.py --workers 4 --top-n-candidates 1,2,3 --top-quantile-candidates 0.02 --confidence-quantile-candidates none,0.9 --spread-quantile-candidates none,0.7 --min-assets 30 --min-holding-count 1 --sharpe-target 3
```

作用：

- 读取既有 XGBoost 验证/测试预测。
- 读取完整特征面板 pkl 和日频流通市值 pkl。
- 构造风格中性残差得分、小市值代理得分、真实流通小市值得分及其 alpha 叠加版本。
- 每个策略家族只用验证集选候选和规则。
- 每个策略家族在测试集只做一次最终回测。

第八步，运行全 A 小市值代理策略：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_all_a_small_cap_backtest.py --size-feature-mode proxy --workers 8 --min-assets 1000 --top-n-candidates 5,10,20,50 --top-quantile-candidates 0.002,0.005,0.01 --confidence-quantile-candidates none --spread-quantile-candidates none --rebuild-panel
```

作用：

- 读取本地 2025/2026 都存在的全 A 分钟数据。
- 构造 `reports/all_a_small_cap_backtest/panel_cache/all_a_small_cap_panel_v2_proxy_20241101_20260430_h24_s24_m0.pkl`。
- 只用验证集选择小市值代理候选和 top 规则。
- 对测试集执行一次 long-only 回测。

第九步，运行全 A 小市值机器学习叠加策略：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_all_a_small_cap_model_backtest.py --panel-cache reports/all_a_small_cap_backtest/panel_cache/all_a_small_cap_panel_v2_proxy_20241101_20260430_h24_s24_m0.pkl --train-start 2024-11-01 --validation-start 2025-03-01 --split-date 2025-05-01 --max-train-rows 2500000 --top-n-candidates 20,50,100 --top-quantile-candidates 0.005,0.01 --confidence-quantile-candidates none --spread-quantile-candidates none --min-assets 1000
```

作用：

- 读取已缓存的全 A 小市值代理面板。
- 只用 `2024-11-01 <= datetime < 2025-03-01` 训练 LightGBM。
- 只用 `2025-03-01 <= datetime < 2025-05-01` 选择候选和规则。
- 测试集只做最终验收。

第十步，运行合成中性策略：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_synthetic_neutral_backtest.py --prediction-run-dir reports/xgboost_long_only_backtest/run_20260526_202355 --top-n-candidates 1,2,3 --top-quantile-candidates '' --confidence-quantile-candidates none,0.85,0.9 --spread-quantile-candidates none,0.6,0.7 --hedge-ratio 1.0
```

作用：

- 读取既有 XGBoost 验证和测试预测。
- 只用验证集选择 top-N、置信度门控和分差门控。
- 测试集计算“多头模型组合收益 - 同期股票池等权基准收益 - 成本”。
- 输出主动收益曲线和指标；该结果需要对冲腿，不是现货纯 long-only。

第十一步，运行小市值状态过滤策略：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_all_a_small_cap_regime_backtest.py --panel-cache reports/all_a_small_cap_backtest/panel_cache/all_a_small_cap_panel_v2_proxy_20241101_20260430_h24_s24_m0.pkl --top-n-candidates 20,50,100 --top-quantile-candidates 0.005,0.01 --target-exposure-candidates 0.5,1.0 --confidence-quantile-candidates none --spread-quantile-candidates none --min-assets 1000
```

作用：

- 读取全 A 小市值代理面板。
- 用验证集估计市场收益、广度、波动和流动性状态阈值。
- 构造小市值 + 短期反转 + 低波动 + 状态过滤候选。
- 测试集只做最终验收。

第十二步，运行小市值开盘反转策略：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_all_a_small_cap_open_reversal_backtest.py --panel-cache reports/all_a_small_cap_backtest/panel_cache/all_a_small_cap_panel_v2_proxy_20241101_20260430_h24_s24_m0.pkl --signal-time 09:55 --top-n-candidates 20,50,100 --top-quantile-candidates 0.005,0.01 --target-exposure-candidates 0.5,1.0 --confidence-quantile-candidates none --spread-quantile-candidates none --min-assets 1000
```

作用：

- 只保留每天 09:55 的信号。
- 入场价格使用下一分钟价格。
- 退出价格使用下一交易日同一入场时点价格，避免把 T+1 策略误写成日内平仓。
- 构造小市值开盘反转、价格健康度过滤、市场状态过滤等候选。

第十三步，运行多因子 CTA 风格策略：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_multi_factor_cta_backtests.py --sharpe-target 3
```

作用：

- 读取 `reports/xgboost_long_only_backtest/run_20260526_202355/` 下既有 XGBoost 验证和测试预测。
- 读取 `reports/feature_panel_cache/feature_panel_7226a2c8f3dd389a.pkl`，不重新构建大特征面板。
- 构造两个 A 股现货 long-only 的 CTA 风格多因子策略：
  - `cta_trend_breakout`：XGBoost 多因子 alpha 预测 + 趋势斜率、均线差、通道突破、动量、ADX 趋势质量、量价资金流。
  - `cta_volatility_scaled`：XGBoost 多因子 alpha 预测 + 风险调整动量、低波动、通道突破、量价资金流。
- 只用验证集选择 top-N、置信度门控和分差门控；测试集只做最终验收。
- 继续执行 signal_lag=1、T+1、100 股整数手、万1成本、不做空、不加杠杆。

原始完整结果目录：

`reports/multi_factor_cta_backtests/run_20260527_192923/`

同命令复现目录：

`reports/multi_factor_cta_backtests/run_20260527_204600/`

最终测试集结果：

- `cta_trend_breakout`：Sharpe 3.909777，年化收益 3.148437，最大回撤 -0.127325，short_exposure 0，交易笔数 163。
- `cta_volatility_scaled`：Sharpe 3.909777，年化收益 3.148437，最大回撤 -0.127325，short_exposure 0，交易笔数 163。
- 两个命名策略的测试 Sharpe 都大于 3，但 `run_20260527_204600` 对 `run_20260527_192923` 的同命令重跑完全复现：`test_strategy_metrics.csv`、`test_equity_curve.csv` 和 `test_trade_events.csv` 一致。
- 去重后的可执行交易路径数量为 1，不能说两个独立策略都成功，也不能把二者当作两个 alpha 加倍配仓。
- 项目历史测试集已经被多次查看，所以该结果仍是当前数据口径下的研究回测；生产判断需要新增未来样本或预先声明的 walk-forward。

关键产物：

- `final_report.md`：标准入口中文总结报告，对齐 `run_20260526_202355` 的报告形式。
- `multi_factor_cta_report.md`：策略专属中文总结报告副本。
- `输出文件说明.md`：根目录所有输出文件用途说明。
- `资料依据.md`：CTA/趋势跟踪资料和本地因子映射。
- `数据切分审计.md`、`split_audit.csv`：验证集和测试集边界。
- `strategy_validation_summary.csv`：两个 CTA 策略在验证集上选中的规则。
- `test_strategy_metrics.csv`：两个 CTA 策略的最终测试指标。
- `交易路径一致性审计.md`、`trade_path_audit.csv`：确认两个 CTA 命名策略是否共享同一条可执行交易路径。
- `cta_trend_breakout/`、`cta_volatility_scaled/`：各策略的 `test_equity_curve.csv`、`测试集收益曲线.png`、`test_trade_events.csv`、`test_round_trips.csv`、`test_open_positions.csv`、`test_metrics.csv` 和 `strategy_metadata.json`。

第十四步，语法和产物检查：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile scripts/run_xgboost_long_only_backtest.py scripts/tune_xgboost_long_only_parameters.py scripts/build_daily_size_features.py scripts/run_researched_strategy_backtests.py scripts/run_all_a_small_cap_backtest.py scripts/run_all_a_small_cap_model_backtest.py scripts/run_all_a_small_cap_regime_backtest.py scripts/run_all_a_small_cap_open_reversal_backtest.py scripts/run_synthetic_neutral_backtest.py scripts/run_multi_factor_cta_backtests.py scripts/audit_factor_feature_library.py scripts/build_factor_features.py quant_backtest/factor_library.py
python3 -m py_compile scripts/run_xgboost_long_only_backtest.py scripts/tune_xgboost_long_only_parameters.py scripts/build_daily_size_features.py scripts/run_researched_strategy_backtests.py scripts/run_all_a_small_cap_backtest.py scripts/run_all_a_small_cap_model_backtest.py scripts/run_all_a_small_cap_regime_backtest.py scripts/run_all_a_small_cap_open_reversal_backtest.py scripts/run_synthetic_neutral_backtest.py scripts/run_multi_factor_cta_backtests.py scripts/audit_factor_feature_library.py scripts/build_factor_features.py quant_backtest/factor_library.py
```

还检查了：

- `test_holdings.csv` 中 `shares` 是否全部为 100 股整数倍。
- `test_metrics.csv` 中 `short_exposure` 是否为 0。
- `test_equity_curve.csv` 是否包含累计收益和回撤列。

## 十、后续更严格的建议

- 保留一段全新的未来数据作为完全未触碰测试集。
- 做 walk-forward：每个窗口只用过去训练和验证，再评估下一个未来窗口。
- 不要因为本次参数调优测试 Sharpe 未达 3 就回头选择测试表现更好的旧参数；旧测试结果只能作为已经触碰过的研究记录。
- 如果要做真正市场中性，必须引入股指期货、ETF 或融券数据，并单独建模对冲腿成本、保证金、基差和可交易约束。
- 如果要做真正小市值，应扩展到全 A 或中证 1000/2000 等更小市值股票池；在稳定沪深 300 大盘股票池内，小市值暴露实证较弱。
- 在已实现 T+1 的基础上，继续补充涨跌停、ST、退市、最低佣金、冲击成本和停牌约束。
- 对持仓数量设置下限，例如至少 10 只股票，以降低单票风险。
- 增加容量和成交金额过滤，防止策略在低流动性股票上不可执行。

## 十一、XGBoost 与多因子 CTA 实盘交接说明

本节专门给后续实盘工程使用。这里的“实盘”不是让脚本直接下真实订单，而是说明如何从现有模型和策略产物生成可审计的目标持仓/委托意图。正式接券商接口前，必须先完成 dry-run、模拟盘、风控和人工复核。

### 11.1 两个策略的标准报告目录

XGBoost long-only 策略目录：

`/home/lxl/seed-emulator/topology/quant/reports/xgboost_long_only_backtest/run_20260526_202355/`

多因子 CTA 风格策略目录：

`/home/lxl/seed-emulator/topology/quant/reports/multi_factor_cta_backtests/run_20260527_192923/`

同命令复现目录：

`/home/lxl/seed-emulator/topology/quant/reports/multi_factor_cta_backtests/run_20260527_204600/`

两个目录都需要重点阅读：

- `final_report.md`：人类可读的标准报告入口。
- `输出文件说明.md`：目录内每个文件的用途。
- `test_metrics.csv` 或 `test_strategy_metrics.csv`：测试集指标。
- `test_equity_curve.csv`：测试集权益曲线。
- `测试集收益曲线.png` 和 `test_return_curve.png`：收益图。
- `test_trade_events.csv`：逐笔买卖事件。
- `test_round_trips.csv`：平仓交易复盘。
- `test_open_positions.csv`：测试期末未平仓持仓。
- `validation_rule_metrics.csv` 或 `validation_rules_*.csv`：验证集规则选择结果。
- `model_metadata.json`、`strategy_metadata.json`、`resolved_config.json`：机器可读配置和参数。

### 11.2 XGBoost Long-Only 策略参数

核心文件：

- 模型文件：`reports/xgboost_long_only_backtest/run_20260526_202355/xgboost_model.joblib`
- XGBoost 原生模型：`reports/xgboost_long_only_backtest/run_20260526_202355/xgboost_model.json`
- 特征顺序：`reports/xgboost_long_only_backtest/run_20260526_202355/feature_list.txt`
- 特征清单：`reports/xgboost_long_only_backtest/run_20260526_202355/used_feature_manifest.csv`
- 稳定股票池：`reports/stable_hs300_factor_correlation/run_20260518_223759/stable_hs300_symbols.csv`
- 可复用历史特征面板：`reports/feature_panel_cache/feature_panel_7226a2c8f3dd389a.pkl`

模型参数：

- 模型：`XGBRegressor`
- `objective=reg:squarederror`
- `n_estimators=220`
- `max_depth=4`
- `learning_rate=0.04`
- `subsample=0.85`
- `colsample_bytree=0.85`
- `min_child_weight=20`
- `reg_lambda=5.0`
- `reg_alpha=0.1`
- `tree_method=hist`
- `eval_metric=rmse`
- `random_state=42`

策略规则：

- 规则名：`top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`
- 每次只选择预测分最高的 1 只股票。
- 目标仓位：`1.0`，但只有门控通过时才持仓。
- 置信度门控：top1 预测分必须大于等于 `0.5530999541282654`。
- 分差门控：top1-top2 预测分差必须大于等于 `0.006096088886260986`。
- 调仓频率：每 1 个采样周期调仓一次。
- 采样周期：24 根分钟 bar。
- 执行延迟：`signal_lag=1`，bar t 生成信号，bar t+1 执行。
- 交易约束：A 股 long-only、不做空、不加杠杆、T+1 卖出、100 股整数手。
- 回测成本：成交金额 `0.0001`，滑点为 0。实盘前必须替换为真实佣金、卖出印花税、过户费、滑点和冲击成本。

测试集结果：

- Sharpe：3.706244
- 年化收益：2.874256
- 年化波动率：0.384842
- 最大回撤：-0.127287
- 成本后总收益：2.669561
- 平均多头暴露：0.107651
- 已平仓胜率：0.641975
- 交易事件数：163
- 总交易手数：263537
- short_exposure：0

重要特征方向：

- 特征总数为 156 个，全部为 `cs_rank_*` 截面 rank 特征。
- 特征必须按 `feature_list.txt` 顺序传入模型。
- 特征生成必须使用当前及历史已完成 bar，不能使用未来收益、未来价格或测试期重估阈值。
- 重要性靠前的特征包括 `cs_rank_range_std_120`、`cs_rank_boll_lower_gap_60`、`cs_rank_boll_lower_gap_20`、`cs_rank_dist_low_120`、`cs_rank_dist_low_60`、`cs_rank_amihud_illiq_20` 等。

### 11.3 多因子 CTA 风格策略参数

核心文件：

- 脚本：`scripts/run_multi_factor_cta_backtests.py`
- 输入预测目录：`reports/xgboost_long_only_backtest/run_20260526_202355/`
- 输入特征面板：`reports/feature_panel_cache/feature_panel_7226a2c8f3dd389a.pkl`
- 根目录配置：`reports/multi_factor_cta_backtests/run_20260527_192923/resolved_config.json`
- 同命令复现配置：`reports/multi_factor_cta_backtests/run_20260527_204600/resolved_config.json`
- 交易路径审计：`reports/multi_factor_cta_backtests/run_20260527_192923/交易路径一致性审计.md` 和 `reports/multi_factor_cta_backtests/run_20260527_204600/交易路径一致性审计.md`
- 策略元数据：
  - `reports/multi_factor_cta_backtests/run_20260527_192923/cta_trend_breakout/strategy_metadata.json`
  - `reports/multi_factor_cta_backtests/run_20260527_192923/cta_volatility_scaled/strategy_metadata.json`

两个 CTA 风格策略：

- `cta_trend_breakout_alpha`：XGBoost 多因子 alpha 预测 + 小权重趋势斜率、均线差、通道突破、动量、ADX 趋势质量、量价资金流。
- `cta_volatility_scaled_alpha`：XGBoost 多因子 alpha 预测 + 小权重风险调整动量、低波动、通道突破、量价资金流。

共同参数：

- `score_blend_strength=0.0005`
- `initial_cash=1000000`
- `commission_rate=0.0001`
- `forward_horizon_bars=24`
- `evaluation_step_bars=24`
- `top_n_candidates=[1, 2, 3]`
- `top_quantile_candidates=[0.02]`
- `confidence_quantile_candidates=[None, 0.9]`
- `spread_quantile_candidates=[None, 0.7]`
- `lot_size=100`

当前验证集选中规则：

- 两个 CTA 风格策略都选中 `top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`。
- `cta_trend_breakout_alpha` 置信度阈值为 `0.5529991269111634`，分差阈值为 `0.0060886263847351`。
- `cta_volatility_scaled_alpha` 置信度阈值为 `0.5529544472694397`，分差阈值为 `0.0061044573783874`。
- 两个 CTA 策略测试交易路径实际一致；2026-05-27 20:46 重跑结果与原始目录完全复现。
- 去重后的可执行交易路径数量为 1，应视为两个 CTA 风格包装共享同一个 156 因子 XGBoost alpha 核心，不应当按两个独立 alpha 加倍配仓。

测试集结果：

- 两个 CTA 风格策略 Sharpe 均为 3.909777。
- 年化收益均为 3.148437。
- 年化波动均为 0.382021。
- 最大回撤均为 -0.127325。
- 成本后总收益均为 2.918502。
- 平均多头暴露均为 0.108982。
- 已平仓胜率均为 0.654321。
- 交易事件数均为 163。
- short_exposure 均为 0。
- `test_equity_curve.csv` 和 `test_trade_events.csv` 完全一致；`test_holdings.csv` 与 `test_round_trips.csv` 只在预测分解释字段不同。

### 11.4 实盘调用的最小技术流程

不要在实盘中读取 `test_predictions.csv` 作为信号。`test_predictions.csv` 只是历史测试产物。实盘必须基于最新已完成分钟 bar 重新生成特征、调用模型、生成目标持仓。

最小流程：

1. 读取稳定股票池 `stable_hs300_symbols.csv`。
2. 为股票池内每只股票加载最新分钟 OHLCV/amount 数据，至少覆盖最大滚动窗口 120 根 bar，建议保留更多冗余。
3. 对每只股票调用 `quant_backtest.factor_library.build_factor_library` 生成基础因子。
4. 按 `used_feature_manifest.csv` 中的 `factor_base` 取基础因子。
5. 在同一个最新已完成 `datetime` 横截面上计算 `cs_rank_*` 特征。
6. 按 `feature_list.txt` 的顺序组成模型输入矩阵。
7. 用 `xgboost_model.joblib` 预测每只股票的 alpha 分数。
8. 对 XGBoost 策略，直接用预测分排序并套用第 11.2 节门控阈值。
9. 对 CTA 风格策略，先在预测分基础上调用 `scripts.run_multi_factor_cta_backtests.add_cta_factor_scores` 生成 `cta_trend_breakout_score` 或 `cta_volatility_scaled_score`，再套用第 11.3 节门控阈值。
10. 若 top1 分数和 top1-top2 分差都通过门控，目标持仓为 top1 股票；否则目标持仓为空仓。
11. 根据账户权益、目标仓位和 100 股整数手生成目标股数。
12. 卖出时必须检查 T+1 批次，今日买入的批次不能卖出。
13. 如果股票停牌、涨停无法买入、跌停无法卖出、行情缺失、特征缺失过多、可交易股票数小于 30、或风控不通过，则不生成真实委托。
14. 先输出 `order_intents.csv` 和 `risk_check.json`，由人工或风控服务确认后再接券商接口。

实盘生成目标持仓时可以复用这些函数：

- `scripts.run_xgboost_long_only_backtest.PortfolioRule`
- `scripts.run_xgboost_long_only_backtest.select_symbols_for_rule`
- `scripts.run_xgboost_long_only_backtest.create_desired_shares`
- `scripts.run_xgboost_long_only_backtest.reduce_buy_plan_to_cash`
- `scripts.run_multi_factor_cta_backtests.add_cta_factor_scores`

### 11.5 实盘前必须补齐的改进点

当前两个策略能跑出高 Sharpe，但仍是研究回测，不是完整实盘系统。实盘前建议按优先级补齐：

1. 成本模型：当前回测把所有税费简化为成交金额 `0.0001`，且滑点为 0。实盘必须拆分买卖佣金、卖出印花税、过户费、最低佣金、滑点和冲击成本。
2. 涨跌停和停牌：当前回测没有完整阻止涨停买入、跌停卖出和停牌成交。实盘必须读取实时盘口、涨跌停价、停牌状态。
3. 单票集中度：当前策略门控通过时只买 1 只股票，目标仓位 100%。如果要降低单票风险，应该先在验证集或未来样本上重测 top-N、最大单票权重、目标仓位上限，不能看测试集后直接改规则。
4. CTA 独立性：两个 CTA 风格策略当前交易路径相同，不能当作两个独立策略叠加仓位。实盘可把 XGBoost 策略作为主策略，把 CTA 分数作为辅助解释或二次确认。
5. 测试集污染：项目历史中测试集已被多次查看。上线前最好做新增未来数据 forward test，或设计 walk-forward，避免把当前测试集 Sharpe 当作稳定生产收益。
6. 历史成分股偏差：当前 257 只稳定沪深300股票池不是严格点时成分股，存在生存者偏差。实盘可以继续使用当前股票池，但研究报告中不能把它描述为完全无偏。
7. 复权和公司行动：当前分钟数据主要按已有 close/open/high/low 使用。实盘前必须确认实盘行情和训练数据复权口径一致，除权除息日必须专项处理。
8. 数据质量监控：每次生成信号前检查时间戳、缺失率、重复 bar、价格非正、high/low 关系、成交量和 amount 异常。
9. 账户和订单状态：必须维护持仓批次、买入日期、可卖数量、冻结资金、撤单、部分成交、废单和成交回报。
10. 风控开关：至少包含最大单票市值、最大日亏损、最大连续亏损、最大换手、最大委托金额、最大持仓数量、禁止交易名单和紧急全局停止。
11. 灰度流程：先历史回放复现，再 dry-run，再模拟盘，再小资金实盘，不允许直接满仓上线。

外部制度参考，实盘前需要重新核对：

- 上海证券交易所交易规则：`https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml`
- 深圳证券交易所交易规则 PDF：`https://docs.static.szse.cn/www/lawrules/rule/stock/trade/W020230217564423808793.pdf`
- 财政部、税务总局关于减半征收证券交易印花税公告：`https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm`

### 11.6 给另一个 Codex 的实盘交接 Prompt

下面这段可以直接复制给另一个 Codex。要求它先做 dry-run 订单生成器，不要直接下真实单。

```text
你现在接手 /home/lxl/seed-emulator/topology/quant 这个 A 股量化项目。请使用 quant-project-context 和 quant-standards。你的目标不是重新调参，也不是看测试集优化 Sharpe，而是把已有两个策略封装成可实盘前 dry-run 的信号和订单意图生成器。

必须先阅读：
1. /home/lxl/seed-emulator/topology/quant/1method/README.md 的第十一节。
2. /home/lxl/seed-emulator/topology/quant/reports/xgboost_long_only_backtest/run_20260526_202355/final_report.md
3. /home/lxl/seed-emulator/topology/quant/reports/xgboost_long_only_backtest/run_20260526_202355/model_metadata.json
4. /home/lxl/seed-emulator/topology/quant/reports/multi_factor_cta_backtests/run_20260527_192923/final_report.md
5. /home/lxl/seed-emulator/topology/quant/reports/multi_factor_cta_backtests/run_20260527_192923/交易路径一致性审计.md
6. /home/lxl/seed-emulator/topology/quant/reports/multi_factor_cta_backtests/run_20260527_204600/final_report.md
7. /home/lxl/seed-emulator/topology/quant/reports/multi_factor_cta_backtests/run_20260527_192923/*/strategy_metadata.json
8. scripts/run_xgboost_long_only_backtest.py
9. scripts/run_multi_factor_cta_backtests.py
10. quant_backtest/factor_library.py

已有模型和参数：
- XGBoost 模型文件：reports/xgboost_long_only_backtest/run_20260526_202355/xgboost_model.joblib
- 特征顺序：reports/xgboost_long_only_backtest/run_20260526_202355/feature_list.txt
- 特征清单：reports/xgboost_long_only_backtest/run_20260526_202355/used_feature_manifest.csv
- 股票池：reports/stable_hs300_factor_correlation/run_20260518_223759/stable_hs300_symbols.csv
- XGBoost 规则：top1，target_exposure=1.0，confidence_threshold=0.5530999541282654，spread_threshold=0.006096088886260986，signal_lag=1，horizon=24，lot_size=100，long-only。
- CTA trend 规则：score_col=cta_trend_breakout_score，confidence_threshold=0.5529991269111634，spread_threshold=0.0060886263847351。
- CTA volatility 规则：score_col=cta_volatility_scaled_score，confidence_threshold=0.5529544472694397，spread_threshold=0.0061044573783874。
- CTA overlay strength=0.0005。注意两个 CTA 策略当前测试交易路径一致，不能当作两个独立策略加倍配仓。

请实现一个新的脚本，建议命名为 scripts/generate_live_order_intents.py。要求：
1. 只生成 dry-run 的 order_intents.csv、target_positions.csv、risk_check.json、live_signal_snapshot.csv，不接券商接口，不发真实订单。
2. 输入参数包括 --strategy xgboost_long_only/cta_trend_breakout/cta_volatility_scaled，--data-root，--account-state，--current-cash，--portfolio-value，--as-of-datetime，--output-dir，--dry-run。
3. 从最新已完成分钟 bar 构造特征，不能读取 test_predictions.csv 作为实盘信号。
4. 对每只股票调用 quant_backtest.factor_library.build_factor_library，按 used_feature_manifest.csv 取 factor_base，再在同一 datetime 横截面计算 cs_rank_*。
5. 按 feature_list.txt 顺序加载特征，用 xgboost_model.joblib 预测 alpha。
6. 对 XGBoost 策略直接用 prediction 排序；对 CTA 策略调用 scripts.run_multi_factor_cta_backtests.add_cta_factor_scores 后用对应 score_col 排序。
7. 应用 top1、置信度门控、top1-top2 分差门控、min_assets=30、target_exposure=1.0、100 股整数手、T+1 可卖数量约束。
8. risk_check.json 必须列出数据新鲜度、缺失率、可交易股票数、停牌/涨跌停过滤、ST/黑名单过滤、当前持仓、目标持仓、买卖金额、现金约束、单票集中度、是否允许下单。
9. 若缺少实时涨跌停、停牌、ST 或盘口字段，脚本必须标记 can_trade=false 或 warning，不能假装实盘约束已完整覆盖。
10. 所有输出写入新的时间戳目录，不覆盖历史报告。
11. 写 README 或 final_report.md 说明调用命令、输入字段、输出字段、失败条件和实盘前人工检查流程。
12. 修改 Python 文件后运行 python3 -m py_compile 和 /home/lxl/anaconda3/envs/quant/bin/python -m py_compile。

验收标准：
- 在一小段历史数据上 dry-run，能生成目标持仓和订单意图。
- 不使用测试集收益或测试集预测做任何调参。
- 输出中明确标记 short_exposure=0、signal_lag=1、T+1、100 股整数手。
- 若没有完整实盘行情字段，默认只允许 dry-run，不允许真实交易。
```

## 新增：固定底仓日内做 T

[4method_inventory_t/README.md](../4method_inventory_t/README.md) 记录固定底仓做 T 的独立研究流程：保留现有 XGBoost + CTA 入场思路，每个年度折重新训练；买入后保留底仓，按昨日可卖股份做日内买卖配对，正常情况下每日收盘恢复日初股数。它与相同底仓买入持有比较净增量收益，并输出 T+1、股数恢复及未完成配对审计。年度留出为 2023—2026；较早年份是含未来训练数据的回顾式 LOYO，2026 只评价截至 04-30 的可用数据。配置入口为 `config/inventory_t_cross_validation.yaml`。
