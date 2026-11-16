# 均衡底仓做T目标研究：v12

本实验把“按原规则建立底仓、持续持有、用昨日可卖股数做T”作为账户合同：每只股票首次自动建仓的实际交易名义额在 **1万—2万元**之间，以 **1.5万元**为目标；账户初始资金100万元，自动底仓使用资金的上限约为85%，不再另设同时持股数量上限。做T收益必须扣除费用和滑点，并按**全部底仓原始含费购入成本**计算降低成本的比例。

当前状态为**四折完整内部研究已完成，未达到进入正式外层的条件，正式外层未运行**。目标仍是各验证年度额外降低底仓成本至少5%，同时真实记录手数、覆盖率和每日股数恢复情况；不能把实现完成、少量成交或内部选参成绩称为目标达成。完整验收口径见[收益与手数协议草案](../reports/weekly_t_goal_diagnosis/capital_lot_acceptance_20260910/protocol_draft.md)。

完整[内部研究报告](../reports/balanced_inventory_t/run_20260910_140512_310726/final_report.md)于 `run_20260910_140512_310726` 保存。四折各55候选，共220项内部账户结果，均见[完整候选表](../reports/balanced_inventory_t/run_20260910_140512_310726/validation_rule_metrics.csv)。下表是用于选参的内部成绩，**不是表中留出年份的外层收益**；多个折使用同一内部年度，不应相加为连续账户利润。

| 拟留出年 | 实际内部年 | 所选规则 | 完整组 / 配对手 | 内部T净额（元） | 内部成本降低率 |
|---|---|---|---:|---:|---:|
| 2023 | 2026 | 49：动量、偏离1%、20分钟、50%旧股 | 2204 / 18773 | −52,440.43 | −6.1697% |
| 2024 | 2026 | hold：无执行合格主动候选 | 0 / 0 | 0.00 | 0.0000% |
| 2025 | 2026 | hold：无执行合格主动候选 | 0 / 0 | 0.00 | 0.0000% |
| 2026 | 2025 | 50：动量、偏离1%、20分钟、100%旧股 | 3969 / 52429 | −155,938.23 | −18.4777% |

[外层决策记录](../reports/balanced_inventory_t/run_20260910_140512_310726/outer_evaluation_decision.json)明确 `all_inner_cost_targets_met=false`、`outer_evaluation_started=false`；[运行身份核对](../reports/balanced_inventory_t/run_20260910_140512_310726/runtime_integrity_audit.json)记录91个输入、38份源码结束时未变。此轮目标未达成；后续动态恢复是独立新实验，不回写上述规则、账户或结论。

此前首轮[烟测报告](../reports/balanced_inventory_t_smoke/run_20260910_135532_126122/final_report.md)覆盖2026外层及该折内部年度各最早15个市场日、最早12只入选股票，55个预声明候选未因烟测收益调整。原运行核验82个输入与37份源码结束时未变；这是流程证据，不是正式收益验证。随后仅修正图表展示，新增[指标与日期预览](../reports/balanced_inventory_t_smoke/run_20260910_135532_126122/engineering_validation/chart_preview/test_return_curve.png)和[修订核对记录](../reports/balanced_inventory_t_smoke/run_20260910_135532_126122/engineering_validation/chart_preview/verification.json)：保留原图及原快照，原1007个文件SHA、年度指标和曲线数值不变；报告器当前SHA与原烟测版本不同，差异仅为图形展示。双环境编译和17项报告测试通过，不代表55候选或正式年度盈利。

## 架构与文件

数据层只读既有、已审计的市场缓存和本折周度入选名单。无T账户先按当时可用的周名单逐步建仓，冻结实际成交的基础计划；各做T候选复制同一计划，在同一现金与行情条件下执行。任何基础成交差异都保留为失败记录。

```text
v4本折周名单及周模型 ── v8对应来源审计、年度拆分
                  │
v11完整输入/执行审计 ── 既有daily/intraday与只读恢复报价
                  │
       无T账户逐步建仓并冻结基础成交计划
                  │
       内部年度55候选真实账户 → 选择执行合格主动规则
                  │
       全部目标年度规则冻结；各内部年成本降低率均≥5%
                  │
       才进入正式外层账户 → 原底仓/做T增量/手数/约束验收
```

| 文件 | 作用 |
|---|---|
| [balanced_t_data.py](../quant_backtest/balanced_t_data.py) | 核对旧审计与输入SHA；读取年度拆分、对应周模型与周名单；保留完整市场日历和持股后的全部信号日。 |
| [balanced_t_manual.py](../quant_backtest/balanced_t_manual.py) | 按年度与声明时间范围处理人工买单，补充请求当前分钟的少量真实报价，以只读覆盖交给规划和账户；记录人工输入及报价来源。 |
| [balanced_t_policy.py](../quant_backtest/balanced_t_policy.py) | 预声明55候选，用已完成信号分钟生成三类透明规则；按内部实际净收益选择执行合格的主动候选。 |
| [balanced_t_engine.py](../quant_backtest/balanced_t_engine.py) | 每股金额约束、持续底仓、共同基础计划、T+1旧股额度、多次T、真实容量和分批恢复。 |
| [balanced_t_reporting.py](../quant_backtest/balanced_t_reporting.py) | 从原始账表重建全部底仓成本、完整配对手数、双腿手数、日分母、各股贡献与目标状态。 |
| [run_balanced_inventory_t.py](../scripts/run_balanced_inventory_t.py) | 配置、来源核验、源码快照、内部候选并行执行、统一冻结及外层评估条件。 |
| [balanced_inventory_t_v12.yaml](../config/balanced_inventory_t_v12.yaml) | 本次实验的完整参数来源；入口校验固定协议和保留的成本参数。 |
| [test_balanced_t_data.py](../scripts/test_balanced_t_data.py) | 市场缓存、来源身份、完整日历与输入不变性。 |
| [test_balanced_t_source_contracts.py](../scripts/test_balanced_t_source_contracts.py) | 本折年度、周名单和周模型训练/排除年份的来源合同。 |
| [test_balanced_t_manual.py](../scripts/test_balanced_t_manual.py) | 5项测试已通过：未启用时不读来源、年度/烟测切片、非法日期/午休/股数/代码拒绝、当前分钟缺价不取未来价、只读覆盖与spawn传递；已有零量报价不得绕过。 |
| [test_balanced_t_policy.py](../scripts/test_balanced_t_policy.py) | 候选枚举、当前信号因果性、方向与恢复期限、内部选择。 |
| [test_balanced_t_engine.py](../scripts/test_balanced_t_engine.py) | 固定底仓、实际金额、整手、旧股预算、恢复及账户约束。 |
| [test_balanced_t_reporting.py](../scripts/test_balanced_t_reporting.py) | 成本分母、人工加仓、零T股票、配对片段、日分母和目标验收；当前17项报告测试已通过。 |

既有 [DenseRecoveryBarStore](../quant_backtest/weekly_t_quote_store.py) 提供只读内存映射报价；交易时钟和当前涨跌停方向检查复用旧公共接口。v12不重建全市场因子、不重新训练周模型，也不训练或复用旧T预测模型。源文件快照以每次运行实际保存的 `source_hashes.json` 为准。

## 底仓与做T合同

原周模型仍按70%下一周涨幅预测排名、30%下一周日均振幅预测排名产生每周top100名单。v12只在名单的实际入场日期到来后尝试新建仓，不能把全年入选股票的并集提前放到年初买入。每只股票自动建仓首次成功后持续持有；掉出周名单不卖出，也不停止读取该股后续持有期的信号。尚未成功建仓的股票可在之后合法入选时再尝试。

计划股数使用已知参考价、买入滑点和100股整手，选择接近1.5万元的数量；实际执行再受当前价格、现金、涨跌停及分钟容量约束。自动成交名义额 `实际成交价×股数` 必须在1万—2万元内，费用另计。容量或现金只允许小于1万元的成交时，拒绝该次自动建仓，保留原因。85%是基础现金预算，账户保留初始本金15%的现金底线供做T使用；并不保证恰好投资85%，也不保证每股相同数量或持有期间市值一直均衡。

不设额外持股数量上限不等于无限资金。实际股票数由周信号、1万—2万元金额区间、整手、可成交容量与现金共同决定。未发生自动再平衡；市值随股价变化，后续人工加仓也会改变均衡程度。

人工加仓文件通过 `manual_buys_csv` 指定，包含 `datetime,signal_datetime,symbol,shares`。请求时间必须晚于其信号时间，数量为正的100股整数倍，执行时间须为引擎支持的实际交易时刻。人工加仓使用账户已有现金，并与无T基准一起冻结；它可以超出首次自动金额区间，但会单列例外，实际含费成本仍增加目标分母。当日新增股数当日不能卖出。

完整人工CSV作为事先提供的外部指令保存SHA，不是模型输入。每折只按评价年份、配置起止日期及烟测实际截止日期取对应请求；范围内的非交易日、午休时刻和非法数量会明确报错，不能按“是否能成交”把指令静默删除。代码须使用本地主板 `SH.600000` 等格式；若提供 `side` 列，只允许 `buy`。人工股票必须在本次评价年度真实日行情支持池内，成功进入固定基础计划的股票还必须被该年度恢复报价缓存覆盖；人工请求不能绕过这些数据覆盖要求。

09:30和15:00请求使用日行情中的开盘/收盘执行报价。其他合法分钟优先读取恢复缓存中的**同股票、同分钟**报价；该键不存在时，只从相应原始年度股票CSV补取请求当前分钟的OHLCV与成交额，通过 `ManualRecoveryStore` 形成稀疏只读覆盖。10:09缺失时不能用10:10报价补成交，已有零量报价也不能当作缺失后寻找更有利价格；缺文件或缺分钟保留为缺价，让执行层真实拒单。原缓存与原行情文件均不修改，读到的完整人工CSV和相关原始股票文件SHA进入输入清单。

`validation_context/manual_context_audit.json` 及外评时的 `test_context/manual_context_audit.json` 保存请求总数/本折范围、日行情请求数、已有缓存命中数、请求原始分钟数/命中/缺失、缺原始文件、股票池覆盖与文件身份。人工计划只注入一次；冻结后重放清空 `manual_buys_csv`，复用同一已冻结买单，避免重复加仓。未配置人工CSV时快速返回原报价store，不额外读取人工文件或原始分钟。

人工首次买入发生在10:09、当日开盘和收盘报价均缺失时，持仓使用截至当时真实成交的**参考价**估值，记录 `mark_datetime=10:09` 并标为陈旧；这不是补造15:00成交价。若之后有有效收盘价，则由收盘价覆盖并清除当日陈旧标记。该场景已纳入28项引擎测试，另5项人工模块测试通过；[真实无T账户前后核对](../reports/weekly_t_goal_diagnosis/run_20260910_134100_engine_inventory/engine_validation/mark_fix_default_comparison.json)显示56股、77日的曲线、实际/基准账本、4312行库存及期末持仓逐值不变。该证据是缺价估值修复及无T回归核对，不能外推为所有做T账户收益都不变；有未恢复股数差时，较新的真实参考价会据实改变对应增量估值。

每日卖出额度来自开盘前已经持有的旧股。卖出再买入、买入再卖出均可，但买入不会补充当天的旧股卖出额度；同股最多同时存在一个待恢复组。`max_cycles_per_day=0` 取消人为次数上限，仍受剩余旧股、现金、容量、时钟和恢复义务限制。例如日初1000股，先后完成两组各500股的T后，当日旧股卖出额度用尽，不能再把买回的股票作为新额度。

首腿使用信号之后的下一交易分钟报价，实际首腿不得晚于14:30；默认最后一个信号时刻为14:00。20或60交易分钟恢复从首腿计划执行时刻起算，跳过午休、不跨日；`close` 从14:55开始恢复。恢复先使用计划时刻及随后可用重试时钟，再使用已声明的公共恢复时钟和尾盘真实报价，可分批成交。缺报价、零量、限价或容量不足不会补造成交；未完成义务继续保留、后续真实时刻再尝试，并计入库存/恢复失败。

无新增底仓的日子，目标是收盘股数等于开盘股数；建仓或人工加仓日应等于开盘股数加实际基础新增股数。持续持有底仓和期末留仓是本实验的正常状态，不能再按旧周度策略将其称为“周末应卖未卖”。但T未恢复、每日股数偏差、负现金、T+1违规和基础计划差异仍是失败，不能被正常留仓掩盖。原事件表中基础订单的 `event_type=weekly` 只是兼容旧表名，`base_order_type` 区分 `automatic/manual`，不表示存在周度卖出。

## 三类信号与55个内部候选

信号只读取当前已完成分钟的VWAP偏离、5/15分钟收益、相对开盘收益及该分钟成交额/量。要求所需值有限、当前成交量和成交额为正。VWAP偏离是信号强度，**不是预期净收益**。

| 规则 | 买先后卖 | 卖先后买 |
|---|---|---|
| `vwap_reversal` | 价格低于当日截至当前VWAP，偏离绝对值达到阈值。 | 价格高于当前VWAP，偏离达到阈值。 |
| `confirmed_reversal` | 满足VWAP低位偏离，且最近5分钟收益严格为正。 | 满足VWAP高位偏离，且最近5分钟收益严格为负。 |
| `momentum` | VWAP偏离为正，且5分钟、15分钟、相对开盘收益全部严格为正。 | 上述四个方向全部为负。 |

三类规则各组合偏离阈值0.3%/0.6%/1%、恢复期限20分钟/60分钟/14:55、旧股比例50%/100%，共 `3×3×3×2=54` 个主动候选；再加一个无T的 `hold` 对照，共55个。每个候选保留后续信号，允许首次因容量受阻后在后续时刻再请求；同刻按偏离绝对值降序、股票代码升序分配有限执行资源。

所有候选继续检查当前已知的恢复方向是否已受限，但不读取未来恢复价格来决定是否发出首腿。v12不再使用旧T模型概率阈值、z分数、模型支持域或绝对日涨跌3%/振幅6%的信号门槛。这些信号改变，以及85%底仓、等金额、持续持有、多次T和期限候选，均为本次显式实验变量；与旧策略的收益差不能全部归因于“择时能力改善”。

## 年度隔离与进入外层的条件

| 外层验证年 | 周模型外层训练年 | 内部选参年 | 内部周名单对应核心训练年 |
|---|---|---|---|
| 2023 | 2024、2025、2026 | 2026 | 2024、2025 |
| 2024 | 2023、2025、2026 | 2026 | 2023、2025 |
| 2025 | 2023、2024、2026 | 2026 | 2023、2024 |
| 2026 | 2023、2024、2025 | 2025 | 2023、2024 |

沿用v4/v8已审计的本折模型、名单和年度拆分；内部周模型排除内部年及外层年，外层周模型使用该折三个训练年。旧周模型原始特征来源区间及跨留出年隔离由原审计提供证据，本次再核对来源SHA及训练/排除年份。透明T规则只在内部年度比较真实账户净收益，不重新拟合T模型。

每折完整运行55个内部候选；主动候选须有完整配对，且库存偏差日、恢复失败日、期末待恢复组、T+1违规和基础计划差异五项均为0，才有选择资格。在此集合中按内部净收益降序、候选编号升序选取；选择器允许负收益，没有100组、置信区间或频率门槛自动关闭。若没有执行合格的主动候选，明确选择hold并记录原因。

**选择规则后，还有正式外层的研究条件：全部目标年度所选规则必须启用，且各自内部底仓成本降低率均达到5%。**全部规则与SHA统一冻结后，才允许开始任何正式外层评估。未满足时保存全部内部候选、失败和负收益，输出“内部研究结果”，不生成外层达标结论。`--internal-only` 则无论内部结果如何都只保留内部研究。烟测允许绕过内部收益条件执行小范围外层，以验证连接与账户流程；该例外明确写入 `outer_evaluation_decision.json`，不能作正式收益证据。

每个外层年度重新使用独立100万元账户，不将四年曲线拼接成连续复利。2023—2025折训练中包含未来年份，是回顾式留一年验证；2026折在时间顺序上前推，但数据仅到2026年4月30日。上述历史区间已经在此前研究中反复查看，本次仍是历史复测，不能称全新、未触碰的测试集。即使内部5%通过，也不保证外层通过。

## 收益、手数与验收分母

令 `B` 为无T基准所有自动建仓和人工加仓的**实际买入名义额加买入费用之和**，包含从未发生T的股票。令 `G` 为完整T组按实际成交价格计算、扣除全部相关费用后的净额。主要成本降低率为 `G/B`；原始账户本金比例 `G/1,000,000` 同时列示。现金流扩展时账户分母另记录原始本金和正向外部投入；本入口人工加仓使用已有现金，不冒充新入金。

T买回不会重新增加或减少原始底仓成本，盈利也不递减分母。成交总额、只发生T的少数股票成本、平均占用资金、期末市值或时间加权本金都不能替代 `B`。平均及峰值持仓市值只作资金占用描述。每股分别列原始含费成本、自动/人工成本、T净额、降低成本率、最终股数、成本权重与盈亏贡献，零T股票保留。

参考价毛差减去滑点成本、再减去手续费，应与成交价净额核对。`net_pnl` 已使用含滑点的实际成交价，不能再次重复扣滑点。账户增量为同一基础计划下“做T账户净值−无T基准净值”；存在未恢复组时，该增量可能包含未完成头寸估值，必须与完整组净额分列，此时不得判5%目标达成。

| 指标 | 含义 |
|---|---|
| `completed_t_pair_lots` | 完整组原始股数÷100，只计一次；这是配对手数。 |
| `total_t_traded_lots` | 全部实际T买卖腿股数之和÷100，包含尚未完成的腿。 |
| `complete_t_pairs` / `recovery_fragments` | 完整组数与实际恢复片段数，分批恢复不增加完整组数。 |
| `market_evaluation_days` | 本次整个市场评估日历，含尚未建仓或完全没有T的日子。 |
| `baseline_old_stock_days` | 至少一只股票在无T基准日初有100股旧股的日子。 |
| `baseline_old_stock_opportunity_stock_days` | 上述旧股资格逐股票、逐日计数；不按未来成交或恢复成功筛分母。 |
| `active_days` / `active_stock_days` | 实际发生成交首腿的日数/股票日数。 |
| `active_stock_day_coverage` | 实际活动与基准旧股机会股票日的交集÷全部机会股票日；交集外活动另列。 |
| `reference_cost_hhi` / `max_reference_cost_weight` | 底仓原始成本集中度及最大股票权重；并列盈利/亏损贡献集中度。 |

例如一组600股，恢复时先100股、后500股，是1完整组、2恢复片段、6配对手、12双腿成交手。日均配对手同时按全部市场日、持仓日、旧股可T日和实际活跃日报告，不能只挑较小分母宣称频率足够。

5%须每个配置中的外层验证年度独立达到，不能用跨年平均补齐，2026也不能年化后判通过。每日库存、T+1、恢复、资金和共同基础计划必须通过核对；自动底仓金额区间另验收。`minimum_completed_pair_lots` 和 `minimum_active_stock_days` 暂未指定数值时，`frequency_target_status=not_quantified`，报告实际活动但不把少量成交称为“足够多”，因此整体 `goal_achieved` 不会宣称完成。这是报告语义，不阻止内部研究或候选交易，也不引入额外审批步骤。

## YAML字段全集

以下逐项对应当前[配置文件](../config/balanced_inventory_t_v12.yaml)。入口会拒绝擅自改变本版本固定候选、金额、期限和收益目标；旧费用、初始本金、信号时刻等还须与来源配置一致。表中列为参数说明，不表示可以在看过外层成绩后修改并仍称同一实验。

| 字段 | 默认值/用途 |
|---|---|
| `research_version` | `balanced_inventory_t_v12`；固定实验身份。 |
| `data_root` | `ashare_minute_data_2023_2026_04/extracted`；原始行情目录，核对与旧缓存来源一致及文件状态。 |
| `market_cache_dir` | `reports/weekly_t_cross_validation/cache/weekly_t_9b6b9bfb3309dd53`；只读 `daily.pkl`、`intraday.pkl` 和元数据。 |
| `source_audit_report_dir` | `run_20260909_182945_388303`所在原报告路径；完整v11模型/输入/执行审计链。 |
| `source_training_report_dir` | `run_20260908_235818_404243`所在路径；v8拆分、原成本配置及冻结周上下文审计。不是重新训练T的入口。 |
| `frozen_selection_report_dir` | `run_20260908_212722_188543`所在路径；v4本折内部/外层原始周名单及对应周模型。 |
| `output_dir` | `reports/balanced_inventory_t`；每次新增唯一运行目录。 |
| `start_date` | `2023-01-01`；与原缓存覆盖合同一致。 |
| `end_date` | `2026-04-30`；2026实际评估截止，不能视作全年。 |
| `test_years` | `[2023,2024,2025,2026]`；目标外层年度，CLI可缩小烟测范围。 |
| `initial_cash` | `1000000`；每折独立原始本金，与原配置一致。 |
| `base_fraction` | `0.85`；基础建仓现金预算，保留初始本金15%的现金底线。 |
| `min_position_notional` | `10000`；自动首次成交名义额下限，不含费用。 |
| `target_position_notional` | `15000`；整手计划建仓的名义额目标。 |
| `max_position_notional` | `20000`；自动首次成交名义额上限，不含费用。 |
| `max_cycles_per_day` | `0`；无额外每日组数上限，仍消耗旧股余额且同股不得重叠待恢复组。 |
| `manual_buys_csv` | `null`；可选人工加仓请求，字段、年度切片、日行情/恢复缓存覆盖及当前分钟补价要求见上文。 |
| `lot_size` | `100`；标准手股数，所有股数按整手处理。 |
| `commission_rate` | `0.0008`；每腿成交额的佣金率，即万分之八。 |
| `min_commission` | `0.0`；本实验不模拟每笔最低佣金，属于成本假设。 |
| `transfer_rate` | `0.00001`；双边成交额过户费率。 |
| `stamp_rate_before_20230828` | `0.001`；2023-08-28之前卖出腿的印花税假设。 |
| `stamp_rate_from_20230828` | `0.0005`；该日期起卖出腿的印花税假设。 |
| `slippage_rate` | `0.0005`；买价上调、卖价下调，各腿5bp。 |
| `max_bar_amount_fraction` | `0.05`；当前分钟成交额最多5%用于本账户成交，首腿和恢复均受限。 |
| `repurchase_buffer` | `0.12`；卖先组按上涨12%的回购参考价及费用预留现金，不是止损收益保证。 |
| `flat_st_limit_rate` | `0.05`；缺真实限价时，单价分钟精确命中前收±5%候选限价的方向限制。 |
| `nonflat_limit_proxy` | `0.098`；缺真实限价时，同向涨跌幅达到9.8%的保守成交限制，对任何bar适用。 |
| `t_avoid_locked_exit_side` | `true`；只按当前信号报价排除恢复方向已受限的首腿；不能预知未来是否锁板。 |
| `max_pending_t_notional_fraction` | `0.20`；同时未恢复T组按首腿参考价计的名义额上限为初始本金20%。 |
| `last_entry_time` | `14:30`；实际首腿执行最晚时间，规则也检查信号时钟。 |
| `fixed_recovery_time` | `14:55`；`close`候选开始恢复时刻，且计划恢复不得晚于此时。 |
| `use_minute_recovery` | `true`；使用真实分钟恢复报价和分批/重试机制。 |
| `signal_times` | `09:35,09:45,10:00,10:30,11:00,13:05,13:30,14:00`；原缓存可用的固定信号时刻。 |
| `horizons` | `[20,40,60]`；用于核对已有恢复缓存及生成公共恢复时钟；不是本次主动候选期限列表。 |
| `signal_families` | `[vwap_reversal,confirmed_reversal,momentum]`；三种透明方向规则。 |
| `entry_deviations` | `[0.003,0.006,0.01]`；当前VWAP偏离绝对值阈值，不是预测净利润。 |
| `recovery_horizons` | `[20,60,close]`；主动候选的计划恢复期限，`close`表示14:55开始。 |
| `t_trade_fractions` | `[0.5,1.0]`；每次申请日初股数的50%或100%，实际还受剩余旧股及容量限制。 |
| `candidate_activation_policy` | `best_execution_valid_active`；内部执行合格主动候选按真实净额优先，无合格者hold。 |
| `profit_target_rate` | `0.05`；原始底仓含费成本的5%，用于内部进入外层条件及各外层收益验收。 |
| `minimum_completed_pair_lots` | `null`；可选每年度最低完整配对手数，未指定则不擅自量化“足够多”。 |
| `minimum_active_stock_days` | `null`；可选每年度最低实际活动股票日数，与上项一样只用于目标报告。 |
| `report_scope` | `formal`；入口根据 `max_symbols/smoke_days` 实际覆盖重设为formal或smoke；内部未外评另明确标记。 |
| `candidate_workers` | `6`；通过spawn并行执行内部候选，共享只读mmap报价，不改变规则选择顺序。 |
| `max_symbols` | `0`；0不截断名单；正值按合法入场日期/排名/代码选前N只作烟测，不是正式持股数量上限。 |
| `smoke_days` | `0`；0用完整年度；正值保留各评估年度最早N个市场交易日，自动标记smoke。 |
| `random_state` | `42`；保留可复现配置字段；本次透明规则与排序没有随机训练。 |
| `dense_recovery_cache_paths` | 下列年度只读mmap缓存映射；路径内容身份进入运行输入清单。 |
| `dense_recovery_cache_paths.2023` | YAML中2023目录的 `dense_recovery_d4bad427…`；2023真实恢复报价。 |
| `dense_recovery_cache_paths.2024` | YAML中2024目录的 `dense_recovery_de8f52a3…`；2024真实恢复报价。 |
| `dense_recovery_cache_paths.2025` | YAML中2025目录的 `dense_recovery_765fb632…`；2025真实恢复报价。 |
| `dense_recovery_cache_paths.2026` | YAML中2026目录的 `dense_recovery_7d3a7435…`；截至4月30日的真实恢复报价。 |

完整缓存目录及内容标识以YAML原值为准。`scheduled_recovery_times`、`internal_only`、`outer_evaluation_policy`、`configuration_source` 是入口派生并保存到 `resolved_config.json` 的字段，不是额外YAML参数。`manual_supported_symbols` 仅由入口按本年真实日行情临时构造，核验人工股票后移除，不是用户另设股票池。运行开始会核对原始CSV大小/修改时间，此项本身不是重新逐个原始CSV计算完整SHA；旧完整执行审计和本次读取PKL/模型/名单的内容SHA分别记录，不能混称。为人工当前分钟补价而读取的原始股票CSV另记录完整内容SHA。

## 复现命令与产物

在项目根目录使用quant环境。先运行各模块测试：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_balanced_t_policy.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_balanced_t_data.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_balanced_t_source_contracts.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_balanced_t_manual.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_balanced_t_engine.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_balanced_t_reporting.py
```

工程烟测保留2026外层和该折对应内部年度的最早15个市场交易日、最早12只合法入选股票；仍逐一运行55个候选：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_balanced_inventory_t.py \
  --config config/balanced_inventory_t_v12.yaml \
  --test-years 2026 --max-symbols 12 --smoke-days 15 \
  --candidate-workers 2 --output-dir reports/balanced_inventory_t_smoke
```

完整内部研究；本命令显式不执行外层：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_balanced_inventory_t.py \
  --config config/balanced_inventory_t_v12.yaml --internal-only
```

正式协议入口。四折内部结果全部达到上述条件后才自动执行外层，否则留下内部研究报告：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_balanced_inventory_t.py \
  --config config/balanced_inventory_t_v12.yaml
```

CLI只提供配置路径、年度、烟测范围、并行度、输出目录及 `--internal-only`；没有跳过正式内部5%条件的参数。完整内部研究和正式入口各自建立新目录，不覆盖旧账户、旧模型或报告。

| 产物 | 检查内容 |
|---|---|
| `resolved_config.json`、`research_protocol.json`、`configuration_source.yaml` | 实际配置及事前实验身份。 |
| `source_snapshot/`、`source_hashes.json`、`input_sources.json`、`runtime_integrity_audit.json` | 运行源码、输入及结束时不变性；当前文件数以实际清单为准。 |
| `year_Y/split.json` | 本折外层、内部和原周模型核心训练年。 |
| `year_Y/validation_context/`、外评时的 `test_context/` | 原周名单、冻结基础请求/成交、规划尝试、基础计划身份及完整估值范围。 |
| 上述上下文目录内 `manual_context_audit.json` | 人工指令年度切片、当前报价来源/缺失、日行情覆盖及输入SHA；未启用也明确记录。 |
| `year_Y/validation_candidates/candidate_NNN/` | 每个候选的规则、决策、因果检查、完整账户、配对、拒单、库存、金额和各股指标。 |
| 各折及根目录 `validation_rule_metrics.csv` | 所有内部候选的正负结果、执行资格及选中标识，不只展示赢家。 |
| `year_Y/selected_rule.json`、`all_rules_frozen_before_outer_test.json` | 每折规则与内部成绩；统一冻结的规则SHA。 |
| `outer_evaluation_decision.json` | 内部5%是否全通过、是否实际外评、是否烟测例外或仅内部研究。 |
| `test_equity_curve.csv`、`test_trade_events.csv`、`test_round_trips.csv` | 无T/做T账户净值、实际各腿成交、恢复片段；通过组ID重建完整配对。 |
| `daily_inventory_audit.csv`、`blocked_orders.csv`、`base_plan_mismatches.csv` | 每日旧股/新增/期末股数、受阻订单和基准基础计划差异。 |
| `position_amount_audit.csv`、`base_entry_plans.csv`、`baseline_trade_events.csv` | 每股规划和实际金额、人工例外、原始成本分母证据。 |
| `test_open_positions.csv` | 正常持续底仓、待恢复T关联、估值时间与期末市值，不能省略。 |
| 完整外评后的 `annual_summary.csv`、`年度.csv`、`goal_evaluation.json` | 各独立账户收益、5%/金额/手数/库存状态及总体目标。 |
| 完整外评后的 `per_stock_cost_reduction.csv`、`t_pair_lot_summary.csv`、`daily_frequency_coverage.csv` | 含零T股票的成本与贡献、配对手数、各日分母与活动覆盖。 |
| `final_report.md`；外评后的 `test_return_curve.png`、`测试集收益曲线.png` | 内部或外层范围明确的报告；年度独立本金收益和人民币T增量曲线。 |

## 研究依据及边界

下列阅读范围为论文摘要；文献给出的是特定模型下的结论，v12据此提出可检验的有限候选，没有估计OU参数、求解最优停止方程，也没有实现论文中的最优交易边界。

| 一手来源及日期 | 文献结论与本实验假设的区别 |
|---|---|
| Richard Martin、Torsten Schöneborn，[Mean Reversion Pays, but Costs](https://arxiv.org/abs/1103.4934)，2011-03-25 | 摘要指出线性交易成本会侵蚀反转交易，交易缓冲区可减少成本；低成本模型中缓冲宽度与成本立方根相关。这里仅据此检验不同偏离阈值，未把0.3%/0.6%/1%称为理论最优。 |
| Tim Leung、Xin Li，[Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit](https://arxiv.org/abs/1411.5062)，2014-11-18，修订2015-05-14 | 在OU价差及交易成本/止损条件下联合研究入场与离场区间。这里不假定单只A股VWAP偏离服从OU，也未实现其止损或最优止盈边界。 |
| Yerkin Kitapbayev、Tim Leung，[Mean Reversion Trading with Sequential Deadlines and Transaction Costs](https://arxiv.org/abs/1707.03498)，2017-07-11，修订2018-01-06 | 在指定扩散模型与成本下，入场、离场期限影响最优停止策略。这里用固定20/60分钟和14:55期限作有限对照；有限期限不保证T+1、容量和涨跌停条件下实际恢复成功。 |

此前[实际交易质量诊断](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/report.md)区分了完整组与恢复片段；[固定标签端点诊断](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_endpoint_diagnosis/report.md)进一步区分原预测目标是否兑现和真实延迟执行。它们是研究动机及历史事后描述，不是v12外层收益，也不能据外层结果反向挑选规则。

本次继续使用分钟OHLCV和既有交易限制近似，没有完整逐笔委托队列、历史逐日ST/特殊限价、停复牌与公司行为的全套点时还原；最低佣金为0也须与实际账户费用区别。期末持仓及缺当日真实价格时的陈旧估值仍应公开。价差、成交和恢复均可能失败，论文、内部5%条件、旧审计通过或工程烟测均不能保证正式验证集获利。
