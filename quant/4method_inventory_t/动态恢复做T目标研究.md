# 动态恢复做T目标研究：v13

v13在[均衡底仓v12](均衡底仓做T目标研究.md)基础上，新增含费用的动态恢复信号，并统一使用完整交易会话的分钟报价。它仍按原周名单建立每股1万—2万元、目标1.5万元的持续底仓，保持85%基础现金预算和原T+1旧股额度。**完整四折内部研究已结束：每折163候选，共652个。两个选定主动规则亏损，另两折hold，未达到各内部年5%条件，正式外层未运行，目标未达成。**

此前v12四折各55个候选的[完整内部研究](../reports/balanced_inventory_t/run_20260910_140512_310726/final_report.md)未达到5%条件，外层未运行。v13是另一个预声明实验，不回写v12规则或结果，也不把“增加动态退出”预先等同于赚钱。0.3%利润触发和1%止损触发是本次固定工程参数，不是论文求解出的最优边界。

## 完整内部结果：未达到5%，未执行正式外层

完整目录 [run_20260910_143829_552610](../reports/dynamic_inventory_t/run_20260910_143829_552610/final_report.md) 已exit0；[运行完整性记录](../reports/dynamic_inventory_t/run_20260910_143829_552610/runtime_integrity_audit.json)确认294份输入和43份运行源码始末不变。以下均为用于选参的内部成绩，年份第一列是拟留出年，不能读作该年的外层收益。

| 拟留出年 | 实际内部年 | 选中规则 | 完整T组 | 配对标准手 | T净额（元） | 全部底仓成本降低率 |
|---|---|---:|---:|---:|---:|---:|
| 2023 | 2026 | 153 | 1778 | 15566 | −41,977.23 | −4.9387% |
| 2024 | 2026 | 0（hold） | 0 | 0 | 0.00 | 0.0000% |
| 2025 | 2026 | 0（hold） | 0 | 0 | 0.00 | 0.0000% |
| 2026 | 2025 | 153 | 5351 | 41041 | −129,497.76 | −15.3446% |

规则153为momentum、1%入场偏离、原60分钟期限、0.5旧股份额及profit_buffer_stop。两个主动账户的库存/恢复/待恢复/T+1/共同底仓失败均为0，但净额仍为负；另外两折没有执行合格主动候选。所有候选和失败记录见[完整内部表](../reports/dynamic_inventory_t/run_20260910_143829_552610/validation_rule_metrics.csv)，未因未选中或亏损而删除。

[外层决定](../reports/dynamic_inventory_t/run_20260910_143829_552610/outer_evaluation_decision.json)明确 `all_inner_cost_targets_met=false`、`outer_evaluation_started=false`，不是手动只跑内部后省略外层。这四行不能相加冒充四年独立外层投资组合；前三折实际内部年份都是仅到4月30日的2026年，参数选择和股票名单不同。运行完整性通过不等于新的完整账户独立审计或盈利目标通过，旧烟测独立证据范围如下保留。

## 完整内部退出对照诊断

[只读诊断解释](../reports/weekly_t_goal_diagnosis/dynamic_inner_20260910/final_20260910_145431_207941/interpretation.md)与[独立身份/结果核对](../reports/weekly_t_goal_diagnosis/dynamic_inner_20260910/final_20260910_145431_207941/verification.json)已完成：652账户中648个主动账户净额全部为负；432个同入场退出对照中，profit相对fixed的216个账户仅66个改善、净额差中位数−2782.66元，profit+stop仅46个改善、中位数−5321.69元。已核3924份内部账户输入和43份源快照，无外层读取、账户重跑或参数更改。

共同且双方完成的配对子集可以改善，但退出时刻同时改变后续现金、旧股额度和待恢复状态，整账户新增/缺失首腿及数量也随之变化，不能据这个条件子集称整体择时改善。上述跨候选统计是重叠实验的对照分布，不是一只组合的累计收益。

## 会话边界修复与烟测证据

完整首启 `run_20260910_143303_013604` 在读取首个内部年原始行情时失败，尚未计算候选或评价外层。[失败记录](../reports/dynamic_inventory_t/run_20260910_143303_013604/failure_summary.json)保留异常原行 `SH.600228 / 2026-04-01 13:00`：新reader原先只排09:30、随后拒绝其他会话外记录，与原市场缓存按09:31—11:30、13:01—15:00过滤的合同不一致。

修复只恢复原会话时间选择，并保存每股排除条数、时钟及原时间戳；13:00原行不移动到13:01，不创造报价，也不按收益或未来可成交性筛样本。会话内非整分钟、重复和无效OHLC仍拒绝。原raw、首启失败目录和首次烟测均保留，旧身份不覆盖。

第二次[烟测报告](../reports/dynamic_inventory_t_smoke/run_20260910_143633_408118/final_report.md)与首次烟测进行了[逐账户对照](../reports/dynamic_inventory_t_smoke/run_20260910_143633_408118/engineering_validation/session_fix_comparison.json)：2357份账户CSV逐字节相等。随后[独立身份复核](../reports/dynamic_inventory_t_smoke/run_20260910_143633_408118/engineering_validation/independent_session_fix_review/review.md)确认168份metrics JSON相等，选中规则仅输出目录重定位、经济参数和指标相同，并核验43份新运行源码/快照、110份原登记输入。它通过内容身份映射继承[首次168账户完整独立审计](../reports/dynamic_inventory_t_smoke/run_20260910_142405_397251/independent_dynamic_audit/run_20260910_142902_822973/verification.json)的核账证据，没有把身份检查冒称重新执行了全部账户审计。

[四内部上下文数据预检](../reports/dynamic_inventory_t/data_validation/all_inner_source_preflight/verification.json)和上述独立复核按拟留出年2023/2024/2025/2026分别核到1/3/3/1条会话外原记录，均为13:00；这些是逐折统计，不能相加当作不重复的原始记录数。排除记录直接与原CSV核对，会话内真实行及13:01原有缺失保留。

上述修复后完整运行使用 `--candidate-workers 24`，日志 `/tmp/dynamic_t_full_v13_session_fix.log`，已完成全部内部候选，结果见本页首表。43份运行源码和失败首启身份继续保留，当前文档维护未修改它们；烟测及数据预检通过均不表示5%目标达到。

## 架构和文件角色

```text
既有已审计周名单、成本及年度拆分
       ↓
v13动态核＋旧稀疏报价/人工当前分钟补价 → 无T规划 → planning_baseline
       ↓
对冻结股票读取本评价期全部真实分钟 → 新独立只读缓存
       ↓
完整分钟无T基准 baseline
       ↓
同一完整报价/重试/估值流：54个入场候选×3种退出＋hold
       ↓
内部真实账户选择 → 全部规则冻结且各内部年5%通过 → 才正式外评
```

| 文件 | 用途 |
|---|---|
| [dynamic_t_data.py](../quant_backtest/dynamic_t_data.py) | 对因果冻结的底仓股票读取评价期完整分钟；核原始文件SHA、真实缺失和会话时钟，构建独立mmap缓存。 |
| [dynamic_t_policy.py](../quant_backtest/dynamic_t_policy.py) | 枚举163候选，首腿完全沿用v12当前信号规则；退出不是事先从未来路径生成的决策。 |
| [dynamic_t_engine.py](../quant_backtest/dynamic_t_engine.py) | `plan_dynamic_entries` 用同一容量规则规划基础买单；`run_dynamic_portfolio` 负责当前bar观察、下一分钟成交、锁存分批及共同估值。 |
| [dynamic_t_reporting.py](../quant_backtest/dynamic_t_reporting.py) | 复用原始成本、手数和验收函数，保留标准账表；新增v13图形、退出信号与触发计数。 |
| [run_dynamic_inventory_t.py](../scripts/run_dynamic_inventory_t.py) | 来源和源码冻结、基础计划、完整分钟上下文、内部候选并行与正式外层条件。 |
| [dynamic_inventory_t_v13.yaml](../config/dynamic_inventory_t_v13.yaml) | 所有数据、成本、入场和退出参数的完整来源。 |
| [test_dynamic_t_data.py](../scripts/test_dynamic_t_data.py) | 7项数据测试已通过：内容身份、逐原始键/六列缓存核对、完整时钟、缺失保留、13:00会话外排除审计及不补13:01等。 |
| [test_dynamic_t_policy.py](../scripts/test_dynamic_t_policy.py) | 3项规则测试已通过：候选数量/身份、首腿继承及配置约束。 |
| [test_dynamic_t_engine.py](../scripts/test_dynamic_t_engine.py) | 21项测试已通过：动态信号时序、费用/分片/T+1、缺价、09:30/09:31同bar容量及新规划冻结回放。 |
| [test_dynamic_t_reporting.py](../scripts/test_dynamic_t_reporting.py) | 7项报告测试及双环境编译已通过：原成本/目标不变、完整组与触发/片段分开、失败/hold保留、原逐年退出CSV不覆盖、错误时序拒绝。 |

旧 `balanced_t_data/engine/manual/policy/reporting` 继续作为被复用的独立依赖，不修改其原实现。新报告函数是 `write_dynamic_report(directory, cfg, annual_results, validation_table=None)`，其中 `annual_results` 为 `(year, result)` 列表。报告只消费已完成账户，不重跑训练或替换候选。

## 完整分钟fixed对照与两个基准目录

先由 `plan_dynamic_entries` 使用**v13动态账户核**、旧稀疏恢复store及人工请求当前分钟的精确补价，运行无T规划并冻结实际基础买单，保存于每折上下文的 `planning_baseline/`。规划与回放使用同一容量记账规则，包含人工加仓合法时间、真实报价和现金约束；不再调用原v12账户核生成计划。之后才按成功冻结的底仓股票集合读取本期全部分钟行情，不能用未来T收益或能否恢复决定纳入哪些股票、日期和分钟。

`baseline/` 是把同一冻结买单交给v13引擎、用完整分钟流运行的无T基准。它与各T候选使用同一标价流：每分钟按当前未加滑点的参考价更新共同估值，日末有有效收盘价则覆盖；缺收盘时保留最近已知标价和真实时间。`planning_baseline/` 保留稀疏规划报价下的无T账目，`baseline/` 才是本实验完整分钟账户比较口径。两者应保持冻结基础股数、成交费用及现金计划一致；缺价时较完整的估值可以不同，不能误算成T价差。无T规划不使用T利润，也不据其盈亏扩大底仓。

一个必须共同处理的边界是：本项目09:30的 `daily.open` 和开盘容量来自09:31第一根原始分钟bar。完整时间线走到09:31时，不能清零已使用额度，再为同一根bar提供第二份5%成交容量。09:30开盘买入/恢复和09:31人工买入/恢复应在各自账户内共享该bar的已用容量。新规划器与实际回放均使用此规则，避免旧规划先允诺超出同bar容量的人工股数、再让新账户复制时失配。这是执行容量修正，不是利用后续收益改变规则或把额外容量当成择时优势。

相关合成测试已通过：同bar容量为16000元时，09:30自动买入1500股占用15000元，09:31请求人工买入500股仅再成交100股；冻结回放的现金、股数、价格和费用逐值相等。另一个跨日恢复场景剩余200股，首bar只容纳09:30恢复100股，09:31受容量限制拒单，09:32才恢复余下100股。这些是执行合同测试，不能当作实际策略利润；动态引擎当前21项测试通过，收益与完整数据集验证另行报告。

所有退出模式均遍历09:31—11:30、13:01—15:00共240个合法会话分钟，使用相同真实报价、逐分钟恢复重试和共同标价。不要求每股每分钟均有实际数据：缺行保留缺行，零量保留零量，没有插值、向后找可成交价或制造午休bar。09:30、13:00等会话外原始记录依声明时段排除并审计；这与会话内真实缺失不同。缺所持股票原始文件则报错，缺具体分钟则真实保留并在撮合时处理。

因此，v13的 `fixed` 是**完整分钟覆盖下**的固定恢复对照，不等于v12旧稀疏报价/重试。v13与v12差异包含报价覆盖、重试时机和共同估值的改变；应先比较同一v13完整分钟环境中的退出模式，不能把跨版本变化全归因于择时能力。`full_minute_coverage_audit.json` 逐股票列应有分钟格数、实际行数、缺失、原始来源及缓存身份。

## 163个候选和动态退出

旧54个主动候选保持不变：3种入场（VWAP反转、5分钟确认反转、动量）×3个偏离（0.3%/0.6%/1%）×3个原恢复期限（20交易分钟/60交易分钟/14:55）×2个旧股比例（50%/100%）。每个分别搭配下表3种退出，共162个主动候选，再加一个hold。首腿仍由8个原信号时刻的已完成分钟产生，实际首腿最晚14:30；没有新增旧模型概率、z分数或支持域门槛。

| 模式 | 提前恢复条件 | 原期限 |
|---|---|---|
| `fixed` | 不根据价格提前触发。 | 到原预定恢复时刻开始尝试。 |
| `profit_buffer` | 当前估计完整组净额率≥`0.003`。 | 未提前触发则沿用原期限。 |
| `profit_buffer_stop` | 同上；或当前估计完整组净额率≤`-0.01`。 | 未提前触发则沿用原期限。 |

当前估计净额使用首腿实际价格与费用、截至当前已完成的片段净额，再加剩余股份按当前参考价和方向滑点估计恢复的价差，减去相应剩余首腿费用和估计恢复费用。分母固定为 `首腿原始参考价×首腿原始实际股数`，不因分批恢复缩小，也不是全部账户本金或底仓购入成本。此净额率只是退出触发指标；最终5%验收仍使用全部底仓原始含费成本。

只有当前价格/高低/成交量/成交额有限、价格有效且有正成交量和成交额时才用其触发价格条件。不能读取未来分钟收盘价，不能看到当前高低价触及阈值便假装在该价成交。手续费与滑点仍照原设置扣除，0.3%是**估计净额**缓冲，并不保证下一分钟实际净赚0.3%；1%止损也不是损失保证上限。

恢复状态从 `monitoring` 变为 `exit_latched` 后锁存，不因价格回撤、限价或缺量撤单等待更有利价格。最早恢复时刻严格晚于退出信号，按下一交易分钟真实报价执行；不足时逐分钟恢复剩余整手，所有片段最终合并为同一个完整组。期限是首腿前已声明的时间条件，在原期限的前一交易分钟发出恢复信号，即使该观察分钟缺价也保留时钟义务。例如13:01期限对应11:30信号，午休不生成成交。

同分钟顺序为先执行先前锁存的恢复，再执行基础买入和新T首腿，然后更新共同标价，最后观察当前bar产生新退出意图；当前新信号不能返回本分钟前面的成交阶段。未恢复则持有真实债务至后续合法时刻并报告失败，不补造成日终股数一致。

买先后卖仍须预留并消耗昨日旧股；卖先后买的买回不补当天卖出额度。动态提前完成只释放实际相应现金/恢复义务，不重置旧股卖出预算。每股同时最多一个未恢复组，恢复前不另开组；人工当日买入股份也不能被动态卖出。

## 年度、成本和目标协议

年度拆分、v4/v8冻结周模型和名单、人工输入来源要求均沿用[v12完整说明](均衡底仓做T目标研究.md)。2023/2024/2025各折在2026内部选参，2026折在2025内部选参；内部周名单使用该折核心两年模型，外层周模型使用对应三个训练年，v13不重新拟合T模型。

每折只从内部有完整配对、且库存偏差日/恢复失败日/期末待恢复/T+1违规/共同底仓差五项全0的主动候选中，按实际内部净额降序、编号升序选择；净额为负也保留真实选中结果，没有执行合格者才hold。之后要求全部配置年度所选规则均启用、且各自内部成本降低率≥5%，统一冻结后才正式外评。未达时只输出内部报告。烟测是工程流程例外，不能替代外层收益证据。

5%的原始分母是所有无T基准自动/人工底仓实际买入金额加买入费用，包含零T股票；T买回、成交额、平均占用及期末市值均不能替代。完整组实际净额和账户净值增量分列；库存未恢复时不能把带有敞口估值的增量冒称完成降成本。每股贡献、成本权重与集中度均继续保存。

同时列完整组、恢复片段、配对100股标准手、全部双腿手数，以及全部市场日/持仓日/旧股日/活跃日的频率。尚未量化“足够多”的最低手数或活动股票日时，报告 `frequency_target_status=not_quantified`，不宣称整体目标完成，也不另加停用交易的审批或样本门槛。

各年度使用独立100万元，四折净额之和不是复利账户；必须逐年达到目标，不能平均补齐。2026仅到4月30日，不年化后过5%。前三折含未来年份训练，是回顾式留年；2026虽顺序前推，所有相关历史已经被反复研究，仍是复测而非全新未触碰测试。

## 配置字段

[v13 YAML](../config/dynamic_inventory_t_v13.yaml)保留[v12字段全集](均衡底仓做T目标研究.md#yaml字段全集)中的全部49个顶层字段及4个年度缓存路径。除下列身份/目录和新增4项外，其余默认值与语义不变；这样可逐字段核对原现金、金额、费用、手数、来源及时间规则，不从文档省略项推断新的默认。

| 字段 | v13值及用途 |
|---|---|
| `research_version` | 改为 `dynamic_inventory_t_v13`，与新入口固定版本一致。 |
| `output_dir` | 改为 `reports/dynamic_inventory_t`，新实验输出与v12隔离。 |
| `full_minute_cache_dir` | 新增 `reports/dynamic_inventory_t/full_minute_cache`；冻结底仓的完整分钟独立内容寻址缓存。 |
| `dynamic_exit_modes` | 新增 `[fixed,profit_buffer,profit_buffer_stop]`，固定顺序与候选身份，不能事后增删。 |
| `dynamic_take_net_return` | 新增 `0.003`，当前估计完整组净额/原始首腿参考金额的利润触发值。 |
| `dynamic_stop_net_return` | 新增 `0.01`，stop模式触发条件使用其负值，即净额率≤−1%。 |

特别保留：每股 `min_position_notional/target_position_notional/max_position_notional=10000/15000/20000`，`base_fraction=0.85`，`max_cycles_per_day=0`，`lot_size=100`；佣金 `commission_rate=0.0008`、最低佣金0、过户费0.00001、卖出印花税按原日期0.001/0.0005、每腿滑点0.0005、分钟容量5%、回购缓冲12%、未恢复名义额上限20%。`last_entry_time=14:30`、`fixed_recovery_time=14:55`、原三种信号/三个偏离/期限和两种份额均不变。

`dense_recovery_cache_paths` 仍为因果底仓规划和人工上下文提供旧稀疏报价，但规划账户核已改用 `plan_dynamic_entries` 的v13同bar容量规则；新候选实际恢复与共同估值使用 `full_minute_cache_dir` 产生的完整store。`horizons=[20,40,60]` 保留原时钟/缓存合同，主动候选原期限仍为 `[20,60,close]`。实际 `dynamic_exit_mode` 由每个选定规则派生后传给账户，不是另一个可任意覆盖的全局YAML字段。

所有新阈值和既有固定协议均由入口校验；原成本还须与已审计来源配置一致。缓存身份包含真实原始文件SHA、年份、股票、市场日历、读取代码SHA和CSV解析方式；损坏或身份冲突报错，不覆盖旧缓存。配置中 `minimum_completed_pair_lots`、`minimum_active_stock_days` 默认仍为null，只影响目标报告。

## 复现与输出

在项目根目录运行新模块检查；其实际完成状态以保存日志为准：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_dynamic_t_data.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_dynamic_t_policy.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_dynamic_t_engine.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_dynamic_t_reporting.py
```

小范围工程烟测：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_dynamic_inventory_t.py \
  --config config/dynamic_inventory_t_v13.yaml \
  --test-years 2026 --max-symbols 12 --smoke-days 15 \
  --candidate-workers 2 --output-dir reports/dynamic_inventory_t_smoke
```

完整内部研究，明确不执行外层：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_dynamic_inventory_t.py \
  --config config/dynamic_inventory_t_v13.yaml --internal-only
```

按协议运行，所有内部年度均达条件后才自动外评：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_dynamic_inventory_t.py \
  --config config/dynamic_inventory_t_v13.yaml
```

CLI继承配置路径、年度、股票/日期烟测截断、候选并行数、输出目录及 `--internal-only`。完整四折为163×4个内部候选记录，不仅保存赢家；没有跳过正式内部5%条件的开关。

全部原规范账表和成本/手数表沿用v12输出；新增或需区分的产物如下：

| 路径/文件 | 用途 |
|---|---|
| `validation_context/planning_baseline/`、外评时对应 `test_context/planning_baseline/` | v13动态核在旧稀疏store/人工精确补价上的无T规划，用同bar容量规则冻结买单；保留规划时的估值。 |
| 同上下文 `baseline/` | 同一冻结底仓在完整分钟共同标价流下的无T比较账户。 |
| `full_minute_coverage_audit.json` | 完整会话应有格数、实际/缺失行、逐股来源、缓存命中及指纹。 |
| `validation_candidates/candidate_NNN/rule.json`、`t_decisions.csv` | 完整候选及因果首腿请求；退出不从未来路径提前写入首腿表。 |
| 各账户 `exit_decisions.csv` | 每个pair首次锁存的一行退出信号，普通观察和后续重试不重复记为触发。 |
| 根 `exit_decisions.csv`、`dynamic_exit_summary.csv/json` | 外评后的年度聚合及利润/止损/期限触发计数、早触发数和协议。 |
| `test_trade_events.csv`、`test_round_trips.csv` | 实际成交与恢复片段；附原期限/信号/最早时刻/原因，完整组由pair_id重建。 |
| `test_open_positions.csv` | 正常底仓、待恢复剩余量和退出状态；不能只展示已盈利平仓组。 |
| `annual_summary.csv`、`年度.csv`、`goal_evaluation.json` | 原分母、完整配对手、所有日分母、库存与目标状态；动态摘要不替代原失败字段。 |
| `test_return_curve.png`、`测试集收益曲线.png`、`final_report.md` | v13版本及formal/smoke/internal范围明确的图文报告。 |

退出表的关键字段为 `pair_id,symbol,direction,exit_mode,first_signal_time,first_execution_time,original_deadline,exit_signal_time,exit_earliest_time,exit_reason,observed_price,estimated_net_pnl,estimated_net_return,normalization_notional,remaining_shares,realized_fragment_net_pnl`。`exit_reason` 为 `profit_buffer/stop_loss/hard_deadline`。价格条件缺可用观察时净额估计可缺失，但预定期限仍能触发；`estimated_net_return` 不是实际成交利润或预测胜率。

运行保存原配置、全部源码快照及输入SHA，末尾核对未变。包装报告器先复用原canonical输出，再只在当前新目录重绘v13图并追加恢复说明；不会改写balanced源码、v12目录或已由入口保存的逐年退出原CSV。

## 文献依据与限制

[Leung/Li](https://arxiv.org/pdf/1411.5062)讨论含成本的入场/退出及止损区间，[Kitapbayev/Leung](https://arxiv.org/pdf/1707.03498)讨论有限期限下随时间变化的退出边界。它们的过程和成交假设不等同于本项目单只A股库存T；v13没有估计OU参数或求解最优停止方程。完整论证和因果协议建议分别见[动态退出研究备忘](../reports/weekly_t_goal_diagnosis/dynamic_exit_research_20260910/research_memo.md)及[执行与数据覆盖设计](../reports/weekly_t_goal_diagnosis/run_20260910_134100_engine_inventory/dynamic_recovery_design.md)。

固定利润/止损触发会改变成交数量、持续时间、现金可用性和余下旧股使用，不保证每组或总体净收益改善。近似涨跌停、分钟容量、缺价和陈旧估值继续适用；完整240个会话时钟不表示240根真实有效bar，更不表示可保证成交。只有在原分母和真实库存合同下逐年验证，才能报告收益目标是否达到。
