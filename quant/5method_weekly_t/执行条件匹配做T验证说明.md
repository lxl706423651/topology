# 按已知执行条件匹配做 T 训练域（v7）

v7 只调整核心 OOF 队列中的 T 训练行：每个恢复周期分别保留计划恢复不晚于原 14:30 截止、且初始资金静态槽位可以支持至少一手 T 的信号。周选股、模型输入定义、27 个候选、费用、实际执行和内部选择门槛沿用 [v6 队列协议](选股队列做T验证说明.md)及 [v5 短周期协议](短周期做T验证说明.md)。[正式运行](../reports/weekly_t_cross_validation/run_20260908_231721_245847/final_report.md)已完成，四折内部均未选出合格候选，外层全部关闭 T、净增量为 0，未达盈利目标。正式独立模型、训练条件和执行审计均已完成并通过；审计通过不代表盈利目标达到。

这是训练域匹配假设。静态槽位不包含后续权益变化、实际已成交底仓、现金、成交量、价格限制或未恢复债务，不能称为实际可交易保证。内部和外层推断行不经过这个新过滤，实际交易仍按当天真实旧库存及原引擎规则判断，也没有增加静态价格门槛。

## 文件角色与入口

| 文件或目录 | 类型 | 作用 |
| --- | --- | --- |
| 本文件 | 方法文档 | v7 的训练过滤、来源边界、配置、输出和验证说明 |
| [`config/weekly_t_cross_validation_v7.yaml`](../config/weekly_t_cross_validation_v7.yaml) | 用户输入 | 选择本轮训练域，其余经济参数保持 v6 |
| [`scripts/run_weekly_t_train_domain.py`](../scripts/run_weekly_t_train_domain.py) | 新入口 | 构建 OOF 队列，逐周期过滤后训练，内部选参全部冻结后统一外层评价 |
| [`quant_backtest/weekly_t_train_domain.py`](../quant_backtest/weekly_t_train_domain.py) | 新辅助库 | 只使用已知周价及确定性时钟，返回保留行、逐行审计与元数据 |
| [`scripts/test_weekly_t_train_domain.py`](../scripts/test_weekly_t_train_domain.py) | 新验证脚本 | 13 项独立时钟、股数、输入不变与失败关闭检查 |
| [`scripts/test_weekly_t_train_domain_workflow.py`](../scripts/test_weekly_t_train_domain_workflow.py) | 新流程验证 | 4 项过滤/标签顺序、支持域与模型行一致、推断不加静态门槛及空训练域检查 |
| [`scripts/audit_weekly_t_train_domain.py`](../scripts/audit_weekly_t_train_domain.py) | 新只读模型审计 | 独立重建 OOF、已知训练域、最终训练行、模型/支持域及输入身份，不读取外层收益 |
| [`scripts/audit_weekly_t_train_domain_execution.py`](../scripts/audit_weekly_t_train_domain_execution.py) | 新只读执行审计 | 完整运行结束后，独立重放费用、现金、容量、T+1、恢复时序和共同底仓 |
| [`quant_backtest/weekly_t_cohort.py`](../quant_backtest/weekly_t_cohort.py) | 共用辅助库 | 同折核心年互为 OOF 的周 top100 队列、四项周预测上下文及四周来源 |
| [`quant_backtest/weekly_t_short_horizon.py`](../quant_backtest/weekly_t_short_horizon.py) | 共用辅助库 | 计划时钟、短周期标签、支持域、原净边际和候选资格 |
| 原数据、来源、模型、报价及账户模块 | 共用辅助库 | 保留既有行为；本轮共保存 17 份实际运行源码快照 |
| `year_Y/training_domain_h*.csv` / `.json` | 新生成输出 | 每个周期进入标签及年度过滤之前的已知训练域审计 |

新接口为：

```python
filtered, audit, metadata = filter_core_t_training_domain(
    frame, weekly_universe, horizon, cfg
)
```

`frame` 是已经接好 v6 周预测上下文的核心 OOF 分钟队列。`filtered` 只取保留的原行，原列、特征、标签、索引、顺序、来源时点和属性保持不变；`audit` 包含所有传入行，与原索引一一对应。函数不写文件、不拟合模型，也不代替年度标签和来源隔离。

## 两个条件怎样计算

第一个条件复用生产函数 `planned_recovery_times`：从当前信号的下一交易分钟开始，再前进 `horizon` 个交易分钟。09:31–11:30、13:01–15:00 是合法分钟，午休跳过，不跨自然日。由该时钟计算的计划恢复须在同日且不晚于 `latest_planned_recovery_time`，默认 14:30。

| 当前信号 | 周期 | 下一成交分钟 | 计划恢复 | 时间条件 |
| --- | --- | --- | --- | --- |
| 11:20 | 20 | 11:21 | 13:11 | 通过，跳过午休 |
| 13:49 | 40 | 13:50 | 14:30 | 通过，包含截止分钟 |
| 13:50 | 40 | 13:51 | 14:31 | 拒绝 |
| 13:30 | 60 | 13:31 | 14:31 | 拒绝 |
| 14:40 | 20 | 14:41 | 无同日合法端点 | 拒绝，不补次日标签 |

函数不读取输入已有的 `execution_time`、`recovery_time`、`planned_recovery_time`、未来价格或标签有效性。即使未来恢复价缺失，当前时钟也只由信号及固定交易分钟计算。标签是否适合拟合留给后续原训练筛选处理。

第二个条件使用同 `trade_week / symbol` 周资格表中的 `f_last_close`，即原引擎计划使用的 `universe_previous_close`，按初始资金计算静态股数：

```text
slot = initial_cash × base_fraction / top_n
base_shares = int(slot / (f_last_close × (1 + slippage_rate)) / lot_size) × lot_size
t_shares = int(base_shares × short_t_fraction / lot_size) × lot_size
keep_slot = t_shares >= lot_size
```

默认初始资金 100 万元、底仓比例 95%、100 个固定槽位，每槽 9,500 元；100 股整手、T 比例 25%。参考价 20 元时计划底仓 400 股、T 为 100 股；参考价 30 元时底仓 300 股、T 向下取整为 0，训练行被排除。价格 23.73 元仍能计划 400 股，这是专门验证没有误加佣金分母的边界例子。

公式精确复用引擎的计划数量口径：买入滑点进入参考价格分母，佣金不进入这个计划分母，之后由实际现金检查处理。周资格池不足 100 只时，分母也不改成实际入选数量。若配置改为 200 股整手，则至少需要一手 200 股 T；两级取整规则保持一致。

实际引擎使用当时基准估值预算和已经成交的底仓，可能受容量影响只买到更少股数，也可能因周仓退出逾期等条件禁止 T。这里使用初始资金的机械代理不模拟这些动态状态。因此既不能说保留行一定可成交，也不能把被排除训练行解释为外层永远不能交易的股票。

## 来源、年份与训练顺序

沿用 v6 的四折拆分：

| 外层年份 | T 核心拟合年 | 内部选择年 |
| --- | --- | --- |
| 2023 | 2024、2025 | 2026 |
| 2024 | 2023、2025 | 2026 |
| 2025 | 2023、2024 | 2026 |
| 2026 | 2023、2024 | 2025 |

两个核心年互为 OOF：每个目标年的周 selector 只拟合另一个核心年，排除自己的目标年、内部年、外层年和其余所有可用年份。按原 70% 收益预测排名、30% 振幅预测排名选择周 top100，再取周首日和末日之间的信号。不能使用另一外层折的 selector 代替本折模型。

原四个 `f_weekly_pred_return_rank`、`f_weekly_pred_amplitude_rank`、`f_weekly_score`、`f_weekly_selection_rank` 保持定义。分钟 `feature_source_start` 已取原分钟来源和周模型四个实际观测周来源的最早值。新静态参考价来自同一周表，过滤器要求原来源已经覆盖该周来源，并且周信号早于当周；若缺失、重复、来源不明或尚未覆盖，直接报错，不偷偷修改来源时间。

模型参数训练身份与原始价格来源分开记录。比如给核心目标 2024 生成周预测的参数来自 2025，其输入量价仍可以全部早于当前信号，但参数不属于 2024 当时可用信息。这是回顾性交叉拟合，不会因原始来源早于信号就变成向前部署实验。`source_model_training_years` 与 `source_model_sha256` 继续记录该区别。

逐折流程如下：

1. 在只读缓存上生成会话、周执行及严格来源，按原资格规则准备周池和分钟表，构建两个核心年的 OOF 队列。
2. 对 20、40、60 分钟分别应用已知静态股数及计划截止过滤，保存完整 `training_domain_h*.csv/json`。
3. 过滤完成后才接对应未来标签，再按原 `select_training_rows` 检查核心年份、标签、完整来源区间及排除年份前后 7 个自然日。输入支持域和模型特征筛选只拟合这批最终合法核心样本。
4. 内部年份保留本折 v4 已冻结选股及周预测上下文，用原支持域、净边际及执行规则评价 27 个候选。推断输入不应用新静态过滤。合格规则按内部净增量选择，无合格候选则关闭 T。
5. 四折内部规则全部冻结后才统一评价外层。T 权重直接复制选中核心模型，`outer_t_refit=false`；外层也保留原推断及实际旧库存执行。完整运行逐日核对共同持有基准与 v4 一致。

“其余三年用于训练流程”仍包含两个核心拟合年及一个内部选择年。内部年参与规则选择；外层所用的 v4 周模型按原协议用这三个训练年拟合，包含内部年。T 权重只有两个核心拟合年，不能把这两类模型混同。

## 配置及未改变的门槛

逐项比较 v6 与 v7 YAML，只有两个差异：

| 字段 | 状态和值 | 消费者及作用 |
| --- | --- | --- |
| `t_training_domain` | 新增必需；`known_slot_and_recovery_deadline` | 入口及过滤模块；只接受两个已知条件构成的训练域 |
| `research_version` | 更新必需；`weekly_t_v7_train_domain` | 入口及协议；标记新的研究版本 |

过滤消费的既有字段如下，其值没有为 v7 改动；完整其他参数解释见 v5、v6 文档。

| 既有字段 | 默认值 | 在训练域中的作用 |
| --- | --- | --- |
| `horizons` | `[20, 40, 60]` | 每个周期单独计算计划时钟和训练子集 |
| `latest_planned_recovery_time` | `14:30` | 既有同日计划恢复截止，包含该分钟 |
| `initial_cash` | `1000000` | 静态槽位的初始资金 |
| `base_fraction` | `0.95` | 初始底仓预算比例 |
| `top_n` / `weekly_top_n` | `100` / `100` | 固定槽位数量；两者须一致 |
| `lot_size` | `100` | 底仓及 T 两次向下取整的股数单位 |
| `short_t_fraction` | `0.25` | 静态 T 数量占计划底仓比例 |
| `slippage_rate` | `0.0005` | 计划买入价格中的滑点 |
| `t_training_cohort` | `core_year_oof_weekly_top100_middle_days` | 原同折 OOF 周中训练队列，过滤在其后进行 |

27 个候选仍为 3 个周期 × L2 depth4/L2 depth6/Huber depth4 三种模型 × 10/20/30 bp 双腿费用后净边际。方向双向、每股每天最多一次请求、25% 库存比例及未恢复参考名义额最多初始资金 10% 不变。

费用仍为每腿佣金 0.08%、过户 0.001%、滑点 0.05%、最低佣金 0，卖出印花税按日期 0.1%/0.05%；容量仍为分钟成交额 5%，原限价代理及真实部分恢复保留。净利润已通过含滑点成交价计算，报告拆分成本时不能再次扣滑点。固定 130 个盘中恢复时钟加尾盘六分钟，以及受阻后真实重试、真实遗留仓位的规则不变。

内部硬门槛仍要求至少 100 个真实完整配对、20 活跃日、8 活跃周、20 只股票；日/周配对和绝对损益集中度受限，实际净增量及周块 95% 区间下界均为正，并满足库存恢复、T+1、共同底仓和期末未恢复组合约束。原 0.1%/99.9% 核心输入支持域规则保持，只在过滤后的合法训练数据上重估。不能为得到成交或利润降低这些门槛。

## 输出与独立审计

`year_Y/training_domain_h20.csv`、`h40.csv`、`h60.csv` 包含过滤前的全部核心队列行。主要字段：

| 字段 | 用途 |
| --- | --- |
| `datetime`、`symbol`、`trade_week`、`feature_source_start` | 原信号键及不变的最早来源 |
| `universe_previous_close`、`weekly_reference_signal_datetime`、`weekly_reference_source_start` | 实际使用的已知周参考价和来源 |
| `planned_recovery_time`、`planned_recovery_deadline` | 独立由时钟计算的恢复及截止时间 |
| `same_day_planned_recovery`、`recovery_deadline_eligible` | 同日端点及截止条件 |
| `static_slot_budget`、`static_base_shares`、`static_t_shares`、`static_slot_eligible` | 两级整手结果及资格 |
| `keep`、`drop_reason` | 是否进入后续训练筛选及原因；可能同时有两个拒绝原因 |

`drop_reason` 使用 `keep`、`recovery_after_deadline`、`no_same_day_planned_recovery`、`static_t_below_one_lot`，多个失败以分号连接。每周期 JSON 记录消费配置、保留/剔除计数、条件重叠计数、已知输入与审计 SHA、公式及四份相关源码 SHA。摘要包含原队列索引，独立重建应使用对应原队列，不能随意重排后计算摘要。

其余 OOF selector、预测、`cohort_lineage.csv`、`t_training_cohort_manifest.json`、内外冻结周上下文 manifest、模型双来源 SHA、`validation_rule_metrics.csv`、`selected_rule.json`、`all_rules_frozen_before_outer_test.json`、标准成交/库存/收益文件继续保留。根目录 `source_snapshot/` 与 `source_hashes.json` 保存 17 份实际依赖；过滤模块自己记录的四份依赖身份不能代替完整快照。

本次存在需明确披露的并发源码变化：[runtime_notice.json](../reports/weekly_t_cross_validation/run_20260908_231721_245847/runtime_notice.json)记录 2026-09-08 23:23:26，`annual_portfolio.py` 中本轮未调用的 `run_lot_portfolio` 内嵌 `calculate_fill` 检查被另一任务修改。17 份启动快照仍全部匹配，当前工作区只有其余 16 份文件逐字节相同，不能声称当前 17 份均一致。实际消费的 `calculate_fee`、`calculate_daily_metrics` 源文本及 AST 不变，除该未调用函数体外的模块 AST、该函数签名/默认值/注解/装饰器也不变。复现依据原快照及身份文件；审计使用限定到这个已核对差异的例外，不放宽其他来源、实际费用或模型检查。

专用独立模型审计输出 `independent_train_domain_audit.json/md`，重建当前条件、过滤顺序、原索引、最终训练样本及来源身份；不读取外层收益，可对模型齐全的首折使用 `--test-years 2023`。独立执行审计输出 `independent_execution_audit.json/md`，只在 `final_report.md` 及所有年度结果完成后读取外层成交和收益、核对真实逐笔账目。程序验证、独立审计和盈利目标分别报告。

已完成小样本的复现审计命令为：

```bash
report_dir='reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787'
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_train_domain.py \
  --report-dir "$report_dir" --smoke
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_train_domain_execution.py \
  --report-dir "$report_dir"
```

正式运行的模型审计不加 `--smoke`；模型产物齐全后可执行第一条，第二条须等整个正式运行结束：

```bash
report_dir='reports/weekly_t_cross_validation/run_20260908_231721_245847'
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_train_domain.py \
  --report-dir "$report_dir"
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_train_domain_execution.py \
  --report-dir "$report_dir"
```

## 复现与当前验证状态

从 quant 仓库根目录运行新增模块测试：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_train_domain.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_train_domain_workflow.py
```

模块 13 项、工作流 4 项共 17 项已通过，两个 Python 环境的 `py_compile` 也通过。模块验证覆盖午休、下一分钟延迟、三个周期的 14:30/14:31 边界、无跨日、未误加佣金分母、100/200 股整手、非法价格/配置、缺失/重复周来源、原来源不被改写、非默认重复索引，以及全部未来列扰动后保留结果与审计 SHA 不变。工作流验证三个周期的过滤子集分别正确，接标签及年度筛选后模型和支持域使用同一批行，审计 CSV/JSON 持久化，内部/外层保留高价、晚信号和未来标签缺失的推断行；空训练域会拒绝拟合，不回退到未过滤队列。

真实四折小样本命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_train_domain.py \
  --config config/weekly_t_cross_validation_v7.yaml \
  --max-symbols 8 --smoke-weeks 3 --workers 4 \
  --output-dir reports/weekly_t_cross_validation_smoke
```

小样本已完成于 [`run_20260908_231228_518787`](../reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787/final_report.md)，启动时 17 份源码均匹配快照；后续当前工作区的 16/17 身份及精确例外见上文。四年均关闭 T，增量为 0，[盈利目标](../reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787/goal_evaluation.json)未达成。它验证真实数据接入、逐周期过滤和完整流程，不能证明实际主动 T 或盈利。日志 `/tmp/weekly_t_v7_smoke.log`；小样本仅缩减股票、内部/外层周数及 T 模型资源，核心 OOF 周队列仍完整生成，支持门槛不降低。

小样本[独立模型及训练域审计](../reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787/independent_train_domain_audit.md)已通过：8 个周 selector、36 个 T 模型、12 个支持域、12 份逐周期训练域的已知输入/审计 SHA，以及 8 份冻结周上下文和 4 份外层权重复制均核对一致，全程没有读取外层收益。并发来源变化后，含精确例外的新审计器已重新审完该小样本，8 项来源例外反例及 5 项训练域反例均通过，[审计源码归档](../reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787/independent_train_domain_audit_source.json)保存脚本、测试和身份。[独立执行审计](../reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787/independent_execution_audit.md)也已通过，重放 22 条实际成交、22 条基准成交及 52 条库存记录；本次没有主动 T 成交，这些通过项不能证明 T 策略盈利。正式全股票池的[模型及训练条件审计](../reports/weekly_t_cross_validation/run_20260908_231721_245847/independent_train_domain_audit.md)和[执行审计](../reports/weekly_t_cross_validation/run_20260908_231721_245847/independent_execution_audit.md)均已通过；分别核对8/36/12/12/8/4项模型与上下文证据，以及7039份原CSV、63686笔成交和76084行库存。最大现金/权益误差约1.397e-9元，持有基准与v4逐日误差为0。17份快照匹配、当前16/17及精确来源例外也在正式审计中如实保留。

该小样本首折原始 OOF 队列为 12,544 行，已知条件过滤后 h20/h40/h60 分别为 10,208 / 8,932 / 7,656 行，之后仍须年度和标签筛选。h60 的原队列中，13:30 计划 14:31、14:00 无同日恢复端点，各占 12.5%，合计 25% 被时钟条件拒绝；这一分母是接标签前的原队列。它与 v6 诊断中“已有合法 h60 标签训练行里 14.286% 超过截止”的分母不同，不能直接比较为过滤强弱变化。

已完成正式运行使用以下命令，运行日志为 `/tmp/weekly_t_full_cv_v7.log`：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_train_domain.py \
  --config config/weekly_t_cross_validation_v7.yaml
```

入口默认使用 v7 配置，新建独立带时间目录。应同时阅读 `final_report.md`、`annual_summary.csv`、`goal_evaluation.json` 和各项独立审计，不能把四个独立初始账户的年度曲线拼成连续实盘组合。

## 正式结果与训练内部解释

每折独立初始资金为 100 万元，[年度汇总 CSV](../reports/weekly_t_cross_validation/run_20260908_231721_245847/annual_summary.csv)的费用后账户收益如下。关闭 T 后，持有基准与加 T 策略的账户结果相同。

| 外层年份 | 只持有收益 | 加 T 收益 | T 净增量（元） | 实际完整 T 配对 | 周退出失败账户日 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | −11.6656% | −11.6656% | 0.00 | 0 | 57 |
| 2024 | −16.4992% | −16.4992% | 0.00 | 0 | 37 |
| 2025 | +31.4697% | +31.4697% | 0.00 | 0 | 55 |
| 2026（截至 4 月 30 日） | +8.2277% | +8.2277% | 0.00 | 0 | 22 |

上表为各自实际估值区间的累计收益：2023 为 2 月 13 日至 12 月 29 日（218 个交易日），2024 为 1 月 2 日至 12 月 31 日（242 日），2025 为 1 月 6 日至 12 月 31 日（241 日），2026 为 1 月 5 日至 4 月 30 日（77 日），不是统一完整自然年的年化收益。

[目标判定](../reports/weekly_t_cross_validation/run_20260908_231721_245847/goal_evaluation.json)中净 T 增量合计 0，正增量年份 0/4，`gain_criterion_met=false`。全部关闭 T 导致增量周区间也是 `[0, 0]`，不提供主动做 T 获利的证据。T 股数差异、恢复失败和期末待恢复组合均为 0，但底仓周退出仍累计失败 171 个账户日；2025、2026 年末分别留下 3、4 笔周底仓，`weekly_execution_complete=false`。不能把没有 T 交易造成的库存检查通过称为每日主动恢复能力或完整周退出成功。

[108 个内部候选](../reports/weekly_t_cross_validation/run_20260908_231721_245847/validation_rule_metrics.csv)中，费用后净增量 23 个为正、79 个为负、6 个为 0，全部未通过联合资格。最高正增量来自外层 2024 折的内部 2026：h40 / `l2_depth6` / 20 bp，净增量 +474.04 元，有 153 个完整配对、22 活跃日、12 活跃周和 139 只股票。它并非小于最低样本量的候选，但周块 95% 区间为 `[−682.04, +1,726.48]` 元，同时有 2 个库存不一致日和 1 个恢复失败日，因此仍被原规则拒绝；没有因正收益取消库存或置信门槛。

首折[内部预测校准诊断](../reports/weekly_t_cross_validation/run_20260908_231721_245847/inner_calibration_v7.md)事前固定 h20 / `l2_depth4` 与 h60 / `l2_depth6`，仅分析内部 2026。固定分箱中，两个模型平均预测净价差为正的非空箱，实际平均方向价差均为负，按原费用计算的标称净价差也为负；静态整手代理子集相同。h20 高幅箱只有 35 行、h60 最高幅箱只有 6 行，且信号共享日期与未来窗口。这是内部预测误差的描述，标称两腿价差没有模拟实际容量、延迟恢复或动态资金，不能当作成交收益，也不能据此反向交易或重新挑阈值。

v7 假设来自 v6 首折核心与内部验证中已知时钟/静态股数分布差异，不读取本轮外层收益选择过滤阈值。但同一历史已经经历多个版本研究，这仍是历史复测：2023–2025 的训练包含后续年份；2026 数据只到 4 月 30 日且此前已查看。原始数据的历史 ST、精确限价/盘口、公司行动和证券主表限制继续存在。缩小训练域可能减少样本或改变分布，不能据此承诺净收益、每天股数恢复或周末完全退出。
