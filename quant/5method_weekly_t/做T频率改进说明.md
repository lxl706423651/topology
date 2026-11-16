# 做 T 频率改进研究（v10）

v10 将内部选参目标改为接近预声明的交易频率，并增加当前信号候选和首腿受阻后的后续尝试。周选股、核心模型权重、实际收费、T 比例和成交引擎沿用原版本。目标是检验能否在执行约束下增加真实完整配对，不能预先认定频率或收益改善。

[正式四折回测](../reports/weekly_t_cross_validation/run_20260909_135906_004933/final_report.md)已完成：完整 T 从 v9 的 34 组增至 **1760 组（51.76 倍）**，但扣费净增量由 +276.31 元变为 **−10,291.38 元**。本轮五项 T 执行失败均为 0，[独立执行审计](../reports/weekly_t_cross_validation/run_20260909_135906_004933/independent_execution_audit.md)已完成通过；周底仓退出风险仍在。[v9 已完成结果](主动做T验证说明.md)仍保留为独立历史记录，没有回写其模型或账本。

## 正式频率与收益：交易增加，成本超过毛价差

来源为本轮 [annual_summary.csv](../reports/weekly_t_cross_validation/run_20260909_135906_004933/annual_summary.csv)、[v9 对照表](../reports/weekly_t_cross_validation/run_20260909_135906_004933/frequency_comparison_v9.csv)和 [验证摘要](../reports/weekly_t_cross_validation/run_20260909_135906_004933/validation_summary.json)。四年各自独立初始资金 100 万元；2026 仅至 4 月 30 日，不是完整年度。

| 年份 | v9→v10 完整 T 组 | 活跃日/可计划日 | 组/可计划日 | 相同底仓收益 | 底仓＋T收益 | T净增量（元） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 1→435 | 126/128 | 3.3984 | −11.665581% | −11.928025% | −2624.447073 |
| 2024 | 9→574 | 136/139 | 4.1295 | −16.499154% | −16.798789% | −2996.353420 |
| 2025 | 21→603 | 138/138 | 4.3696 | +31.469682% | +31.089894% | −3797.871195 |
| 2026 | 3→148 | 45/45 | 3.2889 | +8.227703% | +8.140432% | −872.706469 |
| 合计 | 34→1760 | 445/450 | 3.9111 | — | — | −10291.378158 |

450 个可计划日中有 445 日实际开始 T，覆盖 98.89%；1760 组除以可计划日为 **3.911 组/日**。如果分母换成全部 **778 个估值交易日**，则为 **2.262 组/日**，不能把前一个口径写成“全部交易日每天 3—5 组”。股票日层面为 1760/35675，覆盖率 4.93%；“多数可计划日有交易”不代表每只持仓股票每天都做 T。

实际提交请求为 852/978/897/293 行，共 3020 行；形成 1760 次首腿、1829 个恢复片段和 1760 个完整配对，没有把部分恢复多算成独立组。实际成交较多，扣费结果四年均为负。

| T价差与成本归因 | 合计（元） |
| --- | ---: |
| 参考价毛价差 | +352.000000 |
| 滑点 | −3337.408000 |
| T专属手续费 | −7305.970158 |
| 扣费净T收益 | **−10291.378158** |

参考毛价差仅 352 元，费用与滑点合计 10,643.38 元，超过了毛价差；这解释了本轮频率增加却净亏损。它不是漏扣或重复扣滑点，账户差与配对口径仅有浮点舍入误差。独立年度周增量重抽样的 95% 区间为 **[−11,783.34，−8793.10] 元**，四个年度均为负，未实现盈利目标。图见 [年度收益与净增量](../reports/weekly_t_cross_validation/run_20260909_135906_004933/test_return_curve.png)及 [净增量图](../reports/weekly_t_cross_validation/run_20260909_135906_004933/t_increment_curve.png)。

四年库存差异、恢复失败、期末待恢复、T+1 违规、底仓计划差异五项均为 0；所有 T 都在当日恢复。四个共同基准与 v9 逐日完全相同。本轮恢复目标与周底仓退出是不同约束：周退出仍失败 **171 个年度日/307 个股票日**，2025 年末 3 笔持仓估值 14,464 元，2026 评估期末（4 月 30 日）4 笔持仓估值 36,325 元，其中陈旧报价估值 9750 元。不能把本轮 T 恢复全部成功写成周末已全部退出或持仓均可按估值变现。

本折选定的完整规则均来自内部：2023/2024 为候选 17，L2 depth6、相对分位 0.95、VWAP 偏离 0.0035；2025/2026 为候选 18，同一 profile/分位、偏离 0.005。其内部完整组数分别 153/153/197/641，内部净增量分别 −518.51/−882.38/−1156.27/−4516.46 元。负内部收益照样按事前频率政策启用，没有使用外层结果挑参数。

[工程验证记录](../reports/weekly_t_cross_validation/run_20260909_135906_004933/engineering_validation/verification.json)保存 48 项通过的信号、候选账户、工作流和报告测试及双 Python 编译结果；[运行身份核对](../reports/weekly_t_cross_validation/run_20260909_135906_004933/runtime_integrity_audit.json)记录 234 个输入与 27 份运行源码 SHA 全部一致。[正式独立执行审计](../reports/weekly_t_cross_validation/run_20260909_135906_004933/independent_execution_audit.md)和[独立保存账表复核](../reports/weekly_t_cross_validation/run_20260909_135906_004933/independent_frequency_review/review.md)均已完成通过。独立核对包含 100 个内层候选的保存账本及选择，四个外层账户的 7039 个原始行情 CSV、35,432 个实际成交事件、31,843 个基准事件和 76,084 行库存记录。确认 1760 个完整配对、1829 个恢复片段，最大现金/净值误差约 1.40×10⁻⁹ 元，27 份冻结源码一致，原结果未修改。

训练隔离引用已完成 v8 审计的保存 SHA，本次核对复用核心权重、所选外层模型和支持域身份，没有重新训练。29 项独立审计反例通过，详见[审计归档](../reports/weekly_t_cross_validation/run_20260909_135906_004933/independent_frequency_audit_archive/README.md)。审核确认 T 五项约束本轮全部满足，同时保留周退出失败 171 日/307 股日、期末 7 笔持仓及 9750 元陈旧估值；通过范围是来源、账目和实际执行核算，不能将其写成盈利目标或周退出目标达成。

## 25 个固定候选

每折采用自己的 v8 核心 h20 模型，三个 profile 为 `l2_depth4`、`l2_depth6`、`huber_depth4`。h20 指从信号下一交易分钟开始，再经过 20 个交易分钟，跨午休但不跨日；计划恢复必须不晚于 14:30。

| 候选类别 | 数量 | 信号条件 | 每股每日请求处理 |
| --- | ---: | --- | --- |
| 精确 v9 参考 | 1 | 原来由本折内部选择的模型、净边际和方向规则 | 保留原每日最早请求去重 |
| 绝对净边际 | 3 个模型 × 2 个门槛 = 6 | 按原双腿费用与滑点公式计算预测净边际，要求至少为 0 或 5 bp | 保留所有后来满足条件的信号时刻 |
| VWAP 相对排名 | 3 个模型 × 3 个分位 × 2 个偏离 = 18 | 当前偏离、短期方向、日内风险条件成立后，按预测强度排名 | 保留所有后来满足条件的信号时刻 |

0 bp 绝对门槛表示预测净收益至少覆盖原双腿成本，不表示免收成本。相对排名候选则**没有预测收益必须覆盖费用的门槛**；即使预测幅度很小，也可能形成请求，因此交易增多可能增加净亏损，报告使用实际扣费后的账户收益评价。

相对排名的条件全部来自当前已完成的信号分钟：

- 买先：预测回报大于 0，价格相对当前 VWAP 的偏离不高于 `−gap`，最近 5 分钟收益不小于 0。
- 卖先：预测回报小于 0，VWAP 偏离不低于 `gap`，最近 5 分钟收益不大于 0。
- 两个方向都要求已知日内收益绝对值不超过 3%，已知日内振幅在 0 至 6% 之间；`gap` 固定取 0.0035 或 0.005。
- 在同一信号时刻、本折冻结股票池内满足当前条件的股票中，按 `abs(prediction)` 降序、股票代码升序，选择前 `ceil((1−q)×N)` 个；`q` 固定取 0.70、0.85、0.95。非空候选组至少保留一个。

所有类别继续使用原核心训练支持域、同日计划恢复时钟和当前可见的恢复方向限价检查。排名不读取未来执行价格、恢复价格、标签是否存在或将来能否成交；代码会删除未来字段后核对请求不变。

保留后续信号的含义是：某次首腿因现金、容量或其他执行条件受阻，后面的合格时刻仍可尝试。原引擎每股每天最多一个成功首腿，先前失败不消耗一次成功配对额度；当天买入仍不能新增当天可卖股数。保留多条请求并不允许反复使用当日卖额。

## 频率分母与内部选择

每折先运行相同周底仓的无 T 基准，保存日初可卖旧股。对每个股票日，按原 25% 比例取 100 股整手，至少得到 100 股，且该日属于冻结持仓周的中间交易日、存在合法 h20 计划恢复时钟，才计为一个可计划股票日。

`eligible_inventory_stock_days` 是这样的股票日期组合数；`eligible_portfolio_days` 是至少含一个这样的股票的日期数。分母不使用预测大小、实际现金、是否成交、未来恢复报价或恢复成功筛选。因此它是“具备旧库存和时钟条件”，并不保证实际可交易。新买入的底仓不增加当日旧库存，整手条件也不能用计划 top100 名单替代实际持仓数量。

每折 25 个候选都运行真实内部账户，按以下预声明顺序选择：

1. 只保留 `t_pair_count > 0`，且 `inventory_mismatch_days`、`recovery_failure_days`、`pending_pairs`、`base_plan_mismatch_count`、`t1_violation_count` 五项均为 0 的候选。
2. 计算 `pairs_per_eligible_portfolio_day = 完整配对数 / eligible_portfolio_days`，频率分数为 `min(该值, 3)`。
3. 按频率分数降序，再按内部实际扣费净增量降序、`candidate_index` 升序选择。达到频率目标后，额外交易次数不继续提高第一排序项。

**3 组/可计划日是账户平均频率目标，不是每股每日 3 组，也不是保证每天成交 3 组。**允许选择负净收益候选；不以 100 组、活跃日、集中度、置信区间或净收益为正作为开关。没有符合执行要求的主动候选时明确报错并保留内部记录，不能暗中回退到关闭 T 或删除失败候选。

统计完整配对时按 `pair_id` 去重，部分恢复的多个片段不能多算。实际活跃日和股票日取首腿时间，跨日恢复不能增加主动活动天数。报告的覆盖率采用实际首腿 `(date, symbol)` 与保存的共同基准机会表的交集；实际股数可能因恢复失败而偏离基准，故基准机会之外的活动另列，全部真实首腿仍保留。不能通过移除发生恢复债务的日期来抬高覆盖率。

## 同折来源与执行流程

| 外层年份 | 原核心 T 拟合年份 | 本折内部选择年份 |
| --- | --- | --- |
| 2023 | 2024、2025 | 2026 |
| 2024 | 2023、2025 | 2026 |
| 2025 | 2023、2024 | 2026 |
| 2026 | 2023、2024 | 2025 |

本轮不重新训练模型。读取已审计 v8 的三个 h20 核心模型和 h20 支持域；原训练已包括同折核心年份互为 OOF 的周 top100 队列、静态执行训练域、严格特征来源隔离及同刻队列残差标签。模型预测仍采用残差加固定零共同项先验，不把未来同刻均值加回，更不能称为实际做了市场对冲。

周评分保持下一周收益预测排名 70% 加日均振幅预测排名 30%。内部、外层分别使用本折 v4 冻结周名单及对应模型上下文；连接四项已知周排名特征时，原四周最早来源继续保留。`frozen_selection_report_dir` 指向 v4 原始来源，v8 的导出名单不能冒充另一文件的 SHA 身份。

运行时先在完整同刻截面计算会话特征，并保留完整历史的来源追踪，再限制到冻结名单及执行日期。依次进行本折内部基准、25 个内部候选、选择并冻结；**全部目标年度规则冻结后**，才统一进行外层预测和账户回测。正式全范围还要求精确 v9 参考候选复现原内部请求序列、完整配对数和净增量，不能在外层挑选另一个“参考”。

原执行设置保持：每年独立 100 万元、95% 周底仓预算、固定 top100 分母；T 为当日开盘旧股的 25% 取整手，未恢复名义额上限为初始资金的 10%。首腿下一分钟成交，原真实恢复时钟、尾盘及跨日受阻重试、5% 分钟成交额容量、现金和买回预留继续执行。选出 100 只股票不代表实际每周都成交 100 只。

每腿佣金 0.08%、过户费 0.001%、滑点 0.05%，最低佣金为 0；卖出印花税按原日期配置，2023-08-28 前为 0.1%，此后为 0.05%。费用和滑点均由原引擎实际计入，不因相对排名取消成本覆盖门槛而改变收费。

逐笔 `net_pnl` 已使用含滑点成交价并扣手续费；毛参考价差减滑点再减手续费可以用于归因，但不能从已含滑点的净利重复扣一次滑点。账户净增量为实际权益减共同基准权益，包含未恢复仓位估值及底仓差异，与仅已恢复片段的净利不是同一口径。

## 新文件及配置

| 文件 | 作用 |
| --- | --- |
| [`scripts/run_weekly_t_frequency.py`](../scripts/run_weekly_t_frequency.py) | 验证来源，复用权重，运行内部候选，统一冻结后评价外层 |
| [`config/weekly_t_cross_validation_v10.yaml`](../config/weekly_t_cross_validation_v10.yaml) | 本轮固定候选、频率目标与旧经济参数 |
| [`quant_backtest/weekly_t_frequency.py`](../quant_backtest/weekly_t_frequency.py) | 当前信号漏斗、25 个规则、请求保留及频率排序 |
| [`quant_backtest/weekly_t_frequency_validation.py`](../quant_backtest/weekly_t_frequency_validation.py) | 无 T 基准机会分母、真实内部候选执行及明细保存 |
| [`quant_backtest/weekly_t_frequency_reporting.py`](../quant_backtest/weekly_t_frequency_reporting.py) | 频率、覆盖、v9 对照、实际收益与执行失败报告 |
| [`scripts/audit_weekly_t_frequency_execution.py`](../scripts/audit_weekly_t_frequency_execution.py) | 独立核对保存结果、选择与来源、逐笔账户和约束 |

旧 v9 模块保持原版本；v10 报告只复用已核账的年度数量汇总、周增量计算和图形函数，使用独立频率语义。完整继承参数见 YAML 和 [v9 说明](主动做T验证说明.md)，主要新增或改变字段如下。

| 字段 | 固定值 / 默认来源 | 用途 |
| --- | --- | --- |
| `research_version` | `weekly_t_v10_frequency` | 区分本次研究身份 |
| `candidate_activation_policy` | `frequency_execution_valid` | 五项执行资格后，以封顶频率、净增量、编号排序 |
| `source_active_report_dir` | `run_20260909_124918_833635` | 原 v9 参考规则和真实结果对照 |
| `frequency_horizon_minutes` | `20` | 新候选统一采用 h20 已有模型 |
| `frequency_absolute_net_margins` | `[0.0, 0.0005]` | 绝对候选双腿成本后的额外预测边际 |
| `frequency_rank_quantiles` | `[0.70, 0.85, 0.95]` | 当前合格组中保留的排名尾部比例 |
| `frequency_min_vwap_gaps` | `[0.0035, 0.005]` | 买先向下、卖先向上的已知 VWAP 偏离 |
| `frequency_max_abs_day_return` | `0.03` | 相对排名候选当前日收益绝对值上界 |
| `frequency_max_known_amplitude` | `0.06` | 相对排名候选当前已知振幅上界 |
| `frequency_target_pairs_per_day` | `3.0` | 每可计划日完整配对目标，排序分数在此封顶 |

`source_training_report_dir` 仍指向 v8 正式核心模型；`frozen_selection_report_dir` 仍指向 v4 冻结周选择；四个 `dense_recovery_cache_paths` 复用原真实只读恢复报价。YAML 保留旧模型和 `[20,40,60]` 恢复时钟配置用于兼容验证，不能据此误称本轮重新训练或测试了三种新周期。入口拒绝旧 `candidate_support_requirements` 停用门槛；没有 `--workers` CLI。

## 运行与验证

从项目根目录启动正式四折：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/run_weekly_t_frequency.py \
  --config config/weekly_t_cross_validation_v10.yaml
```

小样本只限制执行股票和周数，复用相同正式核心权重；它会在自己的缩小内部样本重新选择规则，不作为正式选参依据：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/run_weekly_t_frequency.py \
  --config config/weekly_t_cross_validation_v10.yaml \
  --test-years 2026 --max-symbols 20 --smoke-weeks 3 \
  --output-dir reports/weekly_t_cross_validation_smoke
```

已完成的 [2026 年 20 股、3 周小样本](../reports/weekly_t_cross_validation_smoke/run_20260909_135533_893081/final_report.md)内部选中候选 13，完成 23 组、净增量 −148.74 元；小样本外层完成 17 组、净增量 −149.33 元，库存偏差和恢复失败均为 0。外层有 9 个可计划日、66 个可计划股票日，实际活跃 8 日、17 个股票日。这只说明该执行样本产生了真实 T 并保留亏损，不能证明正式四折盈利或每日必有交易。

测试入口如下。频率信号 13 项、候选账户与机会分母 15 项、工作流 6 项和报告 14 项测试，共 48 项已通过；报告测试覆盖负净收益仍启用、片段计数、跨日失败、真实覆盖交集、基准不同时不冒称直接可比，以及输入不变。独立审计与工程测试分别记录，不把单元测试称为策略盈利证明。

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_frequency.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_frequency_validation.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_frequency_workflow.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_frequency_reporting.py
```

以下固定目录的独立审计已完成通过，复现命令为：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/audit_weekly_t_frequency_execution.py \
  --report-dir reports/weekly_t_cross_validation/run_20260909_135906_004933
```

## 输出与应当一起看的指标

根目录 `annual_summary.csv` / `test_metrics.csv` 保存真实配对、请求、频率、交集覆盖、扣费净增量及执行失败；`frequency_comparison_v9.csv` 保存旧 v9 数值和逐日共同基准精确比较。日期或基准不同时 `directly_comparable=false`，尤其不能把缩小股票范围的 smoke 与完整 v9 差额称为规则贡献。

逐年 `validation_rule_metrics.csv` 和 `selected_rule.json` 是**完整参数的精确记录**，包含 `signal_mode`、`rank_quantile`、`min_vwap_gap`、`frequency_score` 等实际适用字段。当前 `validation_summary.json` 的 `selected_inner_candidates` 是部分字段摘要，不能用它替代完整规则文件；完整 CSV 和 JSON 不受摘要省略影响。

| 输出 | 主要核对内容 |
| --- | --- |
| `validation_baseline/`、`validation_eligible_inventory_stock_days.csv`、`validation_frequency_opportunities.json` | 内部无 T 基准、逐股票日机会和分母 |
| `validation_signal_funnels/candidate_*.csv/.json`、`validation_signal_funnel_summary.csv` | 每层当前门控剩余行数、股票日数、日期数、股票数；信号不等于成交 |
| `validation_candidate_audits/` | 各内部候选的真实执行、受阻及库存失败记录 |
| `candidate_rules_frozen_before_inner_execution.json`、`all_rules_frozen_before_outer_test.json` | 候选集合及全部外层规则的冻结时点 |
| `t_predictions.csv`、`test_signal_funnel.csv`、`t_decisions.csv` | 原保存预测、当前信号和实际提交请求；参考每日去重、新候选保留后续时刻 |
| `eligible_inventory_stock_days.csv`、`frequency_opportunities.json` | 外层共同基准机会分母，用于交集覆盖 |
| `test_trade_events.csv`、`test_round_trips.csv` | 首腿、恢复片段、完整配对、真实费用及配对净利 |
| `daily_inventory_audit.csv`、`blocked_orders.csv`、`test_open_positions.csv` | 每日与每股日库存差异、恢复受阻、期末债务和陈旧估值 |
| `baseline_trade_events.csv`、`weekly_entry_plans.csv`、`base_plan_mismatches.csv` | 同一周底仓计划、实际成交与基准偏差 |
| `input_sources.json`、`source_hashes.json`、`source_snapshot/`、`runtime_integrity_audit.json` | 实际读取输入、运行源码和运行前后身份 |
| `final_report.md`、`test_return_curve.png`、`t_increment_curve.png` | 年度收益、频率、成本、恢复失败及风险的共同展示 |

`validation_summary.json`、`frequency_reporting_status.json` 和 `goal_evaluation.json` 都是只读报告结果，标记 `report_only=true`、`affects_activation=false`。净增量正负、周块置信区间、频率目标是否达到不会反向改变已冻结规则。报告保留 T+1、期末待恢复、底仓差异、周退出失败及陈旧估值；不能只展示成交数增长或把期末 `pending=0` 写成每日股数均恢复。

## 研究边界

本轮依据既有历史研究和用户的频率需求设计，旧外层结果已经查看。虽然新候选排序仅使用各折内部数据，仍属于历史重新评价，不能称为全新独立未见测试。2023–2025 的留年折含未来训练年份，2026 仅覆盖现有数据至 4 月 30 日；四个年度各自独立本金，不拼接成连续复利组合。

频率高低、扣费净收益和每日恢复约束是三项不同结果。训练内部零恢复失败不保证外层零失败；频率增加可能放大交易费用、模型误差或极端价格下无法恢复的暴露。缺真实可交易价格、停牌/无量或限价受阻时保留实际库存并重试，不补造买卖或删除这些日期。

原历史 ST、公司行动、股票主表、近似限价和缺乏盘口的限制保留，代码和账目核对通过也不能消除这些数据限制。周块统计区间不能充分覆盖历次研究选择偏差；本次结果表明频率提升而扣费亏损，不能据此保证未来盈利。
