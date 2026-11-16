# 主动做 T 验证（v9）

v9 按用户要求移除阻止开启 T 的额外统计门槛，复用 [v8](市场残差做T验证说明.md) 已训练的核心模型和内部回测记录，重新选择主动规则并回测。内部样本不足 100 对、活跃日/周/股票较少、集中度偏高、周置信下界不为正或内部净收益为负，都不再自动关闭 T。费用、首腿信号规则、T+1、现金、整手、容量和实际恢复约束保留。[正式四折回测](../reports/weekly_t_cross_validation/run_20260909_124918_833635/final_report.md)已完成：四年均有实际 T 成交，共 34 个完整配对，净增量合计 +276.31 元；2025 年仍出现 4 个股数偏差日、2 个恢复失败日，不能称每日股数合同已满足。

这不是把旧 `enabled=false` 一律改成 `true`。每折先从原内部候选中选择实际完成过 T 且满足执行要求的一项；不存在这样的候选时明确报错，不静默回退为零交易。启用后的未来交易仍可能亏损或恢复失败，必须与净收益一起披露。

## 从内部结果选择主动规则

固定政策名为 `active_execution_valid`，对每折原 27 个内部候选按以下顺序处理：

1. 要求 `t_pair_count > 0`，即至少有一个实际完成的 T 配对。
2. 要求 `inventory_mismatch_days`、`recovery_failure_days`、`pending_pairs`、`base_plan_mismatch_count`、`t1_violation_count` 五项均为 0。这保留用户的库存及实际交易合同，不能以零交易候选代替主动策略。
3. 对符合条件的候选按内部 `incremental_net_pnl` 从高到低排序；并列按原 `candidate_index` 从小到大。允许选择负净收益候选，不读取外层结果排序。
4. 写入新的 `selected_rule.json`，明确 `enabled=true`；复制对应核心模型和同周期支持域。所有评价折都冻结后才进行本轮外层预测及账户回测。

原 `strict_eligible`、`rejection_reasons`、样本量、集中度、周置信区间等列保留为历史参考。新注释列 `active_execution_eligible` 和 `execution_rejection_reasons` 单独表示本轮执行资格，`selected` 表示本轮选择；不能把旧未达标记录改写成已通过旧验收。若没有符合条件的主动候选，入口抛出错误，不能暗中放松五项条件或改用空仓。

内部周底仓自身的退出受阻不纳入这五项 T 差异筛选，因为共同基准也承受它；实际外层仍报告周退出失败和遗留持仓。训练内部恢复合法不保证未来合法，外层出现库存差异时照实计入，不重新选参或删除失败日期。

## 模型、特征与执行复用

| 外层年份 | 原核心 T 拟合年 | 原内部选参年 |
| --- | --- | --- |
| 2023 | 2024、2025 | 2026 |
| 2024 | 2023、2025 | 2026 |
| 2025 | 2023、2024 | 2026 |
| 2026 | 2023、2024 | 2025 |

三个非外层年份仍分别承担原核心拟合和内部选择，但 v9 不重新训练任何权重。每折从 v8 的 `inner_t_h周期_profile/` 复制所选模型；同周期 `support_h周期.json` 作为输入支持域。保留原严格年度来源、核心 OOF 周队列、静态训练域和残差标签构造的训练证据。未来同刻队列均值只属于原训练标签，推断仍使用残差预测加固定零共同项先验，不把未来均值加回。

原 v8 外层 `t_predictions.csv` 只包含当时选中默认模型的输出，不能用于其他候选模型。v9 对本轮选中的模型重新推断：读取原 `weekly.pkl`、`intraday.pkl`、`daily.pkl`，在完整同刻截面上增加会话特征，并用完整历史日历追踪来源；然后按本折冻结周名单选择执行样本，连接本折 v4 外层周模型的四个已知排名/分数特征及四周最早来源。不会借其他折的周模型或重选股票。

首腿仍执行原五项特征支持域、双腿费用后 10/20/30 bp 边际、当前恢复方向风险过滤和每股每日最早请求。这些是既有逐信号规则，移除统计晋级门槛不等于取消成本或每个信号都成交。候选仍为 20/40/60 分钟 × 三个回归 profile × 三个净边际，合计每折 27 项；T 数量为旧股的 25% 再取 100 股整手，计划恢复不晚于 14:30。

实际成交使用现有四年 `DenseRecoveryBarStore` 只读报价，校验原数组 SHA；不重建短周期标签或两亿多行恢复行情。下一分钟首腿、计划恢复及其后五个交易分钟、固定盘中时钟和尾盘重试、10% 初始资金待恢复名义额上限继续保留。缺行情、零量、容量不足或限价受阻不能补造成交；当日新买入股数不能重新成为可卖旧股。

每腿佣金 0.08%、过户费 0.001%、滑点 0.05%，最低佣金为 0；卖出印花税在 2023-08-28 前为 0.1%，此后为 0.05%。底仓仍按共同基准计划生成，原初始资金、95% 底仓预算、固定 top100 和 5% 分钟成交额容量不变。过去周价格、日均额、收盘流动性及历史资格也保留。

## 文件与配置

| 文件 | 作用 |
| --- | --- |
| [`scripts/run_weekly_t_active.py`](../scripts/run_weekly_t_active.py) | 验证源报告、选择并冻结主动规则、复用模型预测、执行新账户及保存来源 |
| [`config/weekly_t_cross_validation_v9.yaml`](../config/weekly_t_cross_validation_v9.yaml) | 主动政策、源报告和恢复缓存路径；保留原经济与特征配置 |
| [`quant_backtest/weekly_t_activation.py`](../quant_backtest/weekly_t_activation.py) | 仅从内部候选表判断执行资格、排序，拒绝用零交易回退 |
| [`quant_backtest/weekly_t_active_reporting.py`](../quant_backtest/weekly_t_active_reporting.py) | 报告年度实际收益、主动状态、T 专属次数及失败，统计结果不改变开关 |
| [`quant_backtest/weekly_t_residual.py`](../quant_backtest/weekly_t_residual.py) | 复用原残差模型推断，不重新拟合 |
| [`quant_backtest/weekly_t_engine.py`](../quant_backtest/weekly_t_engine.py) | 原周底仓和库存 T 成交账本、T+1、资金、容量及恢复失败 |

相对 v8 的新设置如下；完整继承值见 YAML，以及 [短周期规则](短周期做T验证说明.md)、[OOF 队列](选股队列做T验证说明.md)、[训练域](执行条件匹配做T验证说明.md)和 [残差模型](市场残差做T验证说明.md)。旧文档中的统计停用逻辑只描述旧版本，不适用于 v9。

| 配置字段 | 默认值 / 要求 | 用途 |
| --- | --- | --- |
| `research_version` | `weekly_t_v9_active_execution_valid` | 新独立研究版本，入口严格校验 |
| `candidate_activation_policy` | `active_execution_valid`；当前唯一支持值 | 完整 T 大于 0、五项执行失败为 0，再按内部净增量排序；统计门槛不控制开关 |
| `source_training_report_dir` | `reports/weekly_t_cross_validation/run_20260908_235818_404243` | 读取 v8 模型、各折内部候选、拆分、缓存和源码身份 |
| `dense_recovery_cache_paths.2023` / `.2024` / `.2025` / `.2026` | YAML 中各年已发布的完整 dense 目录 | 复用相应年份真实恢复报价，打开时校验身份和数组 SHA |
| `candidate_support_requirements` | v9 配置中删除，入口拒绝接收 | 旧阈值只留在源报告供解释，不作为本轮停用参数 |
| `bootstrap_samples` / `random_state` | 沿用 2000 / 42 | 计算可复现的统计参考区间，不参与启用 |
| `max_symbols` / `smoke_weeks` / `test_years` | 默认 0 / 0 / 四年 | 仅控制本次执行样本；不会缩小已训练模型或改变候选、费用 |

入口核对源报告的模型、训练、特征、费用和信号配置必须兼容；不能改费用后继续把旧内部利润作为新费用下的选择依据。旧训练超参和 `workers` 等保留用于来源一致性，本轮不执行训练；CLI 没有 `--workers`。

## 运行与验证

从项目根目录运行完整四折：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_active.py \
  --config config/weekly_t_cross_validation_v9.yaml
```

小样本仍从正式 v8 来源选择规则，先保持完整截面生成特征，再只取 2026 年前 3 个冻结持仓周及按代码排序前 8 只已选股票执行：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_active.py \
  --config config/weekly_t_cross_validation_v9.yaml \
  --max-symbols 8 --smoke-weeks 3 --test-years 2026 \
  --output-dir reports/weekly_t_cross_validation_smoke
```

`--source-training-report-dir` 可以显式指定兼容的已完成源报告，不能用来在外层结果之间挑选；旧 8 股 v8 smoke 来源内部没有主动成交候选，会按本轮规则报错，因此上述小样本使用正式训练来源。即使只取几周发起底仓，估值和受阻退出仍延伸至该年现有真实交易日末尾。

报告模块已双 Python 编译，并用既有开关诊断 B 的已保存账本作只读验证：9 首腿/9 恢复/9 完整配对、负净增量和真实股数失败均完整展示，统计未达标不会改动 `t_enabled`；关闭年不能被悄悄标成主动，配对计数与原表冲突会拒绝，输入表逐值不变。此验证没有重跑账户或训练，证据见 [报告验证记录](../reports/weekly_t_active_reporting_validation/verified/reporting_validation.json)。它只验证报告实现，不是 v9 正式成绩。

2026 年 8 股、前 3 周执行小样本已完成于 [run_20260909_124808_789045](../reports/weekly_t_cross_validation_smoke/run_20260909_124808_789045/final_report.md)。规则保持主动启用，240 行预测在这一小范围内没有触发首腿，实际请求和 T 成交均为 0；这是小样本信号结果，不是统计门槛再次关闭策略。正式全范围结果及已完成的独立审计在下一节单独列出。

## 正式四折结果

来源为 [annual_summary.csv](../reports/weekly_t_cross_validation/run_20260909_124918_833635/annual_summary.csv)。每年独立初始资金 100 万元，四年不是一个连续账户；收益包含期末仍持有股票的估值，2026 只有 77 个交易日。

| 年份 | 实际区间 / 日数 | 底仓收益 | 主动账户收益 | T 净增量（元） | 请求 / 完整 T 配对 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2023 | 02-13—12-29 / 218 | −11.665581% | −11.665570% | +0.108077 | 1 / 1 |
| 2024 | 01-02—12-31 / 242 | −16.499154% | −16.498386% | +7.676313 | 9 / 9 |
| 2025 | 01-06—12-31 / 241 | +31.469682% | +31.495128% | +254.466438 | 27 / 21 |
| 2026 | 01-05—04-30 / 77 | +8.227703% | +8.229109% | +14.059694 | 7 / 3 |
| 合计 | 四个独立账户 | — | — | +276.310523 | 44 / 34 |

实际共有 34 次首腿、34 次恢复、34 个完整配对及 34 个恢复片段，没有把周底仓买卖算作 T。四年的逐行信号数为 1/9/32/7，每日去重后的请求数为 1/9/27/7；启用、信号、请求和实际成交是不同层次。

| 外层折 / 内部年 | 已冻结候选 | 内部完整对数 / 净增量（元） |
| --- | --- | ---: |
| 2023 / 2026 | h20、L2 depth4、20 bp | 1 / +36.58 |
| 2024 / 2026 | h20、L2 depth4、20 bp | 2 / −21.93 |
| 2025 / 2026 | h20、L2 depth6、20 bp | 15 / +220.21 |
| 2026 / 2025 | h20、L2 depth4、20 bp | 14 / +207.92 |

2024 折确实允许负内部净增量候选启用，其余折也没有因少于原 100 对而关闭。各折选择均来自本折内部表，不是按上表外层收益反选模型。

[报告参考状态](../reports/weekly_t_cross_validation/run_20260909_124918_833635/goal_evaluation.json)记录正净增量年度为 4/4，汇总周块 95% 区间为 [−276.02，+909.76] 元，下界仍为负。这只用于解释历史不确定性，没有触发停用或回退。净增量为正不等于已经证明明显、稳定的额外收益。

2025 年存在两次跨日恢复：[独立逐笔核对](../reports/weekly_t_cross_validation/run_20260909_124918_833635/independent_cross_day_execution_2025.md)记录 SH.603215 在 06-18 发起、06-19 恢复，净损失约 122.65 元；SH.601137 在 08-13 发起、08-14 恢复，净损失约 66.69 元。两次合计 −189.34 元，原限价及无量阻单保留；恢复次日相对该日开盘的股数变化也计入偏差，因此共有 4 个股数偏差日、2 个恢复失败日。四年期末待恢复配对、T+1 违规及共同底仓计划偏差均为 0，不能据此抹去此前的每日失败。

共同底仓仍有 171 个周退出失败年度日、307 个股票日。2025 期末 3 项持仓、市值 14,464 元；2026 期末 4 项、市值 36,325 元，其中 9,750 元使用前日价格估值，不能当作可按该价格即时变现。

完整[独立执行审计](../reports/weekly_t_cross_validation/run_20260909_124918_833635/independent_execution_audit.md)已通过，覆盖 7,039 个原始行情 CSV、31,911 条实际事件、31,843 条基准事件和 76,084 行库存。34 对的参考价毛差合计 500 元，减去滑点 71.40 元和费用 152.2894775 元，得到净增量 276.3105225 元；现金/权益最大复算误差约 1.397×10⁻⁹ 元，各年基准与冻结 v4 逐日差异为 0。通过的是来源、成交和记账核验，2025 年股数恢复失败和周度退出风险并未因此消失。

模型来源引用原 v8 已完成的严格训练隔离审计，本次逐文件验证 4 个选中核心模型副本、对应支持域和内部选择，并未重新训练或重新审查完整 36 个模型。23 份原运行快照、报告结束后修订、186 项原输入不变和逐年核算证据绑定在 [final_review.json](../reports/weekly_t_cross_validation/run_20260909_124918_833635/independent_active_audit_archive/final_review.json)；完整逐项数据见 [独立审计 JSON](../reports/weekly_t_cross_validation/run_20260909_124918_833635/independent_execution_audit.json)。

## 怎样读取输出

| 产物 / 字段 | 含义 |
| --- | --- |
| `final_report.md` / `输出文件说明.md` | 实际收益、执行次数、库存及周退出风险；文件用途 |
| `annual_summary.csv` / `test_metrics.csv` | 每个独立年度账户的基准与主动收益、净增量和失败 |
| `t_enabled` / `t_has_actual_trades` | 分别表示开关开启、是否实际执行首腿；开启不保证当期一定成交 |
| `prediction_count` / `signal_count` / `request_count` | 入口记录保存预测、逐行触发及每日最早一次去重请求；报告同时核对已保存预测/请求文件行数 |
| `t_first_leg_count` / `t_recovery_leg_count` / `t_pair_count` / `t_fragment_count` | T 首腿、恢复腿、完整配对和恢复片段；不混入周底仓成交 |
| `t_fee` / `t_slippage_cost` | 全部实际 T 成交腿的费用与滑点归因；含尚未完成配对已产生的费用 |
| `realized_pair_*` | 只按已恢复片段分摊的参考毛差、滑点、费用和净利，不能替代含未恢复仓位估值的账户净增量 |
| `daily_inventory_audit.csv` / `blocked_orders.csv` / `base_plan_mismatches.csv` | 每日库存差异、恢复失败、受阻原因及共同底仓差异，均带年度 |
| `test_trade_events.csv` / `test_round_trips.csv` / `test_open_positions.csv` | 原始成交、恢复片段和期末真实仓位；合并表带 `test_year` |
| `test_return_curve.png` / `测试集收益曲线.png` / `t_increment_curve.png` | 每年的实际账户与基准曲线，以及 T 净增量；不能将四年拼成连续资金曲线 |
| `goal_evaluation.json` / `active_reporting_status.json` | `report_only=true`、`affects_activation=false`；参考统计与执行状态不回流至开关 |
| `validation_rule_metrics.csv` | 原 27 个内部候选的完整统计、旧参考资格、新执行资格及本轮选中标记 |
| `year_*/selected_rule.json` / `t_model/` / `selected_support.json` | 主动规则、所选核心模型副本与同周期支持域 |
| `all_rules_frozen_before_outer_test.json` | 所有评价折启用规则及 SHA 在新外层评价前统一冻结 |
| `prediction_causality_audit.json` / `test_weekly_context_manifest.json` / `dense_recovery_reference.json` | 原生和 bundle 预测一致、清除未来字段后预测/请求不变、周上下文及恢复报价身份 |
| `source_hashes.json` / `source_snapshot/` | 原 20 份核心/公共接口及本轮入口、启用、报告共 23 份源码身份 |

`event_type='t'` 同时用于首腿和恢复腿，`fragment_id=0` 是首腿，`fragment_id>0` 是恢复。完整配对按 `pair_id` 计数；多次恢复不能当作多组独立交易。净利已使用含滑点执行价再减费用，不能再次扣除 `slippage_cost`。

统计参考包括各年实际净增量、正增量年份数及周块 95% 区间。兼容字段 `gain_criterion_met` 保留旧统计与执行条件的汇总含义，但为真或为假都不改变本轮规则。更直接的状态是 `profitability_reference_met`、`t_execution_contract_met`、`weekly_execution_complete`。恢复失败日、库存差异日和其他失败计数可能来自同一事件，不能相加当作独立事故数。

正式运行结束后，报告模块已修复陈旧估值时间边界：`final_stale_position_count` / `final_stale_market_value` 按持仓 `mark_datetime` 与末次权益估值时点精确比较，同日仅有开盘价而缺收盘价也会标记。[4 项回归测试](../scripts/test_weekly_t_active_reporting.py)覆盖同日旧价、跨日旧价、末时点现价及零实际股数。

本次只从保存的 CSV 在独立目录重新渲染，没有重跑账户或模型。实际正式持仓没有新增同日旧价案例，四年陈旧市值仍为 0/0/0/9,750 元；全部年度非陈旧字段逐值精确一致，9 类派生报告也逐字节相同，未替换任何正式输出。原运行结束时 23 份源码一致；报告模块只在结束后由 `1a4e8c80…bb1d` 修订为 `3b841c74…c1a7`，当前为 22/23 整文件相同，原 23 份 `source_snapshot` 保留原身份。旧报告副本、源码、测试、前后 SHA 和 186 项原账本/输入不变证明见 [报告修订记录](../reports/weekly_t_cross_validation/run_20260909_124918_833635/report_revision/README.md)与 [manifest.json](../reports/weekly_t_cross_validation/run_20260909_124918_833635/report_revision/manifest.json)。

## 研究限制

v8 全关闭 T 说明旧晋级逻辑未启用交易；[2026 开关 A/B 诊断](../reports/weekly_t_zero_increment_diagnosis/gate_toggle_2026/final_report.md)已证明只打开原开关会产生实际交易和非零损益，但那只是一个固定模型的故障诊断，不能替代本轮四折结果。v9 会按训练内部执行资格选择对应模型，不根据已见外层收益切换候选。

本轮是在用户查看旧结果后调整启用政策，对同一历史重新评价，不能称为全新未接触测试。较早三折使用未来年份训练，2026 仅到 4 月 30 日。零共同项先验不代表已对冲市场，周统计区间没有完全处理周间相关及历次研究选择偏差。历史 ST、证券主表、精确限价/盘口、复权与公司行动信息仍不完备。

移除统计门槛只让主动策略进入账户检验，不保证盈利，也不能保证每日日终恢复成功。恢复或周退出失败必须保留真实仓位、费用、估值和失败记录；通过代码检查不能写成用户的每日股数和盈利目标已经达成。
