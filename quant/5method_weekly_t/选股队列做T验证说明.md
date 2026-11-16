# 同折 OOF 选股队列上的做 T 验证（v6）

v6 研究一个具体变化：让 T 模型在本折核心年份中由周模型选出的前 100 只股票、周中交易日信号上训练，并加入当周已经生成的周预测排名。费用、27 个候选、库存比例、恢复时间和支持门槛全部延续 [v5 短周期协议](短周期做T验证说明.md)。内外层实际选股名单仍精确复用对应折的 v4 文件。

架构是在原短周期数据、模型和账户执行之间增加 OOF 队列层。OOF 指目标年份的周选股预测来自没有拟合该年标签的模型；这里每个核心年由另一个核心年训练的周模型生成队列。队列层同时传递周特征的四周原始行情来源和模型参数训练身份，随后由原 T 模型执行严格年度过滤。当前正式运行目录为 [`run_20260908_225457_819320`](../reports/weekly_t_cross_validation/run_20260908_225457_819320/)，16 份运行源码与启动快照一致，四折最终均关闭 T，额外净收益为 0。

## 文件角色

| 文件或目录 | 类型 | 用途 |
| --- | --- | --- |
| 本文件 | 方法文档 | v6 队列、来源、配置、验证与复现说明 |
| [`config/weekly_t_cross_validation_v6.yaml`](../config/weekly_t_cross_validation_v6.yaml) | 用户输入 | 选择本折核心年 OOF 队列；其余经济参数与 v5 相同 |
| [`scripts/run_weekly_t_cohort.py`](../scripts/run_weekly_t_cohort.py) | 新入口 | 生成队列、接入本折冻结周预测、训练内部选择、全部冻结后评价外层并生成报告 |
| [`quant_backtest/weekly_t_cohort.py`](../quant_backtest/weekly_t_cohort.py) | 新辅助库 | 核心年互为 OOF 的周选择、四个预测特征、四周来源传播和模型溯源 |
| [`scripts/test_weekly_t_cohort.py`](../scripts/test_weekly_t_cohort.py) | 新验证脚本 | 7 项队列、来源、缺失标签、索引和非法输入验证 |
| [`scripts/test_weekly_t_cohort_workflow.py`](../scripts/test_weekly_t_cohort_workflow.py) | 新流程验证 | 5 项本折模型身份、上下文、拟合边界和先冻结后评价验证 |
| [`scripts/audit_weekly_t_cohort.py`](../scripts/audit_weekly_t_cohort.py) | 新只读审计入口 | 独立重建 OOF 周预测、T 训练行、上下文及模型 SHA；不读取外层收益 |
| [`scripts/audit_weekly_t_cohort_execution.py`](../scripts/audit_weekly_t_cohort_execution.py) | 新只读执行审计入口 | 在v5重放逻辑上核对v6来源身份；完整运行结束后，逐笔核对实际价、费用、现金、T+1、恢复、容量和共同底仓 |
| `year_Y/oof_weekly_cohort/` | 生成输出 | 本折两个周选择模型、目标年预测/选择、队列来源和配置 |
| `year_Y/*weekly_context_manifest.json` | 生成审计记录 | 内部/外层对应冻结周模型身份，以及四个新特征和扩展来源的输入 SHA |
| 原周度、短周期、恢复、模型和账户模块 | 共用辅助库 | 继续使用 v5 的 14 份运行依赖；v6 另加入口与队列模块，共 16 份源码快照 |

队列库的两个接口为：

```python
training, metadata = build_core_t_cohort(
    weekly_universe, intraday, core_years, excluded_years, cfg, output_dir
)
enriched = attach_weekly_context(intraday, weekly_predictions, weekly_universe, cfg)
```

第一个接口负责只保留 OOF 入选股周的中间交易日，返回尚未按 T 标签清除的队列。第二个接口严格保留调用者给定的分钟行、索引顺序和未来标签，只附加唯一对应周预测；缺预测、重复键、未知或未来来源、冲突模型身份都报错。已有上下文不能静默覆盖。

## 每折的年份与数据流程

| 外层留出年 | 核心年份 | 内部规则选择年 | 两个核心年 OOF 周选择模型 |
| --- | --- | --- | --- |
| 2023 | 2024、2025 | 2026 | 用 2025 训练后选 2024；用 2024 训练后选 2025 |
| 2024 | 2023、2025 | 2026 | 用 2025 训练后选 2023；用 2023 训练后选 2025 |
| 2025 | 2023、2024 | 2026 | 用 2024 训练后选 2023；用 2023 训练后选 2024 |
| 2026 | 2023、2024 | 2025 | 用 2024 训练后选 2023；用 2023 训练后选 2024 |

1. 读取原始数据的独立缓存，先增加会话、过去完整周执行特征及真实来源，再按历史价格、成交额、历史天数和收盘流动性选择周资格池。原缓存保持只读。
2. 对每个核心目标年，重新拟合本折自己的周双模型，训练集只有另一个核心年。排除集合为其余全部可用年份，包含目标年、内部年和外层年；周模型的原始量价来源至标签结束区间仍须通过前后 7 个自然日的严格隔离。不能借用其他外层折已训练的周模型。
3. 周模型在目标年的当前合格周上预测，按收益/振幅预测截面排名的 70% / 30% 组合选前 100。目标周只检查已知的特征周完整性、首末交易日在目标年且首日在末日之前；不因个股未来收益、振幅标签或恢复价缺失删除当前预测。队列只保留所选股票周首日和末日之间的分钟信号，核心周不受 `smoke_weeks` 的前三周限制。
4. 附加四个周预测特征，把分钟来源扩展到两类原始输入的最早时点。再接 20/40/60 分钟标签，用扩展后的来源对 T 训练行做严格过滤。支持域、特征筛选和 T 权重都只拟合两个核心年份的合法队列。
5. 在本折最新训练年份上，用对应 v4 内部选股和周预测评价 27 个候选。全部门槛通过后按内部净增量选择，否则关闭 T。选定的 T 模型直接保留核心权重，`outer_t_refit=false`。
6. 全部四折内部规则与模型身份冻结后，才统一评价外层年份。外层继续用本折 v4 已冻结选股和对应周预测；正式完整运行逐日核对共同持有基准与 v4 日期及权益一致。

“另外三年属于训练流程”包含两个核心拟合年和一个内部选择年，内部年没有被遗漏。T 权重只用核心两年拟合；内部年用于挑选规则。外层使用的 v4 周选股模型则按原协议用这三个训练年份拟合，包含内部年。两种模型承担不同任务，不能混用或把内部选参收益称为外层收益。

## 四个周预测特征与真实来源

| 新增模型输入 | 数值来源 | 含义 |
| --- | --- | --- |
| `f_weekly_pred_return_rank` | `pred_return_rank` | 当周周模型收益预测的全资格池截面百分位排名 |
| `f_weekly_pred_amplitude_rank` | `pred_amplitude_rank` | 当周周模型振幅预测的全资格池截面百分位排名 |
| `f_weekly_score` | `score` | 上述两个预测排名按配置 70% / 30% 组合 |
| `f_weekly_selection_rank` | `rank / top_n` | 周选择顺序相对前 100 的归一化值；默认第 1 名为 0.01，第 100 名为 1 |

这里使用的是模型预测排名，没有使用下一周真实收益或真实振幅排名。内部/外层冻结文件的百分位排名保留原 v4 全资格池口径，不能先缩到已选 100 只或烟测 8 只再重新排名。四列送入原 T 特征筛选后，是否全部进入最终模型由核心训练数据的覆盖率、相关性等规则决定，并非强制每列都有模型权重。

周预测读取上一周及此前四个实际观测周的量价，旧分钟来源只追踪前一周、前一日及同刻市场依赖，因此直接拼接四列会低估历史来源。v6 明确更新：

```text
feature_source_start = min(
    原分钟 feature_source_start,
    同 trade_week / symbol 的周特征 feature_source_start
)
```

如果某年 1 月信号的周输入实际来自上一年 12 月，而上一年属于排除年份，该 T 训练行必须被后续严格过滤剔除。不能仅检查信号和标签年份，也不能把 `trade_week` 当作周预测全部原始输入的最早来源。累计 `f_history_days` 仍只用于当前历史资格，不进入模型。

原始量价来源时点和模型参数训练年份分别记录。比如本折核心目标 2024 的周模型用 2025 的标签拟合，其输入价格可以全部早于当前 2024 信号，但模型参数不是当时已经训练好的参数。该样本是回顾性的 OOF 特征，不能描述为历史当时可以部署。模型不包含本目标年、内部年或外层年的拟合标签，是一种交叉拟合边界；它不会把未来年份训练变成向前验证。

## 冻结周模型上下文与溯源字段

固定 v4 来源为 [`run_20260908_212722_188543`](../reports/weekly_t_cross_validation/run_20260908_212722_188543/final_report.md)。外层折 `Y` 的内部上下文只能来自 `year_Y/validation_weekly_selections.csv` 与 `year_Y/inner_weekly_model/`；外层只能来自 `year_Y/weekly_selections.csv` 与 `year_Y/weekly_model/`。入口核对实际模型元数据中的训练年和排除年，并计算 bundle SHA；预测表若已携带不同训练年或 SHA，就拒绝接入。

| 非模型字段或审计文件 | 内容与用途 |
| --- | --- |
| `source_model_training_years` | 稳定 JSON 字符串，如 `[2025]`；标记生成这一周上下文的参数拟合年 |
| `source_model_sha256` | 对应 `model_bundle.joblib` 的 SHA256 |
| `weekly_context_signal_datetime` | 该周预测所使用的已完成周信号时间 |
| `intraday_feature_source_start` | 加周预测之前的分钟原始来源 |
| `weekly_context_feature_source_start` | 同股票周预测原始量价的最早来源 |
| `cohort_target_year` | 核心队列的 OOF 目标年；不进入 `f_*` |
| `validation_weekly_context_manifest.json` | 本折内部冻结周模型路径、bundle/metadata SHA、训练与排除年、行数、四列名称及上下文输入 SHA |
| `test_weekly_context_manifest.json` | 相同结构，记录本折外层冻结周上下文 |

两份上下文输入 SHA 包含 `datetime`、`symbol`、扩展后的 `feature_source_start`、四个周预测特征和两个模型身份字段。它们记录实际送入 T 推断的上下文，不能用检查原模型文件存在来代替。

直接调用辅助库时，可在预测表给出两个模型身份字段，或者通过调用配置的 `weekly_context_model_training_years`、`weekly_context_model_sha256` 填补缺列。正式入口只从本折对应的实际冻结模型计算这两个值，不让用户手工猜测。它们是运行中注入的身份信息，不是新增研究超参数。

## 配置与 v5 的对照

本轮 YAML 与 v5 相比只有下列两个变化，其余键和值保持一致：

| 字段 | 必需/默认 | 消费者及作用 |
| --- | --- | --- |
| `t_training_cohort` | 新增必需；`core_year_oof_weekly_top100_middle_days` | v6 入口；只接受同折核心年 OOF 周 top100 周中队列 |
| `research_version` | 必需；`weekly_t_v6_oof_cohort` | 入口及报告；标记新的研究版本 |

候选仍为 20/40/60 交易分钟 × D4/D6/Huber 三模型 × 10/20/30 bp 费用后边际，共 27 个；方向双向，首腿比例 25%，每股每天最多一个请求。计划恢复不晚于 14:30，使用固定 130 个盘中时钟加尾盘六分钟，未恢复参考名义额上限为初始资金的 10%。

成本仍为每腿佣金 0.08%、过户 0.001%、滑点 0.05%，卖出印花税按日期 0.1%/0.05%，最低佣金 0；100 股整手、分钟成交额 5% 容量及原方向性限价代理保留。精确双腿净边际公式、跨日恢复和不重复扣滑点的口径见 [v5 说明](短周期做T验证说明.md#双腿费用与净边际)。

内部仍要求至少 100 个真实完整配对、20 活跃日、8 活跃周、20 只股票，限制日/周组合数量及净利润集中度，实际净增量和周块 95% 区间下界均为正，并且 T 股数恢复、T+1、期末未恢复组合和共同底仓差异均合格。由于队列变了，核心支持域和模型数值会重新拟合，但支持域的列、0.1%/99.9% 分位规则和硬门槛没有放宽。无合格候选就保持底仓，不把关闭 T 解释成盈利。

## 运行、输出与审计

从 quant 仓库根目录运行。新增 12 项验证使用合成数据和临时模型，不改正式数据或报告：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_cohort.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_cohort_workflow.py
```

四折小样本命令如下。它保留 8 只数据股票和内部/外层冻结名单的前 3 个可用周，把 T 模型缩小至 20 棵树、最多训练 30,000 行、候选进程改为 2；周选择模型仍用原 300 棵树参数，核心队列不截成前三周，样本支持门槛不降低。

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_cohort.py \
  --config config/weekly_t_cross_validation_v6.yaml \
  --max-symbols 8 --smoke-weeks 3 --workers 4 \
  --output-dir reports/weekly_t_cross_validation_smoke
```

正式运行命令如下，每次新建带时间的目录：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_cohort.py \
  --config config/weekly_t_cross_validation_v6.yaml
```

入口默认使用 v6 配置，也支持 `--test-years 2026` 限定留出折、`--workers` 调整资源和 `--output-dir` 更换输出根目录。受限小样本或单折结果不代替完整四折结论。

| 新增输出 | 内容 |
| --- | --- |
| `year_Y/oof_weekly_cohort/selector_目标年/` | 原生周收益/振幅模型、bundle、模型元数据、特征筛选、训练来源审计，以及 `oof_weekly_predictions.csv` / `oof_weekly_selections.csv` |
| `year_Y/oof_weekly_cohort/cohort_lineage.csv` | 每条队列信号、四个新特征、扩展来源、模型训练年/SHA及 OOF 目标年 |
| `year_Y/oof_weekly_cohort/cohort_metadata.json` | 本折核心/排除年、两个选择模型、队列行数、来源键及上下文 SHA、配置和模块身份 |
| `year_Y/t_training_cohort_manifest.json` | 入口持久保存的同一队列元数据，后续 T 拟合及审计使用 |
| `year_Y/validation_weekly_context_manifest.json` / `test_weekly_context_manifest.json` | 实际内部/外层冻结周上下文及参数身份 |
| `independent_cohort_audit.json` / `.md` | 独立来源与模型审计结果；不等于执行成功或获得收益 |

其余 `validation_rule_metrics.csv`、`selected_rule.json`、`all_rules_frozen_before_outer_test.json`、`inner_t_h*`、`t_model/`、标准成交/库存/收益文件与 v5 相同。主报告为 `final_report.md`，最终盈利目标由 `goal_evaluation.json` 判断，审计与收益目标分开。

模型与队列审计命令可先针对已完成的折执行，不读取外层收益。例如验证本次小样本：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_cohort.py \
  --report-dir reports/weekly_t_cross_validation_smoke/run_20260908_225020_968607 \
  --smoke
```

正式运行将下面的 `report_dir` 改成入口实际打印的运行目录；首折模型齐全时可给队列审计加 `--test-years 2023`，完整模型审计则不限制年份。执行审计必须等 `final_report.md` 和所有年度结果完成后运行，它会读取外层成交和收益。

```bash
report_dir='reports/weekly_t_cross_validation/run_YYYYMMDD_HHMMSS_ffffff'
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_cohort.py \
  --report-dir "$report_dir"
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_cohort_execution.py \
  --report-dir "$report_dir"
```

审计分别核对 OOF 周模型的真实训练样本/来源 SHA、目标年预测排名和选择、四周来源传播、T 样本及参数、所选核心权重的精确复制、冻结上下文身份，以及真实执行中的现金、整手、旧库存、容量、费用和共同基准。审计只新增自己的文件，不能重新拟合、选参或修改原结果。

## 已完成验证与结果边界

模块 7 项及流程 5 项测试均通过，覆盖核心年互为 OOF、其余年份全部排除、缺失未来标签保留、周中选择、输入不变、四周来源导致后续跨年 T 样本被清除、预测/模型身份冲突拒绝，以及全部规则冻结后才评价外层、外层 T 不重新拟合。

真实 8 股验证使用原只读缓存和 300 棵树周模型。目标 2024 的选择模型仅在 2025 的 260 条合法周样本上拟合，目标 2025 的模型仅在 2024 的 238 条合法周样本上拟合，合计生成 12,544 条周中 T 队列信号，涉及 7 只股票；四个新特征及最早来源传播均通过检查。证据已复制到正式报告 `engineering_validation/real_cohort_smoke.json`，模型仅写临时目录；这不是收益回测结论。

四折集成小样本已完成于 [`run_20260908_225020_968607`](../reports/weekly_t_cross_validation_smoke/run_20260908_225020_968607/final_report.md)。各年 T 均关闭，净增量为 0，[目标判断](../reports/weekly_t_cross_validation_smoke/run_20260908_225020_968607/goal_evaluation.json)未达成；该小样本只验证流程和身份传递，不能证明真实 T 成交和盈利。真实提前恢复的执行能力另有 [固定真实行情执行诊断](../reports/weekly_t_cross_validation_smoke/real_t_v5_early_execution/README.md)，其固定方向和样本不参与本轮选参。

这次小样本的[独立模型与数据审计](../reports/weekly_t_cross_validation_smoke/run_20260908_225020_968607/independent_cohort_audit.md)已通过：8 个 OOF 周选择器、36 个 T 模型、12 个支持域、8 份冻结周上下文和 4 份外层核心权重复制均与重建证据一致，原生树叶权重及保存预测也逐项核对。另有 7 项反例测试覆盖错误年份、权重和输入篡改、缺失/未来来源及标签缺失推断。审计没有读取外层收益，只证明此次小样本接入和隔离正确；正式全股票池仍须独立审计。

已完成的 [v5 正式报告](../reports/weekly_t_cross_validation/run_20260908_221301_842709/final_report.md)四折均关闭 T、增量为 0。v5 的这一结果不表示 v6 能盈利；v6 队列假设来自此前的[训练内分布诊断](../reports/weekly_t_cross_validation/run_20260908_221301_842709/inner_diagnostic_v5.md)，本轮正式评价也未产生合格 T 规则，不能据此宣称盈利。

v4、v5 和更早版本的同一历史已经查看，v6 属于继续研究。严格隔离和 OOF 不会恢复一个从未看过的留出集：2023–2025 外层折仍含未来年份训练，核心 OOF 中给较早年份生成预测也会用到较晚年份参数；2026 的训练年份较早，但测试数据只到 4 月 30 日且这段历史已被反复研究。原始行情缺少完整历史 ST、精确限价/盘口、公司行动及完备历史证券主表的限制继续存在。

最终判断仍须同时查看费用后净增量、不确定区间、每日股数恢复、周末退出和期末遗留持仓。关闭 T、通过源码或模型审计、生成更匹配的训练队列，都不代表盈利目标已经达成。

## 正式四折结果

[正式主报告](../reports/weekly_t_cross_validation/run_20260908_225457_819320/final_report.md)已生成，程序正常结束。各年从独立100万元开始，2026只到4月30日；T净增量合计0，周增量95%自助区间为[0, 0]，正增益年0/4。

| 留出年 | 周度持有净收益 | 叠加 T 净收益 | T 净增量（元） |
| --- | ---: | ---: | ---: |
| 2023 | -11.665581% | -11.665581% | 0.00 |
| 2024 | -16.499154% | -16.499154% | 0.00 |
| 2025 | 31.469682% | 31.469682% | 0.00 |
| 2026 | 8.227703% | 8.227703% | 0.00 |

108个内部配置中81个负增量、5个零增量、22个正增量，全部未通过完整门槛。最高内部净增量474.531905元，只有87个完整配对且集中在2日/2周，周置信下界不为正；没有将内部偶然盈利提升为正式策略。四折外层T组数均为0，持有基准与v4逐日完全一致。原周末退出受阻仍为57/37/55/22个账户日，不能把关闭T解释为所有周末都已平仓。

[完整模型与队列独立审计](../reports/weekly_t_cross_validation/run_20260908_225457_819320/independent_cohort_audit.md)已通过8个OOF周模型、36个T模型、12个支持域、8个冻结上下文和4份最终权重复制；16份运行源未变。[完整独立执行审计](../reports/weekly_t_cross_validation/run_20260908_225457_819320/independent_execution_audit.md)也已通过：7039份原CSV、63686笔实际/基准成交及76084行库存核对一致，最大现金/权益误差约1.397e-9元；模型及记账审计本身不证明盈利。

[首折训练内部诊断](../reports/weekly_t_cross_validation/run_20260908_225457_819320/inner_diagnostic_v6.md)仅使用核心年与内部年。4个新增特征都实际入模，训练来源哈希核对一致。按初始固定预算机械估计，核心训练行82.269%可形成至少100股T，内部实际开仓映射为67.060%；这仍不是容量/现金/未来恢复的保证。40分钟训练中12.5%的行计划14:41恢复，60分钟训练中14.286%的行计划14:31恢复，均超出原14:30截止。这些是继续匹配训练条件的依据，没有修改本轮模型、门槛或外层结果。
