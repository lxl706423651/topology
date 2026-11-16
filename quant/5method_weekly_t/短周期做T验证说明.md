# 冻结周度选股上的短周期做 T 验证（v5）

v5 保留 v4 各年度已经冻结的周度选股和底仓订单，只研究把做 T 的计划恢复时间缩短为 20、40、60 个交易分钟。每日先买或先卖，另一腿恢复原股数；周度建仓、退出仍按共同底仓计划执行。本说明记录实验协议、新增配置、执行方式和复现命令，正式四折已完成，全部规则关闭T，净增量均为0，未达到明显收益目标。既有 v4 结果继续见 [README.md](README.md)。

## 架构与文件角色

流程由四层组成：原有因果特征与严格历史来源隔离；独立的短周期未来标签和真实恢复报价；仅用训练数据拟合的分钟模型、支持域及候选规则；维护 T+1、现金、分钟容量和共同底仓的账户引擎。恢复报价按实际年度建立只读 mmap，候选子进程共享文件，不复制巨大的逐行行情字典。

| 文件或目录 | 类型 | 输入、输出和用途 |
| --- | --- | --- |
| 本文件 | 方法文档 | 说明 v5 新增规则；不替代 v4 已完成结果 |
| [`config/weekly_t_cross_validation_v5.yaml`](../config/weekly_t_cross_validation_v5.yaml) | 用户输入 | 数据路径、冻结选股来源、27 个候选、支持门槛、费用及资源配置 |
| [`scripts/run_weekly_t_short_horizon.py`](../scripts/run_weekly_t_short_horizon.py) | 可执行入口 | 先完成全部折的内部选择并冻结，再统一评价外层；生成独立报告目录 |
| [`quant_backtest/weekly_t_horizon_data.py`](../quant_backtest/weekly_t_horizon_data.py) | 新增数据辅助库 | 从只读分钟数据生成短周期标签和事前固定恢复时钟的 OHLCVA；缓存不含 `f_*` |
| [`quant_backtest/weekly_t_short_horizon.py`](../quant_backtest/weekly_t_short_horizon.py) | 新增策略及选择辅助库 | 标签连接、训练支持域、精确双腿净边际、最早有效请求、样本与集中度门槛 |
| [`quant_backtest/weekly_t_quote_store.py`](../quant_backtest/weekly_t_quote_store.py) | 新增执行存储辅助库 | `build_dense_recovery_store(bars, cache_dir)` 返回继承引擎接口的 `DenseRecoveryBarStore`；`get_quote(datetime, symbol)` 仅读指定当前行 |
| [`quant_backtest/weekly_t_engine.py`](../quant_backtest/weekly_t_engine.py) | 共用账户引擎 | 新增较早计划恢复、同刻先恢复再开新 T、共享容量和未恢复名义额上限；保留原周底仓逻辑 |
| [`quant_backtest/weekly_t_models.py`](../quant_backtest/weekly_t_models.py) | 共用模型辅助库 | 只读 `f_*`、训练筛选、分钟模型拟合和推断；v5 外层直接复用核心训练权重 |
| [`quant_backtest/weekly_t_validation.py`](../quant_backtest/weekly_t_validation.py) | 共用候选评价库 | 6 个 spawn 进程完整模拟候选，返回账户指标和支持统计，保存失败审计 |
| [`weekly_t_data.py`](../quant_backtest/weekly_t_data.py)、[`weekly_t_features.py`](../quant_backtest/weekly_t_features.py)、[`weekly_t_provenance.py`](../quant_backtest/weekly_t_provenance.py) | 共用数据与特征辅助库 | 原周度/盘中/日行情、当前及过去周特征、逐行真实历史来源；原缓存保持只读 |
| [`weekly_t_recovery_data.py`](../quant_backtest/weekly_t_recovery_data.py) | 共用尾盘执行数据辅助库 | 提供 14:55 至 15:00 六个真实分钟，与新增盘中恢复报价合并 |
| [`test_weekly_t_horizon_data.py`](../scripts/test_weekly_t_horizon_data.py) | 验证脚本 | 交易分钟、午休、缺失标签、全年报价覆盖、冻结文件与缓存 |
| [`test_weekly_t_short_horizon.py`](../scripts/test_weekly_t_short_horizon.py) | 验证脚本 | 清除未来列后决策不变、成本、支持域、候选硬门槛 |
| [`test_weekly_t_scheduled.py`](../scripts/test_weekly_t_scheduled.py) | 验证脚本 | 较早分批恢复、现金释放、同分钟容量、时序与未恢复金额上限 |
| [`test_weekly_t_quote_store.py`](../scripts/test_weekly_t_quote_store.py) | 验证脚本 | 真实 spawn、只读 mmap、双精度、零量/缺行、缓存身份及引擎结果一致 |
| [`test_weekly_t_short_workflow.py`](../scripts/test_weekly_t_short_workflow.py) | 验证脚本 | 同折选股、核心年份隔离、27 候选去重、模型周期及不重拟合、先全冻结后评价、基准偏差拒绝 |
| `horizon_cache/horizon_指纹/` | 生成缓存 | 分股结果、标签、恢复报价、数据质量和元数据；与旧缓存分开 |
| `dense_quote_cache/年份/dense_recovery_指纹/` | 生成执行缓存 | `quotes.npy` 为 `(时间, 股票, 6)` 的 float64 mmap；`present.npy` 标记真实行，`times.npy` 和 `metadata.json` 保存索引及身份 |
| `reports/weekly_t_cross_validation/run_时间/` | 生成报告 | 实际配置、源码快照、冻结规则、全部候选及外层成交/收益；每次新建 |

## 年度拆分与选股冻结

周度选股继续使用上一周及更早行情，预测下一周收益和日均振幅，按预测截面排名的 70% / 30% 组合选前 100。v5 不重新训练周模型，不重新排名，不根据新的 T 收益改选股票。原始文件共覆盖 3,302 只主板股票、804 个市场交易日，数据到 `2026-04-30`。

固定来源为 [`run_20260908_212722_188543`](../reports/weekly_t_cross_validation/run_20260908_212722_188543/final_report.md)。对外层年份 `Y`，内部模拟只读同一目录 `year_Y/validation_weekly_selections.csv`，外层模拟只读 `year_Y/weekly_selections.csv`。四折共 8 个选股文件分别记录 SHA；不能拿其他折的选股文件代替本折内部股池。报价缓存可预载这些文件对应的年度股票并集，但策略每周仍只能交易本折当周入选股，缓存存在一行报价不表示该股入选。

| 外层留出年 | T 模型与支持域拟合年份 | 内部规则选择年 | 外层对应的 v4 目录 |
| --- | --- | --- | --- |
| 2023 | 2024、2025 | 2026 | `year_2023/` |
| 2024 | 2023、2025 | 2026 | `year_2024/` |
| 2025 | 2023、2024 | 2026 | `year_2025/` |
| 2026 | 2023、2024 | 2025 | `year_2026/` |

T 拟合使用核心两年中满足过去价格、历史天数、日均成交额和过去完整周收盘流动性资格的股票，不能用某个已拟合周模型的未来选股结果反选训练样本。特征筛选、模型权重和支持域均只拟合核心年份；内部年只用其对应的冻结周股池做规则选择。内部选择完毕直接复制核心模型到 `t_model/`，不再用内部年重拟合，避免改变已经选定阈值对应的预测刻度。

沿用 v4 严格来源协议：核心训练每行从 `feature_source_start` 到 `label_end` 的完整区间，不能与内部年或外层年及其前后 7 个自然日相交。来源追踪包含前 4 个实际观测周、前一实际周/日及同刻市场收益使用的前收盘来源。累计 `f_history_days` 不进入模型，只用于事前历史资格；缺失来源或未来来源不能进入训练。真实来源审计不等于全市场证券主表、历史 ST 或公司行动信息完备。

本轮先完成全部四折的内部模拟，写出每折 `selected_rule.json`、所选模型 SHA 和支持域，再写 `all_rules_frozen_before_outer_test.json`；之后才开始任何本轮外层组合评价。若内部没有合格候选，该折关闭 T，保留共同底仓并记录全部拒绝原因。

## 27 个候选与当前信息限制

候选为 `3 个周期 × 3 个模型 × 3 个费用后边际 = 27`：周期 20/40/60 个交易分钟；模型 `l2_depth4`、`l2_depth6`、`huber_depth4`；净边际 10/20/30 bp。Huber 参数固定为 `0.01`。每次请求比例固定为开盘库存的 25%，方向同时允许先买和先卖，不再搜索方向或库存比例。所有候选都做完整账户评价；完全相同的决策 SHA 只模拟一次，但 27 行指标及复用指针仍全部保存。

标签定义为 `计划恢复分钟 close / 首腿执行分钟 close - 1`，正值表示上涨。首腿在已完成信号后的下一交易分钟收盘模拟成交，再向后数固定交易分钟，跳过午休、不得跨日。例如 `09:35` 信号在 `09:36` 执行，20 分钟目标为 `09:56`；`11:00` 信号的 60 分钟目标为 `13:31`。`14:00` 信号的 60 分钟目标超出会话，按已知时钟拒绝首腿。

当前输入还必须位于核心训练数据拟合的支持域：`f_market_day_return`、`f_day_return`、`f_open_return`、`f_overnight_gap`、`f_vwap_deviation` 每列均处于训练的 0.1% 至 99.9% 分位区间，且数值有限。边界按周期保存，内部或外层都不重估。这个过滤能减少明显域外场景，但单列区间不能证明多维联合分布一致，也不把预测收益当作可靠概率。

信号还经过只看当前已完成分钟的恢复方向风险过滤。例如预计先卖后买时，当前买回方向已明显受限，就不发起首腿。所有首腿条件均不读取下一分钟价格、未来恢复量价、未来标签是否存在或是否实际恢复成功。未来标签缺失的当前行仍可推断；只在训练时排除不可监督的标签。每股每日在全部当前门控通过后保留最早的一次请求，该请求被实际执行约束阻挡后不会改选当日更晚的请求。

## 双腿费用与净边际

令 `r` 为模型预测的参考价格回报，`c = 佣金率 + 过户费率`，`s` 为单腿滑点率，`d` 为卖出日印花税率。按每 1 元首腿参考名义额计算：

```text
A = (1 + s) × (1 + c)
B = (1 - s) × (1 - c - d)
先买后卖的预期净边际 = (1 + r) × B - A
先卖后买的预期净边际 = B - (1 + r) × A
```

取两方向中较大的净边际，达到候选阈值才请求该方向。这里的 10/20/30 bp 是扣除模型化费用和滑点后的门槛，不是原始预测涨跌幅。默认每腿佣金 0.08%、过户费 0.001%、滑点 0.05%；卖出印花税在 `2023-08-28` 前为 0.1%，当天起为 0.05%。零最低佣金下，上式对应双腿现金流的精确比例；当前辅助库对 `min_commission != 0` 明确拒绝，不能用比例公式低估小单费用。

以 `2023-08-28` 起的税率为例，预测价格不变时双腿成本为首腿参考名义额的约 31.1975 bp，对应门槛为：

| 费用后净边际 | 先买后卖所需预测回报至少 | 先卖后买所需预测回报至多 |
| --- | ---: | ---: |
| 10 bp | +41.2722 bp | −41.1436 bp |
| 20 bp | +51.2903 bp | −51.1305 bp |
| 30 bp | +61.3084 bp | −61.1174 bp |

这只是首腿信号的预测门槛。实际成交仍逐腿按实际日期、价格、股数和费用记账；跨日受阻不能按原计划收益结算。报告 `net_pnl` 已使用含滑点的执行价，再扣手续费；`slippage_cost` 是归因字段，不能从 `net_pnl` 再扣一次。

## 恢复时间、容量与共同底仓

计划恢复必须不晚于 `14:30`，它是计划期限，不能保证一定在该时刻完成。首腿执行还受 `last_entry_time=14:30` 限制。每个组合按其计划时刻及随后 5 个交易分钟尝试部分恢复；同时，入口根据固定 8 个信号时刻与 3 个周期，事前生成 130 个去重盘中恢复时钟，并始终保留 14:55 至 15:00 六个尾盘时钟。130 个时钟是默认配置的计算结果，不从真实成交成功时点反推；改变信号时刻或周期必须视为新的预声明实验。

只有已经到期的未恢复组合才使用这些时钟，其他组合新增的决策不会给它临时增加额外恢复机会。每个实际时刻先偿还到期组合，再按当前预测净边际由高到低、代码稳定排序发起新组合；同一分钟对同一股票共享容量，卖出共同扣减昨日可卖库存。旧组合部分恢复释放现金和未恢复金额额度后，同刻的新组合才可使用释放部分。

全部未恢复组合按 `剩余股数 × 首腿当时参考价格` 加总，不得超过初始资金的 10%，默认 100 万元账户为 10 万元。超过剩余额度时按 100 股整手缩量，不足一手则拒绝。这个上限不随未来价格更新，不是按当日净值计算，也不是最大亏损承诺。先卖所得另受 12% 回购价格缓冲的资金预留，避免用回购资金重复开仓。

先买后卖只能卖出开盘已有的昨日股份；当天新买的股份继续锁定。先卖后买扣减昨日可卖额度，回购不补回当天卖出额度。每股每日最多一个组合；建仓首日无昨日底仓，周末退出日保留清仓额度，不开新 T。100 股整手、每个实际分钟 5% 成交额容量、现金及同底仓约束继续有效。选出 100 只股票不表示每只都能成交：小账户、整手、分钟容量或行情受阻可能导致实际持股数少于 100。

缺失分钟不补价，真实零量/零额保留并由成交逻辑拒绝。涨跌停优先用真实限价；缺失时沿用方向性代理：同向达到 9.8% 拒绝，对平价线仅在价格精确等于前收盘 ±5% 且按分取整的候选限价时额外拒绝。正常 +6%/+7% 平价线不会仅因平价而被当作 5% 限价。代理规则不等于完整的历史 ST 和交易所限价数据。

仍未恢复的股份如实跨日保留，在后续真实开盘、已声明盘中时钟和尾盘重试；按实际年度剩余交易日继续估值及偿还。停牌或缺行情不能虚构成交，期末未完成组合、库存差异日、恢复失败日和周退出失败分别报告。底仓由独立纯持有账户确定，T 盈亏不放大以后的周度买单；正式外层逐日核对 v4 `baseline_equity` 和日期完全一致，金额容差为 `1e-6` 元。

## 训练内部的硬门槛与外层目标

内部候选必须同时满足以下条件，才参与净增量排名：

| 条件 | 默认门槛或定义 |
| --- | --- |
| 真实完整配对 | 至少 100 组；分批恢复片段不能重复算独立组 |
| 分散样本 | 至少 20 个实际首腿成交日、8 个成交周、20 只成交股票 |
| 组合数量集中度 | 单日发起组数占比不超过 25%，单周不超过 40% |
| 盈亏集中度 | 先按首腿日/周汇总实际配对净利润，再取绝对值；最大日占总绝对日利润不超过 35%，最大周不超过 50% |
| 净收益 | 同底仓账户净增量严格大于 0 |
| 不确定性 | 对账户逐周 T 净增量做 2,000 次周块自助重采样，95% 区间下界严格大于 0 |
| T 执行约束 | 库存差异日、恢复失败日、期末未恢复组合、共同底仓订单差异、T+1 违规均为 0；必要统计必须有限 |

全部通过后，按内部净增量最高选择；相同净增量时按事前候选顺序稳定取先。没有合格候选就关闭 T，不能降低门槛让小样本赢家通过。共同底仓自身的周末退出失败不作为内部 T 候选可修复的差异，但仍独立记录，并进入最终整体目标。

外层盈利目标单独判断：四个独立年度账户的 T 净增量合计为正、正增量年份超过半数、汇总周块 95% 区间下界为正、T 约束无失败，同时两账户周退出全部完成。只有全部满足，`goal_evaluation.json` 的 `gain_criterion_met` 才为真。关闭 T、通过代码测试或通过数据身份审计，都不能称为获得盈利增益。年度收益率不能相加或拼接成一个连续账户；金额合计以每年分别投入 100 万元为前提。

## v5 新增配置

以下字段均由 [`weekly_t_cross_validation_v5.yaml`](../config/weekly_t_cross_validation_v5.yaml) 提供；表中默认值构成当前协议。其余延续字段详见 [v4 配置说明](README.md#配置字段)，本轮费用不降低。入口保存完整 `resolved_config.json`。

| 字段 | 必需/默认 | 消费者与作用 |
| --- | --- | --- |
| `research_version` | 必需；`weekly_t_v5_short_horizon` | 入口；标记本轮协议版本，取代 v4 名称 |
| `scheduled_recovery_mode` | 必需；`fixed_horizon_clock_union` | 入口；只接受事前信号×周期导出的固定时钟 |
| `horizons` | 必需；`[20, 40, 60]` | 标签库/入口/策略；正整数交易分钟周期，定义 3 个训练目标 |
| `frozen_selection_report_dir` | 必需；`reports/weekly_t_cross_validation/run_20260908_212722_188543` | 标签库和入口；定位同折内外层冻结选股及基准，记录原文件 SHA |
| `horizon_cache_dir` | 必需；`reports/weekly_t_cross_validation/horizon_cache` | 标签库；独立短周期标签和 130 时钟报价缓存 |
| `dense_quote_cache_dir` | 必需；`reports/weekly_t_cross_validation/dense_quote_cache` | 入口/报价存储；按实际年合并新增报价与旧尾盘后建立 mmap |
| `short_t_net_margins` | 必需；`[0.001, 0.002, 0.003]` | 入口/策略；双腿费用及滑点后 10/20/30 bp 门槛 |
| `short_t_fraction` | 必需；`0.25` | 策略/引擎；首腿请求为开盘库存 25%，再受整手、卖出额度、现金和容量限制 |
| `latest_planned_recovery_time` | 必需；`14:30` | 策略；计划恢复超过此时即不发首腿，不读取未来行情 |
| `max_pending_t_notional_fraction` | 必需；`0.1` | 引擎；未恢复参考名义额相对初始资金上限，允许部分缩量 |
| `t_support_quantiles` | 必需；`[0.001, 0.999]` | 支持域拟合；只从合格核心训练行拟合各特征上下分位界 |
| `t_support_features` | 必需；五列 `f_market_day_return`、`f_day_return`、`f_open_return`、`f_overnight_gap`、`f_vwap_deviation` | 支持域/策略；当前行须全部有限且处于冻结边界内 |
| `outer_t_refit` | 必需；`false` | 入口；必须为 false，外层复用被选核心模型，不改变预测刻度 |
| `candidate_support_requirements.min_complete_pairs` | 必需；`100` | 候选选择；完整配对下限 |
| `candidate_support_requirements.min_active_days` | 必需；`20` | 候选选择；实际首腿成交日下限 |
| `candidate_support_requirements.min_active_weeks` | 必需；`8` | 候选选择；实际首腿成交周下限 |
| `candidate_support_requirements.min_active_symbols` | 必需；`20` | 候选选择；实际首腿成交股票下限 |
| `candidate_support_requirements.max_day_pair_fraction` | 必需；`0.25` | 候选选择；最大日发起组数占比 |
| `candidate_support_requirements.max_week_pair_fraction` | 必需；`0.4` | 候选选择；最大周发起组数占比 |
| `candidate_support_requirements.max_day_abs_pnl_fraction` | 必需；`0.35` | 候选选择；最大绝对日利润占比 |
| `candidate_support_requirements.max_week_abs_pnl_fraction` | 必需；`0.5` | 候选选择；最大绝对周利润占比 |
| `scheduled_recovery_times` | 自动生成；默认 130 个时钟 | 写入实际配置及协议；由入口计算，用户不据收益修改时钟列表 |
| `t_label_horizon_minutes` | 自动生成；当前候选周期 | 入口传模型；模型元数据记录标签周期 |

延续的 `t_model_profiles` 保持 3 种模型，`t_model` 保持 300 棵树、学习率 0.04、叶节点最少 500 样本、L2 正则 20；`t_max_train_rows=1500000`，筛选参数和随机种子沿用配置。`candidate_workers=6` 使用 spawn；`workers=12` 用于数据和模型资源。v5 不再消费 v4 的 `t_thresholds`、`t_directions`、`t_trade_fractions`、`candidate_shortlist_size`：对应选择由净边际、固定双向、固定 25% 和完整 27 组模拟取代。周模型参数仅保留原协议上下文，本入口不会重新拟合周选股模型。

## 运行命令与输出

从 quant 仓库根目录运行，使用项目 `quant` 环境。新增验证脚本仅读合成样本、写临时目录；原有来源、模型及引擎验证命令见 README。

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_horizon_data.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_short_horizon.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_scheduled.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_quote_store.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_short_workflow.py
```

已执行的四折小样本命令如下。它只保留前 8 只数据股票及冻结选股中前 3 个可用周，把 T 模型缩至 20 棵树、叶节点最少 20 样本，最多训练 30,000 行，候选进程改为 2；样本支持硬门槛不降低。

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_short_horizon.py \
  --config config/weekly_t_cross_validation_v5.yaml \
  --max-symbols 8 --smoke-weeks 3 --workers 4 \
  --output-dir reports/weekly_t_cross_validation_smoke
```

正式四折命令如下；入口不会覆盖已有运行目录。短周期全量缓存和 mmap 可复用，但仍须有足够内存和磁盘，初次构建不是小样本操作。

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_short_horizon.py \
  --config config/weekly_t_cross_validation_v5.yaml
```

CLI 还支持 `--test-years 2026` 限定留出折、`--workers` 调整资源、`--output-dir` 改独立输出根目录。正式四折结论使用完整配置；`--max-symbols` 或 `--smoke-weeks` 的结果不能代替它。指定更少折时，“全部规则冻结”仅指该次明确指定的折。

| 输出 | 后续用途 |
| --- | --- |
| `research_protocol.json`、`resolved_config.json`、`source_snapshot/`、`source_hashes.json` | 保存执行前规则、完整配置和 14 份实际依赖源码身份 |
| `data_manifest.json`、`recovery_manifest.json`、`horizon_manifest.json`、`feature_provenance_manifest.json`、`frozen_selection_sources.json` | 追踪原数据、各独立缓存、真实历史来源及 8 份冻结选股文件 |
| `year_Y/split.json`、根目录 `split_audit.csv` | 明确核心拟合、内部选择、外层年份和是否使用未来训练年份 |
| `year_Y/inner_t_h*/`、`support_h*.json`、`validation_predictions_h*.csv` | 9 个核心模型、原生模型/特征及训练来源审计、每周期支持域、内部预测 |
| 根目录及每折 `validation_rule_metrics.csv`、`validation_candidate_audits/` | 全 27 候选收益/样本/集中度/拒绝原因、重复决策指针、受阻订单及库存失败 |
| `year_Y/t_model/`、`selected_rule.json`、`selected_support.json`、根目录 `all_rules_frozen_before_outer_test.json` | 被选核心模型的精确副本、规则与支持边界，以及本轮先冻结后评价的记录 |
| `year_Y/weekly_selections.csv`、`t_predictions.csv`、`t_decisions.csv` | 对应本折的冻结外层选股、模型预测及真实发起请求 |
| 每折标准 `test_*`、`daily_inventory_audit.csv`、`blocked_orders.csv`、`weekly_entry_plans.csv`、`base_plan_mismatches.csv`、`execution_audit.json`、`frozen_baseline_audit.json` | 复核成交、完整配对和恢复片段、持仓/现金、T+1、股数恢复和固定底仓一致性 |
| 根目录 `annual_summary.csv`、`test_metrics.csv`、`test_return_curve.png`、`测试集收益曲线.png`、`goal_evaluation.json`、`final_report.md`、`输出文件说明.md` | 年度同底仓比较、配对净增量及不确定区间、最终目标判断和文件说明 |

## 已完成验证与研究边界

独立短周期缓存已完成构建，证据见 [全量数据验证](../reports/weekly_t_horizon_data_full_validation.json) 和 [缓存元数据](../reports/weekly_t_cross_validation/horizon_cache/horizon_22363501d63027ff/metadata.json)：20,454,854 个原有信号扩展为 61,364,562 行三周期标签，新增 217,139,520 条盘中恢复报价，130 个固定时钟。按 2023/2024/2025/2026 的冻结选股并集分别有 1,828/1,939/2,533/1,973 只股票，覆盖它们被选年度的全部真实日期；并非每周交易这么多只股票。

标签中首腿执行价缺失为 0；恢复价缺失共 2,556,859 行，其中 2,556,856 行是按已知会话时钟越界，另 3 行是实际分钟缺失。它们没有被回填，未来缺价不会反向删除当前信号。[4 股数据烟测](../reports/weekly_t_horizon_data_smoke_validation.json) 验证 77,184 行标签、240,760 条报价与 25,728 个旧信号键一致，重载一致且未修改旧缓存。

只读 mmap 的 18 项测试全部通过，包括真实 spawn 子进程、pickle 只传路径及身份、零量/NaN 与缺行区分、来源/索引/文件变化拒绝、并发原子发布，以及与原字典存储的实际成交和净值完全一致。另以 4 股 19,296 条真实尾盘行核对全部六列双精度值，4,946 条零量行保留；mmap 形状 `(4824, 4, 6)`，pickle 载荷 379 字节。此项临时验证记录为 `/tmp/weekly_t_dense_real_validation.json`，永久可重复验证入口是上述测试脚本。

本阶段 164 项相关持久验证通过，另有 4 项独立流程测试通过；测试脚本可能带入共享旧用例，执行项数不能当作互不重复的独立证据数量。四折集成小样本已完成于 [`run_20260908_221134_755324`](../reports/weekly_t_cross_validation_smoke/run_20260908_221134_755324/final_report.md)，14 份运行源码快照身份一致。小样本各候选没有实际 T 成交，因此四年均关闭 T，增量为 0，[目标判断](../reports/weekly_t_cross_validation_smoke/run_20260908_221134_755324/goal_evaluation.json)为未达成。这些是工程和数据验证，不是正式收益结论，也没有降低门槛来制造小样本收益。

此前 v2/v3/v4 外层历史收益已经查看，v5 的短周期、支持域和样本门槛设计也受先前研究审计启发。本轮虽然对每个模型严格隔离来源、只用训练内部选择，并在本轮任何外层评价前冻结全部规则，仍是对同一历史的后续研究，不能称为未接触过的新留出证据。2023–2025 的拆分还使用未来年份训练，属于回顾性交叉验证；2026 的训练年份早于测试，但数据仅到 4 月 30 日，且历史本身已反复查看。

周块自助区间只度量当前历史样本的不确定性，没有消除多候选选择、版本反复研究或市场状态变化的影响。原行情缺少完整历史 ST、精确限价和盘口、公司行动与完备历史可交易证券主表，基准开盘仍采用 09:31 分钟的 open 代理。无法据此保证每日必然恢复、周末必然退出或未来获利；正式完整结果、执行审计和目标判断须一起查看。


## 正式四折结果与独立审计

正式报告：[run_20260908_221301_842709/final_report.md](../reports/weekly_t_cross_validation/run_20260908_221301_842709/final_report.md)。每年独立100万元；当前费用和训练内硬门槛下，四折均关闭T，不能将零增量称为明显增益。

|留出年|持有净收益|加入T净收益|T净增量|完整T组数|
|---|---:|---:|---:|---:|
|2023|−11.67%|−11.67%|0元|0|
|2024|−16.50%|−16.50%|0元|0|
|2025|+31.47%|+31.47%|0元|0|
|2026截至4月30日|+8.23%|+8.23%|0元|0|

108个内部配置中，92个净亏、9个零收益，7个正增量配置最高99.05元，均不满足支持、周置信下界或库存门槛。[首折内部诊断](../reports/weekly_t_cross_validation/run_20260908_221301_842709/inner_diagnostic_v5.md)重建27请求并拆出毛价差、滑点和手续费；没有把计划端点代理收益当实际账户收益。

[完整模型审计](../reports/weekly_t_cross_validation/run_20260908_221301_842709/independent_short_horizon_audit.md)覆盖36模型、12支持域、4外层复制、14源码，训练/来源SHA一致且排除年份区间重叠为0，4608条原生预测对照误差0。[独立执行审计](../reports/weekly_t_cross_validation/run_20260908_221301_842709/independent_execution_audit.md)核对63686笔事件、76084库存行，原持有基准逐日完全相同。

底仓周退出仍受真实限制：171个账户日/307个股票日未在目标时点清完；2025年末3笔持仓14464元，2026年末4笔36325元，其中9750元按前一交易日真价估值。没有按未来是否可清仓删股票，也没有把未卖出的市值记为现金。真实提前恢复通路另见[固定4股执行诊断](../reports/weekly_t_cross_validation_smoke/real_t_v5_early_execution/README.md)，买先/卖先各12组按相同预设日期执行，不能据两方向诊断收益选择正式方向。

可重复审计命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/audit_weekly_t_short_horizon.py --report-dir reports/weekly_t_cross_validation/run_20260908_221301_842709
/home/lxl/anaconda3/envs/quant/bin/python scripts/audit_weekly_t_short_execution.py --report-dir reports/weekly_t_cross_validation/run_20260908_221301_842709
```

审计脚本分别为[模型及来源审计](../scripts/audit_weekly_t_short_horizon.py)和[独立执行审计](../scripts/audit_weekly_t_short_execution.py)，副本、SHA与日志均在报告目录。后续源码改变时应使用冻结快照或相应版本环境；不要把当前源与旧运行静默混用。本轮源码、模型和结果作为已完成研究保留，后续训练队列实验另建版本。
