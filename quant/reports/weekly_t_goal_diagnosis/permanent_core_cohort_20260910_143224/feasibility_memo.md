# 永久底仓训练队列：只读可行性审查

可以复用已审计的周 OOF 模型和名单，建立每股目标 1.5 万元、不清底仓的纯训练账户，再取全部持有期信号。但不能把旧 `restore_core_cohort` 简单改成取消周五过滤：入场字段、永久状态来源、周排名上下文和上游模型的年度依赖都必须明确。

本次只读取源码、模型元数据/身份、周名单和缓存的来源/日历/已知价格字段；没有拟合、运行账户、选择参数或读取 v13 收益。8 个 selector 的 bundle、16 个原生头、入选 CSV、训练样本/特征来源 SHA 与 v8 已完成独立审计一致。38,700 条入选股周全部能以 `(trade_week,symbol)` 唯一连接原周缓存，信号时间和 feature_week 完全一致；这不是重新执行完整模型审计。

## 原队列的限制

`quant_backtest/weekly_t_timing_data.py:99` 将分钟限制为 `entry_date < date < exit_date`；`weekly_t_cohort.py:252` 的原生成过程也是同一规则。两者还要求当前分钟属于当次入选股周。因此旧队列同时排除了入场日、周末退出日和以后离开周榜的全部持有日，适用于旧周度持仓，不符合现在的永久底仓状态。

可复用来源：

- 周名单：`reports/weekly_t_cross_validation/run_20260908_235818_404243/year_OUTER/oof_weekly_cohort/selector_CORE/oof_weekly_selections.csv`。
- 同目录 `model_bundle.joblib`、`model_metadata.json`、`return_model.txt`、`amplitude_model.txt` 和 `oof_weekly_predictions.csv`；后者包含该年每周全部合资格预测对象，适合追踪排名供体，不能先缩成持有股票再重排名。
- 年度合同：该外层目录的 `split.json`，以及 OOF 目录的 `cohort_metadata.json`；已完成审计为源报告根目录 `independent_residual_audit.json`。
- 原行情：`reports/weekly_t_cross_validation/cache/weekly_t_9b6b9bfb3309dd53/{weekly,daily,intraday}.pkl`，或同目录逐股 `SYMBOL.pkl`。`weekly.pkl` 有 541,043 行，其字节 SHA 本次与 v11 已审计输入清单一致。
- 逐股原分钟 CSV：`ashare_minute_data_2023_2026_04/extracted/YEAR/SYMBOL.csv`。新净价值标签若模拟动态/部分恢复，应为新的 OOF 实际持股集合建立完整分钟只读缓存，不能假定 v13 原账户持股缓存已经覆盖它们。

## 安全字段连接和账本

| 字段 | 来源及检查 |
|---|---|
| `trade_week,symbol,rank` | 原 OOF selected CSV 原样保留；键唯一，rank 正整数，不重新挑股票。 |
| `signal_datetime,feature_week` | CSV 与原 weekly 表同键逐值相等；信号必须早于实际 `entry_date 09:30`。 |
| `entry_date,exit_date` | 从完整原 weekly/universe 表一对一补齐；按全市场日历生成，不能寻找该股未来第一个可成交日。`exit_date` 只留溯源，不再触发卖出。 |
| `universe_previous_close` | 原同股同周 `f_last_close`，即信号周最后已观测 close；检查 `observed_last_datetime <= signal_datetime`，有限且正。保留原 float32 数值，不以入场日价格、未来 close 或当前报价倒填。 |
| `source_model_training_years,source_model_sha256` | 原 selected CSV 与本折 selector metadata、已审计 bundle SHA 三方一致。不能借用其它折的同名模型。 |
| `entry_feature_source_start` | 从完整逐股观测周历恢复真实前四观测周起点；不是自然日简单减 28 天，也不是只读截年后的历史。 |
| `entry_rank_source_start` | 同周完整 OOF 排名供体的最早真实特征来源，与自身来源取较早值。排名/score 会受其它股票影响。 |
| `base_entry_execution_time,shares,reference_price,fee` | 新无 T 账本的真实成交结果；同股仅首次成功自动入场，失败请求保留；现金/当前容量不足下限 1 万元则拒绝。 |
| `baseline_opening_shares,opening_sellable` | 每日真实账本快照；入场日新股为 0 可卖额度。不得用最终持仓股数倒推早期可卖量。 |

OOF CSV 缺 `entry_date` 和 `universe_previous_close` 是正常的旧输出合同，不需要重新训练来补。`scripts/run_weekly_t_cross_validation.py:142` 已将 `f_last_close` 定义为 `universe_previous_close`，可按同一数值合同接回。

每个 core 年使用独立初始现金和该年的 OOF 名单；账户不能跨过被排除年份持仓。按当前固定金额/费用/参与率配置，用新 `plan_dynamic_entries` 生成并冻结基础买单，再载入这些股票从实际入场至该年实际研究末日的所有现有信号。先保留账本和信号键，T+1/旧股不足由当时账本标记，不按未来恢复价格、未来标签完整性或是否仍在周榜筛掉持有期。

现有 intraday 特征缓存是每天 **8 个固定信号时钟**，不是 240 个完整特征时钟；全会话报价可供恢复/标签模拟。“保留所有时钟”应先明确为保留全部 8 个既有时钟。若希望每分钟入场推断，则需独立的新特征缓存与来源审计，不能把稀疏信号特征前向填充成完整分钟特征。

## 必须先解决的来源延续

第一折的 `selector_2024` 首批名单对应 2023-12-29 信号，最早量价来源为 2023-11-27；2023 是这折的排除年。旧周中队列可在 T 拟合时丢弃早周样本；永久底仓若先用该名单建仓，后来持股、建仓成本、占用现金及其它股票能否买到都会继续受它影响。仅在二月以后检查当前分钟来源，并不能恢复严格隔离。

最小安全办法是：**先按来源和既有 7 日 embargo 筛入场请求，再从干净初始现金运行纯训练规划**。这是按已知来源日期限制训练资料，不是按未来成交或盈利筛选。保留被排除请求和原因；新账本传播全部基础决策/现金状态的最早来源，后续分钟再与当前自身/市场特征来源取最早值，最后接标签并执行完整 `[feature_source_start,label_end]` 隔离。来源不明应失败关闭。若保留污染过的早期交易，必须把污染继续传播，而非仅删除相关股票几天的 T 行。

排名也存在同周其它股票的依赖。本次以原完整 OOF prediction **键集合**关联真实四周来源，机械量化如下；没有重算预测或入场收益：

| 外层 / core 年 | 入选股周 | 自身来源＋7日隔离不合格股周 | 加完整排名供体后不合格股周 | 首个排名来源干净的入场日 |
|---|---:|---:|---:|---|
| 2023 / 2024 | 5,000 | 600 | 600 | 2024-02-19 |
| 2023 / 2025 | 5,100 | 100 | 100 | 2025-01-06 |
| 2024 / 2023 | 4,500 | 100 | 100 | 2023-02-13 |
| 2024 / 2025 | 5,100 | 600 | 800 | 2025-03-03 |
| 2025 / 2023 | 4,500 | 0 | 0 | 2023-02-13 |
| 2025 / 2024 | 5,000 | 0 | 0 | 2024-01-02 |
| 2026 / 2023 | 4,500 | 0 | 0 | 2023-02-13 |
| 2026 / 2024 | 5,000 | 0 | 0 | 2024-01-02 |

计数可能包括年末进入下一排除年之前的 7 日边界；“首个干净日”不表示以后每一周必定合格。这里只检查已保存预测供体键的当前来源，不声称重做整个 universe 或预测数值审计。实现时还应绑定完整供体输入 SHA、资格规则和来源，并用“改写排除年价格不改变干净训练账本/样本 SHA”的反例验证。

旧 `attach_weekly_context` 要求分钟和预测属于同一股周，不能直接用来处理永久持股离榜后的分钟。最小方案是新 T 模型不使用旧四个周排名特征，入场来源仍作为不可省略的状态元数据；如保留，可显式改为“首次入场时已知排名/score”并传播其来源。若使用最新周预测，则只能按信号时点做向后 as-of，保留缺失/陈旧年龄，不得用后来一周预测倒填。以上需在拟合前选择一个固定合同。

## 两个 core 年只能共同作为拟合资料

每折 target core A 的名单由只训练 core B 的周模型生成，反向亦然。例如第一折：2024 名单用 2025 模型，2025 名单用 2024 模型；内部 2026 和外层 2023 都不进入这两个 selector 的拟合。即使新的 T 模型只直接拟合 A 的行，它也通过 A 的选股/排名间接使用了 B 的标签。因此 **不能再把 B 宣称为独立校准或验证年**。

安全沿用方式是 A、B 共同拟合新 T 模型，原 inner 年只用于后续校准/选择，原 outer 年保持最后独立评价；所有折规则先冻结。若必须把 B 当独立校准，就需要重新组织上游 selector，使生成训练 A 的全过程也不使用 B，例如额外独立年或嵌套时间拆分。现有两份互相依赖的 OOF 名单不能提供这个独立性。较早 A 由较晚 B 预测仍是回顾性年度 OOF，不能称为真实向前交易验证。

## 新净价值标签与验收

原 v8/v11 标签是固定端点价格收益，旧训练域还有 95%/100 槽位、25%旧股的静态代理，不等于新 1.5 万元永久底仓的实际 T 净价值。新标签应显式包含请求量、实际首腿量、逐腿费用/滑点、恢复片段、未恢复义务及固定评价时点；数量和成本以该训练账本当前可卖旧股/现金为条件。标签可用未来 core 行情计算，但这些未来执行字段不得进入输入、信号资格或股池。

缺未来价、首腿不成交和恢复失败应保留在机会/状态审计中；不得只用完整盈利配对训练而把执行失败静默删掉。若失败用当日收盘标记价值、另学恢复失败概率，或把标签延长至实际完成，必须在拟合前固定定义；跨日标签需要按实际标签终点隔离，不能沿用“保证同日”的旧筛选后宣称没有失败。单个信号的独立反事实价值不能直接相加为整日组合收益，最终仍要按现金、旧股预算和单股 pending 状态逐时运行账户。

实施前至少应验证：缺入场行情不向未来找首笔价格；不入选下一周仍有持仓样本；入场日可卖量为零；人工加仓不补当日预算；排除年报价改写不改变训练账本和样本；完整排名供体来源传播；core B 不能被误标为独立校准；有限标签筛选不改变推断资格；费用/分片/共同基准净额恒等。当前审查没有实施这些新训练步骤，也没有收益结论。

## 下一轮已明确的最小实现合同

下一轮先重训固定 20/40/60 分钟端点收益，不在本轮增加反事实账户状态分支，也不把标签称为真实组合成交净利润。纯训练队列来自永久底仓的当前旧股状态；端点标签由调用方随后连接，费用转换仍为原先声明的期望净边际近似，实际内部收益最终由完整动态引擎验收。

8 份列表的准确相对路径（均在上述 v8 报告根目录内）：

```
year_2023/oof_weekly_cohort/selector_2024/oof_weekly_selections.csv
year_2023/oof_weekly_cohort/selector_2025/oof_weekly_selections.csv
year_2024/oof_weekly_cohort/selector_2023/oof_weekly_selections.csv
year_2024/oof_weekly_cohort/selector_2025/oof_weekly_selections.csv
year_2025/oof_weekly_cohort/selector_2023/oof_weekly_selections.csv
year_2025/oof_weekly_cohort/selector_2024/oof_weekly_selections.csv
year_2026/oof_weekly_cohort/selector_2023/oof_weekly_selections.csv
year_2026/oof_weekly_cohort/selector_2024/oof_weekly_selections.csv
```

建议新接口 `build_inventory_core_cohort(universe, daily, intraday, fold, cfg, out, inputs, record_input, recovery_store=None) -> (cohort, metadata)`。原 `load_frozen_market_data` 一次读完整年度截面，先 session/provenance/timing 特征，随后才进本模块。模块按折校验完整 universe 来源键 SHA、全部已保存排名供体键与各 selector 身份，补齐入场字段；按 `min(自身来源, 同 trade_week 全排名供体来源)` 与 core/排除年及 7 日边界先筛请求，从每年干净现金调用 `plan_dynamic_entries`。

排名来源要保存 `entry_rank_source_start`、供体数、供体键与来源 SHA；供体必须包含未入选的股票。若原 universe 资格本身依赖更早价格/流动性，则仅排名供体来源还不覆盖资格变化；应绑定原 universe 筛选合同，对资格输入的真实来源也保守传播。不能先缩成成功持股再计算来源。

规划后以 `daily_inventory_audit.baseline_opening_shares > 0` 连接全部现有分钟行，不再按当前周榜或周退出日过滤。输出明确区分 `intraday_feature_source_start`、首次成功入场的自身/排名来源与组合资金竞争的累计最早来源；最终 `feature_source_start` 取三者最早值，缺失当前来源保留给后续严格模型隔离处理。原四个“同周排名”特征不加入新模型。纯训练关闭人工买单，另记原配置是否提供人工路径；真实评价仍支持用户人工指令。

最终 `attach_horizon_labels` 放在 root 调用端；所有可卖持股机会先留审计，再做有限标签/完整时间来源 purge，只允许拟合行变化，不能借此改变推断资格。训练两 core 年共同拟合，原 inner 年才是独立选择集。`plan_dynamic_entries` 没有状态快照恢复 API，当前 `run_dynamic_portfolio` 因而不能直接逐行生成等价于组合现金/T+1状态的独立反事实净利润；如未来需要该标签，需另建经过验证的快照/分支接口或事前固定的完整因果训练轨迹。

可以控制训练量：先完整保存机会及来源/账本摘要，再按固定种子对 `(core_year,date,symbol,signal_time)` 做键哈希抽样或按日期/时钟分层等额抽样；同一键的三个期限保持同一抽样决定。抽样只依当前键/预先固定上限，不能依未来收益、恢复成功、方向盈亏或最终标签大小。最简单沿用模型严格 purge 后的确定性训练行上限；若为了减小标签连接成本而提前抽样，也应在连接标签之前冻结抽样清单和 SHA。它减少拟合/标签物化成本，不改变完整内部账户的机会集合或真实交易频率。未来发生后的真实标签仅用于监督训练，不回流当前状态/入场名单。
