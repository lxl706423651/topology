# 以同期队列残差为训练目标的做 T 验证（v8）

v8 只研究 T 模型的训练目标：在每个周期、本折严格核心 OOF 选择且通过已知训练域与年度来源过滤的股票队列中，减去同一信号时刻其他入选股票与自身的平均未来收益，拟合相对这一队列的收益残差。推断时将未来队列均值固定假设为 0，以残差预测近似股票原始收益预测，再使用既有费用后净边际规则。[正式四折验证](../reports/weekly_t_cross_validation/run_20260908_235818_404243/final_report.md)已完成：内部候选均未通过联合资格，四折全部关闭 T，新增净收益均为 0，盈利目标未达成。完整独立模型与执行审计通过，验证的是来源、模型和账目一致性，不是策略盈利。

周选股池、70% 收益排名 / 30% 振幅排名、[v6 OOF 队列](选股队列做T验证说明.md)、[v7 已知训练域](执行条件匹配做T验证说明.md)及 [v5 短周期执行规则](短周期做T验证说明.md)保持不变。这个改动不建立市场对冲，也不保证市场中性或证明 alpha 存在。最终收益必须来自真实股票两腿的模拟成交账本，不能把残差标签或残差预测当成实际股票利润。

## 目标与预测假设

对固定 horizon，先完成已知训练域、未来标签有效性及严格年度来源过滤。对留下的核心训练行，令 `r(i,t,h)` 为股票 i 从下一成交分钟到计划恢复分钟的原始价格收益，定义：

```text
C(t,h) = 此时刻、此周期严格过滤后仍合法的核心 OOF 股票队列
m(t,h) = mean(r(j,t,h), j in C(t,h))
y_residual(i,t,h) = r(i,t,h) - m(t,h)
```

这里的均值是当前 OOF 训练队列的等权均值，包含自身，不是全 A 股指数收益，也不是无条件的全市场收益。同一时刻不足 2 条合法行时不用于拟合。这一最小组数只作用于训练，不能据推断年份的未来标签数量删除当前信号。

GBDT 学习 `y_residual`。推断阶段不计算、不读取未来 `m(t,h)`，固定采用：

```text
predicted_stock_return ≈ predicted_residual_return + 0
```

“未来队列均值为 0”是本轮明确的预测假设，不是由标签均值为 0 推出的事实。例如，假设某股真实残差为 −0.4%，同期队列实际上涨 1%，股票仍会上涨约 0.6%；残差方向正确也可能让先卖后买产生亏损。这个例子只解释假设的局限，不是本轮结果，更不能据事后涨跌切换方向。

未来队列均值只用于构造核心训练标签，不加入任何 `f_*` 特征，不在内部或外层预测时加回，也不把未来市场均值当作已知市场状态。现有信号时刻的市场特征仍是已知输入，沿用原因果构造和来源审计；它们与这里的未来标签均值不同。

## 每折的数据与模型边界

| 外层年份 | 核心模型拟合年 | 内部选择年 |
| --- | --- | --- |
| 2023 | 2024、2025 | 2026 |
| 2024 | 2023、2025 | 2026 |
| 2025 | 2023、2024 | 2026 |
| 2026 | 2023、2024 | 2025 |

两个核心年仍互为 OOF 周选择：每个目标年的周 selector 只训练另一个核心年，排除自身标签年、内部年和外层年。核心周 top100 的中间交易日形成 T 队列。四个周预测排名/score 特征保留，最早来源继续取周预测四个实际观测周与旧分钟来源的最早值；周模型参数训练年/SHA 另行记录，不能与原始价格来源时点混淆。

每个 horizon 独立执行下列顺序：

1. 对核心 OOF 队列应用 v7 已知静态槽位和原 14:30 计划恢复截止过滤；不依据未来成交状态形成当前资格。
2. 接入本周期原始未来收益标签，执行原严格筛选：核心拟合年份、有效标签、信号/标签时序，以及完整真实来源至标签结束区间对排除年份前后 7 个自然日的隔离。
3. 只对第 2 步留下的核心行按 `datetime` 计算未来收益均值，排除合法组数不足 2 的训练组，再形成残差训练目标。不能先用全池或被排除年份的标签求均值后才筛选核心行。
4. 用去掉不足两股组之后的同一批合法核心样本，完成原训练内特征筛选、输入支持域和三个固定 GBDT profile 的拟合。训练抽样或样本上限不能让未来推断年份参与标签构造。
5. 内部年仍使用本折 v4 冻结周选择和对应周模型上下文；不对推断行计算未来均值，不增加静态价格门槛或最小标签组数过滤。以固定零先验下的原始收益近似生成原 27 个候选，并在真实账户执行后按原联合门槛选择。
6. 四折规则全部冻结后统一评价外层，选中的核心 T 权重直接复制，`outer_t_refit=false`。外层沿用本折 v4 周选股和实际旧库存执行；其周模型按原协议使用另外三个训练年份，其中包含内部选择年。

因此，三个训练年份都参与训练流程：两年拟合 T 权重，一年选择规则。残差变换不会把内部选参年变成独立最终测试。较早核心年的周预测仍可能来自较晚核心年参数，2023–2025 的外层训练仍包含后续年份；这些是回顾性的年度研究。

## 费用与实际账户收益

模型输出转换成 `prediction` 后，原双腿净边际函数继续按股票价格比计算买先或卖先预期净利，包含佣金、过户费、历史卖出印花税及双腿滑点。费用后边际门槛仍是 10/20/30 bp，不因标签变为残差就减少成本或将门槛解释为相对指数收益。

实际执行仍按真实股票行情、旧库存、现金、整手、限价代理及分钟成交额容量完成两腿。实际净 T 增量为同一底仓计划下“加 T 账户权益减持有基准权益”，包括未能恢复时真实遗留仓位的估值影响。配对净利润从含滑点的成交价格减真实费用得到，滑点不能再次扣除。

如果残差预测带来了正确的相对排序，但共同市场波动主导股票原始收益，股票 T 仍可能亏损。对残差误差的改善、组内均值为零或相对价差为正，都不能替代实际股票净增量、每日股数恢复及置信区间的判断。

保持的候选和约束如下：

| 项目 | 沿用设置 |
| --- | --- |
| 候选总数 | 20/40/60 分钟 × L2 depth4/L2 depth6/Huber depth4 × 10/20/30 bp，共 27 个 |
| 方向及数量 | 双向；25% 库存；每股每天最多一个请求；100 股整手 |
| 费用 | 每腿佣金 0.08%、过户 0.001%、滑点 0.05%；最低佣金 0；印花税按原日期规则 |
| 资金与容量 | 初始资金 100 万、原 95% 底仓预算及固定 top100；5% 分钟成交额容量；待恢复参考名义额上限初始资金 10% |
| 时间与 T+1 | 原周首末日限制、计划恢复不晚于 14:30、固定盘中/尾盘恢复时钟、真实部分恢复与受阻重试；当日新买不能变成可卖旧股 |
| 输入支持 | 原核心训练 0.1%/99.9% 分位支持域及当前恢复方向风险门控 |
| 内部样本门槛 | 至少 100 完整配对、20 活跃日、8 活跃周、20 股票，原日/周配对及损益集中度限制 |
| 联合资格 | 实际净增量及周块 95% 区间下界为正，并通过库存恢复、T+1、共同底仓及期末恢复要求；无合格候选则关闭 T |

## 新接口、配置与产物状态

独立入口 [`scripts/run_weekly_t_residual.py`](../scripts/run_weekly_t_residual.py)、配置 [`config/weekly_t_cross_validation_v8.yaml`](../config/weekly_t_cross_validation_v8.yaml)和目标模块 [`quant_backtest/weekly_t_residual.py`](../quant_backtest/weekly_t_residual.py)已完成模块、工作流、小样本与正式四折验证。本文件记录已核对的接口，正式运行的源码快照保存在报告目录。

相对 v7 YAML，当前仅有下列四处差异，其余值保持一致：

| 字段 | 必需值 | 消费者和用途 |
| --- | --- | --- |
| `research_version` | `weekly_t_v8_residual_zero_market` | 入口及协议；独立研究版本 |
| `t_target_mode` | `same_clock_cohort_residual_zero_market` | 残差模块；仅接受同刻核心队列去均值目标 |
| `t_market_prior` | `0.0` | 残差推断；未来队列共同项固定为零 |
| `t_residual_min_group_size` | `2` | 训练目标构造；不足两条合法核心标签的同刻组不拟合 |

已确认的模块接口：

```python
validate_residual_config(cfg)
training, metadata, split_audit, decomposition = prepare_residual_training(
    frame, train_years, cfg, excluded_years=()
)
bundle = fit_residual_t_model(frame, train_years, cfg, output_dir, excluded_years=())
predictions = predict_residual_t(bundle, frame, cfg)
```

`training` 保留原始 `target_return`，另加非 `f_*` 的 `target_residual_return`；支持域使用同一批最终合法行。推断保留原字段，新增 `residual_prediction`、固定 0 的 `market_prediction` 及供原交易规则消费的 `prediction`。模型仍使用原 `return_model.txt` 及 bundle 中 `models['return']` 保存方式，不能只凭文件名推断目标仍为原始收益。

每个 `inner_t_h周期_profile/` 模型目录新增 `target_decomposition.csv`，包含 `datetime`、`symbol`、`raw_target_return`、`cohort_mean_return`、`residual_target_return`、`cohort_size`、`feature_source_start`、`label_start`、`label_end`、`kept_for_training`。组数不足的合法标签供体也保留在分解审计中，由 `kept_for_training` 明确区分。完整供体及目标、原始/残差拟合样本和来源 SHA 分别记录，不能把变换后的训练 SHA 冒充原始股票标签身份。

`validation_predictions_h*_*.csv` 与 `t_predictions.csv` 同时保存 `prediction`、`residual_prediction`、`market_prediction`。原 OOF lineage、四周来源、训练域 CSV/JSON、固定周上下文 manifest、训练年份、原生模型、支持域、所有候选与冻结规则记录继续保存。入口在共同依赖基础上增加本轮入口、cohort、训练域、残差模块，以及实际消费的 `annual_panel.py` 与 `strategy.py` 公共接口，共 20 份策略核心及公共接口源码快照。它不表示已枚举所有未调用的传递导入；本次正式完成核验中，20 份启动快照和当前文件 SHA 全部精确一致。

## 验证、运行与审计

新增模块 [9 项测试](../scripts/test_weekly_t_residual.py)和 [4 项工作流测试](../scripts/test_weekly_t_residual_workflow.py)共 13 项通过。覆盖先严格过滤再构造均值、完整供体先于抽样、保留原标签、singleton 审计但不进入训练/支持域、排除年扰动不改变模型、缺失未来字段及推断组大小不改变预测、原生/bundle 预测一致、零先验检查，以及全部内部冻结后再统一外层、核心权重不重拟合。

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_residual.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_residual_workflow.py
```

真实 8 股、内部/外层前 3 个可用周的小样本已完成于 [`run_20260908_235631_167038`](../reports/weekly_t_cross_validation_smoke/run_20260908_235631_167038/final_report.md)：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_residual.py \
  --config config/weekly_t_cross_validation_v8.yaml \
  --max-symbols 8 --smoke-weeks 3 --workers 4 \
  --output-dir reports/weekly_t_cross_validation_smoke
```

小样本四年都关闭 T，净增量为 0，[盈利目标](../reports/weekly_t_cross_validation_smoke/run_20260908_235631_167038/goal_evaluation.json)未达成。它只验证小样本训练与执行流程；没有主动 T 成交，不能作为盈利或实际主动恢复能力的证据。

小样本[独立残差模型审计](../reports/weekly_t_cross_validation_smoke/run_20260908_235631_167038/independent_residual_audit.md)通过，覆盖 8 个 OOF 周 selector、36 个残差 T 模型、12 个训练域、12 个支持域、8 份冻结周上下文与 4 份核心权重复制。原始/残差目标、供体、来源、特征筛选、原生权重和保存预测分别核对，模型审计不读取外层收益。[独立执行审计](../reports/weekly_t_cross_validation_smoke/run_20260908_235631_167038/independent_execution_audit.md)通过，重放 22 条实际成交、22 条基准成交及 52 条库存记录。另有 8 项残差审计器、5 项训练域审计器、10 项执行审计器正反例，共 23 项通过，[归档清单](../reports/weekly_t_cross_validation_smoke/run_20260908_235631_167038/independent_v8_audit_archive.json)保存脚本、日志和 SHA。这些通过项与策略盈利结论分开。

正式运行已完成，使用以下入口，目录为 `reports/weekly_t_cross_validation/run_20260908_235818_404243`，终端日志 `/tmp/weekly_t_full_cv_v8.log`：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_residual.py \
  --config config/weekly_t_cross_validation_v8.yaml
```

独立模型入口 [`scripts/audit_weekly_t_residual.py`](../scripts/audit_weekly_t_residual.py)输出 `independent_residual_audit.json/md`；模型齐全的单折可加 `--test-years 2023`，小样本须加 `--smoke`。独立执行入口 [`scripts/audit_weekly_t_residual_execution.py`](../scripts/audit_weekly_t_residual_execution.py)输出 `independent_execution_audit.json/md`，只在完整主报告及所有年度结果齐全后读取外层账目。

```bash
report_dir='reports/weekly_t_cross_validation/run_20260908_235818_404243'
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_residual.py \
  --report-dir "$report_dir"
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_residual_execution.py \
  --report-dir "$report_dir"
```

上述为正式审计的复现命令；小样本改用其目录并在模型审计增加 `--smoke`。本次完整[残差模型审计](../reports/weekly_t_cross_validation/run_20260908_235818_404243/independent_residual_audit.md)与[执行审计](../reports/weekly_t_cross_validation/run_20260908_235818_404243/independent_execution_audit.md)均已完成通过；[完成核验](../reports/weekly_t_cross_validation/run_20260908_235818_404243/independent_audit_completion_verification.md)与[归档 SHA 清单](../reports/weekly_t_cross_validation/run_20260908_235818_404243/independent_v8_audit_archive.json)保存实际范围及证据。

## 正式结果与未启用原因

每年独立初始资金 100 万元。[年度结果](../reports/weekly_t_cross_validation/run_20260908_235818_404243/annual_summary.csv)中的收益含底仓未退出持仓的估值；四年的净值曲线不能拼接为一个连续账户。

| 外层年份 | 实际评价日期 | 交易日 | 持有基准收益 | 加 T 账户收益 | T 净增量（元） | 完整 T 配对 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | 02-13—12-29 | 218 | −11.6656% | −11.6656% | 0.00 | 0 |
| 2024 | 01-02—12-31 | 242 | −16.4992% | −16.4992% | 0.00 | 0 |
| 2025 | 01-06—12-31 | 241 | +31.4697% | +31.4697% | 0.00 | 0 |
| 2026 | 01-05—04-30 | 77 | +8.2277% | +8.2277% | 0.00 | 0 |

[目标判断](../reports/weekly_t_cross_validation/run_20260908_235818_404243/goal_evaluation.json)为未达成：净增量合计 0 元、正增量年份 0/4，周块 bootstrap 95% 区间为 [0，0] 元。该区间来自全部关闭 T 的策略，不能解释为主动做 T 的盈利稳定性。实际持有基准逐日与冻结 v4 基准精确相同。

每折在各自内部年评价 27 个候选，合计 108 个候选记录。它们可能共享模型或请求，并非 108 个独立实验；正负按实际账户净增量计算，全部未通过原联合资格。

| 外层折 / 内部年 | 净增量为正 | 为负 | 为零 | 内部最高净增量（元） | 合格候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [2023 / 2026](../reports/weekly_t_cross_validation/run_20260908_235818_404243/year_2023/validation_rule_metrics.csv) | 6 | 18 | 3 | +127.35 | 0 |
| [2024 / 2026](../reports/weekly_t_cross_validation/run_20260908_235818_404243/year_2024/validation_rule_metrics.csv) | 0 | 21 | 6 | 0.00 | 0 |
| [2025 / 2026](../reports/weekly_t_cross_validation/run_20260908_235818_404243/year_2025/validation_rule_metrics.csv) | 12 | 11 | 4 | +486.36 | 0 |
| [2026 / 2025](../reports/weekly_t_cross_validation/run_20260908_235818_404243/year_2026/validation_rule_metrics.csv) | 9 | 14 | 4 | +824.61 | 0 |
| 合计 | 27 | 64 | 17 | +824.61 | 0 |

108 个记录中，106 个周块置信下界不为正、102 个完整配对不足 100 组、70 个出现每日 T 股数偏差、70 个出现恢复失败、48 个偏离共同底仓计划。这些拒绝原因可以重叠，不能相加当作候选总数。仅有两个候选的周块置信下界为正，均在 2025 折内部 2026 年、h20/L2 depth6：10 bp 候选净增量 +486.36 元，但只有 34 组完整配对且存在恢复及底仓偏差；20 bp 候选净增量 +220.21 元，只有 15 组、12 个活跃日、12 只股票，未满足样本支持门槛。

内部最高 +824.61 元来自 2026 折内部 2025 年的 h60/L2 depth6/10 bp，拥有 295 组完整配对、108 个活跃日、46 周和 217 只股票，但周块 95% 区间为 [−1,442.96，+3,265.00] 元，另有 24 个库存偏差日、14 个恢复失败日和 6 项共同底仓偏差。它不满足原合同，不能因账面净增量为正而启用，更不能据外层收益放宽门槛。

已冻结的[首折内部校准诊断](../reports/weekly_t_cross_validation/run_20260908_235818_404243/inner_calibration_v8.md)只使用核心 2024/2025 模型和内部 2026 年，未读取外层或重选规则。在 v7/v8 支持域交集的相同信号上，h20/L2 depth4 的同刻共同预测方差占比从 70.52% 降至 3.21%，h60/L2 depth6 从 60.25% 降至 3.26%；这描述预测结构，不是盈利贡献。v8 的 h60 绝对预测 60–100 bp 箱有 76 行，实际方向毛价差均值约 +28.33 bp，按计划价格和原双腿成本计算的标称净价差仍约 −2.90 bp。标称价差假定计划成交，不等于实际账户 PnL；两版按各自预测重新分箱，箱内成员不同，不能解释为同一组股票的改善，极少量更高预测样本也不能证明盈利。[诊断 JSON](../reports/weekly_t_cross_validation/run_20260908_235818_404243/inner_calibration_v8.json)保存 118 项输入 SHA 与固定规则复现记录，相关统计未用于修改本轮参数。

正式账本无主动 T 成交，因此 T+1 违规、每日 T 股数偏差、T 恢复失败和待恢复配对均为 0。这不代表周度退出成功：实际账户与基准共有 171 个周退出失败年度日、307 个股票日，分年为 57/37/55/22 日。2025 期末仍有 3 项持仓、市值 14,464 元；2026 期末 4 项、市值 36,325 元。其中 SH.603718 的 2,500 股按 04-29 最后价格 3.90 元估值 9,750 元，04-30 时已陈旧一天，不能假定能按此价格变现。

完整独立审计核对 8 个 OOF selector、36 个残差 T 模型、12 个训练域、12 个支持域、8 份周上下文与 4 份外层核心权重复制及预测；执行从 7,039 个原始 CSV 重放 31,843 条实际事件、31,843 条基准事件和 76,084 行库存。现金/净值最大数值误差约 1.397×10⁻⁹ 元。审计通过说明这些产物与规则、数据和账目一致，用户要求的额外净收益仍未实现。

## 研究限制

这个假设来自已完成的训练内部诊断，并非通过本轮外层收益挑选。旧 v7 的全关闭 T 和旧周选择诊断均不能证明新残差目标会盈利。同期队列可能带有共同风格、行业或资金偏好，其均值不是无偏的市场基准；推断时零先验可能在单边市场明显失准。

现有市场数据已经反复用于研究。即使严格排除外层标签及来源、只在核心内变换标签，也不能把这段历史重新称为未接触测试；2026 只有截至 4 月 30 日的数据。原历史 ST、精确限价/盘口、公司行动及证券主表不完备的限制保留。正式评价必须如实报告成本后的股票收益、实际 T 样本、恢复失败和遗留底仓，不把残差统计或程序检查通过写成用户的盈利目标已经完成。
