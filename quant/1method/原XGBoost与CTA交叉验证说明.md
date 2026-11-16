# 原XGBoost与CTA的年度交叉验证

本轮[正式研究结论](../reports/original_method_year_validation/run_20260909_124531_658067/研究结论.md)已生成：约束版四年平均夏普-4.368453/-4.372274，未达标。16组收益图、交易记录、4模型/8门槛及独立落盘审计均已保存；不是旧模型的原区间逐值复现。

## 架构与版本

本轮对应用户确认的原策略：Method1使用156列因子截面排名训练XGBoost，Method2在同一个模型分数上加原CTA趋势因子。
它们不是此前平均夏普0.056的LightGBM路径净收益模型族，不改成29列、净收益监督、20/40倍ATR或3/5日期限退出。

对应旧版本是`reports/weekly_high_vol_lot_exit_backtests/run_20260702_123216/`的周度高波动前300、前5门控、每批总资产10%、独立ATR2/4退出。
旧报告记录的5.694637/5.720239属于此前区间和引擎口径，不能直接当本轮年度交叉验证成绩。

流程：复用有限原因子 -> 分年重置递推指标/原24分钟标签并缓存 -> 四折训练核心拟合 -> 内部校准两门槛
-> 内部验证只诊断 -> 模型/规则/源码冻结 -> 实际输入与重载审计 -> 单次四年双执行回顾 -> 中文图表及账本核对。
旧二进制模型不用于重叠年份自测；旧报告不覆盖。

## 原策略保留什么

- 模型目标是t+1到t+25两个观测分钟收盘之间收益的截面百分位，不是收益基点或3/5日收益。
- XGBoost固定220棵树、深度4、学习率0.04、叶权重20、行列采样0.85、L2为5、L1为0.1、种子42，缺失特征仍交给原生模型处理。
- 特征为旧模型的156列，按原列序。每折重新拟合权重，不声称重新独立挖掘156个因子；历史特征筛选偏差保留并披露。
- CTA调用原`add_cta_factor_scores`，叠加强度0.0005；主权重为趋势25%、突破25%、动量20%、趋势质量15%、量价流15%。两方法不是独立风险来源。
- 每折校准期分别计算XGB/CTA的top1分数90%分位门槛，不照搬旧模型0.553左右的绝对阈值，也不读取测试收益或未来成交资格求门槛。
- 周池为上一自然周高波动前300沪深主板股；每24交易分钟选前5且过门槛，单批预算为总资产10%，资金不足缩减，不做空、不加杠杆。
- 同股同日不重复买，隔日可以建立新批次；不因离开股票池或目标组合卖出。
- 每批初始止损为买价减2倍入场ATR240，跟踪线为持仓后已观察最高价减4倍当前ATR240，采用更严格者。仍是原Wilder ATR，不是简单均值ATR。
- 原动态公式在ATR变大时可以放松跟踪线，不擅自改成单向收紧。没有新加最大持仓天数。

## 两种执行口径

| 项目 | legacy（旧式有偏对照） | constrained（信号后成交与约束） |
|---|---|---|
| 核心引擎 | 直接调用原run_lot_backtest | 复用共同账户引擎，退出回调实现原ATR2/4 |
| 退出判断 | 使用成交分钟价格/高点/ATR并同刻卖出，存在偏差 | 当前完成分钟发信号，精确下一分钟尝试成交 |
| 入场ATR | 成交分钟ATR | 入场信号分钟ATR |
| 单批预算 | 成交时卖出后总资产10% | 信号时总资产10% |
| 佣金和过户 | 万八免5、双边十万分之一 | 相同 |
| 印花税 | 旧固定卖出万五 | 2023-08-28前千一，之后万五 |
| 滑点 | 无 | 双边万五 |
| 分钟容量 | 不限制 | 同股买卖共享分钟成交额5% |
| 涨跌停 | 原函数未限制 | 前收±9.5%保守阻成交代理，非真实排队 |
| 受阻退出 | 原规则下次重新判断 | 保存待退出，可能延期/部分成交 |
| 15:00 | 只补日终估值，不新增交易 | 相同 |

两口径在外层之前固定并完整报告，不挑更高者作为最终策略。旧式即使数值较高，也不能作为可执行夏普的证据。
两者的差异不只成本；不能把时序、入场ATR、预算和退出保留机制的差异全部归因滑点。

## 分片和数据边界

2023、2024、2025、2026各留一年，其他年份各自按交易日60%核心、20%校准、20%验证，段起点隔离7自然日。
2026允许进入其他折的训练；但前三折因此包含未来年份信息，只是用户接受的回顾检验，不是历史当时可以部署的成绩。
模型只拟合核心，校准后不再重训，以免门槛分数尺度漂移。内部验证不搜索新规则，也不因为负收益自动停用原策略。

为使后续年份的输入不暗含留出年，EMA/RSI/ADX/MACD/ATR/OBV及趋势数值按年初始化。
有限窗口原因子仍取原单股缓存，但必须该年已有241条合法分钟；训练/校准/验证剔除前周选股来源跨年的行。
按同刻全截面传播原始来源；训练先剔除跨段标签，再对剩余完整标签重新计算收益排名，避免其他股票的越界标签污染目标。

已有年度公共时钟为09:31、09:55、10:19、10:43、11:07、13:01、13:25、13:49、14:13、14:37及15:00估值。
不沿用原逐股全历史行号采样，推断也不按未来标签是否缺失删股票。因此本轮是原策略定义下的隔离重训，不是旧报告完全逐值复现。
原7月版本把257只稳定沪深300股票训练出的模型迁移到全A周池；本轮各折直接在当时周池样本训练。
保留的是模型/因子/退出定义，不是原训练股票集合或原模型权重。这项训练分布变化也可能影响收益，不能只归因市场年份。
缓存来自当前ST/退名称过滤，不是历史点时状态；分红拆股、退市和真实可成交性仍未完整建模。

## 文件角色

| 文件 | 角色和用途 |
|---|---|
| config/original_method_year_validation.yaml | 用户输入，固定数据、模型、门槛、费用和切分 |
| quant_backtest/original_method_data.py | 辅助库，只补原递推指标和24分钟标签，生成带指纹pkl |
| quant_backtest/original_method_portfolio.py | 辅助库，原lot接口、因果ATR回调、标准日终账本转换 |
| scripts/run_original_method_year_validation.py | 可执行入口，build/train/audit/evaluate四阶段 |
| scripts/audit_original_method_results.py | 可执行只读审计，重新读16组CSV核对账务/汇总并统计交易重合，不重跑测试 |
| tests/test_original_method_data.py | 原公式、年界隔离及标签测试 |
| tests/test_original_method_validation.py | 原目标重排、两门槛、退出时序、现金/预算和中文特征解释测试 |
| tests/test_original_method_result_audit.py | 相同/不同交易及现金账本的重合比较测试 |
| reports/original_method_year_validation/cache/ | 可复用生成数据，单股pkl、面板、行情与来源元数据，Git忽略 |
| reports/original_method_year_validation/run_*/ | 正式模型、冻结、内部验证、收益图、逐笔交易及审计 |
| reports/original_method_year_validation_smoke/ | 小样本流程检查，不用于收益验收 |

每轮`输出文件说明.md`逐项解释模型、参数、标准CSV和两口径目录。
`feature_manifest.csv`保存英文机器名及逐列中文含义，不再只提供英文分类。

## 调用顺序

在quant根目录运行，所有计算使用quant环境；build只新建/复用派生缓存，不改原行情。

```bash
/home/lxl/anaconda3/envs/quant/bin/python -m unittest discover -s tests -q
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_original_method_year_validation.py --phase build
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_original_method_year_validation.py --smoke --workers 2
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_original_method_year_validation.py --phase audit --run-dir 实际烟测目录
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_original_method_year_validation.py --phase evaluate --run-dir 实际烟测目录
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_original_method_year_validation.py --workers 4
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_original_method_year_validation.py --phase audit --run-dir 实际正式目录
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_original_method_year_validation.py --phase evaluate --run-dir 实际正式目录
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_original_method_results.py --run-dir 实际正式目录
```

每次train产生新目录，以实际输出替换占位路径；audit重建实际核心和校准门槛并重载原生模型；evaluate只允许一次，禁止复用已看结果改参数重测。
正式4个XGBoost模型供两策略共享，共8个门槛；外层为4年×2方法×2口径共16组，不把8个方法年度组合当8个独立模型。
最后的只读审计可重复执行，每次新增`csv_audit_*/`，包含账务结果、输入/代码指纹、中文交易重合表及审计说明。
约束版账务按另一套公式重建；旧式明确保留同刻退出和旧预算的有偏标记，不能把“账务通过”理解成“可实盘”。

## YAML字段

以下均为必需字段；默认值就是当前YAML，均由新入口读取并保存为resolved_config.yaml，不需手工export。

| 字段 | 当前默认 | 消费者和作用 |
|---|---|---|
| data_root / start_date / end_date | 本地2023至2026年4月行情 | 数据库，限制只读日期范围 |
| annual_cache | 年度282322e813cbf3c6 | 数据库，复用有限因子、周池和公共时钟 |
| original_model_run | 20260526_202355 | 数据库，只读旧156列顺序，不用旧模型预测 |
| output_dir / cache_dir | reports/original_method_year_validation及cache | 入口/数据库，分别新报告与复用数据 |
| test_years | 2023/2024/2025/2026 | 切分，逐年留出不得静默丢年 |
| workers / max_symbols | 4 / 0（全部） | 数据构建/模型线程，烟测覆写2线程与8股 |
| fit_fraction / calibration_fraction / embargo_days | 0.6 / 0.2 / 7 | 切分，余下部分验证，起点隔离自然日 |
| gate_quantile / cta_blend | 0.9 / 0.0005 | 校准/打分，原策略固定值 |
| sample_minutes / min_cross_section | 24 / 30 | 执行/拟合，公共时钟和最小截面；烟测最小1 |
| initial_cash / entry_top_n / entry_equity_fraction / lot_size | 100万元 / 5 / 0.1 / 100 | 两账户执行，资金、前5、单批预算和整手 |
| commission_rate / min_commission / transfer_rate | 0.0008 / 0 / 0.00001 | 两执行口径费用 |
| stamp_rate_before_20230828 / stamp_rate_from_20230828 | 0.001 / 0.0005 | 约束版历史卖出税率；旧式固定后者 |
| slippage_rate / max_bar_amount_fraction | 0.0005 / 0.05 | 只在约束版生效，滑点和共享容量 |
| sharpe_target | 3.0 | 汇总，仅判定不参与调参 |
| model.n_estimators / max_depth / learning_rate | 220 / 4 / 0.04 | XGB固定模型；烟测树数5 |
| model.min_child_weight / subsample / colsample_bytree | 20 / 0.85 / 0.85 | XGB叶约束和行列抽样 |
| model.reg_lambda / reg_alpha / random_state | 5 / 0.1 / 42 | XGB正则化与随机种子 |

回顾指标统一用包含现金日的日频夏普，四年等权；无波动年不删除，2026短年单独注明。
旧式原bar年化另存legacy_native_metrics.json，不拿旧5.7和新日频夏普直接混比。
