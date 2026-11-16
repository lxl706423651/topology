# 5method：周度预测前 100 与学习型库存做 T（v4）

最新择时研究见[做T择时能力改进说明](做T择时能力改进说明.md)：v11结合一手研究新增原始价差回归、扣费盈利三分类、反转/趋势状态及20/40/60分钟选择，按内部实际净收益冻结四折规则。[正式报告](../reports/weekly_t_cross_validation/run_20260909_182945_388303/final_report.md)完成86组T，净增量−573.72元；每组平均−6.67元，比v10的−5.85元更差，毛价差−131元。**本轮未验证择时改善；不能把交易减少导致的总亏损收窄当成有效优化。**四年均开启并实际做T，本轮T库存恢复/T+1失败均为0；原底仓周退出风险保留。入口为 `scripts/run_weekly_t_timing.py`，配置 `config/weekly_t_cross_validation_v11.yaml`；[逐笔质量对比](../reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/report.md)保留全部负结果。

主动版本见[主动做 T 验证说明](主动做T验证说明.md)：v9 复用 v8 核心模型和内部候选，移除样本量、置信区间、集中度及正收益的统计停用门槛，按实际完成过 T 且符合库存执行要求的内部候选选择并开启交易。[正式四折结果](../reports/weekly_t_cross_validation/run_20260909_124918_833635/final_report.md)已完成 34 个 T 配对，额外净收益合计 +276.31 元；2025 年仍有 4 个股数偏差日和 2 个恢复失败日。费用和实际成交规则保留，统计只作参考。[完整独立执行审计](../reports/weekly_t_cross_validation/run_20260909_124918_833635/independent_execution_audit.md)及 4 个复用模型身份核对已通过，审计通过不等于每天股数恢复成功；历史 v4—v8 结论保留。

当前训练条件匹配实验见[执行条件匹配做T验证说明](执行条件匹配做T验证说明.md)：v7按已知静态股数和原恢复截止时间筛选T训练样本，四折正式验证均关闭T、额外净收益为0。

已完成的选股队列实验见[选股队列做T验证说明](选股队列做T验证说明.md)：v6只在核心年度交叉拟合选出的周前100周中样本上训练做T，加入已知周预测上下文；四折正式验证均关闭T、额外净收益为0。费用、候选及执行约束沿用[短周期做T验证说明](短周期做T验证说明.md)中的v5协议；v5四折均关闭T、净增量为0。本页保留已完成的v4结果和复现说明。

当前共用引擎已增加v5可选功能，缺省行为已与v4冻结引擎逐表对照。需要严格逐文件复现旧运行时，应使用相应报告中的源码快照。

本方法用第 N 周及此前已知行情预测第 N+1 周的收益与振幅，按预测排名的 `70% / 30%` 组合选出前 100 只股票。下一周首日建立等金额底仓、末日退出；周中根据单独训练的分钟模型做 T，与完全相同底仓订单的周度买入持有基准比较净增量。

架构分成数据、模型和账户执行三层：数据层从只读分钟行情生成周度、盘中及日行情缓存，另建真实尾盘分钟缓存；模型层先追踪每行特征的最早行情来源，再严格排除来源区间与留出年相交的训练行，仅用合格的 `f_*` 特征筛选及拟合；执行层维护两套独立现金账本、昨日可卖股数、做 T 配对及共同底仓订单差异。入口脚本负责训练内部选参、冻结规则、留出年评价和报告。每个年度分别从初始现金开始。

v4 保留过去完整周的收盘流动性门槛、当日及隔夜行情特征、当前恢复方向风险过滤、同刻预测强度排序，以及 14:55 至 15:00 的真实分钟分批恢复。新增严格历史来源隔离，覆盖前 4 个实际观测周、前一实际观测周/日及同刻市场特征的前收来源；累计 `f_history_days` 从模型输入排除，只保留事前历史资格用途。

v2/v3 曾只检查样本当周与标签日期，滚动特征仍可能包含留出年的行情；旧版保留为旧隔离协议参考。v4 修复这一模型输入边界问题，但历史外层结果已经查看，工程改进受前轮审计启发，因此仍属于研究复测，不能称为从未查看过的全新独立测试。最终四折结果及未达到目标的原因见下方。

## 文件角色

| 文件或目录 | 类型 | 用途 |
| --- | --- | --- |
| 本目录 `README.md` | 方法文档 | 架构、命令、配置、执行语义与验证口径 |
| [`config/weekly_t_cross_validation_v4.yaml`](../config/weekly_t_cross_validation_v4.yaml) | 当前默认输入 | 固定 v4 数据范围、选股权重、候选模型、严格来源隔离、恢复方式及费用 |
| [`config/weekly_t_cross_validation_v3.yaml`](../config/weekly_t_cross_validation_v3.yaml) | 保留的 v3 输入 | 分钟分批恢复等工程改进的旧隔离协议；不具备完整滚动来源排除 |
| [`config/weekly_t_cross_validation.yaml`](../config/weekly_t_cross_validation.yaml) | 保留的 v2 输入 | 保留旧配置；精确复现旧运行还须使用对应报告的源码快照 |
| [`scripts/run_weekly_t_cross_validation.py`](../scripts/run_weekly_t_cross_validation.py) | 可执行脚本 | 数据调度、年度拟合、内部候选比较、规则冻结和报告 |
| [`quant_backtest/weekly_t_data.py`](../quant_backtest/weekly_t_data.py) | 数据辅助库 | 分钟聚合、因果特征、未来标签、周历及独立缓存 |
| [`quant_backtest/weekly_t_features.py`](../quant_backtest/weekly_t_features.py) | 特征与信号辅助库 | 过去周收盘成交额、当前及隔夜行情特征、当前恢复方向风险过滤；不修改基础缓存 |
| [`quant_backtest/weekly_t_provenance.py`](../quant_backtest/weekly_t_provenance.py) | 历史来源辅助库 | 按完整观测日历追踪模型量价特征最早来源和同期市场依赖；不改变特征数值 |
| [`quant_backtest/weekly_t_recovery_data.py`](../quant_backtest/weekly_t_recovery_data.py) | 执行数据辅助库 | 单独缓存 14:55 至 15:00 六个真实分钟的量价，不生成特征或回填缺价 |
| [`quant_backtest/weekly_t_models.py`](../quant_backtest/weekly_t_models.py) | 模型辅助库 | 周双 LightGBM 模型、分钟回报模型、真实来源至标签的年份隔离与模型持久化 |
| [`quant_backtest/weekly_t_engine.py`](../quant_backtest/weekly_t_engine.py) | 组合引擎 | 周度底仓、T+1、同刻排序、分批恢复、片段费用及共同底仓审计 |
| [`quant_backtest/weekly_t_validation.py`](../quant_backtest/weekly_t_validation.py) | 候选评价辅助库 | 用 spawn 进程并行完整账户评价，保存受阻单和库存审计；异常向入口传播 |
| [`quant_backtest/annual_portfolio.py`](../quant_backtest/annual_portfolio.py) | 共享辅助库 | 费用和每日绩效计算 |
| [`quant_backtest/inventory_t.py`](../quant_backtest/inventory_t.py) | 共享辅助库 | 默认费用设置及下一交易分钟时钟；本方法使用独立的周度引擎 |
| [`scripts/test_weekly_t_data.py`](../scripts/test_weekly_t_data.py) | 验证脚本 | 特征因果性、周历、缺失数据、首末分钟及缓存复用 |
| [`scripts/test_weekly_t_features.py`](../scripts/test_weekly_t_features.py) | 验证脚本 | 隔夜特征、当前方向过滤及过去周流动性的因果性 |
| [`scripts/test_weekly_t_provenance.py`](../scripts/test_weekly_t_provenance.py) | 验证脚本 | 实际观测周、停牌长间隔、市场前收依赖、未知来源及原特征不变 |
| [`scripts/test_weekly_t_recovery_data.py`](../scripts/test_weekly_t_recovery_data.py) | 验证脚本 | 六个精确恢复分钟、缺失/零量保留、缓存指纹和复用 |
| [`scripts/test_weekly_t_models.py`](../scripts/test_weekly_t_models.py) | 验证脚本 | 年份隔离、无标签推断、70/30 排名、前 100、模型保存及重载 |
| [`scripts/test_weekly_t_strict_models.py`](../scripts/test_weekly_t_strict_models.py) | 验证脚本 | 跨年来源清除、缺失/未来来源拒绝、累计历史列排除与训练结果不变性 |
| [`scripts/test_weekly_t_engine.py`](../scripts/test_weekly_t_engine.py) | 验证脚本 | 周度底仓与配对账本、T+1、真实恢复失败、预算及限价代理 |
| [`scripts/test_weekly_t_partial.py`](../scripts/test_weekly_t_partial.py) | 验证脚本 | 分钟分批恢复、片段费用与现金恒等式、容量共享、优先级及跨日剩余股数 |
| [`scripts/test_weekly_t_validation.py`](../scripts/test_weekly_t_validation.py) | 验证脚本 | 单进程与 spawn 评价一致、异常传播、执行约束和失败审计 |
| [`scripts/test_weekly_t_workflow.py`](../scripts/test_weekly_t_workflow.py) | 验证脚本 | 因果股票池、动作方向、周边界及增量统计 |
| `reports/weekly_t_cross_validation/cache/` | 可复用缓存 | 分股 pkl、`weekly.pkl`、`intraday.pkl`、`daily.pkl`、数据质量与元数据 |
| `reports/weekly_t_cross_validation/recovery_cache/recovery_指纹/` | 可复用执行缓存 | 分股 pkl、`recovery_bars.pkl`、`metadata.json`、`data_quality.csv`；与基础缓存独立 |
| `reports/weekly_t_cross_validation/run_日期_时间/` | 生成输出 | 独立运行的年度模型、交易、收益、审计与代码快照 |

## 流程与命令

从仓库根目录运行，建议使用项目的 `quant` Python 环境。原始行情路径按年份/股票组织，包含分钟 `datetime,open,high,low,close,volume,amount`；股票代码由文件名取得。中国交易时段、午休、节假日及缺失分钟按原始行情和全市场日历处理，不生成虚构的可成交价格。

1. 数据层读取全部历史文件的主板股票并集，按已完成周构造特征和下一周标签，保存独立缓存。全量数据候选为 3,302 只主板股票，时间范围覆盖 804 个市场交易日，截至 `2026-04-30`。另读真实尾盘分钟执行缓存，运行时先增加过去周执行特征与盘中行情特征，再用完整历史日历及完整同期截面补充 `feature_source_start`，之后才筛选股票池和拆分年份；不改基础缓存。
2. 每周候选需要满足过去价格、日均成交额、完整周各日收盘分钟成交额及历史天数门槛。模型层按每折核心年份严格清除跨排除年的完整来源区间后训练周模型，预测内部验证年的周排名；T 模型在核心年份全部符合过去周资格且来源合格的股票上拟合，再仅对内部验证年选中的周度股票评价候选。
3. 用内部验证年的配对收益近似排列候选，默认对全部 162 个配置做完整账户评价；完全相同的请求只模拟一次，默认以 6 个 spawn 进程并行，但保留每个候选的指标与复用指针。筛选有净增量且约束合格的规则，冻结模型 profile、方向、阈值和库存比例后，用全部外层训练年重拟合模型，评价一个留出年。
4. 根目录生成年度比较；每个 `year_年份/` 保留完整模型、选股、盘中预测、成交及库存审计。主报告为 `final_report.md`。

先运行不接触真实数据的合成验证：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_data.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_features.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_provenance.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_recovery_data.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_models.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_strict_models.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_engine.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_partial.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_validation.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_weekly_t_workflow.py
```

小样本烟测只验证流程，不代表正式收益；它把模型缩小到 20 棵树，放宽价格/日均及收盘成交额门槛，限制训练行数，仅完整评价代理排序前 2 个候选，并将候选进程数设为 2、仅用前几个完整持仓周：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_cross_validation.py \
  --config config/weekly_t_cross_validation_v4.yaml \
  --max-symbols 20 --top-n 5 --smoke-weeks 3 --test-years 2026 --workers 2 \
  --output-dir reports/weekly_t_cross_validation_smoke
```

只构建或复用基础数据及真实尾盘分钟缓存，不拟合模型、不评价测试收益；运行时补充特征与来源追踪在后续正式流程中生成：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_cross_validation.py \
  --config config/weekly_t_cross_validation_v4.yaml --cache-only
```

完整运行会构建或复用独立缓存、训练年度模型、选参并评价四个留出年，耗时和内存需求较大：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_t_cross_validation.py \
  --config config/weekly_t_cross_validation_v4.yaml
```

入口省略 `--config` 时也使用 v4。一次性 CLI 覆盖包括 `--workers`、`--max-symbols`、`--top-n`、`--smoke-weeks`、`--test-years`、`--output-dir`。`--test-years 2026` 只减少要执行的外层折，仍使用数据范围内其他年份训练。持久参数以 YAML 为准，解析后的配置和源码快照写入运行目录。候选并行度单独由 `candidate_workers` 控制；`--workers` 不直接覆盖它。

v2/v3 配置及旧报告保留，但当前引擎的两端恢复模式也已经支持部分成交；只把 `--config` 改回 v2 文件不能精确重现旧版整笔恢复实现。追溯任一旧结果应使用其 `resolved_config.json` 和 `source_snapshot/`，不要覆盖旧报告。仍在执行的旧进程使用其启动时加载的代码，不能依据后来切换的工作区默认版本判断该运行的协议。

## 配置字段

下表默认值对应 [`weekly_t_cross_validation_v4.yaml`](../config/weekly_t_cross_validation_v4.yaml)。完整 YAML 是运行输入；除标明可选的限制项外，字段为解析后必需的研究配置。入口脚本消费全部配置并分别传给数据、模型和执行层，路径按仓库根目录解释。模型 profile 中的固定参数及 LightGBM 实际合并参数另保存在各模型的 `model_metadata.json`。

| 字段 | 默认值 | 用途及输出影响 |
| --- | --- | --- |
| `research_version` | `weekly_t_v4` | 写入研究协议及报告，明确严格来源隔离版本及历史外层已查看的复测身份 |
| `data_root` | `ashare_minute_data_2023_2026_04/extracted` | 只读分钟行情输入 |
| `cache_dir` | `reports/weekly_t_cross_validation/cache` | 独立数据缓存；不改旧年度因子面板 |
| `recovery_cache_dir` | `reports/weekly_t_cross_validation/recovery_cache` | 六个真实尾盘分钟的独立缓存 |
| `output_dir` | `reports/weekly_t_cross_validation` | 新建带时间戳的报告根目录 |
| `start_date` / `end_date` | `2023-01-01` / `2026-04-30` | 数据范围和来源指纹 |
| `known_complete_end_date` | `true` | 事前确认最后观测日是当周最后交易日，允许该周标签视为完整 |
| `test_years` | `[2023,2024,2025,2026]` | 需要执行的外层留出年 |
| `workers` | `12` | 数据并行进程和 LightGBM CPU 线程数 |
| `candidate_workers` | `6` | 完整候选账户评价的 spawn 进程数，公共输入按进程初始化一次；增大需要更多内存 |
| `max_symbols` | `0` | 可选股票数截断，0 表示全量主板并集 |
| `signal_times` | `09:35,09:45,10:00,10:30,11:00,13:05,13:30,14:00` | 盘中模型的已完成分钟信号时钟；首腿下一分钟执行 |
| `min_history_days` | `20` | 周末已知的最低历史交易日数 |
| `min_avg_daily_amount` | `50000000` | 上一周平均日成交额最低 5,000 万元 |
| `min_prior_close_auction_amount` | `500000` | 完整上一周每个市场交易日的有效收盘分钟成交额均须至少 50 万元；缺失记不合格 |
| `min_price` | `3.0` | 上一周末价格下限；不使用下一周价格筛选 |
| `use_weekly_execution_features` | `true` | 补充过去周收盘分钟成交额最小值、均值和缺失天数；启用正数收盘门槛时须提供这些特征 |
| `add_session_context` | `true` | 补充当前及隔夜涨跌、同期市场差异、前日振幅标准化等盘中特征 |
| `top_n` | `100` | 每周选股数上限及等金额预算分母 |
| `weekly_top_n` | `100` | 周模型标记 `selected` 的上限；入口总是令其等于 `top_n` |
| `weekly_return_weight` | `0.70` | 预测周收益排名权重 |
| `weekly_amplitude_weight` | `0.30` | 预测日均振幅排名权重；两权重之和须为 1 |
| `weekly_max_features` / `t_max_features` | `60` / `60` | 周模型及 T 模型训练筛选特征数上限 |
| `feature_min_coverage` | `0.80` | 训练特征非空覆盖率门槛 |
| `feature_max_correlation` | `0.95` | 训练特征 Spearman 相关性去重阈值 |
| `feature_selection_sample_rows` | `20000` | 特征相关性和标签相关性分析抽样上限；覆盖率仍按训练数据 |
| `min_training_rows` | `100` | 清除跨年标签及边界样本后最小训练行数 |
| `weekly_max_train_rows` | `0` | 可选周模型行数上限，0 表示全部训练样本 |
| `t_max_train_rows` | `1500000` | 可选 T 模型训练行数上限，固定种子抽样，特征选择仍在训练范围 |
| `embargo_days` | `7` | 严格模式下，最早真实行情来源 `feature_source_start` 至 `label_end` 的整段区间不得与排除年份及前后 7 个日历日相交 |
| `strict_feature_provenance` | `true` | 先追踪完整历史量价依赖，再据来源清除训练行；未知、晚于信号或跨被排除年的来源不能训练 |
| `excluded_model_features` | `[f_history_days]` | 从候选模型列中排除累计历史天数；保留该原始列用于 `min_history_days` 事前资格；严格入口要求此排除项 |
| `exclude_flat_signal_bars` | `false` | 是否在 T 训练/预测时排除当前平价分钟；默认允许，执行时另查真实限价或代理 |
| `t_avoid_locked_exit_side` | `true` | 当前已知行情命中恢复方向的限价代理或报价无效时，不发该方向首腿；不查看未来恢复价 |
| `t_signal_priority` | `predicted_edge` | 同一执行分钟按预测收益绝对值降序、股票代码升序分配有限现金；不跨时刻排序 |
| `use_minute_recovery` | `true` | 使用真实 14:55 至 15:00 分钟逐次分批恢复；关闭时仅用 14:55 与收盘两个时点 |
| `initial_cash` | `1000000` | 每折两个账户分别独立起始现金（元） |
| `base_fraction` | `0.95` | 纯持有基准权益用于周度底仓的目标比例，剩余现金支持做 T 和费用 |
| `lot_size` | `100` | A 股整数手股数 |
| `commission_rate` / `min_commission` | `0.0008` / `0.0` | 双边佣金与单笔最低佣金（元） |
| `transfer_rate` | `0.00001` | 双边过户费率 |
| `stamp_rate_before_20230828` | `0.001` | 2023-08-28 前卖出印花税 |
| `stamp_rate_from_20230828` | `0.0005` | 2023-08-28 起卖出印花税 |
| `slippage_rate` | `0.0005` | 每边方向滑点，叠加在实际参考价 |
| `max_bar_amount_fraction` | `0.05` | 单分钟成交额参与上限，影响实际成交股数 |
| `flat_st_limit_rate` | `0.05` | 缺真实限价时，对平价分钟额外检查前收盘 ±5% 四舍五入到分的候选限价 |
| `nonflat_limit_proxy` | `0.098` | 缺真实限价时，所有分钟距前收方向涨跌幅达到此值即保守阻成交 |
| `last_entry_time` | `14:30` | 最晚允许做 T 首腿执行时刻；默认信号表最晚为 14:00 |
| `repurchase_buffer` | `0.12` | 先卖后买的回补价格现金缓冲 |
| `t_model_profiles` | `l2_depth4,l2_depth6,huber_depth4` | 预声明分钟模型候选，只在训练内部比较 |
| `t_huber_alpha` | `0.01` | Huber 原始收益残差截断值；1% 残差之外采用稳健损失 |
| `t_thresholds` | `0.003,0.005,0.0075,0.01,0.015,0.02` | 预期回报绝对值候选门槛；最终选择按费用后净增量 |
| `t_directions` | `sell_first,buy_first,both` | 先卖、先买或双向库存做 T 候选 |
| `t_trade_fractions` | `[0.25,0.5,1.0]` | 每只股票每日请求使用的可卖库存比例候选，受真实可卖量/现金限制 |
| `candidate_shortlist_size` | `0` | 0 表示全部候选做完整账户评价；正数只评价代理排序前 N 个，烟测设为 2 |
| `bootstrap_samples` | `2000` | 周净增量重抽样次数，输出 95% 区间 |
| `random_state` | `42` | 特征抽样、模型、训练行抽样及 bootstrap 随机种子 |
| `smoke_weeks` | `0` | 可选烟测周数，0 表示完整区间；启用时还缩小模型和放宽池门槛 |
| `weekly_model.n_estimators` | `300` | 周双模型树数 |
| `weekly_model.learning_rate` | `0.04` | 周模型学习率 |
| `weekly_model.max_depth` / `weekly_model.num_leaves` | `5` / `24` | 周模型深度及叶子上限 |
| `weekly_model.min_child_samples` | `200` | 周模型叶子最小样本数 |
| `weekly_model.reg_lambda` | `10.0` | 周模型 L2 正则 |
| `weekly_model.verbosity` | `-1` | 周模型库日志详细程度 |
| `t_model.n_estimators` | `300` | T 模型树数 |
| `t_model.learning_rate` | `0.04` | T 模型学习率 |
| `t_model.min_child_samples` | `500` | T 模型叶子最小样本数 |
| `t_model.reg_lambda` | `20.0` | T 模型 L2 正则 |
| `t_model.verbosity` | `-1` | T 模型库日志详细程度 |

## 选股、底仓与做 T 语义

周收益标签是下一周首个交易日开盘到最后交易日收盘的回报。振幅标签是下一周每天 `(high-low)/open` 的平均值，始终非负。两个模型分别学习标签的周截面排名，`pred_return` 和 `pred_amplitude` 是排名尺度的预测，不能直接解释为未来上涨百分比。推断时再次在该周候选中分别排名，按 70/30 加权；总分相同按股票代码稳定排序。

股票池只用过去周末的价格、均额、历史天数和过去完整周的收盘分钟成交额筛选，不依赖下一周标签或未来是否存续，也不使用 2026 年名称快照筛选过去 ST 状态。当前配置要求第 N 周全部市场交易日都有有效收盘分钟，且每一天该分钟成交额至少 50 万元；缺少任何一天有效报价或成交额时，最小额特征记为 0。这个历史门槛不能保证第 N+1 周未来收盘能成交。

周初每只候选的预算固定为 `基准账户权益 × base_fraction / top_n`，不把缺股、容量不足或不满一手留下的预算分给其他股票。前 100 是信号排名，实际成交可能少于 100 只；查看 `weekly_selections.csv`、`weekly_entry_plans.csv` 和成交记录才能知道实际仓位。

首日 09:30 预定开仓使用首个 09:31 分钟的 `open` 作为开盘价代理，末日 15:00 预定退出使用收盘价。首末分钟的容量及价格是成交模拟输入，订单预算和选股已事先确定；这不是当前分钟模型使用未来价格发信号。第一天新股不可卖，最后一天禁止新做 T，以保留周末退出额度。只有两个交易日的周只持有；只有一个交易日的周不能兼顾 T+1 与当周退出，事前跳过。

T 模型预测 `recovery_price / execution_price - 1`，其中首腿为信号后一交易分钟，标签终点仍为当日 14:55，实际恢复成交可以分布在更晚分钟。负预测达到门槛时可先卖旧股再买回，正预测达到门槛时可先买后卖旧股。盘中特征包含隔夜跳空、相对前收涨跌、信号价格对数、前日振幅标准化和同期市场差异。同期市场特征只在同一已完成信号时刻、筛选周度持仓前的盘中股票集合上计算，训练与推断口径一致；不使用当日收盘或未来尾盘信息。v4 同时追踪市场日收益贡献者的前收来源，避免个股自身历史合格但市场特征仍引用排除年行情。

先用当前信号分钟的价格、高低价和前收盘检查恢复方向：先买后卖需要将来卖出，当前已触及向下限价代理时不启动；先卖后买需要将来买回，当前已触及向上代理时不启动。当前报价无效也不启动。此过滤不查看后来的恢复分钟；之后的信号时刻解除限制时仍可入选。每股每日仅采用通过该过滤且最早达到冻结阈值的一次信号，最多一组。同一执行分钟内按 `abs(prediction)` 从大到小竞争有限现金，平手按股票代码；不能用稍后信号重排较早委托。

当天买入不增加可卖额度；先买后卖的卖出对象也是昨日库存。`predict_t` 输出保留未来成交/恢复字段供账户引擎使用，但模型输入严格限于训练选中的 `f_*`；未来标签缺失不会删除原本可推断的当前样本。

恢复从 14:55 到 15:00 逐个真实分钟，按当时容量、现金和剩余可卖库存处理整数手；可以分批，缺分钟不补，也不以其他时刻价格替代。零量或零成交额行情仍由执行层拒绝成交。首腿费用及滑点成本按恢复股数分摊，第二腿每个真实片段各自计费；15:00 的恢复与周度退出共享该分钟容量。首腿不能利用这些未来分钟的容量提前缩量或删单。

当天仍不能恢复时，把剩余真实股数和配对债务带到次日开盘、尾盘分钟及收盘继续修复，保持 T+1 约束；恢复失败不会被写成已完成。周末底仓退出失败也按实际持仓跨日估值并重试。每日日终目标为 `日初股数 + 周度买入 - 周度卖出`，周中没有底仓变动时应等于日初股数。当前风险过滤和过去流动性门槛均不能保证未来每天恢复成功，必须以库存审计确认。

每折估值和重试覆盖最后持仓周之后、该年度数据中剩余的真实交易日，不在预定末周退出日截断尚未完成的债务。`evaluation_start`、`evaluation_end`、`evaluated_trading_days` 记录实际评价范围；到年度数据终点仍未完成的项目保留为期末未完成项目。

纯持有基准独立决定后续周度底仓计划，做 T 盈亏不放大开仓预算。若做 T 账户因为现金、容量或 T+1 不能完成与基准相同的底仓订单，引擎保留实际成交并记录 `base_plan_mismatches`；此时两个账户净值差还包含底仓偏离影响，不能只称作已实现做 T 价差。未平仓和未配对项目必须同时查看。

涨跌停优先读取真实限价；缺少时以当前成交价相对前收盘的方向变化近似阻成交：买单向上涨幅或卖单向下跌幅达到 `9.8%` 时拒绝，无论是否平价分钟。平价分钟另检查价格是否恰好等于“前收盘 ±5% 四舍五入到分”的候选限价，容差 `0.000001` 元；匹配则保守拒绝对应方向。普通 `+6%/+7%` 的平价集合竞价不会仅因此被误认成 ST 涨停。普通无波动分钟和收盘集合竞价单一价不自动等同涨跌停，仍可按容量成交；这个近似不完整模拟历史 ST、特殊上市期或真实排队。

## 年度协议与收益判断

| 留出评价年 | 外层训练年 | 内部选参年 | 核心拟合年 | 解释 |
| --- | --- | --- | --- | --- |
| 2023 | 2024、2025、2026 | 2026 | 2024、2025 | 回顾式留一年，训练包含未来年份 |
| 2024 | 2023、2025、2026 | 2026 | 2023、2025 | 回顾式留一年，训练包含未来年份 |
| 2025 | 2023、2024、2026 | 2026 | 2023、2024 | 回顾式留一年，训练包含未来年份 |
| 2026 | 2023、2024、2025 | 2025 | 2023、2024 | 训练时间早于评价期 |

v4 在完整历史日历和构建市场特征时的完整盘中截面上追踪来源，先完成周度/盘中增强与来源标记，再筛选股票池、拆分年份。周模型的 `feature_source_start` 回溯本股之前最多 4 个实际有观测周的最早周一起点，覆盖 `shift(1).rolling(4)`；停牌缺周不占窗口，不能用固定 28 日替代。盘中模型从本股前一个实际观测周起点、前一个实际观测日和当前日取最早者，再纳入同刻有限市场日收益贡献者的前收来源。有限市场贡献若缺少可追溯前收日，受影响时刻的来源记为未知，不允许训练。

每折要求最早来源、信号、标签起点与终点属于允许的训练年，周模型还单独检查 `feature_week`。从 `feature_source_start` 至 `label_end` 的完整区间与任何排除年及其前后 7 个日历日相交时，整行清除；未知来源或来源晚于信号的行也清除。内部拟合同时排除外层年与内部选参年，最终重拟合排除外层年。覆盖率、标签相关性、特征去重、目标排名和拟合只在这之后的合格训练行上完成。整段区间检查较为保守，即使中间有停牌空白也可能清除该行。

累计 `f_history_days` 的来源单列为 `history_metadata_source_start`，不会被错误地包含在有限窗口来源证明中。v4 从模型候选列中排除它，但不改原始值，仍用已知观测历史判断 `min_history_days` 资格。因此严格隔离的对象是模型量价输入与标签；历史资格元数据保留全历史计数用途。新增来源列不以 `f_` 开头，不进入模型。模型记录实际拟合行的 `feature_source_sha256`，配合特征列表、样本哈希和清除原因审计复核。

预测时仍可使用信号当时已知的过去行情，包括评价年内较早的行情；不把训练排除规则用于删除正常评价行，也不因未来标签或恢复价格缺失而删推断样本。严格来源筛选修复跨留出年模型输入边界，但 2023–2025 含后续训练年的折仍然是回顾性评价。

T 训练样本来自符合过去周价格/流动性/历史资格的股票，不用已拟合周模型挑出的训练期前 100 删样本。内部验证时才用核心周模型的样本外前 100 选股，比较 `3 个 T profile × 6 个回报阈值 × 3 个方向 × 3 个库存比例 = 162` 个预声明候选。代理收益计入预估双边费用/滑点，未来恢复缺失按保守损失处理；默认只据此确定评价顺序，不删除候选。`candidate_shortlist_size: 0` 让全部候选进入完整账户评价；事前设为正数时才只评价代理排序前 N 个。

候选按整张请求表计算 `decision_sha256`，包括 `datetime,symbol,action,fraction,priority,date`，只有全部内容相同才复用账户模拟；不同预测优先级可能改变资金分配，不能忽略。`validation_rule_metrics.csv` 仍保留各配置，并用 `same_decisions_as_candidate` 指向原始模拟。独立模拟通过 spawn 并行，结果按候选输入顺序汇总；单进程与多进程应给出相同指标和审计。进程或执行校验异常会终止评价，不当作零收益候选。

每次独立模拟的做 T 受阻单和库存失败行写入 `validation_candidate_audits/`，便于分清收益损失、成交失败和库存比例的影响。只有净增量为正且库存恢复、共同底仓和 T+1 约束合格者可以被选中，否则关闭 T。规则在外层评价前冻结，最终重拟合后不按外层分数分布或收益调整。v4 仍使用原有 162 个训练内候选和佣金、印花税、过户费、滑点参数；历史外层结果已查看，严格隔离及执行改进也属于研究过程，不能消除重复使用这些历史数据的影响。

收益判断以相同周度底仓的净值差为准，单列金额、相对初始资金的百分点和周净增量区间。`goal_evaluation.json` 分别判断两个方面：做 T 净增益要求四折净增量金额合计为正、严格多数留出年增量为正、汇总周增量 bootstrap 的 95% 区间下界为正，并且没有 T+1、库存恢复或共同底仓差异；周度完整执行要求实际账户和基准都没有周末退出失败或到期未清仓持仓。完整目标必须两者同时通过。

内部 T 选参只按做 T 恢复及共同底仓约束过滤候选；基准自身也会遭遇的周末退出失败独立报告，不把它当成 T 模型可以解决的问题。没有达到完整标准时应明确报告未达到，不能把基准上涨、少数正收益年份或主动关闭 T 的零增量说成做 T 有效。各年度账户分别重置，合计金额用于比较这些年度实验，不是连续复利路径，也不表示训练数据与评价年完全独立。

## v4 最终结果与旧协议参考

v4 完整四折已完成，**尚未达到预声明盈利及完整执行目标**。结果见 [v4 主报告](../reports/weekly_t_cross_validation/run_20260908_212722_188543/final_report.md)、[年度明细](../reports/weekly_t_cross_validation/run_20260908_212722_188543/annual_summary.csv)及[收益曲线](../reports/weekly_t_cross_validation/run_20260908_212722_188543/test_return_curve.png)。以下收益均已计入配置中的佣金、税费和方向滑点，每折分别从 100 万元起步。

| 留出年 | 仅周度持有收益 | 加入 T 收益 | T 净增量（元） | 完整 T 配对数 |
| --- | ---: | ---: | ---: | ---: |
| 2023 | -11.6656% | -11.5339% | +1,316.39 | 17 |
| 2024 | -16.4992% | -17.7295% | -12,303.25 | 151 |
| 2025 | +31.4697% | +26.0217% | -54,479.57 | 115 |
| 2026（至 04-30） | +8.2277% | +8.2277% | 0.00 | 0 |

四折净增量合计 **-65,466.42 元**，仅 **1/4 年**为正；汇总周增量 bootstrap 的 95% 区间为 **[-205,549.22，+17,921.48] 元**，下界未大于零。2026 没有实际 T 成交，零增量不能作为有效做 T 的证据。[目标评价](../reports/weekly_t_cross_validation/run_20260908_212722_188543/goal_evaluation.json)中的做 T 盈利与周度完整执行两项均为 `false`。

实际估值范围为 2023-02-13 至 12-29（218 日）、2024-01-02 至 12-31（242 日）、2025-01-06 至 12-31（241 日）、2026-01-05 至 04-30（77 日）；不是四个完整自然年。合计 11 个年度交易日存在日初/日末股数偏差、6 日未能完成当日 T 恢复、171 日周度退出受阻，各项计数可能重叠。期末 T 均已配对，也不能抹去中途恢复失败；2025、2026 期末仍分别有 3、4 批周度底仓未退出。

108 项验证用例和 [v4 小样本四折烟测](../reports/weekly_t_cross_validation_smoke/run_20260908_212630_916413/)通过。[独立执行审计](../reports/weekly_t_cross_validation/run_20260908_212722_188543/independent_execution_audit.md)核对成交、现金、费用、T+1 及估值；[全部四折严格来源审计](../reports/weekly_t_cross_validation/run_20260908_212722_188543/independent_strict_feature_audit_full.md)逐行重建 24 个模型的训练集合，训练数据与来源 SHA256 均匹配，真实来源至标签区间与排除年及隔离窗口无交集，10 份冻结源码哈希一致。[完整机器审计](../reports/weekly_t_cross_validation/run_20260908_212722_188543/independent_strict_feature_audit_full.json)保留核对细节。这些通过证明核对范围内实现和记录一致，**不表示策略盈利或每天恢复股数成功**。

[2024–2025 损失归因](../reports/weekly_t_cross_validation/run_20260908_212722_188543/t_loss_attribution_2024.md)进一步解释了主要亏损：2024 的 33 个跨日配对合计 -12,276.85 元，包含尾盘前已经走错方向及之后恢复延迟的价格风险；2025 主要是先卖后价格上涨，110 个当日完成配对仍亏 -48,862.47 元。滑点已计入执行价，没有再次从净利扣除。归因仅用于解释已发生结果，不据此事后删日期、删股票或重新选规则。

v2/v3 只检查训练信号、标签日期及当周 `feature_week`，7 日隔离没有追溯更早滚动来源。例如留出 2024 年时，2025 年 1 月越过隔离后的训练行仍可能通过 `shift(1).rolling(4)` 使用 2024 年 12 月行情，停牌缺周时回溯可能更长。累计 `f_history_days` 也仍是这两版的候选模型输入。旧元数据中的 `embargo_limitation` 虽已披露更长历史不重置，但不等于实现完整年份排除。

这些行情在对应信号时刻已知，仍属于被留出年的数据；**v2/v3 不满足模型量价训练输入完整排除留出年的严格要求，旧日期检查通过不能证明严格年度隔离。** 两版原运行及源码快照保留，不能把 v4 修复追溯描述为旧运行已使用。

旧版完整运行见 [v2 报告](../reports/weekly_t_cross_validation/run_20260908_203530_622139/final_report.md)及[年度明细](../reports/weekly_t_cross_validation/run_20260908_203530_622139/annual_summary.csv)。下表为计入既定费用和滑点后的账户净值增量，受上述滚动特征跨留出年边界限制，且尚有执行失败；它既不是严格独立测试结果，也不表示每天都恢复成功。

| 留出年 | 加入 T 相对基准净增量（元） |
| --- | ---: |
| 2023 | -145.39 |
| 2024 | -4,874.60 |
| 2025 | +888.84 |
| 2026（至 04-30） | 0.00 |

四折各自重置资金，净增量合计 **-4,131.16 元**，仅 1/4 年为正，汇总周增量 bootstrap 的 95% 区间为 **[-28,177.10，+17,110.83] 元**；T 库存与周末退出均有失败，[预声明目标未达到](../reports/weekly_t_cross_validation/run_20260908_203530_622139/goal_evaluation.json)。这些旧结果保留，不通过新版覆盖。

[v3 旧协议运行目录](../reports/weekly_t_cross_validation/run_20260908_211507_784203/)保留分钟分批恢复等工程改进的参考结果；其滚动来源隔离缺口仍然存在，不作为 v4 严格协议的收益证据。

## 输出与限制

| 文件或目录 | 内容与后续用途 |
| --- | --- |
| 根 `final_report.md` / `输出文件说明.md` | 主结论及产物入口 |
| `annual_summary.csv` / `test_metrics.csv` | 各年度持有/加 T 收益、净增量、风险与约束统计 |
| `goal_evaluation.json` / `research_protocol.json` | 目标评价、本次评价前声明的候选、标准、年度协议、严格来源开关、排除模型列及旧外层结果已查看标记 |
| `split_audit.csv` / `year_年份/split.json` | 内外层年份及时间顺序标记 |
| `resolved_config.json` / `data_manifest.json` | 实际配置、数据范围、基础缓存指纹及本次补充后的 `runtime_*` 特征列表/配置 |
| `recovery_manifest.json` | 启用真实分钟恢复时生成；独立恢复缓存路径、源数据及代码指纹、行数和数据质量 |
| `feature_provenance_manifest.json` | 严格来源追踪规则、输入范围、未知来源行数、必须排除的模型列及原特征未改变标记；正式流程生成 |
| `source_snapshot/` / `source_hashes.json` | 运行时源码副本及哈希 |
| `test_return_curve.png` / `测试集收益曲线.png` | 各年度独立收益和做 T 增量图 |
| `year_年份/weekly_predictions.csv` / `weekly_selections.csv` | 全候选双模型预测、排名及选中前 100 |
| `year_年份/weekly_prediction_metrics.csv` | 规则冻结后的周收益/振幅 Rank IC、前 100 标签均值和实际可评价股票数；不用于再次选参 |
| `year_年份/validation_weekly_selections.csv` | 内部验证年周度选股 |
| `year_年份/t_predictions.csv` / `t_decisions.csv` | 留出年盘中预测、通过当前方向过滤的最早触发请求及同刻优先级 |
| `year_年份/validation_proxy_candidates.csv` / `validation_rule_metrics.csv` | 全部内部候选代理排序、完整引擎指标、请求哈希及相同请求复用指针 |
| `year_年份/validation_candidate_audits/candidate_编号_t_attempts.csv` | 原始独立模拟的做 T 受阻或延迟委托；重复候选通过指标表指针查阅 |
| `year_年份/validation_candidate_audits/candidate_编号_t_inventory_failures.csv` | 同一次模拟的每日库存偏差或恢复失败行 |
| `year_年份/selected_rule.json` | 外层运行前冻结的 profile、阈值、方向、库存比例和是否启用 |
| `year_年份/weekly_entry_plans.csv` | 基准生成的周度预算、请求股数和实际底仓计划 |
| `year_年份/base_plan_mismatches.csv` | 做 T 账户偏离基准底仓订单的原因和差异 |
| `test_equity_curve.csv` | 每日真实账户及基准净值，根表带 `test_year` |
| `test_trade_events.csv` / `baseline_trade_events.csv` | 实际账户与独立基准的成交、费用、可卖量 |
| `test_round_trips.csv` / `test_open_positions.csv` | 已成交恢复片段，以及期末实际库存、滞留周度计划；同一 `pair_id` 可有多个 `fragment_id` |
| `daily_inventory_audit.csv` / `blocked_orders.csv` / `execution_audit.json` | 每日数量、失败委托、现金、整数手及可卖预算 |
| `weekly_increment.csv` | 每周账户净值差变化，用于配对收益分析和 bootstrap |
| `inner_weekly_model/` / `inner_t_profile/` | 核心年份模型，仅供内部验证 |
| `weekly_model/` / `t_model/` | 全部外层训练年份重拟合模型 |
| 各模型目录 `model_bundle.joblib` / `return_model.txt` / `amplitude_model.txt` | Python bundle 与 LightGBM 原生权重；振幅模型仅周模型生成 |
| 各模型目录 `model_metadata.json` / `resolved_config.json` | 模型版本、实际超参、年份、严格来源范围、`feature_source_sha256`、训练样本指纹、标签定义及配置 |
| 各模型目录 `training_split_audit.csv` / `feature_selection_audit.csv` | 标签清除/边界原因计数和特征筛选；清除原因可以重叠 |
| 各模型目录 `feature_list.txt` / `used_feature_manifest.csv` | 选中 `f_*` 的名称、顺序及训练依据 |

分批记录中的 `original_pair_shares` 是最初首腿股数，`remaining_shares_after` 是该次恢复后的待恢复量，`pair_fully_recovered` 标记该片段是否完成整个配对。`t_pair_count` / `completed_pair_count` 只计完整恢复的配对，`fragment_count` / `t_fragment_count` 计恢复片段；胜率按完整配对的合计净收益计算。片段净收益合计可能包含尚未全部恢复配对中的已恢复部分，账户净值差还包含未恢复部分的市值风险，不能互相替代。

2026 数据仅到 04-30，该年评价不是完整年度。`known_complete_end_date` 是事先给定的日历确认；不能机械地把周四当作不完整周。2023–2025 的折包含未来年份训练，只是回顾式迁移检验；2026 虽按时间先训练后评价，但前轮外层结果已经查看，v4 仍是研究复测，不能因修复历史来源隔离而重新称为全新未见样本。周 bootstrap 假设周样本可独立重抽样，没有完全处理跨周相关，也不校正候选搜索和历次研究的选择偏差。进一步确认效果需要未参与当前研究的新数据。

来源追踪覆盖当前已实现的特征定义；新增更长窗口、跨股特征或累计量价特征时，必须同步扩展依赖来源并补充验证，不能沿用旧来源边界。完整基础日历和构建同期市场特征时的完整截面必须先于股票池及年度筛选提供；提前裁剪会丢失真实历史来源。累计历史资格仍使用当时已知全历史计数，不能把模型列排除写成所有资格元数据都已重新从零计算。

未完整实现历史 ST 状态、特殊限价、盘口排队、复权调整和公司行动现金分红。缺少行情时不能成交，估值可沿用最近有效价格并标记陈旧；不能把陈旧净值当作真实可变现价格。正式收益必须与每日股数偏差、挂起配对、周末未清仓及共同底仓差异一起读取。

## v8 已完成：同刻队列残差目标

针对四折 T 净增量均为 0 的进一步核查见 [零增量原因与开关对照诊断](../reports/weekly_t_zero_increment_diagnosis/final_report.md)：原结果是统计/样本/执行联合门槛导致四折均关闭 T，不能解读为主动交易的收益恰为 0。2026 年保持原模型和预测、仅打开开关，实际完成 9 对、扣费净增量 −32.74 元，同时出现 2 个日股数偏差日，仍不满足每日恢复要求。此为独立故障诊断，原正式模型、配置和结果未改写。

[市场残差做 T 验证说明](市场残差做T验证说明.md)记录保持原周池、70/30、费用和每折 27 个候选，仅将 T 训练目标改为严格核心队列同刻收益残差的研究。推断采用固定零队列共同项先验，实际盈亏仍按股票两腿成交计算。[正式四折结果](../reports/weekly_t_cross_validation/run_20260908_235818_404243/final_report.md)已完成：108 个内部候选记录有 27 正、64 负、17 零，但均未通过收益置信区间、样本支持和库存恢复等联合门槛，四折全部关闭 T、净增量为 0，未达盈利目标。13 项新增测试及完整独立模型/执行审计通过；[审计完成核验](../reports/weekly_t_cross_validation/run_20260908_235818_404243/independent_audit_completion_verification.md)同时记录周底仓仍有 171 个退出失败年度日，不能将审计通过理解为盈利或退出无风险。此前 v7 及其他版本的结果保留，v8 仍是已接触历史上的研究复测。


## v10 已完成：增加做 T 频率及实际成本

新增 [做T频率改进说明](做T频率改进说明.md)，入口为 `scripts/run_weekly_t_frequency.py`、配置为 `config/weekly_t_cross_validation_v10.yaml`。每折25个固定候选在内部真实账户中比较，执行合格者按封顶的每可计划日配对数优先、再按扣费净增量排序；保持原周池、核心权重、T比例和实际费用，允许负内部收益，全部规则先冻结再评价外层。

[正式四折结果](../reports/weekly_t_cross_validation/run_20260909_135906_004933/final_report.md)完成1760组（435/574/603/148），是v9的34组的51.76倍；450个可计划日中445日活跃，平均3.911组/可计划日。以全部778个估值日计算为2.262组/日，不能写成全部交易日每天3—5组。参考价毛差仅352元，滑点3337.41元和T手续费7305.97元后，净增量为 **−10,291.38元**，原v9为+276.31元。频率增加，但未实现盈利。

本轮T库存差异、恢复失败、期末待恢复、T+1违规和底仓计划差异均为0，四年共同基准逐日完全相同；周底仓退出仍失败171个年度日/307个股票日，保留2025年末3仓、2026评估期末（4月30日）4仓及9750元陈旧估值。234输入/27运行源码身份核对和48项工程测试通过；[正式独立执行审计](../reports/weekly_t_cross_validation/run_20260909_135906_004933/independent_execution_audit.md)及[保存账表复核](../reports/weekly_t_cross_validation/run_20260909_135906_004933/independent_frequency_review/review.md)均完成通过，覆盖100内层候选和四年原始行情核账，不能将审计通过当作盈利或周退出成功。详细[年度统计](../reports/weekly_t_cross_validation/run_20260909_135906_004933/annual_summary.csv)、[v9对照](../reports/weekly_t_cross_validation/run_20260909_135906_004933/frequency_comparison_v9.csv)和[收益图](../reports/weekly_t_cross_validation/run_20260909_135906_004933/test_return_curve.png)保留真实代价；本轮仍属于历史复测。
