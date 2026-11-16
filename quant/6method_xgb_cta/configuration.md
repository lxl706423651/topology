# YAML配置字段

输入：`config/xgb_cta_walk_forward.yaml`；本地覆盖子集使用`config/xgb_cta_local_history.yaml`。
除标注可选字段外，下表字段为
当前入口必需配置；“默认值”指仓库模板值。所有路径相对quant根目录，不要求手动export。
消费模块简称：数据=cta_research_data，训练=cta_research_training，撮合=cta_research_engine，
报告=cta_research_reporting，入口=run_xgb_cta_walk_forward。

| 字段 | 默认值 | 消费者/含义 |
|---|---|---|
| data_root | ashare_minute_data_2023_2026_04/extracted | 数据，只读原始分钟目录 |
| output_dir | reports/xgb_cta_walk_forward | 训练/报告，新运行目录的父目录 |
| cache_dir | reports/xgb_cta_cache | 数据，历史状态、特征PKL和只读报价矩阵 |
| reuse_factor_cache | reports/annual_method_validation/panel_cache/282322e813cbf3c6 | 数据，按来源及时间键复用旧原因子；也支持CTA缓存的symbols/*.pkl，不复用旧日状态/截面秩。本地配置指向本轮首次生成的3fa003ac93c17d4acdcc缓存 |
| history_data_dir | null，可选 | 数据，授权本地日线目录；非空时禁止网络补缺；撮合同时要求历史tradestatus=1 |
| history_coverage_policy | require_all，可选 | 数据，默认要求全部股票；available_local只在显式本地目录内取文件可用子集，逐日未知ST仍拒绝新买。保存入选/排除清单，不能称完整市场 |
| start_date / end_date | 2023-01-01 / 2026-04-30 | 数据/训练，分钟数据范围；结束日包含在内 |
| test_years | [2023,2024,2025,2026] | 训练，2023仅第四季度、2026仅已有月份 |
| workers | 12 | 数据/训练，进程或XGBoost线程数量 |
| download_workers | 4 | 数据，公开接口进程数；拒访时停止，不绕过限额 |
| network_timeout | 30 | 数据，单次请求秒数与硬超时 |
| max_symbols | 0 | 数据，0为所选覆盖政策内全部股票，正数为每板块调试股票数；非零不能称完整研究 |
| universe_size | 300 | 数据，上周高波动周池数量上限，不足时不虚构补足 |
| min_week_days | 3 | 数据，上周至少多少有效交易日，每日还需≥120根分钟 |
| base_signal_minutes | 24 | 数据，特征与信号的统一采样步长 |
| entry_minutes_candidates | [24,48,120,240] | 训练，内部比较的开仓检查间隔；不代表强制换仓 |
| entry_styles | [trend,breakout,pullback] | 训练，趋势延续/通道突破/趋势回撤恢复三候选 |
| net_return_gates | [0,0.003,0.006] | 训练，预测扣费收益须严格超过门槛 |
| minimum_history_days | 60 | 数据，至少已有历史交易日数量，屏蔽上市初期 |
| atr_days | 14 | 数据，上一已完成日的简单平均真实波幅窗口 |
| max_features | 154 | 训练，每模型最多入选特征数量，不固定使用旧特征清单 |
| feature_min_coverage | 0.80 | 训练，训练区间最低非缺失比例 |
| feature_max_correlation | 0.90 | 训练，与已入选特征相关性绝对值上限 |
| labels_per_symbol_day | 2 | 训练，每股每日按稳定代码/日期轮换选择多少监督时点 |
| initial_cash | 1000000 | 撮合，每个独立年度/连续账户的初始人民币资金 |
| entry_top_n | 5 | 撮合，每个信号时点最多买入候选数 |
| entry_equity_fraction | 0.10 | 撮合，每笔按决策时总资产计算的预算比例 |
| max_symbol_exposure | 0.20 | 撮合，新买入后单股累计市值上限 |
| max_total_exposure | 0.80 | 撮合，新买入后总多头市值上限，价格上涨不强制卖出 |
| commission_rate | 0.0008 | 撮合，买卖各万八佣金 |
| min_commission | 0 | 撮合，每次成交最低佣金；用户要求免5 |
| transfer_rate | 0.00001 | 撮合，双边过户费率 |
| stamp_rate_before_20230828 | 0.001 | 撮合，日期调整前的卖出印花税率 |
| stamp_rate_from_20230828 | 0.0005 | 撮合，自2023-08-28起的卖出印花税率 |
| slippage_rate | 0.0005 | 撮合，双边不利滑点比例，价格向不利方向按分取整 |
| max_bar_amount_fraction | 0.05 | 撮合，同分钟同股票全部买卖累计成交额上限 |
| minimum_signal_amount | 200000 | 数据，生成开仓候选的当前分钟最低成交额，单位元 |
| validation_min_closed_lots | 5 | 训练，内层合格所需完整批次；不足时明确诊断未晋级 |
| benchmark_symbol | sh.000300 | 报告，沪深300基准，用于市况分层 |
| regime_lookback_days | 60 | 报告，上一完成日基准收益观察窗口 |
| regime_return_threshold | 0.05 | 报告，超过±5%分别标记牛/熊趋势，其他震荡 |
| sharpe_target | 3 | 报告，连续净日收益点估计目标，不承诺达成 |
| fit_deployment_model | true | 训练，生成包含2026的全量候选，不能继承年度成绩实盘 |
| random_seed | 42 | 训练/报告，模型与块自助法固定种子 |
| exit_profiles[].name | atr2_trail3_time5 / atr3_trail4_time10 | 撮合/训练，退出方案稳定标识 |
| exit_profiles[].initial_atr | 2 / 3 | 撮合/标签，初始ATR止损倍数 |
| exit_profiles[].trailing_atr | 3 / 4 | 撮合/标签，跟踪ATR倍数，只收紧 |
| exit_profiles[].max_days | 5 / 10 | 撮合/标签，交易分钟累计达到该交易日数后申请退出 |
| exit_profiles[].trend_exit | true | 撮合/标签，跌破上一完成日EMA20时申请退出 |
| model_profiles[].name | shallow / regularized | 训练，浅层/较深强正则模型稳定标识 |
| model_profiles[].n_estimators | 180 / 220 | 训练，树的数量 |
| model_profiles[].max_depth | 3 / 5 | 训练，树深度 |
| model_profiles[].min_child_weight | 60 / 100 | 训练，叶节点最低权重约束 |
| model_profiles[].reg_lambda | 12 / 20 | 训练，L2正则强度 |

固定实现参数在源码快照和model_metadata.actual_model_parameters中保存：学习率0.04，
行/列抽样0.85，L1正则0.2，hist直方图树，平方误差目标。未经测试的新配置必须另建
实验，不得看过外层后改配置继续沿用原实验名。

CLI的`--workers`临时覆盖线程数；`--smoke`固定每板3只、最多16特征、8树、单个
候选和每日报名时点，仅验证工程流程；`--prepare-only`仅建缓存，不训练或回测。
