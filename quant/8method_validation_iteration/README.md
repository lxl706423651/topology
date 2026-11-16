# CTA逐验证集持续迭代

## 架构与目标

目标保持为：同一套交易规则在所有固定验证区间的扣费日收益夏普都严格大于2。
不删除弱年份、不缩短区间、不用平均值代替，不为每个年份事后选不同赢家。
第一阶段复用第七方法已经审计的过去训练模型及724列特征，扩展开仓/市场条件；
第二阶段进一步比较较低模型门槛和开仓频率，已完成全部候选、标准回放及独立审计。

**本轮结果：[三个原协议内部验证区间均达到夏普>2](results.md)，分别为3.2386、2.5242、2.4170。**
这是优化过的验证结果，不是2026测试结果，也不是实盘通过证明。

固定验证区间为2023年第三季度、2024全年、2025全年，源自第七方法原协议。
模型分别在各区间之前拟合；2023—2025验证期间正在用于选择策略，必须称为优化过的验证结果。
2026仍不重新预测或回测；即使验证全部超过2，也不证明独立测试或未来实盘达到2。
每折至少20个完整平仓批次且日波动非零，以避免把全现金或极少交易当成功。

## 文件角色

| 文件 | 用途 |
|---|---|
| README.md | 架构、字段、调用与边界 |
| results.md | 生成后的人工说明：两阶段结果、共同策略、训练来源、逐笔复盘与验收限制 |
| ../config/cta_validation_search.yaml | 用户输入：过去模型来源、候选空间及逐折目标 |
| ../config/cta_validation_search_lower_gates.yaml | 用户输入：第二阶段较低门槛及24/48/120/240分钟检查候选；本轮达标规则来源 |
| ../config/cta_validation_search_timing.yaml | 未执行的第三阶段预案，不可传给现有搜索入口；相关字段见下文 |
| ../quant_backtest/cta_validation_search.py | 预测PKL、共同规则、原Numba执行内核快速指标、标准回放 |
| ../quant_backtest/cta_timing_rules.py | 第三阶段预案的时钟辅助库；只通过单元测试，未用于本轮选优 |
| ../scripts/run_cta_validation_search.py | 可执行入口；新增输出，不改旧研究模型/报告 |
| ../tests/test_cta_validation_search.py | 全折不可删、严格门槛、标准与快速执行一致性、信号接口测试 |
| ../tests/test_cta_timing_rules.py | 预案时钟/午休/索引对齐测试，不代表第三阶段收益实验 |
| ../scripts/audit_cta_validation_search.py | 独立重算所有候选逐折门槛、过去模型预测及所选三折账本 |
| ../scripts/verify_cta_validation_completion.py | 可执行只读验收入口；对比旧检查点、复算所选三折与输出中文批次表，不修改交易规则 |
| ../reports/cta_validation_prediction_cache/ | 生成的仅2023—2025验证预测PKL及输入指纹，不是新用户输入 |
| ../reports/cta_validation_search/run_*/ | 每阶段全部候选、三张曲线、逐笔记录、统一规则与验证结论 |
| ../reports/cta_validation_search_lower_gates/run_20260910_135032_887749/ | 第二阶段最终报告、模型来源、三折曲线/逐笔记录、审计及engineering_validation工程证据 |

## 调用流程

在quant根目录：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/run_cta_validation_search.py --prepare-only
/home/lxl/anaconda3/envs/quant/bin/python scripts/run_cta_validation_search.py --prepare-only
/home/lxl/anaconda3/envs/quant/bin/python -m pytest -q tests/test_cta_validation_search.py tests/test_cta_timing_rules.py tests/test_cta_expanded.py tests/test_cta_research.py
/home/lxl/anaconda3/envs/quant/bin/python scripts/run_cta_validation_search.py --smoke
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_cta_validation_search.py
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_cta_validation_search.py --config config/cta_validation_search_lower_gates.yaml
/home/lxl/anaconda3/envs/quant/bin/python scripts/audit_cta_validation_search.py --report-dir reports/cta_validation_search_lower_gates/run_20260910_135032_887749
```

1. 读取过去模型原生JSON及元数据，验证训练特征、标签截止均早于所属验证期；仅生成三个验证区间预测，不读取2026旧预测。
2. 第二次调用复用预测PKL。之后改交易候选无需重算原因子，也不必反复做模型预测。
3. 加速入口和完整公开回测入口逐折核对后才开始正式枚举。加速只省去重复报告包装，仍调用同一个真实账户内核。
4. 所有候选在全部三折运行，完整保存失败项；按最差折夏普排序，以每折>2判定，而非平均值。
5. 所选共同规则再走标准回测、独立现金/T+1/费用/仓位核对，并生成每折曲线和逐笔记录。
6. 未达标就保留阶段证据并继续后续研究，不能将阶段完成写成目标完成。

完成阶段后运行`scripts/audit_cta_validation_search.py --report-dir 阶段目录`，
重新从原特征和过去模型计算验证预测，再核验统一规则、现金、费用、T+1和日夏普。
根目录兼容test_文件仅复制最弱验证折，不是2026测试；总报告始终列出全部三个验证区间。

### 性能优化说明

搜索首轮遇到长期不可卖批次时，原撮合内核仍逐分钟回溯历史报价。
仅增加已知不可卖的提前拒绝分支，返回与calculate_fill原本相同的原因1；
T+1优先次序、费用、止损、可成交时的旧价引用均未改变。旧运行检查点保留，
新运行仍评估完全相同的2000规则；旧640套规则的1920个账户与新运行全部指标逐值一致，
不能借优化更换研究范围。中断轮保留在run_20260910_133954_798660，不视为完整阶段。

## 交易合同与首阶段候选

周池仍是上一周高波动前300，实际范围为当前已验证历史状态的755只股票中722只曾入池，
创业板/科创板各3只。股票允许扩展到全部非ST沪深股票，但缺失历史ST状态不能自动当成非ST；
本阶段不声称已覆盖所有股票，不访问被拒绝的数据接口。

前5开仓、每批总资产10%、单股20%、总仓80%、T+1、下一分钟执行、佣金万八免五、
历史ST/异常状态及涨跌停阻挡、分钟成交容量限制全部沿用审计过的执行层。
平仓继续独立管理初始ATR止损、只上移吊顶保护、到期与可选趋势退出，不因掉出目标池平仓。

首阶段模型为净收益与盈利概率，各有两种树复杂度、两种退出标签。
开仓类型：trend趋势、breakout突破、pullback回调、recovery中期趋势中的短期回撤转涨、
model_only仅模型门控。上下文过滤：none无附加条件、breadth池内上涨占比>55%且当日均涨幅>0、
market_trend池内20日动量及当日均涨幅>0、quiet个股ATR/价格在0.5%—3.5%、
avoid_chase当日涨幅在±2.5%且相对EMA5偏离<3%。这些均只用当时已完成信息，不按事后牛熊日期人工开关。

首阶段固定预算10%不随波动改变；quiet只是开仓过滤，不能把它称为完整波动率目标仓位模型。
开仓频率24/120指检查，不保证每次成交；概率未校准，不能视为保证胜率。

## YAML字段

全部为必填，默认值见随附YAML，统一由run_cta_validation_search.py消费：

| 字段 | 默认与含义 |
|---|---|
| source_report | 第七方法run_20260910_122142_766807，只读取internal过去模型及数据合同 |
| prediction_cache / output_dir | 验证预测缓存及新阶段报告根目录 |
| sharpe_target | 2.0，每折严格大于，不是大于等于 |
| minimum_closed_lots_per_fold | 20，每个验证区间至少完整平仓20批 |
| objectives / profiles / exit_ids | return/probability；shallow/regularized；0/1 |
| styles | trend,breakout,pullback,recovery,model_only |
| frequencies | [24,120]交易分钟 |
| contexts | none,breadth,market_trend,quiet,avoid_chase |
| return_gates | [-0.003,0,0.003,0.006,0.01]，净收益小数门槛 |
| probability_gates | [0.40,0.45,0.50,0.55,0.60]，原始模型概率输出门槛 |
| workers | 12，预测线程数 |
| save_every | 20，每20套共同规则落盘全部已完成指标 |

默认首阶段2000套共同规则、6000个实际账户。增加候选次数会增加验证过拟合风险，
所有尝试都必须保留，不能把最大验证夏普当未来收益承诺。

第二阶段使用相同字段，差异是：`styles=[breakout,trend]`（突破/趋势）、
`frequencies=[24,48,120,240]`、`return_gates=[0]`、
`probability_gates=[0,0.1,0.2,0.3,0.325,0.35,0.375,0.4]`、独立`output_dir`。
共1440套规则、4320个账户。因第一阶段最优门槛落在0.4边界而扩展，不依据2026表现决定。
搜索共3440个候选槽位，其中一些规则重复；不能将10320个账户当作独立样本量。

### 未执行的第三阶段预案

第二阶段尚未结束时曾准备固定日内时点预案；第二阶段达标后没有执行它，也没有生成第三阶段模型或收益。
辅助库及测试保留，不能把它当成获胜规则的组成部分。没有单独的第三阶段执行器。
预案所有字段均为必填且无执行器默认值：

| 字段 | 预案值与消费者 |
|---|---|
| prediction_config | 第一阶段YAML路径；预定给未来执行器，当前无消费者 |
| output_dir | reports/cta_validation_search_timing；预定给未来执行器，当前无消费者 |
| timing_model_names | 两种复杂度、两套退出的四个概率模型；create_timing_rules读取 |
| styles | [breakout,trend]；create_timing_rules读取 |
| offsets | [0,24,48,72,96,120,144,168,192,216]交易分钟；create_timing_rules读取 |
| probability_gates | [0.30,0.325,0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.40,0.425]；create_timing_rules读取 |
| sharpe_target / minimum_closed_lots_per_fold | 2.0/20；预定给未来执行器，当前无消费者 |
| workers / save_every | 12/20；预定给未来执行器，当前无消费者 |

预案函数固定`frequency=240`、`context=none`，仅在原24分钟采样网格上筛选时点，不移动特征时间。

## 调研参考

[AQR时间序列动量原始研究数据说明](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly)
说明其原研究为多资产期货/远期的多空动量。这里只借鉴历史趋势信息，不移植其业绩或做空假设。
[Moreira与Muir的波动管理研究](https://www.nber.org/papers/w22208)
提供风险状态管理方向；本阶段仅测试波动过滤，不宣称复现该论文，正文访问返回403未绕过。
