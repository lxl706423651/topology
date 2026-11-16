# 同日14:55参考标签的只读可行性备忘

原缓存已经保存“信号下一分钟 close → 同日14:55 close”的真实参考端点，可在后续独立研究中复用。无需改造冻结的 v14 源码、重新生成因子或重建完整分钟恢复缓存。本备忘只确认数据与接口可行性，不实施下一轮、不拟合、不读取外层收益，也不判断较长持有是否盈利。

**核对证据。** 原聚合缓存 [intraday.pkl](/data/lxl/seed-emulator/topology/quant/reports/weekly_t_cross_validation/cache/weekly_t_9b6b9bfb3309dd53/intraday.pkl) 的 SHA256 为 `48839f6681133068bf18740f8f1b5280c5866db003f3996b0f0587612d4fcc8b`，与此前 v11 已审计输入清单一致。metadata 声明20,454,854个信号样本和“同日14:55bar close / 信号下一分钟bar close - 1”合同。

按 metadata 顺序固定取前8个股票代码，仅核2024、2025核心年份：31,040行的执行时刻、恢复时刻、标签区间和价格比均一致；两个端点在本样本中都没有缺价。再独立读取 SH.600000 两个核心年的原始CSV，对下一分钟及14:55的 close/high/low/volume/amount 共38,800个字段进行核对，最大误差0、缺失模式一致。完整证据见 [checks.json](checks.json) 和 [sample_counts.csv](sample_counts.csv)。这不是全市场缺失率普查，分股样本也不代替全市场逐价审计。

样本中有21行在下一分钟成交量为0时仍存在有限参考标签。这直接说明：**有价格标签不保证可以开仓或恢复，也不保证实际赚到该价差。** 正式引擎仍必须按真实量额、现金、旧股预算、方向限价代理和部分恢复逐笔验收。

**原数据合同与缺失处理。** [weekly_t_data.py:203](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_data.py:203) 用精确时间 `reindex` 读取下一分钟和当日14:55；缺行保留 NaN，不寻找后续可用报价。当前信号仅按已观察到的信号 close 选取，未来端点不参与当前样本保留。`target_return = recovery_price / execution_price - 1`，价格及收益为 float64；只有 `f_*` 转为 float32。

原聚合表还在 [weekly_t_data.py:279](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_data.py:279) 根据旧“完整交易周”规则清空部分 target_return，端点价格仍保留。此次缓存 metadata 的最后周完整标记为真，但永久底仓的新标签合同仍不应继承周退出资格。应从精确端点重新计算价格比，并明确记录这种标签语义；不能覆盖原PKL。缺端点只在后置训练标签有效性检查中排除，不能删除冻结持仓机会或推断行。

当前8个信号时钟都可直接用 `signal + 1 minute`，不落午休。若以后扩时钟，必须统一改用下一合法交易分钟；不能把11:30的下一分钟写成11:31。

|当前信号|首腿参考时刻|14:55距离首腿的交易分钟数|
|---|---|---:|
|09:35|09:36|229|
|09:45|09:46|219|
|10:00|10:01|204|
|10:30|10:31|174|
|11:00|11:01|144|
|13:05|13:06|109|
|13:30|13:31|84|
|14:00|14:01|54|

因此14:55是**固定钟点目标**，不是固定长度 horizon，也不是15:00收盘价；最后一个信号的持有长度甚至短于60分钟。

**两种方案的实际区别。** 以下都只是待决的独立研究设计，不是当前运行变更。

|项目|仅14:55目标（close-only）|20/40/60再加14:55目标|
|---|---|---|
|每折模型|2个profile各1个参考收益模型，共2个|2个profile各4个目标，共8个；当前为6个|
|每个当前键的期限选择|不做跨期限选择，期限始终14:55|比较4个预测净边际，按当前预测选期限|
|仍存在的选择|同一期限的交易方向，以及内部37个候选|交易方向、逐行期限，以及内部37个候选|
|检验的变化|永久底仓队列下改用统一尾盘目标|扩大期限备选，并改变逐行选择机会|
|误差解释|减少跨期限取最大值这一来源，仍有模型误差和候选选择误差|多次预测取最大值可能放大选择误差，增加模型不等于增加实际优势|

若只替换目标接口，2 profile × 3净边际 × 2旧股比例 × 3退出模式再加hold，仍为37候选。close-only也不等于所有实际T必须持有至14:55：`profit_buffer` / `profit_buffer_stop` 仍可能提前退出；只有 `fixed` 以14:55为首次硬恢复时刻。实际恢复也可能因容量等原因晚于14:55甚至跨日。因此两个方案的标签都只是参考端点目标，不能改名为完整组合实际利润。

**最小独立接入点。** 若下一轮获得实施决定，建议新增一份独立 close 标签/模型模块、一份 close 政策模块、一份 runner、一份配置及相应测试，旧模块保持原样；报告可以复用已有指标生成逻辑，但另写本轮目标和时钟说明。此处未创建这些生产文件。

1. **加载后立即分离标签。** 新 runner 可复用 [run_inventory_learning_t.py:234](/data/lxl/seed-emulator/topology/quant/scripts/run_inventory_learning_t.py:234) 的全市场加载和全截面当前因子增强。在进入永久底仓队列之前，从已加载的原 intraday 单独取 `datetime,symbol,execution_time,execution_price,recovery_time,recovery_price`，验证唯一键和端点身份，计算标签。记录原聚合缓存、metadata、提取表的SHA，不重新读全市场原CSV或改缓存。
2. **先冻结当前队列，再左联。** 直接复用 `build_inventory_core_cohort` 的入场前来源purge、无T永久底仓规划和全部旧股持有日；其 [未来字段剥离](/data/lxl/seed-emulator/topology/quant/quant_backtest/inventory_t_learning_data.py:186) 保持不变。新增 `attach_close_labels` 在核心机会冻结之后一对一左联，保持所有键；缺失不补价，NaT当前来源也继续由严格训练purge处理。标签不得进入 `f_*`。
3. **明确目标身份。** 使用例如 `recovery_kind='fixed_clock'`、`recovery_clock='14:55'`、`recovery_key='close'`。不要伪装成恒定整数 `horizon_minutes=0/240`；如需展示每行实际时长，另存由当前日历计算的 `planned_holding_minutes`。旧 [load_labels_only](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_timing_data.py:124) 将配置 horizons 与旧20/40/60缓存严格匹配，close-only无需调用它；不能为了接入close修改旧horizon缓存身份。增加第四期限的方案则保留旧三期限加载，另附独立close表。
4. **新模型只替换目标及时间校验。** 复用严格 `select_training_rows`、来源隔离、特征选择、确定性抽样、原LightGBM参数、训练矩阵/SHA/原生重载流程。当前 [模型校验](/data/lxl/seed-emulator/topology/quant/quant_backtest/inventory_t_learning_models.py:87) 绑定整数horizon，需要独立版本验证 `label_start=下一合法交易分钟`、`label_end=recovery_time=同日14:55`、`execution_time=label_start`、首腿早于尾盘目标、正有限价格和价格比。全体来源→信号→标签仍须位于合法核心年份并避开 excluded ±7日；人工加仓继续不作为训练历史。不能把相互依赖的两个OOF核心年再次当成独立校准折。
5. **政策只从当前时钟决定期限。** close-only要求两profile各自覆盖相同当前机会，不再要求同时提供三个整数horizon预测。当前门控、方向限制、正净边际和共享现金优先级保持原定义；`planned_recovery_time` 从当前日期生成14:55，不能读取未来价格或恢复是否成功。增加第四期限则要验证四者当前键一致，并按真实计划时刻或实际时长稳定处理并列，不可按事后最优端点挑期限。
6. **实际账户仍用现有动态引擎。** [dynamic_t_engine.py:301](/data/lxl/seed-emulator/topology/quant/quant_backtest/dynamic_t_engine.py:301) 已接受14:55硬期限，在14:54发出期限意图、14:55起真实尝试；缺当前报价也不取消硬期限。现有完整240分钟报价及旧恢复缓存均包含14:55，无需因这个目标重建行情或改引擎。T+1、单股待恢复义务、旧股卖额、共享容量、费用、共同基准估值、失败及期末残仓全部保留。

现有 [predicted_pair_edges](/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_short_horizon.py:78) 在最低佣金为0时可以原样将同日价格比转换成首腿参考金额归一化的买先/卖先净边际。它不是实际组合收益：真实数量、现金竞争、拒单、滑点成交价、部分恢复和跨日费用都由账户决定。若将来采用非零最低佣金，必须按实际可请求数量另建费用估计，不能把当前比例公式继续称为精确。

最低必要验证应覆盖：缺未来端点仍保留机会与推断键；标签错误日期/错误14:55/重复键被拒；没有继承 `trade_week_complete`；未来字段改写不改变当前预测和首腿；11:30下一分钟午休跳转；14:55真实拒单保留恢复义务；原生模型与保存矩阵重载一致；同一冻结底仓在两方案下逐日基准一致。所有候选先用内部数据决定并冻结，再评价外层；反复使用已有年份只能称为历史研究，不能声称未接触的新测试。

文献范围只支持提出时段假设：[Chu、Gu、Zhou（2019）的作者机构摘要](https://engagedscholarship.csuohio.edu/bus_facpub/288/) 描述中国股市盘中动量与反转的时段关系，并指出交易成本限制套利。这里只核阅摘要，没有据其选择14:55或断言更长持有会赚钱。

本次归档核对48份当前源码与正在运行的 v14 冻结清单及快照全部相同，见 [runtime_identity.json](runtime_identity.json)。检查脚本、输出和命令记录均存于本目录；没有读取外层账户或改变任何运行输入。
