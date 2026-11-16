# v13动态恢复引擎工程验证

新增独立 dynamic_t_engine.py/test_dynamic_t_engine.py，原balanced/weekly引擎与缓存保持不变。run_dynamic_portfolio保留固定底仓、旧股卖额、现金/准备金、真实分片和失败；当前完成分钟锁存退出，下一交易分钟执行，预定期限前一分钟即使无报价也发出时间意图。三模式都使用完整240时钟和逐分钟重试，固定模式是新的完整覆盖对照。

原首腿参考价×实际首腿股数为固定净率分母；当前估计净額包含已恢复片段净额、剩余首腿费用份额、当前反向含滑点价格和当前反向费用。未来分片/行情不进入触发。exit_decisions每pair一条锁存行，重试和恢复片段不重复计触发组。

共同市场mark由相同完整当前报价流提供，T是否成交不增加模式专属估值；基础/人工买单仍有共同成交参考价兜底，有效日收盘最后覆盖。无仓候选不查分钟报价。无T规划仅决定现金/冻结订单；稀疏规划估值和最终full共同基准分开。

验证命令：

- /home/lxl/anaconda3/envs/quant/bin/python scripts/test_dynamic_t_engine.py
- python3 -m py_compile quant_backtest/dynamic_t_engine.py scripts/test_dynamic_t_engine.py
- /home/lxl/anaconda3/envs/quant/bin/python -m py_compile quant_backtest/dynamic_t_engine.py scripts/test_dynamic_t_engine.py
- git diff --check -- quant_backtest/dynamic_t_engine.py scripts/test_dynamic_t_engine.py
- /home/lxl/anaconda3/envs/quant/bin/python /tmp/profile_dynamic_t_engine.py （执行副本已保存；输出目录当时由/tmp/dynamic_t_engine_archive_path.txt记录为本目录）

21项语义测试通过，涵盖下一分钟/未来独立、当前close而非极值、真实费用和填单量分母、两方向/止损、锁存后限价/零量不撤、部分恢复、共享容量、旧股/人工加仓、午休/末分钟/跨日、共同陈旧估值、等覆盖v12经济恒等、新增恢复覆盖差异与冻结计划。

冻结前修正：首轮分片测试金额19600在980/9.8的浮点整手边界被原保守floor判为不足1手；测试改为20000以明确只能容纳1手，容量算法未放宽。独立审查发现09:30开盘与09:31收盘是同一原始首bar，原复制循环会清空used重复分配容量；新引擎现保留该一次used至09:31，09:32再重置。反例为隔夜剩200股、首bar仅100股容量：09:30卖100，09:31拒单，09:32卖100。此修复仅新v13生效。

新增plan_dynamic_entries保证相同容量规划。首bar额度16000元，09:30自动1500股×10元后，09:31人工500股只填100；冻结重放现金、数量、价格、费用一致。源v12 planner没有被改动。

真实12股15日完整Dense包含43200真实行。预声明0/1/2/3四种调用各约0.41/0.55/0.53/0.52秒；主动请求同为833，完整组374/375/376，恢复片段400/395/394，pending与T1均0。各模式共同baseline权益逐值相同；配对净利与实际现金最大误差5.10e-11元。本检查不选参数，不评价5%收益目标。

before_opening_capacity_fix_profile.*只保留早期性能记录，其源码身份未完整归档，不作为当前冻结实现验证；当前real_profile.json绑定准确引擎SHA。最终4旧相关源SHA复核未变。root随后执行163候选smoke及年度内部研究，本工程验证不冒称完整策略实验已经通过。
