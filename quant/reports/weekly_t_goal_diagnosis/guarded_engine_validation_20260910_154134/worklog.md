# 新执行条件验证

仅新增 guarded_t_engine.py 和 test_guarded_t_engine.py，旧48源/旧引擎/缓存/模型保持原样。

执行命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_guarded_t_engine.py > /tmp/guarded_t_engine_tests.log 2>&1
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile quant_backtest/guarded_t_engine.py scripts/test_guarded_t_engine.py
python -m py_compile quant_backtest/guarded_t_engine.py scripts/test_guarded_t_engine.py
```

15项最终测试通过，耗时6.719秒。初轮14项通过、1项manual fixture缺少原API必填signal_datetime而报错；只补新测试fixture后全套通过，未改变生产人工接口。真实案例fixture的信号high核成19.78。

关闭控制对三种退出模式、最低佣金和部分恢复的全部原result字段逐值精确相等。开关开启后验证买先跌停/卖先涨停取消、真实限价优先、6%普通平价允许、5%候选平价拒绝、零数量检查不要求虚拟反向容量、取消不消耗现金/费用/旧股/次数/预留、同刻现金留给下一股票、后续请求继续、未来报价独立、已有pending不被取消/隐藏、原次分钟检查、原planner与人工当日不可卖，以及布尔配置拒绝错误字符串。

本条件沿用现有“下一分钟close为到达报价”的成交模型。它不是上一信号分钟已知因子，也不依赖后来是否恢复。完整OHLCV只是该分钟的模拟报价证据，不证明真实盘口队列和零延迟撤单一定可实现。新报告应称新增执行报价条件，不能回写旧结果或当作恢复保证。
