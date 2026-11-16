# v17 报告器命令与验证记录

- 授权：只新增 inventory_t_robust_reporting.py 与对应测试；旧57运行源和原报告不动。
- 实现：原cfg直接进入balanced核算，纯guard/exit统计复用；新Huber六模型、固定alpha与三期限身份，reference score不冒充账户利润。
- 保持：原始含费底仓成本及人工买入分母、5%目标、未量化频率、完整组/片段、取消请求/实际成交、pending与亏损。
- 验证：10测试通过，3.965秒；午休/三期限/非法期限、参数身份、图题、输入不变、原年度CSV不变均覆盖。仅一个两日合成guard执行核案例，无真实行情回测或模型拟合。

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_inventory_t_robust_reporting.py
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile quant_backtest/inventory_t_robust_reporting.py scripts/test_inventory_t_robust_reporting.py
python -m py_compile quant_backtest/inventory_t_robust_reporting.py scripts/test_inventory_t_robust_reporting.py
sha256sum quant_backtest/inventory_t_robust_reporting.py scripts/test_inventory_t_robust_reporting.py
```

- 归档核对：原v16 source_hashes.json中57路径的实际SHA全部相同；只读文件身份，不读取任何账户收益。
- 报告器陈述的是协议身份；实际训练matrix/labels与v14的同源核对由入口和独审负责，报告不替代其证据。
- 边界：Huber不保证条件均值、预测更小或盈利；本测试通过不表示年度5%目标已达到。
