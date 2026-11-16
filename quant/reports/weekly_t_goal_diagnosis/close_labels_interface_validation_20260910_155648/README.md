# 固定14:55参考标签数据接口

仅准备纯数据接口，没有新配置、模型拟合或账户执行；不修改v15烟测51份运行源码。当前尚未决定用该目标启动新模型。

```python
labels, audit = extract_fixed_close_labels(full_original_intraday, cfg)
# core必须先按原严格来源/真实无T旧股账本构造并冻结。
training_frame = attach_fixed_close_labels(core, labels, cfg)
```

cfg只消费原8个signal_times及fixed_recovery_time=14:55；旧horizons=[20,40,60]不改变本标签身份。返回标签明确recovery_key=close、recovery_kind=fixed_clock、recovery_clock=14:55，没有horizon_minutes。

extract要求唯一datetime/symbol、精确execution_time/price与recovery_time/price；原label_start/end存在时一起验证。用价格重新计算target_return，原完整周资格及旧target不参与样本保留。缺未来价格保留NaN和label_available=False；零量仍可有参考目标，audit明确不能当成交或恢复保证。非正/无限价格、错误端点、非法信号或重复身份拒绝。

attach拒绝core已有execution_/recovery_/target_/label_及旧trade_week_complete/horizon_minutes；只附本schema的标签价格和由当前键确定的时钟。缺整条label键不删core，仍有确定的下一合法分钟和当日14:55，但price/target为NaN。保留core原索引（包括重复位置索引）、行序、f32因子及NaT来源；来源资格仍由后续模型训练purge负责。该接口不把未来端点可用性引入当前推断。

输入文件SHA由未来调用者原数据加载层记录，本纯函数不读取缓存、raw文件或模型。本次只跑13个合成反例：时钟与固定身份、旧周条件忽略、两侧缺价、零量参考、错误时间与重复键、配置与非法价格、缺行左联/NaT来源、未来改写独立、不原地修改、core未来列拒绝、标签篡改和空表。

验证命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_inventory_t_close_labels.py
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile quant_backtest/inventory_t_close_labels.py scripts/test_inventory_t_close_labels.py
python -m py_compile quant_backtest/inventory_t_close_labels.py scripts/test_inventory_t_close_labels.py
```

13项通过（0.216秒），两套Python编译通过。源副本、测试日志、原51源身份和接口SHA见validation.json及manifest.json。此前原缓存/原始端点核对独立保存在 ../close_label_feasibility_20260910_151622/，本次没有重复加载全市场。
