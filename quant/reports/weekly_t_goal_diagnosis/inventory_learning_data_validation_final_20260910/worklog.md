# 验证命令与结果

- 12项合成测试：`/home/lxl/anaconda3/envs/quant/bin/python scripts/test_inventory_t_learning_data.py`，exit 0。
- 双编译：系统 python3 与 quant Python 的 `-m py_compile`，两文件均通过。
- 最终真实首折：`/home/lxl/anaconda3/envs/quant/bin/python /tmp/check_inventory_learning_real_final.py > /tmp/check_inventory_learning_real_final.log 2>&1`，exit 0，165.96秒。数据模块在运行始末 SHA 一致；旧动态引擎未改。
- `python /tmp/finalize_inventory_learning_validation.py` 核全部18份CSV逐字节相同，并独立复核自动底仓每腿现金/费用/整手/金额区间和无T库存审计。

首次真实预检的 source_sha256 是完成时文件身份，加载后曾仅删旧trade_week_complete字段；最终轮已保存准确启动源码并再次完整构建，作为最终验证依据。两轮的账户和lineage均相同，没有拟合、标签连接或外层收益读取。
