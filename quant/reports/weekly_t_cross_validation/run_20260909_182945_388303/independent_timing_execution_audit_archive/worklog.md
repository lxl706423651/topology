# 本轮独立执行审计工作记录

- 任务：默认完整核查 v11 全部内层候选和四折真实执行；不训练、不调参、不改原回测。
- 审计器：`scripts/audit_weekly_t_timing_execution.py`，SHA `eec81401ea1c2c1a4029f63fc59418501dc9361d38e6ec6ba0a497261b522623`。
- 审计前读入修正：恢复冻结源码确定的 39 个 float32 当前特征，以复现支持域边界；旧审计器、diff 和精确边界反例已保留。策略与账户 CSV 未改。正式完整审核一次通过。
- 验证：quant Python 与系统 Python 双编译；7 个门控/选择/类型反例、10 个独立 raw 重放用例均通过。辅助 dtype 合同 AST 提取在两次解析错误后改用实际字符串 split 节点，最终与独立模型审计 24 表逐项一致；这不是回测或正式审计失败。
- 命令：`/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_timing_execution.py --report-dir reports/weekly_t_cross_validation/run_20260909_182945_388303`，终端 exit 0，完整输出 `audit.log`。未使用 accounting-only。
- 结果：100 个候选账本、4 折当前请求精确复算；4 年逐笔原始价格/容量/费用/现金/T+1/片段与每日库存独立重放通过；数值见 verification.json 与主目录 independent_execution_audit.json/md。
- 归档：启动和结束分别核对 2269 个原文件 SHA；32 当前运行源和原快照全部一致；独立模型审核 JSON 另绑定 SHA。本目录只新增审计资料。
- 限制：已验证会计恒等式和冻结执行语义；不能据此声称盈利或全部周末均能清仓。原周退出失败和陈旧价格持仓继续披露。完整训练隔离重建由单独模型审计负责。
