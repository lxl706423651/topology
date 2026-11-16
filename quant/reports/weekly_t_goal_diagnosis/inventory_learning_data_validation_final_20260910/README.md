# 永久底仓训练数据验证

本目录仅验证数据构建和无 T 规划，不训练模型、不连接未来标签、不读取外层收益。最终结果由 verification.json 和 identity_comparison.json 给出。

12 个合成反例覆盖入场日 T+1、离开周榜后的持仓、完整排名供体来源、规划前隔离、其它股票的现金竞争来源、人工训练禁用、未来标签擦除、当前 NaT 来源保留、缺开盘不补未来价格、源身份变更拒绝、全现金年度、年末 embargo。独立 agent 只读复核了模型/供体身份、入场来源筛查和持有期来源传播。

真实预检命令：`/home/lxl/anaconda3/envs/quant/bin/python /tmp/check_inventory_learning_real_final.py > /tmp/check_inventory_learning_real_final.log 2>&1`。脚本副本为 check_real_first_fold.py；重现时须把其中 out 改为新的目录，脚本拒绝覆盖已有结果。

原第一轮通过后，仅对新模块删除旧周标签用的 trade_week_complete 并增加测试。最终轮使用冻结源码重新完成同一纯训练规划，之后逐字节比对原轮的全部 CSV。两轮都未拟合模型。

数据接口为 build_inventory_core_cohort(universe, daily, intraday, fold, cfg, out, inputs, record_input, recovery_store=None)，返回 (cohort, metadata)。完整截面特征由调用方事先生成，标签由调用方后置左联。元数据不宣称固定端点标签等于真实组合执行利润；真实引擎负责之后的内部验收。
