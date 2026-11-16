# 只读审查工作记录

执行：`/home/lxl/anaconda3/envs/quant/bin/python /tmp/check_permanent_core_cohort.py > /tmp/permanent_core_cohort_check.log`，退出码 0。审查脚本只读取并核对已保存模型身份、周缓存当前价格/日历/来源、完整排名供体键；未拟合、未运行账户、未读取 v13 收益表、未改冻结源码。

核对范围：8 selectors、16 原生头、38,700 入选股周；全部连接成功。量化结果见 CSV/JSON，来源边界和最小实现建议见备忘。模块和测试的后续实施属于新的授权工作，未包含在本只读结果中。

归档前再次核对本次实际读取的 5 份生产源 SHA，均与审查开始时一致。
