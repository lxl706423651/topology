# 固定底仓引擎与人工上下文工作记录

- 任务：只读诊断旧95%/100slot与固定底仓金额要求的差距，随后按root明确委派新增固定底仓引擎和人工CSV上下文，保持旧源码/缓存不变。
- 诊断命令：`/home/lxl/anaconda3/envs/quant/bin/python reports/weekly_t_goal_diagnosis/run_20260910_134100_engine_inventory/diagnose.py`，只读20个基准持仓/计划/库存/拒单表，不读取收益列；4个原源和20个原账表字节SHA始末未变。
- 新源码：quant_backtest/balanced_t_engine.py、quant_backtest/balanced_t_manual.py；对应tests为scripts/test_balanced_t_engine.py、scripts/test_balanced_t_manual.py。
- 引擎语义：普通每股目标1.5万元、实际交易名义额1—2万元/整手；资金决定只数，初始85%部署约56只；底仓不按周或期末主动卖出；每日多组共用旧股可卖预算，人工买入不补预算；真实分片恢复/费用/现金和失败状态全部保留。
- 基础规划：plan_balanced_entries仅no-T基准因果决定成功底仓，同一量价费计划冻结重放；未用T盈利扩仓。selections.signal_datetime真实源信号优先保留。
- 人工接口：prepare_manual_context(cfg,evaluation_year,days,base_store,inputs)。完整CSV是用户外部指令，不是模型特征；整CSV/raw完整SHA记录。按年度/配置起止/研究截止切片，范围内非交易日/非法时刻显式拒绝，缓存外时钟只补当前原始分钟且缺行保留。overlay通过spawn测试。
- 验证命令：quant Python分别执行两个test脚本，26+5项通过；quant与系统Python对4个新文件py_compile通过；git diff --check通过。
- 初始开发修正：规划事件plan_id为字符串时先转整数以与计划表精确关联；现金边界测试最初预期6只，手算扣费后第6只不足1万元，修正为5只。两者均发生在冻结前，原回测未修改。
- 真实机械规划检查：2026全市场3198股、245276日行情、1600候选，6.15秒冻结56只，77日窄表重放0.249秒；逐日现金/基准权益/持仓值完全相同。名义额13736.865—16088.04、中位14971.482、金额违规0、底仓卖出0。这是执行与性能检查，不是收益证据。
- 旧dense真实缓存完整SHA校验并兼容新引擎，1.04秒构造；旧engine和metadata未变。
- 归档：engine_validation保存冻结源/测试副本、测试日志、机械规划/dense验证及validation.json。完整策略年度训练和5%收益目标尚由主流程验证，本任务未宣称已达收益目标。

- 估值修复（root明确授权，仅新引擎）：每笔真实成交使用当时原始参考价更新共同市场mark/time，不使用含滑点成交价；随后有效15:00收盘继续覆盖。新增10:09首次人工买入而日开收缺价、有效收盘覆盖两项反例，28+5项通过，双Python编译与diff检查再次通过。旧26项版本和证据保存在engine_validation/before_mark_fix，不覆盖初始证据。
- 修复影响范围：完整收盘数据的日终估值不变；缺收盘时使用更新的真实成交参考价并保留陈旧时间。若实际/基准股数相同，共同mark变化不改变T增量；存在待恢复差股时按股数差×价差真实计值。独立no-T规划在该类缺价日可能只有较旧mark，不宣称所有缺价T场景的绝对基准权益都不变。
- 修复前后真实对照：2026年56股77日no-T，曲线77行、实际/基准各56笔、库存4312行、金额56行、期末持仓56行逐值精确相同；耗时6.224秒。证据engine_validation/mark_fix_default_comparison.json。本对照只证明账户/估值兼容性，不是盈利证据。
- 最终自有源冻结：engine436f78b02006777d5e2de9f805ff7f829160c54c61645c3028ee58bc67e888a4、engine tests e7a2f29719ac26f0fd406b7f24501b4af80a505afa294448c017febc88439f18；manual b4405f75f7f3dfb7fa328515fff093e0f924ec51572f0caa4c4c43e6ce496ccc、manual tests7756a5453e82816db277c40d10b61cb14db235ecca7d0355199e619ea2008ebc。再次核对原4源/20账表SHA未变。

- 后续只读设计委派：检查新引擎 active/recover_pair/timeline/mark 和旧dense时钟索引；未读收益、未改源码。新增 dynamic_recovery_design.md/json，提出完成分钟观察→下一交易分钟成交→锁存分片状态机、共同估值和10个必要反例。只读实际2026 times.npy确认136现有/104缺失分钟；56股77日需补时钟上限448448条。四个自有源及原四源SHA仍一致。备忘只建议新模块与独立对照，没有修改当前v12或选择收益参数。
