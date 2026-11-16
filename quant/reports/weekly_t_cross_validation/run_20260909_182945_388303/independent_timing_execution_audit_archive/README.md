# v11 独立执行审计归档

本目录仅新增独立核验资料，未修改生产策略、模型或原始回测结果。

实际命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_weekly_t_timing_execution.py --report-dir reports/weekly_t_cross_validation/run_20260909_182945_388303
```

审计范围为默认完整模式：100 个内层候选账户、全部当前信号门控和 20/40/60 分钟期限选择、正净利优先的内部选择、4 个外层账户逐笔原始行情重放。`audit.log` 保存终端输出；最终状态以 `verification.json` 为准。

CSV 文本不携带 dtype。本审计依照冻结 data/session/timing 源码还原 39 个原 float32 当前特征，4 个周上下文特征保持原 float64。新增支持域边缘反例验证未放宽门槛。`dtype_contract_verification.json` 同时核对独立模型审计实际来源重建的 24 张表全部类型一致。仅改审计读取逻辑，未修改回测或重算账户。修改前审计器在相邻 `independent_timing_execution_audit_reader_revision` 目录保留。

7 个当前门控/选择反例与 10 个 raw 账户重放用例全部通过，副本及日志在本目录。`started_manifest.json` 绑定启动前 2269 个原报告/模型/候选文件；完成后逐个复核字节 SHA，32 个运行源及快照另外核对。

本轮先前旧 smoke 仅保存部分模型入选列，故 smoke 运行时明确使用 accounting-only；正式报告保存全部当前门控列，本次未使用该选项。完整训练样本、原生模型与预测隔离核验见独立模型审计，不由本执行审计冒称重训。

最终结果：默认完整审计 exit 0，通过全部 100 候选、4 外层账户、7,039 原始行情文件的 32,004 成交分钟。86 完成组/87 恢复片段，账户净增量 −573.7200510555413 元。T+1、日终股数偏差、恢复失败、pending 和基准计划偏差全部为 0；现金/权益最大浮点误差约 1.397e−9 元，冻结 v4 基准逐日误差 0。底仓原有周退出失败 171 日、期末周持仓 7 笔仍保留，其中 1 笔陈旧估值市值 9750 元。32 运行源及快照、2269 个原报告/模型/候选文件的字节 SHA 全部未变。
