# v16 独立模型审计辅助

新增 `scripts/inventory_close_audit_models.py`，不修改旧生产/审计模块。主审计器调用 `audit_close_model(auditor, directory, training, features, feature_audit, fold, profile, full_frame=joined)`，返回可归档结果。`training`、`features` 和完整 `joined` 来自主审计独立恢复；辅助不调用生产 close 标签、模型或冻结队列构造器，不再重复市场/队列/账户。

检查包括完整机会的来源至标签区间与7日隔离、逐原因训练split及metadata、原8个精确信号到下一交易分钟/同日14:55的端点身份、原始CSV的精确价格和零量参考标签计数、全部实际训练矩阵列序/dtype/值、实际标签及来源、文件和结构化SHA、完整预声明模型参数、bundle内外metadata、全部原生树与joblib一致，以及最多1024个确定性训练点的重载。全部树相同是全权重身份核查，重载点数单独列出，不冒称逐行重新计算全部训练预测。

6个纯反例与双Python编译已通过。反例覆盖合法两个core、混入整数期限、错误截止时钟、用扣费目标代替raw标签、无日初旧股、来源越界及缺标签行只影响训练而不消灭原机会。自检未拟合模型、未读取市场，也未运行账户。

本记录仅证明辅助脚本工程检查；完整真实smoke审计随后由 `audit_inventory_close_t.py` 联合运行并单独归档，不能以此记录提前声称全审通过。
