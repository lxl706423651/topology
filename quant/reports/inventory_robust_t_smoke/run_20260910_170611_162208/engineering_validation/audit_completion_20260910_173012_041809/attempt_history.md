# v17烟测独立审计尝试记录

三次均为只读审计尝试；失败不计验收通过，原错误不覆盖。生产模型、账户、策略配置和行情未因审计修复而改变。

## run_20260910_171807_816821

状态：peer_review_terminated；exit=143。
脚本SHA：`0d6738eb7973befd5e078c674e9f998bbaba9eef773e5c51e9f32b2e81fab9e0`。

- 未完成组的marked账户增量不能充当完整组成本降低率分子。
- hold的来源L2档位须为JSON null/CSV空值，不能改成字符串hold。
- 同轮只读复核另发现旧account未返回T滑点；新子类须从逐腿价格差独立补值。
- 回归用例：`test_pending_marked_pnl_is_not_complete_profit_ratio`。
- 回归用例：`test_hold_report_source_profile_is_null`。
- 回归用例：`test_account_adds_independent_t_slippage_not_base_slippage`。

原证据：[verification.json](/data/lxl/seed-emulator/topology/quant/reports/inventory_robust_t_smoke/run_20260910_170611_162208/independent_robust_audit/run_20260910_171807_816821/verification.json)。

## run_20260910_172125_408966

状态：auditor_exception；exit=1。
脚本SHA：`811715b883e90ab67449700969e2c9a4ca87f3d5b4574c991c3a6a683089536d`。

- 外层账户scope与分流统计scope同名，dict打包发生TypeError；已改为funnel_definition。
- 回归用例：`test_zero_request_funnel_retains_late_profitable_score`。

原证据：[verification.json](/data/lxl/seed-emulator/topology/quant/reports/inventory_robust_t_smoke/run_20260910_170611_162208/independent_robust_audit/run_20260910_172125_408966/verification.json)。

## run_20260910_172441_003843

状态：complete_passed；exit=0。
脚本SHA：`a9348c1ad6f0ecb8f81019bd3ee0bb193eec0e0e642cc2ee22cda93184ca5984`。


原证据：[verification.json](/data/lxl/seed-emulator/topology/quant/reports/inventory_robust_t_smoke/run_20260910_170611_162208/independent_robust_audit/run_20260910_172441_003843/verification.json)。

