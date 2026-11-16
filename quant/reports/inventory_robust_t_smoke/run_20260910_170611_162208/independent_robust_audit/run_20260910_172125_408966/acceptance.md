# v17 Huber原始短期限预测与实际执行独立审计

结果：失败，保留本次原错误与脚本。

只读保存产物及原始行情，不训练、不重跑账户、不改原报告或执行源。

```text
Traceback (most recent call last):
  File "/data/lxl/seed-emulator/topology/quant/scripts/audit_inventory_robust_t.py", line 536, in main
    result=auditor.run()
  File "/data/lxl/seed-emulator/topology/quant/scripts/audit_inventory_robust_t.py", line 350, in run
    self.decision_funnels.append(dict(test_year=year, scope='validation', candidate_index=index,
TypeError: dict() got multiple values for keyword argument 'scope'

```
