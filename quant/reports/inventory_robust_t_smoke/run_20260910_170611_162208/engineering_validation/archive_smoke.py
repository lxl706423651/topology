"""归档已结束v17烟测实现证据及零成交原因，不改策略、不再拟合或执行账户。"""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path('/data/lxl/seed-emulator/topology/quant')
sys.path.insert(0, str(ROOT))
import pandas as pd
from quant_backtest.weekly_t_short_horizon import predicted_pair_edges, planned_recovery_times
from quant_backtest.weekly_t_features import select_recoverable_signal_sides
from quant_backtest.weekly_t_quote_store import calculate_file_sha256
from quant_backtest.annual_panel import write_json

report = ROOT/'reports/inventory_robust_t_smoke/run_20260910_170611_162208'
out = report/'engineering_validation'
out.mkdir(exist_ok=False)
shutil.copy2(__file__, out/'archive_smoke.py')
for name in ('inventory_robust_smoke_v17.log', 'inventory_robust_source_preflight_v17.log',
             'inventory_t_robust_modules_validation_summary.json',
             'inventory_t_robust_workflow_validation_summary.json'):
    shutil.copy2(Path('/tmp')/name, out/name)
cfg = json.loads((report/'resolved_config.json').read_text())
identity = {}
rows, exceptions = [], []
for h in (20, 40, 60):
    path = report/f'year_2026/test_predictions/h{h}_huber_depth6/current_predictions.csv'
    identity[str(path)] = calculate_file_sha256(path)
    f = pd.read_csv(path, float_precision='round_trip', parse_dates=['datetime'])
    buy, sell = predicted_pair_edges(f, cfg)
    f['edge'] = pd.concat([buy, sell], axis=1).max(axis=1)
    f['action'] = buy.ge(sell).map({True: 1, False: -1})
    f['side_ok'] = select_recoverable_signal_sides(f, f.action, cfg)
    f['deadline'] = planned_recovery_times(f.datetime, h)
    f['time_ok'] = f.deadline.le(f.datetime.dt.normalize()+pd.Timedelta(hours=14, minutes=55))
    q = f[f.edge.ge(.003)].copy()
    rows.append({'horizon': h, 'current_rows': len(f), 'maximum_cost_adjusted_score': f.edge.max(),
        'margin_003_rows': len(q), 'legal_deadline_margin_rows': int(q.time_ok.sum())})
    if len(q):
        exceptions.append(q[['datetime', 'symbol', 'prediction', 'edge', 'action', 'side_ok',
            'signal_amount', 'signal_volume', 'deadline', 'time_ok']].assign(horizon=h))
pd.DataFrame(rows).to_csv(out/'engineering_zero_signal_funnel.csv', index=False)
pd.concat(exceptions, ignore_index=True).to_csv(out/'engineering_margin_pass_rows.csv', index=False)
for path, digest in identity.items():
    assert calculate_file_sha256(Path(path)) == digest
write_json(out/'validation_summary.json', {'passed': True, 'distinct_implementation_tests': 42,
    'model_policy_tests': 20, 'reporting_tests': 10, 'workflow_tests': 12,
    'policy_repeat_tests_not_double_counted': 10, 'both_python_compiles_passed': True,
    'source_preflight_input_count': 357, 'real_smoke_exit_code': 0,
    'runtime_integrity': json.loads((report/'runtime_integrity_audit.json').read_text()),
    'plot_visually_inspected': True, 'independent_audit_status': 'pending',
    'goal_achieved': False, 'outer_result_is_engineering_only': True,
    'engineering_current_prediction_inputs_sha256': identity,
    'no_refit_or_account_rerun_for_diagnosis': True})
(out/'README.md').write_text('''# v17真实烟测工程验收

42个不同实现用例、双Python编译及357输入来源预检通过。真实烟测主进程exit0，392输入、56源码、48新模型产物始末一致；独立复核尚待完成。原图已人工查看，明确工程用途、零净增量与不可评价胜率。

内部36主动均亏损；所选33完成1组4配对手，净−51.933459元、成本降低−0.030147%。工程外层2026-01-05至01-23为0请求/0成交/0净额，所有库存约束为0失败，不能视为改善。

零成交原因在当前预测即可复算：h20/h40各1440行均未达到原30bp净边际；h60只有2026-01-14 14:00的SZ.000665达到31.153347bp，但下一分钟再推进60交易分钟已跨当日收盘，原合法期限返回NaT，因此不发请求。该股当前限价方向与量正常。没有用未来恢复价格作这项判断，不因零成交调整门槛或读取工程外层择参。

原模型/决策/入口32项证据位于`reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/robust_implementation_validation_20260910_170719_038763/`，报告器10项位于同父目录`robust_reporting_validation_20260910_165855/`。原/tmp日志保留。

实际命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_inventory_robust_t.py --source-report reports/inventory_guarded_t_smoke/run_20260910_154550_255425 --candidate-workers 8 --output-dir reports/inventory_robust_t_smoke
python3 -m py_compile scripts/run_inventory_robust_t.py quant_backtest/inventory_t_robust_models.py quant_backtest/inventory_t_robust_policy.py quant_backtest/inventory_t_robust_reporting.py
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile scripts/run_inventory_robust_t.py quant_backtest/inventory_t_robust_models.py quant_backtest/inventory_t_robust_policy.py quant_backtest/inventory_t_robust_reporting.py
/home/lxl/anaconda3/envs/quant/bin/python /tmp/archive_inventory_robust_smoke.py
```
''')
write_json(out/'manifest.json', {str(p.relative_to(out)): calculate_file_sha256(p)
    for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'})
print(out)
