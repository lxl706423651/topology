"""保存v18已完成烟测工程记录；仅写辅助新目录，不修改原回测或运行源码。"""
from pathlib import Path
import json, hashlib, shutil
import pandas as pd

root = Path('/data/lxl/seed-emulator/topology/quant')
report = root/'reports/inventory_coverage_t_smoke/run_20260910_181434_954042'
out = report/'engineering_validation'
out.mkdir(exist_ok=False)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
references = [
 'reports/weekly_t_goal_diagnosis/inventory_coverage_workflow_validation_20260910_181427/verification.json',
 'reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/coverage_implementation_validation_20260910_181330/verification.json',
 'reports/weekly_t_goal_diagnosis/inventory_coverage_reporting_validation_20260910/run_20260910_180616_085947/validation.json',
 'reports/weekly_t_goal_diagnosis/inventory_coverage_source_preflight_20260910_181326/verification.json',
 'reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/coverage_selector_preflight_20260910/run_20260910_180402_766080/verification.json']
identity={name:sha(root/name) for name in references}
originals={name:sha(report/name) for name in ['resolved_config.json','source_hashes.json','input_sources.json','runtime_integrity_audit.json','all_rules_frozen_before_outer_test.json','final_report.md','annual_summary.csv','goal_evaluation.json']}
shutil.copy2(__file__,out/'archive.py')
shutil.copy2('/tmp/inventory_coverage_smoke_v18.log',out/'main.log')
shutil.copy2('/tmp/inventory_coverage_source_preflight.log',out/'source_preflight.log')
inner=json.loads((report/'year_2026/selected_rule.json').read_text())
annual=pd.read_csv(report/'annual_summary.csv',float_precision='round_trip').iloc[0]
fields=['t_pair_count','completed_t_pair_lots','holding_cost_reduction_rate','all_reference_acquisition_cost','inventory_mismatch_days','recovery_failure_days','pending_pairs','t1_violation_count','base_plan_mismatch_count']
proof={'main_exit_code':0,'distinct_implementation_tests':46,'runtime_integrity':json.loads((report/'runtime_integrity_audit.json').read_text()),
 'independent_smoke_audit_status':'pending','references_sha256':identity,'original_report_files_sha256':originals,
 'command':'/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_inventory_coverage_t.py --source-report reports/inventory_guarded_t_smoke/run_20260910_154550_255425 --candidate-workers 8 --output-dir reports/inventory_coverage_t_smoke',
 'inner_year':2025,'inner_selected_rule':inner['rule'],'inner_metrics':{k:inner['inner_metrics'][k] for k in fields+['incremental_net_pnl']},
 'engineering_outer_year':2026,'engineering_outer_metrics':{k:float(annual[k]) for k in fields+['incremental_marked_pnl']},
 'engineering_slice_is_profit_validation':False,'goal_achieved':False,'curve_visually_checked':True}
(out/'validation_summary.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
(out/'README.md').write_text('# v18真实烟测工程记录\n\n主进程exit0，六模型/60产物、392登记输入与56运行源完整性通过；46项实现反例、双Python编译、CLI及来源预检在真实运行前通过。完整核心训练，工程评价仍12股/15交易日。\n\n内部选15：35组368手，净−635.7192元，原成本降低−0.3690%；工程外层5组69手，净+323.67元、成本降低+0.1782%。两者五项执行失败0。工程片段不证明年度盈利或手数达标。独立审核待完成，不以主运行完整性替代独立验收。\n\n本目录只保存原结束日志、命令、前序证据摘要及原产物SHA；原报告、模型、CSV和运行代码保持冻结。\n')
assert originals=={name:sha(report/name) for name in originals}
(out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},ensure_ascii=False,indent=2)+'\n')
print(out)
print('summary_sha256',sha(out/'validation_summary.json'))
