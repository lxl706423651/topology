"""仅归档v18完整内部主运行的终态及命令，不改原研究产物。"""
from pathlib import Path
import json, hashlib, shutil
import pandas as pd
root=Path('/data/lxl/seed-emulator/topology/quant')
report=root/'reports/inventory_coverage_t/run_20260910_182452_383731'
out=report/'engineering_validation'
out.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
refs=[
'reports/weekly_t_goal_diagnosis/inventory_coverage_workflow_validation_20260910_181427/verification.json',
'reports/weekly_t_goal_diagnosis/inventory_forecast_objective_research_20260910/coverage_implementation_validation_20260910_181330/verification.json',
'reports/weekly_t_goal_diagnosis/inventory_coverage_reporting_validation_20260910/run_20260910_180616_085947/validation.json',
'reports/weekly_t_goal_diagnosis/inventory_coverage_source_preflight_20260910_181326/verification.json',
'reports/inventory_coverage_t_smoke/run_20260910_181434_954042/independent_coverage_audit/run_20260910_182018_637252/verification.json']
originals={p.name:sha(p) for p in sorted(report.iterdir()) if p.is_file()}
fields=['candidate_index','t_pair_count','t_fragment_count','completed_t_pair_lots','incremental_net_pnl','all_reference_acquisition_cost','holding_cost_reduction_rate','inventory_mismatch_days','recovery_failure_days','pending_pairs','t1_violation_count','base_plan_mismatch_count','selection_reason']
rows=[]
for year in range(2023,2027):
 p=report/f'year_{year}/selected_rule.json'
 selected=json.loads(p.read_text())
 originals[str(p.relative_to(report))]=sha(p)
 row={'test_year_label':year,'inner_year':2025 if year==2026 else 2026}
 row.update({k:selected['inner_metrics'].get(k) for k in fields})
 rows.append(row)
integrity=json.loads((report/'runtime_integrity_audit.json').read_text())
assert integrity=={'all_unchanged':True,'input_files_verified':990,'runtime_sources_verified':56,'model_artifacts_verified':240}
assert not json.loads((report/'outer_evaluation_decision.json').read_text())['outer_evaluation_started']
assert [r['candidate_index'] for r in rows]==[0,15,0,0]
assert all(r['holding_cost_reduction_rate']<.05 for r in rows)
shutil.copy2(__file__,out/'archive.py')
shutil.copy2('/tmp/inventory_coverage_full_v18.log',out/'main.log')
pd.DataFrame(rows).to_csv(out/'selected_internal_results.csv',index=False,encoding='utf-8-sig')
proof={'main_exit_code':0,'main_pty':26036,'command':'/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_inventory_coverage_t.py --candidate-workers 24','distinct_implementation_tests':46,'runtime_integrity':integrity,'inner_model_count':24,'inner_candidate_account_count':148,'formal_outer_evaluation_started':False,'independent_full_audit_status':'running','independent_full_audit_dir':'independent_coverage_audit/run_20260910_184346_798266','goal_achieved':False,'profit_target_met':False,'frequency_target_status':'not_quantified','selected_internal_results':rows,'references_sha256':{name:sha(root/name) for name in refs},'original_report_files_sha256':originals}
(out/'validation_summary.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
(out/'README.md').write_text('# v18完整内部主运行验收\n\n主PTY26036实际exit0：24新L2模型、148内部候选账户，990输入/56源/240新模型产物始末一致。每模型由30变为33特征，只改覆盖门槛0.80至0.70。\n\n按留出年标签2023/2024/2025/2026，内部所选0/15/0/0。三个hold因没有执行合格的主动候选，不能视作择时改善；15完成378组、3597配对手，净−3887.1194元，全部原底仓成本841953.8566650599元，成本降低−0.4616784%，五项执行失败均0。正式外层未执行，四折内部均未5%，手数要求未量化。\n\n完整独立审计进行中，不以主运行完整性替代独立核验。本目录保存实际日志、命令、所选内部结果与来源SHA；原代码、模型、报告及所有候选保持冻结。\n')
assert originals=={name:sha(report/name) for name in originals}
(out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},ensure_ascii=False,indent=2)+'\n')
print(out)
print('validation_summary_sha256',sha(out/'validation_summary.json'))
print('manifest_sha256',sha(out/'manifest.json'))
