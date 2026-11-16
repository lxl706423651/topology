"""完整v17终态后只更新工程独审状态，保留主运行和原审计证据。"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent
ROOT=Path('/data/lxl/seed-emulator/topology/quant')
AUDIT=REPORT/'independent_robust_audit/run_20260910_174250_496951'
AUDIT_SHA='5d5e48aa04977eb436d7d3f5241ff0b61d79c1ae502becb73269186f32f085ff'

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    proof=json.loads((AUDIT/'verification.json').read_text())
    assert sha(AUDIT/'verification.json')==AUDIT_SHA
    assert all(proof[k] is True for k in ('passed','complete','all_read_inputs_unchanged','audit_dependencies_unchanged'))
    assert proof['outer_accounts']==0 and proof['goal_achieved'] is False
    original=json.loads((HERE/'manifest.json').read_text())
    assert all(sha(HERE/name)==value for name,value in original.items())
    runtime=json.loads((REPORT/'source_hashes.json').read_text())
    frozen={str(ROOT/name):value for name,value in runtime.items()}
    frozen.update(proof['audit_dependencies_sha256'])
    assert len(runtime)==56 and len(proof['audit_dependencies_sha256'])==11
    assert all(sha(name)==value for name,value in frozen.items())
    artifact=json.loads((AUDIT/'artifact_sha256.json').read_text())
    assert all(sha(AUDIT/name)==value for name,value in artifact.items())
    out=HERE/f'audit_completion_{datetime.now():%Y%m%d_%H%M%S_%f}'
    previous=out/'previous';previous.mkdir(parents=True,exist_ok=False)
    for name in ('validation_summary.json','README.md','manifest.json'):shutil.copy2(HERE/name,previous/name)
    selected=[]
    for choice in proof['choices']:
        path=f"year_{choice['test_year']}/validation_candidates/candidate_{choice['candidate_index']:03d}"
        row=next(item for item in proof['accounts'] if item['path']==path)
        selected.append({**choice,'completed_pairs':row['t_pair_count'],'completed_pair_lots':row['completed_t_pair_lots'],
            'reference_acquisition_cost':row['all_reference_acquisition_cost'],
            'realized_complete_pair_pnl':row['realized_complete_pair_pnl'],'t_fee':row['t_fee'],
            't_slippage_cost':row['t_slippage_cost'],'market_days':row['market_days'],
            'evaluation_start':row['first_market_day'],'evaluation_end':row['last_market_day'],
            'execution_failure_counts':{name:row[name] for name in ('inventory_mismatch_days','recovery_failure_days','pending_pairs','t1_violation_count','base_plan_mismatch_count')},
            'completed_pnl_vs_account_increment_absolute_error':abs(row['realized_complete_pair_pnl']-row['incremental_net_pnl'])})
    summary=json.loads((HERE/'validation_summary.json').read_text())
    assert summary['independent_full_audit_status']=='running'
    summary.update(independent_full_audit_status='passed',independent_full_audit_path=str(AUDIT/'verification.json'),
        independent_full_audit_sha256=AUDIT_SHA,independent_full_audit_exit_code=0,
        independent_full_audit_scope=dict(models=len(proof['models']),current_prediction_tables=len(proof['predictions']),
            internal_candidates=proof['candidate_accounts'],accounts_including_baselines=proof['checked_accounts'],
            outer_accounts=proof['outer_accounts'],read_inputs=len(proof['inputs_sha256']),
            raw_trade_event_checks=proof['raw_event_checks'],raw_first_exit_intent_checks=proof['raw_exit_intent_checks'],
            guard_rows=proof['guard_rows_verified'],guard_cancellations=proof['cancellations_verified'],
            cross_candidate_counts_are_not_one_portfolio=True),
        independent_full_audit_tests=proof['self_tests'],independently_checked_selected_internal=selected,
        production_retrained_or_accounts_rerun_by_auditor=False)
    write(HERE/'validation_summary.json',summary)
    write(out/'completion_identity.json',dict(passed=True,exit_code=0,updated_only_engineering_auxiliary_status=True,
        previous_summary_sha256=sha(previous/'validation_summary.json'),current_summary_sha256=sha(HERE/'validation_summary.json'),
        unchanged_root_readme_sha256=sha(HERE/'README.md'),audit_verification_sha256=AUDIT_SHA,
        audit_artifact_manifest_sha256=sha(AUDIT/'artifact_sha256.json'),
        runtime_sources_verified=56,audit_dependencies_verified=11,selected_internal_accounts=selected,
        formal_outer_evaluated=False,goal_achieved=False))
    shutil.copy2(__file__,out/'complete_independent_audit.py')
    (out/'command.txt').write_text('python '+str(Path(__file__).resolve())+'\n')
    assert all(sha(name)==value for name,value in frozen.items())
    assert sha(AUDIT/'verification.json')==AUDIT_SHA
    assert sha(HERE/'README.md')==sha(previous/'README.md')
    write(HERE/'manifest.json',{str(path.relative_to(HERE)):sha(path) for path in sorted(HERE.rglob('*'))
        if path.is_file() and path!=HERE/'manifest.json'})
    print(out)
    print('summary SHA',sha(HERE/'validation_summary.json'))
    print('engineering manifest SHA',sha(HERE/'manifest.json'))

if __name__=='__main__':main()
