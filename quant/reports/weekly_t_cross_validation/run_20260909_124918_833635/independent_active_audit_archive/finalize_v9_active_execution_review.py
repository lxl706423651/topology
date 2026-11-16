"""Bind independent v9 execution review to unchanged result and reporting revision hashes."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
ROOT=Path('/data/lxl/seed-emulator/topology/quant');REPORT=ROOT/'reports/weekly_t_cross_validation/run_20260909_124918_833635'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((REPORT/'independent_execution_audit.json').read_text());assert audit['passed'] and audit['completed_all_four_years']
archive=REPORT/'independent_active_audit_archive';assert (archive/'verification.json').is_file()
revision=REPORT/'report_revision/manifest.json';rev=json.loads(revision.read_text())
assert rev['republished_files']==[]
assert all(sha(REPORT/name)==value for name,value in rev['protected_original_files_sha256'].items())
declared=json.loads((REPORT/'source_hashes.json').read_text())
assert all(sha(REPORT/'source_snapshot'/Path(name).name)==value for name,value in declared.items())
changed=[name for name,value in declared.items() if sha(ROOT/name)!=value]
assert changed==['quant_backtest/weekly_t_active_reporting.py']
assert sha(ROOT/changed[0])==rev['revised_reporter_sha256']
summary=pd.read_csv(REPORT/'annual_summary.csv');annual=[]
marker=REPORT/'all_rules_frozen_before_outer_test.json'
for r in audit['annual']:
    year=r['year'];directory=REPORT/f'year_{year}';reported=summary.loc[summary.test_year.eq(year)].iloc[0]
    for key in ['t_first_leg_events','t_recovery_events','completed_pairs','fragments','pending_pairs']:
        assert r[key]>=0
    assert int(reported.final_stale_position_count)==r['final_stale_position_count']
    assert abs(reported.final_stale_market_value-r['final_stale_market_value_CNY'])<1e-8
    assert (directory/'selected_rule.json').stat().st_mtime_ns<=marker.stat().st_mtime_ns<=(directory/'t_predictions.csv').stat().st_mtime_ns
    decisions=pd.read_csv(directory/'t_decisions.csv');events=pd.read_csv(directory/'test_trade_events.csv')
    trips=pd.read_csv(directory/'test_round_trips.csv')
    annual.append({'year':year,'requests':len(decisions),'actual_events':len(events),'baseline_events':r['baseline_events'],'t_first_legs':r['t_first_leg_events'],'t_recovery_legs':r['t_recovery_events'],'completed_pairs':r['completed_pairs'],'fragments':r['fragments'],'increment_CNY':r['T_increment_CNY'],'paired_net_pnl_CNY':float(trips.net_pnl.sum()),'T_fees_CNY':r['T_fees_CNY'],'T_slippage_CNY':r['T_slippage_CNY'],'inventory_mismatch_days':r['inventory_mismatch_days'],'recovery_failure_days':r['recovery_failure_days'],'weekly_exit_failure_days':r['weekly_exit_failure_days'],'weekly_unclosed_positions':r['weekly_unclosed_positions'],'final_stale_market_value_CNY':r['final_stale_market_value_CNY'],'frozen_v4_baseline_max_abs_error_CNY':r['frozen_v4_baseline_max_abs_error_CNY'],'cash_max_abs_error_CNY':r['cash_max_abs_error_CNY'],'equity_max_abs_error_CNY':r['equity_max_abs_error_CNY']})
for p in [Path('/tmp/review_active_cross_day.py'),Path('/tmp/review_active_cross_day.log'),Path(__file__),Path('/tmp/v9_active_execution_audit_orchestration.py')]:
    shutil.copy2(p,archive/p.name)
value={'passed':True,'accounting_audit_passed':True,'daily_inventory_contract_met_all_years':all(r['strict_daily_inventory_contract_met'] for r in audit['annual']),'full_weekly_exit_contract_met_all_years':all(r['full_weekly_exit_contract_met'] for r in audit['annual']),'warning':'Passing independent accounting audit does not erase the real 2025 daily inventory failures or inherited weekly exit failures.','audit_output_sha256':{p.name:sha(p) for p in [REPORT/'independent_execution_audit.json',REPORT/'independent_execution_audit.md',REPORT/'independent_cross_day_execution_2025.json',REPORT/'independent_cross_day_execution_2025.md']},'snapshot_verified_count':len(declared),'current_snapshot_match_count':len(declared)-len(changed),'current_changed_files':changed,'postrun_report_revision_sha256':sha(revision),'original_report_inputs_verified_unchanged':len(rev['protected_original_files_sha256']),'report_stale_fields_equal_independent_exact_timestamp':True,'all_selected_rule_hashes_frozen_before_prediction_files':True,'selected_model_copy_count':len(audit['model_isolation_audit_source']['selected_models']),'annual':annual,'totals':audit['totals'],'sum_independent_annual_increment_CNY':sum(r['increment_CNY'] for r in annual),'supplementary_archive_sha256':{p.name:sha(p) for p in archive.iterdir() if p.is_file()}}
(archive/'final_review.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
print(json.dumps(value,ensure_ascii=False,indent=2))
