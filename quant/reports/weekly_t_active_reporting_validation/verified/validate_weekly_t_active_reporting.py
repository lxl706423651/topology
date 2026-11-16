"""只读既有A/B结果验证主动报告；不重新训练或运行账户。"""
from pathlib import Path
import copy, hashlib, json, pickle, shutil, sys, tempfile
sys.path.insert(0, '/data/lxl/seed-emulator/topology/quant')
import pandas as pd
from quant_backtest.weekly_t_active_reporting import RESULT_FILES, write_active_report
p=Path('/data/lxl/seed-emulator/topology/quant/reports/weekly_t_zero_increment_diagnosis/gate_toggle_2026')
def frame(path):
    try: f=pd.read_csv(path,float_precision='round_trip')
    except pd.errors.EmptyDataError: return pd.DataFrame()
    for c in ['datetime','date','mark_datetime','first_datetime','second_datetime']:
        if c in f: f[c]=pd.to_datetime(f[c])
    return f
result={k:frame(p/'b'/v) for k,v in RESULT_FILES.items()}
metrics=frame(p/'b/test_metrics.csv').iloc[0].to_dict();result['metrics']=metrics
annual=pd.DataFrame([{**metrics,'test_year':2026,'t_enabled':True,'prediction_count':35960,'signal_count':21,'request_count':18}])
cfg=json.loads((p/'resolved_config.json').read_text());cfg.update(candidate_activation_policy='active_execution_valid',research_version='weekly_t_v9_active_execution_valid')
validation=frame(Path('/data/lxl/seed-emulator/topology/quant/reports/weekly_t_cross_validation/run_20260908_235818_404243/year_2026/validation_rule_metrics.csv'))
inputs=(cfg,annual,result,validation);before=copy.deepcopy(inputs)
checks=[]
with tempfile.TemporaryDirectory(prefix='weekly_t_active_reporting_') as work:
    out=Path(work)/'successful_report';(out/'year_2026').mkdir(parents=True)
    shutil.copyfile(p/'b/t_decisions.csv',out/'year_2026/t_decisions.csv')
    shutil.copyfile('/data/lxl/seed-emulator/topology/quant/reports/weekly_t_cross_validation/run_20260908_235818_404243/year_2026/t_predictions.csv',out/'year_2026/t_predictions.csv')
    write_active_report(out,cfg,annual,[(2026,result)],validation)
    status=json.loads((out/'goal_evaluation.json').read_text());summary=pd.read_csv(out/'annual_summary.csv').iloc[0]
    assert status['all_years_t_enabled'] is True and status['affects_activation'] is False
    assert status['profitability_reference_met'] is False and status['t_execution_contract_met'] is False
    assert summary.t_first_leg_count==9 and summary.t_recovery_leg_count==9 and summary.t_pair_count==9
    assert summary.inventory_mismatch_days==2 and summary.recovery_failure_days==1
    assert abs(summary.t_added_equity+32.739287754986435)<1e-9
    assert abs(summary.t_fee-41.951787755)<1e-9 and abs(summary.t_slippage_cost-19.7875)<1e-9
    assert summary.prediction_row_count==35960 and summary.t_request_count==18
    assert abs(summary.realized_pair_reference_gross_pnl-29)<1e-9
    assert len(frame(out/'test_trade_events.csv'))==3187 and len(frame(out/'baseline_trade_events.csv'))==3169
    checks.append('existing_real_AB_negative_PnL_active_counts_fees_failures_csv_and_plot')
    disabled=annual.copy();disabled['t_enabled']=False
    try: write_active_report(Path(work)/'disabled',cfg,disabled,[(2026,result)],validation)
    except ValueError as e: assert 't_enabled=True' in str(e)
    else: raise AssertionError('silent active relabel')
    checks.append('disabled_annual_not_silently_relabeled_active')
    broken=copy.deepcopy(result);broken['metrics']['t_pair_count']=10
    try: write_active_report(Path(work)/'broken',cfg,annual,[(2026,broken)],validation)
    except ValueError as e: assert '完整配对' in str(e)
    else: raise AssertionError('pair count mismatch not rejected')
    checks.append('pair_metric_vs_raw_fragments_mismatch_rejected')
    destination=Path('/data/lxl/seed-emulator/topology/quant/reports/weekly_t_active_reporting_validation/verified')
    if destination.exists(): raise FileExistsError(destination)
    shutil.copytree(out,destination)
assert cfg==before[0]
pd.testing.assert_frame_equal(annual,before[1],check_exact=True)
pd.testing.assert_frame_equal(validation,before[3],check_exact=True)
for key,value in result.items():
    if isinstance(value,pd.DataFrame): pd.testing.assert_frame_equal(value,before[2][key],check_exact=True)
    else: assert value==before[2][key]
checks.append('all_input_objects_unchanged')
proof={'passed':True,'checks':checks,'inputs_logically_unchanged':True,'module_sha256':hashlib.sha256(Path('/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_active_reporting.py').read_bytes()).hexdigest(),'no_account_or_training_rerun':True,'fixture':'already_completed_gate_toggle_2026_B'}
(destination/'reporting_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
shutil.copyfile(__file__,destination/'validate_weekly_t_active_reporting.py')
print(json.dumps(proof,indent=2))
