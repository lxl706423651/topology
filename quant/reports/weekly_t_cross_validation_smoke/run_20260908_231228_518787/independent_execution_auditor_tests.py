"""在固定真实早恢复诊断的临时副本验证独立审计器及故意损坏检测，不碰正式报告。"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path('/data/lxl/seed-emulator/topology/quant')
sys.path.insert(0,str(ROOT))
from scripts.audit_weekly_t_train_domain_execution import run_audit
import pandas as pd

SOURCE=ROOT/'reports/weekly_t_cross_validation_smoke/real_t_v5_early_execution'
SMOKE=ROOT/'reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787'

def fixture(directory,direction):
    directory.mkdir()
    out=directory/'year_2024';out.mkdir()
    for path in (SOURCE/direction).glob('*.csv'):shutil.copyfile(path,out/path.name)
    shutil.copyfile(SOURCE/'weekly_selections.csv',out/'weekly_selections.csv')
    cfg=json.loads((SOURCE/'resolved_config.json').read_text());cfg.update(max_symbols=0,smoke_weeks=0,test_years=[2024],research_version='weekly_t_v7_train_domain',t_training_cohort='core_year_oof_weekly_top100_middle_days',t_training_domain='known_slot_and_recovery_deadline')
    frozen=directory/'frozen/year_2024';frozen.mkdir(parents=True)
    shutil.copyfile(out/'test_equity_curve.csv',frozen/'test_equity_curve.csv')
    cfg['frozen_selection_report_dir']=str(frozen.parent)
    (directory/'resolved_config.json').write_text(json.dumps(cfg))
    shutil.copyfile(SMOKE/'data_manifest.json',directory/'data_manifest.json')
    shutil.copyfile(SMOKE/'source_hashes.json',directory/'source_hashes.json')
    shutil.copytree(SMOKE/'source_snapshot',directory/'source_snapshot')
    (out/'selected_rule.json').write_text(json.dumps({'enabled':True,'horizon_minutes':20,'direction':direction}))
    (directory/'all_rules_frozen_before_outer_test.json').write_text(json.dumps({'test_years':[2024],
        'selected_rule_sha256':{'2024':hashlib.sha256((out/'selected_rule.json').read_bytes()).hexdigest()}}))
    (directory/'final_report.md').write_text('Completed temporary harness of the predeclared real early recovery diagnostic.')
    return directory

with tempfile.TemporaryDirectory(prefix='v7_auditor_harness_') as temporary:
    temporary=Path(temporary);results=[]
    for direction in ['sell_first','buy_first']:
        directory=fixture(temporary/direction,direction)
        with redirect_stdout(io.StringIO()):audit=run_audit(directory)
        assert audit['totals']['completed_pairs']==12 and audit['totals']['fragments']==27
        results.append({'case':direction,'pass':True,'independent_increment':audit['sum_independent_annual_T_increment_CNY']})
    checks={
      'fee_corruption':('year_2024/test_trade_events.csv','fee',lambda frame:frame.index[0],1.),
      'cash_corruption':('year_2024/test_trade_events.csv','cash_after',lambda frame:frame.index[0],1.),
      'equity_corruption':('year_2024/test_equity_curve.csv','equity',lambda frame:frame.index[0],1.),
      'baseline_corruption':('frozen/year_2024/test_equity_curve.csv','baseline_equity',lambda frame:frame.index[0],1.),
      'pending_peak_corruption':('year_2024/test_metrics.csv','peak_pending_t_notional',lambda frame:frame.index[0],1.),
    }
    for name,(filename,column,index,delta) in checks.items():
        directory=fixture(temporary/name,'sell_first');path=directory/filename
        frame=pd.read_csv(path);frame.loc[index(frame),column]+=delta;frame.to_csv(path,index=False)
        try:
            with redirect_stdout(io.StringIO()):run_audit(directory)
        except AssertionError:results.append({'case':name,'detected':True})
        else:raise AssertionError('corruption missed '+name)
    directory=fixture(temporary/'planned_corruption','sell_first');path=directory/'year_2024/test_trade_events.csv'
    frame=pd.read_csv(path);mask=frame.event_type.eq('t')&frame.fragment_id.eq(0)
    frame.loc[mask,'planned_recovery_time']=pd.to_datetime(frame.loc[mask,'planned_recovery_time'])-pd.Timedelta(minutes=1)
    frame.to_csv(path,index=False)
    try:
        with redirect_stdout(io.StringIO()):run_audit(directory)
    except AssertionError:results.append({'case':'planned_corruption','detected':True})
    else:raise AssertionError('invalid planned horizon missed')
    try:run_audit(temporary/'not_complete')
    except ValueError:results.append({'case':'unfinished_report_blocked_before_read','detected':True})
    else:raise AssertionError('unfinished report read')
    directory=fixture(temporary/'wrong_version','sell_first')
    config_path=directory/'resolved_config.json';cfg=json.loads(config_path.read_text());cfg['research_version']='unknown';config_path.write_text(json.dumps(cfg))
    try:run_audit(directory)
    except ValueError:results.append({'case':'wrong_engine_protocol_blocked','detected':True})
    else:raise AssertionError('incompatible version accepted')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    (SMOKE/'independent_train_domain_execution_tests.json').write_text(json.dumps({'classification':'auditor_only_not_profit_evidence','cases':results,
          'source_sha256':hashlib.sha256((ROOT/'scripts/audit_weekly_t_train_domain_execution.py').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
