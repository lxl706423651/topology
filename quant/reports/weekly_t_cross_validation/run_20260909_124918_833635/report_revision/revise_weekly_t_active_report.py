"""正式运行结束后仅从CSV重新渲染主动报告，备份旧产物并严格保护原始账本。"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, sys
sys.path.insert(0, '/data/lxl/seed-emulator/topology/quant')
import pandas as pd
from quant_backtest.weekly_t_active_reporting import RESULT_FILES, write_active_report
ROOT=Path('/data/lxl/seed-emulator/topology/quant')
SOURCE=ROOT/'reports/weekly_t_cross_validation/run_20260909_124918_833635'
REVISION=SOURCE/'report_revision'
STAGE=REVISION/'staged_report'
REPORTER=ROOT/'quant_backtest/weekly_t_active_reporting.py'
DERIVED=['annual_summary.csv','test_metrics.csv','goal_evaluation.json','active_reporting_status.json',
         'test_return_curve.png','测试集收益曲线.png','t_increment_curve.png','final_report.md','输出文件说明.md']
STALE=['final_stale_position_count','final_stale_market_value']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_frame(path, dates=False):
    try: frame=pd.read_csv(path,float_precision='round_trip')
    except pd.errors.EmptyDataError:return pd.DataFrame()
    if dates:
        for c in ['datetime','date','first_datetime','second_datetime','mark_datetime','signal_datetime',
                  'entry_date','exit_date','trade_week','planned_recovery_time']:
            if c in frame:frame[c]=pd.to_datetime(frame[c],errors='raise',format='mixed')
    return frame

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')

if (REVISION/'manifest.json').exists() or STAGE.exists():raise FileExistsError('禁止覆盖既有修订')
protocol=json.loads((REVISION/'protocol.json').read_text())
original_files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and REVISION not in p.parents}
original_sources=json.loads((SOURCE/'source_hashes.json').read_text())
reporter_key='quant_backtest/weekly_t_active_reporting.py'
assert original_sources[reporter_key]==protocol['old_reporter_sha256']
current_sources={name:sha(ROOT/name) for name in original_sources}
changed_sources=[name for name in original_sources if original_sources[name]!=current_sources[name]]
assert changed_sources==[reporter_key],changed_sources
for name,digest in original_sources.items():assert sha(SOURCE/'source_snapshot'/Path(name).name)==digest
for name in DERIVED:shutil.copy2(SOURCE/name,REVISION/'previous'/name)
shutil.copy2(REPORTER,REVISION/'weekly_t_active_reporting_revised.py')
shutil.copy2(ROOT/'scripts/test_weekly_t_active_reporting.py',REVISION/'test_weekly_t_active_reporting.py')
shutil.copy2(__file__,REVISION/'revise_weekly_t_active_report.py')
STAGE.mkdir()
cfg=json.loads((SOURCE/'resolved_config.json').read_text())
annual=read_frame(SOURCE/'annual_summary.csv');validation=read_frame(SOURCE/'validation_rule_metrics.csv')
results=[];render_inputs={}
for year in annual.test_year:
    y=int(year); origin=SOURCE/f'year_{y}'; target=STAGE/f'year_{y}';target.mkdir()
    result={}
    for key,name in RESULT_FILES.items():
        path=origin/name
        if path.exists():
            result[key]=read_frame(path,dates=True);render_inputs[str(path.relative_to(SOURCE))]=sha(path)
    metric_path=origin/'test_metrics.csv';result['metrics']=read_frame(metric_path).iloc[0].to_dict()
    render_inputs[str(metric_path.relative_to(SOURCE))]=sha(metric_path)
    results.append((y,result))
    for name in ['t_predictions.csv','t_decisions.csv']:
        shutil.copy2(origin/name,target/name);render_inputs[str((origin/name).relative_to(SOURCE))]=sha(origin/name)
write_active_report(STAGE,cfg,annual,results,validation)
new=read_frame(STAGE/'annual_summary.csv')
assert set(new.columns)==set(annual.columns),(set(new.columns)-set(annual.columns),set(annual.columns)-set(new.columns))
cols=[c for c in annual.columns if c not in STALE]
pd.testing.assert_frame_equal(annual[cols],new[cols],check_exact=True,check_dtype=False)
pd.testing.assert_frame_equal(read_frame(SOURCE/'test_metrics.csv')[cols],read_frame(STAGE/'test_metrics.csv')[cols],check_exact=True,check_dtype=False)
stale_changes=[]
for oldrow,newrow in zip(annual.to_dict('records'),new.to_dict('records')):
    stale_changes.append({'test_year':int(oldrow['test_year']),'before':{k:oldrow[k] for k in STALE},
                          'after':{k:newrow[k] for k in STALE},'changed':any(oldrow[k]!=newrow[k] for k in STALE)})
# 原始聚合/逐年账本只读取，即使stage生成相同文件，也绝不发布这些文件。
for name,digest in original_files.items():assert sha(SOURCE/name)==digest,name
planned={name:{'before_sha256':sha(SOURCE/name),'after_sha256':sha(STAGE/name),
               'old_copy':str((REVISION/'previous'/name).relative_to(SOURCE))} for name in DERIVED}
changed=[]
for name,item in planned.items():
    if item['before_sha256']!=item['after_sha256']:
        shutil.copy2(STAGE/name,SOURCE/name);changed.append(name)
for name,digest in original_files.items():
    if name not in changed:assert sha(SOURCE/name)==digest,name
for name,digest in original_sources.items():assert sha(SOURCE/'source_snapshot'/Path(name).name)==digest
protected=[name for name in original_files if name not in DERIVED]
manifest={'classification':'postrun_reporting_only_stale_timestamp_revision','completed_at_utc':datetime.now(timezone.utc).isoformat(),
          'account_rerun':False,'model_refit':False,'rendered_from_saved_csv':True,
          'original_reporter_sha256':protocol['old_reporter_sha256'],'revised_reporter_sha256':sha(REPORTER),
          'original_runtime_source_count':len(original_sources),'original_runtime_sources':original_sources,
          'runtime_source_snapshot_unchanged':True,'current_source_match_count':len(original_sources)-len(changed_sources),
          'current_changed_source_files':changed_sources,'source_change_phase':'after_original_runtime_exit0_integrity_check',
          'original_runtime_integrity_files':[str(p.relative_to(SOURCE)) for p in SOURCE.glob('*integrity*')],
          'derived_files':planned,'republished_files':changed,'annual_nonstale_columns_exact':True,
          'allowed_annual_change_columns':STALE,'annual_stale_changes':stale_changes,
          'protected_original_files_sha256':{name:original_files[name] for name in protected},
          'all_protected_original_files_unchanged':True,'render_inputs_sha256':render_inputs,
          'staged_aggregate_trade_curve_files_not_published':True,'regression_fixture_cases':4,
          'stale_definition':'actual shares>0 and mark_datetime != final curve datetime; same-day open-only prices are stale',
          'interpretation':'23-source snapshot remains original completed runtime; current reporter changed only postrun reporting, not model/execution'}
dump(REVISION/'manifest.json',manifest)
lines=['# 主动报告结束后修订记录','',
       '只修复报告按自然日比较价格时间的边界，改为逐仓 mark_datetime 与最后权益估值时点精确比较。',
       '同日只有开盘价、缺少收盘价的持仓现在也计为陈旧估值。四项合成持仓测试通过，没有重跑账户或训练。','',
       f'原reporter SHA：`{protocol["old_reporter_sha256"]}`；修订后：`{sha(REPORTER)}`。',
       f'原运行23份source_snapshot全部保持原身份；当前22/23相同，仅reporter在正式结束后修订。',
       f'实际重新发布的文件：{changed or "无（本次正式数据的报告输出全部逐字节相同）"}。',
       '所有潜在替换报告已备份previous/，每个文件的前后SHA见manifest.json。原始逐年及聚合成交、曲线、预测、决策等文件未写入。','',
       '|年度|原陈旧项/市值|新陈旧项/市值|','|---|---:|---:|']
for item in stale_changes:
    a,b=item['before'],item['after'];lines.append(f"|{item['test_year']}|{a[STALE[0]]}/{a[STALE[1]]}|{b[STALE[0]]}/{b[STALE[1]]}|")
lines+=['','除两个陈旧估值字段外，年度汇总所有字段按浮点round_trip逐值精确一致；实际净值、收益、交易及失败没有改变。',
        '本轮是否实际出现同日旧价应以表格差异和随后独立执行审计为准，不把修复通用边界当成更改本轮收益。']
(REVISION/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'passed':True,'changed':changed,'stale_changes':stale_changes,'protected_files':len(protected),
                  'reporter_sha256':sha(REPORTER)},ensure_ascii=False,indent=2))
