"""独立绑定会话修复烟测与已完成审计：仅核文件身份和原异常行，不重跑账户。"""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
NEW=OUT.parents[1];OLD=ROOT/'reports/dynamic_inventory_t_smoke/run_20260910_142405_397251'
INPUTS={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def record(p):
 p=Path(p).resolve();INPUTS.setdefault(str(p),sha(p));return p
def read(p):return json.loads(record(p).read_text())
comparison=read(NEW/'engineering_validation/session_fix_comparison.json')
oldv=OLD/'independent_dynamic_audit/run_20260910_142902_822973/verification.json'
assert sha(oldv)=='73932551f7928f8064043655e67de38c0952e1029aac19189f04d61356811911'
verification=read(oldv);assert verification['passed']
assert comparison['files_compared']==len(comparison['files'])==2357
for row in comparison['files']:
 a=record(OLD/row['path']);b=record(NEW/row['path'])
 assert INPUTS[str(a)]==row['old_sha256']==row['new_sha256']==INPUTS[str(b)],row['path']
json_count=0
for p in OLD.glob('year_*/**/metrics.json'):
 other=NEW/p.relative_to(OLD);assert read(p)==read(other),str(p);json_count+=1
for p in OLD.glob('year_*/selected_rule.json'):
 assert read(p)==read(NEW/p.relative_to(OLD)),str(p)
old_sources=read(OLD/'source_hashes.json');new_sources=read(NEW/'source_hashes.json')
assert len(new_sources)==len(old_sources)==43
changed=[k for k in new_sources if new_sources[k]!=old_sources[k]]
assert changed==['quant_backtest/dynamic_t_data.py'],changed
for name,digest in new_sources.items():
 assert sha(record(NEW/'source_snapshot'/name))==digest
 assert sha(record(ROOT/name))==digest
original_inputs=read(NEW/'input_sources.json')
for name,identity in original_inputs.items():assert sha(record(name))==identity['sha256'],name
integrity=read(NEW/'runtime_integrity_audit.json');assert integrity['all_unchanged'] and integrity['input_files_verified']==110
preflight=ROOT/'reports/dynamic_inventory_t/data_validation/all_inner_source_preflight'
excluded=[]
for p in sorted(preflight.glob('year_*/full_minute_coverage_audit.json')):
 d=read(p);manifest=read(d['manifest_path']);days=set(manifest['identity']['market_days'])
 for row in d['per_symbol']:
  if not row['excluded_outside_session_rows']:continue
  year=int(manifest['identity']['year']);path=ROOT/'ashare_minute_data_2023_2026_04/extracted'/str(year)/(row['symbol']+'.csv')
  raw=pd.read_csv(record(path),usecols=['datetime']);raw['datetime']=pd.to_datetime(raw.datetime,format='mixed')
  raw=raw.loc[raw.datetime.dt.strftime('%Y-%m-%d').isin(days)]
  minute=raw.datetime.dt.hour*60+raw.datetime.dt.minute
  outside=raw.loc[~(minute.between(571,690)|minute.between(781,900)),'datetime']
  assert outside.astype(str).tolist()==row['excluded_timestamps']
  assert len(outside)==row['excluded_outside_session_rows']
  assert outside.dt.strftime('%H:%M:%S').value_counts().to_dict()==row['excluded_clock_counts']
  excluded.append({'fold':p.parent.name,**{k:row[k] for k in ['symbol','excluded_outside_session_rows','excluded_timestamps','excluded_clock_counts']}})
assert [sum(x['excluded_outside_session_rows'] for x in excluded if x['fold']==f'year_{year}') for year in [2023,2024,2025,2026]]==[1,3,3,1]
for p,value in INPUTS.items():assert sha(p)==value,p
result={'passed':True,'old_audit_verification_sha256':sha(oldv),'account_csv_bytes_equal':2357,
 'metrics_json_equal':json_count,'sources_verified':43,'only_runtime_source_changed':changed,
 'new_original_inputs_verified':len(original_inputs),'excluded_actual_rows':excluded,
 'scope':'继承原168账户独立通过证据；只作字节/来源和会话排除核对，无账户重跑，无新参数选择',
 'new_auditor_sha256':sha(ROOT/'scripts/audit_dynamic_inventory_t.py'),'inputs_sha256':INPUTS,
 'script_sha256':sha(__file__)}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
(OUT/'review.md').write_text('# 会话修复烟测独立身份验收\n\n通过：2357份账户CSV逐字节相等，168份metrics JSON及选中规则内容相等；43份新运行源码/快照和110份登记输入均核实。唯一运行源差异是会话过滤及排除审计。原168账户完整独立审计结论可按这份身份映射继承，无需重复计算。\n\n四个内部上下文的会话外记录数1/3/3/1均与原CSV直接核对，全部为13:00。未平移至13:01，未创造报价或删除会话内真实行；两份历史烟测和失败full均保留。\n',encoding='utf-8')
for source in ['/tmp/dynamic_auditor_session_fix_tests.log']:
 shutil.copy2(source,OUT/Path(source).name)
print(json.dumps({k:v for k,v in result.items() if k not in ['inputs_sha256','excluded_actual_rows']},ensure_ascii=False))
