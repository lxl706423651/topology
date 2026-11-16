"""一次性只读验证8股core年close接口与8份完整lineage；不重跑raw审计/市场规划。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path.cwd()))
from quant_backtest.inventory_t_close_labels import extract_fixed_close_labels,attach_fixed_close_labels
root=Path.cwd();source=root/'reports/inventory_learning_t/run_20260910_151232_841321'
cfg=json.loads((source/'resolved_config.json').read_text());cache=Path(cfg['market_cache_dir'])
prior=root/'reports/weekly_t_goal_diagnosis/close_label_feasibility_20260910_151622/checks.json'
known=json.loads(prior.read_text());metadata=json.loads((cache/'metadata.json').read_text())
out=root/'reports/weekly_t_goal_diagnosis'/f'close_labels_real_and_lineage_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir()
Path('/tmp/close_labels_real_and_lineage_path.txt').write_text(str(out))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(8<<20),b''):h.update(part)
 return h.hexdigest()
def frame_sha(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()
inputs={str(prior):sha(prior),str(source/'resolved_config.json'):sha(source/'resolved_config.json')}
lineages=[];rows=[]
time_cols=['datetime','feature_source_start','intraday_feature_source_start','entry_feature_source_start','entry_rank_source_start','portfolio_cash_source_start','base_entry_datetime']
for fold in cfg['test_years']:
 parent=source/f'year_{fold}/training_cohort/inventory_learning_cohort'
 parentmeta=json.loads((parent/'cohort_metadata.json').read_text());inputs[str(parent/'cohort_metadata.json')]=sha(parent/'cohort_metadata.json')
 assert parentmeta['labels_attached'] is False and parentmeta['future_fill_or_return_filter'] is False and parentmeta['core_sampling_performed'] is False
 foldcount=0
 for core in parentmeta['core_years']:
  p=parent/f'core_{core}/cohort_lineage.csv';m=p.parent/'metadata.json'
  inputs[str(p)]=sha(p);inputs[str(m)]=sha(m)
  lineage=pd.read_csv(p,float_precision='round_trip')
  for col in time_cols:lineage[col]=pd.to_datetime(lineage[col],format='mixed')
  meta=json.loads(m.read_text())
  assert frame_sha(lineage)==meta['cohort_lineage_sha256'],(fold,core,'frame SHA')
  assert len(lineage)==meta['training_opportunity_rows']
  assert not lineage.duplicated(['datetime','symbol']).any()
  assert lineage.baseline_opening_shares.gt(0).all()
  assert not any(c.startswith(('execution_','recovery_','target_','label_')) for c in lineage)
  assert not {'trade_week_complete','horizon_minutes'}.intersection(lineage)
  rows.append({'fold':fold,'core_year':core,'rows':len(lineage),'columns':len(lineage.columns),'file':str(p),'file_sha256':inputs[str(p)],'frame_sha256':meta['cohort_lineage_sha256'],'frame_sha_matches':True,'earliest_signal':str(lineage.datetime.min()),'latest_signal':str(lineage.datetime.max()),'missing_feature_source_start':int(lineage.feature_source_start.isna().sum())})
  foldcount+=len(lineage)
  if fold==2023:lineages.append(lineage)
 assert foldcount==parentmeta['training_opportunity_rows']
# 完全相同的固定首8股票和2024/2025样本；不读原CSV，不根据结果扩充样本。
parts=[]
for symbol in metadata['symbols'][:8]:
 p=cache/f'{symbol}.pkl';actual=sha(p)
 key=str(p.relative_to(root))
 assert actual==known['inputs_sha256'][key]
 inputs[str(p)]=actual
 frame=pd.read_pickle(p)['intraday']
 parts.append(frame.loc[frame.datetime.dt.year.isin([2024,2025])])
sample=pd.concat(parts,ignore_index=True);assert len(sample)==31040
labels,audit=extract_fixed_close_labels(sample,cfg)
assert len(labels)==31040 and audit['finite_reference_labels_with_zero_execution_volume']==21
np.testing.assert_array_equal(labels.target_return.to_numpy(),(sample.recovery_price/sample.execution_price-1).to_numpy())
zero=sample.execution_volume.eq(0);assert labels.loc[zero,'label_available'].all()
# 实际当前因子类型通过接口不变；这份fixture保留全部8股，不声称其全是实际持仓。
current_cols=[c for c in sample if not c.startswith(('execution_','recovery_','target_','label_')) and c!='trade_week_complete']
current=sample[current_cols].copy();same=attach_fixed_close_labels(current,labels,cfg)
pd.testing.assert_frame_equal(current,same[current.columns],check_exact=True)
# 用首折真正未联标签的全部lineage验证rank/cash来源，不把h20有效行当全集。
core=pd.concat(lineages,ignore_index=True).sort_values(['datetime','symbol'],kind='stable').reset_index(drop=True)
attached=attach_fixed_close_labels(core,labels.iloc[1:],cfg)
pd.testing.assert_frame_equal(core,attached[core.columns],check_exact=True)
assert len(attached)==len(core)
matched=pd.MultiIndex.from_frame(core[['datetime','symbol']]).isin(pd.MultiIndex.from_frame(labels.iloc[1:][['datetime','symbol']]))
assert attached.loc[~matched,['execution_price','recovery_price','target_return']].isna().all().all()
assert not attached.loc[~matched,'label_available'].any()
assert attached.loc[~matched,'label_end'].dt.strftime('%H:%M').eq('14:55').all()
# 缺一条当前8股标签必须不删当前行，独立于是否曾进入纯核心持仓。
missing=attach_fixed_close_labels(current,labels.iloc[1:],cfg)
assert len(missing)==len(current) and pd.isna(missing.iloc[0].target_return) and not missing.iloc[0].label_available
pd.testing.assert_frame_equal(current,missing[current.columns],check_exact=True)
verification=list((source/'independent_learning_audit').glob('run_*/verification.json'))
complete=[]
for path in verification:
 value=json.loads(path.read_text())
 if value.get('passed') is True and value.get('complete') is True and value.get('all_read_inputs_unchanged') is True:complete.append((path,value))
anchored=False
if complete:
 path,value=sorted(complete,key=lambda item:str(item[0]))[-1]
 for row in rows:assert value['inputs_sha256'][row['file']]==row['file_sha256']
 anchored=True;inputs[str(path)]=sha(path)
summary={'passed':True,'scope':'fixed_first8_symbols_core2024_2025_close_data_interface_and_saved_lineage_identity_only','real_sample_rows':len(sample),'zero_next_volume_labels_retained':int(zero.sum()),'target_max_abs_error':float(np.nanmax(np.abs(labels.target_return-(sample.recovery_price/sample.execution_price-1)))),'current_factor_float32_columns_preserved':[c for c in current if current[c].dtype==np.dtype('float32')],'first_fold_full_unlabelled_opportunities':len(core),'first_fold_full_lineage_sources_and_dtypes_preserved':True,'first_fold_missing_sample_label_keys_kept':int((~matched).sum()),'all_8_lineage_files_metadata_frame_sha_match':True,'full_v14_independent_audit_currently_available':anchored,'full_v14_audit_anchor_required_before_automated_reuse':True,'read_raw_price_csv':False,'reran_permanent_baseline_planner':False,'models_fit':0,'read_outer_returns':False,'inputs_sha256':inputs,'extraction_audit':audit,'lineage_files':rows}
pd.DataFrame(rows).to_csv(out/'lineage_files.csv',index=False)
(out/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['passed','real_sample_rows','zero_next_volume_labels_retained','target_max_abs_error','first_fold_full_unlabelled_opportunities','first_fold_missing_sample_label_keys_kept','all_8_lineage_files_metadata_frame_sha_match','full_v14_independent_audit_currently_available']},ensure_ascii=False,indent=2))
print(out)
