"""只读校验原14:55标签；按固定首8代码和core2024/25取样，不拟合、不读账户收益。"""
from pathlib import Path
from datetime import datetime
import hashlib,json
import numpy as np
import pandas as pd
cache=Path('reports/weekly_t_cross_validation/cache/weekly_t_9b6b9bfb3309dd53')
metadata=json.loads((cache/'metadata.json').read_text())
known=json.loads(Path('reports/weekly_t_cross_validation/run_20260909_182945_388303/input_sources.json').read_text())
out=Path('reports/weekly_t_goal_diagnosis')/f'close_label_feasibility_{datetime.now():%Y%m%d_%H%M%S}'
out.mkdir()
Path('/tmp/close_label_feasibility_path.txt').write_text(str(out))
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):d.update(block)
 return d.hexdigest()
path=cache/'intraday.pkl';expected=known[str(path.resolve())]
assert sha(path)==expected['sha256']
rows=[];inputs={str(path):expected['sha256'],str(cache/'metadata.json'):sha(cache/'metadata.json')}
raw_checks=[]
for symbol in metadata['symbols'][:8]:
 p=cache/f'{symbol}.pkl';inputs[str(p)]=sha(p)
 frame=pd.read_pickle(p)['intraday']
 frame=frame.loc[frame.datetime.dt.year.isin([2024,2025])]
 assert not frame.duplicated(['datetime','symbol']).any()
 assert frame.execution_time.eq(frame.datetime+pd.Timedelta(minutes=1)).all()
 assert frame.recovery_time.eq(frame.datetime.dt.normalize()+pd.Timedelta(hours=14,minutes=55)).all()
 assert frame.label_start.eq(frame.execution_time).all() and frame.label_end.eq(frame.recovery_time).all()
 expected_return=frame.recovery_price/frame.execution_price-1
 np.testing.assert_array_equal(expected_return.to_numpy(),frame.target_return.to_numpy())
 for year, part in frame.groupby(frame.datetime.dt.year):
  rows.append({'symbol':symbol,'year':year,'current_signal_rows':len(part),
               'missing_execution_price_rows':int(part.execution_price.isna().sum()),
               'missing_recovery_price_rows':int(part.recovery_price.isna().sum()),
               'missing_target_rows':int(part.target_return.isna().sum()),
               'finite_target_despite_zero_execution_volume':int((np.isfinite(part.target_return)&part.execution_volume.eq(0)).sum()),
               'finite_target_despite_zero_recovery_volume':int((np.isfinite(part.target_return)&part.recovery_volume.eq(0)).sum())})
 # 原始行情只校验首代码的2个core年，以端点数值误差作为证据，不输出盈利统计。
 if symbol!=metadata['symbols'][0]:continue
 for year, part in frame.groupby(frame.datetime.dt.year):
  rawpath=Path(metadata['data_root'])/str(year)/f'{symbol}.csv';inputs[str(rawpath)]=sha(rawpath)
  raw=pd.read_csv(rawpath,usecols=['datetime','open','high','low','close','volume','amount'])
  raw.datetime=pd.to_datetime(raw.datetime)
  raw=raw.sort_values('datetime',kind='stable').drop_duplicates('datetime',keep='last')
  nums=['open','high','low','close','volume','amount']
  raw[nums]=raw[nums].apply(pd.to_numeric,errors='coerce')
  ok=np.isfinite(raw[nums]).all(axis=1)&raw[['open','high','low','close']].gt(0).all(axis=1)
  ok&=raw[['volume','amount']].ge(0).all(axis=1)&raw.high.ge(raw.low)
  ok&=raw.high.add(1e-6).ge(raw[['open','close']].max(axis=1))
  ok&=raw.low.sub(1e-6).le(raw[['open','close']].min(axis=1))
  minute=raw.datetime.dt.hour*60+raw.datetime.dt.minute
  raw=raw.loc[ok&(minute.between(571,690)|minute.between(781,900))].set_index('datetime')
  errors=[]
  for prefix in ['execution','recovery']:
   for column in ['close','high','low','volume','amount']:
    actual=raw[column].reindex(pd.DatetimeIndex(part[prefix+'_time'])).to_numpy()
    saved=part[prefix+('_price' if column=='close' else '_'+column)].to_numpy()
    assert np.array_equal(np.isnan(actual),np.isnan(saved))
    finite=np.isfinite(actual)&np.isfinite(saved)
    assert np.allclose(actual[finite],saved[finite],rtol=0,atol=1e-10)
    errors.append(float(np.max(np.abs(actual[finite]-saved[finite]))) if finite.any() else 0.)
  raw_checks.append({'symbol':symbol,'year':int(year),'current_rows':len(part),'endpoint_field_comparisons':len(part)*10,
                     'max_abs_endpoint_error':max(errors),'missing_patterns_exact':True})
summary={'passed':True,'scope':'read_only_first_8_symbols_core_2024_2025_only_no_account_returns',
         'aggregate_intraday_sha_matches_v11_audited_input':True,'aggregate_rows_declared':metadata['intraday_rows'],
         'aggregate_label_contract':metadata['intraday_label'],'aggregate_calendar':metadata['calendar'],
         'sample_rows':sum(x['current_signal_rows'] for x in rows),'sample':rows,'raw_checks':raw_checks,
         'labels_are_executable_fill_guarantees':False,'full_missing_label_census_performed':False,
         'model_fit_or_parameter_selection':False,'original_sources_modified':False,'inputs_sha256':inputs}
pd.DataFrame(rows).to_csv(out/'sample_counts.csv',index=False)
(out/'checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'out':str(out),'sample_rows':summary['sample_rows'],'missing_execution':sum(x['missing_execution_price_rows'] for x in rows),'missing_recovery':sum(x['missing_recovery_price_rows'] for x in rows),'finite_on_zero_recovery_volume':sum(x['finite_target_despite_zero_recovery_volume'] for x in rows),'raw_checks':raw_checks},ensure_ascii=False,indent=2),flush=True)
