"""只读复用首折4股真实OOF核心样本；临时双头拟合，不读取外层或改缓存。"""
from pathlib import Path
import sys,json,hashlib,tempfile,time
sys.path.insert(0,'/data/lxl/seed-emulator/topology/quant')
import joblib
import numpy as np
import pandas as pd
from quant_backtest.weekly_t_timing_models import fit_timing_model,predict_timing_model,TARGET_MODE
from quant_backtest.weekly_t_short_horizon import attach_horizon_labels
base=Path('/data/lxl/seed-emulator/topology/quant/reports/weekly_t_cross_validation')
source=base/'run_20260908_235818_404243/year_2023'
lineage_path=source/'oof_weekly_cohort/cohort_lineage.csv'
lineage=pd.read_csv(lineage_path,parse_dates=['datetime','feature_source_start'])
symbols=sorted(lineage.symbol.unique())[:4]
inputs={str(lineage_path):hashlib.sha256(lineage_path.read_bytes()).hexdigest()}
parts=[];label_parts=[]
for symbol in symbols:
 for directory,key in [(base/'cache/weekly_t_9b6b9bfb3309dd53','intraday'),(base/'horizon_cache/horizon_22363501d63027ff','labels')]:
  path=directory/(symbol+'.pkl');inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();payload=pd.read_pickle(path)
  if key=='intraday':parts.append(payload[key])
  else:label_parts.append(payload[key])
  del payload
frame=pd.concat(parts,ignore_index=True).merge(lineage[lineage.symbol.isin(symbols)],on=['datetime','symbol'],how='inner',validate='one_to_one')
domain_path=source/'training_domain_h20.csv';inputs[str(domain_path)]=hashlib.sha256(domain_path.read_bytes()).hexdigest()
domain=pd.read_csv(domain_path,usecols=['datetime','symbol','keep'],parse_dates=['datetime'])
frame=frame.merge(domain[domain.keep][['datetime','symbol']],on=['datetime','symbol'],how='inner',validate='one_to_one')
frame=attach_horizon_labels(frame,pd.concat(label_parts,ignore_index=True),20)
original=frame.copy(deep=True)
cfg=json.loads((base/'run_20260908_235818_404243/resolved_config.json').read_text())
cfg.update(t_target_mode=TARGET_MODE,t_model_profile='l2_depth4',t_label_horizon_minutes=20,workers=1,
           min_training_rows=12,t_max_train_rows=0,t_max_features=60,
           t_support_features=['f_open_return','f_vwap_deviation'],
           t_model={**cfg['t_model'],'n_estimators':16,'min_child_samples':5})
started=time.monotonic()
with tempfile.TemporaryDirectory(prefix='weekly_t_timing_real4_') as temporary:
 out=Path(temporary)/'model';bundle=fit_timing_model(frame,[2024,2025],cfg,out,[2023,2026])
 predicted=predict_timing_model(bundle,frame,cfg)
 restored=predict_timing_model(joblib.load(out/'model_bundle.joblib'),frame,cfg)
 pd.testing.assert_frame_equal(predicted,restored)
 future=['execution_time','execution_price','execution_high','execution_low','execution_amount','execution_volume',
         'recovery_time','recovery_price','recovery_high','recovery_low','recovery_amount','recovery_volume',
         'label_start','label_end','target_return','label_available']
 cleared=predict_timing_model(bundle,frame.drop(columns=[key for key in future if key in frame]),cfg)
 fields=['datetime','symbol','prediction','raw_prediction','prob_neither','prob_buy_profit','prob_sell_profit','horizon_minutes']
 pd.testing.assert_frame_equal(predicted[fields],cleared[fields]);pd.testing.assert_frame_equal(frame,original)
 result={'passed':True,'scope':'首折2024/25核心OOF的4股已知slot域，只做工程接口验证，无外层或全局新特征数值结论',
         'symbols':symbols,'input_rows':len(frame),'training_rows':bundle['metadata']['training_rows'],
         'training_day_count':bundle['metadata']['training_day_count'],'training_clock_count':bundle['metadata']['training_clock_count'],
         'class_counts':bundle['metadata']['training_class_counts'],'feature_names':bundle['features'],
         'model_native_reload':bundle['metadata']['native_reload_audit'],'joblib_and_future_deletion_equal':True,
         'input_unchanged':True,'input_sha256':inputs,'elapsed_seconds':time.monotonic()-started,
         'module_sha256':hashlib.sha256(Path('/data/lxl/seed-emulator/topology/quant/quant_backtest/weekly_t_timing_models.py').read_bytes()).hexdigest()}
 print(json.dumps(result,ensure_ascii=False,indent=2))
 Path('/tmp/weekly_t_timing_models_real_sample_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
