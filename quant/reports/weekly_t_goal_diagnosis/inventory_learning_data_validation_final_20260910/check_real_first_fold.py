"""真实第一折纯训练数据预检；不拟合、不连接标签、不读取外层收益。"""
from pathlib import Path
import json,time,hashlib,sys,yaml
sys.path.insert(0,str(Path.cwd()))
from scripts.run_weekly_t_active import load_frozen_market_data
from quant_backtest.weekly_t_timing import add_timing_features
from quant_backtest.balanced_t_data import record_balanced_input
from quant_backtest.inventory_t_learning_data import build_inventory_core_cohort
out=Path('reports/weekly_t_goal_diagnosis/inventory_learning_data_validation_final_20260910')
out.mkdir(exist_ok=False)
cfg=json.loads(Path('reports/weekly_t_cross_validation/run_20260909_182945_388303/resolved_config.json').read_text())
cfg.update(yaml.safe_load(Path('config/dynamic_inventory_t_v13.yaml').read_text()))
inputs={}
fold=json.loads((Path(cfg['source_training_report_dir'])/'year_2023/split.json').read_text())
source_path=Path('quant_backtest/inventory_t_learning_data.py')
source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
(out/'inventory_t_learning_data.py').write_bytes(source_path.read_bytes())
start=time.monotonic()
print('load full current market',flush=True)
u,i,d=load_frozen_market_data(cfg,out,inputs)
i=add_timing_features(i,cfg)
print('loaded',len(u),len(i),len(d),'seconds',time.monotonic()-start,flush=True)
c,m=build_inventory_core_cohort(u,d,i,fold,cfg,out,inputs,record_balanced_input)
print(json.dumps(m,ensure_ascii=False,indent=2),flush=True)
(out/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
checks={'passed':True,'outer_results_read':False,'model_fit':False,'labels_attached':False,'full_market_rows':[len(u),len(i),len(d)],'opportunities':len(c),'elapsed_seconds':time.monotonic()-start,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('quant_backtest/inventory_t_learning_data.py'),Path('quant_backtest/dynamic_t_engine.py')]}}
(out/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
assert source_sha==checks['source_sha256'][str(source_path)]
assert 'trade_week_complete' not in c
print('done',checks,flush=True)
