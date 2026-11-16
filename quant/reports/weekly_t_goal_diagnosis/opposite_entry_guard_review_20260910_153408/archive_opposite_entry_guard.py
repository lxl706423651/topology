"""归档首腿反向限价复核建议及原函数反例；不修改生产源。"""
from pathlib import Path
import hashlib, json, shutil, ast
import numpy as np
out=Path(Path('/tmp/opposite_entry_guard_path.txt').read_text())
report=Path('reports/inventory_learning_t/run_20260910_151232_841321')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((report/'resolved_config.json').read_text())
source=Path('quant_backtest/dynamic_t_engine.py')
tree=ast.parse(source.read_text());top=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_dynamic_portfolio')
node=next(x for x in top.body if isinstance(x,ast.FunctionDef) and x.name=='calculate_quote_fill')
ns={'np':np,'cfg':cfg,'lot':100};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
fill=ns['calculate_quote_fill']
q=dict(price=19.6,high=19.68,low=19.6,amount=8010775.,volume=4084.,previous_close=21.78,limit_up=np.nan,limit_down=np.nan)
assert fill(q,'sell',0,0.)[2]=='conservative_price_limit_proxy'
cap={**q,'price':20.,'high':20.,'low':20.,'previous_close':20.,'amount':1.}
assert fill(cap,'sell',0,0.)[2]==''
frozen=json.loads((report/'source_hashes.json').read_text());assert all(sha(k)==v and sha(report/'source_snapshot'/k)==v for k,v in frozen.items())
inputs={str(report/'source_hashes.json'):sha(report/'source_hashes.json'),str(report/'resolved_config.json'):sha(report/'resolved_config.json'),'quant_backtest/dynamic_t_engine.py':sha(source),'quant_backtest/weekly_t_features.py':sha('quant_backtest/weekly_t_features.py'),'quant_backtest/inventory_t_learning_policy.py':sha('quant_backtest/inventory_t_learning_policy.py')}
case=Path('reports/weekly_t_goal_diagnosis/v14_internal_progress_20260910/third_fold_failure_20260910_153117_987738')
for name in ['case_raw_quotes.csv','case_pair_events.csv','diagnosis.md']:
 inputs[str(case/name)]=sha(case/name)
(out/'input_identity.json').write_text(json.dumps({'current_runtime_files':len(frozen),'all_current_and_snapshot_match':True,'zero_quantity_check_avoids_capacity_filter':True,'inputs_sha256':inputs},ensure_ascii=False,indent=2)+'\n')
shutil.copy2('/tmp/check_opposite_entry_guard.py',out/'check_opposite_entry_guard.py')
shutil.copy2('/tmp/archive_opposite_entry_guard.py',out/'archive_opposite_entry_guard.py')
print(out)
