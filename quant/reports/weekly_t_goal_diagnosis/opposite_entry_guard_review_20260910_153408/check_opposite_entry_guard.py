"""只读复核现有首腿限价函数和最小合成反例；不实现新引擎或改模型信号。"""
from pathlib import Path
import ast, hashlib, json, sys
from datetime import datetime
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'scripts'))
from quant_backtest.weekly_t_features import select_recoverable_signal_sides
from quant_backtest.dynamic_t_engine import run_dynamic_portfolio, SESSION_MINUTES
from test_balanced_t_engine import selection, market, request, quote
report=Path('reports/inventory_learning_t/run_20260910_151232_841321')
cfg=json.loads((report/'resolved_config.json').read_text())
source=Path('quant_backtest/dynamic_t_engine.py')
module=ast.parse(source.read_text()); top=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run_dynamic_portfolio')
fill_node=next(n for n in top.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_quote_fill')
namespace={'np':np,'cfg':cfg,'lot':100}
exec(compile(ast.Module(body=[fill_node],type_ignores=[]),str(source),'exec'),namespace)
fill=namespace['calculate_quote_fill']
case=Path('reports/weekly_t_goal_diagnosis/v14_internal_progress_20260910/third_fold_failure_20260910_153117_987738/case_raw_quotes.csv')
rows=pd.read_csv(case,float_precision='round_trip');rows['datetime']=pd.to_datetime(rows.datetime)
signal=rows.loc[rows.datetime.eq('2026-03-23 09:45')].iloc[0]
arrived=rows.loc[rows.datetime.eq('2026-03-23 09:46')].iloc[0]
pdframe=pd.DataFrame([{'signal_price':signal.close,'signal_high':signal.high,'signal_low':signal.low,'previous_close':signal.previous_close}])
assert bool(select_recoverable_signal_sides(pdframe,pd.Series([1]),cfg).iloc[0])
q={'price':arrived.close,'high':arrived.high,'low':arrived.low,'amount':arrived.amount,'volume':arrived.volume,'previous_close':arrived.previous_close,'limit_up':np.nan,'limit_down':np.nan}
first=fill(q,'buy',400,0.);opposite=fill(q,'sell',400,0.)
assert first[0]==400 and opposite[0]==0 and opposite[2]=='conservative_price_limit_proxy'
original=q.copy(); fill(q,'sell',400,arrived.amount*.05); assert all((q[k]==v) or (pd.isna(q[k]) and pd.isna(v)) for k,v in original.items())
real={'signal_time':str(signal.datetime),'signal_price':float(signal.close),'signal_return':float(signal.close/signal.previous_close-1),'signal_buy_first_gate_pass':True,'execution_time':str(arrived.datetime),'quote':q,'existing_buy_fill':list(first),'same_quote_opposite_sell_check':list(opposite),'execution_guard_reads_recovery_future':False}
# 两日/一只股/一个当刻信号：前日底仓1500股，本日9:45尚-9.6%，9:46跌至-10%。
synthetic_cfg={'initial_cash':100000.,'base_fraction':.85,'slippage_rate':0.,'commission_rate':0.,'min_commission':0.,'transfer_rate':0.,'stamp_rate_before_20230828':0.,'stamp_rate_from_20230828':0.,'max_bar_amount_fraction':.05,'max_cycles_per_day':0,'dynamic_exit_mode':'fixed','dynamic_take_net_return':.003,'dynamic_stop_net_return':.01,'t_avoid_locked_exit_side':True}
daily=market().iloc[:2].copy();daily.loc[daily.index[-1],['close','close_high','close_low']]=9.
sample,decision=request(clock='09:45',action=1,fraction=.25,price=9.,horizon=20)
sample.update(signal_price=9.04,signal_high=9.05,signal_low=9.03,signal_volume=1000.,signal_amount=9040.)
bars=pd.DataFrame([quote(day+pd.Timedelta(minutes=m),price=9. if day==daily.date.iloc[1] and m>=586 else 10.) for day in daily.date for m in SESSION_MINUTES])
result=run_dynamic_portfolio(pd.DataFrame([selection()]),daily,pd.DataFrame([sample]),pd.DataFrame([decision]),synthetic_cfg,recovery_bars=bars)
firstevents=result['trade_events'].query("event_type=='t' and fragment_id==0")
assert len(firstevents)==1 and firstevents.iloc[0].side=='buy' and firstevents.iloc[0].shares==300
assert result['metrics']['pending_pairs']==1 and result['metrics']['inventory_mismatch_days']==1
assert result['metrics']['t1_violation_count']==0
# 新风险检查只计算纯报价/限价拒因，不能拿虚拟反向容量作为取消理由。
capacity_q={**q,'price':20.,'high':20.,'low':20.,'amount':100.,'previous_close':20.}
capacity_reason=fill(capacity_q,'sell',400,0.)[2];assert capacity_reason=='capacity'
summary={'status':'existing_gap_reproduced_read_only_no_fix_implemented','real_case':real,'synthetic':{'first_leg_count':len(firstevents),'first_shares':int(firstevents.iloc[0].shares),'pending_pairs':result['metrics']['pending_pairs'],'inventory_mismatch_days':result['metrics']['inventory_mismatch_days'],'t1_violation_count':result['metrics']['t1_violation_count']},'capacity_reason_must_not_be_opposite_guard_failure':capacity_reason,'quote_unchanged_by_pure_check':True,'original_signal_guard_documented_scope':'signal_time_only','no_production_source_modified':True,'read_outer_returns':False}
out=Path('reports/weekly_t_goal_diagnosis')/f'opposite_entry_guard_review_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir()
Path('/tmp/opposite_entry_guard_path.txt').write_text(str(out))
result['trade_events'].to_csv(out/'synthetic_events.csv',index=False)
result['order_attempts'].to_csv(out/'synthetic_attempts.csv',index=False)
result['daily_inventory_audit'].to_csv(out/'synthetic_inventory_audit.csv',index=False)
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
(out/'checks.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'out':str(out),'real_case':clean(real),'synthetic':summary['synthetic']},ensure_ascii=False,indent=2))
