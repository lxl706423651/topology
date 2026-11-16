"""诊断配对反例：共同/单边/未完整/不同数量均保留，不调用引擎。"""
from pathlib import Path
import importlib.util,pandas as pd
p=Path('reports/weekly_t_goal_diagnosis/dynamic_inner_20260910/diagnose.py')
spec=importlib.util.spec_from_file_location('diagnosis',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def pair(s,n,complete=True,quantity=100):
 return {'symbol':s,'first_datetime':pd.Timestamp('2026-01-06 10:01'),'direction':'buy_first',
  'original_shares':quantity,'pair_id':ord(s),'complete':complete,'fragment_net':n,
  'reference_gross':n+3,'allocated_fees':2,'slippage':1,'actual_fees':2,
  'first_reference_notional':quantity*10,'reference_price':10.,'fee':1.}
l=pd.DataFrame([pair('A',10),pair('B',-5),pair('C',-3,False),pair('D',8),pair('E',12)])
r=pd.DataFrame([pair('A',20),pair('B',-1,False),pair('C',-4),pair('F',-9),pair('E',24,quantity=200)])
def rule(index,mode):return {'candidate_index':index,'exit_mode':mode,'family':'fixed_test','entry_deviation':.003,'recovery_horizon':20,'fraction':.5}
def metric(net,groups,lots):return {'incremental_net_pnl':net,'t_pair_count':groups,'completed_t_pair_lots':lots,**{k:0 for k in m.FAIL}}
d=m.Diagnosis.__new__(m.Diagnosis);d.year=2023;d.inner=2026;d.tables={n:[] for n in m.TABLE_NAMES}
d.contrast(1,(rule(1,'fixed'),metric(22,4,4),l,pd.DataFrame()),(rule(2,'profit_buffer'),metric(30,4,5),r,pd.DataFrame()))
c=d.tables['paired_exit_contrasts'][0]
assert c['common_exact_first_legs']==3 and c['common_both_complete']==1
assert c['only_fixed_first_legs']==2 and c['only_dynamic_first_legs']==2
assert c['same_key_incomplete_at_least_one']==2
assert c['common_complete_net_delta']==10 and c['all_complete_net_delta']==6
assert c['remaining_complete_net_delta']==-4 and c['incremental_marked_net_delta']==8
assert sum(x['complete_net_delta'] for x in d.tables['pair_match_categories'])==6
causes=d.tables['unmatched_entry_causes']
assert sum(x['first_legs'] for x in causes if x['other_account_observed_reason']=='same_signal_different_filled_quantity')==2
assert any(x['complete_net']<0 for x in causes)
print('PASS: 共同键、不同股数、单边亏损、两种未完整、完整净额分解和marked差额分开等9项断言')
