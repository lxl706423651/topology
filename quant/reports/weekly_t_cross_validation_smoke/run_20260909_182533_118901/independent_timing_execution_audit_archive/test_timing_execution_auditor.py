"""独立择时审计的当前门控、逐请求期限与透明hold反例；只写临时合成CSV。"""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import pandas as pd
ROOT=Path('/data/lxl/seed-emulator/topology/quant');sys.path.insert(0,str(ROOT))
from scripts.test_weekly_t_timing import current_features,prediction_inputs,timing_cfg
from quant_backtest.weekly_t_timing import timing_candidates,make_timing_decisions,select_timing_candidate
from scripts.audit_weekly_t_timing_execution import rebuild_timing_decisions,verify_timing_requests,select_independent_timing_candidate,independent_timing_rules

class TimingAuditorTests(unittest.TestCase):
 def setUp(self):
  self.cfg=timing_cfg();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)
 def frames(self):
  frame=pd.concat([current_features(('10:00','13:30','14:00')),current_features(('10:00','13:30','14:00'))],ignore_index=True)
  frame.loc[3:,'symbol']='SELL';frame.loc[3:,'raw_prediction']=-.01
  frame.loc[3:,'f_vwap_deviation']=.004;frame.loc[3:,'f_return_5m']=-.001
  frame.loc[3:,'f_timing_vwap_z']=1.
  p,s=prediction_inputs(frame)
  p[40]['raw_prediction']*=1.1;p[60]['raw_prediction']*=1.2
  return p,s
 def test_current_reconstruction_matches_all_24_candidates_and_future_deletion(self):
  p,s=self.frames()
  for rule in timing_candidates(self.cfg)[1:]:
   with self.subTest(index=rule['candidate_index']):
    actual,_=make_timing_decisions(p,s,rule,self.cfg)
    expected=rebuild_timing_decisions(p,s,rule,self.cfg)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-15)
    changed={h:f.drop(columns=[c for c in f if c.startswith(('execution_','recovery_','target_','label_'))]) for h,f in p.items()}
    pd.testing.assert_frame_equal(rebuild_timing_decisions(changed,s,rule,self.cfg),expected)
 def write_requests(self,empty=False):
  p,s=self.frames();rule=timing_candidates(self.cfg)[1]
  decisions,_=make_timing_decisions(p,s,rule,self.cfg)
  if empty:decisions=decisions.iloc[:0]
  decisions.to_csv(self.path/'decisions.csv',index=False)
  columns=['datetime','signal_datetime','symbol','event_type','fragment_id','pair_id','planned_recovery_time','side']
  events=[]
  for i,r in enumerate(decisions.itertuples(index=False),1):
   # Only one first leg per stock/day, matching the engine contract despite later requests.
   if r.datetime.hour!=10:continue
   events.append(dict(datetime=r.datetime+pd.Timedelta(minutes=1),signal_datetime=r.datetime,symbol=r.symbol,event_type='t',fragment_id=0,pair_id=i,planned_recovery_time=r.planned_recovery_time,side='buy' if r.action==1 else 'sell'))
   events.append(dict(datetime=r.planned_recovery_time,signal_datetime=r.planned_recovery_time-pd.Timedelta(minutes=1),symbol=r.symbol,event_type='t',fragment_id=i,pair_id=i,planned_recovery_time=r.planned_recovery_time,side='sell' if r.action==1 else 'buy'))
  pd.DataFrame(events,columns=columns).to_csv(self.path/'events.csv',index=False)
  return rule
 def test_dynamic_horizon_and_empty_active_csv_are_legal(self):
  rule=self.write_requests();a=verify_timing_requests(self.path/'decisions.csv',self.path/'events.csv',self.cfg,rule)
  self.assertEqual(a['horizon_counts'],{'60':2,'40':2,'20':2})
  self.assertEqual(a['first_legs'],2)
  self.write_requests(empty=True);a=verify_timing_requests(self.path/'decisions.csv',self.path/'events.csv',self.cfg,rule)
  self.assertFalse(a['explicit_hold']);self.assertEqual(a['requests'],0)
 def test_wrong_horizon_plan_and_probability_are_rejected(self):
  for field,value in [('horizon_minutes',20),('predicted_net_edge',.00001),('prob_profit',.1),('priority',99.)]:
   rule=self.write_requests();path=self.path/'decisions.csv';df=pd.read_csv(path);df.loc[0,field]=value;df.to_csv(path,index=False)
   with self.subTest(field=field),self.assertRaises(AssertionError):verify_timing_requests(path,self.path/'events.csv',self.cfg,rule)
 def test_recovery_before_its_selected_horizon_is_rejected(self):
  rule=self.write_requests();p=self.path/'events.csv';d=pd.read_csv(p);row=d.index[d.fragment_id.gt(0)][0];d.loc[row,'datetime']='2024-01-09 10:02:00';d.to_csv(p,index=False)
  with self.assertRaises(AssertionError):verify_timing_requests(self.path/'decisions.csv',p,self.cfg,rule)
 def test_missing_gate_archive_is_explicitly_rejected(self):
  p,s=self.frames();p[20]=p[20].drop(columns='f_timing_trend_efficiency')
  with self.assertRaises(AssertionError):rebuild_timing_decisions(p,s,timing_candidates(self.cfg)[5],self.cfg)
 def table(self):
  rows=[]
  for rule in independent_timing_rules():
   rows.append({**rule,'t_pair_count':0,'incremental_net_pnl':0.,'inventory_mismatch_days':0,'recovery_failure_days':0,'pending_pairs':0,'base_plan_mismatch_count':0,'t1_violation_count':0})
  return pd.DataFrame(rows)
 def test_independent_positive_net_selection_and_hold_match(self):
  rows=self.table();rows.loc[2,['t_pair_count','incremental_net_pnl']]=[1,50.];rows.loc[5,['t_pair_count','incremental_net_pnl']]=[100,40.]
  chosen,annotated=select_timing_candidate(rows);self.assertEqual(chosen['candidate_index'],2)
  self.assertEqual(select_independent_timing_candidate(annotated).candidate_index,2)
  rows.loc[2,'recovery_failure_days']=1;rows.loc[5,'incremental_net_pnl']=-1.
  _,annotated=select_timing_candidate(rows);self.assertEqual(select_independent_timing_candidate(annotated).candidate_index,0)
  annotated.loc[5,'timing_eligible']=True
  with self.assertRaises(AssertionError):select_independent_timing_candidate(annotated)

if __name__=='__main__':unittest.main(verbosity=2)
