"""Use completed smoke ledgers to test independent candidate/denominator audit corruption detection."""
from pathlib import Path
import json,shutil,sys,tempfile,unittest
import pandas as pd
ROOT=Path('/data/lxl/seed-emulator/topology/quant');sys.path.insert(0,str(ROOT))
from scripts.audit_weekly_t_frequency_execution import verify_saved_frequency_account,verify_frequency_denominator,select_independent_frequency_candidate
SOURCE=ROOT/'reports/weekly_t_cross_validation_smoke/run_20260909_135533_893081'
class FrequencyAuditTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.cfg=json.loads((SOURCE/'resolved_config.json').read_text());self.fold=SOURCE/'year_2026'
  rows=pd.read_csv(self.fold/'validation_rule_metrics.csv');index=int(rows.loc[rows.t_pair_count.gt(0),'candidate_index'].min())
  self.candidate=self.fold/f'validation_candidate_audits/candidate_{index:03d}'
 def copied(self):
  p=self.root/'account';shutil.copytree(self.candidate,p);return p
 def test_real_candidate_and_pure_baseline(self):
  a=verify_saved_frequency_account(self.candidate,self.cfg);self.assertTrue(a['passed']);self.assertGreater(a['t_pair_count'],0)
  b=verify_saved_frequency_account(self.fold/'validation_baseline',self.cfg,no_t=True);self.assertEqual(b['t_pair_count'],0)
 def test_actual_mixed_midnight_signal_strings_preserve_same_instant(self):
  origin=ROOT/'reports/weekly_t_cross_validation/run_20260909_135906_004933/year_2024/validation_candidate_audits/candidate_019'
  original=verify_saved_frequency_account(origin,self.cfg)
  p=self.root/'mixed_account';shutil.copytree(origin,p)
  for name in ['test_trade_events.csv','baseline_trade_events.csv']:
   f=p/name;d=pd.read_csv(f,float_precision='round_trip');d['signal_datetime']=pd.to_datetime(d.signal_datetime,format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S');d.to_csv(f,index=False)
  normalized=verify_saved_frequency_account(p,self.cfg)
  self.assertEqual(original,normalized)
 def test_future_signal_timestamp_still_fails_after_mixed_parser_fix(self):
  p=self.copied();f=p/'test_trade_events.csv';d=pd.read_csv(f);d.loc[0,'signal_datetime']='2099-01-01';d.to_csv(f,index=False)
  with self.assertRaises(AssertionError):verify_saved_frequency_account(p,self.cfg)
 def test_fee_tamper_detected(self):
  p=self.copied();f=p/'test_trade_events.csv';d=pd.read_csv(f);d.loc[0,'fee']+=1.;d.to_csv(f,index=False)
  with self.assertRaises(AssertionError):verify_saved_frequency_account(p,self.cfg)
 def test_cash_tamper_detected(self):
  p=self.copied();f=p/'test_trade_events.csv';d=pd.read_csv(f);d.loc[0,'cash_after']+=1.;d.to_csv(f,index=False)
  with self.assertRaises(AssertionError):verify_saved_frequency_account(p,self.cfg)
 def test_inventory_budget_tamper_detected(self):
  p=self.copied();f=p/'daily_inventory_audit.csv';d=pd.read_csv(f);d.loc[d.day_sold.gt(0),'opening_sellable']=0;d.to_csv(f,index=False)
  with self.assertRaises(AssertionError):verify_saved_frequency_account(p,self.cfg)
 def test_pair_net_tamper_detected(self):
  p=self.copied();f=p/'test_round_trips.csv';d=pd.read_csv(f);d.loc[0,'net_pnl']+=1.;d.to_csv(f,index=False)
  with self.assertRaises(AssertionError):verify_saved_frequency_account(p,self.cfg)
 def test_fractional_count_tamper_detected(self):
  p=self.copied();f=p/'metrics.json';d=json.loads(f.read_text());d['t_pair_count']+=.5;f.write_text(json.dumps(d))
  with self.assertRaises(AssertionError):verify_saved_frequency_account(p,self.cfg)
 def test_active_days_tamper_detected(self):
  p=self.copied();f=p/'summary.json';d=json.loads(f.read_text());d['active_days']+=1;f.write_text(json.dumps(d))
  with self.assertRaises(AssertionError):verify_saved_frequency_account(p,self.cfg)
 def denominator(self):
  p=self.root/'fold';p.mkdir();shutil.copytree(self.fold/'validation_baseline',p/'validation_baseline')
  for filename in ['validation_weekly_selections.csv','validation_eligible_inventory_stock_days.csv','validation_frequency_opportunities.json',*[f'validation_h20_{profile}_predictions.csv' for profile in ['l2_depth4','l2_depth6','huber_depth4']]]:shutil.copyfile(self.fold/filename,p/filename)
  keys=pd.read_csv(p/'validation_h20_l2_depth4_predictions.csv',usecols=['datetime','symbol'],parse_dates=['datetime'])
  keys['date']=keys.datetime.dt.normalize();keys['trade_week']=keys.date-pd.to_timedelta(keys.date.dt.dayofweek,unit='d')
  return p,keys
 def test_denominator_known_keys_reproduce_saved_rows(self):
  p,k=self.denominator();a=verify_frequency_denominator(p,self.cfg,k);self.assertGreater(a['eligible_portfolio_days'],0)
 def test_denominator_row_tamper_detected(self):
  p,k=self.denominator();f=p/'validation_eligible_inventory_stock_days.csv';d=pd.read_csv(f);d.loc[0,'eligible_t_shares']+=100;d.to_csv(f,index=False)
  with self.assertRaises(AssertionError):verify_frequency_denominator(p,self.cfg,k)
 def test_denominator_summary_tamper_detected(self):
  p,k=self.denominator();f=p/'validation_frequency_opportunities.json';d=json.loads(f.read_text());d['eligible_portfolio_days']+=1;f.write_text(json.dumps(d))
  with self.assertRaises(AssertionError):verify_frequency_denominator(p,self.cfg,k)
 def scores(self,pairs,pnl):
  r=pd.DataFrame({'candidate_index':range(len(pairs)),'t_pair_count':pairs,'incremental_net_pnl':pnl})
  for name in ['inventory_mismatch_days','recovery_failure_days','pending_pairs','base_plan_mismatch_count','t1_violation_count']:r[name]=0
  r['frequency_score']=(r.t_pair_count/10).clip(upper=3);r['active_execution_eligible']=r.t_pair_count.gt(0);return r
 def test_negative_profit_higher_frequency_before_target(self):
  r=self.scores([5,20],[1000,-50]);self.assertEqual(select_independent_frequency_candidate(r,10).candidate_index,1)
 def test_frequency_is_capped_then_pnl_then_index(self):
  r=self.scores([50,32,30],[1,2,2]);self.assertEqual(select_independent_frequency_candidate(r,10).candidate_index,1)
 def test_inventory_failure_disqualifies_high_frequency(self):
  r=self.scores([5,20],[1000,2000]);r.loc[1,'inventory_mismatch_days']=1;r.loc[1,'active_execution_eligible']=False
  self.assertEqual(select_independent_frequency_candidate(r,10).candidate_index,0)
 def test_frequency_score_tamper_and_zero_denominator_fail(self):
  r=self.scores([5,20],[1000,2000]);r.loc[0,'frequency_score']=3
  with self.assertRaises(AssertionError):select_independent_frequency_candidate(r,10)
  with self.assertRaises(AssertionError):select_independent_frequency_candidate(r,0)
if __name__=='__main__':unittest.main(verbosity=2)
