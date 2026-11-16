"""对独立审计器注入账簿/日历/手数/成本反例，只改临时副本。"""
from pathlib import Path
import json,shutil,sys,tempfile,unittest
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,'/data/lxl/seed-emulator/topology/quant')
from scripts.audit_balanced_inventory_t import Auditor
SOURCE=Path('/data/lxl/seed-emulator/topology/quant/reports/balanced_inventory_t_smoke/run_20260910_135532_126122')
class AuditCounterexamples(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.auditor=Auditor(SOURCE,Path('/tmp'))
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.auditor.report=Path(self.temp.name)
  self.folder=self.auditor.report/'candidate_002'
  shutil.copytree(SOURCE/'year_2026/validation_candidates/candidate_002',self.folder)
 def change_metric(self,key,delta=None,value=None):
  p=self.folder/'metrics.json';z=json.loads(p.read_text())
  z[key]=z[key]+delta if delta is not None else value;p.write_text(json.dumps(z))
 def test_control(self):
  z=self.auditor.account(self.folder)
  self.assertEqual(z['t_pair_count'],151);self.assertEqual(z['completed_t_pair_lots'],2909)
  self.assertLess(z['incremental_net_pnl'],0)
 def test_cash_mutation(self):
  p=self.folder/'test_trade_events.csv';z=pd.read_csv(p,float_precision='round_trip');z.loc[0,'cash_after']+=.01;z.to_csv(p,index=False)
  with self.assertRaisesRegex(AssertionError,'事件后现金'):self.auditor.account(self.folder)
 def test_old_share_counter_mutation(self):
  p=self.folder/'daily_inventory_audit.csv';z=pd.read_csv(p,float_precision='round_trip');z.loc[0,'day_sold']+=100;z.to_csv(p,index=False)
  with self.assertRaisesRegex(AssertionError,'day_sold'):self.auditor.account(self.folder)
 def test_cash_period_cannot_be_removed(self):
  p=self.folder/'test_equity_curve.csv';z=pd.read_csv(p,float_precision='round_trip');z.iloc[1:].to_csv(p,index=False)
  with self.assertRaisesRegex(AssertionError,'全市场日历'):self.auditor.account(self.folder)
 def test_pair_lots_not_fragments(self):
  self.change_metric('completed_t_pair_lots',value=177)
  with self.assertRaisesRegex(AssertionError,'completed_t_pair_lots'):self.auditor.account(self.folder)
 def test_original_cost_includes_fees(self):
  self.change_metric('all_reference_acquisition_cost',delta=-100)
  with self.assertRaisesRegex(AssertionError,'all_reference_acquisition_cost'):self.auditor.account(self.folder)
 def test_negative_result_not_zero(self):
  self.change_metric('incremental_net_pnl',value=0)
  with self.assertRaisesRegex(AssertionError,'incremental_net_pnl'):self.auditor.account(self.folder)
 def test_raw_capacity_not_trusted_from_ledger(self):
  original=self.auditor.raw_quote
  def reduced(stamp,symbol):
   t,p,a=original(stamp,symbol);return t,p,.01
  with patch.object(self.auditor,'raw_quote',reduced):
   with self.assertRaisesRegex(AssertionError,'容量超限'):self.auditor.account(self.folder)
 def test_complete_groups_not_fragments(self):
  self.change_metric('t_pair_count',value=177)
  with self.assertRaisesRegex(AssertionError,'t_pair_count'):self.auditor.account(self.folder)
if __name__=='__main__':unittest.main(verbosity=2)
