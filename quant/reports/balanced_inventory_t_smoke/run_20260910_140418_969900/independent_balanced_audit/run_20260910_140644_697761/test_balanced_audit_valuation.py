"""独立审计估值反例：无日开收盘的人工成交、待恢复残仓和更晚真实收盘。"""
from collections import defaultdict
from pathlib import Path
import json,sys,tempfile,unittest
import numpy as np,pandas as pd
sys.path.insert(0,'/data/lxl/seed-emulator/topology/quant')
from scripts.audit_balanced_inventory_t import Auditor,fee,update_known_mark,valuation_contract
ROOT=Path('/data/lxl/seed-emulator/topology/quant')
class ValuationTests(unittest.TestCase):
 def create(self,pending=False,closing=None,wrong_mark=False):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);folder=root/'case';folder.mkdir()
  cfg={'initial_cash':100000.,'lot_size':100,'commission_rate':.0008,'transfer_rate':.00001,'min_commission':0.,'stamp_rate_before_20230828':.001,'stamp_rate_from_20230828':.0005,'slippage_rate':.0005,'max_bar_amount_fraction':.05,'min_position_notional':10000.,'max_position_notional':20000.,'max_cycles_per_day':0,'smoke_days':0}
  d1=pd.Timestamp('2025-01-02');d2=pd.Timestamp('2025-01-03');days=pd.DatetimeIndex([d1,d2] if pending else [d1]);capital=cfg['initial_cash']
  def event(dt,signal,qty,ref,kind,pair,cash,old,planned=pd.NaT):
   price=ref*(1+cfg['slippage_rate']);value=qty*price;cost=fee(value,'buy',dt,cfg)
   return {'datetime':dt,'signal_datetime':signal,'planned_recovery_time':planned,'symbol':'SH.600000','plan_id':1,'side':'buy','shares':qty,'price':price,'reference_price':ref,'notional':value,'fee':cost,'slippage_cost':qty*(price-ref),'event_type':kind,'base_order_type':'manual' if kind!='t' else '', 'cash_after':cash-value-cost,'sellable_before':old,'sellable_after':old,'pair_id':pair,'fragment_id':0}
  base=event(d1+pd.Timedelta(hours=10,minutes=1),d1+pd.Timedelta(hours=10),1000,11.,'weekly',0,capital,0)
  events=[base.copy()];base_events=[base.copy()];cash=base['cash_after'];base_cash=cash;mark=11.;qty=1000
  curves=[{'datetime':d1+pd.Timedelta(hours=15),'equity':cash+11000,'cash':cash,'holdings_value':11000.,'baseline_equity':cash+11000,'baseline_cash':cash,'t_added_equity':0.,'pending_pair_count':0,'completed_pair_count':0}]
  def inventory(day,opening,amount,target,baseqty,buy,manual,pending,cycles):
   return {'date':day,'symbol':'SH.600000','opening_shares':opening,'opening_sellable':opening,'closing_shares':amount,'baseline_opening_shares':opening,'baseline_closing_shares':baseqty,'weekly_buys':buy,'weekly_sells':0,'target_shares':target,'inventory_difference':amount-target,'baseline_inventory_difference':amount-baseqty,'day_sold':0,'closing_sellable':opening,'cycles_started':cycles,'manual_buys':manual,'base_buys':0,'inventory_preserved':amount==target,'recovery_failure':pending,'weekly_liquidation_failure':False}
  inventories=[inventory(d1,0,1000,1000,1000,1000,1000,False,0)]
  raw=[{'datetime':base['datetime'],'open':11.,'high':11.1,'low':10.9,'close':11.,'volume':1e6,'amount':1e8}]
  if pending:
   trade=event(d2+pd.Timedelta(hours=10,minutes=31),d2+pd.Timedelta(hours=10,minutes=30),500,12.,'t',1,cash,1000,d2+pd.Timedelta(hours=10,minutes=51));events.append(trade);cash=trade['cash_after'];qty=1500;mark=12.
   raw.append({'datetime':trade['datetime'],'open':12.,'high':12.1,'low':11.9,'close':12.,'volume':1e6,'amount':1e8})
   if closing is not None:
    raw.append({'datetime':d2+pd.Timedelta(hours=15),'open':closing,'high':closing,'low':closing,'close':closing,'volume':1e6,'amount':1e8});mark=closing
   inventories.append(inventory(d2,1000,1500,1000,1000,0,0,True,1))
   curves.append({'datetime':d2+pd.Timedelta(hours=15),'equity':cash+qty*mark,'cash':cash,'holdings_value':qty*mark,'baseline_equity':base_cash+1000*mark,'baseline_cash':base_cash,'t_added_equity':cash-base_cash+500*mark,'pending_pair_count':1,'completed_pair_count':0})
  if wrong_mark:
   bad=mark*(1+cfg['slippage_rate']);curves[-1]['equity']=cash+qty*bad;curves[-1]['holdings_value']=qty*bad;curves[-1]['baseline_equity']=base_cash+1000*bad;curves[-1]['t_added_equity']=cash-base_cash+(qty-1000)*bad;mark=bad
  position={'symbol':'SH.600000','shares':qty,'baseline_shares':1000,'mark_price':mark,'market_value':qty*mark,'exit_date':pd.NaT,'overdue_weekly_exit':False}
  basecost=base['notional']+base['fee'];increment=curves[-1]['equity']-curves[-1]['baseline_equity']
  metrics={'inventory_mismatch_days':int(pending),'recovery_failure_days':int(pending),'pending_pairs':int(pending),'t1_violation_count':0,'base_plan_mismatch_count':0,'t_pair_count':0,'completed_pair_count':0,'started_pair_count':int(pending),'fragment_count':0,'completed_t_pair_lots':0.,'total_t_traded_lots':5. if pending else 0.,'all_reference_acquisition_cost':basecost,'realized_complete_pair_pnl':0.,'realized_all_fragment_pnl':0.,'t_fee':events[-1]['fee'] if pending else 0.,'incremental_net_pnl':increment,'holding_cost_reduction_rate':0.}
  (folder/'metrics.json').write_text(json.dumps(metrics))
  for name,frame in {'test_trade_events.csv':pd.DataFrame(events),'baseline_trade_events.csv':pd.DataFrame(base_events),'test_equity_curve.csv':pd.DataFrame(curves),'daily_inventory_audit.csv':pd.DataFrame(inventories),'test_open_positions.csv':pd.DataFrame([position]),'test_round_trips.csv':pd.DataFrame(columns=['pair_id','second_datetime','shares','net_pnl']),'position_amount_audit.csv':pd.DataFrame(),'base_plan_mismatches.csv':pd.DataFrame()}.items():frame.to_csv(folder/name,index=False)
  auditor=Auditor.__new__(Auditor);auditor.report=root;auditor.out=root;auditor.cfg=cfg;auditor.market_calendar=days;auditor.valuation_mode='successful_fill_reference_then_daily_close';auditor.inputs={};auditor.raw={(2025,'SH.600000'):pd.DataFrame(raw).set_index('datetime')};auditor.raw_files={};auditor.results=[];auditor.checked_raw_events=0
  return auditor,folder
 def test_manual_without_daily_endpoints_uses_unslipped_fill(self):
  auditor,folder=self.create();result=auditor.account(folder)
  self.assertEqual(result['incremental_net_pnl'],0);self.assertGreater(result['all_reference_acquisition_cost'],11000)
 def test_pending_extra_shares_without_close_use_last_fill_reference(self):
  auditor,folder=self.create(pending=True);result=auditor.account(folder)
  self.assertEqual(result['pending_pairs'],1);self.assertLess(result['incremental_net_pnl'],0);self.assertGreater(result['incremental_net_pnl'],-20)
 def test_later_real_close_overrides_fill(self):
  auditor,folder=self.create(pending=True,closing=13.);result=auditor.account(folder)
  self.assertGreater(result['incremental_net_pnl'],480)
 def test_slipped_execution_price_is_not_market_mark(self):
  auditor,folder=self.create(pending=True,wrong_mark=True)
  with self.assertRaisesRegex(AssertionError,'收盘权益'):auditor.account(folder)
 def test_older_baseline_fill_does_not_overwrite_later_actual_price(self):
  marks,times={},{};late=pd.Timestamp('2025-01-03 10:31')
  update_known_mark(marks,times,'A',12.,late);update_known_mark(marks,times,'A',11.,late-pd.Timedelta(minutes=30))
  self.assertEqual(marks['A'],12.)
 def test_declared_mark_contract_requires_snapshot_implementation(self):
  old=ROOT/'reports/balanced_inventory_t_smoke/run_20260910_135532_126122/source_snapshot/quant_backtest/balanced_t_engine.py'
  self.assertEqual(valuation_contract({},old.read_text()),'daily_open_close_only')
  with self.assertRaises(AssertionError):valuation_contract({'valuation_contract':'successful_fill_reference_then_daily_close'},old.read_text())
  with self.assertRaises(AssertionError):valuation_contract({},'print(1)')
if __name__=='__main__':unittest.main(verbosity=2)
