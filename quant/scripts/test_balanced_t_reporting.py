"""固定底仓报告口径测试：费用分母、配对标准手、零T、人工加仓和目标状态。

仅用小型合成已保存账户结果及临时输出，不训练模型、不读取或改写正式结果。
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics,write_balanced_report


def config(**changes):
    """显式低成本账表fixture；手数门槛默认不量化。"""
    return {"initial_cash":100000.,"lot_size":100,"profit_target_rate":.05,
            "min_position_notional":10000.,"target_position_notional":15000.,"max_position_notional":20000.,
            "test_years":[2024],"report_scope":"formal",**changes}


def fixture(net=200.,fragments=(100,),pending=False,hold=False,year=2024):
    """两只各1.5万底仓，B始终不做T；可模拟分片、待恢复和无T。"""
    start=pd.Timestamp(f"{year}-01-02 15:00");end=pd.Timestamp(f"{year}-01-03 15:00")
    bases=[]
    for symbol,shares,price in (("A",1500,10.),("B",1000,15.)):
        bases.append({"datetime":start.normalize()+pd.Timedelta(hours=9,minutes=31),"symbol":symbol,"side":"buy",
                      "shares":shares,"notional":15000.,"fee":1.,"slippage_cost":0.,"event_type":"weekly",
                      "base_order_type":"automatic","fragment_id":0,"pair_id":0,"price":price,"reference_price":price})
    baseline=pd.DataFrame(bases)
    event_rows=deepcopy(bases);trip_rows=[]
    q=600 if pending else sum(fragments)
    if not hold:
        first=end.normalize()+pd.Timedelta(hours=10,minutes=1)
        event_rows.append({"datetime":first,"symbol":"A","side":"buy","shares":q,"notional":q*10.,
                           "fee":1.,"slippage_cost":.5,"event_type":"t","base_order_type":"",
                           "fragment_id":0,"pair_id":1,"price":10.,"reference_price":10.})
        for n,shares in enumerate(fragments,1):
            proportion=shares/sum(fragments);recover=first+pd.Timedelta(minutes=20+n-1)
            event_rows.append({"datetime":recover,"symbol":"A","side":"sell","shares":shares,"notional":shares*10.+net*proportion,
                               "fee":proportion,"slippage_cost":.5*proportion,"event_type":"t","base_order_type":"",
                               "fragment_id":n,"pair_id":1,"price":10.,"reference_price":10.})
            trip_rows.append({"pair_id":1,"symbol":"A","shares":shares,"first_datetime":first,"second_datetime":recover,
                              "pair_fully_recovered":not pending,"gross_pnl":(net+2)*proportion,"net_pnl":net*proportion,
                              "buy_fee":proportion,"sell_fee":proportion,"slippage_cost":proportion})
    events=pd.DataFrame(event_rows)
    trips=pd.DataFrame(trip_rows,columns=["pair_id","symbol","shares","first_datetime","second_datetime","pair_fully_recovered",
                                       "gross_pnl","net_pnl","buy_fee","sell_fee","slippage_cost"])
    gain=0 if hold else net
    curve=pd.DataFrame({"datetime":[start,end],"equity":[99998.,99998.+gain],"baseline_equity":[99998.,99998.],
                        "cash":[69998.,69998.+gain],"holdings_value":[30000.,30000.]})
    audits=[]
    for day in [start.normalize(),end.normalize()]:
        for symbol,shares in [("A",1500),("B",1000)]:
            diff=500 if pending and day==end.normalize() and symbol=="A" else 0
            audits.append({"date":day,"symbol":symbol,"baseline_opening_shares":0 if day==start.normalize() else shares,
                           "inventory_difference":diff,"recovery_failure":bool(diff),"t1_violation_count":0})
    positions=pd.DataFrame({"symbol":["A","B"],"shares":[1500,1000],"market_value":[15000.,15000.],"mark_datetime":[end,end]})
    amounts=pd.DataFrame({"order_type":["automatic","automatic"],"symbol":["A","B"],"actual_shares":[1500,1000],
                          "actual_notional":[15000.,15000.],"baseline_notional":[15000.,15000.],"actual_in_band":[True,True]})
    pairs=0 if hold or pending else 1;pair_lots=0. if hold or pending else q/100
    traded_lots=0. if hold else (q+sum(fragments))/100
    metrics={"inventory_mismatch_days":int(pending),"recovery_failure_days":int(pending),"pending_pairs":int(pending),
             "t1_violation_count":0,"base_plan_mismatch_count":0,"t_pair_count":pairs,
             "total_t_traded_lots":traded_lots,"completed_t_pair_lots":pair_lots}
    return {"equity_curve":curve,"trade_events":events,"round_trips":trips,"open_positions":positions,
            "baseline_events":baseline,"daily_inventory_audit":pd.DataFrame(audits),"position_amount_audit":amounts,
            "metrics":metrics,"market_dates":pd.DatetimeIndex([start.normalize(),end.normalize()])}


class BalancedReportingTests(unittest.TestCase):
    def test_all_original_costs_include_non_trading_stock(self):
        summary,stocks=calculate_balanced_metrics(fixture(net=1600),config())
        self.assertEqual(summary['all_reference_acquisition_cost'],30002.)
        self.assertAlmostEqual(summary['holding_cost_reduction_rate'],1600/30002)
        self.assertEqual(summary['zero_t_reference_stock_count'],1)
        self.assertEqual(stocks.loc[stocks.symbol.eq('B'),'complete_t_net_pnl'].iloc[0],0.)
        self.assertAlmostEqual(summary['account_increment_return'],.016)

    def test_t_repurchase_never_added_to_cost(self):
        small,_=calculate_balanced_metrics(fixture(net=100,fragments=(100,)),config())
        large,_=calculate_balanced_metrics(fixture(net=100,fragments=(600,)),config())
        self.assertEqual(small['all_reference_acquisition_cost'],large['all_reference_acquisition_cost'])
        self.assertEqual(small['holding_cost_reduction_rate'],large['holding_cost_reduction_rate'])

    def test_split_recovery_is_one_pair_six_paired_lots(self):
        summary,_=calculate_balanced_metrics(fixture(fragments=(100,500)),config())
        self.assertEqual(summary['complete_t_pairs'],1)
        self.assertEqual(summary['completed_t_pair_lots'],6)
        self.assertEqual(summary['total_t_traded_lots'],12)
        self.assertEqual(summary['recovery_fragments'],2)
        self.assertAlmostEqual(summary['realized_fragment_reference_gross_pnl']-summary['t_slippage_cost']-summary['t_fee'],200)

    def test_unquantified_frequency_does_not_claim_goal(self):
        summary,_=calculate_balanced_metrics(fixture(net=2000),config())
        self.assertTrue(summary['profit_target_met'])
        self.assertEqual(summary['frequency_target_status'],'not_quantified')
        self.assertFalse(summary['goal_achieved'])
        self.assertFalse(summary['report_controls_activation'])

    def test_explicit_frequency_and_profit_can_be_evaluated(self):
        summary,_=calculate_balanced_metrics(fixture(net=2000),config(minimum_completed_pair_lots=1,minimum_active_stock_days=1))
        self.assertTrue(summary['goal_achieved'])
        low,_=calculate_balanced_metrics(fixture(net=2000),config(minimum_completed_pair_lots=2))
        self.assertEqual(low['frequency_target_status'],'not_met')

    def test_pending_never_passes_with_positive_marked_gain(self):
        summary,_=calculate_balanced_metrics(fixture(net=3000,fragments=(100,),pending=True),config(minimum_completed_pair_lots=1))
        self.assertFalse(summary['profit_target_met'])
        self.assertFalse(summary['inventory_contract_met'])
        self.assertEqual(summary['completed_t_pair_lots'],0)
        self.assertEqual(summary['pending_t_lots'],5)
        self.assertEqual(summary['total_t_traded_lots'],7)

    def test_manual_addition_increases_whole_cost_without_new_deposit(self):
        result=fixture(net=2000)
        manual=result['baseline_events'].iloc[0].copy();manual['base_order_type']='manual';manual['shares']=500;manual['notional']=5000.;manual['fee']=1.
        result['baseline_events']=pd.concat([result['baseline_events'],manual.to_frame().T],ignore_index=True)
        summary,_=calculate_balanced_metrics(result,config())
        self.assertEqual(summary['manual_addition_cost'],5001)
        self.assertEqual(summary['all_reference_acquisition_cost'],35003)
        self.assertEqual(summary['account_capital_denominator'],100000)
        self.assertAlmostEqual(summary['holding_cost_reduction_rate'],2000/35003)

    def test_zero_trade_is_not_enough_and_keeps_normal_bottom(self):
        summary,_=calculate_balanced_metrics(fixture(hold=True),config())
        self.assertEqual(summary['complete_t_pairs'],0)
        self.assertEqual(summary['zero_t_reference_stock_count'],2)
        self.assertEqual(summary['holding_cost_reduction_rate'],0)
        self.assertEqual(summary['final_open_position_rows'],2)
        self.assertTrue(summary['inventory_contract_met'])
        self.assertFalse(summary['goal_achieved'])

    def test_market_days_and_old_stock_days_are_distinct(self):
        summary,_=calculate_balanced_metrics(fixture(fragments=(600,)),config())
        self.assertEqual(summary['market_evaluation_days'],2)
        self.assertEqual(summary['baseline_old_stock_days'],1)
        self.assertEqual(summary['completed_pair_lots_per_market_day'],3)
        self.assertEqual(summary['completed_pair_lots_per_old_stock_day'],6)
        self.assertEqual(summary['active_stock_day_coverage'],.5)
        bad=fixture();bad['market_dates']=pd.date_range('2024-01-01','2024-01-03')
        with self.assertRaisesRegex(ValueError,'全部声明市场'):calculate_balanced_metrics(bad,config())

    def test_amount_band_uses_actual_notional_and_manual_exception(self):
        result=fixture(net=2000);result['position_amount_audit'].loc[0,'actual_notional']=9999.
        summary,_=calculate_balanced_metrics(result,config(minimum_completed_pair_lots=1))
        self.assertFalse(summary['allocation_goal_met']);self.assertFalse(summary['goal_achieved'])
        result['position_amount_audit'].loc[0,'order_type']='manual'
        summary,_=calculate_balanced_metrics(result,config())
        self.assertTrue(summary['allocation_goal_met'])

    def test_duplicate_engine_lot_count_rejected(self):
        result=fixture(fragments=(100,500));result['metrics']['completed_t_pair_lots']=2
        with self.assertRaisesRegex(ValueError,'completed_t_pair_lots'):calculate_balanced_metrics(result,config())

    def test_source_frames_are_not_mutated(self):
        result=fixture();original=deepcopy(result)
        calculate_balanced_metrics(result,config())
        for key in result:
            if isinstance(result[key],pd.DataFrame):pd.testing.assert_frame_equal(result[key],original[key],check_exact=True)

    def test_each_year_must_pass_and_smoke_cannot_claim_formal_goal(self):
        with tempfile.TemporaryDirectory() as folder:
            status=write_balanced_report(Path(folder),config(test_years=[2024,2025],minimum_completed_pair_lots=1),
                                         [(2024,fixture(net=2000)),(2025,fixture(net=-10,year=2025))])
            self.assertFalse(status['goal_achieved']);self.assertFalse(status['each_validation_year_profit_target_met'])
            self.assertTrue((Path(folder)/'test_return_curve.png').is_file())
            saved=json.loads((Path(folder)/'goal_evaluation.json').read_text());self.assertEqual(saved['sum_complete_pairs'],2)
        with tempfile.TemporaryDirectory() as folder:
            status=write_balanced_report(Path(folder),config(report_scope='smoke',minimum_completed_pair_lots=1),{2024:fixture(net=2000)})
            self.assertFalse(status['goal_achieved']);self.assertTrue(status['each_validation_year_profit_target_met'])
            self.assertIn('工程烟测',(Path(folder)/'final_report.md').read_text())

    def test_optional_threshold_must_not_default_to_zero(self):
        for value in (0,-1,True,float('nan')):
            with self.assertRaises(ValueError):calculate_balanced_metrics(fixture(),config(minimum_completed_pair_lots=value))

    def test_stock_mismatch_cannot_be_hidden_by_zero_metric(self):
        result=fixture();result['daily_inventory_audit'].loc[3,'inventory_difference']=100
        with self.assertRaisesRegex(ValueError,'真实逐日库存'):calculate_balanced_metrics(result,config())

    def test_hold_state_and_external_cash_are_explicit(self):
        result=fixture(hold=True);result['selected_rule']={'enabled':False}
        result['external_cash_flows']=pd.DataFrame({'amount':[20000.,-1000.]})
        summary,_=calculate_balanced_metrics(result,config())
        self.assertEqual(summary['t_activity_status'],'hold')
        self.assertEqual(summary['account_capital_denominator'],120000.)
        self.assertEqual(summary['all_reference_acquisition_cost'],30002.)

    def test_real_balanced_engine_result_contract(self):
        from test_balanced_t_engine import BalancedEngineTests,request,quote
        case=BalancedEngineTests();case.setUp()
        reqs=[request(clock=x) for x in ['10:00','10:30']]
        bars=pd.DataFrame([quote(r[1]['planned_recovery_time']) for r in reqs])
        result=case.run_case(reqs,bars=bars)
        summary,stocks=calculate_balanced_metrics(result,{**config(),**case.cfg})
        self.assertEqual(summary['complete_t_pairs'],2)
        self.assertEqual(summary['completed_t_pair_lots'],6)
        self.assertEqual(summary['total_t_traded_lots'],12)
        self.assertEqual(summary['all_reference_acquisition_cost'],15000)
        self.assertEqual(summary['market_evaluation_days'],10)
        self.assertEqual(summary['baseline_old_stock_days'],9)
        self.assertEqual(summary['automatic_base_sell_count'],0)
        self.assertTrue(summary['inventory_contract_met'])


if __name__=='__main__':
    unittest.main(verbosity=2)
