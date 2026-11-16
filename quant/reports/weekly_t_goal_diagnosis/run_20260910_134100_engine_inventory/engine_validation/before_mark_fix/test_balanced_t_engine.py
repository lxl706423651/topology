"""固定金额底仓/多次T的经济语义回归；合成价量，不训练或修改历史缓存。"""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import tempfile
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.balanced_t_engine import run_balanced_portfolio, plan_balanced_entries
from quant_backtest.weekly_t_engine import build_recovery_store


def selection(symbol="A", day="2024-01-08", price=10., rank=1):
    date = pd.Timestamp(day)
    return {"symbol":symbol,"trade_week":date,"entry_date":date,"exit_date":date+pd.Timedelta(days=4),
            "rank":rank,"universe_previous_close":price}


def market(symbol="A", price=10.):
    return pd.DataFrame([{"date":date,"symbol":symbol,"open":price,"close":price,"previous_close":price,
                          "open_high":price+.01,"open_low":price-.01,"open_amount":2e6,"open_volume":2e5,
                          "close_high":price,"close_low":price,"close_amount":2e6,"close_volume":2e5}
                         for date in pd.bdate_range("2024-01-08","2024-01-19")])


def request(clock="10:00",day="2024-01-09",action=-1,fraction=.25,symbol="A",horizon=20,price=10.):
    signal=pd.Timestamp(f"{day} {clock}");execution=signal+pd.Timedelta(minutes=1)
    planned=execution+pd.Timedelta(minutes=horizon)
    sample={"datetime":signal,"date":signal.normalize(),"symbol":symbol,"execution_time":execution,
            "execution_price":price,"execution_high":price+.01,"execution_low":price-.01,"execution_amount":2e6,
            "execution_volume":2e5,"previous_close":10.,"recovery_time":planned,"recovery_price":10.,
            "recovery_high":10.,"recovery_low":10.,"recovery_amount":2e6,"recovery_volume":2e5}
    decision={"datetime":signal,"symbol":symbol,"action":action,"fraction":fraction,
              "planned_recovery_time":planned,"priority":1.}
    return sample,decision


def quote(timestamp,symbol="A",price=10.,amount=2e6,volume=2e5):
    return {"datetime":pd.Timestamp(timestamp),"symbol":symbol,"open":price,"high":price,"low":price,
            "close":price,"volume":volume,"amount":amount}


class BalancedEngineTests(unittest.TestCase):
    def setUp(self):
        self.cfg={"initial_cash":100000.,"base_fraction":.85,"slippage_rate":0.,"commission_rate":0.,
                  "min_commission":0.,"transfer_rate":0.,"stamp_rate_before_20230828":0.,"stamp_rate_from_20230828":0.,
                  "max_bar_amount_fraction":.05,"max_cycles_per_day":0}
        self.selected=pd.DataFrame([selection()]);self.daily=market()

    def run_case(self,requests=(),cfg=None,daily=None,selected=None,bars=None,manual=None):
        return run_balanced_portfolio(self.selected if selected is None else selected,
                                     self.daily if daily is None else daily,
                                     pd.DataFrame([x[0] for x in requests]),pd.DataFrame([x[1] for x in requests]),
                                     {**self.cfg,**(cfg or {})},recovery_bars=bars,manual_entries=manual)

    def assert_safe(self,result,restored=True):
        events=result['trade_events'];audit=result['daily_inventory_audit']
        self.assertTrue(events.cash_after.ge(-1e-7).all())
        self.assertTrue(events.shares.mod(100).eq(0).all())
        self.assertTrue(events.datetime.gt(events.signal_datetime).all())
        self.assertTrue(audit.day_sold.le(audit.opening_sellable).all())
        self.assertEqual(result['metrics']['t1_violation_count'],0)
        if restored:
            self.assertTrue(audit.inventory_preserved.all())
            self.assertAlmostEqual(result['metrics']['total_t_marked_pnl'],result['round_trips'].net_pnl.sum(),places=6)

    def test_target_amount_no_week_exit_no_duplicate_automatic_add(self):
        selected=pd.DataFrame([selection(),selection(day='2024-01-15')])
        result=self.run_case(selected=selected)
        self.assertEqual(list(result['trade_events'].shares),[1500])
        self.assertEqual(result['open_positions'].shares.sum(),1500)
        self.assertEqual(result['metrics']['weekly_sell_count'],0)
        self.assertEqual(result['metrics']['weekly_unclosed_positions'],0)
        self.assertTrue(result['metrics']['no_terminal_liquidation'])
        self.assertEqual(len(result['market_dates']),10)
        self.assert_safe(result)

    def test_nearest_lot_and_strict_lower_upper_bound(self):
        for price,shares in [(100.,100),(200.,100),(201.,0)]:
            with self.subTest(price=price):
                result=self.run_case(daily=market(price=price),selected=pd.DataFrame([selection(price=price)]))
                self.assertEqual(result['trade_events'].shares.sum(),shares)
                filled=result['position_amount_audit'].query('actual_shares>0')
                self.assertTrue(filled.actual_in_band.all())

    def test_slippage_counts_in_amount_band(self):
        result=self.run_case(daily=market(price=200.),selected=pd.DataFrame([selection(price=200.)]),cfg={'slippage_rate':.0005})
        self.assertEqual(len(result['trade_events']),0)

    def test_capacity_under_minimum_rejected_not_small_position(self):
        daily=self.daily.copy();daily.loc[0,'open_amount']=190000.
        result=self.run_case(daily=daily)
        self.assertEqual(len(result['trade_events']),0)
        self.assertIn('below_min_position_notional',set(result['order_attempts'].reason))

    def test_failed_entry_can_retry_next_candidate_week(self):
        daily=self.daily.copy();daily.loc[0,'open_amount']=0.
        result=self.run_case(daily=daily,selected=pd.DataFrame([selection(),selection(day='2024-01-15')]))
        self.assertEqual(result['trade_events'].datetime.iloc[0],pd.Timestamp('2024-01-15 09:30'))
        self.assertEqual(len(result['trade_events']),1)

    def test_cash_decides_count_without_top_n_limit_and_keeps_base_reserve(self):
        selected=pd.DataFrame([selection(str(n),rank=n) for n in range(10)])
        daily=pd.concat([market(str(n)) for n in range(10)],ignore_index=True)
        result=self.run_case(selected=selected,daily=daily,cfg={'top_n':1,'commission_rate':.0008,'min_commission':5.})
        # 第六只的剩余部署现金不足10000元名义额加手续费，必须整笔拒绝。
        self.assertEqual(result['metrics']['base_buy_count'],5)
        self.assertGreaterEqual(result['equity_curve'].cash.min(),15000.)
        self.assertTrue(result['position_amount_audit'].query('actual_shares>0').actual_in_band.all())

    def test_multiple_cycles_consume_old_budget_never_replenish(self):
        requests=[request(clock=x,action=(-1 if i%2==0 else 1)) for i,x in enumerate(['09:35','10:00','10:30','11:00','13:05','13:30'])]
        bars=pd.DataFrame([quote(x[1]['planned_recovery_time']) for x in requests])
        result=self.run_case(requests,bars=bars)
        self.assertEqual(result['metrics']['completed_pair_count'],5)
        inventory=result['daily_inventory_audit'];day=inventory.loc[inventory.date.eq(pd.Timestamp('2024-01-09'))].iloc[0]
        self.assertEqual(day.day_sold,1500);self.assertEqual(day.closing_sellable,0)
        self.assertEqual(result['metrics']['total_t_traded_lots'],30.)
        self.assertEqual(result['metrics']['completed_t_pair_lots'],15.)
        self.assert_safe(result)

    def test_explicit_daily_limit_and_legacy_fraction_small_inventory(self):
        requests=[request(clock='10:00'),request(clock='11:00')]
        bars=pd.DataFrame([quote(x[1]['planned_recovery_time']) for x in requests])
        result=self.run_case(requests,bars=bars,cfg={'max_cycles_per_day':1})
        self.assertEqual(result['metrics']['t_pair_count'],1)
        self.assertIn('daily_pair_limit',set(result['order_attempts'].reason))
        small=self.run_case([request()],daily=market(price=50.),selected=pd.DataFrame([selection(price=50.)]))
        self.assertEqual(small['metrics']['started_pair_count'],0)

    def test_first_day_no_old_inventory_but_old_week_exit_day_t_allowed(self):
        requests=[request(day='2024-01-08'),request(day='2024-01-12')]
        result=self.run_case(requests)
        self.assertEqual(result['metrics']['t_pair_count'],1)
        self.assertEqual(result['round_trips'].first_datetime.iloc[0].date(),pd.Timestamp('2024-01-12').date())
        self.assert_safe(result)

    def test_manual_adds_change_target_but_not_today_sellable(self):
        manual=pd.DataFrame([{'datetime':'2024-01-09 09:30','signal_datetime':'2024-01-08 15:00','symbol':'A','shares':500,'reason':'user_add'}])
        result=self.run_case([request(fraction=1.)],manual=manual)
        inventory=result['daily_inventory_audit'];day=inventory.loc[inventory.date.eq(pd.Timestamp('2024-01-09'))].iloc[0]
        self.assertEqual(day.opening_sellable,1500);self.assertEqual(day.baseline_opening_shares,1500)
        self.assertEqual(day.manual_buys,500);self.assertEqual(day.target_shares,2000)
        self.assertEqual(day.day_sold,1500);self.assertEqual(day.closing_sellable,0)
        self.assertEqual(result['metrics']['manual_buy_count'],1)
        self.assert_safe(result)

    def test_manual_csv_explicitly_supported_and_duplicate_source_rejected(self):
        manual=pd.DataFrame([{'datetime':'2024-01-09 09:30','signal_datetime':'2024-01-08','symbol':'A','shares':100}])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manual.csv';manual.to_csv(path,index=False)
            result=self.run_case(cfg={'manual_buys_csv':str(path)})
            self.assertEqual(result['metrics']['manual_buy_count'],1)
            with self.assertRaises(ValueError):self.run_case(cfg={'manual_buys_csv':str(path)},manual=manual)

    def test_partial_recovery_minimum_fee_and_account_identity(self):
        one=request(fraction=.25,price=10.1)
        bars=pd.DataFrame([quote(one[1]['planned_recovery_time'],amount=20000.),
                           quote(one[1]['planned_recovery_time']+pd.Timedelta(minutes=1),amount=40000.)])
        result=self.run_case([one],bars=bars,cfg={'commission_rate':.0008,'min_commission':5.,'transfer_rate':.00001,'slippage_rate':.0005})
        self.assertEqual(result['metrics']['t_pair_count'],1)
        self.assertEqual(list(result['round_trips'].shares),[100,200])
        self.assertTrue(result['round_trips'].buy_fee.ge(5.).all())
        self.assert_safe(result)

    def test_unrecoverable_debt_is_preserved_across_days(self):
        one=request();bars=pd.DataFrame([quote(one[1]['planned_recovery_time'],amount=0.,volume=0.)])
        daily=self.daily.copy();mask=daily.date.ge(pd.Timestamp('2024-01-09'))
        daily.loc[mask,'open_amount']=0.;daily.loc[mask,'close_amount']=0.
        result=self.run_case([one],bars=bars,daily=daily)
        self.assertEqual(result['metrics']['pending_pairs'],1)
        self.assertGreater(result['metrics']['inventory_mismatch_days'],0)
        self.assertLess(result['open_positions'].shares.sum(),1500)
        self.assert_safe(result,restored=False)

    def test_future_recovery_quotes_do_not_change_first_leg(self):
        one=request();good=pd.DataFrame([quote(one[1]['planned_recovery_time'])]);bad=good.copy();bad['amount']=0.;bad['volume']=0.
        a=self.run_case([one],bars=good);b=self.run_case([one],bars=bad)
        first=lambda x:x['trade_events'].query("event_type=='t' and fragment_id==0").reset_index(drop=True)
        pd.testing.assert_frame_equal(first(a),first(b))

    def test_recovery_and_new_pair_same_minute_share_capacity(self):
        first=request(clock='10:00');second=request(clock='10:20')
        second[0]['execution_amount']=60000.
        bars=pd.DataFrame([quote(first[1]['planned_recovery_time'],amount=60000.),quote(second[1]['planned_recovery_time'])])
        result=self.run_case([first,second],bars=bars)
        self.assertEqual(result['metrics']['t_pair_count'],1)
        self.assertIn('capacity',set(result['order_attempts'].reason))
        self.assert_safe(result)

    def test_freeze_successful_base_plan_exact_replay_and_config_guard(self):
        selected=pd.DataFrame([selection(),selection(day='2024-01-15')])
        plan=plan_balanced_entries(selected,self.daily,self.cfg)
        self.assertEqual(len(plan['frozen_selections']),1)
        result=self.run_case(selected=plan['frozen_selections'])
        for name in ['equity_curve','baseline_events']:
            pd.testing.assert_frame_equal(plan['baseline_result'][name],result[name])
        with self.assertRaises(ValueError):self.run_case(selected=plan['frozen_selections'],cfg={'commission_rate':.001})

    def test_frozen_replay_rejects_changed_base_quote(self):
        plan=plan_balanced_entries(self.selected,self.daily,self.cfg)
        daily=self.daily.copy();daily.loc[0,['open','open_high','open_low']]=[11.,11.01,10.99]
        with self.assertRaises(ValueError):self.run_case(selected=plan['frozen_selections'],daily=daily)

    def test_reusable_store_readonly_and_original_engine_unchanged(self):
        old=Path(__file__).resolve().parents[1]/'quant_backtest/weekly_t_engine.py';before=hashlib.sha256(old.read_bytes()).hexdigest()
        one=request();store=build_recovery_store(pd.DataFrame([quote(one[1]['planned_recovery_time'])]))
        first=self.run_case([one],bars=store);second=self.run_case([one],bars=store)
        pd.testing.assert_frame_equal(first['trade_events'],second['trade_events'])
        self.assertEqual(before,hashlib.sha256(old.read_bytes()).hexdigest())

    def test_T_losses_cannot_change_baseline_plan_or_borrow_base_funding(self):
        selected=pd.DataFrame([selection(),selection(symbol='B',day='2024-01-15')])
        daily=pd.concat([self.daily,market('B')],ignore_index=True)
        one=request(fraction=1.,price=9.8)
        result=self.run_case([one],selected=selected,daily=daily,cfg={'initial_cash':40000.,'base_fraction':.75})
        self.assertEqual(result['metrics']['base_plan_mismatch_count'],1)
        self.assertEqual(result['baseline_events'].query("symbol=='B'").shares.sum(),1500)
        self.assertEqual(result['trade_events'].query("symbol=='B'").shares.sum(),0)
        self.assertTrue(result['trade_events'].cash_after.ge(0).all())

    def test_positive_T_does_not_expand_frozen_base_quantity(self):
        selected=pd.DataFrame([selection(),selection(symbol='B',day='2024-01-15')])
        daily=pd.concat([self.daily,market('B')],ignore_index=True)
        one=request(fraction=1.,price=10.2)
        cfg={'initial_cash':40000.,'base_fraction':.75}
        result=self.run_case([one],selected=selected,daily=daily,cfg=cfg)
        self.assertEqual(result['trade_events'].query("symbol=='B'").shares.sum(),1500)
        self.assertEqual(result['baseline_events'].query("symbol=='B'").shares.sum(),1500)
        self.assert_safe(result)

    def test_manual_frozen_replay_preserves_all_additions(self):
        manual=pd.DataFrame([{'datetime':'2024-01-09 09:30','signal_datetime':'2024-01-08','symbol':'A','shares':100}])
        plan=plan_balanced_entries(self.selected,self.daily,self.cfg,manual_entries=manual)
        result=self.run_case(selected=plan['frozen_selections'])
        self.assertEqual(result['metrics']['manual_buy_count'],1)
        self.assertEqual(result['open_positions'].shares.sum(),1600)
        pd.testing.assert_frame_equal(plan['baseline_result']['baseline_events'],result['baseline_events'])

    def test_manual_intraday_fill_uses_current_quote_and_old_inventory(self):
        manual=pd.DataFrame([{'datetime':'2024-01-09 10:21','signal_datetime':'2024-01-09 10:00','symbol':'A','shares':100}])
        bars=pd.DataFrame([quote('2024-01-09 10:21')])
        result=self.run_case(manual=manual,bars=bars)
        event=result['trade_events'].query("base_order_type=='manual'").iloc[0]
        self.assertEqual(event.sellable_before,1500);self.assertEqual(event.sellable_after,1500)
        self.assertEqual(event.datetime,pd.Timestamp('2024-01-09 10:21'))

    def test_invalid_cycle_or_band_config_is_rejected(self):
        for cfg in [{'max_cycles_per_day':-.1},{'max_cycles_per_day':1.5},{'min_position_notional':20001.}]:
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):self.run_case(cfg=cfg)

    def test_future_base_signal_and_duplicate_frozen_manual_are_rejected(self):
        bad=self.selected.copy();bad['_signal_time']=pd.Timestamp('2024-01-08 10:00')
        with self.assertRaises(ValueError):self.run_case(selected=bad)
        plan=plan_balanced_entries(self.selected,self.daily,self.cfg)
        manual=pd.DataFrame([{'datetime':'2024-01-09 09:30','signal_datetime':'2024-01-08','symbol':'A','shares':100}])
        with self.assertRaises(ValueError):self.run_case(selected=plan['frozen_selections'],manual=manual)

    def test_manual_intraday_planning_uses_same_store(self):
        manual=pd.DataFrame([{'datetime':'2024-01-09 10:21','signal_datetime':'2024-01-09 10:00','symbol':'A','shares':100}])
        bars=pd.DataFrame([quote('2024-01-09 10:21')])
        plan=plan_balanced_entries(self.selected,self.daily,self.cfg,manual_entries=manual,recovery_bars=bars)
        result=self.run_case(selected=plan['frozen_selections'],bars=bars)
        pd.testing.assert_frame_equal(plan['baseline_result']['baseline_events'],result['baseline_events'])

    def test_source_weekly_signal_datetime_is_preserved(self):
        selected=self.selected.copy();selected['signal_datetime']=pd.Timestamp('2024-01-05 15:00')
        result=self.run_case(selected=selected)
        self.assertEqual(result['trade_events'].signal_datetime.iloc[0],pd.Timestamp('2024-01-05 15:00'))
        plan=plan_balanced_entries(selected,self.daily,self.cfg)
        self.assertEqual(plan['frozen_selections']._signal_time.iloc[0],pd.Timestamp('2024-01-05 15:00'))


if __name__=='__main__':
    unittest.main(verbosity=2)
