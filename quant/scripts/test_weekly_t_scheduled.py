"""验证正式周度T引擎的固定早恢复时钟、容量与未恢复名义金额约束。"""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_backtest import weekly_t_engine as engine
from scripts import test_weekly_t_engine as legacy
from scripts import test_weekly_t_partial as partial


def quote(clock,quantity,price=10.,symbol='TEST'):
    """以默认参与率100%生成明确股数容量；零容量真实零量，不补行情。"""
    return {'datetime':pd.Timestamp(clock),'symbol':symbol,'open':price,'high':price,'low':price,
            'close':price,'volume':float(quantity),'amount':quantity*price+(1e-4 if quantity else 0.)}


def request(day='2024-01-09',minute='10:00',planned='10:20',side=-1,fraction=.1,symbol='TEST',price=10.):
    """创建先前已知的恢复计划，与执行/恢复价格是否存在无关。"""
    sample,decision=legacy.create_request(day=day,minute=minute,side=side,execution=price,fraction=fraction)
    sample['symbol']=decision['symbol']=symbol
    decision['planned_recovery_time']=pd.Timestamp(f'{day} {planned}')
    return sample,decision


class ScheduledRecoveryTests(unittest.TestCase):
    """核对时间排序、资金与容量共享、午休、无未来筛选和默认兼容。"""

    def setUp(self):
        legacy.WeeklyTTests.setUp(self)

    def run_case(self,requests=(),quotes=None,daily=None,selected=None,cfg=None,store=None):
        return engine.run_weekly_portfolio(self.selected if selected is None else selected,
                                           self.daily if daily is None else daily,
                                           pd.DataFrame([r[0] for r in requests]),pd.DataFrame([r[1] for r in requests]),
                                           {**self.cfg,**(cfg or {})},recovery_bars=store if store is not None else pd.DataFrame(quotes) if quotes is not None else None)

    def assert_safe(self,result):
        events=result['trade_events'];audit=result['daily_inventory_audit']
        self.assertTrue(events.datetime.is_monotonic_increasing)
        self.assertTrue((events.cash_after>=-1e-7).all())
        self.assertTrue((audit.day_sold<=audit.opening_sellable).all())
        self.assertEqual(result['metrics']['t1_violation_count'],0)
        if result['metrics']['same_base_plan'] and result['metrics']['pending_pairs']==0:
            self.assertAlmostEqual(result['metrics']['total_t_marked_pnl'],result['round_trips'].net_pnl.sum(),places=6)

    def two_stocks(self):
        daily=pd.concat([legacy.create_daily(self.dates,symbol=s) for s in ['AAA','ZZZ']],ignore_index=True)
        selected=pd.DataFrame([{**legacy.create_selection(),'symbol':s,'rank':n} for n,s in enumerate(['AAA','ZZZ'],1)])
        return daily,selected,{'initial_cash':10000.,'base_fraction':.8,'top_n':2}

    def test_early_recovery_releases_cash_before_later_first_leg(self):
        daily,selected,cfg=self.two_stocks()
        requests=[request(symbol='AAA',side=1,fraction=.5),request(symbol='ZZZ',minute='10:30',planned='10:50',side=1,fraction=.5)]
        quotes=[quote('2024-01-09 10:20',200,symbol='AAA'),quote('2024-01-09 10:50',200,symbol='ZZZ')]
        result=self.run_case(requests,quotes=quotes,daily=daily,selected=selected,cfg=cfg)
        events=result['trade_events'].query("event_type=='t'")
        self.assertEqual(events.datetime.dt.strftime('%H:%M').tolist(),['10:01','10:20','10:31','10:50'])
        self.assertEqual(events.shares.tolist(),[200]*4)
        self.assertEqual(result['metrics']['completed_pair_count'],2)
        self.assert_safe(result)
        unavailable=self.run_case(requests,quotes=[{**q,'amount':0.,'volume':0.} for q in quotes],daily=daily,selected=selected,cfg=cfg)
        self.assertEqual(unavailable['metrics']['started_pair_count'],1)

    def test_same_minute_recovery_precedes_new_stock_cash_request(self):
        daily,selected,cfg=self.two_stocks()
        requests=[request(symbol='AAA',side=1,fraction=.5),request(symbol='ZZZ',minute='10:19',planned='10:40',side=1,fraction=.5)]
        result=self.run_case(requests,quotes=[quote('2024-01-09 10:20',200,symbol='AAA'),quote('2024-01-09 10:40',200,symbol='ZZZ')],
                             daily=daily,selected=selected,cfg=cfg)
        same=result['trade_events'].loc[result['trade_events'].datetime.eq(pd.Timestamp('2024-01-09 10:20'))]
        self.assertEqual(same.symbol.tolist(),['AAA','ZZZ'])
        self.assertEqual(same.side.tolist(),['sell','buy'])
        self.assertEqual(same.cash_after.tolist(),[2000.,0.])
        self.assert_safe(result)

    def test_pending_recovery_and_new_first_leg_share_same_symbol_capacity(self):
        daily=self.daily.copy()
        daily.loc[daily.date.eq(pd.Timestamp('2024-01-10')),'open_amount']=0.
        previous=request(side=1,fraction=.4)
        following=request(day='2024-01-10',minute='10:19',planned='10:40',side=-1,fraction=.5)
        following[0]['execution_amount']=6000.0001
        result=self.run_case([previous,following],quotes=[quote('2024-01-10 10:20',600),quote('2024-01-10 10:40',200)],
                             daily=daily,cfg={'base_fraction':.1,'scheduled_recovery_times':['10:20']})
        same=result['trade_events'].loc[result['trade_events'].datetime.eq(pd.Timestamp('2024-01-10 10:20'))]
        self.assertEqual(same.shares.tolist(),[400,200])
        self.assertEqual(same.pair_id.tolist(),[1,2])
        self.assertTrue(same.side.eq('sell').all())
        self.assertEqual(same.sellable_before.tolist(),[1400,1000])
        self.assertEqual(same.sellable_after.tolist(),[1000,800])
        self.assertEqual(result['metrics']['completed_pair_count'],2)
        self.assert_safe(result)

    def test_retry_clock_skips_lunch_and_uses_partial_current_bar(self):
        result=self.run_case([request(minute='11:00',planned='11:29',side=1,fraction=.5)],
                             quotes=[quote('2024-01-09 11:29',100),quote('2024-01-09 11:30',0),quote('2024-01-09 13:01',300)],
                             cfg={'base_fraction':.08})
        trips=result['round_trips']
        self.assertEqual(trips.second_datetime.dt.strftime('%H:%M').tolist(),['11:29','13:01'])
        self.assertEqual(trips.shares.tolist(),[100,300])
        attempts=result['order_attempts'].query("event_type=='t_recovery'")
        self.assertFalse(attempts.datetime.dt.hour.eq(12).any())
        clock=engine.create_planned_recovery_clock(pd.Timestamp('2024-01-09 11:29'))
        self.assertEqual([t.strftime('%H:%M') for t in clock],['11:29','11:30','13:01','13:02','13:03','13:04'])
        self.assert_safe(result)

    def test_missing_early_quotes_fall_back_to_original_tail_store(self):
        quotes=partial.create_recovery('2024-01-09',[400,0,0,0,0,0],fraction=1.)
        result=self.run_case([request(side=1,fraction=.5)],store=engine.build_recovery_store(quotes),cfg={'base_fraction':.08})
        self.assertEqual(result['round_trips'].second_datetime.dt.strftime('%H:%M').tolist(),['14:55'])
        attempts=result['order_attempts'].query("event_type=='t_recovery'")
        early=attempts.loc[attempts.datetime.dt.hour.eq(10)]
        self.assertEqual(len(early),6)
        self.assertTrue(early.reason.eq('missing_quote').all())
        self.assertEqual(result['metrics']['inventory_mismatch_days'],0)
        self.assert_safe(result)

    def test_future_quote_changes_never_resize_or_remove_first_leg(self):
        requested=request(side=-1,fraction=.5)
        good=[quote('2024-01-09 10:20',10000,price=10.1)]
        absent=[quote('2024-01-09 10:20',0,price=99.)]
        results=[self.run_case([requested],quotes=q) for q in [good,absent]]
        first=[r['trade_events'].query("event_type=='t'").iloc[:1].reset_index(drop=True) for r in results]
        pd.testing.assert_frame_equal(*first)
        self.assertEqual(int(first[0].shares.iloc[0]),2500)

    def test_config_clock_cannot_recover_before_pair_plan(self):
        requested=request(side=1,fraction=.5)
        requested[0]['planned_recovery_time']=pd.Timestamp('2024-01-09 10:10')
        result=self.run_case([requested],quotes=[quote('2024-01-09 10:10',400),quote('2024-01-09 10:20',400)],
                             cfg={'base_fraction':.08,'scheduled_recovery_times':['10:10','10:20']})
        self.assertEqual(result['round_trips'].second_datetime.dt.strftime('%H:%M').tolist(),['10:20'])
        self.assertTrue(result['round_trips'].planned_recovery_time.eq(pd.Timestamp('2024-01-09 10:20')).all())

    def test_another_decision_does_not_add_retries_to_old_pair(self):
        daily,selected,cfg=self.two_stocks()
        requests=[request(symbol='AAA',side=1,fraction=.25),request(symbol='ZZZ',minute='10:15',planned='10:30',side=1,fraction=.25)]
        quotes=[quote('2024-01-09 10:30',100,symbol='AAA'),quote('2024-01-09 10:30',100,symbol='ZZZ'),
                quote('2024-01-09 14:55',100,symbol='AAA')]
        result=self.run_case(requests,quotes=quotes,daily=daily,selected=selected,cfg=cfg)
        a=result['round_trips'].loc[result['round_trips'].symbol.eq('AAA')]
        self.assertEqual(a.second_datetime.dt.strftime('%H:%M').tolist(),['14:55'])
        self.assert_safe(result)

    def test_full_session_store_and_subclass_reuse(self):
        frame=pd.DataFrame([quote('2024-01-09 10:20',400),quote('2024-01-09 12:00',400),quote('2024-01-09 14:55',400)])
        original=engine.build_recovery_store(frame)
        self.assertEqual(original.row_count,2)
        self.assertIsNone(original.get_quote(pd.Timestamp('2024-01-09 12:00'),'TEST'))
        class DenseStub(engine.RecoveryBarStore):
            def __init__(self):
                self.calls=[]
            def get_quote(self,timestamp,symbol):
                self.calls.append((timestamp,symbol))
                return original.get_quote(timestamp,symbol)
        store=DenseStub()
        with mock.patch.object(engine,'build_recovery_store',side_effect=AssertionError('subclass rebuilt')):
            result=self.run_case([request(side=1,fraction=.5)],store=store,cfg={'base_fraction':.08})
        self.assertEqual(store.calls,[(pd.Timestamp('2024-01-09 10:20'),'TEST')])
        self.assert_safe(result)

    def test_early_partial_recovery_keeps_full_per_fragment_minimum_fee(self):
        result=self.run_case([request(price=10.4,side=-1,fraction=.1)],
                             quotes=[quote('2024-01-09 10:20',100,price=10.2),quote('2024-01-09 10:21',400,price=10.1)],
                             cfg={'min_commission':5.})
        trips=result['round_trips']
        self.assertEqual(trips.shares.tolist(),[100,400])
        self.assertEqual(trips.buy_fee.tolist(),[5.,5.])
        self.assertEqual(trips.sell_fee.tolist(),[1.,4.])
        self.assertAlmostEqual(trips.net_pnl.sum(),125.)
        self.assert_safe(result)

    def test_default_schedule_matches_frozen_v4_even_with_extra_early_quotes(self):
        path=ROOT/'reports/weekly_t_cross_validation/run_20260908_212722_188543/source_snapshot/weekly_t_engine.py'
        spec=importlib.util.spec_from_file_location('quant_backtest.frozen_weekly_t_v4',path)
        previous=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=previous
        spec.loader.exec_module(previous)
        old_request=legacy.create_request(side=-1,execution=10.4,fraction=.1)
        quotes=pd.DataFrame([quote('2024-01-09 10:05',500),quote('2024-01-09 14:55',100),quote('2024-01-09 14:56',400)])
        old=previous.run_weekly_portfolio(self.selected,self.daily,pd.DataFrame([old_request[0]]),
                                          pd.DataFrame([old_request[1]]),{**self.cfg,'min_commission':5.},recovery_bars=quotes)
        new=self.run_case([old_request],store=engine.build_recovery_store(quotes),cfg={'min_commission':5.})
        for key in ['equity_curve','trade_events','round_trips','open_positions','daily_inventory_audit','order_attempts','baseline_events']:
            pd.testing.assert_frame_equal(new[key][old[key].columns],old[key])
        self.assertFalse(new['metrics']['scheduled_recovery_enabled'])

    def test_shared_pending_notional_limit_floors_lots_and_preserves_priority(self):
        daily,selected,cfg=self.two_stocks()
        cfg.update(base_fraction=.4,max_pending_t_notional_fraction=.15)
        requests=[request(symbol=s,side=1,fraction=1.) for s in ['AAA','ZZZ']]
        requests[1][1]['priority']=1.
        result=self.run_case(requests,quotes=[quote('2024-01-09 10:20',100,symbol='ZZZ')],
                             daily=daily,selected=selected,cfg=cfg)
        first=result['trade_events'].query("event_type=='t' and side=='buy'")
        self.assertEqual(first.symbol.tolist(),['ZZZ'])
        self.assertEqual(first.shares.tolist(),[100])
        self.assertEqual(first.planned_shares.tolist(),[200])
        capped=result['order_attempts'].query("reason=='pending_t_notional_limit'")
        self.assertEqual(capped.symbol.tolist(),['ZZZ','AAA'])
        self.assertEqual(capped.filled_shares.tolist(),[100,0])
        self.assertEqual(result['metrics']['pending_t_notional_limit'],1500.)
        self.assertEqual(result['metrics']['peak_pending_t_notional'],1000.)
        self.assertEqual(result['metrics']['final_pending_t_notional'],0.)
        self.assert_safe(result)

    def test_partial_recovery_releases_shared_notional_limit_before_same_time_first_leg(self):
        daily,selected,cfg=self.two_stocks()
        cfg.update(max_pending_t_notional_fraction=.2)
        requests=[request(symbol='AAA',side=1,fraction=.5),
                  request(symbol='ZZZ',minute='10:19',planned='10:40',side=1,fraction=.5)]
        quotes=[quote('2024-01-09 10:20',100,symbol='AAA'),quote('2024-01-09 10:21',100,symbol='AAA'),
                quote('2024-01-09 10:40',100,symbol='ZZZ')]
        result=self.run_case(requests,quotes=quotes,daily=daily,selected=selected,cfg=cfg)
        same=result['trade_events'].loc[result['trade_events'].datetime.eq(pd.Timestamp('2024-01-09 10:20'))]
        self.assertEqual(same.symbol.tolist(),['AAA','ZZZ'])
        self.assertEqual(same.side.tolist(),['sell','buy'])
        self.assertEqual(same.shares.tolist(),[100,100])
        self.assertEqual(same.planned_shares.tolist(),[200,200])
        self.assertEqual(result['metrics']['peak_pending_t_notional'],2000.)
        self.assertEqual(result['metrics']['completed_pair_count'],2)
        self.assertEqual(result['metrics']['fragment_count'],3)
        self.assert_safe(result)

    def test_zero_pending_notional_limit_disables_entries_and_invalid_cap_rejected(self):
        result=self.run_case([request()],cfg={'max_pending_t_notional_fraction':0.})
        self.assertEqual(result['metrics']['started_pair_count'],0)
        self.assertEqual(result['order_attempts'].reason.tolist(),['pending_t_notional_limit'])
        for invalid in [-.1,float('nan'),float('inf')]:
            with self.assertRaisesRegex(ValueError,'max_pending_t_notional_fraction'):
                self.run_case(cfg={'max_pending_t_notional_fraction':invalid})

    def test_invalid_schedule_is_rejected(self):
        for value in ['10:01','12:00','2024-01-10 10:20']:
            requested=request()
            requested[1]['planned_recovery_time']=pd.Timestamp(value if value.startswith('2024') else f'2024-01-09 {value}')
            with self.assertRaisesRegex(ValueError,'planned_recovery_time'):
                self.run_case([requested])
        requested=request()
        requested[1]['planned_recovery_time']=pd.NaT
        with self.assertRaisesRegex(ValueError,'planned_recovery_time'):
            self.run_case([requested])
        for times in [['12:00'],['09:30'],['10:20:01'],'10:20']:
            with self.assertRaises(ValueError):
                self.run_case(cfg={'scheduled_recovery_times':times})

    def test_missing_execution_sample_cannot_bypass_plan_validation(self):
        """未来首腿行情缺失仍须拒绝越日/午休/过早计划，不能把本日时间线延到次日。"""
        for planned in ['2024-01-10 10:20','2024-01-09 12:00','2024-01-09 10:01']:
            sample,decision=request()
            decision['planned_recovery_time']=pd.Timestamp(planned)
            with self.subTest(planned=planned), self.assertRaisesRegex(ValueError,'planned_recovery_time'):
                engine.run_weekly_portfolio(self.selected,self.daily,pd.DataFrame([sample]).iloc[:0],
                                            pd.DataFrame([decision]),self.cfg)

    def test_missing_execution_sample_with_legal_plan_remains_a_blocked_order(self):
        """合法固定计划不要求未来sample存在；缺失时真实挡单且保留本日审计。"""
        sample,decision=request()
        result=engine.run_weekly_portfolio(self.selected,self.daily,pd.DataFrame([sample]).iloc[:0],
                                           pd.DataFrame([decision]),self.cfg)
        self.assertEqual(result['metrics']['started_pair_count'],0)
        self.assertEqual(result['order_attempts'].reason.tolist(),['missing_execution_minute'])
        self.assert_safe(result)


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(legacy),
                             unittest.defaultTestLoader.loadTestsFromTestCase(partial.WeeklyTPartialTests),
                             unittest.defaultTestLoader.loadTestsFromTestCase(ScheduledRecoveryTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
