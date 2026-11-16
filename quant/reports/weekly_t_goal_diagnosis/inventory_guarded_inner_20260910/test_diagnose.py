"""独立诊断最小反例；不训练模型或运行生产账户。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).with_name('diagnose.py')
SPEC = importlib.util.spec_from_file_location('guarded_diagnosis', PATH)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
CFG = {'flat_st_limit_rate': .05, 'nonflat_limit_proxy': .098}


class Tests(unittest.TestCase):
    def quote(self, price=11.):
        return dict(reference_price=price, previous_close=10., high=price, low=price,
                    limit_up=np.nan, limit_down=np.nan, recovery_price=1000.)

    def test_opposite_direction_and_future_ignored(self):
        q=self.quote()
        self.assertEqual(D.limit_reason(q, 'sell', CFG), '')
        self.assertEqual(D.limit_reason(q, 'buy', CFG), 'conservative_price_limit_proxy')
        q['recovery_price']=.001
        self.assertEqual(D.limit_reason(q, 'buy', CFG), 'conservative_price_limit_proxy')

    def test_explicit_limit_and_flat_five_are_distinct(self):
        q=self.quote(10.5)
        self.assertEqual(D.limit_reason(q, 'buy', CFG), 'conservative_price_limit_proxy')
        q['high']=10.51
        self.assertEqual(D.limit_reason(q, 'buy', CFG), '')
        q['limit_up']=10.5
        self.assertEqual(D.limit_reason(q, 'buy', CFG), 'price_limit')

    def example_pairs(self):
        time=pd.Timestamp('2026-01-06 10:01')
        events=pd.DataFrame([dict(datetime=time,signal_datetime=time-pd.Timedelta(minutes=1),symbol='a',
            event_type='t',fragment_id=0,pair_id=pid,side='buy',shares=shares,reference_price=10.,fee=1.)
            for pid,shares in [(1,100),(2,200)]])
        events.loc[1,'symbol']='b'
        trips=pd.DataFrame([dict(pair_id=pid,shares=100,net_pnl=-5.,gross_pnl=-3.,slippage_cost=1.,buy_fee=1.,sell_fee=1.)
                            for pid in [1,2]])
        return D.pairs(events,trips)

    def test_partial_and_losing_common_groups_retained(self):
        old=self.example_pairs();new=old.copy(deep=True)
        result=D.pair_contrast(old,new)
        self.assertEqual(len(result),2)
        self.assertEqual(result.both_complete.sum(),1)
        self.assertEqual(result.complete_net_source.iloc[0],-5.)
        self.assertTrue(pd.isna(result.complete_net_source.iloc[1]))
        self.assertEqual(result.fragment_net_source.sum(),-10.)

    def test_different_quantity_is_two_unilateral_groups(self):
        old=self.example_pairs().iloc[:1].copy();new=old.copy()
        new['shares']=200;new['complete']=False;new['complete_net']=np.nan
        result=D.pair_contrast(old,new)
        self.assertEqual(set(result.matching),{'source_only_first','guarded_only_first'})
        self.assertEqual(result.both_complete.sum(),0)

    def test_cancel_raw_quote_and_growth_prefix_flag(self):
        d=object.__new__(D.Diagnosis);d.cfg=CFG;d.inner=2026
        t=pd.Timestamp('2026-01-06 10:01')
        raw=pd.DataFrame([dict(open=10.98,high=11.,low=10.98,close=11.,amount=100000.,volume=1000.)],index=[t])
        d.raw_quotes=lambda symbol:raw
        q=pd.DataFrame([dict(datetime=t,signal_datetime=t-pd.Timedelta(minutes=1),symbol='SZ.300001',
            side='sell',opposite_side='buy',requested_shares=100,own_fillable_shares=100,
            guard_enabled=True,opposite_checked=True,cancelled=True,
            reason='execution_exit_side_conservative_price_limit_proxy',opposite_quote_reason='conservative_price_limit_proxy',
            reference_price=11.,high=11.,low=10.98,amount=100000.,volume=1000.,previous_close=10.,limit_up=np.nan,limit_down=np.nan)])
        before=q.copy(deep=True)
        previous=pd.DataFrame([dict(signal_datetime=t-pd.Timedelta(minutes=1),symbol='SZ.300001',original_previous_close=10.)])
        result=d.guards(q,previous)
        self.assertEqual(result.board_prefix.iloc[0],'SZ.300')
        self.assertTrue(result.growth_prefix_9_8_proxy_below_19_8.iloc[0])
        self.assertTrue(result.raw_quote_verified.iloc[0])
        pd.testing.assert_frame_equal(q,before)
        q['reference_price']=11.01
        with self.assertRaises(AssertionError):d.guards(q,previous)


if __name__=='__main__':unittest.main(verbosity=2)
