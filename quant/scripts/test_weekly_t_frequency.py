"""验证固定频率候选、当前信号门控、失败后重试与真实费用，不训练模型或读外层。

使用合成分钟/周底仓与原账户引擎验证排名不免除费用、每股每日成功首腿限制，
其他测试只调用纯规则/选择函数。运行本文件不会改动生产配置或缓存。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quant_backtest.weekly_t_frequency import (FrequencyTStrategy, build_frequency_signal_funnel,
    create_frequency_decisions, create_frequency_rules, select_frequency_candidate, summarize_frequency_funnel)
from quant_backtest.weekly_t_short_horizon import ShortHorizonTStrategy, create_short_decisions, predicted_pair_edges
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from scripts.test_weekly_t_activation import make_candidate
from scripts.test_weekly_t_engine import create_daily, create_selection, create_request
from scripts.test_weekly_t_short_horizon import predictions_at, short_cfg


def frequency_cfg() -> dict:
    """固定与本轮一致的信号/成本条件；账户本金另在小型执行测试明确给出。"""
    return {**short_cfg(), 't_model_profiles':['l2_depth4','l2_depth6','huber_depth4'],
            'frequency_absolute_net_margins':[0.,.0005], 'frequency_rank_quantiles':[.70,.85,.95],
            'frequency_min_vwap_gaps':[.0035,.005], 'frequency_max_abs_day_return':.03,
            'frequency_max_known_amplitude':.06, 'frequency_target_pairs_per_day':3.}


def opportunities(clocks=('10:00',), day='2024-01-09') -> pd.DataFrame:
    """构造低预测收益但已偏离VWAP且短线回升的当前买先机会。"""
    frame=predictions_at(list(clocks),day=day)
    frame['symbol']='TEST'
    frame['prediction']=.001
    frame['f_vwap_deviation']=-.005
    frame['f_return_5m']=.0001
    frame['f_day_return']=0.
    frame['f_known_amplitude']=.01
    return frame


def ranked_rule(**updates) -> dict:
    """缺省为本轮合法VWAP排名规则，可用于单个边界覆盖。"""
    return {'signal_mode':'vwap_rank','profile':'l2_depth4','horizon_minutes':20,
            'rank_quantile':.70,'min_vwap_gap':.0035,'fraction':.25,'direction':'both','enabled':True,**updates}


class TestFrequencySignals(unittest.TestCase):
    """当前机会排序与参考身份测试，不将未来恢复条件用于首腿判断。"""

    def setUp(self):
        self.cfg=frequency_cfg()
        self.support={'bounds':{'f_input':{'lower':-1.,'upper':1.}}}
        self.reference={'profile':'l2_depth4','horizon_minutes':20,'net_margin':.0005,
                        'fraction':.25,'direction':'both','enabled':True}

    def test_fixed_25_rules_reference_nan_options_and_inputs_preserved(self):
        reference={**self.reference,'rank_quantile':np.nan,'min_vwap_gap':np.nan}
        before=deepcopy(reference)
        cfg_before=deepcopy(self.cfg)
        rules=create_frequency_rules(self.cfg,reference)
        self.assertEqual(len(rules),25)
        self.assertEqual([rule['candidate_index'] for rule in rules],list(range(25)))
        self.assertEqual([sum(rule['signal_mode']==mode for rule in rules) for mode in ('reference','absolute','vwap_rank')],[1,6,18])
        self.assertTrue(all(rule['enabled'] is True and rule['fraction']==.25 for rule in rules))
        self.assertTrue(np.isnan(rules[0]['rank_quantile']))
        self.assertEqual(reference.keys(),before.keys())
        self.assertEqual(self.cfg,cfg_before)
        rules[0]['profile']='changed'
        self.assertEqual(reference['profile'],'l2_depth4')

    def test_reference_exact_old_decisions_new_absolute_retains_later(self):
        frame=opportunities(('09:35','09:45','10:00'))
        frame.prediction=.01
        frame=frame.iloc[[2,0,1]].copy()
        reference={**self.reference,'signal_mode':'reference'}
        expected=create_short_decisions(frame,self.reference,self.cfg,self.support)
        actual=create_frequency_decisions(frame,reference,self.cfg,self.support)
        pd.testing.assert_frame_equal(actual,expected)
        new={**self.reference,'signal_mode':'absolute'}
        decisions=create_frequency_decisions(frame,new,self.cfg,self.support)
        self.assertEqual(len(expected),1)
        self.assertEqual(len(decisions),3)
        self.assertTrue(decisions.datetime.is_monotonic_increasing)
        old_signals=ShortHorizonTStrategy(self.reference,self.cfg,self.support).generate_signals(frame)
        new_signals=FrequencyTStrategy(new,self.cfg,self.support).generate_signals(frame)
        pd.testing.assert_frame_equal(new_signals,old_signals)
        funnel=build_frequency_signal_funnel(frame,reference,self.cfg,self.support)
        summary=summarize_frequency_funnel(funnel)
        self.assertEqual(summary['stages']['after_rank']['rows'],3)
        self.assertEqual(summary['stages']['request_eligible']['rows'],1)

    def test_rank_can_request_below_cost_without_claiming_net_edge(self):
        frame=opportunities()
        buy,sell=predicted_pair_edges(frame,self.cfg)
        self.assertLess(max(buy.iloc[0],sell.iloc[0]),0)
        funnel=build_frequency_signal_funnel(frame,ranked_rule(),self.cfg,self.support)
        self.assertTrue(funnel.signal_eligible.iloc[0])
        self.assertTrue(funnel.cost_margin_ok.iloc[0])
        self.assertLess(funnel.predicted_net_edge.iloc[0],0)
        self.assertFalse(funnel.attrs['prediction_cost_coverage_required'])
        self.assertEqual(create_frequency_decisions(frame,ranked_rule(),self.cfg,self.support).action.tolist(),[1])

    def test_vwap_direction_momentum_and_risk_gates(self):
        changes=[({'prediction':.001,'f_vwap_deviation':-.0035,'f_return_5m':0.},1),
                 ({'prediction':-.001,'f_vwap_deviation':.0035,'f_return_5m':0.},-1),
                 ({'prediction':.001,'f_vwap_deviation':.004},0),
                 ({'prediction':-.001,'f_vwap_deviation':-.004},0),
                 ({'f_return_5m':-.00001},0),
                 ({'prediction':-.001,'f_vwap_deviation':.004,'f_return_5m':.00001},0),
                 ({'f_day_return':.030001},0),({'f_day_return':-.030001},0),
                 ({'f_known_amplitude':.060001},0),({'f_known_amplitude':-.001},0),
                 ({'prediction':0.},0),({'f_vwap_deviation':np.nan},0),
                 ({'f_return_5m':np.nan},0),({'f_input':5.},0)]
        for updated,expected in changes:
            with self.subTest(updated=updated):
                frame=opportunities().assign(**updated)
                actual=FrequencyTStrategy(ranked_rule(),self.cfg,self.support).generate_signals(frame)
                self.assertEqual(actual.t_action.iloc[0],expected)
        boundary=opportunities().assign(f_day_return=.03,f_known_amplitude=.06)
        self.assertEqual(len(create_frequency_decisions(boundary,ranked_rule(),self.cfg,self.support)),1)

    def test_current_exit_lock_blocks_before_ranking(self):
        frame=opportunities().assign(signal_price=9.5,signal_high=9.5,signal_low=9.5)
        funnel=build_frequency_signal_funnel(frame,ranked_rule(),self.cfg,self.support)
        self.assertFalse(funnel.exit_guard_ok.iloc[0])
        self.assertFalse(funnel.pre_rank_eligible.iloc[0])
        self.assertFalse(funnel.signal_eligible.iloc[0])

    def test_rank_only_eligible_current_opportunities_and_ties_are_stable(self):
        frame=pd.concat([opportunities() for _ in range(11)],ignore_index=True)
        frame['symbol']=[f'S{i:02d}' for i in range(11)]
        frame.loc[10,'prediction']=1.
        frame.loc[10,'f_vwap_deviation']=.1
        shuffled=frame.sample(frac=1.,random_state=7)
        decisions=create_frequency_decisions(shuffled,ranked_rule(),self.cfg,self.support)
        self.assertEqual(decisions.symbol.tolist(),['S00','S01','S02'])
        funnel=build_frequency_signal_funnel(shuffled,ranked_rule(),self.cfg,self.support)
        self.assertTrue(funnel.eligible_opportunity_count.eq(10).all())
        self.assertTrue(funnel.index.equals(shuffled.index))
        later=frame.copy()
        later['datetime']+=pd.Timedelta(minutes=30)
        combined=pd.concat([frame,later],ignore_index=True)
        actual=create_frequency_decisions(combined,ranked_rule(),self.cfg,self.support)
        pd.testing.assert_frame_equal(actual.loc[actual.datetime.eq(frame.datetime.iloc[0])].reset_index(drop=True),decisions)

    def test_future_deletion_does_not_change_decisions_or_mutate_input(self):
        frame=opportunities(('09:35','10:00'))
        before=frame.copy(deep=True)
        rule=ranked_rule()
        expected=create_frequency_decisions(frame,rule,self.cfg,self.support)
        future=[column for column in frame if column.startswith(('execution_','recovery_','label_','target_'))]
        deleted=frame.drop(columns=future)
        actual=create_frequency_decisions(deleted,rule,self.cfg,self.support)
        pd.testing.assert_frame_equal(actual,expected)
        pd.testing.assert_frame_equal(frame,before)
        self.assertEqual(len(actual),2)

    def test_clock_lunch_deadline_empty_and_invalid_rules(self):
        frame=opportunities(('11:00','14:00','14:15'))
        signals=FrequencyTStrategy(ranked_rule(),self.cfg,self.support).generate_signals(frame)
        self.assertEqual(signals.planned_recovery_time.dt.strftime('%H:%M').tolist(),['11:21','14:21','14:36'])
        self.assertEqual(signals.t_action.tolist(),[1,1,0])
        empty=frame.iloc[:0].copy()
        self.assertTrue(create_frequency_decisions(empty,ranked_rule(),self.cfg,self.support).empty)
        for change in ({'rank_quantile':0.},{'rank_quantile':1.},{'min_vwap_gap':0.},{'horizon_minutes':40}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                create_frequency_decisions(frame,ranked_rule(**change),self.cfg,self.support)
        with self.assertRaises(ValueError):
            create_frequency_rules({**self.cfg,'frequency_absolute_net_margins':[False,.0005]},self.reference)

    def run_engine_case(self,clocks=('10:00',),first_capacity_zero=False):
        """仅生成微型真实价量账户，检验新增请求的费用与成功首腿日限制。"""
        frame=opportunities(clocks)
        samples=[]
        quotes=[]
        for i,clock in enumerate(clocks):
            sample,_=create_request(day='2024-01-09',side=1,execution=10.,recovery=10.,minute=clock)
            if first_capacity_zero and i==0:
                sample['execution_amount']=0.
            samples.append(sample)
            when=pd.Timestamp('2024-01-09 '+clock)+pd.Timedelta(minutes=21)
            quotes.append({'datetime':when,'symbol':'TEST','open':10.,'high':10.01,'low':9.99,
                           'close':10.,'volume':200000.,'amount':2000000.})
        for key in samples[0]:
            if key not in ('datetime','symbol'):
                frame[key]=[sample[key] for sample in samples]
        decisions=create_frequency_decisions(frame,ranked_rule(),self.cfg,self.support)
        cfg={**self.cfg,'initial_cash':100000.,'base_fraction':.3,'top_n':1,
             'max_bar_amount_fraction':1.,'max_pending_t_notional_fraction':.1}
        result=run_weekly_portfolio(pd.DataFrame([create_selection()]),
             create_daily(pd.bdate_range('2024-01-08','2024-01-12')),frame,decisions,cfg,recovery_bars=pd.DataFrame(quotes))
        return result,decisions

    def test_ranked_request_still_pays_actual_engine_fees_and_slippage(self):
        result,decisions=self.run_engine_case()
        self.assertEqual(len(decisions),1)
        self.assertEqual(result['metrics']['t_pair_count'],1)
        trip=result['round_trips'].iloc[0]
        self.assertGreater(result['metrics']['t_fee'],0)
        self.assertLess(trip.net_pnl,0)
        zero=opportunities().assign(prediction=0.)
        edge=predicted_pair_edges(zero,self.cfg)[0].iloc[0]
        self.assertAlmostEqual(trip.net_pnl,float(trip.shares)*10.*edge,places=7)
        self.assertTrue(result['daily_inventory_audit'].inventory_preserved.all())
        self.assertEqual(result['metrics']['t1_violation_count'],0)

    def test_failed_first_can_retry_but_only_one_successful_pair_per_day(self):
        result,decisions=self.run_engine_case(('10:00','10:30','11:00'),first_capacity_zero=True)
        self.assertEqual(len(decisions),3)
        self.assertEqual(result['metrics']['t_pair_count'],1)
        self.assertEqual(result['metrics']['started_pair_count'],1)
        first=result['trade_events'].loc[result['trade_events'].event_type.eq('t')].iloc[0]
        self.assertEqual(first.datetime,pd.Timestamp('2024-01-09 10:31'))
        self.assertTrue(result['daily_inventory_audit'].inventory_preserved.all())


class TestFrequencySelection(unittest.TestCase):
    """频率目标优先、到达目标后收益优先，不以零交易或失败候选占位。"""

    def candidate(self,index,net,pairs,days=10,**changes):
        return make_candidate(index,net,pairs,eligible_portfolio_days=days,**changes)

    def test_frequency_beats_sparse_profit_and_losses_remain_selectable(self):
        rows=pd.DataFrame([self.candidate(0,100000.,1),self.candidate(1,-100.,30),self.candidate(2,0.,0)])
        before=rows.copy(deep=True)
        chosen,annotated=select_frequency_candidate(rows,frequency_cfg())
        self.assertEqual(chosen['candidate_index'],1)
        self.assertEqual(chosen['frequency_score'],3.)
        self.assertEqual(annotated.selected.tolist(),[False,True,False])
        pd.testing.assert_frame_equal(rows,before)

    def test_target_caps_rate_then_net_profit_then_index(self):
        rows=pd.DataFrame([self.candidate(8,10.,40),self.candidate(2,10.,30),self.candidate(1,5.,80)])
        chosen,_=select_frequency_candidate(rows,frequency_cfg())
        self.assertEqual(chosen['candidate_index'],2)
        rows.loc[0,'incremental_net_pnl']=11.
        chosen,_=select_frequency_candidate(rows,frequency_cfg())
        self.assertEqual(chosen['candidate_index'],8)

    def test_execution_failure_denominator_and_no_eligible_cases(self):
        rows=pd.DataFrame([self.candidate(0,100.,100,recovery_failure_days=1),self.candidate(1,-5.,2)])
        chosen,_=select_frequency_candidate(rows,frequency_cfg())
        self.assertEqual(chosen['candidate_index'],1)
        denominator=pd.DataFrame([self.candidate(0,100.,100,days=100),self.candidate(1,0.,30,days=10)])
        chosen,_=select_frequency_candidate(denominator,frequency_cfg())
        self.assertEqual(chosen['candidate_index'],1)
        for value in (0,-1,np.nan,np.inf,True,.5):
            with self.subTest(days=value),self.assertRaises(ValueError):
                select_frequency_candidate(pd.DataFrame([self.candidate(0,1.,1,days=value)]),frequency_cfg())
        with self.assertRaises(ValueError):
            select_frequency_candidate(rows.drop(columns='eligible_portfolio_days'),frequency_cfg())
        with self.assertRaises(ValueError):
            select_frequency_candidate(pd.DataFrame([self.candidate(0,0.,0)]),frequency_cfg())


if __name__=='__main__':
    unittest.main(verbosity=2)
