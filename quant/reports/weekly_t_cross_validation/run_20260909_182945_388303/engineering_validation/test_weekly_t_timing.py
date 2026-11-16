"""验证新增择时特征、当前方向/期限决策和内部净利选择；仅用合成数据，无训练或写缓存。

覆盖完整市场同刻特征、未来独立、真实历史成本边界、三期限时钟、透明hold以及
旧库存引擎的失败后续请求。直接运行使用本项目quant Python环境。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.weekly_t_timing import (DECISION_COLUMNS, TIMING_FEATURE_COLUMNS,
    add_timing_features, make_timing_decisions, select_timing_candidate, timing_candidates)
from quant_backtest.weekly_t_short_horizon import predicted_pair_edges
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from scripts.test_weekly_t_engine import create_daily, create_selection, create_request
from scripts.test_weekly_t_short_horizon import predictions_at, short_cfg


def timing_cfg() -> dict:
    """明确固定候选与旧费用；不会消费运行中YAML或任何历史绩效。"""
    return {**short_cfg(), 't_model_profiles':['l2_depth4', 'l2_depth6'],
            'timing_horizons':[20, 40, 60], 'timing_regimes':['learned', 'reversal', 'momentum'],
            'timing_min_net_edges':[.0005, .001], 'timing_min_profit_probabilities':[.4, .5],
            'timing_max_abs_day_return':.03, 'timing_max_known_amplitude':.06}


def current_features(clocks=('10:00',), day='2024-01-09') -> pd.DataFrame:
    """当前价格回落后反弹，原预测含未来列以检测决策是否误读。"""
    frame = predictions_at(list(clocks), day)
    frame['symbol'] = 'TEST'
    additions = {'raw_prediction':.01, 'prob_buy_profit':.8, 'prob_sell_profit':.8,
                 'f_return_5m':.001, 'f_return_15m':.002, 'f_open_return':.003,
                 'f_day_return':.004, 'f_amount_vs_previous_day':.4, 'f_session_fraction':.25,
                 'f_known_amplitude':.01, 'f_realized_volatility':.003,
                 'f_previous_day_amplitude':.02, 'f_vwap_deviation':-.004}
    for name, value in additions.items():
        frame[name] = value
    frame['feature_source_start'] = pd.Timestamp(day) - pd.Timedelta(days=10)
    return add_timing_features(frame, timing_cfg())


def prediction_inputs(frame: pd.DataFrame) -> tuple[dict, dict]:
    """三期限都覆盖完整当前键，支持域来自合成固定训练边界。"""
    return ({h:frame.copy(deep=True) for h in (20, 40, 60)},
            {h:{'bounds':{'f_input':{'lower':-1., 'upper':1.}}} for h in (20, 40, 60)})


def rule(regime='learned', probability=.4, margin=.0005) -> dict:
    """取预声明候选，测试不通过任意candidate_index旁路参数契约。"""
    return next(c for c in timing_candidates(timing_cfg()) if c['profile']=='l2_depth4'
                and c['regime']==regime and c['min_net_edge']==margin
                and c['min_profit_probability']==probability)


class TimingFeatureTests(unittest.TestCase):
    """当前截面的公式、缺失观测及来源保持语义。"""

    def test_full_same_clock_market_and_formula_before_slice(self):
        frame = pd.concat([current_features(), current_features()], ignore_index=True)
        frame['symbol'] = ['A', 'B']
        frame['f_return_5m'] = [.001, .003]
        frame['f_return_15m'] = [.002, .006]
        frame['f_open_return'] = [.003, -.004]
        frame['f_day_return'] = [.01, .03]
        frame.index = [8, 2]
        before = frame.copy(deep=True)
        out = add_timing_features(frame, timing_cfg())
        self.assertEqual(len(TIMING_FEATURE_COLUMNS), 11)
        np.testing.assert_allclose(out.f_timing_market_return_5m, .002)
        np.testing.assert_allclose(out.f_timing_market_return_15m, .004)
        np.testing.assert_allclose(out.f_timing_excess_return_5m, [-.001, .001])
        np.testing.assert_allclose(out.f_timing_market_breadth, .5)
        np.testing.assert_allclose(out.f_timing_market_day_return_std, .01)
        np.testing.assert_allclose(out.f_timing_volume_rate, 1.6)
        np.testing.assert_allclose(out.f_timing_trend_efficiency, [.3, .4])
        sigma = max(.003*np.sqrt(20/60), .02*.25*np.sqrt(20/240), .0005)
        np.testing.assert_allclose(out.f_timing_sigma20, sigma)
        self.assertAlmostEqual(out.f_timing_vwap_z.iloc[0], -.004/sigma, places=6)
        self.assertAlmostEqual(out.f_timing_reversal_confirmation.iloc[0], .001/sigma, places=6)
        self.assertNotEqual(float(out.f_timing_market_return_5m.iloc[0]),
                            float(add_timing_features(frame.iloc[:1], timing_cfg()).f_timing_market_return_5m.iloc[0]))
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_series_equal(out.feature_source_start, before.feature_source_start)
        self.assertTrue(out.index.equals(before.index))

    def test_future_rows_and_future_columns_cannot_change_current_features(self):
        frame = current_features()
        expected = add_timing_features(frame, timing_cfg())
        later = frame.assign(datetime=frame.datetime+pd.Timedelta(days=1), f_return_5m=999., f_open_return=-999.)
        combined = pd.concat([frame, later], ignore_index=True)
        combined[['recovery_price', 'execution_price', 'target_return']] = -999999.
        actual = add_timing_features(combined, timing_cfg()).iloc[:1]
        pd.testing.assert_frame_equal(actual[TIMING_FEATURE_COLUMNS], expected[TIMING_FEATURE_COLUMNS])

    def test_finite_donors_stability_floor_and_invalid_keys(self):
        frame = pd.concat([current_features(), current_features()], ignore_index=True)
        frame['symbol'] = ['A', 'B']
        frame.loc[1, ['f_return_5m', 'f_open_return', 'f_day_return']] = np.nan
        frame['f_session_fraction'] = 0.
        frame['f_realized_volatility'] = 0.
        frame['f_previous_day_amplitude'] = 0.
        out = add_timing_features(frame, timing_cfg())
        np.testing.assert_allclose(out.f_timing_market_return_5m, .001)
        np.testing.assert_allclose(out.f_timing_market_breadth, 1.)
        np.testing.assert_allclose(out.f_timing_market_day_return_std, 0.)
        np.testing.assert_allclose(out.f_timing_sigma20, .0005)
        np.testing.assert_allclose(out.f_timing_volume_rate, 8.)
        broken = frame.copy();broken.loc[0, 'f_realized_volatility'] = np.nan
        self.assertTrue(pd.isna(add_timing_features(broken, timing_cfg()).f_timing_sigma20.iloc[0]))
        with self.assertRaises(ValueError):
            add_timing_features(pd.concat([frame, frame]), timing_cfg())


class TimingDecisionTests(unittest.TestCase):
    """成本、方向概率、市场状态、多期限以及请求与真实成交区别。"""

    def setUp(self):
        self.cfg = timing_cfg()

    def decide(self, frame=None, candidate=None, inputs=None):
        predictions, support = inputs or prediction_inputs(current_features() if frame is None else frame)
        return make_timing_decisions(predictions, support, candidate or rule(), self.cfg)

    def test_exact_25_candidates_and_transparent_hold(self):
        before = deepcopy(self.cfg)
        candidates = timing_candidates(self.cfg)
        self.assertEqual([c['candidate_index'] for c in candidates], list(range(25)))
        self.assertFalse(candidates[0]['enabled'])
        self.assertEqual(sum(c['enabled'] for c in candidates), 24)
        self.assertEqual(len({(c['profile'], c['regime'], c['min_net_edge'], c['min_profit_probability']) for c in candidates}), 25)
        out, audit = make_timing_decisions({}, {}, candidates[0], self.cfg)
        self.assertTrue(out.empty);self.assertEqual(list(out), DECISION_COLUMNS)
        self.assertEqual(audit['reason'], 'explicit_hold')
        self.assertEqual(self.cfg, before)
        for change in ({'timing_horizons':[20]}, {'timing_min_net_edges':[0.,.001]}, {'short_t_fraction':.5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                timing_candidates({**self.cfg, **change})

    def test_cost_uses_raw_prediction_and_historical_stamp(self):
        frame = current_features().assign(prediction=.9, raw_prediction=.003)
        self.assertTrue(self.decide(frame)[0].empty)
        frame.raw_prediction=.004
        out, audit = self.decide(frame)
        self.assertEqual(out.action.tolist(), [1])
        expected = predicted_pair_edges(frame.assign(prediction=frame.raw_prediction), self.cfg)[0].iloc[0]
        self.assertAlmostEqual(out.predicted_net_edge.iloc[0], expected)
        self.assertEqual(out.candidate_index.iloc[0], rule()['candidate_index'])
        before = frame.assign(datetime=pd.Timestamp('2023-08-25 10:00'))
        after = frame.assign(datetime=pd.Timestamp('2023-08-28 10:00'))
        self.assertTrue(self.decide(before)[0].empty)
        self.assertEqual(len(self.decide(after)[0]), 1)
        with self.assertRaises(ValueError):
            p,s=prediction_inputs(frame);make_timing_decisions(p,s,rule(),{**self.cfg,'min_commission':5.})

    def test_direction_probability_and_invalid_probability_are_side_specific(self):
        buy = current_features().assign(prob_buy_profit=.4, prob_sell_profit=0.)
        self.assertEqual(self.decide(buy)[0].action.tolist(), [1])
        self.assertTrue(self.decide(buy, rule(probability=.5))[0].empty)
        sell = buy.assign(raw_prediction=-.01, prob_buy_profit=0., prob_sell_profit=.5)
        self.assertEqual(self.decide(sell, rule(probability=.5))[0].action.tolist(), [-1])
        for value in (np.nan, np.inf, -.1, 1.1):
            with self.subTest(value=value):
                self.assertTrue(self.decide(buy.assign(prob_buy_profit=value))[0].empty)

    def test_current_risk_support_and_exit_side_still_block(self):
        for change in ({'f_day_return':.03001}, {'f_known_amplitude':.06001},
                       {'f_known_amplitude':-.01}, {'f_input':2.}, {'raw_prediction':np.nan},
                       {'signal_price':9.5, 'signal_high':9.5, 'signal_low':9.5}):
            with self.subTest(change=change):
                self.assertTrue(self.decide(current_features().assign(**change))[0].empty)
        self.assertEqual(len(self.decide(current_features().assign(f_day_return=.03,f_known_amplitude=.06))[0]),1)

    def test_reversal_requires_opposite_vwap_and_strict_confirmation(self):
        frame = current_features().assign(f_timing_trend_efficiency=.65, f_timing_vwap_z=-.75)
        self.assertEqual(self.decide(frame, rule('reversal'))[0].action.tolist(), [1])
        for change in ({'f_return_5m':0.}, {'f_return_5m':-.001}, {'f_vwap_deviation':.004},
                       {'f_timing_trend_efficiency':.651}, {'f_timing_vwap_z':-.749}):
            with self.subTest(change=change):
                self.assertTrue(self.decide(frame.assign(**change), rule('reversal'))[0].empty)
        sell=frame.assign(raw_prediction=-.01,f_vwap_deviation=.004,f_timing_vwap_z=.75,f_return_5m=-.001)
        self.assertEqual(self.decide(sell,rule('reversal'))[0].action.tolist(),[-1])

    def test_momentum_all_four_signs_and_efficiency(self):
        frame=current_features().assign(f_vwap_deviation=.002,f_timing_trend_efficiency=.55)
        self.assertEqual(self.decide(frame,rule('momentum'))[0].action.tolist(),[1])
        for column in ('f_return_15m','f_open_return','f_vwap_deviation','f_return_5m'):
            with self.subTest(column=column):
                self.assertTrue(self.decide(frame.assign(**{column:0.}),rule('momentum'))[0].empty)
        self.assertTrue(self.decide(frame.assign(f_timing_trend_efficiency=.549),rule('momentum'))[0].empty)
        sell=frame.assign(raw_prediction=-.01,f_return_15m=-.002,f_open_return=-.003,f_vwap_deviation=-.002,f_return_5m=-.001)
        self.assertEqual(self.decide(sell,rule('momentum'))[0].action.tolist(),[-1])

    def test_choose_highest_net_horizon_lunch_deadline_and_later_requests(self):
        p,s=prediction_inputs(current_features(('10:00','10:30','13:30','14:00')))
        for h,value in [(20,.005),(40,.007),(60,.01)]:p[h].raw_prediction=value
        out,_=self.decide(inputs=(p,s))
        self.assertEqual(out.horizon_minutes.tolist(),[60,60,40,20])
        self.assertEqual(out.planned_recovery_time.dt.strftime('%H:%M').tolist(),['11:01','13:01','14:11','14:21'])
        self.assertEqual(len(out),4)
        tied=self.decide(current_features())[0]
        self.assertEqual(tied.horizon_minutes.tolist(),[20])
        frame=current_features()
        with patch('quant_backtest.weekly_t_timing.predicted_pair_edges',side_effect=lambda data,cfg:(pd.Series(.01,index=data.index),pd.Series(.01,index=data.index))):
            out,_=self.decide(frame)
        self.assertEqual(out.action.tolist(),[1])

    def test_future_columns_row_order_and_missing_horizon_keys(self):
        frame=current_features(('09:35','10:00')).iloc[::-1].copy()
        p,s=prediction_inputs(frame);before={h:f.copy(deep=True) for h,f in p.items()}
        expected,audit=self.decide(inputs=(p,s))
        for h in p:
            p[h]=p[h].drop(columns=[c for c in p[h] if c.startswith(('execution_','recovery_','label_','target_'))])
        actual,actual_audit=self.decide(inputs=(p,s))
        pd.testing.assert_frame_equal(actual,expected);self.assertEqual(actual_audit,audit)
        pd.testing.assert_frame_equal(frame,before[20])
        p[60]=p[60].iloc[:1]
        with self.assertRaises(ValueError):self.decide(inputs=(p,s))

    def test_empty_predictions_have_stable_schema(self):
        out,audit=self.decide(current_features().iloc[:0])
        self.assertTrue(out.empty);self.assertEqual(list(out),DECISION_COLUMNS)
        self.assertEqual(audit['selected_requests'],0)

    def test_model_horizon_and_profile_identity_are_not_interchangeable(self):
        p,s=prediction_inputs(current_features())
        for h in p:p[h]=p[h].assign(horizon_minutes=h,model_profile='l2_depth4')
        self.assertEqual(len(self.decide(inputs=(p,s))[0]),1)
        p[40]['horizon_minutes']=20
        with self.assertRaises(ValueError):self.decide(inputs=(p,s))
        p[40]['horizon_minutes']=40;p[60]['model_profile']='l2_depth6'
        with self.assertRaises(ValueError):self.decide(inputs=(p,s))

    def test_original_engine_retries_later_then_limits_success_to_one(self):
        frame=current_features(('10:00','10:30','11:00'))
        samples=[];quotes=[]
        for i,clock in enumerate(('10:00','10:30','11:00')):
            sample,_=create_request(day='2024-01-09',side=1,execution=10.,recovery=10.1,minute=clock)
            if i==0:sample['execution_amount']=0.
            samples.append(sample)
            quotes.append({'datetime':pd.Timestamp('2024-01-09 '+clock)+pd.Timedelta(minutes=21),
                           'symbol':'TEST','open':10.1,'high':10.11,'low':10.09,'close':10.1,'volume':200000.,'amount':2000000.})
        for key in samples[0]:
            if key not in ('datetime','symbol'):frame[key]=[sample[key] for sample in samples]
        decisions,_=self.decide(frame)
        cfg={**self.cfg,'initial_cash':100000.,'base_fraction':.3,'top_n':1,'max_bar_amount_fraction':1.,'max_pending_t_notional_fraction':.1}
        result=run_weekly_portfolio(pd.DataFrame([create_selection()]),create_daily(pd.bdate_range('2024-01-08','2024-01-12')),
                                    frame,decisions,cfg,recovery_bars=pd.DataFrame(quotes))
        self.assertEqual(len(decisions),3)
        self.assertEqual(result['metrics']['t_pair_count'],1)
        first=result['trade_events'].query("event_type=='t' and fragment_id==0").iloc[0]
        self.assertEqual(first.datetime,pd.Timestamp('2024-01-09 10:31'))
        self.assertGreater(result['round_trips'].net_pnl.sum(),0)
        self.assertTrue(result['daily_inventory_audit'].inventory_preserved.all())
        self.assertEqual(result['metrics']['t1_violation_count'],0)


class TimingSelectionTests(unittest.TestCase):
    """净利优先、五项失败剔除，缺盈利机会明确选hold。"""

    def row(self,index,net=0.,pairs=0,**changes):
        return {'candidate_index':index,'enabled':index!=0,'incremental_net_pnl':net,'t_pair_count':pairs,
                'inventory_mismatch_days':0,'recovery_failure_days':0,'pending_pairs':0,
                'base_plan_mismatch_count':0,'t1_violation_count':0,**changes}

    def test_profit_beats_frequency_and_no_100_pair_or_ci_gate(self):
        table=pd.DataFrame([self.row(0),self.row(8,10.,1,frequency_score=0.,ci_low=-100.),self.row(2,9.,1000,frequency_score=3.,ci_low=100.)])
        before=table.copy(deep=True);chosen,annotated=select_timing_candidate(table)
        self.assertEqual(chosen['candidate_index'],8);self.assertTrue(chosen['enabled'])
        self.assertEqual(annotated.selected.tolist(),[False,True,False]);pd.testing.assert_frame_equal(table,before)
        table.loc[2,'incremental_net_pnl']=10.
        self.assertEqual(select_timing_candidate(table)[0]['candidate_index'],2)

    def test_all_execution_failures_disqualify_and_hold_reason_is_explicit(self):
        for field in ('inventory_mismatch_days','recovery_failure_days','pending_pairs','base_plan_mismatch_count','t1_violation_count'):
            with self.subTest(field=field):
                table=pd.DataFrame([self.row(0),self.row(1,10.,1,**{field:1}),self.row(2,-1.,100),self.row(3,0.,0)])
                chosen,annotated=select_timing_candidate(table)
                self.assertEqual(chosen['candidate_index'],0);self.assertFalse(chosen['enabled'])
                self.assertEqual(chosen['selection_reason'],'no_profitable_execution_valid_candidate')
                self.assertIn(field,annotated.iloc[1].timing_rejection_reasons)

    def test_invalid_counts_fake_hold_and_pair_fragment_confusion_rejected(self):
        for table in [pd.DataFrame([self.row(1,1.,1)]),pd.DataFrame([self.row(0,1.,0)]),
                      pd.DataFrame([self.row(0),self.row(1,1.,.5)]),
                      pd.DataFrame([self.row(0,completed_pair_count=0),self.row(1,1.,1,completed_pair_count=2)]),
                      pd.DataFrame([self.row(0),self.row(1,np.nan,1)])]:
            with self.subTest(table=table.to_dict()),self.assertRaises(ValueError):select_timing_candidate(table)


if __name__=='__main__':
    unittest.main(verbosity=2)
