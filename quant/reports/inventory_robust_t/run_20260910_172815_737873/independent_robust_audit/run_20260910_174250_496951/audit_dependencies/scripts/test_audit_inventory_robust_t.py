"""v17独审反例：只用合成输入，不训练模型或运行账户。"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_inventory_robust_t import (VERSION, KEYS, robust_rules, robust_decisions,
    attach_short_reference, validate_goal_status, verify_comparison, COMPARISON_FIELDS, score_clock_funnel,
    complete_pair_cost_rate, report_selection, RobustAuditor, AccountAuditor)
from scripts.inventory_robust_audit_models import (independent_robust_training_split,
    validate_native_parameters, compare_source_training, TRAIN_FILES)
from scripts.audit_dynamic_inventory_t import digest


class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(commission_rate=.0008, transfer_rate=.00001, slippage_rate=.0005,
            stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005, min_commission=0.,
            last_entry_time='14:30', fixed_recovery_time='14:55', t_avoid_locked_exit_side=True,
            flat_st_limit_rate=.05, nonflat_limit_proxy=.098)
        current = pd.DataFrame({'datetime': pd.to_datetime(['2026-03-02 09:35', '2026-03-02 14:00']),
            'symbol': ['SH.600001', 'SH.600002'], 'prediction': [.01, -.01], 'model_profile': 'huber_depth4',
            'signal_price': 10., 'signal_high': 10.01, 'signal_low': 9.99, 'previous_close': 10.,
            'signal_amount': 1e6, 'signal_volume': 1e5, 'model_version': VERSION,
            'forecast_loss': 'huber', 'forecast_alpha': .01})
        current['raw_prediction'], current['return_score'] = current.prediction, current.prediction
        self.pred = {h: current.assign(horizon_minutes=h) for h in (20,40,60)}

    def test_grid_exact_numeric_profiles(self):
        rules = robust_rules()
        self.assertEqual(len(rules), 37)
        self.assertEqual([r['candidate_index'] for r in rules], list(range(37)))
        self.assertEqual({r['profile'] for r in rules[1:]}, {'huber_depth4', 'huber_depth6'})
        self.assertEqual({r['min_net_edge'] for r in rules[1:]}, {0., .001, .003})
        self.assertTrue(robust_decisions({}, rules[0], self.cfg).empty)

    def test_shorter_tie_and_next_minute_anchor(self):
        result = robust_decisions(self.pred, robust_rules()[1], self.cfg)
        self.assertEqual(result.horizon_minutes.tolist(), [20,20])
        self.assertEqual(result.planned_recovery_time.tolist(), list(pd.to_datetime(['2026-03-02 09:56','2026-03-02 14:21'])))
        self.assertEqual(result.action.tolist(), [1,-1])

    def test_horizon_current_argmax_and_lunch(self):
        pred = {h: p.iloc[:1].assign(datetime=pd.Timestamp('2026-03-02 11:00')) for h,p in self.pred.items()}
        pred[40] = pred[40].assign(prediction=.03, raw_prediction=.03, return_score=.03)
        result = robust_decisions(pred, robust_rules()[1], self.cfg)
        self.assertEqual(result.horizon_minutes.tolist(), [40])
        self.assertEqual(result.planned_recovery_time.iloc[0], pd.Timestamp('2026-03-02 13:11'))

    def test_late_sixty_minutes_never_rolls_to_next_day(self):
        pred = {h:p.iloc[1:].assign(prediction=.001,raw_prediction=.001,return_score=.001) for h,p in self.pred.items()}
        pred[60] = pred[60].assign(prediction=-.01,raw_prediction=-.01,return_score=-.01)
        self.assertTrue(robust_decisions(pred, robust_rules()[1], self.cfg).empty)

    def test_future_changes_do_not_change_current_decision(self):
        expected = robust_decisions(self.pred, robust_rules()[1], self.cfg)
        changed = {h:p.assign(execution_price=np.nan,recovery_price=0.,label_available=False,target_return=-999.) for h,p in self.pred.items()}
        pd.testing.assert_frame_equal(robust_decisions(changed, robust_rules()[1], self.cfg), expected)

    def test_identity_alias_or_clock_mismatch_fails(self):
        for field,value in [('return_score',.25), ('forecast_loss','regression'), ('forecast_alpha',.02), ('model_profile','l2_depth4'), ('model_version','v16')]:
            pred = copy.deepcopy(self.pred); pred[20][field]=value
            with self.assertRaises(AssertionError): robust_decisions(pred, robust_rules()[1], self.cfg)
        pred=copy.deepcopy(self.pred); pred[60]=pred[60].iloc[:1]
        with self.assertRaises(AssertionError): robust_decisions(pred, robust_rules()[1], self.cfg)
        pred={h:p.assign(datetime=pd.Timestamp('2026-03-02 09:35:01')) for h,p in self.pred.items()}
        with self.assertRaises(AssertionError): robust_decisions(pred, robust_rules()[1], self.cfg)

    def test_cost_and_current_exit_side_gate(self):
        for changes in ({'signal_volume':0.}, {'prediction':.001,'raw_prediction':.001,'return_score':.001},
                        {'signal_price':9.5,'signal_high':9.5,'signal_low':9.5,'prediction':.01,'raw_prediction':.01,'return_score':.01}):
            pred={h:p.assign(**changes) for h,p in self.pred.items()}
            self.assertTrue(robust_decisions(pred, robust_rules()[1], self.cfg).empty)

    def test_attach_keeps_missing_current_key_and_float32(self):
        core=self.pred[20][KEYS].copy()
        core['f_x']=np.array([.1,.2],dtype='float32')
        core['feature_source_start']=pd.to_datetime(['2026-01-01',None])
        labels=core.iloc[:1][KEYS].copy()
        labels['horizon_minutes']=20
        labels['execution_time']=labels.datetime+pd.Timedelta(minutes=1)
        labels['recovery_time']=labels.datetime+pd.Timedelta(minutes=21)
        labels['label_start'],labels['label_end']=labels.execution_time,labels.recovery_time
        labels['execution_price'],labels['recovery_price'],labels['target_return']=10.,10.1,10.1/10.-1
        result=attach_short_reference(core,labels,20)
        pd.testing.assert_frame_equal(result[list(core)],core)
        self.assertEqual(result.label_available.tolist(),[True,False])
        self.assertTrue(pd.isna(result.execution_time.iloc[1]))
        self.assertEqual(result.planned_recovery_time.iloc[1],pd.Timestamp('2026-03-02 14:21'))
        with self.assertRaises(AssertionError): attach_short_reference(core.assign(trade_week_complete=True),labels,20)
        with self.assertRaises(AssertionError): attach_short_reference(core,pd.concat([labels,labels]),20)

    def test_full_core_purge_preserves_missing_reason_and_index(self):
        frame=pd.DataFrame({'datetime':pd.to_datetime(['2024-01-08 09:35','2024-03-04 09:35','2024-03-05 09:35','2025-03-05 09:35']),
            'symbol':'SH.600001','feature_source_start':pd.to_datetime(['2023-12-11','2024-02-05','2024-02-05',None]),
            'horizon_minutes':20,'target_return':[.01,.02,np.nan,.03],'label_available':[True,True,False,True]})
        frame['label_start']=frame.datetime+pd.Timedelta(minutes=1)
        frame['label_end']=frame.datetime+pd.Timedelta(minutes=21)
        cfg=dict(strict_feature_provenance=True,exclude_flat_signal_bars=False,embargo_days=7)
        kept,audit,meta=independent_robust_training_split(frame,dict(core_years=[2024,2025],test_year=2023,inner_year=2026),cfg)
        self.assertEqual(kept.index.tolist(),[1])
        self.assertEqual(meta['input_rows'],4)
        self.assertEqual(audit.set_index('reason').loc['invalid_target_return','excluded_rows'],1)

    def native_fixture(self):
        class Native:
            def __init__(self): self.params={'objective':'huber','alpha':.01}; self.text='real-trees'; self.names=['f_x']
            def feature_name(self): return self.names
            def model_to_string(self): return self.text
        class Model:
            def __init__(self): self.booster_=Native(); self.parameters={'objective':'huber','alpha':.01,'num_leaves':15}
            def get_params(self): return self.parameters
        return Native(),Model(),{'objective':'huber','alpha':.01,'num_leaves':15}

    def test_actual_native_huber_alpha_is_required(self):
        native,model,parameters=self.native_fixture()
        validate_native_parameters(native,model,parameters,['f_x'])
        for field,value in [('objective','regression'),('alpha',.9)]:
            other=copy.deepcopy(native); other.params[field]=value
            with self.assertRaises(AssertionError): validate_native_parameters(other,model,parameters,['f_x'])

    def test_all_native_trees_and_features_must_match(self):
        native,model,parameters=self.native_fixture()
        for field,value in [('text','tampered-leaf'),('names',['f_y'])]:
            other=copy.deepcopy(native); setattr(other,field,value)
            with self.assertRaises(AssertionError): validate_native_parameters(other,model,parameters,['f_x'])

    def test_source_matrix_dtype_tamper_cannot_hide_in_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=Path(tmp)/'old',Path(tmp)/'new'; old.mkdir();new.mkdir()
            meta=dict(research_version='inventory_learning_t_v14',profile='l2_depth4',horizon_minutes=20,
                train_years=[2023,2024],excluded_years=[2025,2026],features=['f_x'],fitted_rows=2,training_rows=2,
                full_training_rows_sha256='a',training_matrix_sha256='b',training_target_sha256='c',
                model_parameters={'objective':'regression','n_jobs':1,'num_leaves':15})
            current={**meta,'research_version':VERSION,'profile':'huber_depth4',
                'model_parameters':{'objective':'huber','alpha':.01,'n_jobs':2,'num_leaves':15}}
            (old/'model_metadata.json').write_text(json.dumps(meta))
            for folder in (old,new):
                pd.DataFrame({'f_x':np.array([.1,.2],dtype='float32')}).to_pickle(folder/TRAIN_FILES[0])
                for name in TRAIN_FILES[1:]: pd.DataFrame({'x':[1,2]}).to_csv(folder/name,index=False)
            class ReadOnly:
                def record(self,path): return Path(path)
                def csv(self,path): return pd.read_csv(path,float_precision='round_trip')
                def json(self,path): return json.loads(Path(path).read_text())
            auditor=ReadOnly()
            auditor.source_audits={'learning':(None,{'inputs_sha256':{str(p.resolve()):digest(p) for p in old.iterdir()}})}
            result=compare_source_training(auditor,new,old,current)
            self.assertEqual(len(result['four_training_artifacts']),4)
            frame=pd.read_pickle(new/TRAIN_FILES[0]).astype('float64');frame.to_pickle(new/TRAIN_FILES[0])
            with self.assertRaises(AssertionError): compare_source_training(auditor,new,old,current)
            frame.astype('float32').to_pickle(new/TRAIN_FILES[0])
            auditor.source_audits['learning'][1]['inputs_sha256'][str((old/TRAIN_FILES[0]).resolve())]='fake'
            with self.assertRaises(AssertionError): compare_source_training(auditor,new,old,current)

    def test_engineering_zero_trades_is_not_goal_success(self):
        cfg={'report_scope':'smoke','minimum_completed_pair_lots':None,'minimum_active_stock_days':None}
        validate_goal_status({'goal_achieved':False,'frequency_target_status':'not_quantified'},cfg,True)
        with self.assertRaises(AssertionError): validate_goal_status({'goal_achieved':True,'frequency_target_status':'not_quantified'},cfg,True)

    def test_same_rule_comparison_rejects_mapping_and_profit_change(self):
        current=pd.DataFrame(robust_rules());original=current.copy()
        original['profile']=original.profile.str.replace('huber_','l2_')
        for name in COMPARISON_FIELDS:current[name]=1.;original[name]=1.
        current['incremental_net_pnl']=2.
        saved=current[['candidate_index']].copy()
        for name in COMPARISON_FIELDS:
            saved[name+'_robust']=current[name];saved[name+'_source']=original[name]
            saved[name+'_change']=current[name]-original[name]
        saved['current_profile'],saved['source_profile']=current.profile,original.profile
        saved['same_forecast_target'],saved['same_forecasts'],saved['same_execution_guard']=True,False,True
        verify_comparison(current,original,saved)
        changed=saved.copy();changed.loc[1,'incremental_net_pnl_change']=999.
        with self.assertRaises(AssertionError):verify_comparison(current,original,changed)
        changed=current.copy();changed.loc[1,'fraction']=.25
        with self.assertRaises(AssertionError):verify_comparison(changed,original,saved)

    def test_zero_request_funnel_retains_late_profitable_score(self):
        pred={h:p.iloc[1:].assign(prediction=.001,raw_prediction=.001,return_score=.001) for h,p in self.pred.items()}
        pred[60]=pred[60].assign(prediction=-.01,raw_prediction=-.01,return_score=-.01)
        packed=dict(test_year=2026,scope='test',candidate_index=1,request_count=0,
            **score_clock_funnel(pred,robust_rules()[1],self.cfg))
        self.assertEqual(packed['scope'],'test')
        result=packed['horizons']
        self.assertEqual([r['score_margin_pass_rows'] for r in result],[0,0,1])
        self.assertEqual(result[2]['score_pass_but_clock_invalid_rows'],1)
        self.assertEqual(len(result[2]['first_clock_rejected_examples']),1)

    def test_pending_marked_pnl_is_not_complete_profit_ratio(self):
        actual={'all_reference_acquisition_cost':10000.,'realized_complete_pair_pnl':20.,'incremental_net_pnl':-100.}
        self.assertEqual(complete_pair_cost_rate(actual),.002)
        self.assertIsNone(complete_pair_cost_rate({**actual,'all_reference_acquisition_cost':0.}))

    def test_hold_report_source_profile_is_null(self):
        self.assertIsNone(report_selection(2026,robust_rules()[0])['source_l2_profile'])
        self.assertEqual(report_selection(2026,robust_rules()[1])['source_l2_profile'],'l2_depth4')

    def test_account_adds_independent_t_slippage_not_base_slippage(self):
        from unittest.mock import patch
        auditor=RobustAuditor.__new__(RobustAuditor)
        auditor.table=lambda path:pd.DataFrame({'event_type':['weekly','t','t'],'slippage_cost':[20.,.3,.4],
            'shares':[100,100,200],'price':[10.2,10.003,9.998],'reference_price':[10.,10.,10.]})
        auditor.json=lambda path:{'t_slippage_cost':.7}
        with patch.object(AccountAuditor,'account',return_value={'passed':True}):
            self.assertAlmostEqual(auditor.account(Path('unused'))['t_slippage_cost'],.7)
        auditor.json=lambda path:{'t_slippage_cost':20.7}
        with patch.object(AccountAuditor,'account',return_value={'passed':True}):
            with self.assertRaises(AssertionError):auditor.account(Path('unused'))


def run_self_tests():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful(): raise AssertionError('v17独立审计反例失败')
    return {'passed':True,'tests':result.testsRun}


if __name__=='__main__': run_self_tests()
