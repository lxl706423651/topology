"""v18独审有界反例；合成数据与伪原生对象，不拟合或重放账户。"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_inventory_coverage_t import (VERSION,KEYS,coverage_rules,coverage_decisions,
    attach_short_reference,verify_comparison,COMPARISON_FIELDS,score_clock_funnel,complete_pair_cost_rate,
    report_selection,CoverageAuditor,AccountAuditor,validate_goal_status,assert_no_outer_products)
from scripts.inventory_coverage_audit_models import (coverage_feature_selection,validate_native_parameters,
    compare_source_training,CONTROL_PAIRS,INPUT_KEYS,independent_training_split)
from scripts.audit_dynamic_inventory_t import digest


class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(commission_rate=.0008,transfer_rate=.00001,slippage_rate=.0005,
            stamp_rate_before_20230828=.001,stamp_rate_from_20230828=.0005,min_commission=0.,
            last_entry_time='14:30',fixed_recovery_time='14:55',t_avoid_locked_exit_side=True,
            flat_st_limit_rate=.05,nonflat_limit_proxy=.098)
        current=pd.DataFrame({'datetime':pd.to_datetime(['2026-03-02 09:35','2026-03-02 14:00']),
            'symbol':['SH.600001','SH.600002'],'prediction':[.01,-.01],'model_profile':'l2_depth4',
            'signal_price':10.,'signal_high':10.01,'signal_low':9.99,'previous_close':10.,
            'signal_amount':1e6,'signal_volume':1e5,'model_version':VERSION,'forecast_loss':'regression_l2','feature_min_coverage':.7})
        current['raw_prediction'],current['return_score']=current.prediction,current.prediction
        self.pred={h:current.assign(horizon_minutes=h) for h in (20,40,60)}
        self.selector=dict(feature_min_coverage=.7,source_feature_min_coverage=.8,
            feature_selection_sample_rows=20000,random_state=42,feature_max_correlation=.95,
            t_max_features=60,excluded_model_features=['f_history_days'])

    def test_37_original_rules_not_38_features_or_huber(self):
        rules=coverage_rules()
        self.assertEqual(len(rules),37)
        self.assertEqual([r['candidate_index'] for r in rules],list(range(37)))
        self.assertEqual({r['profile'] for r in rules[1:]},{'l2_depth4','l2_depth6'})
        self.assertTrue(coverage_decisions({},rules[0],self.cfg).empty)
        self.assertNotIn('source_l2_profile',report_selection(2026,rules[0]))

    def test_current_tie_shorter_next_minute_and_lunch(self):
        result=coverage_decisions(self.pred,coverage_rules()[1],self.cfg)
        self.assertEqual(list(result),['datetime','symbol','action','priority','planned_recovery_time','prediction',
            'horizon_minutes','fraction','candidate_index','signal_strength','predicted_net_edge','regime','model_profile',
            'model_version','forecast_loss','feature_min_coverage'])
        self.assertNotIn('return_score',result)
        self.assertEqual(result.horizon_minutes.tolist(),[20,20])
        self.assertEqual(result.planned_recovery_time.tolist(),list(pd.to_datetime(['2026-03-02 09:56','2026-03-02 14:21'])))
        pred={h:p.iloc[:1].assign(datetime=pd.Timestamp('2026-03-02 11:00')) for h,p in self.pred.items()}
        pred[40]=pred[40].assign(prediction=.03,raw_prediction=.03,return_score=.03)
        result=coverage_decisions(pred,coverage_rules()[1],self.cfg)
        self.assertEqual(result.planned_recovery_time.iloc[0],pd.Timestamp('2026-03-02 13:11'))

    def test_late_h60_zero_request_funnel_real_packing(self):
        pred={h:p.iloc[1:].assign(prediction=.001,raw_prediction=.001,return_score=.001) for h,p in self.pred.items()}
        pred[60]=pred[60].assign(prediction=-.01,raw_prediction=-.01,return_score=-.01)
        self.assertTrue(coverage_decisions(pred,coverage_rules()[1],self.cfg).empty)
        packed=dict(scope='test',request_count=0,**score_clock_funnel(pred,coverage_rules()[1],self.cfg))
        self.assertEqual([r['score_margin_pass_rows'] for r in packed['horizons']],[0,0,1])
        self.assertEqual(packed['horizons'][2]['score_pass_but_clock_invalid_rows'],1)

    def test_future_values_never_delete_current_keys(self):
        expected=coverage_decisions(self.pred,coverage_rules()[1],self.cfg)
        changed={h:p.assign(execution_price=np.nan,recovery_price=0.,target_return=-999.,label_available=False) for h,p in self.pred.items()}
        pd.testing.assert_frame_equal(coverage_decisions(changed,coverage_rules()[1],self.cfg),expected)

    def test_model_coverage_alias_and_clock_identity(self):
        for name,value in [('return_score',.25),('feature_min_coverage',.8),('forecast_loss','huber'),
                           ('model_version','inventory_robust_t_v17'),('model_profile','huber_depth4')]:
            pred=copy.deepcopy(self.pred);pred[20][name]=value
            with self.assertRaises(AssertionError):coverage_decisions(pred,coverage_rules()[1],self.cfg)
        pred=copy.deepcopy(self.pred);pred[20]['forecast_alpha']=.01
        with self.assertRaises(AssertionError):coverage_decisions(pred,coverage_rules()[1],self.cfg)
        pred=copy.deepcopy(self.pred);pred[60]=pred[60].iloc[:1]
        with self.assertRaises(AssertionError):coverage_decisions(pred,coverage_rules()[1],self.cfg)
        pred={h:p.assign(datetime=pd.Timestamp('2026-03-02 09:35:01')) for h,p in self.pred.items()}
        with self.assertRaises(AssertionError):coverage_decisions(pred,coverage_rules()[1],self.cfg)

    def test_cost_current_volume_and_reverse_lock_gate(self):
        for changes in ({'signal_volume':0.},{'prediction':.001,'raw_prediction':.001,'return_score':.001},
            {'signal_price':9.5,'signal_high':9.5,'signal_low':9.5,'prediction':.01,'raw_prediction':.01,'return_score':.01}):
            pred={h:p.assign(**changes) for h,p in self.pred.items()}
            self.assertTrue(coverage_decisions(pred,coverage_rules()[1],self.cfg).empty)

    def test_complete_core_left_join_preserves_missing_label_and_dtype(self):
        core=self.pred[20][KEYS].copy();core['f_x']=np.array([.1,np.nan],dtype='float32')
        core['feature_source_start']=pd.to_datetime(['2026-01-01',None])
        labels=core.iloc[:1][KEYS].copy();labels['horizon_minutes']=20
        labels['execution_time']=labels.datetime+pd.Timedelta(minutes=1)
        labels['recovery_time']=labels.datetime+pd.Timedelta(minutes=21)
        labels['label_start'],labels['label_end']=labels.execution_time,labels.recovery_time
        labels['execution_price'],labels['recovery_price'],labels['target_return']=10.,10.1,10.1/10.-1
        result=attach_short_reference(core,labels,20)
        pd.testing.assert_frame_equal(result[list(core)],core)
        self.assertEqual(result.label_available.tolist(),[True,False])
        with self.assertRaises(AssertionError):attach_short_reference(core.assign(trade_week_complete=True),labels,20)
        with self.assertRaises(AssertionError):attach_short_reference(core,pd.concat([labels,labels]),20)

    def test_full_core_purge_keeps_original_index_for_selection(self):
        frame=pd.DataFrame({'datetime':pd.to_datetime(['2024-01-08 09:35','2024-03-04 09:35','2024-03-05 09:35']),
            'symbol':'SH.600001','feature_source_start':pd.to_datetime(['2023-12-11','2024-02-05','2024-02-05']),
            'horizon_minutes':20,'target_return':[.01,.02,np.nan],'label_available':[True,True,False]})
        frame['label_start']=frame.datetime+pd.Timedelta(minutes=1);frame['label_end']=frame.datetime+pd.Timedelta(minutes=21)
        cfg=dict(strict_feature_provenance=True,exclude_flat_signal_bars=False,embargo_days=7)
        kept,audit,meta=independent_training_split(frame,dict(core_years=[2024,2025],test_year=2023,inner_year=2026),cfg)
        self.assertEqual(kept.index.tolist(),[1]);self.assertEqual(meta['input_rows'],3)
        self.assertEqual(audit.set_index('reason').loc['label_unavailable','excluded_rows'],1)

    def test_75pct_feature_enters_new_selector_without_filling_nan(self):
        rng=np.random.default_rng(42);size=400
        frame=pd.DataFrame({'f_old':rng.normal(size=size).astype('float32'),
            'f_return_15m':rng.normal(size=size).astype('float32'),'target_return':rng.normal(size=size)})
        frame.loc[:99,'f_return_15m']=np.nan;before=frame.copy(deep=True)
        new,audit=coverage_feature_selection(frame,self.selector,.7)
        old,old_audit=coverage_feature_selection(frame,self.selector,.8)
        self.assertIn('f_return_15m',new);self.assertNotIn('f_return_15m',old)
        self.assertEqual(audit.set_index('feature').loc['f_return_15m','coverage'],.75)
        self.assertEqual(old_audit.set_index('feature').loc['f_return_15m','reason'],'coverage')
        pd.testing.assert_frame_equal(before,frame)

    def test_new_feature_can_replace_old_correlated_feature(self):
        frame=pd.DataFrame({'f_old':np.arange(100,dtype='float32'),'f_new':np.arange(100,dtype='float32'),
            'target_return':np.arange(100,dtype=float)})
        frame.loc[:24,'f_new']=np.nan
        new,_=coverage_feature_selection(frame,self.selector,.7)
        old,_=coverage_feature_selection(frame,self.selector,.8)
        self.assertEqual(new,['f_new']);self.assertEqual(old,['f_old'])

    def test_strict_correlation_greater_than_point95(self):
        x=np.arange(16,dtype=float);equal=x.copy();higher=x.copy()
        equal[[0,4]]=equal[[4,0]];equal[[6,7]]=equal[[7,6]]
        higher[[0,4]]=higher[[4,0]]
        frame=pd.DataFrame({'f_a':x,'f_b_equal95':equal,'f_z_higher95':higher,'target_return':x})
        selected,audit=coverage_feature_selection(frame,self.selector)
        self.assertEqual(selected,['f_a','f_b_equal95'])
        self.assertEqual(audit.set_index('feature').loc['f_z_higher95','reason'],'correlation_limit')

    def test_full_coverage_not_sample_fraction_and_fixed_sample_seed(self):
        rng=np.random.default_rng(19);n=20001
        frame=pd.DataFrame({'f_a':rng.normal(size=n).astype('float32'),'f_b':rng.normal(size=n).astype('float32'),
            'target_return':rng.normal(size=n)},index=np.arange(n)*2+7)
        frame.loc[frame.index[:5000],'f_b']=np.nan
        first,audit=coverage_feature_selection(frame,self.selector)
        again,repeat=coverage_feature_selection(frame,self.selector)
        self.assertEqual(first,again);pd.testing.assert_frame_equal(audit,repeat)
        self.assertEqual(audit.set_index('feature').loc['f_b','coverage'],15001/20001)
        indices=frame.sample(n=20000,random_state=42).index
        expected=frame.loc[indices,'f_a'].rank(pct=True).corr(frame.loc[indices,'target_return'].rank(pct=True))
        self.assertEqual(audit.set_index('feature').loc['f_a','target_rank_relevance'],abs(expected))
        with self.assertRaises(AssertionError):coverage_feature_selection(frame,{**self.selector,'random_state':43})

    def test_coverage_constant_inf_and_candidate_filter_reasons(self):
        frame=pd.DataFrame({'f_nan':[np.nan]*20,'f_inf':[np.inf]*20,'f_constant':[1.]*20,
            'f_history_days':np.arange(20),'f_valid':np.arange(20),'target_return':np.arange(20)})
        selected,audit=coverage_feature_selection(frame,self.selector)
        self.assertEqual(selected,['f_valid']);self.assertNotIn('f_history_days',audit.feature.tolist())
        self.assertEqual(audit.set_index('feature').loc['f_constant','reason'],'constant')
        self.assertEqual(audit.set_index('feature').loc['f_inf','coverage'],0.)

    def test_feature_limit_remains_60_not_guaranteed_33(self):
        rng=np.random.default_rng(7)
        frame=pd.DataFrame(rng.normal(size=(300,65)),columns=[f'f_{i:02d}' for i in range(65)])
        frame['target_return']=rng.normal(size=300)
        selected,audit=coverage_feature_selection(frame,self.selector)
        self.assertEqual(len(selected),60)
        self.assertEqual(int(audit.reason.eq('feature_limit').sum()),5)

    def native_fixture(self):
        class Native:
            def __init__(self):self.params={'objective':'regression','alpha':.9};self.names=['f_x'];self.text='real-L2-trees'
            def feature_name(self):return self.names
            def model_to_string(self):return self.text
        class Model:
            def __init__(self):self.booster_=Native();self.parameters={'objective':'regression','num_leaves':15}
            def get_params(self):return self.parameters
        return Native(),Model(),{'objective':'regression','num_leaves':15}

    def test_real_native_l2_not_metadata_and_no_huber_parameter(self):
        native,model,parameters=self.native_fixture()
        validate_native_parameters(native,model,parameters,['f_x'])
        native.params['objective']='huber'
        with self.assertRaises(AssertionError):validate_native_parameters(native,model,parameters,['f_x'])
        native.params['objective']='regression'
        with self.assertRaises(AssertionError):validate_native_parameters(native,model,{**parameters,'alpha':.01},['f_x'])

    def test_native_all_trees_and_feature_sequence(self):
        native,model,parameters=self.native_fixture()
        for field,value in [('text','tampered-tree'),('names',['f_y'])]:
            other=copy.deepcopy(native);setattr(other,field,value)
            with self.assertRaises(AssertionError):validate_native_parameters(other,model,parameters,['f_x'])

    def source_fixture(self,root):
        old,new=root/'old',root/'new';old.mkdir();new.mkdir()
        meta=dict(research_version='inventory_learning_t_v14',profile='l2_depth4',horizon_minutes=20,
            train_years=[2023,2024],excluded_years=[2025,2026],features=['f_x'],fitted_rows=2,training_rows=2,
            full_training_rows_sha256='a',training_matrix_sha256='b',training_target_sha256='c',
            model_parameters={'objective':'regression','n_jobs':1,'num_leaves':15})
        current={**meta,'research_version':VERSION,'features':['f_x','f_new'],'source_features':['f_x'],
            'source_training_matrix_sha256':'b','training_matrix_sha256':'different-new-matrix',
            'feature_min_coverage':.7,'source_feature_min_coverage':.8,
            'model_parameters':{'objective':'regression','n_jobs':2,'num_leaves':15}}
        (old/'model_metadata.json').write_text(json.dumps(meta))
        frame=pd.DataFrame({'datetime':pd.to_datetime(['2024-01-08 09:35','2024-01-08 09:45']),
            'symbol':'SH.600000','feature_source_start':pd.to_datetime(['2024-01-01']*2),
            'cohort_target_year':2024,'baseline_opening_shares':1500,'f_x':np.array([.1,.2],dtype='float32')})
        frame.to_pickle(old/'training_inputs.pkl');frame.to_pickle(new/'source_training_inputs.pkl')
        frame.assign(f_new=np.array([np.nan,.3],dtype='float32')).to_pickle(new/'training_inputs.pkl')
        for newname,oldname in CONTROL_PAIRS[1:]:
            for path in [new/newname,old/oldname]:pd.DataFrame({'x':[1,2]}).to_csv(path,index=False)
        class ReadOnly:
            def record(self,path):return Path(path)
            def csv(self,path):return pd.read_csv(path,float_precision='round_trip')
            def json(self,path):return json.loads(Path(path).read_text())
        auditor=ReadOnly();auditor.source_audits={'learning':(None,{'inputs_sha256':{str(p.resolve()):digest(p) for p in old.iterdir()}})}
        return auditor,new,old,current

    def test_new_full_matrix_changes_allowed_but_source_control_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            auditor,new,old,meta=self.source_fixture(Path(temp))
            result=compare_source_training(auditor,new,old,meta)
            self.assertEqual(result['added_features'],['f_new'])
            self.assertEqual(result['common_features'],['f_x'])
            self.assertEqual(len(result['reconstructed_source_selection_labels_and_split_exact']),4)
            wrong=pd.read_pickle(new/'source_training_inputs.pkl');wrong.loc[0,'f_x']=999.;wrong.to_pickle(new/'source_training_inputs.pkl')
            with self.assertRaises(AssertionError):compare_source_training(auditor,new,old,meta)

    def test_current_common_dtype_and_old_anchor_tampering_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            auditor,new,old,meta=self.source_fixture(Path(temp))
            frame=pd.read_pickle(new/'training_inputs.pkl');frame['f_x']=frame.f_x.astype('float64');frame.to_pickle(new/'training_inputs.pkl')
            with self.assertRaises(AssertionError):compare_source_training(auditor,new,old,meta)
            frame['f_x']=frame.f_x.astype('float32');frame.to_pickle(new/'training_inputs.pkl')
            auditor.source_audits['learning'][1]['inputs_sha256'][str((old/'training_inputs.pkl').resolve())]='fake'
            with self.assertRaises(AssertionError):compare_source_training(auditor,new,old,meta)

    def test_actual_matrix_indices_cannot_be_reset_silently(self):
        for name in ('training_inputs.pkl','source_training_inputs.pkl'):
            with tempfile.TemporaryDirectory() as temp:
                auditor,new,old,meta=self.source_fixture(Path(temp))
                frame=pd.read_pickle(new/name);frame.index=[5,9];frame.to_pickle(new/name)
                with self.assertRaises(AssertionError):compare_source_training(auditor,new,old,meta)

    def test_disabled_outer_requires_real_artifact_absence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'year_2023').mkdir()
            assert_no_outer_products(root,[2023])
            (root/'year_2023/test_predictions').mkdir()
            with self.assertRaises(AssertionError):assert_no_outer_products(root,[2023])
            (root/'year_2023/test_predictions').rmdir()
            (root/'year_2023/test_equity_curve.csv').write_text('equity\n1\n')
            with self.assertRaises(AssertionError):assert_no_outer_products(root,[2023])

    def test_source_labels_and_model_parameters_still_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            auditor,new,old,meta=self.source_fixture(Path(temp))
            changed=copy.deepcopy(meta);changed['model_parameters']['num_leaves']=31
            with self.assertRaises(AssertionError):compare_source_training(auditor,new,old,changed)
            pd.DataFrame({'x':[1,999]}).to_csv(new/'training_labels.csv',index=False)
            with self.assertRaises(AssertionError):compare_source_training(auditor,new,old,meta)

    def test_comparison_actual_change_and_only_threshold_flag(self):
        current=pd.DataFrame(coverage_rules());original=current.copy()
        for name in COMPARISON_FIELDS:current[name]=1.;original[name]=1.
        current['incremental_net_pnl']=2.;saved=current[['candidate_index']].copy()
        for name in COMPARISON_FIELDS:
            saved[name+'_coverage']=current[name];saved[name+'_source']=original[name];saved[name+'_change']=current[name]-original[name]
        saved['current_profile'],saved['source_profile']=current.profile,original.profile
        saved['same_forecast_target'],saved['same_forecasts'],saved['same_execution_guard']=True,False,True
        saved['same_learning_objective'],saved['source_feature_min_coverage'],saved['current_feature_min_coverage']=True,.8,.7
        verify_comparison(current,original,saved)
        for field,value in [('incremental_net_pnl_change',999.),('same_learning_objective',False),('current_feature_min_coverage',.6)]:
            altered=saved.copy();altered.loc[1,field]=value
            with self.assertRaises(AssertionError):verify_comparison(current,original,altered)

    def test_pending_marked_pnl_is_not_cost_target(self):
        self.assertEqual(complete_pair_cost_rate({'all_reference_acquisition_cost':10000.,'realized_complete_pair_pnl':20.,'incremental_net_pnl':-100.}),.002)
        cfg={'report_scope':'smoke','minimum_completed_pair_lots':None,'minimum_active_stock_days':None}
        validate_goal_status({'goal_achieved':False,'frequency_target_status':'not_quantified'},cfg,True)
        with self.assertRaises(AssertionError):validate_goal_status({'goal_achieved':True,'frequency_target_status':'not_quantified'},cfg,True)

    def test_account_adds_all_t_leg_slippage_excludes_base(self):
        auditor=CoverageAuditor.__new__(CoverageAuditor)
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
    if not result.wasSuccessful():raise AssertionError('v18独立审计反例失败')
    return {'passed':True,'tests':result.testsRun}


if __name__=='__main__':run_self_tests()
