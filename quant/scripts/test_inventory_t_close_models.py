"""固定14:55回归模型的隔离、原始目标及原生重载反例；只拟合临时合成小模型。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import lightgbm
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_backtest.inventory_t_close_models as models
from quant_backtest.inventory_t_close_labels import extract_fixed_close_labels, attach_fixed_close_labels


def make_config():
    """保留原费用及模型profile合同，只有合成规模/线程缩小并明确新目标。"""
    cfg=yaml.safe_load((ROOT/'config/inventory_learning_t_v14.yaml').read_text())
    cfg.pop('horizons')
    cfg.update(forecast_target='same_day_1455',recovery_key='close',t_model_profile='l2_depth4',
               workers=1,min_training_rows=20,t_max_features=3,feature_selection_sample_rows=200,
               t_model={'n_estimators':8,'min_child_samples':5,'learning_rate':.1})
    return cfg


def make_frame():
    """两个核心年160个当前键；前10条触及排除年，不可参与任何目标或特征选择。"""
    days=pd.DatetimeIndex(list(pd.bdate_range('2023-03-01',periods=80))+list(pd.bdate_range('2024-03-01',periods=80)))
    x=np.sin(np.arange(len(days))*.37)
    signal=days+pd.Timedelta(hours=9,minutes=35)
    entry=10.+np.arange(len(days))*.01
    frame=pd.DataFrame({'datetime':signal,'symbol':np.where(np.arange(len(days))%2,'SZ.000001','SH.600001'),
        'cohort_target_year':days.year,'baseline_opening_shares':1000,
        'feature_source_start':days-pd.Timedelta(days=7),
        'f_signal':x.astype('float32'),'f_noise':np.cos(np.arange(len(days))*.13).astype('float32'),
        'f_only_excluded':np.where(np.arange(len(days))<10,x,0.).astype('float32'),
        'f_history_days':np.arange(len(days)),'execution_time':signal+pd.Timedelta(minutes=1),
        'recovery_time':days+pd.Timedelta(hours=14,minutes=55),
        'execution_price':entry,'recovery_price':entry*(1.+.012+.01*x),
        'target_return':99.,'trade_week_complete':False})
    frame.loc[:9,'feature_source_start']=pd.Timestamp('2022-12-23')
    labels,_=extract_fixed_close_labels(frame,make_config())
    current=frame.drop(columns=[name for name in frame if name.startswith(('execution_','recovery_','target_','label_')) or name=='trade_week_complete'])
    frame=attach_fixed_close_labels(current,labels,make_config())
    frame.index=np.arange(len(frame))*3+1000
    return frame


def frame_sha(frame):
    """独立重算实际矩阵/标签内容摘要。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()


class CloseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.cfg=make_config();cls.frame=make_frame()
        cls.original=cls.frame.copy(deep=True);cls.path=Path(cls.temp.name)/'model'
        cls.metadata=models.fit_inventory_close_model(cls.frame,[2023,2024],cls.cfg,cls.path,[2022,2025])
        cls.bundle=joblib.load(cls.path/'model_bundle.joblib')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def output(self,label):return Path(self.temp.name)/label

    def test_core_purge_before_feature_selection_and_no_input_mutation(self):
        pd.testing.assert_frame_equal(self.frame,self.original)
        self.assertEqual(self.metadata['training_rows'],150)
        self.assertEqual(self.metadata['fitted_rows'],150)
        self.assertEqual(self.metadata['train_years'],[2023,2024])
        self.assertEqual(self.metadata['excluded_years'],[2022,2025])
        self.assertNotIn('f_history_days',self.bundle['features'])
        self.assertNotIn('f_only_excluded',self.bundle['features'])
        self.assertTrue(all(name.startswith('f_') for name in self.bundle['features']))
        audit=pd.read_csv(self.path/'feature_manifest.csv')
        self.assertEqual(audit.set_index('feature').loc['f_only_excluded','reason'],'constant')

    def test_exact_raw_close_target_saved_not_residual_and_no_future_matrix(self):
        labels=pd.read_csv(self.path/'training_labels.csv',float_precision='round_trip')
        inputs=pd.read_pickle(self.path/'training_inputs.pkl')
        expected=self.frame.iloc[10:].sort_values(['datetime','symbol'])
        np.testing.assert_array_equal(labels.target_return,expected.recovery_price/expected.execution_price-1)
        self.assertGreater(labels.target_return.mean(),.01)
        self.assertTrue(labels.fitted.all())
        self.assertTrue(labels.recovery_clock.eq('14:55').all())
        self.assertEqual(set(inputs),{'datetime','symbol','feature_source_start','cohort_target_year','baseline_opening_shares',*self.bundle['features']})
        pd.testing.assert_frame_equal(inputs[self.bundle['features']].reset_index(drop=True),expected[self.bundle['features']].reset_index(drop=True))
        self.assertEqual(self.metadata['training_matrix_sha256'],frame_sha(inputs[self.bundle['features']]))
        self.assertEqual(self.metadata['training_target_sha256'],frame_sha(expected[['target_return']]))
        for name,key in [('training_inputs.pkl','training_inputs_file_sha256'),('training_labels.csv','training_labels_file_sha256'),('return_model.txt','native_model_sha256'),('model_bundle.joblib','model_bundle_sha256')]:
            self.assertEqual(hashlib.sha256((self.path/name).read_bytes()).hexdigest(),self.metadata[key])

    def test_native_and_joblib_predictions_exact_with_current_feature_dtypes(self):
        x=pd.read_pickle(self.path/'training_inputs.pkl')[self.bundle['features']]
        native=lightgbm.Booster(model_file=str(self.path/'return_model.txt'))
        restored=joblib.load(self.path/'model_bundle.joblib')
        self.assertEqual(native.feature_name(),self.bundle['features'])
        np.testing.assert_array_equal(native.predict(x),restored['model'].predict(x))
        self.assertEqual(self.metadata['model_parameters']['objective'],'regression')
        self.assertEqual(self.metadata['model_parameters']['max_depth'],4)
        self.assertTrue(json.loads((self.path/'native_reload_audit.json').read_text())['passed'])

    def test_missing_or_perturbed_future_does_not_change_prediction_keys(self):
        before=models.predict_inventory_close_model(self.bundle,self.frame,self.cfg)
        stripped=self.frame.drop(columns=[name for name in self.frame if name.startswith(('execution_','recovery_','target_','label_'))])
        after=models.predict_inventory_close_model(self.bundle,stripped,self.cfg)
        self.assertEqual(len(after),len(self.frame));self.assertTrue(after.index.equals(self.frame.index))
        pd.testing.assert_series_equal(before.prediction,after.prediction)
        bad=self.frame.copy()
        for name in ['target_return','execution_price','recovery_price','label_available']:
            bad[name]=np.nan
        bad['label_end']=pd.Timestamp('2099-01-01')
        changed=models.predict_inventory_close_model(self.bundle,bad,self.cfg)
        pd.testing.assert_series_equal(before.prediction,changed.prediction)
        self.assertTrue(changed.recovery_key.eq('close').all())
        self.assertNotIn('horizon_minutes',after)

    def test_missing_targets_and_nat_or_boundary_sources_only_remove_fit_rows(self):
        frame=self.frame.copy()
        frame.loc[frame.index[20],'target_return']=np.nan
        frame.loc[frame.index[21],'feature_source_start']=pd.NaT
        frame.loc[frame.index[22],'feature_source_start']=pd.Timestamp('2023-01-07 23:59')
        frame.loc[frame.index[23],'feature_source_start']=pd.Timestamp('2023-01-08')
        result=models.fit_inventory_close_model(frame,[2023,2024],self.cfg,self.output('missing'),[2022,2025])
        self.assertEqual(result['training_rows'],147)
        saved=pd.read_pickle(self.output('missing')/'training_inputs.pkl')
        keys=set(zip(saved.datetime,saved.symbol))
        for index in [20,21,22]:self.assertNotIn((frame.iloc[index].datetime,frame.iloc[index].symbol),keys)
        self.assertIn((frame.iloc[23].datetime,frame.iloc[23].symbol),keys)
        self.assertEqual(len(models.predict_inventory_close_model(self.bundle,frame,self.cfg)),len(frame))

    def test_label_cross_day_is_purged_and_wrong_same_day_clock_rejected(self):
        frame=self.frame.copy();index=frame.index[20]
        frame.loc[index,'label_end']=frame.loc[index,'label_end']+pd.Timedelta(days=1)
        result=models.fit_inventory_close_model(frame,[2023,2024],self.cfg,self.output('cross_day'),[2022,2025])
        self.assertEqual(result['training_rows'],149)
        split=pd.read_csv(self.output('cross_day')/'training_split_audit.csv').set_index('reason')
        self.assertEqual(split.loc['intraday_label_crosses_day','excluded_rows'],1)
        for field in ['label_end','recovery_time','execution_time']:
            bad=self.frame.copy();bad.loc[index,field]+=pd.Timedelta(minutes=1)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'身份不一致'):
                models.fit_inventory_close_model(bad,[2023,2024],self.cfg,self.output('clock_'+field),[2022,2025])

    def test_wrong_price_target_and_fixed_identity_rejected_before_fit(self):
        cases=[('target_return',.9),('execution_price',0.),('recovery_price',np.inf),
               ('recovery_key','20'),('recovery_kind','fixed_horizon'),('recovery_clock','15:00')]
        for field,value in cases:
            bad=self.frame.copy();bad.loc[bad.index[20],field]=value
            path=self.output('bad_'+field)
            with self.subTest(field=field),self.assertRaises(ValueError):
                models.fit_inventory_close_model(bad,[2023,2024],self.cfg,path,[2022,2025])
            self.assertFalse(path.exists())

    def test_core_inventory_fold_identity_and_overlap_rejected(self):
        for field,value in [('baseline_opening_shares',0),('cohort_target_year',2025)]:
            frame=self.frame.copy();frame.loc[frame.index[20],field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'核心旧股'):
                models.fit_inventory_close_model(frame,[2023,2024],self.cfg,self.output('wrong_'+field),[2022,2025])
        bad=self.frame.copy();bad.loc[bad.index[20],'cohort_target_year']=2024
        with self.assertRaisesRegex(ValueError,'年份与当前时刻'):
            models.fit_inventory_close_model(bad,[2023,2024],self.cfg,self.output('wrong_current_year'),[2022,2025])
        with self.assertRaisesRegex(ValueError,'互斥'):
            models.fit_inventory_close_model(self.frame,[2023,2024],self.cfg,self.output('overlap'),[2024,2025])

    def test_excluded_rows_never_reach_model_or_change_feature_selection(self):
        frame=self.frame.copy();frame.loc[frame.index[:10],'f_signal']=1e8
        frame.loc[frame.index[:10],'target_return']=1234.
        with patch.object(models,'LGBMRegressor',wraps=models.LGBMRegressor) as factory:
            result=models.fit_inventory_close_model(frame,[2023,2024],self.cfg,self.output('excluded_perturbation'),[2022,2025])
        self.assertEqual(factory.call_count,1)
        self.assertEqual(result['features'],self.metadata['features'])
        self.assertEqual(result['training_matrix_sha256'],self.metadata['training_matrix_sha256'])
        self.assertEqual(result['training_target_sha256'],self.metadata['training_target_sha256'])

    def test_fixed_config_no_horizon_sampling_or_loose_provenance(self):
        cases=[('forecast_target','raw_20m'),('recovery_key','20'),('horizons',[20]),('t_label_horizon_minutes',20),
               ('strict_feature_provenance',False),('excluded_model_features',[]),('embargo_days',0),
               ('exclude_flat_signal_bars',True),('t_max_train_rows',1),('t_model_profile','huber_depth4')]
        for key,value in cases:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):models.validate_close_model_config(cfg)

    def test_empty_predictions_wrong_bundle_and_duplicate_keys(self):
        empty=self.frame.iloc[:0].copy()
        result=models.predict_inventory_close_model(self.bundle,empty,self.cfg)
        self.assertTrue(result.empty);self.assertIn('raw_prediction',result)
        with self.assertRaisesRegex(ValueError,'唯一当前键'):
            models.predict_inventory_close_model(self.bundle,pd.concat([self.frame,self.frame.iloc[:1]]),self.cfg)
        bad=copy.deepcopy(self.bundle);bad['metadata']['recovery_clock']='15:00'
        with self.assertRaisesRegex(ValueError,'模型身份'):
            models.predict_inventory_close_model(bad,self.frame,self.cfg)
        cfg={**self.cfg,'t_model_profile':'l2_depth6'}
        with self.assertRaisesRegex(ValueError,'模型身份'):
            models.predict_inventory_close_model(self.bundle,self.frame,cfg)

    def test_depth6_uses_original_l2_parameters_and_no_overwrite(self):
        cfg={**self.cfg,'t_model_profile':'l2_depth6'}
        result=models.fit_inventory_close_model(self.frame,[2023,2024],cfg,self.output('depth6'),[2022,2025])
        self.assertEqual(result['model_parameters']['max_depth'],6)
        self.assertEqual(result['model_parameters']['objective'],'regression')
        self.assertEqual(result['fitted_rows'],150)
        with self.assertRaisesRegex(ValueError,'不能覆盖'):
            models.fit_inventory_close_model(self.frame,[2023,2024],self.cfg,self.path,[2022,2025])

    def test_native_reload_nan_cannot_be_marked_passed(self):
        """NaN最大误差不满足大于阈值，故须在保存通过证据前显式拒绝非有限输出。"""
        real_booster = models.lightgbm.Booster

        def restore(*args, **kwargs):
            native = real_booster(*args, **kwargs)
            if 'model_file' in kwargs:
                native.predict = lambda matrix: np.full(len(matrix), np.nan)
            return native

        directory = self.output('native_nan')
        with patch.object(models.lightgbm, 'Booster', side_effect=restore), \
                self.assertRaisesRegex(AssertionError, '全为有限数'):
            models.fit_inventory_close_model(self.frame,[2023,2024],self.cfg,directory,[2022,2025])
        self.assertFalse((directory/'native_reload_audit.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
