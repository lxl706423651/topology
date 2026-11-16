"""合成残差审计器正例及标签/供体/推断污染反例，只写临时目录，不动正式模型。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT=Path('/data/lxl/seed-emulator/topology/quant');sys.path.insert(0,str(ROOT))
from scripts.audit_weekly_t_residual import (
    reconstruct_residual_targets,reconstruct_residual_feature_selection,audit_model,
    audit_residual_decomposition,audit_saved_t_predictions,
)
from quant_backtest.weekly_t_residual import fit_residual_t_model


class ResidualAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load((ROOT/'config/weekly_t_cross_validation_v8.yaml').read_text())
        cls.cfg.update(min_training_rows=2,workers=1,t_max_train_rows=17,t_max_features=4,
                       t_model_profile='l2_depth4',t_label_horizon_minutes=20)
        cls.cfg['t_model']={'n_estimators':4,'min_child_samples':2,'reg_lambda':0.1,'verbosity':-1}
        rows=[]
        for i,date in enumerate(pd.bdate_range('2024-02-05',periods=20)):
            for j,symbol in enumerate(['A','B','C']):
                stamp=date+pd.Timedelta(hours=10)
                rows.append({'datetime':stamp,'symbol':symbol,'feature_source_start':date-pd.Timedelta(days=7),
                    'label_start':stamp+pd.Timedelta(minutes=1),'label_end':stamp+pd.Timedelta(minutes=21),
                    'target_return':.01*np.sin(i)+(j-1)*.002,'f_side':float(j),
                    'f_clock':float(i),'f_common':float(np.sin(i)),'f_history_days':30,
                    'label_available':True,'trade_week_complete':True,'signal_bar_tradable':True})
        # 两个额外同刻行：一个自身来源触碰排除年，其target不得污染原三股均值。
        bad=dict(rows[0]);bad.update(symbol='Z',feature_source_start=pd.Timestamp('2023-12-20'),target_return=50.)
        singleton=dict(rows[0]);singleton.update(datetime=pd.Timestamp('2024-04-03 10:00'),symbol='S',
            feature_source_start=pd.Timestamp('2024-03-01'),label_start=pd.Timestamp('2024-04-03 10:01'),
            label_end=pd.Timestamp('2024-04-03 10:21'),target_return=.25)
        cls.frame=pd.DataFrame(rows+[bad,singleton])
        cls.training,cls.decomp=reconstruct_residual_targets(cls.frame,[2024],[2023,2025],cls.cfg)
        cls.features=reconstruct_residual_feature_selection(cls.training,cls.cfg)

    def test_raw_purge_precedes_donor_mean(self):
        first=self.decomp.loc[self.decomp.datetime.eq(pd.Timestamp('2024-02-05 10:00'))]
        self.assertEqual(first.cohort_size.tolist(),[3,3,3])
        np.testing.assert_allclose(first.cohort_mean_return,0,atol=1e-16)
        self.assertNotIn('Z',self.decomp.symbol.tolist())

    def test_singleton_is_audited_but_not_trained(self):
        single=self.decomp.loc[self.decomp.symbol.eq('S')].iloc[0]
        self.assertEqual(single.cohort_size,1);self.assertFalse(single.kept_for_training)
        self.assertNotIn('S',self.training.symbol.tolist())

    def test_excluded_target_cannot_change_donors(self):
        changed=self.frame.copy();changed.loc[changed.symbol.eq('Z'),'target_return']=-100000.
        other,donors=reconstruct_residual_targets(changed,[2024],[2023,2025],self.cfg)
        pd.testing.assert_frame_equal(other,self.training);pd.testing.assert_frame_equal(donors,self.decomp)

    def test_mixed_horizon_donors_rejected(self):
        changed=self.frame.copy();changed.loc[0,'label_end']+=pd.Timedelta(minutes=20)
        with self.assertRaises(AssertionError):reconstruct_residual_targets(changed,[2024],[2023,2025],self.cfg)

    def test_residual_features_differ_from_raw_common_target(self):
        self.assertEqual(self.features[0][0],'f_side')

    def test_native_hash_and_target_decomposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model'
            fit_residual_t_model(self.frame,[2024],self.cfg,path,[2023,2025])
            result=audit_model(path,self.training,self.cfg,[2024],[2023,2025],20,'l2_depth4',self.decomp,self.features)
            self.assertTrue(result['passed']);self.assertEqual(result['fitted_rows'],17)
            self.assertNotEqual(result['training_sample_sha256'],result['raw_training_sample_sha256'])

    def test_donor_csv_and_metadata_corruption_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model';fit_residual_t_model(self.frame,[2024],self.cfg,path,[2023,2025])
            donor=path/'target_decomposition.csv';original=donor.read_bytes()
            bad=pd.read_csv(donor);bad.loc[0,'cohort_mean_return']+=.01;bad.to_csv(donor,index=False)
            with self.assertRaises(AssertionError):audit_residual_decomposition(path,self.training,self.decomp,self.cfg)
            donor.write_bytes(original)
            metadata=path/'model_metadata.json';bad=json.loads(metadata.read_text());bad['donor_source_sha256']='0'*64
            metadata.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):audit_residual_decomposition(path,self.training,self.decomp,self.cfg)

    def test_prediction_future_market_addback_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model';bundle=fit_residual_t_model(self.frame,[2024],self.cfg,path,[2023,2025])
            current=self.frame.iloc[[0]].copy();current['target_return']=np.nan;current['cohort_size']=1
            prediction=bundle['models']['return'].predict(current[bundle['features']],num_threads=1)
            stored=current[['datetime','symbol']].copy();stored['prediction']=prediction
            stored['residual_prediction']=prediction;stored['market_prediction']=0.
            output=Path(temporary)/'prediction.csv';stored.to_csv(output,index=False)
            self.assertTrue(audit_saved_t_predictions(output,path,current,self.cfg)['passed'])
            stored['prediction']+=.1;stored['market_prediction']=.1;stored.to_csv(output,index=False)
            with self.assertRaises(AssertionError):audit_saved_t_predictions(output,path,current,self.cfg)


if __name__=='__main__':unittest.main(verbosity=2)
