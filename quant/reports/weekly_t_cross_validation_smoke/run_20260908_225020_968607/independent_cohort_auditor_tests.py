"""只变更临时副本/内存的v6独立审计反例；不改正式权重、CSV、缓存或源码。"""
from pathlib import Path
import copy
import json
import re
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path('/data/lxl/seed-emulator/topology/quant')
sys.path.insert(0,str(ROOT))
from scripts.audit_weekly_t_cohort import (audit_selector,audit_parameters,audit_training_lineage,
    independent_context,file_digest,read_json)
from quant_backtest.weekly_t_features import add_session_context_features,add_weekly_execution_features
from quant_backtest.weekly_t_provenance import add_feature_provenance


class CohortAuditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=ROOT/'reports/weekly_t_cross_validation_smoke/run_20260908_225020_968607'
        cls.cfg=read_json(cls.report/'resolved_config.json')
        manifest=read_json(cls.report/'data_manifest.json');cache=Path(manifest['cache_path'])
        weekly,intraday,daily=[pd.read_pickle(cache/f'{name}.pkl') for name in ['weekly','intraday','daily']]
        weekly=add_weekly_execution_features(weekly,daily);intraday=add_session_context_features(intraday)
        weekly,intraday,_=add_feature_provenance(weekly,intraday,daily)
        m=(weekly.f_history_days.ge(cls.cfg['min_history_days'])&weekly.f_last_close.ge(cls.cfg['min_price'])
           &weekly.f_avg_daily_amount.ge(cls.cfg['min_avg_daily_amount'])&weekly.feature_week_complete.fillna(False)
           &weekly.f_prior_close_auction_amount_min.ge(cls.cfg['min_prior_close_auction_amount']))
        cls.universe=weekly.loc[m].copy()
        cls.path=cls.report/'year_2023/oof_weekly_cohort/selector_2024'

    def test_target_future_labels_cannot_filter_oof(self):
        frame=self.universe.copy();own=frame.trade_week.dt.year.eq(2024)
        frame.loc[own,['target_return','target_amplitude']]=np.nan
        frame.loc[own,['trade_week_complete','label_available']]=False
        selected,check=audit_selector(self.path,frame,2024,[2025],[2023,2024,2026],self.cfg)
        self.assertTrue(check['passed']);self.assertGreater(check['missing_future_return_labels_still_predicted'],0)
        self.assertGreater(len(selected),0)

    def test_training_feature_mutation_changes_signature(self):
        frame=self.universe.copy();feature=read_json(self.path/'model_metadata.json')['feature_names'][0]
        frame.loc[frame.trade_week.dt.year.eq(2025),feature]+=123.0
        with self.assertRaises(AssertionError):audit_selector(self.path,frame,2024,[2025],[2023,2024,2026],self.cfg)

    def test_selector_own_year_training_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model';shutil.copytree(self.path,path)
            metadata=read_json(path/'model_metadata.json');metadata['train_years']=[2024]
            (path/'model_metadata.json').write_text(json.dumps(metadata))
            with self.assertRaises(AssertionError):audit_selector(path,self.universe,2024,[2025],[2023,2024,2026],self.cfg)

    def test_native_leaf_weight_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model';shutil.copytree(self.path,path)
            native=path/'return_model.txt';text=native.read_text()
            text,count=re.subn(r'leaf_value=([-+0-9.eE]+)',lambda m:'leaf_value='+str(float(m.group(1))+1),text,count=1)
            self.assertEqual(count,1);native.write_text(text)
            with self.assertRaises(AssertionError):audit_parameters(path,self.cfg,'weekly','weekly_dual_rank')

    def context_case(self):
        minute=pd.DataFrame({'datetime':[pd.Timestamp('2024-02-06 10:00')],
            'trade_week':[pd.Timestamp('2024-02-05')],'symbol':['SH600000'],
            'feature_source_start':[pd.Timestamp('2024-01-29')],
            'target_return':[np.nan]})
        universe=pd.DataFrame({'trade_week':[pd.Timestamp('2024-02-05')],'symbol':['SH600000'],
            'signal_datetime':[pd.Timestamp('2024-02-02 15:00')],
            'feature_source_start':[pd.Timestamp('2024-01-08')]})
        p=pd.DataFrame({'trade_week':[pd.Timestamp('2024-02-05')],'symbol':['SH600000'],
            'pred_return_rank':[.8],'pred_amplitude_rank':[.4],'score':[.68],'rank':[5],
            'source_model_training_years':['[2025]'],
            'source_model_sha256':[file_digest(self.path/'model_bundle.joblib')]})
        return minute,p,universe

    def test_context_min_source_and_original_rank_kept(self):
        minute,p,universe=self.context_case()
        context=independent_context(minute,p,universe,{'top_n':100})
        self.assertEqual(context.feature_source_start.iloc[0],pd.Timestamp('2024-01-08'))
        self.assertEqual(context.f_weekly_selection_rank.iloc[0],.05)
        self.assertTrue(context.target_return.isna().all());self.assertEqual(len(context),1)

    def test_context_missing_or_future_week_source_rejected(self):
        minute,p,universe=self.context_case()
        with self.assertRaises(AssertionError):independent_context(minute,p,universe.iloc[:0],{'top_n':100})
        universe.loc[0,'signal_datetime']=pd.Timestamp('2024-02-06 11:00')
        with self.assertRaises(AssertionError):independent_context(minute,p,universe,{'top_n':100})

    def test_t_parameter_lineage_own_or_inner_year_rejected(self):
        minute,p,universe=self.context_case()
        training=independent_context(minute,p,universe,{'top_n':100})
        training['cohort_target_year']=2024
        self.assertTrue(audit_training_lineage(training,self.report/'year_2023',[2024,2025],[2023,2026])['passed'])
        for invalid in ['[2024]','[2026]']:
            altered=training.copy();altered['source_model_training_years']=invalid
            with self.assertRaises(AssertionError):audit_training_lineage(altered,self.report/'year_2023',[2024,2025],[2023,2026])


if __name__=='__main__':unittest.main(verbosity=2)
