"""独立v7训练域审计的时间/整手/来源/未来不干预反例；只改内存与临时文件。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
ROOT=Path('/data/lxl/seed-emulator/topology/quant');sys.path.insert(0,str(ROOT))
from scripts.audit_weekly_t_train_domain import reconstruct_known_domain,audit_training_domain

class DomainAuditTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787/resolved_config.json').read_text())
        self.frame=pd.DataFrame({'datetime':pd.to_datetime(['2024-02-06 10:00','2024-02-06 11:00','2024-02-06 14:00']),
            'symbol':['A','B','C'],'trade_week':pd.Timestamp('2024-02-05'),'feature_source_start':pd.Timestamp('2024-01-08'),
            'target_return':[.1,-.5,np.nan],'execution_price':[1.,2.,3.],
            'recovery_price':[30.,40.,np.nan],'signal_bar_tradable':[True,False,True]},index=[9,3,11])
        self.weekly=pd.DataFrame({'symbol':['A','B','C'],'trade_week':pd.Timestamp('2024-02-05'),
            'signal_datetime':pd.Timestamp('2024-02-02 15:00'),'feature_source_start':pd.Timestamp('2024-01-08'),
            'f_last_close':[23.5,24.,10.]})

    def test_two_floor_lot_formula(self):
        kept,audit,_=reconstruct_known_domain(self.frame,self.weekly,20,self.cfg)
        self.assertEqual(audit.static_base_shares.tolist(),[400,300,900])
        self.assertEqual(audit.static_t_shares.tolist(),[100,0,200])
        self.assertEqual(kept.index.tolist(),[9,11]);self.assertEqual(audit.loc[3,'drop_reason'],'static_t_below_one_lot')

    def test_lunch_deadline_and_same_day(self):
        self.weekly['f_last_close']=10.
        _,audit,_=reconstruct_known_domain(self.frame,self.weekly,40,self.cfg)
        self.assertEqual(audit.loc[3,'planned_recovery_time'],pd.Timestamp('2024-02-06 13:11'))
        self.assertEqual(audit.loc[11,'drop_reason'],'recovery_after_deadline')
        _,audit,_=reconstruct_known_domain(self.frame,self.weekly,60,self.cfg)
        self.assertTrue(pd.isna(audit.loc[11,'planned_recovery_time']))
        self.assertEqual(audit.loc[11,'drop_reason'],'no_same_day_planned_recovery')

    def test_future_values_and_fillability_not_consumed(self):
        original,audit,metadata=reconstruct_known_domain(self.frame,self.weekly,20,self.cfg)
        changed=self.frame.copy();changed[['target_return','execution_price','recovery_price']]=-9999.
        changed['signal_bar_tradable']=~changed.signal_bar_tradable
        changed['planned_recovery_time']=pd.Timestamp('2030-01-01')
        kept,other,second=reconstruct_known_domain(changed,self.weekly,20,self.cfg)
        pd.testing.assert_frame_equal(audit,other);self.assertEqual(metadata,second)
        self.assertEqual(original.index.tolist(),kept.index.tolist())

    def test_future_week_source_rejected(self):
        self.weekly.loc[0,'signal_datetime']=pd.Timestamp('2024-02-06 10:01')
        with self.assertRaises(AssertionError):reconstruct_known_domain(self.frame,self.weekly,20,self.cfg)

    def test_metadata_and_row_corruption_rejected(self):
        _,audit,metadata=reconstruct_known_domain(self.frame,self.weekly,20,self.cfg)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'domain'
            path.with_suffix('.json').write_text(json.dumps(metadata));audit.to_csv(path.with_suffix('.csv'),index=False)
            audit_training_domain(self.frame,self.weekly,20,self.cfg,path)
            wrong=dict(metadata);wrong['known_inputs_sha256']='0'*64;path.with_suffix('.json').write_text(json.dumps(wrong))
            with self.assertRaises(AssertionError):audit_training_domain(self.frame,self.weekly,20,self.cfg,path)
            path.with_suffix('.json').write_text(json.dumps(metadata));audit.loc[9,'static_t_shares']+=100
            audit.to_csv(path.with_suffix('.csv'),index=False)
            with self.assertRaises(AssertionError):audit_training_domain(self.frame,self.weekly,20,self.cfg,path)

if __name__=='__main__':unittest.main(verbosity=2)
