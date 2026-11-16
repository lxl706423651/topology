"""独立微型报价反例：规划稀疏口径与最终完整口径各自计价，不调用引擎。"""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import audit_dynamic_inventory_t as audit
import pandas as pd
import numpy as np

class ValuationScopes(unittest.TestCase):
    def fixture(self):
        a=audit.Auditor.__new__(audit.Auditor)
        a.cfg={};a.mark_cache={}
        day=pd.Timestamp('2025-01-06')
        times=pd.DatetimeIndex(['2025-01-06 14:00','2025-01-06 14:30','2025-01-06 14:59'])
        raw=pd.DataFrame({'open':[10.2,10.4,10.5],'high':[10.2,10.4,10.5],
            'low':[10.2,10.4,10.5],'close':[10.2,10.4,10.5],
            'volume':[1000.]*3,'amount':[10000.]*3},index=times)
        a.bars=lambda year,symbol:raw
        a.full_bars=lambda year,symbol:raw
        a.sparse_arrays={2025:({'symbols':['A']},times[:1].asi8,np.array([[1]],dtype='uint8'),
            raw.iloc[:1].to_numpy().reshape(1,1,6))}
        return a,day
    def mark(self,a,day,planning):
        marks={'A':10.};times={'A':day+pd.Timedelta(minutes=570)}
        a.update_closing_mark(day,'A',planning,marks,times)
        return marks['A'],times['A']
    def test_planning_sparse_and_final_full_quotes_have_separate_correct_values(self):
        a,day=self.fixture()
        self.assertEqual(self.mark(a,day,True),(10.2,pd.Timestamp('2025-01-06 14:00')))
        self.assertEqual(self.mark(a,day,False),(10.5,pd.Timestamp('2025-01-06 14:59')))
    def test_unfilled_manual_current_quote_can_update_shared_planning_mark(self):
        a,day=self.fixture();a.cfg={'manual_buys_csv':'synthetic.csv'}
        a.csv=lambda path:pd.DataFrame({'symbol':['A'],'datetime':['2025-01-06 14:30']})
        self.assertEqual(self.mark(a,day,True),(10.4,pd.Timestamp('2025-01-06 14:30')))
    def test_pending_share_difference_uses_full_current_market_mark(self):
        a,day=self.fixture();price,_=self.mark(a,day,False)
        self.assertEqual(1100*price-1000*price,1050.)

if __name__=='__main__':unittest.main(verbosity=2)
