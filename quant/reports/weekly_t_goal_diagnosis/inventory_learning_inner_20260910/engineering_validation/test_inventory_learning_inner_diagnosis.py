import importlib.util,unittest
from pathlib import Path
import pandas as pd,numpy as np
path=Path('reports/weekly_t_goal_diagnosis/inventory_learning_inner_20260910/diagnose.py').resolve()
spec=importlib.util.spec_from_file_location('diag',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
CFG=dict(min_commission=0,commission_rate=.0008,transfer_rate=.00001,slippage_rate=.0005,stamp_rate_before_20230828=.001,stamp_rate_from_20230828=.0005)
class Tests(unittest.TestCase):
 def setup_diag(self):
  a=object.__new__(d.Diagnosis);a.cfg=CFG;a.inner=2026
  a.inventory=pd.Series([200],index=pd.MultiIndex.from_tuples([(pd.Timestamp('2026-01-06'),'a')]))
  a.closes=lambda s:pd.Series([10.,10.1],index=pd.to_datetime(['2026-01-06 10:01','2026-01-06 10:21']))
  return a
 def test_cost_date(self):
  b,s=d.edges(pd.Series([0.,0.]),pd.Series(pd.to_datetime(['2023-08-25','2023-08-28'])),CFG)
  self.assertAlmostEqual(b.iloc[1]*1e4,-31.1975);self.assertAlmostEqual((b.iloc[1]-b.iloc[0])*1e4,4.9975)
 def test_label_missing_keeps_all_rows(self):
  a=self.setup_diag();p=pd.DataFrame({'datetime':pd.to_datetime(['2026-01-06 10:00','2026-01-06 14:00']),'symbol':['a','a'],'prediction':[.01,.01]})
  x=a.annotate(p,60);self.assertEqual(len(x),2);self.assertFalse(x.endpoint_available.any());self.assertFalse(x.endpoint_clock_valid.iloc[1]);self.assertEqual(d.summary(x)['missing_endpoint_rows'],2)
 def test_direction_not_future_return(self):
  a=self.setup_diag();p=pd.DataFrame({'datetime':pd.to_datetime(['2026-01-06 10:00']),'symbol':['a'],'prediction':[-.01]})
  x=a.annotate(p,20);self.assertEqual(x.direction.iloc[0],'sell_first');self.assertAlmostEqual(x.fixed_gross_bp.iloc[0],-100);self.assertLess(x.fixed_net_bp.iloc[0],-100)
 def test_input_unchanged_and_inventory(self):
  a=self.setup_diag();p=pd.DataFrame({'datetime':pd.to_datetime(['2026-01-06 10:00']),'symbol':['a'],'prediction':[.01]});before=p.copy(deep=True)
  x=a.annotate(p,20);pd.testing.assert_frame_equal(p,before);self.assertEqual(x.opening_old_shares.iloc[0],200);self.assertAlmostEqual(d.summary(x)['mean_net_optimism_bp'],0)
 def test_future_not_change_direction(self):
  a=self.setup_diag();p=pd.DataFrame({'datetime':pd.to_datetime(['2026-01-06 10:00']),'symbol':['a'],'prediction':[.01]})
  x=a.annotate(p,20);a.closes=lambda s:pd.Series([10.,9.],index=pd.to_datetime(['2026-01-06 10:01','2026-01-06 10:21']))
  y=a.annotate(p,20);self.assertEqual(x.direction.iloc[0],y.direction.iloc[0]);self.assertEqual(x.predicted_net_bp.iloc[0],y.predicted_net_bp.iloc[0]);self.assertNotEqual(x.fixed_net_bp.iloc[0],y.fixed_net_bp.iloc[0])
if __name__=='__main__':unittest.main()
