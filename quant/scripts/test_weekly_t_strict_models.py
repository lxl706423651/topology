"""严格验证年度排除包含滚动历史来源，而不仅是标签年份。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.test_weekly_t_models import create_weekly_frame, create_intraday_frame
from quant_backtest import weekly_t_models as models


class StrictModelTests(unittest.TestCase):
    """跨边界来源剔除、关闭未知来源及拟合不受被排除量价变化影响。"""

    def setUp(self):
        """小模型使用确定随机种子，文件均写临时目录。"""
        self.cfg = {'workers': 1, 'random_state': 42, 'embargo_days': 7,
                    'min_training_rows': 2, 'strict_feature_provenance': True,
                    'excluded_model_features': ['f_history_days'],
                    'weekly_model': {'n_estimators': 10, 'min_child_samples': 2},
                    'weekly_return_weight': .7, 'weekly_amplitude_weight': .3}

    def test_actual_lookback_rejects_prior_year_and_long_suspension(self):
        """信号和标签虽同年，最早量价落在排除年仍必须剔除。"""
        frame = create_weekly_frame()
        frame['feature_source_start'] = frame.feature_week - pd.Timedelta(days=28)
        rows = frame.index[frame.signal_datetime.dt.year.eq(2024)]
        frame.loc[rows[:2], 'feature_source_start'] = pd.to_datetime(['2023-12-18', '2023-09-04'])
        selected, meta, _ = models.select_training_rows(frame, [2024], self.cfg, 'weekly', [2023])
        self.assertFalse(selected.index.isin(rows[:2]).any())
        self.assertTrue(selected.feature_source_start.ge(pd.Timestamp('2024-01-08')).all())
        self.assertTrue(meta['strict_feature_provenance'])

    def test_market_previous_close_origin_can_exclude_intraday_row(self):
        """同刻市场前收来源跨排除年时，当日标签合法也不能训练该行。"""
        frame = create_intraday_frame()
        frame['feature_source_start'] = frame.date - pd.Timedelta(days=1)
        rows = frame.index[frame.datetime.dt.year.eq(2024)]
        frame.loc[rows[0], 'feature_source_start'] = pd.Timestamp('2023-11-03')
        selected, _, _ = models.select_training_rows(frame, [2024], self.cfg, 'intraday', [2023])
        self.assertNotIn(rows[0], selected.index)
        self.assertEqual(len(selected), len(rows)-1)

    def test_missing_or_future_provenance_cannot_train(self):
        """缺少来源列时报错；来源无效或晚于信号的行不得进入模型。"""
        frame = create_weekly_frame()
        with self.assertRaisesRegex(ValueError, 'feature_source_start'):
            models.select_training_rows(frame, [2024], self.cfg, 'weekly', [2023])
        frame['feature_source_start'] = frame.feature_week
        rows = frame.index[frame.signal_datetime.dt.year.eq(2024)]
        frame.loc[rows[0], 'feature_source_start'] = pd.NaT
        frame.loc[rows[1], 'feature_source_start'] = frame.loc[rows[1], 'label_end']
        selected, _, _ = models.select_training_rows(frame, [2024], self.cfg, 'weekly', [2023])
        self.assertFalse(selected.index.isin(rows[:2]).any())

    def test_excluded_raw_dependency_changes_do_not_change_fitted_model(self):
        """重叠行与累计历史计数任意改变，不影响严格隔离后的特征、样本和预测。"""
        frame = create_weekly_frame()
        frame['feature_source_start'] = frame.feature_week
        frame['f_history_days'] = frame.f_signal * 100
        overlapping = frame.index[frame.signal_datetime.dt.year.eq(2024)][:10]
        frame.loc[overlapping, 'feature_source_start'] = pd.Timestamp('2023-12-18')
        altered = frame.copy()
        altered.loc[overlapping, ['f_signal', 'f_second', 'target_return']] = 999.
        altered['f_history_days'] *= -1e6
        with tempfile.TemporaryDirectory() as directory:
            a = models.fit_weekly_models(frame, [2024], self.cfg, Path(directory)/'a', [2023])
            b = models.fit_weekly_models(altered, [2024], self.cfg, Path(directory)/'b', [2023])
            self.assertNotIn('f_history_days', a['features'])
            self.assertEqual(a['features'], b['features'])
            self.assertEqual(a['metadata']['training_sample_sha256'], b['metadata']['training_sample_sha256'])
            for key in ('return', 'amplitude'):
                np.testing.assert_array_equal(a['models'][key].predict(frame[a['features']]), b['models'][key].predict(frame[b['features']]))


if __name__ == '__main__':
    unittest.main()
