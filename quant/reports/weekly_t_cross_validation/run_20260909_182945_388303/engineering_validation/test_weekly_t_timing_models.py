"""验证 v11 原始收益/扣费三分类的隔离、成本标签、类序及双模型持久化。

从 quant 根目录运行；只用四年合成特征训练小 LightGBM，在临时目录写模型。
不读取真实外层、不更改行情/旧模型；独立手算目标和哈希，覆盖财务语义反例。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import joblib
import lightgbm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.weekly_t_models import select_training_rows
from quant_backtest.weekly_t_short_horizon import fit_feature_support, planned_recovery_times
from quant_backtest.weekly_t_timing_models import (
    CLASS_MAPPING, TARGET_COLUMNS, TARGET_MODE, calculate_timing_frame_sha256,
    fit_timing_model, predict_timing_model, validate_timing_config,
)


def timing_config():
    """预声明三类俱全的小模型，费用严格保持真实协议量级。"""
    return {"workers": 1, "random_state": 7, "embargo_days": 7, "min_training_rows": 12,
            "strict_feature_provenance": True, "excluded_model_features": ["f_history_days"],
            "exclude_flat_signal_bars": False, "t_target_mode": TARGET_MODE,
            "t_model_profile": "l2_depth4", "t_label_horizon_minutes": 20,
            "t_max_train_rows": 0, "t_max_features": 10,
            "feature_selection_sample_rows": 1000, "feature_min_coverage": .8,
            "feature_max_correlation": 1., "commission_rate": .0008, "transfer_rate": .00001,
            "slippage_rate": .0005, "stamp_rate_before_20230828": .001,
            "stamp_rate_from_20230828": .0005, "min_commission": 0.,
            "t_support_features": ["f_signal", "f_clock"], "t_support_quantiles": [.001, .999],
            "t_model": {"n_estimators": 16, "min_child_samples": 3, "reg_lambda": .1, "reg_alpha": 0.}}


def timing_frame():
    """生成含共同正收益偏移和三个成本类别的四年多股数据，标签是未来20分钟。"""
    rows = []
    returns = [-.015, -.01, -.006, -.002, -.0005, 0., .001, .002, .005, .009, .02, .03]
    for year in [2023, 2024, 2025, 2026]:
        for date in [f"{year}-03-12", f"{year}-06-13", f"{year}-09-14"]:
            for clock in ["09:35", "10:30", "13:05", "14:00"]:
                for stock, target in enumerate(returns):
                    timestamp = pd.Timestamp(f"{date} {clock}")
                    rows.append({"datetime": timestamp, "symbol": f"TEST{stock:02d}",
                        "feature_source_start": timestamp.normalize()-pd.Timedelta(days=28),
                        "target_return": target, "f_signal": stock/10., "f_clock": timestamp.hour+timestamp.minute/60.,
                        "f_calendar": timestamp.month, "f_market": .002+timestamp.month*.0001,
                        "f_history_days": stock*100., "execution_price": 10.+stock,
                        "signal_bar_tradable": True, "horizon_minutes": 20})
    frame = pd.DataFrame(rows)
    frame["execution_time"] = frame.datetime+pd.Timedelta(minutes=1)
    frame["recovery_time"] = planned_recovery_times(frame.datetime, 20)
    frame["label_start"], frame["label_end"] = frame.execution_time, frame.recovery_time
    frame["recovery_price"] = frame.execution_price*(1+frame.target_return)
    frame["label_available"] = True
    return frame


def frame_sha(frame):
    """独立于生产哈希函数，以保存列顺序核实际拟合内容。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


class TimingModelTests(unittest.TestCase):
    """共享一次实际双模型拟合，其余只在有明确边界变化时另行拟合。"""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="weekly_t_timing_")
        cls.root = Path(cls.temporary.name)
        cls.frame, cls.cfg = timing_frame(), timing_config()
        cls.original = cls.frame.copy(deep=True)
        cls.bundle = fit_timing_model(cls.frame, [2024, 2025], cls.cfg, cls.root/"base", [2023, 2026])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def saved_targets(self, folder="base"):
        return pd.read_csv(self.root/folder/"training_targets.csv", float_precision="round_trip",
                           parse_dates=["datetime", "feature_source_start", "label_start", "label_end"])

    def test_core_source_label_and_embargo_boundaries_precede_target_fit(self):
        """被排除标签和四周来源不可通过同年signal或极端值进入拟合。"""
        frame = self.frame.copy(deep=True)
        core = frame.index[frame.datetime.dt.year.isin([2024, 2025])]
        frame.loc[core[0], "feature_source_start"] = pd.Timestamp("2023-12-01")
        frame.loc[core[1], "label_end"] = pd.Timestamp("2026-01-02 10:00")
        frame.loc[core[2], "feature_source_start"] = pd.Timestamp("2024-01-07")
        frame.loc[core[:3], "target_return"] = 1e8
        # 前一年排除区间的右端是01-08，等于右端的来源应保留。
        frame.loc[core[3], "feature_source_start"] = pd.Timestamp("2024-01-08")
        before = frame.copy(deep=True)
        bundle = fit_timing_model(frame, [2024, 2025], self.cfg, self.root/"purge", [2023, 2026])
        saved = self.saved_targets("purge")
        self.assertEqual(len(saved), len(core)-3)
        self.assertTrue(saved.datetime.dt.year.isin([2024, 2025]).all())
        for excluded in [2023, 2026]:
            left = pd.Timestamp(excluded, 1, 1)-pd.Timedelta(days=7)
            right = pd.Timestamp(excluded+1, 1, 1)+pd.Timedelta(days=7)
            self.assertFalse(((saved.feature_source_start < right) & (saved.label_end >= left)).any())
        self.assertEqual(bundle['metadata']['excluded_years'], [2023, 2026])
        self.assertLess(saved.raw_target_return.max(), 1.)
        pd.testing.assert_frame_equal(frame, before)

    def test_profit_class_requires_cost_coverage_and_date_stamp_math(self):
        """正涨幅仍可属于两向都亏；2023印花税按实际标签日期切换。"""
        bundle = fit_timing_model(self.frame, [2023, 2024], self.cfg, self.root/"fee_dates", [2025, 2026])
        target = self.saved_targets("fee_dates")
        stamp = np.where(target.datetime < pd.Timestamp("2023-08-28"), .001, .0005)
        ratio = 1+target.raw_target_return
        buy = ratio*.9995*(1-.00081-stamp)-1.0005*1.00081
        sell = .9995*(1-.00081-stamp)-ratio*1.0005*1.00081
        np.testing.assert_allclose(target.buy_net_return, buy, atol=0, rtol=0)
        np.testing.assert_allclose(target.sell_net_return, sell, atol=0, rtol=0)
        expected = np.where(buy > 0, 1, np.where(sell > 0, 2, 0))
        np.testing.assert_array_equal(target.profit_class, expected)
        self.assertTrue(target.loc[target.raw_target_return.eq(.001), 'profit_class'].eq(0).all())
        self.assertTrue(target.loc[target.raw_target_return.eq(.009), 'profit_class'].eq(1).all())
        self.assertTrue(target.loc[target.raw_target_return.eq(-.01), 'profit_class'].eq(2).all())
        self.assertEqual(bundle['metadata']['class_mapping'], CLASS_MAPPING)

    def test_raw_l2_is_unmodified_not_residual_and_full_targets_not_inputs(self):
        """用另一个实际L2对原始标签拟合，精确确认生产回归没有偷偷居中。"""
        training, _, _ = select_training_rows(self.frame, [2024, 2025], self.cfg, 'intraday', [2023, 2026])
        saved = self.saved_targets()
        np.testing.assert_array_equal(saved.raw_target_return, training.target_return)
        x = training[self.bundle['features']]
        reference = lightgbm.LGBMRegressor(**self.bundle['metadata']['model_parameters']).fit(x, training.target_return)
        np.testing.assert_array_equal(reference.predict(x), self.bundle['models']['return'].predict(x))
        self.assertGreater(abs(training.groupby('datetime').target_return.mean().mean()), .001)
        self.assertTrue(self.bundle['metadata']['raw_target_unchanged'])
        self.assertFalse(self.bundle['metadata']['cohort_mean_subtracted'])
        self.assertNotIn('f_history_days', self.bundle['features'])
        self.assertTrue(all(name.startswith('f_') for name in self.bundle['features']))
        self.assertFalse(set(TARGET_COLUMNS) & set(self.bundle['features']))
        pd.testing.assert_frame_equal(self.frame, self.original)

    def test_future_fields_deleted_or_corrupted_leave_current_predictions_identical(self):
        """删除未来成交价/标签、改变未来完整性不影响任何预测或当前行选择。"""
        frame = self.frame.loc[self.frame.datetime.dt.year.eq(2026)].sample(frac=1., random_state=9)
        expected = predict_timing_model(self.bundle, frame, self.cfg)
        future = ['execution_time', 'recovery_time', 'execution_price', 'recovery_price',
                  'label_start', 'label_end', 'target_return', 'label_available']
        absent = frame.drop(columns=future)
        actual = predict_timing_model(self.bundle, absent, self.cfg)
        columns = ['datetime', 'symbol', 'prediction', 'raw_prediction', 'prob_neither', 'prob_buy_profit', 'prob_sell_profit', 'horizon_minutes']
        pd.testing.assert_frame_equal(actual[columns], expected[columns])
        absent['label_available'], absent['target_return'], absent['recovery_price'] = False, -1e9, np.nan
        pd.testing.assert_frame_equal(predict_timing_model(self.bundle, absent, self.cfg)[columns], expected[columns])
        self.assertEqual(len(actual), len(frame))
        single = predict_timing_model(self.bundle, frame.iloc[:1], self.cfg)
        match = expected[expected.datetime.eq(single.datetime.iloc[0]) & expected.symbol.eq(single.symbol.iloc[0])]
        np.testing.assert_array_equal(single[columns[2:]], match[columns[2:]])

    def test_three_probabilities_and_joblib_native_exactly_reload(self):
        """两份native逐值对应joblib；三类输出顺序固定、和为1且均有限。"""
        restored = joblib.load(self.root/'base/model_bundle.joblib')
        frame = self.frame[self.frame.datetime.dt.year.eq(2026)]
        result = predict_timing_model(restored, frame, self.cfg)
        pd.testing.assert_frame_equal(result, predict_timing_model(self.bundle, frame, self.cfg))
        probs = result[['prob_neither', 'prob_buy_profit', 'prob_sell_profit']].to_numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1., atol=1e-12, rtol=0)
        self.assertTrue(np.isfinite(probs).all())
        for name, expected in [('return', result.prediction.to_numpy()), ('profit_class', probs)]:
            native = lightgbm.Booster(model_file=str(self.root/f'base/{name}_model.txt'))
            self.assertEqual(native.feature_name(), restored['features'])
            np.testing.assert_array_equal(native.predict(result[restored['features']]), expected)
            self.assertTrue(restored['metadata']['native_reload_audit'][name]['passed'])
        self.assertTrue(self.bundle['models']['profit_class'].get_params()['class_weight'] is None)
        self.assertFalse(restored['metadata']['probability_calibration'])

    def test_fitted_hash_cap_and_support_use_actual_purged_rows(self):
        """cap仅缩小双头拟合行，特征支持域和完整标签保持全部严格核心。"""
        cfg = {**self.cfg, 't_max_train_rows': 36}
        bundle = fit_timing_model(self.frame, [2024, 2025], cfg, self.root/'cap', [2023, 2026])
        training, _, _ = select_training_rows(self.frame, [2024, 2025], cfg, 'intraday', [2023, 2026])
        fitted = training.sample(n=36, random_state=7).sort_index()
        saved = self.saved_targets('cap')
        self.assertEqual(saved.columns.tolist(), TARGET_COLUMNS)
        self.assertEqual(saved.fitted.sum(), 36)
        self.assertEqual(bundle['metadata']['training_rows'], len(training))
        self.assertEqual(bundle['support'], fit_feature_support(training, cfg))
        target = saved.set_index(['datetime', 'symbol']).loc[pd.MultiIndex.from_frame(fitted[['datetime', 'symbol']])]
        target.index = fitted.index
        targets = target[['raw_target_return', 'buy_net_return', 'sell_net_return', 'profit_class']]
        x = fitted[bundle['features']].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        expected = pd.concat([fitted[['symbol', 'datetime']], x, targets], axis=1)
        self.assertEqual(frame_sha(expected), bundle['metadata']['training_sample_sha256'])
        self.assertEqual(frame_sha(fitted[['symbol', 'datetime', 'feature_source_start']]), bundle['metadata']['feature_source_sha256'])
        self.assertEqual(hashlib.sha256((self.root/'cap/training_targets.csv').read_bytes()).hexdigest(), bundle['metadata']['training_targets_file_sha256'])
        self.assertEqual(frame_sha(saved.drop(columns='fitted')), bundle['metadata']['training_target_sha256'])

    def test_excluded_year_values_cannot_change_features_support_targets_or_models(self):
        """inner/outer异常值不能改变核心输入选择、支持域或任一模型权重。"""
        changed = self.frame.copy()
        excluded = changed.datetime.dt.year.isin([2023, 2026])
        changed.loc[excluded, ['f_signal', 'f_market', 'target_return']] = 1e8
        bundle = fit_timing_model(changed, [2024, 2025], self.cfg, self.root/'heldout_changed', [2023, 2026])
        for key in ['training_target_sha256', 'training_input_sha256', 'training_sample_sha256', 'feature_source_sha256']:
            self.assertEqual(bundle['metadata'][key], self.bundle['metadata'][key])
        self.assertEqual(bundle['features'], self.bundle['features'])
        self.assertEqual(bundle['support'], self.bundle['support'])
        for name in ['return_model.txt', 'profit_class_model.txt']:
            self.assertEqual((self.root/'base'/name).read_bytes(), (self.root/'heldout_changed'/name).read_bytes())

    def test_missing_class_and_cap_losing_class_refuse_fit(self):
        """任何一类缺失都明确拒fit，不能出现概率列隐式换序或偷偷重采样。"""
        changed = self.frame[self.frame.target_return.le(.002)].copy()
        with self.assertRaisesRegex(ValueError, '全部三类'):
            fit_timing_model(changed, [2024, 2025], self.cfg, self.root/'missing', [2023, 2026])
        with self.assertRaisesRegex(ValueError, 'cap采样训练行缺少'):
            fit_timing_model(self.frame, [2024, 2025], {**self.cfg, 't_max_train_rows': 1}, self.root/'missing_cap', [2023, 2026])

    def test_invalid_protocol_horizon_and_existing_output_refused(self):
        """拒残差模式、概率重权、非法周期/成本以及覆盖已冻结模型。"""
        for changes in [{'t_target_mode': 'residual'}, {'strict_feature_provenance': False},
                        {'excluded_model_features': []}, {'t_model_profile': 'huber_depth4'},
                        {'t_max_features': 61}, {'min_commission': 5}, {'slippage_rate': np.nan},
                        {'t_label_horizon_minutes': True}, {'horizon_minutes': 40},
                        {'t_model': {'class_weight': 'balanced'}}, {'t_model': {'is_unbalance': True}},
                        {'t_model': {'scale_pos_weight': 2}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_timing_config({**self.cfg, **changes})
        changed = self.frame.copy()
        changed.loc[changed.datetime.dt.year.eq(2024), 'label_end'] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, '端点'):
            fit_timing_model(changed, [2024, 2025], self.cfg, self.root/'wrong_h', [2023, 2026])
        before = (self.root/'base/model_bundle.joblib').read_bytes()
        with self.assertRaisesRegex(ValueError, '不能覆盖'):
            fit_timing_model(self.frame, [2024, 2025], self.cfg, self.root/'base', [2023, 2026])
        self.assertEqual(before, (self.root/'base/model_bundle.joblib').read_bytes())

    def test_wrong_class_order_invalid_probabilities_and_empty_current_filter(self):
        """类序或模型损坏不能静默解释概率；当前资格空集合返回固定空schema。"""
        frame = self.frame.iloc[:2]
        for classes, probs in [(np.array([2, 1, 0]), np.ones((2, 3))/3),
                               (np.array([0, 1, 2]), np.full((2, 3), np.nan)),
                               (np.array([0, 1, 2]), np.ones((2, 3))),
                               (np.array([0, 1, 2]), np.ones((2, 2))/2)]:
            bundle = {**self.bundle, 'models': {**self.bundle['models'], 'profit_class':
                       types.SimpleNamespace(classes_=classes, predict_proba=lambda x, p=probs: p)}}
            with self.assertRaises(ValueError):
                predict_timing_model(bundle, frame, self.cfg)
        empty = frame.copy()
        empty['signal_bar_tradable'] = False
        result = predict_timing_model(self.bundle, empty, {**self.cfg, 'exclude_flat_signal_bars': True})
        self.assertTrue(result.empty)
        self.assertTrue({'prediction', 'raw_prediction', 'prob_neither', 'prob_buy_profit', 'prob_sell_profit'}.issubset(result))

    def test_prediction_rejects_changed_cost_definition_or_model_horizon(self):
        """成本后事件必须沿用训练定义，不能更换费率后仍冒称同一盈利类别。"""
        frame = self.frame.iloc[:2]
        for key, value in [('commission_rate', .0009), ('slippage_rate', .0004),
                           ('stamp_rate_from_20230828', .0004)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '成本定义不一致'):
                predict_timing_model(self.bundle, frame, {**self.cfg, key: value})
        with self.assertRaisesRegex(ValueError, '周期不一致'):
            predict_timing_model(self.bundle, frame, {**self.cfg, 't_label_horizon_minutes': 40})
        without_h = {key: value for key, value in self.cfg.items() if key != 't_label_horizon_minutes'}
        pd.testing.assert_frame_equal(predict_timing_model(self.bundle, frame, without_h),
                                      predict_timing_model(self.bundle, frame, self.cfg))


if __name__ == '__main__':
    unittest.main(verbosity=2)
