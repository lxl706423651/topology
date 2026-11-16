"""验证残差T训练真实居中、严格供体隔离、采样溯源与零市场先验推断。

从quant根目录执行。合成数据与小LightGBM仅写临时目录，不读取真实外层结果，
不改变任何缓存；覆盖会影响财务结论的标签、供体和模型持久化边界。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import lightgbm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_models import select_training_rows
from quant_backtest.weekly_t_residual import (
    DECOMPOSITION_COLUMNS, TARGET_MODE, calculate_residual_frame_sha256,
    fit_residual_t_model, predict_residual_t, prepare_residual_training, validate_residual_config,
)
from scripts.test_weekly_t_models import create_intraday_frame


def residual_config() -> dict:
    """固定小模型参数与正式残差协议；不通过内部收益选择任何配置。"""
    return {"workers": 1, "random_state": 7, "embargo_days": 7, "min_training_rows": 12,
            "strict_feature_provenance": True, "excluded_model_features": ["f_history_days"],
            "exclude_flat_signal_bars": False, "t_target_mode": TARGET_MODE, "t_market_prior": 0.0,
            "t_residual_min_group_size": 2, "t_model_profile": "l2_depth4", "t_max_train_rows": 0,
            "t_max_features": 8, "feature_selection_sample_rows": 500, "feature_min_coverage": .8,
            "feature_max_correlation": 1.,
            "t_model": {"n_estimators": 12, "min_child_samples": 3, "reg_lambda": .1, "reg_alpha": 0.}}


def residual_frame() -> pd.DataFrame:
    """四年合成股票当前特征，raw目标含明显共同项，以检验实际训练标签是否被居中。"""
    frame = create_intraday_frame()
    frame["feature_source_start"] = frame.datetime.dt.normalize() - pd.Timedelta(days=28)
    frame["f_history_days"] = 100.
    frame["f_market_day_return"] = frame.datetime.dt.day * .001 + frame.datetime.dt.hour * .0001
    frame["target_return"] += frame.datetime.dt.day * .003 + frame.datetime.dt.hour * .001
    frame["recovery_price"] = frame.execution_price * (1 + frame.target_return)
    return frame


class WeeklyTResidualTests(unittest.TestCase):
    """独立手算均值与输入哈希，不通过调用旧raw模型伪装残差模型。"""

    @classmethod
    def setUpClass(cls):
        """只训练一个小残差模型，供重载、未来标签独立与原生预测共用。"""
        cls.temporary = tempfile.TemporaryDirectory(prefix="weekly_t_residual_")
        cls.root = Path(cls.temporary.name)
        cls.cfg, cls.frame = residual_config(), residual_frame()
        cls.bundle = fit_residual_t_model(cls.frame, [2024, 2025], cls.cfg, cls.root / "base", [2023, 2026])

    @classmethod
    def tearDownClass(cls):
        """仅清理测试创建的临时模型目录。"""
        cls.temporary.cleanup()

    def test_raw_purge_precedes_cohort_mean_and_inputs_are_unchanged(self):
        frame = self.frame.copy(deep=True)
        first_clock = frame.loc[frame.datetime.dt.year.eq(2024), "datetime"].min()
        group = frame.index[frame.datetime.eq(first_clock)]
        # 这个极大标签必须先被真实来源隔离删除，不能污染同刻其他供体的均值。
        frame.loc[group[0], "feature_source_start"] = pd.Timestamp("2023-12-01")
        frame.loc[group[0], "target_return"] = 1000000.
        # 同刻另一标签跨入内部年，亦须在居中前删除。
        frame.loc[group[1], "label_end"] = pd.Timestamp("2026-01-02 14:55")
        frame.loc[group[1], "target_return"] = -1000000.
        original = frame.copy(deep=True)
        training, metadata, _, donors = prepare_residual_training(frame, [2024, 2025], self.cfg, [2023, 2026])
        observed = donors.loc[donors.datetime.eq(first_clock)]
        expected = frame.loc[group[2:], "target_return"].mean()
        self.assertEqual(len(observed), 8)
        np.testing.assert_allclose(observed.cohort_mean_return, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(observed.residual_target_return,
                                   observed.raw_target_return - expected, rtol=0, atol=1e-15)
        self.assertTrue(donors.datetime.dt.year.isin([2024, 2025]).all())
        self.assertEqual(metadata["raw_training_rows"], 238)
        self.assertLess(metadata["max_absolute_donor_clock_mean_residual"], 1e-14)
        pd.testing.assert_series_equal(training.target_return, original.loc[training.index, "target_return"])
        pd.testing.assert_frame_equal(frame, original)

    def test_singleton_is_audited_but_excluded_from_training_and_support_rows(self):
        frame = self.frame.copy(deep=True)
        clock = frame.loc[frame.datetime.dt.year.eq(2024), "datetime"].min()
        group = frame.index[frame.datetime.eq(clock)]
        frame.loc[group[1:], "feature_source_start"] = pd.Timestamp("2023-12-01")
        training, metadata, audit, donors = prepare_residual_training(frame, [2024, 2025], self.cfg, [2023, 2026])
        singleton = donors.loc[donors.datetime.eq(clock)]
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton.cohort_size.iloc[0], 1)
        self.assertFalse(singleton.kept_for_training.iloc[0])
        self.assertEqual(singleton.residual_target_return.iloc[0], 0.)
        self.assertFalse(training.datetime.eq(clock).any())
        self.assertEqual(metadata["raw_training_rows"], 231)
        self.assertEqual(metadata["training_rows"], 230)
        self.assertEqual(metadata["singleton_rows_excluded"], 1)
        self.assertEqual(metadata["singleton_clock_count"], 1)
        self.assertEqual(audit.loc[audit.reason.eq("retained_residual_training_rows"), "excluded_rows"].iloc[0], 230)

    def test_full_donors_center_before_sampling_and_residual_and_raw_hashes_differ(self):
        cfg = {**self.cfg, "t_max_train_rows": 20}
        bundle = fit_residual_t_model(self.frame, [2024, 2025], cfg, self.root / "capped", [2023, 2026])
        training, _, _, donors = prepare_residual_training(self.frame, [2024, 2025], cfg, [2023, 2026])
        fitted = training.sample(n=20, random_state=7).sort_index()
        metadata = bundle["metadata"]
        self.assertEqual(metadata["donor_rows"], 240)
        self.assertEqual(metadata["fitted_rows"], 20)
        self.assertTrue(donors.cohort_size.eq(10).all())
        self.assertEqual(metadata["target_decomposition_sha256"], calculate_residual_frame_sha256(donors))
        x = fitted[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        keys = fitted[["symbol", "datetime"]]
        residual_hash = calculate_residual_frame_sha256(pd.concat([keys, x, fitted.target_residual_return.rename("return")], axis=1))
        raw_hash = calculate_residual_frame_sha256(pd.concat([keys, x, fitted.target_return.rename("return")], axis=1))
        self.assertEqual(metadata["training_sample_sha256"], residual_hash)
        self.assertEqual(metadata["raw_training_sample_sha256"], raw_hash)
        self.assertNotEqual(residual_hash, raw_hash)
        self.assertEqual(metadata["feature_source_sha256"], calculate_residual_frame_sha256(fitted[["symbol", "datetime", "feature_source_start"]]))
        persisted = pd.read_csv(self.root / "capped/target_decomposition.csv", float_precision="round_trip",
                                parse_dates=["datetime", "feature_source_start", "label_start", "label_end"])
        self.assertEqual(persisted.columns.tolist(), DECOMPOSITION_COLUMNS)
        pd.testing.assert_frame_equal(persisted, donors.reset_index(drop=True))
        self.assertEqual(metadata["target_decomposition_file_sha256"], hashlib.sha256((self.root / "capped/target_decomposition.csv").read_bytes()).hexdigest())

    def test_future_missing_prices_labels_and_inference_group_size_do_not_change_predictions(self):
        frame = self.frame.loc[self.frame.datetime.dt.year.eq(2026)].copy()
        original = frame.copy(deep=True)
        expected = predict_residual_t(self.bundle, frame, self.cfg)
        changed = frame.copy(deep=True)
        for name in ["execution_price", "recovery_price", "target_return", "cohort_mean_return", "cohort_size", "target_residual_return"]:
            changed[name] = np.nan
        for name in ["execution_time", "recovery_time", "label_start", "label_end"]:
            changed[name] = pd.NaT
        changed["label_available"] = False
        actual = predict_residual_t(self.bundle, changed, self.cfg)
        fields = ["datetime", "symbol", "prediction", "residual_prediction", "market_prediction"]
        pd.testing.assert_frame_equal(actual[fields], expected[fields])
        single = changed.iloc[[0]].copy()
        single_result = predict_residual_t(self.bundle, single, self.cfg)
        self.assertEqual(len(single_result), 1)
        self.assertEqual(single_result.prediction.iloc[0], expected.prediction.iloc[0])
        self.assertTrue(expected.market_prediction.eq(0).all())
        pd.testing.assert_series_equal(expected.prediction, expected.residual_prediction, check_names=False)
        self.assertTrue(actual.label_end.isna().all())
        pd.testing.assert_frame_equal(frame, original)

    def test_saved_native_and_joblib_predict_exact_same_residual_and_market_features_remain_allowed(self):
        restored = joblib.load(self.root / "base/model_bundle.joblib")
        frame = self.frame.loc[self.frame.datetime.dt.year.eq(2026)].copy()
        expected = predict_residual_t(self.bundle, frame, self.cfg)
        actual = predict_residual_t(restored, frame, self.cfg)
        pd.testing.assert_frame_equal(actual, expected)
        native = lightgbm.Booster(model_file=str(self.root / "base/return_model.txt"))
        self.assertEqual(native.feature_name(), restored["features"])
        np.testing.assert_array_equal(native.predict(actual[restored["features"]]), actual.residual_prediction)
        self.assertIn("f_market_day_return", restored["features"])
        self.assertNotIn("f_history_days", restored["features"])
        metadata = json.loads((self.root / "base/model_metadata.json").read_text())
        self.assertEqual(metadata["t_target_mode"], TARGET_MODE)
        self.assertIn("残差", metadata["label_definition"])
        self.assertNotEqual(metadata["training_sample_sha256"], metadata["raw_training_sample_sha256"])

    def test_excluded_year_values_cannot_change_donors_features_or_fitted_model(self):
        changed = self.frame.copy(deep=True)
        outside = changed.datetime.dt.year.isin([2023, 2026])
        changed.loc[outside, ["target_return", "f_signal", "f_market_day_return"]] = 1e8
        bundle = fit_residual_t_model(changed, [2024, 2025], self.cfg, self.root / "heldout_changed", [2023, 2026])
        for key in ["raw_training_target_sha256", "residual_training_target_sha256", "target_decomposition_sha256",
                    "training_sample_sha256", "raw_training_sample_sha256", "feature_source_sha256"]:
            self.assertEqual(bundle["metadata"][key], self.bundle["metadata"][key])
        self.assertEqual(bundle["features"], self.bundle["features"])
        self.assertEqual((self.root / "base/return_model.txt").read_bytes(), (self.root / "heldout_changed/return_model.txt").read_bytes())

    def test_invalid_fixed_prior_group_policy_and_provenance_fail_closed(self):
        for change in [{"t_market_prior": value} for value in [True, "0", np.nan, np.inf, .000001, -1.]] + [
                {"t_target_mode": "raw"}, {"t_residual_min_group_size": 1}, {"t_residual_min_group_size": 3},
                {"t_residual_min_group_size": True}, {"t_residual_min_group_size": 2.0},
                {"strict_feature_provenance": False}, {"excluded_model_features": []}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_residual_config({**self.cfg, **change})
        old = deepcopy(self.bundle)
        old["target_mode"] = "raw"
        with self.assertRaises(ValueError):
            predict_residual_t(old, self.frame.iloc[:1], self.cfg)
        old = deepcopy(self.bundle)
        old["metadata"]["t_market_prior"] = .001
        with self.assertRaises(ValueError):
            predict_residual_t(old, self.frame.iloc[:1], self.cfg)

    def test_mixed_horizons_in_one_clock_and_all_singletons_are_rejected(self):
        changed = self.frame.copy(deep=True)
        first = changed.index[changed.datetime.dt.year.eq(2024)][0]
        changed.loc[first, "label_end"] -= pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "不能混合周期"):
            prepare_residual_training(changed, [2024, 2025], self.cfg, [2023, 2026])
        one = self.frame.drop_duplicates("datetime")
        with self.assertRaisesRegex(ValueError, "去除不足2股"):
            prepare_residual_training(one, [2024, 2025], self.cfg, [2023, 2026])

    def test_existing_model_directory_cannot_be_overwritten(self):
        before = (self.root / "base/model_bundle.joblib").read_bytes()
        with self.assertRaisesRegex(ValueError, "不能覆盖"):
            fit_residual_t_model(self.frame, [2024, 2025], self.cfg, self.root / "base", [2023, 2026])
        self.assertEqual(before, (self.root / "base/model_bundle.joblib").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
