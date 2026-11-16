"""0.70覆盖率消融的真实合成拟合、来源和原生身份反例；不读取评价数据。"""
from __future__ import annotations

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

from quant_backtest import inventory_t_coverage_models as coverage
from quant_backtest.inventory_t_learning_models import fit_inventory_return_model
from quant_backtest.weekly_t_models import create_model_parameters
from quant_backtest.weekly_t_short_horizon import planned_recovery_times


def make_config():
    """只缩小合成测试树数，原经济、来源隔离及特征选择合同保持。"""
    cfg = yaml.safe_load((ROOT / "config/inventory_learning_t_v14.yaml").read_text())
    cfg.update(research_version=coverage.VERSION, training_loss="regression_l2",
        feature_min_coverage=.70, source_feature_min_coverage=.80,
        forecast_target="fixed_20_40_60_reference_return", learning_model_profiles=list(coverage.PROFILES),
        source_learning_model_profiles=list(coverage.PROFILES), t_recheck_exit_side_at_execution=True,
        t_model_profile="l2_depth4", t_label_horizon_minutes=20, workers=1,
        min_training_rows=20, t_max_features=10, feature_selection_sample_rows=400,
        t_model={"n_estimators": 8, "min_child_samples": 5, "learning_rate": .1})
    return cfg


def make_frame():
    """两核心年八时钟：早两钟15m因果缺失，整日跨排除年来源须被剔除。"""
    days = list(pd.bdate_range("2023-03-01", periods=30)) + list(pd.bdate_range("2024-03-01", periods=30))
    clocks = ("09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00")
    times = pd.DatetimeIndex([d + pd.to_timedelta(clock + ":00") for d in days for clock in clocks])
    rng = np.random.default_rng(42)
    old = rng.normal(size=len(times)).astype("float32")
    fifteen = rng.normal(size=len(times)).astype("float32")
    fifteen[np.arange(len(times)) % 8 < 2] = np.nan
    raw = .015 * np.nan_to_num(fifteen) + .004 * old
    raw[32:37] = [.14, -.12, .17, -.15, .18]
    frame = pd.DataFrame({"datetime": times, "symbol": "SH.600001", "cohort_target_year": times.year,
        "baseline_opening_shares": 1000, "feature_source_start": times.normalize() - pd.Timedelta(days=7),
        "f_signal": old, "f_return_15m": fifteen, "f_noise": rng.normal(size=len(times)).astype("float32"),
        "f_only_excluded": np.where(np.arange(len(times)) < 16, old, 0).astype("float32"),
        "f_history_days": np.arange(len(times)), "execution_price": 10.,
        "recovery_price": 10 * (1 + raw.astype("float64")), "horizon_minutes": 20})
    frame["target_return"] = frame.recovery_price / frame.execution_price - 1
    frame.loc[:15, "feature_source_start"] = pd.Timestamp("2022-12-23")
    frame["label_start"] = frame.datetime + pd.Timedelta(minutes=1)
    frame["label_end"] = planned_recovery_times(frame.datetime, 20)
    return frame


class CoverageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.frame, cls.cfg = make_frame(), make_config()
        cls.original = cls.frame.copy(deep=True)
        cls.path = cls.root / "coverage"
        cls.meta = coverage.fit_inventory_coverage_model(cls.frame, [2023, 2024], cls.cfg, cls.path, [2022, 2025])
        cls.bundle = joblib.load(cls.path / "model_bundle.joblib")
        cls.old = fit_inventory_return_model(cls.frame, [2023, 2024], {**cls.cfg, "feature_min_coverage": .8},
                                             cls.root / "source", [2022, 2025])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_actual_l2_parameters_are_completely_unchanged_for_both_profiles(self):
        self.assertEqual(coverage.SOURCE_PROFILE_MAP, {p: p for p in coverage.PROFILES})
        for profile in coverage.PROFILES:
            cfg = {**self.cfg, "t_model_profile": profile}
            expected, _ = create_model_parameters(cfg, "intraday")
            self.assertEqual(coverage.create_coverage_model_parameters(cfg), expected)
            self.assertEqual(expected["objective"], "regression")
            self.assertNotIn("alpha", expected)

    def test_source_control_is_exact_old_matrix_manifest_labels_and_split(self):
        for actual, source in (("source_training_inputs.pkl", "training_inputs.pkl"),
                               ("source_feature_manifest.csv", "feature_manifest.csv"),
                               ("training_labels.csv", "training_labels.csv"),
                               ("training_split_audit.csv", "training_split_audit.csv")):
            self.assertEqual((self.path / actual).read_bytes(), (self.root / "source" / source).read_bytes(), actual)
        self.assertEqual(len(list(self.path.iterdir())), 10)
        self.assertEqual(self.meta["source_features"], self.old["features"])
        self.assertEqual(self.meta["source_training_matrix_sha256"], self.old["training_matrix_sha256"])
        for name in ("full_training_rows_sha256", "training_target_sha256", "training_rows", "fitted_rows"):
            self.assertEqual(self.meta[name], self.old[name])
        self.assertEqual(self.meta["training_rows"], 464)

    def test_structural_missing_feature_enters_without_filling_or_changing_common_columns(self):
        self.assertIn("f_return_15m", self.bundle["features"])
        self.assertNotIn("f_return_15m", self.meta["source_features"])
        actual = pd.read_pickle(self.path / "training_inputs.pkl")
        old = pd.read_pickle(self.path / "source_training_inputs.pkl")
        pd.testing.assert_frame_equal(actual[list(old)], old, check_exact=True)
        pd.testing.assert_series_equal(actual.f_return_15m, self.frame.loc[actual.index, "f_return_15m"], check_exact=True)
        self.assertEqual(actual.f_return_15m.dtype, np.dtype("float32"))
        self.assertEqual(actual.f_return_15m.isna().mean(), .25)
        self.assertFalse(self.meta["nan_values_filled"])
        self.assertNotIn("f_history_days", actual)
        self.assertNotIn("f_only_excluded", actual)
        pd.testing.assert_frame_equal(self.frame, self.original, check_exact=True)

    def test_native_weights_feature_order_and_saved_matrix_are_actual_fitted_inputs(self):
        native = lightgbm.Booster(model_file=str(self.path / "return_model.txt"))
        matrix = pd.read_pickle(self.path / "training_inputs.pkl")[self.bundle["features"]]
        self.assertEqual(native.params["objective"], "regression")
        self.assertEqual(native.feature_name(), list(matrix))
        self.assertEqual(native.model_to_string(), self.bundle["model"].booster_.model_to_string())
        np.testing.assert_array_equal(native.predict(matrix), self.bundle["model"].predict(matrix))
        self.assertEqual(coverage.calculate_learning_frame_sha256(matrix), self.meta["training_matrix_sha256"])
        labels = pd.read_csv(self.path / "training_labels.csv", float_precision="round_trip")
        np.testing.assert_array_equal(labels.target_return, labels.recovery_price / labels.execution_price - 1)
        self.assertEqual(labels.target_return.abs().gt(.1).sum(), 5)
        self.assertTrue(labels.fitted.all())
        self.assertFalse(self.meta["training_targets_clipped"])

    def test_inference_keeps_unlabeled_rows_and_is_invariant_to_future_erasure(self):
        before = coverage.predict_inventory_coverage_model(self.bundle, self.frame, self.cfg)
        stripped = self.frame.drop(columns=[c for c in self.frame if c.startswith(("execution_", "recovery_", "target_", "label_"))])
        after = coverage.predict_inventory_coverage_model(self.bundle, stripped, self.cfg)
        self.assertEqual(len(after), len(self.frame))
        pd.testing.assert_series_equal(before.prediction, after.prediction, check_exact=True)
        np.testing.assert_array_equal(after.prediction, after.return_score)
        np.testing.assert_array_equal(after.prediction, after.raw_prediction)
        self.assertTrue(after.model_version.eq(coverage.VERSION).all())
        self.assertTrue(after.feature_min_coverage.eq(.7).all())
        self.assertTrue(after.forecast_loss.eq("regression_l2").all())
        self.assertNotIn("forecast_alpha", after)
        self.assertTrue(coverage.predict_inventory_coverage_model(self.bundle, stripped.iloc[:0], self.cfg).empty)

    def test_original_weights_or_wrong_version_coverage_parameters_cannot_disguise(self):
        bad = {**self.bundle, "model": joblib.load(self.root / "source/model_bundle.joblib")["model"]}
        with self.assertRaises(ValueError):
            coverage.predict_inventory_coverage_model(bad, self.frame, self.cfg)
        for key, value in (("research_version", "inventory_learning_t_v14"), ("feature_min_coverage", .8),
                           ("source_feature_min_coverage", .7), ("profile", "l2_depth6"),
                           ("horizon_minutes", 40), ("training_loss", "huber"), ("source_l2_profile", "l2_depth6")):
            bad = {**self.bundle, "metadata": {**self.bundle["metadata"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                coverage.predict_inventory_coverage_model(bad, self.frame, self.cfg)
        with self.assertRaises(ValueError):
            coverage.predict_inventory_coverage_model(self.bundle, self.frame, {**self.cfg, "workers": 2})

    def test_invalid_config_cannot_turn_single_change_into_new_loss_target_or_sampling(self):
        for key, value in (("feature_min_coverage", .6), ("source_feature_min_coverage", .9),
                           ("training_loss", "huber"), ("t_max_train_rows", 100),
                           ("forecast_target", "same_day_1455"), ("strict_feature_provenance", False),
                           ("embargo_days", 0), ("excluded_model_features", []),
                           ("t_model_profile", "huber_depth4"), ("horizons", [20, 60]),
                           ("t_model", {"objective": "regression"}), ("t_model", {"alpha": .01})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                coverage.validate_coverage_model_config({**self.cfg, key: value})

    def test_wrong_core_inventory_endpoint_or_price_identity_is_rejected(self):
        for key, value in (("baseline_opening_shares", 0), ("cohort_target_year", 2025),
                           ("target_return", .8), ("horizon_minutes", 40),
                           ("label_start", pd.Timestamp("2023-03-07 09:37")), ("recovery_price", -1.)):
            changed = self.frame.copy()
            changed.loc[32, key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                coverage.fit_inventory_coverage_model(changed, [2023, 2024], self.cfg, self.root / f"bad_{key}", [2022, 2025])
        with self.assertRaises(ValueError):
            coverage.fit_inventory_coverage_model(self.frame, [2023, 2024], self.cfg, self.root / "overlap", [2024, 2025])

    def test_missing_future_or_source_affects_only_fitting_and_never_truncates_tail(self):
        frame = self.frame.copy()
        frame.loc[40, "target_return"] = np.nan
        frame.loc[41, "feature_source_start"] = pd.NaT
        meta = coverage.fit_inventory_coverage_model(frame, [2023, 2024], self.cfg, self.root / "missing", [2022, 2025])
        self.assertEqual(meta["training_rows"], 462)
        self.assertEqual(len(coverage.predict_inventory_coverage_model(self.bundle, frame, self.cfg)), len(frame))
        self.assertEqual(pd.read_csv(self.root / "missing/training_labels.csv").target_return.abs().gt(.1).sum(), 5)

    def test_native_nan_cannot_write_success_certificate(self):
        original = coverage.lightgbm.Booster
        def restore(*args, **kwargs):
            native = original(*args, **kwargs)
            if "model_file" in kwargs:
                native.predict = lambda matrix: np.full(len(matrix), np.nan)
            return native
        out = self.root / "nan_native"
        with patch.object(coverage.lightgbm, "Booster", side_effect=restore), self.assertRaisesRegex(AssertionError, "全为有限数"):
            coverage.fit_inventory_coverage_model(self.frame, [2023, 2024], self.cfg, out, [2022, 2025])
        self.assertFalse((out / "native_reload_audit.json").exists())

    def test_depth6_late_horizon_drops_label_only_but_keeps_current_inference(self):
        frame = self.frame.copy()
        frame["horizon_minutes"] = 60
        frame["label_end"] = planned_recovery_times(frame.datetime, 60)
        cfg = {**self.cfg, "t_model_profile": "l2_depth6", "t_label_horizon_minutes": 60}
        out = self.root / "h60"
        meta = coverage.fit_inventory_coverage_model(frame, [2023, 2024], cfg, out, [2022, 2025])
        self.assertEqual(meta["fitted_rows"], 406)
        inputs = pd.read_pickle(out / "training_inputs.pkl")
        self.assertFalse(inputs.datetime.dt.strftime("%H:%M").eq("14:00").any())
        self.assertEqual(inputs.f_return_15m.notna().mean(), 5/7)
        result = coverage.predict_inventory_coverage_model(joblib.load(out / "model_bundle.joblib"), frame, cfg)
        self.assertEqual(len(result), len(frame))
        self.assertEqual(meta["model_parameters"]["max_depth"], 6)
        with self.assertRaisesRegex(ValueError, "不能覆盖"):
            coverage.fit_inventory_coverage_model(frame, [2023, 2024], cfg, out, [2022, 2025])

    def test_inference_missing_duplicate_keys_or_nonfinite_native_output_rejected(self):
        for frame in (self.frame.drop(columns=self.bundle["features"][0]), pd.concat([self.frame, self.frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                coverage.predict_inventory_coverage_model(self.bundle, frame, self.cfg)
        with patch.object(self.bundle["model"], "predict", return_value=np.full(len(self.frame), np.nan)), self.assertRaises(ValueError):
            coverage.predict_inventory_coverage_model(self.bundle, self.frame, self.cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
