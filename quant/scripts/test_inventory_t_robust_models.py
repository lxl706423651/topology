"""Huber与原L2训练输入一致性、损失身份和因果反例；只拟合临时合成小模型。"""
from __future__ import annotations

import copy
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import quant_backtest.inventory_t_robust_models as robust
from quant_backtest.inventory_t_learning_models import fit_inventory_return_model
from quant_backtest.weekly_t_short_horizon import planned_recovery_times
from quant_backtest.weekly_t_models import create_model_parameters


def make_config():
    """合成测试仅减小树数和叶片样本，经济字段与实际原配置保持。"""
    cfg = yaml.safe_load((ROOT/"config/inventory_learning_t_v14.yaml").read_text())
    cfg.update(research_version=robust.VERSION, training_loss="huber", t_huber_alpha=.01,
        forecast_target="fixed_20_40_60_reference_return", learning_model_profiles=list(robust.PROFILES),
        source_learning_model_profiles=["l2_depth4", "l2_depth6"], t_recheck_exit_side_at_execution=True,
        t_model_profile="huber_depth4", t_label_horizon_minutes=20, workers=1, min_training_rows=20,
        t_max_features=3, feature_selection_sample_rows=200,
        t_model={"n_estimators": 8, "min_child_samples": 5, "learning_rate": .1})
    return cfg


def make_frame():
    """两core年240行，10行跨排除年被purge；合法极端标签必须完整保留。"""
    days = pd.DatetimeIndex(list(pd.bdate_range("2023-03-01", periods=120))+
                            list(pd.bdate_range("2024-03-01", periods=120)))
    x = np.sin(np.arange(len(days))*.37)
    raw = .012+.025*x
    raw[30:35] = [.14, -.12, .17, -.15, .18]
    frame = pd.DataFrame({"datetime": days+pd.Timedelta(hours=9, minutes=35),
        "symbol": np.where(np.arange(len(days))%2, "SH.600001", "SZ.000001"),
        "cohort_target_year": days.year, "baseline_opening_shares": 1000,
        "feature_source_start": days-pd.Timedelta(days=7), "f_signal": x.astype("float32"),
        "f_noise": np.cos(np.arange(len(days))*.13).astype("float32"),
        "f_only_excluded": np.where(np.arange(len(days))<10, x, 0.).astype("float32"),
        "f_history_days": np.arange(len(days)), "execution_price": 10.,
        "recovery_price": 10*(1+raw), "horizon_minutes": 20})
    frame["target_return"] = frame.recovery_price/frame.execution_price-1
    frame.loc[:9, "feature_source_start"] = pd.Timestamp("2022-12-23")
    frame["label_start"] = frame.datetime+pd.Timedelta(minutes=1)
    frame["label_end"] = planned_recovery_times(frame.datetime, 20)
    return frame


class RobustModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.frame, cls.cfg = make_frame(), make_config()
        cls.original = cls.frame.copy(deep=True)
        cls.path = cls.root/"huber"
        cls.meta = robust.fit_inventory_robust_model(cls.frame, [2023, 2024], cls.cfg, cls.path, [2022, 2025])
        cls.bundle = joblib.load(cls.path/"model_bundle.joblib")
        source_cfg = {**cls.cfg, "t_model_profile": "l2_depth4"}
        cls.source_meta = fit_inventory_return_model(cls.frame, [2023, 2024], source_cfg, cls.root/"l2", [2022, 2025])

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_only_loss_and_alpha_change_and_all_other_parameters_exact(self):
        for profile, old in robust.SOURCE_PROFILE_MAP.items():
            cfg = {**self.cfg, "t_model_profile": profile}
            actual = robust.create_robust_model_parameters(cfg)
            original, _ = create_model_parameters({**cfg, "t_model_profile": old}, "intraday")
            self.assertEqual({k: v for k, v in actual.items() if k not in ("objective", "alpha")},
                             {k: v for k, v in original.items() if k not in ("objective", "alpha")})
            self.assertEqual(actual["objective"], "huber")
            self.assertEqual(actual["alpha"], .01)

    def test_actual_matrix_raw_labels_manifest_and_purge_byte_equal_original_l2(self):
        for name in ("training_inputs.pkl", "training_labels.csv", "feature_manifest.csv", "training_split_audit.csv"):
            self.assertEqual((self.path/name).read_bytes(), (self.root/"l2"/name).read_bytes(), name)
        self.assertEqual(len(list(self.path.iterdir())), 8)
        self.assertEqual(self.meta["training_rows"], 230)
        self.assertEqual(self.meta["fitted_rows"], 230)
        self.assertNotIn("f_history_days", self.bundle["features"])
        self.assertNotIn("f_only_excluded", self.bundle["features"])
        labels = pd.read_csv(self.path/"training_labels.csv", float_precision="round_trip")
        self.assertGreater(labels.target_return.abs().max(), .15)
        self.assertTrue(labels.fitted.all())
        self.assertFalse(self.meta["training_targets_clipped"])
        pd.testing.assert_frame_equal(self.frame, self.original)

    def test_native_huber_objective_and_all_saved_weights_reload_exact(self):
        native = lightgbm.Booster(model_file=str(self.path/"return_model.txt"))
        self.assertEqual(native.params["objective"], "huber")
        self.assertEqual(native.params["alpha"], .01)
        self.assertEqual(native.model_to_string(), self.bundle["model"].booster_.model_to_string())
        matrix = pd.read_pickle(self.path/"training_inputs.pkl")[self.bundle["features"]]
        np.testing.assert_array_equal(native.predict(matrix), self.bundle["model"].predict(matrix))
        labels = pd.read_csv(self.path/"training_labels.csv", float_precision="round_trip")
        expected = robust.calculate_fit_residual_statistics(labels.target_return.to_numpy(), native.predict(matrix))
        self.assertEqual(expected, self.meta["fit_residual_statistics"])
        self.assertFalse(self.meta["score_is_calibrated_conditional_mean"])

    def test_current_inference_never_uses_future_targets_or_removes_unlabeled_keys(self):
        before = robust.predict_inventory_robust_model(self.bundle, self.frame, self.cfg)
        stripped = self.frame.drop(columns=[c for c in self.frame if c.startswith(("execution_", "recovery_", "target_", "label_"))])
        after = robust.predict_inventory_robust_model(self.bundle, stripped, self.cfg)
        self.assertEqual(len(after), len(self.frame))
        pd.testing.assert_series_equal(before.prediction, after.prediction)
        self.assertTrue(after.model_version.eq(robust.VERSION).all())
        self.assertTrue(after.forecast_alpha.eq(.01).all())
        np.testing.assert_array_equal(after.prediction, after.return_score)
        empty = robust.predict_inventory_robust_model(self.bundle, stripped.iloc[:0], self.cfg)
        self.assertTrue(empty.empty)
        self.assertIn("return_score", empty)

    def test_wrong_actual_l2_weights_cannot_disguise_as_huber_metadata(self):
        bad = {**self.bundle, "model": joblib.load(self.root/"l2/model_bundle.joblib")["model"]}
        with self.assertRaisesRegex(ValueError, "实际参数/原生损失"):
            robust.predict_inventory_robust_model(bad, self.frame, self.cfg)
        for key, value in (("profile", "huber_depth6"), ("horizon_minutes", 60),
                           ("research_version", "inventory_learning_t_v14"), ("huber_alpha", .02)):
            bad = {**self.bundle, "metadata": {**self.bundle["metadata"], key: value}}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "模型身份"):
                robust.predict_inventory_robust_model(bad, self.frame, self.cfg)

    def test_illegal_alpha_sampling_wrong_core_or_endpoint_rejected(self):
        for key, value in (("t_huber_alpha", .02), ("training_loss", "regression"), ("t_max_train_rows", 20),
                           ("forecast_target", "same_day_1455"), ("embargo_days", 0),
                           ("strict_feature_provenance", False), ("t_model_profile", "l2_depth4")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                robust.validate_robust_model_config({**self.cfg, key: value})
        for key, value in (("baseline_opening_shares", 0), ("cohort_target_year", 2025),
                           ("target_return", .8), ("horizon_minutes", 40)):
            frame = self.frame.copy()
            frame.loc[30, key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                robust.fit_inventory_robust_model(frame, [2023, 2024], self.cfg, self.root/f"invalid_{key}", [2022, 2025])
        with self.assertRaises(ValueError):
            robust.fit_inventory_robust_model(self.frame, [2023, 2024], self.cfg, self.root/"overlap", [2024, 2025])

    def test_missing_future_fit_rows_removed_but_extreme_labels_not_truncated(self):
        frame = self.frame.copy()
        frame.loc[40, "target_return"] = np.nan
        frame.loc[41, "feature_source_start"] = pd.NaT
        meta = robust.fit_inventory_robust_model(frame, [2023, 2024], self.cfg, self.root/"missing", [2022, 2025])
        self.assertEqual(meta["training_rows"], 228)
        labels = pd.read_csv(self.root/"missing/training_labels.csv")
        self.assertEqual(labels.target_return.abs().gt(.1).sum(), 5)
        self.assertEqual(len(robust.predict_inventory_robust_model(self.bundle, frame, self.cfg)), len(frame))

    def test_native_nan_never_writes_success_proof(self):
        original = robust.lightgbm.Booster
        def restore(*args, **kwargs):
            native = original(*args, **kwargs)
            if "model_file" in kwargs:
                native.predict = lambda matrix: np.full(len(matrix), np.nan)
            return native
        out = self.root/"nan_native"
        with patch.object(robust.lightgbm, "Booster", side_effect=restore), self.assertRaisesRegex(AssertionError, "全为有限数"):
            robust.fit_inventory_robust_model(self.frame, [2023, 2024], self.cfg, out, [2022, 2025])
        self.assertFalse((out/"native_reload_audit.json").exists())

    def test_depth6_three_horizons_and_no_overwrite(self):
        for horizon in (40, 60):
            frame = self.frame.copy()
            frame["horizon_minutes"] = horizon
            frame["label_end"] = planned_recovery_times(frame.datetime, horizon)
            cfg = {**self.cfg, "t_model_profile": "huber_depth6", "t_label_horizon_minutes": horizon}
            out = self.root/f"depth6_{horizon}"
            meta = robust.fit_inventory_robust_model(frame, [2023, 2024], cfg, out, [2022, 2025])
            self.assertEqual(meta["source_l2_profile"], "l2_depth6")
            self.assertEqual(meta["model_parameters"]["max_depth"], 6)
            actual = robust.predict_inventory_robust_model(joblib.load(out/"model_bundle.joblib"), frame, cfg)
            self.assertTrue(actual.horizon_minutes.eq(horizon).all())
        with self.assertRaisesRegex(ValueError, "不能覆盖"):
            robust.fit_inventory_robust_model(self.frame, [2023, 2024], self.cfg, self.path, [2022, 2025])

    def test_residual_tail_denominators_use_all_rows_and_keep_zero_loss_finite(self):
        result = robust.calculate_fit_residual_statistics(np.array([0., .02, -.03]), np.zeros(3))
        self.assertEqual(result["abs_residual_gt_alpha_rows"], 2)
        self.assertEqual(result["abs_residual_gt_alpha_fraction"], 2/3)
        self.assertEqual(result["tail_squared_residual_share"], 1.)
        self.assertEqual(robust.calculate_fit_residual_statistics(np.zeros(2), np.zeros(2))["tail_squared_residual_share"], 0.)


if __name__ == "__main__": unittest.main(verbosity=2)
