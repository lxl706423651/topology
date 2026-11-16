"""用固定合成数据验证做T开仓模型的年度隔离和推断契约，不读取原始行情。

从quant项目根目录执行quant环境Python scripts/test_inventory_t_models.py。
只拟合几个微型CPU模型，模型文件写系统临时目录并自动清理；不修改现有缓存。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.factor_library import validate_no_future_leakage
from quant_backtest.inventory_t_models import (
    build_year_folds, calibrate_entry_gate, fit_entry_model, predict_entry_scores,
    select_entry_candidates, select_model_training_rows, select_training_features,
)


def create_model_panel() -> pd.DataFrame:
    """创建四年、六只股票的确定性截面：含有效因子、完全重复因子和独立噪声。"""
    random = np.random.default_rng(5)
    rows = []
    for year in (2023, 2024, 2025, 2026):
        for day in ("01-02", "01-08", "02-01", "02-02"):
            for hour in (10, 14):
                timestamp = pd.Timestamp(f"{year}-{day} {hour}:00")
                for asset in range(6):
                    rank = (asset + 1) / 6
                    rows.append({"datetime": timestamp, "symbol": f"SH.{600000 + asset}",
                                 "label_end": timestamp + pd.Timedelta(days=1),
                                 "forward_return": rank / 100, "target_rank": rank,
                                 "cross_section_count": 6, "cs_rank_alpha": rank,
                                 "cs_rank_duplicate": rank, "cs_rank_noise": random.random()})
    return pd.DataFrame(rows)


def create_model_config() -> dict:
    """微型模型配置保留正式隔离/选特征/门控流程，只减少树数量及训练样本要求。"""
    return {"min_cross_section": 5, "min_training_rows": 20, "workers": 1,
            "embargo_days": 7, "max_features": 2, "feature_selection_sample_rows": 200,
            "feature_min_coverage": 0.8, "feature_max_correlation": 0.9,
            "entry_method": "cta_trend", "gate_quantile": 0.5, "entry_top_n": 5,
            "model": {"n_estimators": 5, "max_depth": 2,
                      "min_child_weight": 1, "random_state": 7}}


class InventoryTModelTests(unittest.TestCase):
    """检查经济研究需要的样本隔离，而不是只检查模型文件是否能够生成。"""

    @classmethod
    def setUpClass(cls):
        """拟合共享的小模型；额外重拟合仅用于证明改变留出数据不会改变训练产物。"""
        cls.temporary = tempfile.TemporaryDirectory(prefix="inventory_t_model_tests_")
        cls.directory = Path(cls.temporary.name)
        cls.panel = create_model_panel()
        cls.original = cls.panel.copy(deep=True)
        cls.cfg = create_model_config()
        cls.original_cfg = copy.deepcopy(cls.cfg)
        cls.model, cls.features, cls.metadata = fit_entry_model(
            cls.panel, [2023, 2025], cls.cfg, cls.directory / "original", [2024, 2026])

    @classmethod
    def tearDownClass(cls):
        """清理本组测试创建的全部模型产物，不保留临时实验报告。"""
        cls.temporary.cleanup()

    def test_outer_folds_exclude_requested_year_and_label_forward_protocol(self):
        folds = build_year_folds(self.panel, [2023, 2024, 2025, 2026])
        by_year = {fold["test_year"]: fold for fold in folds}
        self.assertEqual(by_year[2024]["train_years"], [2023, 2025, 2026])
        self.assertEqual(by_year[2026]["train_years"], [2023, 2024, 2025])
        self.assertEqual(by_year[2026]["protocol"], "forward")
        for year in (2023, 2024, 2025):
            self.assertEqual(by_year[year]["protocol"], "leave_one_year_out_retrospective")
        for fold in folds:
            self.assertNotIn(fold["test_year"], fold["train_years"])

    def test_inner_validation_is_latest_train_year_and_core_excludes_both_holdouts(self):
        for fold in build_year_folds(self.panel, [2023, 2024, 2025, 2026]):
            inner = fold["inner_validation_year"]
            self.assertEqual(inner, max(fold["train_years"]))
            core = select_model_training_rows(self.panel, fold["core_train_years"], self.cfg,
                                              [fold["test_year"], inner])
            self.assertNotIn(inner, set(core.datetime.dt.year))
            self.assertNotIn(fold["test_year"], set(core.datetime.dt.year))
            self.assertTrue(set(core.label_end.dt.year).issubset(fold["core_train_years"]))

    def test_label_end_crossing_holdout_or_nonfuture_timestamp_is_purged(self):
        frame = self.panel.loc[self.panel.datetime.dt.year == 2023].head(6).copy()
        frame["datetime"] = pd.Timestamp("2023-12-29 10:00")
        frame["label_end"] = pd.Timestamp("2023-12-29 14:00")
        frame.loc[frame.index[0], "label_end"] = pd.Timestamp("2024-01-02 10:00")
        frame.loc[frame.index[1], "label_end"] = frame.datetime.iloc[1]
        frame.loc[frame.index[2], "forward_return"] = np.nan
        frame.loc[frame.index[3], "target_rank"] = np.inf
        training = select_model_training_rows(frame, [2023], self.cfg, [2024])
        self.assertEqual(list(training.symbol), list(frame.symbol.iloc[4:]))

    def test_seven_calendar_day_embargo_ends_on_january_eighth(self):
        frame = self.panel.loc[self.panel.datetime.dt.year == 2025].copy()
        frame.loc[frame.datetime.dt.day == 2, "datetime"] = pd.Timestamp("2025-01-07 23:59")
        frame["label_end"] = frame.datetime + pd.Timedelta(days=1)
        training = select_model_training_rows(frame, [2025], self.cfg, [2024])
        self.assertFalse((training.datetime < pd.Timestamp("2025-01-08")).any())
        self.assertTrue((training.datetime == pd.Timestamp("2025-01-08 10:00")).any())
        unrestricted = select_model_training_rows(frame, [2025], {**self.cfg, "embargo_days": 0}, [2024])
        self.assertGreater(len(unrestricted), len(training))

    def test_small_cross_sections_are_training_only_filter(self):
        frame = self.panel.copy()
        blocked_time = pd.Timestamp("2023-02-01 10:00")
        frame.loc[frame.datetime == blocked_time, "cross_section_count"] = 1
        training = select_model_training_rows(frame, [2023], self.cfg, [2024])
        self.assertFalse((training.datetime == blocked_time).any())
        scores = predict_entry_scores(self.model, self.features, frame, [2023], self.cfg)
        self.assertEqual(len(scores), int((frame.datetime.dt.year == 2023).sum()))

    def test_heldout_feature_and_label_changes_do_not_change_training(self):
        frame = self.panel.copy(deep=True)
        heldout = frame.datetime.dt.year.isin([2024, 2026])
        for column in ("cs_rank_alpha", "cs_rank_duplicate", "cs_rank_noise"):
            frame.loc[heldout, column] = np.arange(heldout.sum()) * 1e6
        frame.loc[heldout, "target_rank"] = -1e9
        frame.loc[heldout, "forward_return"] = 1e9
        changed, features, metadata = fit_entry_model(frame, [2023, 2025], self.cfg,
                                                     self.directory / "changed_holdout", [2024, 2026])
        self.assertEqual(features, self.features)
        self.assertEqual(metadata["fitted_sample_sha256"], self.metadata["fitted_sample_sha256"])
        np.testing.assert_array_equal(changed.predict(self.panel[features]), self.model.predict(self.panel[features]))
        first_audit = pd.read_csv(self.directory / "original/feature_selection_audit.csv")
        second_audit = pd.read_csv(self.directory / "changed_holdout/feature_selection_audit.csv")
        pd.testing.assert_frame_equal(first_audit, second_audit)

    def test_correlated_factors_reduced_and_used_manifest_matches_native_model(self):
        self.assertEqual(self.features, ["cs_rank_alpha", "cs_rank_noise"])
        manifest = pd.read_csv(self.directory / "original/used_feature_manifest.csv")
        self.assertEqual(manifest.feature.tolist(), self.features)
        self.assertTrue(manifest.selected.all())
        metadata = json.loads((self.directory / "original/model_metadata.json").read_text())
        self.assertEqual(metadata["train_years"], [2023, 2025])
        self.assertEqual(metadata["excluded_years"], [2024, 2026])
        self.assertEqual(self.model.get_booster().feature_names, self.features)

    def test_inference_requires_no_future_labels_and_preserves_unlabeled_rows(self):
        reference = predict_entry_scores(self.model, self.features, self.panel, [2026], self.cfg)
        unlabeled = self.panel.drop(columns=["target_rank", "forward_return", "label_end"])
        scores = predict_entry_scores(self.model, self.features, unlabeled, [2026], self.cfg)
        pd.testing.assert_frame_equal(reference, scores)
        self.assertEqual(len(scores), int((self.panel.datetime.dt.year == 2026).sum()))
        missing_features = unlabeled.copy()
        missing_features.loc[missing_features.datetime.dt.year == 2026, self.features] = np.nan
        missing_scores = predict_entry_scores(self.model, self.features, missing_features, [2026], self.cfg)
        self.assertEqual(len(missing_scores), len(scores))
        self.assertTrue(np.isfinite(missing_scores.score).all())

    def test_joblib_and_native_json_reload_produce_identical_predictions(self):
        loaded = joblib.load(self.directory / "original/model.joblib")
        native = XGBRegressor()
        native.load_model(self.directory / "original/model.json")
        expected = self.model.predict(self.panel[self.features])
        np.testing.assert_array_equal(loaded.predict(self.panel[self.features]), expected)
        np.testing.assert_array_equal(native.predict(self.panel[self.features]), expected)

    def test_gates_are_frozen_on_training_scores_and_keep_top_five_structure(self):
        training = select_model_training_rows(self.panel, [2023, 2025], self.cfg, [2024, 2026])
        scores = predict_entry_scores(self.model, self.features, training, [2023, 2025], self.cfg)
        gate = calibrate_entry_gate(scores, self.cfg)
        snapshot = copy.deepcopy(gate)
        test_scores = predict_entry_scores(self.model, self.features, self.panel, [2026], self.cfg)
        candidates = select_entry_candidates(test_scores, gate, self.cfg)
        self.assertGreater(len(candidates), 0)
        self.assertTrue((candidates.score >= gate["threshold"]).all())
        self.assertLessEqual(candidates.groupby("datetime").size().max(), 5)
        self.assertIsNone(gate["spread_threshold"])
        self.assertEqual(gate, snapshot)
        spread_gate = calibrate_entry_gate(scores, {**self.cfg, "spread_quantile": 0.7})
        self.assertTrue(np.isfinite(spread_gate["spread_threshold"]))

    def test_invalid_fold_feature_gate_and_prediction_config_fails(self):
        for kwargs in ({"years": [2030]}, {"years": [2026], "embargo_days": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_year_folds(self.panel, **kwargs)
        with self.assertRaises(ValueError):
            build_year_folds(self.panel.loc[self.panel.datetime.dt.year < 2025], [2024])
        with self.assertRaises(ValueError):
            select_model_training_rows(self.panel, [2023], self.cfg, [2023])
        training = select_model_training_rows(self.panel, [2023], self.cfg, [2024])
        for override in ({"max_features": 0}, {"feature_selection_sample_rows": 1},
                         {"feature_min_coverage": -0.1}, {"feature_max_correlation": 0}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                select_training_features(training, {**self.cfg, **override})
        scores = predict_entry_scores(self.model, self.features, self.panel, [2026], self.cfg)
        for override in ({"gate_quantile": 1.1}, {"spread_quantile": -0.1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                calibrate_entry_gate(scores, {**self.cfg, **override})
        with self.assertRaises(ValueError):
            predict_entry_scores(self.model, self.features, self.panel, [2026], {**self.cfg, "entry_method": "unknown"})
        with self.assertRaises(ValueError):
            predict_entry_scores(self.model, self.features, self.panel.drop(columns=self.features[0]), [2026], self.cfg)

    def test_feature_generation_does_not_change_when_future_bars_change(self):
        index = pd.date_range("2025-01-02 09:31", periods=800, freq="min")
        close = 10 + np.arange(800) * 0.001 + np.sin(np.arange(800) / 15) * 0.1
        frame = pd.DataFrame({"datetime": index, "symbol": "SYNTHETIC", "open": close - 0.01,
                              "high": close + 0.03, "low": close - 0.03, "close": close,
                              "volume": 1000 + np.arange(800) % 43,
                              "amount": close * (1000 + np.arange(800) % 43) * 100})
        outcome = validate_no_future_leakage(frame)
        self.assertTrue(outcome["passed"], outcome)
        self.assertGreater(outcome["checked_feature_count"], 150)

    def test_training_and_prediction_do_not_mutate_input_panel_or_configuration(self):
        predict_entry_scores(self.model, self.features, self.panel, [2023, 2026], self.cfg)
        pd.testing.assert_frame_equal(self.panel, self.original)
        self.assertEqual(self.cfg, self.original_cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
