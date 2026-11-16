"""永久底仓模型/信号的必要因果与成本反例；仅写临时小模型，不读市场或旧报告。"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import lightgbm
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_backtest.inventory_t_learning_models import fit_inventory_return_model, predict_inventory_return_model, attach_learning_labels
from quant_backtest.inventory_t_learning_policy import build_inventory_learning_candidates, generate_inventory_learning_decisions
from quant_backtest.weekly_t_short_horizon import planned_recovery_times


def make_config() -> dict:
    """读取实际候选合同，只将合成测试的训练规模和线程缩小。"""
    cfg = yaml.safe_load((ROOT/"config/inventory_learning_t_v14.yaml").read_text())
    cfg.update(t_model_profile="l2_depth4", t_label_horizon_minutes=20, workers=1,
               min_training_rows=20, t_max_features=3, t_avoid_locked_exit_side=False,
               feature_selection_sample_rows=100, t_model={"n_estimators": 8, "min_child_samples": 5})
    return cfg


def make_training_frame() -> pd.DataFrame:
    """生成固定期限的核心年当前特征和独立标签，首10行来源触及排除年。"""
    days = pd.bdate_range("2023-03-01", periods=120)
    x = np.sin(np.arange(120)*.4)
    frame = pd.DataFrame({"datetime": days+pd.Timedelta(hours=9, minutes=35), "symbol": "SH.600001",
        "cohort_target_year": 2023, "baseline_opening_shares": 1000,
        "feature_source_start": days-pd.Timedelta(days=7), "f_signal": x,
        "f_noise": np.cos(np.arange(120)*.17), "f_history_days": np.arange(120),
        "execution_price": 10., "recovery_price": 10.*(1.+.015*x),
        "target_return": .015*x, "horizon_minutes": 20})
    frame.loc[:9, "feature_source_start"] = pd.Timestamp("2022-12-23")
    frame["label_start"] = frame.datetime+pd.Timedelta(minutes=1)
    frame["label_end"] = planned_recovery_times(frame.datetime, 20)
    return frame


def make_predictions(cfg: dict, returns=(.01, .01, .02)) -> dict:
    """每期限使用相同三个当前键，另附不可用于信号的未来字段。"""
    result = {}
    for horizon, value in zip((20, 40, 60), returns):
        result[horizon] = pd.DataFrame({"datetime": pd.to_datetime(["2025-03-03 09:35", "2025-03-03 09:45", "2025-03-03 14:00"]),
            "symbol": ["SH.600001"]*3, "prediction": [value]*3, "model_profile": cfg["t_model_profile"],
            "horizon_minutes": horizon, "signal_amount": 1e6, "signal_volume": 1e5,
            "execution_price": [11., np.nan, 9.], "recovery_price": [5., 18., np.nan]})
    return result


class LearningModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cfg, cls.frame = make_config(), make_training_frame()
        cls.path = Path(cls.temp.name)/"model"
        cls.metadata = fit_inventory_return_model(cls.frame, [2023], cls.cfg, cls.path, [2022, 2024])
        cls.bundle = joblib.load(cls.path/"model_bundle.joblib")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_source_purge_and_native_artifact(self):
        self.assertEqual(self.metadata["training_rows"], 110)
        self.assertNotIn("f_history_days", self.bundle["features"])
        inputs = pd.read_pickle(self.path/"training_inputs.pkl")
        self.assertTrue(pd.to_datetime(inputs.feature_source_start).dt.year.eq(2023).all())
        native = lightgbm.Booster(model_file=str(self.path/"return_model.txt"))
        matrix = inputs[self.bundle["features"]]
        np.testing.assert_array_equal(native.predict(matrix), self.bundle["model"].predict(matrix))

    def test_future_labels_do_not_change_current_predictions_or_drop_rows(self):
        one = predict_inventory_return_model(self.bundle, self.frame, self.cfg)
        changed = self.frame.copy()
        for column in ("label_start", "label_end", "target_return", "execution_price", "recovery_price"):
            changed[column] = np.nan
        two = predict_inventory_return_model(self.bundle, changed, self.cfg)
        self.assertEqual(len(two), len(self.frame))
        pd.testing.assert_series_equal(one.prediction, two.prediction)

    def test_target_price_mismatch_and_wrong_horizon_rejected(self):
        bad = self.frame.copy()
        bad.loc[20, "target_return"] = .4
        with self.assertRaisesRegex(ValueError, "收益标签"):
            fit_inventory_return_model(bad, [2023], self.cfg, Path(self.temp.name)/"bad_price", [2022, 2024])
        bad = self.frame.copy()
        bad["horizon_minutes"] = 40
        with self.assertRaisesRegex(ValueError, "固定期限"):
            fit_inventory_return_model(bad, [2023], self.cfg, Path(self.temp.name)/"bad_horizon", [2022, 2024])

    def test_no_old_inventory_or_holdout_cohort_is_rejected(self):
        for field, value in (("baseline_opening_shares", 0), ("cohort_target_year", 2024)):
            bad = self.frame.copy()
            bad.loc[20, field] = value
            with self.assertRaisesRegex(ValueError, "核心旧股"):
                fit_inventory_return_model(bad, [2023], self.cfg, Path(self.temp.name)/field, [2024])

    def test_labels_attach_after_current_cohort_and_missing_endpoints_keep_keys(self):
        current = self.frame.drop(columns=[name for name in self.frame
            if name.startswith(("execution_", "recovery_", "target_", "label_"))])
        labels = self.frame.copy()
        labels["recovery_time"], labels["execution_time"] = labels.label_end, labels.label_start
        labels = labels.iloc[:-1]
        attached = attach_learning_labels(current, labels, 20)
        self.assertEqual(len(attached), len(current))
        self.assertTrue(pd.isna(attached.execution_price.iloc[-1]))
        self.assertFalse(attached.label_available.iloc[-1])
        pd.testing.assert_series_equal(attached.f_signal, current.f_signal)
        self.assertTrue(attached.execution_price.iloc[:-1].eq(10.).all())
        with self.assertRaisesRegex(ValueError, "去除全部未来"):
            attach_learning_labels(self.frame, labels, 20)


class LearningPolicyTest(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.rules = build_inventory_learning_candidates(self.cfg)
        self.rule = self.rules[1]

    def test_costs_reject_flat_expectation_and_choose_sell(self):
        flat = make_predictions(self.cfg, (0., 0., 0.))
        result, _ = generate_inventory_learning_decisions(flat, self.rule, self.cfg)
        self.assertTrue(result.empty)
        down = make_predictions(self.cfg, (-.02, -.02, -.02))
        result, _ = generate_inventory_learning_decisions(down, self.rule, self.cfg)
        self.assertTrue(result.action.eq(-1).all())
        expected = (1-.0005)*(1-.00081-.0005)-.98*(1+.0005)*(1+.00081)
        np.testing.assert_allclose(result.priority, expected, rtol=0, atol=1e-14)

    def test_current_best_horizon_and_late_clock(self):
        result, _ = generate_inventory_learning_decisions(make_predictions(self.cfg), self.rule, self.cfg)
        self.assertEqual(result.horizon_minutes.tolist(), [60, 60, 20])
        self.assertTrue(result.planned_recovery_time.gt(result.datetime).all())
        self.assertTrue((result.planned_recovery_time.dt.hour*60+result.planned_recovery_time.dt.minute).le(14*60+55).all())

    def test_future_endpoint_changes_and_frame_order_do_not_change_decisions(self):
        original = make_predictions(self.cfg)
        expected, _ = generate_inventory_learning_decisions(original, self.rule, self.cfg)
        changed = copy.deepcopy(original)
        for horizon, frame in changed.items():
            frame[["execution_price", "recovery_price"]] = np.nan
            changed[horizon] = frame.iloc[::-1].copy()
        actual, _ = generate_inventory_learning_decisions(changed, self.rule, self.cfg)
        pd.testing.assert_frame_equal(expected, actual)

    def test_missing_current_key_in_one_horizon_rejected(self):
        changed = make_predictions(self.cfg)
        changed[60] = changed[60].iloc[1:]
        with self.assertRaisesRegex(ValueError, "当前键必须完全相同"):
            generate_inventory_learning_decisions(changed, self.rule, self.cfg)

    def test_all_exit_modes_keep_same_entry_requests(self):
        decisions = [generate_inventory_learning_decisions(make_predictions(self.cfg), rule, self.cfg)[0]
                     for rule in self.rules[1:4]]
        for frame in decisions[1:]:
            pd.testing.assert_frame_equal(decisions[0].drop(columns="candidate_index"), frame.drop(columns="candidate_index"))
        self.assertEqual(len(self.rules), 37)


if __name__ == "__main__":
    unittest.main()
