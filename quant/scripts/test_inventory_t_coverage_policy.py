"""覆盖率版本包装的37规则与原L2逐值一致；不训练模型或撮合账户。"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.test_inventory_t_coverage_models import make_config
from quant_backtest import inventory_t_coverage_policy as policy
from quant_backtest.inventory_t_coverage_models import VERSION
from quant_backtest.inventory_t_learning_policy import build_inventory_learning_candidates, generate_inventory_learning_decisions


def make_predictions(profile="l2_depth4", scores=(.01, .012, .025)):
    """仅构造三个同键当前分数；未来列故意缺失或不合理以证明不参与门控。"""
    return {h: pd.DataFrame({"datetime": pd.to_datetime(["2025-03-03 09:35", "2025-03-03 11:00", "2025-03-03 14:00"]),
        "symbol": ["SH.600001", "SZ.000001", "SH.600001"], "prediction": score, "raw_prediction": score,
        "return_score": score, "model_profile": profile, "horizon_minutes": h, "model_version": VERSION,
        "forecast_loss": "regression_l2", "feature_min_coverage": .7,
        "signal_amount": 1e6, "signal_volume": 1e5, "signal_price": 10., "previous_close": 10.,
        "signal_high": 10.1, "signal_low": 9.9, "execution_price": [np.nan, 10., -8.],
        "recovery_price": [np.nan, 18., -3.], "target_return": [np.nan, 999., -999.]})
        for h, score in zip((20, 40, 60), scores)}


class CoveragePolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.rules = policy.build_inventory_coverage_candidates(self.cfg)

    def test_full37_grid_and_rule_configs_are_original_numeric_l2(self):
        self.assertEqual(len(self.rules), 37)
        self.assertEqual(self.rules, build_inventory_learning_candidates(self.cfg))
        self.assertEqual({r["profile"] for r in self.rules}, {"hold", "l2_depth4", "l2_depth6"})
        self.assertEqual(sum(r["enabled"] for r in self.rules), 36)
        original = copy.deepcopy(self.cfg)
        for rule in self.rules:
            result = policy.create_inventory_coverage_rule_config(self.cfg, rule)
            self.assertEqual({k: v for k, v in result.items() if k != "dynamic_exit_mode"},
                             {k: v for k, v in self.cfg.items() if k != "dynamic_exit_mode"})
            self.assertEqual(result["dynamic_exit_mode"], rule["exit_mode"])
        self.assertEqual(original, self.cfg)

    def test_same_current_scores_reproduce_every_original_candidate_exactly(self):
        for rule in self.rules:
            frames = make_predictions(rule["profile"]) if rule["enabled"] else {}
            expected, old_audit = generate_inventory_learning_decisions(frames, rule, self.cfg)
            actual, audit = policy.generate_inventory_coverage_decisions(frames, rule, self.cfg)
            pd.testing.assert_frame_equal(actual[list(expected)], expected, check_exact=True)
            self.assertEqual({k: audit[k] for k in old_audit}, old_audit)
            self.assertFalse(audit["score_is_calibrated_conditional_mean"])
            self.assertTrue(audit["same_numeric_l2_policy"])
            self.assertTrue(actual.model_version.eq(VERSION).all())

    def test_future_erasure_and_row_reordering_preserve_decisions_and_input(self):
        frames = make_predictions()
        before = copy.deepcopy(frames)
        expected, _ = policy.generate_inventory_coverage_decisions(frames, self.rules[1], self.cfg)
        changed = {h: p.drop(columns=[c for c in p if c.startswith(("execution_", "recovery_", "target_", "label_"))]).iloc[::-1].copy()
                   for h, p in frames.items()}
        actual, _ = policy.generate_inventory_coverage_decisions(changed, self.rules[1], self.cfg)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        for h in frames:
            pd.testing.assert_frame_equal(frames[h], before[h], check_exact=True)

    def test_original_cost_direction_and_nonpositive_edge_remain(self):
        empty, _ = policy.generate_inventory_coverage_decisions(make_predictions(scores=(0., 0., 0.)), self.rules[1], self.cfg)
        self.assertTrue(empty.empty)
        actual, _ = policy.generate_inventory_coverage_decisions(make_predictions(scores=(-.02, -.02, -.02)), self.rules[1], self.cfg)
        self.assertTrue(actual.action.eq(-1).all())
        edge = (1-.0005)*(1-.00081-.0005)-.98*(1+.0005)*(1+.00081)
        np.testing.assert_allclose(actual.priority, edge, rtol=0, atol=1e-14)

    def test_original_lunch_horizon_deadline_and_shorter_tie_order(self):
        actual, _ = policy.generate_inventory_coverage_decisions(make_predictions(), self.rules[1], self.cfg)
        self.assertEqual(actual.horizon_minutes.tolist(), [60, 60, 40])
        self.assertEqual(actual.planned_recovery_time.dt.strftime("%H:%M").tolist(), ["10:36", "13:31", "14:41"])
        tied, _ = policy.generate_inventory_coverage_decisions(make_predictions(scores=(.02, .02, .02)), self.rules[1], self.cfg)
        self.assertTrue(tied.horizon_minutes.eq(20).all())

    def test_current_price_limit_and_zero_volume_are_still_required(self):
        frames = make_predictions()
        for frame in frames.values():
            frame.loc[0, ["signal_price", "signal_high", "signal_low"]] = 9.5
            frame.loc[1, "signal_volume"] = 0
        result, _ = policy.generate_inventory_coverage_decisions(frames, self.rules[1], self.cfg)
        self.assertEqual(result.datetime.dt.strftime("%H:%M").tolist(), ["14:00"])

    def test_mixed_loss_version_coverage_score_or_profile_is_rejected(self):
        for name, value in (("model_version", "inventory_learning_t_v14"), ("forecast_loss", "huber"),
                            ("feature_min_coverage", .8), ("return_score", .9), ("forecast_alpha", .01),
                            ("model_profile", "l2_depth6"), ("horizon_minutes", 60)):
            frames = make_predictions()
            frames[40][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                policy.generate_inventory_coverage_decisions(frames, self.rules[1], self.cfg)

    def test_future_label_intersection_cannot_hide_missing_horizon_current_keys(self):
        frames = make_predictions()
        frames[60] = frames[60].iloc[1:]
        with self.assertRaisesRegex(ValueError, "当前键必须完全相同"):
            policy.generate_inventory_coverage_decisions(frames, self.rules[1], self.cfg)
        with self.assertRaises(ValueError):
            policy.generate_inventory_coverage_decisions({20: frames[20]}, self.rules[1], self.cfg)

    def test_invalid_clocks_nonfinite_scores_and_changed_rules_are_rejected(self):
        for name, value in (("datetime", pd.Timestamp("2025-03-03 09:36")), ("datetime", pd.Timestamp("2025-03-03 09:35:01")),
                            ("prediction", np.nan), ("prediction", -1.)):
            frames = make_predictions()
            frames[20].loc[0, name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                policy.generate_inventory_coverage_decisions(frames, self.rules[1], self.cfg)
        with self.assertRaises(ValueError):
            policy.create_inventory_coverage_rule_config(self.cfg, {**self.rules[1], "fraction": .7})

    def test_coverage_loss_clock_grid_and_fee_configuration_boundaries(self):
        for name, value in (("feature_min_coverage", .8), ("source_feature_min_coverage", .7),
                            ("training_loss", "huber"), ("min_commission", 5.), ("commission_rate", np.nan),
                            ("t_recheck_exit_side_at_execution", False), ("last_entry_time", "14:00"),
                            ("horizons", [20, 60]), ("learning_model_profiles", ["huber_depth4", "huber_depth6"])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                policy.build_inventory_coverage_candidates({**self.cfg, name: value})


if __name__ == "__main__":
    unittest.main(verbosity=2)
