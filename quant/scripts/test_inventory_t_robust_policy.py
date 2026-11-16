"""Huber37候选与原L2择时规则的对应反例；仅构造当前分数，不训练或运行账户。"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.test_inventory_t_robust_models import make_config
from quant_backtest import inventory_t_robust_policy as policy
from quant_backtest.inventory_t_robust_models import VERSION, SOURCE_PROFILE_MAP
from quant_backtest.inventory_t_learning_policy import build_inventory_learning_candidates, generate_inventory_learning_decisions


def make_predictions(profile="huber_depth4", scores=(.01, .012, .025)):
    """三期限完整同键当前分数；附未来字段供删除/污染反例，不包含实际结果。"""
    frames = {}
    for horizon, score in zip((20, 40, 60), scores):
        frames[horizon] = pd.DataFrame({"datetime": pd.to_datetime(["2025-03-03 09:35", "2025-03-03 11:00", "2025-03-03 14:00"]),
            "symbol": ["SH.600001", "SZ.000001", "SH.600001"], "prediction": score,
            "raw_prediction": score, "return_score": score, "model_profile": profile,
            "horizon_minutes": horizon, "model_version": VERSION, "forecast_loss": "huber", "forecast_alpha": .01,
            "signal_amount": 1e6, "signal_volume": 1e5, "signal_price": 10., "previous_close": 10.,
            "signal_high": 10.1, "signal_low": 9.9, "execution_price": [10., np.nan, 8.],
            "recovery_price": [3., 18., np.nan], "target_return": [np.nan, 999., -999.]})
    return frames


class RobustPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.rules = policy.build_inventory_robust_candidates(self.cfg)

    def test_grid_is_numeric_v15_with_explicit_actual_huber_profile_map(self):
        original_cfg = {**self.cfg, "learning_model_profiles": ["l2_depth4", "l2_depth6"]}
        original = build_inventory_learning_candidates(original_cfg)
        self.assertEqual(len(self.rules), 37)
        for current, old in zip(self.rules, original):
            mapped = {**current, "profile": SOURCE_PROFILE_MAP.get(current["profile"], "hold")}
            self.assertEqual(mapped, old)
        self.assertEqual({rule["profile"] for rule in self.rules}, {"hold", "huber_depth4", "huber_depth6"})

    def test_same_current_scores_reproduce_every_original_numeric_entry_rule(self):
        original_cfg = {**self.cfg, "learning_model_profiles": ["l2_depth4", "l2_depth6"]}
        originals = build_inventory_learning_candidates(original_cfg)
        for rule, old in zip(self.rules, originals):
            profile = rule["profile"] if rule["enabled"] else "huber_depth4"
            forecasts = make_predictions(profile)
            source = {h: p.assign(model_profile=SOURCE_PROFILE_MAP[profile]) for h, p in forecasts.items()}
            expected, _ = generate_inventory_learning_decisions(source, old, original_cfg)
            actual, audit = policy.generate_inventory_robust_decisions(forecasts, rule, self.cfg)
            if len(expected):
                expected["model_profile"] = profile
                pd.testing.assert_frame_equal(actual[list(expected)], expected, check_exact=True)
            else:
                self.assertTrue(actual.empty)
            self.assertFalse(audit["score_is_calibrated_conditional_mean"])

    def test_cost_formula_rejects_zero_score_and_original_sell_edge_exact(self):
        empty, _ = policy.generate_inventory_robust_decisions(make_predictions(scores=(0., 0., 0.)), self.rules[1], self.cfg)
        self.assertTrue(empty.empty)
        result, _ = policy.generate_inventory_robust_decisions(make_predictions(scores=(-.02, -.02, -.02)), self.rules[1], self.cfg)
        self.assertTrue(result.action.eq(-1).all())
        edge = (1-.0005)*(1-.00081-.0005)-.98*(1+.0005)*(1+.00081)
        np.testing.assert_allclose(result.priority, edge, rtol=0, atol=1e-14)

    def test_legal_lunch_late_horizon_and_stable_shortest_tie(self):
        actual, _ = policy.generate_inventory_robust_decisions(make_predictions(), self.rules[1], self.cfg)
        self.assertEqual(actual.horizon_minutes.tolist(), [60, 60, 40])
        self.assertEqual(actual.planned_recovery_time.dt.strftime("%H:%M").tolist(), ["10:36", "13:31", "14:41"])
        tied, _ = policy.generate_inventory_robust_decisions(make_predictions(scores=(.02, .02, .02)), self.rules[1], self.cfg)
        self.assertTrue(tied.horizon_minutes.eq(20).all())
        self.assertTrue(tied.action.eq(1).all())

    def test_future_erasure_and_row_reordering_do_not_change_decisions_or_inputs(self):
        frames = make_predictions()
        before = copy.deepcopy(frames)
        expected, _ = policy.generate_inventory_robust_decisions(frames, self.rules[1], self.cfg)
        changed = {h: p.drop(columns=[name for name in p if name.startswith(("execution_", "recovery_", "target_", "label_"))]).iloc[::-1].copy()
                   for h, p in frames.items()}
        actual, _ = policy.generate_inventory_robust_decisions(changed, self.rules[1], self.cfg)
        pd.testing.assert_frame_equal(actual, expected)
        for h in frames: pd.testing.assert_frame_equal(frames[h], before[h])

    def test_partial_horizon_keyset_and_wrong_actual_loss_version_are_rejected(self):
        frames = make_predictions()
        frames[60] = frames[60].iloc[1:]
        with self.assertRaisesRegex(ValueError, "当前键必须完全相同"):
            policy.generate_inventory_robust_decisions(frames, self.rules[1], self.cfg)
        for name, value in (("forecast_loss", "regression"), ("forecast_alpha", .02),
                            ("model_version", "inventory_learning_t_v14"), ("model_profile", "l2_depth4"),
                            ("return_score", .9)):
            frames = make_predictions()
            frames[40][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                policy.generate_inventory_robust_decisions(frames, self.rules[1], self.cfg)

    def test_current_reverse_price_limit_and_zero_volume_remain_original_guards(self):
        frames = make_predictions()
        for p in frames.values():
            p.loc[0, ["signal_price", "signal_high", "signal_low"]] = 9.5
            p.loc[1, "signal_volume"] = 0
        result, _ = policy.generate_inventory_robust_decisions(frames, self.rules[1], self.cfg)
        self.assertEqual(result.datetime.dt.strftime("%H:%M").tolist(), ["14:00"])

    def test_unknown_signal_clock_and_nonfinite_score_cannot_create_request(self):
        for name, value in (("datetime", pd.Timestamp("2025-03-03 09:36")), ("datetime", pd.Timestamp("2025-03-03 09:35:01")),
                            ("prediction", np.nan), ("prediction", -1.)):
            frames = make_predictions()
            frames[20].loc[0, name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                policy.generate_inventory_robust_decisions(frames, self.rules[1], self.cfg)

    def test_hold_and_exit_modes_keep_same_requests_without_silent_rule_edit(self):
        hold, _ = policy.generate_inventory_robust_decisions({}, self.rules[0], self.cfg)
        self.assertTrue(hold.empty)
        frames = [policy.generate_inventory_robust_decisions(make_predictions(), rule, self.cfg)[0] for rule in self.rules[1:4]]
        for actual in frames[1:]:
            pd.testing.assert_frame_equal(actual.drop(columns="candidate_index"), frames[0].drop(columns="candidate_index"))
        original = copy.deepcopy(self.cfg)
        changed = policy.create_inventory_robust_rule_config(self.cfg, self.rules[2])
        self.assertEqual(changed["dynamic_exit_mode"], "profit_buffer")
        self.assertEqual(self.cfg, original)
        with self.assertRaises(ValueError):
            policy.create_inventory_robust_rule_config(self.cfg, {**self.rules[1], "fraction": .7})

    def test_loss_fee_clock_configuration_is_fixed(self):
        for name, value in (("training_loss", "regression"), ("t_huber_alpha", .02), ("min_commission", 5.),
                            ("t_recheck_exit_side_at_execution", False), ("last_entry_time", "14:00"),
                            ("horizons", [20, 60]), ("learning_model_profiles", ["l2_depth4", "l2_depth6"])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                policy.build_inventory_robust_candidates({**self.cfg, name: value})


if __name__ == "__main__": unittest.main(verbosity=2)
