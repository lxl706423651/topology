"""动态退出候选首腿因果性与配置隔离；小型当前信号，不读取任何真实外层数据。"""
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.balanced_t_policy import build_balanced_candidates, generate_balanced_decisions
from quant_backtest.dynamic_t_policy import (build_dynamic_candidates, generate_dynamic_decisions,
                                            create_dynamic_rule_config)


class DynamicPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"t_avoid_locked_exit_side": True, "flat_st_limit_rate": .05,
            "nonflat_limit_proxy": .098, "last_entry_time": "14:30", "fixed_recovery_time": "14:55"}
        self.frame = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-06 10:00", "2025-01-06 11:00"]),
            "symbol": ["SH.600000"] * 2, "signal_price": [10., 10.], "signal_high": [10.1, 10.1],
            "signal_low": [9.9, 9.9], "previous_close": [10., 10.], "signal_volume": [1e5, 1e5],
            "signal_amount": [1e6, 1e6], "f_vwap_deviation": [-.012, .012],
            "f_return_5m": [.002, .002], "f_return_15m": [-.004, .004], "f_open_return": [-.01, .01]})

    def test_every_exit_mode_preserves_original_first_leg_and_deadline(self):
        rules = build_dynamic_candidates(self.cfg)
        self.assertEqual(len(rules), 163)
        for rule in rules:
            original = build_balanced_candidates(self.cfg)[rule["entry_candidate_index"]]
            expected, _ = generate_balanced_decisions(self.frame, original, self.cfg)
            actual, _ = generate_dynamic_decisions(self.frame, rule, self.cfg)
            pd.testing.assert_frame_equal(actual.drop(columns="candidate_index"),
                                          expected.drop(columns="candidate_index"), check_exact=True)
            self.assertNotIn("exit_signal_time", actual)

    def test_future_path_fields_cannot_preselect_first_legs(self):
        rule = build_dynamic_candidates(self.cfg)[3]
        actual, _ = generate_dynamic_decisions(self.frame, rule, self.cfg)
        changed = self.frame.assign(future_high=999., target_net_return=-999., recovery_price=0.)
        repeated, _ = generate_dynamic_decisions(changed, rule, self.cfg)
        pd.testing.assert_frame_equal(actual, repeated, check_exact=True)
        with self.assertRaises(ValueError):
            generate_dynamic_decisions(self.frame, {**rule, "take_net_return": .0001}, self.cfg)

    def test_single_account_config_does_not_mutate_shared_rule_or_fee(self):
        cfg = {**self.cfg, "commission_rate": .0008}
        rule = build_dynamic_candidates(cfg)[3]
        local = create_dynamic_rule_config(cfg, rule)
        self.assertEqual(local["dynamic_exit_mode"], "profit_buffer_stop")
        self.assertEqual(local["commission_rate"], cfg["commission_rate"])
        self.assertNotIn("dynamic_exit_mode", cfg)
        self.assertNotIn("dynamic_exit_mode", rule)


if __name__ == "__main__":
    unittest.main(verbosity=2)
