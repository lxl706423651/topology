"""验证新等金额做T的因果信号/内部选择，使用小型确定数据，不读写真实行情。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_backtest.balanced_t_policy import (
    build_balanced_candidates, generate_balanced_decisions, select_balanced_candidate,
)


class BalancedTPolicyTests(unittest.TestCase):
    """覆盖未来字段独立性、买卖方向确认、午休时钟和实际净利选择。"""

    def setUp(self):
        self.cfg = {"t_avoid_locked_exit_side": True, "flat_st_limit_rate": .05,
                    "nonflat_limit_proxy": .098, "last_entry_time": "14:00", "fixed_recovery_time": "14:55"}
        self.frame = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-02-04 10:30", "2025-02-04 11:00", "2025-02-04 14:00"]),
            "symbol": ["SH.600000"] * 3, "signal_price": [10.] * 3,
            "signal_high": [10.1] * 3, "signal_low": [9.9] * 3, "previous_close": [10.] * 3,
            "signal_amount": [1e6] * 3, "signal_volume": [1e5] * 3,
            "f_vwap_deviation": [-.012, .012, .012], "f_return_5m": [.003, .002, -.002],
            "f_return_15m": [-.004, .004, -.004], "f_open_return": [-.01, .01, .01],
            "execution_price": [500., 200., 300.], "target_return": [.9, -.9, .8]})

    def get_rule(self, family="vwap_reversal", horizon=20, fraction=.5):
        return next(r for r in build_balanced_candidates(self.cfg) if r["family"] == family and
                    r["entry_deviation"] == .003 and r["recovery_horizon"] == horizon and r["fraction"] == fraction)

    def test_predeclared_full_grid_and_hold(self):
        rules = build_balanced_candidates(self.cfg)
        self.assertEqual(len(rules), 55)
        empty, audit = generate_balanced_decisions(self.frame, rules[0], self.cfg)
        self.assertTrue(empty.empty)
        self.assertEqual(audit["request_rows"], 0)
        self.assertEqual(list(empty.columns)[:4], ["datetime", "symbol", "action", "fraction"])

    def test_reversal_confirmation_excludes_continuing_direction(self):
        result, _ = generate_balanced_decisions(self.frame, self.get_rule("confirmed_reversal"), self.cfg)
        self.assertEqual(result.action.tolist(), [1, -1])
        self.assertEqual(result.datetime.dt.strftime("%H:%M").tolist(), ["10:30", "14:00"])

    def test_momentum_requires_all_current_directions(self):
        result, _ = generate_balanced_decisions(self.frame, self.get_rule("momentum"), self.cfg)
        self.assertEqual(result.action.tolist(), [1])
        self.assertEqual(result.datetime.iloc[0], pd.Timestamp("2025-02-04 11:00"))

    def test_future_endpoints_do_not_change_actions_or_size(self):
        rule = self.get_rule()
        original, _ = generate_balanced_decisions(self.frame, rule, self.cfg)
        changed = self.frame.copy()
        changed[["execution_price", "target_return"]] = np.nan
        changed["recovery_price"] = -999.
        actual, _ = generate_balanced_decisions(changed, rule, self.cfg)
        pd.testing.assert_frame_equal(original, actual, check_exact=True)
        removed, _ = generate_balanced_decisions(changed.drop(columns=["execution_price", "target_return", "recovery_price"]), rule, self.cfg)
        pd.testing.assert_frame_equal(original, removed, check_exact=True)

    def test_lunch_and_close_deadline_are_fixed_before_future_quotes(self):
        result, _ = generate_balanced_decisions(self.frame, self.get_rule(horizon=60), self.cfg)
        self.assertEqual(result.planned_recovery_time.dt.strftime("%H:%M").tolist(), ["13:01", "13:31"])
        close, _ = generate_balanced_decisions(self.frame, self.get_rule(horizon="close"), self.cfg)
        self.assertEqual(close.planned_recovery_time.dt.strftime("%H:%M").tolist(), ["14:55"] * 3)

    def test_nontrading_or_locked_current_bar_is_not_an_opportunity(self):
        frame = self.frame.copy()
        frame.loc[0, "signal_volume"] = 0
        frame.loc[1, ["signal_price", "signal_high", "signal_low"]] = 11.
        frame.loc[2, "f_vwap_deviation"] = np.nan
        result, _ = generate_balanced_decisions(frame, self.get_rule(), self.cfg)
        self.assertTrue(result.empty)

    def test_reject_changed_rule_and_duplicate_keys(self):
        rule = self.get_rule()
        with self.assertRaises(ValueError):
            generate_balanced_decisions(self.frame, {**rule, "fraction": .2}, self.cfg)
        with self.assertRaises(ValueError):
            generate_balanced_decisions(pd.concat([self.frame, self.frame.iloc[:1]]), rule, self.cfg)

    def test_inner_selection_uses_actual_net_not_count_and_rejects_failures(self):
        rows = pd.DataFrame({"candidate_index": [0, 1, 2, 3], "incremental_net_pnl": [0., -10., -5., 1000.],
                             "t_pair_count": [0, 500, 30, 100], "inventory_mismatch_days": [0, 0, 0, 1],
                             "recovery_failure_days": [0] * 4, "pending_pairs": [0] * 4,
                             "t1_violation_count": [0] * 4, "base_plan_mismatch_count": [0] * 4})
        selected, checked = select_balanced_candidate(rows)
        self.assertEqual(selected["candidate_index"], 2)
        self.assertEqual(selected["incremental_net_pnl"], -5.)
        self.assertFalse(checked.loc[3, "execution_eligible"])
        rows.loc[1:, "t_pair_count"] = 0
        selected, _ = select_balanced_candidate(rows)
        self.assertEqual(selected["candidate_index"], 0)
        rows.loc[0, "incremental_net_pnl"] = 1.
        with self.assertRaises(ValueError):
            select_balanced_candidate(rows)


if __name__ == "__main__":
    unittest.main()
