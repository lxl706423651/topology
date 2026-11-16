"""固定尾盘策略的成本、当前时序和未来信息反例；不训练或读取回测账户。"""
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

from quant_backtest.inventory_t_close_labels import IDENTITY, SIGNAL_TIMES
from quant_backtest.inventory_t_close_policy import (build_inventory_close_candidates,
    create_inventory_close_rule_config, generate_inventory_close_decisions)


def config() -> dict:
    """单独给出事前合同；费用期望在测试中另按逐腿现金计算。"""
    return {"research_version": "inventory_close_t_v16", "forecast_target": "same_day_1455",
        "recovery_key": "close", "signal_times": list(SIGNAL_TIMES), "fixed_recovery_time": "14:55",
        "last_entry_time": "14:30", "t_recheck_exit_side_at_execution": True,
        "learning_model_profiles": ["l2_depth4", "l2_depth6"], "learning_min_net_edges": [0., .001, .003],
        "t_trade_fractions": [.5, 1.], "dynamic_exit_modes": ["fixed", "profit_buffer", "profit_buffer_stop"],
        "dynamic_take_net_return": .003, "dynamic_stop_net_return": .01,
        "commission_rate": .0008, "transfer_rate": .00001, "slippage_rate": .0005,
        "stamp_rate_before_20230828": .001, "stamp_rate_from_20230828": .0005, "min_commission": 0.,
        "t_avoid_locked_exit_side": True, "flat_st_limit_rate": .05, "nonflat_limit_proxy": .098}


def predictions() -> pd.DataFrame:
    """不同买卖方向及尾盘信号，额外未来列只用于扰动检验。"""
    frame = pd.DataFrame({"datetime": pd.to_datetime(["2026-03-02 09:35", "2026-03-02 09:45", "2026-03-02 14:00"]),
        "symbol": ["SH.600001", "SH.600002", "SH.600003"], "prediction": [.01, -.01, .02],
        "model_profile": "l2_depth4", "signal_price": 10., "signal_high": 10.01, "signal_low": 9.99,
        "previous_close": 10., "signal_amount": 1e6, "signal_volume": 1e5,
        "execution_time": pd.to_datetime(["2026-03-02 09:36", "2026-03-02 09:46", "2026-03-02 14:01"]),
        "execution_price": [10., np.nan, 10.], "recovery_price": [100., 1., np.nan],
        "recovery_time": pd.to_datetime(["2026-03-02 14:55"]*3), "label_available": [True, False, False],
        "target_return": [9., np.nan, -.9]})
    for key, value in IDENTITY.items():
        frame[key] = value
    frame.index = [7, 3, 12]
    return frame


class ClosePolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.frame = config(), predictions()
        self.rules = build_inventory_close_candidates(self.cfg)

    def test_predeclared_37_grid_and_copied_config(self):
        self.assertEqual(len(self.rules), 37)
        self.assertEqual([r["candidate_index"] for r in self.rules], list(range(37)))
        original = copy.deepcopy(self.cfg)
        result = create_inventory_close_rule_config(self.cfg, self.rules[3])
        self.assertEqual(result["dynamic_exit_mode"], "profit_buffer_stop")
        self.assertEqual(self.cfg, original)
        self.assertIsNot(result, self.cfg)
        for patch in ({"fraction": .25}, {"candidate_index": True}, {"profile": "l2_depth6"}):
            with self.assertRaises(ValueError):
                create_inventory_close_rule_config(self.cfg, {**self.rules[1], **patch})

    def test_single_close_identity_current_times_and_directions(self):
        result, audit = generate_inventory_close_decisions(self.frame, self.rules[1], self.cfg)
        self.assertEqual(result.action.tolist(), [1, -1, 1])
        self.assertTrue(result.planned_recovery_time.eq(pd.Timestamp("2026-03-02 14:55")).all())
        self.assertNotIn("horizon_minutes", result)
        for key, value in IDENTITY.items():
            self.assertTrue(result[key].eq(value).all())
        self.assertFalse(audit["cross_horizon_argmax"])
        self.assertFalse(audit["prediction_is_execution_profit"])
        self.assertEqual(audit["request_rows"], 3)
        with self.assertRaises(ValueError):
            generate_inventory_close_decisions({20: self.frame}, self.rules[1], self.cfg)

    def test_future_price_time_label_changes_and_missing_values_have_no_effect(self):
        original = self.frame.copy(deep=True)
        expected, expected_audit = generate_inventory_close_decisions(self.frame, self.rules[1], self.cfg)
        changed = self.frame.copy(deep=True)
        for column in ("execution_time", "execution_price", "recovery_price", "recovery_time", "target_return", "label_available"):
            changed[column] = np.nan
        changed["execution_amount"], changed["recovery_volume"] = 0., 0.
        changed["planned_recovery_time"] = pd.Timestamp("2099-01-01")
        actual, audit = generate_inventory_close_decisions(changed, self.rules[1], self.cfg)
        pd.testing.assert_frame_equal(expected, actual)
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(audit, expected_audit)

    def test_profit_edges_equal_independent_two_leg_cash_and_stamp_date(self):
        fee = .0008 + .00001
        for day, stamp in (("2023-08-25", .001), ("2023-08-28", .0005)):
            frame = self.frame.copy()
            frame["datetime"] = pd.Timestamp(day)+pd.to_timedelta(["09:35:00", "09:45:00", "14:00:00"])
            result, _ = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
            reference, shares, slip = 10., 100, .0005
            for row in result.itertuples(index=False):
                if row.action == 1:
                    paid = reference*shares*(1+slip)*(1+fee)
                    received = reference*(1+row.prediction)*shares*(1-slip)*(1-fee-stamp)
                else:
                    received = reference*shares*(1-slip)*(1-fee-stamp)
                    paid = reference*(1+row.prediction)*shares*(1+slip)*(1+fee)
                self.assertAlmostEqual(row.predicted_net_edge, (received-paid)/(reference*shares), places=14)

    def test_zero_and_weak_positive_forecast_do_not_cover_cost(self):
        frame = self.frame.copy()
        frame["prediction"] = [0., .001, -.001]
        result, audit = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        self.assertTrue(result.empty)
        self.assertEqual(audit["positive_net_predictions"], 0)
        frame["prediction"] = .004
        low, _ = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        high_rule = next(rule for rule in self.rules if rule["profile"] == "l2_depth4"
            and rule["min_net_edge"] == .003 and rule["fraction"] == .5 and rule["exit_mode"] == "fixed")
        high, _ = generate_inventory_close_decisions(frame, high_rule, self.cfg)
        self.assertEqual(len(low), 3)
        self.assertTrue(high.empty)

    def test_wrong_target_profile_and_legacy_horizon_rejected(self):
        for column, value in (("model_profile", "l2_depth6"), ("recovery_key", "20"),
                              ("recovery_kind", "trading_minutes"), ("recovery_clock", "15:00"), ("horizon_minutes", 20)):
            frame = self.frame.copy()
            frame[column] = value
            with self.assertRaises(ValueError):
                generate_inventory_close_decisions(frame, self.rules[1], self.cfg)

    def test_wrong_configuration_and_costs_rejected(self):
        changes = ({"forecast_target": "raw_return"}, {"recovery_key": "20"}, {"horizons": [20, 40, 60]},
            {"t_label_horizon_minutes": 20}, {"fixed_recovery_time": "15:00"}, {"last_entry_time": "14:45"},
            {"t_recheck_exit_side_at_execution": False}, {"min_commission": 5.},
            {"commission_rate": np.nan}, {"slippage_rate": -.01}, {"transfer_rate": True},
            {"stamp_rate_from_20230828": np.inf}, {"learning_min_net_edges": [0., .001, .002]})
        for patch in changes:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                build_inventory_close_candidates({**self.cfg, **patch})

    def test_illegal_current_times_and_duplicate_keys_rejected(self):
        for value in ("2026-03-02 12:00", "2026-03-02 13:00", "2026-03-02 14:30", "2026-03-02 09:35:01", None):
            frame = self.frame.copy()
            frame.loc[7, "datetime"] = pd.Timestamp(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        frame = self.frame.copy()
        frame["datetime"] = frame.datetime.dt.tz_localize("Asia/Shanghai")
        with self.assertRaises(ValueError):
            generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        with self.assertRaises(ValueError):
            generate_inventory_close_decisions(pd.concat([self.frame, self.frame.iloc[:1]]), self.rules[1], self.cfg)
        duplicate_column = pd.concat([self.frame, self.frame[["prediction"]]], axis=1)
        with self.assertRaises(ValueError):
            generate_inventory_close_decisions(duplicate_column, self.rules[1], self.cfg)

    def test_invalid_prediction_and_current_quote_fail_closed(self):
        for value in (np.nan, np.inf, -1., -1.1):
            frame = self.frame.copy()
            frame.loc[7, "prediction"] = value
            with self.assertRaises(ValueError):
                generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        for column, value in (("signal_amount", 0.), ("signal_volume", 0.), ("signal_price", np.nan),
                              ("signal_high", 9.), ("signal_low", 11.), ("previous_close", 0.)):
            frame = self.frame.copy()
            frame.loc[7, column] = value
            result, _ = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
            self.assertNotIn("SH.600001", result.symbol.tolist())

    def test_current_reverse_limit_proxy_retained(self):
        for price, raw_prediction in ((9., .02), (11., -.02), (9.5, .02), (10.5, -.02)):
            frame = self.frame.iloc[:1].copy()
            frame["signal_price"], frame["signal_high"], frame["signal_low"] = price, price, price
            frame["prediction"] = raw_prediction
            result, _ = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
            self.assertTrue(result.empty)
        frame = self.frame.iloc[:1].copy()
        frame["signal_price"], frame["signal_high"], frame["signal_low"] = 10.6, 10.6, 10.6
        frame["prediction"] = -.02
        result, _ = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        self.assertEqual(len(result), 1)

    def test_stable_shared_cash_order_and_all_same_day_requests_preserved(self):
        frame = self.frame.copy()
        frame["datetime"] = pd.Timestamp("2026-03-02 09:35")
        frame["symbol"], frame["prediction"] = ["SH.600003", "SH.600002", "SH.600001"], [.01, .02, .01]
        result, _ = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        self.assertEqual(result.symbol.tolist(), ["SH.600002", "SH.600001", "SH.600003"])
        frame = self.frame.copy()
        frame["symbol"] = "SH.600001"
        result, _ = generate_inventory_close_decisions(frame, self.rules[1], self.cfg)
        self.assertEqual(len(result), 3)

    def test_hold_and_empty_current_table_remain_typed_empty(self):
        hold, audit = generate_inventory_close_decisions(pd.DataFrame(), self.rules[0], self.cfg)
        self.assertTrue(hold.empty)
        self.assertEqual(audit["request_rows"], 0)
        self.assertEqual(str(hold.datetime.dtype), "datetime64[ns]")
        result, audit = generate_inventory_close_decisions(self.frame.iloc[:0], self.rules[1], self.cfg)
        self.assertTrue(result.empty)
        self.assertEqual(audit["current_rows"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
