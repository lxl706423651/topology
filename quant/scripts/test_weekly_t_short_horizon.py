"""验证短周期T的当前信号因果性、精确成本、核心训练支持域和内部选参门槛。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_horizon_data import build_symbol_horizon_data
from quant_backtest.weekly_t_short_horizon import (
    ShortHorizonTStrategy, attach_horizon_labels, candidate_passes_support,
    candidate_support_statistics, create_short_decisions, feature_support_mask,
    fit_feature_support, planned_recovery_times, predicted_pair_edges,
)


def short_cfg() -> dict:
    """提供显式固定成本、训练支持域和候选门槛，测试不依赖运行中YAML。"""
    return {"commission_rate": .0008, "transfer_rate": .00001, "slippage_rate": .0005,
            "min_commission": 0., "stamp_rate_before_20230828": .001,
            "stamp_rate_from_20230828": .0005, "short_t_fraction": .25,
            "latest_planned_recovery_time": "14:30", "t_avoid_locked_exit_side": True,
            "flat_st_limit_rate": .05, "nonflat_limit_proxy": .098,
            "t_support_features": ["f_input"], "t_support_quantiles": [.05, .95],
            "random_state": 17, "bootstrap_samples": 1000,
            "candidate_support_requirements": {"min_complete_pairs": 4,
                "min_active_days": 4, "min_active_weeks": 4, "min_active_symbols": 4,
                "max_day_pair_fraction": .5, "max_week_pair_fraction": .5,
                "max_day_abs_pnl_fraction": .5, "max_week_abs_pnl_fraction": .5}}


def predictions_at(clocks: list[str], day: str = "2026-04-29") -> pd.DataFrame:
    """生成合法当前行情和独立预测，未来执行列显式给定但不应用于首腿门控。"""
    times = pd.to_datetime([day + " " + clock for clock in clocks])
    return pd.DataFrame({"datetime": times, "symbol": "SH.600000", "prediction": .01,
                         "f_input": 0., "signal_price": 10., "signal_high": 10.01,
                         "signal_low": 9.99, "previous_close": 10.,
                         "execution_price": 10., "recovery_price": 10.1,
                         "target_return": .01, "label_available": True})


def supported_result() -> dict:
    """四周四完整pair，各pair拆两片段；逐周净增量均为10元。"""
    days = pd.to_datetime(["2026-03-03", "2026-03-10", "2026-03-17", "2026-03-24"])
    events, trips = [], []
    for pair_id, day in enumerate(days, 1):
        symbol = f"SH.60000{pair_id}"
        first = day + pd.Timedelta(hours=9, minutes=36)
        events.append({"datetime": first, "event_type": "t", "pair_id": pair_id, "symbol": symbol})
        for offset, pnl in ((20, 4.), (21, 6.)):
            events.append({"datetime": first + pd.Timedelta(minutes=offset), "event_type": "t", "pair_id": pair_id, "symbol": symbol})
            trips.append({"first_datetime": first, "second_datetime": first + pd.Timedelta(minutes=offset),
                          "pair_id": pair_id, "symbol": symbol, "net_pnl": pnl, "pair_fully_recovered": True})
    events.append({"datetime": days[0], "event_type": "weekly", "pair_id": 0, "symbol": "BASE"})
    return {"trade_events": pd.DataFrame(events), "round_trips": pd.DataFrame(trips),
            "equity_curve": pd.DataFrame({"datetime": days, "baseline_equity": 1000., "equity": [1010., 1020., 1030., 1040.]}),
            "metrics": {"t_pair_count": 4, "t_fragment_count": 8, "pending_pairs": 0,
                        "inventory_mismatch_days": 0, "recovery_failure_days": 0,
                        "base_plan_mismatch_count": 0, "t1_violation_count": 0}}


class WeeklyTShortHorizonTests(unittest.TestCase):
    """对会改变经济结论或年度隔离的门控行为进行独立核对。"""

    def setUp(self):
        """每例使用独立配置和冻结支持域，不共享可变输入。"""
        self.cfg = short_cfg()
        self.support = {"bounds": {"f_input": {"lower": -1., "upper": 1.}}}
        self.rule = {"horizon_minutes": 20, "net_margin": .001, "enabled": True}

    def test_future_prices_labels_and_availability_never_change_decisions(self):
        original = predictions_at(["09:35", "09:45"])
        frozen = original.copy(deep=True)
        expected = create_short_decisions(original, self.rule, self.cfg, self.support)
        for value in (np.nan, -999999., 999999.):
            changed = original.copy(deep=True)
            changed[["execution_price", "recovery_price", "target_return"]] = value
            changed["label_available"] = False
            changed["label_end"] = pd.NaT
            changed["recovery_time"] = pd.NaT
            changed["execution_time"] = pd.NaT
            actual = create_short_decisions(changed, self.rule, self.cfg, self.support)
            pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(original, frozen)
        self.assertEqual(len(expected), 1)

    def test_missing_label_attachment_keeps_current_features_and_clock_only_plan(self):
        current = predictions_at(["09:35", "11:00", "14:00"])
        current["f_second"] = [1., 2., 3.]
        raw = pd.DataFrame({"datetime": current.datetime, "open": 10., "high": 10.01,
                            "low": 9.99, "close": 10., "volume": 1000., "amount": 10000.})
        labels, _ = build_symbol_horizon_data(raw, "SH.600000", [], {})
        attached = attach_horizon_labels(current, labels.iloc[:0], 20)
        self.assertEqual(len(attached), len(current))
        self.assertTrue(attached.target_return.isna().all())
        self.assertFalse(attached.label_available.any())
        pd.testing.assert_frame_equal(attached[["f_input", "f_second"]], current[["f_input", "f_second"]])
        self.assertEqual(attached.planned_recovery_time.tolist(),
                         pd.to_datetime(["2026-04-29 09:56", "2026-04-29 11:21", "2026-04-29 14:21"]).tolist())
        expected = create_short_decisions(current, self.rule, self.cfg, self.support)
        pd.testing.assert_frame_equal(create_short_decisions(attached, self.rule, self.cfg, self.support), expected)

    def test_cost_edges_match_exact_cashflows_on_both_sides_and_tax_boundary(self):
        predictions = pd.DataFrame({"datetime": pd.to_datetime(["2023-08-27 15:00", "2023-08-28 09:35"]),
                                    "prediction": [.0123, -.0145]})
        buy, sell = predicted_pair_edges(predictions, self.cfg)
        for index, row in predictions.iterrows():
            first, later, shares = 13.21, 13.21 * (1 + row.prediction), 300
            tax = .001 if index == 0 else .0005
            fee = self.cfg["commission_rate"] + self.cfg["transfer_rate"]
            slip = self.cfg["slippage_rate"]
            cash_bought = first * (1 + slip) * shares * (1 + fee)
            cash_sold = later * (1 - slip) * shares * (1 - fee - tax)
            self.assertAlmostEqual(buy.iloc[index], (cash_sold - cash_bought) / (first * shares), places=14)
            cash_sold_first = first * (1 - slip) * shares * (1 - fee - tax)
            cash_bought_later = later * (1 + slip) * shares * (1 + fee)
            self.assertAlmostEqual(sell.iloc[index], (cash_sold_first - cash_bought_later) / (first * shares), places=14)
        with self.assertRaisesRegex(ValueError, "min_commission"):
            predicted_pair_edges(predictions, {**self.cfg, "min_commission": 5})

    def test_clock_lunch_deadline_inclusive_and_no_cross_day(self):
        current = predictions_at(["11:00", "13:05", "13:30", "14:00"])
        times = planned_recovery_times(current.datetime, 60)
        self.assertEqual(times.iloc[0], pd.Timestamp("2026-04-29 13:31"))
        self.assertEqual(times.iloc[1], pd.Timestamp("2026-04-29 14:06"))
        self.assertEqual(times.iloc[2], pd.Timestamp("2026-04-29 14:31"))
        self.assertTrue(pd.isna(times.iloc[3]))
        actions = ShortHorizonTStrategy({**self.rule, "horizon_minutes": 60}, self.cfg, self.support).generate_signals(current)
        self.assertEqual(actions.t_action.tolist(), [1, 1, 0, 0])
        edge = predictions_at(["14:09", "14:10"])
        self.assertEqual(ShortHorizonTStrategy(self.rule, self.cfg, self.support).generate_signals(edge).t_action.tolist(), [1, 0])

    def test_empty_predictions_return_empty_datetime_plan_and_decisions(self):
        empty = predictions_at([])
        times = planned_recovery_times(empty.datetime, 20)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(times.dtype))
        decisions = create_short_decisions(empty, self.rule, self.cfg, self.support)
        self.assertTrue(decisions.empty)
        self.assertIn("planned_recovery_time", decisions)

    def test_support_is_fit_only_on_supplied_core_not_changed_by_validation_outliers(self):
        core = predictions_at(["09:35", "09:45", "10:00"], "2023-04-12")
        core["f_input"] = [-1., 0., 1.]
        original = core.copy(deep=True)
        support = fit_feature_support(core, self.cfg)
        validation = predictions_at(["09:35"], "2024-04-12")
        validation["f_input"] = 1000.
        self.assertFalse(feature_support_mask(validation, support).iloc[0])
        validation["f_input"] = -1000.
        self.assertFalse(feature_support_mask(validation, support).iloc[0])
        self.assertEqual(support, fit_feature_support(core, self.cfg))
        self.assertEqual(support["training_rows"], 3)
        self.assertAlmostEqual(support["bounds"]["f_input"]["lower"], -.9)
        self.assertAlmostEqual(support["bounds"]["f_input"]["upper"], .9)
        self.assertEqual(len(support["training_inputs_sha256"]), 64)
        pd.testing.assert_frame_equal(core, original)
        for column in ("target_return", "recovery_price", "f_missing"):
            with self.subTest(column=column), self.assertRaises(ValueError):
                fit_feature_support(core, {**self.cfg, "t_support_features": [column]})

    def test_first_valid_signal_after_all_current_gates_and_index_alignment(self):
        current = predictions_at(["10:00", "09:35", "09:45"])
        current.index = [50, 12, 97]
        current.loc[12, "f_input"] = 2.
        current.loc[50, "prediction"] = .03
        actual = create_short_decisions(current, self.rule, self.cfg, self.support)
        self.assertEqual(actual.datetime.tolist(), [pd.Timestamp("2026-04-29 09:45")])
        self.assertEqual(actual.action.tolist(), [1])
        self.assertEqual(actual.fraction.tolist(), [.25])
        current.loc[97, "prediction"] = -.02
        actual = create_short_decisions(current, self.rule, self.cfg, self.support)
        self.assertEqual(actual.action.tolist(), [-1])
        disabled = create_short_decisions(current, {**self.rule, "enabled": False}, self.cfg, self.support)
        self.assertTrue(disabled.empty)

    def test_only_current_locked_recovery_side_can_reject_request(self):
        current = predictions_at(["09:35", "09:45"])
        current.loc[0, ["signal_price", "signal_high", "signal_low"]] = 9.
        current.loc[1, "prediction"] = -.01
        current.loc[1, ["signal_price", "signal_high", "signal_low"]] = 11.
        actions = ShortHorizonTStrategy(self.rule, self.cfg, self.support).generate_signals(current)
        self.assertEqual(actions.t_action.tolist(), [0, 0])
        current[["signal_price", "signal_high", "signal_low"]] = [10., 10.01, 9.99]
        self.assertEqual(ShortHorizonTStrategy(self.rule, self.cfg, self.support).generate_signals(current).t_action.tolist(), [1, -1])

    def test_pair_support_never_counts_recovery_fragments_as_pairs(self):
        result = supported_result()
        original = deepcopy(result)
        support = candidate_support_statistics(result, self.cfg)
        self.assertEqual(support["initiated_pairs"], 4)
        self.assertEqual(support["active_days"], 4)
        self.assertEqual(support["active_weeks"], 4)
        self.assertEqual(support["active_symbols"], 4)
        for key in ("max_day_pair_fraction", "max_week_pair_fraction", "max_day_abs_pnl_fraction", "max_week_abs_pnl_fraction"):
            self.assertEqual(support[key], .25)
        self.assertEqual(support["weekly_increment_ci_low_rmb"], 40.)
        self.assertEqual(support["weekly_increment_ci_high_rmb"], 40.)
        evaluation = {"metrics": result["metrics"], "support": support, "incremental_net_pnl": 40.}
        self.assertEqual(candidate_passes_support(evaluation, self.cfg), (True, []))
        insufficient = deepcopy(evaluation)
        insufficient["metrics"]["t_pair_count"] = 3
        insufficient["metrics"]["t_fragment_count"] = 100
        passed, reasons = candidate_passes_support(insufficient, self.cfg)
        self.assertFalse(passed)
        self.assertIn("insufficient_complete_pairs", reasons)
        for name in ("trade_events", "round_trips", "equity_curve"):
            pd.testing.assert_frame_equal(result[name], original[name])

    def test_concentration_ci_and_execution_failures_all_block_selection(self):
        result = supported_result()
        support = candidate_support_statistics(result, self.cfg)
        evaluation = {"metrics": result["metrics"], "support": support, "incremental_net_pnl": 40.}
        for key in ("inventory_mismatch_days", "recovery_failure_days", "pending_pairs", "base_plan_mismatch_count", "t1_violation_count"):
            changed = deepcopy(evaluation)
            changed["metrics"][key] = 1
            passed, reasons = candidate_passes_support(changed, self.cfg)
            self.assertFalse(passed)
            self.assertIn(key, reasons)
        for key in ("max_day_pair_fraction", "max_week_pair_fraction", "max_day_abs_pnl_fraction", "max_week_abs_pnl_fraction"):
            changed = deepcopy(evaluation)
            changed["support"][key] = .51
            passed, reasons = candidate_passes_support(changed, self.cfg)
            self.assertFalse(passed)
            self.assertIn(key, reasons)
        changed = deepcopy(evaluation)
        changed["support"]["weekly_increment_ci_low_rmb"] = 0.
        self.assertIn("nonpositive_weekly_bootstrap_lower", candidate_passes_support(changed, self.cfg)[1])
        changed = deepcopy(evaluation)
        changed["incremental_net_pnl"] = 0.
        self.assertIn("nonpositive_net_pnl", candidate_passes_support(changed, self.cfg)[1])

    def test_nonfinite_profit_ci_or_support_cannot_pass_hard_gate(self):
        result = supported_result()
        support = candidate_support_statistics(result, self.cfg)
        evaluation = {"metrics": result["metrics"], "support": support, "incremental_net_pnl": 40.}
        for key in ("incremental_net_pnl", "weekly_increment_ci_low_rmb", "active_days", "max_day_pair_fraction"):
            for value in (np.nan, np.inf, -np.inf):
                changed = deepcopy(evaluation)
                destination = changed if key == "incremental_net_pnl" else changed["support"]
                destination[key] = value
                with self.subTest(key=key, value=value):
                    self.assertFalse(candidate_passes_support(changed, self.cfg)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
