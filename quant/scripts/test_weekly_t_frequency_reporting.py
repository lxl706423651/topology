"""v10 只读报告回归：真实计数、共同分母、执行失败和旧版本对照不可混淆。"""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quant_backtest.weekly_t_active_reporting import EXECUTION_FAILURE_KEYS
from quant_backtest.weekly_t_frequency_reporting import write_frequency_report


def make_fixture(root: Path, cross_day: bool = False) -> tuple:
    """两组T、三片恢复；源v9是另一个固定账本，账户无需重跑。"""
    dates = pd.to_datetime(["2026-01-06 15:00", "2026-01-07 15:00", "2026-01-08 15:00"])
    curve = pd.DataFrame({"datetime": dates, "equity": [1e6, 1e6 - 3, 1e6 - 9],
                          "baseline_equity": [1e6] * 3})
    events = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-06 10:01", "2026-01-06 10:21", "2026-01-06 10:22",
                                     "2026-01-07 10:01", "2026-01-08 09:30" if cross_day else "2026-01-07 10:21"]),
        "symbol": ["SH.600001"] * 3 + ["SH.600002"] * 2,
        "event_type": ["t"] * 5, "fragment_id": [0, 1, 2, 0, 1], "pair_id": [1, 1, 1, 2, 2],
        "fee": [1.] * 5, "slippage_cost": [.2] * 5,
    })
    trips = pd.DataFrame({
        "pair_id": [1, 1, 2], "pair_fully_recovered": [False, True, True],
        "first_datetime": pd.to_datetime(["2026-01-06 10:01"] * 2 + ["2026-01-07 10:01"]),
        "second_datetime": events.datetime.iloc[[1, 2, 4]].to_numpy(),
        "gross_pnl": [-9., 2., 3.], "slippage_cost": [.3, .3, .4],
        "buy_fee": [.5, .5, 1.], "sell_fee": [1., 1., 1.], "net_pnl": [-10.5, .5, 1.],
    })
    audit = pd.DataFrame({"date": dates.normalize(), "inventory_difference": [0, -100, 100] if cross_day else [0] * 3,
                          "recovery_failure": [False, cross_day, False], "weekly_liquidation_failure": [False] * 3})
    metrics = {key: 0 for key in EXECUTION_FAILURE_KEYS}
    metrics.update(t_pair_count=2, weekly_liquidation_failure_days=0, weekly_unclosed_positions=0,
                   baseline_weekly_unclosed_positions=0, inventory_mismatch_days=2 if cross_day else 0,
                   recovery_failure_days=int(cross_day))
    positions = pd.DataFrame(columns=["shares", "mark_datetime", "market_value"])
    result = {"metrics": metrics, "equity_curve": curve, "trade_events": events, "round_trips": trips,
              "daily_inventory_audit": audit, "open_positions": positions,
              "order_attempts": pd.DataFrame({"reason": ["below_one_lot"]}),
              "baseline_events": pd.DataFrame({"datetime": [dates[0]], "event_type": ["weekly"]}),
              "weekly_entry_plans": pd.DataFrame({"symbol": ["SH.600001"]}),
              "base_plan_mismatches": pd.DataFrame(columns=["symbol"])}
    annual = pd.DataFrame([{"test_year": 2026, "enabled": True, "t_enabled": True,
                            "eligible_portfolio_days": 3, "eligible_inventory_stock_days": 6,
                            "active_days": 2, "active_stock_days": 2, "submitted_t_requests": 7,
                            "runtime_signal_rows": 8, "prediction_rows": 30,
                            "preexisting_statistical_reference": "keep unchanged"}])
    validation = pd.DataFrame([{"test_year": 2026, "inner_year": 2025, "candidate_index": 0, "selected": False,
                                "t_pair_count": 1, "incremental_net_pnl": 40.},
                               {"test_year": 2026, "inner_year": 2025, "candidate_index": 1, "selected": True,
                                "t_pair_count": 3, "incremental_net_pnl": -20.}])
    source = root / "v9"
    (source / "year_2026").mkdir(parents=True)
    pd.DataFrame([{"test_year": 2026, "t_pair_count": 9, "t_added_equity": 14., "t_first_leg_count": 9,
                   "t_recovery_leg_count": 9, "t_fee": 30., "inventory_mismatch_days": 0,
                   "recovery_failure_days": 0}]).to_csv(source / "annual_summary.csv", index=False)
    curve.assign(equity=1e6 + 14).to_csv(source / "year_2026" / "test_equity_curve.csv", index=False)
    cfg = {"research_version": "weekly_t_v10_frequency", "candidate_activation_policy": "frequency_execution_valid",
           "initial_cash": 1e6, "frequency_target_pairs_per_day": 3., "bootstrap_samples": 10,
           "random_state": 42, "source_active_report_dir": str(source), "commission_rate": .0008,
           "transfer_rate": .00001, "slippage_rate": .0005, "min_commission": 0.}
    output_year = root / "v10" / "year_2026"
    output_year.mkdir(parents=True)
    pd.DataFrame([{"date": date.normalize(), "symbol": symbol} for date in dates
                  for symbol in ("SH.600001", "SH.600002")]).to_csv(
        output_year / "eligible_inventory_stock_days.csv", index=False)
    return cfg, annual, [(2026, result)], validation


class FrequencyReportingTest(unittest.TestCase):
    """报告只解释已完成账本，不依照盈亏、样本量或外层状态重新选参。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg, self.annual, self.results, self.validation = make_fixture(self.root)
        self.output = self.root / "v10"

    def tearDown(self):
        self.temp.cleanup()

    def render(self, real_plot: bool = False):
        if real_plot:
            return write_frequency_report(self.output, self.cfg, self.annual, self.results, self.validation)
        with patch("quant_backtest.weekly_t_frequency_reporting.plot_active_results"):
            return write_frequency_report(self.output, self.cfg, self.annual, self.results, self.validation)

    def test_fewer_pairs_negative_profit_remains_enabled_and_reported(self):
        status = self.render(real_plot=True)
        self.assertTrue(status["all_years_t_enabled"])
        self.assertFalse(status["profitability_reference_met"])
        self.assertFalse(status["affects_activation"])
        self.assertEqual(status["selected_inner_candidates"][0]["candidate_index"], 1)
        self.assertEqual(status["complete_pairs"], 2)
        self.assertEqual(status["aggregate_increment_rmb"], -9.)
        compare = pd.read_csv(self.output / "frequency_comparison_v9.csv")
        self.assertEqual(compare.pair_count_difference.iloc[0], -7)
        self.assertEqual(compare.t_added_equity_difference.iloc[0], -23.)
        self.assertTrue(compare.directly_comparable.iloc[0])
        for filename in ("test_return_curve.png", "测试集收益曲线.png", "t_increment_curve.png"):
            self.assertGreater((self.output / filename).stat().st_size, 1000)
        report = (self.output / "final_report.md").read_text()
        self.assertIn("先按 min(完整pair/可计划日", report)
        self.assertIn("相对排名候选没有预测净收益覆盖费用的门槛", report)
        self.assertNotIn("按内部净增量降序、原候选编号升序选择", report)

    def test_fragments_not_pairs_and_no_double_slippage(self):
        self.render()
        row = pd.read_csv(self.output / "annual_summary.csv").iloc[0]
        self.assertEqual((row.t_first_leg_count, row.t_recovery_leg_count, row.t_pair_count), (2, 3, 2))
        self.assertEqual(row.realized_pair_reference_gross_pnl, -3.)
        self.assertEqual(row.realized_pair_slippage_cost, 1.)
        self.assertEqual(row.realized_pair_fees, 5.)
        self.assertEqual(row.realized_pair_net_pnl, -9.)
        self.assertEqual(row.t_added_equity, -9.)
        self.assertEqual(row.active_stock_days, 2)
        self.assertAlmostEqual(row.pairs_per_eligible_portfolio_day, 2 / 3)
        self.assertAlmostEqual(row.active_stock_day_coverage, 1 / 3)
        self.assertEqual(row.preexisting_statistical_reference, "keep unchanged")

    def test_cross_day_failure_not_hidden_by_zero_pending(self):
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg, self.annual, self.results, self.validation = make_fixture(self.root, cross_day=True)
        self.output = self.root / "v10"
        status = self.render()
        row = pd.read_csv(self.output / "annual_summary.csv").iloc[0]
        self.assertEqual(row.pending_pairs, 0)
        self.assertEqual(row.t_cross_day_recovery_pair_count, 1)
        self.assertEqual(row.inventory_mismatch_days, 2)
        self.assertEqual(row.recovery_failure_days, 1)
        self.assertEqual(row.active_days, 2)
        self.assertFalse(status["t_execution_contract_met"])
        self.assertTrue(status["all_years_t_enabled"])

    def test_changed_baseline_is_reported_not_claimed_same(self):
        path = Path(self.cfg["source_active_report_dir"]) / "year_2026" / "test_equity_curve.csv"
        frame = pd.read_csv(path)
        frame.loc[1, "baseline_equity"] += 1.
        frame.to_csv(path, index=False)
        status = self.render()
        self.assertFalse(status["all_years_v9_baseline_daily_exact_match"])
        row = pd.read_csv(self.output / "frequency_comparison_v9.csv").iloc[0]
        self.assertTrue(row.same_evaluation_dates)
        self.assertFalse(row.directly_comparable)
        self.assertEqual(row.baseline_common_max_abs_difference_rmb, 1.)

    def test_smoke_different_dates_not_directly_comparable(self):
        path = Path(self.cfg["source_active_report_dir"]) / "year_2026" / "test_equity_curve.csv"
        pd.read_csv(path).iloc[:2].to_csv(path, index=False)
        self.render()
        row = pd.read_csv(self.output / "frequency_comparison_v9.csv").iloc[0]
        self.assertFalse(row.same_evaluation_dates)
        self.assertFalse(row.directly_comparable)
        self.assertEqual(row.common_valuation_rows, 2)

    def test_all_inputs_and_source_files_unchanged(self):
        cfg, annual, result, validation = deepcopy((self.cfg, self.annual, self.results, self.validation))
        source = Path(self.cfg["source_active_report_dir"])
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.csv")}
        status = self.render()
        self.assertEqual(self.cfg, cfg)
        pd.testing.assert_frame_equal(self.annual, annual, check_exact=True)
        pd.testing.assert_frame_equal(self.validation, validation, check_exact=True)
        for key, value in result[0][1].items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(self.results[0][1][key], value, check_exact=True)
            else:
                self.assertEqual(self.results[0][1][key], value)
        self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.csv")})
        self.assertEqual(status["v9_reference_source"]["input_sha256"], hashes)
        self.assertEqual(json.loads((self.output / "validation_summary.json").read_text()), status)

    def test_missing_fractional_boolean_or_small_denominator_rejected(self):
        for bad in (np.nan, -1, True, 2.5, 1):
            with self.subTest(bad=bad):
                self.annual.loc[0, "eligible_portfolio_days"] = bad
                with self.assertRaises(ValueError):
                    self.render()
        self.annual = self.annual.drop(columns="eligible_portfolio_days")
        with self.assertRaises(ValueError):
            self.render()

    def test_active_coverage_cannot_silently_use_future_recovery_day(self):
        self.annual.loc[0, "active_days"] = 3
        with self.assertRaisesRegex(ValueError, "真实首腿"):
            self.render()

    def test_saved_prediction_or_request_count_mismatch_rejected(self):
        year = self.output / "year_2026"
        year.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"datetime": ["2026-01-06"]}).to_csv(year / "t_predictions.csv", index=False)
        with self.assertRaisesRegex(ValueError, "保存行数"):
            self.render()

    def test_nonunique_selected_candidate_rejected_without_output(self):
        self.validation.loc[:, "selected"] = True
        with self.assertRaisesRegex(ValueError, "恰有一个"):
            self.render()
        self.assertFalse((self.output / "annual_summary.csv").exists())

    def test_wrong_version_and_disabled_rule_rejected(self):
        self.cfg["research_version"] = "weekly_t_v9_active_execution_valid"
        with self.assertRaises(ValueError):
            self.render()
        self.cfg["research_version"] = "weekly_t_v10_frequency"
        self.annual.loc[0, "enabled"] = False
        with self.assertRaisesRegex(ValueError, "enabled=True"):
            self.render()

    def test_same_day_open_mark_stays_stale(self):
        self.results[0][1]["open_positions"] = pd.DataFrame({
            "shares": [100], "mark_datetime": [pd.Timestamp("2026-01-08 09:30")], "market_value": [1000.]})
        self.render()
        row = pd.read_csv(self.output / "annual_summary.csv").iloc[0]
        self.assertEqual(row.final_stale_market_value, 1000.)

    def test_actual_activity_outside_baseline_is_retained(self):
        path = self.output / "year_2026" / "eligible_inventory_stock_days.csv"
        pd.DataFrame({"date": ["2026-01-06"], "symbol": ["SH.600001"]}).to_csv(path, index=False)
        self.annual.loc[0, ["eligible_portfolio_days", "eligible_inventory_stock_days"]] = 1
        status = self.render()
        row = pd.read_csv(self.output / "annual_summary.csv").iloc[0]
        self.assertEqual(row.active_stock_days, 2)
        self.assertEqual(row.active_days, 2)
        self.assertEqual(row.active_stock_days_in_baseline_opportunities, 1)
        self.assertEqual(row.active_stock_days_outside_baseline_opportunities, 1)
        self.assertEqual(row.active_stock_day_coverage, 1.)
        self.assertEqual(status["active_stock_days_outside_baseline_opportunities"], 1)
        self.assertEqual(row.pairs_per_eligible_portfolio_day, 2.)

    def test_missing_row_opportunities_marks_coverage_unknown(self):
        (self.output / "year_2026" / "eligible_inventory_stock_days.csv").unlink()
        status = self.render()
        self.assertFalse(status["coverage_exact_available"])
        self.assertIsNone(status["active_stock_day_coverage"])
        self.assertEqual(status["active_stock_days"], 2)


if __name__ == "__main__":
    unittest.main()
