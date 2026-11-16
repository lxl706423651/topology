"""v13包装报告测试；合成账户验证原分母/目标与退出组计数，不运行策略账户。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics
from quant_backtest.dynamic_t_reporting import EXIT_COLUMNS, VERSION, write_dynamic_report
from test_balanced_t_reporting import config, fixture


def dynamic_config(**changes):
    return config(research_version=VERSION, dynamic_take_net_return=.003,
                  dynamic_stop_net_return=.01, **changes)


def dynamic_fixture(hold=False, pending=False, fragments=(100,), reason="profit_buffer", net=200.):
    """每实际pair只构造一个锁存退出信号；恢复片段可独立变化。"""
    result = fixture(hold=hold, pending=pending, fragments=fragments, net=net)
    rows = []
    if not hold:
        first = result["trade_events"].loc[lambda x: x.event_type.eq("t") & x.fragment_id.eq(0)].iloc[0]
        mode = "fixed" if reason == "hard_deadline" else "profit_buffer_stop"
        deadline = first.datetime + pd.Timedelta(minutes=20)
        signal = deadline - pd.Timedelta(minutes=1 if mode == "fixed" else 2)
        rows.append(dict(zip(EXIT_COLUMNS, [
            1, "A", "buy_first", mode, first.datetime-pd.Timedelta(minutes=1), first.datetime,
            deadline, signal, signal+pd.Timedelta(minutes=1), reason, 10.1, 3., .003,
            float(first.reference_price*first.shares), int(first.shares), 0.
        ])))
    result["exit_decisions"] = pd.DataFrame(rows, columns=EXIT_COLUMNS)
    result["selected_rule"] = {"enabled": not hold, "exit_mode": "fixed" if hold else mode}
    return result


class DynamicReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_source = (ROOT / "quant_backtest/balanced_t_reporting.py").read_bytes()

    @classmethod
    def tearDownClass(cls):
        assert (ROOT / "quant_backtest/balanced_t_reporting.py").read_bytes() == cls.old_source

    def write(self, result, cfg=None, directory=None):
        if directory is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            directory = Path(temporary.name)
        status = write_dynamic_report(directory, cfg or dynamic_config(), [(2024, result)])
        return directory, status

    def test_canonical_metrics_costs_goals_and_inputs_unchanged(self):
        result = dynamic_fixture(net=2000.)
        original = deepcopy(result)
        expected, _ = calculate_balanced_metrics(result, dynamic_config())
        directory, status = self.write(result)
        for key in ("all_reference_acquisition_cost", "holding_cost_reduction_rate", "incremental_marked_pnl",
                    "completed_t_pair_lots", "total_t_traded_lots", "goal_achieved", "frequency_target_status"):
            self.assertEqual(status["annual_results"][0][key], expected[key])
        self.assertFalse(status["goal_achieved"])
        for key, value in original.items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(result[key], value, check_exact=True)
        for filename in ("annual_summary.csv", "年度.csv", "test_trade_events.csv", "test_round_trips.csv",
                         "test_equity_curve.csv", "test_open_positions.csv", "daily_inventory_audit.csv",
                         "baseline_trade_events.csv", "per_stock_cost_reduction.csv", "t_pair_lot_summary.csv",
                         "daily_frequency_coverage.csv", "goal_evaluation.json", "exit_decisions.csv"):
            self.assertTrue((directory / filename).is_file(), filename)

    def test_split_pair_is_one_trigger_not_two_fragments(self):
        directory, status = self.write(dynamic_fixture(fragments=(100,500)))
        row = status["annual_results"][0]
        self.assertEqual(row["recovery_fragments"], 2)
        self.assertEqual(row["complete_t_pairs"], 1)
        self.assertEqual(row["completed_t_pair_lots"], 6.)
        summary = pd.read_csv(directory / "dynamic_exit_summary.csv").iloc[0]
        self.assertEqual(summary.latched_exit_pairs, 1)
        self.assertEqual(summary.profit_buffer_trigger_pairs, 1)
        self.assertEqual(summary.early_trigger_pairs, 1)

    def test_pending_loss_and_inventory_failure_remain_visible(self):
        directory, status = self.write(dynamic_fixture(pending=True, net=-50., reason="stop_loss"))
        row = status["annual_results"][0]
        self.assertFalse(row["profit_target_met"])
        self.assertEqual(row["inventory_mismatch_days"], 1)
        self.assertEqual(row["pending_pairs"], 1)
        self.assertEqual(row["complete_t_pairs"], 0)
        summary = pd.read_csv(directory / "dynamic_exit_summary.csv").iloc[0]
        self.assertEqual(summary.stop_loss_trigger_pairs, 1)

    def test_hold_empty_exit_table_and_smoke_do_not_claim_profit(self):
        directory, status = self.write(dynamic_fixture(hold=True), dynamic_config(report_scope="smoke"))
        self.assertFalse(status["goal_achieved"])
        self.assertEqual(len(pd.read_csv(directory / "exit_decisions.csv")), 0)
        report = (directory / "final_report.md").read_text()
        self.assertIn("v13", report.splitlines()[0])
        self.assertIn("工程烟测", report.splitlines()[0])
        self.assertIn("无成交时没有可评价", report)
        self.assertIn("v12旧稀疏", report)
        self.assertEqual((directory / "test_return_curve.png").read_bytes(),
                         (directory / "测试集收益曲线.png").read_bytes())

    def test_existing_year_exit_csv_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = dynamic_fixture(reason="hard_deadline")
            folder = directory / "year_2024"
            folder.mkdir()
            path = folder / "exit_decisions.csv"
            result["exit_decisions"].to_csv(path, index=False, encoding="utf-8")
            previous = path.read_bytes()
            _, status = self.write(result, directory=directory)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(status["dynamic_exit"]["annual_exit_summaries"][0]["hard_deadline_trigger_pairs"], 1)
            self.assertEqual(status["dynamic_exit"]["annual_exit_summaries"][0]["early_trigger_pairs"], 0)

    def test_duplicate_unknown_pair_and_noncausal_time_rejected(self):
        for failure in ("duplicate", "unknown_pair", "same_bar_execution", "future_first", "bad_notional"):
            result = dynamic_fixture()
            table = result["exit_decisions"]
            if failure == "duplicate":
                result["exit_decisions"] = pd.concat([table, table], ignore_index=True)
            elif failure == "unknown_pair":
                table.loc[0, "pair_id"] = 999
            elif failure == "same_bar_execution":
                table.loc[0, "exit_earliest_time"] = table.loc[0, "exit_signal_time"]
            elif failure == "future_first":
                table.loc[0, "first_execution_time"] = table.loc[0, "exit_signal_time"] + pd.Timedelta(minutes=1)
            else:
                table.loc[0, "normalization_notional"] = 0.
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.write(result)

    def test_missing_schema_wrong_version_and_invalid_threshold_rejected(self):
        result = dynamic_fixture()
        del result["exit_decisions"]
        with self.assertRaises(ValueError):
            self.write(result)
        for key, value in (("research_version", "v12"), ("dynamic_take_net_return", float('nan')),
                           ("dynamic_stop_net_return", True)):
            cfg = dynamic_config()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.write(dynamic_fixture(), cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
