"""v14报告口径测试：使用合成已保存账户，不训练、不运行或覆盖历史账户。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_backtest import inventory_t_learning_reporting as reporting
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics
from test_dynamic_t_reporting import dynamic_config, dynamic_fixture


def learning_config(**changes):
    cfg = dynamic_config()
    cfg.update(research_version=reporting.VERSION)
    cfg.update(changes)
    return cfg


def learning_fixture(hold=False, **changes):
    result = dynamic_fixture(hold=hold, **changes)
    result["selected_rule"].update(profile="hold" if hold else "l2_depth4",
                                   min_net_edge=0., fraction=0. if hold else .5)
    return result


class InventoryLearningReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_sources = {name: (ROOT/name).read_bytes() for name in
                           ("quant_backtest/balanced_t_reporting.py", "quant_backtest/dynamic_t_reporting.py")}

    @classmethod
    def tearDownClass(cls):
        for name, content in cls.old_sources.items():
            assert (ROOT/name).read_bytes() == content, name

    def write(self, result, cfg=None, directory=None):
        if directory is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            directory = Path(temporary.name)
        status = reporting.write_inventory_learning_report(directory, cfg or learning_config(), [(2024, result)])
        return directory, status

    def test_original_version_metrics_costs_and_inputs_preserved(self):
        result, cfg = learning_fixture(net=2000.), learning_config()
        original, original_cfg = deepcopy(result), deepcopy(cfg)
        expected, _ = calculate_balanced_metrics(result, cfg)
        original_writer = reporting.write_balanced_report
        versions = []
        def observed_writer(directory, actual_cfg, entries, validation_table):
            versions.append(actual_cfg["research_version"])
            return original_writer(directory, actual_cfg, entries, validation_table)
        with patch.object(reporting, "write_balanced_report", observed_writer):
            directory, status = self.write(result, cfg)
        self.assertEqual(versions, [reporting.VERSION])
        self.assertEqual(cfg, original_cfg)
        for key in ("all_reference_acquisition_cost", "holding_cost_reduction_rate", "incremental_marked_pnl",
                    "t_fee", "t_slippage_cost", "completed_t_pair_lots", "total_t_traded_lots",
                    "goal_achieved", "frequency_target_status"):
            self.assertEqual(status["annual_results"][0][key], expected[key])
        for key, value in original.items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(result[key], value, check_exact=True)
            elif isinstance(value, pd.Index):
                pd.testing.assert_index_equal(result[key], value, exact=True)
            else:
                self.assertEqual(result[key], value)
        for filename in ("annual_summary.csv", "年度.csv", "test_trade_events.csv", "test_round_trips.csv",
                         "test_equity_curve.csv", "test_open_positions.csv", "daily_inventory_audit.csv",
                         "baseline_trade_events.csv", "per_stock_cost_reduction.csv", "t_pair_lot_summary.csv",
                         "daily_frequency_coverage.csv", "goal_evaluation.json", "exit_decisions.csv",
                         "dynamic_exit_summary.csv", "selected_learning_rules.csv", "inventory_learning_summary.json"):
            self.assertTrue((directory/filename).is_file(), filename)

    def test_split_pair_is_one_exit_and_six_lots_not_two_groups(self):
        directory, status = self.write(learning_fixture(fragments=(100, 500)))
        row = status["annual_results"][0]
        self.assertEqual((row["complete_t_pairs"], row["recovery_fragments"], row["completed_t_pair_lots"]), (1, 2, 6.))
        self.assertEqual(pd.read_csv(directory/"dynamic_exit_summary.csv").iloc[0].latched_exit_pairs, 1)
        meta = json.loads((directory/"inventory_learning_summary.json").read_text())
        self.assertEqual(meta["candidate_count"], 37)
        self.assertEqual(meta["prediction_horizons_minutes"], [20, 40, 60])
        self.assertFalse(meta["target_is_portfolio_execution_net_pnl"])

    def test_hold_smoke_has_no_profit_win_claim_and_correct_title(self):
        directory, status = self.write(learning_fixture(hold=True), learning_config(report_scope="smoke"))
        self.assertFalse(status["goal_achieved"])
        self.assertEqual(len(pd.read_csv(directory/"exit_decisions.csv")), 0)
        text = (directory/"final_report.md").read_text()
        self.assertIn("永久底仓训练择时（v14）", text.splitlines()[0])
        self.assertIn("工程烟测", text.splitlines()[0])
        self.assertIn("没有可评价的做T胜率", text)
        self.assertNotIn("v13恢复协议", text)
        self.assertNotIn("163候选", text)
        self.assertIn("不是共享现金账户的真实组合利润", text)
        self.assertEqual((directory/"test_return_curve.png").read_bytes(), (directory/"测试集收益曲线.png").read_bytes())

    def test_pending_and_negative_pnl_remain_in_goal_and_report(self):
        directory, status = self.write(learning_fixture(pending=True, net=-50., reason="stop_loss"))
        row = status["annual_results"][0]
        self.assertFalse(row["profit_target_met"])
        self.assertFalse(status["goal_achieved"])
        self.assertEqual((row["inventory_mismatch_days"], row["pending_pairs"], row["complete_t_pairs"]), (1, 1, 0))
        self.assertEqual(row["incremental_marked_pnl"], -50.)
        self.assertEqual(pd.read_csv(directory/"dynamic_exit_summary.csv").iloc[0].stop_loss_trigger_pairs, 1)

    def test_saved_exit_csv_and_exact_selected_rule_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = learning_fixture(reason="hard_deadline")
            result["selected_rule"].update(profile="l2_depth6", min_net_edge=.003, fraction=1.)
            folder = directory/"year_2024"
            folder.mkdir()
            path = folder/"exit_decisions.csv"
            result["exit_decisions"].to_csv(path, index=False, encoding="utf-8")
            original = path.read_bytes()
            self.write(result, directory=directory)
            self.assertEqual(path.read_bytes(), original)
            selected = pd.read_csv(directory/"selected_learning_rules.csv").iloc[0]
            self.assertEqual((selected.profile, selected.min_net_edge, selected.fraction, selected.exit_mode),
                             ("l2_depth6", .003, 1., "fixed"))

    def test_wrong_version_candidate_or_exit_timing_rejected(self):
        for failure in ("version", "profile", "margin", "fraction", "missing_enabled", "time", "duplicate"):
            result, cfg = learning_fixture(), learning_config()
            if failure == "version": cfg["research_version"] = "dynamic_inventory_t_v13"
            elif failure == "profile": result["selected_rule"]["profile"] = "huber"
            elif failure == "margin": result["selected_rule"]["min_net_edge"] = .0005
            elif failure == "fraction": result["selected_rule"]["fraction"] = True
            elif failure == "missing_enabled": del result["selected_rule"]["enabled"]
            elif failure == "time":
                result["exit_decisions"].loc[0, "exit_earliest_time"] = result["exit_decisions"].loc[0, "exit_signal_time"]
            else:
                result["exit_decisions"] = pd.concat([result["exit_decisions"]]*2, ignore_index=True)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.write(result, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
