"""v17报告合同合成测试；不拟合、不读取历史账户，产物仅写临时目录。

覆盖六模型/三期限身份、实际cfg、原成本分母、guard请求、分片及失败保留。
一个两日合成执行核案例验证实际固定短期限/费用与报告衔接，不运行真实行情。
"""
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
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from quant_backtest import inventory_t_robust_reporting as reporting
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics
from test_guarded_t_reporting import guarded_config, guarded_fixture


def robust_config(**changes):
    """独立v17配置；固定loss/alpha，保留旧fixture的经济及执行字段。"""
    cfg = guarded_config()
    cfg.update(research_version=reporting.VERSION, training_loss="huber", t_huber_alpha=.01,
               horizons=[20, 40, 60], forecast_target="fixed_20_40_60_reference_return",
               frozen_t_models_reused=False, source_execution_report_dir="synthetic_v15_execution",
               source_learning_report_dir="synthetic_v14_core")
    cfg.update(changes)
    return cfg


def robust_fixture(**changes):
    """只把规则profile映射到Huber；合成账目、短期deadline保持原样。"""
    result = guarded_fixture(**changes)
    if result["selected_rule"]["enabled"]:
        result["selected_rule"]["profile"] = "huber_depth4"
    return result


class RobustReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = {name: (ROOT/name).read_bytes() for name in (
            "quant_backtest/guarded_t_reporting.py", "quant_backtest/inventory_t_close_reporting.py",
            "quant_backtest/dynamic_t_reporting.py", "quant_backtest/balanced_t_reporting.py",
            "quant_backtest/guarded_t_engine.py", "quant_backtest/inventory_t_learning_models.py")}

    @classmethod
    def tearDownClass(cls):
        for name, content in cls.frozen.items():
            assert (ROOT/name).read_bytes() == content, name

    def write(self, result, cfg=None, directory=None):
        """实际渲染规范报告/PNG，始终写测试临时目录。"""
        if directory is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            directory = Path(temporary.name)
        status = reporting.write_robust_inventory_report(directory, cfg or robust_config(), [(2024, result)])
        return directory, status

    def test_v17_actual_cfg_model_identity_plot_and_cost_contract(self):
        result, cfg = robust_fixture(net=1600., fragments=(100, 500)), robust_config()
        original, original_cfg = deepcopy(result), deepcopy(cfg)
        expected, _ = calculate_balanced_metrics(result, cfg)
        writer, seen, titles = reporting.write_balanced_report, [], []
        from matplotlib.figure import Figure
        original_title = Figure.suptitle
        def observe_writer(directory, actual, entries, table):
            self.assertIs(actual, cfg)
            seen.append(actual["research_version"])
            return writer(directory, actual, entries, table)
        def observe_title(figure, title, *args, **kwargs):
            titles.append(title)
            return original_title(figure, title, *args, **kwargs)
        with patch.object(reporting, "write_balanced_report", observe_writer), patch.object(Figure, "suptitle", observe_title):
            directory, status = self.write(result, cfg)
        self.assertEqual(seen, [reporting.VERSION])
        self.assertIn("v17", titles[-1])
        self.assertIn("Huber", titles[-1])
        self.assertNotIn("固定14:55", titles[-1])
        self.assertEqual(cfg, original_cfg)
        for name, value in original.items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(result[name], value, check_exact=True)
            elif isinstance(value, pd.Index):
                pd.testing.assert_index_equal(result[name], value, exact=True)
            else:
                self.assertEqual(result[name], value)
        actual = status["annual_results"][0]
        for name in ("all_reference_acquisition_cost", "holding_cost_reduction_rate", "t_fee", "t_slippage_cost",
                     "complete_t_pairs", "recovery_fragments", "completed_t_pair_lots", "total_t_traded_lots"):
            self.assertEqual(actual[name], expected[name])
        self.assertEqual(actual["all_reference_acquisition_cost"], 30002.)
        self.assertAlmostEqual(actual["holding_cost_reduction_rate"], 1600/30002)
        self.assertTrue(actual["profit_target_met"])
        self.assertFalse(status["goal_achieved"])
        self.assertEqual(status["frequency_target_status"], "not_quantified")
        meta = status["inventory_robust"]
        self.assertEqual((meta["model_learning_version"], meta["newly_fitted_models_per_fold"], meta["candidate_count"]), (17, 6, 37))
        self.assertEqual((meta["objective"], meta["t_huber_alpha"]), ("huber", .01))
        self.assertEqual(meta["prediction_horizons_minutes"], [20, 40, 60])
        self.assertEqual(meta["profile_source_mapping"]["huber_depth4"], "l2_depth4")
        self.assertFalse(meta["frozen_t_models_reused"])
        self.assertFalse(meta["prediction_is_conditional_mean_guaranteed"])
        self.assertFalse(meta["training_labels_winsorized_or_deleted_for_robust_loss"])
        self.assertFalse(meta["uniform_1455_forecast_target"])
        self.assertNotIn("recovery_key", meta)
        self.assertTrue(meta["cross_horizon_argmax"])
        text = (directory/"final_report.md").read_text()
        self.assertIn("v17", text.splitlines()[0])
        self.assertIn("不保证条件均值、预测更小或盈利", text)
        self.assertIn("predicted_edge只是参考评分", text)
        self.assertIn("14:55只保留原日末恢复底线", text)
        self.assertNotIn("模型学习版本14原样复用", text)
        self.assertNotIn("新拟合2个14:55", text)
        for name in ("final_report.md", "输出文件说明.md", "annual_summary.csv", "年度.csv", "test_trade_events.csv",
                     "test_round_trips.csv", "test_equity_curve.csv", "test_open_positions.csv", "daily_inventory_audit.csv",
                     "baseline_trade_events.csv", "per_stock_cost_reduction.csv", "daily_frequency_coverage.csv",
                     "first_leg_exit_side_audit.csv", "guard_summary.csv", "dynamic_exit_summary.csv", "exit_decisions.csv",
                     "selected_robust_rules.csv", "inventory_robust_summary.json", "planned_horizon_summary.csv", "goal_evaluation.json"):
            self.assertTrue((directory/name).is_file(), name)
        self.assertEqual((directory/"test_return_curve.png").read_bytes(), (directory/"测试集收益曲线.png").read_bytes())

    def test_guard_and_partial_recovery_counts(self):
        directory, status = self.write(robust_fixture(fragments=(100, 500)))
        row = status["annual_results"][0]
        self.assertEqual((row["complete_t_pairs"], row["recovery_fragments"], row["completed_t_pair_lots"]), (1, 2, 6.))
        guard = pd.read_csv(directory/"guard_summary.csv").iloc[0]
        self.assertEqual((guard.guard_audit_requests, guard.guard_cancelled_requests, guard.guard_passed_requests), (3, 1, 2))
        self.assertEqual((guard.guard_actual_first_leg_groups, guard.guard_passed_without_actual_first_leg), (1, 1))
        exits = status["inventory_robust"]["annual_exit_summaries"][0]
        self.assertEqual((exits["latched_exit_pairs"], exits["early_trigger_pairs"]), (1, 1))

    def test_three_planned_horizons_and_lunch_are_not_close_clock(self):
        first = pd.to_datetime(["2024-01-03 10:01", "2024-01-03 11:01", "2024-01-03 13:06"])
        end = pd.to_datetime(["2024-01-03 10:21", "2024-01-03 13:11", "2024-01-03 14:06"])
        result = reporting._horizon_summary(2024, pd.DataFrame({"first_execution_time": first, "original_deadline": end}))
        self.assertEqual((result["planned_h20_pairs"], result["planned_h40_pairs"], result["planned_h60_pairs"]), (1, 1, 1))
        self.assertEqual(result["latched_pairs_with_checked_deadline"], 3)

    def test_illegal_deadline_is_not_silently_labeled_a_valid_horizon(self):
        for start, end in (("10:01", "10:31"), ("11:01", "11:41"), ("10:01", "14:55"), ("14:00", "15:00")):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "20/40/60"):
                reporting._horizon_summary(2024, pd.DataFrame({"first_execution_time": [f"2024-01-03 {start}"],
                                                               "original_deadline": [f"2024-01-03 {end}"]}))

    def test_hold_smoke_is_not_goal_success(self):
        directory, status = self.write(robust_fixture(hold=True), robust_config(report_scope="smoke"))
        self.assertFalse(status["goal_achieved"])
        self.assertEqual(status["inventory_robust"]["report_scope"], "smoke")
        text = (directory/"final_report.md").read_text()
        self.assertIn("工程烟测", text.splitlines()[0])
        self.assertIn("没有可评价的做T胜率", text)
        self.assertTrue(pd.read_csv(directory/"exit_decisions.csv").empty)

    def test_pending_loss_and_inventory_failure_not_suppressed(self):
        _, status = self.write(robust_fixture(pending=True, net=-50., reason="stop_loss"))
        row = status["annual_results"][0]
        self.assertEqual((row["inventory_mismatch_days"], row["pending_pairs"], row["complete_t_pairs"]), (1, 1, 0))
        self.assertEqual(row["incremental_marked_pnl"], -50.)
        self.assertFalse(row["profit_target_met"])

    def test_manual_cost_stays_in_five_percent_denominator(self):
        result = robust_fixture(net=2000.)
        manual = result["baseline_events"].iloc[0].copy()
        manual[["base_order_type", "shares", "notional", "fee"]] = ["manual", 500, 5000., 1.]
        result["baseline_events"] = pd.concat([result["baseline_events"], manual.to_frame().T], ignore_index=True)
        _, status = self.write(result)
        row = status["annual_results"][0]
        self.assertEqual(row["all_reference_acquisition_cost"], 35003.)
        self.assertAlmostEqual(row["holding_cost_reduction_rate"], 2000/35003)

    def test_wrong_loss_version_source_or_execution_contract_rejected(self):
        changes = [{"research_version": "inventory_close_t_v16"}, {"training_loss": "regression"},
                   {"t_huber_alpha": .02}, {"t_huber_alpha": True}, {"horizons": [20]},
                   {"forecast_target": "same_day_1455"}, {"recovery_key": "close"},
                   {"frozen_t_models_reused": True}, {"t_recheck_exit_side_at_execution": False},
                   {"source_learning_report_dir": ""}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.write(robust_fixture(), robust_config(**change))
        for mode in ("old_profile", "hold_with_fill", "guard_missing", "cancelled_fill"):
            result = robust_fixture()
            if mode == "old_profile": result["selected_rule"]["profile"] = "l2_depth4"
            elif mode == "hold_with_fill": result["selected_rule"]["enabled"] = False
            elif mode == "guard_missing": del result["first_leg_exit_side_audit"]
            else:
                result["first_leg_exit_side_audit"].loc[0, ["cancelled", "reason", "opposite_quote_reason"]] = [True, "execution_exit_side_price_limit", "price_limit"]
            with self.subTest(mode=mode), self.assertRaises(ValueError) as caught:
                self.write(result)
            self.assertIn("v17", str(caught.exception))

    def test_original_year_exit_and_guard_csv_bytes_preserved(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        folder = directory/"year_2024"
        folder.mkdir()
        result, original = robust_fixture(reason="hard_deadline"), {}
        for name in ("exit_decisions", "first_leg_exit_side_audit"):
            result[name].to_csv(folder/f"{name}.csv", index=False)
            original[name] = (folder/f"{name}.csv").read_bytes()
        _, status = self.write(result, directory=directory)
        for name, content in original.items():
            self.assertEqual((folder/f"{name}.csv").read_bytes(), content)
        summary = status["inventory_robust"]["annual_exit_summaries"][0]
        self.assertEqual((summary["early_trigger_pairs"], summary["hard_deadline_trigger_pairs"]), (0, 1))

    def test_two_day_synthetic_guarded_execution_and_actual_cost(self):
        from quant_backtest.guarded_t_engine import run_guarded_portfolio
        from quant_backtest.dynamic_t_engine import SESSION_MINUTES
        from test_balanced_t_engine import market, quote, request, selection
        daily = market().iloc[:2].copy()
        cfg = robust_config(base_fraction=.85, commission_rate=.0008, min_commission=0., transfer_rate=.00001,
                            stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005, slippage_rate=.0005,
                            dynamic_exit_mode="fixed", max_cycles_per_day=0, max_bar_amount_fraction=.05)
        sample, decision = request(action=-1, fraction=.5, horizon=40)
        bars = pd.DataFrame([quote(day+pd.Timedelta(minutes=minute)) for day in daily.date for minute in SESSION_MINUTES])
        result = run_guarded_portfolio(pd.DataFrame([selection()]), daily, pd.DataFrame([sample]),
                                      pd.DataFrame([decision]), cfg, recovery_bars=bars)
        result["selected_rule"] = {"enabled": True, "candidate_index": 1, "profile": "huber_depth4",
                                   "min_net_edge": 0., "fraction": .5, "exit_mode": "fixed"}
        _, status = self.write(result, cfg)
        self.assertEqual(result["exit_decisions"].exit_earliest_time.iloc[0], decision["planned_recovery_time"])
        self.assertEqual(result["round_trips"].second_datetime.iloc[0], decision["planned_recovery_time"])
        row = status["annual_results"][0]
        self.assertEqual((row["complete_t_pairs"], row["inventory_mismatch_days"]), (1, 0))
        self.assertGreater(row["t_fee"], 0.)
        self.assertLess(row["incremental_marked_pnl"], 0.)
        self.assertAlmostEqual(row["incremental_marked_pnl"], result["round_trips"].net_pnl.sum(), places=7)
        self.assertEqual(status["inventory_robust"]["annual_planned_horizon_summaries"][0]["planned_h40_pairs"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
