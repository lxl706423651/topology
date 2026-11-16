"""v16报告身份/成本/分片/守卫测试；仅合成账表和一个两日合成真实执行核。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from quant_backtest import inventory_t_close_reporting as reporting
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics
from test_guarded_t_reporting import guarded_config, guarded_fixture


def close_config(**changes):
    """使用新目标身份，绝不把传入公共核算的配置伪装成旧版本。"""
    cfg = guarded_config()
    cfg.update(research_version=reporting.VERSION, forecast_target="same_day_1455", recovery_key="close",
               fixed_recovery_time="14:55", frozen_t_models_reused=False,
               source_execution_report_dir="synthetic_v15_execution", source_learning_report_dir="synthetic_v14_core")
    cfg.update(changes)
    return cfg


def close_fixture(**changes):
    """沿用已核共同账目夹具，但真实原计划期限改为同日14:55。"""
    result = guarded_fixture(**changes)
    exits = result["exit_decisions"]
    if len(exits):
        deadline = pd.to_datetime(exits.first_execution_time).dt.normalize()+pd.Timedelta(hours=14, minutes=55)
        exits["original_deadline"] = deadline
        if exits.exit_reason.eq("hard_deadline").all():
            exits["exit_signal_time"] = deadline-pd.Timedelta(minutes=1)
            exits["exit_earliest_time"] = deadline
            trips = result["round_trips"]
            if len(trips):
                trips["second_datetime"] = [deadline.iloc[0]+pd.Timedelta(minutes=i) for i in range(len(trips))]
                recoveries = result["trade_events"].fragment_id.gt(0)
                result["trade_events"].loc[recoveries, "datetime"] = trips.second_datetime.to_numpy()
    return result


class CloseReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = {name: (ROOT / name).read_bytes() for name in (
            "quant_backtest/guarded_t_reporting.py", "quant_backtest/inventory_t_learning_reporting.py",
            "quant_backtest/dynamic_t_reporting.py", "quant_backtest/balanced_t_reporting.py",
            "quant_backtest/guarded_t_engine.py")}

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.frozen.items():
            assert (ROOT / name).read_bytes() == value, name

    def write(self, result, cfg=None, directory=None):
        """真实生成规范报告和PNG，输出限临时目录。"""
        if directory is None:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            directory = Path(temp.name)
        status = reporting.write_close_inventory_report(directory, cfg or close_config(), [(2024, result)])
        return directory, status

    def test_v16_identity_actual_cfg_plot_title_and_cost_contract(self):
        result, cfg = close_fixture(net=1600., fragments=(100, 500)), close_config()
        original, original_cfg = deepcopy(result), deepcopy(cfg)
        expected, _ = calculate_balanced_metrics(result, cfg)
        seen, titles = [], []
        writer = reporting.write_balanced_report
        from matplotlib.figure import Figure
        original_title = Figure.suptitle
        def observe_writer(directory, actual, entries, table):
            seen.append(actual["research_version"])
            return writer(directory, actual, entries, table)
        def observe_title(figure, text, *args, **kwargs):
            titles.append(text)
            return original_title(figure, text, *args, **kwargs)
        with patch.object(reporting, "write_balanced_report", observe_writer), patch.object(Figure, "suptitle", observe_title):
            directory, status = self.write(result, cfg)
        self.assertEqual(seen, [reporting.VERSION])
        self.assertIn("v16", titles[-1])
        self.assertIn("14:55", titles[-1])
        self.assertEqual(cfg, original_cfg)
        for name, value in original.items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(value, result[name], check_exact=True)
            elif isinstance(value, pd.Index):
                pd.testing.assert_index_equal(value, result[name], exact=True)
            else:
                self.assertEqual(value, result[name])
        actual = status["annual_results"][0]
        for name in ("all_reference_acquisition_cost", "holding_cost_reduction_rate", "t_fee", "t_slippage_cost",
                     "complete_t_pairs", "recovery_fragments", "completed_t_pair_lots", "total_t_traded_lots"):
            self.assertEqual(actual[name], expected[name])
        self.assertEqual(actual["all_reference_acquisition_cost"], 30002.)
        self.assertAlmostEqual(actual["holding_cost_reduction_rate"], 1600/30002)
        self.assertTrue(actual["profit_target_met"])
        self.assertFalse(status["goal_achieved"])
        self.assertEqual(status["frequency_target_status"], "not_quantified")
        meta = status["inventory_close"]
        self.assertFalse(meta["frozen_t_models_reused"])
        self.assertTrue(meta["models_retrained"])
        self.assertEqual(meta["newly_fitted_close_models_per_fold"], 2)
        self.assertEqual(meta["candidate_count"], 37)
        self.assertEqual((meta["recovery_key"], meta["recovery_kind"], meta["recovery_clock"]), ("close", "fixed_clock", "14:55"))
        self.assertFalse(meta["cross_horizon_argmax"])
        self.assertNotIn("prediction_horizons_minutes", meta)
        text = (directory / "final_report.md").read_text()
        self.assertIn("v16", text.splitlines()[0])
        self.assertIn("新拟合", text)
        self.assertIn("不能把账户差额单独归因于持有更久", text)
        self.assertNotIn("模型学习版本14原样复用", text)
        for filename in ("final_report.md", "输出文件说明.md", "annual_summary.csv", "年度.csv", "test_trade_events.csv",
                         "test_round_trips.csv", "test_equity_curve.csv", "test_open_positions.csv", "daily_inventory_audit.csv",
                         "baseline_trade_events.csv", "per_stock_cost_reduction.csv", "daily_frequency_coverage.csv",
                         "first_leg_exit_side_audit.csv", "guard_summary.csv", "dynamic_exit_summary.csv",
                         "exit_decisions.csv", "selected_close_rules.csv", "inventory_close_summary.json", "goal_evaluation.json"):
            self.assertTrue((directory / filename).is_file(), filename)
        self.assertEqual((directory / "test_return_curve.png").read_bytes(), (directory / "测试集收益曲线.png").read_bytes())

    def test_guard_cancellations_and_partial_recoveries_not_extra_pairs(self):
        directory, status = self.write(close_fixture(fragments=(100, 500)))
        row = status["annual_results"][0]
        self.assertEqual((row["complete_t_pairs"], row["recovery_fragments"], row["completed_t_pair_lots"]), (1, 2, 6.))
        guard = pd.read_csv(directory / "guard_summary.csv").iloc[0]
        self.assertEqual((guard.guard_audit_requests, guard.guard_cancelled_requests, guard.guard_passed_requests), (3, 1, 2))
        self.assertEqual((guard.guard_actual_first_leg_groups, guard.guard_passed_without_actual_first_leg), (1, 1))
        self.assertEqual(pd.read_csv(directory / "dynamic_exit_summary.csv").iloc[0].latched_exit_pairs, 1)

    def test_hold_smoke_is_engineering_and_not_success(self):
        directory, status = self.write(close_fixture(hold=True), close_config(report_scope="smoke"))
        self.assertFalse(status["goal_achieved"])
        self.assertEqual(status["inventory_close"]["report_scope"], "smoke")
        text = (directory / "final_report.md").read_text()
        self.assertIn("工程烟测", text.splitlines()[0])
        self.assertIn("没有可评价的做T胜率", text)
        self.assertTrue(pd.read_csv(directory / "exit_decisions.csv").empty)

    def test_inventory_failure_and_marked_loss_remain_visible(self):
        _, status = self.write(close_fixture(pending=True, net=-50., reason="stop_loss"))
        row = status["annual_results"][0]
        self.assertEqual((row["inventory_mismatch_days"], row["pending_pairs"], row["complete_t_pairs"]), (1, 1, 0))
        self.assertEqual(row["incremental_marked_pnl"], -50.)
        self.assertFalse(row["profit_target_met"])

    def test_all_cost_including_manual_buy_remains_denominator(self):
        result = close_fixture(net=2000.)
        manual = result["baseline_events"].iloc[0].copy()
        manual[["base_order_type", "shares", "notional", "fee"]] = ["manual", 500, 5000., 1.]
        result["baseline_events"] = pd.concat([result["baseline_events"], manual.to_frame().T], ignore_index=True)
        _, status = self.write(result)
        row = status["annual_results"][0]
        self.assertEqual(row["all_reference_acquisition_cost"], 35003.)
        self.assertAlmostEqual(row["holding_cost_reduction_rate"], 2000/35003)

    def test_old_deadline_or_model_identity_is_rejected(self):
        for failure in ("version", "target", "horizon", "reused", "no_guard", "source", "deadline", "cancelled_fill", "hold_with_fill"):
            result, cfg = close_fixture(), close_config()
            if failure == "version": cfg["research_version"] = "inventory_guarded_t_v15"
            elif failure == "target": cfg["forecast_target"] = "h20"
            elif failure == "horizon": cfg["horizons"] = [20]
            elif failure == "reused": cfg["frozen_t_models_reused"] = True
            elif failure == "no_guard": cfg["t_recheck_exit_side_at_execution"] = False
            elif failure == "source": cfg["source_learning_report_dir"] = ""
            elif failure == "deadline": result["exit_decisions"]["original_deadline"] -= pd.Timedelta(minutes=10)
            elif failure == "hold_with_fill": result["selected_rule"]["enabled"] = False
            else:
                result["first_leg_exit_side_audit"].loc[0, ["cancelled", "reason", "opposite_quote_reason"]] = [True, "execution_exit_side_price_limit", "price_limit"]
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.write(result, cfg)

    def test_existing_year_raw_exit_and_guard_tables_kept(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        folder = directory / "year_2024"
        folder.mkdir()
        result = close_fixture(reason="hard_deadline")
        originals = {}
        for name in ("exit_decisions", "first_leg_exit_side_audit"):
            result[name].to_csv(folder / f"{name}.csv", index=False)
            originals[name] = (folder / f"{name}.csv").read_bytes()
        _, status = self.write(result, directory=directory)
        for name, content in originals.items():
            self.assertEqual((folder / f"{name}.csv").read_bytes(), content)
        exit_summary = status["inventory_close"]["annual_exit_summaries"][0]
        self.assertEqual((exit_summary["early_trigger_pairs"], exit_summary["hard_deadline_trigger_pairs"]), (0, 1))

    def test_two_day_actual_guarded_engine_close_deadline_report(self):
        from quant_backtest.guarded_t_engine import run_guarded_portfolio
        from quant_backtest.dynamic_t_engine import SESSION_MINUTES
        from test_balanced_t_engine import market, quote, request, selection
        daily = market().iloc[:2].copy()
        cfg = close_config(base_fraction=.85, commission_rate=.0008, min_commission=0., transfer_rate=.00001,
                           stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005, slippage_rate=.0005,
                           dynamic_exit_mode="fixed", max_cycles_per_day=0, max_bar_amount_fraction=.05)
        sample, decision = request(action=-1, fraction=.5)
        planned = sample["datetime"].normalize()+pd.Timedelta(hours=14, minutes=55)
        sample["recovery_time"] = decision["planned_recovery_time"] = planned
        bars = pd.DataFrame([quote(day+pd.Timedelta(minutes=minute)) for day in daily.date for minute in SESSION_MINUTES])
        result = run_guarded_portfolio(pd.DataFrame([selection()]), daily, pd.DataFrame([sample]),
                                      pd.DataFrame([decision]), cfg, recovery_bars=bars)
        result["selected_rule"] = {"enabled": True, "candidate_index": 1, "profile": "l2_depth4",
                                   "min_net_edge": 0., "fraction": .5, "exit_mode": "fixed"}
        directory, status = self.write(result, cfg)
        self.assertEqual(result["exit_decisions"].exit_earliest_time.iloc[0], planned)
        self.assertEqual(result["round_trips"].second_datetime.iloc[0], planned)
        row = status["annual_results"][0]
        self.assertEqual(row["complete_t_pairs"], 1)
        self.assertEqual(row["inventory_mismatch_days"], 0)
        self.assertGreater(row["t_fee"], 0.)
        self.assertLess(row["incremental_marked_pnl"], 0.)
        self.assertEqual(status["inventory_close"]["annual_guard_summaries"][0]["guard_actual_first_leg_groups"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
