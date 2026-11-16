"""验证机会分母、完整候选账户、spawn一致性及失败留证；仅写临时合成数据。

在quant根目录运行本脚本，无模型训练、无正式报告或原始行情修改。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from quant_backtest.weekly_t_engine import build_recovery_store, run_weekly_portfolio
from quant_backtest.weekly_t_frequency_validation import (
    RESULT_FILES, calculate_frequency_opportunities, run_frequency_candidates,
)
from test_weekly_t_engine import create_daily, create_request, create_selection


class FrequencyOpportunityTests(unittest.TestCase):
    """分母只由旧股和已知会话时钟决定，故未来行情或当日买入不能扩大分母。"""

    def setUp(self):
        """覆盖新买0旧股、100/300股不足整手、400股恰一手及午休边界。"""
        self.inventory = pd.DataFrame([
            {"date": "2024-01-09", "symbol": symbol, "opening_sellable": old, "weekly_buys": added}
            for symbol, old, added in [("NEW", 0, 800), ("A", 100, 0), ("B", 300, 0), ("C", 400, 0), ("D", 800, 0)]
        ] + [{"date": "2024-01-10", "symbol": "C", "opening_sellable": 400, "weekly_buys": 0},
             {"date": "2024-01-11", "symbol": "C", "opening_sellable": 400, "weekly_buys": 0}])
        self.baseline = {"daily_inventory_audit": self.inventory,
                         "trade_events": pd.DataFrame(columns=["event_type"])}
        self.intraday = pd.DataFrame([
            {"datetime": "2024-01-09 10:00", "symbol": symbol} for symbol in ["NEW", "A", "B", "C", "D"]
        ] + [{"datetime": "2024-01-09 14:15", "symbol": "C"},
             {"datetime": "2024-01-09 13:55", "symbol": "D"},
             {"datetime": "2024-01-10 14:15", "symbol": "C"},
             {"datetime": "2024-01-11 11:20", "symbol": "C"}])
        self.cfg = {"lot_size": 100, "short_t_fraction": .25, "latest_planned_recovery_time": "14:30"}

    def test_new_buys_and_small_old_positions_not_counted(self):
        summary, frame = calculate_frequency_opportunities(self.baseline, self.intraday, self.cfg)
        self.assertEqual(summary["eligible_portfolio_days"], 2)
        self.assertEqual(summary["eligible_inventory_stock_days"], 3)
        self.assertEqual(summary["eligible_signal_rows"], 4)
        self.assertEqual(frame.symbol.tolist(), ["C", "D", "C"])
        self.assertEqual(frame.eligible_t_shares.tolist(), [100, 200, 100])
        self.assertNotIn(pd.Timestamp("2024-01-10"), set(frame.date))
        self.assertEqual(frame.last_eligible_signal.iloc[-1], pd.Timestamp("2024-01-11 11:20"))

    def test_future_fields_cash_and_closing_inventory_do_not_change_denominator(self):
        original = calculate_frequency_opportunities(self.baseline, self.intraday, self.cfg)
        altered = self.intraday.assign(execution_price=np.nan, recovery_price=-100., recovery_amount=0., target_return=99.)
        baseline = deepcopy(self.baseline)
        baseline["daily_inventory_audit"]["closing_shares"] = 999999
        baseline["equity_curve"] = pd.DataFrame({"cash": [0.0]})
        changed = calculate_frequency_opportunities(baseline, altered, self.cfg)
        self.assertEqual(original[0], changed[0])
        pd.testing.assert_frame_equal(original[1], changed[1])

    def test_empty_legal_clocks_give_zero_not_missing_statistics(self):
        summary, frame = calculate_frequency_opportunities(self.baseline, self.intraday.iloc[:0], self.cfg)
        self.assertEqual(summary["eligible_portfolio_days"], 0)
        self.assertEqual(summary["eligible_inventory_stock_days"], 0)
        self.assertTrue(frame.empty)

    def test_duplicates_negative_old_shares_and_nonbaseline_rejected(self):
        altered = deepcopy(self.baseline)
        altered["daily_inventory_audit"].loc[0, "opening_sellable"] = -100
        with self.assertRaises(ValueError):
            calculate_frequency_opportunities(altered, self.intraday, self.cfg)
        altered = deepcopy(self.baseline)
        altered["trade_events"] = pd.DataFrame({"event_type": ["t"]})
        with self.assertRaisesRegex(ValueError, "未做T"):
            calculate_frequency_opportunities(altered, self.intraday, self.cfg)
        with self.assertRaisesRegex(ValueError, "唯一"):
            calculate_frequency_opportunities(self.baseline, pd.concat([self.intraday, self.intraday.iloc[:1]]), self.cfg)

    def test_invalid_denominator_configuration_or_symbol_is_rejected(self):
        for changed in ({"short_t_fraction": "0.25"}, {"lot_size": True},
                        {"latest_planned_recovery_time": "25:00"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                calculate_frequency_opportunities(self.baseline, self.intraday, {**self.cfg, **changed})
        altered = self.intraday.copy()
        altered.loc[0, "symbol"] = None
        with self.assertRaisesRegex(ValueError, "股票代码"):
            calculate_frequency_opportunities(self.baseline, altered, self.cfg)


class FrequencyCandidateTests(unittest.TestCase):
    """所有候选保存完整账户；相同SHA仍独立执行，异常不能成为0收益结果。"""

    def setUp(self):
        """使用免费用合成行情，使相反动作价差可以手算。"""
        temporary = tempfile.TemporaryDirectory(prefix="frequency_validation_")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.selected = pd.DataFrame([create_selection()])
        self.daily = create_daily(pd.bdate_range("2024-01-08", "2024-01-12"))
        sample, decision = create_request(side=-1, fraction=.5)
        self.intraday = pd.DataFrame([sample]); self.sell = pd.DataFrame([decision])
        self.cfg = {"initial_cash": 100000., "base_fraction": .5, "top_n": 1, "lot_size": 100,
                    "commission_rate": 0., "min_commission": 0., "transfer_rate": 0.,
                    "stamp_rate_before_20230828": 0., "stamp_rate_from_20230828": 0.,
                    "slippage_rate": 0., "max_bar_amount_fraction": 1., "short_t_fraction": .25}
        self.tasks = [self.task(self.sell, 7), self.task(self.sell, 3),
                      self.task(self.sell.assign(action=1), 11), self.task(self.sell.iloc[:0], 0)]

    def task(self, decisions, index):
        """候选索引唯一，但允许相同决策SHA出现于不同候选。"""
        digest = hashlib.sha256(pd.util.hash_pandas_object(decisions, index=False).to_numpy().tobytes()).hexdigest()
        return {"candidate_index": index, "decision_sha256": digest, "decisions": decisions.copy()}

    def evaluate(self, workers=1, tasks=None, name=None, **kwargs):
        """每次评估使用独立临时目录，不覆盖先前结果。"""
        return run_frequency_candidates(self.tasks if tasks is None else tasks, self.selected, self.daily,
                                        self.intraday, {**self.cfg, "candidate_workers": workers},
                                        self.directory / (name or f"workers_{workers}"), **kwargs)

    def result(self):
        """生成用于错误注入的真实已完成账户，不手造生产返回结果。"""
        return run_weekly_portfolio(self.selected, self.daily, self.intraday, self.sell, self.cfg)

    def test_same_hash_candidates_and_spawn_have_identical_complete_books(self):
        serial, parallel = self.evaluate(1), self.evaluate(2)
        self.assertEqual(list(serial), [7, 3, 11, 0])
        self.assertEqual(list(serial), list(parallel))
        self.assertEqual([serial[key]["incremental_net_pnl"] for key in serial], [1000., 1000., -1000., 0.])
        for index in serial:
            self.assertEqual(serial[index]["execution_audit"], parallel[index]["execution_audit"])
            pd.testing.assert_series_equal(pd.Series(serial[index]["metrics"]), pd.Series(parallel[index]["metrics"]))
            for filename in [*RESULT_FILES.values(), "decisions.csv", "metrics.json", "test_metrics.csv", "execution_audit.json", "candidate.json"]:
                first = self.directory / "workers_1" / f"candidate_{index:03d}" / filename
                second = self.directory / "workers_2" / f"candidate_{index:03d}" / filename
                self.assertTrue(first.is_file(), str(first))
                self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(serial[7]["active_days"], 1)
        self.assertEqual(serial[7]["active_stock_days"], 1)
        self.assertEqual(serial[0]["active_days"], 0)

    def test_public_frames_tasks_and_cfg_are_unchanged(self):
        before = [frame.copy(deep=True) for frame in [self.selected, self.daily, self.intraday, self.sell]]
        tasks, cfg = deepcopy(self.tasks), deepcopy(self.cfg)
        self.evaluate(tasks=self.tasks[:1])
        for first, second in zip(before, [self.selected, self.daily, self.intraday, self.sell]):
            pd.testing.assert_frame_equal(first, second)
        for first, second in zip(tasks, self.tasks):
            pd.testing.assert_frame_equal(first["decisions"], second["decisions"])
        self.assertEqual(cfg, self.cfg)
        self.assertEqual(list(self.directory.glob("**/*.tmp")), [])

    def test_real_no_t_baseline_excludes_day_zero_purchases(self):
        baseline = run_weekly_portfolio(self.selected, self.daily, self.intraday, self.sell.iloc[:0], self.cfg)
        clocks = pd.DataFrame({"datetime": ["2024-01-08 10:00", "2024-01-09 10:00"], "symbol": ["TEST", "TEST"]})
        summary, frame = calculate_frequency_opportunities(baseline, clocks, self.cfg)
        self.assertEqual(summary["eligible_inventory_stock_days"], 1)
        self.assertEqual(frame.date.tolist(), [pd.Timestamp("2024-01-09")])
        self.assertEqual(frame.opening_sellable.tolist(), [5000])

    def test_pending_failure_keeps_all_result_frames_and_active_day(self):
        self.intraday["recovery_amount"] = 0.
        self.daily.loc[self.daily.date.gt(self.daily.date.min()), ["open_amount", "close_amount"]] = 0.
        output = self.evaluate(tasks=[self.task(self.sell.assign(action=1), 5)])[5]
        folder = self.directory / "workers_1/candidate_005"
        self.assertEqual(output["metrics"]["pending_pairs"], 1)
        self.assertGreater(output["metrics"]["recovery_failure_days"], 0)
        self.assertFalse(output["execution_audit"]["paired_pnl_identity_checked"])
        self.assertEqual(output["active_days"], 1)
        self.assertTrue(all((folder / filename).exists() for filename in RESULT_FILES.values()))
        self.assertFalse((folder / "failure.json").exists())

    def test_engine_exception_propagates_from_spawn_with_input_evidence(self):
        task = self.task(self.sell.assign(action=99), 1)
        with self.assertRaisesRegex(ValueError, "action"):
            self.evaluate(2, tasks=[task])
        folder = self.directory / "workers_2/candidate_001"
        self.assertTrue((folder / "decisions.csv").exists())
        self.assertEqual(json.loads((folder / "failure.json").read_text())["error_type"], "ValueError")
        self.assertFalse((folder / "metrics.json").exists())

    def test_bad_accounting_preserves_evidence_and_never_returns_zero_result(self):
        for name in ["negative_cash", "wrong_pair_profit"]:
            modified = deepcopy(self.result())
            if name == "negative_cash":
                modified["trade_events"].loc[0, "cash_after"] = -1.
            else:
                modified["round_trips"].loc[0, "net_pnl"] += 1.
            with patch("quant_backtest.weekly_t_frequency_validation.run_weekly_portfolio", return_value=modified):
                with self.subTest(name=name), self.assertRaises(AssertionError):
                    self.evaluate(tasks=self.tasks[:1], name=name)
            folder = self.directory / name / "candidate_007"
            self.assertTrue((folder / "test_trade_events.csv").exists())
            self.assertTrue((folder / "failure.json").exists())
            self.assertFalse((folder / "summary.json").exists())

    def test_nonzero_fees_keep_cashflow_and_paired_pnl_identity(self):
        self.cfg.update(commission_rate=.0008, transfer_rate=.00001, stamp_rate_from_20230828=.0005,
                        slippage_rate=.0005, min_commission=5.)
        out = self.evaluate(tasks=self.tasks[:1])[7]
        trips = pd.read_csv(self.directory / "workers_1/candidate_007/test_round_trips.csv")
        self.assertGreater(out["metrics"]["t_fee"], 0.)
        self.assertAlmostEqual(out["incremental_net_pnl"], trips.net_pnl.sum(), places=8)

    def test_shared_recovery_store_partial_fragments_count_one_active_stock_day(self):
        bars = pd.DataFrame([
            {"datetime": pd.Timestamp(f"2024-01-09 {clock}"), "symbol": "TEST", "open": 10., "high": 10.01,
             "low": 9.99, "close": 10., "volume": amount / 10, "amount": amount}
            for clock, amount in [("14:55", 10000.), ("14:56", 15000.)]])
        store = build_recovery_store(bars)
        state = deepcopy(store.__getstate__())
        out = self.evaluate(2, tasks=self.tasks[:2], recovery_store=store)
        self.assertEqual(store.__getstate__(), state)
        for value in out.values():
            self.assertEqual(value["metrics"]["fragment_count"], 2)
            self.assertEqual(value["t_first_leg_count"], 1)
            self.assertEqual(value["active_stock_days"], 1)
            self.assertEqual(value["active_days"], 1)

    def test_invalid_task_or_worker_config_fails_before_output_creation(self):
        for workers in (0, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.evaluate(workers)
        with self.assertRaisesRegex(ValueError, "重复"):
            self.evaluate(tasks=[self.tasks[0], self.tasks[0]])
        self.assertEqual(self.evaluate(2, tasks=[]), {})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_existing_candidate_directory_not_overwritten(self):
        self.evaluate(tasks=self.tasks[:1])
        path = self.directory / "workers_1/candidate_007/test_equity_curve.csv"
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.evaluate(tasks=self.tasks[:1])
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
