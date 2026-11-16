"""验证内部候选spawn并行结果一致、执行不变量和失败透传；所有账户使用合成行情。

在quant项目根目录执行quant环境Python scripts/test_weekly_t_validation.py。
审计CSV只写临时目录；不训练模型、不读取正式候选或任何外层测试结果。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_validation import run_candidate_evaluations, validate_candidate_result
from test_weekly_t_engine import create_daily, create_request, create_selection


class WeeklyTValidationTests(unittest.TestCase):
    """检查并行账户互不污染、审计不遗漏及错误不能被替换为零收益。"""

    def setUp(self):
        """冻结同一周股票池与一分钟样本；两相反动作的收益可直接手算。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="weekly_t_validation_tests_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.selected = pd.DataFrame([create_selection()])
        self.daily = create_daily(pd.bdate_range("2024-01-08", "2024-01-12"))
        sample, sell = create_request(side=-1, execution=10.4, fraction=0.5)
        self.intraday = pd.DataFrame([sample])
        self.sell = pd.DataFrame([sell])
        self.buy = self.sell.assign(action=1)
        self.cfg = {"initial_cash": 100_000.0, "base_fraction": 0.5, "top_n": 1, "lot_size": 100,
                    "commission_rate": 0.0, "min_commission": 0.0, "transfer_rate": 0.0,
                    "stamp_rate_before_20230828": 0.0, "stamp_rate_from_20230828": 0.0,
                    "slippage_rate": 0.0, "max_bar_amount_fraction": 1.0}
        self.tasks = [self.task(self.sell, 7), self.task(self.buy, 3), self.task(self.sell.iloc[:0], 11)]

    def task(self, decisions, index):
        """使用与调用方一致的逐行哈希生成去重键，保留原候选索引。"""
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(decisions, index=False).to_numpy().tobytes()).hexdigest()
        return {"candidate_index": index, "decision_sha256": fingerprint, "decisions": decisions.copy()}

    def result(self):
        """生成一个已完成且基准计划一致的真实引擎结果供篡改测试。"""
        return run_weekly_portfolio(self.selected, self.daily, self.intraday, self.sell, self.cfg)

    def evaluate(self, workers, tasks=None, directory=None, **kwargs):
        """把同样的冻结数据传入候选评价器，默认输出到独立临时子目录。"""
        return run_candidate_evaluations(self.tasks if tasks is None else tasks,
                                         self.selected, self.daily, self.intraday,
                                         {**self.cfg, "candidate_workers": workers},
                                         self.directory / (directory or f"workers_{workers}"), **kwargs)

    def test_single_and_spawn_workers_return_identical_metrics_and_audits(self):
        serial = self.evaluate(1)
        parallel = self.evaluate(2)
        self.assertEqual(list(serial), [task["decision_sha256"] for task in self.tasks])
        self.assertEqual(list(parallel), list(serial))
        self.assertEqual([item["original_index"] for item in serial.values()], [7, 3, 11])
        self.assertEqual([item["incremental_net_pnl"] for item in serial.values()], [1000.0, -1000.0, 0.0])
        for task in self.tasks:
            key, index = task["decision_sha256"], task["candidate_index"]
            self.assertEqual(set(serial[key]), {"incremental_net_pnl", "metrics", "original_index", "execution_audit"})
            self.assertEqual(serial[key]["execution_audit"], parallel[key]["execution_audit"])
            self.assertEqual(serial[key]["incremental_net_pnl"], parallel[key]["incremental_net_pnl"])
            pd.testing.assert_series_equal(pd.Series(serial[key]["metrics"]), pd.Series(parallel[key]["metrics"]))
            for suffix in ("t_attempts", "t_inventory_failures"):
                filename = f"candidate_{index:03d}_{suffix}.csv"
                a, b = self.directory / "workers_1" / filename, self.directory / "workers_2" / filename
                self.assertTrue(a.exists() and b.exists())
                self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_shared_frames_tasks_and_configuration_are_not_mutated(self):
        frames = [self.selected, self.daily, self.intraday, self.sell, self.buy]
        expected = [frame.copy(deep=True) for frame in frames]
        tasks, cfg = deepcopy(self.tasks), deepcopy(self.cfg)
        self.evaluate(1)
        for before, after in zip(expected, frames):
            pd.testing.assert_frame_equal(before, after)
        for before, after in zip(tasks, self.tasks):
            self.assertEqual(before["candidate_index"], after["candidate_index"])
            self.assertEqual(before["decision_sha256"], after["decision_sha256"])
            pd.testing.assert_frame_equal(before["decisions"], after["decisions"])
        self.assertEqual(cfg, self.cfg)
        self.assertEqual(list(self.directory.glob("**/*.tmp")), [])

    def test_invalid_engine_decision_exception_propagates_from_spawn(self):
        invalid = self.task(self.sell.assign(action=99), 1)
        with self.assertRaisesRegex(ValueError, "action"):
            self.evaluate(2, tasks=[invalid])
        self.assertEqual(list((self.directory / "workers_2").glob("*.csv")), [])

    def test_invalid_inputs_rejected_before_audit_files_are_created(self):
        for workers in (0, -1, 1.5, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.evaluate(workers)
        for tasks in ([self.tasks[0], self.tasks[0]],
                      [self.tasks[0], {**self.tasks[1], "candidate_index": 7}],
                      [{**self.tasks[0], "candidate_index": -1}],
                      [{"candidate_index": 1}]):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                self.evaluate(1, tasks=tasks)
        self.assertEqual(self.evaluate(2, tasks=[]), {})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_negative_cash_and_nonlot_shares_fail_closed(self):
        original = self.result()
        for account in ("trade_events", "baseline_events"):
            for column, value in (("cash_after", -0.01), ("shares", 101)):
                modified = deepcopy(original)
                modified[account].loc[0, column] = value
                with self.subTest(account=account, column=column), self.assertRaises(AssertionError):
                    validate_candidate_result(modified, self.cfg)
        modified = deepcopy(original)
        modified["equity_curve"].loc[0, "baseline_cash"] = -1.0
        with self.assertRaisesRegex(AssertionError, "nonnegative_cash"):
            validate_candidate_result(modified, self.cfg)

    def test_t1_cumulative_and_buyback_sellable_replenishment_are_rejected(self):
        original = self.result()
        modified = deepcopy(original)
        inventory = modified["daily_inventory_audit"]
        inventory.loc[0, "day_sold"] = inventory.opening_sellable.iloc[0] + 100
        with self.assertRaisesRegex(AssertionError, "t1_legal"):
            validate_candidate_result(modified, self.cfg)
        modified = deepcopy(original)
        events = modified["trade_events"]
        buyback = events.index[(events.event_type == "t") & (events.side == "buy")][0]
        events.loc[buyback, "sellable_after"] += 100
        with self.assertRaisesRegex(AssertionError, "t1_legal"):
            validate_candidate_result(modified, self.cfg)

    def test_first_leg_latency_and_reverse_pair_chronology_are_rejected(self):
        original = self.result()
        for timestamp in ("2024-01-09 10:00", "2024-01-09 10:02"):
            modified = deepcopy(original)
            events = modified["trade_events"]
            first = events.index[events.event_type == "t"][0]
            events.loc[first, "datetime"] = pd.Timestamp(timestamp)
            with self.subTest(timestamp=timestamp), self.assertRaises(AssertionError):
                validate_candidate_result(modified, self.cfg)
        modified = deepcopy(original)
        modified["round_trips"].loc[0, "second_datetime"] = pd.Timestamp("2024-01-09 09:00")
        with self.assertRaisesRegex(AssertionError, "paired_time_order"):
            validate_candidate_result(modified, self.cfg)

    def test_completed_pair_equity_identity_and_conditional_scope(self):
        result = self.result()
        self.assertTrue(validate_candidate_result(result, self.cfg)["paired_pnl_identity_checked"])
        result["round_trips"].loc[0, "net_pnl"] += 1.0
        with self.assertRaisesRegex(AssertionError, "paired_pnl_identity"):
            validate_candidate_result(result, self.cfg)
        for changed in ({"pending_pairs": 1}, {"same_base_plan": False}):
            altered = deepcopy(result)
            altered["metrics"].update(changed)
            self.assertFalse(validate_candidate_result(altered, self.cfg)["paired_pnl_identity_checked"])

    def test_recovery_store_forwarded_only_when_provided(self):
        valid = self.result()
        with patch("quant_backtest.weekly_t_validation.run_weekly_portfolio", return_value=valid) as mocked:
            self.evaluate(1, tasks=self.tasks[:1])
            self.assertEqual(mocked.call_args.kwargs, {})
        store = object()
        with patch("quant_backtest.weekly_t_validation.run_weekly_portfolio", return_value=valid) as mocked:
            self.evaluate(1, tasks=self.tasks[:1], directory="store", recovery_store=store)
            self.assertIs(mocked.call_args.kwargs["recovery_bars"], store)

    def test_legitimate_pending_candidate_retained_with_nonempty_failure_audits(self):
        self.intraday["recovery_amount"] = 0.0
        after_entry = self.daily.date > self.daily.date.min()
        self.daily.loc[after_entry, ["open_amount", "close_amount"]] = 0.0
        task = self.task(self.buy, 15)
        output = self.evaluate(1, tasks=[task])[task["decision_sha256"]]
        self.assertEqual(output["metrics"]["pending_pairs"], 1)
        self.assertGreater(output["metrics"]["recovery_failure_days"], 0)
        self.assertFalse(output["execution_audit"]["paired_pnl_identity_checked"])
        attempts = pd.read_csv(self.directory / "workers_1/candidate_015_t_attempts.csv")
        failures = pd.read_csv(self.directory / "workers_1/candidate_015_t_inventory_failures.csv")
        self.assertGreater(len(attempts), 0)
        self.assertTrue(attempts.event_type.str.startswith("t").all())
        self.assertGreater(len(failures), 0)
        self.assertTrue(failures.recovery_failure.all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
