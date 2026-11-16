"""验证当前信息训练域的交易时钟、计划整手及来源边界，不读行情、不拟合模型。

在 quant 仓库根目录直接运行；只用合成表，期望时刻与股数独立列出，未来标签
扰动不得影响过滤。输入和冻结模块保持只读，不生成正式报告或修改缓存。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_train_domain import filter_core_t_training_domain


def create_domain_fixture(clocks=("10:00",), prices=(20.,)):
    """构造周二信号及前四个实际周范围内已知参考价，未来标签故意全部缺失。"""
    rows, weeks = [], []
    for number, (clock, price) in enumerate(zip(clocks, prices)):
        symbol = f"SH.{600000 + number:06d}"
        rows.append({"datetime": pd.Timestamp("2024-03-05 " + clock), "trade_week": pd.Timestamp("2024-03-04"),
                     "symbol": symbol, "feature_source_start": pd.Timestamp("2024-02-05"),
                     "f_alpha": float(number), "target_return": np.nan, "label_available": False,
                     "execution_price": np.nan, "recovery_price": np.nan, "label_start": pd.NaT,
                     "label_end": pd.NaT, "planned_recovery_time": pd.NaT})
        weeks.append({"trade_week": pd.Timestamp("2024-03-04"), "symbol": symbol,
                      "signal_datetime": pd.Timestamp("2024-03-01 15:00"),
                      "feature_source_start": pd.Timestamp("2024-02-05"), "f_last_close": price,
                      "target_return": np.nan, "label_available": False})
    return pd.DataFrame(rows), pd.DataFrame(weeks)


def create_domain_config():
    """固定与真实协议相同的初始槽位、25% T 和费用后期限；佣金仅用于反例。"""
    return {"t_training_domain": "known_slot_and_recovery_deadline", "initial_cash": 1_000_000.,
            "base_fraction": .95, "top_n": 100, "weekly_top_n": 100, "lot_size": 100,
            "short_t_fraction": .25, "slippage_rate": .0005, "latest_planned_recovery_time": "14:30",
            "commission_rate": .0008, "transfer_rate": .00001}


class TrainingDomainTests(unittest.TestCase):
    """覆盖机械训练资格而非执行成功；所有未来有效性均不作为保留依据。"""

    def setUp(self):
        """每例新建独立配置，避免测试间状态污染。"""
        self.cfg = create_domain_config()

    def test_lunch_and_next_minute_are_independent_expected_clocks(self):
        frame, weeks = create_domain_fixture(("10:00", "11:20", "11:30", "13:01"), (20.,) * 4)
        result, audit, _ = filter_core_t_training_domain(frame, weeks, 20, self.cfg)
        expected = pd.to_datetime(["2024-03-05 10:21", "2024-03-05 13:11", "2024-03-05 13:21", "2024-03-05 13:22"])
        self.assertEqual(audit.planned_recovery_time.tolist(), expected.tolist())
        self.assertEqual(len(result), 4)

    def test_exact_1430_kept_and_next_minute_rejected_for_each_horizon(self):
        for horizon, clocks in [(20, ("14:09", "14:10")), (40, ("13:49", "13:50")), (60, ("13:29", "13:30"))]:
            with self.subTest(horizon=horizon):
                frame, weeks = create_domain_fixture(clocks, (20., 20.))
                result, audit, _ = filter_core_t_training_domain(frame, weeks, horizon, self.cfg)
                self.assertEqual(audit.planned_recovery_time.dt.strftime("%H:%M").tolist(), ["14:30", "14:31"])
                self.assertEqual(result.index.tolist(), [0])
                self.assertEqual(audit.drop_reason.tolist(), ["keep", "recovery_after_deadline"])

    def test_no_cross_day_recovery_is_not_replaced_with_future_label_time(self):
        frame, weeks = create_domain_fixture(("14:40", "15:00"), (20., 20.))
        frame["recovery_time"] = pd.Timestamp("2024-03-06 10:00")
        frame["planned_recovery_time"] = pd.Timestamp("2024-03-05 14:00")
        result, audit, _ = filter_core_t_training_domain(frame, weeks, 20, self.cfg)
        self.assertTrue(result.empty)
        self.assertTrue(audit.planned_recovery_time.isna().all())
        self.assertEqual(audit.drop_reason.tolist(), ["no_same_day_planned_recovery"] * 2)

    def test_static_quantity_matches_engine_plan_without_commission_denominator(self):
        frame, weeks = create_domain_fixture(("10:00",) * 4, (20., 30., 23.73, 95.))
        result, audit, metadata = filter_core_t_training_domain(frame, weeks, 20, self.cfg)
        self.assertEqual(audit.static_slot_budget.tolist(), [9500.] * 4)
        self.assertEqual(audit.static_base_shares.tolist(), [400, 300, 400, 0])
        self.assertEqual(audit.static_t_shares.tolist(), [100, 0, 100, 0])
        self.assertEqual(result.index.tolist(), [0, 2])
        # 23.73 正好能计划 400 股；把佣金塞进分母会错误截成 300 股。
        changed = {**self.cfg, "commission_rate": .9, "transfer_rate": .9}
        _, after, after_metadata = filter_core_t_training_domain(frame, weeks, 20, changed)
        pd.testing.assert_frame_equal(audit, after)
        self.assertEqual(metadata, after_metadata)

    def test_lot_multiple_and_fraction_use_original_engine_rounding(self):
        frame, weeks = create_domain_fixture(("10:00", "10:00"), (10., 20.))
        config = {**self.cfg, "lot_size": 200}
        _, audit, _ = filter_core_t_training_domain(frame, weeks, 20, config)
        self.assertEqual(audit.static_base_shares.tolist(), [800, 400])
        self.assertEqual(audit.static_t_shares.tolist(), [200, 0])
        self.assertEqual(audit.keep.tolist(), [True, False])

    def test_audit_reports_both_independent_reasons_and_keeps_original_index(self):
        frame, weeks = create_domain_fixture(("14:10", "14:40", "10:00"), (30., 30., 20.))
        frame.index = pd.Index([7, 7, 2], name="source_row")
        result, audit, metadata = filter_core_t_training_domain(frame, weeks, 20, self.cfg)
        self.assertTrue(audit.index.equals(frame.index))
        self.assertEqual(audit.drop_reason.tolist(), ["recovery_after_deadline;static_t_below_one_lot",
                                                     "no_same_day_planned_recovery;static_t_below_one_lot", "keep"])
        self.assertEqual(metadata["failed_condition_counts"], {"recovery_deadline": 2, "static_t_below_one_lot": 2, "both": 2})
        self.assertEqual((metadata["input_rows"], metadata["kept_rows"], metadata["dropped_rows"]), (3, 1, 2))
        pd.testing.assert_frame_equal(result, frame.iloc[[2]])

    def test_future_columns_and_week_labels_do_not_change_domain_or_hashes(self):
        frame, weeks = create_domain_fixture(("10:00", "14:10"), (20., 30.))
        baseline, audit, metadata = filter_core_t_training_domain(frame, weeks, 20, self.cfg)
        changed, changed_weeks = frame.copy(deep=True), weeks.copy(deep=True)
        for column in ("execution_price", "recovery_price", "target_return"):
            changed[column] = [1e8, -1e8]
        for column in ("execution_time", "recovery_time", "label_start", "label_end", "planned_recovery_time"):
            changed[column] = pd.Timestamp("2099-01-01")
        changed["label_available"] = True
        changed_weeks["target_return"] = [999., -999.]
        changed_weeks["label_available"] = True
        changed_weeks["execution_price"] = -999.
        selected, altered_audit, altered_metadata = filter_core_t_training_domain(changed, changed_weeks, 20, self.cfg)
        pd.testing.assert_frame_equal(audit, altered_audit)
        self.assertEqual(metadata, altered_metadata)
        self.assertTrue(selected.index.equals(baseline.index))
        pd.testing.assert_frame_equal(selected, changed.iloc[[0]])

    def test_input_features_labels_sources_and_attrs_are_not_modified(self):
        frame, weeks = create_domain_fixture(("14:10", "10:00"), (20., 20.))
        frame.index = pd.Index([15, 2], name="original")
        frame.attrs = {"scope": "core_oof"}
        before, week_before, cfg_before = frame.copy(deep=True), weeks.copy(deep=True), deepcopy(self.cfg)
        result, _, _ = filter_core_t_training_domain(frame, weeks, 20, self.cfg)
        pd.testing.assert_frame_equal(result, before.iloc[[1]])
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(weeks, week_before)
        self.assertEqual(result.attrs, frame.attrs)
        self.assertEqual(self.cfg, cfg_before)
        result.loc[2, "f_alpha"] = 999.
        self.assertEqual(frame.loc[2, "f_alpha"], 1.)

    def test_missing_and_duplicate_weekly_join_or_signal_fail_closed(self):
        frame, weeks = create_domain_fixture()
        for altered in (weeks.iloc[:0], pd.concat([weeks, weeks]), weeks.assign(symbol="SH.699999")):
            with self.subTest(weekly_rows=len(altered)), self.assertRaises(ValueError):
                filter_core_t_training_domain(frame, altered, 20, self.cfg)
        with self.assertRaises(ValueError):
            filter_core_t_training_domain(pd.concat([frame, frame]), weeks, 20, self.cfg)

    def test_unknown_future_or_not_yet_covered_source_fails_without_rewriting(self):
        frame, weeks = create_domain_fixture()
        bad = [(frame.assign(feature_source_start=pd.Timestamp("2024-02-06")), weeks),
               (frame, weeks.assign(feature_source_start=pd.NaT)),
               (frame, weeks.assign(feature_source_start=pd.Timestamp("2024-03-02"))),
               (frame, weeks.assign(signal_datetime=pd.Timestamp("2024-03-05"))),
               (frame.assign(feature_source_start=pd.Timestamp("2025-01-01")), weeks),
               (frame.assign(datetime=pd.Timestamp("2024-03-12 10:00")), weeks)]
        for minutes, weekly in bad:
            with self.subTest(minutes=len(minutes)), self.assertRaises(ValueError):
                filter_core_t_training_domain(minutes, weekly, 20, self.cfg)

    def test_lunch_nonminute_timezone_and_invalid_week_are_rejected(self):
        frame, weeks = create_domain_fixture()
        for stamp in (pd.Timestamp("2024-03-05 12:00"), pd.Timestamp("2024-03-05 09:30"),
                      pd.Timestamp("2024-03-05 10:00:01"), pd.Timestamp("2024-03-05 10:00", tz="Asia/Shanghai"), pd.NaT):
            with self.subTest(timestamp=stamp), self.assertRaises(ValueError):
                filter_core_t_training_domain(frame.assign(datetime=stamp), weeks, 20, self.cfg)
        with self.assertRaises(ValueError):
            filter_core_t_training_domain(frame.assign(trade_week=pd.Timestamp("2024-03-05")), weeks, 20, self.cfg)

    def test_invalid_prices_and_configs_are_rejected(self):
        frame, weeks = create_domain_fixture()
        for price in (0., -1., np.inf, -np.inf, np.nan, True, "bad"):
            with self.subTest(price=price), self.assertRaises(ValueError):
                filter_core_t_training_domain(frame, weeks.assign(f_last_close=price), 20, self.cfg)
        invalid = {"initial_cash": [0., -1., np.inf, True], "base_fraction": [0., 1.1, np.nan, True],
                   "short_t_fraction": [0., -1., 1.1, True], "slippage_rate": [-1., np.inf, True],
                   "top_n": [0, 1.5, True], "lot_size": [0, 150, 100., True], "weekly_top_n": [99],
                   "latest_planned_recovery_time": ["12:00", "24:00", "14:30:00", "bad", None],
                   "t_training_domain": ["unknown", None]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    filter_core_t_training_domain(frame, weeks, 20, {**self.cfg, key: value})
        for horizon in (0, -1, 241, True, 20., "20"):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                filter_core_t_training_domain(frame, weeks, horizon, self.cfg)

    def test_empty_input_has_stable_schema_and_no_future_dependency(self):
        frame, weeks = create_domain_fixture()
        result, audit, metadata = filter_core_t_training_domain(frame.iloc[:0], weeks, 20, self.cfg)
        pd.testing.assert_frame_equal(result, frame.iloc[:0])
        self.assertTrue(audit.empty)
        self.assertEqual(metadata["input_rows"], 0)
        self.assertEqual(metadata["future_columns_consumed"], [])
        self.assertTrue(metadata["feature_source_unchanged"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
