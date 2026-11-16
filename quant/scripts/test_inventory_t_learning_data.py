"""永久底仓纯训练队列的隔离/T+1/未来独立回归；只写临时合成来源，不拟合模型。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.inventory_t_learning_data import (
    SOURCE_KEYS, build_inventory_core_cohort, select_clean_entry_requests)
from quant_backtest.weekly_t_timing_data import frame_sha256


def record_input(path, inputs):
    """测试采用与正式 runner 同签名的只读输入绑定。"""
    path = Path(path)
    inputs[str(path.resolve())] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class InventoryLearningTests(unittest.TestCase):
    """通过完整入口和真实无T规划核实持仓/来源语义，来源模型仅为固定身份夹具。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.fold = {"test_year": 2023, "inner_year": 2026, "core_years": [2024, 2025]}
        self.cfg = {"source_training_report_dir": str(self.source), "initial_cash": 40000., "base_fraction": .85,
                    "slippage_rate": 0., "commission_rate": 0., "min_commission": 0., "transfer_rate": 0.,
                    "stamp_rate_before_20230828": 0., "stamp_rate_from_20230828": 0., "embargo_days": 7}
        weekly, daily, intraday = [], [], []
        for year in self.fold["core_years"]:
            monday = pd.Timestamp(f"{year}-03-11")
            monday += pd.Timedelta(days=(-monday.dayofweek) % 7)
            for offset, symbols in ((0, ["A", "B"]), (7, ["C"])):
                entry = monday + pd.Timedelta(days=offset)
                for symbol in symbols:
                    weekly.append({"symbol": symbol, "trade_week": entry, "feature_week": entry-pd.Timedelta(days=7),
                                   "signal_datetime": entry-pd.Timedelta(days=3)+pd.Timedelta(hours=15),
                                   "feature_source_start": pd.Timestamp(year, 2 if offset == 0 else 3, 1),
                                   "entry_date": entry, "exit_date": entry+pd.Timedelta(days=4),
                                   "f_last_close": np.float32(10.), "feature_week_complete": True,
                                   "observed_last_datetime": entry-pd.Timedelta(days=3)+pd.Timedelta(hours=15)})
            for date in pd.bdate_range(monday, monday+pd.Timedelta(days=18)):
                for symbol in ("A", "B", "C"):
                    daily.append({"date": date, "symbol": symbol, "open": 10., "close": 10., "previous_close": 10.,
                                  "open_high": 10.01, "open_low": 9.99, "open_amount": 2e6, "open_volume": 2e5,
                                  "close_high": 10., "close_low": 10., "close_amount": 2e6, "close_volume": 2e5})
                    for clock in ("10:00", "11:00", "14:00"):
                        intraday.append({"datetime": pd.Timestamp(f"{date.date()} {clock}"), "symbol": symbol,
                                         "feature_source_start": date-pd.Timedelta(days=3),
                                         "f_current": np.float32(.1), "signal_bar_tradable": False,
                                         "target_return": np.nan, "recovery_price": np.nan,
                                         "execution_price": np.nan, "label_available": False})
        self.universe = pd.DataFrame(weekly)
        self.daily = pd.DataFrame(daily)
        self.intraday = pd.DataFrame(intraday)
        self.create_source()

    def create_source(self):
        """冻结完整排名集合/选股/model身份，允许每个测试声明不同来源边界后重建夹具。"""
        base = self.source / "year_2023/oof_weekly_cohort"
        base.mkdir(parents=True, exist_ok=True)
        audit = {"passed": True, "complete": True, "full_configured_scope_complete": True, "selectors": {}}
        for target in self.fold["core_years"]:
            directory = base / f"selector_{target}"
            directory.mkdir(exist_ok=True)
            train = [year for year in self.fold["core_years"] if year != target]
            excluded = sorted(set([2023, 2024, 2025, 2026])-set(train))
            for name in ("model_bundle.joblib", "return_model.txt", "amplitude_model.txt"):
                (directory/name).write_text(f"frozen fixture {target} {name}")
            sha = lambda name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
            targets = self.universe.loc[self.universe.trade_week.dt.year.eq(target)]
            selected = targets.loc[targets.symbol.ne("B"), ["symbol", "trade_week", "feature_week", "signal_datetime"]].copy()
            selected["rank"] = 1
            selected["source_model_training_years"] = json.dumps(train, separators=(",", ":"))
            selected["source_model_sha256"] = sha("model_bundle.joblib")
            selected.to_csv(directory/"oof_weekly_selections.csv", index=False)
            targets[["symbol", "trade_week"]].to_csv(directory/"oof_weekly_predictions.csv", index=False)
            metadata = {"train_years": train, "excluded_years": excluded,
                        "training_sample_sha256": "1"*64, "feature_source_sha256": "2"*64}
            (directory/"model_metadata.json").write_text(json.dumps(metadata))
            audit["selectors"][str(directory.relative_to(self.source))] = {
                "passed": True, "train_years": train, "excluded_years": excluded,
                "sample_sha256": "1"*64, "source_sha256": "2"*64,
                "model_bundle_sha256": sha("model_bundle.joblib"),
                "selected_csv_sha256": sha("oof_weekly_selections.csv"), "target_rows": len(targets),
                "native_checks": {name: {"native_sha256": sha(f"{name}_model.txt")} for name in ("return", "amplitude")}}
        (self.source/"independent_residual_audit.json").write_text(json.dumps(audit))
        (base/"cohort_metadata.json").write_text(json.dumps({
            "core_years": self.fold["core_years"], "excluded_years": [2023, 2026],
            "available_years": [2023, 2024, 2025, 2026],
            "weekly_source_keys_sha256": frame_sha256(self.universe[SOURCE_KEYS])}))

    def run_case(self, name="run", **kwargs):
        """每次独立输出，避免覆盖已有输入/结果；允许替换未来栏和日行情反例。"""
        return build_inventory_core_cohort(self.universe, kwargs.get("daily", self.daily),
            kwargs.get("intraday", self.intraday), self.fold, {**self.cfg, **kwargs.get("cfg", {})},
            self.root/name, {}, record_input)

    def test_holding_continues_after_week_exit_and_new_inventory_is_not_sellable(self):
        frame, meta = self.run_case()
        for year in self.fold["core_years"]:
            selected = self.universe.loc[self.universe.trade_week.dt.year.eq(year) & self.universe.symbol.eq("A")].iloc[0]
            held = frame.loc[frame.cohort_target_year.eq(year) & frame.symbol.eq("A")]
            self.assertTrue(held.datetime.dt.normalize().gt(selected.entry_date).all())
            self.assertTrue(held.datetime.dt.normalize().gt(selected.exit_date).any())
            self.assertEqual(set(held.datetime.dt.strftime("%H:%M")), {"10:00", "11:00", "14:00"})
            self.assertTrue(held.baseline_opening_shares.eq(1500).all())
        self.assertEqual([item["actual_base_entry_count"] for item in meta["selectors"]], [2, 2])
        self.assertTrue(frame.source_model_training_years.ne("[]").all())
        self.assertTrue(meta["all_core_years_joint_fit_only_not_independent_calibration"])

    def test_future_fields_and_invalid_labels_do_not_change_opportunities_or_states(self):
        expected, _ = self.run_case()
        changed = self.intraday.copy()
        changed[["target_return", "execution_price", "recovery_price"]] = 1e9
        changed["label_available"] = True
        changed["trade_week_complete"] = False
        actual, _ = self.run_case("changed", intraday=changed)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        self.assertFalse(any(name.startswith(("target_", "execution_", "recovery_", "label_")) for name in actual))
        self.assertNotIn("trade_week_complete", actual)

    def test_current_nat_source_is_retained_for_later_model_purge(self):
        source = self.intraday.copy()
        source.loc[source.symbol.eq("A"), "feature_source_start"] = pd.NaT
        frame, metadata = self.run_case(intraday=source)
        self.assertTrue(frame.loc[frame.symbol.eq("A"), "feature_source_start"].isna().all())
        self.assertGreater(sum(item["missing_current_provenance_rows"] for item in metadata["selectors"]), 0)

    def test_unselected_ranking_donor_can_reject_early_entries_before_account_planning(self):
        self.universe.loc[(self.universe.trade_week.dt.year==2024)&self.universe.symbol.eq("B"), "feature_source_start"] = pd.Timestamp("2023-12-01")
        self.create_source()
        frame, metadata = self.run_case()
        current = frame.loc[frame.cohort_target_year.eq(2024)]
        self.assertEqual(set(current.symbol), {"C"})
        self.assertEqual(metadata["selectors"][0]["source_rejected_requests"], 1)
        entries = pd.read_csv(self.root/"run/inventory_learning_cohort/core_2024/base_entries.csv")
        self.assertEqual(entries.symbol.tolist(), ["C"])
        self.assertEqual(entries.cash_after.iloc[0], 25000.)

    def test_cash_competition_source_from_other_stock_persists(self):
        frame, _ = self.run_case()
        c = frame.loc[frame.symbol.eq("C")]
        self.assertTrue(c.entry_feature_source_start.dt.month.eq(3).all())
        self.assertTrue(c.portfolio_cash_source_start.dt.month.eq(2).all())
        self.assertTrue(c.feature_source_start.dt.month.eq(2).all())

    def test_manual_instruction_does_not_enter_training_and_smoke_does_not_trim_core(self):
        expected, _ = self.run_case()
        actual, metadata = self.run_case("manual", cfg={"manual_buys_csv": "does_not_exist.csv", "smoke_days": 1})
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        self.assertTrue(all(item["manual_input_configured_but_disabled_for_training"] for item in metadata["selectors"]))

    def test_missing_open_does_not_backfill_or_make_future_entry(self):
        changed = self.daily.copy()
        changed.loc[changed.symbol.eq("A"), ["open", "open_amount", "open_volume"]] = np.nan
        frame, metadata = self.run_case(daily=changed)
        self.assertEqual(set(frame.symbol), {"C"})
        self.assertEqual([item["actual_base_entry_count"] for item in metadata["selectors"]], [1, 1])
        # 失败的 A 请求也参加组合决策来源，不能因为未成交就事后丢弃其依赖。
        self.assertTrue(frame.portfolio_cash_source_start.dt.month.eq(2).all())

    def test_all_source_rejected_keeps_empty_cohort_and_complete_cash_calendar(self):
        self.universe["feature_source_start"] = pd.Timestamp("2022-01-01")
        self.create_source()
        frame, metadata = self.run_case()
        self.assertEqual(len(frame), 0)
        self.assertEqual([item["actual_base_entry_count"] for item in metadata["selectors"]], [0, 0])
        for year in self.fold["core_years"]:
            curve = pd.read_csv(self.root/f"run/inventory_learning_cohort/core_{year}/planning_equity_curve.csv")
            self.assertEqual(len(curve), 15)
            self.assertTrue(curve.baseline_equity.eq(self.cfg["initial_cash"]).all())

    def test_wrong_training_year_and_changed_selected_bytes_are_rejected(self):
        directory = self.source/"year_2023/oof_weekly_cohort/selector_2024"
        path = directory/"model_metadata.json"
        metadata = json.loads(path.read_text()); metadata["train_years"] = [2024]
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "年份"):
            self.run_case()
        self.create_source()
        path = directory/"oof_weekly_selections.csv"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "文件被改变"):
            self.run_case("changed")

    def test_dropping_unselected_prediction_donor_cannot_pass(self):
        path = self.source/"year_2023/oof_weekly_cohort/selector_2024/oof_weekly_predictions.csv"
        frame = pd.read_csv(path); frame.loc[frame.symbol.ne("B")].to_csv(path, index=False)
        with self.assertRaises(AssertionError):
            self.run_case()

    def test_full_universe_cannot_be_shrunk_to_actual_positions(self):
        self.universe = self.universe.loc[self.universe.symbol.ne("B")]
        with self.assertRaisesRegex(ValueError, "来源键 SHA"):
            self.run_case()

    def test_embargo_uses_entry_time_and_missing_source_fails_closed(self):
        rows = pd.DataFrame({"entry_rank_source_start": pd.to_datetime(["2024-02-01", "2024-02-01", None]),
                             "signal_datetime": pd.to_datetime(["2024-12-20 15:00", "2024-12-27 15:00", "2024-03-01 15:00"]),
                             "entry_date": pd.to_datetime(["2024-12-23", "2024-12-30", "2024-03-04"])})
        kept, audit = select_clean_entry_requests(rows, [2023, 2024], [2025, 2026], 7)
        self.assertEqual(len(kept), 1)
        self.assertTrue(audit.entry_source_excluded_or_embargo.iloc[1])
        self.assertTrue(audit.entry_source_invalid.iloc[2])


if __name__ == "__main__":
    unittest.main()
