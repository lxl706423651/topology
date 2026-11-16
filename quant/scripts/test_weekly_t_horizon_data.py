"""短周期标签、真实恢复报价与独立缓存的持久测试，不训练或改既有缓存。"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_horizon_data import (
    LABEL_COLUMNS, QUOTE_COLUMNS, TRADING_MINUTES, advance_trading_minutes,
    build_symbol_horizon_data, load_frozen_selections, load_horizon_data,
    next_trading_minute, validate_horizon_config,
)


def create_horizon_day(day: str) -> pd.DataFrame:
    """生成完整240分钟行情，价格按交易分钟递增，午休没有记录。"""
    times = pd.Timestamp(day) + pd.to_timedelta(TRADING_MINUTES, unit="min")
    close = 10.0 + np.arange(len(times)) / 100.0
    return pd.DataFrame({"datetime": times, "open": close - 0.01, "high": close + 0.02,
                         "low": close - 0.02, "close": close, "volume": 1000.0, "amount": close * 1000.0})


class WeeklyTHorizonDataTests(unittest.TestCase):
    """检查交易时钟、精确缺价、全年度恢复覆盖、冻结文件和缓存校验。"""

    def setUp(self):
        """每个用例的行情、选股和缓存都位于独立临时目录。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="weekly_t_horizon_tests_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "raw/2026"
        self.source.mkdir(parents=True)
        self.raw = create_horizon_day("2026-04-29")
        self.source_file = self.source / "SH.600000.csv"
        self.raw.to_csv(self.source_file, index=False)
        self.report = self.directory / "frozen/year_2026"
        self.report.mkdir(parents=True)
        self.selection_path = self.report / "weekly_selections.csv"
        pd.DataFrame([{"symbol": "SH.600000", "trade_week": "2026-04-27",
                       "entry_date": "2026-04-27", "exit_date": "2026-04-30"}]).to_csv(self.selection_path, index=False)
        self.cfg = {"data_root": str(self.source.parent), "start_date": "2023-01-01",
                    "end_date": "2026-04-30", "workers": 1,
                    "frozen_selection_report_dir": str(self.report.parent),
                    "horizon_cache_dir": str(self.directory / "cache")}

    def test_trade_minute_anchor_lunch_and_same_day_boundary(self):
        self.assertEqual(next_trading_minute("2026-04-29 09:35"), pd.Timestamp("2026-04-29 09:36"))
        self.assertEqual(advance_trading_minutes("2026-04-29 09:36", 20), pd.Timestamp("2026-04-29 09:56"))
        self.assertEqual(next_trading_minute("2026-04-29 11:30"), pd.Timestamp("2026-04-29 13:01"))
        self.assertEqual(advance_trading_minutes("2026-04-29 11:01", 60), pd.Timestamp("2026-04-29 13:31"))
        self.assertEqual(advance_trading_minutes("2026-04-29 14:01", 59), pd.Timestamp("2026-04-29 15:00"))
        self.assertTrue(pd.isna(advance_trading_minutes("2026-04-29 14:01", 60)))
        self.assertTrue(pd.isna(next_trading_minute("2026-04-29 15:00")))
        self.assertTrue(pd.isna(advance_trading_minutes(pd.NaT, 20)))
        self.assertEqual(advance_trading_minutes("2026-04-29 13:01", 0), pd.Timestamp("2026-04-29 13:01"))

    def test_invalid_clocks_steps_and_configuration_fail(self):
        for timestamp in ("2026-04-29 09:30", "2026-04-29 12:00", "2026-04-29 09:35:01",
                          "2026-04-29 09:35:00.000000001", "2026-04-29 09:35+08:00"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                advance_trading_minutes(timestamp, 1)
        for steps in (-1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                advance_trading_minutes("2026-04-29 09:35", steps)
        for changes in ({"horizons": []}, {"horizons": [20, 20]}, {"horizons": [0]},
                        {"horizons": [True]}, {"signal_times": ["15:00"]},
                        {"signal_times": ["09:35", "09:35"]}, {"signal_times": ["9:35"]},
                        {"workers": 0}, {"max_symbols": -1}, {"start_date": "2027-01-01"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_horizon_data({**self.cfg, **changes})

    def test_exact_prices_and_all_current_signal_horizons_retained(self):
        original = self.raw.copy(deep=True)
        labels, quotes = build_symbol_horizon_data(self.raw, "SH.600000", [2026], self.cfg)
        self.assertEqual(list(labels), LABEL_COLUMNS)
        self.assertEqual(list(quotes), QUOTE_COLUMNS)
        self.assertEqual(len(labels), 24)
        self.assertFalse(labels.duplicated(["datetime", "symbol", "horizon_minutes"]).any())
        row = labels.loc[labels.datetime.eq(pd.Timestamp("2026-04-29 09:35")) & labels.horizon_minutes.eq(20)].iloc[0]
        self.assertEqual(row.execution_time, pd.Timestamp("2026-04-29 09:36"))
        self.assertEqual(row.recovery_time, pd.Timestamp("2026-04-29 09:56"))
        self.assertAlmostEqual(row.execution_price, 10.05)
        self.assertAlmostEqual(row.recovery_price, 10.25)
        self.assertAlmostEqual(row.target_return, 10.25 / 10.05 - 1)
        pd.testing.assert_series_equal(labels.label_start, labels.execution_time, check_names=False)
        pd.testing.assert_series_equal(labels.label_end, labels.recovery_time, check_names=False)
        final = labels.loc[labels.datetime.eq(pd.Timestamp("2026-04-29 14:00")) & labels.horizon_minutes.eq(60)].iloc[0]
        self.assertTrue(pd.isna(final.label_end))
        self.assertTrue(pd.isna(final.target_return))
        self.assertFalse(any(column.startswith("f_") for column in labels.columns.union(quotes.columns)))
        pd.testing.assert_frame_equal(self.raw, original)

    def test_future_missing_never_filters_current_or_fills_prices(self):
        remove = pd.to_datetime(["2026-04-29 09:36", "2026-04-29 10:06"])
        raw = self.raw.loc[~self.raw.datetime.isin(remove)].copy()
        labels, _ = build_symbol_horizon_data(raw, "SH.600000", [2026], self.cfg)
        self.assertEqual(len(labels), 24)
        missing_execution = labels.loc[labels.datetime.eq(pd.Timestamp("2026-04-29 09:35"))]
        self.assertTrue(missing_execution.execution_price.isna().all())
        self.assertTrue(missing_execution.target_return.isna().all())
        missing_recovery = labels.loc[labels.datetime.eq(pd.Timestamp("2026-04-29 09:45")) & labels.horizon_minutes.eq(20)]
        self.assertTrue(missing_recovery.recovery_price.isna().all())
        truncated = raw.loc[raw.datetime.le(pd.Timestamp("2026-04-29 09:35"))]
        prefix, _ = build_symbol_horizon_data(truncated, "SH.600000", [2026], self.cfg)
        self.assertEqual(len(prefix), 3)
        self.assertTrue(prefix.target_return.isna().all())
        self.assertEqual(prefix.datetime.unique().tolist(), [pd.Timestamp("2026-04-29 09:35")])

    def test_only_exact_current_minutes_count_no_subsecond_approximation(self):
        raw = self.raw.copy()
        raw.loc[raw.datetime.eq(pd.Timestamp("2026-04-29 09:35")), "datetime"] += pd.Timedelta(seconds=1)
        labels, _ = build_symbol_horizon_data(raw, "SH.600000", [2026], self.cfg)
        self.assertEqual(len(labels), 21)
        self.assertNotIn(pd.Timestamp("2026-04-29 09:35"), labels.datetime.tolist())

    def test_quote_windows_cover_whole_selected_year_and_skip_tail_without_future_filters(self):
        raw = pd.concat([create_horizon_day("2025-04-29"), create_horizon_day("2026-01-05"),
                         create_horizon_day("2026-04-27"), self.raw, create_horizon_day("2026-04-30")], ignore_index=True)
        _, _, minutes = validate_horizon_config(self.cfg)
        _, quotes = build_symbol_horizon_data(raw, "SH.600000", [2026], self.cfg)
        self.assertEqual(len(quotes), 4 * len(minutes))
        self.assertEqual(set(quotes.datetime.dt.normalize()),
                         set(pd.to_datetime(["2026-01-05", "2026-04-27", "2026-04-29", "2026-04-30"])))
        self.assertFalse((quotes.datetime.dt.hour * 60 + quotes.datetime.dt.minute).ge(895).any())
        self.assertIn(pd.Timestamp("2026-04-29 09:56"), quotes.datetime.tolist())
        self.assertIn(pd.Timestamp("2026-04-29 10:01"), quotes.datetime.tolist())
        # 没有实际当前信号的日期仍有恢复报价，可处理从此前日期遗留的债务。
        without_signals = raw.loc[~raw.datetime.dt.strftime("%H:%M").isin(self.cfg.get("signal_times", ["09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00"]))]
        labels_missing, quotes_missing = build_symbol_horizon_data(without_signals, "SH.600000", [2026], self.cfg)
        self.assertTrue(labels_missing.empty)
        self.assertFalse(quotes_missing.empty)

    def test_zero_volume_flat_quotes_retained_and_missing_quotes_never_filled(self):
        raw = self.raw.loc[~self.raw.datetime.eq(pd.Timestamp("2026-04-29 09:57"))].copy()
        zero = raw.datetime.eq(pd.Timestamp("2026-04-29 09:56"))
        raw.loc[zero, ["volume", "amount"]] = 0.0
        raw.loc[zero, ["open", "high", "low", "close"]] = 10.0
        _, quotes = build_symbol_horizon_data(raw, "SH.600000", [2026], self.cfg)
        self.assertNotIn(pd.Timestamp("2026-04-29 09:57"), quotes.datetime.tolist())
        retained = quotes.loc[quotes.datetime.eq(pd.Timestamp("2026-04-29 09:56"))].iloc[0]
        self.assertEqual(retained.amount, 0)
        self.assertEqual(retained.volume, 0)
        self.assertEqual(retained.high, retained.low)

    def test_selection_union_hashes_and_conflicting_dates_fail(self):
        original = pd.read_csv(self.selection_path)
        validation_path = self.report / "validation_weekly_selections.csv"
        original.to_csv(validation_path, index=False)
        selected, sources = load_frozen_selections(self.cfg)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in sources))
        original.loc[0, "exit_date"] = "2026-04-29"
        original.to_csv(validation_path, index=False)
        with self.assertRaisesRegex(ValueError, "冲突"):
            load_frozen_selections(self.cfg)
        original.loc[0, "exit_date"] = "2026-05-04"
        original.to_csv(validation_path, index=False)
        with self.assertRaisesRegex(ValueError, "自然周"):
            load_frozen_selections(self.cfg)

    def test_cache_reuse_input_immutable_and_source_selection_horizon_invalidation(self):
        original = deepcopy(self.cfg)
        first_labels, first_quotes, first = load_horizon_data(self.cfg)
        with patch("quant_backtest.weekly_t_horizon_data.load_weekly_symbol_bars", side_effect=AssertionError("不能重读CSV")):
            labels, quotes, reused = load_horizon_data(self.cfg)
        pd.testing.assert_frame_equal(first_labels, labels)
        pd.testing.assert_frame_equal(first_quotes, quotes)
        self.assertTrue(reused["cache_hit"])
        self.assertFalse(first["cache_hit"])
        self.assertEqual(self.cfg, original)
        self.assertEqual(list((self.directory / "cache").glob("**/*.tmp")), [])
        changed = self.raw.copy()
        changed.loc[0, "amount"] *= 2
        changed.to_csv(self.source_file, index=False)
        _, _, source_changed = load_horizon_data(self.cfg)
        self.assertNotEqual(first["fingerprint"], source_changed["fingerprint"])
        with self.selection_path.open("a") as stream:
            stream.write("\n")
        _, _, selection_changed = load_horizon_data(self.cfg)
        self.assertNotEqual(source_changed["fingerprint"], selection_changed["fingerprint"])
        shorter, _, horizon_changed = load_horizon_data({**self.cfg, "horizons": [20]})
        self.assertEqual(len(shorter), 8)
        self.assertNotEqual(selection_changed["fingerprint"], horizon_changed["fingerprint"])

    def test_serial_spawn_equal_mainboard_history_union_and_cache_metadata_rejection(self):
        self.raw.to_csv(self.source / "SZ.000001.csv", index=False)
        self.raw.to_csv(self.source / "SZ.300001.csv", index=False)
        historical = self.source.parent / "2023"
        historical.mkdir()
        create_horizon_day("2023-04-28").to_csv(historical / "SH.600001.csv", index=False)
        serial_labels, serial_quotes, manifest = load_horizon_data(self.cfg)
        parallel_cfg = {**self.cfg, "workers": 2, "horizon_cache_dir": str(self.directory / "parallel")}
        parallel_labels, parallel_quotes, parallel_manifest = load_horizon_data(parallel_cfg)
        pd.testing.assert_frame_equal(serial_labels, parallel_labels)
        pd.testing.assert_frame_equal(serial_quotes, parallel_quotes)
        self.assertEqual(manifest["fingerprint"], parallel_manifest["fingerprint"])
        self.assertEqual(set(serial_labels.symbol), {"SH.600000", "SH.600001", "SZ.000001"})
        self.assertEqual(set(serial_quotes.symbol), {"SH.600000"})
        metadata = Path(manifest["cache_path"]) / "metadata.json"
        metadata.write_text(json.dumps({**manifest, "label_rows": manifest["label_rows"] + 1}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "行数"):
            load_horizon_data(self.cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
