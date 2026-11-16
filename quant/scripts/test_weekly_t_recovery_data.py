"""验证独立恢复行情只包含精确实际分钟，缓存可重用且不污染冻结的预测数据。

在quant目录执行quant环境Python scripts/test_weekly_t_recovery_data.py。
所有CSV和缓存均位于临时目录；测试不改历史行情、不训练模型、不读取正式回测。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_recovery_data import (
    RECOVERY_COLUMNS, RECOVERY_MINUTES, load_recovery_data, select_recovery_bars,
)


def create_recovery_day(day: str = "2026-04-30") -> pd.DataFrame:
    """生成恢复窗口及前后各一分钟的合法行情，价格可直接核对。"""
    timestamp = pd.date_range(day + " 14:54", day + " 15:01", freq="min")
    price = pd.Series([10.0 + index * 0.01 for index in range(len(timestamp))])
    return pd.DataFrame({"datetime": timestamp, "open": price, "high": price + 0.02,
                         "low": price - 0.02, "close": price, "volume": 1000.0,
                         "amount": price * 1000.0})


class WeeklyTRecoveryDataTests(unittest.TestCase):
    """检查时间边界、缺价和零量语义、历史主板范围及缓存指纹。"""

    def setUp(self):
        """每个测试使用独立临时目录；默认只放一只历史主板股票。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="weekly_t_recovery_tests_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "raw/2026"
        self.source.mkdir(parents=True)
        self.raw = create_recovery_day()
        self.source_file = self.source / "SH.600000.csv"
        self.raw.to_csv(self.source_file, index=False)
        self.cfg = {"data_root": str(self.source.parent), "start_date": "2026-04-01",
                    "end_date": "2026-04-30", "workers": 1,
                    "recovery_cache_dir": str(self.directory / "cache")}

    def test_exact_six_minutes_and_eight_execution_only_columns(self):
        offsets = self.raw.iloc[[1, 1, 1]].copy()
        offsets["datetime"] = [pd.Timestamp("2026-04-30 14:55:30"),
                                pd.Timestamp("2026-04-30 14:55:00.000001"),
                                pd.Timestamp("2026-04-30 14:55:00.000000001")]
        raw = pd.concat([self.raw, offsets], ignore_index=True)
        original = raw.copy(deep=True)
        bars = select_recovery_bars(raw, "SH.600000")
        self.assertEqual(bars.datetime.dt.strftime("%H:%M").tolist(), RECOVERY_MINUTES)
        self.assertEqual(list(bars), RECOVERY_COLUMNS)
        self.assertEqual(bars.symbol.unique().tolist(), ["SH.600000"])
        self.assertFalse(any(column.startswith("f_") for column in bars))
        pd.testing.assert_frame_equal(raw, original)

    def test_missing_minutes_never_filled_zero_volume_and_flat_bars_preserved(self):
        raw = self.raw.loc[self.raw.datetime.dt.strftime("%H:%M") != "14:57"].copy()
        zero = raw.datetime.dt.strftime("%H:%M") == "14:56"
        raw.loc[zero, ["volume", "amount"]] = 0.0
        raw.loc[zero, ["open", "high", "low", "close"]] = 10.0
        raw.to_csv(self.source_file, index=False)
        bars, manifest = load_recovery_data(self.cfg)
        self.assertEqual(len(bars), 5)
        self.assertNotIn("14:57", bars.datetime.dt.strftime("%H:%M").tolist())
        retained = bars.loc[bars.datetime.dt.strftime("%H:%M") == "14:56"].iloc[0]
        self.assertEqual(retained.volume, 0)
        self.assertEqual(retained.amount, 0)
        self.assertEqual(retained.high, retained.low)
        self.assertEqual(manifest["zero_volume_rows"], 1)
        self.assertEqual(manifest["zero_amount_rows"], 1)

    def test_shared_reader_keeps_latest_duplicate_and_rejects_invalid_price(self):
        duplicate = self.raw.iloc[[1]].copy()
        duplicate[["open", "high", "low", "close"]] += 1.0
        raw = pd.concat([self.raw, duplicate], ignore_index=True)
        raw.loc[raw.datetime.dt.strftime("%H:%M") == "14:58", "low"] = 100.0
        raw.to_csv(self.source_file, index=False)
        bars, manifest = load_recovery_data(self.cfg)
        self.assertEqual(len(bars), 5)
        self.assertAlmostEqual(bars.close.iloc[0], 11.01)
        self.assertEqual(manifest["source_duplicates"], 1)
        self.assertEqual(manifest["source_invalid_bars"], 1)
        self.assertNotIn("14:58", bars.datetime.dt.strftime("%H:%M").tolist())

    def test_history_union_excludes_growth_boards_without_current_snapshot(self):
        historical = self.source.parent / "2023"
        historical.mkdir()
        create_recovery_day("2023-06-30").to_csv(historical / "SZ.000001.csv", index=False)
        self.raw.to_csv(self.source / "SH.688001.csv", index=False)
        self.raw.to_csv(self.source / "SZ.300001.csv", index=False)
        cfg = {**self.cfg, "start_date": "2023-01-01", "name_snapshot": "/must/not/read.csv"}
        bars, manifest = load_recovery_data(cfg)
        self.assertEqual(manifest["symbols"], ["SH.600000", "SZ.000001"])
        self.assertEqual(set(bars.symbol), {"SH.600000", "SZ.000001"})
        self.assertEqual(len(bars), 12)

    def test_cache_load_is_identical_and_does_not_reopen_csv_or_mutate_configuration(self):
        original = deepcopy(self.cfg)
        first, first_manifest = load_recovery_data(self.cfg)
        with patch("quant_backtest.weekly_t_recovery_data.load_weekly_symbol_bars",
                   side_effect=AssertionError("缓存复用不得重读CSV")):
            second, second_manifest = load_recovery_data(self.cfg)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first_manifest["cache_hit"])
        self.assertTrue(second_manifest["cache_hit"])
        self.assertEqual(first_manifest["fingerprint"], second_manifest["fingerprint"])
        self.assertEqual(self.cfg, original)
        self.assertEqual(list((self.directory / "cache").glob("**/*.tmp")), [])
        pd.testing.assert_frame_equal(pd.read_csv(self.source_file, parse_dates=["datetime"]), self.raw)

    def test_source_and_effective_date_or_symbol_changes_invalidate_cache(self):
        _, original = load_recovery_data(self.cfg)
        changed = self.raw.copy()
        changed.loc[1, "amount"] *= 2
        changed.to_csv(self.source_file, index=False)
        _, modified = load_recovery_data(self.cfg)
        self.assertNotEqual(original["fingerprint"], modified["fingerprint"])
        _, date_filtered = load_recovery_data({**self.cfg, "start_date": "2026-04-30"})
        self.assertNotEqual(modified["fingerprint"], date_filtered["fingerprint"])
        self.raw.to_csv(self.source / "SZ.000001.csv", index=False)
        all_bars, all_symbols = load_recovery_data(self.cfg)
        limited_bars, limited = load_recovery_data({**self.cfg, "max_symbols": 1})
        self.assertNotEqual(all_symbols["fingerprint"], limited["fingerprint"])
        self.assertEqual(len(all_bars), 12)
        self.assertEqual(len(limited_bars), 6)

    def test_complete_cache_rejects_metadata_identity_or_count_mismatch(self):
        _, manifest = load_recovery_data(self.cfg)
        path = Path(manifest["cache_path"]) / "metadata.json"
        altered = {**manifest, "fingerprint": "not-the-input-fingerprint"}
        path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "指纹"):
            load_recovery_data(self.cfg)
        altered = {**manifest, "rows": manifest["rows"] + 1}
        path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "行数"):
            load_recovery_data(self.cfg)

    def test_empty_recovery_window_retains_fixed_schema(self):
        raw = self.raw.iloc[[0]].copy()
        raw.to_csv(self.source_file, index=False)
        bars, manifest = load_recovery_data(self.cfg)
        self.assertTrue(bars.empty)
        self.assertEqual(list(bars), RECOVERY_COLUMNS)
        self.assertEqual(manifest["source_symbol_count"], 1)
        self.assertEqual(manifest["symbol_count"], 0)
        self.assertEqual(manifest["rows"], 0)

    def test_invalid_configuration_fails_before_building_cache(self):
        for changes in ({"workers": 0}, {"max_symbols": -1},
                        {"start_date": "2026-05-01", "end_date": "2026-04-30"},
                        {"data_root": str(self.directory / "missing")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_recovery_data({**self.cfg, **changes})


if __name__ == "__main__":
    unittest.main(verbosity=2)
