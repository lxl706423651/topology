"""完整分钟缓存的真实缺失/复用/来源变更测试；临时CSV，不读生产大行情。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.dynamic_t_data import load_full_minute_store, audit_signal_execution_coverage
from quant_backtest.weekly_t_quote_store import build_dense_recovery_store, calculate_file_sha256


class DynamicMinuteDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw/2025/SH.600000.csv"
        self.raw.parent.mkdir(parents=True)
        self.days = pd.DatetimeIndex(["2025-01-06", "2025-01-07"])
        self.cfg = {"data_root": str(self.root / "raw"), "full_minute_cache_dir": str(self.root / "cache")}
        self.frame = pd.DataFrame([{"datetime": stamp, "open": 10., "high": 10.1, "low": 9.9,
            "close": price, "volume": 1000., "amount": 10000.} for stamp, price in
            [("2025-01-06 10:08", 10.), ("2025-01-06 10:10", 10.1), ("2025-01-07 15:00", 10.)]])
        self.frame.to_csv(self.raw, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, inputs=None):
        return load_full_minute_store(self.cfg, 2025, ["SH.600000"], self.days,
                                      inputs if inputs is not None else {}, self.root / "audit")

    def test_build_then_reuse_preserves_missing_minute_and_sources(self):
        inputs = {}
        first, audit = self.load(inputs)
        self.assertFalse(audit["cache_hit"])
        self.assertEqual(audit["missing_rows"], 477)
        self.assertIsNone(first.get_quote(pd.Timestamp("2025-01-06 10:09"), "SH.600000"))
        self.assertEqual(first.get_quote(pd.Timestamp("2025-01-06 10:10"), "SH.600000")["close"], 10.1)
        second, again = self.load()
        self.assertTrue(again["cache_hit"])
        self.assertEqual(first.cache_path, second.cache_path)
        self.assertIn(str(self.raw.resolve()), inputs)
        self.assertEqual(len(inputs), 3)

    def test_changed_raw_creates_new_identity_and_preserves_old_cache(self):
        first, audit = self.load()
        old_manifest = Path(audit["manifest_path"]).read_bytes()
        self.frame.loc[0, "close"] = 10.05
        self.frame.to_csv(self.raw, index=False)
        second, changed = self.load()
        self.assertNotEqual(first.cache_path, second.cache_path)
        self.assertNotEqual(audit["request_fingerprint"], changed["request_fingerprint"])
        self.assertEqual(Path(audit["manifest_path"]).read_bytes(), old_manifest)
        self.assertEqual(first.get_quote(pd.Timestamp("2025-01-06 10:08"), "SH.600000")["close"], 10.)

    def test_corrupt_manifest_fails_without_rebuild_or_overwrite(self):
        _, audit = self.load()
        path = Path(audit["manifest_path"])
        value = json.loads(path.read_text())
        value["dense_metadata_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_ohlc_duplicates_and_nonwhole_session_minutes_are_rejected(self):
        variants = [pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True),
                    self.frame.assign(high=9.),
                    self.frame.assign(datetime="2025-01-06 10:08:30")]
        for frame in variants:
            with self.subTest(rows=len(frame)):
                frame.to_csv(self.raw, index=False)
                with self.assertRaises(ValueError):
                    self.load()

    def test_outside_session_rows_are_audited_without_shifting_or_filling(self):
        extra = self.frame.iloc[:1].copy()
        extra["datetime"] = "2025-01-06 13:00:00"
        pd.concat([self.frame, extra], ignore_index=True).to_csv(self.raw, index=False)
        store, audit = self.load()
        self.assertEqual(audit["present_rows"], 3)
        self.assertEqual(audit["per_symbol"][0]["excluded_outside_session_rows"], 1)
        self.assertEqual(audit["per_symbol"][0]["excluded_clock_counts"], {"13:00:00": 1})
        self.assertIsNone(store.get_quote(pd.Timestamp("2025-01-06 13:00"), "SH.600000"))
        self.assertIsNone(store.get_quote(pd.Timestamp("2025-01-06 13:01"), "SH.600000"))

    def test_self_consistent_manifest_cannot_repoint_to_other_prices(self):
        _, audit = self.load()
        changed = self.frame.copy()
        changed["symbol"] = "SH.600000"
        changed.loc[0, "close"] = 10.05
        alternate = build_dense_recovery_store(changed, self.root / "alternate")
        manifest = Path(audit["manifest_path"])
        value = json.loads(manifest.read_text())
        value["dense_cache_path"] = str(alternate.cache_path)
        value["dense_metadata_sha256"] = calculate_file_sha256(alternate.cache_path / "metadata.json")
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "数值与已绑定原始切片不同"):
            self.load()

    def test_execution_alignment_rejects_bad_values_or_missing_rows_without_filtering(self):
        store, _ = self.load()
        signals = pd.DataFrame({"symbol": ["SH.600000"] * 2,
            "execution_time": pd.to_datetime(["2025-01-06 10:08", None]),
            "execution_price": [10., float("nan")], "execution_high": [10.1, float("nan")],
            "execution_low": [9.9, float("nan")], "execution_volume": [1000., float("nan")],
            "execution_amount": [10000., float("nan")]})
        original = signals.copy(deep=True)
        audit = audit_signal_execution_coverage(signals, store)
        self.assertEqual(audit["verified_execution_rows"], 1)
        self.assertEqual(audit["missing_original_execution_times_retained"], 1)
        pd.testing.assert_frame_equal(signals, original)
        with self.assertRaises(ValueError):
            audit_signal_execution_coverage(signals.assign(execution_price=9.), store)
        signals.loc[0, "execution_time"] = pd.Timestamp("2025-01-06 10:09")
        with self.assertRaises(ValueError):
            audit_signal_execution_coverage(signals, store)


if __name__ == "__main__":
    unittest.main(verbosity=2)
