"""验证当前信号交易资格缓存的因果字段、失败关闭及独立缓存复用，不读取真实行情。

从项目根目录以quant环境Python运行本脚本；所有合成单股票pkl及输出均位于自动
清理的临时目录，不修改既有年度面板或原始行情，不训练模型或连接外部服务。
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.inventory_t_data import apply_signal_eligibility, load_signal_eligibility


class InventoryTDataTests(unittest.TestCase):
    """独立检查当前信息与未来成交信息分离、样本键完整性及缓存来源追踪。"""

    def setUp(self):
        """建立两个单股缓存；当前hl_range决定资格，未来执行栏故意与它相反。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="inventory_t_data_tests_")
        self.root = Path(self.temporary.name)
        self.source = self.root / "annual_cache"
        self.source.mkdir()
        self.cfg = {"signal_cache_dir": str(self.root / "own_cache"), "min_cross_section": 2}
        self.manifest = {"cache_path": str(self.source), "factor_code_sha256": "factor-test",
                         "panel_code_sha256": "panel-test"}
        self.dates = pd.to_datetime(["2026-01-05 09:31", "2026-01-05 09:55", "2026-01-05 10:19"])
        frames = []
        for symbol, ranges in (("SH.600001", [0.0, 0.02, np.nan]),
                               ("SZ.000001", [0.03, np.inf, -0.01])):
            frame = pd.DataFrame({"datetime": self.dates, "symbol": symbol, "hl_range": ranges,
                                  "forward_return": [0.9, -0.9, 0.0],
                                  "label_end": self.dates + pd.Timedelta(days=1)})
            # 用与当前分钟相反的未来执行状态验证资格不偷看下一分钟能否成交。
            market = pd.DataFrame({"datetime": self.dates, "symbol": symbol,
                                   "execution_high": [99.0, 1.0, 99.0],
                                   "execution_low": [1.0, 1.0, 1.0],
                                   "execution_volume": [10000.0, 0.0, 10000.0]})
            pd.to_pickle({"panel": frame, "market": market}, self.source / f"{symbol}.pkl")
            frames.append(frame[["datetime", "symbol"]])
        self.panel = pd.concat(frames, ignore_index=True)
        self.panel["cs_rank_momentum"] = np.arange(len(self.panel)) / len(self.panel)
        self.panel["target_rank"] = 1 - self.panel.cs_rank_momentum
        self.panel["cross_section_count"] = 2

    def tearDown(self):
        """删除本用例临时合成数据和辅助缓存。"""
        self.temporary.cleanup()

    def test_only_finite_positive_current_range_permits_signal(self):
        flags, metadata = load_signal_eligibility(self.manifest, self.cfg)
        permitted = flags.loc[flags.signal_bar_tradable]
        expected = {(pd.Timestamp("2026-01-05 09:31"), "SZ.000001"),
                    (pd.Timestamp("2026-01-05 09:55"), "SH.600001")}
        self.assertEqual(set(zip(permitted.datetime, permitted.symbol)), expected)
        self.assertEqual(metadata["rows"], 6)
        self.assertEqual(metadata["tradable_rows"], 2)
        self.assertEqual(metadata["blocked_rows"], 4)
        self.assertEqual(metadata["source_stock_count"], 2)
        self.assertFalse(any("execution" in column or "label" in column for column in flags))

    def test_cache_build_then_load_reads_only_auxiliary_pickle(self):
        expected, first = load_signal_eligibility(self.manifest, self.cfg)
        reader = pd.read_pickle
        with patch("quant_backtest.inventory_t_data.pd.read_pickle", wraps=reader) as read:
            loaded, second = load_signal_eligibility(self.manifest, self.cfg)
        pd.testing.assert_frame_equal(expected, loaded)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(read.call_count, 1)
        self.assertEqual(Path(read.call_args.args[0]), Path(first["cache_path"]))
        self.assertTrue(Path(first["metadata_path"]).exists())
        self.assertEqual(list((self.root / "own_cache").glob("*.tmp")), [])

    def test_changing_future_prices_labels_cannot_change_current_flags(self):
        first, before = load_signal_eligibility(self.manifest, self.cfg)
        for path in self.source.glob("*.pkl"):
            payload = pd.read_pickle(path)
            payload["panel"]["forward_return"] = [1e9, -1e9, np.nan]
            payload["panel"]["label_end"] += pd.Timedelta(days=300)
            payload["market"]["execution_volume"] = 0
            payload["market"]["execution_high"] = 1
            payload["market"]["execution_low"] = 1
            pd.to_pickle(payload, path)
        second, after = load_signal_eligibility(self.manifest, self.cfg)
        pd.testing.assert_frame_equal(first, second)
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])

    def test_source_current_range_change_invalidates_cache_and_changes_flag(self):
        _, first = load_signal_eligibility(self.manifest, self.cfg)
        path = self.source / "SH.600001.pkl"
        payload = pd.read_pickle(path)
        payload["panel"].loc[0, "hl_range"] = 0.01
        pd.to_pickle(payload, path)
        flags, second = load_signal_eligibility(self.manifest, self.cfg)
        self.assertFalse(second["cache_hit"])
        self.assertNotEqual(first["cache_path"], second["cache_path"])
        self.assertEqual(int(flags.signal_bar_tradable.sum()), 3)

    def test_filter_recounts_remaining_assets_but_preserves_original_ranks_and_input(self):
        flags, _ = load_signal_eligibility(self.manifest, self.cfg)
        original = self.panel.copy(deep=True)
        original_flags = flags.copy(deep=True)
        result = apply_signal_eligibility(self.panel, flags, self.cfg)
        self.assertEqual(len(result), 2)
        # 虽低于min_cross_section，仍保留给后续训练/排序统一处理。
        self.assertTrue((result.cross_section_count == 1).all())
        pd.testing.assert_frame_equal(result[["cs_rank_momentum", "target_rank"]],
                                      original.loc[result.index, ["cs_rank_momentum", "target_rank"]])
        pd.testing.assert_frame_equal(self.panel, original)
        pd.testing.assert_frame_equal(flags, original_flags)

    def test_missing_flags_and_nonboolean_or_invalid_positive_claims_fail_closed(self):
        flags, _ = load_signal_eligibility(self.manifest, self.cfg)
        # 删除一个有资格记录，同时把无资格记录伪装为True；价差校验仍不允许它通过。
        remaining = flags.loc[~((flags.symbol == "SZ.000001") & (flags.datetime == self.dates[0]))].copy()
        remaining.loc[remaining.signal_hl_range == 0, "signal_bar_tradable"] = True
        result = apply_signal_eligibility(self.panel, remaining, self.cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.symbol.iloc[0], "SH.600001")
        invalid = flags.copy()
        invalid["signal_bar_tradable"] = "False"
        self.assertTrue(apply_signal_eligibility(self.panel, invalid, self.cfg).empty)
        invalid["signal_bar_tradable"] = pd.Series(pd.NA, index=invalid.index, dtype="boolean")
        self.assertTrue(apply_signal_eligibility(self.panel, invalid, self.cfg).empty)

    def test_duplicate_flags_and_duplicate_panel_keys_raise(self):
        flags, _ = load_signal_eligibility(self.manifest, self.cfg)
        with self.assertRaisesRegex(ValueError, "重复"):
            apply_signal_eligibility(self.panel, pd.concat([flags, flags.iloc[[0]]]), self.cfg)
        with self.assertRaisesRegex(ValueError, "重复"):
            apply_signal_eligibility(pd.concat([self.panel, self.panel.iloc[[0]]]), flags, self.cfg)

    def test_duplicate_or_missing_source_fields_raise_without_publishing_bad_cache(self):
        path = self.source / "SH.600001.pkl"
        payload = pd.read_pickle(path)
        payload["panel"] = pd.concat([payload["panel"], payload["panel"].iloc[[0]]])
        pd.to_pickle(payload, path)
        with self.assertRaisesRegex(ValueError, "重复"):
            load_signal_eligibility(self.manifest, self.cfg)
        self.assertEqual(list((self.root / "own_cache").glob("*.pkl")), [])
        payload["panel"] = payload["panel"].drop(columns="hl_range")
        pd.to_pickle(payload, path)
        with self.assertRaisesRegex(ValueError, "缺少"):
            load_signal_eligibility(self.manifest, self.cfg)

    def test_source_files_and_caller_manifest_are_unchanged(self):
        original_manifest = copy.deepcopy(self.manifest)
        original_cfg = copy.deepcopy(self.cfg)
        before = {path.name: path.read_bytes() for path in self.source.glob("*.pkl")}
        load_signal_eligibility(self.manifest, self.cfg)
        after = {path.name: path.read_bytes() for path in self.source.glob("*.pkl")}
        self.assertEqual(before, after)
        self.assertEqual(self.manifest, original_manifest)
        self.assertEqual(self.cfg, original_cfg)
        self.assertTrue(all(path.parent == self.root / "own_cache" for path in (self.root / "own_cache").iterdir()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
