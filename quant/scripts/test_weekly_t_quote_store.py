"""验证只读mmap恢复行情、缓存身份及spawn共享；仅写临时目录，不读取真实行情。

从quant仓库运行quant环境Python scripts/test_weekly_t_quote_store.py。
"""

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest import weekly_t_engine as engine
from quant_backtest import weekly_t_quote_store as stores
from scripts import test_weekly_t_engine as legacy
from scripts.test_weekly_t_partial import create_recovery


def create_quotes() -> pd.DataFrame:
    """小型不规则完整时段行情；特意缺少部分时间/股票组合。"""
    rows = []
    for timestamp, symbol in [("2026-04-29 09:31", "AAA"), ("2026-04-29 11:30", "AAA"),
                              ("2026-04-29 13:01", "BBB"), ("2026-04-29 15:00", "AAA")]:
        rows.append({"datetime": pd.Timestamp(timestamp), "symbol": symbol,
                     "open": 10.1234567890123, "high": 11., "low": 9., "close": 10.5,
                     "volume": 1000., "amount": 10500.})
    return pd.DataFrame(rows)


def read_spawn_quote(store: stores.DenseRecoveryBarStore) -> dict:
    """spawn子进程仅读一个当前报价并报告映射模式，不回传整个数组。"""
    return {"quote": dict(store.get_quote(pd.Timestamp("2026-04-29 09:31"), "AAA")),
            "mmap": isinstance(store._quotes, np.memmap), "writeable": store._quotes.flags.writeable,
            "row_count": store.row_count, "pickle_bytes": len(pickle.dumps(store))}


class DenseQuoteStoreTests(unittest.TestCase):
    """覆盖共享缓存和真实数据缺失语义，不通过与实现逐行镜像的测试替代执行验证。"""

    def setUp(self):
        """为每项校验隔离临时缓存，避免触碰研究缓存。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.bars = create_quotes()

    def test_engine_instance_passthrough_and_float64_readonly_quotes(self):
        """继承接口让引擎直通，报价保持双精度且无法通过返回值修改存储。"""
        store = stores.build_dense_recovery_store(self.bars, self.cache)
        self.assertIsInstance(store, engine.RecoveryBarStore)
        self.assertIs(engine.prepare_recovery_bars(store, pd.DatetimeIndex([])), store)
        self.assertIsInstance(store._quotes, np.memmap)
        self.assertEqual(store._quotes.shape, (4, 2, 6))
        self.assertEqual(store._quotes.dtype, np.dtype("float64"))
        quote = store.get_quote(pd.Timestamp("2026-04-29 09:31"), "AAA")
        self.assertEqual(quote["open"], self.bars.open.iloc[0])
        self.assertEqual(set(quote), set(stores.QUOTE_COLUMNS))
        with self.assertRaises(TypeError):
            quote["amount"] = 0
        with self.assertRaises(ValueError):
            store._quotes[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            store._quotes.flags.writeable = True

    def test_session_edges_accepted_lunch_and_nonminute_rejected(self):
        """午休/时段外与秒、微秒、纳秒标签都不能被静默取整或过滤。"""
        stores.build_dense_recovery_store(self.bars, self.cache)
        invalid = ["09:30", "11:31", "12:00", "13:00", "15:01", "09:31:01",
                   "09:31:00.000001", "09:31:00.000000001"]
        for clock in invalid:
            with self.subTest(clock=clock):
                changed = self.bars.copy()
                changed.loc[0, "datetime"] = pd.Timestamp(f"2026-04-29 {clock}")
                with self.assertRaises(ValueError):
                    stores.build_dense_recovery_store(changed, self.cache)

    def test_missing_or_timezone_dates_and_empty_symbols_rejected(self):
        """缺失键和带时区日期不能转成看似合法的股票或本地时钟。"""
        for value in [None, "", "   "]:
            changed = self.bars.copy()
            changed.loc[0, "symbol"] = value
            with self.assertRaises(ValueError):
                stores.build_dense_recovery_store(changed, self.cache)
        changed = self.bars.copy()
        changed.loc[0, "datetime"] = pd.NaT
        with self.assertRaises(ValueError):
            stores.build_dense_recovery_store(changed, self.cache)
        changed = self.bars.copy()
        changed["datetime"] = changed.datetime.dt.tz_localize("Asia/Shanghai")
        with self.assertRaises(ValueError):
            stores.build_dense_recovery_store(changed, self.cache)

    def test_duplicate_keys_rejected_after_symbol_normalization(self):
        """重复行及数字/字符串归一后碰撞都必须显式拒绝。"""
        with self.assertRaisesRegex(ValueError, "唯一"):
            stores.build_dense_recovery_store(pd.concat([self.bars, self.bars.iloc[[0]]]), self.cache)
        changed = pd.concat([self.bars.iloc[[0]], self.bars.iloc[[0]]]).reset_index(drop=True)
        changed["symbol"] = [1, "1"]
        with self.assertRaisesRegex(ValueError, "唯一"):
            stores.build_dense_recovery_store(changed, self.cache)

    def test_negative_infinite_and_nonnumeric_values_rejected(self):
        """所有量价列遵循旧Store非负/不含无穷值契约，不默默丢弃坏行。"""
        for column in stores.QUOTE_COLUMNS:
            for value in [-1., np.inf, -np.inf]:
                with self.subTest(column=column, value=value):
                    changed = self.bars.copy()
                    changed.loc[0, column] = value
                    with self.assertRaises(ValueError):
                        stores.build_dense_recovery_store(changed, self.cache)
        changed = self.bars.copy()
        changed["open"] = "bad"
        with self.assertRaises(ValueError):
            stores.build_dense_recovery_store(changed, self.cache)
        changed["open"] = 1 + 2j
        with self.assertRaises(ValueError):
            stores.build_dense_recovery_store(changed, self.cache)

    def test_actual_zero_and_nan_rows_are_distinct_from_missing_rows(self):
        """真实0和全NaN行仍有报价；不存在的(time,symbol)组合才返回None。"""
        changed = self.bars.copy()
        changed.loc[0, list(stores.QUOTE_COLUMNS)] = 0.
        changed.loc[1, list(stores.QUOTE_COLUMNS)] = np.nan
        store = stores.build_dense_recovery_store(changed, self.cache)
        self.assertTrue(all(value == 0 for value in store.get_quote(changed.datetime.iloc[0], "AAA").values()))
        self.assertTrue(all(np.isnan(value) for value in store.get_quote(changed.datetime.iloc[1], "AAA").values()))
        self.assertIsNone(store.get_quote(changed.datetime.iloc[0], "BBB"))
        self.assertIsNone(store.get_quote(pd.Timestamp("2026-04-29 09:32"), "AAA"))

    def test_cache_reuse_preserves_input_and_ignores_row_order_extra_columns(self):
        """相同行情只读复用，额外未来标签不参与报价身份，原输入不被修改。"""
        before = self.bars.copy(deep=True)
        first = stores.build_dense_recovery_store(self.bars, self.cache)
        changed = self.bars.iloc[::-1].assign(future_label=999.)
        stats = {x.name: (x.stat().st_size, x.stat().st_mtime_ns) for x in first.cache_path.iterdir()}
        second = stores.build_dense_recovery_store(changed, self.cache)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.cache_path, second.cache_path)
        self.assertEqual(stats, {x.name: (x.stat().st_size, x.stat().st_mtime_ns) for x in first.cache_path.iterdir()})
        pd.testing.assert_frame_equal(self.bars, before)

    def test_future_quote_changes_identity_but_not_current_lookup(self):
        """未来价格可改变整个缓存身份，但不能回流当前分钟的读取结果。"""
        first = stores.build_dense_recovery_store(self.bars, self.cache)
        changed = self.bars.copy()
        changed.loc[changed.datetime == changed.datetime.max(), list(stores.QUOTE_COLUMNS)] = 100.
        second = stores.build_dense_recovery_store(changed, self.cache)
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(dict(first.get_quote(self.bars.datetime.iloc[0], "AAA")),
                         dict(second.get_quote(self.bars.datetime.iloc[0], "AAA")))

    def test_pickle_is_small_and_does_not_hash_or_copy_payload_on_reload(self):
        """反序列化仅重开mmap及轻量索引，不调用父构造或扫描大型数组。"""
        store = stores.build_dense_recovery_store(self.bars, self.cache)
        state = store.__getstate__()
        self.assertEqual(set(state), {"cache_path", "fingerprint", "metadata_sha256"})
        payload = pickle.dumps(store)
        self.assertLess(len(payload), 2000)
        original_hash = stores.calculate_file_sha256
        paths = []
        def record_hash(path):
            paths.append(path.name)
            return original_hash(path)
        with mock.patch.object(stores, "calculate_file_sha256", side_effect=record_hash), \
             mock.patch.object(engine.RecoveryBarStore, "__init__", side_effect=AssertionError("parent rebuilt")):
            restored = pickle.loads(payload)
        self.assertNotIn("quotes.npy", paths)
        self.assertNotIn("present.npy", paths)
        self.assertIsInstance(restored._quotes, np.memmap)
        self.assertFalse(restored._quotes.flags.writeable)
        self.assertEqual(dict(restored.get_quote(self.bars.datetime.iloc[0], "AAA")),
                         dict(store.get_quote(self.bars.datetime.iloc[0], "AAA")))

    def test_spawn_reopens_readonly_mmap(self):
        """真正spawn进程取得同一行情文件的只读映射，而非序列化数组副本。"""
        store = stores.build_dense_recovery_store(self.bars, self.cache)
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            result = pool.apply(read_spawn_quote, (store,))
        self.assertTrue(result["mmap"])
        self.assertFalse(result["writeable"])
        self.assertEqual(result["row_count"], len(self.bars))
        self.assertEqual(result["quote"]["open"], self.bars.open.iloc[0])
        self.assertLess(result["pickle_bytes"], 2000)

    def test_metadata_tampering_or_missing_array_fails_closed(self):
        """缓存身份损坏或文件缺失不能被当作可用行情。"""
        store = stores.build_dense_recovery_store(self.bars, self.cache)
        metadata_path = store.cache_path / "metadata.json"
        original = metadata_path.read_bytes()
        metadata = json.loads(original)
        metadata["identity"]["input_sha256"] = "incorrect"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            stores.DenseRecoveryBarStore(store.cache_path)
        metadata_path.write_bytes(original)
        (store.cache_path / "present.npy").unlink()
        with self.assertRaises(ValueError):
            stores.DenseRecoveryBarStore(store.cache_path)

    def test_payload_changes_rejected_by_master_and_pickle_reload(self):
        """数组在发布后变化时，完整校验和worker的stat校验均应拒绝。"""
        store = stores.build_dense_recovery_store(self.bars, self.cache)
        payload = pickle.dumps(store)
        target = store.cache_path / "quotes.npy"
        changed = np.load(target, mmap_mode="r+")
        changed[0, 0, 0] = 999.
        changed.flush()
        del changed
        with self.assertRaises(ValueError):
            stores.DenseRecoveryBarStore(store.cache_path)
        with self.assertRaises(ValueError):
            pickle.loads(payload)

    def test_symbol_index_and_identity_are_bound_to_cache_directory(self):
        """即使数组未变，索引改名或自行改写身份摘要也不能把报价分配给其他股票。"""
        store = stores.build_dense_recovery_store(self.bars, self.cache)
        metadata_path = store.cache_path / "metadata.json"
        original = metadata_path.read_bytes()
        metadata = json.loads(original)
        metadata["symbols"] = ["CCC", "DDD"]
        metadata_path.write_bytes(stores.encode_metadata(metadata))
        with self.assertRaises(ValueError):
            stores.DenseRecoveryBarStore(store.cache_path)
        metadata = json.loads(original)
        metadata["identity"]["row_count"] = 3
        metadata["fingerprint"] = stores.hashlib.sha256(stores.encode_metadata(metadata["identity"])).hexdigest()
        metadata_path.write_bytes(stores.encode_metadata(metadata))
        with self.assertRaises(ValueError):
            stores.DenseRecoveryBarStore(store.cache_path)

    def test_source_version_changes_identity_and_old_cache_is_rejected(self):
        """实现来源变化必须构造新缓存，不能悄悄重用旧版本数组。"""
        first = stores.build_dense_recovery_store(self.bars, self.cache)
        changed_hashes = {**stores.calculate_source_hashes(), "weekly_t_quote_store.py": "different_source"}
        with mock.patch.object(stores, "calculate_source_hashes", return_value=changed_hashes):
            second = stores.build_dense_recovery_store(self.bars, self.cache)
            self.assertNotEqual(first.fingerprint, second.fingerprint)
            with self.assertRaises(ValueError):
                stores.DenseRecoveryBarStore(first.cache_path)

    def test_failed_build_never_publishes_partial_cache(self):
        """构建中断清理自己的临时目录，不能留下被其他worker误用的半成品。"""
        with mock.patch.object(stores, "write_array_file", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                stores.build_dense_recovery_store(self.bars, self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_concurrent_builders_publish_one_complete_cache(self):
        """并发构建同一输入只保留一个完整内容寻址目录，败者不覆盖赢家文件。"""
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: stores.build_dense_recovery_store(self.bars, self.cache), range(2)))
        self.assertEqual(results[0].cache_path, results[1].cache_path)
        self.assertEqual(len(list(self.cache.iterdir())), 1)
        self.assertEqual(results[0].row_count, len(self.bars))

    def test_empty_input_and_invalid_query_are_safe(self):
        """空缓存无需映射不存在的数据区；缺失/带时区/非法查询不寻找近邻报价。"""
        empty = stores.build_dense_recovery_store(self.bars.iloc[:0], self.cache)
        self.assertEqual(empty.row_count, 0)
        self.assertEqual(empty.symbols, frozenset())
        self.assertIsNone(empty.get_quote(pd.Timestamp("2026-04-29 09:31"), "AAA"))
        store = stores.build_dense_recovery_store(self.bars, self.cache)
        for value in [None, pd.NaT, "not a date", pd.Timestamp("2026-04-29 09:31", tz="Asia/Shanghai")]:
            self.assertIsNone(store.get_quote(value, "AAA"))

    def test_legacy_tail_execution_matches_dictionary_store(self):
        """真实组合回放比较旧行字典与新mmap，确保接口替换不改变成交和净值。"""
        fixture = legacy.WeeklyTTests()
        fixture.setUp()
        sample, decision = legacy.create_request(side=-1, execution=10.1, fraction=.1)
        bars = create_recovery("2024-01-09", [100, 100, 200, 0, 0, 0])
        old = engine.build_recovery_store(bars)
        new = stores.build_dense_recovery_store(bars, self.cache)
        def evaluate(store):
            return engine.run_weekly_portfolio(fixture.selected, fixture.daily,
                pd.DataFrame([sample]), pd.DataFrame([decision]), fixture.cfg, recovery_bars=store)
        before, after = evaluate(old), evaluate(new)
        pd.testing.assert_frame_equal(before["trade_events"], after["trade_events"])
        pd.testing.assert_frame_equal(before["equity_curve"], after["equity_curve"])
        pd.testing.assert_frame_equal(before["round_trips"], after["round_trips"])


if __name__ == "__main__":
    unittest.main()
