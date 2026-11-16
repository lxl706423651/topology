"""小缓存验证新数据复用时的身份/完整日历和未来标签剥离，不读取真实大缓存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from quant_backtest.balanced_t_data import INTRADAY_COLUMNS, load_balanced_market, record_balanced_input


class BalancedTDataTests(unittest.TestCase):
    """防止缓存被改后静默复用以及只按持仓日期缩小评价分母。"""

    def test_market_cache_identity_calendar_and_feature_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, report = root / "cache", root / "report"
            cache.mkdir()
            report.mkdir()
            dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
            daily = pd.DataFrame({"date": dates, "symbol": ["SH.600000"] * 2})
            data = {column: [1., 1.] for column in INTRADAY_COLUMNS}
            data.update({"datetime": dates + pd.Timedelta(hours=10), "date": dates,
                         "symbol": ["SH.600000"] * 2, "trade_week": dates,
                         "execution_time": dates + pd.Timedelta(hours=10, minutes=1),
                         "target_return": [999., -999.], "recovery_price": [1000., .1]})
            pd.DataFrame(data).to_pickle(cache / "intraday.pkl")
            daily.to_pickle(cache / "daily.pkl")
            known = {str(path.resolve()): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in cache.glob("*.pkl")}
            (report / "input_sources.json").write_text(json.dumps(known))
            (cache / "metadata.json").write_text(json.dumps({"calendar": {"trading_days": 2}}))
            cfg = {"market_cache_dir": str(cache), "source_audit_report_dir": str(report)}
            actual_daily, signals, calendar = load_balanced_market(cfg, {})
            self.assertEqual(len(calendar), 2)
            self.assertEqual(len(actual_daily), 2)
            self.assertEqual(signals.columns.tolist(), INTRADAY_COLUMNS)
            self.assertNotIn("target_return", signals)
            self.assertNotIn("recovery_price", signals)
            daily.iloc[:1].to_pickle(cache / "daily.pkl")
            with self.assertRaisesRegex(ValueError, "已审计来源"):
                load_balanced_market(cfg, {})

    def test_record_input_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.json"
            path.write_text("original")
            inputs = {}
            record_balanced_input(path, inputs)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "期间输入发生变化"):
                record_balanced_input(path, inputs)


if __name__ == "__main__":
    unittest.main()
