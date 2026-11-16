"""检验原两方法只读重合比较，使用合成账本，不读取真实测试预测或修改策略。"""

import unittest

import pandas as pd

from scripts.audit_original_method_results import compare_method_trades


def create_frames(values, symbols):
    """用values日终权益及symbols买入股票构造三日账本，初始资金由调用者固定为100。"""
    dates = pd.date_range("2023-02-01 15:00", periods=3)
    return {"curve": pd.DataFrame({"datetime": dates, "equity": values}),
            "events": pd.DataFrame({"datetime": dates[:len(symbols)], "symbol": symbols, "side": "buy"})}


class OriginalMethodResultAuditTests(unittest.TestCase):
    def test_identical_books(self):
        frames = create_frames([100, 101, 99], ["000001.SZ", "000002.SZ"])
        result = compare_method_trades(frames, frames, 100)
        self.assertTrue(result["净值逐点完全相同"])
        self.assertEqual(result["买入交并比"], 1)
        self.assertAlmostEqual(result["日收益相关系数"], 1)

    def test_different_books(self):
        first = create_frames([100, 101, 99], ["000001.SZ", "000002.SZ"])
        second = create_frames([100, 102, 99], ["000001.SZ", "000003.SZ"])
        result = compare_method_trades(first, second, 100)
        self.assertFalse(result["净值逐点完全相同"])
        self.assertEqual(result["共同买入键数"], 1)
        self.assertAlmostEqual(result["买入交并比"], 1 / 3)

    def test_cash_books(self):
        frames = create_frames([100, 100, 100], [])
        result = compare_method_trades(frames, frames, 100)
        self.assertIsNone(result["买入交并比"])
        self.assertIsNone(result["日收益相关系数"])


if __name__ == "__main__":
    unittest.main()
