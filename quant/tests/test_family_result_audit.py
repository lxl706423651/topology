"""用内存账户和故意损坏的CSV形状检查独立审计，不读取真实测试预测或改动报告。"""

import unittest

import numpy as np
import pandas as pd

from quant_backtest.annual_portfolio import MarketStore, run_lot_portfolio
from scripts.audit_family_path_results import audit_account_frames


def create_account_frames(cash_only: bool = False, partial_exit: bool = True) -> tuple[dict, dict]:
    """生成三日内存账户；cash_only控制空仓，partial_exit控制税率变更后是否部分卖出。"""
    times = pd.to_datetime(["2023-08-25 09:31", "2023-08-25 15:00", "2023-08-28 09:31",
                            "2023-08-28 15:00", "2023-08-29 09:31", "2023-08-29 15:00"])
    raw = pd.DataFrame({"datetime": times, "symbol": "SH.600000", "close": [10., 10.1, 10.2, 10.2, 10.3, 10.3],
        "atr": 1., "volume": 100000., "amount": 1000000., "execution_volume": 100000.,
        "execution_amount": [200000., 200000., 2100., 2100., 2100., 2100.], "previous_close": 10.})
    raw["interval_high"] = raw.close
    raw["execution_close"] = raw.close
    if not partial_exit:
        raw["execution_amount"] = 200000.
    cfg = {"initial_cash": 100000., "commission_rate": .0008, "min_commission": 0., "transfer_rate": .00001,
           "stamp_rate_before_20230828": .001, "stamp_rate_from_20230828": .0005, "slippage_rate": .0005,
           "lot_size": 100, "max_bar_amount_fraction": 1., "entry_top_n": 5, "entry_equity_fraction": .1,
           "min_cross_section": 1}
    scores = raw.iloc[:1][["datetime", "symbol"]].assign(prediction=-1. if cash_only else 1.)
    result = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2023-08-25"), pd.Timestamp("2023-08-30"),
        "prediction", {"threshold": 0., "initial": 20., "trailing": 40., "max_days": 1}, cfg)
    frames = {k: result[k].copy() for k in ["curve", "events", "trips", "open", "holdings"]}
    frames["metrics"] = pd.DataFrame([result["metrics"]])
    return frames, cfg


class FamilyResultAuditTests(unittest.TestCase):
    """证明完整/空成交能够核对，并能识别费用、信号、库存、盈亏和指标损坏。"""

    def test_partial_inventory_and_historical_fees(self):
        """部分卖出及税率变更不应丢失剩余成本，未完全平仓不计批次胜率。"""
        frames, cfg = create_account_frames()
        self.assertEqual(len(frames["trips"]), 2)
        self.assertEqual(len(frames["open"]), 1)
        result = audit_account_frames(frames, cfg)
        self.assertTrue(result["passed"])
        self.assertEqual(result["closed_lot_count"], 0)
        self.assertIsNone(result["mean_final_close_days"])
        self.assertLess(result["cash_error"], 1e-7)

    def test_empty_csv_columns_preserve_undefined_sharpe(self):
        """模拟空CSV全object列，现金年夏普未定义而不是删除该年。"""
        frames, cfg = create_account_frames(cash_only=True)
        for key in ["events", "trips", "open", "holdings"]:
            frames[key] = pd.DataFrame(columns=frames[key].columns)
        result = audit_account_frames(frames, cfg)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["sharpe"])

    def test_completed_lot_and_final_holding_duration(self):
        """完全退出时按批次计胜率，并以最后卖出减买入计算自然日。"""
        frames, cfg = create_account_frames(partial_exit=False)
        result = audit_account_frames(frames, cfg)
        self.assertEqual(result["closed_lot_count"], 1)
        self.assertEqual(result["win_rate"], 1.)
        self.assertEqual(result["mean_final_close_days"], 3.)
        self.assertEqual(result["pending_lots"], 0)

    def test_tampered_ledgers_and_metrics_rejected(self):
        """不同表分别损坏必须失败，不能只校验回测器自带的passed标志。"""
        for key, column, increment in [("events", "fee", 1.), ("curve", "cash", 1.),
            ("trips", "net_pnl", 1.), ("open", "shares", 100), ("holdings", "mark_price", .01),
            ("metrics", "sharpe", .1)]:
            with self.subTest(table=key, column=column):
                frames, cfg = create_account_frames()
                frames[key].loc[0, column] += increment
                with self.assertRaises(AssertionError):
                    audit_account_frames(frames, cfg)

    def test_lag_t1_and_budget_rejected(self):
        """同刻成交、同日平仓、超10%预算均不得被最终盈利掩盖。"""
        for change in ["lag", "t1", "budget"]:
            frames, cfg = create_account_frames()
            if change == "lag":
                frames["events"].loc[0, "signal_datetime"] = frames["events"].loc[0, "datetime"]
            elif change == "t1":
                frames["trips"].loc[0, "buy_datetime"] = frames["trips"].loc[0, "sell_datetime"]
            else:
                frames["events"].loc[0, "signal_equity"] = 1000.
            with self.subTest(change=change), self.assertRaises(AssertionError):
                audit_account_frames(frames, cfg)

    def test_nonfinite_price_rejected(self):
        """NaN不应通过金额或价格校验。"""
        frames, cfg = create_account_frames()
        frames["events"].loc[0, "price"] = np.nan
        with self.assertRaises(AssertionError):
            audit_account_frames(frames, cfg)


if __name__ == "__main__":
    unittest.main()
