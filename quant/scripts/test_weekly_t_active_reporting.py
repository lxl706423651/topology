"""验证主动报告按末次估值时点识别陈旧价格；仅合成持仓，不运行交易账户。"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quant_backtest.weekly_t_active_reporting import EXECUTION_FAILURE_KEYS, summarize_active_year


def create_reporting_fixture(mark_datetime: str, shares: int = 100) -> tuple[dict, dict, dict]:
    """构造末日只有一个估值点的空成交账本，单独测试持仓价格时间而非成交行为。"""
    metrics = {key: 0 for key in EXECUTION_FAILURE_KEYS}
    metrics.update(t_pair_count=0, weekly_liquidation_failure_days=0, weekly_unclosed_positions=0,
                   baseline_weekly_unclosed_positions=0)
    result = {
        "metrics": metrics,
        "equity_curve": pd.DataFrame({"datetime": [pd.Timestamp("2026-04-30 15:00:00")],
                                      "equity": [1000000.0], "baseline_equity": [1000000.0]}),
        "trade_events": pd.DataFrame(columns=["event_type", "fragment_id", "pair_id", "fee", "slippage_cost"]),
        "round_trips": pd.DataFrame(columns=["pair_id", "pair_fully_recovered", "first_datetime", "second_datetime",
                                             "gross_pnl", "slippage_cost", "buy_fee", "sell_fee", "net_pnl"]),
        "daily_inventory_audit": pd.DataFrame(columns=["date", "inventory_difference", "recovery_failure",
                                                       "weekly_liquidation_failure"]),
        "open_positions": pd.DataFrame({"shares": [shares], "mark_datetime": [pd.Timestamp(mark_datetime)],
                                        "market_value": [shares * 10.0]}),
        "order_attempts": pd.DataFrame(), "baseline_events": pd.DataFrame(),
    }
    return result, {"test_year": 2026, "t_enabled": True}, {
        "initial_cash": 1000000.0, "candidate_activation_policy": "active_execution_valid"}


class ActiveReportingStaleTest(unittest.TestCase):
    """同日开盘价缺收盘、跨日旧价、真实末价和无实际持仓四种边界。"""

    def summarize(self, timestamp: str, shares: int = 100) -> dict:
        """调用只读年度汇总，验证原持仓表不变。"""
        result, annual, cfg = create_reporting_fixture(timestamp, shares)
        before = result["open_positions"].copy(deep=True)
        summary = summarize_active_year(2026, result, annual, Path("/nonexistent_reporting_test"), cfg)
        pd.testing.assert_frame_equal(before, result["open_positions"], check_exact=True)
        return summary

    def test_same_day_open_only_is_stale(self):
        result = self.summarize("2026-04-30 09:30:00")
        self.assertEqual(result["final_stale_position_count"], 1)
        self.assertEqual(result["final_stale_market_value"], 1000.0)

    def test_previous_day_price_is_stale(self):
        result = self.summarize("2026-04-29 15:00:00")
        self.assertEqual(result["final_stale_position_count"], 1)

    def test_exact_final_timestamp_is_current(self):
        result = self.summarize("2026-04-30 15:00:00")
        self.assertEqual(result["final_stale_position_count"], 0)
        self.assertEqual(result["final_stale_market_value"], 0.0)

    def test_baseline_only_row_is_not_actual_stale_position(self):
        result = self.summarize("2026-04-30 09:30:00", shares=0)
        self.assertEqual(result["final_stale_position_count"], 0)
        self.assertEqual(result["final_stale_market_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
