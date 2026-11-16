"""用合成周度选股、真实时刻价量样本验证周度底仓做 T；不读写行情或训练模型。

在仓库根目录执行 python scripts/test_weekly_t_engine.py；覆盖下一分钟成交、周末
T+1 预算、集合竞价平价成交、缺价/容量失败、跨周滞留及独立基准资金计划。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_engine import run_weekly_portfolio


def create_daily(dates, symbol="TEST", price=10.0) -> pd.DataFrame:
    """构造普通有量集合竞价平价收盘，首分钟非平价；可由测试局部替换成交条件。"""
    return pd.DataFrame([{"date": date, "symbol": symbol, "open": price, "close": price,
                          "previous_close": price, "open_amount": 2_000_000.0,
                          "open_high": price + 0.01, "open_low": price - 0.01,
                          "close_amount": 2_000_000.0, "close_high": price, "close_low": price}
                         for date in pd.to_datetime(dates)])


def create_selection(week="2024-01-08", exit_date="2024-01-12", previous_close=10.0):
    """创建上周末已知参考价的独立周度计划。"""
    return {"trade_week": pd.Timestamp(week), "entry_date": pd.Timestamp(week),
            "exit_date": pd.Timestamp(exit_date), "symbol": "TEST", "rank": 1,
            "universe_previous_close": previous_close}


def create_request(day="2024-01-09", side=-1, execution=10.4, recovery=10.0, fraction=0.5, minute="10:00"):
    """返回预先冻结决策和对应分钟样本；恢复价量可缺失，不应影响首腿是否发起。"""
    signal = pd.Timestamp(f"{day} {minute}")
    sample = {"datetime": signal, "date": signal.normalize(), "symbol": "TEST",
              "execution_time": signal + pd.Timedelta(minutes=1), "execution_price": execution,
              "execution_high": execution + 0.01, "execution_low": execution - 0.01,
              "execution_amount": 2_000_000.0, "recovery_time": signal.normalize() + pd.Timedelta(hours=14, minutes=55),
              "recovery_price": recovery, "recovery_high": recovery, "recovery_low": recovery,
              "recovery_amount": 2_000_000.0, "previous_close": 10.0}
    decision = {"datetime": signal, "symbol": "TEST", "action": side, "fraction": fraction}
    return sample, decision


class WeeklyTTests(unittest.TestCase):
    """以手算价差和可卖额度验证账户结果，不复制引擎内部实现。"""

    def setUp(self):
        """默认五日一周、底仓5000股、免成本，便于逐笔核对价差现金。"""
        self.dates = pd.bdate_range("2024-01-08", "2024-01-12")
        self.daily = create_daily(self.dates)
        self.selected = pd.DataFrame([create_selection()])
        self.cfg = {"initial_cash": 100_000.0, "base_fraction": 0.5, "top_n": 1,
                    "commission_rate": 0.0, "min_commission": 0.0, "transfer_rate": 0.0,
                    "stamp_rate_before_20230828": 0.0, "stamp_rate_from_20230828": 0.0,
                    "slippage_rate": 0.0, "max_bar_amount_fraction": 1.0}

    def run_case(self, requests=(), daily=None, selected=None, cfg=None):
        """运行独立的冻结计划及稀疏分钟输入；空决策代表相同周度纯持有。"""
        samples = pd.DataFrame([request[0] for request in requests])
        decisions = pd.DataFrame([request[1] for request in requests])
        return run_weekly_portfolio(self.selected if selected is None else selected,
                                    self.daily if daily is None else daily, samples, decisions,
                                    {**self.cfg, **(cfg or {})})

    def assert_safe(self, result, preserved=True):
        """核查真实成交合法性；仅成功情形要求全部每日股数目标达成。"""
        events, audit = result["trade_events"], result["daily_inventory_audit"]
        self.assertEqual(result["metrics"]["t1_violation_count"], 0)
        self.assertTrue((events.cash_after >= -1e-7).all())
        self.assertTrue((events.shares % 100 == 0).all())
        self.assertTrue((audit.day_sold <= audit.opening_sellable).all())
        self.assertTrue((events.datetime > events.signal_datetime).all())
        if preserved:
            self.assertTrue(audit.inventory_preserved.all())
            self.assertEqual(result["metrics"]["base_plan_mismatch_count"], 0)

    def test_sell_first_next_minute_recovery_and_weekly_close(self):
        result = self.run_case([create_request()])
        trip = result["round_trips"].iloc[0]
        self.assertEqual(trip.sell_datetime, pd.Timestamp("2024-01-09 10:01"))
        self.assertEqual(trip.buy_datetime, pd.Timestamp("2024-01-09 14:55"))
        self.assertEqual(trip.shares, 2500)
        self.assertAlmostEqual(trip.net_pnl, 1000.0)
        self.assertAlmostEqual(result["metrics"]["total_t_marked_pnl"], 1000.0)
        self.assertEqual(len(result["open_positions"]), 0)
        self.assertEqual(result["trade_events"].iloc[-1].datetime, pd.Timestamp("2024-01-12 15:00"))
        self.assert_safe(result)

    def test_buy_first_sells_yesterday_shares(self):
        result = self.run_case([create_request(side=1, execution=9.6)])
        self.assertAlmostEqual(result["round_trips"].iloc[0].net_pnl, 1000.0)
        event = result["trade_events"].loc[result["trade_events"].event_type == "t"].iloc[-1]
        self.assertEqual(event.sellable_before, 5000)
        self.assertEqual(event.sellable_after, 2500)
        self.assert_safe(result)

    def test_first_and_last_day_forbid_t(self):
        result = self.run_case([create_request(day="2024-01-08"), create_request(day="2024-01-12")])
        self.assertEqual(result["metrics"]["t_pair_count"], 0)
        reasons = set(result["order_attempts"].reason)
        self.assertIn("no_opening_t1_inventory", reasons)
        self.assertIn("weekly_exit_day_t1_reserve", reasons)
        self.assertEqual(result["metrics"]["weekly_unclosed_positions"], 0)
        self.assert_safe(result)

    def test_only_one_pair_started_per_day(self):
        result = self.run_case([create_request(), create_request(minute="11:00")])
        self.assertEqual(result["metrics"]["t_pair_count"], 1)
        self.assertEqual(result["daily_inventory_audit"].cycles_started.sum(), 1)
        self.assert_safe(result)

    def test_normal_flat_auction_fills_and_directional_limit_blocks(self):
        result = self.run_case([create_request()])
        self.assertGreaterEqual(result["metrics"]["flat_bar_fill_count"], 2)
        daily = self.daily.copy()
        mask = daily.date == pd.Timestamp("2024-01-12")
        daily.loc[mask, ["close", "close_high", "close_low"]] = 9.5
        blocked = self.run_case(daily=daily)
        self.assertEqual(blocked["metrics"]["weekly_unclosed_positions"], 1)
        self.assertIn("conservative_price_limit_proxy", set(blocked["order_attempts"].reason))
        # 涨停方向不阻卖出，普通平价反向交易也无需虚构流动性判定。
        daily.loc[mask, ["close", "close_high", "close_low"]] = 10.5
        allowed = self.run_case(daily=daily)
        self.assertEqual(allowed["metrics"]["weekly_unclosed_positions"], 0)

    def test_fixed_top_n_denominator_and_past_reference_sizing(self):
        daily = self.daily.copy()
        daily.loc[0, ["open", "open_high", "open_low"]] = [10.4, 10.41, 10.39]
        result = self.run_case(daily=daily, cfg={"top_n": 2})
        plan = result["weekly_entry_plans"].iloc[0]
        self.assertEqual(plan.known_reference_price, 10.0)
        self.assertEqual(plan.planned_shares, 2500)
        self.assertEqual(plan.entry_budget, 25_000.0)
        self.assertEqual(result["trade_events"].iloc[0].shares, 2500)
        self.assertAlmostEqual(result["trade_events"].iloc[0].price, 10.4)

    def test_flat_six_percent_auction_allowed_but_exact_five_percent_buy_blocked(self):
        normal = self.run_case([create_request(recovery=10.6)])
        self.assertEqual(normal["round_trips"].iloc[0].buy_datetime, pd.Timestamp("2024-01-09 14:55"))
        candidate_st_limit = self.run_case([create_request(recovery=10.5)])
        self.assertEqual(candidate_st_limit["round_trips"].iloc[0].buy_datetime, pd.Timestamp("2024-01-09 15:00"))
        attempts = candidate_st_limit["order_attempts"]
        self.assertTrue(((attempts.side == "buy") & (attempts.reason == "conservative_price_limit_proxy")).any())
        self.assert_safe(normal)
        self.assert_safe(candidate_st_limit)

    def test_capacity_shrinks_entry_and_first_leg_only_on_current_bar(self):
        daily = self.daily.copy()
        daily.loc[0, "open_amount"] = 26_000.0
        sample, decision = create_request(fraction=1.0)
        sample["execution_amount"] = 10.4 * 900
        sample["recovery_amount"] = 0.0
        result = self.run_case([(sample, decision)], daily=daily)
        event = result["trade_events"].loc[result["trade_events"].event_type == "t"].iloc[0]
        self.assertEqual(result["weekly_entry_plans"].iloc[0].planned_shares, 5000)
        self.assertEqual(result["trade_events"].iloc[0].shares, 2600)
        self.assertEqual(event.shares, 900)
        self.assertEqual(result["round_trips"].iloc[0].buy_datetime, pd.Timestamp("2024-01-09 15:00"))
        self.assert_safe(result)

    def test_missing_recovery_no_future_filter_and_next_open_restore(self):
        sample, decision = create_request()
        sample["recovery_price"] = np.nan
        daily = self.daily.copy()
        daily.loc[daily.date == pd.Timestamp("2024-01-09"), "close_amount"] = 0.0
        result = self.run_case([(sample, decision)], daily=daily)
        self.assertEqual(result["round_trips"].iloc[0].buy_datetime, pd.Timestamp("2024-01-10 09:30"))
        self.assertEqual(result["metrics"]["recovery_failure_days"], 1)
        self.assertEqual(result["metrics"]["inventory_mismatch_days"], 2)
        self.assert_safe(result, preserved=False)

    def test_weekly_exit_failure_carries_to_next_week_open(self):
        daily = create_daily(pd.bdate_range("2024-01-08", "2024-01-15"))
        daily.loc[daily.date == pd.Timestamp("2024-01-12"), "close_amount"] = 0.0
        result = self.run_case(daily=daily)
        self.assertEqual(result["trade_events"].iloc[-1].datetime, pd.Timestamp("2024-01-15 09:30"))
        self.assertEqual(result["metrics"]["weekly_exit_failure_days"], 1)
        self.assertEqual(result["metrics"]["weekly_unclosed_positions"], 0)
        self.assert_safe(result)

    def test_t_debt_weekend_repair_cannot_sell_newly_rebought_shares(self):
        daily = create_daily(pd.bdate_range("2024-01-08", "2024-01-15"))
        sample, decision = create_request()
        sample["recovery_amount"] = 0.0
        daily.loc[daily.date.between("2024-01-09", "2024-01-11"), "close_amount"] = 0.0
        daily.loc[daily.date.between("2024-01-10", "2024-01-12"), "open_amount"] = 0.0
        result = self.run_case([(sample, decision)], daily=daily)
        friday = result["daily_inventory_audit"].loc[result["daily_inventory_audit"].date == pd.Timestamp("2024-01-12")].iloc[0]
        self.assertEqual(friday.closing_shares, 2500)
        self.assertEqual(friday.weekly_sells, 2500)
        self.assertEqual(friday.weekly_sell_target, 5000)
        self.assertGreater(result["metrics"]["base_plan_mismatch_count"], 0)
        self.assertEqual(result["metrics"]["weekly_unclosed_positions"], 0)
        self.assert_safe(result, preserved=False)

    def test_t_profit_never_compounds_into_next_week_base_size(self):
        daily = create_daily(pd.bdate_range("2024-01-08", "2024-01-19"))
        selected = pd.DataFrame([create_selection(), create_selection("2024-01-15", "2024-01-19")])
        result = self.run_case([create_request()], daily=daily, selected=selected)
        buys = result["trade_events"].loc[(result["trade_events"].event_type == "weekly") & (result["trade_events"].side == "buy")]
        self.assertEqual(list(buys.shares), [5000, 5000])
        self.assertAlmostEqual(result["metrics"]["total_t_marked_pnl"], 1000.0)
        self.assert_safe(result)

    def test_actual_cash_shortfall_rejects_baseline_order_and_is_audited(self):
        dates = pd.bdate_range("2024-01-08", "2024-01-19")
        daily = create_daily(dates)
        # 明示限价隔离本例的资金测试，不让价格保护先挡住预定周末卖出。
        daily["limit_down"] = 1.0
        mask = daily.date >= pd.Timestamp("2024-01-12")
        daily.loc[mask, ["close", "close_high", "close_low"]] = 8.0
        after = daily.date >= pd.Timestamp("2024-01-15")
        daily.loc[after, ["open", "open_high", "open_low", "previous_close"]] = [8.0, 8.01, 7.99, 8.0]
        selected = pd.DataFrame([create_selection(), create_selection("2024-01-15", "2024-01-19", 8.0)])
        sample, decision = create_request(execution=10.0, recovery=10.49, fraction=1.0)
        sample["recovery_high"], sample["recovery_low"] = 10.50, 10.48
        result = self.run_case([(sample, decision)], daily=daily,
                               selected=selected, cfg={"base_fraction": 0.95, "repurchase_buffer": 0.0})
        attempts = result["order_attempts"]
        self.assertTrue(((attempts.account == "actual") & (attempts.reason == "cash_or_t_repurchase_reserve")).any())
        self.assertGreater(result["metrics"]["base_plan_mismatch_count"], 0)
        self.assertTrue((result["equity_curve"].cash >= 0).all())
        self.assertEqual(result["weekly_entry_plans"].iloc[-1].actual_filled_shares, 0)

    def test_fee_and_slippage_net_increment_match_pair_cash_profit(self):
        result = self.run_case([create_request()], cfg={"commission_rate": 0.0008, "min_commission": 5.0,
                                                      "transfer_rate": 0.00001, "stamp_rate_from_20230828": 0.0005,
                                                      "slippage_rate": 0.0005})
        trip = result["round_trips"].iloc[0]
        expected = trip.shares * (10.4 * 0.9995 - 10.0 * 1.0005) - trip.buy_fee - trip.sell_fee
        self.assertAlmostEqual(trip.net_pnl, expected)
        self.assertAlmostEqual(result["metrics"]["total_t_marked_pnl"], expected)
        self.assert_safe(result)

    def test_invalid_same_minute_execution_rejected(self):
        sample, decision = create_request()
        sample["execution_time"] = sample["datetime"]
        with self.assertRaisesRegex(ValueError, "下一交易分钟"):
            self.run_case([(sample, decision)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
