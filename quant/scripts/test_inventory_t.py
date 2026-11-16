"""用独立合成分钟行情验证库存做 T 的因果成交、T+1 和真实失败记账；不读写数据。

从仓库根目录执行 python scripts/test_inventory_t.py；不训练模型、不连接外部服务。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.inventory_t import InventoryTStrategy, run_inventory_t


def create_bars(day: str, overrides: dict[str, float] | None = None,
                missing: list[str] | None = None, locked: list[str] | None = None) -> pd.DataFrame:
    """生成收盘标签的一分钟 OHLCV；价格默认为 10，支持局部改变和缺失/一字线。"""
    index = pd.date_range(f"{day} 09:31", f"{day} 11:30", freq="min")
    index = index.append(pd.date_range(f"{day} 13:01", f"{day} 15:00", freq="min"))
    close = np.array([(overrides or {}).get(t.strftime("%H:%M"), 10.0) for t in index])
    frame = pd.DataFrame({"datetime": index, "open": close, "high": close + 0.01,
                          "low": close - 0.01, "close": close, "volume": 1_000_000.0,
                          "amount": close * 1_000_000.0})
    for minute in locked or []:
        mask = frame.datetime == pd.Timestamp(f"{day} {minute}")
        frame.loc[mask, "high"] = frame.loc[mask, "close"]
        frame.loc[mask, "low"] = frame.loc[mask, "close"]
    if missing:
        frame = frame.loc[~frame.datetime.isin(pd.to_datetime([f"{day} {t}" for t in missing]))]
    return frame.reset_index(drop=True)


class InventoryTTests(unittest.TestCase):
    """以确定价格路径检查经济收益、时序和库存预算，而不是镜像实现细节。"""

    def setUp(self):
        """建立无费用基准，便于按交易价手算利润；另有费用单测。"""
        self.cfg = {"initial_cash": 100_000.0, "commission_rate": 0.0, "min_commission": 0.0,
                    "transfer_rate": 0.0, "stamp_rate_before_20230828": 0.0,
                    "stamp_rate_from_20230828": 0.0, "slippage_rate": 0.0,
                    "max_bar_amount_fraction": 1.0, "max_cycles_per_day": 2}
        self.rule = {"entry_deviation": 0.02, "take_profit": 0.002, "trade_fraction": 0.5}
        self.first_day = create_bars("2024-01-02")
        self.entries = pd.DataFrame([{"datetime": pd.Timestamp("2024-01-02 09:32"), "symbol": "TEST",
                                      "shares": 1000, "price": 10.0, "reason": "原策略开底仓"}])

    def run_case(self, next_day, *, cfg=None, rule=None, entries=None, first_day=None):
        """以同一确定底仓运行两日或多日路径。"""
        frames = [first_day if first_day is not None else self.first_day]
        frames += next_day if isinstance(next_day, list) else [next_day]
        frame = pd.concat(frames, ignore_index=True)
        return run_inventory_t({"TEST": frame}, self.entries if entries is None else entries,
                               pd.DatetimeIndex(frame.datetime.dt.normalize().unique()),
                               {**self.cfg, **(cfg or {})}, {**self.rule, **(rule or {})})

    def assert_inventory_valid(self, result):
        """所有成功日应保持目标库存、非负现金以及可卖股数守恒。"""
        audit = result["daily_inventory_audit"]
        self.assertTrue(audit.inventory_preserved.all())
        self.assertTrue((audit.day_sold <= audit.opening_sellable).all())
        self.assertEqual(result["metrics"]["t1_violation_count"], 0)
        self.assertTrue((result["trade_events"].cash_after >= -1e-7).all())
        self.assertTrue((result["trade_events"].shares % 100 == 0).all())

    def test_buy_first_uses_prior_inventory_and_next_bar_price(self):
        day = create_bars("2024-01-03", {"09:40": 9.0, "09:41": 9.1, "09:42": 10.0})
        result = self.run_case(day, rule={"direction": "buy_first"})
        trip = result["round_trips"].iloc[0]
        self.assertEqual(trip.buy_datetime, pd.Timestamp("2024-01-03 09:41"))
        self.assertEqual(trip.sell_datetime, pd.Timestamp("2024-01-03 09:43"))
        self.assertAlmostEqual(trip.net_pnl, 450.0)
        self.assertEqual(result["metrics"]["total_t_marked_pnl"], 450.0)
        self.assert_inventory_valid(result)

    def test_sell_first_repurchase_profit_and_same_inventory(self):
        day = create_bars("2024-01-03", {"09:40": 11.0, "09:41": 10.9})
        result = self.run_case(day, rule={"direction": "sell_first"})
        trip = result["round_trips"].iloc[0]
        self.assertEqual(trip.sell_datetime, pd.Timestamp("2024-01-03 09:41"))
        self.assertEqual(trip.buy_datetime, pd.Timestamp("2024-01-03 09:43"))
        self.assertAlmostEqual(trip.net_pnl, 450.0)
        self.assertEqual(result["open_positions"].shares.iloc[0], 1000)
        self.assert_inventory_valid(result)

    def test_repeated_cycles_exhaust_daily_sale_budget(self):
        overrides = {}
        for minute in [40, 45, 50, 55]:
            overrides[f"09:{minute}"] = 9.0
            overrides[f"09:{minute + 1}"] = 9.1
        result = self.run_case(create_bars("2024-01-03", overrides),
                               cfg={"max_cycles_per_day": 10}, rule={"direction": "buy_first"})
        self.assertEqual(len(result["round_trips"]), 2)
        day = result["daily_inventory_audit"].iloc[-1]
        self.assertEqual(day.day_sold, 1000)
        self.assertEqual(day.closing_sellable, 0)
        self.assert_inventory_valid(result)

    def test_mixed_directions_share_sale_budget(self):
        day = create_bars("2024-01-03", {"09:40": 11.0, "09:41": 10.9,
                                         "10:00": 9.0, "10:01": 9.1,
                                         "10:10": 11.0, "10:11": 10.9})
        result = self.run_case(day, cfg={"max_cycles_per_day": 10})
        self.assertEqual(list(result["round_trips"].direction), ["sell_first", "buy_first"])
        self.assertEqual(result["daily_inventory_audit"].iloc[-1].day_sold, 1000)
        self.assert_inventory_valid(result)

    def test_day_zero_and_manual_adds_never_increase_sale_budget(self):
        initial = create_bars("2024-01-02", {"09:40": 11.0, "09:41": 11.0})
        day = create_bars("2024-01-03", {"09:40": 11.0, "09:41": 11.0})
        manual = pd.DataFrame([{"datetime": pd.Timestamp("2024-01-03 09:32"), "symbol": "TEST",
                                "shares": 2000, "price": 10.0, "reason": "人工加仓"}])
        result = self.run_case(day, entries=pd.concat([self.entries, manual], ignore_index=True),
                               first_day=initial, rule={"trade_fraction": 1.0, "direction": "sell_first"})
        trades = result["trade_events"]
        self.assertEqual(len(trades.loc[(trades.event_type == "t") & (trades.datetime.dt.day == 2)]), 0)
        audit = result["daily_inventory_audit"].iloc[-1]
        self.assertEqual(audit.opening_sellable, 1000)
        self.assertEqual(audit.day_sold, 1000)
        self.assertEqual(audit.closing_shares, 3000)
        self.assert_inventory_valid(result)

    def test_missing_next_minute_cancels_first_leg(self):
        day = create_bars("2024-01-03", {"09:40": 11.0}, missing=["09:41"])
        result = self.run_case(day, rule={"direction": "sell_first"})
        self.assertEqual(len(result["round_trips"]), 0)
        self.assertEqual(result["order_attempts"].iloc[0].reason, "missing_bar")
        self.assert_inventory_valid(result)

    def test_missing_reverse_is_real_failure_then_recovers_next_day(self):
        overrides = {t.strftime("%H:%M"): 9.0 for t in pd.date_range("2024-01-03 14:20", "2024-01-03 14:54", freq="min")}
        missing = [f"14:{minute}" for minute in range(55, 60)] + ["15:00"]
        day = create_bars("2024-01-03", overrides, missing=missing)
        failure = self.run_case(day, rule={"direction": "buy_first", "take_profit": 0.5})
        self.assertEqual(failure["metrics"]["pending_pairs"], 1)
        self.assertEqual(failure["metrics"]["total_t_net_profit"], 0)
        self.assertEqual(failure["open_positions"].shares.iloc[0], 1500)
        self.assertEqual(failure["daily_inventory_audit"].iloc[-1].inventory_difference, 500)
        self.assertEqual(failure["metrics"]["stale_valuation_days"], 1)
        self.assertEqual(len(failure["trade_events"].loc[failure["trade_events"].side == "sell"]), 0)
        repaired = self.run_case([day, create_bars("2024-01-04")], rule={"direction": "buy_first", "take_profit": 0.5})
        self.assertEqual(repaired["metrics"]["pending_pairs"], 0)
        self.assertFalse(repaired["round_trips"].iloc[0].recovered_same_day)
        self.assertEqual(repaired["open_positions"].shares.iloc[0], 1000)
        self.assertEqual(repaired["daily_inventory_audit"].iloc[-1].inventory_difference, -500)

    def test_locked_reverse_retries_without_fictitious_fill(self):
        overrides = {t.strftime("%H:%M"): 9.0 for t in pd.date_range("2024-01-03 14:20", "2024-01-03 15:00", freq="min")}
        day = create_bars("2024-01-03", overrides, locked=["14:55", "14:56"])
        result = self.run_case(day, rule={"direction": "buy_first", "take_profit": 0.5})
        self.assertEqual(result["round_trips"].iloc[0].sell_datetime, pd.Timestamp("2024-01-03 14:57"))
        self.assertEqual(result["daily_inventory_audit"].iloc[-1].recovery_blocked_attempts, 2)
        self.assert_inventory_valid(result)

    def test_cash_and_sell_first_repurchase_reserve(self):
        dip = create_bars("2024-01-03", {"09:40": 9.0, "09:41": 9.1})
        buy = self.run_case(dip, cfg={"initial_cash": 10_000.0}, rule={"direction": "buy_first"})
        self.assertEqual(len(buy["round_trips"]), 0)
        peak = create_bars("2024-01-03", {"09:40": 11.0, "09:41": 10.9})
        sell = self.run_case(peak, cfg={"initial_cash": 10_000.0}, rule={"direction": "sell_first"})
        self.assertEqual(len(sell["round_trips"]), 0)
        self.assert_inventory_valid(buy)
        self.assert_inventory_valid(sell)

    def test_extreme_repurchase_cash_failure_keeps_shortfall(self):
        overrides = {"14:20": 11.0, "14:21": 11.0}
        overrides.update({t.strftime("%H:%M"): 100.0 for t in pd.date_range("2024-01-03 14:22", "2024-01-03 15:00", freq="min")})
        day = create_bars("2024-01-03", overrides)
        result = self.run_case(day, cfg={"initial_cash": 11_000.0},
                               rule={"direction": "sell_first", "take_profit": 0.5})
        self.assertEqual(result["metrics"]["pending_pairs"], 1)
        self.assertEqual(result["open_positions"].shares.iloc[0], 500)
        self.assertEqual(result["daily_inventory_audit"].iloc[-1].inventory_difference, -500)
        self.assertTrue((result["order_attempts"].reason == "cash").all())
        self.assertGreaterEqual(result["equity_curve"].cash.iloc[-1], 0)
        self.assertLess(result["metrics"]["total_t_marked_pnl"], 0)

    def test_manual_addition_cannot_borrow_from_t_profit_for_baseline(self):
        day = create_bars("2024-01-03", {"09:40": 9.0, "09:41": 9.1, "10:00": 10.04})
        manual = pd.DataFrame([{"datetime": pd.Timestamp("2024-01-03 10:00"), "symbol": "TEST",
                                "shares": 9000, "price": 10.04, "reason": "人工加仓"}])
        # T 盈利450元足以让实际账户支付90360元，但纯持有账户只剩90000元。
        with self.assertRaisesRegex(ValueError, "超过纯持有基准现金"):
            self.run_case(day, entries=pd.concat([self.entries, manual], ignore_index=True),
                          rule={"direction": "buy_first"})

    def test_capacity_limits_first_leg_and_blocks_reverse(self):
        day = create_bars("2024-01-03", {"09:40": 9.0, "09:41": 9.1})
        day.loc[day.datetime == pd.Timestamp("2024-01-03 09:41"), "amount"] = 9.1 * 250
        day.loc[day.datetime == pd.Timestamp("2024-01-03 09:43"), "amount"] = 10.0 * 100
        result = self.run_case(day, rule={"direction": "buy_first"})
        trip = result["round_trips"].iloc[0]
        self.assertEqual(trip.shares, 200)
        self.assertEqual(trip.sell_datetime, pd.Timestamp("2024-01-03 09:44"))
        self.assertEqual(result["order_attempts"].iloc[0].reason, "capacity")
        self.assert_inventory_valid(result)

    def test_last_entry_cutoff_applies_to_fill_time(self):
        day = create_bars("2024-01-03", {"14:30": 9.0, "14:31": 9.1})
        result = self.run_case(day, rule={"direction": "buy_first"})
        self.assertEqual(len(result["round_trips"]), 0)
        self.assert_inventory_valid(result)

    def test_causal_signals_and_lunch_execution(self):
        day = create_bars("2024-01-03", {"11:30": 11.0, "13:01": 10.9})
        strategy = InventoryTStrategy(self.rule)
        early = day.iloc[:120].copy()
        pd.testing.assert_frame_equal(strategy.generate_signals(day).iloc[:120], strategy.generate_signals(early))
        result = self.run_case(day, rule={"direction": "sell_first"})
        self.assertEqual(result["round_trips"].iloc[0].sell_datetime, pd.Timestamp("2024-01-03 13:01"))
        self.assert_inventory_valid(result)

    def test_fees_slippage_and_baseline_comparison(self):
        day = create_bars("2024-01-03", {"09:40": 9.0, "09:41": 9.1})
        cfg = {"commission_rate": 0.0008, "transfer_rate": 0.00001,
               "stamp_rate_from_20230828": 0.0005, "slippage_rate": 0.001,
               "min_commission": 5.0}
        result = self.run_case(day, cfg=cfg, rule={"direction": "buy_first"})
        baseline = self.run_case(day, cfg=cfg, rule={"enabled": False})
        trip = result["round_trips"].iloc[0]
        expected = 500 * (10 * 0.999 - 9.1 * 1.001) - trip.buy_fee - trip.sell_fee
        self.assertAlmostEqual(trip.net_pnl, expected)
        self.assertAlmostEqual(result["equity_curve"].equity.iloc[-1] - baseline["equity_curve"].equity.iloc[-1], expected)
        self.assertGreater(trip.slippage_cost, 0)
        self.assert_inventory_valid(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
