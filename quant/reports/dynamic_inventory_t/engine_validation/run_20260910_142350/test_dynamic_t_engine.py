"""独立v13执行语义反例：只用合成行情，不修改旧回测、缓存或训练模型。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_backtest.balanced_t_engine import plan_balanced_entries, run_balanced_portfolio
from quant_backtest.dynamic_t_engine import run_dynamic_portfolio, plan_dynamic_entries, SESSION_MINUTES
from quant_backtest.weekly_t_engine import RecoveryBarStore, build_recovery_store
from test_balanced_t_engine import selection, market, request, quote


class DynamicEngineTests(unittest.TestCase):
    """比较账目、因果顺序和真实失败状态，不按实现细节伪造收益断言。"""

    def setUp(self):
        self.cfg = {"initial_cash": 100000., "base_fraction": .85, "slippage_rate": 0.,
                    "commission_rate": 0., "min_commission": 0., "transfer_rate": 0.,
                    "stamp_rate_before_20230828": 0., "stamp_rate_from_20230828": 0.,
                    "max_bar_amount_fraction": .05, "max_cycles_per_day": 0,
                    "dynamic_exit_mode": "profit_buffer", "dynamic_take_net_return": .003,
                    "dynamic_stop_net_return": .01}
        self.daily = market().iloc[:2].copy()
        self.selected = pd.DataFrame([selection()])
        self.bars = pd.DataFrame([quote(day + pd.Timedelta(minutes=minute))
                                  for day in self.daily.date for minute in SESSION_MINUTES])

    def change(self, bars, clock, price=None, amount=None, volume=None, day="2024-01-09"):
        """只修改指定测试bar，不向邻接分钟复制数据。"""
        mask = bars.datetime.eq(pd.Timestamp(f"{day} {clock}"))
        if price is not None:
            bars.loc[mask, ["open", "high", "low", "close"]] = price
        if amount is not None:
            bars.loc[mask, "amount"] = amount
        if volume is not None:
            bars.loc[mask, "volume"] = volume

    def run_case(self, requests=(), *, bars=None, cfg=None, daily=None, selected=None, manual=None):
        return run_dynamic_portfolio(self.selected if selected is None else selected,
                                     self.daily if daily is None else daily,
                                     pd.DataFrame([row[0] for row in requests]),
                                     pd.DataFrame([row[1] for row in requests]),
                                     {**self.cfg, **(cfg or {})},
                                     recovery_bars=self.bars if bars is None else bars,
                                     manual_entries=manual)

    def assert_account(self, result, complete=True):
        """独立检查净现金、整手、日初卖额、组净利和恢复时间。"""
        events, trips = result["trade_events"], result["round_trips"]
        self.assertTrue(events.cash_after.ge(-1e-7).all())
        self.assertTrue(events.shares.mod(100).eq(0).all())
        self.assertTrue(events.datetime.gt(events.signal_datetime).all())
        inv = result["daily_inventory_audit"]
        self.assertTrue(inv.day_sold.le(inv.opening_sellable).all())
        self.assertEqual(result["metrics"]["t1_violation_count"], 0)
        decisions = result["exit_decisions"]
        self.assertFalse(decisions.pair_id.duplicated().any())
        self.assertTrue(decisions.exit_signal_time.gt(decisions.first_signal_time).all())
        self.assertTrue(decisions.exit_earliest_time.gt(decisions.exit_signal_time).all())
        for row in trips.itertuples(index=False):
            self.assertGreater(row.second_datetime, row.exit_signal_time)
            self.assertGreaterEqual(row.second_datetime, row.exit_earliest_time)
        flow = sum((1 if row.side == "sell" else -1) * row.notional - row.fee
                   for row in events.itertuples(index=False) if row.event_type == "t")
        self.assertAlmostEqual(flow, result["metrics"]["t_cashflow"], places=7)
        if complete:
            self.assertEqual(result["metrics"]["pending_pairs"], 0)
            self.assertAlmostEqual(flow, trips.net_pnl.sum(), places=7)
            self.assertAlmostEqual(result["metrics"]["total_t_marked_pnl"], flow, places=7)

    def test_profit_requires_next_minute_and_future_prices_cannot_change_intent(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=9.9)
        first = self.run_case([request()], bars=bars)
        changed = bars.copy(); self.change(changed, "10:03", price=10.2)
        second = self.run_case([request()], bars=changed)
        pd.testing.assert_frame_equal(first["exit_decisions"], second["exit_decisions"])
        for result in (first, second):
            intent = result["exit_decisions"].iloc[0]
            self.assertEqual(intent.exit_signal_time, pd.Timestamp("2024-01-09 10:02"))
            self.assertEqual(intent.exit_earliest_time, pd.Timestamp("2024-01-09 10:03"))
            self.assertEqual(result["round_trips"].second_datetime.iloc[0], intent.exit_earliest_time)
            self.assert_account(result)
        self.assertNotEqual(first["round_trips"].net_pnl.sum(), second["round_trips"].net_pnl.sum())

    def test_close_signal_does_not_use_intrabar_extreme(self):
        bars = self.bars.copy()
        bars.loc[bars.datetime.eq(pd.Timestamp("2024-01-09 10:02")), "low"] = 9.8
        result = self.run_case([request()], bars=bars)
        self.assertEqual(result["exit_decisions"].exit_reason.tolist(), ["hard_deadline"])
        self.assertEqual(result["round_trips"].second_datetime.iloc[0], pd.Timestamp("2024-01-09 10:21"))

    def test_minimum_commission_is_in_profit_estimate(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=9.95)
        result = self.run_case([request()], bars=bars, cfg={"min_commission": 5.})
        self.assertEqual(result["exit_decisions"].exit_reason.tolist(), ["hard_deadline"])
        self.change(bars, "10:02", price=9.9)
        result = self.run_case([request()], bars=bars, cfg={"min_commission": 5.})
        intent = result["exit_decisions"].iloc[0]
        self.assertAlmostEqual(intent.normalization_notional, 3000.)
        self.assertAlmostEqual(intent.estimated_net_pnl, 20.)
        self.assertAlmostEqual(intent.estimated_net_return, 20. / 3000.)
        self.assertEqual(intent.exit_reason, "profit_buffer")
        self.assert_account(result)

    def test_normalizer_uses_actual_first_fill_and_buy_first_cash_is_bounded(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=9.9)
        one = request(); one[0]["execution_amount"] = 40000.
        clipped = self.run_case([one], bars=bars)
        self.assertEqual(clipped["exit_decisions"].normalization_notional.iloc[0], 2000.)
        self.assertEqual(clipped["exit_decisions"].remaining_shares.iloc[0], 200)
        cash_limited = self.run_case([request(action=1, fraction=1.)], cfg={"initial_cash": 20000.})
        first = cash_limited["trade_events"].query("event_type=='t' and fragment_id==0").iloc[0]
        self.assertEqual(first.shares, 500)
        self.assertEqual(first.cash_after, 0.)
        self.assert_account(clipped); self.assert_account(cash_limited)

    def test_stop_only_enabled_in_stop_mode_and_buy_first_profit_direction(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=10.11)
        stop = self.run_case([request()], bars=bars, cfg={"dynamic_exit_mode": "profit_buffer_stop"})
        take = self.run_case([request()], bars=bars)
        buy = self.run_case([request(action=1)], bars=bars)
        self.assertEqual(stop["exit_decisions"].exit_reason.tolist(), ["stop_loss"])
        self.assertEqual(take["exit_decisions"].exit_reason.tolist(), ["hard_deadline"])
        self.assertEqual(buy["exit_decisions"].exit_reason.tolist(), ["profit_buffer"])
        for result in (stop, take, buy): self.assert_account(result)

    def test_latched_order_survives_limit_and_zero_volume_then_price_reversal(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=9.9)
        self.change(bars, "10:03", price=10.5)  # 买侧5%候选限价。
        self.change(bars, "10:04", price=10.2, amount=0., volume=0.)
        self.change(bars, "10:05", price=10.2)
        result = self.run_case([request()], bars=bars)
        self.assertEqual(len(result["exit_decisions"]), 1)
        self.assertEqual(result["round_trips"].second_datetime.iloc[0], pd.Timestamp("2024-01-09 10:05"))
        self.assertLess(result["round_trips"].net_pnl.sum(), 0.)
        self.assertTrue({"conservative_price_limit_proxy", "zero_price_or_volume"}.issubset(set(result["order_attempts"].reason)))
        self.assert_account(result)

    def test_zero_volume_current_bar_cannot_trigger_but_deadline_still_latches(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=9.9, amount=0., volume=0.)
        bars = bars.loc[~bars.datetime.eq(pd.Timestamp("2024-01-09 10:20"))]
        result = self.run_case([request()], bars=bars)
        intent = result["exit_decisions"].iloc[0]
        self.assertEqual(intent.exit_signal_time, pd.Timestamp("2024-01-09 10:20"))
        self.assertEqual(intent.exit_reason, "hard_deadline")
        self.assertTrue(pd.isna(intent.observed_price))
        self.assert_account(result)

    def test_partial_recovery_charges_each_fee_and_preserves_one_pair_lot_count(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=9.8)
        self.change(bars, "10:03", price=9.8, amount=20000.)
        self.change(bars, "10:04", price=9.8, amount=20000.)
        self.change(bars, "10:05", price=9.8, amount=20000.)
        result = self.run_case([request()], bars=bars,
                               cfg={"min_commission": 5., "commission_rate": .0008,
                                    "transfer_rate": .00001, "slippage_rate": .0005,
                                    "stamp_rate_from_20230828": .0005})
        trips = result["round_trips"]
        self.assertEqual(trips.shares.tolist(), [100, 100, 100])
        self.assertTrue(trips.buy_fee.gt(5.).all())
        first_fee = result["trade_events"].query("event_type=='t' and fragment_id==0").fee.iloc[0]
        self.assertAlmostEqual(trips.sell_fee.sum(), first_fee)
        self.assertEqual(result["metrics"]["t_pair_count"], 1)
        self.assertEqual(result["metrics"]["fragment_count"], 3)
        self.assertEqual(result["metrics"]["completed_t_pair_lots"], 3.)
        self.assertEqual(result["metrics"]["total_t_traded_lots"], 6.)
        self.assert_account(result)

    def test_recovery_and_new_first_leg_share_same_minute_capacity(self):
        bars = self.bars.copy(); self.change(bars, "10:03", amount=60000.)
        one = request(horizon=2); two = request(clock="10:02")
        two[0]["execution_amount"] = 60000.
        result = self.run_case([one, two], bars=bars, cfg={"dynamic_exit_mode": "fixed"})
        self.assertEqual(result["metrics"]["started_pair_count"], 1)
        self.assertIn("capacity", set(result["order_attempts"].reason))
        self.assert_account(result)

    def test_multiple_pairs_exhaust_old_shares_and_manual_buy_does_not_replenish(self):
        requests = [request(clock=f"10:{minute:02d}", fraction=.2, horizon=2, action=(-1 if i % 2 == 0 else 1))
                    for i, minute in enumerate(range(0, 24, 4))]
        manual = pd.DataFrame([{"datetime": "2024-01-09 09:30", "signal_datetime": "2024-01-08", "symbol": "A", "shares": 500}])
        result = self.run_case(requests, manual=manual, cfg={"dynamic_exit_mode": "fixed"})
        inv = result["daily_inventory_audit"].iloc[-1]
        self.assertEqual(result["metrics"]["completed_pair_count"], 5)
        self.assertEqual(inv.opening_sellable, 1500)
        self.assertEqual(inv.day_sold, 1500)
        self.assertEqual(inv.closing_shares, 2000)
        self.assertEqual(inv.closing_sellable, 0)
        self.assert_account(result)

    def test_lunch_and_last_minute_deadline_are_trading_minute_ordered(self):
        for signal, deadline, exit_signal in [("11:28", "13:01", "11:30"), ("14:00", "15:00", "14:59")]:
            with self.subTest(signal=signal):
                one = request(clock=signal)
                one[1]["planned_recovery_time"] = pd.Timestamp("2024-01-09 " + deadline)
                result = self.run_case([one], cfg={"dynamic_exit_mode": "fixed"})
                intent = result["exit_decisions"].iloc[0]
                self.assertEqual(intent.exit_signal_time, pd.Timestamp("2024-01-09 " + exit_signal))
                self.assertEqual(result["round_trips"].second_datetime.iloc[0], pd.Timestamp("2024-01-09 " + deadline))
                self.assert_account(result)

    def test_failed_end_of_day_recovery_carries_true_debt_and_original_signal(self):
        one = request(clock="14:00"); one[1]["planned_recovery_time"] = pd.Timestamp("2024-01-09 15:00")
        bars = self.bars.copy(); self.change(bars, "15:00", amount=0., volume=0.)
        daily = market().iloc[:3].copy()
        result = self.run_case([one], bars=bars, daily=daily, cfg={"dynamic_exit_mode": "fixed"})
        trip = result["round_trips"].iloc[0]
        self.assertEqual(trip.second_datetime, pd.Timestamp("2024-01-10 09:30"))
        self.assertEqual(trip.exit_signal_time, pd.Timestamp("2024-01-09 14:59"))
        self.assertEqual(result["metrics"]["inventory_mismatch_days"], 2)
        self.assertEqual(result["metrics"]["recovery_failure_days"], 1)
        self.assert_account(result)
        unfinished = self.run_case([one], bars=bars, cfg={"dynamic_exit_mode": "fixed"})
        self.assertEqual(unfinished["metrics"]["pending_pairs"], 1)
        self.assertEqual(unfinished["open_positions"].pending_shares.sum(), 300)
        self.assertEqual(unfinished["open_positions"].exit_state.iloc[0], "exit_latched")
        self.assert_account(unfinished, complete=False)

    def test_buy_first_partial_across_day_uses_new_opening_budget_and_same_intent(self):
        one = request(clock="14:00", action=1)
        one[1]["planned_recovery_time"] = pd.Timestamp("2024-01-09 15:00")
        bars = self.bars.copy(); self.change(bars, "15:00", amount=20000.)
        bars = pd.concat([bars, pd.DataFrame([quote("2024-01-10 09:31", amount=20000.),
                                              quote("2024-01-10 09:32")])], ignore_index=True)
        daily = market().iloc[:3].copy()
        daily.loc[daily.date.eq(pd.Timestamp("2024-01-10")), "open_amount"] = 20000.
        result = self.run_case([one], bars=bars, daily=daily,
                               cfg={"dynamic_exit_mode": "fixed", "min_commission": 5.})
        trips = result["round_trips"]
        self.assertEqual(trips.shares.tolist(), [100, 100, 100])
        self.assertEqual(trips.exit_signal_time.nunique(), 1)
        self.assertEqual(trips.second_datetime.iloc[-1], pd.Timestamp("2024-01-10 09:32"))
        first_raw_bar = result["trade_events"].loc[result["trade_events"].datetime.isin(
            [pd.Timestamp("2024-01-10 09:30"), pd.Timestamp("2024-01-10 09:31")])]
        self.assertEqual(first_raw_bar.shares.sum(), 100)
        self.assertTrue(result["order_attempts"].loc[result["order_attempts"].datetime.eq(
            pd.Timestamp("2024-01-10 09:31")), "reason"].eq("capacity").all())
        inv = result["daily_inventory_audit"].iloc[-1]
        self.assertEqual(inv.opening_sellable, 1700)
        self.assertEqual(inv.day_sold, 200)
        self.assertEqual(inv.closing_shares, 1500)
        self.assertAlmostEqual(trips.buy_fee.sum(), 5.)
        self.assertAlmostEqual(trips.sell_fee.sum(), 15.)
        self.assert_account(result)

    def test_same_common_marks_without_close_for_hold_and_all_exit_modes(self):
        bars = self.bars.copy(); self.change(bars, "10:02", price=9.9)
        bars = bars.loc[~((bars.datetime.dt.normalize().eq(pd.Timestamp("2024-01-09"))) & (bars.datetime.dt.time > pd.Timestamp("14:50").time()))]
        self.change(bars, "14:50", price=10.4)
        daily = self.daily.copy(); daily.loc[daily.date.eq(pd.Timestamp("2024-01-09")), "close"] = np.nan
        hold = self.run_case(bars=bars, daily=daily)
        for mode in ("fixed", "profit_buffer", "profit_buffer_stop"):
            result = self.run_case([request()], bars=bars, daily=daily, cfg={"dynamic_exit_mode": mode})
            pd.testing.assert_series_equal(hold["equity_curve"].baseline_equity, result["equity_curve"].baseline_equity)
            pos = result["open_positions"].iloc[0]
            self.assertEqual(pos.mark_price, 10.4)
            self.assertEqual(pos.mark_datetime, pd.Timestamp("2024-01-09 14:50"))
            self.assertTrue(result["daily_inventory_audit"].iloc[-1].stale_price)
            self.assert_account(result)

    def test_T_fill_reference_does_not_add_strategy_specific_common_mark(self):
        # 刻意使store缺末段，但稀疏首腿样本仍有执行quote；共同估值只认共同流。
        bars = self.bars.loc[~((self.bars.datetime.dt.normalize().eq(pd.Timestamp("2024-01-09"))) & (self.bars.datetime.dt.time >= pd.Timestamp("14:01").time()))].copy()
        daily = self.daily.copy(); daily.loc[daily.date.eq(pd.Timestamp("2024-01-09")), "close"] = np.nan
        one = request(clock="14:00", price=10.2)
        hold = self.run_case(bars=bars, daily=daily)
        result = self.run_case([one], bars=bars, daily=daily)
        pd.testing.assert_series_equal(hold["equity_curve"].baseline_equity, result["equity_curve"].baseline_equity)
        self.assertEqual(result["open_positions"].mark_datetime.iloc[0], pd.Timestamp("2024-01-09 14:00"))
        self.assertEqual(result["metrics"]["pending_pairs"], 1)
        self.assert_account(result, complete=False)

    def test_v12_fixed_control_economic_identity_with_equal_full_retry_clocks(self):
        bars = self.bars.copy(); self.change(bars, "10:21", amount=20000.)
        one = request()
        cfg = {**self.cfg, "dynamic_exit_mode": "fixed", "min_commission": 5., "slippage_rate": .0005,
               "scheduled_recovery_times": [f"{m//60:02d}:{m%60:02d}" for m in SESSION_MINUTES]}
        new = self.run_case([one], bars=bars, cfg=cfg)
        old = run_balanced_portfolio(self.selected, self.daily, pd.DataFrame([one[0]]), pd.DataFrame([one[1]]), cfg, recovery_bars=bars)
        pd.testing.assert_frame_equal(old["equity_curve"], new["equity_curve"])
        economic = ["datetime", "symbol", "side", "shares", "price", "reference_price", "fee", "slippage_cost", "cash_after"]
        pd.testing.assert_frame_equal(old["trade_events"][economic], new["trade_events"][economic])
        self.assert_account(new)

    def test_added_retry_coverage_is_visible_vs_v12_sparse_control(self):
        bars = self.bars.copy()
        for minute in range(21, 27): self.change(bars, f"10:{minute}", amount=0., volume=0.)
        one = request(); cfg = {**self.cfg, "dynamic_exit_mode": "fixed"}
        new = self.run_case([one], bars=bars, cfg=cfg)
        old = run_balanced_portfolio(self.selected, self.daily, pd.DataFrame([one[0]]), pd.DataFrame([one[1]]), cfg, recovery_bars=bars)
        self.assertEqual(new["round_trips"].second_datetime.iloc[0], pd.Timestamp("2024-01-09 10:27"))
        self.assertEqual(old["round_trips"].second_datetime.iloc[0], pd.Timestamp("2024-01-09 14:55"))
        self.assertEqual(new["metrics"]["recovery_session_minute_count"], 240)
        self.assert_account(new)

    def test_v12_frozen_base_and_store_are_reusable_and_unchanged(self):
        paths = [Path(__file__).resolve().parents[1]/"quant_backtest"/name
                 for name in ("balanced_t_engine.py", "weekly_t_engine.py", "weekly_t_quote_store.py")]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        plan = plan_balanced_entries(self.selected, self.daily, self.cfg)
        store = build_recovery_store(self.bars)
        one = self.run_case([request()], bars=store, selected=plan["frozen_selections"])
        two = self.run_case([request()], bars=store, selected=plan["frozen_selections"])
        pd.testing.assert_frame_equal(one["trade_events"], two["trade_events"])
        pd.testing.assert_frame_equal(plan["baseline_result"]["baseline_events"], one["baseline_events"][plan["baseline_result"]["baseline_events"].columns])
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])

    def test_dynamic_manual_planning_and_replay_share_opening_raw_bar_capacity(self):
        bars = self.bars.copy(); self.change(bars, "09:31", day="2024-01-08", amount=320000.)
        daily = self.daily.copy(); daily.loc[daily.date.eq(pd.Timestamp("2024-01-08")), "open_amount"] = 320000.
        manual = pd.DataFrame([{"datetime": "2024-01-08 09:31", "signal_datetime": "2024-01-08 09:30",
                                "symbol": "A", "shares": 500}])
        planned = plan_dynamic_entries(self.selected, daily, self.cfg, manual_entries=manual, recovery_bars=bars)
        frozen = planned["frozen_selections"]
        self.assertEqual(frozen.loc[frozen._manual, "frozen_base_shares"].tolist(), [100])
        self.assertEqual(planned["base_entries"].shares.sum(), 1600)
        actual = self.run_case(bars=bars, daily=daily, selected=frozen)
        self.assertEqual(actual["metrics"]["base_plan_mismatch_count"], 0)
        self.assertEqual(actual["open_positions"].shares.sum(), 1600)
        common = ["datetime", "symbol", "shares", "price", "reference_price", "notional", "fee", "cash_after"]
        pd.testing.assert_frame_equal(planned["base_entries"][common], actual["baseline_events"][common])

    def test_unfunded_candidates_do_not_query_full_minute_marks(self):
        class CountingStore(RecoveryBarStore):
            def __init__(self, base):
                self.base, self.calls = base, []
            def get_quote(self, timestamp, symbol):
                self.calls.append((timestamp, symbol))
                return self.base.get_quote(timestamp, symbol)
        store = CountingStore(build_recovery_store(self.bars))
        selected = pd.DataFrame([selection(), selection(symbol="B", rank=2)])
        daily = pd.concat([self.daily, market("B").iloc[:2]], ignore_index=True)
        result = self.run_case(selected=selected, daily=daily, bars=store, cfg={"initial_cash": 20000.})
        self.assertEqual(result["baseline_events"].symbol.tolist(), ["A"])
        self.assertEqual({symbol for _, symbol in store.calls}, {"A"})
        self.assertEqual(len(store.calls), 2 * 240)

    def test_invalid_modes_and_nonpositive_thresholds_rejected(self):
        for cfg in ({"dynamic_exit_mode": "best"}, {"dynamic_take_net_return": 0.},
                    {"dynamic_stop_net_return": np.nan}, {"dynamic_take_net_return": True}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError): self.run_case(cfg=cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
