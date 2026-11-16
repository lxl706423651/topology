"""验证逐分钟分批恢复、费用、T+1及预测优先级；仅用合成分钟数据，不写原行情。"""

from pathlib import Path
import pickle
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from quant_backtest import weekly_t_engine as engine
from scripts import test_weekly_t_engine as legacy


def create_recovery(day, capacities, price=10.0, symbol="TEST", fraction=0.05):
    """按给定的每分钟100股容量生成有量/零量真实时间标签，容量无未来聚合。"""
    index = pd.date_range(f"{day} 14:55", f"{day} 15:00", freq="min")
    prices = [price] * 6 if np.isscalar(price) else list(price)
    return pd.DataFrame([{"datetime": timestamp, "symbol": symbol, "open": value,
                          "high": value, "low": value, "close": value,
                          "volume": capacity / fraction if capacity else 0.0,
                          "amount": value * capacity / fraction + (0.01 if capacity else 0.0)}
                         for timestamp, value, capacity in zip(index, prices, capacities)])


class WeeklyTPartialTests(unittest.TestCase):
    """独立验证分批还原、T+1、准备金和同刻资金优先级，不以外层收益选测试。"""

    def setUp(self):
        legacy.WeeklyTTests.setUp(self)

    def run_case(self, requests=(), daily=None, selected=None, cfg=None, recovery=None):
        samples = pd.DataFrame([request[0] for request in requests])
        decisions = pd.DataFrame([request[1] for request in requests])
        return engine.run_weekly_portfolio(self.selected if selected is None else selected,
                                           self.daily if daily is None else daily,
                                           samples, decisions, {**self.cfg, **(cfg or {})}, recovery_bars=recovery)

    def assert_identity(self, result):
        self.assertEqual(result["metrics"]["pending_pairs"], 0)
        self.assertTrue(result["metrics"]["same_base_plan"])
        self.assertAlmostEqual(result["metrics"]["total_t_marked_pnl"], result["round_trips"].net_pnl.sum(), places=6)
        events = result["trade_events"]
        self.assertTrue((events.cash_after >= -1e-7).all())
        self.assertTrue((events.shares % 100 == 0).all())
        audit = result["daily_inventory_audit"]
        self.assertTrue((audit.day_sold <= audit.opening_sellable).all())

    def test_pickled_recovery_store_rebuilds_only_readonly_shell(self):
        recovery = create_recovery("2024-01-09", [100, 100, 200, 0, 0, 0])
        store = engine.build_recovery_store(recovery)
        with mock.patch.object(engine.RecoveryBarStore, "__init__", side_effect=AssertionError("store revalidated")):
            restored = pickle.loads(pickle.dumps(store))
        self.assertEqual(restored.row_count, store.row_count)
        self.assertEqual(restored.symbols, store.symbols)
        timestamp = pd.Timestamp("2024-01-09 14:55")
        self.assertEqual(dict(restored.get_quote(timestamp, "TEST")), dict(store.get_quote(timestamp, "TEST")))
        with self.assertRaises(TypeError):
            restored.get_quote(timestamp, "TEST")["amount"] = 0
        request = legacy.create_request(side=-1, execution=10.1, fraction=0.1)
        expected = self.run_case([request], recovery=store)
        actual = self.run_case([request], recovery=restored)
        pd.testing.assert_frame_equal(actual["trade_events"], expected["trade_events"])
        pd.testing.assert_frame_equal(actual["equity_curve"], expected["equity_curve"])

    def test_repurchase_cash_shortfall_fills_affordable_lots_and_keeps_debt(self):
        """回购上涨超出已有现金时只恢复付得起的900股，余下100股不得虚构成交。"""
        request = legacy.create_request(side=-1, execution=10.0, fraction=1.0)
        recovery = create_recovery("2024-01-09", [1000] * 6, price=10.6, fraction=1.0)
        result = self.run_case([request], cfg={"initial_cash": 10_000., "base_fraction": 1.,
                                               "repurchase_buffer": 0.}, recovery=recovery)
        first_day = result["trade_events"].loc[
            result["trade_events"].datetime.dt.normalize().eq(pd.Timestamp("2024-01-09"))]
        self.assertEqual(first_day.shares.tolist(), [1000, 900])
        self.assertAlmostEqual(first_day.cash_after.iloc[-1], 460.)
        partial = result["round_trips"].iloc[0]
        self.assertEqual(partial.remaining_shares_after, 100)
        self.assertAlmostEqual(partial.net_pnl, -540.)
        self.assertEqual(result["metrics"]["completed_pair_count"], 0)
        self.assertEqual(result["metrics"]["pending_pairs"], 1)
        attempt = result["order_attempts"].loc[
            result["order_attempts"].datetime.eq(pd.Timestamp("2024-01-09 14:55"))].iloc[0]
        self.assertEqual(attempt.reason, "cash")
        self.assertEqual(attempt.filled_shares, 900)
        events, audits = result["trade_events"], result["daily_inventory_audit"]
        self.assertTrue((events.cash_after >= -1e-7).all())
        self.assertTrue((audits.day_sold <= audits.opening_sellable).all())

    def test_reproduce_400_shares_capacity_requires_two_intermediate_minutes(self):
        daily = legacy.create_daily(self.dates, price=21.61)
        daily.loc[daily.date == pd.Timestamp("2024-01-09"), ["close", "close_high", "close_low", "close_amount"]] = [21.53, 21.53, 21.53, 116262.0]
        selected = pd.DataFrame([legacy.create_selection(previous_close=21.61)])
        sample, decision = legacy.create_request(side=1, execution=21.56, fraction=1.0)
        sample["previous_close"] = 21.61
        recovery = create_recovery("2024-01-09", [0, 300, 300, 0, 0, 200], price=[21.56, 21.55, 21.55, 21.55, 21.55, 21.53])
        recovery["amount"] = [23679.0, 148594.0, 163728.0, 0.0, 0.0, 116262.0]
        recovery["volume"] = [11.0, 69.0, 76.0, 0.0, 0.0, 54.0]
        result = self.run_case([(sample, decision)], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.087, "max_bar_amount_fraction": 0.05}, recovery=recovery)
        trips = result["round_trips"]
        self.assertEqual(list(trips.shares), [300, 100])
        self.assertEqual(list(trips.second_datetime.dt.strftime("%H:%M")), ["14:56", "14:57"])
        self.assertEqual(result["metrics"]["t_pair_count"], 1)
        self.assertEqual(result["metrics"]["fragment_count"], 2)
        self.assertTrue(trips.pair_fully_recovered.all())
        self.assert_identity(result)

    def test_reproduce_2700_shares_split_into_800_900_1000(self):
        daily = legacy.create_daily(self.dates, price=3.51)
        daily.loc[daily.date == pd.Timestamp("2024-01-09"), ["close", "close_high", "close_low", "close_amount"]] = [3.86, 3.86, 3.86, 205352.0]
        selected = pd.DataFrame([legacy.create_selection(previous_close=3.51)])
        sample, decision = legacy.create_request(side=1, execution=3.70, fraction=1.0)
        sample["previous_close"] = 3.51
        recovery = create_recovery("2024-01-09", [800, 900, 1100, 0, 0, 2600], price=3.86)
        recovery["amount"] = [64462.0, 76042.0, 86078.0, 6176.0, 0.0, 205352.0]
        recovery["volume"] = [167.0, 197.0, 223.0, 16.0, 0.0, 532.0]
        result = self.run_case([(sample, decision)], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.095, "max_bar_amount_fraction": 0.05}, recovery=recovery)
        self.assertEqual(list(result["round_trips"].shares), [800, 900, 1000])
        self.assertEqual(result["round_trips"].second_datetime.iloc[-1], pd.Timestamp("2024-01-09 14:57"))
        self.assertEqual(result["metrics"]["t_pair_count"], 1)
        self.assertEqual(result["metrics"]["fragment_count"], 3)
        self.assert_identity(result)

    def test_insufficient_six_minute_capacity_preserves_partial_debt(self):
        dates = pd.bdate_range("2024-01-08", "2024-01-10")
        daily = legacy.create_daily(dates)
        daily.loc[daily.date == pd.Timestamp("2024-01-10"), ["open_amount", "close_amount"]] = 0.0
        selected = pd.DataFrame([legacy.create_selection(exit_date="2024-01-10")])
        request = legacy.create_request(side=1, execution=9.6, fraction=1.0)
        recovery = create_recovery("2024-01-09", [100, 100, 0, 0, 0, 100])
        result = self.run_case([request], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.04, "max_bar_amount_fraction": 0.05}, recovery=recovery)
        self.assertEqual(result["metrics"]["started_pair_count"], 1)
        self.assertEqual(result["metrics"]["completed_pair_count"], 0)
        self.assertEqual(result["metrics"]["t_pair_count"], 0)
        self.assertEqual(result["metrics"]["fragment_count"], 3)
        self.assertEqual(result["metrics"]["pending_pairs"], 1)
        self.assertEqual(result["open_positions"].pending_shares.sum(), 100)
        self.assertEqual(result["round_trips"].remaining_shares_after.iloc[-1], 100)
        self.assertFalse(result["round_trips"].pair_fully_recovered.any())
        self.assertEqual(result["open_positions"].inventory_difference.sum(), 100)

    def test_missing_future_recovery_does_not_drop_or_resize_first_leg(self):
        request = legacy.create_request(side=1, execution=9.6, fraction=1.0)
        filled = create_recovery("2024-01-09", [400, 0, 0, 0, 0, 0])
        absent = filled.iloc[:0].copy()
        cfg = {"base_fraction": 0.04, "max_bar_amount_fraction": 0.05}
        first = self.run_case([request], cfg=cfg, recovery=filled)
        missing = self.run_case([request], cfg=cfg, recovery=absent)
        columns = ["datetime", "symbol", "side", "shares", "price", "fee", "cash_after"]
        a = first["trade_events"].loc[first["trade_events"].event_type == "t", columns].iloc[0]
        b = missing["trade_events"].loc[missing["trade_events"].event_type == "t", columns].iloc[0]
        pd.testing.assert_series_equal(a, b)

    def test_partial_fees_slippage_allocation_and_base_equity_identity(self):
        request = legacy.create_request(side=1, execution=9.6, fraction=1.0)
        recovery = create_recovery("2024-01-09", [300, 100, 0, 0, 0, 0])
        result = self.run_case([request], cfg={"base_fraction": 0.041, "max_bar_amount_fraction": 0.05,
                               "commission_rate": 0.0008, "min_commission": 5.0, "transfer_rate": 0.00001,
                               "stamp_rate_from_20230828": 0.0005, "slippage_rate": 0.0005}, recovery=recovery)
        events = result["trade_events"].loc[result["trade_events"].event_type == "t"]
        first = events.iloc[0]
        trips = result["round_trips"]
        self.assertEqual(list(trips.shares), [300, 100])
        self.assertAlmostEqual(trips.buy_fee.sum(), first.fee)
        self.assertAlmostEqual(trips.buy_fee.iloc[0], first.fee * .75)
        self.assertAlmostEqual(trips.slippage_cost.sum(), events.slippage_cost.sum())
        self.assertTrue((trips.sell_fee >= 5.0).all())
        self.assertAlmostEqual(trips.net_pnl.sum(), sum((r.notional-r.fee) if r.side == "sell" else (-r.notional-r.fee) for r in events.itertuples()))
        self.assert_identity(result)

    def test_sell_first_partial_release_of_reserve(self):
        dates = pd.bdate_range("2024-01-08", "2024-01-10")
        daily = legacy.create_daily(dates)
        daily.loc[daily.date == pd.Timestamp("2024-01-10"), ["open_amount", "close_amount"]] = 0.0
        selected = pd.DataFrame([legacy.create_selection(exit_date="2024-01-10")])
        request = legacy.create_request(side=-1, execution=10.4, fraction=1.0)
        recovery = create_recovery("2024-01-09", [200, 0, 0, 0, 0, 0])
        result = self.run_case([request], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.04, "max_bar_amount_fraction": 0.05}, recovery=recovery)
        original_reserve = 400 * 10.4 * 1.12
        curve = result["equity_curve"].iloc[1]
        self.assertAlmostEqual(curve.repurchase_reserved_cash, original_reserve * .5)
        self.assertEqual(result["open_positions"].pending_shares.sum(), 200)
        self.assertEqual(result["metrics"]["t_pair_count"], 0)
        self.assertEqual(result["metrics"]["fragment_count"], 1)

    def test_remaining_repurchase_reserve_keeps_whole_minimum_commission(self):
        dates = pd.bdate_range("2024-01-08", "2024-01-10")
        daily = legacy.create_daily(dates)
        daily.loc[daily.date == pd.Timestamp("2024-01-10"), ["open_amount", "close_amount"]] = 0.0
        selected = pd.DataFrame([legacy.create_selection(exit_date="2024-01-10")])
        request = legacy.create_request(side=-1, execution=10.4, fraction=1.0)
        recovery = create_recovery("2024-01-09", [200, 0, 0, 0, 0, 0])
        result = self.run_case([request], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.04, "max_bar_amount_fraction": 0.05, "min_commission": 5.0}, recovery=recovery)
        expected = 200 * 10.4 * 1.12 + 5.0
        self.assertAlmostEqual(result["equity_curve"].repurchase_reserved_cash.iloc[-1], expected)
        self.assertEqual(result["open_positions"].pending_shares.sum(), 200)

    def test_reusable_store_is_readonly_and_does_not_rebuild_between_runs(self):
        request = legacy.create_request(side=1, execution=9.6, fraction=1.0)
        recovery = create_recovery("2024-01-09", [300, 100, 0, 0, 0, 0])
        store = engine.build_recovery_store(recovery)
        quote = store.get_quote(pd.Timestamp("2024-01-09 14:55"), "TEST")
        self.assertEqual(store.row_count, 6)
        with self.assertRaises(TypeError):
            quote["close"] = 999.0
        recovery.loc[:, "close"] = 999.0
        self.assertEqual(quote["close"], 10.0)
        cfg = {"base_fraction": 0.04, "max_bar_amount_fraction": 0.05}
        with mock.patch.object(engine, "build_recovery_store", side_effect=AssertionError("不应重复构建store")):
            first = self.run_case([request], cfg=cfg, recovery=store)
            second = self.run_case([request], cfg=cfg, recovery=store)
        for key in ["equity_curve", "trade_events", "round_trips", "daily_inventory_audit", "open_positions"]:
            pd.testing.assert_frame_equal(first[key], second[key])
        self.assertEqual(store.row_count, 6)
        self.assertEqual(store.get_quote(pd.Timestamp("2024-01-09 14:55"), "TEST")["close"], 10.0)
        self.assert_identity(first)

    def test_sell_first_three_fragments_allocate_original_fee_once(self):
        request = legacy.create_request(side=-1, execution=10.4, fraction=1.0)
        recovery = create_recovery("2024-01-09", [100, 100, 200, 0, 0, 0])
        result = self.run_case([request], cfg={"base_fraction": 0.041, "max_bar_amount_fraction": 0.05,
                               "commission_rate": 0.0008, "min_commission": 5.0, "transfer_rate": 0.00001,
                               "stamp_rate_from_20230828": 0.0005, "slippage_rate": 0.0005}, recovery=recovery)
        first = result["trade_events"].loc[result["trade_events"].event_type == "t"].iloc[0]
        trips = result["round_trips"]
        self.assertEqual(list(trips.shares), [100, 100, 200])
        self.assertAlmostEqual(trips.sell_fee.sum(), first.fee)
        self.assertAlmostEqual(trips.sell_fee.iloc[0], first.fee / 4)
        self.assertTrue((trips.buy_fee >= 5.0).all())
        self.assertEqual(result["metrics"]["completed_pair_count"], 1)
        self.assertEqual(result["metrics"]["fragment_count"], 3)
        self.assert_identity(result)

    def test_cross_day_partial_buys_do_not_replenish_exit_day_sellable(self):
        daily = legacy.create_daily(pd.bdate_range("2024-01-08", "2024-01-11"))
        selected = pd.DataFrame([legacy.create_selection(exit_date="2024-01-10")])
        daily.loc[daily.date == pd.Timestamp("2024-01-10"), "open_amount"] = 20_000.01
        recovery = pd.concat([create_recovery("2024-01-09", [200, 0, 0, 0, 0, 0]),
                              create_recovery("2024-01-10", [100, 0, 0, 0, 0, 10000])], ignore_index=True)
        result = self.run_case([legacy.create_request(side=-1, fraction=1.0)], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.04, "max_bar_amount_fraction": 0.05}, recovery=recovery)
        audit = result["daily_inventory_audit"].set_index("date")
        wednesday = audit.loc[pd.Timestamp("2024-01-10")]
        self.assertEqual(wednesday.opening_sellable, 200)
        self.assertEqual(wednesday.day_sold, 200)
        self.assertEqual(wednesday.closing_shares, 200)
        self.assertEqual(result["metrics"]["pending_pairs"], 0)
        self.assertEqual(result["metrics"]["fragment_count"], 3)
        self.assertEqual(len(result["open_positions"]), 0)
        self.assertTrue((result["trade_events"].cash_after >= 0).all())

    def test_fifteen_oclock_partial_t_and_weekly_exit_share_capacity(self):
        daily = legacy.create_daily(pd.bdate_range("2024-01-08", "2024-01-10"))
        selected = pd.DataFrame([legacy.create_selection(exit_date="2024-01-10")])
        daily.loc[daily.date == pd.Timestamp("2024-01-10"), "open_amount"] = 0.0
        daily.loc[daily.date == pd.Timestamp("2024-01-10"), "close_amount"] = 60_000.01
        recovery = pd.concat([create_recovery("2024-01-09", [0, 0, 0, 0, 0, 0]),
                              create_recovery("2024-01-10", [0, 0, 0, 0, 0, 300])], ignore_index=True)
        result = self.run_case([legacy.create_request(side=1, execution=9.6, fraction=1.0)], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.04, "max_bar_amount_fraction": 0.05}, recovery=recovery)
        at_close = result["trade_events"].loc[result["trade_events"].datetime == pd.Timestamp("2024-01-10 15:00")]
        self.assertEqual(at_close.shares.sum(), 300)
        self.assertTrue((at_close.event_type == "t").all())
        self.assertEqual(result["metrics"]["pending_pairs"], 1)
        self.assertEqual(result["open_positions"].pending_shares.sum(), 100)

    def test_pending_buy_first_extra_inventory_not_consumed_by_weekly_exit(self):
        daily = legacy.create_daily(pd.bdate_range("2024-01-08", "2024-01-10"))
        selected = pd.DataFrame([legacy.create_selection(exit_date="2024-01-10")])
        daily.loc[daily.date == pd.Timestamp("2024-01-10"), "open_amount"] = 0.0
        empty = create_recovery("2024-01-09", [0, 0, 0, 0, 0, 0]).iloc[:0]
        result = self.run_case([legacy.create_request(side=1, execution=9.6, fraction=1.0)], daily=daily, selected=selected,
                               cfg={"base_fraction": 0.04, "max_bar_amount_fraction": 0.05}, recovery=empty)
        self.assertEqual(result["metrics"]["pending_pairs"], 1)
        self.assertEqual(result["open_positions"].shares.sum(), 400)
        self.assertEqual(result["open_positions"].pending_shares.sum(), 400)
        self.assertEqual(result["open_positions"].baseline_shares.sum(), 0)
        self.assertTrue(result["metrics"]["same_base_plan"])

    def test_priority_controls_same_time_cash_competition_and_default_is_symbol(self):
        selected = pd.DataFrame([{**legacy.create_selection(), "symbol": "AAA", "rank": 1},
                                 {**legacy.create_selection(), "symbol": "ZZZ", "rank": 2}])
        daily = pd.concat([legacy.create_daily(self.dates, "AAA"), legacy.create_daily(self.dates, "ZZZ")], ignore_index=True)
        requests = []
        for symbol, priority in [("AAA", 1.0), ("ZZZ", 9.0)]:
            sample, decision = legacy.create_request(side=1, execution=9.6, fraction=1.0)
            sample["symbol"], decision["symbol"], decision["priority"] = symbol, symbol, priority
            requests.append((sample, decision))
        cfg = {"base_fraction": 0.95, "top_n": 2}
        ranked = self.run_case(requests, daily=daily, selected=selected, cfg=cfg)
        first = ranked["trade_events"].loc[(ranked["trade_events"].event_type == "t") & (ranked["trade_events"].side == "buy")]
        self.assertEqual(list(first.symbol), ["ZZZ"])
        self.assertEqual(first.signal_priority.iloc[0], 9.0)
        self.assertEqual(first.shares.iloc[0], 600)
        for sample, decision in requests:
            decision.pop("priority")
            sample["priority"] = 999.0 if sample["symbol"] == "ZZZ" else 0.0
        fallback = self.run_case(requests, daily=daily, selected=selected, cfg=cfg)
        first = fallback["trade_events"].loc[(fallback["trade_events"].event_type == "t") & (fallback["trade_events"].side == "buy")]
        self.assertEqual(list(first.symbol), ["AAA"])
        self.assert_identity(ranked)
        self.assert_identity(fallback)

    def test_endpoint_mode_also_permits_partial_recovery(self):
        sample, decision = legacy.create_request(side=1, execution=9.6, fraction=1.0)
        sample["recovery_amount"] = 10 * 100 / .05 + .01
        daily = self.daily.copy()
        daily.loc[daily.date == pd.Timestamp("2024-01-09"), "close_amount"] = 10 * 300 / .05 + .01
        result = self.run_case([(sample, decision)], daily=daily,
                               cfg={"base_fraction": 0.04, "max_bar_amount_fraction": 0.05})
        self.assertEqual(list(result["round_trips"].shares), [100, 300])
        self.assertEqual(list(result["round_trips"].second_datetime.dt.strftime("%H:%M")), ["14:55", "15:00"])
        self.assertEqual(result["metrics"]["recovery_execution_mode"], "endpoint_partial")
        self.assert_identity(result)


if __name__ == "__main__":
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(legacy),
                               unittest.defaultTestLoader.loadTestsFromTestCase(WeeklyTPartialTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
