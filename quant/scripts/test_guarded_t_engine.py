"""独立首腿到达报价条件反例；只运行合成账户，不修改旧源或历史账本。

从仓库根运行 python scripts/test_guarded_t_engine.py。引用旧测试的纯行情构造器，
以原动态引擎作为关闭控制；真实案例只固化事前/到达分钟报价，不按后续盈亏选样。
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_backtest.dynamic_t_engine import run_dynamic_portfolio, plan_dynamic_entries, SESSION_MINUTES
from quant_backtest.guarded_t_engine import run_guarded_portfolio, FIRST_LEG_EXIT_SIDE_AUDIT_COLUMNS
from quant_backtest.weekly_t_features import select_recoverable_signal_sides
from test_balanced_t_engine import market, selection, request, quote


class GuardedEngineTests(unittest.TestCase):
    """核对取消条件、零状态消耗及原账户关闭模式的精确相等。"""

    def setUp(self):
        self.cfg = {"initial_cash": 100000., "base_fraction": .85, "slippage_rate": .0005,
                    "commission_rate": .0008, "min_commission": 0., "transfer_rate": .00001,
                    "stamp_rate_before_20230828": .001, "stamp_rate_from_20230828": .0005,
                    "max_bar_amount_fraction": .05, "max_cycles_per_day": 0,
                    "dynamic_exit_mode": "fixed", "dynamic_take_net_return": .003,
                    "dynamic_stop_net_return": .01, "t_avoid_locked_exit_side": True}
        self.daily = market().iloc[:2].copy()
        self.selected = pd.DataFrame([selection()])
        self.bars = pd.DataFrame([quote(day + pd.Timedelta(minutes=m))
                                  for day in self.daily.date for m in SESSION_MINUTES])

    def run_case(self, requests=(), *, cfg=None, bars=None, daily=None, selected=None,
                 guarded=True, enabled=True, manual=None):
        engine = run_guarded_portfolio if guarded else run_dynamic_portfolio
        return engine(self.selected if selected is None else selected,
                      self.daily if daily is None else daily,
                      pd.DataFrame([item[0] for item in requests]),
                      pd.DataFrame([item[1] for item in requests]),
                      {**self.cfg, **(cfg or {}), "t_recheck_exit_side_at_execution": enabled},
                      recovery_bars=self.bars if bars is None else bars, manual_entries=manual)

    def assert_old_fields_equal(self, old, new):
        self.assertEqual(set(new), set(old) | {"first_leg_exit_side_audit"})
        for key, value in old.items():
            with self.subTest(result_key=key):
                if isinstance(value, pd.DataFrame):
                    pd.testing.assert_frame_equal(value, new[key], check_exact=True)
                elif isinstance(value, pd.Index):
                    pd.testing.assert_index_equal(value, new[key], exact=True)
                elif isinstance(value, dict):
                    self.assertEqual(set(value), set(new[key]))
                    for name, item in value.items():
                        if isinstance(item, float) and np.isnan(item):
                            self.assertTrue(np.isnan(new[key][name]))
                        else:
                            self.assertEqual(item, new[key][name], name)
                else:
                    self.assertEqual(value, new[key])

    def first_events(self, result):
        return result["trade_events"].query("event_type=='t' and fragment_id==0")

    def test_off_exactly_reproduces_original_all_fields_with_partial_fees(self):
        requests = [request(action=-1), request(clock="11:00", action=1)]
        bars = self.bars.copy()
        mask = bars.datetime.eq(pd.Timestamp("2024-01-09 10:21"))
        bars.loc[mask, "amount"] = 20000.  # 第一段只能恢复100股；后续仍计真实最低佣金。
        for mode in ("fixed", "profit_buffer", "profit_buffer_stop"):
            with self.subTest(mode=mode):
                cfg = {"dynamic_exit_mode": mode, "min_commission": 5.}
                old = self.run_case(requests, cfg=cfg, bars=bars, guarded=False, enabled=False)
                new = self.run_case(requests, cfg=cfg, bars=bars, enabled=False)
                self.assert_old_fields_equal(old, new)
                self.assertGreater(new["metrics"]["fragment_count"], new["metrics"]["t_pair_count"])
                self.assertFalse(new["first_leg_exit_side_audit"].opposite_checked.any())

    def test_default_is_off_and_empty_result_has_stable_audit_schema(self):
        old = run_dynamic_portfolio(self.selected, self.daily, pd.DataFrame(), pd.DataFrame(), self.cfg,
                                    recovery_bars=self.bars)
        new = run_guarded_portfolio(self.selected, self.daily, pd.DataFrame(), pd.DataFrame(), self.cfg,
                                    recovery_bars=self.bars)
        self.assert_old_fields_equal(old, new)
        self.assertEqual(list(new["first_leg_exit_side_audit"]), FIRST_LEG_EXIT_SIDE_AUDIT_COLUMNS)
        self.assertTrue(new["first_leg_exit_side_audit"].empty)

    def test_buy_first_at_lower_limit_and_sell_first_at_upper_limit_cancel(self):
        for action, price, opposite in [(1, 9., "sell"), (-1, 11., "buy")]:
            with self.subTest(action=action):
                result = self.run_case([request(clock="09:45", action=action, price=price)])
                audit = result["first_leg_exit_side_audit"].iloc[0]
                self.assertTrue(audit.cancelled)
                self.assertEqual(audit.opposite_side, opposite)
                self.assertEqual(audit.reason, "execution_exit_side_conservative_price_limit_proxy")
                self.assertEqual(audit.datetime, pd.Timestamp("2024-01-09 09:46"))
                self.assertEqual(audit.signal_datetime, pd.Timestamp("2024-01-09 09:45"))
                self.assertEqual(len(self.first_events(result)), 0)
                self.assertEqual(result["metrics"]["pending_pairs"], 0)
                self.assertEqual(result["metrics"]["inventory_mismatch_days"], 0)

    def test_explicit_directional_limit_has_priority_over_proxy(self):
        sample, decision = request(action=1, price=9.)
        sample["execution_limit_down"] = 8.5  # 同方向真实限价存在，应覆盖-10%代理。
        result = self.run_case([(sample, decision)])
        self.assertFalse(result["first_leg_exit_side_audit"].cancelled.any())
        self.assertEqual(len(self.first_events(result)), 1)
        sample, decision = request(action=1, price=9.6)
        sample["execution_limit_down"] = 9.6
        result = self.run_case([(sample, decision)])
        self.assertEqual(result["first_leg_exit_side_audit"].reason.tolist(), ["execution_exit_side_price_limit"])
        self.assertTrue(self.first_events(result).empty)

    def test_normal_flat_six_percent_allowed_and_exact_flat_five_percent_blocked(self):
        for price, flat, expected in [(9.4, True, False), (9.5, True, True), (9.5, False, False)]:
            with self.subTest(price=price, flat=flat):
                sample, decision = request(action=1, price=price)
                if flat:
                    sample["execution_high"] = sample["execution_low"] = price
                result = self.run_case([(sample, decision)])
                self.assertEqual(bool(result["first_leg_exit_side_audit"].cancelled.iloc[0]), expected)

    def test_zero_quantity_probe_does_not_require_full_opposite_capacity(self):
        sample, decision = request(action=1, fraction=1.)
        sample["execution_amount"] = 20000.  # 只容100股，不是请求的1500股。
        result = self.run_case([(sample, decision)])
        audit = result["first_leg_exit_side_audit"].iloc[0]
        self.assertEqual(audit.requested_shares, 1500)
        self.assertEqual(audit.own_fillable_shares, 100)
        self.assertFalse(audit.cancelled)
        self.assertEqual(audit.opposite_quote_reason, "")
        self.assertEqual(self.first_events(result).shares.tolist(), [100])

    def test_cancel_has_no_cash_fee_old_inventory_or_cycle_effect(self):
        cancelled = self.run_case([request(action=1, fraction=1., price=9.)])
        no_t = self.run_case()
        pd.testing.assert_frame_equal(cancelled["equity_curve"], no_t["equity_curve"], check_exact=True)
        pd.testing.assert_frame_equal(cancelled["trade_events"], no_t["trade_events"], check_exact=True)
        pd.testing.assert_frame_equal(cancelled["baseline_events"], no_t["baseline_events"], check_exact=True)
        pd.testing.assert_frame_equal(cancelled["daily_inventory_audit"], no_t["daily_inventory_audit"], check_exact=True)
        audit = cancelled["first_leg_exit_side_audit"].iloc[0]
        self.assertEqual(audit.minute_amount_used_before, 0.)
        self.assertEqual(audit.old_sellable_before, 1500)
        self.assertEqual(cancelled["metrics"]["t_fee"], 0.)
        self.assertEqual(cancelled["metrics"]["peak_pending_t_notional"], 0.)

    def test_cancel_preserves_shared_cash_for_next_same_clock_symbol(self):
        daily = pd.concat([self.daily, market("B").iloc[:2]], ignore_index=True)
        selected = pd.DataFrame([selection(), selection("B", rank=2)])
        first = request(action=1, fraction=1., price=9.)
        second = request(action=1, fraction=1., symbol="B")
        first[1]["priority"], second[1]["priority"] = 2., 1.
        bars = pd.concat([self.bars, self.bars.assign(symbol="B")], ignore_index=True)
        result = self.run_case([first, second], selected=selected, daily=daily, bars=bars,
                               cfg={"initial_cash": 48000., "max_pending_t_notional_fraction": None})
        entries = self.first_events(result)
        self.assertEqual(entries.symbol.tolist(), ["B"])
        self.assertEqual(entries.shares.tolist(), [1500])
        audit = result["first_leg_exit_side_audit"]
        self.assertEqual(audit.cash_before.iloc[0], audit.cash_before.iloc[1])
        self.assertEqual(audit.minute_amount_used_before.tolist(), [0., 0.])
        self.assertTrue(result["trade_events"].cash_after.ge(0).all())

    def test_cancel_allows_later_request_with_full_old_budget(self):
        requests = [request(action=1, fraction=1., price=9.),
                    request(clock="10:30", action=1, fraction=1.)]
        result = self.run_case(requests, cfg={"max_cycles_per_day": 1})
        self.assertEqual(self.first_events(result).shares.tolist(), [1500])
        self.assertEqual(self.first_events(result).datetime.tolist(), [pd.Timestamp("2024-01-09 10:31")])
        audit = result["daily_inventory_audit"].iloc[-1]
        self.assertEqual(audit.cycles_started, 1)
        self.assertEqual(audit.day_sold, 1500)
        self.assertEqual(result["metrics"]["t1_violation_count"], 0)

    def test_future_prices_and_recovery_fields_do_not_change_first_or_cancel(self):
        for price in (9., 10.):
            with self.subTest(first_price=price):
                sample, decision = request(action=1, price=price)
                first = self.run_case([(sample, decision)])
                changed_sample = {**sample, "recovery_price": 500., "recovery_amount": 0.,
                                  "recovery_high": 500., "recovery_low": 500., "recovery_volume": 0.}
                bars = self.bars.copy()
                later = bars.datetime.gt(sample["execution_time"])
                bars.loc[later, ["open", "high", "low", "close"]] = 9.8
                second = self.run_case([(changed_sample, decision)], bars=bars)
                pd.testing.assert_frame_equal(first["first_leg_exit_side_audit"], second["first_leg_exit_side_audit"], check_exact=True)
                pd.testing.assert_frame_equal(self.first_events(first), self.first_events(second), check_exact=True)

    def test_new_guard_does_not_cancel_or_hide_existing_recovery_obligation(self):
        bars = self.bars.copy()
        later = bars.datetime.gt(pd.Timestamp("2024-01-09 10:01"))
        bars.loc[later, ["open", "high", "low", "close"]] = 9.
        daily = self.daily.copy()
        daily.loc[daily.index[-1], ["close", "close_high", "close_low"]] = 9.
        result = self.run_case([request(action=1)], bars=bars, daily=daily)
        self.assertFalse(result["first_leg_exit_side_audit"].cancelled.any())
        self.assertEqual(result["metrics"]["pending_pairs"], 1)
        self.assertEqual(result["metrics"]["inventory_mismatch_days"], 1)
        self.assertEqual(result["metrics"]["recovery_failure_days"], 1)
        self.assertEqual(result["metrics"]["t1_violation_count"], 0)
        self.assertIn("conservative_price_limit_proxy", set(result["order_attempts"].reason))

    def test_first_leg_original_next_minute_validation_cannot_be_bypassed(self):
        sample, decision = request(action=1, price=9.)
        sample["execution_time"] = sample["datetime"]
        with self.assertRaisesRegex(ValueError, "下一交易分钟"):
            self.run_case([(sample, decision)])

    def test_original_frozen_planner_and_manual_new_shares_contract(self):
        manual = pd.DataFrame([{"datetime": pd.Timestamp("2024-01-09 09:30"),
                                "signal_datetime": pd.Timestamp("2024-01-08 15:00"),
                                "symbol": "A", "shares": 100}])
        plan = plan_dynamic_entries(self.selected, self.daily, self.cfg, manual_entries=manual,
                                    recovery_bars=self.bars)
        requests = [request(action=1, fraction=1., price=9.), request(clock="10:30", action=-1, fraction=1.)]
        new = run_guarded_portfolio(plan["frozen_selections"], self.daily, pd.DataFrame([p[0] for p in requests]),
                                    pd.DataFrame([p[1] for p in requests]),
                                    {**plan["replay_cfg"], "t_recheck_exit_side_at_execution": True}, recovery_bars=self.bars)
        self.assertEqual(self.first_events(new).shares.tolist(), [1500])
        self.assertEqual(new["metrics"]["base_plan_mismatch_count"], 0)
        self.assertEqual(new["metrics"]["t1_violation_count"], 0)
        self.assertEqual(new["daily_inventory_audit"].iloc[-1].day_sold, 1500)
        pd.testing.assert_frame_equal(plan["base_entries"], new["baseline_events"], check_exact=True)

    def test_real_sh603949_arrival_quote_reproduces_signal_execution_gap(self):
        # 原内部案例2026-03-23 09:45→09:46；这里只固化已核raw报价，不读未来路径。
        known = pd.DataFrame([{"signal_price": 19.68, "signal_high": 19.78,
                               "signal_low": 19.68, "previous_close": 21.78}])
        cfg = {**self.cfg, "flat_st_limit_rate": .05, "nonflat_limit_proxy": .098}
        self.assertTrue(select_recoverable_signal_sides(known, pd.Series([1]), cfg).iloc[0])
        daily = market(price=21.78).iloc[:2]
        selected = pd.DataFrame([selection(price=21.78)])
        sample, decision = request(clock="09:45", action=1, fraction=.5, price=19.60)
        sample.update(previous_close=21.78, execution_high=19.68, execution_low=19.60,
                      execution_amount=8010775., execution_volume=4084.)
        old = self.run_case([(sample, decision)], daily=daily, selected=selected, enabled=False)
        new = self.run_case([(sample, decision)], daily=daily, selected=selected)
        self.assertEqual(len(self.first_events(old)), 1)
        self.assertTrue(self.first_events(new).empty)
        audit = new["first_leg_exit_side_audit"].iloc[0]
        self.assertEqual(audit.reference_price, 19.6)
        self.assertEqual(audit.reason, "execution_exit_side_conservative_price_limit_proxy")

    def test_switch_requires_boolean_not_truthy_string(self):
        with self.assertRaisesRegex(ValueError, "布尔"):
            self.run_case(enabled="false")


if __name__ == "__main__":
    unittest.main(verbosity=2)
