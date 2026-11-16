"""验证周度工作流的因果股票池、T动作、周边界与增量统计；仅用合成DataFrame。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_weekly_t_cross_validation import (
    WeeklyLearnedTStrategy, calculate_proxy_candidates, calculate_weekly_increment, create_decisions,
    create_fold_recovery_store, select_execution_data, select_weekly_universe, select_year_rows,
)


class WeeklyWorkflowTests(unittest.TestCase):
    """覆盖真实语义边界，而非重复模型或账户实现。"""

    def test_actions_preserve_input_index_and_old_inventory_target(self):
        """负预测是卖旧仓请求，不能表现为空头目标或改变输入表。"""
        frame = pd.DataFrame({"prediction": [-.01, .012, .001, np.nan]}, index=[8, 2, 9, 4])
        original = frame.copy(deep=True)
        strategy = WeeklyLearnedTStrategy({"threshold": .005, "direction": "both", "fraction": .5})
        output = strategy.generate_signals(frame)
        self.assertEqual(output.index.tolist(), frame.index.tolist())
        self.assertEqual(output.t_action.tolist(), [-1, 1, 0, 0])
        self.assertTrue(output.target_position.eq(1).all())
        pd.testing.assert_frame_equal(original, frame)

    def test_first_qualifying_signal_is_independent_of_future_profit(self):
        """同日最早信号优先，不因晚些信号的未来收益更高而重选。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2024-03-05 10:00", "2024-03-05 09:45"]),
                              "symbol": ["SH.600000"] * 2, "prediction": [-.02, -.006],
                              "target_return": [-.1, .03]})
        rule = {"threshold": .005, "direction": "sell_first", "fraction": 1.0}
        decisions = create_decisions(frame, rule)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions.datetime.iloc[0], pd.Timestamp("2024-03-05 09:45"))
        frame.target_return *= -100
        pd.testing.assert_frame_equal(decisions, create_decisions(frame, rule))

    def test_weekly_universe_does_not_require_future_labels(self):
        """推断股票池仅看历史资格，无标签的候选仍保留。"""
        frame = pd.DataFrame({"f_history_days": [20, 19, 200], "f_last_close": [4, 10, 2],
                              "f_avg_daily_amount": [6e7, 8e7, 9e7],
                              "target_return": [np.nan, 1, 3], "feature_week_complete": [True] * 3})
        cfg = {"min_history_days": 20, "min_price": 3, "min_avg_daily_amount": 5e7}
        output = select_weekly_universe(frame, cfg)
        self.assertEqual(output.index.tolist(), [0])
        self.assertTrue(output.target_return.isna().all())

    def test_week_selection_excludes_one_day_and_cross_year(self):
        """预知单日交易周不能实现T+1周内买卖，跨年周不能混入独立年度账户。"""
        frame = pd.DataFrame({"entry_date": pd.to_datetime(["2024-01-02", "2024-02-19", "2024-12-30"]),
                              "exit_date": pd.to_datetime(["2024-01-02", "2024-02-23", "2025-01-03"]),
                              "trade_week": pd.to_datetime(["2024-01-01", "2024-02-19", "2024-12-30"]),
                              "trade_week_complete": [True] * 3})
        self.assertEqual(select_year_rows(frame, 2024, {"smoke_weeks": 0}).index.tolist(), [1])

    def test_increment_statistics_include_first_week_and_do_not_compound_difference(self):
        """权益差100→300归因为100和200，不丢掉第一周也不把差额当组合收益连乘。"""
        curve = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-05", "2024-01-12"]),
                              "equity": [1100, 1500], "baseline_equity": [1000, 1200]})
        weekly, metrics = calculate_weekly_increment(curve, 1000, {"random_state": 42, "bootstrap_samples": 100})
        self.assertEqual(weekly.t_increment.tolist(), [100, 200])
        self.assertAlmostEqual(weekly.increment_return.sum(), .3)
        self.assertGreater(metrics["increment_ci_low"], 0)

    def test_all_declared_trade_fractions_are_evaluated(self):
        """减量不能只停留在配置；每个方向和阈值必须产生对应比例候选。"""
        predictions = pd.DataFrame({"datetime": pd.to_datetime(["2024-03-05 09:45"]),
                                    "symbol": ["SH.600000"], "prediction": [-.02]})
        intraday = predictions[["datetime", "symbol"]].assign(execution_price=10., recovery_price=9.8)
        cfg = {"t_thresholds": [.005, .01], "t_directions": ["sell_first", "both"],
               "t_trade_fractions": [.25, .5, 1.], "commission_rate": .0008, "transfer_rate": .00001,
               "slippage_rate": .0005, "stamp_rate_before_20230828": .001,
               "stamp_rate_from_20230828": .0005}
        result = pd.DataFrame(calculate_proxy_candidates(predictions, intraday, "l2_depth4", cfg))
        self.assertEqual(len(result), 12)
        self.assertEqual(set(result.fraction), {.25, .5, 1.})

    def test_locked_signal_can_be_followed_by_unlocked_signal(self):
        """当前卖出方向受限时可跳过，稍后恢复常态再买；未来回补价不参与选择。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2024-03-05 09:45", "2024-03-05 10:00"]),
                              "symbol": ["SH.600000"] * 2, "prediction": [.02, .01],
                              "signal_price": [9., 9.4], "signal_high": [9., 9.5],
                              "signal_low": [9., 9.3], "previous_close": [10., 10.],
                              "recovery_price": [11., 9.]}, index=[9, 2])
        cfg = {"t_avoid_locked_exit_side": True, "t_signal_priority": "predicted_edge",
               "flat_st_limit_rate": .05, "nonflat_limit_proxy": .098}
        rule = {"threshold": .005, "direction": "buy_first", "fraction": .5}
        decision = create_decisions(frame, rule, cfg)
        self.assertEqual(decision.datetime.tolist(), [pd.Timestamp("2024-03-05 10:00")])
        self.assertEqual(decision.priority.tolist(), [.01])
        frame.recovery_price *= 10
        pd.testing.assert_frame_equal(decision, create_decisions(frame, rule, cfg))

    def test_past_auction_gate_cannot_depend_on_future_close_amount(self):
        """周度流动性只能读取已完成周，下一周实际收盘成交额变化不影响候选。"""
        frame = pd.DataFrame({"f_history_days": [20, 20], "f_last_close": [4, 4],
                              "f_avg_daily_amount": [6e7, 6e7],
                              "f_prior_close_auction_amount_min": [6e5, 4e5],
                              "future_close_amount": [0., 1e10]})
        cfg = {"min_history_days": 20, "min_price": 3, "min_avg_daily_amount": 5e7,
               "min_prior_close_auction_amount": 5e5}
        self.assertEqual(select_weekly_universe(frame, cfg).index.tolist(), [0])
        frame.future_close_amount = [1e10, 0.]
        self.assertEqual(select_weekly_universe(frame, cfg).index.tolist(), [0])

    def test_recovery_quotes_continue_through_remaining_test_year(self):
        """末个计划平仓日后仍给未偿库存提供真实年度末行情，不跨年使用未来数据。"""
        selected = pd.DataFrame({"symbol": ["SH.600000"], "trade_week": pd.to_datetime(["2024-12-23"]),
                                 "entry_date": pd.to_datetime(["2024-12-23"]),
                                 "exit_date": pd.to_datetime(["2024-12-27"])})
        daily = pd.DataFrame({"symbol": ["SH.600000"] * 3,
                              "date": pd.to_datetime(["2024-12-27", "2024-12-30", "2025-01-02"])})
        intraday = pd.DataFrame({"symbol": ["SH.600000"], "date": pd.to_datetime(["2024-12-26"]),
                                 "trade_week": pd.to_datetime(["2024-12-23"])})
        fold_daily, _ = select_execution_data(selected, daily, intraday)
        self.assertEqual(fold_daily.date.max(), pd.Timestamp("2024-12-30"))
        quotes = pd.DataFrame({"datetime": pd.to_datetime(["2024-12-30 14:56", "2025-01-02 14:56"]),
                               "symbol": ["SH.600000"] * 2, "open": [10., 10.], "high": [10.1, 10.1],
                               "low": [9.9, 9.9], "close": [10., 10.], "volume": [100, 100],
                               "amount": [1e5, 1e5]})
        store = create_fold_recovery_store(selected, quotes, fold_daily)
        self.assertIsNotNone(store.get_quote(pd.Timestamp("2024-12-30 14:56"), "SH.600000"))
        self.assertIsNone(store.get_quote(pd.Timestamp("2025-01-02 14:56"), "SH.600000"))


if __name__ == "__main__":
    unittest.main()
