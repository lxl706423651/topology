"""验证新增会话特征和当前锁板门控的因果性，仅用合成盘中记录。"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_backtest.weekly_t_features import add_session_context_features, add_weekly_execution_features, select_recoverable_signal_sides


class SessionFeatureTests(unittest.TestCase):
    """测试缺失、同刻截面和未来扰动，避免回测可成交性混入预测输入。"""

    def create_frame(self):
        """返回两个当前股票与一个较晚样本，带不应被读取的未来价格标签。"""
        return pd.DataFrame({"datetime": pd.to_datetime(["2024-03-05 10:00"] * 2 + ["2024-03-05 11:00"]),
                             "symbol": ["A", "B", "A"], "signal_price": [10.5, 10., 11.],
                             "previous_close": [10., 10., 10.], "f_open_return": [0., 0., 11 / 10.5 - 1],
                             "f_vwap_deviation": [0., -.01, .02], "f_previous_day_amplitude": [.02] * 3,
                             "target_return": [1., 2., 3.], "recovery_price": [20., 20., 20.]})

    def test_gap_distinguishes_same_intraday_return(self):
        """相同open内涨幅可以有不同隔夜跳空，模型必须能区别。"""
        frame = self.create_frame()
        result = add_session_context_features(frame)
        self.assertAlmostEqual(result.f_overnight_gap.iloc[0], .05, places=6)
        self.assertAlmostEqual(result.f_market_day_return.iloc[0], .025, places=6)
        self.assertAlmostEqual(result.f_relative_day_return.iloc[0], .025, places=6)
        self.assertNotIn("f_day_return", frame)

    def test_future_changes_do_not_affect_earlier_features(self):
        """修改未来时间行情及所有标签/恢复价不能改变当前两股特征。"""
        frame = self.create_frame()
        old = add_session_context_features(frame)
        frame.loc[2, "signal_price"] = 1000
        frame["target_return"] = -999
        frame["recovery_price"] = 999
        new = add_session_context_features(frame)
        pd.testing.assert_frame_equal(old.filter(like="f_").iloc[:2], new.filter(like="f_").iloc[:2])

    def test_locked_recovery_direction_only_uses_current_bar(self):
        """正常+6%平价不会被当5%锁板；买先在当前跌停代理价被拒绝，与未来价无关。"""
        prices = [9.5, 10.5, 10.6, 9.4]
        frame = pd.DataFrame({"signal_price": prices, "signal_high": prices, "signal_low": prices,
                              "previous_close": [10.] * 4, "recovery_price": [100.] * 4})
        actions = pd.Series([1, -1, -1, 1])
        cfg = {"t_avoid_locked_exit_side": True, "flat_st_limit_rate": .05, "nonflat_limit_proxy": .098}
        permitted = select_recoverable_signal_sides(frame, actions, cfg)
        self.assertEqual(permitted.tolist(), [False, False, True, True])
        frame.recovery_price = np.nan
        pd.testing.assert_series_equal(permitted, select_recoverable_signal_sides(frame, actions, cfg))

    def test_nonflat_at_ten_percent_and_missing_quote_are_blocked(self):
        """10%代理与执行器一致，不因该分钟先前曾有其他价格而允许不利恢复方向。"""
        frame = pd.DataFrame({"signal_price": [11., 10.], "signal_high": [11., np.nan],
                              "signal_low": [10.9, 9.9], "previous_close": [10., 10.]})
        cfg = {"t_avoid_locked_exit_side": True, "flat_st_limit_rate": .05, "nonflat_limit_proxy": .098}
        self.assertEqual(select_recoverable_signal_sides(frame, pd.Series([-1, 1]), cfg).tolist(), [False, False])
        with self.assertRaises(ValueError):
            select_recoverable_signal_sides(frame, pd.Series([-1, 1], index=[2, 3]), cfg)

    def test_weekly_auction_liquidity_uses_only_feature_week(self):
        """未来周巨大成交额不能改善过去周资格；过去缺失一天收盘量则失败关闭。"""
        weekly = pd.DataFrame({"feature_week": pd.to_datetime(["2024-03-04", "2024-03-04"]),
                               "symbol": ["A", "B"], "f_week_days": [2, 2]})
        daily = pd.DataFrame({"date": pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-04", "2024-03-11"]),
                              "symbol": ["A", "A", "B", "B"], "close": [10.] * 4,
                              "close_amount": [600000., 900000., 800000., 1e9]})
        result = add_weekly_execution_features(weekly, daily)
        self.assertEqual(result.f_prior_close_auction_amount_min.tolist(), [600000., 0.])
        self.assertEqual(result.f_missing_prior_close_days.tolist(), [0., 1.])


if __name__ == "__main__":
    unittest.main()
