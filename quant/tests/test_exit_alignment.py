"""退出期限配对研究的合成回归测试；不读取真实测试预测或收益，不改外部行情。"""

from unittest.mock import Mock
import unittest

import numpy as np
import pandas as pd

from quant_backtest.annual_panel import create_sampling_clock
from quant_backtest.annual_portfolio import MarketStore, PositionLot, run_lot_portfolio
from quant_backtest.exit_alignment import HorizonExitPolicy, create_alignment_policy
from quant_backtest.learned_exit_models import LearnedExitPolicy
from scripts.run_exit_alignment_year_validation import validate_runner_compatibility
from scripts.run_learned_exit_year_validation import create_learned_policy, select_rules
from test_annual_validation import create_config, create_market


def create_alignment_market(days: int = 20) -> MarketStore:
    """生成days工作日完整公共时钟和恒定报价，含下一分钟容量；节假日映射由传入日历负责。"""
    times = create_sampling_clock(pd.bdate_range("2025-01-02", periods=days), 24)
    raw = create_market().iloc[[0]].loc[lambda x: x.index.repeat(len(times))].reset_index(drop=True)
    raw["datetime"] = times
    return MarketStore(raw)


def create_test_lot(market: MarketStore, index: int, lot_id: int = 1) -> PositionLot:
    """在market的index信号下一分钟构造lot_id批次，供独立期限及预测器路由验证。"""
    signal = market.times[index]
    return PositionLot(lot_id, 0, 100, signal + pd.Timedelta(minutes=1), 10., .81, signal, .01, 10., 0., index)


class ExitAlignmentTests(unittest.TestCase):
    """到期时刻、独立批次、T+1和ATR提前保护必须同时成立。"""

    def test_deadline_is_trading_days_at_signal_clock(self):
        """周四下午的3交易日到期在周二同刻，不是周日或周二开盘。"""
        market = create_alignment_market()
        lot = create_test_lot(market, 7)
        policy = HorizonExitPolicy(market, 3, "matched_horizon")
        self.assertEqual(policy(39, [lot]), {})
        self.assertEqual(market.times[40], pd.Timestamp("2025-01-07 13:49"))
        self.assertIn(lot.lot_id, policy(40, [lot]))
        self.assertEqual(policy(7, [lot]), {})

    def test_addons_have_independent_deadlines(self):
        """同股隔日加仓不能共享第一笔的到期时刻。"""
        market = create_alignment_market()
        lots = [create_test_lot(market, 7, 1), create_test_lot(market, 18, 2)]
        policy = HorizonExitPolicy(market, 3, "matched_horizon")
        self.assertEqual(set(policy(40, lots)), {1})
        self.assertEqual(set(policy(51, lots)), {1, 2})

    def test_minimum_horizon_filters_model_input(self):
        """最短期限前不调用模型，到期后仅把已到期批次交给模型，而不是必然卖出。"""
        market = create_alignment_market()
        lots = [create_test_lot(market, 7, 1), create_test_lot(market, 18, 2)]
        learned = Mock(return_value={})
        policy = HorizonExitPolicy(market, 3, "minimum_horizon_model", learned)
        self.assertEqual(policy(39, lots), {})
        learned.assert_not_called()
        self.assertEqual(policy(40, lots), {})
        self.assertEqual([p.lot_id for p in learned.call_args.args[1]], [1])

    def test_deadline_signal_does_not_inspect_execution_fields(self):
        """到期意图不以未来成交价/量决定；真实成交限制仍留给引擎。"""
        market = create_alignment_market()
        lot = create_test_lot(market, 7)
        policy = HorizonExitPolicy(market, 3, "matched_horizon")
        before = policy(40, [lot])
        market.values["execution_close"][:] = np.nan
        market.values["execution_volume"][:] = 0
        self.assertEqual(before, policy(40, [lot]))

    def test_next_minute_fill_matches_three_day_label(self):
        """仅一次入场时，到期卖出须精确落在3交易日后同刻下一分钟，即旧标签出口。"""
        market, cfg = create_alignment_market(), create_config()
        signal = market.times[7]
        scores = pd.DataFrame({"datetime": [signal], "symbol": market.symbols[0], "prediction": .1})
        rule = {"threshold": 0., "initial": 1000., "trailing": 1000., "max_days": 0}
        result = run_lot_portfolio(scores, market, signal, market.times[-1], "prediction", rule, cfg,
                                   HorizonExitPolicy(market, 3, "matched_horizon"))
        self.assertEqual(len(result["trips"]), 1)
        self.assertEqual(result["trips"].sell_datetime.iloc[0], market.times[40] + pd.Timedelta(minutes=1))

    def test_missing_exit_delays_without_changing_deadline(self):
        """到期下一分钟无量不能成交，待平仓意图在下一采样点继续尝试。"""
        market, cfg = create_alignment_market(), create_config()
        market.values["execution_volume"][40, 0] = 0
        signal = market.times[7]
        scores = pd.DataFrame({"datetime": [signal], "symbol": market.symbols[0], "prediction": .1})
        rule = {"threshold": 0., "initial": 1000., "trailing": 1000., "max_days": 0}
        result = run_lot_portfolio(scores, market, signal, market.times[-1], "prediction", rule, cfg,
                                   HorizonExitPolicy(market, 3, "matched_horizon"))
        self.assertEqual(result["trips"].sell_datetime.iloc[0], market.times[41] + pd.Timedelta(minutes=1))
        self.assertIn("预测期限3", result["trips"].sell_reason.iloc[0])

    def test_atr_protection_can_precede_model_minimum(self):
        """同日ATR触发仍不能卖，但次日应卖，不会被最短5日模型等待拦住。"""
        market, cfg = create_alignment_market(), create_config()
        market.values["close"][1, 0] = 9.1
        scores = pd.DataFrame({"datetime": [market.times[0]], "symbol": market.symbols[0], "prediction": .1})
        rule = {"threshold": 0., "initial": 2., "trailing": 4., "max_days": 20}
        result = run_lot_portfolio(scores, market, market.times[0], market.times[-1], "prediction", rule, cfg,
                                   HorizonExitPolicy(market, 5, "minimum_horizon_model", Mock(return_value={})))
        self.assertEqual(result["trips"].sell_datetime.iloc[0], market.times[11] + pd.Timedelta(minutes=1))
        self.assertIn("ATR", result["trips"].sell_reason.iloc[0])

    def test_factory_modes_and_default_compatibility(self):
        """旧接口缺省仍返回相同持仓模型，固定期限和T+1对照不假装启用了模型退出。"""
        market = create_alignment_market()
        model, matrices = object(), {}
        rule = {"exit_threshold": -.001, "profile": "three", "alignment_mode": "model_control"}
        cfg = {"model_profiles": [{"name": "three", "horizon": 3}]}
        old = create_learned_policy(model, matrices, market, rule, cfg)
        control = create_alignment_policy(model, matrices, market, rule, cfg)
        self.assertIsInstance(old, LearnedExitPolicy)
        self.assertIs(control.model, old.model)
        self.assertEqual(control.threshold, old.threshold)
        self.assertIsNone(create_alignment_policy(model, matrices, market, {**rule, "alignment_mode": "next_day_control"}, cfg))
        matched = create_alignment_policy(model, matrices, market, {**rule, "alignment_mode": "matched_horizon"}, cfg)
        self.assertEqual(matched.horizon, 3)

    def test_invalid_deadline_configuration_rejected(self):
        """非整数、零期限或缺少到期后模型都不能默默变为无限持仓。"""
        market = create_alignment_market()
        for horizon in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                HorizonExitPolicy(market, horizon, "matched_horizon")
        with self.assertRaises(ValueError):
            HorizonExitPolicy(market, 3, "minimum_horizon_model")

    def test_parent_compatibility_rejects_training_change(self):
        """退出工厂扩展可接受，但拟合函数或顶层配置变化必须阻止旧模型复用。"""
        old = "VALUE=1\ndef fit_return_model():\n return 1\ndef select_rules():\n return 1\n"
        permitted = "VALUE=1\ndef fit_return_model():\n return 1\ndef select_rules(policy_factory=None):\n return 2\n"
        self.assertTrue(validate_runner_compatibility(old, permitted))
        self.assertFalse(validate_runner_compatibility(old, permitted.replace("return 1", "return 3")))
        self.assertFalse(validate_runner_compatibility(old, permitted.replace("VALUE=1", "VALUE=2")))

    def test_wrong_runner_cannot_silently_use_old_exit(self):
        """不能把新期限规则误交给默认旧退出器；在任何预测和成交前拒绝错误入口。"""
        cfg = {"exit_candidates": [{"alignment_mode": "matched_horizon"}]}
        with self.assertRaisesRegex(ValueError, "对应退出工厂"):
            select_rules({}, None, [], None, {}, None, {}, cfg, None)


if __name__ == "__main__":
    unittest.main()
