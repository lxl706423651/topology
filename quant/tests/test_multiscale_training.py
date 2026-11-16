"""多尺度模型的因果性、目标期限、特征契约与配置继承检查；仅用合成行情。"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from quant_backtest.learned_exit_models import calculate_base_matrices, fit_return_model
from quant_backtest.multiscale_training import (build_context_panel, calculate_multiday_labels,
    select_context_features, select_exit_samples, MACRO_COLUMNS)
from scripts.run_multiscale_year_validation import build_horizon_training, load_layered_config, select_outer_feature_rows
from scripts.run_learned_exit_year_validation import generate_scores
from scripts.audit_multiscale_training_execution import calculate_execution_diagnostics
from test_learned_exit_validation import create_long_market, create_research_config


class MultiscaleTrainingTests(unittest.TestCase):
    """避免把未来多日标签或其他股票/年度混入当前因果特征。"""

    def test_multiday_labels_use_trading_calendar_and_delay(self):
        """3日标签要跨3个实际交易日、买卖均下一分钟，不能当成3行或3自然日。"""
        market = create_long_market()
        frame = pd.DataFrame({"datetime": market.times[[0, 11]], "symbol": market.symbols[0]})
        ret, end = calculate_multiday_labels(frame, market, 3)
        np.testing.assert_allclose(ret, market.values["execution_close"][[33, 44], 0] / market.values["execution_close"][[0, 11], 0] - 1)
        self.assertEqual(end[0], market.times[33] + pd.Timedelta(minutes=1))

    def test_future_labels_never_remove_prediction_rows(self):
        """尾部未来价格缺失只影响训练标签，不得删除最新可预测时间。"""
        market, cfg = create_long_market(), create_research_config()
        cfg["entry_horizons"] = [3, 5]
        frame = pd.DataFrame({"datetime": market.times[market.times.hour != 15], "symbol": market.symbols[0]})
        out = build_context_panel(frame, market, calculate_base_matrices(market), cfg)
        self.assertEqual(len(out), len(frame))
        self.assertTrue(out.future_return_5d.tail(50).isna().all())

    def test_future_changes_do_not_change_context_history(self):
        """未来价格改变可以改变标签，但过去的截面/环境特征必须完全不变。"""
        market, cfg = create_long_market(), create_research_config()
        cfg["entry_horizons"] = [3, 5]
        frame = pd.DataFrame({"datetime": market.times[market.times.hour != 15], "symbol": market.symbols[0]})
        first = build_context_panel(frame, market, calculate_base_matrices(market), cfg)
        market.values["close"][200:] *= 1.5
        market.values["execution_close"][200:] *= 1.5
        second = build_context_panel(frame, market, calculate_base_matrices(market), cfg)
        cols = [c for c in first if "context" in c]
        pd.testing.assert_frame_equal(first.loc[first.datetime < market.times[200], cols], second.loc[second.datetime < market.times[200], cols])

    def test_horizon_label_purge_and_input_immutable(self):
        """3日/5日标签的清理按各自结束时间，不能沿用旧1日label_end越过验证边界。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                              "future_return_3d": [.01, .02], "future_end_3d": pd.to_datetime(["2025-01-07", "2025-01-08"]),
                              "future_return_5d": [.02, .03], "future_end_5d": pd.to_datetime(["2025-01-09", "2025-01-10"]),
                              "label_end": pd.to_datetime(["2025-01-03", "2025-01-06"]), "forward_return": 1000.})
        original = frame.copy()
        windows = [{"start": pd.Timestamp("2025-01-01"), "end": pd.Timestamp("2025-01-10")}]
        self.assertEqual(len(build_horizon_training(frame, windows, 3)), 2)
        self.assertEqual(len(build_horizon_training(frame, windows, 5)), 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_current_eligible_entry_filters_virtual_lots(self):
        """持仓样本由历史入场时资格筛选，不要求现在仍在池，也不读取未来收益。"""
        samples = pd.DataFrame({"entry_datetime": pd.to_datetime(["2025-01-02 09:32", "2025-01-02 09:56"]),
                                "datetime": pd.to_datetime(["2025-02-01", "2025-02-01"]), "symbol": "SH.600000"})
        panel = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-02 09:31"]), "symbol": "SH.600000"})
        self.assertEqual(list(select_exit_samples(samples, panel).index), [0])

    def test_context_columns_reach_model_without_labels(self):
        """非cs_rank的环境特征必须按契约送入模型，任何未来标签都不能成为输入。"""
        cfg = create_research_config()
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=300), "symbol": "SH.600000",
                              "cs_rank_context_momentum_3d": rng.uniform(size=300), "target_rank": rng.normal(size=300)})
        for c in MACRO_COLUMNS:
            frame[c] = rng.uniform(size=300)
        frame["future_return_5d"] = 1e9
        features, _ = select_context_features(frame, {**cfg, "max_features": 8})
        self.assertTrue(set(MACRO_COLUMNS).issubset(features))
        self.assertNotIn("future_return_5d", features)
        model = fit_return_model(frame, features, frame.target_rank.to_numpy(), cfg, {})
        first = generate_scores(model, frame, features, cfg)
        changed = generate_scores(model, frame.assign(future_return_5d=-1e9), features, cfg)
        pd.testing.assert_frame_equal(first, changed)

    def test_layered_config_preserves_cost_and_position_constraints(self):
        """新增YAML要继承已有万八免5和10%预算，不能因为分层配置而回退到乐观费用。"""
        cfg = load_layered_config(Path("config/multiscale_year_validation.yaml"))
        self.assertEqual(cfg["commission_rate"], .0008)
        self.assertEqual(cfg["min_commission"], 0.)
        self.assertEqual(cfg["entry_equity_fraction"], .1)
        self.assertEqual(cfg["entry_top_n"], 5)
        self.assertEqual(cfg["slippage_rate"], .0005)

    def test_diagnostics_label_profit_cannot_imply_possible_fill(self):
        """未来标签即使盈利，下一分钟涨停或容量不足也不能记作可成交意图。"""
        cfg, market = create_research_config(), create_long_market()
        indices = [0, 11, 22]
        market.values["execution_close"][11, 0] = 11.5
        market.values["execution_amount"][22, 0] = 0
        frame = pd.DataFrame({"datetime": market.times[indices], "symbol": market.symbols[0], "prediction": .02,
                              "future_return_3d": .03, "future_end_3d": market.times[[33, 44, 55]] + pd.Timedelta(minutes=1)})
        out = calculate_execution_diagnostics(frame, market, cfg, 3, market.times[-1])
        self.assertEqual(out["candidate_count"], 3)
        self.assertEqual(out["buyable_count"], 1)
        self.assertEqual(out["entry_price_limit_count"], 1)
        self.assertEqual(out["insufficient_board_lot_count"], 1)
        self.assertGreater(out["mean_label_return"], out["capacity_weighted_label_proxy"])
        empty = calculate_execution_diagnostics(frame.iloc[:0], market, cfg, 3, market.times[-1])
        self.assertEqual(empty["buyable_count"], 0)

    def test_feature_embargo_excludes_heldout_price_lookback(self):
        """留出年后前60交易日也须排除；第61日的60日动量只读取非留出年价格。"""
        dates = pd.bdate_range("2023-09-01", "2024-07-01")
        times = dates + pd.Timedelta(hours=9, minutes=31)
        frame = pd.DataFrame({"datetime": times, "symbol": "SH.600000"})
        original = np.arange(len(frame), dtype=float) + 100
        perturbed = original.copy()
        perturbed[times.year == 2023] *= 2
        first = pd.Series(original).pct_change(60, fill_method=None)
        second = pd.Series(perturbed).pct_change(60, fill_method=None)
        filtered = select_outer_feature_rows(frame, times, 2023, 60)
        np.testing.assert_allclose(first.loc[filtered.index], second.loc[filtered.index])
        following = times[times.year == 2024]
        self.assertEqual(filtered.datetime.min(), following[60])
        self.assertFalse(filtered.datetime.isin(following[:60]).any())
        with self.assertRaises(ValueError):
            select_outer_feature_rows(frame, times, 2023, 7)

    def test_embargo_does_not_discard_2026_from_all_folds(self):
        """2026仍可进入2023/2024留出折拟合；2025折严格隔离时不能声称2026核心行仍然存在。"""
        dates = pd.bdate_range("2023-01-03", "2026-04-30")
        frame = pd.DataFrame({"datetime": dates})
        for year in [2023, 2024]:
            out = select_outer_feature_rows(frame, dates, year, 60)
            self.assertTrue((out.datetime.dt.year == 2026).any())
        out = select_outer_feature_rows(frame, dates, 2026, 60)
        self.assertFalse((out.datetime.dt.year == 2026).any())


if __name__ == "__main__":
    unittest.main()
