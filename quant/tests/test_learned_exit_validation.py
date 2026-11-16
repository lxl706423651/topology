"""双模型年度实验的非破坏性检查；内存行情验证隔离、批次状态、缓存、成本与四年均值口径。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from quant_backtest.annual_panel import create_sampling_clock
from quant_backtest.annual_portfolio import MarketStore, PositionLot, run_lot_portfolio
from quant_backtest.learned_exit_models import (EXIT_FEATURES, build_research_folds, select_window_rows,
    calculate_entry_target, calculate_base_matrices, build_lot_features, fit_return_model, load_exit_cache, LearnedExitPolicy)
from scripts.run_learned_exit_year_validation import calculate_four_year_summary, generate_scores, select_rules
from test_annual_validation import create_config, create_market


def create_research_config() -> dict:
    """构造小规模配置，不改变实际研究YAML；控制虚拟持仓抽样和四年切片。"""
    cfg = create_config()
    cfg.update(test_years=[2023, 2024, 2025, 2026], fit_fraction=.6, calibration_fraction=.2,
               embargo_days=7, label_clip=.2, exit_ages=[1, 3], exit_entry_stride=1,
               entry_min_net_return=0., sharpe_target=3.)
    return cfg


def create_long_market(days: int = 30) -> MarketStore:
    """生成days个工作日的完整24分钟时钟、单股非恒定行情，供因果特征与训练缓存检查。"""
    times = create_sampling_clock(pd.bdate_range("2025-01-02", periods=days), 24)
    raw = create_market().iloc[[0]].loc[lambda x: x.index.repeat(len(times))].reset_index(drop=True)
    raw["datetime"] = times
    raw["close"] = 10 + np.sin(np.arange(len(raw)) / 13) * .1
    raw["execution_close"] = raw.close + .002
    raw["interval_high"] = raw.close + .01
    return MarketStore(raw)


class LearnedExitTests(unittest.TestCase):
    """检查四折训练隔离与实际模型平仓路径，而不是仅检查语法。"""

    def test_method_thresholds_are_not_overwritten(self):
        """同名规则须按方法隔离；各方法的校准、验证和冻结门槛应完全一致。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2024-03-01", "2024-03-02", "2025-03-01", "2025-03-02"]),
                              "symbol": "SH.600000", "prediction": [.01, .03, .01, .03],
                              "cta_trend_breakout_score": [.1, .3, .1, .3]})
        fold = {"year": 2026, "calibration": [{"year": 2024, "start": pd.Timestamp("2024-01-01"), "end": pd.Timestamp("2025-01-01")}],
                "validation": [{"year": 2025, "start": pd.Timestamp("2025-01-01"), "end": pd.Timestamp("2026-01-01")}]}
        cfg = create_research_config()
        cfg.update(gate_quantiles=[.5, .9], train_shortlist=2, min_closed_lots=1, minimum_positive_validation_segments=1,
                   exit_candidates=[{"name": "test_exit", "exit_threshold": 0.}])
        summary = {"mean_sharpe": 1., "worst_drawdown": -.01, "closed_lots": 2, "positive_segments": 1, "active_segments": 1}
        module = "scripts.run_learned_exit_year_validation"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch(module + ".generate_scores", side_effect=lambda model, part, features, config: part.copy()), \
                 patch(module + ".LearnedExitPolicy", return_value=None), \
                 patch(module + ".evaluate_segments", return_value=(summary, [])):
                winners = select_rules({"profile": object()}, object(), [], frame, fold, None, {}, cfg, directory)
            calibration = pd.read_csv(directory / "training_rule_metrics.csv")
            validation = pd.read_csv(directory / "validation_rule_metrics.csv")
            joined = calibration.merge(validation, on=["method", "rule_id"], suffixes=("_calibration", "_validation"))
            self.assertEqual(len(joined), 4)
            np.testing.assert_allclose(joined.threshold_calibration, joined.threshold_validation, rtol=0, atol=0)
            for method, winner in winners.items():
                row = calibration.loc[(calibration.method == method) & (calibration.rule_id == winner["rule_id"])].iloc[0]
                self.assertAlmostEqual(winner["threshold"], row.threshold)
            self.assertNotEqual(winners["method1"]["threshold"], winners["method2"]["threshold"])

    def test_four_folds_include_2026_training(self):
        """四年均为留出年一次；前三折含2026拟合，最后一折不能拟合2026。"""
        times = pd.bdate_range("2023-01-03", "2026-04-30")
        folds = build_research_folds(times, create_research_config())
        self.assertEqual([f["year"] for f in folds], [2023, 2024, 2025, 2026])
        for fold in folds:
            self.assertNotIn(fold["year"], fold["training_years"])
            self.assertEqual(fold["future_trained"], fold["year"] < 2026)
            if fold["year"] < 2026:
                self.assertIn(2026, [w["year"] for w in fold["fit"]])

    def test_label_and_lot_entry_purge(self):
        """买入早于训练窗口或收益标签触及窗口边界的虚拟持仓均不得拟合。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2025-02-01"] * 3),
                              "entry_datetime": pd.to_datetime(["2025-01-05", "2025-01-20", "2025-01-20"]),
                              "label_end": pd.to_datetime(["2025-02-02", "2025-03-01", "2025-02-02"])})
        window = [{"start": pd.Timestamp("2025-01-10"), "end": pd.Timestamp("2025-03-01")}]
        self.assertEqual(list(select_window_rows(frame, window).index), [2])

    def test_entry_target_accounts_costs_and_tax_date(self):
        """无价格收益也应亏损全套成本；印花税减半之前净收益更低。"""
        cfg = create_research_config()
        cfg["slippage_rate"] = .0005
        frame = pd.DataFrame({"forward_return": [0., 0.], "label_end": pd.to_datetime(["2023-08-25", "2023-08-28"])})
        target = calculate_entry_target(frame, cfg)
        self.assertTrue((target < -30).all())
        self.assertLess(target[0], target[1])

    def test_future_data_does_not_change_features(self):
        """修改未来行情及全部下一分钟价格，都不能改变截止点前的持仓特征。"""
        market = create_long_market()
        first = calculate_base_matrices(market)
        market.values["close"][200:] *= 2
        market.values["execution_close"][:] *= 100
        second = calculate_base_matrices(market)
        for col in first:
            np.testing.assert_allclose(first[col][:200], second[col][:200], equal_nan=True)

    def test_lot_price_and_age_are_independent(self):
        """同股不同买价和入场日期产生不同状态，不能把加仓混成一个均价退出。"""
        market = create_long_market()
        matrices = calculate_base_matrices(market)
        features = build_lot_features(matrices, market, np.array([100, 100]), np.array([0, 0]),
                                      np.array([9., 11.]), np.array([0, 80]))
        self.assertNotEqual(features[0, -3], features[1, -3])
        self.assertNotEqual(features[0, -2], features[1, -2])
        self.assertEqual(features.shape[1], len(EXIT_FEATURES))

    def test_exit_cache_roundtrip_and_invalidation(self):
        """首次生成持仓pkl，第二次复用逐值一致，改变成本后换key；不触碰真实年度面板。"""
        cfg, market = create_research_config(), create_long_market()
        panel = pd.DataFrame({"datetime": market.times[market.times.hour != 15], "symbol": market.symbols[0]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text("{}")
            panel.to_pickle(root / "panel.pkl")
            panel.to_pickle(root / "market.pkl")
            cfg["aux_cache_dir"] = str(root / "cache")
            a, samples, info = load_exit_cache(panel, market, {"cache_path": tmp}, cfg)
            b, again, info2 = load_exit_cache(panel, market, {"cache_path": tmp}, cfg)
            pd.testing.assert_frame_equal(samples, again)
            self.assertEqual(info, info2)
            np.testing.assert_allclose(a["momentum_1"], b["momentum_1"], equal_nan=True)
            cfg["slippage_rate"] += .001
            _, _, other = load_exit_cache(panel, market, {"cache_path": tmp}, cfg)
            self.assertNotEqual(info["key"], other["key"])

    def test_heldout_mutation_does_not_change_model(self):
        """修改外层年份的特征/标签，选出的拟合行和保存模型预测应完全一致。"""
        cfg = create_research_config()
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=800),
                              "feature": rng.normal(size=800), "target": rng.normal(size=800)})
        frame["label_end"] = frame.datetime + pd.Timedelta(days=1)
        window = [{"start": pd.Timestamp("2024-01-10"), "end": pd.Timestamp("2025-01-01")}]
        original = select_window_rows(frame, window)
        frame.loc[frame.datetime.dt.year == 2025, ["feature", "target"]] = 1e9
        changed = select_window_rows(frame, window)
        pd.testing.assert_frame_equal(original, changed)
        one = fit_return_model(original, ["feature"], original.target.to_numpy(), cfg, {})
        two = fit_return_model(changed, ["feature"], changed.target.to_numpy(), cfg, {})
        np.testing.assert_array_equal(one.predict(original[["feature"]]), two.predict(original[["feature"]]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            one.save_model(path)
            restored = XGBRegressor()
            restored.load_model(path)
            np.testing.assert_array_equal(one.predict(original[["feature"]]), restored.predict(original[["feature"]]))

    def test_learned_exit_after_leaving_universe_and_t1(self):
        """实际模型可在没有新入池分数时卖出旧持仓，且不能同日卖出。"""
        cfg, market = create_research_config(), create_long_market()
        matrices = calculate_base_matrices(market)
        training = pd.DataFrame(np.zeros((100, len(EXIT_FEATURES))), columns=EXIT_FEATURES)
        training["datetime"] = pd.date_range("2024-01-01", periods=100)
        model = fit_return_model(training, EXIT_FEATURES, np.full(100, -100.), cfg, {})
        policy = LearnedExitPolicy(model, matrices, market, 0.)
        entry = market.times[70]
        scores = pd.DataFrame({"datetime": [entry], "symbol": market.symbols[0], "prediction": .01})
        rule = {"initial": 1000., "trailing": 1000., "max_days": 0, "threshold": 0.}
        result = run_lot_portfolio(scores, market, entry, market.times[-1] + pd.Timedelta(days=1), "prediction", rule, cfg, policy)
        self.assertEqual(len(result["trips"]), 1)
        trip = result["trips"].iloc[0]
        self.assertGreater(trip.sell_datetime.normalize(), trip.buy_datetime.normalize())
        self.assertTrue(trip.sell_reason.startswith("持仓模型"))

    def test_four_year_summary_never_drops_cash_year(self):
        """不能靠丢弃空仓年/亏损年让平均夏普达标。"""
        table = pd.DataFrame([{"method": m, "year": y, "sharpe": 4., "annual_volatility": .1, "total_return": .1}
                              for m in ["method1", "method2"] for y in [2023, 2024, 2025, 2026]])
        cfg = create_research_config()
        self.assertTrue(calculate_four_year_summary(table, cfg).target_passed.all())
        table.loc[table.year == 2026, "annual_volatility"] = 0
        out = calculate_four_year_summary(table, cfg)
        self.assertTrue(out.four_year_mean_sharpe.isna().all())
        self.assertFalse(out.target_passed.any())
        table.loc[table.year == 2026, ["annual_volatility", "sharpe"]] = [.1, -5.]
        self.assertTrue((calculate_four_year_summary(table, cfg).four_year_mean_sharpe == 1.75).all())


if __name__ == "__main__":
    unittest.main()
