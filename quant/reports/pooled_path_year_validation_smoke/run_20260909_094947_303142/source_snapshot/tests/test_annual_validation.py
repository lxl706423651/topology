"""年度流程的非破坏性单元检查；使用内存行情验证时间切分、缓存时钟和资金/T+1。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from quant_backtest.annual_panel import create_sampling_clock
from quant_backtest.annual_portfolio import MarketStore, run_lot_portfolio, calculate_daily_metrics
from quant_backtest.annual_training import create_year_folds, select_training_rows, select_training_features, fit_rank_model, select_signal_tradable_panel
from quant_backtest.factor_library import build_factor_library


def create_config() -> dict:
    """读取仓库YAML，仅覆盖内存烟测规模与容量，不读取或修改真实股票行情。"""
    cfg = yaml.safe_load(Path("config/annual_method_validation.yaml").read_text())
    cfg.update(min_cross_section=1, max_bar_amount_fraction=1.0, slippage_rate=0.0, min_closed_lots=1)
    cfg["model"]["n_estimators"] = 3
    cfg["workers"] = 1
    return cfg


def create_market() -> pd.DataFrame:
    """构造三天的一只股票行情，含同日止损触发和次日可卖出情形。"""
    times = pd.to_datetime(["2025-01-02 09:31", "2025-01-02 09:55", "2025-01-02 15:00",
                            "2025-01-03 09:31", "2025-01-03 09:55", "2025-01-03 15:00",
                            "2025-01-06 09:31", "2025-01-06 15:00"])
    close = np.array([10., 9.96, 9.96, 9.97, 9.96, 9.96, 9.95, 9.95])
    return pd.DataFrame({"datetime": times, "symbol": "SH.600000", "close": close, "atr": 0.01,
                         "interval_high": close, "volume": 100000., "amount": 10000000.,
                         "execution_close": close, "execution_high": close, "execution_low": close,
                         "execution_volume": 100000., "execution_amount": 10000000., "previous_close": 10.})


class AnnualValidationTests(unittest.TestCase):
    """检查逐年前向、标签隔离、模型序列化和独立批次账本的关键不变量。"""

    def test_missing_or_invalid_execution_inputs_fail_closed(self):
        """成交量、成交额、执行价或前收缺失/非正时不委托成交，不能让NaN比较绕过保护。"""
        for column in ["execution_volume", "execution_amount", "execution_close", "previous_close"]:
            for value in [np.nan, np.inf, 0.0, -1.0]:
                raw = create_market()
                raw.loc[0, column] = value
                scores = raw.iloc[:1][["datetime", "symbol"]].assign(prediction=1.0)
                rule = {"rule_id": "invalid_execution", "threshold": 0., "initial": 20., "trailing": 40., "max_days": 1}
                result = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"),
                    pd.Timestamp("2025-01-07"), "prediction", rule, create_config())
                self.assertEqual(result["metrics"]["buy_count"], 0, (column, value))

    def test_forward_folds_and_2026_separation(self):
        """2026可以在最终拟合中使用，但年度折的任何训练截止都早于测试年。"""
        cfg = create_config()
        for fold in create_year_folds(cfg):
            self.assertLess(fold["train_end"], fold["validation_start"])
            self.assertEqual(fold["validation_end"], fold["test_start"])
            self.assertLess(fold["train_end"].year, fold["year"])
        self.assertEqual(create_year_folds(cfg)[-1]["test_end"], pd.Timestamp("2026-05-01"))

    def test_purge_cross_boundary_labels(self):
        """训练标签触及边界即剔除，包括时间在训练期但未来收益来自验证期的行。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2024-06-28", "2024-06-30", "2024-07-01"]),
                              "label_end": pd.to_datetime(["2024-06-29", "2024-07-01", "2024-07-02"]),
                              "target_rank": [0.3, 0.5, 0.8], "cross_section_count": 100})
        out = select_training_rows(frame, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01"), 30)
        self.assertEqual(len(out), 1)

    def test_common_clock(self):
        """时钟不依赖股票有效行数，每天10个信号与1个日终估值点。"""
        dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
        clock = create_sampling_clock(dates, 24)
        self.assertEqual(len(clock), 22)
        self.assertIn(pd.Timestamp("2025-01-02 13:01"), clock)
        self.assertFalse(((clock.hour == 12) | ((clock.hour == 11) & (clock.minute > 30))).any())

    def test_signal_filter_ignores_execution_and_future_labels(self):
        """可买筛选只读取信号时行情，修改下一分钟成交字段或标签不改变候选行。"""
        cfg, market = create_config(), create_market()
        panel = market[["datetime", "symbol"]].assign(forward_return=0.01)
        market.loc[1, "close"] = 11.0
        first = select_signal_tradable_panel(panel, market, cfg)
        market["execution_close"] = 10000.0
        market["execution_amount"] = 0.0
        changed = panel.assign(forward_return=-1.0)
        second = select_signal_tradable_panel(changed, market, cfg)
        pd.testing.assert_frame_equal(first[["datetime", "symbol"]], second[["datetime", "symbol"]])
        self.assertNotIn(1, first.index)

    def test_raw_factors_do_not_change_when_future_is_appended(self):
        """完整274因子在追加未来行情后，已经完成的历史行必须不变。"""
        rng = np.random.default_rng(7)
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.001, 600)))
        raw = pd.DataFrame({"datetime": pd.date_range("2023-01-03 09:31", periods=600, freq="min"),
                            "open": close, "high": close + 0.02, "low": close - 0.02, "close": close,
                            "volume": rng.integers(100, 1000, 600)})
        raw["amount"] = raw.volume * raw.close * 100
        prefix = build_factor_library(raw.iloc[:400])
        full = build_factor_library(raw)
        pd.testing.assert_frame_equal(prefix, full.iloc[:400], check_exact=False, atol=1e-10, rtol=1e-8)

    def test_t1_independent_lots_and_no_membership_exit(self):
        """同日止损不能卖出；从目标池移除仍应按独立止损在次日卖，而不是目标变化强平。"""
        cfg = create_config()
        raw = create_market()
        scores = raw.iloc[[0]][["datetime", "symbol"]].assign(prediction=0.8)
        rule = {"initial": 2, "trailing": 4, "max_days": 0, "threshold": 0.5}
        result = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-07"), "prediction", rule, cfg)
        self.assertEqual(len(result["events"]), 2)
        trip = result["trips"].iloc[0]
        self.assertGreater(trip.sell_datetime.normalize(), trip.buy_datetime.normalize())
        self.assertIn("ATR", trip.sell_reason)
        self.assertTrue((result["events"].datetime > result["events"].signal_datetime).all())
        buy = result["events"].query("side == 'buy'").iloc[0]
        self.assertLessEqual(buy.notional, buy.signal_equity * 0.1)
        self.assertTrue((result["curve"].cash >= 0).all())

    def test_missing_execution_never_forward_filled(self):
        """下一分钟缺失时取消买入，不拿下一条任意时间的行情替代成交。"""
        cfg, raw = create_config(), create_market()
        raw.loc[0, "execution_close"] = np.nan
        scores = raw.iloc[[0]][["datetime", "symbol"]].assign(prediction=0.8)
        result = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-07"),
                                   "prediction", {"initial": 2, "trailing": 4, "max_days": 0, "threshold": 0.5}, cfg)
        self.assertTrue(result["events"].empty)
        self.assertEqual(result["metrics"]["total_return"], 0)

    def test_next_day_addons_have_independent_stops(self):
        """不同交易日同票买入形成不同批次，同日不能重复加仓，各自保留入场止损。"""
        cfg, raw = create_config(), create_market()
        raw["atr"] = 1.0
        prices = [10., 10., 10., 10.1, 10.1, 10.1, 10.2, 10.2]
        for column in ["close", "execution_close", "interval_high"]:
            raw[column] = prices
        scores = raw[["datetime", "symbol"]].assign(prediction=0.8)
        result = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-07"),
                                   "prediction", {"initial": 2, "trailing": 4, "max_days": 0, "threshold": 0.5}, cfg)
        self.assertEqual(len(result["open"]), 3)
        self.assertEqual(result["open"].stop_price.nunique(), 3)
        self.assertEqual(result["events"].datetime.dt.normalize().nunique(), len(result["events"]))

    def test_partial_exit_keeps_cost_basis_and_budget(self):
        """成交额不足时分批卖出，成本按股数摊销，剩余批次不能凭空消失。"""
        cfg, raw = create_config(), create_market()
        raw.loc[3:, "execution_amount"] = 10000.0
        scores = raw.iloc[[0]][["datetime", "symbol"]].assign(prediction=0.8)
        result = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-07"),
                                   "prediction", {"initial": 2, "trailing": 4, "max_days": 0, "threshold": 0.5}, cfg)
        buy = result["events"].query("side == 'buy'").iloc[0]
        self.assertEqual(buy.shares, result["trips"].shares.sum() + result["open"].shares.sum())
        self.assertAlmostEqual(buy.fee, result["trips"].buy_fee.sum() + result["open"].buy_fee.sum())
        self.assertEqual(result["metrics"]["closed_lot_count"], 0)
        unrealized = (result["open"].shares * (result["open"].mark_price - result["open"].buy_price) - result["open"].buy_fee).sum()
        reconciled = cfg["initial_cash"] + result["trips"].net_pnl.sum() + unrealized
        self.assertAlmostEqual(result["curve"].equity.iloc[-1], reconciled)

    def test_future_prices_do_not_change_earlier_orders(self):
        """任意修改未来行情，不得改变截止时刻之前的委托结果。"""
        cfg, raw = create_config(), create_market()
        scores = raw[["datetime", "symbol"]].assign(prediction=0.8)
        rule = {"initial": 2, "trailing": 4, "max_days": 0, "threshold": 0.5}
        baseline = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-07"), "prediction", rule, cfg)
        changed = raw.copy()
        changed.loc[changed.datetime >= "2025-01-06", ["close", "execution_close", "interval_high"]] *= 1.2
        perturbed = run_lot_portfolio(scores, MarketStore(changed), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-07"), "prediction", rule, cfg)
        pd.testing.assert_frame_equal(baseline["events"].query("datetime < '2025-01-06'"), perturbed["events"].query("datetime < '2025-01-06'"))

    def test_missing_mark_retains_value(self):
        """股票没有新行情时使用过去估值，不能凭空把已有持仓变为零。"""
        raw = create_market()
        raw.loc[1:, ["close", "execution_close", "interval_high"]] = np.nan
        cfg = create_config()
        scores = raw.iloc[[0]][["datetime", "symbol"]].assign(prediction=0.8)
        result = run_lot_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-07"),
                                   "prediction", {"initial": 2, "trailing": 4, "max_days": 0, "threshold": 0.5}, cfg)
        self.assertGreater(result["curve"].equity.min(), cfg["initial_cash"] * 0.99)

    def test_first_day_fees_in_metrics(self):
        """只有一天时也要把相对初始现金的亏损和回撤计算进去。"""
        curve = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-02 15:00"]), "equity": [990.], "exposure": [0.], "open_lots": [0]})
        metric = calculate_daily_metrics(curve, pd.DataFrame(), pd.DataFrame(), 1000.)
        self.assertAlmostEqual(metric["total_return"], -0.01)
        self.assertAlmostEqual(metric["max_drawdown"], -0.01)

    def test_final_training_contains_2026_but_year_fold_does_not(self):
        """同一面板可以用于2026历史测试和最终全量训练，但必须由不同的模型切片消费。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2025-05-01", "2026-04-28", "2026-04-30"]),
                              "label_end": pd.to_datetime(["2025-05-02", "2026-04-29", None]),
                              "target_rank": [0.3, 0.6, np.nan], "cross_section_count": 100})
        historical = select_training_rows(frame, pd.Timestamp("2023-01-01"), pd.Timestamp("2025-07-01"), 30)
        final = select_training_rows(frame, pd.Timestamp("2023-01-01"), pd.Timestamp("2026-05-01"), 30)
        self.assertFalse((historical.datetime.dt.year == 2026).any())
        self.assertTrue((final.datetime.dt.year == 2026).any())
        self.assertEqual(len(final), 2)

    def test_train_feature_selection_and_model_roundtrip(self):
        """训练特征不包含标签；JSON保存/重载保持预测一致。"""
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"datetime": pd.date_range("2023-01-01", periods=200, freq="h"), "symbol": "SH.600000",
                              "cs_rank_momentum_20": rng.uniform(size=200), "cs_rank_ret_std_20": rng.uniform(size=200),
                              "target_rank": rng.uniform(size=200), "forward_return": 1000.})
        cfg = create_config()
        features, _ = select_training_features(frame, cfg)
        self.assertNotIn("forward_return", features)
        model = fit_rank_model(frame, features, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save_model(path)
            restored = XGBRegressor()
            restored.load_model(path)
            np.testing.assert_allclose(model.predict(frame[features]), restored.predict(frame[features]))


if __name__ == "__main__":
    unittest.main()
