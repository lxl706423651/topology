"""原XGBoost/CTA年度训练和执行契约测试，全部使用内存行情，不查看真实外层。"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import yaml

from quant_backtest.annual_portfolio import MarketStore
from quant_backtest.original_method_portfolio import run_original_portfolio
from scripts.run_original_method_year_validation import select_original_rows, calculate_original_gates, audit_original_result, describe_original_feature
from quant_backtest.original_method_data import read_original_features


class OriginalMethodValidationTests(unittest.TestCase):
    """覆盖截面目标来源、方法独立门槛，以及原2/4退出和不同撮合时序。"""

    def create_inputs(self):
        """返回两日内存报价、只一次入场信号和仓库费用配置；不复用历史模型。"""
        times = pd.to_datetime(["2025-01-02 09:31", "2025-01-02 09:55", "2025-01-02 15:00",
            "2025-01-03 09:31", "2025-01-03 09:55", "2025-01-03 15:00"])
        prices = np.array([10., 10., 10., 10.1, 9.85, 9.8])
        raw = pd.DataFrame({"datetime": times, "symbol": "SH.600000", "close": prices, "high": prices,
            "atr": .05, "execution_atr": .05, "interval_high": prices, "volume": 100000., "amount": 1000000.,
            "execution_close": prices, "execution_high": prices, "execution_volume": 100000., "execution_amount": 1000000., "previous_close": 10.})
        scores = raw[["datetime", "symbol"]].loc[raw.datetime.dt.hour != 15].copy()
        scores["prediction"] = [-1.] * len(scores)
        scores.iloc[0, scores.columns.get_loc("prediction")] = 1.
        scores["cta_trend_breakout_score"] = scores.prediction
        cfg = yaml.safe_load(Path("config/original_method_year_validation.yaml").read_text())
        cfg.update(min_cross_section=1, initial_cash=100000.)
        return raw, scores, cfg

    def test_target_rank_rebuilt_after_label_purging(self):
        """另一股票标签超出核心边界，不能通过预计算截面排名污染合法核心目标。"""
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2025-02-03"] * 3), "symbol": ["a", "b", "c"],
            "source_start": pd.to_datetime(["2025-01-02"] * 3), "pool_source_start": pd.to_datetime(["2025-01-27"] * 3),
            "label_end": pd.to_datetime(["2025-02-04", "2025-02-04", "2025-02-06"]),
            "forward_return": [.1, .2, .3], "target_rank": [1/3, 2/3, 1.], "cross_section_count": 3})
        windows = [{"start": pd.Timestamp("2025-02-01"), "end": pd.Timestamp("2025-02-05")}]
        result = select_original_rows(frame, windows, {"min_cross_section": 2}, True)
        self.assertEqual(result.symbol.tolist(), ["a", "b"])
        self.assertEqual(result.target_rank.tolist(), [.5, 1.])

    def test_gate_uses_own_method_and_no_future_fields(self):
        """CTA与XGB分别校准，增删未来收益/成交列不能改变门槛。"""
        raw, scores, cfg = self.create_inputs()
        scores["cta_trend_breakout_score"] += .01
        original = calculate_original_gates(scores, cfg)
        self.assertNotEqual(original["method1"], original["method2"])
        self.assertEqual(original, calculate_original_gates(scores.assign(forward_return=1e9, entry_price=np.nan), cfg))

    def test_two_modes_report_timing_and_accounting(self):
        """原2/4ATR真实触发退出；两种口径账本一致，但旧卖出明确标记同刻决策。"""
        raw, scores, cfg = self.create_inputs()
        for mode in ["legacy", "constrained"]:
            result = run_original_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04"), "method1", .5, mode, cfg)
            self.assertEqual(result["metrics"]["buy_count"], 1)
            self.assertEqual(result["metrics"]["sell_count"], 1)
            self.assertTrue(audit_original_result(result, mode, cfg)["passed"])
            sell = result["events"].loc[result["events"].side == "sell"].iloc[0]
            expected = pd.Timedelta(minutes=0 if mode == "legacy" else 1)
            self.assertEqual(sell.datetime - sell.signal_datetime, expected)

    def test_execution_price_does_not_generate_strict_exit_signal(self):
        """下一分钟大跌只会影响撮合，不允许约束版在上一分钟凭此提前决定卖出。"""
        raw, scores, cfg = self.create_inputs()
        raw.loc[3, ["execution_close", "execution_high"]] = 9.4
        results = {mode: run_original_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"),
            pd.Timestamp("2025-01-04"), "method1", .5, mode, cfg) for mode in ["legacy", "constrained"]}
        self.assertEqual(results["legacy"]["trips"].sell_datetime.iloc[0], pd.Timestamp("2025-01-03 09:32"))
        self.assertEqual(results["constrained"]["trips"].sell_datetime.iloc[0], pd.Timestamp("2025-01-03 09:56"))

    def test_no_trade_is_valid_cash_account(self):
        """无买入的年度仍输出现金曲线，不因旧函数缺成交列而崩溃。"""
        raw, scores, cfg = self.create_inputs()
        for mode in ["legacy", "constrained"]:
            result = run_original_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04"), "method1", 9., mode, cfg)
            self.assertEqual(result["metrics"]["buy_count"], 0)
            self.assertTrue(audit_original_result(result, mode, cfg)["passed"])

    def test_empty_period_still_outputs_cash(self):
        """完全没有候选行的年份片段也必须保留，不能因旧函数空表排序失败而丢弃。"""
        raw, scores, cfg = self.create_inputs()
        for mode in ["legacy", "constrained"]:
            result = run_original_portfolio(scores.iloc[:0], MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04"), "method1", .5, mode, cfg)
            self.assertTrue(audit_original_result(result, mode, cfg)["passed"])

    def test_close_mark_does_not_add_an_exit_rule(self):
        """日终大跌只用于估值，不能额外产生原策略没有的次日待平仓信号。"""
        raw, scores, cfg = self.create_inputs()
        raw.loc[2, ["close", "high", "interval_high"]] = 8.
        result = run_original_portfolio(scores, MarketStore(raw), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04"), "method1", .5, "constrained", cfg)
        self.assertEqual(result["trips"].sell_datetime.iloc[0], pd.Timestamp("2025-01-03 09:56"))
        self.assertTrue(audit_original_result(result, "constrained", cfg)["passed"])

    def test_all_original_features_have_chinese_explanations(self):
        """旧156列全部有中文含义，不能沿用不易读的英文元数据作为最终说明。"""
        features, _ = read_original_features(Path("reports/xgboost_long_only_backtest/run_20260526_202355"))
        self.assertTrue(all("截面百分位" in describe_original_feature(c) for c in features))


if __name__ == "__main__":
    unittest.main()
