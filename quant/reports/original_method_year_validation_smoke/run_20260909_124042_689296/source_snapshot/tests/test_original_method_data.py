"""原模型年界数据的非破坏性测试：检查公式、标签和未来年份隔离，不读取测试收益。"""

import unittest

import numpy as np
import pandas as pd

from quant_backtest.factor_library import build_factor_library
from quant_backtest.annual_panel import create_sampling_clock
from quant_backtest.original_method_data import calculate_recursive_columns, build_original_symbol


class OriginalMethodDataTests(unittest.TestCase):
    """使用完整原库作为数值对照，确保没有换指标或把其他年份的递推状态带入。"""

    def create_inputs(self):
        """返回两年各三天内存分钟、全历史原库缓存和有限/递推混合列。"""
        rng = np.random.default_rng(17)
        days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2025-01-02", "2025-01-03", "2025-01-06"])
        offsets = np.r_[np.arange(571, 691), np.arange(781, 901)]
        dates = pd.DatetimeIndex((days.to_numpy()[:, None] + offsets[None, :] * np.timedelta64(1, "m")).ravel())
        prices = 10 * np.exp(np.cumsum(rng.normal(0, .001, len(dates))))
        raw = pd.DataFrame({"datetime": dates, "open": prices, "close": prices, "high": prices + .01,
                            "low": prices - .01, "volume": 1000., "amount": prices * 100000})
        factors = build_factor_library(raw)
        factors.index = dates
        clock = create_sampling_clock(days, 24)
        market = raw.set_index("datetime").reindex(clock)[["close", "volume", "amount"]].assign(symbol="SH.600000")
        selected = factors.reindex(clock[clock.hour != 15]).reset_index(names="datetime")
        columns = ["cs_rank_momentum_20", "cs_rank_ema_gap_20", "cs_rank_rsi_14", "cs_rank_adx_14",
                   "cs_rank_atr_ratio_14", "cs_rank_macd_hist_12_26_9", "cs_rank_trend_slope_20", "cs_rank_obv_z_60"]
        return raw, {"panel": selected, "market": market.reset_index(names="datetime"), "audit": {"symbol": "SH.600000"}}, columns

    def test_recursive_formulas_equal_original_library(self):
        """新代码只分年初始化，不改变原库EMA/Wilder指标和趋势公式。"""
        raw, _, cols = self.create_inputs()
        year = raw.loc[raw.datetime.dt.year == 2025].reset_index(drop=True)
        names = [c.removeprefix("cs_rank_") for c in cols[1:]]
        pd.testing.assert_frame_equal(calculate_recursive_columns(year, names), build_factor_library(year)[names], check_dtype=False)

    def test_other_year_price_changes_do_not_change_recursive_features(self):
        """大幅改变前一年价量也不改变下一年重置的递推列；有限列在241根后复用原缓存。"""
        raw, cached, cols = self.create_inputs()
        expected = build_original_symbol(raw, cached, cols)
        changed = raw.copy()
        changed.loc[changed.datetime.dt.year == 2024, ["open", "close", "high", "low", "amount"]] *= 7
        actual = build_original_symbol(changed, cached, cols)
        for key in ["panel", "market"]:
            a = expected[key].loc[expected[key].datetime.dt.year == 2025].reset_index(drop=True)
            b = actual[key].loc[actual[key].datetime.dt.year == 2025].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)

    def test_labels_are_24_observed_minutes_and_missing_future_not_removed(self):
        """标签用t+1至t+25，年末缺标签行保留推断，来源不跨年。"""
        raw, cached, cols = self.create_inputs()
        panel = build_original_symbol(raw, cached, cols)["panel"]
        row = panel.loc[panel.label_end.notna()].iloc[0]
        i = raw.index[raw.datetime == row.datetime][0]
        self.assertEqual(row.label_end, raw.datetime.iloc[i + 25])
        self.assertAlmostEqual(row.forward_return, raw.close.iloc[i + 25] / raw.close.iloc[i + 1] - 1, places=7)
        self.assertTrue(panel.label_end.isna().any())
        self.assertTrue((panel.source_start.dt.year == panel.datetime.dt.year).all())


if __name__ == "__main__":
    unittest.main()
