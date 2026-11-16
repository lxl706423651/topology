"""有限来源和ATR修正的合成回归测试，不读取任何真实测试预测、收益或交易。"""

import unittest

import numpy as np
import pandas as pd

from quant_backtest.annual_panel import create_sampling_clock
from quant_backtest.bounded_training import (
    build_symbol_support, create_finite_factor_names, propagate_feature_sources, select_bounded_rows, select_virtual_entries,
)
from quant_backtest.factor_library import build_factor_library, calculate_atr
from scripts.run_multiscale_year_validation import run_training
from scripts.audit_bounded_training import validate_core_sources


def create_raw(times: pd.DatetimeIndex) -> pd.DataFrame:
    """在times上构造非平价合法单股OHLCV，固定随机种子，足够测试递推与有限窗口差别。"""
    rng = np.random.default_rng(17)
    close = 20 * np.exp(np.cumsum(rng.normal(0, .002, len(times))))
    return pd.DataFrame({"datetime": times, "symbol": "SH.600000", "open": close * .999,
                         "high": close * 1.002, "low": close * .998, "close": close,
                         "volume": rng.uniform(1000, 5000, len(times)), "amount": close * 200000})


class BoundedTrainingTests(unittest.TestCase):
    """不能把固定交易日隔离当成稀疏股票的实际观测窗口，也不能只隔离模型而遗漏ATR。"""

    def test_finite_values_ignore_prefix_price_mutation(self):
        """保留尾部241条原始观测后，白名单数学窗口不再需要更早价格；检查数值容差。"""
        raw = create_raw(pd.date_range("2025-01-02 09:31", periods=850, freq="min"))
        changed = raw.copy()
        changed.loc[:399, ["open", "high", "low", "close", "amount"]] *= 1.15
        a, b = build_factor_library(raw), build_factor_library(changed)
        names = sorted(create_finite_factor_names())
        self.assertTrue(set(names).issubset(a.columns))
        np.testing.assert_allclose(a.loc[641:, names], b.loc[641:, names], rtol=1e-5, atol=1e-6, equal_nan=True)
        self.assertTrue(any((a.ema_gap_120.iloc[641:700] - b.ema_gap_120.iloc[641:700]).abs() > 1e-5))

    def test_finite_atr_and_source_ignore_old_prefix(self):
        """有限ATR只需241条观测，修改更早前缀不改变；旧Wilder ATR仍可变化。"""
        times = pd.date_range("2025-01-02 09:31", periods=850, freq="min")
        raw = create_raw(times)
        changed = raw.copy()
        changed.loc[:399, ["open", "high", "low", "close"]] *= 1.15
        a, b = build_symbol_support(raw, times), build_symbol_support(changed, times)
        np.testing.assert_allclose(a.finite_atr.iloc[641:], b.finite_atr.iloc[641:], rtol=1e-12)
        self.assertEqual(a.raw_source_start.iloc[641], times[401])
        old_a = calculate_atr(raw.high, raw.low, raw.close, 240)
        old_b = calculate_atr(changed.high, changed.low, changed.close, 240)
        self.assertGreater(abs(old_a.iloc[641] - old_b.iloc[641]), 1e-4)

    def test_missing_clock_never_fills_atr(self):
        """无当前报价不得把过去ATR当本时刻有效新计算；有限起点也应未知。"""
        times = pd.date_range("2025-01-02 09:31", periods=300, freq="min")
        out = build_symbol_support(create_raw(times.delete(270)), times)
        self.assertTrue(pd.isna(out.finite_atr.iloc[270]))
        self.assertTrue(pd.isna(out.raw_source_start.iloc[270]))

    def test_sparse_stock_source_survives_sixty_day_gap(self):
        """长停牌后仅少数新观测，即便过60共同交易日，前收和有限窗口仍可来自留出年。"""
        old = pd.date_range("2024-12-31 13:01", periods=120, freq="min")
        resumed = pd.DatetimeIndex(["2025-04-15 09:31"])
        raw = create_raw(old.append(resumed))
        clock = create_sampling_clock(pd.bdate_range("2024-12-31", "2025-04-15"), 24)
        support = build_symbol_support(raw, clock).assign(symbol="SH.600000")
        panel = pd.DataFrame({"datetime": resumed, "symbol": "SH.600000", "cs_rank_momentum_5": .5})
        enriched = propagate_feature_sources(panel, support, clock)
        self.assertLess(enriched.feature_source_start.iloc[0], pd.Timestamp("2025-01-01"))
        self.assertTrue(select_bounded_rows(enriched, clock, 2024, 60).empty)

    def test_cross_section_propagates_all_contributors(self):
        """即使本股自身来源干净，原截面排名的另一贡献者来源不干净，也应拒绝该时点。"""
        time = pd.Timestamp("2025-04-15 09:31")
        clock = create_sampling_clock(pd.bdate_range("2024-12-02", "2025-04-15"), 24)
        panel = pd.DataFrame({"datetime": [time, time], "symbol": ["A", "B"], "cs_rank_momentum_5": [.5, 1.]})
        source = panel[["datetime", "symbol"]].assign(raw_source_start=pd.to_datetime(["2025-04-14", "2024-12-31"]),
                                                     previous_close_source=pd.Timestamp("2025-04-14"))
        enriched = propagate_feature_sources(panel, source, clock)
        self.assertEqual(enriched.feature_source_start.nunique(), 1)
        self.assertTrue(select_bounded_rows(enriched, clock, 2024, 60).empty)
        source.loc[1, "raw_source_start"] = pd.NaT
        unknown = propagate_feature_sources(panel, source, clock)
        self.assertTrue(unknown.feature_source_start.isna().all())

    def test_virtual_sampling_does_not_shift_when_other_year_removed(self):
        """删除整个无关年份，不得通过全局iloc步长改变其他年份虚拟入口集合。"""
        frame = pd.DataFrame({"datetime": pd.date_range("2024-11-01", "2025-02-28"), "symbol": "A"})
        all_selected = select_virtual_entries(frame, 8)
        remaining = frame.loc[frame.datetime.dt.year == 2025]
        a = select_virtual_entries(remaining, 8)
        b = all_selected.loc[all_selected.datetime.dt.year == 2025]
        pd.testing.assert_frame_equal(a, b)

    def test_core_fit_cannot_read_prior_year_validation(self):
        """非外层留出年的前一年尾部仍可能是内部验证，核心拟合应额外拒绝该来源。"""
        clock = create_sampling_clock(pd.bdate_range("2023-12-01", "2025-04-15"), 24)
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-15 09:31", "2025-02-15 09:31"]),
                              "feature_source_start": pd.to_datetime(["2024-12-20", "2025-01-10"])})
        out = select_bounded_rows(frame, clock, 2023, 60)
        self.assertEqual(out.fit_source_allowed.tolist(), [False, True])

    def test_disallowed_features_and_wrong_runner_rejected(self):
        """递推或累计列不能进入白名单，有界配置不能被默认旧数据路径消费。"""
        names = create_finite_factor_names()
        for column in ["ema_gap_120", "rsi_14", "atr_ratio_14", "adx_14", "macd_dif_12_26_9", "obv_z_60", "trend_r2_120"]:
            self.assertNotIn(column, names)
        with self.assertRaisesRegex(ValueError, "对应数据"):
            run_training({"data_contract": "bounded_observation_v1"})

    def test_core_audit_checks_sources_and_label_year(self):
        """实际拟合来源审计同时阻止上年价格、未知来源、跨年标签及留出年直接拟合。"""
        valid = pd.DataFrame({"datetime": pd.to_datetime(["2025-02-01 09:31"]),
                              "feature_source_start": pd.to_datetime(["2025-01-03"]),
                              "label_end": pd.to_datetime(["2025-02-05 09:32"])})
        validate_core_sources(valid, 2024)
        for field, value in [("feature_source_start", pd.Timestamp("2024-12-31")),
                             ("feature_source_start", pd.NaT),
                             ("feature_source_start", pd.Timestamp("2025-02-02")),
                             ("label_end", pd.Timestamp("2026-01-02")),
                             ("label_end", pd.Timestamp("2025-02-01 09:30"))]:
            changed = valid.copy()
            changed.loc[0, field] = value
            with self.assertRaises(ValueError):
                validate_core_sources(changed, 2024)
        with self.assertRaises(ValueError):
            validate_core_sources(valid, 2025)


if __name__ == "__main__":
    unittest.main()
