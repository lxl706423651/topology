"""路径结果标签的非破坏性测试；以既有逐批引擎为参照，不使用真实测试收益调参。"""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from quant_backtest.annual_panel import create_sampling_clock
from quant_backtest.annual_portfolio import MarketStore
from quant_backtest.pathwise_labels import (create_path_arrays, validate_path_case, calculate_path_targets,
    select_daily_entries, load_path_labels)


def create_path_config() -> dict:
    """构造标签测试配置，保留万八免5、10%总资产、T+1、100股和5%成交容量。"""
    return {"initial_cash": 1000000., "entry_equity_fraction": .1, "commission_rate": .0008,
        "transfer_rate": .00001, "slippage_rate": .0005, "max_bar_amount_fraction": .05,
        "lot_size": 100, "min_commission": 0., "entry_top_n": 5, "min_cross_section": 1,
        "stamp_rate_from_20230828": .0005, "stamp_rate_before_20230828": .001,
        "path_horizons": [1, 3], "path_exit_grace_days": 2, "path_initial_atr": 20.,
        "path_trailing_atr": 40., "workers": 1}


def create_path_market() -> pd.DataFrame:
    """创建跨印花税调整日的12日双股票已完成行情，价格、容量和时钟均确定性。"""
    times = create_sampling_clock(pd.bdate_range("2023-08-21", periods=12), 24)
    rng = np.random.default_rng(23)
    frames = []
    for symbol in ["SH.600000", "SZ.000001"]:
        close = 10 * np.exp(np.cumsum(rng.normal(0, .007, len(times))))
        frames.append(pd.DataFrame({"datetime": times, "symbol": symbol, "close": close,
            "atr": .01, "interval_high": close * 1.004, "volume": 100000.,
            "amount": 10000000., "execution_close": close * 1.0003, "execution_volume": 100000.,
            "execution_amount": 10000000., "previous_close": close}))
    return pd.concat(frames, ignore_index=True)


class PathwiseLabelTests(unittest.TestCase):
    """检查虚拟独立账户、部分卖出、待卖、异常行情及pkl重用与原引擎一致。"""

    def test_random_paths_match_engine(self):
        """多个股票/钟点/预测期限与原始逐批引擎逐例比收益、买入及剩余股数和卖出事件。"""
        market, cfg = MarketStore(create_path_market()), create_path_config()
        arrays = create_path_arrays(market, cfg)
        for asset in [0, 1]:
            for start in [0, 3, 11, 18, 22]:
                for horizon in [1, 3]:
                    validate_path_case(market, cfg, arrays, asset, start, start + 55, horizon)

    def test_t1_partial_and_terminal_marks(self):
        """同日跌破止损不能卖，后续小容量部分退出、缺行情和期末余仓估值保持相同结果。"""
        raw, cfg = create_path_market(), create_path_config()
        symbol_rows = raw.symbol.eq("SH.600000")
        indices = raw.index[symbol_rows]
        raw.loc[indices[1:10], "close"] = 9.
        raw.loc[indices[1:10], "interval_high"] = 9.
        raw.loc[indices[11:], "execution_amount"] = 25000.
        raw.loc[indices[12], "execution_volume"] = np.nan
        raw.loc[indices[13], "execution_close"] = np.nan
        raw.loc[indices[14], "previous_close"] = 100.
        market = MarketStore(raw)
        arrays = create_path_arrays(market, cfg)
        result = validate_path_case(market, cfg, arrays, 0, 0, 55, 3)
        self.assertGreater(result["remaining_shares"], 0)
        self.assertGreater(result["sell_events"], 1)

    def test_bad_entry_and_limit_block(self):
        """缺失/零量、不可判断涨跌停、涨停和不足整手容量不能虚构买入；失败意图收益0。"""
        cfg = create_path_config()
        for field, value in [("execution_volume", np.nan), ("execution_close", 20.),
                             ("execution_amount", 10.), ("previous_close", np.nan)]:
            raw = create_path_market()
            raw.loc[0, field] = value
            market = MarketStore(raw)
            result = validate_path_case(market, cfg, create_path_arrays(market, cfg), 0, 0, 55, 3)
            self.assertEqual(result["entered_shares"], 0)
            self.assertEqual(result["net_return"], 0)

    def test_batch_matches_single_and_ignores_after_window(self):
        """批量加速与逐例结果一致，窗口后的未来行情突变不得改变已完成标签。"""
        market, cfg = MarketStore(create_path_market()), create_path_config()
        q, offsets, executable, stamp, parameters = create_path_arrays(market, cfg)
        args = (np.array([0, 11]), np.array([0, 1]), np.array([55, 66]), market.day_index,
                offsets, executable, stamp, 3, 20., 40., parameters)
        original = calculate_path_targets(q, *args)
        q[:, 67:] = np.nan
        changed = calculate_path_targets(q, *args)
        np.testing.assert_allclose(original, changed)
        for n, (asset, start, end) in enumerate([(0, 0, 55), (1, 11, 66)]):
            one = validate_path_case(market, cfg, create_path_arrays(market, cfg), asset, start, end, 3)
            self.assertAlmostEqual(original[n, 0], one["net_return"])

    def test_stable_clock_and_label_cache(self):
        """入口不依标签/行序；缓存再次读取逐值一致，潜在窗口终点不早于实际读取终点。"""
        raw, cfg = create_path_market(), create_path_config()
        market = MarketStore(raw)
        panel = raw[["datetime", "symbol"]].assign(cross_section_count=2)
        one = select_daily_entries(panel, market.times)
        two = select_daily_entries(panel.sample(frac=1, random_state=8).assign(forward_return=999), market.times)
        pd.testing.assert_frame_equal(one, two.drop(columns="forward_return"))
        self.assertFalse(one.assign(day=one.datetime.dt.normalize()).duplicated(["symbol", "day"]).any())
        with tempfile.TemporaryDirectory() as tmp:
            cfg["path_cache_dir"] = tmp
            first, meta = load_path_labels(panel, market, {"key": "synthetic"}, cfg)
            second, repeat = load_path_labels(panel, market, {"key": "synthetic"}, cfg)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(meta, repeat)
            self.assertTrue(Path(meta["path"]).exists())
            for h in cfg["path_horizons"]:
                complete = first.dropna(subset=[f"path_h{h}_window_end"])
                self.assertTrue((complete[f"path_h{h}_observed_end"] <= complete[f"path_h{h}_window_end"]).all())


if __name__ == "__main__":
    unittest.main()

