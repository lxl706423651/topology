"""验证周/T数据的完整分钟聚合、交易周边界与因果特征；合成缓存只写临时目录。

从quant项目根目录执行quant环境Python scripts/test_weekly_t_data.py；不训练模型，
不下载行情，不改原始分钟CSV。另由实际4股烟测核对真实文件格式与资源消耗。
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_data import (
    DEFAULT_SIGNAL_TIMES, attach_weekly_calendar_labels,
    build_symbol_weekly_features, load_weekly_t_data,
)


def create_minute_day(day: str, price: float = 10.0) -> pd.DataFrame:
    """生成240条合法交易分钟；非信号分钟也有独立成交额及可核算价格变化。"""
    dates = pd.date_range(day + " 09:31", day + " 11:30", freq="min")
    dates = dates.append(pd.date_range(day + " 13:01", day + " 15:00", freq="min"))
    sequence = np.arange(len(dates))
    close = price + sequence * 0.001
    volume = 1000.0 + sequence
    return pd.DataFrame({"datetime": dates, "open": close, "high": close + 0.02,
                         "low": close - 0.02, "close": close, "volume": volume,
                         "amount": close * volume * 100})


class WeeklyTDataTests(unittest.TestCase):
    """检查未来价格不进入特征、节假日周映射正确且缺成交不伪造标签。"""

    def setUp(self):
        """两周各两交易日用于模拟合法的短交易周，不按五天判断完整性。"""
        self.raw = pd.concat([create_minute_day(day, price) for day, price in (
            ("2026-04-20", 10.0), ("2026-04-24", 10.5),
            ("2026-04-27", 11.0), ("2026-04-30", 12.0))], ignore_index=True)
        self.symbol = "SH.600000"

    def build_tables(self, raw=None, complete=True):
        """构造单股三表再接全市场日历，返回完整接口结果用于独立断言。"""
        weekly, intraday, daily = build_symbol_weekly_features(
            self.raw if raw is None else raw, self.symbol, DEFAULT_SIGNAL_TIMES)
        return attach_weekly_calendar_labels(weekly, intraday, daily,
                                              {"known_complete_end_date": complete})

    def test_all_minutes_contribute_to_week_amount_volatility_and_concentration(self):
        weekly, _, daily, _ = self.build_tables()
        first_week = self.raw.loc[self.raw.datetime < pd.Timestamp("2026-04-27")]
        expected_amount = first_week.amount.sum()
        expected_hhi = first_week.amount.pow(2).sum() / expected_amount ** 2
        self.assertAlmostEqual(float(weekly.f_week_amount.iloc[0]) / expected_amount, 1.0, places=6)
        self.assertAlmostEqual(float(weekly.f_amount_concentration.iloc[0]), expected_hhi, places=9)
        self.assertEqual(int(weekly.f_week_bar_count.iloc[0]), 480)
        self.assertEqual(int(daily.bar_count.iloc[0]), 240)
        changed = self.raw.copy()
        unsampled = changed.datetime == pd.Timestamp("2026-04-20 10:17")
        changed.loc[unsampled, "amount"] *= 100
        adjusted, _, _, _ = self.build_tables(changed)
        self.assertGreater(adjusted.f_week_amount.iloc[0], weekly.f_week_amount.iloc[0])
        self.assertGreater(adjusted.f_amount_concentration.iloc[0], weekly.f_amount_concentration.iloc[0])

    def test_weekly_labels_use_next_week_first_open_last_close_and_mean_daily_amplitude(self):
        weekly, _, daily, _ = self.build_tables()
        row = weekly.iloc[0]
        self.assertEqual(row.trade_week, pd.Timestamp("2026-04-27"))
        self.assertEqual(row.entry_date, pd.Timestamp("2026-04-27"))
        self.assertEqual(row.exit_date, pd.Timestamp("2026-04-30"))
        self.assertEqual(row.label_start, pd.Timestamp("2026-04-27 09:31"))
        self.assertEqual(row.label_end, pd.Timestamp("2026-04-30 15:00"))
        self.assertAlmostEqual(row.target_return, 12.239 / 11.0 - 1)
        next_days = daily.loc[daily.trade_week == pd.Timestamp("2026-04-27")]
        expected = ((next_days.high - next_days.low) / next_days.open).mean()
        self.assertAlmostEqual(row.target_amplitude, expected)
        self.assertGreaterEqual(row.target_amplitude, 0)
        self.assertTrue(weekly.iloc[-1].target_return != weekly.iloc[-1].target_return)
        self.assertFalse(weekly.iloc[-1].future_week_observed)

    def test_confirmed_holiday_thursday_is_complete_but_unknown_final_week_is_not(self):
        known, _, known_daily, known_meta = self.build_tables(complete=True)
        unknown, unknown_intraday, _, unknown_meta = self.build_tables(complete=False)
        self.assertTrue(known.iloc[0].trade_week_complete)
        self.assertTrue(known_daily.trade_week_complete.all())
        self.assertEqual(known.iloc[-1].f_week_days, 2)
        self.assertTrue(known_meta["last_week_complete"])
        self.assertFalse(unknown_meta["last_week_complete"])
        self.assertTrue(np.isnan(unknown.iloc[0].target_return))
        last_week = unknown_intraday.trade_week == pd.Timestamp("2026-04-27")
        self.assertTrue(unknown_intraday.loc[last_week, "target_return"].isna().all())

    def test_empty_holiday_week_is_skipped_in_next_trade_week_mapping(self):
        raw = pd.concat([create_minute_day("2026-02-13"), create_minute_day("2026-02-23", 11)], ignore_index=True)
        weekly, _, _, _ = self.build_tables(raw)
        self.assertEqual(weekly.iloc[0].feature_week, pd.Timestamp("2026-02-09"))
        self.assertEqual(weekly.iloc[0].trade_week, pd.Timestamp("2026-02-23"))
        self.assertEqual(weekly.iloc[0].entry_date, pd.Timestamp("2026-02-23"))

    def test_daily_open_close_are_exact_and_never_forward_filled(self):
        missing = [pd.Timestamp("2026-04-27 09:31"), pd.Timestamp("2026-04-30 15:00")]
        raw = self.raw.loc[~self.raw.datetime.isin(missing)].copy()
        weekly, _, daily, _ = self.build_tables(raw)
        self.assertTrue(np.isnan(daily.loc[daily.date == pd.Timestamp("2026-04-27"), "open"].iloc[0]))
        self.assertTrue(np.isnan(daily.loc[daily.date == pd.Timestamp("2026-04-30"), "close"].iloc[0]))
        self.assertTrue(np.isnan(weekly.iloc[0].target_return))
        self.assertTrue(np.isnan(weekly.iloc[0].target_amplitude))
        self.assertEqual(len(weekly), 2)

    def test_missing_future_execution_or_recovery_preserves_current_inference(self):
        missing = [pd.Timestamp("2026-04-20 09:36"), pd.Timestamp("2026-04-24 14:55")]
        raw = self.raw.loc[~self.raw.datetime.isin(missing)].copy()
        _, intraday, _, _ = self.build_tables(raw)
        first = intraday.loc[intraday.datetime == pd.Timestamp("2026-04-20 09:35")].iloc[0]
        self.assertTrue(np.isnan(first.execution_price))
        self.assertTrue(np.isnan(first.target_return))
        self.assertTrue(np.isfinite(first.f_open_return))
        next_day = intraday.loc[intraday.date == pd.Timestamp("2026-04-24")]
        self.assertEqual(len(next_day), 8)
        self.assertTrue(next_day.recovery_price.isna().all())
        self.assertTrue(next_day.target_return.isna().all())

    def test_intraday_targets_use_next_minute_and_1455_with_previous_close_known(self):
        _, intraday, daily, _ = self.build_tables()
        row = intraday.loc[intraday.datetime == pd.Timestamp("2026-04-24 10:00")].iloc[0]
        indexed = self.raw.set_index("datetime")
        self.assertEqual(row.execution_time, pd.Timestamp("2026-04-24 10:01"))
        self.assertEqual(row.recovery_time, pd.Timestamp("2026-04-24 14:55"))
        self.assertEqual(row.label_start, row.execution_time)
        self.assertEqual(row.label_end, row.recovery_time)
        self.assertAlmostEqual(row.execution_price, indexed.loc[row.execution_time, "close"])
        self.assertAlmostEqual(row.target_return, indexed.loc[row.recovery_time, "close"] / indexed.loc[row.execution_time, "close"] - 1)
        self.assertAlmostEqual(row.previous_close, daily.close.iloc[0])

    def test_future_bar_perturbation_does_not_change_earlier_weekly_or_intraday_features(self):
        weekly, intraday, _, _ = self.build_tables()
        modified = self.raw.copy()
        cutoff = pd.Timestamp("2026-04-27 10:00")
        future = modified.datetime > cutoff
        modified.loc[future, ["open", "high", "low", "close"]] *= 1.5
        modified.loc[future, ["volume", "amount"]] *= 2.0
        changed_weekly, changed_intraday, _, _ = self.build_tables(modified)
        weekly_features = [column for column in weekly if column.startswith("f_")]
        pd.testing.assert_frame_equal(weekly.loc[weekly.feature_week < cutoff.normalize(), weekly_features],
                                      changed_weekly.loc[changed_weekly.feature_week < cutoff.normalize(), weekly_features])
        intraday_features = [column for column in intraday if column.startswith("f_")]
        pd.testing.assert_frame_equal(intraday.loc[intraday.datetime <= cutoff, intraday_features],
                                      changed_intraday.loc[changed_intraday.datetime <= cutoff, intraday_features])

    def test_cross_section_states_use_only_same_completed_signal(self):
        first = build_symbol_weekly_features(self.raw, "SH.600000", DEFAULT_SIGNAL_TIMES)
        second_raw = self.raw.copy()
        sequence = second_raw.groupby(second_raw.datetime.dt.normalize()).cumcount().to_numpy()
        second_raw[["open", "high", "low", "close"]] += sequence[:, None] * 0.005
        second = build_symbol_weekly_features(second_raw, "SZ.000001", DEFAULT_SIGNAL_TIMES)
        tables = [pd.concat([first[index], second[index]], ignore_index=True) for index in range(3)]
        _, intraday, _, _ = attach_weekly_calendar_labels(*tables, {"known_complete_end_date": True})
        group = intraday.loc[intraday.datetime == pd.Timestamp("2026-04-20 10:00")]
        self.assertTrue(np.allclose(group.f_market_return, group.f_open_return.mean()))
        ranks = group.set_index("symbol").f_cs_return_rank
        self.assertEqual(ranks["SH.600000"], 0.5)
        self.assertEqual(ranks["SZ.000001"], 1.0)

    def test_cache_build_load_union_universe_and_source_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix="weekly_t_data_tests_") as location:
            directory = Path(location)
            source = directory / "raw/2026"
            source.mkdir(parents=True)
            for symbol in ("SH.600000", "SZ.000001", "SH.688001", "SZ.300001"):
                self.raw.to_csv(source / f"{symbol}.csv", index=False)
            cfg = {"data_root": str(source.parent), "start_date": "2026-04-01", "end_date": "2026-04-30",
                   "cache_dir": str(directory / "cache"), "workers": 1,
                   "known_complete_end_date": True, "name_snapshot": "/this/path/must/not/be/read.csv"}
            first = load_weekly_t_data(cfg)
            second = load_weekly_t_data(cfg)
            self.assertEqual(first[3]["symbols"], ["SH.600000", "SZ.000001"])
            self.assertFalse(first[3]["cache_hit"])
            self.assertTrue(second[3]["cache_hit"])
            for expected, actual in zip(first[:3], second[:3]):
                pd.testing.assert_frame_equal(expected, actual)
            path = source / "SH.600000.csv"
            changed = self.raw.copy()
            changed.loc[0, "amount"] *= 2
            changed.to_csv(path, index=False)
            third = load_weekly_t_data(cfg)
            self.assertNotEqual(first[3]["fingerprint"], third[3]["fingerprint"])
            self.assertEqual(list((directory / "cache").glob("**/*.tmp")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
