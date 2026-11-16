"""验证实际观测窗口、跨股市场依赖及未来追加不改历史来源；不读取正式回测。

在quant目录执行quant环境Python scripts/test_weekly_t_provenance.py。
所有输入为合成表，原始f_*和索引保留；测试不修改缓存或模型。
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_provenance import add_feature_provenance


def create_tables(observations: dict[str, list[str]], market: bool = False):
    """以逐股观测日期生成周、盘中和日历表，保留明显未来列用于扰动测试。"""
    daily = pd.DataFrame([{"symbol": symbol, "date": pd.Timestamp(date)}
                          for symbol, dates in observations.items() for date in dates])
    weekly = daily.copy()
    weekly["feature_week"] = weekly.date - pd.to_timedelta(weekly.date.dt.dayofweek, unit="D")
    weekly = weekly[["symbol", "feature_week"]].drop_duplicates().reset_index(drop=True)
    weekly["f_return_mean_prior4w"] = np.arange(len(weekly), dtype="float32")
    weekly["f_history_days"] = np.arange(1, len(weekly) + 1, dtype="float32")
    weekly["target_return"] = 999.0
    intraday = daily.copy()
    intraday["datetime"] = intraday.date + pd.Timedelta(hours=10)
    intraday["f_previous_week_return"] = np.arange(len(intraday), dtype="float32")
    intraday["f_open_return"] = np.float32(.01)
    intraday["f_market_return"] = np.float32(.01)
    intraday["f_cs_return_rank"] = np.float32(.5)
    intraday["f_history_days"] = np.arange(len(intraday), dtype="float32")
    intraday["recovery_price"] = 999.0
    intraday["target_return"] = 777.0
    if market:
        intraday["f_day_return"] = np.where(intraday.groupby("symbol").cumcount() > 0, .02, np.nan)
        intraday["f_market_day_return"] = intraday.groupby("datetime").f_day_return.transform("mean")
        intraday["f_relative_day_return"] = intraday.f_day_return - intraday.f_market_day_return
    return weekly, intraday, daily


class WeeklyTProvenanceTests(unittest.TestCase):
    """按真实日期手算依赖起点，防止固定天数隔离漏掉跨年历史价格。"""

    def test_four_observed_weeks_cross_year_and_short_history(self):
        tables = create_tables({"AAA": ["2023-12-04", "2023-12-11", "2023-12-18", "2023-12-25",
                                         "2024-01-02", "2024-01-08", "2024-01-15"]})
        weekly, _, meta = add_feature_provenance(*tables)
        expected = pd.to_datetime(["2023-12-04"] * 5 + ["2023-12-11", "2023-12-18"])
        pd.testing.assert_index_equal(pd.DatetimeIndex(weekly.feature_source_start), expected, check_names=False)
        self.assertEqual(weekly.history_metadata_source_start.iloc[-1], pd.Timestamp("2023-12-04"))
        self.assertEqual(meta["required_model_feature_exclusions"], ["f_history_days"])

    def test_long_suspension_uses_observed_weeks_not_twenty_eight_days(self):
        tables = create_tables({"AAA": ["2023-09-04", "2023-10-02", "2023-11-06", "2023-12-04", "2024-03-04"]})
        weekly, intraday, _ = add_feature_provenance(*tables)
        self.assertEqual(weekly.feature_source_start.iloc[-1], pd.Timestamp("2023-09-04"))
        self.assertEqual(intraday.feature_source_start.iloc[-1], pd.Timestamp("2023-12-04"))
        self.assertLess(weekly.feature_source_start.iloc[-1], weekly.feature_week.iloc[-1] - pd.Timedelta(days=28))

    def test_previous_day_and_previous_week_are_both_included(self):
        tables = create_tables({"AAA": ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]})
        _, intraday, _ = add_feature_provenance(*tables)
        expected = pd.to_datetime(["2024-01-04", "2024-01-04", "2024-01-01", "2024-01-01"])
        pd.testing.assert_index_equal(pd.DatetimeIndex(intraday.feature_source_start), expected, check_names=False)

    def test_market_day_return_propagates_only_contributor_previous_close(self):
        tables = create_tables({"AAA": ["2024-01-08", "2024-01-15"],
                                "BBB": ["2023-11-30", "2024-01-15"]}, market=True)
        _, intraday, meta = add_feature_provenance(*tables)
        at_signal = intraday.datetime == pd.Timestamp("2024-01-15 10:00")
        self.assertTrue(intraday.loc[at_signal, "market_previous_close_source_start"].eq(pd.Timestamp("2023-11-30")).all())
        a = intraday.loc[at_signal & intraday.symbol.eq("AAA")].iloc[0]
        b = intraday.loc[at_signal & intraday.symbol.eq("BBB")].iloc[0]
        self.assertEqual(a.feature_source_start, pd.Timestamp("2023-11-30"))
        self.assertEqual(b.feature_source_start, pd.Timestamp("2023-11-27"))
        self.assertTrue(meta["market_previous_close_dependency_enabled"])

    def test_missing_market_contribution_and_old_open_return_do_not_spread_history(self):
        weekly, intraday, daily = create_tables({"AAA": ["2024-01-08", "2024-01-15"],
                                               "BBB": ["2023-11-30", "2024-01-15"]}, market=True)
        intraday.loc[intraday.symbol.eq("BBB"), "f_day_return"] = np.nan
        _, missing, _ = add_feature_provenance(weekly, intraday, daily)
        a = missing.loc[missing.symbol.eq("AAA")].iloc[-1]
        self.assertEqual(a.feature_source_start, pd.Timestamp("2024-01-08"))
        original = intraday.drop(columns=["f_market_day_return", "f_relative_day_return"])
        _, old, meta = add_feature_provenance(weekly, original, daily)
        a = old.loc[old.symbol.eq("AAA")].iloc[-1]
        self.assertEqual(a.feature_source_start, pd.Timestamp("2024-01-08"))
        self.assertTrue(old.market_previous_close_source_start.isna().all())
        self.assertFalse(meta["market_previous_close_dependency_enabled"])

    def test_unknown_finite_market_source_fails_closed_for_entire_same_clock(self):
        weekly, intraday, daily = create_tables({"AAA": ["2024-01-08", "2024-01-15"],
                                               "BBB": ["2024-01-15"]}, market=True)
        intraday.loc[intraday.symbol.eq("BBB"), "f_day_return"] = .03
        _, result, meta = add_feature_provenance(weekly, intraday, daily)
        same_clock = result.datetime == pd.Timestamp("2024-01-15 10:00")
        self.assertTrue(result.loc[same_clock, "feature_source_start"].isna().all())
        self.assertEqual(meta["unknown_market_source_rows"], 2)

    def test_planned_session_features_propagate_before_market_columns_exist(self):
        weekly, intraday, daily = create_tables({"AAA": ["2024-01-08", "2024-01-15"],
                                               "BBB": ["2023-11-30", "2024-01-15"]})
        intraday["signal_price"] = 10.2
        intraday["previous_close"] = np.where(intraday.groupby("symbol").cumcount() > 0, 10.0, np.nan)
        _, result, meta = add_feature_provenance(weekly, intraday, daily, {"add_session_context": True})
        a = result.loc[result.symbol.eq("AAA")].iloc[-1]
        self.assertEqual(a.feature_source_start, pd.Timestamp("2023-11-30"))
        self.assertTrue(meta["market_previous_close_dependency_enabled"])
        for column in ("signal_price", "previous_close"):
            for value in (0., -1., np.inf, np.nan):
                changed = intraday.copy()
                changed.loc[changed.symbol.eq("BBB"), column] = value
                _, filtered, _ = add_feature_provenance(weekly, changed, daily, {"add_session_context": True})
                with self.subTest(column=column, value=value):
                    self.assertEqual(filtered.loc[filtered.symbol.eq("AAA")].iloc[-1].feature_source_start,
                                     pd.Timestamp("2024-01-08"))

    def test_future_append_and_future_column_changes_leave_past_provenance_unchanged(self):
        observations = {"AAA": ["2023-12-04", "2023-12-11", "2024-01-08"],
                        "BBB": ["2023-12-18", "2024-01-08"]}
        tables = create_tables(observations, market=True)
        original = add_feature_provenance(*tables)
        observations = {symbol: dates + ["2024-03-04", "2024-03-11"] for symbol, dates in observations.items()}
        expanded = create_tables(observations, market=True)
        expanded[0]["target_return"] = -99999.
        expanded[1][["recovery_price", "target_return"]] = -99999.
        actual = add_feature_provenance(*expanded)
        for old, new, key in zip(original[:2], actual[:2], ["feature_week", "datetime"]):
            columns = [column for column in old if column.endswith("source_start")]
            selected = new.loc[new[key] <= pd.Timestamp("2024-01-08 23:59")]
            expected = old.set_index(["symbol", key])[columns].sort_index()
            observed = selected.set_index(["symbol", key])[columns].sort_index()
            pd.testing.assert_frame_equal(expected, observed)

    def test_full_daily_history_used_even_if_weekly_rows_are_preselected(self):
        weekly, intraday, daily = create_tables({"AAA": ["2023-11-06", "2023-11-13", "2023-11-20", "2023-12-04", "2024-01-08"]})
        result, _, _ = add_feature_provenance(weekly.iloc[[-1]], intraday.iloc[[-1]], daily)
        self.assertEqual(result.feature_source_start.iloc[0], pd.Timestamp("2023-11-06"))

    def test_input_values_dtypes_and_indices_preserved_without_future_columns(self):
        tables = list(create_tables({"AAA": ["2023-12-04", "2024-01-08"]}))
        tables[0] = tables[0].drop(columns="target_return")
        tables[1] = tables[1].drop(columns=["target_return", "recovery_price"])
        tables[0].index = [9, 2]
        tables[1].index = [6, 1]
        original = [frame.copy(deep=True) for frame in tables]
        result = add_feature_provenance(*tables)
        for old, current in zip(original, tables):
            pd.testing.assert_frame_equal(old, current)
        for old, new in zip(original[:2], result[:2]):
            pd.testing.assert_frame_equal(old, new[old.columns])
            self.assertFalse(any(column.startswith("f_") for column in set(new).difference(old)))

    def test_missing_or_duplicate_calendar_keys_and_missing_market_input_rejected(self):
        weekly, intraday, daily = create_tables({"AAA": ["2024-01-08", "2024-01-15"]}, market=True)
        cases = [(weekly, intraday, pd.concat([daily, daily.iloc[[0]]])),
                 (weekly, intraday, daily.iloc[:1]),
                 (pd.concat([weekly, weekly.iloc[[0]]]), intraday, daily),
                 (weekly, pd.concat([intraday, intraday.iloc[[0]]]), daily),
                 (weekly, intraday.drop(columns="f_day_return"), daily)]
        for case in cases:
            with self.subTest(shapes=[frame.shape for frame in case]), self.assertRaises(ValueError):
                add_feature_provenance(*case)


if __name__ == "__main__":
    unittest.main(verbosity=2)
