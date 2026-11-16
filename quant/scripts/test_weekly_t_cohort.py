"""验证核心年度互为OOF的周选股队列、上下文来源及未来标签独立性。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_cohort import CONTEXT_FEATURES, attach_weekly_context, build_core_t_cohort
from quant_backtest.weekly_t_models import select_training_rows


def cohort_config() -> dict:
    """提供小样本真实隔离配置；选股固定一只，不降低任何年度隔离要求。"""
    return {"strict_feature_provenance": True, "excluded_model_features": ["f_history_days"],
            "top_n": 1, "weekly_top_n": 1, "weekly_return_weight": .7,
            "weekly_amplitude_weight": .3, "embargo_days": 7, "min_training_rows": 2,
            "exclude_flat_signal_bars": False, "random_state": 42, "smoke_weeks": 0}


def cohort_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造四年各两周两只股票及全周分钟；周量价来源比旧分钟来源早四周。"""
    weekly, intraday = [], []
    for year, first in ((2023, "2023-03-06"), (2024, "2024-03-04"),
                        (2025, "2025-03-03"), (2026, "2026-03-02")):
        for offset in (0, 7):
            week = pd.Timestamp(first) + pd.Timedelta(days=offset)
            for rank, symbol in ((1, "SH.600000"), (2, "SH.600004")):
                weekly.append({"symbol": symbol, "trade_week": week, "feature_week": week - pd.Timedelta(days=7),
                    "signal_datetime": week - pd.Timedelta(days=3) + pd.Timedelta(hours=15),
                    "feature_source_start": week - pd.Timedelta(days=35), "entry_date": week,
                    "exit_date": week + pd.Timedelta(days=4), "feature_week_complete": True,
                    "trade_week_complete": True, "label_available": True,
                    "label_start": week + pd.Timedelta(hours=9, minutes=30),
                    "label_end": week + pd.Timedelta(days=4, hours=15),
                    "target_return": .02 / rank, "target_amplitude": .03,
                    "f_quality": 3 - rank, "f_history_days": 100.})
                for day in range(5):
                    timestamp = week + pd.Timedelta(days=day, hours=9, minutes=35)
                    intraday.append({"datetime": timestamp, "date": timestamp.normalize(), "symbol": symbol,
                        "trade_week": week, "feature_source_start": week - pd.Timedelta(days=7),
                        "label_start": timestamp + pd.Timedelta(minutes=1),
                        "label_end": timestamp + pd.Timedelta(minutes=21), "target_return": .001,
                        "label_available": True, "recovery_price": 10.01, "f_input": float(day),
                        "signal_bar_tradable": True})
    return pd.DataFrame(weekly), pd.DataFrame(intraday)


def fake_weekly_predictions(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """只凭当前f_quality排序，返回满足70/30恒等式的固定预测。"""
    result = frame[["symbol", "feature_week", "signal_datetime", "trade_week", "f_quality"]].copy()
    result = result.sort_values(["trade_week", "f_quality"], ascending=[True, False], kind="stable")
    result["rank"] = result.groupby("trade_week").cumcount() + 1
    result["pred_return_rank"] = np.where(result.f_quality.eq(2), .9, .5)
    result["pred_amplitude_rank"] = np.where(result.f_quality.eq(2), .8, .6)
    result["score"] = .7 * result.pred_return_rank + .3 * result.pred_amplitude_rank
    result["selected"] = result["rank"] <= cfg["top_n"]
    return result.drop(columns="f_quality").reset_index(drop=True)


class WeeklyTCohortTests(unittest.TestCase):
    """使用实际年度purge和mock模型隔离队列行为，不训练额外模型。"""

    def setUp(self):
        """每例独立数据、配置及输出目录，fake fit保留真实训练筛选审计。"""
        self.weekly, self.intraday = cohort_frames()
        self.cfg = cohort_config()
        self.temporary = tempfile.TemporaryDirectory(prefix="weekly_t_cohort_test_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.calls = []

    def fake_fit(self, frame, train_years, cfg, output_dir, excluded_years=()):
        """执行生产的严格标签purge，仅模型权重写入用固定字节替代训练。"""
        training, metadata, _ = select_training_rows(frame, train_years, cfg, "weekly", excluded_years)
        self.calls.append({"years": list(train_years), "excluded": list(excluded_years),
                           "training": training.copy(deep=True), "directory": str(output_dir)})
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / "model_bundle.joblib").write_bytes(json.dumps({"years": list(train_years), "excluded": list(excluded_years)}).encode())
        (path / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return {"task": "weekly", "metadata": metadata}

    def build(self, directory="cohort", weekly=None, intraday=None):
        """用因果fake预测运行完整队列构造，返回真实接口结果。"""
        with patch("quant_backtest.weekly_t_cohort.fit_weekly_models", side_effect=self.fake_fit), patch(
            "quant_backtest.weekly_t_cohort.predict_weekly", side_effect=lambda bundle, frame, cfg: fake_weekly_predictions(frame, cfg)):
            return build_core_t_cohort(self.weekly if weekly is None else weekly,
                                      self.intraday if intraday is None else intraday,
                                      [2023, 2024], [2025, 2026], self.cfg, self.directory / directory)

    def context_inputs(self):
        """提供2023单股周的两行，非默认索引用于检查返回顺序。"""
        weekly = self.weekly.loc[self.weekly.trade_week.eq(pd.Timestamp("2023-03-06"))].copy()
        rows = self.intraday.loc[self.intraday.trade_week.eq(pd.Timestamp("2023-03-06"))
                                 & self.intraday.symbol.eq("SH.600000")].iloc[[3, 1]].copy()
        rows.index = pd.Index([101, 7], name="original_position")
        predictions = fake_weekly_predictions(weekly, self.cfg)
        cfg = {**self.cfg, "weekly_context_model_training_years": [2024], "weekly_context_model_sha256": "a" * 64}
        return weekly, rows, predictions, cfg

    def test_build_uses_other_core_only_and_excludes_every_other_available_year(self):
        original_weekly, original_intraday, original_cfg = self.weekly.copy(deep=True), self.intraday.copy(deep=True), deepcopy(self.cfg)
        training, metadata = self.build()
        self.assertEqual([(call["years"], call["excluded"]) for call in self.calls],
                         [([2024], [2023, 2025, 2026]), ([2023], [2024, 2025, 2026])])
        self.assertEqual(metadata["core_years"], [2023, 2024])
        self.assertEqual(metadata["selector_count"], 2)
        self.assertEqual(len(training), 12)
        self.assertEqual(set(training.symbol), {"SH.600000"})
        self.assertEqual(set(training.datetime.dt.dayofweek), {1, 2, 3})
        self.assertEqual(set(training.datetime.dt.year), {2023, 2024})
        for selector in metadata["selectors"]:
            target = selector["target_year"]
            self.assertNotIn(target, selector["source_model_training_years"])
            group = training.loc[training.cohort_target_year.eq(target)]
            self.assertEqual(set(group.source_model_training_years.map(json.loads).map(tuple)),
                             {tuple(selector["source_model_training_years"])})
            digest = hashlib.sha256(Path(selector["source_model_path"]).read_bytes()).hexdigest()
            self.assertEqual(set(group.source_model_sha256), {digest})
        self.assertTrue(all(column in training for column in CONTEXT_FEATURES.values()))
        pd.testing.assert_frame_equal(self.weekly, original_weekly)
        pd.testing.assert_frame_equal(self.intraday, original_intraday)
        self.assertEqual(self.cfg, original_cfg)

    def test_unknown_future_labels_do_not_remove_selected_core_inference_rows(self):
        missing = self.weekly.trade_week.eq(pd.Timestamp("2023-03-06")) & self.weekly.symbol.eq("SH.600000")
        altered = self.weekly.copy(deep=True)
        altered.loc[missing, ["target_return", "target_amplitude"]] = np.nan
        altered.loc[missing, ["label_start", "label_end"]] = pd.NaT
        altered.loc[missing, ["label_available", "trade_week_complete"]] = False
        minutes = self.intraday.copy(deep=True)
        minutes.loc[minutes.trade_week.eq(pd.Timestamp("2023-03-06")), "target_return"] = np.nan
        minutes.loc[minutes.trade_week.eq(pd.Timestamp("2023-03-06")), "label_available"] = False
        baseline, _ = self.build("baseline")
        actual, _ = self.build("missing", altered, minutes)
        pd.testing.assert_frame_equal(baseline[["datetime", "symbol", *CONTEXT_FEATURES.values()]], actual[["datetime", "symbol", *CONTEXT_FEATURES.values()]])
        self.assertEqual(len(actual.loc[actual.trade_week.eq(pd.Timestamp("2023-03-06"))]), 3)
        self.assertTrue(actual.loc[actual.trade_week.eq(pd.Timestamp("2023-03-06")), "target_return"].isna().all())

    def test_context_preserves_index_order_future_fields_and_uses_earliest_source(self):
        weekly, rows, predictions, cfg = self.context_inputs()
        before = rows.copy(deep=True)
        rows.loc[7, "feature_source_start"] = pd.Timestamp("2023-01-01")
        result = attach_weekly_context(rows, predictions, weekly, cfg)
        self.assertTrue(result.index.equals(rows.index))
        pd.testing.assert_frame_equal(result[["datetime", "symbol", "target_return", "label_start", "label_end", "recovery_price"]],
                                      rows[["datetime", "symbol", "target_return", "label_start", "label_end", "recovery_price"]])
        self.assertEqual(result.feature_source_start.tolist(), [pd.Timestamp("2023-01-30"), pd.Timestamp("2023-01-01")])
        self.assertEqual(result.f_weekly_selection_rank.tolist(), [1., 1.])
        self.assertEqual(result.source_model_training_years.tolist(), ["[2024]", "[2024]"])
        pd.testing.assert_frame_equal(rows.drop(columns="feature_source_start"), before.drop(columns="feature_source_start"))

    def test_future_weekly_labels_do_not_change_context_and_prediction_trace_can_supply_identity(self):
        weekly, rows, predictions, cfg = self.context_inputs()
        predictions["source_model_training_years"] = "[2024]"
        predictions["source_model_sha256"] = "b" * 64
        no_fallback = {key: value for key, value in cfg.items() if not key.startswith("weekly_context_model_")}
        first = attach_weekly_context(rows, predictions, weekly, no_fallback)
        changed = weekly.copy(deep=True)
        changed[["target_return", "target_amplitude"]] = np.nan
        changed[["label_start", "label_end"]] = pd.NaT
        changed[["label_available", "trade_week_complete"]] = False
        actual = attach_weekly_context(rows, predictions, changed, no_fallback)
        pd.testing.assert_frame_equal(first, actual)
        self.assertEqual(set(actual.source_model_sha256), {"b" * 64})

    def test_four_week_source_propagation_is_used_by_subsequent_t_training_purge(self):
        weekly, intraday = cohort_frames()
        weekly = weekly.loc[weekly.trade_week.eq(pd.Timestamp("2024-03-04"))].copy()
        normal = intraday.loc[intraday.trade_week.eq(pd.Timestamp("2024-03-04")) & intraday.date.eq(pd.Timestamp("2024-03-06"))].copy()
        early_week = weekly.copy(deep=True)
        early_week["trade_week"] = pd.Timestamp("2024-01-08")
        early_week["feature_week"] = pd.Timestamp("2024-01-01")
        early_week["signal_datetime"] = pd.Timestamp("2024-01-05 15:00")
        early_week["feature_source_start"] = pd.Timestamp("2023-12-11")
        early = normal.copy(deep=True)
        early["datetime"] = pd.Timestamp("2024-01-10 09:35")
        early["trade_week"] = pd.Timestamp("2024-01-08")
        early["date"] = pd.Timestamp("2024-01-10")
        early["feature_source_start"] = pd.Timestamp("2024-01-08")
        early["label_start"] = early.datetime + pd.Timedelta(minutes=1)
        early["label_end"] = early.datetime + pd.Timedelta(minutes=21)
        source = pd.concat([weekly, early_week], ignore_index=True)
        minutes = pd.concat([normal, early], ignore_index=True)
        predictions = fake_weekly_predictions(source, {**self.cfg, "top_n": 2})
        cfg = {**self.cfg, "top_n": 2, "weekly_top_n": 2,
               "weekly_context_model_training_years": [2023], "weekly_context_model_sha256": "c" * 64}
        old_training, _, _ = select_training_rows(minutes, [2024], cfg, "intraday", [2023, 2025, 2026])
        context = attach_weekly_context(minutes, predictions, source, cfg)
        training, _, audit = select_training_rows(context, [2024], cfg, "intraday", [2023, 2025, 2026])
        self.assertEqual(len(old_training), 4)
        self.assertEqual(len(training), 2)
        self.assertTrue(training.datetime.dt.month.eq(3).all())
        self.assertGreater(audit.loc[audit.reason.eq("feature_year_not_training"), "excluded_rows"].iloc[0], 0)

    def test_missing_duplicate_unknown_future_or_conflicting_context_fails_closed(self):
        weekly, rows, predictions, cfg = self.context_inputs()
        cases = []
        cases.append((weekly, rows, predictions.loc[predictions.symbol.ne("SH.600000")], cfg))
        cases.append((weekly, rows, pd.concat([predictions, predictions.iloc[[0]]]), cfg))
        cases.append((weekly.loc[weekly.symbol.ne("SH.600000")], rows, predictions, cfg))
        cases.append((pd.concat([weekly, weekly.iloc[[0]]]), rows, predictions, cfg))
        for value in (pd.NaT, pd.Timestamp("2023-03-10")):
            bad = weekly.copy(deep=True)
            bad["feature_source_start"] = value
            cases.append((bad, rows, predictions, cfg))
        for changes in ({"weekly_context_model_training_years": [2023]}, {"weekly_context_model_sha256": "invalid"}):
            cases.append((weekly, rows, predictions, {**cfg, **changes}))
        reserved = rows.assign(f_weekly_score=.5)
        cases.append((weekly, reserved, predictions, cfg))
        wrong_score = predictions.copy()
        wrong_score["score"] += .01
        cases.append((weekly, rows, wrong_score, cfg))
        for index, (w, i, p, c) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                attach_weekly_context(i, p, w, c)

    def test_invalid_core_configuration_or_returned_model_years_are_rejected(self):
        for core, excluded, changes in (([2023], [2024, 2025, 2026], {}),
            ([2023, 2024, 2025], [2026], {}), ([2023, 2024], [2024, 2026], {}),
            ([2023, 2024], [2025, 2026], {"strict_feature_provenance": False}),
            ([2023, 2024], [2025, 2026], {"weekly_top_n": 2})):
            with self.subTest(core=core, changes=changes), self.assertRaises(ValueError):
                build_core_t_cohort(self.weekly, self.intraday, core, excluded, {**self.cfg, **changes}, self.directory / "invalid")
        def wrong_fit(*args, **kwargs):
            result = self.fake_fit(*args, **kwargs)
            result["metadata"]["train_years"] = [2023, 2024]
            return result
        with patch("quant_backtest.weekly_t_cohort.fit_weekly_models", side_effect=wrong_fit), self.assertRaisesRegex(ValueError, "元数据"):
            build_core_t_cohort(self.weekly, self.intraday, [2023, 2024], [2025, 2026], self.cfg, self.directory / "wrong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
