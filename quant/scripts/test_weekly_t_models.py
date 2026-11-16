"""用小型合成周/分钟数据验证周选股和库存 T 模型，不改真实行情或研究缓存。

从 quant 根目录执行 python scripts/test_weekly_t_models.py；小模型及原生权重
只写临时目录。覆盖年度隔离、因果输入、未来标签缺失、排名和模型可复现加载。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import lightgbm
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_models import (
    build_t_model_profiles, fit_t_model, fit_weekly_models, predict_t, predict_weekly,
    select_training_rows,
)


def create_weekly_frame() -> pd.DataFrame:
    """生成四年份的已完成周及下一周标签，所有 f_* 只来自信号周状态。"""
    random = np.random.default_rng(2026)
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        for week in pd.date_range(f"{year}-02-01", periods=6, freq="W-MON"):
            for stock in range(12):
                signal = random.normal()
                feature2 = random.normal()
                rows.append({"symbol": f"SH.{600000 + stock}", "feature_week": week,
                             "signal_datetime": week + pd.Timedelta(days=4, hours=15),
                             "trade_week": week + pd.Timedelta(days=7),
                             "entry_date": week + pd.Timedelta(days=7),
                             "exit_date": week + pd.Timedelta(days=11),
                             "label_start": week + pd.Timedelta(days=7, hours=9, minutes=31),
                             "label_end": week + pd.Timedelta(days=11, hours=15),
                             "feature_week_complete": True, "trade_week_complete": True,
                             "target_return": .03 * signal + random.normal(0, .002),
                             "target_amplitude": .02 + .02 * abs(feature2),
                             "f_signal": signal, "f_signal_duplicate": signal,
                             "f_second": feature2, "f_constant": 1.0, "f_missing": np.nan,
                             "future_price": 10.0 + 2 * signal})
    return pd.DataFrame(rows)


def create_intraday_frame() -> pd.DataFrame:
    """生成同日下一分钟至14:55标签，负 prediction 对应卖出后等待回补的方向。"""
    random = np.random.default_rng(42)
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        for date in pd.bdate_range(f"{year}-03-05", periods=4):
            for minutes in [600, 660, 810]:
                for stock in range(10):
                    signal = random.normal()
                    timestamp = date + pd.Timedelta(minutes=minutes)
                    execution = timestamp + pd.Timedelta(minutes=1)
                    recovery = date + pd.Timedelta(hours=14, minutes=55)
                    price = 10 + random.uniform()
                    target = .02 * signal + random.normal(0, .001)
                    rows.append({"datetime": timestamp, "symbol": f"SH.{600000 + stock}",
                                 "date": date, "trade_week": date - pd.Timedelta(days=date.weekday()),
                                 "execution_time": execution, "recovery_time": recovery,
                                 "label_start": execution, "label_end": recovery,
                                 "execution_price": price, "recovery_price": price * (1 + target),
                                 "target_return": target, "signal_bar_tradable": True,
                                 "f_signal": signal, "f_clock": minutes / 900,
                                 "f_constant": 1.0, "f_missing": np.nan})
    return pd.DataFrame(rows)


class ConstantModel:
    """返回固定分数，用于独立验证top100和稳定同分排序，不依赖训练器巧合。"""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """保持输入行数，令全部股票两个模型均同分。"""
        return np.zeros(len(frame))


class FeatureModel:
    """直接返回选定的合成特征，用于手工验证70/30排名公式。"""

    def __init__(self, feature: str):
        """保存本测试需要返回的列名。"""
        self.feature = feature

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """返回合成值，不读取价格和标签。"""
        return frame[self.feature].to_numpy()


class WeeklyTModelTests(unittest.TestCase):
    """验证训练和推断边界，而非通过同一实现重算预期收益。"""

    @classmethod
    def setUpClass(cls):
        """在临时目录仅训练两个小模型bundle供推断和加载测试复用。"""
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.weekly, cls.intraday = create_weekly_frame(), create_intraday_frame()
        cls.cfg = {"workers": 1, "random_state": 7, "embargo_days": 7,
                   "min_training_rows": 20, "weekly_max_features": 8, "t_max_features": 8,
                   "feature_selection_sample_rows": 200, "feature_min_coverage": .8,
                   "feature_max_correlation": .95, "weekly_top_n": 100,
                   "weekly_return_weight": .7, "weekly_amplitude_weight": .3,
                   "weekly_model": {"n_estimators": 20, "min_child_samples": 3, "reg_lambda": .1},
                   "t_model": {"n_estimators": 20, "min_child_samples": 3, "reg_lambda": .1}}
        cls.weekly_bundle = fit_weekly_models(cls.weekly, [2023, 2025], cls.cfg,
                                              cls.directory / "weekly", excluded_years=[2024, 2026])
        cls.t_bundle = fit_t_model(cls.intraday, [2023, 2025], cls.cfg,
                                   cls.directory / "intraday", excluded_years=[2024, 2026])

    @classmethod
    def tearDownClass(cls):
        """移除临时模型，真实研究产物从未写入。"""
        cls.temporary.cleanup()

    def test_feature_and_both_label_endpoints_must_be_training_years(self):
        altered = self.weekly.copy()
        index = altered.index[altered.signal_datetime.dt.year == 2023][:3]
        altered.loc[index[0], "feature_week"] = pd.Timestamp("2022-12-26")
        altered.loc[index[1], "label_start"] = pd.Timestamp("2024-01-01")
        altered.loc[index[2], "label_end"] = pd.Timestamp("2024-01-05")
        selected, metadata, audit = select_training_rows(altered, [2023, 2025], self.cfg, "weekly", [2024, 2026])
        self.assertEqual(len(selected), 144 - 3)
        self.assertTrue(selected.feature_week.dt.year.isin([2023, 2025]).all())
        self.assertEqual(metadata["train_years"], [2023, 2025])
        self.assertGreater(audit.loc[audit.reason == "label_end_year_not_training", "excluded_rows"].iloc[0], 0)

    def test_embargo_covers_both_sides_of_excluded_year(self):
        source = self.weekly.iloc[:3].copy()
        dates = [pd.Timestamp("2023-12-22"), pd.Timestamp("2025-01-03"), pd.Timestamp("2025-02-07")]
        for index, signal in zip(source.index, dates):
            source.loc[index, "signal_datetime"] = signal + pd.Timedelta(hours=15)
            source.loc[index, "feature_week"] = signal - pd.Timedelta(days=4)
            source.loc[index, "label_start"] = signal + pd.Timedelta(days=3, hours=9, minutes=31)
            source.loc[index, "label_end"] = signal + pd.Timedelta(days=7, hours=15)
        # 添加第二个远离边界的合法样本，避免触发最小训练样本门槛。
        safe = source.iloc[[2]].assign(symbol="SH.699999")
        source = pd.concat([source, safe], ignore_index=True)
        selected, _, _ = select_training_rows(source, [2023, 2025], {**self.cfg, "min_training_rows": 2}, "weekly", [2024])
        self.assertEqual(len(selected), 2)
        self.assertTrue((selected.signal_datetime.dt.month == 2).all())

    def test_intraday_labels_cannot_cross_day_or_run_before_signal(self):
        source = self.intraday.copy()
        source.loc[0, "label_end"] += pd.Timedelta(days=1)
        source.loc[1, "label_start"] = source.loc[1, "datetime"]
        selected, _, _ = select_training_rows(source, [2023], self.cfg, "intraday")
        self.assertEqual(len(selected), 120 - 2)

    def test_overlapping_train_and_excluded_years_rejected(self):
        with self.assertRaises(ValueError):
            select_training_rows(self.weekly, [2023, 2024], self.cfg, "weekly", [2024])

    def test_future_label_completeness_only_filters_training(self):
        future = self.weekly.loc[self.weekly.signal_datetime.dt.year == 2026].copy()
        expected = predict_weekly(self.weekly_bundle, future, self.cfg)
        future["target_return"] = np.nan
        future["target_amplitude"] = np.nan
        future["trade_week_complete"] = False
        future["label_start"] = pd.NaT
        future["label_end"] = pd.NaT
        actual = predict_weekly(self.weekly_bundle, future, self.cfg)
        assert_frame_equal(expected, actual)
        with self.assertRaises(ValueError):
            fit_weekly_models(future, [2026], self.cfg, self.directory / "invalid_weekly")

    def test_weekly_prediction_never_reads_future_prices(self):
        future = self.weekly.loc[self.weekly.signal_datetime.dt.year == 2026].copy()
        first = predict_weekly(self.weekly_bundle, future, self.cfg)
        future["future_price"] = -999999
        future["target_return"] = 10000
        future["target_amplitude"] = 30000
        assert_frame_equal(first, predict_weekly(self.weekly_bundle, future, self.cfg))

    def test_t_prediction_keeps_unlabeled_rows_and_future_fields_without_using_them(self):
        future = self.intraday.loc[self.intraday.datetime.dt.year == 2026].copy()
        original = predict_t(self.t_bundle, future, self.cfg)
        future["recovery_price"] = np.nan
        future["target_return"] = np.nan
        future["label_end"] = pd.NaT
        actual = predict_t(self.t_bundle, future, self.cfg)
        self.assertEqual(len(actual), len(future))
        self.assertTrue(actual.recovery_price.isna().all())
        np.testing.assert_allclose(original.prediction, actual.prediction)

    def test_current_eligibility_filters_training_and_inference_consistently(self):
        future = self.intraday.loc[self.intraday.datetime.dt.year == 2026].copy()
        future.iloc[:4, future.columns.get_loc("signal_bar_tradable")] = False
        selected, _, _ = select_training_rows(future, [2026], self.cfg, "intraday")
        predicted = predict_t(self.t_bundle, future, self.cfg)
        self.assertEqual(len(selected), len(future) - 4)
        self.assertEqual(len(predicted), len(selected))
        self.assertEqual(len(predict_t(self.t_bundle, future, {**self.cfg, "exclude_flat_signal_bars": False})), len(future))

    def test_training_feature_selection_drops_constant_missing_and_duplicate_columns(self):
        features = self.weekly_bundle["features"]
        self.assertNotIn("f_constant", features)
        self.assertNotIn("f_missing", features)
        self.assertEqual(sum(name in features for name in ["f_signal", "f_signal_duplicate"]), 1)
        self.assertTrue(all(name.startswith("f_") for name in features))

    def test_unseen_year_changes_cannot_change_training_or_feature_selection(self):
        altered = self.weekly.copy()
        outside = ~altered.signal_datetime.dt.year.isin([2023, 2025])
        altered.loc[outside, ["target_return", "target_amplitude", "f_signal", "f_second"]] = 9999.0
        fitted = fit_weekly_models(altered, [2023, 2025], self.cfg, self.directory / "heldout_changed", [2024, 2026])
        self.assertEqual(fitted["features"], self.weekly_bundle["features"])
        self.assertEqual(fitted["metadata"]["training_sample_sha256"], self.weekly_bundle["metadata"]["training_sample_sha256"])
        untouched = self.weekly.loc[self.weekly.signal_datetime.dt.year == 2026]
        assert_frame_equal(predict_weekly(fitted, untouched, self.cfg), predict_weekly(self.weekly_bundle, untouched, self.cfg))

    def test_training_and_inference_do_not_mutate_input(self):
        original = self.weekly.copy(deep=True)
        predict_weekly(self.weekly_bundle, self.weekly, self.cfg)
        fit_weekly_models(self.weekly, [2023], self.cfg, self.directory / "no_mutation")
        assert_frame_equal(original, self.weekly)

    def test_top100_is_per_week_and_ties_are_symbol_stable(self):
        source = pd.concat([self.weekly.iloc[[0]].assign(symbol=f"SH.{600000+i}") for i in range(105)], ignore_index=True)
        source = pd.concat([source, source.assign(trade_week=source.trade_week + pd.Timedelta(days=7),
                                               signal_datetime=source.signal_datetime + pd.Timedelta(days=7))], ignore_index=True)
        bundle = {"task": "weekly", "features": ["f_signal"], "models": {"return": ConstantModel(), "amplitude": ConstantModel()},
                  "metadata": {"train_years": [2023]}}
        result = predict_weekly(bundle, source.sample(frac=1, random_state=5), self.cfg)
        self.assertEqual(result.groupby("trade_week").selected.sum().tolist(), [100, 100])
        for _, group in result.groupby("trade_week"):
            self.assertEqual(group.symbol.tolist(), sorted(group.symbol))
            self.assertEqual(group.loc[group.selected].symbol.iloc[-1], "SH.600099")

    def test_rank_combination_is_seventy_thirty(self):
        source = self.weekly.iloc[:3].copy()
        source["f_signal"] = [3., 1., 2.]
        source["f_second"] = [1., 3., 2.]
        bundle = {"task": "weekly", "features": ["f_signal", "f_second"],
                  "models": {"return": FeatureModel("f_signal"), "amplitude": FeatureModel("f_second")},
                  "metadata": {"train_years": [2023]}}
        result = predict_weekly(bundle, source, self.cfg).set_index("symbol")
        self.assertAlmostEqual(result.loc["SH.600000", "score"], .7 + .3 / 3)
        self.assertAlmostEqual(result.loc["SH.600001", "score"], .7 / 3 + .3)
        self.assertEqual(result.loc["SH.600000", "rank"], 1)

    def test_weekly_top100_cannot_include_duplicate_stock_from_different_signal_times(self):
        duplicate = self.weekly.iloc[[0]].copy()
        duplicate["signal_datetime"] += pd.Timedelta(minutes=1)
        source = pd.concat([self.weekly, duplicate], ignore_index=True)
        with self.assertRaises(ValueError):
            predict_weekly(self.weekly_bundle, source, self.cfg)

    def test_invalid_weights_and_missing_feature_are_rejected(self):
        with self.assertRaises(ValueError):
            predict_weekly(self.weekly_bundle, self.weekly, {**self.cfg, "weekly_return_weight": .8})
        with self.assertRaises(ValueError):
            predict_t(self.t_bundle, self.intraday.drop(columns=self.t_bundle["features"][0]), self.cfg)

    def test_duplicate_keys_rejected_but_duplicate_index_labels_supported(self):
        with self.assertRaises(ValueError):
            predict_t(self.t_bundle, pd.concat([self.intraday, self.intraday.iloc[[0]]]), self.cfg)
        source = self.intraday.copy()
        source.index = [0] * len(source)
        assert_frame_equal(predict_t(self.t_bundle, source, self.cfg), predict_t(self.t_bundle, self.intraday, self.cfg))

    def test_native_model_and_joblib_bundle_reload_identically(self):
        restored = joblib.load(self.directory / "weekly/model_bundle.joblib")
        assert_frame_equal(predict_weekly(restored, self.weekly, self.cfg), predict_weekly(self.weekly_bundle, self.weekly, self.cfg))
        native = lightgbm.Booster(model_file=str(self.directory / "intraday/return_model.txt"))
        expected = predict_t(self.t_bundle, self.intraday, self.cfg).prediction.to_numpy()
        source = self.intraday.sort_values(["datetime", "symbol"])
        np.testing.assert_allclose(native.predict(source[self.t_bundle["features"]]), expected)

    def test_t_prediction_sign_is_raw_future_return(self):
        source = self.intraday.copy()
        result = predict_t(self.t_bundle, source, self.cfg)
        self.assertGreater(result.loc[result.f_signal > 1, "prediction"].mean(), 0)
        self.assertLess(result.loc[result.f_signal < -1, "prediction"].mean(), 0)

    def test_task_specific_training_caps_and_profile_candidates(self):
        self.assertEqual([profile["name"] for profile in build_t_model_profiles()],
                         ["l2_depth4", "l2_depth6", "huber_depth4"])
        cfg = {**deepcopy(self.cfg), "t_max_train_rows": 50, "weekly_max_train_rows": 30, "max_train_rows": 25}
        for profile in ["l2_depth6", "huber_depth4"]:
            fitted = fit_t_model(self.intraday, [2023], {**cfg, "t_model_profile": profile}, self.directory / profile)
            self.assertEqual(fitted["metadata"]["fitted_rows"], 50)
            self.assertEqual(fitted["metadata"]["profile"], profile)
            if profile == "huber_depth4":
                self.assertEqual(fitted["metadata"]["model_parameters"]["alpha"], .01)
            self.assertTrue(np.isfinite(predict_t(fitted, self.intraday, cfg).prediction).all())
        fitted = fit_weekly_models(self.weekly, [2023], cfg, self.directory / "weekly_capped")
        self.assertEqual(fitted["metadata"]["fitted_rows"], 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
