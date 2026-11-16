"""执行路径模型的合成训练检查：验证标签窗口、固定特征、概率幅度及保存重载，不评价真实外层。"""

import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from quant_backtest.pathwise_labels import PATH_FEATURES
from quant_backtest.pathwise_models import (fit_path_model, generate_path_scores, select_path_rows, create_path_exits)


class PathwiseModelTests(unittest.TestCase):
    """检查执行净收益回归及概率模型只用训练信息，退出与对应标签期限相同。"""

    def create_frame(self):
        """返回确定性的600行特征和净账户收益标签，部分意图为未成交零收益。"""
        rng = np.random.default_rng(47)
        frame = pd.DataFrame({c: rng.normal(size=600) for c in PATH_FEATURES})
        frame["datetime"] = pd.date_range("2024-02-01 09:31", periods=600, freq="h")
        frame["symbol"] = "SH.600000"
        frame["path_target"] = .004 * np.tanh(frame.context_pool_momentum5) + rng.normal(0, .001, 600)
        frame.loc[:10, "path_target"] = 0.
        frame.loc[20:40, "cs_rank_boll_pctb_20"] = np.nan
        return frame

    def create_config(self):
        """创建小型模型/标签边界所需配置，交易费用不在这里重复扣除。"""
        return {"simple_min_training_rows": 20, "label_clip": .2, "entry_equity_fraction": .1,
                "workers": 1, "model": {"random_state": 42}, "path_exit_grace_days": 5,
                "path_initial_atr": 20., "path_trailing_atr": 40.}

    def test_model_units_schema_and_reload(self):
        """回归和概率两种模型保存重载后完全一致，未来标签删除或修改不会改变推断。"""
        frame, cfg = self.create_frame(), self.create_config()
        for algorithm in ["regression", "classification"]:
            profile = {"name": algorithm, "algorithm": algorithm, "horizon": 3,
                "parameters": {"n_estimators": 10, "num_leaves": 4, "max_depth": 2, "min_child_samples": 10}}
            bundle, details = fit_path_model(frame, profile, cfg)
            self.assertEqual(len(details), 29)
            self.assertEqual(bundle["model"].booster_.feature_name(), list(PATH_FEATURES))
            first = generate_path_scores(bundle, frame, cfg)
            current = frame.drop(columns="path_target").assign(forward_return=999)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.joblib"
                joblib.dump(bundle, path)
                second = generate_path_scores(joblib.load(path), current, cfg)
            pd.testing.assert_frame_equal(first, second)
            self.assertTrue(np.isfinite(first.prediction).all())
            self.assertGreater(bundle["positive_mean"], 0)
            self.assertLess(bundle["nonpositive_mean"], 0)
            if algorithm == "classification":
                self.assertTrue(first.prediction.between(bundle["nonpositive_mean"], bundle["positive_mean"]).all())

    def test_fixed_potential_window_purges_early_winners(self):
        """实际提前盈利平仓仍不能让潜在标签窗口跨入验证，避免按退出结果放宽边界。"""
        frame = self.create_frame().iloc[:3].copy()
        frame["datetime"] = pd.to_datetime(["2024-02-01", "2024-02-05", "2024-02-09"])
        frame["path_h3_net_return"] = [.01, .01, 0.]
        frame["path_h3_window_end"] = pd.to_datetime(["2024-02-09", "2024-02-15", "2024-02-20"])
        frame["path_h3_observed_end"] = pd.to_datetime(["2024-02-02 00:00", "2024-02-06 00:00", "2024-02-09 00:01"])
        windows = [{"start": pd.Timestamp("2024-02-01"), "end": pd.Timestamp("2024-02-10")}]
        selected = select_path_rows(frame, windows, 3, self.create_config())
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.label_end.iloc[0], pd.Timestamp("2024-02-09"))

    def test_exit_is_specific_to_model_horizon(self):
        """每个模型只有对应期限的退出候选，不能训练3日路径再根据验证套10日退出。"""
        cfg = self.create_config()
        exits = create_path_exits({"horizon": 3}, cfg)
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["name"], "path_h3")
        self.assertEqual(exits[0]["exit_mode"], "matched_horizon")
        self.assertEqual(exits[0]["max_days"], 0)

    def test_classifier_rejects_single_class(self):
        """没有正/非正两类时拒绝训练概率模型，不能静默伪造常数盈利概率。"""
        frame = self.create_frame().assign(path_target=0.)
        profile = {"name": "single", "algorithm": "classification", "horizon": 3, "parameters": {}}
        with self.assertRaises(ValueError):
            fit_path_model(frame, profile, self.create_config())


if __name__ == "__main__":
    unittest.main()
