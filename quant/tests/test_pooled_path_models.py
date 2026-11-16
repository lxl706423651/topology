"""完整核心模型的合成检查：标签并发降权、窗口隔离、分解恒等式与无未来输入推断。"""

import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from quant_backtest.pathwise_labels import PATH_FEATURES
from quant_backtest.multiscale_training import MACRO_COLUMNS
from quant_backtest.pooled_path_models import (calculate_interval_uniqueness, select_pooled_training,
    prepare_pooled_targets, fit_pooled_model, generate_pooled_scores, summarize_pooled_training,
    STOCK_FEATURES)


class PooledPathModelTests(unittest.TestCase):
    """对确定性小样本验证实际权重/标签/模型接口，不使用真实外层数据。"""

    def create_data(self):
        """生成40日共同钟和30日、20股的完整路径，环境同刻相同，每股一天仅一个入口。"""
        rng = np.random.default_rng(68)
        times = pd.bdate_range("2024-02-01", periods=40) + pd.Timedelta(hours=9, minutes=31)
        frame = pd.DataFrame([(t, f"SH.{600000+s}", times[i+3]+pd.Timedelta(minutes=1))
            for i,t in enumerate(times[:30]) for s in range(20)],
            columns=["datetime", "symbol", "path_h1_window_end"])
        for c in PATH_FEATURES:
            frame[c] = rng.normal(size=len(frame))
        for c in MACRO_COLUMNS:
            values = pd.Series(rng.normal(size=len(times)), index=times)
            frame[c] = frame.datetime.map(values)
        frame["path_h1_net_return"] = .003 * np.tanh(frame.context_pool_return) + .004 * frame[STOCK_FEATURES[0]]
        frame["feature_source_start"] = frame.datetime - pd.Timedelta(days=2)
        cfg = {"pooled_min_label_group": 5, "path_exit_grace_days": 2,
            "simple_min_training_rows": 20, "label_clip": .2, "entry_equity_fraction": .1,
            "workers": 1, "model": {"random_state": 42},
            "pooled_environment_model": {"num_leaves": 3, "max_depth": 2, "n_estimators": 8,
                                         "min_child_samples": 2, "reg_lambda": 5.}}
        windows = [{"start": times[0], "end": times[25]}]
        return frame, times, cfg, windows

    def test_uniqueness_matches_brute_indicator(self):
        """差分区间算法与逐格并发矩阵一致，闭区间接触与完全重合均计算降权。"""
        starts, ends, size = np.array([0, 2, 4, 0]), np.array([2, 3, 5, 2]), 6
        matrix = np.array([(np.arange(size)>=s)&(np.arange(size)<=e) for s,e in zip(starts,ends)])
        count = matrix.sum(axis=0)
        expected = np.array([(1/count[row]).mean() for row in matrix])
        np.testing.assert_allclose(calculate_interval_uniqueness(starts, ends, size), expected)
        np.testing.assert_allclose(calculate_interval_uniqueness(np.array([0,0]),np.array([2,2]),3),[.5,.5])
        with self.assertRaises(ValueError):
            calculate_interval_uniqueness(np.array([2]),np.array([1]),3)

    def test_full_core_boundaries_and_weights(self):
        """窗口外标签扰动不改变核心；同股存在重叠但唯一性降权，每个核心日期总权重相同。"""
        frame,times,cfg,windows = self.create_data()
        selected = select_pooled_training(frame,windows,1,cfg,times)
        self.assertGreater(len(selected),300)
        self.assertTrue((selected.label_end < windows[0]["end"]).all())
        self.assertTrue(selected.sample_uniqueness.between(0,1).all())
        self.assertLess(selected.sample_uniqueness.mean(),.6)
        day_weight = selected.groupby(selected.datetime.dt.normalize()).sample_weight.sum()
        np.testing.assert_allclose(day_weight,day_weight.iloc[0])
        self.assertAlmostEqual(selected.sample_weight.mean(),1.)
        other = frame.copy()
        other.loc[other.datetime>=windows[0]["end"],"path_h1_net_return"] = 999
        altered = select_pooled_training(other.sample(frac=1,random_state=2),windows,1,cfg,times)
        pd.testing.assert_frame_equal(selected,altered)
        shifted = pd.DatetimeIndex([times[0]-pd.Timedelta(days=100)]).append(times)
        pd.testing.assert_frame_equal(selected,select_pooled_training(frame,windows,1,cfg,shifted))

    def test_decomposition_is_weighted_identity(self):
        """同刻核心共同目标加个股残差精确恢复截断目标，残差加权和为0，不对外层标签做归一化。"""
        frame,times,cfg,windows = self.create_data()
        train = select_pooled_training(frame,windows,1,cfg,times)
        target,baseline,residual = prepare_pooled_targets(train,cfg)
        np.testing.assert_allclose(residual+train.datetime.map(baseline.target),target)
        totals = (residual*train.sample_weight).groupby(train.datetime).sum()
        np.testing.assert_allclose(totals,0,atol=1e-12)
        train.loc[1,MACRO_COLUMNS[0]] = np.nan
        with self.assertRaises(ValueError):
            prepare_pooled_targets(train,cfg)

    def test_reload_and_future_label_independence(self):
        """单分量/双分量均保存重载，预测只取实际入模列，任意分批推断与完整推断一致。"""
        frame,times,cfg,windows = self.create_data()
        train = select_pooled_training(frame,windows,1,cfg,times)
        for algorithm in ["direct","decomposed"]:
            profile = {"name": algorithm, "algorithm": algorithm, "horizon": 1,
                "parameters": {"n_estimators": 8, "num_leaves": 4, "max_depth": 2, "min_child_samples": 10}}
            bundle,details = fit_pooled_model(train,profile,cfg)
            first = generate_pooled_scores(bundle,train,cfg)
            visible = train[["datetime","symbol"]+list(PATH_FEATURES)]
            with tempfile.TemporaryDirectory() as temp:
                file = Path(temp)/"model.joblib"
                joblib.dump(bundle,file)
                reloaded = joblib.load(file)
                second = generate_pooled_scores(reloaded,visible.assign(forward_return=999),cfg)
                pd.testing.assert_frame_equal(first,second)
            partial = generate_pooled_scores(bundle,visible.iloc[:17],cfg)
            pd.testing.assert_frame_equal(first.iloc[:17],partial)
            expected = sum(m.predict(visible[m.booster_.feature_name()])/10000 for m in bundle["components"].values())
            np.testing.assert_allclose(first.prediction,expected)
            self.assertEqual(set(details["特征代码"]),set(PATH_FEATURES))
            self.assertEqual(bundle["core_keys_sha256"],summarize_pooled_training(train,cfg)["core_keys_sha256"])
            if algorithm=="decomposed":
                self.assertEqual(bundle["components"]["stock"].booster_.feature_name(),STOCK_FEATURES)
                self.assertEqual(bundle["components"]["environment"].booster_.feature_name(),MACRO_COLUMNS)

    def test_core_hashes_detect_inputs_and_supervision(self):
        """拟合摘要覆盖特征、收益、权重和日期来源，不只核对样本数。"""
        frame,times,cfg,windows = self.create_data()
        train = select_pooled_training(frame,windows,1,cfg,times)
        original = summarize_pooled_training(train,cfg)
        changed = train.copy()
        changed.loc[0,STOCK_FEATURES[0]] += .1
        self.assertNotEqual(original["core_features_sha256"],summarize_pooled_training(changed,cfg)["core_features_sha256"])
        changed = train.copy()
        changed.loc[0,"path_target"] += .001
        self.assertNotEqual(original["core_supervision_sha256"],summarize_pooled_training(changed,cfg)["core_supervision_sha256"])
        changed = train.copy()
        changed.loc[0,"sample_weight"] *= .5
        self.assertNotEqual(original["core_supervision_sha256"],summarize_pooled_training(changed,cfg)["core_supervision_sha256"])


if __name__ == "__main__":
    unittest.main()
