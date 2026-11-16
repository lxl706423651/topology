"""可解释模型单元检查：使用合成已完成特征，不读取真实测试收益或连接券商。"""

import unittest

import numpy as np
import pandas as pd

from quant_backtest.simple_signals import (FEATURE_NAMES, calculate_family_signal, calculate_day_weights,
    calculate_gate, select_spaced_training, select_horizon_rows, fit_simple_model, generate_simple_scores)


class SimpleSignalsTests(unittest.TestCase):
    """验证固定特征、拟合边界、同股非重叠抽样及无标签推断。"""

    def create_frame(self, rows=300):
        """创建rows行确定性特征及未来标签，用于局部接口测试而非盈利评估。"""
        rng = np.random.default_rng(8)
        frame = pd.DataFrame({c: rng.uniform(0, 1, rows) for c in FEATURE_NAMES})
        frame["datetime"] = pd.date_range("2024-03-01 09:31", periods=rows, freq="D")
        frame["symbol"] = "SH.600000"
        frame["signal_daily_return"] = -.01
        frame["forward_return"] = rng.normal(.006, .02, rows)
        frame["label_end"] = frame.datetime + pd.Timedelta(days=1, minutes=1)
        return frame

    def create_config(self):
        """返回训练/标签测试所需费用和线程参数，不覆盖正式YAML。"""
        return {"simple_min_training_rows": 10, "isotonic_bins": 10, "workers": 1,
                "commission_rate": .0008, "transfer_rate": .00001, "slippage_rate": .0005,
                "stamp_rate_from_20230828": .0005, "stamp_rate_before_20230828": .001, "label_clip": .2}

    def test_fixed_schema(self):
        self.assertEqual(len(FEATURE_NAMES), 26)
        self.assertFalse(any("future" in c or "label" in c for c in FEATURE_NAMES))

    def test_day_weights_equal(self):
        frame = self.create_frame(4)
        frame["datetime"] = pd.to_datetime(["2024-03-01"] * 3 + ["2024-03-02"])
        weights = calculate_day_weights(frame)
        self.assertAlmostEqual(weights[:3].sum(), weights[3])
        self.assertAlmostEqual(weights.mean(), 1)

    def test_signals_fail_closed(self):
        frame = self.create_frame(8)
        frame.loc[0, "signal_daily_return"] = np.nan
        frame.loc[1, "cs_rank_context_momentum_5d"] = np.nan
        frame.loc[2, "cs_rank_money_flow_imbalance_20"] = np.nan
        _, valid = calculate_family_signal(frame, "pullback_trend")
        self.assertFalse(valid[:3].any())
        with self.assertRaises(ValueError):
            calculate_family_signal(frame, "unknown")

    def test_gate_ignores_ineligible_and_losses(self):
        scores = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"]),
            "prediction": [.01, 5, -.1, .02, .04], "eligible": [True, False, True, True, True]})
        self.assertAlmostEqual(calculate_gate(scores, .5, 0), .03)
        self.assertEqual(calculate_gate(scores.assign(eligible=False), .5, 0), 1e9)

    def test_nonoverlap_and_permutation(self):
        days = pd.bdate_range("2024-01-01", periods=90)
        offsets = [pd.Timedelta(hours=9, minutes=31 + i) for i in range(10)]
        times = pd.DatetimeIndex([d + o for d in days for o in offsets])
        frame = pd.DataFrame([(t, s) for t in times for s in ["SH.600000", "SZ.000001"]],
                             columns=["datetime", "symbol"])
        day_index = days.get_indexer(frame.datetime.dt.normalize())
        frame["label_end"] = pd.DatetimeIndex([days[min(i + 3, len(days) - 1)] for i in day_index]) + (
            frame.datetime - frame.datetime.dt.normalize()) + pd.Timedelta(minutes=1)
        selected = select_spaced_training(frame, times, 3)
        self.assertGreater(len(selected), 30)
        previous = selected.groupby("symbol").label_end.shift()
        self.assertTrue((previous.isna() | (previous < selected.datetime)).all())
        reordered = select_spaced_training(frame.sample(frac=1, random_state=2), times, 3)
        pd.testing.assert_frame_equal(selected, reordered)
        extra_times = pd.DatetimeIndex([pd.Timestamp("2023-12-29") + o for o in offsets]).append(times)
        pd.testing.assert_frame_equal(selected, select_spaced_training(frame, extra_times, 3))
        with self.assertRaises(ValueError):
            select_spaced_training(pd.concat([frame, frame.head(1)]), times, 3)

    def test_labels_contained(self):
        frame = self.create_frame(5)
        selected = select_horizon_rows(frame, [{"start": pd.Timestamp("2024-03-01"), "end": pd.Timestamp("2024-03-04")}], 1)
        self.assertEqual(len(selected), 2)
        self.assertTrue((selected.label_end < pd.Timestamp("2024-03-04")).all())

    def test_models_no_label_reads_or_refit(self):
        frame = self.create_frame()
        cfg = self.create_config()
        for profile in [
            {"name": "ridge", "algorithm": "ridge", "penalty": .1, "horizon": 1},
            {"name": "isotonic", "algorithm": "isotonic", "family": "oversold_reversal", "horizon": 1},
        ]:
            bundle, details = fit_simple_model(frame, profile, cfg)
            self.assertTrue(len(details) > 0)
            source_copy = frame.copy(deep=True)
            original = generate_simple_scores(bundle, frame, cfg)
            stripped = frame.drop(columns=["forward_return", "label_end"])
            pd.testing.assert_frame_equal(original, generate_simple_scores(bundle, stripped, cfg))
            changed = frame.assign(forward_return=999, label_end=pd.Timestamp("2099-01-01"))
            pd.testing.assert_frame_equal(original, generate_simple_scores(bundle, changed, cfg))
            pd.testing.assert_frame_equal(frame, source_copy)
            self.assertTrue(np.isfinite(original.prediction).all())
            if profile["algorithm"] == "ridge":
                np.testing.assert_allclose(bundle["model"]["imputer"].statistics_, frame[list(FEATURE_NAMES)].median())
            else:
                knot = bundle["model"].y_thresholds_
                self.assertTrue((np.diff(knot) >= -1e-10).all())


if __name__ == "__main__":
    unittest.main()
