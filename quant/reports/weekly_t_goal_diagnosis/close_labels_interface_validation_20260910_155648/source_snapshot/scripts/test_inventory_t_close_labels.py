"""固定14:55数据接口反例；仅合成小表，不训练、回测或修改已有运行输入。"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.inventory_t_close_labels import (
    SIGNAL_TIMES, LABEL_COLUMNS, extract_fixed_close_labels, attach_fixed_close_labels)


def original_minutes() -> pd.DataFrame:
    """同一固定日期的原8信号；旧target故意清空，模拟不完整周条件。"""
    times = pd.to_datetime(["2024-01-08 " + clock for clock in SIGNAL_TIMES])
    return pd.DataFrame({"datetime": times, "symbol": "SH.600000",
                         "execution_time": times + pd.Timedelta(minutes=1),
                         "execution_price": np.linspace(10., 11., len(times)),
                         "recovery_time": times.normalize() + pd.Timedelta(hours=14, minutes=55),
                         "recovery_price": 11., "label_start": times + pd.Timedelta(minutes=1),
                         "label_end": times.normalize() + pd.Timedelta(hours=14, minutes=55),
                         "target_return": np.nan, "trade_week_complete": False,
                         "execution_volume": 100., "recovery_volume": 100.,
                         "execution_amount": 1000., "recovery_amount": 1100.,
                         "f_current": pd.Series(np.arange(len(times)), dtype="float32")})


class FixedCloseLabelTests(unittest.TestCase):
    """核端点身份、缺失保留、纯当前来源及单独标签左联的经济语义。"""

    def setUp(self):
        self.cfg = {"signal_times": list(SIGNAL_TIMES), "fixed_recovery_time": "14:55",
                    "horizons": [20, 40, 60]}
        self.original = original_minutes()
        self.core = self.original[["datetime", "symbol", "f_current"]].copy()
        self.core["feature_source_start"] = pd.to_datetime(["2024-01-01"] * 7 + [None])
        self.core["baseline_opening_shares"] = 700
        self.core["source_model_training_years"] = "2025"

    def test_original_eight_clocks_and_no_fake_integer_horizon(self):
        labels, audit = extract_fixed_close_labels(self.original, self.cfg)
        self.assertEqual(list(labels), LABEL_COLUMNS)
        self.assertNotIn("horizon_minutes", labels)
        self.assertNotIn("f_current", labels)
        self.assertTrue(labels.execution_time.eq(labels.datetime + pd.Timedelta(minutes=1)).all())
        self.assertTrue(labels.recovery_time.dt.strftime("%H:%M").eq("14:55").all())
        self.assertEqual(labels.recovery_key.unique().tolist(), ["close"])
        self.assertEqual(labels.recovery_kind.unique().tolist(), ["fixed_clock"])
        self.assertEqual(labels.recovery_clock.unique().tolist(), ["14:55"])
        self.assertFalse(audit["fixed_integer_horizon_used"])
        self.assertEqual(labels.iloc[-1].recovery_time-labels.iloc[-1].execution_time, pd.Timedelta(minutes=54))

    def test_old_complete_week_flag_and_old_target_do_not_remove_reference_labels(self):
        labels, audit = extract_fixed_close_labels(self.original, self.cfg)
        self.assertEqual(len(labels), len(self.original))
        expected = self.original.recovery_price/self.original.execution_price-1
        pd.testing.assert_series_equal(labels.target_return, expected, check_names=False, check_exact=True)
        self.assertTrue(labels.label_available.all())
        self.assertNotIn("trade_week_complete", labels)
        self.assertEqual(audit["incomplete_legacy_week_rows_retained"], 8)
        self.assertEqual(audit["legacy_missing_target_with_available_prices"], 8)
        changed = self.original.assign(target_return=999., trade_week_complete=True)
        new, _ = extract_fixed_close_labels(changed, self.cfg)
        pd.testing.assert_frame_equal(labels, new, check_exact=True)

    def test_missing_either_endpoint_keeps_current_key_and_explicit_unavailable_label(self):
        original = self.original.copy()
        original.loc[0, "execution_price"] = np.nan
        original.loc[1, "recovery_price"] = np.nan
        labels, audit = extract_fixed_close_labels(original, self.cfg)
        self.assertEqual(len(labels), 8)
        self.assertTrue(labels.loc[:1, "target_return"].isna().all())
        self.assertFalse(labels.loc[:1, "label_available"].any())
        self.assertTrue(labels[TIME_COLUMNS := ["execution_time", "recovery_time", "label_start", "label_end"]].notna().all().all())
        self.assertEqual(audit["label_unavailable_rows"], 2)

    def test_zero_future_amount_and_volume_are_reference_labels_not_fill_guarantees(self):
        original = self.original.copy()
        original.loc[0, ["execution_volume", "execution_amount", "recovery_volume", "recovery_amount"]] = 0.
        labels, audit = extract_fixed_close_labels(original, self.cfg)
        self.assertTrue(labels.label_available.all())
        for field in ("execution_volume", "execution_amount", "recovery_volume", "recovery_amount"):
            self.assertEqual(audit["finite_reference_labels_with_zero_" + field], 1)
        self.assertFalse(audit["label_is_executable_fill_guarantee"])
        self.assertFalse(audit["target_is_portfolio_execution_net_pnl"])
        self.assertFalse(audit["future_volume_used_for_opportunity_selection"])

    def test_wrong_endpoint_times_are_rejected_even_when_prices_missing(self):
        for field, value in [("execution_time", "2024-01-08 09:37"),
                             ("recovery_time", "2024-01-08 15:00"),
                             ("label_start", "2024-01-08 09:35"),
                             ("label_end", "2024-01-09 14:55"),
                             ("execution_time", None)]:
            with self.subTest(field=field, value=value):
                changed = self.original.copy()
                changed.loc[0, field] = pd.Timestamp(value) if value is not None else pd.NaT
                changed.loc[0, "execution_price"] = np.nan
                with self.assertRaisesRegex(ValueError, "身份"):
                    extract_fixed_close_labels(changed, self.cfg)

    def test_invalid_current_clock_duplicate_or_missing_identity_rejected(self):
        for value in ["2024-01-08 11:30", "2024-01-08 13:00", "2024-01-08 10:00:01", None]:
            with self.subTest(value=value):
                changed = self.original.copy()
                changed.loc[0, "datetime"] = pd.Timestamp(value) if value is not None else pd.NaT
                with self.assertRaises(ValueError):
                    extract_fixed_close_labels(changed, self.cfg)
        duplicated = pd.concat([self.original, self.original.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "重复"):
            extract_fixed_close_labels(duplicated, self.cfg)
        with self.assertRaisesRegex(ValueError, "当前键"):
            extract_fixed_close_labels(self.original.assign(symbol=None), self.cfg)

    def test_config_cannot_redefine_signal_set_or_clock(self):
        for cfg in ({**self.cfg, "fixed_recovery_time": "15:00"},
                    {**self.cfg, "signal_times": ["10:00"]},
                    {**self.cfg, "signal_times": list(reversed(SIGNAL_TIMES))}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                extract_fixed_close_labels(self.original, cfg)

    def test_nonpositive_and_infinite_prices_rejected_nan_is_not_error(self):
        for value in (0., -1., np.inf, -np.inf):
            with self.subTest(value=value):
                changed = self.original.copy()
                changed.loc[0, "recovery_price"] = value
                with self.assertRaisesRegex(ValueError, "正有限"):
                    extract_fixed_close_labels(changed, self.cfg)

    def test_left_join_preserves_missing_rows_order_index_features_and_nat_provenance(self):
        labels, _ = extract_fixed_close_labels(self.original, self.cfg)
        labels = labels.iloc[[0, 2, 3, 4, 5, 6]].copy()
        core = self.core.iloc[::-1].copy()
        core.index = [40, 40, 7, 8, 9, 10, 11, 12]  # 非默认且重复index不能错配因子。
        joined = attach_fixed_close_labels(core, labels, self.cfg)
        pd.testing.assert_frame_equal(joined[core.columns], core, check_exact=True)
        missing = ~core.datetime.isin(labels.datetime)
        self.assertEqual(len(joined), len(core))
        self.assertTrue(joined.loc[missing, "target_return"].isna().all())
        self.assertFalse(joined.loc[missing, "label_available"].any())
        self.assertTrue(joined.loc[missing, "label_start"].notna().all())
        self.assertTrue(joined.loc[missing, "label_end"].dt.strftime("%H:%M").eq("14:55").all())
        self.assertTrue(joined.feature_source_start.iloc[0] is pd.NaT)
        self.assertEqual(joined.f_current.dtype, np.dtype("float32"))

    def test_future_price_rewrite_changes_only_labels_not_core_or_opportunity_keys(self):
        before = self.original.copy(deep=True)
        core_before = self.core.copy(deep=True)
        labels, _ = extract_fixed_close_labels(self.original, self.cfg)
        changed = self.original.assign(recovery_price=12., execution_amount=0., recovery_volume=0.)
        other, _ = extract_fixed_close_labels(changed, self.cfg)
        first = attach_fixed_close_labels(self.core, labels, self.cfg)
        second = attach_fixed_close_labels(self.core, other, self.cfg)
        pd.testing.assert_frame_equal(first[self.core.columns], second[self.core.columns], check_exact=True)
        self.assertFalse(first.target_return.equals(second.target_return))
        pd.testing.assert_frame_equal(self.original, before, check_exact=True)
        pd.testing.assert_frame_equal(self.core, core_before, check_exact=True)

    def test_core_future_fields_and_legacy_week_qualification_rejected(self):
        labels, _ = extract_fixed_close_labels(self.original, self.cfg)
        for field in ("execution_price", "recovery_price", "target_return", "label_available",
                      "trade_week_complete", "horizon_minutes"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "剥离"):
                attach_fixed_close_labels(self.core.assign(**{field: 1}), labels, self.cfg)

    def test_attached_label_identity_value_and_duplicate_tampering_rejected(self):
        labels, _ = extract_fixed_close_labels(self.original, self.cfg)
        for field, value in [("recovery_key", "h20"), ("recovery_kind", "duration"),
                             ("recovery_clock", "15:00"), ("target_return", 123.),
                             ("label_available", False)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                changed = labels.copy()
                changed.loc[0, field] = value
                attach_fixed_close_labels(self.core, changed, self.cfg)
        with self.assertRaisesRegex(ValueError, "重复"):
            attach_fixed_close_labels(self.core, pd.concat([labels, labels.iloc[:1]]), self.cfg)
        with self.assertRaises(ValueError):
            attach_fixed_close_labels(self.core, labels.assign(horizon_minutes=240), self.cfg)

    def test_empty_label_and_core_inputs_keep_schema_and_current_rows(self):
        empty, audit = extract_fixed_close_labels(self.original.iloc[:0], self.cfg)
        self.assertEqual(list(empty), LABEL_COLUMNS)
        self.assertEqual(audit["rows"], 0)
        joined = attach_fixed_close_labels(self.core, empty, self.cfg)
        self.assertEqual(len(joined), 8)
        self.assertFalse(joined.label_available.any())
        self.assertTrue(joined.target_return.isna().all())
        none = attach_fixed_close_labels(self.core.iloc[:0], empty, self.cfg)
        self.assertTrue(none.empty)
        self.assertEqual(set(none), set(self.core) | set(LABEL_COLUMNS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
