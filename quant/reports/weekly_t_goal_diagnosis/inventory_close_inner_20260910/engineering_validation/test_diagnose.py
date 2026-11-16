"""v16内部诊断的纯公式/缺价/未配对/范围反例；不读任何市场报告或运行账户。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

FILE = Path(__file__).with_name("diagnose.py")
SPEC = importlib.util.spec_from_file_location("inventory_close_inner_diagnosis", FILE)
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def configuration():
    """独立合成配置使用原费税/时钟，仅供公式验证。"""
    return dict(min_commission=0., commission_rate=.0008, transfer_rate=.00001,
        slippage_rate=.0005, stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005,
        t_avoid_locked_exit_side=True, flat_st_limit_rate=.05, nonflat_limit_proxy=.098,
        last_entry_time="14:30")


def predictions():
    """两个相反方向当前预测，当前价格区间合法，无未来输入。"""
    return pd.DataFrame({"datetime": pd.to_datetime(["2026-03-11 09:35", "2026-03-11 10:00"]),
        "symbol": ["SH.600001", "SZ.000001"], "prediction": [.02, -.02],
        "signal_price": [10., 10.], "previous_close": [10., 10.],
        "signal_high": [10.1, 10.1], "signal_low": [9.9, 9.9],
        "signal_amount": [1e6, 1e6], "signal_volume": [1e5, 1e5]})


def inventory():
    """只为首只股票提供昨日库存；第二只真实零旧股，不按未来端点补入。"""
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-03-11"), "SH.600001")])
    return pd.Series([1000.], index=index)


class DiagnosisTests(unittest.TestCase):
    def test_exact_two_direction_costs_and_stamp_change(self):
        cfg = configuration()
        dates = pd.Series(pd.to_datetime(["2023-08-27 09:35", "2023-08-28 09:35"]))
        gross = pd.Series([.02, -.02])
        buy, sell = diagnosis.edges(gross, dates, cfg)
        for i, stamp in enumerate((.001, .0005)):
            fee, slip = .00081, .0005
            q, first, second = 1000, 10., 10.*(1+gross.iloc[i])
            bought = q*first*(1+slip)
            sold = q*second*(1-slip)
            self.assertAlmostEqual(buy.iloc[i], (sold*(1-fee-stamp)-bought*(1+fee))/(q*first), places=14)
            sold_first = q*first*(1-slip)
            bought_second = q*second*(1+slip)
            self.assertAlmostEqual(sell.iloc[i], (sold_first*(1-fee-stamp)-bought_second*(1+fee))/(q*first), places=14)
        flat_buy, flat_sell = diagnosis.edges(pd.Series([0., 0.]), dates, cfg)
        self.assertAlmostEqual(-flat_buy.iloc[0]*1e4, 36.195, places=6)
        self.assertAlmostEqual(-flat_sell.iloc[1]*1e4, 31.1975, places=6)
        with self.assertRaises(AssertionError): diagnosis.edges(gross, dates, {**cfg, "min_commission": 5.})

    def test_future_prices_missing_dont_change_current_eligibility_or_direction(self):
        raw = predictions()
        before = raw.copy(deep=True)
        valid = diagnosis.annotate_values(raw, [10., 10.], [10.2, 9.8], inventory(), configuration())
        missing = diagnosis.annotate_values(raw, [np.nan, 10.], [np.nan, np.nan], inventory(), configuration())
        self.assertEqual(len(valid), len(missing))
        self.assertEqual(missing.endpoint_available.sum(), 0)
        self.assertTrue(missing.fixed_net_bp.isna().all())
        pd.testing.assert_series_equal(valid.action, missing.action)
        for margin in (0., .001, .003):
            pd.testing.assert_series_equal(diagnosis.current_eligibility(valid, margin, configuration()),
                diagnosis.current_eligibility(missing, margin, configuration()))
        pd.testing.assert_frame_equal(raw, before)

    def test_summary_common_endpoints_and_distinct_dates_do_not_hide_missing(self):
        p = diagnosis.annotate_values(predictions(), [10., 10.], [10.2, np.nan], inventory(), configuration())
        summary = diagnosis.summarize_predictions(p)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["valid_endpoint_rows"], 1)
        self.assertEqual(summary["missing_recovery_rows"], 1)
        self.assertEqual(summary["distinct_days"], 1)
        self.assertNotIn("independent_days", summary)
        self.assertEqual(summary["mean_predicted_net_bp_common"], p.iloc[0].predicted_net_bp)
        self.assertEqual(p.opening_old_shares.tolist(), [1000., 0.])

    def test_current_limit_volume_and_exact_margin_before_bp_rounding(self):
        p = diagnosis.annotate_values(predictions(), [10., 10.], [10.2, 9.8], inventory(), configuration())
        self.assertTrue(diagnosis.current_eligibility(p, .001, configuration()).all())
        p.loc[0, ["signal_price", "signal_high", "signal_low"]] = 9.5
        p.loc[1, "signal_volume"] = 0
        self.assertFalse(diagnosis.current_eligibility(p, 0., configuration()).any())
        p = diagnosis.annotate_values(predictions(), [10., 10.], [10.2, 9.8], inventory(), configuration())
        p["predicted_net_bp"] = 1e9
        p["predicted_net_return"] = np.nextafter(.001, -np.inf)
        self.assertFalse(diagnosis.current_eligibility(p, .001, configuration()).any())

    def test_pair_full_partial_and_unrecovered_are_not_mixed_with_fragments(self):
        p = diagnosis.annotate_values(predictions(), [10., 10.], [10.2, 9.8], inventory(), configuration())
        first = pd.DataFrame({"datetime": p.fixed_start, "signal_datetime": p.datetime,
            "symbol": p.symbol, "pair_id": [1, 2], "event_type": "t", "fragment_id": 0,
            "shares": [1000, 1000], "reference_price": [10., 10.], "side": ["buy", "sell"], "fee": [10., 10.]})
        trips = pd.DataFrame({"pair_id": [1, 1, 2], "shares": [400, 600, 200],
            "second_datetime": pd.to_datetime(["2026-03-11 14:55", "2026-03-11 14:56", "2026-03-11 14:55"]),
            "net_pnl": [5., -20., -4.], "gross_pnl": [10., -10., 0.],
            "slippage_cost": [4., 6., 2.], "buy_fee": [2., 4., 2.], "sell_fee": [3., 6., 2.]})
        actual = diagnosis.pair_details(first, trips, p)
        self.assertEqual(actual.complete.tolist(), [True, False])
        self.assertEqual(actual.fragments.tolist(), [2, 1])
        self.assertEqual(actual.unrecovered_shares.tolist(), [0, 800])
        self.assertEqual(actual.loc[0, "actual_complete_net_currency"], -15.)
        self.assertTrue(pd.isna(actual.loc[1, "actual_complete_net_currency"]))
        self.assertEqual(actual.loc[1, "fragment_net"], -4.)
        self.assertGreater(actual.loc[0, "fixed_nominal_net_currency"], 0.)
        none = diagnosis.pair_details(first, trips.iloc[:0], p)
        self.assertEqual(len(none), 2)
        self.assertFalse(none.complete.any())
        self.assertTrue(none.actual_complete_net_currency.isna().all())
        self.assertEqual(none.fragment_net.sum(), 0.)
        self.assertEqual(none.all_actual_pair_fees.sum(), 20.)

    def test_actual_first_price_or_direction_mismatch_fails(self):
        p = diagnosis.annotate_values(predictions(), [10., 10.], [10.2, 9.8], inventory(), configuration())
        event = pd.DataFrame({"datetime": p.fixed_start, "signal_datetime": p.datetime,
            "symbol": p.symbol, "pair_id": [1, 2], "event_type": "t", "fragment_id": 0,
            "shares": [1000, 1000], "reference_price": [11., 10.], "side": ["buy", "sell"], "fee": [10., 10.]})
        with self.assertRaises(AssertionError): diagnosis.pair_details(event, pd.DataFrame(), p)

    def test_same_numeric_candidate_comparison_preserves_negative_differences(self):
        rows = []
        for index in range(37):
            row = dict(candidate_index=index, enabled=index > 0, profile="hold" if index == 0 else "l2_depth4",
                min_net_edge=0., fraction=0. if index == 0 else .5, exit_mode="fixed")
            row.update({name: 0. for name in diagnosis.METRIC_FIELDS})
            row["all_reference_acquisition_cost"] = 10000.
            rows.append(row)
        original = pd.DataFrame(rows)
        current = original.copy()
        current.loc[1, "incremental_net_pnl"] = -50.
        compared = diagnosis.compare_candidates(current, original)
        self.assertEqual(compared.loc[1, "incremental_net_pnl_change"], -50.)
        self.assertTrue(compared.same_forecasts.eq(False).all())
        current.loc[1, "all_reference_acquisition_cost"] = 9999.
        with self.assertRaises(AssertionError): diagnosis.compare_candidates(current, original)

    def test_read_policy_rejects_outer_before_opening_file(self):
        with tempfile.TemporaryDirectory() as directory:
            d = diagnosis.Diagnosis.__new__(diagnosis.Diagnosis)
            root = Path(directory)
            d.report, d.source, d.year, d.inner = root/"new", root/"old", 2023, 2026
            d.cfg, d.inputs = {"data_root": str(root/"raw")}, {}
            for path in (d.report/"year_2023/test_trade_events.csv",
                         d.report/"year_2023/test_predictions/current_predictions.csv",
                         d.report/"year_2024/validation_rule_metrics.csv",
                         d.source/"year_2023/test_metrics.csv", root/"raw/2023/SH.600001.csv"):
                with self.subTest(path=path), self.assertRaisesRegex(AssertionError, "禁止读取"):
                    d.record(path)
            self.assertEqual(d.inputs, {})

    def test_inventory_failure_rows_keep_recovery_base_difference_and_old_share_budget(self):
        daily = pd.DataFrame({"inventory_preserved": [True, True, False, True],
            "recovery_failure": [False, True, False, False], "baseline_inventory_difference": [0, 0, 100, 0],
            "t1_violation_count": [0, 0, 0, 0], "day_sold": [0, 100, 100, 1100], "opening_sellable": 1000})
        before = daily.copy(deep=True)
        failures = diagnosis.inventory_failures(daily)
        self.assertEqual(list(failures.index), [1, 2, 3])
        pd.testing.assert_frame_equal(daily, before)


if __name__ == "__main__": unittest.main(verbosity=2)
