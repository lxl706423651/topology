"""v18内部诊断的纯公式/缺价/未配对/范围反例；不读任何市场报告或运行账户。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

FILE = Path(__file__).with_name("diagnose.py")
SPEC = importlib.util.spec_from_file_location("inventory_coverage_inner_diagnosis", FILE)
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def configuration():
    """独立合成配置使用原费税/时钟，仅供公式验证。"""
    return dict(min_commission=0., commission_rate=.0008, transfer_rate=.00001,
        slippage_rate=.0005, stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005,
        t_avoid_locked_exit_side=True, flat_st_limit_rate=.05, nonflat_limit_proxy=.098,
        last_entry_time="14:30", fixed_recovery_time="14:55")


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
    def coverage_metadata(self):
        """合成0.80/0.70证据，允许新列及矩阵身份改变而保持共同列精确。"""
        evidence = {name: {"values_and_dtype_exact": True} for name in
            ("training_inputs.pkl", "feature_manifest.csv", "training_labels.csv", "training_split_audit.csv")}
        meta = dict(research_version=diagnosis.VERSION, training_loss="regression_l2",
            source_l2_profile="l2_depth4", profile="l2_depth4", horizon_minutes=20,
            feature_min_coverage=.7, source_feature_min_coverage=.8,
            model_parameters={"objective":"regression"}, source_features=["f_one", "f_two"],
            features=["f_one", "f_new"], added_features=["f_new"], removed_features=["f_two"])
        identity = dict(passed=True, current_profile="l2_depth4", source_profile="l2_depth4", horizon_minutes=20,
            feature_min_coverage=.7, source_feature_min_coverage=.8, source_objective="regression",
            current_objective="regression", labels_winsorized_or_deleted=False,
            common_feature_values_and_dtype_exact=True, same_selected_feature_matrix_required=False,
            current_features=meta["features"], source_features=meta["source_features"],
            added_features=["f_new"], removed_features=["f_two"], common_features=["f_one"],
            reconstructed_source_selection_labels_and_split_exact=evidence)
        return meta, identity

    def ledger(self):
        """完整pair两片段和一个pending；完整首腿1000元，不能双腿/分片重复计量。"""
        events = pd.DataFrame(dict(event_type=["t"]*5, pair_id=[1,1,1,2,2],
            fragment_id=[0,1,2,0,1], shares=[100,40,60,100,50], reference_price=[10.]*5,
            fee=[.2,.3,.5,.4,.5], slippage_cost=[.5,.2,.3,.5,.25]))
        # Use legal100share pieces, scale shares/denominator by10; monetary values are independently synthetic.
        events["shares"] *= 10
        trips = pd.DataFrame(dict(pair_id=[1,1,2], shares=[400,600,500],
            net_pnl=[2.,3.,-1.], gross_pnl=[2.4,3.6,-.3], slippage_cost=[.4,.6,.5],
            buy_fee=[.1,.1,.2], sell_fee=[.3,.5,.5]))
        return events, trips

    def test_configuration_and_saved_prediction_l2_identity(self):
        cfg = dict(research_version=diagnosis.VERSION, training_loss="regression_l2",
            feature_min_coverage=.7, source_feature_min_coverage=.8, t_recheck_exit_side_at_execution=True)
        diagnosis.validate_configuration(cfg)
        for key, value in (("training_loss", "huber"), ("feature_min_coverage", .8),
                ("source_feature_min_coverage", .7), ("t_recheck_exit_side_at_execution", False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                diagnosis.validate_configuration({**cfg, key:value})
        p = predictions().assign(model_version=diagnosis.VERSION, forecast_loss="regression_l2",
            feature_min_coverage=.7)
        p["return_score"], p["raw_prediction"] = p.prediction, p.prediction
        diagnosis.validate_prediction_identity(p)
        with self.assertRaises(AssertionError): diagnosis.validate_prediction_identity(p.assign(forecast_alpha=.01))
        with self.assertRaises(AssertionError): diagnosis.validate_prediction_identity(p.assign(return_score=.5))

    def test_model_evidence_accepts_new_feature_matrix_but_rejects_changed_labels_or_loss(self):
        meta, identity = self.coverage_metadata()
        diagnosis.validate_model_identity(meta, identity, "l2_depth4", 20)
        for key, value in (("same_selected_feature_matrix_required", True), ("labels_winsorized_or_deleted", True),
                ("common_feature_values_and_dtype_exact", False), ("current_features", ["f_one"]),
                ("added_features", [])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                diagnosis.validate_model_identity(meta, {**identity,key:value}, "l2_depth4", 20)
        with self.assertRaises(AssertionError):
            diagnosis.validate_model_identity({**meta,"model_parameters":{"objective":"huber"}}, identity, "l2_depth4", 20)
        broken = dict(identity["reconstructed_source_selection_labels_and_split_exact"])
        broken.pop("training_labels.csv")
        with self.assertRaises(AssertionError):
            diagnosis.validate_model_identity(meta, {**identity,"reconstructed_source_selection_labels_and_split_exact":broken}, "l2_depth4", 20)

    def test_complete_reference_notional_once_and_pending_fee_retained(self):
        events, trips = self.ledger()
        before, trip_before = events.copy(deep=True), trips.copy(deep=True)
        actual = diagnosis.complete_notional_statistics(events, trips)
        self.assertEqual(actual["started_groups"], 2)
        self.assertEqual(actual["complete_groups"], 1)
        self.assertEqual(actual["incomplete_groups"], 1)
        self.assertEqual(actual["recovery_fragments"], 3)
        self.assertEqual(actual["complete_pair_reference_first_notional"], 10000.)
        self.assertAlmostEqual(actual["complete_reference_gross"], 7.)
        self.assertAlmostEqual(actual["complete_slippage"], 1.)
        self.assertAlmostEqual(actual["complete_fees"], 1.)
        self.assertAlmostEqual(actual["complete_net"], 5.)
        self.assertAlmostEqual(actual["complete_net_bp"], 5.)
        self.assertAlmostEqual(actual["all_t_event_fees"], 1.9)
        self.assertAlmostEqual(actual["all_fragment_net"], 4.)
        pd.testing.assert_frame_equal(events, before)
        pd.testing.assert_frame_equal(trips, trip_before)

    def test_empty_or_only_pending_quality_is_unknown_not_perfect(self):
        blank = diagnosis.complete_notional_statistics(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(np.isnan(blank["complete_net_bp"]))
        events, _ = self.ledger()
        pending = diagnosis.complete_notional_statistics(events.loc[events.fragment_id.eq(0)], pd.DataFrame())
        self.assertEqual(pending["incomplete_groups"], 2)
        self.assertEqual(pending["complete_pair_reference_first_notional"], 0.)
        self.assertAlmostEqual(pending["all_t_event_fees"], .6)
        self.assertTrue(np.isnan(pending["complete_gross_bp"]))

    def test_complete_notional_rejects_double_slippage_and_bad_fragments(self):
        events, trips = self.ledger()
        with self.assertRaises(AssertionError):
            diagnosis.complete_notional_statistics(events, trips.assign(net_pnl=trips.net_pnl-trips.slippage_cost))
        with self.assertRaises(AssertionError):
            diagnosis.complete_notional_statistics(pd.concat([events, events.iloc[:1]]), trips)
        with self.assertRaises(AssertionError):
            diagnosis.complete_notional_statistics(events, trips.assign(shares=trips.shares*10))
        with self.assertRaises(AssertionError):
            diagnosis.complete_notional_statistics(events.assign(fee=events.fee+.1), trips)
        with self.assertRaises(AssertionError):
            diagnosis.complete_notional_statistics(events.iloc[0:0], trips)

    def test_source_full_file_and_snapshot_identity_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, source = root/"report", root/"project"
            report.mkdir(); source.mkdir()
            identities = {}
            for i in range(56):
                name = f"source_{i}.py"
                for p in (source/name, report/"source_snapshot"/name):
                    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(f"NUMBER={i}\n")
                identities[name] = diagnosis.sha(source/name)
            (report/"source_hashes.json").write_text(json.dumps(identities))
            previous = diagnosis.ROOT
            try:
                diagnosis.ROOT = source
                actual = diagnosis.verify_runtime_sources(report)
                self.assertEqual(len(actual), 113)
                (source/"source_0.py").write_text("CHANGED=True\n")
                with self.assertRaises(AssertionError): diagnosis.verify_runtime_sources(report)
            finally: diagnosis.ROOT = previous

    def test_source_notional_access_only_current_selected_numeric_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d = diagnosis.Diagnosis.__new__(diagnosis.Diagnosis)
            d.report, d.source, d.year, d.inner = root/"new", root/"old", 2023, 2026
            d.cfg, d.inputs = {"data_root":str(root/"raw")}, {}
            d.chosen = {"rule":{"candidate_index":17}}
            allowed = d.source/"year_2023/validation_candidates/candidate_017/test_trade_events.csv"
            allowed.parent.mkdir(parents=True); allowed.write_text("event_type\n")
            self.assertEqual(d.record(allowed), allowed)
            for path in (d.source/"year_2023/validation_candidates/candidate_016/test_trade_events.csv",
                    d.source/"year_2023/test_context/baseline/test_trade_events.csv",
                    d.source/"year_2024/validation_candidates/candidate_017/test_trade_events.csv"):
                with self.assertRaisesRegex(AssertionError, "禁止读取"): d.record(path)

    def three_horizons(self, missing=False):
        """三个完整当前键，评分随期限增加；事后恢复价与评分无强制一致性。"""
        result = {}
        for horizon in diagnosis.HORIZONS:
            raw = predictions()
            raw["prediction"] = [horizon*.001, -horizon*.001]
            recovery = [10.+(80-horizon)*.01, 9.8] if not missing else [np.nan, np.nan]
            result[horizon] = diagnosis.annotate_values(raw, [10., 10.], recovery,
                inventory(), configuration(), horizon)
        return result

    def test_trading_horizon_skips_lunch_preserves_invalid_late_clock(self):
        stamps = pd.Series(pd.to_datetime(["2026-03-11 11:00", "2026-03-11 14:00"]))
        expected = pd.Series(pd.to_datetime(["2026-03-11 13:11", "2026-03-11 14:41"]))
        pd.testing.assert_series_equal(diagnosis.fixed_end_times(stamps, 40), expected)
        late = diagnosis.fixed_end_times(stamps, 60)
        self.assertEqual(late.iloc[0], pd.Timestamp("2026-03-11 13:31"))
        self.assertTrue(pd.isna(late.iloc[1]))
        with self.assertRaises(AssertionError): diagnosis.fixed_end_times(stamps+pd.Timedelta(seconds=1), 40)
        raw = predictions().iloc[:1].copy()
        raw["datetime"] = stamps.iloc[1]
        annotated = diagnosis.annotate_values(raw, [10.], [10.2], inventory(), configuration(), 60)
        self.assertEqual(len(annotated), 1)
        self.assertFalse(annotated.endpoint_available.any())
        self.assertFalse(diagnosis.current_eligibility(annotated, 0., configuration()).any())

    def test_argmax_never_drops_missing_future_horizon_and_requires_all_current_keys(self):
        original = self.three_horizons()
        copies = {h:p.copy(deep=True) for h,p in original.items()}
        missing = self.three_horizons(missing=True)
        a = diagnosis.select_current_argmax(original, 0., configuration())
        b = diagnosis.select_current_argmax(missing, 0., configuration())
        columns = diagnosis.KEYS+["horizon_minutes", "action", "predicted_net_return"]
        pd.testing.assert_frame_equal(a[columns], b[columns])
        self.assertEqual(a.horizon_minutes.tolist(), [60, 60])
        self.assertFalse(b.endpoint_available.any())
        for h in original: pd.testing.assert_frame_equal(original[h], copies[h])
        missing[60] = missing[60].iloc[:1]
        with self.assertRaises(AssertionError): diagnosis.select_current_argmax(missing, 0., configuration())

    def test_argmax_tie_chooses_shorter_horizon_and_saved_request_is_exact(self):
        inputs = self.three_horizons()
        for h,p in inputs.items():
            p["prediction"], p["predicted_net_return"] = .02, .016
        selected = diagnosis.select_current_argmax(inputs, 0., configuration())
        self.assertTrue(selected.horizon_minutes.eq(20).all())
        saved = selected.rename(columns={"fixed_end":"planned_recovery_time", "predicted_net_return":"predicted_net_edge"})
        diagnosis.assert_saved_decisions(saved, selected)
        broken = saved.copy()
        broken.loc[0, "predicted_net_edge"] = np.nextafter(broken.loc[0, "predicted_net_edge"], np.inf)
        with self.assertRaises(AssertionError): diagnosis.assert_saved_decisions(broken, selected)
        with self.assertRaises(AssertionError): diagnosis.assert_saved_decisions(saved.iloc[:0], selected)

    def test_argmax_premium_uses_common_available_alternatives_without_changing_selection(self):
        inputs = self.three_horizons()
        inputs[20].loc[0, ["endpoint_available", "fixed_net_bp"]] = [False, np.nan]
        selected = diagnosis.select_current_argmax(inputs, 0., configuration())
        alternatives = pd.concat(inputs.values(), ignore_index=True)
        before = selected.copy(deep=True)
        result = diagnosis.argmax_premium(selected, alternatives)
        self.assertEqual(result["argmax_rows"], 2)
        self.assertEqual(result["missing_any_alternative_endpoint_rows"], 1)
        self.assertEqual(result["same_key_all_alternatives_available_rows"], 1)
        self.assertGreater(result["score_selection_premium_bp"], 0)
        self.assertTrue(result["no_future_filter_in_actual_argmax"])
        pd.testing.assert_frame_equal(selected, before)

    def test_original_selection_accepts_negative_active_but_excludes_execution_failure(self):
        rows = []
        for index in range(37):
            rows.append(dict(candidate_index=index, incremental_net_pnl=-100.-index if index else 0.,
                t_pair_count=int(index > 0), **{name:0 for name in diagnosis.FAIL}))
        metrics = pd.DataFrame(rows)
        metrics["execution_eligible"] = metrics.candidate_index.gt(0)
        metrics["selected"] = metrics.candidate_index.eq(1)
        diagnosis.validate_saved_selection(metrics, {"candidate_index":1})
        metrics.loc[1, "recovery_failure_days"] = 1
        with self.assertRaises(AssertionError): diagnosis.validate_saved_selection(metrics, {"candidate_index":1})
        metrics.loc[1, "execution_eligible"] = False
        metrics["selected"] = metrics.candidate_index.eq(2)
        diagnosis.validate_saved_selection(metrics, {"candidate_index":2})
        metrics.loc[metrics.candidate_index.gt(0), "recovery_failure_days"] = 1
        metrics["execution_eligible"] = False
        metrics["selected"] = metrics.candidate_index.eq(0)
        diagnosis.validate_saved_selection(metrics, {"candidate_index":0})

    def test_profile_mapping_preserves_l2_and_original_cost_denominator(self):
        rows = []
        for index in range(37):
            row = dict(candidate_index=index, enabled=index > 0, profile="hold" if index == 0 else "l2_depth4",
                min_net_edge=0., fraction=0. if index == 0 else .5, exit_mode="fixed")
            row.update({name:0. for name in diagnosis.METRIC_FIELDS})
            row["all_reference_acquisition_cost"] = 10000.
            rows.append(row)
        original = pd.DataFrame(rows)
        current = original.copy()
        current.loc[current.enabled, "profile"] = "l2_depth4"
        result = diagnosis.compare_candidates(current, original)
        self.assertTrue(result.same_forecast_target.all())
        self.assertEqual(result.loc[1, "current_profile"], "l2_depth4")
        self.assertEqual(result.loc[1, "source_profile"], "l2_depth4")
        current.loc[1, "profile"] = "l2_depth6"
        with self.assertRaises(AssertionError): diagnosis.compare_candidates(current, original)

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
        current.loc[current.enabled, "profile"] = "l2_depth4"
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

    def test_saved_namedtuple_metrics_row_with_hold_does_not_manufacture_pair(self):
        d = diagnosis.Diagnosis.__new__(diagnosis.Diagnosis)
        d.folder, d.year, d.inner = Path('/unused'), 2023, 2026
        d.cfg = {"lot_size": 100}
        values = {name: 0. for name in diagnosis.METRIC_FIELDS}
        values.update(candidate_index=0, profile='hold', all_reference_acquisition_cost=10000.)
        row = next(pd.DataFrame([values]).itertuples())
        d.json = lambda path: values
        d.csv = lambda path: pd.DataFrame()
        actual = d.account(row, {})
        self.assertEqual(actual['started_groups'], 0)
        self.assertEqual(actual['cost_denominator'], 10000.)
        self.assertEqual(actual['marked_increment'], 0.)


if __name__ == "__main__": unittest.main(verbosity=2)
