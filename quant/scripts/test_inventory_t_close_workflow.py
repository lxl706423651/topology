"""固定尾盘入口的来源和调度反例；只写临时文件，模型及账户均用替身，不读取市场收益。"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_inventory_close_t as runner
from scripts.test_inventory_t_close_models import make_config


def write_json(path, value):
    """在独立临时目录保存可按真实合同核验的JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str), encoding="utf-8")


def calculate_sha(path):
    """使用入口已有字节摘要算法，但不替换任何来源验证函数。"""
    return runner.calculate_file_sha256(Path(path))


def create_current_frame():
    """四年每年两股的当前键及固定端点；小表只用于调度，不形成实际收益评估。"""
    rows = []
    for year in range(2023, 2027):
        for symbol in ("SH.600001", "SZ.000001"):
            stamp = pd.Timestamp(f"{year}-03-11 09:35")
            row = {name: 1. for name in runner.INTRADAY_COLUMNS}
            row.update(datetime=stamp, symbol=symbol, date=stamp.normalize(),
                trade_week=stamp.normalize()-pd.Timedelta(days=stamp.weekday()),
                execution_time=stamp+pd.Timedelta(minutes=1), execution_price=10.,
                recovery_time=stamp.normalize()+pd.Timedelta(hours=14, minutes=55), recovery_price=10.1,
                feature_source_start=stamp-pd.Timedelta(days=10), f_signal=np.float32(.2),
                cohort_target_year=year, baseline_opening_shares=1000,
                signal_price=10., signal_high=10.1, signal_low=9.9, previous_close=10.)
            rows.append(row)
    return pd.DataFrame(rows)


class LinearCurrentPredictor:
    """可序列化的确定性当前特征替身，不拟合或读取标签。"""
    def predict(self, matrix):
        return matrix.f_signal.to_numpy(dtype=float)*.01


class NativeCurrentPredictor(LinearCurrentPredictor):
    """原生接口替身，专门让异常数组合同可被测试。"""
    def feature_name(self):
        return ["f_signal"]


class CloseWorkflowTests(unittest.TestCase):
    """覆盖错误来源、无未来推断、统一冻结和未达标不外评等跨模块边界。"""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/"v15"
        self.cohort_source = self.root/"v14"
        self.source.mkdir()
        self.original = yaml.safe_load((ROOT/"config/inventory_learning_t_v14.yaml").read_text())
        self.original.update(research_version="inventory_guarded_t_v15", configuration_source="/frozen/v15.yaml",
            source_learning_report_dir=str(self.cohort_source), source_learning_version="inventory_learning_t_v14",
            test_years=[2023, 2026], internal_only=False, report_scope="smoke", max_symbols=12,
            smoke_days=15, workers=1, candidate_workers=1, output_dir="/frozen/output",
            t_recheck_exit_side_at_execution=True)
        write_json(self.source/"resolved_config.json", self.original)
        self.cohort_audit = self.cohort_source/"independent_learning_audit/run_01/verification.json"
        write_json(self.cohort_audit, dict(research_version="inventory_learning_t_v14", passed=True,
            complete=True, all_read_inputs_unchanged=True))
        self.declared = dict(research_version=runner.VERSION, source_execution_report_dir=str(self.source),
            forecast_target="same_day_1455", recovery_key="close", t_recheck_exit_side_at_execution=True,
            output_dir=str(self.root/"output"), workers=1, candidate_workers=1)
        self.config_path = self.root/"config.yaml"

    def load(self, *args, reconstructed=None):
        """只替代旧入口的配置重建，保留新入口的完整比较和参数验证。"""
        self.config_path.write_text(yaml.safe_dump(self.declared))
        with patch.object(runner, "load_guarded_configuration", return_value=copy.deepcopy(
                self.original if reconstructed is None else reconstructed)):
            return runner.load_configuration(["--config", str(self.config_path), *args])

    def test_configuration_inherits_cost_scope_and_pins_complete_core_audit(self):
        cfg = self.load("--test-years", "2026", "--candidate-workers", "2")
        for name in ("commission_rate", "slippage_rate", "transfer_rate", "min_commission", "base_fraction",
                     "profit_target_rate", "max_symbols", "smoke_days", "report_scope", "signal_times",
                     "weekly_return_weight", "weekly_amplitude_weight", "learning_model_profiles"):
            self.assertEqual(cfg[name], self.original[name], name)
        self.assertEqual(cfg["source_learning_audit_sha256"], calculate_sha(self.cohort_audit))
        self.assertEqual(cfg["source_prediction_horizons"], [20, 40, 60])
        self.assertNotIn("horizons", cfg)
        self.assertEqual(cfg["scheduled_recovery_times"], ["14:55"])
        self.assertFalse(cfg["frozen_t_models_reused"])
        self.assertEqual(len(runner.build_inventory_close_candidates(cfg)), 37)

    def test_configuration_rejects_new_fee_wrong_source_and_year_expansion(self):
        self.declared["commission_rate"] = 0
        with self.assertRaises(ValueError): self.load()
        self.declared.pop("commission_rate")
        changed = {**self.original, "slippage_rate": 0.}
        with self.assertRaisesRegex(ValueError, "精确复现"): self.load(reconstructed=changed)
        for args in (("--test-years", "2025"), ("--test-years", "2026", "2026")):
            with self.subTest(args=args), self.assertRaises(ValueError): self.load(*args)
        write_json(self.source/"resolved_config.json", {**self.original, "research_version": "wrong"})
        with self.assertRaisesRegex(ValueError, "v15"): self.load()

    def test_incomplete_or_wrong_version_audit_cannot_replace_passed_anchor(self):
        folder = self.cohort_audit.parent.parent
        for index, delta in enumerate(({"complete": False}, {"all_read_inputs_unchanged": False},
                                      {"research_version": "inventory_learning_t_v99"})):
            write_json(folder/f"run_9{index}/verification.json", {**dict(research_version="inventory_learning_t_v14",
                passed=True, complete=True, all_read_inputs_unchanged=True), **delta})
        path, _ = runner.find_complete_source_audit(self.cohort_source, "learning", "inventory_learning_t_v14")
        self.assertEqual(path, self.cohort_audit)
        self.cohort_audit.unlink()
        with self.assertRaises(ValueError): self.load()

    def create_source_anchor(self):
        """小型独审输入清单绑定源配置、冻结权重及当前/快照源码。"""
        old_model = self.root/"old_model.joblib"
        old_model.write_bytes(b"frozen old model, never used for new prediction")
        source_name = "scripts/run_inventory_close_t.py"
        source_sha = calculate_sha(ROOT/source_name)
        snapshot = self.source/"source_snapshot"/source_name
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes((ROOT/source_name).read_bytes())
        write_json(self.source/"runtime_integrity_audit.json", {"all_unchanged": True})
        write_json(self.source/"source_hashes.json", {source_name: source_sha})
        write_json(self.source/"input_sources.json", {})
        write_json(self.source/"research_protocol.json", {"version": "inventory_guarded_t_v15"})
        write_json(self.source/"all_rules_frozen_before_outer_test.json",
            {"model_artifacts_sha256": {str(old_model): calculate_sha(old_model)}})
        anchors = {str(path.resolve()): calculate_sha(path) for path in self.source.glob("*.json")}
        audit = self.source/"independent_guarded_audit/run_01/verification.json"
        write_json(audit, dict(research_version="inventory_guarded_t_v15", passed=True, complete=True,
            all_read_inputs_unchanged=True, inputs_sha256=anchors))
        return old_model, audit

    def test_source_manifest_self_consistent_rewrite_still_fails_independent_anchor(self):
        old_model, _ = self.create_source_anchor()
        cfg = {"source_execution_report_dir": str(self.source)}
        (self.root/"accepted").mkdir()
        runner.validate_execution_source(cfg, self.root/"accepted", {})
        old_model.write_bytes(b"different old model")
        write_json(self.source/"all_rules_frozen_before_outer_test.json",
            {"model_artifacts_sha256": {str(old_model): calculate_sha(old_model)}})
        with self.assertRaisesRegex(ValueError, "已审计来源"):
            runner.validate_execution_source(cfg, self.root/"rejected", {})

    def test_core_labels_follow_frozen_keys_and_keep_missing_endpoint(self):
        frame = create_current_frame()
        core = frame.loc[frame.cohort_target_year.isin([2023, 2024])].iloc[[0, 2]].copy()
        frame.loc[frame.cohort_target_year.eq(2026), "recovery_time"] = pd.Timestamp("2099-01-01")
        frame.loc[core.index[0], "recovery_price"] = np.nan
        before = frame.copy(deep=True)
        labels, audit = runner.extract_core_labels(frame, core, make_config())
        self.assertEqual(len(labels), len(core))
        self.assertEqual(list(zip(labels.datetime, labels.symbol)), list(zip(core.datetime, core.symbol)))
        self.assertEqual(labels.target_return.isna().sum(), 1)
        self.assertTrue(audit["only_frozen_core_keys_read"])
        pd.testing.assert_frame_equal(frame, before)

    def create_prediction_fixture(self):
        """不用拟合，保存有完整文件/折身份的可重载当前特征模型替身。"""
        model_dir = self.root/"year_2026/models/close_l2_depth4"
        model_dir.mkdir(parents=True)
        (model_dir/"return_model.txt").write_text("synthetic native")
        metadata = dict(research_version=runner.VERSION, profile="l2_depth4", features=["f_signal"],
            train_years=[2023, 2024], excluded_years=[2025, 2026],
            native_model_sha256=calculate_sha(model_dir/"return_model.txt"), **runner.IDENTITY)
        bundle = {"model": LinearCurrentPredictor(), "features": ["f_signal"], "metadata": metadata}
        joblib.dump(bundle, model_dir/"model_bundle.joblib")
        write_json(model_dir/"model_metadata.json", {**metadata,
            "model_bundle_sha256": calculate_sha(model_dir/"model_bundle.joblib")})
        write_json(model_dir.parent.parent/"split.json", dict(core_years=[2023, 2024], test_year=2026, inner_year=2025))
        return model_dir, create_current_frame().iloc[-2:].copy()

    def test_prediction_removes_all_future_fields_and_archives_current_features(self):
        model, frame = self.create_prediction_fixture()
        frame["target_return"] = np.nan
        original = frame.copy(deep=True)
        with patch.object(runner.lightgbm, "Booster", return_value=NativeCurrentPredictor()):
            actual = runner.predict_and_audit(frame, model, make_config(), self.root/"predictions", {})
        self.assertEqual(len(actual), len(frame))
        self.assertNotIn("horizon_minutes", actual)
        self.assertTrue(actual.recovery_key.eq("close").all())
        saved = pd.read_csv(self.root/"predictions/current_predictions.csv")
        self.assertIn("f_signal", saved)
        self.assertNotIn("target_return", saved)
        audit = json.loads((self.root/"predictions/prediction_audit.json").read_text())
        self.assertTrue(audit["future_independent"])
        self.assertIn("recovery_price", audit["future_fields_removed"])
        pd.testing.assert_frame_equal(frame, original)

    def test_nonfinite_or_wrong_shape_native_prediction_never_passes_nan_comparison(self):
        model, frame = self.create_prediction_fixture()
        for index, values in enumerate((np.full(2, np.nan), np.full(2, np.inf), np.zeros((2, 1)))):
            native = NativeCurrentPredictor()
            with patch.object(native, "predict", return_value=values), \
                    patch.object(runner.lightgbm, "Booster", return_value=native), \
                    self.subTest(index=index), self.assertRaisesRegex(AssertionError, "全为有限数"):
                runner.predict_and_audit(frame, model, make_config(), self.root/f"invalid_{index}", {})

    def test_wrong_fold_and_model_metadata_or_weight_identity_rejected_before_prediction(self):
        model, frame = self.create_prediction_fixture()
        split_path = model.parent.parent/"split.json"
        split = json.loads(split_path.read_text())
        write_json(split_path, {**split, "core_years": [2023, 2025], "inner_year": 2024})
        with self.assertRaisesRegex(AssertionError, "当前折"):
            runner.predict_and_audit(frame, model, make_config(), self.root/"wrong_fold", {})
        write_json(split_path, split)
        metadata_path = model/"model_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        write_json(metadata_path, {**metadata, "features": ["f_unrelated"]})
        with self.assertRaisesRegex(AssertionError, "元数据不一致"):
            runner.predict_and_audit(frame, model, make_config(), self.root/"wrong_metadata", {})
        write_json(metadata_path, metadata)
        (model/"return_model.txt").write_text("replaced weights")
        with self.assertRaisesRegex(AssertionError, "文件身份"):
            runner.predict_and_audit(frame, model, make_config(), self.root/"wrong_weights", {})

    @staticmethod
    def candidate_rows(rules, rate=.06):
        """固定合成指标让真实选参函数运行；这些数值不代表策略回测结果。"""
        rows = []
        for rule in rules:
            active = bool(rule["enabled"])
            net = (rate*10000 if rule["candidate_index"] == 1 else -200.) if active else 0.
            rows.append({**rule, "incremental_net_pnl": net, "holding_cost_reduction_rate": net/10000,
                "t_pair_count": int(active), "completed_t_pair_lots": 2*int(active),
                "t_fee": 1.*active, "t_slippage_cost": 2.*active, "inventory_mismatch_days": 0,
                "recovery_failure_days": 0, "pending_pairs": 0, "t1_violation_count": 0,
                "base_plan_mismatch_count": 0, "all_reference_acquisition_cost": 10000.})
        return pd.DataFrame(rows)

    def test_same_numeric_candidate_comparison_allows_new_prediction_but_not_new_cost_or_rule(self):
        cfg = self.load()
        old = self.candidate_rows(runner.build_inventory_close_candidates(cfg))
        source_path = self.root/"old_metrics.csv"
        old.to_csv(source_path, index=False)
        current = old.copy()
        current.loc[1, "incremental_net_pnl"] = -100.
        result = runner.write_source_comparison(current, source_path, self.root, {})
        self.assertTrue(result.same_forecasts.eq(False).all())
        self.assertEqual(result.loc[1, "incremental_net_pnl_change"], -700.)
        for name, value in (("all_reference_acquisition_cost", 10001.), ("fraction", .3)):
            bad = current.copy()
            bad.loc[1, name] = value
            with self.subTest(name=name), self.assertRaises(AssertionError):
                runner.write_source_comparison(bad, source_path, self.root, {})

    def execute_mock_workflow(self, scope="formal", rates=None, tamper_model=False, internal_only=False,
                              missing_source_outer=False):
        """运行真实主调度和选参，替代大数据、拟合、预测及账户；核顺序而不跑实验。"""
        cfg = self.load()
        cfg.update(report_scope=scope, internal_only=internal_only, outer_evaluation_policy="all_inner_5_percent",
            source_audit_report_dir=str(self.root/"market_source"), market_cache_dir=str(self.root/"market"))
        frame = create_current_frame()
        daily = pd.DataFrame({"date": frame.date.unique()})
        known = {}
        for name in ("weekly", "intraday", "daily"):
            path = Path(cfg["market_cache_dir"])/f"{name}.pkl"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
            known[str(path.resolve())] = {"sha256": calculate_sha(path)}
        write_json(Path(cfg["source_audit_report_dir"])/"input_sources.json", known)
        folds = {2023: dict(test_year=2023, inner_year=2026, core_years=[2024, 2025]),
                 2026: dict(test_year=2026, inner_year=2025, core_years=[2023, 2024])}
        rules = runner.build_inventory_close_candidates(cfg)
        rates = rates or {2023: .06, 2026: .06}
        for year in folds:
            destination = self.source/f"year_{year}/validation_rule_metrics.csv"
            destination.parent.mkdir(parents=True)
            self.candidate_rows(rules, .06).to_csv(destination, index=False)
        events, directories = [], []
        real_write = runner.write_json

        def snapshot(directory, unused):
            directories.append(directory)
            real_write(directory/"source_hashes.json", {})

        def cohort(intraday, fold, local, source, destination, inputs, recorder):
            events.append(("cohort", fold["test_year"]))
            current = intraday.loc[intraday.cohort_target_year.isin(fold["core_years"])].copy()
            current = current.drop(columns=[name for name in current if name.startswith(("execution_", "recovery_", "target_", "label_"))])
            self.assertEqual(len(current), 4)
            return current, {"passed": True}

        def fit(training, years, local, model, excluded):
            self.assertEqual(set(training.datetime.dt.year), set(years))
            self.assertFalse(set(years) & set(excluded))
            self.assertEqual(len(training), 4)
            self.assertTrue(training.recovery_clock.eq("14:55").all())
            year = int(model.parent.parent.name.split("_")[-1])
            self.assertIn(("cohort", year), events)
            self.assertFalse(any(e[0] == "test" for e in events))
            events.append(("fit", year, local["t_model_profile"]))
            model.mkdir(parents=True)
            (model/"model_bundle.joblib").write_text(f"{year}:{local['t_model_profile']}")
            return {"fitted_rows": len(training), "features": ["f_signal"]}

        def context(fold, mode, prices, signals, calendar, local, out, inputs, store):
            year = fold["test_year"]
            if mode == "test":
                marker = out.parent.parent/"all_rules_frozen_before_outer_test.json"
                self.assertTrue(marker.exists())
                frozen = json.loads(marker.read_text())
                self.assertEqual(set(frozen["selected_rule_sha256"]), {"2023", "2026"})
                self.assertEqual(len([e for e in events if e[0] == "fit"]), 4)
                self.assertEqual(len([e for e in events if e[0] == "validation"]), 2)
                self.assertTrue(all(not Path(name).is_absolute() for name in frozen["model_artifacts_sha256"]))
            events.append((mode, year))
            selected = pd.DataFrame({"symbol": ["SH.600001"], "shares": [1000]})
            baseline = pd.DataFrame({"datetime": [f"{fold['inner_year'] if mode=='validation' else year}-03-11"], "equity": [1e6]})
            for relative, table in (("frozen_base_selections.csv", selected), ("frozen_base_entries.csv", selected),
                                    ("baseline/test_equity_curve.csv", baseline)):
                destination = out/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(destination, index=False)
                if not (mode == "test" and missing_source_outer):
                    old_path = self.source/f"year_{year}/{mode}_context"/relative
                    old_path.parent.mkdir(parents=True, exist_ok=True)
                    table.to_csv(old_path, index=False)
            signals = frame.loc[frame.datetime.dt.year.eq(fold["inner_year"] if mode == "validation" else year)].copy()
            return selected, prices, signals, local, object()

        def predict(signals, out, local, destination, inputs, profiles=None):
            expected = cfg["learning_model_profiles"] if profiles is None else profiles
            self.assertEqual(len(expected), 2 if "validation" in str(destination) else 1)
            for profile in expected:
                self.assertTrue((out/f"models/close_{profile}/model_bundle.joblib").exists())
            self.assertEqual(local["commission_rate"], self.original["commission_rate"])
            self.assertEqual(local["slippage_rate"], self.original["slippage_rate"])
            return {profile: signals.copy() for profile in expected}

        def evaluate(rule):
            year = runner._WORKER["cfg"]["current_test_year"]
            return self.candidate_rows([rule], rates[year]).iloc[0].to_dict()

        def intercept_write(path, value):
            real_write(path, value)
            if tamper_model and Path(path).name == "all_rules_frozen_before_outer_test.json":
                name = sorted(value["model_artifacts_sha256"])[-1]
                (Path(path).parent/name).write_text("modified after freezing last fold")

        patches = dict(load_configuration=lambda _: cfg, snapshot_sources=snapshot,
            validate_execution_source=lambda *args: None,
            validate_balanced_sources=lambda *args: {"old_t_models_reused": True},
            read_balanced_fold=lambda cfg, year, inputs: copy.deepcopy(folds[year]),
            load_frozen_market_data=lambda *args: (pd.DataFrame(), frame, daily),
            add_timing_features=lambda frame, cfg: frame.copy(),
            load_frozen_inventory_core_cohort=cohort, fit_inventory_close_model=fit,
            prepare_account_context=context, get_store=lambda *args: object(), predict_account_models=predict,
            evaluate_worker=evaluate, write_json=intercept_write,
            generate_inventory_close_decisions=lambda *args: (pd.DataFrame(), {"passed": True}),
            run_guarded_portfolio=lambda *args, **kwargs: {}, save_guarded_result=lambda *args: {},
            write_close_inventory_report=lambda out, *args: (out/"final_report.md").write_text("工程替身"))
        with ExitStack() as stack:
            for name, replacement in patches.items():
                stack.enter_context(patch.object(runner, name, side_effect=replacement))
            try:
                runner.main([])
            finally:
                self.workflow_events = events
                self.workflow_directory = directories[0] if directories else None
        return directories[0], events

    def test_all_folds_two_core_models_and_rules_freeze_before_any_outer_without_refit(self):
        directory, events = self.execute_mock_workflow(scope="smoke", rates={2023: -.01, 2026: -.01})
        self.assertEqual(len([e for e in events if e[0] == "fit"]), 4)
        self.assertEqual(len([e for e in events if e[0] == "test"]), 2)
        protocol = json.loads((directory/"research_protocol.json").read_text())
        self.assertFalse(protocol["cross_horizon_argmax"])
        self.assertFalse(protocol["outer_model_refit"])
        self.assertTrue(protocol["full_core_training_also_in_smoke"])
        marker = json.loads((directory/"all_rules_frozen_before_outer_test.json").read_text())
        for name, digest in marker["model_artifacts_sha256"].items():
            self.assertEqual(calculate_sha(directory/name), digest)
        decision = json.loads((directory/"outer_evaluation_decision.json").read_text())
        self.assertFalse(decision["all_inner_cost_targets_met"])
        self.assertTrue(decision["smoke_permits_engineering_outer_without_profit_claim"])

    def test_one_inner_fold_below_target_prevents_all_formal_outer_accounts(self):
        directory, events = self.execute_mock_workflow(rates={2023: .06, 2026: -.01})
        self.assertFalse(any(e[0] == "test" for e in events))
        goal = json.loads((directory/"goal_evaluation.json").read_text())
        self.assertFalse(goal["formal_outer_evaluated"])
        self.assertFalse(goal["goal_achieved"])
        choice = json.loads((directory/"year_2026/selected_rule.json").read_text())
        self.assertTrue(choice["rule"]["enabled"])
        self.assertLess(choice["inner_metrics"]["incremental_net_pnl"], 0)

    def test_last_fold_model_tampering_blocks_first_outer_even_in_smoke(self):
        with self.assertRaisesRegex(AssertionError, "外评前改变"):
            self.execute_mock_workflow(scope="smoke", tamper_model=True)
        self.assertFalse(any(e[0] == "test" for e in self.workflow_events))

    def test_smoke_requires_existing_source_outer_baseline_and_internal_only_overrides_pass(self):
        with self.assertRaisesRegex(AssertionError, "已审计来源底仓"):
            self.execute_mock_workflow(scope="smoke", missing_source_outer=True)

    def test_internal_only_never_runs_outer_even_when_all_internal_targets_met(self):
        directory, events = self.execute_mock_workflow(internal_only=True)
        self.assertFalse(any(e[0] == "test" for e in events))
        decision = json.loads((directory/"outer_evaluation_decision.json").read_text())
        self.assertTrue(decision["all_inner_cost_targets_met"])
        self.assertFalse(decision["outer_evaluation_started"])

    def test_account_candidate_must_reproduce_exact_baseline(self):
        rule = runner.build_inventory_close_candidates(self.load())[1]
        baseline = pd.DataFrame({"datetime": pd.to_datetime(["2026-03-11"]), "equity": [1e6]})
        runner.initialize_worker(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, self.load(), None,
            baseline, self.root/"candidates")
        changed = {"equity_curve": baseline.rename(columns={"equity": "baseline_equity"})}
        changed["equity_curve"].loc[0, "baseline_equity"] -= .01
        with patch.object(runner, "generate_inventory_close_decisions", return_value=(pd.DataFrame(), {})), \
                patch.object(runner, "run_guarded_portfolio", return_value=changed), \
                self.assertRaisesRegex(AssertionError, "基准发生改变"):
            runner.evaluate_worker(rule)


if __name__ == "__main__":
    unittest.main(verbosity=2)
