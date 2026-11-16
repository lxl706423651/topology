"""Huber入口的源输入、损失映射和统一冻结反例；仅临时小表/接口替身，不运行账户。"""
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts import run_inventory_robust_t as runner
from scripts.test_inventory_t_robust_models import make_config
from scripts.test_inventory_t_close_workflow import create_current_frame, write_json


class CurrentOnlyModel:
    """只读取当前f_signal的可序列化接口替身，无拟合/未来价格依赖。"""
    def __init__(self):
        self.params = {"objective": "huber", "alpha": .01}
    @property
    def booster_(self): return self
    def get_params(self): return self.params.copy()
    def feature_name(self): return ["f_signal"]
    def predict(self, matrix): return matrix.f_signal.to_numpy(dtype=float)*.01


class RobustWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.cohort = self.root/"v15", self.root/"v14"
        self.source.mkdir()
        self.original = yaml.safe_load((ROOT/"config/inventory_learning_t_v14.yaml").read_text())
        self.original.update(research_version="inventory_guarded_t_v15", configuration_source="/frozen/v15.yaml",
            source_learning_report_dir=str(self.cohort), source_learning_version="inventory_learning_t_v14",
            test_years=[2023, 2026], internal_only=False, report_scope="smoke", max_symbols=12, smoke_days=15,
            workers=1, candidate_workers=1, output_dir="/original/output", t_recheck_exit_side_at_execution=True,
            scheduled_recovery_times=["10:16", "10:36", "10:56", "14:55"])
        write_json(self.source/"resolved_config.json", self.original)
        self.audit = self.cohort/"independent_learning_audit/run_01/verification.json"
        write_json(self.audit, dict(research_version="inventory_learning_t_v14", passed=True, complete=True,
            all_read_inputs_unchanged=True))
        self.declared = dict(research_version=runner.VERSION, source_execution_report_dir=str(self.source),
            training_loss="huber", t_huber_alpha=.01, t_recheck_exit_side_at_execution=True,
            output_dir=str(self.root/"output"), workers=1, candidate_workers=1)
        self.config = self.root/"robust.yaml"

    def load(self, *args, old=None):
        """仅替代旧入口重建，真实运行新配置合同检查。"""
        self.config.write_text(yaml.safe_dump(self.declared))
        with patch.object(runner, "load_guarded_configuration", return_value=copy.deepcopy(self.original if old is None else old)):
            return runner.load_configuration(["--config", str(self.config), *args])

    def test_original_horizons_schedules_fees_scope_and_audit_are_exact(self):
        cfg = self.load()
        for name in ("horizons", "scheduled_recovery_times", "signal_times", "commission_rate", "slippage_rate",
                     "base_fraction", "profit_target_rate", "report_scope", "smoke_days", "max_symbols"):
            self.assertEqual(cfg[name], self.original[name])
        self.assertEqual(cfg["source_learning_model_profiles"], ["l2_depth4", "l2_depth6"])
        self.assertEqual(cfg["learning_model_profiles"], ["huber_depth4", "huber_depth6"])
        self.assertEqual(cfg["source_learning_audit_sha256"], runner.calculate_file_sha256(self.audit))
        self.assertEqual(cfg["forecast_target"], "fixed_20_40_60_reference_return")

    def test_loss_declaration_cannot_change_fee_horizon_or_source_scope(self):
        for key, value in (("commission_rate", 0.), ("horizons", [20]), ("t_huber_alpha", .02)):
            original = self.declared.copy()
            self.declared[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.load()
            self.declared = original
        with self.assertRaisesRegex(ValueError, "精确复现"):
            self.load(old={**self.original, "scheduled_recovery_times": ["14:55"]})
        for years in (("2025",), ("2026", "2026")):
            with self.assertRaises(ValueError): self.load("--test-years", *years)

    def test_partial_source_audit_cannot_authorize_core_reuse(self):
        value = json.loads(self.audit.read_text())
        write_json(self.audit, {**value, "complete": False})
        with self.assertRaisesRegex(ValueError, "尚未完成"): self.load()

    def training_fixture(self):
        """两个实际小型输入目录，不拟合；身份清单放模型目录之外。"""
        new, old = self.root/"new_model", self.root/"old_model"
        new.mkdir(); old.mkdir()
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2023-03-01 09:35"]), "symbol": ["SH.600001"],
            "f_x": np.array([.25], dtype="float32")})
        for directory in (new, old):
            frame.to_pickle(directory/"training_inputs.pkl")
            for name in ("training_labels.csv", "feature_manifest.csv", "training_split_audit.csv"):
                pd.DataFrame({"value": [.125], "fitted": [True]}).to_csv(directory/name, index=False)
        common = dict(train_years=[2023, 2024], excluded_years=[2025, 2026], features=["f_x"], fitted_rows=1,
            training_matrix_sha256="a"*64, training_target_sha256="b"*64, horizon_minutes=20)
        a = {**common, "profile": "huber_depth4", "model_parameters": {"objective": "huber", "alpha": .01, "n_jobs": 2, "n_estimators": 300}}
        b = {**common, "profile": "l2_depth4", "model_parameters": {"objective": "regression", "n_jobs": 12, "n_estimators": 300}}
        write_json(new/"model_metadata.json", a); write_json(old/"model_metadata.json", b)
        return new, old, a, b

    def test_training_identity_compares_four_artifacts_and_allows_only_declared_loss_threads(self):
        new, old, _, _ = self.training_fixture()
        target = self.root/"training_identity.json"
        runner.compare_training_to_source(new, old, target, {})
        saved = json.loads(target.read_text())
        self.assertTrue(saved["passed"])
        self.assertEqual(len(saved["matrices_labels_features_and_split_exact"]), 4)
        self.assertEqual(saved["source_workers"], 12)
        self.assertEqual(saved["current_workers"], 2)
        self.assertFalse((new/target.name).exists())

    def test_dtype_change_and_same_metadata_do_not_hide_actual_matrix_change(self):
        new, old, _, _ = self.training_fixture()
        frame = pd.read_pickle(new/"training_inputs.pkl")
        frame["f_x"] = frame.f_x.astype("float64")
        frame.to_pickle(new/"training_inputs.pkl")
        with self.assertRaises(AssertionError): runner.compare_training_to_source(new, old, self.root/"proof.json", {})

    def test_wrong_profile_horizon_target_hash_or_training_parameter_rejected(self):
        new, old, metadata, _ = self.training_fixture()
        for key, value in (("profile", "huber_depth6"), ("horizon_minutes", 40), ("training_target_sha256", "x"*64),
                           ("train_years", [2023, 2025]), ("features", ["f_other"]),
                           ("model_parameters", {**metadata["model_parameters"], "n_estimators": 301})):
            write_json(new/"model_metadata.json", {**metadata, key: value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                runner.compare_training_to_source(new, old, self.root/"proof.json", {})

    def prediction_fixture(self):
        """完整模型/折摘要和当前预测替身，原生接口单独替换以测试异常拒绝。"""
        directory = self.root/"year_2026/models/h20_huber_depth4"
        directory.mkdir(parents=True)
        (directory/"return_model.txt").write_text("current-only fake native")
        meta = dict(research_version=runner.VERSION, profile="huber_depth4", source_l2_profile="l2_depth4",
            horizon_minutes=20, features=["f_signal"], training_loss="huber", huber_alpha=.01,
            train_years=[2023, 2024], excluded_years=[2025, 2026], model_parameters={"objective": "huber", "alpha": .01},
            native_model_sha256=runner.calculate_file_sha256(directory/"return_model.txt"))
        bundle = {"model": CurrentOnlyModel(), "features": ["f_signal"], "metadata": meta}
        joblib.dump(bundle, directory/"model_bundle.joblib")
        write_json(directory/"model_metadata.json", {**meta,
            "model_bundle_sha256": runner.calculate_file_sha256(directory/"model_bundle.joblib")})
        write_json(directory.parent.parent/"split.json", dict(core_years=[2023, 2024], test_year=2026, inner_year=2025))
        return directory, create_current_frame().iloc[-2:].copy()

    def test_future_scrub_preserves_all_keys_and_archives_huber_score_identity(self):
        directory, frame = self.prediction_fixture()
        before = frame.copy(deep=True)
        with patch.object(runner.lightgbm, "Booster", return_value=CurrentOnlyModel()):
            actual = runner.predict_and_audit(frame, directory, make_config(), self.root/"predictions", {})
        self.assertEqual(len(actual), len(frame))
        self.assertTrue(actual.forecast_loss.eq("huber").all())
        self.assertTrue(actual.model_version.eq(runner.VERSION).all())
        np.testing.assert_array_equal(actual.return_score, actual.prediction)
        self.assertTrue(json.loads((self.root/"predictions/prediction_audit.json").read_text())["future_independent"])
        pd.testing.assert_frame_equal(frame, before)

    def test_native_nan_shape_and_wrong_fold_cannot_pass_prediction_audit(self):
        directory, frame = self.prediction_fixture()
        native = CurrentOnlyModel()
        for number, values in enumerate((np.full(2, np.nan), np.zeros((2, 1)))):
            with patch.object(native, "predict", return_value=values), patch.object(runner.lightgbm, "Booster", return_value=native), \
                    self.assertRaisesRegex(AssertionError, "全为有限数"):
                runner.predict_and_audit(frame, directory, make_config(), self.root/f"invalid_{number}", {})
        write_json(directory.parent.parent/"split.json", dict(core_years=[2023, 2025], test_year=2026, inner_year=2024))
        with self.assertRaisesRegex(AssertionError, "当前折"):
            runner.predict_and_audit(frame, directory, make_config(), self.root/"invalid_fold", {})

    @staticmethod
    def rows(rules, rate=.06):
        """真实选参函数只消费合成账户指标，不运行实际组合。"""
        rows = []
        for rule in rules:
            active = rule["enabled"]
            net = (10000*rate if rule["candidate_index"] == 1 else -200.) if active else 0.
            rows.append({**rule, "incremental_net_pnl": net, "holding_cost_reduction_rate": net/10000,
                "t_pair_count": int(active), "completed_t_pair_lots": 2*int(active), "t_fee": 1.*active,
                "t_slippage_cost": 2.*active, "inventory_mismatch_days": 0, "recovery_failure_days": 0,
                "pending_pairs": 0, "t1_violation_count": 0, "base_plan_mismatch_count": 0,
                "all_reference_acquisition_cost": 10000.})
        return pd.DataFrame(rows)

    def test_numeric_candidate_mapping_preserves_economics_without_claiming_equal_predictions(self):
        cfg = self.load(); current = self.rows(runner.build_inventory_robust_candidates(cfg))
        original = current.copy()
        original["profile"] = original.profile.map({**runner.SOURCE_PROFILE_MAP, "hold": "hold"})
        path = self.root/"old_metrics.csv"; original.to_csv(path, index=False)
        actual = runner.write_source_comparison(current, path, self.root, {})
        self.assertTrue(actual.same_forecast_target.all())
        self.assertFalse(actual.same_forecasts.any())
        self.assertEqual(actual.loc[1, "source_profile"], "l2_depth4")
        current.loc[1, "fraction"] = .7
        with self.assertRaises(AssertionError): runner.write_source_comparison(current, path, self.root, {})

    def run_workflow(self, *, rate=-.01, scope="formal", tamper=False):
        """运行主调度；全core/标签/6模型/37候选用小型替身追踪，不拟合账户。"""
        cfg = self.load()
        cfg.update(report_scope=scope, outer_evaluation_policy="all_inner_5_percent",
            source_audit_report_dir=str(self.root/"old_market"), market_cache_dir=str(self.root/"cache"),
            source_training_report_dir=str(self.root/"label_proof"))
        known = {}
        for name in ("weekly", "intraday", "daily"):
            path = Path(cfg["market_cache_dir"])/f"{name}.pkl"
            path.parent.mkdir(exist_ok=True); path.write_bytes(name.encode())
            known[str(path.resolve())] = {"sha256": runner.calculate_file_sha256(path)}
        write_json(Path(cfg["source_audit_report_dir"])/"input_sources.json", known)
        write_json(Path(cfg["source_training_report_dir"])/"independent_residual_audit.json", {"passed": True})
        frame = create_current_frame(); daily = pd.DataFrame({"date": frame.date.unique()})
        folds = {2023: dict(test_year=2023, inner_year=2026, core_years=[2024, 2025]),
                 2026: dict(test_year=2026, inner_year=2025, core_years=[2023, 2024])}
        rules = runner.build_inventory_robust_candidates(cfg)
        for year in folds:
            original = self.rows(rules)
            original["profile"] = original.profile.map({**runner.SOURCE_PROFILE_MAP, "hold": "hold"})
            path = self.source/f"year_{year}/validation_rule_metrics.csv"
            path.parent.mkdir(); original.to_csv(path, index=False)
        events, directories = [], []
        original_write = runner.write_json

        def snapshot(out, cfg):
            directories.append(out); original_write(out/"source_hashes.json", {})

        def cohort(intraday, fold, cfg, source, out, inputs, recorder):
            events.append(("core", fold["test_year"]))
            core = frame.loc[frame.cohort_target_year.isin(fold["core_years"])].copy()
            core = core.drop(columns=[n for n in core if n.startswith(("execution_", "recovery_", "target_", "label_"))])
            return core, {"passed": True}

        def attach(core, labels, horizon):
            self.assertEqual(len(core), 4)
            self.assertFalse(any(n.startswith(("target_", "recovery_", "execution_")) for n in core))
            return core.assign(horizon_minutes=horizon, target_return=.01)

        def fit(training, years, local, model, excluded):
            year = int(model.parent.parent.name.split("_")[-1])
            self.assertIn(("core", year), events)
            self.assertEqual(set(training.datetime.dt.year), set(years))
            self.assertFalse(set(years)&set(excluded))
            self.assertFalse(any(e[0] == "test" for e in events))
            events.append(("fit", year, local["t_model_profile"], local["t_label_horizon_minutes"]))
            model.mkdir(parents=True)
            for name in ("return_model.txt", "model_bundle.joblib", "model_metadata.json", "training_inputs.pkl",
                         "training_labels.csv", "feature_manifest.csv", "training_split_audit.csv", "native_reload_audit.json"):
                (model/name).write_text(model.name+name)
            return dict(fitted_rows=4, features=["f_signal"])

        def compare(model, source, out, inputs):
            horizon, profile = model.name.split("_", 1)
            self.assertEqual(source.name, horizon+"_"+runner.SOURCE_PROFILE_MAP[profile])
            self.assertEqual(len(list(model.iterdir())), 8)
            self.assertEqual(out.parent.name, "training_identity")
            original_write(out, {"passed": True})

        def context(fold, scope, prices, signals, calendar, local, out, inputs, store):
            year = fold["test_year"]
            if scope == "test":
                marker = out.parent.parent/"all_rules_frozen_before_outer_test.json"
                self.assertTrue(marker.exists())
                proof = json.loads(marker.read_text())
                self.assertEqual(set(proof["selected_rule_sha256"]), {"2023", "2026"})
                self.assertEqual(len(proof["model_artifacts_sha256"]), 96)
                self.assertTrue(all(not Path(n).is_absolute() and "training_identity" not in n for n in proof["model_artifacts_sha256"]))
                self.assertEqual(len([e for e in events if e[0] == "fit"]), 12)
                self.assertEqual(len([e for e in events if e[0] == "validation"]), 2)
            events.append((scope, year))
            selected = pd.DataFrame({"symbol": ["SH.600001"], "shares": [1000]})
            baseline = pd.DataFrame({"datetime": ["2025-03-11"], "equity": [1e6]})
            for relative, data in (("frozen_base_selections.csv", selected), ("frozen_base_entries.csv", selected),
                                   ("baseline/test_equity_curve.csv", baseline)):
                for destination in (out/relative, self.source/f"year_{year}/{scope}_context"/relative):
                    destination.parent.mkdir(parents=True, exist_ok=True); data.to_csv(destination, index=False)
            return selected, prices, frame.iloc[:2].copy(), local, object()

        def predict(signals, model, local, out, inputs):
            self.assertEqual(model.name, f"h{local['t_label_horizon_minutes']}_{local['t_model_profile']}")
            self.assertTrue((model/"model_bundle.joblib").exists())
            self.assertEqual(local["commission_rate"], cfg["commission_rate"])
            events.append(("prediction", model.parent.parent.name, model.name))
            return pd.DataFrame()

        def write(path, value):
            original_write(path, value)
            if tamper and Path(path).name == "all_rules_frozen_before_outer_test.json":
                (Path(path).parent/sorted(value["model_artifacts_sha256"])[-1]).write_text("changed")

        replacements = dict(load_configuration=lambda _: cfg, snapshot_sources=snapshot,
            validate_execution_source=lambda *a: None, validate_balanced_sources=lambda *a: {"old_t_models_reused": True},
            read_balanced_fold=lambda cfg, year, inputs: copy.deepcopy(folds[year]),
            load_frozen_market_data=lambda *a: (pd.DataFrame(), frame, daily), add_timing_features=lambda frame, cfg: frame,
            load_labels_only=lambda *a: object(), load_frozen_inventory_core_cohort=cohort, attach_learning_labels=attach,
            fit_inventory_robust_model=fit, compare_training_to_source=compare, prepare_account_context=context,
            get_store=lambda *a: object(), predict_and_audit=predict,
            evaluate_worker=lambda rule: self.rows([rule], rate).iloc[0].to_dict(), write_json=write,
            generate_inventory_robust_decisions=lambda *a: (pd.DataFrame(), {"passed": True}),
            run_guarded_portfolio=lambda *a, **k: {}, save_guarded_result=lambda *a: {},
            write_robust_inventory_report=lambda directory, *a: (directory/"final_report.md").write_text("工程替身"))
        with ExitStack() as stack:
            for name, replacement in replacements.items(): stack.enter_context(patch.object(runner, name, side_effect=replacement))
            try: runner.main([])
            finally:
                self.events = events
                self.directory = directories[0] if directories else None
        return directories[0], events

    def test_all_folds_six_models_freeze_before_first_outer_with_source_identity_external(self):
        directory, events = self.run_workflow(scope="smoke")
        self.assertEqual(len([e for e in events if e[0] == "fit"]), 12)
        self.assertEqual(len([e for e in events if e[0] == "prediction"]), 18)
        self.assertEqual(len([e for e in events if e[0] == "test"]), 2)
        protocol = json.loads((directory/"research_protocol.json").read_text())
        self.assertTrue(protocol["cross_horizon_argmax"])
        self.assertFalse(protocol["outer_model_refit"])
        self.assertFalse(protocol["score_is_calibrated_conditional_mean"])

    def test_negative_active_candidate_is_not_replaced_by_hold_but_formal_gate_stops_outer(self):
        directory, events = self.run_workflow()
        self.assertFalse(any(e[0] == "test" for e in events))
        chosen = json.loads((directory/"year_2023/selected_rule.json").read_text())
        self.assertTrue(chosen["rule"]["enabled"])
        self.assertLess(chosen["inner_metrics"]["incremental_net_pnl"], 0)
        self.assertFalse(json.loads((directory/"goal_evaluation.json").read_text())["formal_outer_evaluated"])

    def test_last_fold_model_tamper_blocks_first_outer_not_just_its_own_fold(self):
        with self.assertRaisesRegex(AssertionError, "外评前改变"):
            self.run_workflow(scope="smoke", tamper=True)
        self.assertFalse(any(e[0] == "test" for e in self.events))


if __name__ == "__main__": unittest.main(verbosity=2)
