"""验证v6入口只用同折OOF队列拟合T、正确接入冻结周身份并先冻结全部规则。"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_cohort import CONTEXT_FEATURES, attach_weekly_context
from scripts import run_weekly_t_cohort as runner
from scripts.test_weekly_t_cohort import cohort_config, cohort_frames, fake_weekly_predictions
from scripts.test_weekly_t_short_horizon import short_cfg


class WeeklyTCohortWorkflowTests(unittest.TestCase):
    """生产流程保留真实连接、标签purge和文件复制，仅替代耗时模型拟合与账户回测。"""

    def setUp(self):
        """构造独立临时的四年输入及本折v4选股/周模型，不读取正式外层结果。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="weekly_t_cohort_workflow_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.weekly, self.intraday = cohort_frames()
        self.daily = self.intraday[["date", "symbol"]].drop_duplicates().reset_index(drop=True)
        self.cfg = {**short_cfg(), **cohort_config(), "test_years": [2023, 2026], "horizons": [20],
                    "t_model_profiles": ["l2_depth4", "l2_depth6"], "short_t_net_margins": [.001, .002],
                    "short_t_fraction": .25, "max_symbols": 8, "smoke_weeks": 1,
                    "t_support_features": ["f_input", "f_weekly_score"], "t_support_quantiles": [.05, .95],
                    "frozen_selection_report_dir": str(self.root / "v4"), "initial_cash": 1000000,
                    "output_dir": str(self.root / "outputs"), "research_version": "synthetic_v6_workflow",
                    "scheduled_recovery_times": ["09:56"], "outer_t_refit": False,
                    "t_training_cohort": "core_year_oof_weekly_top100_middle_days"}
        self.labels = self.intraday[["datetime", "symbol", "label_start", "label_end", "target_return"]].copy()
        self.labels["horizon_minutes"] = 20
        self.labels["recovery_time"] = self.labels.label_end
        self.labels["recovery_price"] = 10.01
        self.stores = {year: object() for year in (2023, 2024, 2025, 2026)}
        self.fit_calls, self.cohort_calls, self.outer_bundles = [], [], []
        self.make_frozen_inputs()

    def make_frozen_inputs(self):
        """写入合法inner/outer训练年与排除年，权重字节用于核对实际bundle SHA。"""
        for year in self.cfg["test_years"]:
            directory = self.root / "v4" / f"year_{year}"
            directory.mkdir(parents=True)
            train = [candidate for candidate in (2023, 2024, 2025, 2026) if candidate != year]
            inner = max(train)
            core = [candidate for candidate in train if candidate != inner]
            for target_year, filename in ((inner, "validation_weekly_selections.csv"), (year, "weekly_selections.csv")):
                target = self.weekly.loc[self.weekly.trade_week.dt.year.eq(target_year)]
                selected = fake_weekly_predictions(target, self.cfg)
                selected = selected.loc[selected.selected].merge(target[["trade_week", "symbol", "entry_date", "exit_date"]], on=["trade_week", "symbol"], validate="one_to_one")
                selected.to_csv(directory / filename, index=False)
            for name, years, excluded in (("inner_weekly_model", core, [year, inner]), ("weekly_model", train, [year])):
                model = directory / name
                model.mkdir()
                metadata = {"train_years": years, "excluded_years": sorted(excluded)}
                (model / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                joblib.dump({"metadata": metadata, "synthetic_weights": name + str(year)}, model / "model_bundle.joblib")
            for name in ("weekly_predictions.csv", "weekly_prediction_metrics.csv"):
                (directory / name).write_text("synthetic\n1\n", encoding="utf-8")

    def frozen_arguments(self, year=2023, inner=True):
        """调用真实执行行筛选器，返回单折冻结上下文所需输入。"""
        directory = self.root / "v4" / f"year_{year}"
        filename = "validation_weekly_selections.csv" if inner else "weekly_selections.csv"
        selected = runner.read_frozen_selections(directory / filename, set(self.daily.symbol), self.cfg)
        _, samples = runner.select_execution_data(selected, self.daily, self.intraday)
        model = directory / ("inner_weekly_model" if inner else "weekly_model")
        expected = json.loads((model / "model_metadata.json").read_text())["train_years"]
        return samples, selected, model, expected

    def test_inner_and_outer_use_same_fold_model_years_sha_and_four_week_sources(self):
        for year in self.cfg["test_years"]:
            for inner in (True, False):
                with self.subTest(year=year, inner=inner):
                    samples, selected, model, expected = self.frozen_arguments(year, inner)
                    result = runner.attach_frozen_context(samples, selected, self.weekly, self.cfg, model, expected)
                    digest = hashlib.sha256((model / "model_bundle.joblib").read_bytes()).hexdigest()
                    self.assertEqual(set(result.source_model_sha256), {digest})
                    self.assertEqual(set(result.source_model_training_years.map(json.loads).map(tuple)), {tuple(expected)})
                    self.assertTrue(all(name in result for name in CONTEXT_FEATURES.values()))
                    self.assertTrue(result.feature_source_start.lt(samples.feature_source_start).all())
                    self.assertTrue(result.feature_source_start.eq(result.weekly_context_feature_source_start).all())
                    manifest = result.attrs["frozen_weekly_context_manifest"]
                    keys = ["datetime", "symbol", "feature_source_start", *CONTEXT_FEATURES.values(),
                            "source_model_training_years", "source_model_sha256"]
                    self.assertEqual(manifest["source_model_sha256"], digest)
                    self.assertEqual(manifest["source_metadata_sha256"], hashlib.sha256((model / "model_metadata.json").read_bytes()).hexdigest())
                    self.assertEqual(manifest["source_model_training_years"], expected)
                    self.assertEqual(manifest["context_rows"], len(result))
                    self.assertEqual(manifest["context_inputs_sha256"], hashlib.sha256(pd.util.hash_pandas_object(result[keys], index=False).to_numpy().tobytes()).hexdigest())
                    pd.testing.assert_frame_equal(result[["datetime", "symbol", "target_return", "label_start", "label_end"]],
                                                  samples[["datetime", "symbol", "target_return", "label_start", "label_end"]])

    def test_frozen_model_training_or_excluded_year_mismatch_fails_closed(self):
        samples, selected, model, expected = self.frozen_arguments()
        path = model / "model_metadata.json"
        original = json.loads(path.read_text())
        for changes in ({"train_years": [2023, 2024]}, {"excluded_years": [2023]}):
            path.write_text(json.dumps({**original, **changes}), encoding="utf-8")
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.attach_frozen_context(samples, selected, self.weekly, self.cfg, model, expected)
        path.write_text(json.dumps(original), encoding="utf-8")

    def test_prediction_trace_cannot_override_actual_frozen_model_identity(self):
        samples, selected, model, expected = self.frozen_arguments()
        digest = hashlib.sha256((model / "model_bundle.joblib").read_bytes()).hexdigest()
        for years, sha in (([2024], digest), (expected, "f" * 64)):
            changed = selected.copy(deep=True)
            changed["source_model_training_years"] = json.dumps(years)
            changed["source_model_sha256"] = sha
            with self.subTest(years=years, sha=sha), self.assertRaises(ValueError):
                runner.attach_frozen_context(samples, changed, self.weekly, self.cfg, model, expected)
        matching = selected.copy(deep=True)
        matching["source_model_training_years"] = json.dumps(list(reversed(expected)))
        matching["source_model_sha256"] = digest
        accepted = runner.attach_frozen_context(samples, matching, self.weekly, self.cfg, model, expected)
        self.assertEqual(set(accepted.source_model_training_years.map(json.loads).map(tuple)), {tuple(expected)})

    def fake_cohort(self, universe, intraday, core_years, excluded_years, cfg, output_dir):
        """只返回互为OOF的选中周中行，加入可识别的四项真实上下文连接结果。"""
        self.cohort_calls.append((list(core_years), sorted(excluded_years)))
        weekly = universe.loc[universe.trade_week.dt.year.isin(core_years)]
        predictions = fake_weekly_predictions(weekly, cfg)
        predictions = predictions.loc[predictions.selected].copy()
        predictions["source_model_training_years"] = predictions.trade_week.dt.year.map(
            lambda year: json.dumps([candidate for candidate in core_years if candidate != year]))
        predictions["source_model_sha256"] = "a" * 64
        selected = intraday.loc[intraday.datetime.dt.year.isin(core_years)
                                & intraday.symbol.eq("SH.600000") & intraday.datetime.dt.dayofweek.isin([1, 2, 3])].copy()
        enriched = attach_weekly_context(selected, predictions, universe, cfg)
        return enriched, {"core_years": list(core_years), "excluded_years": list(excluded_years),
                          "training_rows_before_t_label_purge": len(enriched), "synthetic_sentinel": "OOF_SELECTED_ONLY"}

    def fake_t_fit(self, frame, years, cfg, output_dir, excluded_years=()):
        """验证传给T模型的是周中OOF队列，写可被真实joblib重载的固定权重对象。"""
        self.assertEqual(set(frame.symbol), {"SH.600000"})
        self.assertEqual(set(frame.datetime.dt.year), set(years))
        self.assertEqual(set(frame.datetime.dt.dayofweek), {1, 2, 3})
        self.assertTrue(all(name in frame for name in CONTEXT_FEATURES.values()))
        self.assertTrue(frame.feature_source_start.eq(frame.weekly_context_feature_source_start).all())
        self.assertTrue(all(year not in json.loads(trace) for year, trace in zip(frame.datetime.dt.year, frame.source_model_training_years)))
        self.fit_calls.append((list(years), sorted(excluded_years), cfg["t_model_profile"]))
        bundle = {"metadata": {"train_years": list(years), "excluded_years": sorted(excluded_years),
                               "profile": cfg["t_model_profile"]},
                  "features": list(CONTEXT_FEATURES.values()), "synthetic_weights": cfg["t_model_profile"]}
        directory = Path(output_dir)
        directory.mkdir(parents=True)
        joblib.dump(bundle, directory / "model_bundle.joblib")
        (directory / "model_metadata.json").write_text(json.dumps(bundle["metadata"]), encoding="utf-8")
        return bundle

    def fake_predict(self, bundle, frame, cfg):
        """以profile标记模拟不同预测，完全保留当前周上下文及其溯源。"""
        self.assertTrue(all(name in frame for name in CONTEXT_FEATURES.values()))
        result = frame.copy(deep=True)
        result["prediction"] = .02 if bundle["metadata"]["profile"] == "l2_depth4" else .03
        return result

    def fake_decisions(self, predictions, rule, cfg, support):
        """生成可按内容去重的当前请求；测试不依赖引擎经济表现。"""
        result = predictions[["datetime", "symbol"]].copy()
        result["action"] = 1 if rule.get("enabled", True) else 0
        result["fraction"] = .25
        result["priority"] = predictions.prediction + rule["net_margin"]
        result["planned_recovery_time"] = result.datetime + pd.Timedelta(minutes=21)
        result["date"] = result.datetime.dt.normalize()
        return result

    def fake_evaluations(self, tasks, selected, daily, intraday, cfg, audit_dir, recovery_store=None):
        """返回固定的合法内部评价，让最后一个预声明候选被确定选中。"""
        years = set(intraday.datetime.dt.year)
        self.assertEqual(len(years), 1)
        self.assertTrue(all(name in intraday for name in CONTEXT_FEATURES.values()))
        metrics = {"t_pair_count": 120, "inventory_mismatch_days": 0, "recovery_failure_days": 0,
                   "pending_pairs": 0, "base_plan_mismatch_count": 0, "t1_violation_count": 0}
        support = {"active_days": 20, "active_weeks": 8, "active_symbols": 20,
                   "weekly_increment_ci_low_rmb": 1., "weekly_increment_ci_high_rmb": 20.,
                   "max_day_pair_fraction": .1, "max_week_pair_fraction": .1,
                   "max_day_abs_pnl_fraction": .1, "max_week_abs_pnl_fraction": .1}
        return {task["decision_sha256"]: {"incremental_net_pnl": 10. + task["candidate_index"],
                "original_index": task["candidate_index"], "metrics": metrics, "support": support}
                for task in tasks}

    def test_train_uses_only_returned_oof_cohort_and_outer_reuses_core_weights_after_all_freeze(self):
        directory = self.root / "run"
        directory.mkdir()
        frozen_before = self.intraday.copy(deep=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "build_core_t_cohort", side_effect=self.fake_cohort))
            fit = stack.enter_context(patch.object(runner, "fit_t_model", side_effect=self.fake_t_fit))
            stack.enter_context(patch.object(runner, "predict_t", side_effect=self.fake_predict))
            stack.enter_context(patch.object(runner, "create_short_decisions", side_effect=self.fake_decisions))
            stack.enter_context(patch.object(runner, "run_candidate_evaluations", side_effect=self.fake_evaluations))
            folds, _ = runner.train_all_rules(self.weekly, self.intraday, self.daily, self.labels, self.stores, self.cfg, directory)
            self.assertEqual(self.cohort_calls, [([2024, 2025], [2023, 2026]), ([2023, 2024], [2025, 2026])])
            self.assertEqual(fit.call_count, 4)
            frozen = json.loads((directory / "all_rules_frozen_before_outer_test.json").read_text())
            self.assertEqual(frozen["test_years"], [2023, 2026])
            for fold in folds:
                out = directory / f"year_{fold['test_year']}"
                rule = json.loads((out / "selected_rule.json").read_text())
                self.assertEqual(rule["profile"], "l2_depth6")
                self.assertFalse(rule["fold"]["outer_t_refit"])
                self.assertEqual((out / rule["source_model"] / "model_bundle.joblib").read_bytes(), (out / "t_model/model_bundle.joblib").read_bytes())
                bundle = joblib.load(out / "t_model/model_bundle.joblib")
                self.assertEqual(bundle["metadata"]["train_years"], fold["core_years"])
                context = json.loads((out / "validation_weekly_context_manifest.json").read_text())
                original_model = self.root / "v4" / f"year_{fold['test_year']}" / "inner_weekly_model/model_bundle.joblib"
                self.assertEqual(context["source_model_training_years"], fold["core_years"])
                self.assertEqual(context["source_model_sha256"], hashlib.sha256(original_model.read_bytes()).hexdigest())
                self.assertEqual(hashlib.sha256((out / "selected_rule.json").read_bytes()).hexdigest(), frozen["selected_rule_sha256"][str(fold["test_year"])] )
            def fake_engine(selected, daily, intraday, decisions, cfg, recovery_bars=None):
                self.assertTrue((directory / "all_rules_frozen_before_outer_test.json").is_file())
                self.assertEqual(fit.call_count, 4)
                self.assertTrue(all(name in intraday for name in CONTEXT_FEATURES.values()))
                return {"metrics": {"baseline_total_return": 0., "total_return": 0., "t_added_equity": 0.}}
            stack.enter_context(patch.object(runner, "run_weekly_portfolio", side_effect=fake_engine))
            stack.enter_context(patch.object(runner, "save_result", side_effect=lambda result, out, cfg: result["metrics"]))
            summary, results = runner.evaluate_all_folds(folds, self.weekly, self.intraday, self.daily, self.stores, self.cfg, directory)
            self.assertEqual(fit.call_count, 4)
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(results), 2)
            for fold in folds:
                context = json.loads((directory / f"year_{fold['test_year']}" / "test_weekly_context_manifest.json").read_text())
                original_model = self.root / "v4" / f"year_{fold['test_year']}" / "weekly_model/model_bundle.joblib"
                self.assertEqual(context["source_model_training_years"], fold["train_years"])
                self.assertEqual(context["source_model_sha256"], hashlib.sha256(original_model.read_bytes()).hexdigest())
        pd.testing.assert_frame_equal(self.intraday, frozen_before)

    def test_main_calls_complete_rule_training_before_any_outer_evaluation(self):
        calls = []
        empty = pd.DataFrame()
        def train(universe, intraday, daily, labels, stores, cfg, directory):
            calls.append("all_internal_training")
            (directory / "all_rules_frozen_before_outer_test.json").write_text(json.dumps({"test_years": cfg["test_years"]}))
            return [{"test_year": year} for year in cfg["test_years"]], []
        def evaluate(folds, universe, intraday, daily, stores, cfg, directory):
            self.assertEqual(calls, ["all_internal_training"])
            self.assertEqual(json.loads((directory / "all_rules_frozen_before_outer_test.json").read_text())["test_years"], cfg["test_years"])
            calls.append("outer_evaluation")
            return empty, []
        replacements = {
            "load_configuration": lambda: self.cfg, "snapshot_cohort_sources": lambda directory: None,
            "load_weekly_t_data": lambda cfg: (self.weekly, self.intraday, self.daily, {}),
            "load_recovery_data": lambda cfg: (empty, {}),
            "load_horizon_data": lambda cfg: (self.labels, empty, {"frozen_selection_sources": []}),
            "build_year_stores": lambda early, tail, cfg: self.stores,
            "add_weekly_execution_features": lambda weekly, daily: weekly,
            "add_session_context_features": lambda intraday: intraday,
            "add_feature_provenance": lambda weekly, intraday, daily: (weekly, intraday, {}),
            "select_weekly_universe": lambda weekly, cfg: weekly,
            "train_all_rules": train, "evaluate_all_folds": evaluate,
            "write_final_report": lambda *args: None, "append_cohort_report": lambda *args: None}
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["test", "--config", "synthetic.yaml"]))
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(runner, name, side_effect=replacement))
            runner.main()
        self.assertEqual(calls, ["all_internal_training", "outer_evaluation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
