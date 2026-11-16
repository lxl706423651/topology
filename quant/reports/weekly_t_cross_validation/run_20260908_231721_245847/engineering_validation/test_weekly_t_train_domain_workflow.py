"""验证v7只收窄核心T训练域、模型/支持域一致、推断不加静态价格门槛。

从quant根目录运行。复用四年合成OOF夹具，真实执行域过滤、标签隔离和文件复制；
模型训练和账户评价用确定性替身，不读取正式外层结果，不修改缓存或冻结源码。
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_backtest.weekly_t_cohort import CONTEXT_FEATURES
from quant_backtest.weekly_t_models import select_training_rows
from quant_backtest.weekly_t_short_horizon import fit_feature_support, planned_recovery_times
from quant_backtest.weekly_t_train_domain import filter_core_t_training_domain
from scripts import run_weekly_t_cohort as previous_runner
from scripts import run_weekly_t_train_domain as runner
from scripts import test_weekly_t_cohort_workflow as cohort_fixtures


class WeeklyTTrainDomainWorkflowTests(unittest.TestCase):
    """对新增训练过滤在完整调用链中的位置和作用范围做非重复回归检查。"""

    def setUp(self):
        """建立低价/高价两周、早中晚三信号，标签缺失独立于当前资格。"""
        self.fixture = cohort_fixtures.WeeklyTCohortWorkflowTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.cfg = self.fixture.cfg
        self.cfg.update({"initial_cash": 10000., "base_fraction": .95, "lot_size": 100,
                         "horizons": [20, 40, 60], "short_t_net_margins": [.001],
                         "smoke_weeks": 0, "research_version": "synthetic_v7_workflow",
                         "t_training_domain": "known_slot_and_recovery_deadline"})
        self.weekly = self.fixture.weekly.copy(deep=True)
        first_week = self.weekly.groupby(self.weekly.trade_week.dt.year).trade_week.transform("min")
        self.weekly["f_last_close"] = np.where(self.weekly.trade_week.eq(first_week), 10., 30.)
        parts = []
        for offset, clock in enumerate(["09:35", "13:30", "14:00"]):
            part = self.fixture.intraday.copy(deep=True)
            hour, minute = map(int, clock.split(":"))
            part["datetime"] = part.date + pd.Timedelta(hours=hour, minutes=minute)
            part["f_input"] += offset * 100.
            part["label_start"] = part.datetime + pd.Timedelta(minutes=1)
            part["label_end"] = planned_recovery_times(part.datetime, 20)
            parts.append(part)
        self.intraday = pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
        # 低价周首个周中早信号的未来标签缺失，仍须进入domain及内/外层推断。
        self.missing_label = (self.intraday.datetime.dt.dayofweek.eq(1)
                              & self.intraday.datetime.dt.strftime("%H:%M").eq("09:35")
                              & self.intraday.symbol.eq("SH.600000"))
        self.intraday.loc[self.missing_label, "target_return"] = np.nan
        labels = []
        for horizon in self.cfg["horizons"]:
            part = self.intraday[["datetime", "symbol", "label_start", "target_return"]].copy()
            part["horizon_minutes"] = horizon
            part["label_end"] = planned_recovery_times(part.datetime, horizon)
            part["recovery_time"] = part.label_end
            part["recovery_price"] = 10.01
            part.loc[part.label_end.isna(), "target_return"] = np.nan
            labels.append(part)
        self.labels = pd.concat(labels, ignore_index=True)
        self.daily, self.stores = self.fixture.daily, self.fixture.stores
        self.domain_calls, self.support_calls, self.fit_calls, self.prediction_calls = [], [], [], []

    def record_domain(self, frame, weekly, horizon, cfg):
        """记录真实过滤前后输入，确认未来缺标签不会影响当前domain。"""
        result, audit, metadata = filter_core_t_training_domain(frame, weekly, horizon, cfg)
        self.domain_calls.append({"years": sorted(frame.datetime.dt.year.unique().tolist()),
                                  "horizon": horizon, "frame": result.copy(deep=True),
                                  "audit": audit.copy(deep=True), "metadata": deepcopy(metadata)})
        self.assertTrue(result.target_return.isna().any())
        self.assertEqual(len(result), {20: 18, 40: 12, 60: 6}[horizon])
        pd.testing.assert_frame_equal(result, frame.loc[audit.keep])
        return result, audit, metadata

    def record_support(self, frame, cfg):
        """用实际严格标签子集拟合支持域，保存行键以与模型输入逐项核对。"""
        result = fit_feature_support(frame, cfg)
        self.support_calls.append({"frame": frame.copy(deep=True), "support": deepcopy(result)})
        return result

    def fake_fit(self, frame, years, cfg, output_dir, excluded_years=()):
        """保留真实标签purge，确认模型母表正是domain输出，写确定性权重替身。"""
        horizon = cfg["t_label_horizon_minutes"]
        domain = next(call for call in self.domain_calls
                      if call["years"] == list(years) and call["horizon"] == horizon)
        pd.testing.assert_frame_equal(frame[domain["frame"].columns.difference(
            ["label_start", "label_end", "target_return", "label_available", "recovery_price"])].reset_index(drop=True),
            domain["frame"][domain["frame"].columns.difference(
                ["label_start", "label_end", "target_return", "label_available", "recovery_price"])].reset_index(drop=True))
        actual, metadata, _ = select_training_rows(frame, years, cfg, "intraday", excluded_years)
        pd.testing.assert_frame_equal(actual, self.support_calls[-1]["frame"])
        self.assertEqual(len(actual), {20: 16, 40: 10, 60: 4}[horizon])
        self.assertTrue(all(name in actual for name in CONTEXT_FEATURES.values()))
        self.assertFalse(actual.target_return.isna().any())
        self.fit_calls.append({"horizon": horizon, "years": list(years), "frame": actual.copy(deep=True)})
        metadata["profile"] = cfg["t_model_profile"]
        bundle = {"metadata": metadata, "synthetic_weights": [horizon, cfg["t_model_profile"]]}
        path = Path(output_dir)
        path.mkdir(parents=True)
        joblib.dump(bundle, path / "model_bundle.joblib")
        (path / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return bundle

    def fake_predict(self, bundle, frame, cfg):
        """检查推断仍含高价周、晚时钟和缺标签行，仅给出固定预测用于流程测试。"""
        known = frame[["trade_week", "symbol"]].merge(
            self.weekly[["trade_week", "symbol", "f_last_close"]], on=["trade_week", "symbol"], validate="many_to_one")
        self.assertEqual(set(known.f_last_close), {10., 30.})
        self.assertEqual(set(frame.datetime.dt.strftime("%H:%M")), {"09:35", "13:30", "14:00"})
        self.assertTrue(frame.target_return.isna().any())
        self.assertFalse(any(column.startswith("static_") for column in frame))
        self.prediction_calls.append({"years": sorted(frame.datetime.dt.year.unique().tolist()),
                                      "rows": len(frame), "weights": bundle["synthetic_weights"]})
        result = frame.copy(deep=True)
        result["prediction"] = .02 if bundle["metadata"]["profile"] == "l2_depth4" else .03
        return result

    def fake_decisions(self, predictions, rule, cfg, support):
        """保留固定时钟差异供候选去重，不引入训练域静态股数到推断。"""
        result = self.fixture.fake_decisions(predictions, rule, cfg, support)
        result["planned_recovery_time"] = planned_recovery_times(result.datetime, rule["horizon_minutes"])
        result = result.loc[result.planned_recovery_time.notna()
                            & result.planned_recovery_time.dt.strftime("%H:%M").le(cfg["latest_planned_recovery_time"])]
        return result.drop_duplicates(["date", "symbol"], keep="first").reset_index(drop=True)

    def test_horizon_models_and_support_share_filtered_core_and_inference_remains_unfiltered(self):
        directory = self.root / "v7_run"
        directory.mkdir()
        original_weekly, original_intraday = self.weekly.copy(deep=True), self.intraday.copy(deep=True)
        with ExitStack() as stack:
            for name, replacement in {"build_core_t_cohort": self.fixture.fake_cohort,
                                      "filter_core_t_training_domain": self.record_domain,
                                      "fit_feature_support": self.record_support, "fit_t_model": self.fake_fit,
                                      "predict_t": self.fake_predict, "create_short_decisions": self.fake_decisions,
                                      "run_candidate_evaluations": self.fixture.fake_evaluations}.items():
                stack.enter_context(patch.object(runner, name, side_effect=replacement))
            folds, _ = runner.train_all_rules(self.weekly, self.intraday, self.daily, self.labels,
                                               self.stores, self.cfg, directory)
            self.assertEqual(len(self.domain_calls), 6)
            self.assertEqual(len(self.support_calls), 6)
            self.assertEqual(len(self.fit_calls), 12)
            frozen = json.loads((directory / "all_rules_frozen_before_outer_test.json").read_text())
            self.assertEqual(frozen["test_years"], self.cfg["test_years"])
            for fold in folds:
                out = directory / f"year_{fold['test_year']}"
                for horizon in self.cfg["horizons"]:
                    call = next(c for c in self.domain_calls if c["years"] == fold["core_years"] and c["horizon"] == horizon)
                    self.assertEqual(json.loads((out / f"training_domain_h{horizon}.json").read_text()), call["metadata"])
                    audit = pd.read_csv(out / f"training_domain_h{horizon}.csv")
                    self.assertEqual(audit.keep.tolist(), call["audit"].keep.tolist())
                    self.assertEqual(audit.drop_reason.tolist(), call["audit"].drop_reason.tolist())
                    support = json.loads((out / f"support_h{horizon}.json").read_text())
                    self.assertEqual(support["training_rows"], {20: 16, 40: 10, 60: 4}[horizon])
                rule = json.loads((out / "selected_rule.json").read_text())
                self.assertFalse(rule["fold"]["outer_t_refit"])
                self.assertEqual((out / rule["source_model"] / "model_bundle.joblib").read_bytes(),
                                 (out / "t_model/model_bundle.joblib").read_bytes())
                self.assertEqual(hashlib.sha256((out / "selected_rule.json").read_bytes()).hexdigest(),
                                 frozen["selected_rule_sha256"][str(fold["test_year"])])

            def fake_engine(selected, daily, intraday, decisions, cfg, recovery_bars=None):
                """外层只在全部冻结后执行，预算/费用/决策函数配置没有变化。"""
                self.assertTrue((directory / "all_rules_frozen_before_outer_test.json").exists())
                self.assertEqual(len(self.fit_calls), 12)
                self.assertEqual(len(self.domain_calls), 6)
                self.assertIs(cfg, self.cfg)
                self.assertEqual(set(intraday.datetime.dt.strftime("%H:%M")), {"09:35", "13:30", "14:00"})
                return {"metrics": {"baseline_total_return": 0., "total_return": 0., "t_added_equity": 0.}}

            stack.enter_context(patch.object(runner, "run_weekly_portfolio", side_effect=fake_engine))
            stack.enter_context(patch.object(runner, "save_result", side_effect=lambda result, out, cfg: result["metrics"]))
            summary, _ = runner.evaluate_all_folds(folds, self.weekly, self.intraday, self.daily,
                                                   self.stores, self.cfg, directory)
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(self.fit_calls), 12)
            self.assertEqual(len(self.prediction_calls), 14)
            self.assertEqual(len(self.domain_calls), 6)
        pd.testing.assert_frame_equal(self.weekly, original_weekly)
        pd.testing.assert_frame_equal(self.intraday, original_intraday)

    def test_empty_training_domain_fails_before_fit_without_falling_back_to_all_rows(self):
        directory = self.root / "empty_run"
        directory.mkdir()
        weekly = self.weekly.copy(deep=True)
        weekly["f_last_close"] = 1000.
        with patch.object(runner, "build_core_t_cohort", side_effect=self.fixture.fake_cohort), \
                patch.object(runner, "fit_t_model") as fit, patch.object(runner, "evaluate_all_folds") as outer:
            with self.assertRaisesRegex(ValueError, "有效训练样本 0"):
                runner.train_all_rules(weekly, self.intraday, self.daily, self.labels, self.stores, self.cfg, directory)
            fit.assert_not_called()
            outer.assert_not_called()
        metadata = json.loads((directory / "year_2023/training_domain_h20.json").read_text())
        self.assertEqual(metadata["kept_rows"], 0)
        self.assertGreater(metadata["input_rows"], 0)
        self.assertFalse((directory / "all_rules_frozen_before_outer_test.json").exists())

    def test_main_defaults_to_v7_persists_domain_protocol_and_finishes_all_rules_before_outer(self):
        order, outputs = [], []
        empty = pd.DataFrame()

        def configure():
            """默认配置须指向新增v7文件，不能默默复用v6入口。"""
            self.assertEqual(sys.argv[-2:], ["--config", str(runner.PROJECT_ROOT / "config/weekly_t_cross_validation_v7.yaml")])
            return self.cfg

        def train(universe, intraday, daily, labels, stores, cfg, directory):
            """在任何外层前完成事前协议和全部内部冻结标记。"""
            protocol = json.loads((directory / "research_protocol.json").read_text())
            self.assertEqual(protocol["t_training_domain"], cfg["t_training_domain"])
            self.assertTrue(protocol["previous_outer_results_seen"])
            self.assertIn("不加静态价格门槛", protocol["training_domain_limit"])
            self.assertFalse(protocol["outer_t_refit"])
            order.append("all_internal_rules")
            (directory / "all_rules_frozen_before_outer_test.json").write_text(json.dumps({"test_years": cfg["test_years"]}))
            outputs.append(directory)
            return [{"test_year": year} for year in cfg["test_years"]], []

        def evaluate(folds, universe, intraday, daily, stores, cfg, directory):
            """外层接收到完整折列表与冻结标记，未调用训练域过滤器。"""
            self.assertEqual(order, ["all_internal_rules"])
            self.assertEqual(json.loads((directory / "all_rules_frozen_before_outer_test.json").read_text())["test_years"],
                             [fold["test_year"] for fold in folds])
            order.append("outer")
            return empty, []

        replacements = {"load_configuration": configure, "snapshot_domain_sources": lambda directory: None,
                        "load_weekly_t_data": lambda cfg: (self.weekly, self.intraday, self.daily, {}),
                        "load_recovery_data": lambda cfg: (empty, {}),
                        "load_horizon_data": lambda cfg: (self.labels, empty, {"frozen_selection_sources": []}),
                        "build_year_stores": lambda early, tail, cfg: self.stores,
                        "add_weekly_execution_features": lambda weekly, daily: weekly,
                        "add_session_context_features": lambda intraday: intraday,
                        "add_feature_provenance": lambda weekly, intraday, daily: (weekly, intraday, {}),
                        "select_weekly_universe": lambda weekly, cfg: weekly,
                        "train_all_rules": train, "evaluate_all_folds": evaluate,
                        "write_final_report": lambda *args: None, "append_domain_report": lambda *args: None}
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["synthetic_v7_test"]))
            domain = stack.enter_context(patch.object(runner, "filter_core_t_training_domain"))
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(runner, name, side_effect=replacement))
            runner.main()
            domain.assert_not_called()
        self.assertEqual(order, ["all_internal_rules", "outer"])
        self.assertEqual(len(outputs), 1)

    def test_snapshot_has_exact_v7_dependencies_and_configuration_preserves_v6_economics(self):
        directory = self.root / "snapshot"
        directory.mkdir()
        runner.snapshot_domain_sources(directory)
        hashes = json.loads((directory / "source_hashes.json").read_text())
        self.assertEqual(len(hashes), 17)
        for name in ("scripts/run_weekly_t_train_domain.py", "quant_backtest/weekly_t_cohort.py",
                     "quant_backtest/weekly_t_train_domain.py"):
            self.assertIn(name, hashes)
        self.assertNotIn("scripts/run_weekly_t_cohort.py", hashes)
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((runner.PROJECT_ROOT / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(hashlib.sha256((directory / "source_snapshot" / Path(name).name).read_bytes()).hexdigest(), digest)
        old = yaml.safe_load((runner.PROJECT_ROOT / "config/weekly_t_cross_validation_v6.yaml").read_text())
        new = yaml.safe_load((runner.PROJECT_ROOT / "config/weekly_t_cross_validation_v7.yaml").read_text())
        changed = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
        self.assertEqual(changed, {"research_version", "t_training_domain"})
        for name in ("predict_t", "create_short_decisions", "run_weekly_portfolio", "select_execution_data",
                     "attach_horizon_labels", "candidate_passes_support", "fit_feature_support"):
            self.assertIs(getattr(runner, name), getattr(previous_runner, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
