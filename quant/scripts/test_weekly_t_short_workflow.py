"""独立验证短周期运行流程的隔离、冻结模型身份、标签周期及基准，不训练真实模型。

从 quant 根目录运行；使用临时目录和合成四年面板，原始行情和正式报告不变。
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_weekly_t_short_horizon as runner
from quant_backtest.weekly_t_models import select_training_rows


class ShortWorkflowTests(unittest.TestCase):
    """以真实筛选和决策函数连接mock拟合器，检查调用边界而非模型数值效果。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output = self.root / "frozen", self.root / "new"
        self.output.mkdir()
        self.cfg = yaml.safe_load((PROJECT_ROOT / "config/weekly_t_cross_validation_v5.yaml").read_text())
        self.cfg.update(frozen_selection_report_dir=str(self.source), min_training_rows=2,
                        initial_cash=10000., t_support_features=["f_day_return"], t_avoid_locked_exit_side=False,
                        bootstrap_samples=20, dense_quote_cache_dir=str(self.root / "dense"))
        _, _, minutes = runner.validate_horizon_config(self.cfg)
        self.cfg["scheduled_recovery_times"] = [f"{minute // 60:02d}:{minute % 60:02d}" for minute in minutes]
        self.years = [2023, 2024, 2025, 2026]
        self.weeks = {year: pd.date_range(f"{year}-05-01", periods=1, freq="W-MON")[0] for year in self.years}
        signals, labels, daily, universe = [], [], [], []
        for year, week in self.weeks.items():
            for owner in self.years:
                symbol = f"S{owner}"
                current = week + pd.Timedelta(days=1, hours=10)
                signals.append({"datetime": current, "date": current.normalize(), "symbol": symbol,
                                "trade_week": week, "execution_time": current + pd.Timedelta(minutes=1),
                                "feature_source_start": week, "f_day_return": .005})
                daily.append({"date": current.normalize(), "symbol": symbol})
                universe.append({"trade_week": week, "symbol": symbol})
                for horizon in self.cfg["horizons"]:
                    end = current + pd.Timedelta(minutes=1 + horizon)
                    labels.append({"datetime": current, "symbol": symbol, "horizon_minutes": horizon,
                                   "recovery_time": end, "recovery_price": 10 * (1 + horizon / 10000),
                                   "label_start": current + pd.Timedelta(minutes=1), "label_end": end,
                                   "target_return": horizon / 10000})
        self.intraday, self.labels = pd.DataFrame(signals), pd.DataFrame(labels)
        self.daily, self.universe = pd.DataFrame(daily), pd.DataFrame(universe)
        self.stores = {year: object() for year in self.years}
        for year in self.years:
            directory = self.source / f"year_{year}"
            directory.mkdir(parents=True)
            inner = max(other for other in self.years if other != year)
            for data_year, filename in [(year, "weekly_selections.csv"), (inner, "validation_weekly_selections.csv")]:
                week = self.weeks[data_year]
                pd.DataFrame([{"symbol": f"S{year}", "trade_week": week, "entry_date": week,
                               "exit_date": week + pd.Timedelta(days=4), "rank": 1}]).to_csv(directory / filename, index=False)
            for filename in ["weekly_predictions.csv", "weekly_prediction_metrics.csv"]:
                pd.DataFrame({"source_owner": [year]}).to_csv(directory / filename, index=False)
            pd.DataFrame({"datetime": [self.weeks[year] + pd.Timedelta(days=1, hours=15)],
                          "baseline_equity": [10000.]}).to_csv(directory / "test_equity_curve.csv", index=False)
        self.fits, self.evaluations = [], []

    def fake_fit(self, frame, years, cfg, directory, excluded):
        """仍调用真实严格训练过滤，假模型只保留训练身份和周期以供重载核对。"""
        training, metadata, _ = select_training_rows(frame, years, cfg, "intraday", excluded)
        horizon = cfg["t_label_horizon_minutes"]
        self.assertTrue(training.horizon_minutes.eq(horizon).all())
        self.assertTrue(training.target_return.eq(horizon / 10000).all())
        self.assertEqual(set(training.datetime.dt.year), set(years))
        self.assertFalse(set(training.datetime.dt.year).intersection(excluded))
        self.fits.append({"years": list(years), "excluded": list(excluded), "horizon": horizon,
                          "profile": cfg["t_model_profile"], "rows": len(training), "directory": directory})
        bundle = {"horizon": horizon, "profile": cfg["t_model_profile"], "metadata": metadata}
        directory.mkdir(parents=True)
        joblib.dump(bundle, directory / "model_bundle.joblib")
        return bundle

    def fake_predict(self, bundle, frame, cfg):
        """保持当前信号和特征；所有profile同预测以检查相同决策只执行一次。"""
        return frame.assign(prediction=.02, model_profile=bundle["profile"])

    def fake_evaluate(self, tasks, selected, daily, intraday, cfg, directory, store):
        inner = int(selected.entry_date.dt.year.iloc[0])
        self.assertIs(store, self.stores[inner])
        self.assertTrue(daily.date.dt.year.eq(inner).all())
        self.assertTrue(intraday.datetime.dt.year.eq(inner).all())
        self.evaluations.append({"selected": selected.copy(), "tasks": tasks, "directory": directory})
        return {task["decision_sha256"]: {"metrics": {"t_pair_count": 200},
                "support": {"active_days": 30, "active_weeks": 10},
                "incremental_net_pnl": float(100 + task["candidate_index"]),
                "original_index": task["candidate_index"]} for task in tasks}

    def run_training(self):
        """拟合、推断和成交成本高的边界用mock替代，其余真实训练流程保持原样。"""
        with mock.patch.object(runner, "fit_t_model", side_effect=self.fake_fit), \
                mock.patch.object(runner, "predict_t", side_effect=self.fake_predict), \
                mock.patch.object(runner, "run_candidate_evaluations", side_effect=self.fake_evaluate), \
                mock.patch.object(runner, "candidate_passes_support", return_value=(True, [])), redirect_stdout(io.StringIO()):
            return runner.train_all_rules(self.universe, self.intraday, self.daily, self.labels,
                                          self.stores, self.cfg, self.output)

    def test_each_fold_uses_own_inner_selections_core_years_and_matching_horizon(self):
        """四折各自读取冻结内层文件；模型和支持域只见严格隔离后核心年标签。"""
        folds, rows = self.run_training()
        self.assertEqual(len(self.fits), 36)
        self.assertEqual(len(rows), 4 * 27)
        self.assertEqual(len(folds), 4)
        for fold, evaluation in zip(folds, self.evaluations):
            year, inner = fold["test_year"], fold["inner_year"]
            expected_core = set(self.years).difference({year, inner})
            self.assertEqual(set(fold["core_years"]), expected_core)
            self.assertEqual(evaluation["selected"].symbol.tolist(), [f"S{year}"])
            self.assertEqual(set(evaluation["selected"].entry_date.dt.year), {inner})
            self.assertEqual(len(evaluation["tasks"]), 3)
            fitted = [record for record in self.fits if record["directory"].parent.name == f"year_{year}"]
            for record in fitted:
                self.assertEqual(set(record["years"]), expected_core)
                self.assertEqual(set(record["excluded"]), {year, inner})
            path = self.output / f"year_{year}"
            rule = json.loads((path / "selected_rule.json").read_text())
            self.assertEqual(rule["horizon_minutes"], 60)
            self.assertEqual(rule["profile"], "l2_depth4")
            self.assertTrue(rule["frozen_before_any_outer_test"])
            self.assertEqual((path / "t_model/model_bundle.joblib").read_bytes(),
                             (path / rule["source_model"] / "model_bundle.joblib").read_bytes())
            support = json.loads((path / "selected_support.json").read_text())
            self.assertEqual(support["training_rows"], 8)
        marker = json.loads((self.output / "all_rules_frozen_before_outer_test.json").read_text())
        self.assertEqual(set(marker["selected_rule_sha256"]), set(map(str, self.years)))

    def test_outer_uses_frozen_core_bundle_and_own_selection_and_rejects_changed_baseline(self):
        """外层不重拟合；选中60分钟权重和规则保持同一周期，基准偏差失败关闭。"""
        folds, _ = self.run_training()
        calls = []

        def execute(selected, daily, intraday, decisions, cfg, recovery_bars):
            year = int(selected.entry_date.dt.year.iloc[0])
            self.assertIs(recovery_bars, self.stores[year])
            self.assertEqual(selected.symbol.tolist(), [f"S{year}"])
            self.assertEqual(decisions.planned_recovery_time.dt.strftime("%H:%M").tolist(), ["11:01"])
            calls.append(year)
            return {"equity_curve": pd.DataFrame({"datetime": [self.weeks[year] + pd.Timedelta(days=1, hours=15)],
                                                  "equity": [10010.], "baseline_equity": [10000.]})}

        metrics = {"baseline_total_return": 0., "total_return": .001, "t_added_equity": 10.}
        with mock.patch.object(runner, "fit_t_model", side_effect=AssertionError("outer refitted")), \
                mock.patch.object(runner, "predict_t", side_effect=self.fake_predict), \
                mock.patch.object(runner, "run_weekly_portfolio", side_effect=execute), \
                mock.patch.object(runner, "save_result", return_value=metrics), redirect_stdout(io.StringIO()):
            summary, _ = runner.evaluate_all_folds(folds, self.intraday, self.daily, self.stores, self.cfg, self.output)
            self.assertEqual(calls, self.years)
            self.assertEqual(summary.t_added_equity.tolist(), [10.] * 4)
            curve_path = self.source / "year_2023/test_equity_curve.csv"
            curve = pd.read_csv(curve_path)
            curve.baseline_equity += 1
            curve.to_csv(curve_path, index=False)
            with self.assertRaisesRegex(AssertionError, "冻结持有基准改变"):
                runner.evaluate_all_folds(folds[:1], self.intraday, self.daily, self.stores, self.cfg, self.output)

    def test_year_store_keeps_all_selected_symbols_and_year_end_retry_dates(self):
        """恢复输入按全年保留，不因本周未再入选而丢弃旧债务股票的后续报价。"""
        early = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-09 10:20", "2024-12-31 10:20", "2025-01-02 10:20"]),
                              "symbol": ["OLD", "OLD", "NEW"]})
        tail = pd.DataFrame({"datetime": pd.to_datetime(["2024-12-31 15:00", "2025-01-02 15:00"]),
                             "symbol": ["TAIL_ONLY", "NEW"]})
        with mock.patch.object(runner, "build_dense_recovery_store", side_effect=lambda bars, path: bars.copy()), \
                redirect_stdout(io.StringIO()):
            stores = runner.build_year_stores(early, tail, self.cfg)
        self.assertEqual(set(stores), {2024, 2025})
        self.assertEqual(set(stores[2024].symbol), {"OLD", "TAIL_ONLY"})
        self.assertTrue(stores[2024].datetime.eq(pd.Timestamp("2024-12-31 10:20")).any())
        self.assertTrue(stores[2024].datetime.dt.year.eq(2024).all())

    def test_main_finishes_all_training_before_any_outer_evaluation(self):
        """主流程不是逐折训练后立即看外层；先返回全部冻结规则，再启动外层入口。"""
        order = []
        cfg = {**self.cfg, "output_dir": str(self.root / "main")}
        folds = [{"test_year": year} for year in self.years]

        def train(universe, intraday, daily, labels, stores, config, directory):
            self.assertEqual(order, [])
            protocol = json.loads((directory / "research_protocol.json").read_text())
            self.assertTrue(protocol["created_before_this_run_outer_test"])
            order.append("train_all")
            # 各折实际冻结文件及哈希由上一个真实train_all_rules测试核对。
            return folds, []

        def evaluate(received, intraday, daily, stores, config, directory):
            self.assertEqual(order, ["train_all"])
            self.assertEqual(received, folds)
            order.append("outer")
            return pd.DataFrame(), []

        replacements = {
            "load_configuration": lambda: cfg,
            "snapshot_sources": lambda directory: None,
            "load_weekly_t_data": lambda config: (self.universe, self.intraday, self.daily, {}),
            "load_recovery_data": lambda config: (pd.DataFrame(), {}),
            "load_horizon_data": lambda config: (self.labels, pd.DataFrame(), {"frozen_selection_sources": []}),
            "build_year_stores": lambda early, tail, config: self.stores,
            "add_weekly_execution_features": lambda weekly, daily: weekly,
            "add_session_context_features": lambda intraday: intraday,
            "add_feature_provenance": lambda weekly, intraday, daily: (weekly, intraday, {}),
            "select_weekly_universe": lambda weekly, config: self.universe,
            "train_all_rules": train,
            "evaluate_all_folds": evaluate,
            "write_final_report": lambda *args: order.append("report"),
        }
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(mock.patch.object(runner, name, side_effect=replacement))
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner.main()
        self.assertEqual(order, ["train_all", "outer", "report"])


if __name__ == "__main__":
    unittest.main()
