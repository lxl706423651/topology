"""复用冻结核心模型，以真实执行合格的内部候选开启年度做T回测。

从quant根目录读取v9 YAML、v8训练报告和既有只读行情缓存，不重新训练或构建
未来标签。统计验收只作参考，亏损候选也可选；全部规则先冻结再评价外层。
仅新建独立报告，保留T+1、股数、资金、成本、容量与恢复失败的真实记录。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.weekly_t_activation import select_active_candidate
from quant_backtest.weekly_t_active_reporting import write_active_report
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_features import add_session_context_features, add_weekly_execution_features
from quant_backtest.weekly_t_horizon_data import validate_horizon_config
from quant_backtest.weekly_t_provenance import add_feature_provenance
from quant_backtest.weekly_t_quote_store import DenseRecoveryBarStore, calculate_file_sha256
from quant_backtest.weekly_t_residual import predict_residual_t, validate_residual_config
from quant_backtest.weekly_t_short_horizon import ShortHorizonTStrategy, create_short_decisions
from scripts.run_weekly_t_cross_validation import save_result, select_execution_data, select_weekly_universe, write_frame
from scripts.run_weekly_t_residual import attach_frozen_context, snapshot_residual_sources
from scripts.run_weekly_t_short_horizon import read_frozen_selections

VERSION = "weekly_t_v9_active_execution_valid"
PATH_FIELDS = ("data_root", "cache_dir", "output_dir", "recovery_cache_dir", "horizon_cache_dir",
               "dense_quote_cache_dir", "frozen_selection_report_dir", "source_training_report_dir")


def load_configuration(argv: list[str] | None = None) -> dict:
    """解析YAML及仅影响运行范围/输入路径的CLI；smoke不改变模型、费用或信号。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/weekly_t_cross_validation_v9.yaml")
    parser.add_argument("--source-training-report-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--smoke-weeks", type=int)
    parser.add_argument("--test-years")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("source_training_report_dir", "output_dir", "max_symbols", "smoke_weeks"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    if args.test_years:
        cfg["test_years"] = [int(y) for y in args.test_years.split(",")]
    for key in PATH_FIELDS:
        cfg[key] = str((PROJECT_ROOT / cfg[key]).resolve())
    cfg["dense_recovery_cache_paths"] = {
        str(year): str((PROJECT_ROOT / path).resolve()) for year, path in cfg["dense_recovery_cache_paths"].items()}
    if cfg.get("research_version") != VERSION or cfg.get("candidate_activation_policy") != "active_execution_valid":
        raise ValueError("本入口要求v9 active_execution_valid策略，统计要求不能控制开启")
    if "candidate_support_requirements" in cfg:
        raise ValueError("v9不接收旧统计停用配置；旧值保留在源训练报告中供参考")
    if cfg["max_symbols"] < 0 or cfg["smoke_weeks"] < 0:
        raise ValueError("smoke范围不得为负")
    if not cfg["test_years"] or len(set(cfg["test_years"])) != len(cfg["test_years"]):
        raise ValueError("test_years必须非空且无重复")
    validate_residual_config(cfg)
    _, _, minutes = validate_horizon_config(cfg)
    cfg["scheduled_recovery_times"] = [f"{value // 60:02d}:{value % 60:02d}" for value in minutes]
    return cfg


def validate_training_source(cfg: dict) -> dict:
    """只允许启用政策、输出及评价范围变化；训练、特征、费用和信号必须相同。"""
    source = Path(cfg["source_training_report_dir"])
    previous = json.loads((source / "resolved_config.json").read_text())
    if previous.get("research_version") != "weekly_t_v8_residual_zero_market":
        raise ValueError("冻结训练来源必须为已完成的v8残差实验")
    allowed = {"research_version", "candidate_activation_policy", "source_training_report_dir",
               "dense_recovery_cache_paths", "candidate_support_requirements", "output_dir",
               "max_symbols", "smoke_weeks", "test_years"}
    changed = [key for key in set(cfg) | set(previous)
               if key not in allowed and cfg.get(key) != previous.get(key)]
    if changed:
        raise ValueError(f"不能复用与当前经济/训练配置不一致的内部结果：{sorted(changed)}")
    if not set(cfg["test_years"]).issubset(previous["test_years"]):
        raise ValueError("留出年份不在冻结训练报告中")
    hashes = json.loads((source / "source_hashes.json").read_text())
    for name, expected in hashes.items():
        if calculate_file_sha256(PROJECT_ROOT / name) != expected:
            raise ValueError(f"与源训练实验的公共实现不一致：{name}")
        if calculate_file_sha256(source / "source_snapshot" / Path(name).name) != expected:
            raise ValueError(f"源训练快照身份不一致：{name}")
    for name in ("independent_residual_audit.json", "all_rules_frozen_before_outer_test.json"):
        if not (source / name).is_file():
            raise ValueError(f"缺少已完成训练审计：{name}")
    audit = json.loads((source / "independent_residual_audit.json").read_text())
    if not all(audit.get(key) is True for key in ("passed", "complete", "full_configured_scope_complete")):
        raise ValueError("源训练审计未完整通过，不允许复用其候选或权重")
    return {"source_report": str(source), "configuration_compatible": True,
            "source_implementation_files_verified": len(hashes), "no_model_refit": True,
            "source_model_audit_sha256": calculate_file_sha256(source / "independent_residual_audit.json")}


def snapshot_active_sources(directory: Path) -> None:
    """冻结既有20份实现及本轮启用/报告入口，不改旧实验源码。"""
    snapshot_residual_sources(directory)
    hashes = json.loads((directory / "source_hashes.json").read_text())
    for path in (Path(__file__), PROJECT_ROOT / "quant_backtest/weekly_t_activation.py",
                 PROJECT_ROOT / "quant_backtest/weekly_t_active_reporting.py"):
        shutil.copyfile(path, directory / "source_snapshot" / path.name)
        hashes[str(path.relative_to(PROJECT_ROOT))] = calculate_file_sha256(path)
    write_json(directory / "source_hashes.json", hashes)


def record_input(path: Path, inputs: dict) -> None:
    """记录只读输入的完整身份，运行结束重新核对内容，路径不从交易逻辑推断。"""
    path = path.resolve()
    inputs[str(path)] = {"sha256": calculate_file_sha256(path), "size_bytes": path.stat().st_size}


def freeze_active_rules(cfg: dict, directory: Path, inputs: dict) -> tuple[list, pd.DataFrame]:
    """仅读取本折内部候选选择有交易的执行合格模型；冻结所有折后才允许外评。"""
    source = Path(cfg["source_training_report_dir"])
    audited = json.loads((source / "independent_residual_audit.json").read_text())
    folds, tables = [], []
    for year in cfg["test_years"]:
        origin, out = source / f"year_{year}", directory / f"year_{year}"
        out.mkdir()
        metrics_path = origin / "validation_rule_metrics.csv"
        record_input(metrics_path, inputs)
        record_input(origin / "split.json", inputs)
        fold = json.loads((origin / "split.json").read_text())
        if fold["test_year"] != year or year in fold["train_years"] or fold["inner_year"] in fold["core_years"]:
            raise ValueError("冻结年度拆分与core/inner/outer合同冲突")
        if sorted(fold["train_years"]) != sorted(fold["core_years"] + [fold["inner_year"]]):
            raise ValueError("核心年与内部年必须恰好组成训练集合")
        rows = pd.read_csv(metrics_path, float_precision="round_trip")
        if not rows.test_year.eq(year).all() or not rows.inner_year.eq(fold["inner_year"]).all():
            raise ValueError("候选评价年份与本折不一致")
        chosen, annotated = select_active_candidate(rows)
        horizon, profile = int(chosen["horizon_minutes"]), chosen["profile"]
        model_name = f"inner_t_h{horizon}_{profile}"
        model_source = origin / model_name
        identity = audited["models"][f"year_{year}/{model_name}"]
        if identity.get("passed") is not True:
            raise ValueError("所选核心模型的独立审计未通过")
        for filename, key in (("model_metadata.json", "metadata_sha256"),
                              ("model_bundle.joblib", "model_bundle_sha256"),
                              ("return_model.txt", "native_model_sha256")):
            if calculate_file_sha256(model_source / filename) != identity[key]:
                raise ValueError(f"所选核心模型偏离原审计身份：{filename}")
        metadata = json.loads((model_source / "model_metadata.json").read_text())
        if (sorted(metadata["train_years"]) != sorted(fold["core_years"])
                or sorted(metadata["excluded_years"]) != sorted([year, fold["inner_year"]])
                or metadata["profile"] != profile):
            raise ValueError("候选核心权重的训练/排除年份或profile不一致")
        model_cfg = metadata["resolved_config"]
        if int(model_cfg["t_label_horizon_minutes"]) != horizon:
            raise ValueError("模型标签周期与候选不一致")
        for path in sorted(model_source.rglob("*")):
            if path.is_file():
                record_input(path, inputs)
        shutil.copytree(model_source, out / "t_model")
        support_path = origin / f"support_h{horizon}.json"
        support_identity = audited["supports"][f"year_{year}/support_h{horizon}.json"]
        if (support_identity.get("passed") is not True
                or calculate_file_sha256(support_path) != support_identity["support_file_sha256"]):
            raise ValueError("所选输入支持域偏离原审计身份")
        record_input(support_path, inputs)
        shutil.copyfile(support_path, out / "selected_support.json")
        shutil.copyfile(origin / "split.json", out / "split.json")
        rule = {"enabled": True, "horizon_minutes": horizon, "profile": profile,
                "net_margin": float(chosen["net_margin"]), "fraction": float(chosen["fraction"]),
                "direction": chosen["direction"], "fold": fold, "source_model": model_name,
                "source_model_path": str(model_source.resolve()),
                "model_bundle_sha256": calculate_file_sha256(out / "t_model/model_bundle.joblib"),
                "candidate_index": int(chosen["candidate_index"]),
                "inner_incremental_net_pnl": float(chosen["incremental_net_pnl"]),
                "inner_complete_pairs": int(chosen["t_pair_count"]),
                "statistical_gates_are_informational": True,
                "activation_policy": cfg["candidate_activation_policy"],
                "frozen_before_any_outer_test": True}
        write_json(out / "selected_rule.json", rule)
        write_frame(annotated, out / "validation_rule_metrics.csv")
        tables.append(annotated)
        folds.append(fold)
        print(f"[{year}] 开启T，内部候选{rule['candidate_index']} h{horizon}/{profile}，"
              f"净边际{rule['net_margin']:.4f}，内部{rule['inner_complete_pairs']}对/"
              f"{rule['inner_incremental_net_pnl']:.2f}元；统计条件不控制开关", flush=True)
    combined = pd.concat(tables, ignore_index=True)
    write_frame(combined, directory / "validation_rule_metrics.csv")
    write_frame(pd.DataFrame(folds), directory / "split_audit.csv")
    write_json(directory / "all_rules_frozen_before_outer_test.json", {
        "created_at": datetime.now().isoformat(), "test_years": cfg["test_years"],
        "all_t_enabled": True, "selection_uses_inner_only": True,
        "selected_rule_sha256": {str(f["test_year"]): calculate_file_sha256(
            directory / f"year_{f['test_year']}/selected_rule.json") for f in folds}})
    return folds, combined


def load_frozen_market_data(cfg: dict, directory: Path, inputs: dict) -> tuple:
    """复用三张原行情缓存，先构造完整同刻截面及完整历史来源，再选择交易股票。"""
    source = Path(cfg["source_training_report_dir"])
    manifest_path = source / "data_manifest.json"
    record_input(manifest_path, inputs)
    manifest = json.loads(manifest_path.read_text())
    cache = Path(manifest["cache_path"])
    tables = []
    for name in ("weekly", "intraday", "daily"):
        path = cache / f"{name}.pkl"
        record_input(path, inputs)
        frame = pd.read_pickle(path)
        if len(frame) != manifest[f"{name}_rows"]:
            raise ValueError(f"缓存行数与冻结来源不一致：{name}")
        tables.append(frame)
    weekly, intraday, daily = tables
    del tables
    # 可以删掉完全不评价的年度，但不能先删某一时刻的截面成员；daily保留完整历史。
    if set(cfg["test_years"]) != set(intraday.datetime.dt.year.unique()):
        intraday = intraday.loc[intraday.datetime.dt.year.isin(cfg["test_years"])].copy()
    weekly = add_weekly_execution_features(weekly, daily)
    intraday = add_session_context_features(intraday)
    weekly, intraday, provenance = add_feature_provenance(weekly, intraday, daily)
    write_json(directory / "data_manifest.json", {**manifest, "active_run_cache_reused": True,
               "full_cross_section_before_execution_smoke_slice": True})
    write_json(directory / "feature_provenance_manifest.json", provenance)
    return select_weekly_universe(weekly, cfg), intraday, daily


def evaluate_active_folds(folds: list, universe: pd.DataFrame, intraday: pd.DataFrame,
                          daily: pd.DataFrame, cfg: dict, directory: Path, inputs: dict) -> tuple:
    """执行已冻结规则，原价格/费用/恢复引擎不变；无成交和执行失败均明确记录。"""
    marker = json.loads((directory / "all_rules_frozen_before_outer_test.json").read_text())
    if marker["test_years"] != [f["test_year"] for f in folds]:
        raise ValueError("外评必须等待全部目标年度冻结")
    for year in cfg["test_years"]:
        if calculate_file_sha256(directory / f"year_{year}/selected_rule.json") != marker["selected_rule_sha256"][str(year)]:
            raise ValueError("冻结规则在外评前变化")
    source, weekly_source = Path(cfg["source_training_report_dir"]), Path(cfg["frozen_selection_report_dir"])
    summaries, results = [], []
    for fold in folds:
        year = fold["test_year"]
        origin, out = source / f"year_{year}", directory / f"year_{year}"
        rule = json.loads((out / "selected_rule.json").read_text())
        if rule["enabled"] is not True:
            raise ValueError("主动入口不接受关闭规则")
        support = json.loads((out / "selected_support.json").read_text())
        selection_path = origin / "weekly_selections.csv"
        record_input(selection_path, inputs)
        selected = read_frozen_selections(selection_path, set(daily.symbol), {**cfg, "max_symbols": 0, "smoke_weeks": 0})
        if cfg["smoke_weeks"]:
            selected = selected.loc[selected.trade_week.isin(sorted(selected.trade_week.unique())[:cfg["smoke_weeks"]])].copy()
        if cfg["max_symbols"]:
            symbols = sorted(selected.symbol.unique())[:cfg["max_symbols"]]
            selected = selected.loc[selected.symbol.isin(symbols)].copy()
        if selected.empty:
            raise ValueError("选定执行范围为空")
        write_frame(selected, out / "weekly_selections.csv")
        for name in ("weekly_predictions.csv", "weekly_prediction_metrics.csv"):
            record_input(origin / name, inputs)
            shutil.copyfile(origin / name, out / name)
        actual_daily, actual_intraday = select_execution_data(selected, daily, intraday)
        context_model = weekly_source / f"year_{year}/weekly_model"
        for path in context_model.iterdir():
            if path.is_file():
                record_input(path, inputs)
        actual_intraday = attach_frozen_context(actual_intraday, selected, universe, cfg, context_model, fold["train_years"])
        write_json(out / "test_weekly_context_manifest.json", actual_intraday.attrs["frozen_weekly_context_manifest"])
        bundle = joblib.load(out / "t_model/model_bundle.joblib")
        predictions = predict_residual_t(bundle, actual_intraday, cfg)
        decisions = create_short_decisions(predictions, rule, cfg, support)
        # 独立原生权重预测及删除未来字段复测，防止复制错模型或未来标签控制订单。
        native = lightgbm.Booster(model_file=str(out / "t_model/return_model.txt"))
        if native.feature_name() != bundle["features"]:
            raise AssertionError("原生权重的特征顺序与bundle不一致")
        native_inputs = predictions[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = native.predict(native_inputs) if len(predictions) else np.array([], dtype=float)
        if values.shape != (len(predictions),) or not np.isfinite(values).all():
            raise AssertionError("原生权重预测形状或有限性错误")
        error = float(np.max(np.abs(values - predictions.prediction))) if len(predictions) else 0.0
        if error > 1e-12:
            raise AssertionError(f"原生与序列化模型不一致：{error}")
        future = [c for c in actual_intraday if c.startswith(("execution_", "recovery_", "target_", "label_"))]
        scrubbed = actual_intraday.copy()
        scrubbed[future] = np.nan
        scrubbed_predictions = predict_residual_t(bundle, scrubbed, cfg)
        pd.testing.assert_series_equal(predictions.prediction, scrubbed_predictions.prediction, check_exact=True)
        pd.testing.assert_frame_equal(decisions, create_short_decisions(scrubbed_predictions, rule, cfg, support), check_exact=True)
        del scrubbed, scrubbed_predictions
        write_json(out / "prediction_causality_audit.json", {"native_prediction_max_error": error,
                   "prediction_rows": len(predictions), "future_fields_removed": future,
                   "prediction_and_decisions_unchanged_without_future_fields": True})
        write_frame(predictions[["datetime", "symbol", "prediction", "residual_prediction", "market_prediction"]], out / "t_predictions.csv")
        write_frame(decisions, out / "t_decisions.csv")
        signals = ShortHorizonTStrategy(rule, cfg, support).generate_signals(predictions)
        dense_path = Path(cfg["dense_recovery_cache_paths"][str(year)])
        record_input(dense_path / "metadata.json", inputs)
        store = DenseRecoveryBarStore(dense_path, verify_integrity=True)
        if not set(selected.symbol).issubset(store.symbols):
            raise ValueError("恢复行情未覆盖选定股票")
        write_json(out / "dense_recovery_reference.json", {"path": str(dense_path), "fingerprint": store.fingerprint,
                   "arrays_sha256_verified": True, "time_count": store.metadata["time_count"]})
        result = run_weekly_portfolio(selected, actual_daily, actual_intraday, decisions, cfg, recovery_bars=store)
        metrics = save_result(result, out, cfg)
        if not cfg["max_symbols"] and not cfg["smoke_weeks"]:
            record_input(origin / "test_equity_curve.csv", inputs)
            original_curve = pd.read_csv(origin / "test_equity_curve.csv", parse_dates=["datetime"], float_precision="round_trip")
            curve = result["equity_curve"]
            if not curve.datetime.reset_index(drop=True).equals(original_curve.datetime.reset_index(drop=True)):
                raise AssertionError("原底仓基准日期改变")
            baseline_error = float(np.max(np.abs(curve.baseline_equity.to_numpy() - original_curve.baseline_equity.to_numpy())))
            if baseline_error > 1e-8:
                raise AssertionError("原底仓基准权益改变")
            write_json(out / "frozen_baseline_audit.json", {"same_dates": True, "max_equity_error_rmb": baseline_error})
        activity = "enabled_with_trades" if metrics["t_pair_count"] else "enabled_no_completed_pairs"
        summaries.append({"test_year": year, "inner_year": fold["inner_year"], **metrics,
                          "t_enabled": True, "t_activity_status": activity,
                          "prediction_count": len(predictions), "signal_count": int(signals.t_action.ne(0).sum()),
                          "request_count": len(decisions), "selected_candidate_index": rule["candidate_index"]})
        results.append((year, result))
        print(f"[{year}] enabled=True，{len(decisions)}请求/{metrics['t_pair_count']}完整对，"
              f"净T {metrics['t_added_equity']:.6f}元，股数差{metrics['inventory_mismatch_days']}日", flush=True)
        del store, predictions, bundle, native, actual_intraday
        gc.collect()
    return pd.DataFrame(summaries), results


def main(argv: list[str] | None = None) -> None:
    """先声明用户授权的启用政策，冻结内层选参，再推断/执行/保存独立报告。"""
    cfg = load_configuration(argv)
    source_audit = validate_training_source(cfg)
    directory = Path(cfg["output_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True)
    snapshot_active_sources(directory)
    write_json(directory / "resolved_config.json", cfg)
    write_json(directory / "training_reuse_audit.json", source_audit)
    write_json(directory / "research_protocol.json", {
        "research_version": VERSION, "created_before_this_run_outer_test": True,
        "previous_outer_results_seen": True, "user_authorized_statistical_gate_removal": True,
        "activation_policy": cfg["candidate_activation_policy"],
        "selection": "仅inner：完整pair>0且原5执行失败计数=0，净增量降序/候选序号升序；负收益也可选",
        "statistics": "原100配对/20日/8周/20股、CI、集中度、正收益条件不决定开启，仅保留参考",
        "no_eligible_candidate": "显式报错，不静默关闭、不选0交易、不跳过执行要求",
        "outer_execution_order": "所有目标年度规则先冻结，随后统一推断/执行外层",
        "source_training_report_dir": cfg["source_training_report_dir"], "no_model_refit": True,
        "future_data_not_used_to_select": True, "execution_engine_unchanged": True,
        "smoke_scope": "完整同刻特征后，每折先取前smoke_weeks周，再取按代码排序前max_symbols只已选股票"})
    print(f"输出目录：{directory}", flush=True)
    inputs = {}
    for name in ("resolved_config.json", "source_hashes.json", "independent_residual_audit.json"):
        record_input(Path(cfg["source_training_report_dir"]) / name, inputs)
    folds, validation = freeze_active_rules(cfg, directory, inputs)
    universe, intraday, daily = load_frozen_market_data(cfg, directory, inputs)
    annual, results = evaluate_active_folds(folds, universe, intraday, daily, cfg, directory, inputs)
    write_json(directory / "input_sources.json", inputs)
    for path, identity in inputs.items():
        if calculate_file_sha256(Path(path)) != identity["sha256"]:
            raise AssertionError(f"运行期间输入变化：{path}")
    hashes = json.loads((directory / "source_hashes.json").read_text())
    for name, expected in hashes.items():
        if calculate_file_sha256(PROJECT_ROOT / name) != expected:
            raise AssertionError(f"运行期间源码变化：{name}")
    write_json(directory / "runtime_integrity_audit.json", {"input_files_verified": len(inputs),
               "runtime_sources_verified": len(hashes), "all_unchanged": True})
    write_active_report(directory, cfg, annual, results, validation)
    print(f"报告完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
