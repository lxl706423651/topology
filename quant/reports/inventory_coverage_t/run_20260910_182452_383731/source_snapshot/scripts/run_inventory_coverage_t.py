"""永久底仓15分钟特征研究：原L2训练覆盖门槛固定由0.80改为0.70。

保留v15的20/40/60分钟、原37数字候选、永久底仓、费用和T+1执行。
完整纯core每折新拟合六模型；原训练行和标签保持，重新选择特征并保存真实矩阵。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gc
import json
import multiprocessing
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
from quant_backtest.balanced_t_data import (INTRADAY_COLUMNS, read_balanced_fold,
    record_balanced_input, validate_balanced_sources)
from quant_backtest.balanced_t_policy import select_balanced_candidate
from quant_backtest.guarded_t_engine import run_guarded_portfolio
from quant_backtest.inventory_t_coverage_policy import (build_inventory_coverage_candidates,
    generate_inventory_coverage_decisions, create_inventory_coverage_rule_config)
from quant_backtest.inventory_t_coverage_models import (VERSION, PROFILES, HORIZONS, SOURCE_PROFILE_MAP,
    fit_inventory_coverage_model, predict_inventory_coverage_model, validate_coverage_model_config)
from quant_backtest.inventory_t_learning_models import attach_learning_labels
from quant_backtest.weekly_t_timing_data import load_labels_only
from quant_backtest.inventory_t_frozen_cohort import load_frozen_inventory_core_cohort
from quant_backtest.inventory_t_coverage_reporting import write_coverage_inventory_report
from quant_backtest.weekly_t_quote_store import calculate_file_sha256
from quant_backtest.weekly_t_timing import add_timing_features
from scripts.run_dynamic_inventory_t import (prepare_account_context,
    get_store, write_frame)
from scripts.run_weekly_t_active import load_frozen_market_data
from scripts.run_inventory_guarded_t import (load_configuration as load_guarded_configuration,
    snapshot_sources as snapshot_guarded_sources, assert_source_table_equal, save_guarded_result)


_WORKER = {}


def find_complete_source_audit(source: Path, kind: str, version: str) -> tuple[Path, dict]:
    """选择完整通过的来源独审；实际摘要另行冻结，不能用进度记录当通过。"""
    matches = []
    for path in sorted(source.glob(f"independent_{kind}_audit/run_*/verification.json")):
        data = json.loads(path.read_text())
        if (data.get("research_version") == version and all(data.get(key) is True
                for key in ("passed", "complete", "all_read_inputs_unchanged"))):
            matches.append((path, data))
    if not matches:
        raise ValueError(f"来源尚未完成独立{kind}审计：{source}")
    return matches[-1]


def load_configuration(argv: list[str] | None = None) -> dict:
    """严格继承v15经济、期限、评价范围；只改变训练特征覆盖门槛。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/inventory_coverage_t_v18.yaml")
    parser.add_argument("--source-report")
    parser.add_argument("--test-years", type=int, nargs="+")
    parser.add_argument("--candidate-workers", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--internal-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = (PROJECT_ROOT/args.config).resolve()
    declared = yaml.safe_load(config_path.read_text())
    keys = {"research_version", "source_execution_report_dir", "output_dir", "training_loss",
            "feature_min_coverage", "source_feature_min_coverage", "t_recheck_exit_side_at_execution",
            "candidate_workers", "workers"}
    if (not isinstance(declared, dict) or set(declared) != keys or declared["research_version"] != VERSION
            or declared["training_loss"] != "regression_l2"
            or type(declared["feature_min_coverage"]) not in (int, float) or declared["feature_min_coverage"] != .7
            or type(declared["source_feature_min_coverage"]) not in (int, float) or declared["source_feature_min_coverage"] != .8
            or declared["t_recheck_exit_side_at_execution"] is not True):
        raise ValueError("v18配置只能声明原L2的覆盖门槛0.80至0.70及原guard执行合同")
    source = (PROJECT_ROOT/(args.source_report or declared["source_execution_report_dir"])).resolve()
    original = json.loads((source/"resolved_config.json").read_text())
    if original.get("research_version") != "inventory_guarded_t_v15":
        raise ValueError("执行及基准来源必须是已完成的v15研究")
    if original.get("feature_min_coverage") != .8:
        raise ValueError("原L2执行对照的训练覆盖门槛必须为0.80")
    old_args = ["--config", original["configuration_source"], "--source-report",
                original["source_learning_report_dir"], "--test-years", *map(str, original["test_years"])]
    for key in ("candidate_workers", "workers", "output_dir"):
        old_args += ["--"+key.replace("_", "-"), str(original[key])]
    if original["internal_only"]:
        old_args.append("--internal-only")
    if load_guarded_configuration(old_args) != original:
        raise ValueError("来源配置不能由原v15合同精确复现")
    years = args.test_years or original["test_years"]
    if not years or len(set(years)) != len(years) or not set(years).issubset(original["test_years"]):
        raise ValueError("年份须为来源已研究年份的非空无重复子集")
    cohort_source = Path(original["source_learning_report_dir"])
    cohort_audit, _ = find_complete_source_audit(cohort_source, "learning", "inventory_learning_t_v14")
    cfg = {**original, **declared, "configuration_source": str(config_path),
        "source_execution_report_dir": str(source), "source_execution_version": original["research_version"],
        "source_configuration_source": original["configuration_source"],
        "source_learning_report_dir": str(cohort_source), "source_cohort_version": "inventory_learning_t_v14",
        "source_learning_audit": str(cohort_audit),
        "source_learning_audit_sha256": calculate_file_sha256(cohort_audit),
        "source_report_scope": original["report_scope"], "frozen_t_models_reused": False,
        "t_models_fitted_this_run": True, "test_years": years, "internal_only": args.internal_only,
        "source_learning_model_profiles": original["learning_model_profiles"],
        "learning_model_profiles": list(PROFILES), "forecast_target": "fixed_20_40_60_reference_return",
        "output_dir": str((PROJECT_ROOT/(args.output_dir or declared["output_dir"])).resolve())}
    for key in ("source_learning_version",):
        cfg.pop(key)
    for key in ("candidate_workers", "workers"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("工作进程数必须为正整数")
    build_inventory_coverage_candidates(cfg)
    for profile in cfg["learning_model_profiles"]:
        for horizon in HORIZONS:
            validate_coverage_model_config({**cfg, "t_model_profile": profile, "t_label_horizon_minutes": horizon})
    return cfg


def snapshot_sources(directory: Path, cfg: dict) -> None:
    """保留原51源不变，再绑定完整机会复用与四个独立特征覆盖运行源。"""
    snapshot_guarded_sources(directory, cfg)
    hashes = json.loads((directory/"source_hashes.json").read_text())
    for name in ("scripts/run_inventory_coverage_t.py", "quant_backtest/inventory_t_frozen_cohort.py", "quant_backtest/inventory_t_coverage_models.py",
                 "quant_backtest/inventory_t_coverage_policy.py", "quant_backtest/inventory_t_coverage_reporting.py"):
        target = directory/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT/name, target)
        hashes[name] = calculate_file_sha256(PROJECT_ROOT/name)
    write_json(directory/"source_hashes.json", hashes)


def validate_execution_source(cfg: dict, directory: Path, inputs: dict) -> None:
    """核已审计执行对照、原数据与旧模型身份；只复用队列及基准，不加载旧T权重。"""
    source = Path(cfg["source_execution_report_dir"])
    audit_path, audit = find_complete_source_audit(source, "guarded", "inventory_guarded_t_v15")
    record_balanced_input(audit_path, inputs)
    for name in ("resolved_config.json", "runtime_integrity_audit.json", "source_hashes.json", "input_sources.json",
                 "all_rules_frozen_before_outer_test.json", "research_protocol.json"):
        path = (source/name).resolve()
        expected = audit["inputs_sha256"].get(str(path))
        if expected is None:
            raise ValueError(f"执行来源独审未绑定关键文件：{path}")
        record_balanced_input(path, inputs, expected)
    integrity = json.loads((source/"runtime_integrity_audit.json").read_text())
    if integrity.get("all_unchanged") is not True:
        raise ValueError("执行来源运行未完成完整性核验")
    original_inputs = json.loads((source/"input_sources.json").read_text())
    for name, identity in original_inputs.items():
        record_balanced_input(name, inputs, identity["sha256"])
    hashes = json.loads((source/"source_hashes.json").read_text())
    for name, digest in hashes.items():
        record_balanced_input(PROJECT_ROOT/name, inputs, digest)
        record_balanced_input(source/"source_snapshot"/name, inputs, digest)
    frozen = json.loads((source/"all_rules_frozen_before_outer_test.json").read_text())
    for name, digest in frozen["model_artifacts_sha256"].items():
        record_balanced_input(Path(name), inputs, digest)
    write_json(directory/"execution_source_reuse_audit.json", {"passed": True,
        "source_report": str(source), "independent_audit": str(audit_path),
        "independent_audit_sha256": calculate_file_sha256(audit_path),
        "original_inputs_verified": len(original_inputs), "runtime_sources_verified": len(hashes),
        "source_t_models_identity_verified": len(frozen["model_artifacts_sha256"]),
        "source_t_models_used_for_prediction": False, "new_coverage_models_fitted_per_fold": 6,
        "training_loss": "regression_l2", "feature_min_coverage": .7, "source_feature_min_coverage": .8,
        "same_training_rows_and_labels_required": True, "same_selected_features_required": False,
        "same_baseline_required": True, "same_predictions_and_decisions_required": False,
        "forecast_target_and_deadline_unchanged": True, "only_feature_coverage_threshold_changed": True})


def compare_training_to_source(model_dir: Path, source_model: Path, out: Path, inputs: dict) -> None:
    """核同核心重建的0.80清单、原训练行/标签/隔离与共同列；允许0.70选择改变矩阵。"""
    evidence = {}
    pairs = (("source_training_inputs.pkl", "training_inputs.pkl"),
             ("source_feature_manifest.csv", "feature_manifest.csv"),
             ("training_labels.csv", "training_labels.csv"),
             ("training_split_audit.csv", "training_split_audit.csv"))
    for current_name, source_name in pairs:
        record_balanced_input(source_model/source_name, inputs)
        if source_name.endswith(".pkl"):
            current, original = pd.read_pickle(model_dir/current_name), pd.read_pickle(source_model/source_name)
        else:
            current = pd.read_csv(model_dir/current_name, float_precision="round_trip")
            original = pd.read_csv(source_model/source_name, float_precision="round_trip")
        pd.testing.assert_frame_equal(current, original, check_dtype=True, check_exact=True)
        evidence[source_name] = {"current_file": current_name,
            "current_sha256": calculate_file_sha256(model_dir/current_name),
            "source_sha256": calculate_file_sha256(source_model/source_name), "values_and_dtype_exact": True}
    record_balanced_input(source_model/"model_metadata.json", inputs)
    current = json.loads((model_dir/"model_metadata.json").read_text())
    original = json.loads((source_model/"model_metadata.json").read_text())
    if (SOURCE_PROFILE_MAP[current["profile"]] != original["profile"]
            or current["horizon_minutes"] != original["horizon_minutes"]):
        raise AssertionError("新覆盖模型与原L2架构/期限映射不符")
    for name in ("train_years", "excluded_years", "fitted_rows", "full_training_rows_sha256", "training_target_sha256"):
        if current[name] != original[name]:
            raise AssertionError(f"原核心训练行或标签改变：{name}")
    if (current["source_features"] != original["features"]
            or current["source_training_matrix_sha256"] != original["training_matrix_sha256"]
            or current.get("feature_min_coverage") != .7 or current.get("source_feature_min_coverage") != .8):
        raise AssertionError("同核心重建的原特征身份或覆盖门槛不符")
    actual_inputs = pd.read_pickle(model_dir/"training_inputs.pkl")
    source_inputs = pd.read_pickle(source_model/"training_inputs.pkl")
    keys = ["datetime", "symbol", "feature_source_start", "cohort_target_year", "baseline_opening_shares"]
    if (list(actual_inputs) != keys+current["features"]
            or list(source_inputs) != keys+original["features"]):
        raise AssertionError("真实训练输入列序与模型清单不一致")
    common = [name for name in original["features"] if name in current["features"]]
    pd.testing.assert_frame_equal(actual_inputs[keys+common], source_inputs[keys+common],
                                  check_dtype=True, check_exact=True)
    left, right = current["model_parameters"], original["model_parameters"]
    if left.get("objective") != "regression" or right.get("objective") != "regression":
        raise AssertionError("覆盖对照必须保持原L2损失")
    # n_jobs是显式运行并行度，允许CLI调整；其余训练参数必须逐项与源相同。
    ignored = {"n_jobs"}
    if {k:v for k,v in left.items() if k not in ignored} != {k:v for k,v in right.items() if k not in ignored}:
        raise AssertionError("除显式CPU并行度外，原模型训练参数改变")
    write_json(out, {"passed": True, "source_model": str(source_model), "current_model": str(model_dir),
        "source_profile": original["profile"], "current_profile": current["profile"],
        "horizon_minutes": current["horizon_minutes"], "source_objective": "regression", "current_objective": "regression",
        "feature_min_coverage": .7, "source_feature_min_coverage": .8, "labels_winsorized_or_deleted": False,
        "reconstructed_source_selection_labels_and_split_exact": evidence,
        "current_features": current["features"], "source_features": original["features"],
        "added_features": [name for name in current["features"] if name not in original["features"]],
        "removed_features": [name for name in original["features"] if name not in current["features"]],
        "common_features": common, "common_feature_values_and_dtype_exact": True,
        "same_selected_feature_matrix_required": False,
        "source_workers": right["n_jobs"], "current_workers": left["n_jobs"]})


def predict_and_audit(frame: pd.DataFrame, model_dir: Path, cfg: dict, out: Path, inputs: dict) -> pd.DataFrame:
    """保存每条当前输入与预测；独立原生重载及擦除未来字段复算必须精确相等。"""
    out.mkdir(parents=True, exist_ok=False)
    for name in ("model_bundle.joblib", "return_model.txt", "model_metadata.json"):
        record_balanced_input(model_dir/name, inputs)
    metadata = json.loads((model_dir/"model_metadata.json").read_text())
    for name, key in (("model_bundle.joblib", "model_bundle_sha256"), ("return_model.txt", "native_model_sha256")):
        if calculate_file_sha256(model_dir/name) != metadata[key]:
            raise AssertionError("预测前模型文件身份与拟合元数据不一致")
    split_path = model_dir.parent.parent/"split.json"
    record_balanced_input(split_path, inputs)
    split = json.loads(split_path.read_text())
    if (metadata["train_years"] != sorted(split["core_years"])
            or metadata["excluded_years"] != sorted([split["test_year"], split["inner_year"]])):
        raise AssertionError("模型核心/排除年份与当前折不一致")
    bundle = joblib.load(model_dir/"model_bundle.joblib")
    if json.loads(json.dumps(bundle["metadata"], default=str)) != {
            key: value for key, value in metadata.items() if key != "model_bundle_sha256"}:
        raise AssertionError("bundle与外部模型元数据不一致")
    predictions = predict_inventory_coverage_model(bundle, frame, cfg)
    future = [name for name in frame if name.startswith(("execution_", "recovery_", "target_", "label_"))]
    repeated = predict_inventory_coverage_model(bundle, frame.drop(columns=future), cfg)
    pd.testing.assert_series_equal(predictions.prediction, repeated.prediction, check_exact=True)
    native = lightgbm.Booster(model_file=str(model_dir/"return_model.txt"))
    matrix = frame[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = np.asarray(native.predict(matrix) if len(matrix) else np.array([]))
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise AssertionError("原生重载输出必须逐行且全为有限数")
    error = float(np.max(np.abs(values-predictions.prediction.to_numpy()))) if len(frame) else 0.
    if error > 1e-12 or native.feature_name() != bundle["features"]:
        raise AssertionError("原生模型不能精确复现全部当前预测")
    fields = ["datetime", "symbol", "prediction", "raw_prediction", "model_profile", "horizon_minutes",
              "return_score", "model_version", "forecast_loss", "feature_min_coverage",
              "signal_price", "signal_high", "signal_low", "signal_amount", "signal_volume", "previous_close"]
    archived = list(dict.fromkeys(fields+bundle["features"]+["feature_source_start"]))
    write_frame(predictions[archived], out/"current_predictions.csv")
    write_json(out/"prediction_audit.json", {"passed": True, "current_rows": len(frame), "horizon_minutes": cfg["t_label_horizon_minutes"],
        "forecast_loss": "regression_l2", "feature_min_coverage": .7, "score_is_calibrated_conditional_mean": False,
        "future_fields_removed": future, "future_independent": True, "native_max_error": error,
        "features": bundle["features"], "model_bundle_sha256": calculate_file_sha256(model_dir/"model_bundle.joblib"),
        "predictions_sha256": calculate_file_sha256(out/"current_predictions.csv")})
    return predictions[fields].copy()


def predict_account_models(signals, fold_out, cfg, out, inputs, profiles=None) -> dict:
    """只加载本折六个新L2核心权重；所有期限预测同一当前键集，不在评价年再拟合。"""
    forecasts = {}
    for profile in cfg["learning_model_profiles"] if profiles is None else profiles:
        forecasts[profile] = {}
        for horizon in HORIZONS:
            key = f"h{horizon}_{profile}"
            forecasts[profile][horizon] = predict_and_audit(signals, fold_out/"models"/key,
                {**cfg, "t_model_profile": profile, "t_label_horizon_minutes": horizon}, out/key, inputs)
    return forecasts


def write_source_comparison(current: pd.DataFrame, source_path: Path, out: Path, inputs: dict) -> pd.DataFrame:
    """同数字候选逐项核对L2架构；只比较各真实账户，不相加不同候选。"""
    record_balanced_input(source_path, inputs)
    original = pd.read_csv(source_path, float_precision="round_trip")
    key = "candidate_index"
    fields = [key, "enabled", "profile", "min_net_edge", "fraction", "exit_mode"]
    mapped = current[fields].reset_index(drop=True).copy()
    mapped["profile"] = mapped.profile.map({**SOURCE_PROFILE_MAP, "hold": "hold"})
    if mapped.profile.isna().any():
        raise AssertionError("存在无法对应原L2架构的候选")
    pd.testing.assert_frame_equal(mapped, original[fields], check_dtype=False, check_exact=True)
    values = ["incremental_net_pnl", "holding_cost_reduction_rate", "t_pair_count", "completed_t_pair_lots",
              "t_fee", "t_slippage_cost", "inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
              "all_reference_acquisition_cost"]
    combined = current[[key, *values]].merge(original[[key, *values]], on=key,
        validate="one_to_one", suffixes=("_coverage", "_source"))
    if not np.allclose(combined.all_reference_acquisition_cost_coverage,
                      combined.all_reference_acquisition_cost_source, atol=1e-8, rtol=0):
        raise AssertionError("新特征覆盖研究改变了原底仓取得成本分母")
    for name in values:
        combined[name+"_change"] = combined[name+"_coverage"]-combined[name+"_source"]
    combined["current_profile"] = current.profile.to_numpy()
    combined["source_profile"] = original.profile.to_numpy()
    combined["same_forecast_target"] = True
    combined["same_forecasts"] = False
    combined["same_execution_guard"] = True
    combined["same_learning_objective"] = True
    combined["source_feature_min_coverage"] = .8
    combined["current_feature_min_coverage"] = .7
    write_frame(combined, out/"coverage_vs_source_candidates.csv")
    return combined


def initialize_worker(selected, daily, signals, predictions, cfg, store, baseline, directory):
    """候选进程只接收本账户当前信号和小型模型输出，共用只读完整分钟mmap。"""
    global _WORKER
    _WORKER = dict(selected=selected, daily=daily, signals=signals, predictions=predictions,
                   cfg=cfg, store=store, baseline=baseline, directory=Path(directory))


def evaluate_worker(rule: dict) -> dict:
    """执行预声明候选，逐日核无T基准一致，保存实际成交后才参与内部选参。"""
    c = _WORKER
    out = c["directory"]/f"candidate_{rule['candidate_index']:03d}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"rule.json", rule)
    forecasts = c["predictions"].get(rule["profile"], {})
    decisions, audit = generate_inventory_coverage_decisions(forecasts, rule, c["cfg"])
    write_frame(decisions, out/"t_decisions.csv")
    write_json(out/"signal_causality_audit.json", audit)
    local = create_inventory_coverage_rule_config(c["cfg"], rule)
    result = run_guarded_portfolio(c["selected"], c["daily"], c["signals"], decisions, local, recovery_bars=c["store"])
    curve = result["equity_curve"]
    if (not pd.to_datetime(curve.datetime).reset_index(drop=True).equals(c["baseline"].datetime)
            or not np.allclose(curve.baseline_equity, c["baseline"].equity, atol=1e-8, rtol=0)):
        raise AssertionError("不同T规则的无T基准发生改变")
    metrics = save_guarded_result(result, out, local)
    return {**metrics, **rule, "request_count": len(decisions), "candidate_directory": str(out)}


def write_internal_report(directory, folds, choices, targets_passed):
    """未晋级时仅报告完整内部结果，不制造外层成绩或外层收益图。"""
    lines = ["# 永久底仓15分钟特征覆盖：内部研究结果（v18）", "", "正式外层未执行。下表是内部年度选参结果，不是年度验证集成绩。", "",
        "| 留出年 | 内部年 | 规则 | 完整组 | 配对手 | T净额（元） | 成本降低率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in folds:
        item = choices[fold["test_year"]]["inner_metrics"]
        lines.append(f"| {fold['test_year']} | {fold['inner_year']} | {int(item['candidate_index'])} | {int(item['t_pair_count'])} | {item['completed_t_pair_lots']:.0f} | {item['incremental_net_pnl']:.2f} | {item['holding_cost_reduction_rate']:.4%} |")
    lines += ["", "原20/40/60分钟、训练行、原始标签、来源隔离及L2模型参数保持，仅将训练特征覆盖门槛从0.80改为0.70。原覆盖清单由相同核心数据重建后逐项核来源；新特征集合和矩阵允许改变，共同列数值与dtype保持。每折新拟合六模型；参考收益评分不是已校准的条件均值或实际净利润。原37数字候选、底仓和guard15执行条件不变。",
        "", "所有内部折达到5%底仓成本降低率后才进入统一外评。每股初始名义1万—2万元，底仓永久保留；没有回选历史外层结果。"]
    (directory/"final_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(directory/"goal_evaluation.json", {"goal_achieved": False, "formal_outer_evaluated": False,
        "all_inner_cost_targets_met": targets_passed, "reason": "internal_research_only"})


def main(argv: list[str] | None = None) -> None:
    """冻结单一覆盖门槛与原L2/短期限/执行条件；纯核心拟合后按原5%协议外评。"""
    cfg = load_configuration(argv)
    directory = Path(cfg["output_dir"])/f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory, cfg)
    write_json(directory/"resolved_config.json", cfg)
    rules = build_inventory_coverage_candidates(cfg)
    write_json(directory/"research_protocol.json", {"version": VERSION, "previous_historical_outer_results_seen": True,
        "candidate_rules": rules, "models_per_fold": 6, "models_fitted_this_run": 6*len(cfg["test_years"]),
        "frozen_t_models_reused": False, "outer_model_refit": False,
        "source_execution_report_dir": cfg["source_execution_report_dir"],
        "source_learning_report_dir": cfg["source_learning_report_dir"],
        "training_cohort": "core_only_oof_weekly_entry_purged_before_permanent_baseline_planning",
        "training_labels": "next_minute_to_fixed_20_40_60_minute_reference_return",
        "training_loss": "regression_l2", "source_loss": "regression_l2",
        "feature_min_coverage": .7, "source_feature_min_coverage": .8,
        "only_feature_coverage_threshold_changed": True,
        "same_training_rows_and_labels_required": True, "same_selected_features_required": False,
        "reconstructed_source_feature_selection_required": True,
        "native_missing_values_preserved": True, "coverage_threshold_selected_using_validation_returns": False,
        "labels_winsorized_or_deleted": False, "score_is_calibrated_conditional_mean": False,
        "cross_horizon_argmax": True, "labels_are_actual_shared_account_pnl": False,
        "selection_uses_inner_real_accounts_only": True, "no_weekly_middle_day_restriction": True,
        "no_probability_or_old_support_gate": True, "full_core_training_also_in_smoke": True,
        "execution_guard_unchanged_from_v15": True, "outer_evaluation_policy": cfg["outer_evaluation_policy"],
        "valuation_contract": "full_minute_reference_stream_then_daily_close",
        "sources": ["https://lightgbm.readthedocs.io/en/v4.6.0/Advanced-Topics.html#missing-value-handle"],
        "lower_coverage_threshold_not_proof_of_profit": True,
        "literature_not_a_profit_guarantee": True})
    print(f"输出目录：{directory}", flush=True)
    inputs, stores = {}, {}
    record_balanced_input(cfg["configuration_source"], inputs)
    validate_execution_source(cfg, directory, inputs)
    model_hashes = {}
    market_audit = validate_balanced_sources(cfg, inputs, PROJECT_ROOT)
    market_audit["pre_v14_legacy_t_models_reused"] = market_audit.pop("old_t_models_reused")
    market_audit["frozen_inventory_learning_t_models_reused"] = False
    market_audit["frozen_inventory_core_cohort_reused"] = True
    market_audit["holding_contract_matches_source_learning"] = True
    write_json(directory/"source_reuse_audit.json", market_audit)
    folds = [read_balanced_fold(cfg, year, inputs) for year in cfg["test_years"]]
    for fold in folds:
        fold["t_model_fit"] = True
        fold["core_cohort_reused_from"] = cfg["source_learning_report_dir"]
    needed = sorted({y for f in folds for y in [f["test_year"], f["inner_year"], *f["core_years"]]})
    print("复用完整因子与已审计纯核心机会；原L2和短期限标签不变，仅检验0.70覆盖筛选", flush=True)
    universe, intraday, daily = load_frozen_market_data({**cfg, "test_years": needed}, directory, inputs)
    known = json.loads((Path(cfg["source_audit_report_dir"])/"input_sources.json").read_text())
    for name in ("weekly", "intraday", "daily"):
        path = (Path(cfg["market_cache_dir"])/f"{name}.pkl").resolve()
        if str(path) not in known:
            raise ValueError("原市场PKL缺已审计身份")
        record_balanced_input(path, inputs, known[str(path)]["sha256"])
    intraday = add_timing_features(intraday, cfg)
    record_balanced_input(Path(cfg["source_training_report_dir"])/"independent_residual_audit.json", inputs)
    labels = load_labels_only(cfg, directory, inputs, record_balanced_input)
    del universe
    calendar = pd.DatetimeIndex(daily.date.unique()).sort_values()
    tables, comparisons, choices = [], [], {}
    source = Path(cfg["source_execution_report_dir"])
    for fold in folds:
        year = fold["test_year"]
        out = directory/f"year_{year}"
        out.mkdir()
        write_json(out/"split.json", fold)
        core, cohort_audit = load_frozen_inventory_core_cohort(intraday, fold, cfg,
            Path(cfg["source_learning_report_dir"]), out/"training_cohort", inputs, record_balanced_input)
        write_json(out/"training_cohort_summary.json", cohort_audit)
        print(f"[{year}] 全部永久旧股核心机会{len(core):,}行；开始6个覆盖门槛0.70的L2模型", flush=True)
        identity_dir = out/"training_identity"
        identity_dir.mkdir()
        for horizon in HORIZONS:
            training = attach_learning_labels(core, labels, horizon)
            for profile in PROFILES:
                key = f"h{horizon}_{profile}"
                model = out/"models"/key
                metadata = fit_inventory_coverage_model(training, fold["core_years"],
                    {**cfg, "t_model_profile": profile, "t_label_horizon_minutes": horizon}, model,
                    [fold["inner_year"], year])
                source_model = (Path(cfg["source_learning_report_dir"])/f"year_{year}"
                    /"models"/f"h{horizon}_{SOURCE_PROFILE_MAP[profile]}")
                compare_training_to_source(model, source_model, identity_dir/f"{key}.json", inputs)
                model_hashes.update({str(path.relative_to(directory)): calculate_file_sha256(path)
                    for path in sorted(model.iterdir()) if path.is_file()})
                print(f"[{year}] {key}拟合{metadata['fitted_rows']:,}行/{len(metadata['features'])}特征；原行/标签及0.80重建清单精确同源", flush=True)
            del training
        del core
        gc.collect()
        fold_cfg = {**cfg, "current_test_year": year}
        context = prepare_account_context(fold, "validation", daily, intraday, calendar, fold_cfg,
            out/"validation_context", inputs, get_store(fold["inner_year"], cfg, inputs, stores))
        selected, prices, signals, local_cfg, store = context
        forecasts = predict_account_models(signals, out, local_cfg, out/"validation_predictions", inputs)
        for relative in ("validation_context/frozen_base_selections.csv", "validation_context/frozen_base_entries.csv",
                         "validation_context/baseline/test_equity_curve.csv"):
            compare = pd.read_csv(out/relative, float_precision="round_trip")
            assert_source_table_equal(compare, source/f"year_{year}"/relative, inputs)
        baseline = pd.read_csv(out/"validation_context/baseline/test_equity_curve.csv",
                              parse_dates=["datetime"], float_precision="round_trip")[["datetime", "equity"]]
        candidates = out/"validation_candidates"
        candidates.mkdir()
        args = (selected, prices, signals[INTRADAY_COLUMNS].copy(), forecasts, local_cfg, store, baseline, str(candidates))
        rows = []
        if cfg["candidate_workers"] == 1:
            initialize_worker(*args)
            rows = [evaluate_worker(rule) for rule in rules]
        else:
            with ProcessPoolExecutor(max_workers=cfg["candidate_workers"], mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker, initargs=args) as executor:
                pending = [executor.submit(evaluate_worker, rule) for rule in rules]
                for future in as_completed(pending):
                    rows.append(future.result())
                    if len(rows)%10 == 0 or len(rows) == len(rules):
                        print(f"[{year}] 内部账户{len(rows)}/{len(rules)}", flush=True)
        chosen, annotated = select_balanced_candidate(pd.DataFrame(rows).sort_values("candidate_index").reset_index(drop=True))
        rule = rules[int(chosen["candidate_index"])]
        choices[year] = {"rule": rule, "inner_metrics": chosen}
        write_frame(annotated, out/"validation_rule_metrics.csv")
        comparison = write_source_comparison(annotated,
            source/f"year_{year}/validation_rule_metrics.csv", out, inputs)
        comparisons.append(comparison.assign(test_year=year, inner_year=fold["inner_year"]))
        write_json(out/"selected_rule.json", {**choices[year], "frozen_before_any_outer_test": True})
        tables.append(annotated.assign(test_year=year, inner_year=fold["inner_year"]))
        print(f"[{year}] 冻结{rule['candidate_index']}；内部净额{chosen['incremental_net_pnl']:.2f}元，成本降低{chosen['holding_cost_reduction_rate']:.4%}", flush=True)
        del context, forecasts, signals, args
        gc.collect()
    validation = pd.concat(tables, ignore_index=True)
    write_frame(validation, directory/"validation_rule_metrics.csv")
    write_frame(pd.concat(comparisons, ignore_index=True), directory/"coverage_vs_source_candidates.csv")
    frozen = {str(y): calculate_file_sha256(directory/f"year_{y}/selected_rule.json") for y in cfg["test_years"]}
    write_json(directory/"all_rules_frozen_before_outer_test.json", {"selected_rule_sha256": frozen,
        "test_years": cfg["test_years"], "model_artifacts_sha256": model_hashes})
    passed = all(choices[y]["rule"]["enabled"] and choices[y]["inner_metrics"]["holding_cost_reduction_rate"] >= cfg["profit_target_rate"]
                 for y in cfg["test_years"])
    evaluate_outer = not cfg["internal_only"] and (passed or cfg["report_scope"] == "smoke")
    write_json(directory/"outer_evaluation_decision.json", {"all_inner_cost_targets_met": passed,
        "outer_evaluation_started": evaluate_outer, "internal_only_requested": cfg["internal_only"],
        "smoke_permits_engineering_outer_without_profit_claim": cfg["report_scope"] == "smoke"})
    results = []
    for name, digest in model_hashes.items():
        if calculate_file_sha256(directory/name) != digest:
            raise AssertionError("新拟合模型在外评前改变")
    for year, digest in frozen.items():
        if calculate_file_sha256(directory/f"year_{year}/selected_rule.json") != digest:
            raise AssertionError("统一冻结规则在首次外评前改变")
    if evaluate_outer:
        for fold in folds:
            year, out = fold["test_year"], directory/f"year_{fold['test_year']}"
            if calculate_file_sha256(out/"selected_rule.json") != frozen[str(year)]:
                raise AssertionError("外评前冻结规则改变")
            selected, prices, signals, local_cfg, store = prepare_account_context(fold, "test", daily, intraday,
                calendar, cfg, out/"test_context", inputs, get_store(year, cfg, inputs, stores))
            source_context = source/f"year_{year}/test_context"
            if cfg["report_scope"] == "smoke" and not source_context.exists():
                raise AssertionError("工程外层须有对应已审计来源底仓可比较")
            if source_context.exists():
                for relative in ("frozen_base_selections.csv", "frozen_base_entries.csv",
                                 "baseline/test_equity_curve.csv"):
                    compare = pd.read_csv(out/"test_context"/relative, float_precision="round_trip")
                    assert_source_table_equal(compare, source_context/relative, inputs)
            rule = choices[year]["rule"]
            forecasts = predict_account_models(signals, out, local_cfg, out/"test_predictions", inputs,
                profiles=[rule["profile"]]) if rule["enabled"] else {}
            decisions, audit = generate_inventory_coverage_decisions(forecasts.get(rule["profile"], {}), rule, local_cfg)
            write_frame(decisions, out/"t_decisions.csv")
            write_json(out/"signal_causality_audit.json", audit)
            local_cfg = create_inventory_coverage_rule_config(local_cfg, rule)
            result = run_guarded_portfolio(selected, prices, signals[INTRADAY_COLUMNS].copy(), decisions, local_cfg, recovery_bars=store)
            result["selected_rule"] = rule
            save_guarded_result(result, out, local_cfg)
            results.append((year, result))
        write_coverage_inventory_report(directory, cfg, results, validation)
    else:
        write_internal_report(directory, folds, choices, passed)
    write_json(directory/"input_sources.json", inputs)
    for name, identity in inputs.items():
        if calculate_file_sha256(Path(name)) != identity["sha256"]:
            raise AssertionError(f"运行期间输入改变：{name}")
    sources = json.loads((directory/"source_hashes.json").read_text())
    for name, digest in sources.items():
        if calculate_file_sha256(PROJECT_ROOT/name) != digest:
            raise AssertionError(f"运行期间源码改变：{name}")
    for name, digest in model_hashes.items():
        if calculate_file_sha256(directory/name) != digest:
            raise AssertionError(f"运行期间模型产物改变：{name}")
    write_json(directory/"runtime_integrity_audit.json", {"all_unchanged": True,
        "input_files_verified": len(inputs), "runtime_sources_verified": len(sources), "model_artifacts_verified": len(model_hashes)})
    print(f"研究完成：{directory/'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
