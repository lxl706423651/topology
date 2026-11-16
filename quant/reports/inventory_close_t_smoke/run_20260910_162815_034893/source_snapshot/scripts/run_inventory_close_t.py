"""永久底仓同日14:55择时：纯核心新拟合，内部真实账户选参后按原门槛外评。

复用已审计完整当前机会与原底仓，参考目标独立改为下一分钟到14:55；
保留v15成交时反向限价复核及全部原费用、T+1、容量与每日恢复核算。
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
from quant_backtest.inventory_t_close_policy import (build_inventory_close_candidates,
    generate_inventory_close_decisions, create_inventory_close_rule_config)
from quant_backtest.inventory_t_close_models import (VERSION, fit_inventory_close_model,
    predict_inventory_close_model, validate_close_model_config)
from quant_backtest.inventory_t_close_labels import (IDENTITY, extract_fixed_close_labels,
    attach_fixed_close_labels)
from quant_backtest.inventory_t_frozen_cohort import load_frozen_inventory_core_cohort
from quant_backtest.inventory_t_close_reporting import write_close_inventory_report
from quant_backtest.weekly_t_quote_store import calculate_file_sha256
from quant_backtest.weekly_t_timing import add_timing_features
from scripts.run_dynamic_inventory_t import (prepare_account_context, save_dynamic_result,
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
    """严格继承v15经济合同与评价范围；仅声明新目标、来源和运行并行度。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/inventory_close_t_v16.yaml")
    parser.add_argument("--source-report")
    parser.add_argument("--test-years", type=int, nargs="+")
    parser.add_argument("--candidate-workers", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--internal-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = (PROJECT_ROOT/args.config).resolve()
    declared = yaml.safe_load(config_path.read_text())
    keys = {"research_version", "source_execution_report_dir", "output_dir", "forecast_target",
            "recovery_key", "t_recheck_exit_side_at_execution", "candidate_workers", "workers"}
    if (not isinstance(declared, dict) or set(declared) != keys or declared["research_version"] != VERSION
            or declared["forecast_target"] != "same_day_1455" or declared["recovery_key"] != "close"
            or declared["t_recheck_exit_side_at_execution"] is not True):
        raise ValueError("v16配置只能声明同日14:55新目标及原guard执行合同")
    source = (PROJECT_ROOT/(args.source_report or declared["source_execution_report_dir"])).resolve()
    original = json.loads((source/"resolved_config.json").read_text())
    if original.get("research_version") != "inventory_guarded_t_v15":
        raise ValueError("执行及基准来源必须是已完成的v15研究")
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
        "scheduled_recovery_times": ["14:55"], "source_prediction_horizons": original["horizons"],
        "output_dir": str((PROJECT_ROOT/(args.output_dir or declared["output_dir"])).resolve())}
    for key in ("horizons", "source_learning_version"):
        cfg.pop(key)
    for key in ("candidate_workers", "workers"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("工作进程数必须为正整数")
    build_inventory_close_candidates(cfg)
    for profile in cfg["learning_model_profiles"]:
        validate_close_model_config({**cfg, "t_model_profile": profile})
    return cfg


def snapshot_sources(directory: Path, cfg: dict) -> None:
    """保留原51源不变，再绑定独立标签、完整机会复用、模型、策略、入口和报告器。"""
    snapshot_guarded_sources(directory, cfg)
    hashes = json.loads((directory/"source_hashes.json").read_text())
    for name in ("scripts/run_inventory_close_t.py", "quant_backtest/inventory_t_close_labels.py",
                 "quant_backtest/inventory_t_frozen_cohort.py", "quant_backtest/inventory_t_close_models.py",
                 "quant_backtest/inventory_t_close_policy.py", "quant_backtest/inventory_t_close_reporting.py"):
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
        "source_t_models_used_for_prediction": False, "new_close_models_fitted_per_fold": 2,
        "same_baseline_required": True, "same_predictions_and_decisions_required": False,
        "comparison_changes_forecast_target_and_corresponding_deadline": True})


def extract_core_labels(intraday: pd.DataFrame, core: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """先冻结所有当前core键，再取同键原端点；不构造或读取内部/外层收益标签表。"""
    columns = ["datetime", "symbol", "execution_time", "execution_price", "recovery_time", "recovery_price"]
    optional = [name for name in ("execution_volume", "execution_amount", "recovery_volume", "recovery_amount",
                                  "target_return", "trade_week_complete") if name in intraday]
    current = intraday.loc[intraday.datetime.dt.year.isin(core.cohort_target_year.unique()), columns+optional]
    matched = core[["datetime", "symbol"]].merge(current, on=["datetime", "symbol"],
        how="left", sort=False, validate="one_to_one")
    labels, audit = extract_fixed_close_labels(matched, cfg)
    return labels, {**audit, "only_frozen_core_keys_read": True, "core_rows": len(core)}


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
    predictions = predict_inventory_close_model(bundle, frame, cfg)
    future = [name for name in frame if name.startswith(("execution_", "recovery_", "target_", "label_"))]
    repeated = predict_inventory_close_model(bundle, frame.drop(columns=future), cfg)
    pd.testing.assert_series_equal(predictions.prediction, repeated.prediction, check_exact=True)
    native = lightgbm.Booster(model_file=str(model_dir/"return_model.txt"))
    matrix = frame[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = np.asarray(native.predict(matrix) if len(matrix) else np.array([]))
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise AssertionError("原生重载输出必须逐行且全为有限数")
    error = float(np.max(np.abs(values-predictions.prediction.to_numpy()))) if len(frame) else 0.
    if error > 1e-12 or native.feature_name() != bundle["features"]:
        raise AssertionError("原生模型不能精确复现全部当前预测")
    fields = ["datetime", "symbol", "prediction", "raw_prediction", "model_profile", *IDENTITY,
              "signal_price", "signal_high", "signal_low", "signal_amount", "signal_volume", "previous_close"]
    archived = list(dict.fromkeys(fields+bundle["features"]+["feature_source_start"]))
    write_frame(predictions[archived], out/"current_predictions.csv")
    write_json(out/"prediction_audit.json", {"passed": True, "current_rows": len(frame), **IDENTITY,
        "future_fields_removed": future, "future_independent": True, "native_max_error": error,
        "features": bundle["features"], "model_bundle_sha256": calculate_file_sha256(model_dir/"model_bundle.joblib"),
        "predictions_sha256": calculate_file_sha256(out/"current_predictions.csv")})
    return predictions[fields].copy()


def predict_account_models(signals, fold_out, cfg, out, inputs, profiles=None) -> dict:
    """只加载本折两种新固定尾盘核心权重；内部/外层没有再拟合或校准。"""
    forecasts = {}
    for profile in cfg["learning_model_profiles"] if profiles is None else profiles:
        key = f"close_{profile}"
        forecasts[profile] = predict_and_audit(signals, fold_out/"models"/key,
            {**cfg, "t_model_profile": profile}, out/key, inputs)
    return forecasts


def write_source_comparison(current: pd.DataFrame, source_path: Path, out: Path, inputs: dict) -> pd.DataFrame:
    """同数字候选与相同底仓比较新旧预测策略，不宣称决策相同或相加候选利润。"""
    record_balanced_input(source_path, inputs)
    original = pd.read_csv(source_path, float_precision="round_trip")
    key = "candidate_index"
    fields = [key, "enabled", "profile", "min_net_edge", "fraction", "exit_mode"]
    pd.testing.assert_frame_equal(current[fields].reset_index(drop=True), original[fields],
                                  check_dtype=False, check_exact=True)
    values = ["incremental_net_pnl", "holding_cost_reduction_rate", "t_pair_count", "completed_t_pair_lots",
              "t_fee", "t_slippage_cost", "inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
              "all_reference_acquisition_cost"]
    combined = current[[key, *values]].merge(original[[key, *values]], on=key,
        validate="one_to_one", suffixes=("_close", "_source"))
    if not np.allclose(combined.all_reference_acquisition_cost_close,
                      combined.all_reference_acquisition_cost_source, atol=1e-8, rtol=0):
        raise AssertionError("新预测目标改变了原底仓取得成本分母")
    for name in values:
        combined[name+"_change"] = combined[name+"_close"]-combined[name+"_source"]
    combined["same_forecasts"] = False
    combined["same_execution_guard"] = True
    write_frame(combined, out/"close_vs_source_candidates.csv")
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
    forecasts = c["predictions"].get(rule["profile"], pd.DataFrame())
    decisions, audit = generate_inventory_close_decisions(forecasts, rule, c["cfg"])
    write_frame(decisions, out/"t_decisions.csv")
    write_json(out/"signal_causality_audit.json", audit)
    local = create_inventory_close_rule_config(c["cfg"], rule)
    result = run_guarded_portfolio(c["selected"], c["daily"], c["signals"], decisions, local, recovery_bars=c["store"])
    curve = result["equity_curve"]
    if (not pd.to_datetime(curve.datetime).reset_index(drop=True).equals(c["baseline"].datetime)
            or not np.allclose(curve.baseline_equity, c["baseline"].equity, atol=1e-8, rtol=0)):
        raise AssertionError("不同T规则的无T基准发生改变")
    metrics = save_guarded_result(result, out, local)
    return {**metrics, **rule, "request_count": len(decisions), "candidate_directory": str(out)}


def write_internal_report(directory, folds, choices, targets_passed):
    """未晋级时仅报告完整内部结果，不制造外层成绩或外层收益图。"""
    lines = ["# 永久底仓固定尾盘预测：内部研究结果", "", "正式外层未执行。下表是内部年度选参结果，不是年度验证集成绩。", "",
        "| 留出年 | 内部年 | 规则 | 完整组 | 配对手 | T净额（元） | 成本降低率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in folds:
        item = choices[fold["test_year"]]["inner_metrics"]
        lines.append(f"| {fold['test_year']} | {fold['inner_year']} | {int(item['candidate_index'])} | {int(item['t_pair_count'])} | {item['completed_t_pair_lots']:.0f} | {item['incremental_net_pnl']:.2f} | {item['holding_cost_reduction_rate']:.4%} |")
    lines += ["", "在原纯核心完整永久旧股队列新拟合两种回归模型，预测下一分钟到当天14:55的参考价差。取消跨期限择优，沿用37数字候选、原底仓和v15反向复核执行规则；预测收益不是实际账户净利润。",
        "", "所有内部折达到5%底仓成本降低率后才进入统一外评。每股初始名义1万—2万元，底仓永久保留；没有回选历史外层结果。"]
    (directory/"final_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(directory/"goal_evaluation.json", {"goal_achieved": False, "formal_outer_evaluated": False,
        "all_inner_cost_targets_met": targets_passed, "reason": "internal_research_only"})


def main(argv: list[str] | None = None) -> None:
    """冻结新预测目标与原执行条件；纯核心新拟合、37内部账户后按原5%协议外评。"""
    cfg = load_configuration(argv)
    directory = Path(cfg["output_dir"])/f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory, cfg)
    write_json(directory/"resolved_config.json", cfg)
    rules = build_inventory_close_candidates(cfg)
    write_json(directory/"research_protocol.json", {"version": VERSION, "previous_historical_outer_results_seen": True,
        "candidate_rules": rules, "models_per_fold": 2, "models_fitted_this_run": 2*len(cfg["test_years"]),
        "frozen_t_models_reused": False, "outer_model_refit": False,
        "source_execution_report_dir": cfg["source_execution_report_dir"],
        "source_learning_report_dir": cfg["source_learning_report_dir"],
        "training_cohort": "core_only_oof_weekly_entry_purged_before_permanent_baseline_planning",
        "training_labels": "next_minute_to_same_day_14_55_reference_return", **IDENTITY,
        "cross_horizon_argmax": False, "labels_are_actual_shared_account_pnl": False,
        "selection_uses_inner_real_accounts_only": True, "no_weekly_middle_day_restriction": True,
        "no_probability_or_old_support_gate": True, "full_core_training_also_in_smoke": True,
        "execution_guard_unchanged_from_v15": True, "outer_evaluation_policy": cfg["outer_evaluation_policy"],
        "valuation_contract": "full_minute_reference_stream_then_daily_close",
        "sources": ["https://doi.org/10.1016/j.frl.2019.04.002", "https://arxiv.org/abs/1411.5062"],
        "literature_supports_investigating_time_of_day_not_exact_clock_or_profit": True,
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
    print("复用完整因子与已审计纯核心机会；独立拟合同日14:55新目标", flush=True)
    universe, intraday, daily = load_frozen_market_data({**cfg, "test_years": needed}, directory, inputs)
    known = json.loads((Path(cfg["source_audit_report_dir"])/"input_sources.json").read_text())
    for name in ("weekly", "intraday", "daily"):
        path = (Path(cfg["market_cache_dir"])/f"{name}.pkl").resolve()
        if str(path) not in known:
            raise ValueError("原市场PKL缺已审计身份")
        record_balanced_input(path, inputs, known[str(path)]["sha256"])
    intraday = add_timing_features(intraday, cfg)
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
        labels, label_audit = extract_core_labels(intraday, core, cfg)
        write_json(out/"fixed_close_label_audit.json", label_audit)
        training = attach_fixed_close_labels(core, labels, cfg)
        print(f"[{year}] 全部永久旧股核心机会{len(core):,}行；开始2个固定尾盘回归模型", flush=True)
        for profile in cfg["learning_model_profiles"]:
            key = f"close_{profile}"
            model = out/"models"/key
            metadata = fit_inventory_close_model(training, fold["core_years"],
                {**cfg, "t_model_profile": profile}, model, [fold["inner_year"], year])
            model_hashes.update({str(path.relative_to(directory)): calculate_file_sha256(path)
                for path in sorted(model.iterdir()) if path.is_file()})
            print(f"[{year}] {key}拟合{metadata['fitted_rows']:,}行/{len(metadata['features'])}特征", flush=True)
        del core, labels, training
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
    write_frame(pd.concat(comparisons, ignore_index=True), directory/"close_vs_source_candidates.csv")
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
            decisions, audit = generate_inventory_close_decisions(forecasts.get(rule["profile"], pd.DataFrame()), rule, local_cfg)
            write_frame(decisions, out/"t_decisions.csv")
            write_json(out/"signal_causality_audit.json", audit)
            local_cfg = create_inventory_close_rule_config(local_cfg, rule)
            result = run_guarded_portfolio(selected, prices, signals[INTRADAY_COLUMNS].copy(), decisions, local_cfg, recovery_bars=store)
            result["selected_rule"] = rule
            save_guarded_result(result, out, local_cfg)
            results.append((year, result))
        write_close_inventory_report(directory, cfg, results, validation)
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
