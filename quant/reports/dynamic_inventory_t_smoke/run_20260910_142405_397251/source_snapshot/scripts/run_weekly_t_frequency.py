"""在冻结核心权重上比较提高频率的做T规则，全部内部选参后再统一年度外评。

从quant根目录读取v10 YAML和v8/v9只读报告，复用原完整截面行情与恢复索引。
不重训、不修改费用或库存执行器；完整保留每个内部候选的真实成交与失败。
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
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_frequency import (
    build_frequency_signal_funnel, create_frequency_decisions, create_frequency_rules,
    select_frequency_candidate, summarize_frequency_funnel,
)
from quant_backtest.weekly_t_frequency_reporting import write_frequency_report
from quant_backtest.weekly_t_frequency_validation import (
    calculate_frequency_opportunities, run_frequency_candidates,
)
from quant_backtest.weekly_t_horizon_data import validate_horizon_config
from quant_backtest.weekly_t_quote_store import DenseRecoveryBarStore, calculate_file_sha256
from quant_backtest.weekly_t_residual import predict_residual_t, validate_residual_config
from quant_backtest.weekly_t_validation import validate_candidate_result
from scripts.run_weekly_t_active import (
    PATH_FIELDS, load_frozen_market_data, record_input, snapshot_active_sources, validate_training_source,
)
from scripts.run_weekly_t_cross_validation import save_result, select_execution_data, write_frame
from scripts.run_weekly_t_residual import attach_frozen_context
from scripts.run_weekly_t_short_horizon import read_frozen_selections

VERSION = "weekly_t_v10_frequency"
DECISION_COLUMNS = ["datetime", "symbol", "action", "fraction", "priority", "planned_recovery_time", "date"]


def load_configuration(argv: list[str] | None = None) -> dict:
    """仅允许声明的新规则参数和执行范围；训练、成本、份额与恢复时钟必须可复用。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/weekly_t_cross_validation_v10.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--smoke-weeks", type=int)
    parser.add_argument("--test-years")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("output_dir", "max_symbols", "smoke_weeks"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    if args.test_years:
        cfg["test_years"] = [int(y) for y in args.test_years.split(",")]
    for key in (*PATH_FIELDS, "source_active_report_dir"):
        cfg[key] = str((PROJECT_ROOT / cfg[key]).resolve())
    cfg["dense_recovery_cache_paths"] = {str(y): str((PROJECT_ROOT / p).resolve())
                                         for y, p in cfg["dense_recovery_cache_paths"].items()}
    if cfg.get("research_version") != VERSION or cfg.get("candidate_activation_policy") != "frequency_execution_valid":
        raise ValueError("本入口要求v10 frequency_execution_valid配置")
    if "candidate_support_requirements" in cfg or cfg["max_symbols"] < 0 or cfg["smoke_weeks"] < 0:
        raise ValueError("不接收旧停用门槛或负smoke范围")
    if not cfg["test_years"] or len(set(cfg["test_years"])) != len(cfg["test_years"]):
        raise ValueError("年度必须非空且无重复")
    if cfg.get("frequency_horizon_minutes") != 20:
        raise ValueError("本次协议固定h20，不能混用其他训练目标")
    create_frequency_rules(cfg, {"profile": "l2_depth4", "horizon_minutes": 20, "net_margin": .002,
                                "fraction": cfg["short_t_fraction"], "direction": "both", "enabled": True})
    for key in ("frequency_target_pairs_per_day", "frequency_max_abs_day_return", "frequency_max_known_amplitude"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key}必须为有限正数")
    validate_residual_config(cfg)
    _, _, minutes = validate_horizon_config(cfg)
    cfg["scheduled_recovery_times"] = [f"{v // 60:02d}:{v % 60:02d}" for v in minutes]
    # 用已有严格校验核对全部原字段，新信号配置另由新模块校验。
    original = {k: v for k, v in cfg.items() if not k.startswith("frequency_") and k != "source_active_report_dir"}
    original["candidate_activation_policy"] = "active_execution_valid"
    validate_training_source(original)
    return cfg


def snapshot_frequency_sources(directory: Path) -> None:
    """保存旧运行依赖和本次新增实现，历史报告/快照不改写。"""
    snapshot_active_sources(directory)
    hashes = json.loads((directory / "source_hashes.json").read_text())
    for name in ("scripts/run_weekly_t_frequency.py", "quant_backtest/weekly_t_frequency.py",
                 "quant_backtest/weekly_t_frequency_validation.py", "quant_backtest/weekly_t_frequency_reporting.py"):
        path = PROJECT_ROOT / name
        shutil.copyfile(path, directory / "source_snapshot" / path.name)
        hashes[name] = calculate_file_sha256(path)
    write_json(directory / "source_hashes.json", hashes)


def load_fold_definitions(cfg: dict, directory: Path, inputs: dict) -> list[dict]:
    """只读取已审计训练拆分；保存独立副本，不按新测试表现改变年份。"""
    folds = []
    for year in cfg["test_years"]:
        path = Path(cfg["source_training_report_dir"]) / f"year_{year}/split.json"
        record_input(path, inputs)
        fold = json.loads(path.read_text())
        if (fold["test_year"] != year or year in fold["train_years"]
                or fold["inner_year"] in fold["core_years"]
                or sorted(fold["train_years"]) != sorted(fold["core_years"] + [fold["inner_year"]])):
            raise ValueError("核心/内部/外层拆分合同不一致")
        out = directory / f"year_{year}"
        out.mkdir()
        write_json(out / "split.json", fold)
        folds.append(fold)
    write_frame(pd.DataFrame(folds), directory / "split_audit.csv")
    return folds


def prepare_execution_context(fold: dict, scope: str, universe: pd.DataFrame, intraday: pd.DataFrame,
                              daily: pd.DataFrame, cfg: dict, out: Path, inputs: dict) -> tuple:
    """先保留完整同刻特征，再取已冻结股池；验证相应inner/outer周上下文身份。"""
    year = fold["test_year"]
    inner = scope == "validation"
    filename = "validation_weekly_selections.csv" if inner else "weekly_selections.csv"
    source = Path(cfg["source_training_report_dir"])
    # 审计来源是v4原始冻结名单；v8导出的同值CSV会有序列化差异，不能冒用原SHA。
    path = Path(cfg["frozen_selection_report_dir"]) / f"year_{year}" / filename
    audit = json.loads((source / "independent_residual_audit.json").read_text())["frozen_contexts"][f"year_{year}/{scope}"]
    record_input(path, inputs)
    if audit.get("passed") is not True or calculate_file_sha256(path) != audit["source_selection_sha256"]:
        raise ValueError("冻结周股池偏离原审计身份")
    selected = read_frozen_selections(path, set(daily.symbol), {**cfg, "max_symbols": 0, "smoke_weeks": 0})
    if cfg["smoke_weeks"]:
        selected = selected.loc[selected.trade_week.isin(sorted(selected.trade_week.unique())[:cfg["smoke_weeks"]])].copy()
    if cfg["max_symbols"]:
        selected = selected.loc[selected.symbol.isin(sorted(selected.symbol.unique())[:cfg["max_symbols"]])].copy()
    if selected.empty:
        raise ValueError("执行范围为空")
    write_frame(selected, out / filename)
    if not inner:
        for name in ("weekly_predictions.csv", "weekly_prediction_metrics.csv"):
            original = source / f"year_{year}" / name
            record_input(original, inputs)
            shutil.copyfile(original, out / name)
    actual_daily, actual_intraday = select_execution_data(selected, daily, intraday)
    model_name = "inner_weekly_model" if inner else "weekly_model"
    model = Path(cfg["frozen_selection_report_dir"]) / f"year_{year}" / model_name
    expected = audit["context"]
    for filename, key in (("model_bundle.joblib", "source_model_sha256"), ("model_metadata.json", "source_metadata_sha256")):
        if calculate_file_sha256(model / filename) != expected[key]:
            raise ValueError("冻结周模型上下文身份改变")
    for path in model.iterdir():
        if path.is_file():
            record_input(path, inputs)
    actual_intraday = attach_frozen_context(actual_intraday, selected, universe, cfg, model,
                                            fold["core_years"] if inner else fold["train_years"])
    manifest = actual_intraday.attrs["frozen_weekly_context_manifest"]
    if not cfg["smoke_weeks"] and not cfg["max_symbols"]:
        if manifest["context_inputs_sha256"] != expected["context_inputs_sha256"]:
            raise AssertionError("完整范围周特征与已审计输入改变")
    write_json(out / f"{scope}_weekly_context_manifest.json", manifest)
    return selected, actual_daily, actual_intraday


def copy_core_model(fold: dict, profile: str, cfg: dict, out: Path, inputs: dict) -> Path:
    """只复用精确审计过的h20核心权重，核对训练/排除年份且保存全模型文件。"""
    year = fold["test_year"]
    name = f"inner_t_h20_{profile}"
    source = Path(cfg["source_training_report_dir"])
    origin = source / f"year_{year}" / name
    identity = json.loads((source / "independent_residual_audit.json").read_text())["models"][f"year_{year}/{name}"]
    if identity.get("passed") is not True:
        raise ValueError("核心模型原审计未通过")
    for filename, key in (("model_metadata.json", "metadata_sha256"), ("model_bundle.joblib", "model_bundle_sha256"),
                          ("return_model.txt", "native_model_sha256")):
        if calculate_file_sha256(origin / filename) != identity[key]:
            raise ValueError(f"核心模型身份变化：{filename}")
    metadata = json.loads((origin / "model_metadata.json").read_text())
    if (sorted(metadata["train_years"]) != sorted(fold["core_years"])
            or sorted(metadata["excluded_years"]) != sorted([year, fold["inner_year"]])
            or metadata["profile"] != profile or metadata["resolved_config"]["t_label_horizon_minutes"] != 20):
        raise ValueError("核心模型训练域与本折不一致")
    for path in sorted(origin.rglob("*")):
        if path.is_file():
            record_input(path, inputs)
    shutil.copytree(origin, out / name)
    return out / name


def predict_and_audit(frame: pd.DataFrame, model: Path, cfg: dict, out: Path, prefix: str) -> pd.DataFrame:
    """原生权重复核，并证明删除未来价格/标签字段不改变任何模型输入或预测。"""
    bundle = joblib.load(model / "model_bundle.joblib")
    predictions = predict_residual_t(bundle, frame, cfg)
    native = lightgbm.Booster(model_file=str(model / "return_model.txt"))
    if native.feature_name() != bundle["features"]:
        raise AssertionError("原生与bundle特征顺序不一致")
    matrix = predictions[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = native.predict(matrix) if len(matrix) else np.array([], dtype=float)
    error = float(np.max(np.abs(values - predictions.prediction))) if len(values) else 0.0
    if values.shape != (len(predictions),) or not np.isfinite(values).all() or error > 1e-12:
        raise AssertionError("原生权重与bundle预测不一致")
    future = [c for c in frame if c.startswith(("execution_", "recovery_", "target_", "label_"))]
    scrubbed = frame.copy()
    scrubbed[future] = np.nan
    second = predict_residual_t(bundle, scrubbed, cfg)
    pd.testing.assert_frame_equal(predictions[[*bundle["features"], "prediction"]],
                                  second[[*bundle["features"], "prediction"]], check_exact=True)
    write_json(out / f"{prefix}_causality_audit.json", {"native_prediction_max_error": error,
               "prediction_rows": len(predictions), "future_fields_removed": future,
               "model_features_and_predictions_unchanged_without_future_fields": True})
    write_frame(predictions[["datetime", "symbol", "prediction", "residual_prediction", "market_prediction"]],
                out / f"{prefix}_predictions.csv")
    return predictions


def assert_decision_causality(predictions: pd.DataFrame, decisions: pd.DataFrame, rule: dict, cfg: dict, support: dict) -> None:
    """新信号也须不受执行价、恢复价或未来标签字段影响。"""
    future = [c for c in predictions if c.startswith(("execution_", "recovery_", "target_", "label_"))]
    scrubbed = predictions.copy()
    scrubbed[future] = np.nan
    pd.testing.assert_frame_equal(decisions, create_frequency_decisions(scrubbed, rule, cfg, support), check_exact=True)


def get_store(year: int, cfg: dict, inputs: dict, stores: dict) -> DenseRecoveryBarStore:
    """每个年度只打开并完整校验一次原只读恢复数组。"""
    if year not in stores:
        path = Path(cfg["dense_recovery_cache_paths"][str(year)])
        record_input(path / "metadata.json", inputs)
        stores[year] = DenseRecoveryBarStore(path, verify_integrity=True)
    return stores[year]


def freeze_frequency_rules(folds: list, universe: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                           cfg: dict, directory: Path, inputs: dict, stores: dict) -> pd.DataFrame:
    """完整运行本折25个内部账户；频率达标前按频率排序，之后比较真实净增量。"""
    source, active = Path(cfg["source_training_report_dir"]), Path(cfg["source_active_report_dir"])
    audited = json.loads((source / "independent_residual_audit.json").read_text())
    tables = []
    for fold in folds:
        year = fold["test_year"]
        out = directory / f"year_{year}"
        selected, inner_daily, inner_intraday = prepare_execution_context(
            fold, "validation", universe, intraday, daily, cfg, out, inputs)
        store = get_store(fold["inner_year"], cfg, inputs, stores)
        if not set(selected.symbol).issubset(store.symbols):
            raise ValueError("恢复行情未覆盖内部股票")
        source_metrics = source / f"year_{year}/validation_rule_metrics.csv"
        record_input(source_metrics, inputs)
        reference, _ = select_active_candidate(pd.read_csv(source_metrics, float_precision="round_trip"))
        prior_path = active / f"year_{year}/selected_rule.json"
        record_input(prior_path, inputs)
        prior = json.loads(prior_path.read_text())
        for key in ("candidate_index", "horizon_minutes", "profile", "net_margin", "fraction", "direction"):
            if reference[key] != prior[key]:
                raise AssertionError("旧v9参考规则不能由外层结果重新选择")
        rules = create_frequency_rules(cfg, prior)
        write_json(out / "candidate_rules_frozen_before_inner_execution.json", rules)
        support_path = source / f"year_{year}/support_h20.json"
        record_input(support_path, inputs)
        identity = audited["supports"][f"year_{year}/support_h20.json"]
        if identity.get("passed") is not True or calculate_file_sha256(support_path) != identity["support_file_sha256"]:
            raise ValueError("h20训练支持域身份改变")
        support = json.loads(support_path.read_text())
        shutil.copyfile(support_path, out / "selected_support.json")
        baseline = run_weekly_portfolio(selected, inner_daily, inner_intraday,
                                       pd.DataFrame(columns=DECISION_COLUMNS), cfg, recovery_bars=store)
        opportunities, eligible = calculate_frequency_opportunities(baseline, inner_intraday, cfg, 20)
        baseline_dir = out / "validation_baseline"
        baseline_dir.mkdir()
        save_result(baseline, baseline_dir, cfg)
        write_frame(eligible, out / "validation_eligible_inventory_stock_days.csv")
        write_json(out / "validation_frequency_opportunities.json", opportunities)
        del baseline
        tasks, funnels = [], []
        for profile in cfg["t_model_profiles"]:
            model = copy_core_model(fold, profile, cfg, out, inputs)
            predictions = predict_and_audit(inner_intraday, model, cfg, out, f"validation_h20_{profile}")
            for rule in (r for r in rules if r["profile"] == profile):
                decisions = create_frequency_decisions(predictions, rule, cfg, support)
                assert_decision_causality(predictions, decisions, rule, cfg, support)
                signature = hashlib.sha256(pd.util.hash_pandas_object(decisions, index=False).to_numpy().tobytes()).hexdigest()
                funnel = build_frequency_signal_funnel(predictions, rule, cfg, support)
                funnel_dir = out / "validation_signal_funnels"
                funnel_dir.mkdir(exist_ok=True)
                write_frame(funnel, funnel_dir / f"candidate_{rule['candidate_index']:03d}.csv")
                funnel_summary = summarize_frequency_funnel(funnel)
                write_json(funnel_dir / f"candidate_{rule['candidate_index']:03d}.json", funnel_summary)
                flattened = {f"{stage}_{key}": value for stage, counts in
                             {"input": funnel_summary["input"], **funnel_summary["stages"]}.items()
                             for key, value in counts.items()}
                funnels.append({"candidate_index": rule["candidate_index"], "signal_mode": rule["signal_mode"], **flattened})
                tasks.append({"candidate_index": rule["candidate_index"], "decision_sha256": signature, "decisions": decisions})
            del predictions
        write_frame(pd.DataFrame(funnels), out / "validation_signal_funnel_summary.csv")
        print(f"[{year}] 内部{fold['inner_year']}，{len(tasks)}候选，计划可做T {opportunities['eligible_portfolio_days']}日", flush=True)
        evaluated = run_frequency_candidates(tasks, selected, inner_daily, inner_intraday, cfg,
                                             out / "validation_candidate_audits", recovery_store=store)
        task_map = {t["candidate_index"]: t for t in tasks}
        rows = []
        for rule in rules:
            index = rule["candidate_index"]
            item = evaluated[index]
            task = task_map[index]
            rows.append({"test_year": year, "inner_year": fold["inner_year"], **rule, **item["metrics"],
                         **opportunities, "incremental_net_pnl": item["incremental_net_pnl"],
                         "active_days": item["active_days"], "active_stock_days": item["active_stock_days"],
                         "submitted_t_requests": len(task["decisions"]), "decision_sha256": task["decision_sha256"]})
        if not cfg["smoke_weeks"] and not cfg["max_symbols"]:
            ref = rows[0]
            if task_map[0]["decision_sha256"] != reference["decision_sha256"]:
                raise AssertionError("精确v9参考规则的完整内部请求序列未复现")
            if int(ref["t_pair_count"]) != int(reference["t_pair_count"]) or abs(ref["incremental_net_pnl"] - reference["incremental_net_pnl"]) > 1e-7:
                raise AssertionError("精确v9参考规则内部成交/PnL未复现")
        # 失败时也先保留所有内部指标，不伪造无交易或跳过失败候选。
        write_frame(pd.DataFrame(rows), out / "validation_rule_metrics_unselected.csv")
        chosen, annotated = select_frequency_candidate(pd.DataFrame(rows), cfg)
        rule = next(r for r in rules if r["candidate_index"] == int(chosen["candidate_index"]))
        model_name = f"inner_t_h20_{rule['profile']}"
        shutil.copytree(out / model_name, out / "t_model")
        frozen = {**rule, "fold": fold, "source_model": model_name,
                  "source_model_path": str((source / f"year_{year}" / model_name).resolve()),
                  "model_bundle_sha256": calculate_file_sha256(out / "t_model/model_bundle.joblib"),
                  "inner_incremental_net_pnl": float(chosen["incremental_net_pnl"]),
                  "inner_complete_pairs": int(chosen["t_pair_count"]),
                  "inner_eligible_portfolio_days": int(opportunities["eligible_portfolio_days"]),
                  "activation_policy": cfg["candidate_activation_policy"], "statistical_gates_are_informational": True,
                  "frozen_before_any_outer_test": True}
        write_json(out / "selected_rule.json", frozen)
        write_frame(annotated, out / "validation_rule_metrics.csv")
        tables.append(annotated)
        print(f"[{year}] 冻结候选{rule['candidate_index']} {rule['signal_mode']}/{rule['profile']}："
              f"内部{chosen['t_pair_count']}对，净T {chosen['incremental_net_pnl']:.2f}元", flush=True)
        del tasks, task_map, inner_intraday
        gc.collect()
    combined = pd.concat(tables, ignore_index=True)
    write_frame(combined, directory / "validation_rule_metrics.csv")
    write_json(directory / "all_rules_frozen_before_outer_test.json", {
        "created_at": datetime.now().isoformat(), "test_years": cfg["test_years"], "all_t_enabled": True,
        "selection_uses_inner_only": True, "selected_rule_sha256": {str(f["test_year"]): calculate_file_sha256(
            directory / f"year_{f['test_year']}/selected_rule.json") for f in folds}})
    return combined


def evaluate_frequency_folds(folds: list, universe: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                             cfg: dict, directory: Path, inputs: dict, stores: dict) -> tuple:
    """只运行全部已冻结的外层规则；保存完整真实账户及不使用事后成交的频率分母。"""
    marker = json.loads((directory / "all_rules_frozen_before_outer_test.json").read_text())
    if marker["test_years"] != [f["test_year"] for f in folds]:
        raise ValueError("外评必须等待所有目标年度冻结")
    for year in cfg["test_years"]:
        if calculate_file_sha256(directory / f"year_{year}/selected_rule.json") != marker["selected_rule_sha256"][str(year)]:
            raise ValueError("外评前规则改变")
    summaries, results = [], []
    for fold in folds:
        year = fold["test_year"]
        out = directory / f"year_{year}"
        rule = json.loads((out / "selected_rule.json").read_text())
        support = json.loads((out / "selected_support.json").read_text())
        selected, actual_daily, actual_intraday = prepare_execution_context(
            fold, "test", universe, intraday, daily, cfg, out, inputs)
        predictions = predict_and_audit(actual_intraday, out / "t_model", cfg, out, "t")
        decisions = create_frequency_decisions(predictions, rule, cfg, support)
        assert_decision_causality(predictions, decisions, rule, cfg, support)
        funnel = build_frequency_signal_funnel(predictions, rule, cfg, support)
        write_frame(decisions, out / "t_decisions.csv")
        write_frame(funnel, out / "test_signal_funnel.csv")
        write_json(out / "test_signal_funnel_summary.json", summarize_frequency_funnel(funnel))
        store = get_store(year, cfg, inputs, stores)
        if not set(selected.symbol).issubset(store.symbols):
            raise ValueError("恢复行情未覆盖外层股票")
        write_json(out / "dense_recovery_reference.json", {"path": cfg["dense_recovery_cache_paths"][str(year)],
                   "fingerprint": store.fingerprint, "arrays_sha256_verified": True, "time_count": store.metadata["time_count"]})
        baseline = run_weekly_portfolio(selected, actual_daily, actual_intraday,
                                       pd.DataFrame(columns=DECISION_COLUMNS), cfg, recovery_bars=store)
        opportunities, eligible = calculate_frequency_opportunities(baseline, actual_intraday, cfg, 20)
        write_frame(eligible, out / "eligible_inventory_stock_days.csv")
        write_json(out / "frequency_opportunities.json", opportunities)
        result = run_weekly_portfolio(selected, actual_daily, actual_intraday, decisions, cfg, recovery_bars=store)
        execution_audit = validate_candidate_result(result, cfg)
        write_json(out / "candidate_execution_audit.json", execution_audit)
        metrics = save_result(result, out, cfg)
        curve = result["equity_curve"]
        pd.testing.assert_series_equal(curve.baseline_equity, baseline["equity_curve"].equity, check_exact=True, check_names=False)
        if not cfg["max_symbols"] and not cfg["smoke_weeks"]:
            origin = Path(cfg["source_training_report_dir"]) / f"year_{year}/test_equity_curve.csv"
            record_input(origin, inputs)
            previous = pd.read_csv(origin, parse_dates=["datetime"], float_precision="round_trip")
            if not curve.datetime.reset_index(drop=True).equals(previous.datetime.reset_index(drop=True)):
                raise AssertionError("原底仓基准日期改变")
            error = float(np.max(np.abs(curve.baseline_equity.to_numpy() - previous.baseline_equity.to_numpy())))
            if error > 1e-8:
                raise AssertionError("原底仓基准权益改变")
            write_json(out / "frozen_baseline_audit.json", {"same_dates": True, "max_equity_error_rmb": error})
        first = result["trade_events"].loc[lambda x: x.event_type.eq("t") & x.fragment_id.eq(0)]
        active_keys = first.assign(date=pd.to_datetime(first.datetime).dt.normalize())
        summaries.append({"test_year": year, "inner_year": fold["inner_year"], **metrics, **opportunities,
                          "t_enabled": True, "enabled": True, "submitted_t_requests": len(decisions),
                          "request_count": len(decisions), "runtime_signal_rows": int(funnel.signal_eligible.sum()),
                          "prediction_rows": len(predictions), "active_days": int(active_keys.date.nunique()),
                          "active_stock_days": len(active_keys.drop_duplicates(["date", "symbol"])),
                          "selected_candidate_index": rule["candidate_index"], "signal_mode": rule["signal_mode"]})
        results.append((year, result))
        print(f"[{year}] {len(decisions)}请求/{metrics['t_pair_count']}完整对，净T {metrics['t_added_equity']:.2f}元，"
              f"股数未恢复{metrics['inventory_mismatch_days']}日/恢复失败{metrics['recovery_failure_days']}日", flush=True)
        del predictions, actual_intraday, baseline
        gc.collect()
    return pd.DataFrame(summaries), results


def main(argv: list[str] | None = None) -> None:
    """固定协议、完整内部选择、统一外评并核对全部只读来源后形成独立报告。"""
    cfg = load_configuration(argv)
    directory = Path(cfg["output_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True)
    snapshot_frequency_sources(directory)
    write_json(directory / "resolved_config.json", cfg)
    write_json(directory / "research_protocol.json", {
        "research_version": VERSION, "created_before_inner_execution_and_outer_test": True,
        "previous_outer_results_seen": True, "selection_uses_inner_only": True, "no_model_refit": True,
        "frequency_target_pairs_per_day": cfg["frequency_target_pairs_per_day"],
        "candidate_design": "h20: 1精确v9参考+3模型×2绝对门槛+3模型×3相对名次×2当前VWAP偏差；25候选",
        "selection": "完整对>0且5执行失败计数=0；min(完整对/计划可做T日,目标频率)降序，再净增量降序、索引升序",
        "denominator": "无T基准开盘旧股×25%够100整手，且该股该日有当前合法h20时钟；不按事后成交/现金/恢复删样本",
        "relative_signal": "偏离当前VWAP且5分钟回转，预测同向，当前日涨跌幅<=3%/振幅<=6%，同刻合格机会按abs(pred)排名",
        "relative_cost_gate": "不要求绝对预测收益覆盖费用；实际滑点、税费保持原值，排名不保证盈利",
        "retries": "新候选保留全部有效信号时刻；原引擎每股每日最多1次成功首腿，不重置当日卖额",
        "execution_engine_unchanged": True, "all_rules_frozen_before_outer_test": True,
        "no_eligible_candidate": "显式报错并保留全部内部结果，不能静默关闭或改写成交失败",
        "smoke_scope": "只缩小执行股池与周，不用于正式选参；完整同刻特征仍先生成",
        "is_smoke": bool(cfg["max_symbols"] or cfg["smoke_weeks"]),
        "source_training_report_dir": cfg["source_training_report_dir"], "source_active_report_dir": cfg["source_active_report_dir"]})
    print(f"输出目录：{directory}", flush=True)
    inputs, stores = {}, {}
    for name in ("resolved_config.json", "source_hashes.json", "independent_residual_audit.json"):
        record_input(Path(cfg["source_training_report_dir"]) / name, inputs)
    for name in ("resolved_config.json", "annual_summary.csv", "independent_execution_audit.json"):
        record_input(Path(cfg["source_active_report_dir"]) / name, inputs)
    for year in cfg["test_years"]:
        record_input(Path(cfg["source_active_report_dir"]) / f"year_{year}/test_equity_curve.csv", inputs)
    folds = load_fold_definitions(cfg, directory, inputs)
    data_years = sorted({f[k] for f in folds for k in ("test_year", "inner_year")})
    universe, intraday, daily = load_frozen_market_data({**cfg, "test_years": data_years}, directory, inputs)
    validation = freeze_frequency_rules(folds, universe, intraday, daily, cfg, directory, inputs, stores)
    annual, results = evaluate_frequency_folds(folds, universe, intraday, daily, cfg, directory, inputs, stores)
    write_json(directory / "input_sources.json", inputs)
    for path, identity in inputs.items():
        if calculate_file_sha256(Path(path)) != identity["sha256"]:
            raise AssertionError(f"运行期间只读来源变化：{path}")
    hashes = json.loads((directory / "source_hashes.json").read_text())
    for name, expected in hashes.items():
        if calculate_file_sha256(PROJECT_ROOT / name) != expected:
            raise AssertionError(f"运行期间源码变化：{name}")
    write_json(directory / "runtime_integrity_audit.json", {"input_files_verified": len(inputs),
               "runtime_sources_verified": len(hashes), "all_unchanged": True})
    write_frequency_report(directory, cfg, annual, results, validation)
    print(f"报告完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
