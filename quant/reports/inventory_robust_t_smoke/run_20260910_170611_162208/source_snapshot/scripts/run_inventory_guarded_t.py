"""复用已审计永久底仓模型，独立检验首腿成交报价的反向限价复核。

当前预测、候选决策和底仓与来源v14逐项相同；唯一执行变化是首腿报价到达时，
若其反向交易已受原限价规则约束，则取消首腿。当前报价不证明未来一定可恢复。
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
from quant_backtest.inventory_t_learning_policy import (build_inventory_learning_candidates,
    generate_inventory_learning_decisions, create_inventory_learning_rule_config)
from quant_backtest.guarded_t_reporting import write_guarded_inventory_report
from quant_backtest.weekly_t_quote_store import calculate_file_sha256
from quant_backtest.weekly_t_timing import add_timing_features
from scripts.run_dynamic_inventory_t import (prepare_account_context, save_dynamic_result,
    get_store, write_frame)
from scripts.run_weekly_t_active import load_frozen_market_data
from scripts.run_inventory_learning_t import (load_configuration as load_learning_configuration,
    snapshot_sources as snapshot_learning_sources, predict_account_models)

VERSION = "inventory_guarded_t_v15"


_WORKER = {}


def load_configuration(argv: list[str] | None = None) -> dict:
    """继承来源已冻结配置和评价范围；新YAML只声明同刻执行复核与来源。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/inventory_guarded_t_v15.yaml")
    parser.add_argument("--source-report")
    parser.add_argument("--test-years", type=int, nargs="+")
    parser.add_argument("--candidate-workers", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--internal-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = (PROJECT_ROOT/args.config).resolve()
    declared = yaml.safe_load(config_path.read_text())
    expected_keys = {"research_version", "source_learning_report_dir", "output_dir",
                     "t_recheck_exit_side_at_execution", "candidate_workers", "workers"}
    if (not isinstance(declared, dict) or set(declared) != expected_keys
            or declared["research_version"] != VERSION
            or declared["t_recheck_exit_side_at_execution"] is not True):
        raise ValueError("v15配置只允许已声明来源、工作进程及开启当刻反向复核")
    source = (PROJECT_ROOT/(args.source_report or declared["source_learning_report_dir"])).resolve()
    original = json.loads((source/"resolved_config.json").read_text())
    if original.get("research_version") != "inventory_learning_t_v14":
        raise ValueError("必须复用已完成的v14永久底仓模型")
    old_args = ["--config", original["configuration_source"], "--test-years", *map(str, original["test_years"])]
    for key in ("max_symbols", "smoke_days", "candidate_workers", "workers", "output_dir"):
        old_args += ["--"+key.replace("_", "-"), str(original[key])]
    if original["internal_only"]:
        old_args.append("--internal-only")
    if load_learning_configuration(old_args) != original:
        raise ValueError("来源配置不能由原v14合同精确复现")
    years = args.test_years or original["test_years"]
    if not years or len(set(years)) != len(years) or not set(years).issubset(original["test_years"]):
        raise ValueError("留出年必须是来源已有模型年份的非空无重复子集")
    cfg = {**original, **declared, "configuration_source": str(config_path),
           "source_learning_report_dir": str(source), "test_years": years,
           "source_configuration_source": original["configuration_source"],
           "source_learning_version": original["research_version"],
           "source_report_scope": original["report_scope"],
           "frozen_t_models_reused": True, "t_models_fitted_this_run": False,
           "internal_only": args.internal_only,
           "output_dir": str((PROJECT_ROOT/(args.output_dir or declared["output_dir"])).resolve())}
    for key in ("candidate_workers", "workers"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError("工作进程数必须为正整数")
    return cfg


def snapshot_sources(directory: Path, cfg: dict) -> None:
    """保留全部48个原依赖，加独立执行器、入口及报告器；不改变历史源码。"""
    snapshot_learning_sources(directory, cfg)
    hashes = json.loads((directory/"source_hashes.json").read_text())
    for name in ("scripts/run_inventory_guarded_t.py", "quant_backtest/guarded_t_engine.py",
                 "quant_backtest/guarded_t_reporting.py"):
        target = directory/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT/name, target)
        hashes[name] = calculate_file_sha256(PROJECT_ROOT/name)
    write_json(directory/"source_hashes.json", hashes)


def validate_learning_source(cfg: dict, directory: Path, inputs: dict) -> dict:
    """要求来源完整独立审计，核对其配置、旧输入、48源码及实际模型身份。"""
    source = Path(cfg["source_learning_report_dir"])
    for name in ("resolved_config.json", "runtime_integrity_audit.json", "source_hashes.json",
                 "input_sources.json", "all_rules_frozen_before_outer_test.json", "research_protocol.json"):
        record_balanced_input(source/name, inputs)
    integrity = json.loads((source/"runtime_integrity_audit.json").read_text())
    if integrity.get("all_unchanged") is not True:
        raise ValueError("来源运行未完成全部完整性核验")
    audits = []
    for path in sorted(source.glob("independent_learning_audit/run_*/verification.json")):
        audit = json.loads(path.read_text())
        if audit.get("passed") and audit.get("complete") and audit.get("all_read_inputs_unchanged"):
            audits.append((path, audit))
    if not audits:
        raise ValueError("来源须先完成完整独立模型与账户审计")
    audit_path, audit = audits[-1]
    record_balanced_input(audit_path, inputs)
    # 消费已保存模型前重新核其原训练/执行输入，保留旧审计与本次复用的身份链。
    original_inputs = json.loads((source/"input_sources.json").read_text())
    for name, identity in original_inputs.items():
        record_balanced_input(name, inputs, identity["sha256"])
    for name, digest in json.loads((source/"source_hashes.json").read_text()).items():
        record_balanced_input(PROJECT_ROOT/name, inputs, digest)
        record_balanced_input(source/"source_snapshot"/name, inputs, digest)
    frozen = json.loads((source/"all_rules_frozen_before_outer_test.json").read_text())
    artifacts = {}
    for name, digest in frozen["model_artifacts_sha256"].items():
        path = (source/name).resolve()
        record_balanced_input(path, inputs, digest)
        artifacts[str(path)] = digest
    if len(artifacts) != 48*len(json.loads((source/"resolved_config.json").read_text())["test_years"]):
        raise ValueError("每来源折须有6模型、每模型8项冻结产物")
    for name in ("resolved_config.json", "source_hashes.json", "all_rules_frozen_before_outer_test.json"):
        path = (source/name).resolve()
        expected = audit["inputs_sha256"].get(str(path))
        if expected is None or calculate_file_sha256(path) != expected:
            raise ValueError("来源关键产物已偏离独立审计")
    write_json(directory/"reused_model_artifacts_sha256.json", artifacts)
    write_json(directory/"learning_source_reuse_audit.json", {"passed": True,
        "source_report": str(source), "independent_audit": str(audit_path),
        "model_artifacts_verified": len(artifacts), "original_inputs_verified": len(original_inputs),
        "model_learning_version": "inventory_learning_t_v14", "no_model_refit": True,
        "same_predictions_and_decisions_required": True})
    return artifacts


def assert_source_table_equal(current: pd.DataFrame, path: Path, inputs: dict) -> None:
    """当前预测、决策或基准必须精确复现来源；只允许保存时无意义的dtype差异。"""
    record_balanced_input(path, inputs)
    original = pd.read_csv(path, float_precision="round_trip")
    if list(current.columns) != list(original.columns):
        raise AssertionError(f"来源表列发生改变：{path}")
    left = current.copy().reset_index(drop=True)
    for name in left:
        if pd.api.types.is_datetime64_any_dtype(left[name]):
            original[name] = pd.to_datetime(original[name])
    pd.testing.assert_frame_equal(left, original, check_dtype=False, check_exact=True)


def save_guarded_result(result: dict, out: Path, cfg: dict) -> dict:
    """沿用原核算并附当刻复核次数；报价可填并不代表现金等后续约束已批准。"""
    audit = result["first_leg_exit_side_audit"]
    checks = audit.loc[audit.guard_enabled.eq(True)]
    result["metrics"].update({"t_recheck_exit_side_at_execution": cfg["t_recheck_exit_side_at_execution"],
        "execution_exit_side_checks": len(checks),
        "execution_exit_side_cancellations": int(checks.cancelled.eq(True).sum())})
    metrics = save_dynamic_result(result, out, cfg)
    write_frame(audit, out/"first_leg_exit_side_audit.csv")
    return metrics


def write_source_comparison(current: pd.DataFrame, source_path: Path, out: Path, inputs: dict) -> pd.DataFrame:
    """同候选配对比较执行条件；禁止将不同候选的增量相加当作一个组合。"""
    record_balanced_input(source_path, inputs)
    original = pd.read_csv(source_path, float_precision="round_trip")
    key = "candidate_index"
    fields = [key, "enabled", "profile", "min_net_edge", "fraction", "exit_mode"]
    pd.testing.assert_frame_equal(current[fields].reset_index(drop=True), original[fields],
                                  check_dtype=False, check_exact=True)
    values = ["incremental_net_pnl", "holding_cost_reduction_rate", "t_pair_count",
              "completed_t_pair_lots", "t_fee", "t_slippage_cost", "inventory_mismatch_days",
              "recovery_failure_days", "pending_pairs", "all_reference_acquisition_cost"]
    combined = current[[key, *values, "execution_exit_side_checks", "execution_exit_side_cancellations"]].merge(
        original[[key, *values]], on=key, validate="one_to_one", suffixes=("_guarded", "_source"))
    if not np.allclose(combined.all_reference_acquisition_cost_guarded,
                       combined.all_reference_acquisition_cost_source, atol=1e-8, rtol=0):
        raise AssertionError("反向复核改变了底仓取得成本分母")
    for name in values:
        combined[name+"_change"] = combined[name+"_guarded"]-combined[name+"_source"]
    write_frame(combined, out/"guarded_vs_source_candidates.csv")
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
    decisions, audit = generate_inventory_learning_decisions(forecasts, rule, c["cfg"])
    write_frame(decisions, out/"t_decisions.csv")
    write_json(out/"signal_causality_audit.json", audit)
    source_decisions = (Path(c["cfg"]["source_learning_report_dir"])
        /f"year_{c['cfg']['current_test_year']}"
        /f"validation_candidates/candidate_{rule['candidate_index']:03d}/t_decisions.csv")
    assert_source_table_equal(decisions, source_decisions, {})
    local = create_inventory_learning_rule_config(c["cfg"], rule)
    result = run_guarded_portfolio(c["selected"], c["daily"], c["signals"], decisions, local, recovery_bars=c["store"])
    curve = result["equity_curve"]
    if (not pd.to_datetime(curve.datetime).reset_index(drop=True).equals(c["baseline"].datetime)
            or not np.allclose(curve.baseline_equity, c["baseline"].equity, atol=1e-8, rtol=0)):
        raise AssertionError("不同T规则的无T基准发生改变")
    metrics = save_guarded_result(result, out, local)
    write_json(out/"source_decisions_identity.json", {"same_as_source": True, "path": str(source_decisions),
        "sha256": calculate_file_sha256(source_decisions)})
    return {**metrics, **rule, "request_count": len(decisions), "candidate_directory": str(out)}


def write_internal_report(directory, folds, choices, targets_passed):
    """未晋级时仅报告完整内部结果，不制造外层成绩或外层收益图。"""
    lines = ["# 首腿反向限价复核：内部研究结果", "", "正式外层未执行。下表是内部年度选参结果，不是年度验证集成绩。", "",
        "| 留出年 | 内部年 | 规则 | 完整组 | 配对手 | T净额（元） | 成本降低率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in folds:
        item = choices[fold["test_year"]]["inner_metrics"]
        lines.append(f"| {fold['test_year']} | {fold['inner_year']} | {int(item['candidate_index'])} | {int(item['t_pair_count'])} | {item['completed_t_pair_lots']:.0f} | {item['incremental_net_pnl']:.2f} | {item['holding_cost_reduction_rate']:.4%} |")
    lines += ["", "复用v14核心模型；内部预测、37候选决策和无T底仓逐项相同。唯一变化为首腿当刻按原报价规则复核反向限价，受限则取消首腿。当前可交易仍不能保证未来可恢复。",
        "", "所有内部折达到5%底仓成本降低率后才进入统一外评。每股初始名义1万—2万元，底仓永久保留；没有回选历史外层结果。"]
    (directory/"final_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(directory/"goal_evaluation.json", {"goal_achieved": False, "formal_outer_evaluated": False,
        "all_inner_cost_targets_met": targets_passed, "reason": "internal_research_only"})


def main(argv: list[str] | None = None) -> None:
    """冻结复用模型和同刻执行条件；37内部账户后按原5%协议评价外层。"""
    cfg = load_configuration(argv)
    directory = Path(cfg["output_dir"])/f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory, cfg)
    write_json(directory/"resolved_config.json", cfg)
    rules = build_inventory_learning_candidates(cfg)
    write_json(directory/"research_protocol.json", {"version": VERSION, "previous_historical_outer_results_seen": True,
        "candidate_rules": rules, "reused_models_per_fold": 6, "models_fitted_this_run": 0, "outer_model_refit": False,
        "source_learning_report_dir": cfg["source_learning_report_dir"],
        "only_execution_change": "current_first_leg_quote_opposite_limit_recheck",
        "guard_does_not_predict_future_recoverability": True,
        "source_model_training_cohort": "core_only_oof_weekly_entry_purged_before_permanent_baseline_planning",
        "source_model_training_labels": "next_minute_to_fixed_20_40_60_minute_reference_return",
        "labels_are_actual_shared_account_pnl": False, "selection_uses_inner_real_accounts_only": True,
        "no_weekly_middle_day_restriction": True, "no_probability_or_old_support_gate": True,
        "reused_models_trained_on_full_core_even_for_source_smoke": True,
        "outer_evaluation_policy": cfg["outer_evaluation_policy"],
        "valuation_contract": "full_minute_reference_stream_then_daily_close",
        "sources": ["https://doi.org/10.1016/j.frl.2019.04.002", "https://arxiv.org/abs/1411.5062"],
        "literature_sources_are_for_reused_learning_models": True,
        "literature_not_a_profit_guarantee": True})
    print(f"输出目录：{directory}", flush=True)
    inputs, stores = {}, {}
    record_balanced_input(cfg["configuration_source"], inputs)
    model_hashes = validate_learning_source(cfg, directory, inputs)
    market_audit = validate_balanced_sources(cfg, inputs, PROJECT_ROOT)
    market_audit["pre_v14_legacy_t_models_reused"] = market_audit.pop("old_t_models_reused")
    market_audit["frozen_inventory_learning_t_models_reused"] = True
    market_audit["holding_contract_matches_source_learning"] = True
    write_json(directory/"source_reuse_audit.json", market_audit)
    folds = [read_balanced_fold(cfg, year, inputs) for year in cfg["test_years"]]
    for fold in folds:
        fold["t_model_fit"] = False
        fold["t_models_reused_from"] = cfg["source_learning_report_dir"]
    needed = sorted({y for f in folds for y in [f["test_year"], f["inner_year"], *f["core_years"]]})
    print("复用完整因子与v14冻结模型；当前预测和候选决策逐项对照来源", flush=True)
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
    source = Path(cfg["source_learning_report_dir"])
    for fold in folds:
        year = fold["test_year"]
        out = directory/f"year_{year}"
        out.mkdir()
        write_json(out/"split.json", fold)
        fold_cfg = {**cfg, "current_test_year": year}
        context = prepare_account_context(fold, "validation", daily, intraday, calendar, fold_cfg,
            out/"validation_context", inputs, get_store(fold["inner_year"], cfg, inputs, stores))
        selected, prices, signals, local_cfg, store = context
        forecasts = predict_account_models(signals, source/f"year_{year}", local_cfg, out/"validation_predictions", inputs)
        for path in sorted((out/"validation_predictions").glob("*/current_predictions.csv")):
            compare = pd.read_csv(path, float_precision="round_trip")
            assert_source_table_equal(compare, source/f"year_{year}"/path.relative_to(out), inputs)
        for relative in ("validation_context/frozen_base_selections.csv", "validation_context/frozen_base_entries.csv",
                         "validation_context/baseline/test_equity_curve.csv"):
            compare = pd.read_csv(out/relative, float_precision="round_trip")
            assert_source_table_equal(compare, source/f"year_{year}"/relative, inputs)
        for rule in rules:
            record_balanced_input(source/f"year_{year}/validation_candidates/candidate_{rule['candidate_index']:03d}/t_decisions.csv", inputs)
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
    write_frame(pd.concat(comparisons, ignore_index=True), directory/"guarded_vs_source_candidates.csv")
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
        if calculate_file_sha256(Path(name)) != digest:
            raise AssertionError("冻结复用模型在外评前改变")
    if evaluate_outer:
        for fold in folds:
            year, out = fold["test_year"], directory/f"year_{fold['test_year']}"
            if calculate_file_sha256(out/"selected_rule.json") != frozen[str(year)]:
                raise AssertionError("外评前冻结规则改变")
            selected, prices, signals, local_cfg, store = prepare_account_context(fold, "test", daily, intraday,
                calendar, cfg, out/"test_context", inputs, get_store(year, cfg, inputs, stores))
            rule = choices[year]["rule"]
            forecasts = predict_account_models(signals, source/f"year_{year}", local_cfg, out/"test_predictions", inputs,
                profiles=[rule["profile"]]) if rule["enabled"] else {}
            decisions, audit = generate_inventory_learning_decisions(forecasts.get(rule["profile"], {}), rule, local_cfg)
            write_frame(decisions, out/"t_decisions.csv")
            write_json(out/"signal_causality_audit.json", audit)
            local_cfg = create_inventory_learning_rule_config(local_cfg, rule)
            result = run_guarded_portfolio(selected, prices, signals[INTRADAY_COLUMNS].copy(), decisions, local_cfg, recovery_bars=store)
            result["selected_rule"] = rule
            save_guarded_result(result, out, local_cfg)
            results.append((year, result))
        write_guarded_inventory_report(directory, cfg, results, validation)
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
        if calculate_file_sha256(Path(name)) != digest:
            raise AssertionError(f"运行期间模型产物改变：{name}")
    write_json(directory/"runtime_integrity_audit.json", {"all_unchanged": True,
        "input_files_verified": len(inputs), "runtime_sources_verified": len(sources), "model_artifacts_verified": len(model_hashes)})
    print(f"研究完成：{directory/'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
