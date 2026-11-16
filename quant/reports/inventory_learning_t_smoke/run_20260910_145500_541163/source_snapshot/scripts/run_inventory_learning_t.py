"""在quant环境重训永久底仓做T：纯核心队列拟合、内部真实账户选参、达标后外评。

只新建模型/报告，复用已有只读市场因子与固定端点标签；不覆盖旧研究。每股
名义初始1万—2万元并永久保留底仓，真实T+1及完整分钟动态恢复沿用v13执行器。
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
from quant_backtest.dynamic_t_engine import run_dynamic_portfolio
from quant_backtest.inventory_t_learning_data import build_inventory_core_cohort
from quant_backtest.inventory_t_learning_models import (VERSION, fit_inventory_return_model,
    predict_inventory_return_model, attach_learning_labels)
from quant_backtest.inventory_t_learning_policy import (build_inventory_learning_candidates,
    generate_inventory_learning_decisions, create_inventory_learning_rule_config)
from quant_backtest.inventory_t_learning_reporting import write_inventory_learning_report
from quant_backtest.weekly_t_horizon_data import validate_horizon_config
from quant_backtest.weekly_t_quote_store import calculate_file_sha256
from quant_backtest.weekly_t_timing import add_timing_features
from quant_backtest.weekly_t_timing_data import load_labels_only
from scripts.run_dynamic_inventory_t import (prepare_account_context, save_dynamic_result,
    get_store, snapshot_sources as snapshot_dynamic_sources, write_frame)
from scripts.run_weekly_t_active import load_frozen_market_data


_WORKER = {}


def load_configuration(argv: list[str] | None = None) -> dict:
    """读取预声明YAML；smoke仅缩评价账户，仍用同一完整核心训练队列。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/inventory_learning_t_v14.yaml")
    parser.add_argument("--test-years", type=int, nargs="+")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--smoke-days", type=int)
    parser.add_argument("--candidate-workers", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--internal-only", action="store_true")
    args = parser.parse_args(argv)
    source = (PROJECT_ROOT/args.config).resolve()
    cfg = yaml.safe_load(source.read_text())
    for key in ("test_years", "max_symbols", "smoke_days", "candidate_workers", "workers", "output_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    for key in ("data_root", "market_cache_dir", "source_audit_report_dir", "source_training_report_dir",
                "frozen_selection_report_dir", "output_dir", "manual_buys_csv", "full_minute_cache_dir"):
        if cfg.get(key):
            cfg[key] = str((PROJECT_ROOT/cfg[key]).resolve())
    cfg["dense_recovery_cache_paths"] = {str(y): str((PROJECT_ROOT/p).resolve())
        for y, p in cfg["dense_recovery_cache_paths"].items()}
    if cfg.get("research_version") != VERSION or cfg.get("candidate_activation_policy") != "best_execution_valid_active":
        raise ValueError("本入口要求v14永久底仓训练、内部净额优先协议")
    if not cfg["test_years"] or len(set(cfg["test_years"])) != len(cfg["test_years"]):
        raise ValueError("留出年份须非空且不重复")
    if min(cfg["max_symbols"], cfg["smoke_days"]) < 0 or min(cfg["workers"], cfg["candidate_workers"]) < 1:
        raise ValueError("烟测范围和工作进程数非法")
    original = json.loads((Path(cfg["source_training_report_dir"])/"resolved_config.json").read_text())
    unchanged = ("initial_cash", "lot_size", "commission_rate", "min_commission", "transfer_rate",
        "stamp_rate_before_20230828", "stamp_rate_from_20230828", "slippage_rate", "max_bar_amount_fraction",
        "repurchase_buffer", "flat_st_limit_rate", "nonflat_limit_proxy", "signal_times", "min_history_days",
        "min_avg_daily_amount", "min_price", "top_n", "weekly_top_n", "weekly_return_weight",
        "weekly_amplitude_weight", "min_prior_close_auction_amount", "embargo_days", "excluded_model_features")
    for key in unchanged:
        if cfg[key] != original[key]:
            raise ValueError(f"原选股/数据/费用合同改变：{key}")
    fixed = {"min_position_notional": 10000, "target_position_notional": 15000, "max_position_notional": 20000,
        "base_fraction": .85, "max_cycles_per_day": 0, "profit_target_rate": .05,
        "last_entry_time": "14:30", "fixed_recovery_time": "14:55", "max_pending_t_notional_fraction": .2,
        "strict_feature_provenance": True, "exclude_flat_signal_bars": False, "outer_t_refit": False,
        "t_max_train_rows": 0, "t_max_features": 60}
    for key, value in fixed.items():
        if cfg.get(key) != value:
            raise ValueError(f"v14事前协议不一致：{key}")
    if cfg["t_model"] != {"n_estimators": 300, "learning_rate": .04, "min_child_samples": 500,
                         "reg_lambda": 20., "verbosity": -1}:
        raise ValueError("保持预声明的原回归器参数")
    if len(build_inventory_learning_candidates(cfg)) != 37:
        raise ValueError("须有36主动候选和一个无T对照")
    _, _, minutes = validate_horizon_config(cfg)
    cfg["scheduled_recovery_times"] = [f"{v//60:02d}:{v%60:02d}" for v in minutes]
    cfg["configuration_source"] = str(source)
    cfg["internal_only"] = args.internal_only
    cfg["report_scope"] = "smoke" if cfg["max_symbols"] or cfg["smoke_days"] else "formal"
    cfg["outer_evaluation_policy"] = "all_inner_holding_cost_targets_met_before_outer"
    return cfg


def snapshot_sources(directory: Path, cfg: dict) -> None:
    """继承全部冻结实际依赖，新增本轮源快照，不改v13或更早源码。"""
    snapshot_dynamic_sources(directory, cfg)
    hashes = json.loads((directory/"source_hashes.json").read_text())
    names = ["scripts/run_inventory_learning_t.py", "quant_backtest/inventory_t_learning_data.py",
             "quant_backtest/inventory_t_learning_models.py", "quant_backtest/inventory_t_learning_policy.py",
             "quant_backtest/inventory_t_learning_reporting.py"]
    for name in names:
        target = directory/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT/name, target)
        hashes[name] = calculate_file_sha256(PROJECT_ROOT/name)
    write_json(directory/"source_hashes.json", hashes)


def predict_and_audit(frame: pd.DataFrame, model_dir: Path, cfg: dict, out: Path, inputs: dict) -> pd.DataFrame:
    """以原生模型复算每条当前预测，擦除未来字段后逐值相等；保存实际输入供审计。"""
    out.mkdir(parents=True, exist_ok=False)
    for name in ("model_bundle.joblib", "return_model.txt", "model_metadata.json"):
        record_balanced_input(model_dir/name, inputs)
    bundle = joblib.load(model_dir/"model_bundle.joblib")
    predictions = predict_inventory_return_model(bundle, frame, cfg)
    future = [name for name in frame if name.startswith(("execution_", "recovery_", "target_", "label_"))]
    clean = frame.drop(columns=future)
    repeated = predict_inventory_return_model(bundle, clean, cfg)
    pd.testing.assert_series_equal(predictions.prediction, repeated.prediction, check_exact=True)
    native = lightgbm.Booster(model_file=str(model_dir/"return_model.txt"))
    matrix = frame[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    native_values = native.predict(matrix) if len(matrix) else np.array([])
    error = float(np.max(np.abs(native_values-predictions.prediction.to_numpy()))) if len(frame) else 0.
    if error > 1e-12 or native.feature_name() != bundle["features"]:
        raise AssertionError("原生模型无法复现当前预测")
    policy_fields = ["datetime", "symbol", "prediction", "raw_prediction", "model_profile", "horizon_minutes",
        "signal_price", "signal_high", "signal_low", "signal_amount", "signal_volume", "previous_close"]
    archived = list(dict.fromkeys(policy_fields+bundle["features"]+["feature_source_start"]))
    write_frame(predictions[archived], out/"current_predictions.csv")
    write_json(out/"prediction_audit.json", {"passed": True, "current_rows": len(frame),
        "future_fields_removed": future, "future_independent": True, "native_max_error": error,
        "features": bundle["features"], "model_bundle_sha256": calculate_file_sha256(model_dir/"model_bundle.joblib"),
        "predictions_sha256": calculate_file_sha256(out/"current_predictions.csv")})
    return predictions[policy_fields].copy()


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
    local = create_inventory_learning_rule_config(c["cfg"], rule)
    result = run_dynamic_portfolio(c["selected"], c["daily"], c["signals"], decisions, local, recovery_bars=c["store"])
    curve = result["equity_curve"]
    if (not pd.to_datetime(curve.datetime).reset_index(drop=True).equals(c["baseline"].datetime)
            or not np.allclose(curve.baseline_equity, c["baseline"].equity, atol=1e-8, rtol=0)):
        raise AssertionError("不同T规则的无T基准发生改变")
    metrics = save_dynamic_result(result, out, local)
    return {**metrics, **rule, "request_count": len(decisions), "candidate_directory": str(out)}


def predict_account_models(signals, fold_out, cfg, out, inputs, profiles=None) -> dict:
    """只加载本折已拟合核心权重，不使用评价年重新拟合/校准。"""
    forecasts = {}
    for profile in cfg["learning_model_profiles"] if profiles is None else profiles:
        forecasts[profile] = {}
        for horizon in cfg["horizons"]:
            key = f"h{horizon}_{profile}"
            forecasts[profile][horizon] = predict_and_audit(signals, fold_out/"models"/key,
                {**cfg, "t_model_profile": profile, "t_label_horizon_minutes": horizon}, out/key, inputs)
    return forecasts


def write_internal_report(directory, folds, choices, targets_passed):
    """未晋级时仅报告完整内部结果，不制造外层成绩或外层收益图。"""
    lines = ["# 永久底仓训练择时：内部研究结果", "", "正式外层未执行。下表是内部年度选参结果，不是年度验证集成绩。", "",
        "| 留出年 | 内部年 | 规则 | 完整组 | 配对手 | T净额（元） | 成本降低率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in folds:
        item = choices[fold["test_year"]]["inner_metrics"]
        lines.append(f"| {fold['test_year']} | {fold['inner_year']} | {int(item['candidate_index'])} | {int(item['t_pair_count'])} | {item['completed_t_pair_lots']:.0f} | {item['incremental_net_pnl']:.2f} | {item['holding_cost_reduction_rate']:.4%} |")
    lines += ["", "核心模型用永久底仓开盘旧股机会学习固定端点参考收益；实际共享账户费用、容量、现金、T+1和分钟恢复由独立执行器检验。",
        "", "所有内部折达到5%底仓成本降低率后才进入统一外评。每股初始名义1万—2万元，底仓永久保留；没有回选历史外层结果。"]
    (directory/"final_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(directory/"goal_evaluation.json", {"goal_achieved": False, "formal_outer_evaluated": False,
        "all_inner_cost_targets_met": targets_passed, "reason": "internal_research_only"})


def main(argv: list[str] | None = None) -> None:
    """冻结协议、核心训练、37内部真实账户、统一规则冻结及必要时一次外层。"""
    cfg = load_configuration(argv)
    directory = Path(cfg["output_dir"])/f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory, cfg)
    write_json(directory/"resolved_config.json", cfg)
    rules = build_inventory_learning_candidates(cfg)
    write_json(directory/"research_protocol.json", {"version": VERSION, "previous_historical_outer_results_seen": True,
        "candidate_rules": rules, "models_per_fold": 6, "outer_model_refit": False,
        "training_cohort": "core_only_oof_weekly_entry_purged_before_permanent_baseline_planning",
        "training_labels": "next_minute_to_fixed_20_40_60_minute_reference_return",
        "labels_are_actual_shared_account_pnl": False, "selection_uses_inner_real_accounts_only": True,
        "no_weekly_middle_day_restriction": True, "no_probability_or_old_support_gate": True,
        "full_core_training_also_in_smoke": True, "outer_evaluation_policy": cfg["outer_evaluation_policy"],
        "valuation_contract": "full_minute_reference_stream_then_daily_close",
        "sources": ["https://doi.org/10.1016/j.frl.2019.04.002", "https://arxiv.org/abs/1411.5062"],
        "literature_not_a_profit_guarantee": True})
    print(f"输出目录：{directory}", flush=True)
    inputs, stores = {}, {}
    record_balanced_input(cfg["configuration_source"], inputs)
    write_json(directory/"source_reuse_audit.json", validate_balanced_sources(cfg, inputs, PROJECT_ROOT))
    folds = [read_balanced_fold(cfg, year, inputs) for year in cfg["test_years"]]
    for fold in folds:
        fold["t_model_fit"] = True
    needed = sorted({y for f in folds for y in [f["test_year"], f["inner_year"], *f["core_years"]]})
    print("复用完整因子/来源，准备永久底仓核心训练；不重建原始特征缓存", flush=True)
    universe, intraday, daily = load_frozen_market_data({**cfg, "test_years": needed}, directory, inputs)
    known = json.loads((Path(cfg["source_audit_report_dir"])/"input_sources.json").read_text())
    for name in ("weekly", "intraday", "daily"):
        path = (Path(cfg["market_cache_dir"])/f"{name}.pkl").resolve()
        if str(path) not in known:
            raise ValueError("原市场PKL缺已审计身份")
        record_balanced_input(path, inputs, known[str(path)]["sha256"])
    intraday = add_timing_features(intraday, cfg)
    labels = load_labels_only(cfg, directory, inputs, record_balanced_input)
    calendar = pd.DatetimeIndex(daily.date.unique()).sort_values()
    tables, choices, model_hashes = [], {}, {}
    for fold in folds:
        year = fold["test_year"]
        out = directory/f"year_{year}"
        out.mkdir()
        write_json(out/"split.json", fold)
        core, cohort_audit = build_inventory_core_cohort(universe, daily, intraday, fold,
            cfg, out/"training_cohort", inputs, record_balanced_input)
        write_json(out/"training_cohort_summary.json", cohort_audit)
        print(f"[{year}] 永久旧股核心机会{len(core):,}行；开始6个独立回归模型", flush=True)
        for horizon in cfg["horizons"]:
            training = attach_learning_labels(core, labels, horizon)
            for profile in cfg["learning_model_profiles"]:
                key = f"h{horizon}_{profile}"
                model = out/"models"/key
                metadata = fit_inventory_return_model(training, fold["core_years"],
                    {**cfg, "t_model_profile": profile, "t_label_horizon_minutes": horizon}, model,
                    [fold["inner_year"], year])
                model_hashes.update({str(path.relative_to(directory)): calculate_file_sha256(path)
                    for path in sorted(model.iterdir()) if path.is_file()})
                print(f"[{year}] {key}拟合{metadata['fitted_rows']:,}行/{len(metadata['features'])}特征", flush=True)
            del training
        del core
        gc.collect()
        context = prepare_account_context(fold, "validation", daily, intraday, calendar, cfg,
            out/"validation_context", inputs, get_store(fold["inner_year"], cfg, inputs, stores))
        selected, prices, signals, local_cfg, store = context
        forecasts = predict_account_models(signals, out, local_cfg, out/"validation_predictions", inputs)
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
        write_json(out/"selected_rule.json", {**choices[year], "frozen_before_any_outer_test": True})
        tables.append(annotated.assign(test_year=year, inner_year=fold["inner_year"]))
        print(f"[{year}] 冻结{rule['candidate_index']}；内部净额{chosen['incremental_net_pnl']:.2f}元，成本降低{chosen['holding_cost_reduction_rate']:.4%}", flush=True)
        del context, forecasts, signals, args
        gc.collect()
    validation = pd.concat(tables, ignore_index=True)
    write_frame(validation, directory/"validation_rule_metrics.csv")
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
            raise AssertionError("拟合模型在外评前改变")
    if evaluate_outer:
        for fold in folds:
            year, out = fold["test_year"], directory/f"year_{fold['test_year']}"
            if calculate_file_sha256(out/"selected_rule.json") != frozen[str(year)]:
                raise AssertionError("外评前冻结规则改变")
            selected, prices, signals, local_cfg, store = prepare_account_context(fold, "test", daily, intraday,
                calendar, cfg, out/"test_context", inputs, get_store(year, cfg, inputs, stores))
            rule = choices[year]["rule"]
            forecasts = predict_account_models(signals, out, local_cfg, out/"test_predictions", inputs,
                profiles=[rule["profile"]]) if rule["enabled"] else {}
            decisions, audit = generate_inventory_learning_decisions(forecasts.get(rule["profile"], {}), rule, local_cfg)
            write_frame(decisions, out/"t_decisions.csv")
            write_json(out/"signal_causality_audit.json", audit)
            local_cfg = create_inventory_learning_rule_config(local_cfg, rule)
            result = run_dynamic_portfolio(selected, prices, signals[INTRADAY_COLUMNS].copy(), decisions, local_cfg, recovery_bars=store)
            result["selected_rule"] = rule
            save_dynamic_result(result, out, local_cfg)
            results.append((year, result))
        write_inventory_learning_report(directory, cfg, results, validation)
    else:
        write_internal_report(directory, folds, choices, passed)
    write_json(directory/"input_sources.json", inputs)
    for name, identity in inputs.items():
        if calculate_file_sha256(name) != identity["sha256"]:
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
