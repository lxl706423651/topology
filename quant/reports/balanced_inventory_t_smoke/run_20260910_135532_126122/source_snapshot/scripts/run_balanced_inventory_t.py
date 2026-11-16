"""在quant工作区运行等金额固定底仓T研究；复用已审计PKL，保存新目录。

先执行工程烟测，再完整评价内部年度全部预声明规则。每折按内部实际净收益
选执行合格主动规则；全部内部成本降低率达到目标后统一冻结并自动外评，也可
用--internal-only保留内部研究。不会重建原因子或覆盖历史模型/报告。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
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
from quant_backtest.balanced_t_data import (load_balanced_market, read_balanced_fold,
    read_balanced_selections, record_balanced_input, validate_balanced_sources)
from quant_backtest.balanced_t_engine import plan_balanced_entries, run_balanced_portfolio
from quant_backtest.balanced_t_policy import (VERSION, build_balanced_candidates,
    create_empty_balanced_decisions, generate_balanced_decisions, select_balanced_candidate)
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics, write_balanced_report
from quant_backtest.weekly_t_horizon_data import validate_horizon_config
from quant_backtest.weekly_t_quote_store import DenseRecoveryBarStore, calculate_file_sha256


RESULT_FILES = {"equity_curve": "test_equity_curve.csv", "trade_events": "test_trade_events.csv",
    "round_trips": "test_round_trips.csv", "open_positions": "test_open_positions.csv",
    "daily_inventory_audit": "daily_inventory_audit.csv", "order_attempts": "blocked_orders.csv",
    "baseline_events": "baseline_trade_events.csv", "weekly_entry_plans": "base_entry_plans.csv",
    "base_plan_mismatches": "base_plan_mismatches.csv", "position_amount_audit": "position_amount_audit.csv"}
_WORKER = {}


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    """保存新的标准UTF-8 BOM账表；调用方保证目录唯一。"""
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_configuration(argv: list[str] | None = None) -> dict:
    """YAML作为实验来源，CLI仅覆盖年份、烟测范围、并行度和输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/balanced_inventory_t_v12.yaml")
    parser.add_argument("--test-years", type=int, nargs="+")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--smoke-days", type=int)
    parser.add_argument("--candidate-workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--internal-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    for key in ("test_years", "max_symbols", "smoke_days", "candidate_workers", "output_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    for key in ("data_root", "market_cache_dir", "source_audit_report_dir", "source_training_report_dir",
                "frozen_selection_report_dir", "output_dir", "manual_buys_csv"):
        if cfg.get(key):
            cfg[key] = str((PROJECT_ROOT / cfg[key]).resolve())
    cfg["dense_recovery_cache_paths"] = {str(y): str((PROJECT_ROOT / p).resolve())
                                         for y, p in cfg["dense_recovery_cache_paths"].items()}
    if cfg.get("research_version") != VERSION or cfg.get("candidate_activation_policy") != "best_execution_valid_active":
        raise ValueError("本入口只实现v12预声明的内部净收益优先主动规则")
    if not cfg["test_years"] or len(set(cfg["test_years"])) != len(cfg["test_years"]):
        raise ValueError("年度须非空且不重复")
    if cfg["max_symbols"] < 0 or cfg["smoke_days"] < 0 or cfg["candidate_workers"] < 1:
        raise ValueError("烟测范围或并行数非法")
    frozen_fields = {"signal_families": ["vwap_reversal", "confirmed_reversal", "momentum"],
        "entry_deviations": [.003, .006, .01], "recovery_horizons": [20, 60, "close"],
        "t_trade_fractions": [.5, 1.], "min_position_notional": 10000,
        "target_position_notional": 15000, "max_position_notional": 20000,
        "base_fraction": .85, "max_cycles_per_day": 0, "profit_target_rate": .05,
        "last_entry_time": "14:30", "fixed_recovery_time": "14:55", "horizons": [20, 40, 60]}
    for key, expected in frozen_fields.items():
        if cfg.get(key) != expected:
            raise ValueError(f"v12事前协议{key}与默认候选合同不一致")
    source_cfg = json.loads((Path(cfg["source_training_report_dir"]) / "resolved_config.json").read_text())
    for key in ("initial_cash", "lot_size", "commission_rate", "min_commission", "transfer_rate",
                "stamp_rate_before_20230828", "stamp_rate_from_20230828", "slippage_rate",
                "max_bar_amount_fraction", "repurchase_buffer", "flat_st_limit_rate", "nonflat_limit_proxy", "signal_times"):
        if cfg[key] != source_cfg[key]:
            raise ValueError(f"v12保留原成本/行情合同，{key}发生变化")
    _, _, minutes = validate_horizon_config(cfg)
    cfg["scheduled_recovery_times"] = [f"{value // 60:02d}:{value % 60:02d}" for value in minutes]
    cfg["internal_only"] = args.internal_only
    cfg["report_scope"] = "smoke" if cfg["max_symbols"] or cfg["smoke_days"] else "formal"
    cfg["outer_evaluation_policy"] = "all_inner_holding_cost_targets_met_before_outer"
    cfg["configuration_source"] = str((PROJECT_ROOT / args.config).resolve())
    if len(build_balanced_candidates(cfg)) != 55:
        raise ValueError("v12要求54主动候选和1个hold")
    return cfg


def snapshot_sources(directory: Path, cfg: dict) -> None:
    """保存全部已冻结公共源码和新增研究源码，运行期间不允许它们变化。"""
    old = json.loads((Path(cfg["source_audit_report_dir"]) / "source_hashes.json").read_text())
    added = ["quant_backtest/balanced_t_data.py", "quant_backtest/balanced_t_engine.py",
             "quant_backtest/balanced_t_policy.py", "quant_backtest/balanced_t_reporting.py",
             "scripts/run_balanced_inventory_t.py"]
    names = sorted(set(old) | set(added))
    hashes = {}
    for name in names:
        path = PROJECT_ROOT / name
        target = directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = calculate_file_sha256(path)
    config_path = Path(cfg["configuration_source"])
    shutil.copy2(config_path, directory / "configuration_source.yaml")
    write_json(directory / "source_hashes.json", hashes)


def save_balanced_result(result: dict, directory: Path, cfg: dict) -> dict:
    """保存完整事件、配对、库存、成本分母及各股贡献；返回实际内部选择指标。"""
    directory.mkdir(parents=True, exist_ok=True)
    for key, filename in RESULT_FILES.items():
        if key in result:
            write_frame(result[key], directory / filename)
    summary, per_symbol = calculate_balanced_metrics(result, cfg)
    write_frame(per_symbol, directory / "per_symbol_t_quality.csv")
    curve = result["equity_curve"]
    metrics = {**result["metrics"], **summary,
               "incremental_net_pnl": float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1])}
    write_frame(pd.DataFrame([metrics]), directory / "test_metrics.csv")
    write_json(directory / "metrics.json", metrics)
    return metrics


def initialize_worker(selections: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame,
                      cfg: dict, store: DenseRecoveryBarStore, directory: str) -> None:
    """spawn工作进程只持有该组合的小型输入及只读mmap，不复制全市场大面板。"""
    global _WORKER
    _WORKER = {"selections": selections, "daily": daily, "intraday": intraday,
               "cfg": cfg, "store": store, "directory": Path(directory)}


def evaluate_worker(rule: dict) -> dict:
    """按冻结规则生成当前信号并执行真实账户；先保存证据再返回内部指标。"""
    context = _WORKER
    out = context["directory"] / f"candidate_{rule['candidate_index']:03d}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "rule.json", rule)
    decisions, funnel = generate_balanced_decisions(context["intraday"], rule, context["cfg"])
    future_columns = [key for key in context["intraday"] if key.startswith(("execution_", "recovery_", "target_", "label_"))]
    scrubbed = context["intraday"].drop(columns=future_columns)
    repeated, _ = generate_balanced_decisions(scrubbed, rule, context["cfg"])
    pd.testing.assert_frame_equal(decisions, repeated, check_exact=True)
    write_frame(decisions, out / "t_decisions.csv")
    write_json(out / "signal_causality_audit.json", {**funnel, "future_columns_removed": future_columns,
               "future_fields_independent": True})
    result = run_balanced_portfolio(context["selections"], context["daily"], context["intraday"],
                                    decisions, context["cfg"], recovery_bars=context["store"])
    metrics = save_balanced_result(result, out, context["cfg"])
    return {**metrics, **rule, "request_count": len(decisions), "candidate_directory": str(out)}


def prepare_account_context(fold: dict, scope: str, daily: pd.DataFrame, intraday: pd.DataFrame,
                            calendar: pd.DatetimeIndex, cfg: dict, out: Path, inputs: dict,
                            store: DenseRecoveryBarStore) -> tuple:
    """先按无T账户的因果真实买单冻结底仓，再加载其全持有期行情；保留所有市场日。"""
    evaluation_year = fold["inner_year"] if scope == "validation" else fold["test_year"]
    days = calendar[calendar.year == evaluation_year]
    if cfg["smoke_days"]:
        days = days[:cfg["smoke_days"]]
    if not len(days):
        raise ValueError("年度没有可评价交易日")
    selected = read_balanced_selections(cfg, fold, scope, days, inputs)
    out.mkdir(parents=True, exist_ok=False)
    write_frame(selected, out / "source_weekly_selections.csv")
    # 规划输入先保留全市场日历，避免个股缺价抹去应计现金日/恢复失败日。
    year_daily = daily.loc[daily.date.isin(days)].copy()
    planned = plan_balanced_entries(selected, year_daily, cfg, recovery_bars=store)
    frozen = planned["frozen_selections"]
    write_frame(frozen, out / "frozen_base_selections.csv")
    write_frame(planned["base_entries"], out / "frozen_base_entries.csv")
    write_frame(planned["planning_attempts"], out / "base_planning_attempts.csv")
    write_json(out / "base_plan_identity.json", {"sha256": planned["base_config_sha256"],
               "frozen_csv_sha256": calculate_file_sha256(out / "frozen_base_selections.csv"),
               "market_days": len(days), "no_t_profit_used_for_base": True,
               "scope": scope, "evaluation_year": evaluation_year})
    save_balanced_result(planned["baseline_result"], out / "baseline", planned["replay_cfg"])
    symbols = set(frozen.symbol)
    held_daily = year_daily.loc[year_daily.symbol.isin(symbols)].copy()
    # 在全现金或全部持股同日缺价时仍保留一个只用于日历的空报价哨兵。
    missing_days = days.difference(pd.DatetimeIndex(held_daily.date.unique()))
    if len(missing_days):
        placeholder = pd.DataFrame({"date": missing_days, "symbol": "__MARKET_CALENDAR__"})
        held_daily = pd.concat([held_daily, placeholder], ignore_index=True)
    signals = intraday.loc[intraday.symbol.isin(symbols) & intraday.datetime.dt.normalize().isin(days)].copy()
    write_json(out / "execution_data_scope.json", {"held_symbols": sorted(symbols), "daily_rows": len(held_daily),
               "signal_rows": len(signals), "full_market_dates_retained": True,
               "whole_holding_period_kept_after_weekly_pool_exit": True,
               "no_future_return_or_recovery_filter": True})
    replay_cfg = {**planned["replay_cfg"], "report_scope": "smoke" if cfg["report_scope"] == "smoke" else
                  "internal" if scope == "validation" else "formal"}
    return frozen, held_daily, signals, replay_cfg


def get_store(year: int, cfg: dict, inputs: dict, stores: dict) -> DenseRecoveryBarStore:
    """每年完整核验并打开一次原始恢复报价数组，不生成新行情或补价格。"""
    if year not in stores:
        path = Path(cfg["dense_recovery_cache_paths"][str(year)])
        record_balanced_input(path / "metadata.json", inputs)
        stores[year] = DenseRecoveryBarStore(path, verify_integrity=True)
    return stores[year]


def main(argv: list[str] | None = None) -> None:
    """完整内层比较、冻结所有规则、达标后外评；所有负结果与未外评状态如实保存。"""
    cfg = load_configuration(argv)
    directory = Path(cfg["output_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_sources(directory, cfg)
    write_json(directory / "resolved_config.json", cfg)
    rules = build_balanced_candidates(cfg)
    write_json(directory / "research_protocol.json", {
        "version": VERSION, "previous_historical_outer_results_seen": True,
        "base_contract": "Each stock 10000–20000 CNY at automatic first fill, target 15000; persist shares across weeks; no automatic base sale",
        "entry_ranking": "Original point-in-time weekly ranking; only add after that week's own signal; never buy the annual union at year start",
        "t_rules": rules, "t_model_refit": False, "selection_uses_only_inner_accounts": True,
        "outer_evaluation_policy": cfg["outer_evaluation_policy"],
        "profit_denominator": "All original baseline purchase costs including fees and manual additions; excludes every T repurchase",
        "frequency_target_not_quantified": cfg.get("minimum_completed_pair_lots") is None and cfg.get("minimum_active_stock_days") is None,
        "research_scope": cfg["report_scope"], "economic_protocol_changed_from_v11": True,
        "sources": ["https://arxiv.org/abs/1103.4934", "https://arxiv.org/abs/1411.5062", "https://arxiv.org/abs/1707.03498"],
        "literature_not_a_profit_guarantee": True})
    print(f"输出目录：{directory}", flush=True)
    inputs, stores = {}, {}
    write_json(directory / "source_reuse_audit.json", validate_balanced_sources(cfg, inputs, PROJECT_ROOT))
    print("复用已审计市场PKL；仅读取，不重建特征，不拟合旧T模型", flush=True)
    daily, intraday, calendar = load_balanced_market(cfg, inputs)
    folds = [read_balanced_fold(cfg, year, inputs) for year in cfg["test_years"]]
    tables, choices = [], {}
    for fold in folds:
        year = fold["test_year"]
        out = directory / f"year_{year}"
        out.mkdir()
        write_json(out / "split.json", fold)
        store = get_store(fold["inner_year"], cfg, inputs, stores)
        selected, prices, signals, local_cfg = prepare_account_context(
            fold, "validation", daily, intraday, calendar, cfg, out / "validation_context", inputs, store)
        if not set(selected.symbol).issubset(store.symbols):
            raise ValueError("原恢复行情没有覆盖当前固定持股，不能静默删股")
        candidate_dir = out / "validation_candidates"
        candidate_dir.mkdir()
        print(f"[{year}] 内部{fold['inner_year']}：{selected.symbol.nunique()}只底仓，{len(signals):,}当前信号，{len(rules)}候选", flush=True)
        context_args = (selected, prices, signals, local_cfg, store, str(candidate_dir))
        rows = []
        if cfg["candidate_workers"] == 1:
            initialize_worker(*context_args)
            for rule in rules:
                rows.append(evaluate_worker(rule))
        else:
            with ProcessPoolExecutor(max_workers=cfg["candidate_workers"], mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker, initargs=context_args) as executor:
                pending = {executor.submit(evaluate_worker, rule): rule["candidate_index"] for rule in rules}
                for future in as_completed(pending):
                    rows.append(future.result())
                    if len(rows) % 10 == 0 or len(rows) == len(rules):
                        print(f"[{year}] 已完成内部候选{len(rows)}/{len(rules)}", flush=True)
        table = pd.DataFrame(rows).sort_values("candidate_index").reset_index(drop=True)
        chosen, annotated = select_balanced_candidate(table)
        selected_rule = rules[int(chosen["candidate_index"])]
        choices[year] = {"rule": selected_rule, "inner_metrics": chosen}
        write_frame(annotated, out / "validation_rule_metrics.csv")
        write_json(out / "selected_rule.json", {"rule": selected_rule, "inner_metrics": chosen,
                   "frozen_before_any_outer_test": True})
        tables.append(annotated.assign(test_year=year, inner_year=fold["inner_year"]))
        print(f"[{year}] 冻结候选{selected_rule['candidate_index']}，内部净额{chosen['incremental_net_pnl']:.2f}元，成本降低率{chosen['holding_cost_reduction_rate']:.4%}", flush=True)
    validation = pd.concat(tables, ignore_index=True)
    write_frame(validation, directory / "validation_rule_metrics.csv")
    frozen = {str(year): calculate_file_sha256(directory / f"year_{year}/selected_rule.json") for year in cfg["test_years"]}
    write_json(directory / "all_rules_frozen_before_outer_test.json", {"test_years": cfg["test_years"], "selected_rule_sha256": frozen})
    targets_passed = all(choices[year]["rule"]["enabled"] and
        choices[year]["inner_metrics"]["holding_cost_reduction_rate"] >= cfg["profit_target_rate"] for year in cfg["test_years"])
    evaluate_outer = not cfg["internal_only"] and (targets_passed or cfg["report_scope"] == "smoke")
    write_json(directory / "outer_evaluation_decision.json", {"all_inner_cost_targets_met": targets_passed,
               "outer_evaluation_started": evaluate_outer, "internal_only_requested": cfg["internal_only"],
               "smoke_permits_engineering_outer_without_profit_claim": cfg["report_scope"] == "smoke"})
    results = []
    if evaluate_outer:
        for fold in folds:
            year = fold["test_year"]
            out = directory / f"year_{year}"
            if calculate_file_sha256(out / "selected_rule.json") != frozen[str(year)]:
                raise AssertionError("规则在外评前改变")
            store = get_store(year, cfg, inputs, stores)
            selected, prices, signals, local_cfg = prepare_account_context(
                fold, "test", daily, intraday, calendar, cfg, out / "test_context", inputs, store)
            rule = choices[year]["rule"]
            decisions, funnel = generate_balanced_decisions(signals, rule, local_cfg)
            write_frame(decisions, out / "t_decisions.csv")
            write_json(out / "test_signal_funnel.json", funnel)
            result = run_balanced_portfolio(selected, prices, signals, decisions, local_cfg,
                         recovery_bars=store)
            result["selected_rule"] = rule
            save_balanced_result(result, out, local_cfg)
            results.append((year, result))
        write_balanced_report(directory, cfg, results, validation)
    else:
        lines = ["# 等金额固定底仓做T：内部研究结果", "", "正式外层尚未执行；以下是用于选择规则的内部年度成绩，不能作为验证集达标证据。", "",
                 "| 留出年 | 内部年 | 规则 | 完整组 | 配对手 | T净额（元） | 底仓成本降低率 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for fold in folds:
            item = choices[fold["test_year"]]["inner_metrics"]
            lines.append(f"| {fold['test_year']} | {fold['inner_year']} | {int(item['candidate_index'])} | {int(item['t_pair_count'])} | {item['completed_t_pair_lots']:.0f} | {item['incremental_net_pnl']:.2f} | {item['holding_cost_reduction_rate']:.4%} |")
        lines += ["", "所有内部年度均达到5%成本降低率后才进入统一外评。每股名义建仓1万—2万元、目标1.5万元；底仓不按周退出，多次T仍消耗当日旧股卖出额度。", "", "当前研究及完整目标尚未达成；所有候选、负结果和拒单记录保留。"]
        (directory / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(directory / "goal_evaluation.json", {"goal_achieved": False, "formal_outer_evaluated": False,
                   "all_inner_cost_targets_met": targets_passed, "reason": "internal_research_only"})
    write_json(directory / "input_sources.json", inputs)
    for name, identity in inputs.items():
        if calculate_file_sha256(Path(name)) != identity["sha256"]:
            raise AssertionError(f"运行期间输入改变：{name}")
    sources = json.loads((directory / "source_hashes.json").read_text())
    for name, digest in sources.items():
        if calculate_file_sha256(PROJECT_ROOT / name) != digest:
            raise AssertionError(f"运行期间源码改变：{name}")
    write_json(directory / "runtime_integrity_audit.json", {"all_unchanged": True,
               "input_files_verified": len(inputs), "runtime_sources_verified": len(sources)})
    print(f"研究完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
