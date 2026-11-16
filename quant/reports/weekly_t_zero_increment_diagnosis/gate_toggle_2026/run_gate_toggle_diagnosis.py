"""固定 v8 保存预测，仅切换 enabled 的 A/B 诊断，不训练或修改原报告。

从 quant 根目录执行；读取指定正式或 smoke 报告、已有分钟/日缓存和只读 dense
恢复报价。先把 A 与源报告逐日核对通过，再执行 B；产物只写独立 output-dir。
B 是不满足原晋级条件的受控反事实，不作为合格策略或重新选参依据。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "quant_backtest").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_features import add_session_context_features, select_recoverable_signal_sides
from quant_backtest.weekly_t_quote_store import DenseRecoveryBarStore
from quant_backtest.weekly_t_short_horizon import (
    ShortHorizonTStrategy, create_short_decisions, feature_support_mask,
    planned_recovery_times, predicted_pair_edges,
)
from scripts.run_weekly_t_cross_validation import save_result, select_execution_data, write_frame


def calculate_sha256(path: Path) -> str:
    """按块计算输入文件 SHA，不把大行情数组复制进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """只写当前独立诊断目录的 JSON；不覆盖正式产物。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    """以 round_trip 保留保存数值，读取报告 CSV 的已知时间列。"""
    frame = pd.read_csv(path, float_precision="round_trip")
    for column in ("datetime", "feature_week", "signal_datetime", "trade_week", "entry_date", "exit_date",
                   "date", "planned_recovery_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def verify_runtime_sources(source: Path) -> dict:
    """要求原 20 份运行代码、原快照与当前文件逐字节相同，失败立即停止。"""
    expected = json.loads((source / "source_hashes.json").read_text())
    records = {}
    for name, digest in expected.items():
        current = calculate_sha256(PROJECT_ROOT / name)
        frozen = calculate_sha256(source / "source_snapshot" / Path(name).name)
        if current != digest or frozen != digest:
            raise AssertionError(f"源代码已改变：{name}")
        records[name] = {"sha256": digest, "current_matches": True, "snapshot_matches": True}
    if len(records) != 20:
        raise AssertionError("本诊断要求已声明的 v8 20 份运行源码")
    return records


def compare_curves(actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str]) -> dict:
    """严格核对日期及指定逐日账户量；允许 CSV/浮点复算最多 1e-8 元误差。"""
    if not actual.datetime.reset_index(drop=True).equals(expected.datetime.reset_index(drop=True)):
        raise AssertionError("A 与正式日期不一致")
    errors = {c: float(np.max(np.abs(actual[c].to_numpy() - expected[c].to_numpy()))) for c in columns}
    if any(not np.isfinite(v) or v > 1e-8 for v in errors.values()):
        raise AssertionError(f"A 未复现正式账户：{errors}")
    return {"same_dates": True, "daily_rows": len(actual), "max_absolute_errors": errors,
            "exact_all_compared_values": all(v == 0 for v in errors.values()), "passed": True}


def create_signal_audit(frame: pd.DataFrame, rule: dict, cfg: dict, support: dict) -> pd.DataFrame:
    """拆解固定已知门控，仅用于解释信号数量，不读取未来成交或恢复标签。"""
    audit = frame[["datetime", "symbol", "prediction"]].copy()
    buy, sell = predicted_pair_edges(frame, cfg)
    audit["buy_net_edge"] = buy
    audit["sell_net_edge"] = sell
    audit["support_pass"] = feature_support_mask(frame, support)
    planned = planned_recovery_times(frame.datetime, int(rule["horizon_minutes"]))
    audit["planned_recovery_time"] = planned
    audit["clock_pass"] = planned.notna() & planned.dt.strftime("%H:%M").le(cfg["latest_planned_recovery_time"])
    audit["finite_prediction_pass"] = np.isfinite(frame.prediction) & frame.prediction.gt(-1)
    audit["net_edge_pass"] = np.maximum(buy, sell).ge(float(rule["net_margin"]))
    direction = pd.Series(np.where(buy >= sell, 1, -1), index=frame.index)
    audit["current_exit_side_pass"] = select_recoverable_signal_sides(frame, direction, cfg)
    audit["a_action"] = ShortHorizonTStrategy(rule, cfg, support).generate_signals(frame).t_action
    active = {**rule, "enabled": True}
    audit["b_action"] = ShortHorizonTStrategy(active, cfg, support).generate_signals(frame).t_action
    return audit


def summarize_result(result: dict, metrics: dict, decisions: pd.DataFrame, signal_audit: pd.DataFrame,
                     arm: str) -> dict:
    """区分原预测、逐行触发、每日去重请求、实际首腿、恢复片段和净账户收益。"""
    events = result["trade_events"]
    first = events.loc[events.event_type.eq("t") & events.fragment_id.eq(0)]
    recovery = events.loc[events.event_type.eq("t") & events.fragment_id.gt(0)]
    audit = result["daily_inventory_audit"]
    action = signal_audit[f"{arm}_action"]
    fields = ("total_return", "baseline_total_return", "t_added_equity", "t_pair_count", "fragment_count",
              "pending_pairs", "inventory_mismatch_days", "recovery_failure_days", "base_plan_mismatch_count",
              "t1_violation_count", "weekly_liquidation_failure_days", "weekly_unclosed_positions",
              "baseline_weekly_unclosed_positions", "t_fee", "total_t_net_profit", "total_t_marked_pnl")
    return {"enabled": arm == "b", "prediction_rows": len(signal_audit),
            "nonzero_raw_prediction_rows": int(signal_audit.prediction.ne(0).sum()),
            "finite_prediction_rows": int(signal_audit.finite_prediction_pass.sum()),
            "raw_trigger_rows_before_daily_deduplication": int(action.ne(0).sum()),
            "decision_requests": len(decisions), "buy_first_requests": int(decisions.action.eq(1).sum()),
            "sell_first_requests": int(decisions.action.eq(-1).sum()),
            "all_actual_trade_events": len(events), "t_first_leg_events": len(first),
            "t_recovery_events": len(recovery), "t_all_trade_events": len(first) + len(recovery),
            "round_trip_rows": len(result["round_trips"]),
            "t_first_leg_unique_pairs": int(first.pair_id.nunique()),
            "inventory_audit_rows": len(audit),
            "inventory_mismatch_stock_days": int(audit.inventory_difference.ne(0).sum()),
            "recovery_failure_stock_days": int(audit.recovery_failure.sum()),
            **{key: metrics.get(key) for key in fields}}


def plot_comparison(a: pd.DataFrame, b: pd.DataFrame, initial: float, target: Path) -> None:
    """输出两臂和共同底仓的静态对照曲线，B 标记为未晋级诊断。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(a.datetime, (a.equity / initial - 1) * 100, label="A: original enabled=false")
    axes[0].plot(b.datetime, (b.equity / initial - 1) * 100, label="B: enabled=true diagnostic", linewidth=1.2)
    axes[0].plot(a.datetime, (a.baseline_equity / initial - 1) * 100, label="Identical weekly baseline", linestyle=":")
    axes[0].set_ylabel("Account return (%)")
    axes[0].legend()
    axes[1].plot(b.datetime, b.equity - a.equity, label="B minus A (CNY)")
    axes[1].axhline(0, color="grey", linewidth=.7)
    axes[1].set_ylabel("Increment (CNY)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("2026 gate-only A/B diagnostic - not an eligible strategy")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def run_diagnosis(args: argparse.Namespace) -> None:
    """先冻结协议/输入身份，复现 A，通过后只切一次开关运行 B并归档。"""
    started = time.monotonic()
    source, out, dense = args.source_report.resolve(), args.output_dir.resolve(), args.dense_cache.resolve()
    if source == out or source in out.parents:
        raise ValueError("诊断目录不得写入原正式报告内")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError("协议已存在，须另选独立目录；不覆盖既有诊断")
    runtime = verify_runtime_sources(source)
    year_dir = source / "year_2026"
    cfg = json.loads((source / "resolved_config.json").read_text())
    rule_a = json.loads((year_dir / "selected_rule.json").read_text())
    support = json.loads((year_dir / "selected_support.json").read_text())
    if rule_a["enabled"] is not False or rule_a["fold"]["test_year"] != 2026:
        raise ValueError("必须是源报告 2026 原关闭规则")
    rule_b = {**rule_a, "enabled": True}
    assert [k for k in rule_a if rule_a[k] != rule_b[k]] == ["enabled"]
    if calculate_sha256(year_dir / "t_model/model_bundle.joblib") != rule_a["model_bundle_sha256"]:
        raise AssertionError("原选定模型包SHA不匹配")
    manifest = json.loads((source / "data_manifest.json").read_text())
    cache = Path(manifest["cache_path"])
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "controlled_enabled_only_gate_diagnosis",
                "source_report": str(source), "test_year": 2026, "rule_a": rule_a, "rule_b": rule_b,
                "changed_fields": ["enabled"], "script_sha256": calculate_sha256(Path(__file__)),
                "model_retrained": False, "prediction_recomputed": False, "candidate_reselected": False,
                "protocol_frozen_before_ab_execution": True,
                "a_must_reproduce_source_before_b": True, "numeric_tolerance_rmb": 1e-8,
                "data_route": "complete same-clock 2026 cross-section session features, then original select_execution_data",
                "dense_cache": str(dense), "runtime_sources": runtime,
                "interpretation": "B bypasses original promotion gate; diagnostic only, not a newly eligible strategy; no test-based selection"}
    write_json(out / "protocol.json", protocol)
    write_json(out / "resolved_config.json", cfg)
    for arm, rule in [("a", rule_a), ("b", rule_b)]:
        (out / arm).mkdir()
        write_json(out / arm / "selected_rule.json", rule)
    local_script = out / Path(__file__).name
    if local_script.resolve() != Path(__file__).resolve():
        shutil.copyfile(__file__, local_script)
    input_paths = [source / "resolved_config.json", source / "data_manifest.json", source / "source_hashes.json",
                   year_dir / "selected_rule.json", year_dir / "selected_support.json", year_dir / "weekly_selections.csv",
                   year_dir / "t_predictions.csv", year_dir / "t_decisions.csv", year_dir / "test_equity_curve.csv",
                   year_dir / "test_trade_events.csv", year_dir / "baseline_trade_events.csv",
                   year_dir / "t_model/model_bundle.joblib", year_dir / "t_model/model_metadata.json",
                   cache / "intraday.pkl", cache / "daily.pkl", cache / "metadata.json", dense / "metadata.json"]
    inputs = {str(path): {"sha256": calculate_sha256(path), "size_bytes": path.stat().st_size,
                          "mtime_ns": path.stat().st_mtime_ns} for path in input_paths}
    write_json(out / "input_sources.json", inputs)
    print("Protocol frozen; source SHA verified; loading saved cache", flush=True)
    intraday = pd.read_pickle(cache / "intraday.pkl")
    intraday = intraday.loc[pd.to_datetime(intraday.datetime).dt.year.eq(2026)].copy()
    raw_cross_section_rows = len(intraday)
    intraday = add_session_context_features(intraday)
    daily = pd.read_pickle(cache / "daily.pkl")
    daily = daily.loc[pd.to_datetime(daily.date).dt.year.eq(2026)].copy()
    selections = read_csv(year_dir / "weekly_selections.csv")
    actual_daily, actual_intraday = select_execution_data(selections, daily, intraday)
    del daily, intraday
    gc.collect()
    saved_predictions = read_csv(year_dir / "t_predictions.csv")
    keys = ["datetime", "symbol"]
    if saved_predictions.duplicated(keys).any() or actual_intraday.duplicated(keys).any():
        raise AssertionError("重复预测或行情键")
    predicted_columns = ["prediction", "residual_prediction", "market_prediction"]
    if set(predicted_columns).intersection(actual_intraday):
        raise AssertionError("原行情不应已包含模型预测")
    joined = actual_intraday.merge(saved_predictions, on=keys, how="outer", validate="one_to_one", indicator=True, sort=False)
    if not joined._merge.eq("both").all() or len(joined) != len(saved_predictions):
        raise AssertionError("已保存预测与当前执行信号没有严格一一对应")
    frame = joined.drop(columns="_merge")
    if not frame.market_prediction.eq(0).all() or not frame.prediction.equals(frame.residual_prediction):
        raise AssertionError("保存的v8零先验预测分量不一致")
    signal_audit = create_signal_audit(frame, rule_a, cfg, support)
    write_frame(signal_audit, out / "signal_gate_audit.csv")
    decisions_a = create_short_decisions(frame, rule_a, cfg, support)
    if len(decisions_a) or len(read_csv(year_dir / "t_decisions.csv")):
        raise AssertionError("原关闭开关应该没有T决策")
    store = DenseRecoveryBarStore(dense, verify_integrity=True)
    write_json(out / "data_alignment.json", {"full_2026_current_cross_section_rows": raw_cross_section_rows,
               "selected_prediction_rows": len(frame), "one_to_one_saved_prediction_alignment": True,
               "weekly_selection_rows": len(selections), "selected_symbols": int(selections.symbol.nunique()),
               "dense_fingerprint": store.fingerprint, "dense_arrays_verified": True,
               "full_cross_section_features_before_stock_selection": True})
    print(f"A: {len(frame)} saved predictions, {len(decisions_a)} decisions", flush=True)
    result_a = run_weekly_portfolio(selections, actual_daily, actual_intraday, decisions_a, cfg, recovery_bars=store)
    metrics_a = save_result(result_a, out / "a", cfg)
    write_frame(decisions_a, out / "a/t_decisions.csv")
    source_curve = read_csv(year_dir / "test_equity_curve.csv")
    a_comparison = compare_curves(result_a["equity_curve"], source_curve,
                                 ["equity", "cash", "holdings_value", "baseline_equity", "baseline_cash"])
    write_json(out / "a_reproduction_audit.json", a_comparison)
    print(f"A reproduced: {a_comparison}", flush=True)
    # 只有A复现通过后，才运行B；不依据B结果选择参数或重复筛选其它模型。
    decisions_b = create_short_decisions(frame, rule_b, cfg, support)
    future_columns = [c for c in frame if c.startswith(("execution_", "recovery_", "target_", "label_"))]
    no_future = frame.copy()
    no_future[future_columns] = np.nan
    pd.testing.assert_frame_equal(create_short_decisions(no_future, rule_b, cfg, support), decisions_b, check_exact=True)
    del no_future
    write_json(out / "decision_causality_check.json", {"future_columns_removed": future_columns,
               "same_decisions_after_removing_future_prices_labels": True})
    print(f"B: enabled-only toggle produces {len(decisions_b)} requests", flush=True)
    result_b = run_weekly_portfolio(selections, actual_daily, actual_intraday, decisions_b, cfg, recovery_bars=store)
    metrics_b = save_result(result_b, out / "b", cfg)
    write_frame(decisions_b, out / "b/t_decisions.csv")
    baseline_comparison = compare_curves(result_b["equity_curve"], result_a["equity_curve"], ["baseline_equity", "baseline_cash"])
    summary_a = summarize_result(result_a, metrics_a, decisions_a, signal_audit, "a")
    summary_b = summarize_result(result_b, metrics_b, decisions_b, signal_audit, "b")
    sources_after = verify_runtime_sources(source)
    for path_text, before in inputs.items():
        path = Path(path_text)
        if path.stat().st_size != before["size_bytes"] or path.stat().st_mtime_ns != before["mtime_ns"]:
            raise AssertionError(f"输入在诊断中变化：{path}")
        if calculate_sha256(path) != before["sha256"]:
            raise AssertionError(f"输入SHA在诊断中变化：{path}")
    curve_a, curve_b = result_a["equity_curve"], result_b["equity_curve"]
    combined = pd.DataFrame({"datetime": curve_a.datetime, "a_equity": curve_a.equity, "b_equity": curve_b.equity,
                             "baseline_equity": curve_a.baseline_equity, "b_minus_a": curve_b.equity - curve_a.equity})
    write_frame(combined, out / "ab_equity_curve.csv")
    write_frame(pd.DataFrame([{"arm": "a", **summary_a}, {"arm": "b", **summary_b}]), out / "ab_summary.csv")
    plot_comparison(curve_a, curve_b, cfg["initial_cash"], out / "ab_return_curve.png")
    summary = {"classification": "gate_toggle_causal_diagnostic_not_eligible_strategy", "a": summary_a, "b": summary_b,
               "a_reproduces_formal": a_comparison, "baseline_unchanged_ab": baseline_comparison,
               "only_rule_change": ["enabled"], "runtime_sources_end": sources_after,
               "input_files_unchanged": True, "future_fields_do_not_change_decisions": True,
               "b_minus_a_final_equity": float(curve_b.equity.iloc[-1] - curve_a.equity.iloc[-1]),
               "elapsed_seconds": time.monotonic() - started,
               "not_promoted": True, "outer_data_used_for_diagnosis_only": True}
    write_json(out / "summary.json", summary)
    table_keys = [("保存预测行", "prediction_rows"), ("去重前非零T信号", "raw_trigger_rows_before_daily_deduplication"),
                  ("每日去重T请求", "decision_requests"), ("实际T首腿", "t_first_leg_events"),
                  ("实际T恢复腿", "t_recovery_events"), ("完整配对", "t_pair_count"), ("恢复片段", "fragment_count"),
                  ("每日T股数偏差日", "inventory_mismatch_days"), ("T恢复失败日", "recovery_failure_days"),
                  ("期末待恢复配对", "pending_pairs"), ("底仓计划偏差", "base_plan_mismatch_count"),
                  ("T+1违规", "t1_violation_count"), ("T手续费元", "t_fee"), ("实际账户净T增量元", "t_added_equity")]
    lines = ["# 2026 年 enabled 开关 A/B 诊断", "", "本次只切开关，没有改模型、名单、预测、参数、费用、股数或成交规则。",
             "B 绕过原晋级门槛，仅用于查明代码路径；不是新合格策略，也不依据本次外层结果重新选参。", "",
             f"源报告：`{source}`。协议在两臂运行前保存于 protocol.json；A 复现通过后才运行 B。", "",
             "| 指标 | A 原 enabled=false | B 仅 enabled=true |", "| --- | ---: | ---: |"]
    lines += [f"| {label} | {summary_a[key]} | {summary_b[key]} |" for label, key in table_keys]
    lines += ["", f"A 与原报告 {a_comparison['daily_rows']} 个交易日逐日相同日期，现金/权益最大误差为 "
              f"{max(a_comparison['max_absolute_errors'].values()):.12g} 元。两臂基准最大误差 "
              f"{max(baseline_comparison['max_absolute_errors'].values()):.12g} 元。",
              f"B−A 期末权益差 {summary['b_minus_a_final_equity']:.6f} 元；配对净利使用已含滑点成交价减手续费，不能再次扣滑点。", "",
              "原预测非零不等于有下单资格；先过输入支持域、计划时间、净边际、当前恢复方向，再按每股每日最早信号去重，最终还受旧股、资金、容量和限价影响。",
              f"A/B 的周退出失败日分别 {metrics_a['weekly_liquidation_failure_days']}/{metrics_b['weekly_liquidation_failure_days']}，"
              f"期末未退出底仓 {metrics_a['weekly_unclosed_positions']}/{metrics_b['weekly_unclosed_positions']}；T开关不消除底仓退出风险。", "",
              "原始预测与当前行情严格一一对应；全部同刻截面先构造会话特征。删除未来成交/恢复/标签列不改变B请求。",
              "20份源码和所有记录输入在结束时再次SHA核对未变。原正式文件没有被覆盖；全部原始账本、信号审计和曲线在本目录。", "",
              "2026只有截至4月30日的77日正式数据（smoke则只涵盖其原范围）；这是已接触外层上的故障诊断，不是独立未来业绩。", "",
              "文件：protocol.json 为事前协议，input_sources.json 为输入SHA，a/ 与 b/ 保存同结构成交/配对/库存/受阻/净值CSV；",
              "signal_gate_audit.csv 为逐信号门控，a_reproduction_audit.json 为原结果复现，summary.json 为完整汇总，ab_return_curve.png 为对照图。"]
    (out / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {str(p.relative_to(out)): calculate_sha256(p) for p in out.rglob("*")
               if p.is_file() and p.suffix not in (".pyc", ".log") and p.name != "output_sha256.json"}
    write_json(out / "output_sha256.json", outputs)
    print(json.dumps({"completed": True, "a": summary_a, "b": summary_b,
                      "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run_diagnosis(parser.parse_args())
