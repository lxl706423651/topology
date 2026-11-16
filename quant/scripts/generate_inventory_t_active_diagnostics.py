"""为已完成年度回测生成实际做 T 诊断：读取冻结输入，输出独立诊断目录和跨年汇总。

在 quant 仓库执行 --run-dir REPORT [--year YEAR]。仅按预先冻结协议中的内部验证
净增量选一条启用的做 T 规则，放宽每日库存恢复筛选；不训练模型、不改变正式报告
或原始数据、不按外层收益调参。已完成的同输入诊断重用，未完成目录拒绝覆盖。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.inventory_t import run_inventory_t
from scripts.run_inventory_t_cross_validation import (
    MinuteBarStore, audit_result, create_t_rules, plot_return_curves, save_fold_report, write_frame,
)


CLASSIFICATION = "active_diagnostic_not_strict_inventory_compliant"
RESULT_FILES = {
    "equity_curve": "test_equity_curve.csv", "trade_events": "test_trade_events.csv",
    "round_trips": "test_round_trips.csv", "open_positions": "test_open_positions.csv",
    "daily_inventory_audit": "daily_inventory_audit.csv", "order_attempts": "blocked_orders.csv",
}
REQUIRED_MAIN_FILES = (
    "final_report.md", "selected_rule.json", "test_metrics.csv", "validation_rule_metrics.csv",
    "base_entry_schedule.csv", "test_equity_curve.csv",
)


def read_json(path: Path) -> dict:
    """读取 path 的 UTF-8 JSON 配置或冻结元数据，不修改源文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def calculate_file_hash(path: Path) -> str:
    """计算复现所需源文件的 SHA-256；仅记录指纹，不据测试内容选择参数。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_active_rule(validation: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    """只按内部验证 incremental_net_pnl 降序、rule_name 升序选启用的候选。

    不筛恢复失败或实际触发次数，确保严格执行运行前冻结的诊断协议；没有实际
    触发的启用规则也保留，不用外层结果换成其他规则。
    """
    rules = {rule["name"]: rule for rule in create_t_rules(cfg) if rule.get("enabled", True)}
    required = {"rule_name", "incremental_net_pnl", "pair_count", "inventory_mismatch_days", "recovery_failure_days"}
    if not required.issubset(validation):
        raise ValueError(f"内部验证结果缺列: {required.difference(validation.columns)}")
    candidates = validation.loc[validation.rule_name.isin(rules)].copy()
    if candidates.empty or candidates.rule_name.duplicated().any():
        raise ValueError("内部启用规则结果为空或重复")
    if not np.isfinite(candidates.incremental_net_pnl).all():
        raise ValueError("内部候选净增量必须全部为有限数")
    selected = candidates.sort_values(["incremental_net_pnl", "rule_name"], ascending=[False, True], kind="stable").iloc[0]
    return rules[selected.rule_name], selected.to_dict()


def load_diagnostic_result(directory: Path) -> dict:
    """只读已完成诊断的净值、明细和指标，供重复执行及跨年汇总使用。"""
    result = {key: pd.read_csv(directory / filename) for key, filename in RESULT_FILES.items()}
    result["equity_curve"]["datetime"] = pd.to_datetime(result["equity_curve"].datetime)
    result["metrics"] = pd.read_csv(directory / "test_metrics.csv").iloc[0].to_dict()
    return result


def generate_year_diagnostic(run_dir: Path, year: int, cfg: dict, protocol: dict,
                             store: MinuteBarStore) -> dict:
    """为已完成 year 冻结内部规则并回测同底仓；输出只写 active_t_diagnostic 子目录。"""
    main_dir = run_dir / f"year_{year}"
    missing = [name for name in REQUIRED_MAIN_FILES if not (main_dir / name).is_file()]
    if missing:
        raise ValueError(f"{year} 正式年度尚未完成，缺少 {missing}")
    selection = read_json(main_dir / "selected_rule.json")
    validation = pd.read_csv(main_dir / "validation_rule_metrics.csv")
    rule, training = select_active_rule(validation, cfg)
    inputs = [run_dir / "resolved_config.json", run_dir / "active_diagnostic_protocol.json",
              main_dir / "selected_rule.json", main_dir / "validation_rule_metrics.csv",
              main_dir / "base_entry_schedule.csv", main_dir / "test_equity_curve.csv"]
    fingerprints = {str(path.relative_to(run_dir)): calculate_file_hash(path) for path in inputs}
    code_files = [Path(__file__), PROJECT_ROOT / "quant_backtest/inventory_t.py",
                  PROJECT_ROOT / "scripts/run_inventory_t_cross_validation.py"]
    code_hashes = {str(path.relative_to(PROJECT_ROOT)): calculate_file_hash(path) for path in code_files}
    directory = main_dir / "active_t_diagnostic"
    metadata_file = directory / "diagnostic_metadata.json"
    if directory.exists():
        if not metadata_file.is_file() or not (directory / "final_report.md").is_file():
            raise ValueError(f"诊断目录已存在但尚未完成，拒绝覆盖: {directory}")
        previous = read_json(metadata_file)
        if previous["input_sha256"] != fingerprints or previous["selected_rule"] != rule:
            raise ValueError(f"诊断输入已变化，拒绝覆盖旧结果: {directory}")
        if previous["code_sha256"] != code_hashes:
            raise ValueError(f"诊断代码已变化，拒绝把旧结果混作同版本输出: {directory}")
        print(f"[{year}] 读取已完成诊断，不重跑或覆盖: {directory}", flush=True)
        return load_diagnostic_result(directory)

    # 在读取外层净值和执行实际做 T 之前落盘选择；训练失败信息完整保留。
    directory.mkdir(exist_ok=False)
    frozen = {"created_at": datetime.now().isoformat(), "classification": CLASSIFICATION,
              "selected_rule": rule, "training_candidate": training,
              "selected_from_internal_validation_only": True,
              "outer_returns_used_for_selection": False, "inventory_failure_filter_applied": False,
              "same_base_entry_schedule": True, "input_sha256": fingerprints,
              "code_sha256": code_hashes, "protocol_snapshot": protocol}
    write_json(directory / "frozen_diagnostic_rule_before_test.json", frozen)
    entries = pd.read_csv(main_dir / "base_entry_schedule.csv", dtype={"symbol": str})
    entries["datetime"] = pd.to_datetime(entries.datetime)
    if "signal_datetime" in entries:
        entries["signal_datetime"] = pd.to_datetime(entries.signal_datetime)
    date_frame = pd.read_csv(main_dir / "test_equity_curve.csv", usecols=["datetime"])
    dates = pd.DatetimeIndex(pd.to_datetime(date_frame.datetime)).normalize().unique().sort_values()
    bars = store.load_slice(sorted(entries.symbol.unique()), dates)
    print(f"[{year}] 仅内部选择 {rule['name']}，内部净增量 {training['incremental_net_pnl']:,.2f} 元，"
          f"内部股数偏差 {training['inventory_mismatch_days']} 日；开始独立诊断", flush=True)
    result = run_inventory_t(bars, entries, dates, cfg, rule)
    audit_result(result, entries, cfg)
    # 参数已经冻结。只在执行后比较同一底仓基准，确保没有额外资金或不同初始买单。
    expected_baseline = pd.read_csv(main_dir / "test_equity_curve.csv", usecols=["baseline_equity"])
    if not np.allclose(result["equity_curve"].baseline_equity, expected_baseline.baseline_equity, rtol=1e-10, atol=1e-5):
        raise AssertionError("实际做 T 诊断与正式回测的同底仓基准不一致")
    base_events = result["trade_events"].loc[result["trade_events"].event_type == "baseline"]
    base_notional = float(base_events.notional.sum())
    base_cost = base_notional + float(base_events.fee.sum())
    result["metrics"].update({"diagnostic_classification": CLASSIFICATION,
                              "train_inner_pnl": float(training["incremental_net_pnl"]),
                              "train_inner_inventory_mismatch_days": int(training["inventory_mismatch_days"]),
                              "train_inner_recovery_failure_days": int(training["recovery_failure_days"]),
                              "selected_from_internal_validation_only": True,
                              "base_entry_notional": base_notional, "base_entry_cost": base_cost,
                              "initial_base_exposure": base_cost / cfg["initial_cash"],
                              "incremental_return_on_base_cost": result["metrics"]["total_t_marked_pnl"] / base_cost if base_cost else np.nan})
    diagnostic_validation = validation.copy()
    diagnostic_validation["selected"] = diagnostic_validation.rule_name == rule["name"]
    save_fold_report(directory, year, result, entries, diagnostic_validation,
                     selection["fold"], rule, selection["gate"], cfg)
    original = (directory / "final_report.md").read_text(encoding="utf-8")
    body = original.split("\n", 1)[1] if "\n" in original else original
    header = (f"# {year} 年实际做 T 诊断：放宽每日库存恢复约束\n\n"
              "**此结果属于允许恢复失败的研究诊断。出现开收盘股数偏差即不满足每日库存目标；"
              "不能据此替代严格策略或声称每天都能恢复。**\n\n"
              f"按事先冻结协议，仅用内部验证净增量选出 `{rule['name']}`，"
              f"内部净增量 {training['incremental_net_pnl']:,.2f} 元，"
              f"内部股数偏差 {int(training['inventory_mismatch_days'])} 日、"
              f"恢复失败 {int(training['recovery_failure_days'])} 日。没有用外层收益更换规则。\n\n"
              "与正式策略共用完全相同的底仓买单、现金、费用和行情；T+1预算、整手和无杠杆检查仍然执行。"
              "主策略的选参约束及根报告保持原样。\n\n"
              f"实际底仓买入成交额 {base_notional:,.2f} 元，含费成本 {base_cost:,.2f} 元，"
              f"占初始现金 {base_cost / cfg['initial_cash']:.2%}。容量限制可能让实际仓位远低于目标仓位，"
              "不得将初始现金金额误当全部投入底仓。CSV同时保存做T净增量/实际底仓成本，"
              "该比率是投入成本分母的诊断收益，不是整个账户的收益率。\n")
    (directory / "final_report.md").write_text(header + body, encoding="utf-8")
    description = directory / "输出文件说明.md"
    with description.open("a", encoding="utf-8") as stream:
        stream.write("\n本目录复用上级年度的开仓模型，没有重新训练模型。"
                     "`frozen_diagnostic_rule_before_test.json` 保存执行前选择、协议和输入指纹；"
                     "`diagnostic_metadata.json` 表示诊断已完整结束，附带同底仓核查及结果分类。\n")
    metadata = {**frozen, "completed_at": datetime.now().isoformat(), "same_baseline_verified": True,
                "test_year": year, "protocol": selection["fold"]["protocol"],
                "main_model_reference": "../model/model_metadata.json",
                "main_inner_model_reference": "../inner_model/model_metadata.json",
                "outer_inventory_mismatch_days": result["metrics"]["inventory_mismatch_days"],
                "outer_recovery_failure_days": result["metrics"]["recovery_failure_days"]}
    write_json(metadata_file, metadata)
    print(f"[{year}] 实际 T 净增量 {result['metrics']['total_t_marked_pnl']:,.2f} 元，"
          f"股数偏差 {result['metrics']['inventory_mismatch_days']} 日，"
          f"未恢复 {result['metrics']['recovery_failure_days']} 日", flush=True)
    return result


def save_diagnostic_summary(run_dir: Path, cfg: dict) -> None:
    """汇总所有已完成诊断并刷新独立 CSV/Markdown/图，不改正式 final_report.md。"""
    rows, results = [], []
    for directory in sorted(run_dir.glob("year_[0-9][0-9][0-9][0-9]/active_t_diagnostic")):
        if not (directory / "diagnostic_metadata.json").is_file():
            continue
        metadata = read_json(directory / "diagnostic_metadata.json")
        result = load_diagnostic_result(directory)
        metrics, training = result["metrics"], metadata["training_candidate"]
        year = int(metadata["test_year"])
        rows.append({"test_year": year, "selected_rule": metadata["selected_rule"]["name"],
                     "protocol": metadata["protocol"], "train_inner_pnl": training["incremental_net_pnl"],
                     "train_inner_inventory_mismatch_days": training["inventory_mismatch_days"],
                     "train_inner_recovery_failure_days": training["recovery_failure_days"],
                     "baseline_total_return": metrics["baseline_total_return"], "active_total_return": metrics["total_return"],
                     "incremental_return_points": metrics["total_t_marked_pnl"] / cfg["initial_cash"] * 100,
                     "total_t_marked_pnl": metrics["total_t_marked_pnl"], "total_t_net_profit": metrics["total_t_net_profit"],
                     "base_entry_notional": metrics["base_entry_notional"], "base_entry_cost": metrics["base_entry_cost"],
                     "initial_base_exposure": metrics["initial_base_exposure"],
                     "incremental_return_on_base_cost": metrics["incremental_return_on_base_cost"],
                     "inventory_mismatch_days": metrics["inventory_mismatch_days"],
                     "recovery_failure_days": metrics["recovery_failure_days"], "t_pair_count": metrics["t_pair_count"],
                     "pending_pairs": metrics["pending_pairs"], "t1_violation_count": metrics["t1_violation_count"],
                     "diagnostic_classification": CLASSIFICATION, "start": metrics["start"], "end": metrics["end"],
                     "trading_days": metrics["trading_days"]})
        results.append((year, result))
    if not rows:
        raise ValueError("尚无已完成的实际做 T 诊断")
    summary = pd.DataFrame(rows).sort_values("test_year")
    write_frame(summary, run_dir / "active_t_diagnostic_summary.csv")
    lines = ["# 实际做 T 诊断：允许库存恢复失败\n",
             "**这组诊断放宽了每日股数恢复筛选，只用于观察实际做 T 的收益和失败次数。"
             "存在股数偏差时不满足每日库存目标；不能把此诊断提升为严格策略。**\n",
             "各年规则均按已冻结协议，从启用的候选中选择内部验证净增量最高者，同分按规则名排序。"
             "不按外层收益筛选，没有更改正式策略；所有结果使用相同底仓买单。\n",
             "| 留出年 | 内部选择规则 | 内部T净增量(元) | 持有收益 | 实际T后收益 | T净增量(元) | 增量(百分点) | T组数 | 股数偏差日 | 恢复失败日 | 期末未配对 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.selected_rule} | {row.train_inner_pnl:,.2f} | "
                     f"{row.baseline_total_return:.2%} | {row.active_total_return:.2%} | {row.total_t_marked_pnl:,.2f} | "
                     f"{row.incremental_return_points:+.2f} | {row.t_pair_count} | {row.inventory_mismatch_days} | "
                     f"{row.recovery_failure_days} | {row.pending_pairs} |")
    lines += ["\n实际底仓投入受首笔分钟容量限制，最多买入指定数量的不同股票后不再自动补足目标仓位。"
              "以下成本含底仓开仓及人工加仓费用；比例分母为初始现金，不代表全年度平均市值仓位。\n",
              "| 留出年 | 实际区间 | 交易日 | 底仓成交额(元) | 底仓含费成本(元) | 成本/初始现金 | T增量/底仓成本 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        ratio = f"{row.incremental_return_on_base_cost:+.2%}" if np.isfinite(row.incremental_return_on_base_cost) else "不适用"
        lines.append(f"| {row.test_year} | {row.start} 至 {row.end} | {row.trading_days} | "
                     f"{row.base_entry_notional:,.2f} | {row.base_entry_cost:,.2f} | {row.initial_base_exposure:.2%} | {ratio} |")
    lines += ["\n每年独立重置现金和底仓，各年结果不拼成连续不清仓净值。"
              "净增量包括未配对腿按真实库存估值的影响，已实现配对利润另存CSV。"
              "恢复失败日是日终仍有未配对腿；跨日修复可能让股数偏差日多于恢复失败日。\n",
              "逐年完整文件位于 `year_YYYY/active_t_diagnostic/`。"
              "交易成本、原始数据限制、回顾性留一年及2026部分年度口径继承正式年度报告。\n"]
    (run_dir / "active_t_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    plot_return_curves(results, run_dir / "active_t_diagnostic_return_curve.png", cfg["initial_cash"])
    print(f"已刷新 {len(rows)} 个完成年度的独立诊断汇总: {run_dir / 'active_t_diagnostic_report.md'}", flush=True)


def main() -> None:
    """检查预先冻结协议，生成指定年度或全部已完成年度，再刷新独立汇总。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, help="仅生成此已完成留出年；省略时处理全部已完成年")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    protocol_path = run_dir / "active_diagnostic_protocol.json"
    if not protocol_path.is_file():
        raise ValueError("缺少事先冻结的 active_diagnostic_protocol.json，不能事后按测试结果创建诊断协议")
    protocol = read_json(protocol_path)
    if "active_rule_selection" not in protocol or "outer_diagnostics" not in protocol:
        raise ValueError("冻结诊断协议缺少选择规则或外层诊断说明")
    cfg = read_json(run_dir / "resolved_config.json")
    if args.year is not None:
        years = [args.year]
    else:
        years = sorted(int(directory.name.removeprefix("year_")) for directory in run_dir.glob("year_[0-9][0-9][0-9][0-9]")
                       if all((directory / name).is_file() for name in REQUIRED_MAIN_FILES))
    if not years:
        raise ValueError("尚无已完成的正式年度回测")
    store = MinuteBarStore(cfg)
    for year in years:
        generate_year_diagnostic(run_dir, year, cfg, protocol, store)
    save_diagnostic_summary(run_dir, cfg)


if __name__ == "__main__":
    main()
