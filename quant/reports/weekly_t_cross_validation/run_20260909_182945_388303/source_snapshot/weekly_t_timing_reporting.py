"""从 v11 已保存年度账本生成择时报告；不训练、选参或重新运行账户。

入口 write_timing_report 读取 cross_validation_results.csv 和 year_Y 标准文件，
只写根目录派生摘要、图表和来源指纹。关闭 T、成交受阻和未恢复库存均保留。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


VERSION = "weekly_t_v11_cost_aware_timing"
FAILURE_KEYS = ("inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
                "base_plan_mismatch_count", "t1_violation_count")
RESULT_FILES = {
    "equity_curve": "test_equity_curve.csv", "trade_events": "test_trade_events.csv",
    "round_trips": "test_round_trips.csv", "open_positions": "test_open_positions.csv",
    "inventory": "daily_inventory_audit.csv", "attempts": "blocked_orders.csv",
    "baseline_events": "baseline_trade_events.csv", "plans": "weekly_entry_plans.csv",
    "base_mismatches": "base_plan_mismatches.csv",
}
ALIASES = {"round_trips": "test_t_round_trips.csv", "inventory": "test_inventory_audit.csv"}


def _sha(path: Path) -> str:
    """流式计算文件身份，避免把大账本再次整体载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """保存严格 JSON；缺失参考字段为 null，不输出非标准 NaN。"""
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (Path, pd.Timestamp)):
            return str(item)
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert,
                               allow_nan=False) + "\n", encoding="utf-8")


def _read_csv(path: Path, inputs: dict) -> pd.DataFrame:
    """以 round_trip 精度读取来源；真正空表允许为空，缺文件不冒充空表。"""
    inputs[str(path.resolve())] = _sha(path)
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path, inputs: dict) -> dict:
    """读取并登记 JSON 来源字节身份。"""
    inputs[str(path.resolve())] = _sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _boolean(value, name: str) -> bool:
    """拒绝将字符串 False 或缺失值错误解释为开启。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{name} 必须明确为布尔值")


def _sum(frame: pd.DataFrame, column: str) -> float:
    """空表返回零；非空记录必须有有限数值，不能忽略缺失费用。"""
    if frame.empty:
        return 0.
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} 包含非有限账目")
    return float(values.sum())


def _read_year(directory: Path, year: int, inputs: dict) -> tuple[dict, dict]:
    """读取旧 save_result 名称，兼容显式别名并拒绝两份同义文件冲突。"""
    folder = directory / f"year_{year}"
    result = {}
    for key, name in RESULT_FILES.items():
        paths = [folder / name]
        if key in ALIASES:
            paths.append(folder / ALIASES[key])
        present = [path for path in paths if path.exists()]
        if not present:
            raise FileNotFoundError(paths[0])
        result[key] = _read_csv(present[0], inputs)
        for duplicate in present[1:]:
            pd.testing.assert_frame_equal(result[key], _read_csv(duplicate, inputs), check_exact=True)
    metrics_csv, metrics_json = folder / "test_metrics.csv", folder / "test_metrics.json"
    if metrics_csv.exists():
        table = _read_csv(metrics_csv, inputs)
        if len(table) != 1:
            raise ValueError(f"{year} test_metrics.csv 必须恰有一行")
        result["metrics"] = table.iloc[0].to_dict()
    elif metrics_json.exists():
        result["metrics"] = _read_json(metrics_json, inputs)
    else:
        raise FileNotFoundError(metrics_csv)
    if metrics_csv.exists() and metrics_json.exists():
        alternative = _read_json(metrics_json, inputs)
        for key in set(alternative) & set(result["metrics"]):
            a, b = alternative[key], result["metrics"][key]
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                if not np.isclose(a, b, rtol=1e-12, atol=1e-9):
                    raise ValueError(f"{year} 两份指标冲突：{key}")
            elif a != b:
                raise ValueError(f"{year} 两份指标冲突：{key}")
    return result, _read_json(folder / "selected_rule.json", inputs)


def _summarize_year(year: int, result: dict, rule: dict, original: dict,
                    directory: Path, cfg: dict, inputs: dict) -> tuple[dict, pd.DataFrame]:
    """重算启用状态、独立账户差、片段费用与执行失败；不以失败修改原交易。"""
    states = [_boolean(rule[key], key) for key in ("enabled", "active") if key in rule]
    if "hold" in rule:
        states.append(not _boolean(rule["hold"], "hold"))
    if not states or len(set(states)) != 1:
        raise ValueError(f"{year} selected_rule 缺失或冲突的 enabled/active/hold")
    enabled = states[0]
    for key in ("t_enabled", "enabled", "active"):
        if key in original and _boolean(original[key], key) != enabled:
            raise ValueError(f"{year} 年度摘要与冻结规则开关不符")
    metrics = result["metrics"]
    required = set(FAILURE_KEYS) | {"t_pair_count", "weekly_liquidation_failure_days",
                                  "weekly_unclosed_positions", "baseline_weekly_unclosed_positions"}
    if required.difference(metrics):
        raise ValueError(f"{year} 必须披露的执行指标缺失：{sorted(required.difference(metrics))}")
    for key in required:
        value = float(metrics[key])
        if not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError(f"{year} 非法计数 {key}")
    curve = result["equity_curve"]
    curve["datetime"] = pd.to_datetime(curve.datetime, format="mixed", errors="raise")
    if (curve.empty or curve.datetime.isna().any() or curve.datetime.duplicated().any()
            or not curve.datetime.is_monotonic_increasing or not curve.datetime.dt.year.eq(year).all()
            or not np.isfinite(curve[["equity", "baseline_equity"]].to_numpy(dtype=float)).all()):
        raise ValueError(f"{year} 净值必须为本年有限、升序、唯一的真实估值")
    events, trips, audit = result["trade_events"], result["round_trips"], result["inventory"]
    t_events = events.loc[events.event_type.eq("t")] if len(events) else events
    first = t_events.loc[t_events.fragment_id.eq(0)] if len(t_events) else t_events
    recovery = t_events.loc[t_events.fragment_id.gt(0)] if len(t_events) else t_events
    complete = trips.loc[trips.pair_fully_recovered.map(lambda v: _boolean(v, "pair_fully_recovered"))] if len(trips) else trips
    pair_count = int(complete.pair_id.nunique()) if len(complete) else 0
    if (len(first) + len(recovery) != len(t_events) or len(recovery) != len(trips)
            or pair_count != int(metrics["t_pair_count"])
            or len(first) != pair_count + int(metrics["pending_pairs"])
            or (len(first) and len(first) != first.pair_id.nunique())):
        raise ValueError(f"{year} 首腿、完整组、恢复片段或待恢复计数不符")
    added = float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1])
    if not enabled and (len(t_events) or len(trips) or not np.array_equal(curve.equity, curve.baseline_equity)):
        raise ValueError(f"{year} hold 必须是真实零 T 且逐日账户与基准相同")
    quantities = {}
    for column, key in (("inventory_difference", "inventory_mismatch_days"),
                        ("recovery_failure", "recovery_failure_days"),
                        ("weekly_liquidation_failure", "weekly_liquidation_failure_days")):
        if audit.empty:
            rows = audit
        else:
            mask = audit[column].ne(0) if column == "inventory_difference" else audit[column].map(lambda v: _boolean(v, column))
            rows = audit.loc[mask]
        day_count = int(pd.to_datetime(rows.date, format="mixed").dt.normalize().nunique()) if len(rows) else 0
        if day_count != int(metrics[key]):
            raise ValueError(f"{year} {key} 与逐日库存审计不符")
        quantities[key.replace("_days", "_stock_days")] = len(rows)
    cross_day = 0
    if len(trips):
        cross = (pd.to_datetime(trips.first_datetime, format="mixed").dt.normalize()
                 != pd.to_datetime(trips.second_datetime, format="mixed").dt.normalize())
        cross_day = int(trips.loc[cross, "pair_id"].nunique())
    positions = result["open_positions"]
    opened = positions.loc[positions.shares.gt(0)] if len(positions) else positions
    stale = opened.loc[pd.to_datetime(opened.mark_datetime, format="mixed").ne(curve.datetime.iloc[-1])] if len(opened) else opened
    pair_gross, pair_slip = _sum(trips, "gross_pnl"), _sum(trips, "slippage_cost")
    pair_fees = _sum(trips, "buy_fee") + _sum(trips, "sell_fee")
    pair_net = _sum(trips, "net_pnl")
    if not np.isclose(pair_gross - pair_fees, pair_net, rtol=1e-10, atol=1e-7):
        raise ValueError(f"{year} 配对净收益不符合成交价毛差减费用恒等式")
    row = {**original, **metrics, **quantities, "test_year": year, "t_enabled": enabled,
           "hold": not enabled, "t_activity_status": "active_traded" if len(first) else "active_no_fills" if enabled else "hold",
           "selected_candidate_index": rule.get("candidate_index"), "selected_profile": rule.get("profile"),
           "selected_regime": rule.get("regime"), "selection_reason": rule.get("selection_reason", "unspecified"),
           "all_actual_trade_events": len(events), "all_baseline_trade_events": len(result["baseline_events"]),
           "t_first_leg_count": len(first), "t_recovery_leg_count": len(recovery), "t_pair_count": pair_count,
           "t_fragment_count": len(trips), "t_cross_day_recovery_pair_count": cross_day,
           "t_fee": _sum(t_events, "fee"), "t_slippage_cost": _sum(t_events, "slippage_cost"),
           "realized_pair_reference_gross_pnl": pair_gross + pair_slip,
           "realized_pair_slippage_cost": pair_slip, "realized_pair_fees": pair_fees,
           "realized_pair_net_pnl": pair_net, "t_added_equity": added,
           "unpaired_or_marked_increment": added - pair_net,
           "total_return": float(curve.equity.iloc[-1] / cfg["initial_cash"] - 1),
           "baseline_total_return": float(curve.baseline_equity.iloc[-1] / cfg["initial_cash"] - 1),
           "increment_return": added / cfg["initial_cash"], "evaluated_trading_days": len(curve),
           "evaluation_start": str(curve.datetime.iloc[0]), "evaluation_end": str(curve.datetime.iloc[-1]),
           "active_days": int(pd.to_datetime(first.datetime, format="mixed").dt.normalize().nunique()) if len(first) else 0,
           "final_open_position_rows": len(opened), "final_position_market_value": _sum(opened, "market_value"),
           "final_stale_position_count": len(stale), "final_stale_market_value": _sum(stale, "market_value"),
           "t_execution_contract_met": all(int(metrics[key]) == 0 for key in FAILURE_KEYS)}
    for key in ("t_added_equity", "total_return", "baseline_total_return", "increment_return"):
        for source in (original, metrics):
            if key in source and not np.isclose(float(source[key]), row[key], rtol=1e-10, atol=1e-7):
                raise ValueError(f"{year} {key} 与保存净值不符")
    for key, name, aliases in (("prediction_rows", "t_predictions.csv", ("prediction_rows", "prediction_count", "prediction_row_count")),
                               ("submitted_t_requests", "t_decisions.csv", ("submitted_t_requests", "request_count", "t_request_count"))):
        path = directory / f"year_{year}" / name
        row[key] = len(_read_csv(path, inputs)) if path.exists() else next((original[k] for k in aliases if k in original), None)
    weekly = curve[["datetime", "equity", "baseline_equity"]].copy()
    weekly["trade_week"] = weekly.datetime.dt.to_period("W-SUN").dt.start_time
    weekly = weekly.groupby("trade_week", sort=True).last().reset_index()
    weekly["t_added_equity"] = weekly.equity - weekly.baseline_equity
    weekly["t_increment"] = weekly.t_added_equity.diff().fillna(weekly.t_added_equity)
    weekly["test_year"] = year
    return row, weekly


def _plot(directory: Path, summary: pd.DataFrame, results: list, initial: float) -> None:
    """每个年度独立绘图，明确开启/持有状态，不拼接跨年本金。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    fonts = sorted(Path("/usr/share/fonts/opentype/noto").glob("*CJK*"))
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 3.3 * len(results)), squeeze=False)
    for index, (year, result) in enumerate(results):
        row = summary.loc[summary.test_year.eq(year)].iloc[0]
        curve = result["equity_curve"]
        left, right = axes[index]
        left.plot(curve.datetime, (curve.baseline_equity / initial - 1) * 100, label="相同周度底仓")
        left.plot(curve.datetime, (curve.equity / initial - 1) * 100, "--", label="底仓＋择时 T" if row.t_enabled else "内部选择持有")
        left.set_title(f"{year} | {'启用' if row.t_enabled else '持有'} | T {row.t_pair_count} 组 | 收益 {row.total_return:.2%}")
        left.set_ylabel("独立账户收益（%）")
        left.legend()
        right.plot(curve.datetime, curve.equity - curve.baseline_equity, color="#1c7a70")
        right.axhline(0, color="grey", lw=.7)
        right.set_title(f"净增量 {row.t_added_equity:,.2f} 元 | 股数偏差 {row.inventory_mismatch_days} 日 / 恢复失败 {row.recovery_failure_days} 日")
        right.set_ylabel("净增量（元）")
        for axis in (left, right):
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(directory / "test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory / "test_return_curve.png", directory / "测试集收益曲线.png")


def write_timing_report(output_dir: Path, cfg: dict) -> dict:
    """只读年度原文件，生成 v11 标准汇总、净增量图和可追溯报告状态。

必须在全部目标年度账户保存后调用；本函数不重新选择候选，亦不将统计结论反馈
给 enabled。配置只消费版本、费用、初始资金及 bootstrap 参数，其他字段按原样保留。
"""
    directory = Path(output_dir)
    if cfg.get("research_version") != VERSION or cfg.get("candidate_activation_policy") != "positive_net_execution_valid":
        raise ValueError("报告配置必须为 v11 positive_net_execution_valid")
    initial = float(cfg["initial_cash"])
    samples = cfg.get("bootstrap_samples", 2000)
    if not np.isfinite(initial) or initial <= 0 or isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("初始资金和 bootstrap_samples 必须为正")
    inputs = {}
    original = _read_csv(directory / "cross_validation_results.csv", inputs)
    if original.empty or "test_year" not in original or original.test_year.duplicated().any():
        raise ValueError("cross_validation_results.csv 必须非空且 test_year 唯一")
    if not original.test_year.isin([2023, 2024, 2025, 2026]).all():
        raise ValueError("本研究年度必须属于2023至2026")
    if "test_years" in cfg and set(original.test_year) != set(cfg["test_years"]):
        raise ValueError("结果年度必须覆盖配置全部目标年度")
    results, records, weeks, rules = [], [], [], {}
    for item in original.sort_values("test_year").to_dict("records"):
        year = int(item["test_year"])
        result, rule = _read_year(directory, year, inputs)
        row, weekly = _summarize_year(year, result, rule, item, directory, cfg, inputs)
        results.append((year, result)); records.append(row); weeks.append(weekly); rules[str(year)] = rule
    summary, weekly = pd.DataFrame(records), pd.concat(weeks, ignore_index=True)
    increments = weekly.t_increment.to_numpy(dtype=float)
    rng = np.random.default_rng(int(cfg.get("random_state", 42)))
    bootstrap = rng.choice(increments, size=(samples, len(increments)), replace=True).sum(axis=1)
    low, high = map(float, np.quantile(bootstrap, [.025, .975]))
    failures = {key: int(summary[key].sum()) for key in FAILURE_KEYS}
    total = float(summary.t_added_equity.sum())
    positive = int(summary.t_added_equity.gt(0).sum())
    reference = bool(total > 0 and positive > len(summary) / 2 and low > 0)
    weekly_complete = bool(summary.weekly_liquidation_failure_days.sum() == 0
                           and summary.weekly_unclosed_positions.sum() == 0
                           and summary.baseline_weekly_unclosed_positions.sum() == 0)
    status = {"research_version": VERSION, "report_only": True, "affects_activation": False,
              "candidate_activation_policy": cfg["candidate_activation_policy"],
              "evaluated_years": len(summary), "active_years": int(summary.t_enabled.sum()),
              "hold_years": int(summary.hold.sum()), "years_with_actual_t_trades": int(summary.t_first_leg_count.gt(0).sum()),
              "aggregate_increment_rmb": total, "aggregate_increment_ci_low_rmb": low,
              "aggregate_increment_ci_high_rmb": high, "positive_years": positive,
              "complete_t_pairs": int(summary.t_pair_count.sum()), "t_recovery_fragments": int(summary.t_fragment_count.sum()),
              "realized_pair_reference_gross_pnl": float(summary.realized_pair_reference_gross_pnl.sum()),
              "realized_pair_slippage_cost": float(summary.realized_pair_slippage_cost.sum()),
              "realized_pair_fees": float(summary.realized_pair_fees.sum()),
              "realized_pair_net_pnl": float(summary.realized_pair_net_pnl.sum()),
              "total_t_fees": float(summary.t_fee.sum()), "total_t_slippage_cost": float(summary.t_slippage_cost.sum()),
              "execution_failure_totals": failures, "t_execution_contract_met": not any(failures.values()),
              "profitability_reference_met": reference, "weekly_execution_complete": weekly_complete,
              "gain_criterion_met": reference and not any(failures.values()) and weekly_complete,
              "weekly_exit_failure_days": int(summary.weekly_liquidation_failure_days.sum()),
              "weekly_exit_failure_stock_days": int(summary.weekly_liquidation_failure_stock_days.sum()),
              "final_open_position_rows": int(summary.final_open_position_rows.sum()),
              "final_stale_market_value": float(summary.final_stale_market_value.sum()),
              "historical_re_evaluation": True, "independent_execution_audit_performed_by_reporter": False}
    for filename in ("annual_summary.csv", "年度.csv", "test_metrics.csv"):
        summary.to_csv(directory / filename, index=False, encoding="utf-8-sig")
    weekly.to_csv(directory / "weekly_increment.csv", index=False, encoding="utf-8-sig")
    for key, filename in RESULT_FILES.items():
        pd.concat([result[key].assign(test_year=year) for year, result in results], ignore_index=True).to_csv(
            directory / filename, index=False, encoding="utf-8-sig")
    validation = directory / "validation_rule_metrics.csv"
    validation_summary = {"report_only": True, "selection_scope": "inner_only", "selected_rules": rules,
                          "policy": "完整组>0、净增量>0、五项T执行失败为0，净增量降序/index升序；无候选则hold"}
    if validation.exists():
        table = _read_csv(validation, inputs)
        validation_summary["candidate_rows"] = len(table)
        if "incremental_net_pnl" in table:
            values = pd.to_numeric(table.incremental_net_pnl, errors="raise")
            validation_summary.update(positive_net_rows=int(values.gt(0).sum()), negative_net_rows=int(values.lt(0).sum()), zero_net_rows=int(values.eq(0).sum()))
    _write_json(directory / "validation_summary.json", validation_summary)
    _plot(directory, summary, results, initial)
    _write_report_text(directory, cfg, summary, status)
    changed = [path for path, signature in inputs.items() if _sha(Path(path)) != signature]
    if changed:
        raise RuntimeError(f"汇总期间原始输入发生变化：{changed}")
    status["input_sha256"] = inputs
    status["reporter_sha256"] = _sha(Path(__file__))
    _write_json(directory / "timing_reporting_status.json", status)
    _write_json(directory / "goal_evaluation.json", status)
    return status


def _write_report_text(directory: Path, cfg: dict, summary: pd.DataFrame, status: dict) -> None:
    """撰写实际结果和研究限制；费用拆解基于成交价口径，不重复扣滑点。"""
    lines = ["# 库存做 T 择时验证（v11）", "",
             f"{len(summary)} 个独立年度账户，{status['active_years']} 年启用 T、{status['hold_years']} 年选择持有，"
             f"实际完成 {status['complete_t_pairs']} 组 T。扣费后净增量合计 **{status['aggregate_increment_rmb']:,.2f} 元**。",
             "持有代表内部没有同时满足正净收益、至少一组完整T及五项执行约束的候选；"
             "持有或启用均不证明存在盈利能力。启用也不等于每日必有成交。", "",
             "| 年份 | 状态 | 底仓收益 | 加入择时收益 | T净增量（元） | 完整组/恢复片段 | 活跃日/估值日 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.t_activity_status} | {row.baseline_total_return:.4%} | {row.total_return:.4%} | "
                     f"{row.t_added_equity:,.2f} | {row.t_pair_count}/{row.t_fragment_count} | {row.active_days}/{row.evaluated_trading_days} |")
    lines += ["", "active_traded=启用且成交；active_no_fills=启用但未成交；hold=内部选择持有。",
              f"各年独立初始资金 {cfg['initial_cash']:,.0f} 元，合计是独立账户净增量之和，不是复利或连续实盘收益。",
              "实际区间：" + "；".join(f"{r.test_year}：{r.evaluation_start} 至 {r.evaluation_end}" for r in summary.itertuples()) + "。",
              "2026数据仅截至4月30日，不能称作完整年度结果。", "",
              "| 年份 | 保存预测/请求 | 首腿/恢复成交 | T费用/滑点（元） | 跨日恢复组 | 股数偏差/恢复失败日 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    number = lambda value: "未提供" if value is None or pd.isna(value) else str(int(value))
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.test_year} | {number(row.prediction_rows)}/{number(row.submitted_t_requests)} | "
                     f"{row.t_first_leg_count}/{row.t_recovery_leg_count} | {row.t_fee:.2f}/{row.t_slippage_cost:.2f} | "
                     f"{row.t_cross_day_recovery_pair_count} | {row.inventory_mismatch_days}/{row.recovery_failure_days} |")
    lines += ["", f"已恢复片段：参考价毛差 {status['realized_pair_reference_gross_pnl']:,.2f} − "
              f"滑点 {status['realized_pair_slippage_cost']:,.2f} − 费用 {status['realized_pair_fees']:,.2f} "
              f"= 已实现净额 {status['realized_pair_net_pnl']:,.2f} 元。",
              "gross_pnl 使用含滑点成交价，net_pnl 已扣费用；不能从 net_pnl 再减滑点。"
              "T事件费用包括尚未恢复的首腿，已恢复片段与账户差可能因待恢复库存估值不同；"
              "未配对/估值差保存在 unpaired_or_marked_increment。", "",
              "| 年份 | 待恢复/T+1违规/底仓计划差 | 周退出失败日/股日 | 期末未退仓实际/基准 | 期末市值/陈旧估值（元） |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.pending_pairs}/{row.t1_violation_count}/{row.base_plan_mismatch_count} | "
                     f"{row.weekly_liquidation_failure_days}/{row.weekly_liquidation_failure_stock_days} | "
                     f"{row.weekly_unclosed_positions}/{row.baseline_weekly_unclosed_positions} | "
                     f"{row.final_position_market_value:.2f}/{row.final_stale_market_value:.2f} |")
    lines += ["", "库存偏差、恢复失败与周退出失败可能重叠，不能简单相加视为独立事故数。"
              "末日只有开盘旧价而没有收盘价也计入陈旧估值；未成交和未清仓不会被虚构补齐。", "",
              f"周增量重采样95%区间 [{status['aggregate_increment_ci_low_rmb']:,.2f}, "
              f"{status['aggregate_increment_ci_high_rmb']:,.2f}] 元；正增量年度 {status['positive_years']}/{len(summary)}。"
              f"收益统计参考满足：{status['profitability_reference_met']}；T执行约束满足：{status['t_execution_contract_met']}；"
              f"周退出完整：{status['weekly_execution_complete']}。这些为报告结果，不会反向替换已冻结规则。", "",
              "每折只在核心两年训练原始20/40/60交易分钟收益回归与扣费盈利三分类，最新训练年评估24个主动候选及持有。"
              "候选仅在内部完整组>0、净增量>0且五项T执行失败均为0时，按净增量降序/index升序选中，否则持有。"
              "没有100组、频率目标或置信区间开启门槛。全部目标年度规则冻结之后才运行外层；不重拟合外层T模型。", "",
              "三分类是固定端点的买先盈利/卖先盈利/两者均不盈利事件，输出未经概率校准，不是成交概率。"
              "每个信号依据已训练模型在20/40/60分钟中选择方向和预定恢复时刻，不是动态止盈止损。"
              "实际P&L来自股票两腿成交与估值，不是残差P&L或概率相加。", "",
              f"费用沿用：单边佣金 {cfg['commission_rate']:.4%}（最低 {cfg['min_commission']:g} 元）、"
              f"双边过户费 {cfg['transfer_rate']:.4%}、单边滑点 {cfg['slippage_rate']:.4%}；"
              f"卖出印花税2023-08-28前 {cfg['stamp_rate_before_20230828']:.4%}、之后 {cfg['stamp_rate_from_20230828']:.4%}。"
              f"单分钟成交额上限 {cfg['max_bar_amount_fraction']:.0%}，整手/现金/昨日可卖额度和原涨跌停代理保持不变。", "",
              "2023–2025留一年训练含后续年份，是回顾性迁移检验；2026折训练年份止于2025。"
              "这些年份已被多轮研究，因此包括2026在内均不是从未查看过的全新留出集。"
              "训练从最早真实特征来源到标签终点执行排除年度前后7日隔离，不能消除历次研究选择偏差。"
              "周bootstrap也没有完整校正跨周相关和多次尝试。", "",
              "共同底仓仍使用原冻结周模型70%收益排名＋30%振幅排名。入选前100不等于实际成交100只。"
              "周首新买不可卖、周末禁新T；每天买入不增加当天可卖额度。"
              "涨跌停价代理、缺盘口排队和公司行动校准的限制仍在，OHLCV不能识别真实信息流或保证恢复。", "",
              "本报告只对保存账本作一致性汇总，不等同独立原始行情审计。"
              "精确规则见各年 selected_rule.json，全部内部结果见 validation_rule_metrics.csv（若入口提供）。", ""]
    (directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "输出文件说明.md").write_text(
        "# v11 输出文件\n\n"
        "- final_report.md：启用/持有、收益、费用、失败与研究边界。\n"
        "- annual_summary.csv、年度.csv、test_metrics.csv：逐年机器精度摘要。\n"
        "- test_return_curve.png、测试集收益曲线.png：每年独立底仓与择时账户、净增量。\n"
        "- test_equity_curve.csv、test_trade_events.csv、test_round_trips.csv、test_open_positions.csv：各年原记录合并并附 test_year，不能当作连续组合。\n"
        "- daily_inventory_audit.csv、blocked_orders.csv、baseline_trade_events.csv、weekly_entry_plans.csv、base_plan_mismatches.csv：实际库存、受阻和共同底仓记录。\n"
        "- weekly_increment.csv：每个年度内的周净增量，不跨年续接。\n"
        "- timing_reporting_status.json、goal_evaluation.json：只报告统计和约束，不触发交易；含读取输入及报告器SHA。\n"
        "- validation_summary.json：原冻结规则与内部候选计数，不重新选参。\n"
        "- cross_validation_results.csv 与 year_Y/ 下文件：原始入口结果，本报告器不改写。\n",
        encoding="utf-8")
