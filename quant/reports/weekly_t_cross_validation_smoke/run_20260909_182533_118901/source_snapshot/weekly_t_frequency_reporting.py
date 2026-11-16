"""v10 频率研究的只读报告：固定内层频率目标、真实成交与执行失败分别披露。

不调用模型、账户或选参函数，不修改任何旧版本文件。复用已核账的年度汇总和
绘图工具，另外校验频率分母、记录 v9 对照来源，并保存本次运行的派生报告。
"""

from __future__ import annotations

import hashlib
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from .weekly_t_active_reporting import (
    EXECUTION_FAILURE_KEYS, RESULT_FILES, calculate_active_weekly_increment,
    plot_active_results, summarize_active_year, write_active_json,
)


VERSION = "weekly_t_v10_frequency"
FREQUENCY_COUNT_COLUMNS = (
    "eligible_portfolio_days", "eligible_inventory_stock_days", "submitted_t_requests",
    "runtime_signal_rows", "prediction_rows",
)


def _count(value: object, name: str) -> int:
    """拒绝布尔、缺失、非有限或小数计数，不把未知频率静默当成零。"""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or value != np.floor(value)):
        raise ValueError(f"{name} 必须为有限非负整数")
    return int(value)


def _sha256(path: Path) -> str:
    """记录本报告实际读取的旧结果身份，不改写旧目录。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize_frequency_year(year: int, result: dict, original: dict,
                              directory: Path, cfg: dict) -> dict:
    """核对首腿活动日和共同底仓分母；完整配对仍由原逐笔汇总核账。"""
    original = dict(original)
    enabled = original.get("enabled", original.get("t_enabled"))
    if not isinstance(enabled, (bool, np.bool_)) or not enabled:
        raise ValueError("频率报告要求 enabled=True，不能把关闭规则写成主动执行")
    if "t_enabled" in original and original["t_enabled"] != enabled:
        raise ValueError("enabled 与 t_enabled 不一致")
    original["t_enabled"] = True
    for key in FREQUENCY_COUNT_COLUMNS:
        if key not in original:
            raise ValueError(f"{year} 年缺少频率计数 {key}")
        original[key] = _count(original[key], key)
    original["prediction_row_count"] = original["prediction_rows"]
    original["t_request_count"] = original["submitted_t_requests"]
    summary_cfg = dict(cfg)
    summary_cfg.setdefault("candidate_activation_policy", "frequency_target_then_net_pnl")
    summary = summarize_active_year(year, result, original, directory, summary_cfg)
    for measured, provided in (("prediction_row_count", "prediction_rows"),
                               ("t_request_count", "submitted_t_requests")):
        if summary[measured] != original[provided]:
            raise ValueError(f"{year} 年保存行数与 {provided} 不一致")
    events = result["trade_events"]
    first = events.loc[events.event_type.eq("t") & events.fragment_id.eq(0)].copy()
    # 首腿而非恢复日决定活动覆盖；一次跨日恢复不能新增一个主动交易日。
    if len(first):
        first["activity_date"] = pd.to_datetime(first.datetime).dt.normalize()
        active_days = int(first.activity_date.nunique())
        active_stock_days = int(len(first[["activity_date", "symbol"]].drop_duplicates()))
    else:
        active_days = active_stock_days = 0
    for key, value in (("active_days", active_days), ("active_stock_days", active_stock_days)):
        if key in original and _count(original[key], key) != value:
            raise ValueError(f"{year} 年 {key} 与真实首腿不一致")
    eligible_days = original["eligible_portfolio_days"]
    eligible_stock_days = original["eligible_inventory_stock_days"]
    if eligible_days > eligible_stock_days:
        raise ValueError(f"{year} 年可计划日与股票日分母不一致")
    if eligible_days > summary["evaluated_trading_days"]:
        raise ValueError(f"{year} 年可计划日超过估值交易日")
    if summary["t_first_leg_count"] > original["submitted_t_requests"]:
        raise ValueError(f"{year} 年真实首腿超过提交请求数")
    # 恢复受阻可能改变实际持仓。共同基准机会并非实际活动的硬上界，必须
    # 用逐行交集计算覆盖，另列落在基准机会以外的首腿，不能隐藏或拒绝它们。
    opportunity_path = directory / f"year_{year}" / "eligible_inventory_stock_days.csv"
    coverage = {"coverage_exact_available": False, "active_days_in_baseline_opportunities": None,
                "active_stock_days_in_baseline_opportunities": None,
                "active_stock_days_outside_baseline_opportunities": None,
                "active_days_without_matched_baseline_opportunity": None,
                "active_day_coverage": None, "active_stock_day_coverage": None,
                "opportunity_file_sha256": None}
    if opportunity_path.exists():
        opportunity = pd.read_csv(opportunity_path, usecols=["date", "symbol"])
        opportunity["date"] = pd.to_datetime(opportunity.date, format="mixed", errors="raise").dt.normalize()
        if opportunity.isna().any().any() or opportunity.duplicated(["date", "symbol"]).any():
            raise ValueError(f"{year} 年共同基准机会缺失或重复")
        if len(opportunity) != eligible_stock_days or opportunity.date.nunique() != eligible_days:
            raise ValueError(f"{year} 年共同基准机会逐行数据与分母不一致")
        actual_keys = (first[["activity_date", "symbol"]].rename(columns={"activity_date": "date"}).drop_duplicates()
                       if len(first) else pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                                                       "symbol": pd.Series(dtype=str)}))
        matched = actual_keys.merge(opportunity, on=["date", "symbol"], how="inner", validate="one_to_one")
        matched_days, matched_stock_days = int(matched.date.nunique()), len(matched)
        coverage.update(coverage_exact_available=True,
                        active_days_in_baseline_opportunities=matched_days,
                        active_stock_days_in_baseline_opportunities=matched_stock_days,
                        active_stock_days_outside_baseline_opportunities=active_stock_days - matched_stock_days,
                        active_days_without_matched_baseline_opportunity=active_days - matched_days,
                        active_day_coverage=matched_days / eligible_days if eligible_days else 0.0,
                        active_stock_day_coverage=matched_stock_days / eligible_stock_days if eligible_stock_days else 0.0,
                        opportunity_file_sha256=_sha256(opportunity_path))
    pairs_per_day = summary["t_pair_count"] / eligible_days if eligible_days else 0.0
    summary.update(
        enabled=True, research_version=VERSION, active_days=active_days,
        active_stock_days=active_stock_days,
        pairs_per_eligible_portfolio_day=float(pairs_per_day),
        frequency_selection_score=float(min(pairs_per_day, cfg["frequency_target_pairs_per_day"])),
        frequency_target_pairs_per_day=float(cfg["frequency_target_pairs_per_day"]),
        frequency_target_met=bool(eligible_days and pairs_per_day >= cfg["frequency_target_pairs_per_day"]),
        frequency_denominator_available=bool(eligible_days),
        request_to_first_leg_rate=float(summary["t_first_leg_count"] / original["submitted_t_requests"])
        if original["submitted_t_requests"] else 0.0,
    )
    summary.update(coverage)
    return summary


def _compare_v9(directory: Path, summary: pd.DataFrame, results: list,
                cfg: dict) -> tuple[pd.DataFrame, dict]:
    """按年度读取 v9 实际汇总和共同基准曲线；不同区间不冒称直接可比。"""
    source = Path(cfg["source_active_report_dir"]).expanduser().resolve()
    if source == directory.resolve():
        raise ValueError("v10 输出目录不能与 v9 对照源目录相同")
    annual_path = source / "annual_summary.csv"
    old = pd.read_csv(annual_path, float_precision="round_trip")
    required = {"test_year", "t_pair_count", "t_added_equity", "t_first_leg_count",
                "t_recovery_leg_count", "t_fee", "inventory_mismatch_days", "recovery_failure_days"}
    if required.difference(old) or old.test_year.duplicated().any():
        raise ValueError("v9 对照年度表缺字段或年度重复")
    identity = {str(annual_path): _sha256(annual_path)}
    compared = []
    for year, result in results:
        previous = old.loc[old.test_year.eq(year)]
        if len(previous) != 1:
            raise ValueError(f"v9 对照缺少唯一的 {year} 年结果")
        previous = previous.iloc[0]
        current = summary.loc[summary.test_year.eq(year)].iloc[0]
        path = source / f"year_{year}" / "test_equity_curve.csv"
        base = pd.read_csv(path, usecols=["datetime", "baseline_equity"], float_precision="round_trip")
        identity[str(path)] = _sha256(path)
        base["datetime"] = pd.to_datetime(base.datetime, format="mixed", errors="raise")
        curve = result["equity_curve"][["datetime", "baseline_equity"]].copy()
        curve["datetime"] = pd.to_datetime(curve.datetime, errors="raise")
        if base.empty or base.datetime.duplicated().any() or not base.datetime.is_monotonic_increasing:
            raise ValueError(f"v9 {year} 年基准曲线为空、重复或非升序")
        if not np.isfinite(base.baseline_equity.to_numpy(dtype=float)).all():
            raise ValueError(f"v9 {year} 年基准包含非有限值")
        same_dates = bool(np.array_equal(base.datetime.to_numpy(), curve.datetime.to_numpy()))
        common = curve.merge(base, on="datetime", how="inner", suffixes=("_v10", "_v9"), validate="one_to_one")
        difference = float((common.baseline_equity_v10 - common.baseline_equity_v9).abs().max()) if len(common) else None
        identical = bool(same_dates and difference == 0.0)
        compared.append({
            "test_year": int(year), "v9_completed_pairs": _count(previous.t_pair_count, "v9 t_pair_count"),
            "v10_completed_pairs": int(current.t_pair_count),
            "pair_count_difference": int(current.t_pair_count - previous.t_pair_count),
            "v9_first_leg_count": _count(previous.t_first_leg_count, "v9 t_first_leg_count"),
            "v10_first_leg_count": int(current.t_first_leg_count),
            "v9_recovery_leg_count": _count(previous.t_recovery_leg_count, "v9 t_recovery_leg_count"),
            "v10_recovery_leg_count": int(current.t_recovery_leg_count),
            "v9_t_added_equity": float(previous.t_added_equity), "v10_t_added_equity": float(current.t_added_equity),
            "t_added_equity_difference": float(current.t_added_equity - previous.t_added_equity),
            "v9_t_fee": float(previous.t_fee), "v10_t_fee": float(current.t_fee),
            "v9_inventory_mismatch_days": _count(previous.inventory_mismatch_days, "v9 inventory_mismatch_days"),
            "v10_inventory_mismatch_days": int(current.inventory_mismatch_days),
            "v9_recovery_failure_days": _count(previous.recovery_failure_days, "v9 recovery_failure_days"),
            "v10_recovery_failure_days": int(current.recovery_failure_days),
            "same_evaluation_dates": same_dates, "common_valuation_rows": len(common),
            "baseline_common_max_abs_difference_rmb": difference,
            "baseline_daily_exact_match": identical, "directly_comparable": identical,
        })
    return pd.DataFrame(compared), {"source_active_report_dir": str(source), "input_sha256": identity}


def write_frequency_report(directory: Path, cfg: dict, annual: pd.DataFrame,
                           results: list, validation: pd.DataFrame) -> dict:
    """保存 v10 频率报告并返回 validation_summary；results 为年度、结果字典二元组。

所有输入只读。年度 enabled 来自已冻结规则；频率、收益、置信区间和失败状态
均为结果披露，本函数不会依照外层结果更换候选或关闭交易。
"""
    directory = Path(directory)
    if cfg.get("research_version") != VERSION:
        raise ValueError(f"频率报告只支持 {VERSION}")
    for key in ("initial_cash", "frequency_target_pairs_per_day"):
        value = cfg[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须有限且正")
    sample_count = cfg["bootstrap_samples"]
    if isinstance(sample_count, (bool, np.bool_)) or not isinstance(sample_count, Integral) or sample_count <= 0:
        raise ValueError("bootstrap_samples 必须正整数")
    if annual.empty or "test_year" not in annual or annual.test_year.duplicated().any():
        raise ValueError("年度表必须非空且年份唯一")
    if not results or len({int(year) for year, _ in results}) != len(results):
        raise ValueError("结果年度必须非空且唯一")
    if set(map(int, annual.test_year)) != {int(year) for year, _ in results}:
        raise ValueError("年度表与原始结果的年份不同")
    if validation.empty or not {"test_year", "selected"}.issubset(validation):
        raise ValueError("内部候选表必须非空并保留 test_year/selected")
    if not validation.selected.map(lambda item: isinstance(item, (bool, np.bool_))).all():
        raise ValueError("内部 selected 必须明确为布尔")
    ordered = sorted(results, key=lambda item: int(item[0]))
    records, weeks = [], []
    for year, result in ordered:
        if int(validation.loc[validation.test_year.eq(year), "selected"].sum()) != 1:
            raise ValueError(f"{year} 年必须恰有一个事前选中的内层候选")
        weeks.append(calculate_active_weekly_increment(result["equity_curve"], int(year)))
        records.append(_summarize_frequency_year(
            int(year), result, annual.loc[annual.test_year.eq(year)].iloc[0].to_dict(), directory, cfg))
    summary = pd.DataFrame(records)
    weekly = pd.concat(weeks, ignore_index=True)
    comparison, source_identity = _compare_v9(directory, summary, ordered, cfg)
    summary = summary.merge(comparison[["test_year", "baseline_daily_exact_match", "directly_comparable"]],
                            on="test_year", validate="one_to_one")
    rng = np.random.default_rng(int(cfg["random_state"]))
    increments = weekly.t_increment.to_numpy(dtype=float)
    # 只按年度内权益差得到周增量；合并抽样不是跨年复利组合。
    samples = rng.choice(increments, size=(int(sample_count), len(increments)), replace=True).sum(axis=1)
    low, high = map(float, np.quantile(samples, [.025, .975]))
    added = float(summary.t_added_equity.sum())
    positive_years = int(summary.t_added_equity.gt(0).sum())
    statistical_reference = bool(added > 0 and positive_years > len(summary) / 2 and low > 0)
    failures = {key: int(summary[key].sum()) for key in EXECUTION_FAILURE_KEYS}
    t_contract = not any(failures.values())
    weekly_complete = bool(summary.weekly_liquidation_failure_days.sum() == 0
                           and summary.weekly_unclosed_positions.sum() == 0
                           and summary.baseline_weekly_unclosed_positions.sum() == 0)
    eligible_days = int(summary.eligible_portfolio_days.sum())
    eligible_stock_days = int(summary.eligible_inventory_stock_days.sum())
    total_pairs = int(summary.t_pair_count.sum())
    exact_coverage = bool(summary.coverage_exact_available.all())
    chosen_columns = [name for name in (
        "test_year", "inner_year", "candidate_index", "candidate_kind", "profile", "net_margin", "quantile",
        "vwap_gap", "t_pair_count", "active_days", "eligible_portfolio_days", "eligible_inventory_stock_days",
        "pairs_per_eligible_portfolio_day", "frequency_selection_score", "incremental_net_pnl", *EXECUTION_FAILURE_KEYS,
    ) if name in validation]
    selected_records = validation.loc[validation.selected, chosen_columns].copy()
    # 保留未知候选参数为 JSON null；不能把选中的原始候选重新排序或重写。
    selected_records = selected_records.astype(object).where(pd.notna(selected_records), None).to_dict("records")
    status = {
        "research_version": VERSION, "report_only": True, "affects_activation": False,
        "selection_uses_outer_results": False,
        "selection_order": "min(inner_complete_pairs / inner_eligible_portfolio_days, frequency_target_pairs_per_day) desc; inner_incremental_net_pnl desc; candidate_index asc",
        "execution_eligibility": "inner complete pairs > 0 and all five execution failure counts = 0; negative net profit allowed",
        "frequency_target_pairs_per_day": float(cfg["frequency_target_pairs_per_day"]),
        "candidate_counts_by_year": {str(int(year)): len(group) for year, group in validation.groupby("test_year")},
        "selected_inner_candidates": selected_records,
        "all_years_t_enabled": True, "years_with_actual_t_trades": int(summary.t_has_actual_trades.sum()),
        "evaluated_years": len(summary), "complete_pairs": total_pairs,
        "eligible_portfolio_days": eligible_days, "eligible_inventory_stock_days": eligible_stock_days,
        "active_days": int(summary.active_days.sum()), "active_stock_days": int(summary.active_stock_days.sum()),
        "pairs_per_eligible_portfolio_day": total_pairs / eligible_days if eligible_days else 0.0,
        "coverage_exact_available": exact_coverage,
        "active_day_coverage": (float(summary.active_days_in_baseline_opportunities.sum() / eligible_days)
                                if eligible_days else 0.0) if exact_coverage else None,
        "active_stock_day_coverage": (float(summary.active_stock_days_in_baseline_opportunities.sum() / eligible_stock_days)
                                      if eligible_stock_days else 0.0) if exact_coverage else None,
        "active_stock_days_outside_baseline_opportunities": int(summary.active_stock_days_outside_baseline_opportunities.sum())
        if exact_coverage else None,
        "all_years_frequency_target_met": bool(summary.frequency_target_met.all()),
        "aggregate_increment_rmb": added, "aggregate_increment_ci_low_rmb": low,
        "aggregate_increment_ci_high_rmb": high, "positive_years": positive_years,
        "profitability_reference_met": statistical_reference, "statistical_reference_affects_activation": False,
        "t_execution_contract_met": t_contract, "execution_failure_totals": failures,
        "constraint_count_note": "失败类型及不同独立年度可重叠，不能把各类计数加成独立事件总数",
        "weekly_execution_complete": weekly_complete,
        "weekly_exit_failure_days": int(summary.weekly_liquidation_failure_days.sum()),
        "weekly_exit_failure_stock_days": int(summary.weekly_exit_failure_stock_days.sum()),
        "all_years_v9_baseline_daily_exact_match": bool(comparison.baseline_daily_exact_match.all()),
        "historical_re_evaluation": True, "v9_reference_source": source_identity,
    }
    directory.mkdir(parents=True, exist_ok=True)
    for frame, name in ((summary, "annual_summary.csv"), (summary, "test_metrics.csv"),
                        (validation, "validation_rule_metrics.csv"), (weekly, "weekly_increment.csv"),
                        (comparison, "frequency_comparison_v9.csv")):
        frame.to_csv(directory / name, index=False, encoding="utf-8-sig")
    for key, name in RESULT_FILES.items():
        frames = [result[key].assign(test_year=int(year)) for year, result in ordered if key in result]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(directory / name, index=False, encoding="utf-8-sig")
    for name in ("validation_summary.json", "frequency_reporting_status.json", "goal_evaluation.json"):
        write_active_json(directory / name, status)
    plot_active_results(ordered, summary, cfg, directory)
    _write_frequency_text(directory, cfg, summary, comparison, status)
    return status


def _write_frequency_text(directory: Path, cfg: dict, summary: pd.DataFrame,
                          comparison: pd.DataFrame, status: dict) -> None:
    """使用 v10 独立文字，不沿用 v9 的最高净利优先或旧统计晋级描述。"""
    yes = lambda value: "是" if value else "否"
    frequency = ["| 年份 | 预测/信号/请求 | 首腿/恢复片段/完整组 | 活跃日/可计划日 | 每可计划日完整组 | 活跃股票日/可计划股票日 | 股票日交集覆盖/基准机会外活动 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    returns = ["| 年份 | 相同底仓收益 | 底仓＋T收益 | 扣费T净增量（元） | T手续费/滑点（元） | 实际区间 |",
               "| --- | ---: | ---: | ---: | ---: | --- |"]
    failures = ["| 年份 | 库存差异日/股日 | 恢复失败日/股日 | 跨日恢复组/期末待恢复 | T+1/底仓计划差异 | 周退出失败日/股日 | 未退出持仓实际/基准 | 陈旧估值（元） |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    compares = ["| 年份 | v9→v10 完整组 | v9→v10 净增量（元） | v9→v10 T费用（元） | 逐日基准精确相同 |",
                "| --- | ---: | ---: | ---: | --- |"]
    for row in summary.itertuples(index=False):
        coverage_text = (f"{row.active_stock_day_coverage:.2%}/{int(row.active_stock_days_outside_baseline_opportunities)}"
                         if row.coverage_exact_available else "缺少逐行机会，未知")
        frequency.append(f"| {row.test_year} | {row.prediction_rows}/{row.runtime_signal_rows}/{row.submitted_t_requests} | "
                         f"{row.t_first_leg_count}/{row.t_recovery_leg_count}/{row.t_pair_count} | "
                         f"{row.active_days}/{row.eligible_portfolio_days} | {row.pairs_per_eligible_portfolio_day:.4f} | "
                         f"{row.active_stock_days}/{row.eligible_inventory_stock_days} | {coverage_text} |")
        returns.append(f"| {row.test_year} | {row.baseline_total_return:.6%} | {row.total_return:.6%} | "
                       f"{row.t_added_equity:,.2f} | {row.t_fee:,.2f}/{row.t_slippage_cost:,.2f} | "
                       f"{row.evaluation_start} 至 {row.evaluation_end}（{row.evaluated_trading_days}日） |")
        failures.append(f"| {row.test_year} | {row.inventory_mismatch_days}/{row.inventory_mismatch_stock_days} | "
                        f"{row.recovery_failure_days}/{row.recovery_failure_stock_days} | "
                        f"{row.t_cross_day_recovery_pair_count}/{row.pending_pairs} | "
                        f"{row.t1_violation_count}/{row.base_plan_mismatch_count} | "
                        f"{row.weekly_liquidation_failure_days}/{row.weekly_exit_failure_stock_days} | "
                        f"{row.weekly_unclosed_positions}/{row.baseline_weekly_unclosed_positions} | {row.final_stale_market_value:,.2f} |")
    for row in comparison.itertuples(index=False):
        compares.append(f"| {row.test_year} | {row.v9_completed_pairs}→{row.v10_completed_pairs} | "
                        f"{row.v9_t_added_equity:,.2f}→{row.v10_t_added_equity:,.2f} | "
                        f"{row.v9_t_fee:,.2f}→{row.v10_t_fee:,.2f} | {yes(row.baseline_daily_exact_match)} |")
    text = ["# 频率优先做 T：年度结果与执行约束（v10）", "",
            f"{len(summary)} 个目标年度均启用冻结规则，其中 {status['years_with_actual_t_trades']} 年产生实际首腿。"
            f"完成 {status['complete_pairs']} 组 T；各年独立投入 {cfg['initial_cash']:,.0f} 元，"
            f"扣费后净增量合计 **{status['aggregate_increment_rmb']:,.2f} 元**。实际结果不预设频率改善或盈利。", "",
            "## 频率与覆盖", "", *frequency, "",
            "可计划日分母由无T共同底仓计算：当日开盘昨日股数按25%取整手至少100股，"
            "处于允许做T的周中日期，且有h20计划恢复不晚于14:30的时钟；不以实际成交、现金或未来恢复成功筛选分母。"
            "eligible_portfolio_days 是至少有一只符合条件股票的日期数，eligible_inventory_stock_days 是股票日期组合数。"
            "它们是持仓和时钟条件，不保证容量、现金和涨跌停允许成交。0分母显示为0覆盖，另存 frequency_denominator_available=false。",
            "活跃日及股票日按真实首腿时间去重，跨日恢复不会额外计为主动活动日；完整组按pair_id去重，"
            "恢复片段不能重复当成完整交易。估值日可能含期末恢复重试日，不能直接替代频率分母。"
            "覆盖率分子按真实首腿(date,symbol)与保存的共同底仓机会表取交集；全部实际活动仍列出，"
            "另列基准机会之外的活动，不因恢复失败改变实际股数而隐藏成交。缺逐行机会表则覆盖率未知。"
            "请求可在同一股票同日多次提交，每股每日实际首腿仍受原引擎一组上限约束。", "",
            "## 收益、费用与 v9 对照", "", *returns, "", *compares, "",
            "v9对照取 source_active_report_dir 保存的真实年度汇总，不重新选择v9规则。"
            "表内每年独立账户，不能拼接成年际复利曲线或相加年度收益率；不同日期或底仓的smoke仅列数值，"
            "frequency_comparison_v9.csv 的 directly_comparable=false 明示不能直接归因差异。",
            "T净增量是实际账户权益减共同底仓权益，包含未恢复仓位估值和底仓差异。"
            "realized_pair_net_pnl 使用含滑点成交价后扣实际手续费；参考价毛差−滑点−费用可作归因，"
            "不能从已含滑点的净利再次扣滑点。所有T腿费用另计，包括尚未完整恢复首腿。", "",
            "## 执行约束与遗留仓位", "", *failures, "",
            f"全部T执行约束满足：**{yes(status['t_execution_contract_met'])}**；"
            f"全部周退出完成：**{yes(status['weekly_execution_complete'])}**。"
            "恢复受阻时保留真实仓位并跨日重试，日终股数差异不能用期末pending=0抹去；"
            "次日恢复造成相对该日开盘的股数变化也计入审计。陈旧估值按实际末次报价时点统计，不能当作可实现成交价。", "",
            "## 固定候选与训练内部选择", "",
            "每折25个候选：1个精确v9参考规则；3种h20核心模型×净边际0/5bp的6个绝对信号候选；"
            "3种模型×分位0.70/0.85/0.95×VWAP偏离0.0035/0.005的18个相对排名候选。"
            "核心权重沿用本折v8，周排名和共同底仓不变。相对排名使用信号时当前可见条件内的股票，"
            "取top ceil((1−q)×n)：买先要求预测>0、价格低于VWAP且5分钟收益≥0，卖先方向相反；"
            "已知日收益绝对值≤3%、已知振幅≤6%。相对排名候选没有预测净收益覆盖费用的门槛，真实每腿仍照常收费。",
            "每折只用自己的inner选择：保留完整pair>0且库存差异、恢复失败、期末待恢复、底仓计划差异、T+1违规五项全0的候选，"
            f"先按 min(完整pair/可计划日, {cfg['frequency_target_pairs_per_day']:g}) 降序，"
            "再按扣费净增量降序、candidate_index升序。允许负收益，没有符合执行约束的主动候选时显式报错。"
            "所有目标年度规则先冻结，再执行外层；不按外层表现重排候选。频率达到目标不代表收益达标。",
            f"费用保持：每腿佣金 {cfg['commission_rate']:.4%}、过户 {cfg['transfer_rate']:.4%}、"
            f"滑点 {cfg['slippage_rate']:.4%}、最低佣金 {cfg['min_commission']} 元；印花税按原成交日期规则。", "",
            "## 统计参考与研究边界", "",
            f"正增量年度 {status['positive_years']}/{len(summary)}；独立年度内周增量重抽样95%区间 "
            f"[{status['aggregate_increment_ci_low_rmb']:,.2f}, {status['aggregate_increment_ci_high_rmb']:,.2f}] 元。"
            f"旧利润参考条件满足：{yes(status['profitability_reference_met'])}。"
            "样本数100组、净收益正负、置信区间及集中度均不控制开启；validation_summary.json/goal_evaluation.json "
            "明确 report_only=true、affects_activation=false。报告函数不停止策略或回退持有。",
            "历史外层已在前期研究中查看，本轮是重复历史研究，不是全新未见测试；2023–2025留年折含未来训练年，"
            "2026只覆盖现有数据区间。周块区间未完整处理所有时序依赖或历次研究选择偏差。"
            "现有历史ST/公司行动/精确涨跌停与盘口限制保留。账目或代码检查通过不能代替每日股数约束或盈利证据。", "",
            "[年度统计](annual_summary.csv) · [v9对照](frequency_comparison_v9.csv) · "
            "[收益图](test_return_curve.png) · [净增量图](t_increment_curve.png) · "
            "[内部选择与验证摘要](validation_summary.json) · [库存审计](daily_inventory_audit.csv) · [受阻请求](blocked_orders.csv)"]
    (directory / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (directory / "输出文件说明.md").write_text(
        "# 输出文件说明\n\n"
        "annual_summary.csv/test_metrics.csv 保存年度账目、真实配对、已知持仓时钟分母及覆盖率；"
        "enabled/t_enabled 和 t_has_actual_trades 分别表示规则启用与真实首腿成交。\n\n"
        "frequency_comparison_v9.csv 保存v9既有结果对照、共同日期数量和逐日基准差；"
        "validation_summary.json/frequency_reporting_status.json/goal_evaluation.json 仅报告，不控制交易。"
        "摘要记录对照输入SHA和事前已选候选，validation_rule_metrics.csv保留所有内层候选，不由报告重选。\n\n"
        "test_equity_curve.csv、test_trade_events.csv、test_round_trips.csv、test_open_positions.csv、"
        "daily_inventory_audit.csv、blocked_orders.csv、baseline_trade_events.csv、weekly_entry_plans.csv、"
        "base_plan_mismatches.csv按test_year合并独立年度原始结果。weekly_increment.csv先各年内作周权益差分。\n\n"
        "test_return_curve.png/测试集收益曲线.png和t_increment_curve.png复用实际账目绘图，"
        "不跨年拼接账户。T手续费/滑点按全部T成交腿统计，realized_pair_*按已恢复片段统计；"
        "未恢复首腿费用与剩余仓位估值不能遗漏。\n", encoding="utf-8")
