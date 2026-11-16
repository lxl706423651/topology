"""主动库存做 T 的只读结果汇总：启用、统计参考与真实执行合约分开报告。

由独立 v9 入口传入已完成的年度结果与内部候选表；本模块不选参、不修改开关、
不训练模型、不下单。仅向指定新报告目录写标准 CSV、说明、JSON 和静态收益图。
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


EXECUTION_FAILURE_KEYS = (
    "inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
    "base_plan_mismatch_count", "t1_violation_count",
)
RESULT_FILES = {
    "equity_curve": "test_equity_curve.csv",
    "trade_events": "test_trade_events.csv",
    "round_trips": "test_round_trips.csv",
    "open_positions": "test_open_positions.csv",
    "daily_inventory_audit": "daily_inventory_audit.csv",
    "order_attempts": "blocked_orders.csv",
    "baseline_events": "baseline_trade_events.csv",
    "weekly_entry_plans": "weekly_entry_plans.csv",
    "base_plan_mismatches": "base_plan_mismatches.csv",
}


def write_active_json(path: Path, value: dict) -> None:
    """写 UTF-8 结果 JSON；numpy 标量归一化，不允许静默输出非有限指标。"""
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (Path, pd.Timestamp)):
            return str(item)
        raise TypeError(f"不支持 JSON 类型：{type(item)}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n",
                    encoding="utf-8")


def calculate_active_weekly_increment(curve: pd.DataFrame, year: int) -> pd.DataFrame:
    """各独立年度内先取周末权益差，再作差；不跨年拼接本金或收益率。"""
    selected = curve[["datetime", "equity", "baseline_equity"]].copy()
    selected["datetime"] = pd.to_datetime(selected.datetime, errors="raise")
    if selected.empty or selected.datetime.duplicated().any() or not selected.datetime.is_monotonic_increasing:
        raise ValueError(f"{year} 年净值必须非空、日期唯一且升序")
    if not selected.datetime.dt.year.eq(year).all():
        raise ValueError(f"{year} 年净值混入其他年度")
    if not np.isfinite(selected[["equity", "baseline_equity"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{year} 年净值包含非有限值")
    selected["trade_week"] = selected.datetime.dt.to_period("W-SUN").dt.start_time
    weekly = selected.groupby("trade_week", sort=True).last().reset_index()
    weekly["t_added_equity"] = weekly.equity - weekly.baseline_equity
    weekly["t_increment"] = weekly.t_added_equity.diff().fillna(weekly.t_added_equity)
    weekly["test_year"] = int(year)
    return weekly


def count_saved_active_rows(path: Path) -> int | None:
    """读取入口已保存的预测/请求行数；缺文件返回未知，不能伪装成零请求。"""
    if not path.exists():
        return None
    return len(pd.read_csv(path, usecols=["datetime"]))


def summarize_active_year(year: int, result: dict, original: dict, directory: Path, cfg: dict) -> dict:
    """从原始事件、恢复片段和库存审计计算 T 专属数量，保留原年度参考统计。"""
    required = set(RESULT_FILES).difference({"weekly_entry_plans", "base_plan_mismatches"}) | {"metrics"}
    if required.difference(result):
        raise ValueError(f"{year} 年结果缺表：{sorted(required.difference(result))}")
    enabled = original.get("t_enabled")
    if not isinstance(enabled, (bool, np.bool_)) or not enabled:
        raise ValueError("主动报告要求年度 t_enabled=True；不得把关闭策略静默写成主动")
    events, trips = result["trade_events"], result["round_trips"]
    audit, curve, positions = result["daily_inventory_audit"], result["equity_curve"], result["open_positions"]
    metrics = result["metrics"]
    absent = set(EXECUTION_FAILURE_KEYS).difference(metrics)
    absent |= {"t_pair_count", "weekly_liquidation_failure_days", "weekly_unclosed_positions",
               "baseline_weekly_unclosed_positions"}.difference(metrics)
    if absent:
        raise ValueError(f"{year} 年缺少必须披露的执行指标：{sorted(absent)}")
    t_events = events.loc[events.event_type.eq("t")]
    first = t_events.loc[t_events.fragment_id.eq(0)]
    recovery = t_events.loc[t_events.fragment_id.gt(0)]
    if len(t_events) != len(first) + len(recovery) or len(first) != first.pair_id.nunique():
        raise ValueError(f"{year} 年 T 首腿或恢复片段标识不一致")
    complete = trips.loc[trips.pair_fully_recovered.eq(True)] if len(trips) else trips
    complete_count = int(complete.pair_id.nunique())
    if complete_count != int(metrics["t_pair_count"]) or len(trips) != len(recovery):
        raise ValueError(f"{year} 年完整配对、恢复事件与片段表不一致")
    if len(first) != complete_count + int(metrics["pending_pairs"]):
        raise ValueError(f"{year} 年首腿数不等于完整配对加期末待恢复数")
    initial = float(cfg["initial_cash"])
    added = float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1])
    last_date = pd.Timestamp(curve.datetime.iloc[-1]).normalize()
    same_day = (pd.to_datetime(trips.first_datetime).dt.normalize()
                == pd.to_datetime(trips.second_datetime).dt.normalize()) if len(trips) else pd.Series(dtype=bool)
    late_pairs = int(trips.loc[~same_day, "pair_id"].nunique()) if len(trips) else 0
    stale = (pd.to_datetime(positions.mark_datetime).dt.normalize().lt(last_date)
             & positions.shares.gt(0)) if len(positions) else pd.Series(dtype=bool)
    daily_inventory = audit.loc[audit.inventory_difference.ne(0)]
    recovery_failures = audit.loc[audit.recovery_failure.eq(True)]
    weekly_failures = audit.loc[audit.weekly_liquidation_failure.eq(True)]
    for key, rows in (("inventory_mismatch_days", daily_inventory), ("recovery_failure_days", recovery_failures),
                      ("weekly_liquidation_failure_days", weekly_failures)):
        if int(metrics[key]) != int(rows.date.nunique()):
            raise ValueError(f"{year} 年 {key} 与逐日库存审计不符")
    value = {**original, **metrics, "test_year": int(year), "t_enabled": True,
             "candidate_activation_policy": cfg["candidate_activation_policy"],
             "statistical_reference_affects_activation": False,
             "t_has_actual_trades": bool(len(first)), "all_actual_trade_events": len(events),
             "t_first_leg_count": len(first), "t_recovery_leg_count": len(recovery),
             "t_trade_event_count": len(t_events), "t_pair_count": complete_count,
             "completed_pair_count": complete_count, "t_fragment_count": len(trips), "fragment_count": len(trips),
             "t_cross_day_recovery_pair_count": late_pairs,
             "t_fee": float(t_events.fee.sum()), "t_slippage_cost": float(t_events.slippage_cost.sum()),
             "realized_pair_reference_gross_pnl": float((trips.gross_pnl + trips.slippage_cost).sum()),
             "realized_pair_slippage_cost": float(trips.slippage_cost.sum()),
             "realized_pair_fees": float((trips.buy_fee + trips.sell_fee).sum()),
             "realized_pair_net_pnl": float(trips.net_pnl.sum()),
             "total_return": float(curve.equity.iloc[-1] / initial - 1),
             "baseline_total_return": float(curve.baseline_equity.iloc[-1] / initial - 1),
             "t_added_equity": added, "increment_return": added / initial,
             "evaluation_start": str(curve.datetime.iloc[0]), "evaluation_end": str(curve.datetime.iloc[-1]),
             "evaluated_trading_days": len(curve), "inventory_mismatch_stock_days": len(daily_inventory),
             "recovery_failure_stock_days": len(recovery_failures), "weekly_exit_failure_stock_days": len(weekly_failures),
             "final_open_position_rows": int(positions.shares.gt(0).sum()),
             "final_position_market_value": float(positions.market_value.sum()),
             "final_stale_position_count": int(stale.sum()),
             "final_stale_market_value": float(positions.loc[stale, "market_value"].sum()) if len(positions) else 0.0,
             "t_execution_contract_met": all(int(metrics[k]) == 0 for k in EXECUTION_FAILURE_KEYS)}
    for key, filename in (("prediction_row_count", "t_predictions.csv"), ("t_request_count", "t_decisions.csv")):
        count = count_saved_active_rows(directory / f"year_{year}" / filename)
        if count is None:
            count = original.get(key)
        value[key] = count
    return value


def plot_active_results(results: list, summary: pd.DataFrame, cfg: dict, directory: Path) -> None:
    """每年独立绘制持有/主动账户和净增量；标题包含真实 T 数量与恢复失败。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = sorted(Path("/usr/share/fonts/opentype/noto").glob("*CJK*"))
    chinese = bool(fonts)
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 3.5 * len(results)), squeeze=False)
    extra, incremental_axes = plt.subplots(len(results), 1, figsize=(11, 3.1 * len(results)), squeeze=False)
    for index, (year, result) in enumerate(results):
        curve = result["equity_curve"]
        row = summary.loc[summary.test_year.eq(year)].iloc[0]
        left, right = axes[index]
        left.plot(curve.datetime, (curve.baseline_equity / cfg["initial_cash"] - 1) * 100,
                  label="相同周度底仓" if chinese else "Same weekly baseline")
        left.plot(curve.datetime, (curve.equity / cfg["initial_cash"] - 1) * 100,
                  label="底仓＋主动做 T" if chinese else "Weekly baseline + active T", linestyle="--")
        left.set_ylabel("收益（%）" if chinese else "Return (%)")
        left.set_title(f"{year}｜T {row.t_pair_count} 组｜净增量 {row.t_added_equity:,.2f} 元" if chinese
                       else f"{year}: {row.t_pair_count} T pairs; increment CNY {row.t_added_equity:,.2f}")
        left.legend(fontsize=9)
        for ax in (right, incremental_axes[index, 0]):
            ax.plot(curve.datetime, curve.equity - curve.baseline_equity, color="#1c7a70")
            ax.axhline(0, color="grey", linewidth=.6)
            ax.set_ylabel("净增量（元）" if chinese else "Increment (CNY)")
            ax.set_title(f"{year}｜股数偏差 {row.inventory_mismatch_days} 日｜恢复失败 {row.recovery_failure_days} 日" if chinese
                         else f"{year}: inventory mismatch {row.inventory_mismatch_days}d; recovery failure {row.recovery_failure_days}d")
        for ax in (left, right, incremental_axes[index, 0]):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.grid(alpha=.25)
    fig.tight_layout()
    extra.tight_layout()
    fig.savefig(directory / "test_return_curve.png", dpi=150)
    extra.savefig(directory / "t_increment_curve.png", dpi=150)
    plt.close(fig)
    plt.close(extra)
    shutil.copyfile(directory / "test_return_curve.png", directory / "测试集收益曲线.png")


def write_active_report(directory: Path, cfg: dict, annual: pd.DataFrame, results: list,
                        validation: pd.DataFrame) -> None:
    """保存主动 v9 报告；results 为 [(year, resultdict)]，输入表、配置和开关不变。

内部启用只由入口决定：有实际完整配对且五项执行失败为零，按内部净增量选择，
允许负收益。此处计算统计参考与外层执行状态，任何结果均不反向停止或更换策略。
"""
    directory = Path(directory)
    if cfg.get("candidate_activation_policy") != "active_execution_valid":
        raise ValueError("主动报告只支持 active_execution_valid")
    if not np.isfinite(cfg["initial_cash"]) or cfg["initial_cash"] <= 0:
        raise ValueError("initial_cash 必须有限且正")
    if annual.empty or "test_year" not in annual or annual.test_year.duplicated().any():
        raise ValueError("年度汇总必须非空且 test_year 唯一")
    if not results or len({int(y) for y, _ in results}) != len(results):
        raise ValueError("年度结果必须非空且年份唯一")
    if set(map(int, annual.test_year)) != {int(y) for y, _ in results}:
        raise ValueError("年度汇总与结果年份不一致")
    ordered = sorted(results, key=lambda item: int(item[0]))
    records, weeks = [], []
    for year, result in ordered:
        weekly = calculate_active_weekly_increment(result["equity_curve"], int(year))
        weeks.append(weekly)
        records.append(summarize_active_year(int(year), result, annual.loc[annual.test_year.eq(year)].iloc[0].to_dict(),
                                             directory, cfg))
    summary = pd.DataFrame(records)
    weekly = pd.concat(weeks, ignore_index=True)
    sample_count = int(cfg["bootstrap_samples"])
    if sample_count <= 0:
        raise ValueError("bootstrap_samples 必须正整数")
    increments = weekly.t_increment.to_numpy(dtype=float)
    rng = np.random.default_rng(int(cfg["random_state"]))
    boot = rng.choice(increments, size=(sample_count, len(increments)), replace=True).sum(axis=1)
    low, high = map(float, np.quantile(boot, [.025, .975]))
    positive_years = int(summary.t_added_equity.gt(0).sum())
    positive_net = bool(summary.t_added_equity.sum() > 0)
    statistical_reference = bool(positive_net and positive_years > len(summary) / 2 and low > 0)
    failure_totals = {key: int(summary[key].sum()) for key in EXECUTION_FAILURE_KEYS}
    t_contract = not any(failure_totals.values())
    weekly_complete = bool(summary.weekly_liquidation_failure_days.sum() == 0
                           and summary.weekly_unclosed_positions.sum() == 0
                           and summary.baseline_weekly_unclosed_positions.sum() == 0)
    status = {"report_only": True, "affects_activation": False,
              "candidate_activation_policy": cfg["candidate_activation_policy"],
              "activation_uses_statistical_reference": False, "all_years_t_enabled": True,
              "years_with_actual_t_trades": int(summary.t_has_actual_trades.sum()),
              "aggregate_increment_rmb": float(summary.t_added_equity.sum()),
              "aggregate_increment_ci_low_rmb": low, "aggregate_increment_ci_high_rmb": high,
              "positive_years": positive_years, "evaluated_years": len(summary),
              "profitability_reference_met": statistical_reference,
              "reference_definition": "净增量合计>0、正增量年度超过半数、周块95%区间下界>0；仅报告，不控制启用",
              "t_execution_contract_met": t_contract, "execution_failure_totals": failure_totals,
              "constraint_failures": sum(failure_totals.values()),
              "constraint_count_note": "各类指标计数相加，不是互不重叠的失败事件数",
              "weekly_execution_complete": weekly_complete,
              "weekly_exit_failure_days": int(summary.weekly_liquidation_failure_days.sum()),
              "weekly_exit_failure_stock_days": int(summary.weekly_exit_failure_stock_days.sum()),
              "weekly_unclosed_positions": int(summary.weekly_unclosed_positions.sum()),
              "baseline_weekly_unclosed_positions": int(summary.baseline_weekly_unclosed_positions.sum()),
              "t_gain_criterion_met": statistical_reference and t_contract,
              "gain_criterion_met": statistical_reference and t_contract and weekly_complete,
              "legacy_criterion_note": "兼容参考字段；任一值为false也不会改动enabled或触发持有回退",
              "historical_re_evaluation": True}
    directory.mkdir(parents=True, exist_ok=True)
    for frame, filename in ((summary, "annual_summary.csv"), (summary, "test_metrics.csv"),
                            (validation, "validation_rule_metrics.csv"), (weekly, "weekly_increment.csv")):
        frame.to_csv(directory / filename, index=False, encoding="utf-8-sig")
    for key, filename in RESULT_FILES.items():
        frames = [result[key].assign(test_year=int(year)) for year, result in ordered if key in result]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(directory / filename, index=False, encoding="utf-8-sig")
    write_active_json(directory / "goal_evaluation.json", status)
    write_active_json(directory / "active_reporting_status.json", status)
    plot_active_results(ordered, summary, cfg, directory)
    yes = lambda value: "是" if value else "否"
    overview = ["| 年份 | T启用/实际成交 | 评价日数 | 底仓收益 | 主动账户收益 | T净增量（元） | 完整T组数 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    counts = ["| 年份 | 保存预测/请求 | T首腿/恢复腿 | 片段/跨日恢复组 | T费用/滑点（元） | 股数偏差/恢复失败日 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    failures = ["| 年份 | 期末待恢复 | T+1违规 | 底仓计划差异 | 周退出失败日/股日 | 期末未退仓（实际/基准） | 陈旧估值（元） |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    periods = []
    for row in summary.itertuples(index=False):
        overview.append(f"| {row.test_year} | 是/{yes(row.t_has_actual_trades)} | {row.evaluated_trading_days} | "
                        f"{row.baseline_total_return:.4%} | {row.total_return:.4%} | {row.t_added_equity:,.2f} | {row.t_pair_count} |")
        predicted = "未提供" if pd.isna(row.prediction_row_count) else str(int(row.prediction_row_count))
        requested = "未提供" if pd.isna(row.t_request_count) else str(int(row.t_request_count))
        counts.append(f"| {row.test_year} | {predicted}/{requested} | {row.t_first_leg_count}/{row.t_recovery_leg_count} | "
                      f"{row.t_fragment_count}/{row.t_cross_day_recovery_pair_count} | {row.t_fee:.2f}/{row.t_slippage_cost:.2f} | "
                      f"{row.inventory_mismatch_days}/{row.recovery_failure_days} |")
        failures.append(f"| {row.test_year} | {row.pending_pairs} | {row.t1_violation_count} | {row.base_plan_mismatch_count} | "
                        f"{row.weekly_liquidation_failure_days}/{row.weekly_exit_failure_stock_days} | "
                        f"{row.weekly_unclosed_positions}/{row.baseline_weekly_unclosed_positions} | {row.final_stale_market_value:.2f} |")
        periods.append(f"{row.test_year}：{row.evaluation_start} 至 {row.evaluation_end}")
    document = ["# 主动库存做 T：年度实际收益与执行记录（v9）", "",
                f"{len(summary)} 折均启用 T；其中 {status['years_with_actual_t_trades']} 折有实际 T 成交。"
                f"各年独立投入 {cfg['initial_cash']:,.0f} 元，实际净 T 增量合计 **{status['aggregate_increment_rmb']:,.2f} 元**。",
                "启用不等于保证每天产生请求或成交。样本量、集中度、置信区间及内部净收益正负不再阻止启用，"
                "统计结果仅作参考；实际成交仍必须服从费用、现金、旧库存、整手、容量和限价规则。", "", *overview, "",
                "实际区间：" + "；".join(periods) + "。各年度独立，不拼接年度净值或直接相加收益率。", "",
                "## 实际做 T 数量与费用", "", *counts, "",
                "T首腿和恢复腿的 event_type 都为 t，分别按 fragment_id=0 和 >0 区分；完整配对按 pair_id 去重，"
                "恢复片段不能重复当作独立交易。trade_lots 等通用指标可能包含周底仓交易，不能当成做 T 次数。",
                "账户净增量为实际权益减相同底仓计划基准权益，含未恢复仓位估值及底仓差异影响。"
                "片段 net_pnl 使用含滑点成交价再扣真实费用，slippage_cost 仅归因，不能再次从净利扣除。", "",
                "## 执行合约与退出风险", "", *failures, "",
                f"T执行合约全部满足：**{yes(t_contract)}**；周度退出全部完成：**{yes(weekly_complete)}**。"
                "启用后的恢复失败仍按真实仓位跨日、估值并重试，不补造恢复，不隐藏股数偏差。"
                "恢复次日相对该日开盘增加/减少股数也计偏差，不能只看期末 pending=0。陈旧估值不代表能够按该价变现。", "",
                "## 统计参考，不参与开关", "",
                f"正净增量年度 {positive_years}/{len(summary)}；按各独立年度周增量合并重抽样的95%区间 "
                f"[{low:,.2f}，{high:,.2f}] 元。旧收益统计参考条件满足：{yes(statistical_reference)}。",
                "goal_evaluation.json 和 active_reporting_status.json 的 report_only=true、affects_activation=false。"
                "保留兼容 gain_criterion_met 等字段以说明旧参考条件，但它们不改变策略、不停止交易，也不回退持有。"
                "内部原样本量、集中度、置信区间和收益正负列保留在 validation_rule_metrics.csv，不能把仅移除统计晋级条件写成已通过原验收。", "",
                "## 本轮固定协议", "",
                "复用 v8 每折核心残差模型和本折 v4 冻结周排名，周评分仍为下一周收益预测排名70%加日均振幅预测排名30%。"
                "不重新训练，不使用未来队列均值作输入；预测仍是残差加固定零共同项先验。"
                "每折从原内部候选中保留完整T配对>0且股数偏差、恢复失败、期末pending、共同底仓差异、T+1违规均为0者，"
                "按内部净增量降序、原候选编号升序选择，允许净收益为负；没有符合执行条件的主动候选时显式报错，不静默回退持有。",
                "全部四折规则和复用模型身份先冻结，再统一执行本轮外层推断和账户回测。"
                "首腿仍采用原支持域、费用后净边际、当前恢复方向过滤及每天最早请求；盘中实际股数和现金由原引擎控制。",
                f"费用：每腿佣金 {cfg['commission_rate']:.4%}、过户 {cfg['transfer_rate']:.4%}、滑点 {cfg['slippage_rate']:.4%}，"
                f"最低佣金 {cfg['min_commission']} 元；卖出印花税按原日期规则。配置及源模型目录见 resolved_config.json。", "",
                "## 研究边界", "",
                "这次主动规则是在既有研究和开关诊断后按用户要求启用，旧外层历史已经查看；"
                "本轮属于历史重新评价，不能称为全新未见测试。2023–2025折含未来年份训练，2026数据仅到4月30日。"
                "统计区间不完全处理周间相关或历次研究选择偏差。历史ST、公司行动、精确限价/盘口和证券主表限制保留。"
                "代码与账目审计通过不等于盈利，也不能替代每日股数合同。", "",
                "[年度指标](annual_summary.csv) · [收益与增量图](test_return_curve.png) · "
                "[净增量图](t_increment_curve.png) · [逐日库存](daily_inventory_audit.csv) · [受阻订单](blocked_orders.csv)"]
    (directory / "final_report.md").write_text("\n".join(document) + "\n", encoding="utf-8")
    (directory / "输出文件说明.md").write_text(
        "# 输出文件说明\n\n"
        "final_report.md：主动启用后的真实年度收益、T次数、费用及执行失败；annual_summary.csv/test_metrics.csv："
        "同一年度汇总，t_enabled与t_has_actual_trades分别表示启用和实际成交。统计列保留为参考。\n\n"
        "test_equity_curve.csv、test_trade_events.csv、test_round_trips.csv、test_open_positions.csv、"
        "baseline_trade_events.csv、daily_inventory_audit.csv、blocked_orders.csv、weekly_entry_plans.csv、"
        "base_plan_mismatches.csv按test_year合并各独立账户；weekly_increment.csv先各年内作周权益差分。\n\n"
        "test_return_curve.png/测试集收益曲线.png含持有及主动账户收益和净增量，t_increment_curve.png单列净增量。"
        "goal_evaluation.json/active_reporting_status.json只报告收益参考与执行状态，不控制启用或选参。\n\n"
        "validation_rule_metrics.csv保留原内部候选统计及本轮选择记录；year_*/selected_rule.json为本轮主动规则，"
        "t_model/是对应源核心模型副本，selected_support.json为同周期核心支持域。year_*还保存原始预测、请求、"
        "成交/库存/失败表、execution_audit.json及frozen_baseline_audit.json。源模型、数据、上下文和代码身份由入口manifest记录。\n\n"
        "T实际手续费/滑点按全部T成交腿统计；realized_pair_*仅按已恢复片段分摊统计。未完成配对的费用不能遗漏，"
        "片段净利与含遗留仓位估值的账户净增量不得混为一项。\n",
        encoding="utf-8")
