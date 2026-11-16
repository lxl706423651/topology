"""v16固定14:55目标报告；只消费已有账户并向调用方新目录写派生结果。

新拟合两个close回归模型，复用v14完整无标签核心队列及v15执行保护。
共同成本、手数、库存和目标口径沿用balanced；不伪装旧版本、不运行模型。
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .balanced_t_reporting import write_balanced_report
from .dynamic_t_reporting import EXIT_MODES, SCOPE_LABELS, _exit_summary, _write_json

VERSION = "inventory_close_t_v16"
PROFILES = ("l2_depth4", "l2_depth6")
GUARD_COLUMNS = ("datetime", "signal_datetime", "symbol", "side", "opposite_side", "requested_shares",
                 "own_fillable_shares", "guard_enabled", "opposite_checked", "cancelled", "reason",
                 "opposite_quote_reason", "reference_price", "high", "low", "amount", "volume",
                 "previous_close", "limit_up", "limit_down", "minute_amount_used_before", "cash_before",
                 "old_sellable_before")


def _selection(year: int, result: dict) -> dict:
    """保留本折37候选中的精确规则，不混入旧整数预测期限身份。"""
    rule = result.get("selected_rule")
    if isinstance(rule, dict) and isinstance(rule.get("rule"), dict):
        rule = rule["rule"]
    if not isinstance(rule, dict) or not isinstance(rule.get("enabled"), (bool, np.bool_)):
        raise ValueError("v16结果须有明确enabled的selected_rule")
    if "horizon_minutes" in rule or "horizons" in rule:
        raise ValueError("固定尾盘规则不能混入旧整数期限")
    if rule["enabled"]:
        if rule.get("profile") not in PROFILES or rule.get("exit_mode") not in EXIT_MODES:
            raise ValueError("未知v16新拟合close模型或退出模式")
        for key, choices in (("min_net_edge", (0., .001, .003)), ("fraction", (.5, 1.))):
            if isinstance(rule.get(key), (bool, np.bool_)) or rule.get(key) not in choices:
                raise ValueError(f"{key}不属于原37候选")
    return {"test_year": int(year), "enabled": bool(rule["enabled"]),
            **{k: rule.get(k) for k in ("candidate_index", "profile", "min_net_edge", "fraction", "exit_mode")}}


def _summarize_guard(year: int, result: dict) -> tuple[dict, pd.DataFrame]:
    """核对显式开启的首腿复查表；通过复查可仍被现金等后续限制拒绝。"""
    frame = result.get("first_leg_exit_side_audit")
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any() or not set(GUARD_COLUMNS).issubset(frame):
        raise ValueError("v16须有完整first_leg_exit_side_audit，包括hold空表")
    frame = frame.copy(deep=True)
    dates = pd.to_datetime(frame.datetime, format="mixed", errors="raise")
    signals = pd.to_datetime(frame.signal_datetime, format="mixed", errors="raise")
    for col in ("guard_enabled", "opposite_checked", "cancelled"):
        if not frame[col].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"{col}须为布尔值，不能把字符串False当True")
    if len(frame):
        if dates.isna().any() or signals.isna().any() or dates.dt.tz is not None or signals.dt.tz is not None or not dates.gt(signals).all():
            raise ValueError("首腿到达时刻须晚于信号且为市场本地无时区时间")
        if not frame.guard_enabled.all() or not frame.opposite_checked.all():
            raise ValueError("v16开启报告中出现未开启复查的行")
        if not frame.side.isin(["buy", "sell"]).all() or not frame.opposite_side.eq(frame.side.map({"buy": "sell", "sell": "buy"})).all():
            raise ValueError("反向复查方向错误")
        own = pd.to_numeric(frame.own_fillable_shares, errors="raise")
        requested = pd.to_numeric(frame.requested_shares, errors="raise")
        if not (np.isfinite(own) & own.gt(0) & np.isfinite(requested) & requested.ge(own)).all():
            raise ValueError("复查只记录本方向已有正可填股数的请求")
        opposite = frame.opposite_quote_reason.fillna("")
        expected_cancel = opposite.isin(["price_limit", "conservative_price_limit_proxy"])
        if not opposite.isin(["", "price_limit", "conservative_price_limit_proxy"]).all() or not frame.cancelled.eq(expected_cancel).all():
            raise ValueError("复查取消标志与反向限价原因不一致")
        expected_reason = opposite.map(lambda x: "execution_exit_side_"+x if x else "")
        if not frame.reason.fillna("").eq(expected_reason).all():
            raise ValueError("首腿取消原因不一致")
    keys = pd.DataFrame({"datetime": dates, "symbol": frame.symbol, "side": frame.side})
    if keys.duplicated().any():
        raise ValueError("首腿复查请求重复")
    events = result["trade_events"]
    first = events.loc[events.event_type.eq("t") & events.fragment_id.eq(0)]
    actual_keys = set(zip(pd.to_datetime(first.datetime, format="mixed"), first.symbol, first.side))
    passed_keys = set(map(tuple, keys.loc[~frame.cancelled].to_numpy()))
    if not actual_keys.issubset(passed_keys):
        raise ValueError("实际首腿缺通过复查记录，或已取消请求仍发生首腿成交")
    cancelled = frame.cancelled
    stock_days = pd.DataFrame({"date": dates.dt.normalize(), "symbol": frame.symbol})
    summary = {"test_year": int(year), "guard_audit_requests": len(frame),
               "guard_checked_requests": int(frame.opposite_checked.sum()),
               "guard_passed_requests": int((~cancelled).sum()), "guard_cancelled_requests": int(cancelled.sum()),
               "guard_cancelled_days": int(dates.loc[cancelled].dt.normalize().nunique()),
               "guard_cancelled_stock_days": len(stock_days.loc[cancelled].drop_duplicates()),
               "guard_cancelled_own_fillable_shares": float(frame.loc[cancelled, "own_fillable_shares"].sum()),
               "guard_actual_first_leg_groups": len(first),
               "guard_passed_without_actual_first_leg": len(passed_keys.difference(actual_keys)),
               "guard_cancelled_price_limit": int((cancelled & frame.opposite_quote_reason.eq("price_limit")).sum()),
               "guard_cancelled_limit_proxy": int((cancelled & frame.opposite_quote_reason.eq("conservative_price_limit_proxy")).sum())}
    return summary, frame


def write_close_inventory_report(directory: Path, cfg: dict, annual_results,
                                   validation_table=None) -> dict:
    """生成v16规范报告；两个新模型的结果与真实账户分列，不在报告阶段拟合。"""
    if cfg.get("research_version") != VERSION:
        raise ValueError("本报告器仅用于inventory_close_t_v16")
    if (cfg.get("forecast_target") != "same_day_1455" or cfg.get("recovery_key") != "close"
            or cfg.get("fixed_recovery_time") != "14:55" or "horizons" in cfg
            or "t_label_horizon_minutes" in cfg):
        raise ValueError("v16须明确单一同日14:55固定时钟目标，不能沿用整数期限")
    if cfg.get("frozen_t_models_reused", False) is not False:
        raise ValueError("v16新拟合close模型，不能声明复用旧T模型")
    for key in ("source_execution_report_dir", "source_learning_report_dir"):
        if not isinstance(cfg.get(key), (str, Path)) or not str(cfg[key]).strip():
            raise ValueError(f"v16缺少明确来源路径：{key}")
    if not isinstance(cfg.get("t_recheck_exit_side_at_execution"), (bool, np.bool_)) or not cfg["t_recheck_exit_side_at_execution"]:
        raise ValueError("v16须显式开启t_recheck_exit_side_at_execution")
    entries = sorted(annual_results.items() if isinstance(annual_results, dict) else annual_results, key=lambda x: int(x[0]))
    if not entries or len({int(y) for y, _ in entries}) != len(entries):
        raise ValueError("年度结果须非空且年份唯一")
    selections = [_selection(y, r) for y, r in entries]
    guards = [_summarize_guard(int(y), r) for y, r in entries]
    exits = [_exit_summary(int(y), r) for y, r in entries]
    for selected, (_, result), (_, frame) in zip(selections, entries, exits):
        first = result["trade_events"].loc[lambda x: x.event_type.eq("t") & x.fragment_id.eq(0)]
        if not selected["enabled"] and len(first):
            raise ValueError("hold规则不能含实际T首腿")
        if len(frame):
            first_time = pd.to_datetime(frame.first_execution_time, format="mixed")
            expected = first_time.dt.normalize() + pd.Timedelta(hours=14, minutes=55)
            deadline = pd.to_datetime(frame.original_deadline, format="mixed")
            if not deadline.eq(expected).all():
                raise ValueError("v16实际pair原计划期限必须为首腿同日14:55")
    directory = Path(directory)
    # 保留原v16配置身份；共同指标不需要把cfg伪装为其他实验。
    status = write_balanced_report(directory, cfg, entries, validation_table)
    for filename, items in (("first_leg_exit_side_audit.csv", guards), ("exit_decisions.csv", exits)):
        all_rows = []
        for (year, _), (_, frame) in zip(entries, items):
            path = directory / f"year_{int(year)}" / filename
            if not path.exists():
                frame.to_csv(path, index=False, encoding="utf-8-sig")
            all_rows.append(frame.assign(test_year=int(year)))
        pd.concat(all_rows, ignore_index=True).to_csv(directory/filename, index=False, encoding="utf-8-sig")
    guard_summary = pd.DataFrame([s for s, _ in guards])
    exit_summary = pd.DataFrame([s for s, _ in exits])
    guard_summary.to_csv(directory/"guard_summary.csv", index=False, encoding="utf-8-sig")
    exit_summary.to_csv(directory/"dynamic_exit_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(directory/"selected_close_rules.csv", index=False, encoding="utf-8-sig")
    meta = {"research_version": VERSION, "report_scope": status["report_scope"],
            "model_learning_version": 16, "model_learning_research_version": VERSION,
            "frozen_t_models_reused": False, "frozen_models_reused": False,
            "models_retrained": True, "newly_fitted_close_models_per_fold": 2,
            "newly_fitted_model_profiles": list(PROFILES), "candidate_count": 37,
            "source_execution_report_dir": str(cfg["source_execution_report_dir"]),
            "source_learning_report_dir": str(cfg["source_learning_report_dir"]),
            "training_cohort_reused_from_version": 14, "execution_engine_version": 15,
            "source_report_scope": cfg.get("source_report_scope", "not_recorded"),
            "t_recheck_exit_side_at_execution": True,
            "execution_condition_inherited": "opposite_price_limit_recheck_on_same_first_leg_arrival_quote",
            "uses_future_recovery_quotes_or_success_for_current_eligibility": False,
            "guard_pass_is_actual_fill_guarantee": False, "guard_pass_is_daily_recovery_guarantee": False,
            "guard_rows_are_requests_not_completed_pairs": True,
            "guard_has_no_virtual_reverse_fill_or_inventory_consumption": True,
            "forecast_target": "same_day_1455", "recovery_key": "close",
            "recovery_kind": "fixed_clock", "recovery_clock": "14:55",
            "reference_target": "same_day_1455_reference_price / next_minute_reference_price - 1",
            "cross_horizon_argmax": False, "holding_duration_is_constant_minutes": False,
            "target_is_portfolio_execution_net_pnl": False,
            "actual_exit_may_precede_fixed_clock": True,
            "comparison_changes_prediction_target_and_fitted_weights": True,
            "goal_calculation_unchanged": True, "selected_rules": selections,
            "annual_guard_summaries": guard_summary.to_dict("records"),
            "annual_exit_summaries": exit_summary.to_dict("records")}
    _write_json(directory/"inventory_close_summary.json", meta)
    _write_json(directory/"guard_summary.json", {"research_version": VERSION, "annual_results": meta["annual_guard_summaries"]})
    _write_json(directory/"dynamic_exit_summary.json", {"research_version": VERSION, "annual_results": meta["annual_exit_summaries"]})
    status = {**status, "research_version": VERSION, "inventory_close": meta}
    _write_json(directory/"goal_evaluation.json", status)
    annual = pd.read_csv(directory/"annual_summary.csv", float_precision="round_trip")
    _plot_close(directory, cfg, annual, entries, status["report_scope"])
    _append_close_text(directory, cfg, selections, guard_summary, exit_summary, status["report_scope"])
    return status


def _plot_close(directory: Path, cfg: dict, annual: pd.DataFrame, entries: list, scope: str) -> None:
    """仅绘制已有规范指标，明确v16身份及工程/内部范围。"""
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
    fig, axes = plt.subplots(len(entries), 2, figsize=(15, 5.4*len(entries)), squeeze=False)
    fig.suptitle(f"永久底仓固定14:55目标做T（v16）｜{SCOPE_LABELS[scope]}", fontsize=15, y=.995)
    for i, (year, result) in enumerate(entries):
        row = annual.loc[annual.test_year.eq(int(year))].iloc[0]
        metric = lambda name, fmt='.2f': "不可评价" if pd.isna(row.get(name, np.nan)) else format(row[name], fmt)
        curve = result["equity_curve"]
        time = pd.to_datetime(curve.datetime, format="mixed")
        left, right = axes[i]
        left.plot(time, (curve.baseline_equity/cfg['initial_cash']-1)*100, label="相同永久底仓")
        left.plot(time, (curve.equity/cfg['initial_cash']-1)*100, '--', label="底仓＋v16做T")
        left.set_title(f"{year}｜{time.iloc[0]:%Y-%m-%d}—{time.iloc[-1]:%Y-%m-%d}")
        left.set_ylabel("原始账户本金收益（%）")
        left.legend()
        right.plot(time, curve.equity-curve.baseline_equity, color='#177f75')
        right.axhline(0, color='grey', lw=.7)
        right.set_title(f"真实净增量 {row.incremental_marked_pnl:,.2f}元｜股数差 {row.inventory_mismatch_days}日")
        right.set_ylabel("做T净增量（元）")
        notes = [f"总收益 {metric('total_return','.2%')}｜基准 {metric('baseline_total_return','.2%')}\n"
                 f"年化估计 {metric('annual_return','.2%')}｜波动 {metric('annual_volatility','.2%')}\n"
                 f"Sharpe {metric('sharpe')}｜最大回撤 {metric('max_drawdown','.2%')}",
                 f"完整组 {int(row.complete_t_pairs)}｜配对手 {row.completed_t_pair_lots:g}｜净胜率 "
                 + (metric('win_rate','.2%') if row.complete_t_pairs else "不可评价（无完整T组）") + "\n"
                 f"成本降低 {metric('holding_cost_reduction_rate','.2%')}｜期末持仓 {metric('final_open_position_rows','.0f')}笔\n"
                 f"底仓 {metric('reference_stock_count','.0f')}只｜平均持仓市值 {metric('mean_holdings_market_value',',.0f')}元"]
        short = (time.iloc[-1]-time.iloc[0]).days <= 90 and time.iloc[0].year == time.iloc[-1].year
        for ax, note in zip((left, right), notes):
            ax.text(0, -.22, note, transform=ax.transAxes, fontsize=9, va='top',
                    bbox={'boxstyle': 'round,pad=.4', 'facecolor': '#f2f5f7', 'edgecolor': '#d7dfe5'})
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d' if short else '%Y-%m'))
            ax.grid(alpha=.25)
    fig.text(.5, .012, "各年独立账户；2026截至4月30日。每折新拟合2个14:55目标模型；沿用v15成交保护，参考标签不等于实际利润。",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95), h_pad=3.)
    fig.savefig(directory/"test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory/"test_return_curve.png", directory/"测试集收益曲线.png")


def _append_close_text(directory: Path, cfg: dict, selections: list, guards: pd.DataFrame,
                         exits: pd.DataFrame, scope: str) -> None:
    """追加新模型/目标身份与原成交保护，明确参考收益和实际恢复的区别。"""
    path = directory/"final_report.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(text.splitlines()[0], f"# 永久底仓固定14:55目标做T（v16）：{SCOPE_LABELS[scope]}", 1)
    lines = ["", "## 同日14:55新目标与沿用的成交保护", "",
             "每折使用永久底仓的完整未联标签核心队列，新拟合l2_depth4/l2_depth6两个回归模型；"
             "frozen_t_models_reused=false，未复用v14旧T模型权重。两core年仍联合拟合，"
             "不能把互相依赖的OOF核心年再称为独立校准。37候选保持2个profile×3边际下限×2份额×3退出＋hold。", "",
             "模型标签为下一合法分钟参考价到同日14:55参考价的价格比收益，身份是"
             "recovery_key=close、recovery_kind=fixed_clock、recovery_clock=14:55；不是伪装成整数horizon的固定长度期限。"
             "各信号距14:55的交易分钟数不同。每个profile只产生单一close预测，"
             "不再在20/40/60三个期限间逐行取最大预测，但仍保留方向选择与内部候选选择。", "",
             f"完整训练队列来源（v14）：`{cfg['source_learning_report_dir']}`；"
             f"执行条件来源（v15）：`{cfg['source_execution_report_dir']}`。", "",
             "机会先由日初旧股持仓与已知来源冻结，再左联未来端点；缺参考价保留为NaN，"
             "仅后续核心训练purge决定有效标签。零量仍可有参考价格标签，不代表该端点可成交。"
             "回归目标未扣真实账户费用，推断后按原费用换算期望净边际；实际组合盈亏继续由共享资金、"
             "容量、整手、分片恢复与真实每腿费用计算。", "",
             "沿用v15的t_recheck_exit_side_at_execution=true：原首腿本方向已有正可填股数时，"
             "用同一到达quote检查反方向限价，命中则整笔取消；不读未来恢复行情，不虚拟反向成交或消耗现金/旧股。"
             "通过复查仍可能被现金、整手或待恢复额度拒绝，后续也可能封板或缺量而恢复失败。"
             "取消请求不是实际T组，更不是等额避免亏损。", "",
             "原万八免5、过户费、日期印花税、双腿滑点、永久底仓及T+1旧股预算均保留。"
             "fixed模式原期限为同日14:55；利润缓冲/止损模式仍可观察已完成分钟并在下一交易分钟提前恢复，"
             "意图锁存，受阻继续分片，义务可能跨日。14:55是预测与计划期限，不保证每组恰在14:55全部成交。"
             "参考毛差减滑点和费用，不重复扣除已含在成交价格中的滑点。", "",
             "与v15的差异包含新参考目标、新拟合权重以及取消跨期限逐行argmax；"
             "不能把账户差额单独归因于持有更久，也不能预先断言较长窗口更赚钱。", "",
             "| 年度 | 开启 | profile | 最小净边际 | 份额 | 退出 |",
             "|---|---|---|---:|---:|---|"]
    for r in selections:
        lines.append('| '+ ' | '.join(str(r[k]) for k in ('test_year','enabled','profile','min_net_edge','fraction','exit_mode'))+' |')
    lines += ["", "| 年度 | 复查请求 | 取消请求 | 取消股票日 | 通过但未成交请求 | 实际首腿组 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in guards.itertuples(index=False):
        lines.append(f"| {r.test_year} | {r.guard_audit_requests} | {r.guard_cancelled_requests} | {r.guard_cancelled_stock_days} | "
                     f"{r.guard_passed_without_actual_first_leg} | {r.guard_actual_first_leg_groups} |")
    lines += ["", "| 年度 | 首次锁存 | 提前触发 | 利润/止损/原期限 |",
              "|---|---:|---:|---:|"]
    for r in exits.itertuples(index=False):
        lines.append(f"| {r.test_year} | {r.latched_exit_pairs} | {r.early_trigger_pairs} | "
                     f"{r.profit_buffer_trigger_pairs}/{r.stop_loss_trigger_pairs}/{r.hard_deadline_trigger_pairs} |")
    lines += ["", "hold或无完整组时没有可评价的做T胜率。5%按全部原始含费底仓成本逐内部年验收；"
              "各内部折达到原条件后才统一正式外评，烟测不得当收益证据。库存/恢复/pending/T+1/共同底仓差如实保留，"
              "正常永久底仓期末保留不等于未恢复T；频率未量化不暗判足够。", "",
              "2023–2025留出折为可含未来训练年的回顾研究；2026时间前推但只截至4月30日，不年化跨过5%。"
              "这些历史已反复研究，独立年度账户不拼为跨年复利，也不称全新测试。", ""]
    path.write_text(text+'\n'.join(lines), encoding="utf-8")
    with (directory/"输出文件说明.md").open('a', encoding='utf-8') as stream:
        stream.write("\n- first_leg_exit_side_audit.csv：本方向已有可填股数请求的到达报价复查，包含取消与通过行；非成交组。\n"
                     "- guard_summary.csv/json：复查、取消、股票日和实际首腿分列。\n"
                     "- selected_close_rules.csv、inventory_close_summary.json：v16身份、新拟合close模型、完整队列/执行来源与精确规则。\n"
                     "- exit_decisions.csv、dynamic_exit_summary.csv/json：每组一次首次锁存，恢复片段不能重复计组。\n")
