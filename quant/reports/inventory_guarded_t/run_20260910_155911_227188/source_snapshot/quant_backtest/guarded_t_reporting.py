"""v15首腿反向限价复查报告；消费已有账户，只写调用方的新输出目录。

直接复用balanced的原始成本、手数和验收口径；独立标识v15执行研究与v14
冻结模型复用。复查请求、取消、实际首腿、首次退出意图和恢复片段分别计数。
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .balanced_t_reporting import write_balanced_report
from .dynamic_t_reporting import EXIT_MODES, SCOPE_LABELS, _exit_summary, _write_json

VERSION = "inventory_guarded_t_v15"
PROFILES = ("l2_depth4", "l2_depth6")
GUARD_COLUMNS = ("datetime", "signal_datetime", "symbol", "side", "opposite_side", "requested_shares",
                 "own_fillable_shares", "guard_enabled", "opposite_checked", "cancelled", "reason",
                 "opposite_quote_reason", "reference_price", "high", "low", "amount", "volume",
                 "previous_close", "limit_up", "limit_down", "minute_amount_used_before", "cash_before",
                 "old_sellable_before")


def _selection(year: int, result: dict) -> dict:
    """保留本折原37候选中的精确规则；不把每信号所选期限虚构为单一期限。"""
    rule = result.get("selected_rule")
    if isinstance(rule, dict) and isinstance(rule.get("rule"), dict):
        rule = rule["rule"]
    if not isinstance(rule, dict) or not isinstance(rule.get("enabled"), (bool, np.bool_)):
        raise ValueError("v15结果须有明确enabled的selected_rule")
    if rule["enabled"]:
        if rule.get("profile") not in PROFILES or rule.get("exit_mode") not in EXIT_MODES:
            raise ValueError("未知v15继承模型或退出模式")
        for key, choices in (("min_net_edge", (0., .001, .003)), ("fraction", (.5, 1.))):
            if isinstance(rule.get(key), (bool, np.bool_)) or rule.get(key) not in choices:
                raise ValueError(f"{key}不属于原37候选")
    return {"test_year": int(year), "enabled": bool(rule["enabled"]),
            **{k: rule.get(k) for k in ("candidate_index", "profile", "min_net_edge", "fraction", "exit_mode")}}


def _summarize_guard(year: int, result: dict) -> tuple[dict, pd.DataFrame]:
    """核对显式开启的首腿复查表；通过复查可仍被现金等后续限制拒绝。"""
    frame = result.get("first_leg_exit_side_audit")
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any() or not set(GUARD_COLUMNS).issubset(frame):
        raise ValueError("v15须有完整first_leg_exit_side_audit，包括hold空表")
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
            raise ValueError("v15开启报告中出现未开启复查的行")
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


def write_guarded_inventory_report(directory: Path, cfg: dict, annual_results,
                                   validation_table=None) -> dict:
    """生成独立v15报告；无拟合、选参或账户重跑，不改原cfg和输入表。"""
    if cfg.get("research_version") != VERSION:
        raise ValueError("本报告器仅用于inventory_guarded_t_v15")
    if not isinstance(cfg.get("t_recheck_exit_side_at_execution"), (bool, np.bool_)) or not cfg["t_recheck_exit_side_at_execution"]:
        raise ValueError("v15须显式开启t_recheck_exit_side_at_execution")
    entries = sorted(annual_results.items() if isinstance(annual_results, dict) else annual_results, key=lambda x: int(x[0]))
    if not entries or len({int(y) for y, _ in entries}) != len(entries):
        raise ValueError("年度结果须非空且年份唯一")
    selections = [_selection(y, r) for y, r in entries]
    guards = [_summarize_guard(int(y), r) for y, r in entries]
    exits = [_exit_summary(int(y), r) for y, r in entries]
    directory = Path(directory)
    # 保留原v15配置身份；共同指标不需要把cfg伪装为其他实验。
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
    pd.DataFrame(selections).to_csv(directory/"selected_guarded_rules.csv", index=False, encoding="utf-8-sig")
    meta = {"research_version": VERSION, "report_scope": status["report_scope"],
            "model_learning_version": 14, "model_learning_research_version": "inventory_learning_t_v14",
            "frozen_models_reused": True, "models_retrained": False, "candidate_count": 37,
            "source_learning_report_dir": str(cfg.get("source_learning_report_dir", "")),
            "source_report_scope": cfg.get("source_report_scope", "not_recorded"),
            "t_recheck_exit_side_at_execution": True,
            "only_new_execution_condition": "opposite_price_limit_recheck_on_same_first_leg_arrival_quote",
            "uses_future_recovery_quotes_or_success": False,
            "guard_pass_is_actual_fill_guarantee": False, "guard_pass_is_daily_recovery_guarantee": False,
            "guard_rows_are_requests_not_completed_pairs": True,
            "guard_has_no_virtual_reverse_fill_or_inventory_consumption": True,
            "prediction_horizons_minutes": [20, 40, 60],
            "target_is_portfolio_execution_net_pnl": False,
            "goal_calculation_unchanged": True, "selected_rules": selections,
            "annual_guard_summaries": guard_summary.to_dict("records"),
            "annual_exit_summaries": exit_summary.to_dict("records")}
    _write_json(directory/"inventory_guarded_summary.json", meta)
    _write_json(directory/"guard_summary.json", {"research_version": VERSION, "annual_results": meta["annual_guard_summaries"]})
    _write_json(directory/"dynamic_exit_summary.json", {"research_version": VERSION, "annual_results": meta["annual_exit_summaries"]})
    status = {**status, "research_version": VERSION, "inventory_guarded": meta}
    _write_json(directory/"goal_evaluation.json", status)
    annual = pd.read_csv(directory/"annual_summary.csv", float_precision="round_trip")
    _plot_guarded(directory, cfg, annual, entries, status["report_scope"])
    _append_guarded_text(directory, cfg, selections, guard_summary, exit_summary, status["report_scope"])
    return status


def _plot_guarded(directory: Path, cfg: dict, annual: pd.DataFrame, entries: list, scope: str) -> None:
    """仅绘制已有规范指标，明确v15身份及工程/内部范围。"""
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
    fig.suptitle(f"永久底仓首腿反向限价复核（v15）｜{SCOPE_LABELS[scope]}", fontsize=15, y=.995)
    for i, (year, result) in enumerate(entries):
        row = annual.loc[annual.test_year.eq(int(year))].iloc[0]
        metric = lambda name, fmt='.2f': "不可评价" if pd.isna(row.get(name, np.nan)) else format(row[name], fmt)
        curve = result["equity_curve"]
        time = pd.to_datetime(curve.datetime, format="mixed")
        left, right = axes[i]
        left.plot(time, (curve.baseline_equity/cfg['initial_cash']-1)*100, label="相同永久底仓")
        left.plot(time, (curve.equity/cfg['initial_cash']-1)*100, '--', label="底仓＋v15做T")
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
    fig.text(.5, .012, "各年独立账户；2026截至4月30日。复用v14冻结模型，无重训；复查通过不保证恢复或利润。",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95), h_pad=3.)
    fig.savefig(directory/"test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory/"test_return_curve.png", directory/"测试集收益曲线.png")


def _append_guarded_text(directory: Path, cfg: dict, selections: list, guards: pd.DataFrame,
                         exits: pd.DataFrame, scope: str) -> None:
    """追加新执行条件和取消请求口径，不暗示取消即避免等量亏损。"""
    path = directory/"final_report.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(text.splitlines()[0], f"# 永久底仓首腿反向限价复核（v15）：{SCOPE_LABELS[scope]}", 1)
    lines = ["", "## 冻结模型与唯一新执行条件", "",
             "模型学习版本14原样复用，无重新训练；核心模型仍学习固定20/40/60分钟端点参考价收益，"
             "不是实际共享账户利润。2个profile×3边际下限×2份额×3退出＋hold，仍为原37候选。",
             f"模型来源：`{cfg.get('source_learning_report_dir','未记录')}`；来源范围：`{cfg.get('source_report_scope','not_recorded')}`。", "",
             "唯一新增条件t_recheck_exit_side_at_execution=true：原首腿本方向在到达分钟已有可填股数时，"
             "用同一quote再次检查反方向限价；命中则整笔取消。这是到达报价的成交约束，不是上一信号分钟已知的新因子。"
             "不读未来恢复行情，不试算反向成交容量，不消耗虚拟现金或旧股。通过后仍受现金、整手和未恢复额度约束，"
             "后续封板仍可能恢复失败。取消请求不是实际T组，也不能把被取消金额当作避免的亏损。", "",
             "原万八免5、过户费、日期印花税、双腿滑点、永久底仓、T+1与动态恢复均保留。"
             "利润缓冲/止损观察当前已完成分钟，下一分钟执行并锁存分片，最晚仍按原计划期限。"
             "当前成交价已含滑点；参考毛差减滑点和费用的口径不重复扣费。", "",
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
                     "- selected_guarded_rules.csv、inventory_guarded_summary.json：v15身份、原模型版本/来源范围与精确规则。\n"
                     "- exit_decisions.csv、dynamic_exit_summary.csv/json：每组一次首次锁存，恢复片段不能重复计组。\n")
