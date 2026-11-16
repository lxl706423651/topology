"""v17固定1% Huber短期限报告；只消费已有账户并写调用方的新目录。

直接保留balanced成本/手数/验收口径，复用纯退出及guard统计，不伪装旧版本。
每折新拟合六模型，原始参考标签保持不变；预测评分不等于账户可成交净收益。
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .balanced_t_reporting import write_balanced_report
from .dynamic_t_reporting import EXIT_MODES, SCOPE_LABELS, _exit_summary, _write_json
from .guarded_t_reporting import _summarize_guard

VERSION = "inventory_robust_t_v17"
PROFILES = ("huber_depth4", "huber_depth6")
SOURCE_PROFILES = {"huber_depth4": "l2_depth4", "huber_depth6": "l2_depth6"}
HORIZONS = (20, 40, 60)
SESSION_MINUTES = tuple(range(571, 691)) + tuple(range(781, 901))


def _selection(year: int, result: dict) -> dict:
    """核对37候选的数值参数与新profile，保留逐信号多期限选择的语义。"""
    rule = result.get("selected_rule")
    if isinstance(rule, dict) and isinstance(rule.get("rule"), dict):
        rule = rule["rule"]
    if not isinstance(rule, dict) or not isinstance(rule.get("enabled"), (bool, np.bool_)):
        raise ValueError("v17结果须有明确enabled的selected_rule")
    if rule["enabled"]:
        if rule.get("profile") not in PROFILES or rule.get("exit_mode") not in EXIT_MODES:
            raise ValueError("未知v17新拟合Huber模型或退出模式")
        for key, choices in (("min_net_edge", (0., .001, .003)), ("fraction", (.5, 1.))):
            if isinstance(rule.get(key), (bool, np.bool_)) or rule.get(key) not in choices:
                raise ValueError(f"{key}不属于原37候选")
    return {"test_year": int(year), "enabled": bool(rule["enabled"]),
            **{k: rule.get(k) for k in ("candidate_index", "profile", "min_net_edge", "fraction", "exit_mode")},
            "source_l2_profile": SOURCE_PROFILES.get(rule.get("profile"))}


def _inherited_summary(kind: str, year: int, result: dict) -> tuple[dict, pd.DataFrame]:
    """只复用旧纯统计函数；错误加v17上下文，不调用旧版本writer或改输入cfg。"""
    try:
        return (_summarize_guard if kind == "guard" else _exit_summary)(year, result)
    except ValueError as error:
        detail = str(error).replace("v15", "v17继承的guard格式").replace("v13", "v17继承的动态退出格式")
        raise ValueError(f"v17 {kind}审计：{detail}") from error


def _horizon_summary(year: int, frame: pd.DataFrame) -> dict:
    """逐已锁存pair核对原计划期限：真实首腿后20/40/60交易分钟，跳过午休。"""
    counts = {h: 0 for h in HORIZONS}
    if len(frame):
        first = pd.to_datetime(frame.first_execution_time, format="mixed")
        deadline = pd.to_datetime(frame.original_deadline, format="mixed")
        positions = {minute: i for i, minute in enumerate(SESSION_MINUTES)}
        start_index = (first.dt.hour*60+first.dt.minute).map(positions)
        end_index = (deadline.dt.hour*60+deadline.dt.minute).map(positions)
        delta = end_index-start_index
        if (not first.dt.normalize().eq(deadline.dt.normalize()).all() or start_index.isna().any()
                or end_index.isna().any() or not first.eq(first.dt.floor("min")).all()
                or not deadline.eq(deadline.dt.floor("min")).all() or not delta.isin(HORIZONS).all()
                or not deadline.le(deadline.dt.normalize()+pd.Timedelta(hours=14, minutes=55)).all()):
            raise ValueError("v17原计划须为真实首腿后20/40/60交易分钟且同日不晚于14:55")
        counts = {h: int(delta.eq(h).sum()) for h in HORIZONS}
    return {"test_year": int(year), "latched_pairs_with_checked_deadline": len(frame),
            **{f"planned_h{h}_pairs": counts[h] for h in HORIZONS}}


def write_robust_inventory_report(directory: Path, cfg: dict, annual_results,
                                  validation_table=None) -> dict:
    """生成v17规范报告；保留原cfg身份、原账户数值和原5%含费底仓成本分母。"""
    if cfg.get("research_version") != VERSION:
        raise ValueError("本报告器仅用于inventory_robust_t_v17")
    if (cfg.get("training_loss") != "huber" or isinstance(cfg.get("t_huber_alpha"), (bool, np.bool_))
            or cfg.get("t_huber_alpha") != .01):
        raise ValueError("v17须明确Huber目标和固定t_huber_alpha=0.01")
    if (cfg.get("forecast_target") != "fixed_20_40_60_reference_return"
            or cfg.get("horizons") != list(HORIZONS) or cfg.get("recovery_key") == "close"):
        raise ValueError("v17须保留20/40/60参考端点，不能误用固定尾盘close目标")
    if cfg.get("frozen_t_models_reused", False) is not False:
        raise ValueError("v17新拟合Huber模型，不能声明复用旧T模型")
    for key in ("source_execution_report_dir", "source_learning_report_dir"):
        if not isinstance(cfg.get(key), (str, Path)) or not str(cfg[key]).strip():
            raise ValueError(f"v17缺少明确来源路径：{key}")
    if not isinstance(cfg.get("t_recheck_exit_side_at_execution"), (bool, np.bool_)) or not cfg["t_recheck_exit_side_at_execution"]:
        raise ValueError("v17须显式开启t_recheck_exit_side_at_execution")
    entries = sorted(annual_results.items() if isinstance(annual_results, dict) else annual_results, key=lambda x: int(x[0]))
    if not entries or len({int(y) for y, _ in entries}) != len(entries):
        raise ValueError("年度结果须非空且年份唯一")
    selections = [_selection(y, r) for y, r in entries]
    guards = [_inherited_summary("guard", int(y), r) for y, r in entries]
    exits = [_inherited_summary("exit", int(y), r) for y, r in entries]
    horizons = [_horizon_summary(int(y), item[1]) for (y, _), item in zip(entries, exits)]
    for selected, (_, result) in zip(selections, entries):
        first = result["trade_events"].loc[lambda x: x.event_type.eq("t") & x.fragment_id.eq(0)]
        if not selected["enabled"] and len(first):
            raise ValueError("v17 hold规则不能含实际T首腿")
    directory = Path(directory)
    # 原v17 cfg直接进入共同核算；不调用v15/v16 writer，不宣称旧权重已复用。
    status = write_balanced_report(directory, cfg, entries, validation_table)
    for filename, items in (("first_leg_exit_side_audit.csv", guards), ("exit_decisions.csv", exits)):
        all_rows = []
        for (year, _), (_, frame) in zip(entries, items):
            path = directory/f"year_{int(year)}"/filename
            if not path.exists():
                frame.to_csv(path, index=False, encoding="utf-8-sig")
            all_rows.append(frame.assign(test_year=int(year)))
        pd.concat(all_rows, ignore_index=True).to_csv(directory/filename, index=False, encoding="utf-8-sig")
    guard_summary = pd.DataFrame([s for s, _ in guards])
    exit_summary = pd.DataFrame([s for s, _ in exits])
    horizon_summary = pd.DataFrame(horizons)
    guard_summary.to_csv(directory/"guard_summary.csv", index=False, encoding="utf-8-sig")
    exit_summary.to_csv(directory/"dynamic_exit_summary.csv", index=False, encoding="utf-8-sig")
    horizon_summary.to_csv(directory/"planned_horizon_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(directory/"selected_robust_rules.csv", index=False, encoding="utf-8-sig")
    meta = {"research_version": VERSION, "report_scope": status["report_scope"],
            "model_learning_version": 17, "model_learning_research_version": VERSION,
            "frozen_t_models_reused": False, "frozen_models_reused": False, "models_retrained": True,
            "newly_fitted_models_per_fold": 6, "newly_fitted_model_profiles": list(PROFILES),
            "profile_source_mapping": SOURCE_PROFILES, "prediction_horizons_minutes": list(HORIZONS),
            "training_loss": "huber", "objective": "huber", "t_huber_alpha": .01,
            "alpha_units": "unscaled raw return residual", "candidate_count": 37,
            "source_execution_report_dir": str(cfg["source_execution_report_dir"]),
            "source_learning_report_dir": str(cfg["source_learning_report_dir"]),
            "training_cohort_reused_from_version": 14, "execution_engine_version": 15,
            "source_report_scope": cfg.get("source_report_scope", "not_recorded"),
            "declared_model_change": "L2 to Huber with fixed alpha=0.01; fitted weights change",
            "source_training_identity_contract": "same actual features, fitted rows, labels and nonloss parameters; runner/auditor verifies hashes",
            "training_labels_winsorized_or_deleted_for_robust_loss": False,
            "prediction_is_conditional_mean_guaranteed": False,
            "predicted_edge_is_expected_portfolio_net_pnl": False,
            "predicted_edge_interpretation": "cost_adjusted_reference_score",
            "huber_implies_lower_predictions_or_profit": False,
            "forecast_target": "fixed_20_40_60_reference_return",
            "reference_target": "planned_horizon_reference_price / next_minute_reference_price - 1",
            "cross_horizon_argmax": True, "uniform_1455_forecast_target": False,
            "target_is_portfolio_execution_net_pnl": False,
            "actual_exit_may_precede_planned_deadline": True,
            "t_recheck_exit_side_at_execution": True,
            "execution_condition_inherited": "opposite_price_limit_recheck_on_same_first_leg_arrival_quote",
            "uses_future_recovery_quotes_or_success_for_current_eligibility": False,
            "guard_pass_is_actual_fill_guarantee": False, "guard_pass_is_daily_recovery_guarantee": False,
            "guard_rows_are_requests_not_completed_pairs": True,
            "guard_has_no_virtual_reverse_fill_or_inventory_consumption": True,
            "goal_calculation_unchanged": True, "selected_rules": selections,
            "annual_guard_summaries": guard_summary.to_dict("records"),
            "annual_exit_summaries": exit_summary.to_dict("records"),
            "annual_planned_horizon_summaries": horizons}
    _write_json(directory/"inventory_robust_summary.json", meta)
    _write_json(directory/"guard_summary.json", {"research_version": VERSION, "annual_results": meta["annual_guard_summaries"]})
    _write_json(directory/"dynamic_exit_summary.json", {"research_version": VERSION, "annual_results": meta["annual_exit_summaries"]})
    status = {**status, "research_version": VERSION, "inventory_robust": meta}
    _write_json(directory/"goal_evaluation.json", status)
    annual = pd.read_csv(directory/"annual_summary.csv", float_precision="round_trip")
    _plot_robust(directory, cfg, annual, entries, status["report_scope"])
    _append_robust_text(directory, cfg, selections, guard_summary, exit_summary, status["report_scope"])
    return status


def _plot_robust(directory: Path, cfg: dict, annual: pd.DataFrame, entries: list, scope: str) -> None:
    """仅绘制已有规范指标，用v17/三期限/Huber身份避免与close目标混淆。"""
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
    fig.suptitle(f"永久底仓Huber短期限做T（v17，alpha=0.01）｜{SCOPE_LABELS[scope]}", fontsize=15, y=.995)
    for i, (year, result) in enumerate(entries):
        row = annual.loc[annual.test_year.eq(int(year))].iloc[0]
        metric = lambda name, fmt='.2f': "不可评价" if pd.isna(row.get(name, np.nan)) else format(row[name], fmt)
        curve = result["equity_curve"]
        time = pd.to_datetime(curve.datetime, format="mixed")
        left, right = axes[i]
        left.plot(time, (curve.baseline_equity/cfg['initial_cash']-1)*100, label="相同永久底仓")
        left.plot(time, (curve.equity/cfg['initial_cash']-1)*100, '--', label="底仓＋v17做T")
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
    fig.text(.5, .012, "各年独立账户；2026截至4月30日。每折新拟合6个20/40/60分钟Huber模型；预测参考评分不等于真实账户净收益。",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95), h_pad=3.)
    fig.savefig(directory/"test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory/"test_return_curve.png", directory/"测试集收益曲线.png")


def _append_robust_text(directory: Path, cfg: dict, selections: list, guards: pd.DataFrame,
                        exits: pd.DataFrame, scope: str) -> None:
    """追加唯一声明的损失变更和证据边界，不把参考评分写作可成交利润。"""
    path = directory/"final_report.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(text.splitlines()[0], f"# 永久底仓Huber短期限做T（v17）：{SCOPE_LABELS[scope]}", 1)
    lines = ["", "## 固定1% Huber目标与原执行合同", "",
             "每折新拟合6个模型：20/40/60交易分钟×huber_depth4/huber_depth6。"
             "与来源L2同深度profile对应，唯一声明的拟合机制变化为objective=huber、alpha=0.01；"
             "frozen_t_models_reused=false，新权重没有伪装成旧模型复用。"
             "2个profile×3边际下限×2份额×3退出＋hold，仍为原37个数值参数候选。", "",
             f"完整核心队列来源（v14）：`{cfg['source_learning_report_dir']}`；"
             f"执行条件来源（v15）：`{cfg['source_execution_report_dir']}`。"
             "运行入口及独审须逐项核对原实际矩阵、标签、行序、dtype和非损失参数，报告不代替这些身份检查。"
             "两core年仍联合拟合，不能把互相依赖的OOF核心年再次称为独立校准。", "",
             "Huber在原始收益残差绝对值超过1%时限制梯度影响；不缩尾、不删除真实标签，"
             "不保证条件均值、预测更小或盈利。模型仍使用下一合法分钟到20/40/60分钟端点的原始参考价格收益；"
             "每个信号仍可按三个期限的成本调整参考评分选择方向和期限。predicted_edge只是参考评分，"
             "不能当作期望账户净利润，也不能把固定端点标称净额当实际账户净额。", "",
             "每组original_deadline按真实首腿成交时刻前推20/40/60交易分钟，跳过午休；"
             "14:55只保留原日末恢复底线，并非统一预测目标。利润/止损观察当前完成分钟，"
             "下一交易分钟起锁存并按真实容量分片恢复；受阻会晚于计划或跨日，不能虚构期限价成交。", "",
             "沿用v15的t_recheck_exit_side_at_execution=true：本方向报价已有正可填股数后，"
             "用同一到达quote复查反方向限价；不读未来恢复行情、不虚拟消耗现金、容量或旧股。"
             "通过复查仍可能被资金或待恢复额度拒绝，后续封板或缺量仍会恢复失败；取消请求不是实际T组或避免亏损金额。", "",
             "原万八免5、过户费、日期印花税、双腿滑点、永久底仓和T+1旧股预算保持。"
             "当前成交价已含滑点；参考毛差减滑点及费用不重复扣费。底仓和人工增仓全部原始含费成本仍进入5%分母。", "",
             "| 年度 | 开启 | Huber profile | 来源L2 profile | 最小参考净边际 | 份额 | 退出 |",
             "|---|---|---|---|---:|---:|---|"]
    for row in selections:
        lines.append('| '+' | '.join(str(row[key]) for key in ('test_year','enabled','profile','source_l2_profile','min_net_edge','fraction','exit_mode'))+' |')
    lines += ["", "| 年度 | 复查请求 | 取消请求 | 取消股票日 | 通过但未成交请求 | 实际首腿组 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in guards.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.guard_audit_requests} | {row.guard_cancelled_requests} | {row.guard_cancelled_stock_days} | "
                     f"{row.guard_passed_without_actual_first_leg} | {row.guard_actual_first_leg_groups} |")
    lines += ["", "| 年度 | 首次锁存 | 提前触发 | 利润/止损/原期限 |", "|---|---:|---:|---:|"]
    for row in exits.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.latched_exit_pairs} | {row.early_trigger_pairs} | "
                     f"{row.profit_buffer_trigger_pairs}/{row.stop_loss_trigger_pairs}/{row.hard_deadline_trigger_pairs} |")
    lines += ["", "hold或无完整组时没有可评价的做T胜率。5%仍按全部原始含费底仓成本逐内部年验收；"
              "各内部折达到原条件后才统一正式外评，烟测不得当收益证据。库存/恢复/pending/T+1/共同底仓差如实保留，"
              "正常永久底仓期末持有不等于未恢复T；频率未量化不暗判足够。", "",
              "2023–2025留出折为可含未来训练年的回顾研究；2026时间前推但只截至4月30日，不年化跨过5%。"
              "这些历史已反复研究，独立年度账户不拼为跨年复利，也不称全新测试。", ""]
    path.write_text(text+'\n'.join(lines), encoding="utf-8")
    with (directory/"输出文件说明.md").open('a', encoding='utf-8') as stream:
        stream.write("\n- first_leg_exit_side_audit.csv：原到达报价复查；取消和通过请求不等于实际成交组。\n"
                     "- guard_summary.csv/json：复查、取消、股票日和真实首腿分列。\n"
                     "- selected_robust_rules.csv、inventory_robust_summary.json：v17六个新Huber模型、固定alpha、原矩阵/标签身份合同及来源。\n"
                     "- planned_horizon_summary.csv：已锁存pair的20/40/60原计划期限计数；不能把恢复片段重复计组。\n"
                     "- exit_decisions.csv、dynamic_exit_summary.csv/json：每组首次锁存与真实恢复语义，参考评分不等于账户利润。\n")
