"""v18覆盖率门槛报告：只消费已保存账户并写调用方的新目录。

直接保留balanced成本/手数/验收口径，独立标识原L2的0.80→0.70特征筛选研究。
不拟合或验证模型身份；原训练行/标签不变的合同由入口及独审另外验证。
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .balanced_t_reporting import write_balanced_report
from .dynamic_t_reporting import EXIT_MODES, SCOPE_LABELS, _exit_summary, _write_json
from .guarded_t_reporting import _summarize_guard

VERSION = "inventory_coverage_t_v18"
PROFILES = ("l2_depth4", "l2_depth6")
HORIZONS = (20, 40, 60)
SESSION_MINUTES = tuple(range(571, 691)) + tuple(range(781, 901))
COVERAGE_FEATURES = ("f_return_15m", "f_timing_market_return_15m", "f_timing_excess_return_15m")


def _selection(year: int, result: dict) -> dict:
    """核对37候选的数值参数与新profile，保留逐信号多期限选择的语义。"""
    rule = result.get("selected_rule")
    if isinstance(rule, dict) and isinstance(rule.get("rule"), dict):
        rule = rule["rule"]
    if not isinstance(rule, dict) or not isinstance(rule.get("enabled"), (bool, np.bool_)):
        raise ValueError("v18结果须有明确enabled的selected_rule")
    if rule["enabled"]:
        if rule.get("profile") not in PROFILES or rule.get("exit_mode") not in EXIT_MODES:
            raise ValueError("未知v18新拟合L2模型或退出模式")
        for key, choices in (("min_net_edge", (0., .001, .003)), ("fraction", (.5, 1.))):
            if isinstance(rule.get(key), (bool, np.bool_)) or rule.get(key) not in choices:
                raise ValueError(f"{key}不属于原37候选")
    return {"test_year": int(year), "enabled": bool(rule["enabled"]),
            **{k: rule.get(k) for k in ("candidate_index", "profile", "min_net_edge", "fraction", "exit_mode")}}


def _inherited_summary(kind: str, year: int, result: dict) -> tuple[dict, pd.DataFrame]:
    """只复用旧纯统计函数；错误加v18上下文，不调用旧版本writer或改输入cfg。"""
    try:
        return (_summarize_guard if kind == "guard" else _exit_summary)(year, result)
    except ValueError as error:
        detail = str(error).replace("v15", "v18继承的guard格式").replace("v13", "v18继承的动态退出格式")
        raise ValueError(f"v18 {kind}审计：{detail}") from error


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
            raise ValueError("v18原计划须为真实首腿后20/40/60交易分钟且同日不晚于14:55")
        counts = {h: int(delta.eq(h).sum()) for h in HORIZONS}
    return {"test_year": int(year), "latched_pairs_with_checked_deadline": len(frame),
            **{f"planned_h{h}_pairs": counts[h] for h in HORIZONS}}


def write_coverage_inventory_report(directory: Path, cfg: dict, annual_results,
                                    validation_table=None) -> dict:
    """用真实v18配置生成规范报告；不读模型/选参收益，不改原账户或输入表。"""
    if cfg.get("research_version") != VERSION or cfg.get("training_loss") != "regression_l2":
        raise ValueError("v18须明确inventory_coverage_t_v18及regression_l2")
    for key, expected in (("feature_min_coverage", .70), ("source_feature_min_coverage", .80)):
        if isinstance(cfg.get(key), (bool, np.bool_)) or cfg.get(key) != expected:
            raise ValueError(f"v18 {key}须为预声明{expected}")
    if (cfg.get("forecast_target") != "fixed_20_40_60_reference_return"
            or cfg.get("horizons") != list(HORIZONS) or cfg.get("recovery_key") == "close"):
        raise ValueError("v18须保留20/40/60参考端点，不能误用固定尾盘目标")
    if cfg.get("frozen_t_models_reused", False) is not False:
        raise ValueError("v18每折新拟合六个L2模型，不能声明复用旧T权重")
    if cfg.get("learning_model_profiles", list(PROFILES)) != list(PROFILES):
        raise ValueError("v18仅支持原l2_depth4/l2_depth6")
    for key in ("source_execution_report_dir", "source_learning_report_dir"):
        if not isinstance(cfg.get(key), (str, Path)) or not str(cfg[key]).strip():
            raise ValueError(f"v18缺少明确来源路径：{key}")
    if not isinstance(cfg.get("t_recheck_exit_side_at_execution"), (bool, np.bool_)) or not cfg["t_recheck_exit_side_at_execution"]:
        raise ValueError("v18须显式保留原guard15首腿复查")
    entries = sorted(annual_results.items() if isinstance(annual_results, dict) else annual_results,
                     key=lambda item: int(item[0]))
    if not entries or len({int(year) for year, _ in entries}) != len(entries):
        raise ValueError("年度结果须非空且年份唯一")
    selections = [_selection(year, result) for year, result in entries]
    guards = [_inherited_summary("guard", int(year), result) for year, result in entries]
    exits = [_inherited_summary("exit", int(year), result) for year, result in entries]
    horizons = [_horizon_summary(int(year), item[1]) for (year, _), item in zip(entries, exits)]
    for selected, (_, result) in zip(selections, entries):
        first = result["trade_events"].loc[lambda rows: rows.event_type.eq("t") & rows.fragment_id.eq(0)]
        if not selected["enabled"] and len(first):
            raise ValueError("v18 hold规则不能含实际T首腿")
    directory = Path(directory)
    # 始终传原cfg；不调用旧版本writer然后改写身份。
    status = write_balanced_report(directory, cfg, entries, validation_table)
    for filename, items in (("first_leg_exit_side_audit.csv", guards), ("exit_decisions.csv", exits)):
        frames = []
        for (year, _), (_, frame) in zip(entries, items):
            path = directory/f"year_{int(year)}"/filename
            if not path.exists():
                frame.to_csv(path, index=False, encoding="utf-8-sig")
            frames.append(frame.assign(test_year=int(year)))
        pd.concat(frames, ignore_index=True).to_csv(directory/filename, index=False, encoding="utf-8-sig")
    guard_summary = pd.DataFrame([summary for summary, _ in guards])
    exit_summary = pd.DataFrame([summary for summary, _ in exits])
    guard_summary.to_csv(directory/"guard_summary.csv", index=False, encoding="utf-8-sig")
    exit_summary.to_csv(directory/"dynamic_exit_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(horizons).to_csv(directory/"planned_horizon_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(directory/"selected_coverage_rules.csv", index=False, encoding="utf-8-sig")
    meta = {"research_version": VERSION, "report_scope": status["report_scope"],
            "model_learning_version": 18, "model_learning_research_version": VERSION,
            "frozen_t_models_reused": False, "frozen_models_reused": False, "models_retrained": True,
            "newly_fitted_models_per_fold": 6, "newly_fitted_model_profiles": list(PROFILES),
            "prediction_horizons_minutes": list(HORIZONS), "training_loss": "regression_l2",
            "objective": "regression", "feature_min_coverage": .70, "source_feature_min_coverage": .80,
            "candidate_count": 37, "coverage_candidate_features": list(COVERAGE_FEATURES),
            "source_execution_report_dir": str(cfg["source_execution_report_dir"]),
            "source_learning_report_dir": str(cfg["source_learning_report_dir"]),
            "training_cohort_reused_from_version": 14, "execution_engine_version": 15,
            "source_report_scope": cfg.get("source_report_scope", "not_recorded"),
            "declared_model_change": "feature_min_coverage 0.80 to 0.70; original L2 loss unchanged",
            "same_training_rows_labels_split_required": True,
            "source_feature_screen_reproduction_required": True,
            "common_feature_values_and_dtypes_equal_required": True,
            "complete_training_matrix_identity_required": False,
            "selected_features_and_column_order_may_change": True,
            "source_training_identity_contract": "same raw core, rows, labels, split and shared feature values/dtypes; original 0.80 selection reproduced; new 0.70 manifest/matrix/weights recorded separately",
            "identity_verification_performed_by_reporter": False,
            "additional_15m_features_guaranteed_selected": False, "feature_gain_guaranteed": False,
            "missing_feature_values_preserved_as_nan": True, "missing_feature_rows_deleted": False,
            "forecast_target": "fixed_20_40_60_reference_return",
            "reference_target": "planned_horizon_reference_price / next_minute_reference_price - 1",
            "cross_horizon_argmax": True, "uniform_1455_forecast_target": False,
            "target_is_portfolio_execution_net_pnl": False,
            "predicted_edge_is_expected_portfolio_net_pnl": False,
            "actual_exit_may_precede_planned_deadline": True, "t_recheck_exit_side_at_execution": True,
            "uses_future_recovery_quotes_or_success_for_current_eligibility": False,
            "guard_pass_is_actual_fill_guarantee": False, "guard_pass_is_daily_recovery_guarantee": False,
            "guard_rows_are_requests_not_completed_pairs": True,
            "guard_has_no_virtual_reverse_fill_or_inventory_consumption": True,
            "goal_calculation_unchanged": True, "selected_rules": selections,
            "annual_guard_summaries": guard_summary.to_dict("records"),
            "annual_exit_summaries": exit_summary.to_dict("records"),
            "annual_planned_horizon_summaries": horizons}
    _write_json(directory/"inventory_coverage_summary.json", meta)
    _write_json(directory/"guard_summary.json", {"research_version": VERSION, "annual_results": meta["annual_guard_summaries"]})
    _write_json(directory/"dynamic_exit_summary.json", {"research_version": VERSION, "annual_results": meta["annual_exit_summaries"]})
    status = {**status, "research_version": VERSION, "inventory_coverage": meta}
    _write_json(directory/"goal_evaluation.json", status)
    annual = pd.read_csv(directory/"annual_summary.csv", float_precision="round_trip")
    _plot_coverage(directory, cfg, annual, entries, status["report_scope"])
    _append_coverage_text(directory, cfg, selections, guard_summary, exit_summary, status["report_scope"])
    return status


def _plot_coverage(directory: Path, cfg: dict, annual: pd.DataFrame, entries: list, scope: str) -> None:
    """仅绘制已有规范指标，用v18/三期限/L2覆盖筛选身份避免与close目标混淆。"""
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
    fig.suptitle(f"永久底仓L2覆盖筛选短期限做T（v18，覆盖率0.80→0.70）｜{SCOPE_LABELS[scope]}", fontsize=15, y=.995)
    for i, (year, result) in enumerate(entries):
        row = annual.loc[annual.test_year.eq(int(year))].iloc[0]
        metric = lambda name, fmt='.2f': "不可评价" if pd.isna(row.get(name, np.nan)) else format(row[name], fmt)
        curve = result["equity_curve"]
        time = pd.to_datetime(curve.datetime, format="mixed")
        left, right = axes[i]
        left.plot(time, (curve.baseline_equity/cfg['initial_cash']-1)*100, label="相同永久底仓")
        left.plot(time, (curve.equity/cfg['initial_cash']-1)*100, '--', label="底仓＋v18做T")
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
    fig.text(.5, .012, "各年独立账户；2026截至4月30日。每折新拟合6个20/40/60分钟L2覆盖筛选模型；预测参考评分不等于真实账户净收益。",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95), h_pad=3.)
    fig.savefig(directory/"test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory/"test_return_curve.png", directory/"测试集收益曲线.png")


def _append_coverage_text(directory: Path, cfg: dict, selections: list, guards: pd.DataFrame,
                          exits: pd.DataFrame, scope: str) -> None:
    """解释唯一覆盖门槛变化及验收边界；不把未运行的特征审计写成已通过。"""
    path = directory/"final_report.md"
    original = path.read_text(encoding="utf-8")
    title = f"# 永久底仓L2特征覆盖做T（v18）：{SCOPE_LABELS[scope]}"
    original = original.replace(original.splitlines()[0], title, 1)
    lines = ["", "## 覆盖筛选变化与原执行合同", "",
        "唯一声明的学习机制改动是feature_min_coverage从0.80降至0.70；原L2损失、depth4/depth6、"
        "20/40/60交易分钟端点保持，每折新拟合6模型。2个profile×3净边际下限×2份额×3退出＋hold仍为37个数字候选。"
        "不复用原T权重进行新预测，也不同时更换为Huber或固定14:55目标。", "",
        "原38个候选特征保持。个股/市场/超额15分钟收益仅进入覆盖资格后的相关性筛选，"
        "不保证三列实际入模或改善收益；相关性排序可能替换旧列，不能把本次说成固定追加三列。"
        "早时钟的缺失保留NaN交给原LightGBM处理，不强填零、不删当前样本、不填未来值。", "",
        f"完整核心来源：`{cfg['source_learning_report_dir']}`；原执行与共同底仓来源：`{cfg['source_execution_report_dir']}`。"
        "入口和独审须核原训练行/标签/隔离表/共同特征值及dtype，并在同一核心表重现原0.80清单。"
        "新0.70特征清单、矩阵列序和权重允许改变，必须分别保存真实SHA；本报告器不读取模型、不代替这些身份核验。"
        "两核心年仍联合训练，所有特征选择只用核心行，周选股及原底仓冻结。", "",
        "模型学习下一合法分钟到固定20/40/60分钟的原始参考价格收益；成本调整预测边际仍仅用于方向/期限选择，"
        "不是实际账户利润。original_deadline从真实首腿后计交易分钟并跳过午休；14:55是原恢复底线，"
        "不是统一预测端点。动态利润/止损在当前完成bar观察，下一分钟起锁存、按真实容量分片恢复。", "",
        "沿用guard15的同到达quote反方向限价复查。取消不虚拟扣现金、旧股或容量；复查通过也可能因资金等未成交，"
        "后续封板或无量仍可能恢复失败。取消请求不是完成T组，不等于避免亏损金额。", "",
        "原万八免5、过户费、日期印花税、双腿滑点、永久底仓和T+1旧股额度不变；"
        "成交价已含滑点，参考毛差减滑点及费用只扣一次。全部原始含费底仓成本（含人工加仓及未做T股票）仍为5%分母，"
        "T买回不算新增底仓成本。完整组、恢复片段、配对手和双腿成交手分别列示。", "",
        "| 年度 | 开启 | L2 profile | 最小参考净边际 | 份额 | 退出 |",
        "|---|---|---|---:|---:|---|"]
    for row in selections:
        lines.append('| '+' | '.join(str(row[key]) for key in ('test_year','enabled','profile','min_net_edge','fraction','exit_mode'))+' |')
    lines += ["", "| 年度 | 复查请求 | 取消请求 | 取消股票日 | 通过未成交 | 实际首腿组 |", "|---|---:|---:|---:|---:|---:|"]
    for row in guards.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.guard_audit_requests} | {row.guard_cancelled_requests} | {row.guard_cancelled_stock_days} | {row.guard_passed_without_actual_first_leg} | {row.guard_actual_first_leg_groups} |")
    lines += ["", "| 年度 | 首次锁存 | 提前触发 | 利润/止损/原期限 |", "|---|---:|---:|---:|"]
    for row in exits.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.latched_exit_pairs} | {row.early_trigger_pairs} | {row.profit_buffer_trigger_pairs}/{row.stop_loss_trigger_pairs}/{row.hard_deadline_trigger_pairs} |")
    lines += ["", "hold或无完整组时没有可评价的做T胜率。库存差、恢复失败、期末待恢复、T+1与共同底仓差原样报告；"
        "正常永久底仓期末持有不等于未恢复T。足够多的频率未量化时不暗判达标；每个内部折满足原执行条件并达到"
        "全部底仓成本5%后才允许统一正式外评，工程烟测不证明收益。", "",
        "各年度独立账户不拼接为复利。2023–2025留出折可能使用未来训练年，只是回顾研究；2026时间前推但仅到4月30日，"
        "不年化跨越5%。历史已经反复研究，不能称全新测试；只读结果不能证明本次特征变化有因果盈利优势。", ""]
    path.write_text(original+'\n'.join(lines), encoding="utf-8")
    with (directory/"输出文件说明.md").open('a', encoding='utf-8') as stream:
        stream.write("\n- inventory_coverage_summary.json、selected_coverage_rules.csv：v18六个新L2模型、覆盖门槛和来源身份合同；实际模型核验另见入口/独审。\n"
            "- first_leg_exit_side_audit.csv、guard_summary.csv/json：复查/取消/实际首腿分列，取消不计完成组。\n"
            "- exit_decisions.csv、dynamic_exit_summary.csv/json、planned_horizon_summary.csv：每组首次锁存、提前恢复及20/40/60原计划期限；片段不重复计组。\n")
