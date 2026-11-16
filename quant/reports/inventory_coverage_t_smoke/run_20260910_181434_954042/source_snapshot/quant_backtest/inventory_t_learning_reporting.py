"""v14永久底仓训练择时报告：复用共同账目，不替换旧版本身份或源文件。

模型固定端点预测与账户真实成交分别描述。收益分母、手数和失败验收直接
沿用balanced规范；动态退出表仍按每组首次锁存计数，不把分片重复计组。
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .balanced_t_reporting import write_balanced_report
from .dynamic_t_reporting import EXIT_MODES, SCOPE_LABELS, _exit_summary, _write_json


VERSION = "inventory_learning_t_v14"
PROFILES = ("l2_depth4", "l2_depth6")
HORIZONS = (20, 40, 60)
CANDIDATE_COUNT = 37


def _selection(year: int, result: dict) -> dict:
    """保留所选规则的精确四字段，期限由每条实际决策确定而非虚构单一期限。"""
    rule = result.get("selected_rule")
    if isinstance(rule, dict) and isinstance(rule.get("rule"), dict):
        rule = rule["rule"]
    if not isinstance(rule, dict) or not isinstance(rule.get("enabled"), (bool, np.bool_)):
        raise ValueError("v14结果须有明确enabled的selected_rule")
    if rule["enabled"]:
        if rule.get("profile") not in PROFILES or rule.get("exit_mode") not in EXIT_MODES:
            raise ValueError("未知v14模型或退出模式")
        for key, choices in (("min_net_edge", (0., .001, .003)), ("fraction", (.5, 1.))):
            value = rule.get(key)
            if isinstance(value, (bool, np.bool_)) or value not in choices:
                raise ValueError(f"{key}不属于预声明候选")
    return {"test_year": int(year), "enabled": bool(rule["enabled"]),
            **{key: rule.get(key) for key in ("profile", "min_net_edge", "fraction", "exit_mode")}}


def write_inventory_learning_report(directory: Path, cfg: dict, annual_results,
                                    validation_table=None) -> dict:
    """在新输出目录写v14报告；不训练、选择规则或修改输入账户/版本配置。"""
    if cfg.get("research_version") != VERSION:
        raise ValueError("本报告器仅用于inventory_learning_t_v14")
    for key in ("dynamic_take_net_return", "dynamic_stop_net_return"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key}须为正有限数")
    entries = sorted(annual_results.items() if isinstance(annual_results, dict) else annual_results,
                     key=lambda item: int(item[0]))
    if not entries or len({int(year) for year, _ in entries}) != len(entries):
        raise ValueError("年度结果必须非空且年份唯一")
    selections = [_selection(year, result) for year, result in entries]
    exits = [_exit_summary(int(year), result) for year, result in entries]
    directory = Path(directory)
    # 直接传原v14配置给共同账户报告器，绝不伪装成v12/v13配置。
    status = write_balanced_report(directory, cfg, entries, validation_table)
    tables = []
    for (year, _), (_, frame) in zip(entries, exits):
        path = directory / f"year_{int(year)}/exit_decisions.csv"
        if not path.exists():
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        tables.append(frame.assign(test_year=int(year)))
    pd.concat(tables, ignore_index=True).to_csv(directory/"exit_decisions.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([item for item, _ in exits])
    summary.to_csv(directory/"dynamic_exit_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(directory/"selected_learning_rules.csv", index=False, encoding="utf-8-sig")
    metadata = {"research_version": VERSION, "report_scope": status["report_scope"],
                "candidate_count": CANDIDATE_COUNT, "profiles": list(PROFILES),
                "prediction_horizons_minutes": list(HORIZONS),
                "target_definition": "reference_recovery_price/reference_next_minute_price-1",
                "target_is_portfolio_execution_net_pnl": False,
                "direction_and_horizon_selection": "maximum_current_predicted_net_edge_after_original_costs",
                "probability_daily_amplitude_or_support_gate_used": False,
                "training_cohort": "core_oof_permanent_inventory_opening_old_shares",
                "added_four_week_rank_features": False,
                "dynamic_take_net_return": float(cfg["dynamic_take_net_return"]),
                "dynamic_stop_net_return": float(cfg["dynamic_stop_net_return"]),
                "all_modes_share_full_240_minute_quotes_retries_and_marks": True,
                "threshold_denominator": "original first-leg reference_price * original_shares",
                "exit_decision_rows_are_first_latched_signals_not_recovery_fragments": True,
                "goal_calculation_unchanged": True, "selected_rules": selections,
                "annual_exit_summaries": summary.to_dict("records")}
    _write_json(directory/"inventory_learning_summary.json", metadata)
    _write_json(directory/"dynamic_exit_summary.json", metadata)
    status = {**status, "research_version": VERSION, "inventory_learning": metadata}
    _write_json(directory/"goal_evaluation.json", status)
    annual = pd.read_csv(directory/"annual_summary.csv", float_precision="round_trip")
    _plot_inventory_learning(directory, cfg, annual, entries, status["report_scope"])
    _append_learning_text(directory, cfg, summary, selections, status["report_scope"])
    return status


def _plot_inventory_learning(directory: Path, cfg: dict, annual: pd.DataFrame,
                             entries: list, scope: str) -> None:
    """只画规范账户已有指标；v14身份、短区间日期和无交易胜率明确。"""
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
    fig.suptitle(f"永久底仓训练择时（v14）｜{SCOPE_LABELS[scope]}", fontsize=15, y=.995)
    for index, (year, result) in enumerate(entries):
        row = annual.loc[annual.test_year.eq(int(year))].iloc[0]
        metric = lambda name, spec='.2f': "不可评价" if pd.isna(row.get(name, np.nan)) else format(row[name], spec)
        curve = result["equity_curve"]
        time = pd.to_datetime(curve.datetime, format="mixed")
        left, right = axes[index]
        left.plot(time, (curve.baseline_equity/cfg["initial_cash"]-1)*100, label="相同永久底仓")
        left.plot(time, (curve.equity/cfg["initial_cash"]-1)*100, '--', label="底仓＋本折训练择时")
        left.set_ylabel("原始账户本金收益（%）")
        left.legend()
        left.set_title(f"{int(year)}｜{time.iloc[0]:%Y-%m-%d}—{time.iloc[-1]:%Y-%m-%d}")
        right.plot(time, curve.equity-curve.baseline_equity, color="#177f75")
        right.axhline(0, color='grey', lw=.7)
        right.set_ylabel("真实净增量（元）")
        right.set_title(f"净增量 {row.incremental_marked_pnl:,.2f}｜股数偏差 {row.inventory_mismatch_days}日")
        left_text = (f"总收益 {metric('total_return','.2%')}｜基准 {metric('baseline_total_return','.2%')}\n"
                     f"年化估计 {metric('annual_return','.2%')}｜年化波动 {metric('annual_volatility','.2%')}\n"
                     f"Sharpe {metric('sharpe')}｜最大回撤 {metric('max_drawdown','.2%')}")
        win = metric('win_rate', '.2%') if row.complete_t_pairs else "不可评价（无完整T组）"
        right_text = (f"完整T组 {int(row.complete_t_pairs)}｜配对手 {row.completed_t_pair_lots:g}｜净胜率 {win}\n"
                      f"成本降低率 {metric('holding_cost_reduction_rate','.2%')}｜期末持仓 {metric('final_open_position_rows','.0f')}笔\n"
                      f"底仓股票 {metric('reference_stock_count','.0f')}只｜平均持仓市值 {metric('mean_holdings_market_value',',.0f')}元")
        short = (time.iloc[-1]-time.iloc[0]).days <= 90 and time.iloc[0].year == time.iloc[-1].year
        for ax, content in ((left, left_text), (right, right_text)):
            ax.text(0, -.22, content, transform=ax.transAxes, fontsize=9, va='top',
                    bbox={"boxstyle": "round,pad=.4", "facecolor": "#f2f5f7", "edgecolor": "#d7dfe5"})
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d' if short else '%Y-%m'))
            ax.grid(alpha=.25)
    fig.text(.5, .012, "各年独立账户；2026截至4月30日。固定端点预测不是实际成交利润，年化不用于5%验收。",
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .045, 1, .95), h_pad=3.)
    fig.savefig(directory/"test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory/"test_return_curve.png", directory/"测试集收益曲线.png")


def _append_learning_text(directory: Path, cfg: dict, summary: pd.DataFrame,
                          selections: list[dict], scope: str) -> None:
    """明确学习对象、实际执行与年度范围；不把预测或烟测写成达标。"""
    path = directory/"final_report.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(text.splitlines()[0], f"# 永久底仓训练择时（v14）：{SCOPE_LABELS[scope]}", 1)
    lines = ["", "## v14学习目标和实际账户", "",
             "两种回归器l2_depth4/l2_depth6分别学习20/40/60交易分钟固定端点的原始参考收益："
             "恢复参考价/下一分钟执行参考价−1。这不是共享现金账户的真实组合利润，也不是成交概率。"
             "每个当前信号在方向和三个期限中选择预测扣费净边际最大的方案，再应用本折选定下限。", "",
             "边际估计沿用原万八免5佣金、双向过户费、按日期的卖出印花税和双腿滑点；"
             "真实利润仍由下一分钟及后续实际分片成交、现金、容量与T+1额度决定。"
             "不使用旧概率、日振幅或支持域门控，预测正边际不保证扣费后获利。", "",
             "训练队列由同折核心年份OOF周排名产生，入场前严格剔除跨排除年边界的来源与标签；"
             "永久底仓全部持有日的昨日旧股机会均可进入训练，不再限于周中日。"
             "使用当前全截面的因果特征，不新增四周预测rank因子；模型参数只在核心训练数据拟合，"
             "内部年选择规则，全部规则冻结后才按协议评价外层。", "",
             "预声明37候选：2种profile×3个净边际下限（0/0.001/0.003）×2份额（0.5/1）"
             "×3退出模式（fixed/profit_buffer/profit_buffer_stop）＋hold。当前决策选定期限不等于动态止盈止损；"
             "动态恢复另根据当前已完成bar触发，最早下一交易分钟成交，触发后锁存并按实际容量分片。", "",
             f"利润退出阈值为{cfg['dynamic_take_net_return']:.2%}，止损模式另在净额率≤−{cfg['dynamic_stop_net_return']:.2%}触发；"
             "分母固定为完整首腿原始参考价×原股数，包含已恢复片段及剩余股数的当前扣费估计。"
             "最晚仍按原选定期限恢复；全部模式共享完整240分钟报价、重试和共同估值，不保证按触发净额成交。", "",
             "| 年度 | 开启 | profile | 净边际下限 | 份额 | 退出模式 |",
             "|---|---|---|---:|---:|---|"]
    for row in selections:
        lines.append("| "+" | ".join(str(row[key]) for key in
                     ("test_year", "enabled", "profile", "min_net_edge", "fraction", "exit_mode"))+" |")
    lines += ["", "| 年度 | 实际首腿组 | 首次锁存 | 提前触发 | 利润/止损/期限触发 | 无退出信号首腿组 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.started_pairs} | {row.latched_exit_pairs} | {row.early_trigger_pairs} | "
                     f"{row.profit_buffer_trigger_pairs}/{row.stop_loss_trigger_pairs}/{row.hard_deadline_trigger_pairs} | "
                     f"{row.started_pairs_without_exit_signal} |")
    lines += ["", "exit_decisions.csv每组仅记录一次首次锁存；恢复片段不能重复计完整组。"
              "hold或无完整组时没有可评价的做T胜率。原始底仓成本分母、股数恢复与每年度5%合同均不改变；"
              "全部内部年度满足原条件才正式外评，工程烟测不提供盈利证据。", ""]
    path.write_text(text+"\n".join(lines), encoding="utf-8")
    with (directory/"输出文件说明.md").open("a", encoding="utf-8") as stream:
        stream.write("\n- selected_learning_rules.csv：逐折实际选定profile、净边际下限、份额和退出模式；实际期限见逐信号决策。\n"
                     "- inventory_learning_summary.json：v14版本、固定端点学习目标及真实账户语义。\n"
                     "- exit_decisions.csv、dynamic_exit_summary.csv/json：首次锁存与动态触发统计，不能替代真实成交净收益。\n")
