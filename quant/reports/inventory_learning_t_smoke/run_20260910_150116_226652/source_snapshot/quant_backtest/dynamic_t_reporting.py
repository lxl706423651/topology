"""v13动态恢复的独立报告包装；保留固定底仓成本、手数及目标计算。

只在调用方的新目录生成派生表/图，不改变balanced报告器、输入账户或历史报告。
退出表每个pair只含一次锁存信号，恢复片段由原round_trips独立统计。
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .balanced_t_reporting import write_balanced_report


VERSION = "dynamic_inventory_t_v13"
EXIT_MODES = ("fixed", "profit_buffer", "profit_buffer_stop")
EXIT_REASONS = ("profit_buffer", "stop_loss", "hard_deadline")
EXIT_COLUMNS = ("pair_id", "symbol", "direction", "exit_mode", "first_signal_time",
                "first_execution_time", "original_deadline", "exit_signal_time",
                "exit_earliest_time", "exit_reason", "observed_price", "estimated_net_pnl",
                "estimated_net_return", "normalization_notional", "remaining_shares",
                "realized_fragment_net_pnl")
SCOPE_LABELS = {"formal": "正式年度验证", "smoke": "工程烟测，不得作为收益证据",
                "internal": "训练内部评价，不是外层验证"}


def _json_default(value):
    """仅处理报告中出现的numpy标量与时间，不隐式改写指标。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"不能序列化报告值：{type(value)}")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=_json_default, allow_nan=False) + "\n", encoding="utf-8")


def _exit_summary(year: int, result: dict) -> tuple[dict, pd.DataFrame]:
    """核对退出表基本身份与信号顺序；不重新拟合或选择退出参数。"""
    if "exit_decisions" not in result or not isinstance(result["exit_decisions"], pd.DataFrame):
        raise ValueError("v13结果须包含exit_decisions DataFrame，包括hold的有类型空表")
    frame = result["exit_decisions"].copy(deep=True)
    if frame.columns.duplicated().any() or not set(EXIT_COLUMNS).issubset(frame):
        raise ValueError("exit_decisions缺少规范字段或列名重复")
    if frame.pair_id.duplicated().any():
        raise ValueError("每组只能有一个首次锁存退出信号，恢复片段不能重复计组")
    events = result["trade_events"]
    first = events.loc[events.event_type.eq("t") & events.fragment_id.eq(0)]
    if not set(frame.pair_id).issubset(set(first.pair_id)):
        raise ValueError("退出信号引用了不存在的实际首腿")
    if len(frame):
        if (not frame.exit_mode.isin(EXIT_MODES).all() or
                not frame.exit_reason.isin(EXIT_REASONS).all()):
            raise ValueError("未知动态退出模式或原因")
        dates = {}
        for column in ("first_signal_time", "first_execution_time", "original_deadline",
                       "exit_signal_time", "exit_earliest_time"):
            dates[column] = pd.to_datetime(frame[column], format="mixed", errors="raise")
            if dates[column].isna().any() or dates[column].dt.tz is not None:
                raise ValueError("退出时间须为完整的市场本地无时区时间")
        if (not dates["first_signal_time"].lt(dates["first_execution_time"]).all() or
                not dates["exit_signal_time"].ge(dates["first_execution_time"]).all() or
                not dates["exit_earliest_time"].gt(dates["exit_signal_time"]).all()):
            raise ValueError("退出观察/下一分钟执行时序不合法")
        denominators = pd.to_numeric(frame.normalization_notional, errors="raise")
        if not (np.isfinite(denominators) & denominators.gt(0)).all():
            raise ValueError("退出净额率必须保留完整首腿的正参考金额分母")
    modes = sorted(frame.exit_mode.unique().tolist())
    selection = result.get("selected_rule", {})
    if isinstance(selection, dict) and isinstance(selection.get("rule"), dict):
        selection = selection["rule"]
    selected_mode = selection.get("exit_mode", selection.get("dynamic_exit_mode")) if isinstance(selection, dict) else None
    if selected_mode is None:
        selected_mode = result.get("metrics", {}).get("dynamic_exit_mode", "unknown")
    summary = {"test_year": int(year), "selected_exit_mode": selected_mode,
               "observed_exit_modes": ",".join(modes), "started_pairs": len(first),
               "latched_exit_pairs": len(frame),
               "started_pairs_without_exit_signal": len(set(first.pair_id).difference(frame.pair_id)),
               **{f"{reason}_trigger_pairs": int(frame.exit_reason.eq(reason).sum()) for reason in EXIT_REASONS},
               "early_trigger_pairs": int((pd.to_datetime(frame.exit_earliest_time, format="mixed") <
                                             pd.to_datetime(frame.original_deadline, format="mixed")).sum()) if len(frame) else 0}
    return summary, frame


def write_dynamic_report(directory: Path, cfg: dict, annual_results, validation_table=None) -> dict:
    """写v13规范账户报告及退出审计；收益/频率目标沿用balanced的原始分母。"""
    if cfg.get("research_version") != VERSION:
        raise ValueError("本包装器仅用于dynamic_inventory_t_v13")
    for key in ("dynamic_take_net_return", "dynamic_stop_net_return"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key}须为正有限数")
    entries = sorted(annual_results.items() if isinstance(annual_results, dict) else annual_results,
                     key=lambda item: int(item[0]))
    exits = [_exit_summary(int(year), result) for year, result in entries]
    directory = Path(directory)
    status = write_balanced_report(directory, cfg, entries, validation_table)
    tables = []
    for (year, _), (_, frame) in zip(entries, exits):
        yearly_path = directory / f"year_{int(year)}/exit_decisions.csv"
        # 主入口已保存的逐年退出原表保持不变；独立包装调用时才补写。
        if not yearly_path.exists():
            frame.to_csv(yearly_path, index=False, encoding="utf-8-sig")
        tables.append(frame.assign(test_year=int(year)))
    pd.concat(tables, ignore_index=True).to_csv(directory / "exit_decisions.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([item for item, _ in exits])
    summary.to_csv(directory / "dynamic_exit_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {"research_version": VERSION, "report_scope": status["report_scope"],
                "dynamic_take_net_return": float(cfg["dynamic_take_net_return"]),
                "dynamic_stop_net_return": float(cfg["dynamic_stop_net_return"]),
                "exit_modes": list(EXIT_MODES), "candidate_count": 163,
                "threshold_denominator": "original first-leg reference_price * original_shares",
                "all_modes_share_full_240_minute_quotes_retries_and_marks": True,
                "fixed_mode_is_not_identical_to_v12_sparse_execution": True,
                "exit_decision_rows_are_first_latched_signals_not_recovery_fragments": True,
                "thresholds_are_not_guaranteed_fill_returns": True,
                "parameters_claimed_optimal": False,
                "goal_calculation_unchanged": True,
                "annual_exit_summaries": summary.to_dict("records")}
    _write_json(directory / "dynamic_exit_summary.json", metadata)
    status = {**status, "dynamic_exit": metadata}
    _write_json(directory / "goal_evaluation.json", status)
    annual = pd.read_csv(directory / "annual_summary.csv", float_precision="round_trip")
    _plot_dynamic(directory, cfg, annual, entries, status["report_scope"])
    _append_dynamic_text(directory, cfg, summary, status["report_scope"])
    return status


def _plot_dynamic(directory: Path, cfg: dict, annual: pd.DataFrame, entries: list, scope: str) -> None:
    """独立v13图形，不修改旧plot函数；只展示规范表已有指标。"""
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
    fig, axes = plt.subplots(len(entries), 2, figsize=(15, 5.4 * len(entries)), squeeze=False)
    fig.suptitle(f"均衡底仓动态恢复做T（v13）｜{SCOPE_LABELS[scope]}", fontsize=15, y=.995)
    for index, (year, result) in enumerate(entries):
        row = annual.loc[annual.test_year.eq(int(year))].iloc[0]
        metric = lambda name, spec='.2f': "不可评价" if pd.isna(row.get(name, np.nan)) else format(row[name], spec)
        curve = result["equity_curve"]
        time = pd.to_datetime(curve.datetime, format="mixed")
        left, right = axes[index]
        left.plot(time, (curve.baseline_equity / cfg["initial_cash"] - 1) * 100, label="相同持续底仓")
        left.plot(time, (curve.equity / cfg["initial_cash"] - 1) * 100, '--', label="底仓＋本折选定做T规则")
        left.set_ylabel("原始账户本金收益（%）")
        left.legend()
        left.set_title(f"{int(year)}｜{time.iloc[0]:%Y-%m-%d}—{time.iloc[-1]:%Y-%m-%d}")
        right.plot(time, curve.equity - curve.baseline_equity, color="#177f75")
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
        short = (time.iloc[-1] - time.iloc[0]).days <= 90 and time.iloc[0].year == time.iloc[-1].year
        for ax, content in ((left, left_text), (right, right_text)):
            ax.text(0, -.22, content, transform=ax.transAxes, fontsize=9, va='top',
                    bbox={"boxstyle": "round,pad=.4", "facecolor": "#f2f5f7", "edgecolor": "#d7dfe5"})
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d' if short else '%Y-%m'))
            ax.grid(alpha=.25)
    fig.text(.5, .012, "各年独立账户；2026截至4月30日。年化不用于5%验收；触发净额不等于下一分钟实际成交净额。",
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .045, 1, .95), h_pad=3.)
    fig.savefig(directory / "test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory / "test_return_curve.png", directory / "测试集收益曲线.png")


def _append_dynamic_text(directory: Path, cfg: dict, summary: pd.DataFrame, scope: str) -> None:
    """追加本版本明确变化，不把完整分钟fixed对照冒充v12相同执行。"""
    path = directory / "final_report.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(text.splitlines()[0], f"# 均衡底仓动态恢复做T（v13）：{SCOPE_LABELS[scope]}", 1)
    take = cfg["dynamic_take_net_return"]
    stop = cfg["dynamic_stop_net_return"]
    lines = ["", "## v13恢复协议与实际触发", "",
             f"旧54个入场候选分别配合fixed/profit_buffer/profit_buffer_stop三种退出，另加hold，共163候选。"
             f"profit模式在当前估计完整组净额率≥{take:.2%}时触发，stop模式还允许≤−{stop:.2%}时触发；"
             "分母始终是完整首腿的原参考价×原股数，净额包含已恢复片段和剩余股份的当前估计扣费恢复额。", "",
             "只能观察已完成分钟，触发后最早下一交易分钟执行；意图锁存，按真实价格、容量、费用分批恢复。"
             "未触发则不晚于原固定期限开始恢复，受阻义务仍保留。阈值不是保证成交收益，也不是论文证明的最优参数。", "",
             "所有三种模式使用同一新增完整240分钟报价、重试时钟和共同估值流。fixed是v13内部的执行对照，"
             "与v12旧稀疏报价覆盖不同；两版本差额不能全部归因于动态择时。原底仓、成本分母、T+1旧股预算和年度目标不变。", "",
             "每折只在内部选择规则；全部目标年度内部成本降低率达5%并统一冻结后才正式外评。"
             "烟测可以验证外层连接但不证明盈利。未达条件只保留内部结果，不把hold或少量完整组称为目标达成。", "",
             "| 年度 | 所选退出模式 | 实际首腿组 | 锁存信号 | 提前触发 | 利润/止损/期限触发 | 无退出信号首腿组 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.selected_exit_mode} | {row.started_pairs} | {row.latched_exit_pairs} | "
                     f"{row.early_trigger_pairs} | {row.profit_buffer_trigger_pairs}/{row.stop_loss_trigger_pairs}/{row.hard_deadline_trigger_pairs} | "
                     f"{row.started_pairs_without_exit_signal} |")
    lines.extend(["", "exit_decisions.csv每组仅一条首次锁存信号，不是恢复片段表；estimated_net_return是当前估计，"
                  "实际完成净额继续以test_round_trips.csv及完整组账表为准。无成交时没有可评价的做T胜率。", ""])
    path.write_text(text + "\n".join(lines), encoding="utf-8")
    with (directory / "输出文件说明.md").open("a", encoding="utf-8") as stream:
        stream.write("\n- exit_decisions.csv：按年聚合的首次锁存退出信号，含当前净额估计、完整首腿分母及最早执行时间。\n"
                     "- dynamic_exit_summary.csv/json：按年触发计数、模式和协议；不替代实际完整组/恢复片段/股数失败指标。\n")
