"""从项目根目录运行周度top100与学习型做T的嵌套年度研究。

输入只读分钟行情和YAML；构建独立缓存、训练模型、冻结内部选择后回测留出年，
输出标准CSV/PNG/中文报告。不会连接券商，不修改历史行情或覆盖旧报告。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.strategy import BaseStrategy
from quant_backtest.weekly_t_data import load_weekly_t_data
from quant_backtest.weekly_t_engine import build_recovery_store, run_weekly_portfolio
from quant_backtest.weekly_t_features import (
    add_session_context_features, add_weekly_execution_features, select_recoverable_signal_sides,
)
from quant_backtest.weekly_t_recovery_data import load_recovery_data
from quant_backtest.weekly_t_validation import run_candidate_evaluations, validate_candidate_result
from quant_backtest.weekly_t_models import (
    fit_t_model, fit_weekly_models, predict_t, predict_weekly,
)


class WeeklyLearnedTStrategy(BaseStrategy):
    """主板周度只多底仓上的库存T；预测尾盘回报决定方向，不建立空头。

    data包含prediction；target_position始终1表示维持底仓，t_action仅是库存
    换手请求，卖出预算、开平仓日限制和整手资金均由组合引擎执行。
    """

    name = "weekly_learned_t"

    def __init__(self, rule: dict):
        """rule给出固定threshold、direction、fraction和enabled，不读取验证标签。"""
        self.rule = dict(rule)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """仅读取同索引prediction，返回对齐的长仓目标和T动作；不改变data。"""
        result = pd.DataFrame(index=data.index)
        result["target_position"] = 1.0
        result["t_action"] = 0
        result["fraction"] = float(self.rule.get("fraction", 1.0))
        if self.rule.get("enabled", True):
            threshold = float(self.rule["threshold"])
            direction = self.rule["direction"]
            if direction in {"sell_first", "both"}:
                result.loc[data.prediction <= -threshold, "t_action"] = -1
            if direction in {"buy_first", "both"}:
                result.loc[data.prediction >= threshold, "t_action"] = 1
        return result


def parse_arguments() -> argparse.Namespace:
    """读取持久YAML和单次覆盖项，cache-only只构建数据不训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/weekly_t_cross_validation_v3.yaml")
    for key in ("workers", "max-symbols", "top-n", "smoke-weeks"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--test-years")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> dict:
    """应用CLI显式覆盖并校验权重、资金和路径；原YAML不修改。"""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("workers", "max_symbols", "top_n", "smoke_weeks", "output_dir"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    cfg["weekly_top_n"] = cfg["top_n"]
    if args.test_years:
        cfg["test_years"] = [int(y) for y in args.test_years.split(",")]
    for key in ("data_root", "cache_dir", "output_dir"):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(PROJECT_ROOT / cfg[key])
    if cfg.get("recovery_cache_dir") and not Path(cfg["recovery_cache_dir"]).is_absolute():
        cfg["recovery_cache_dir"] = str(PROJECT_ROOT / cfg["recovery_cache_dir"])
    if not np.isclose(cfg["weekly_return_weight"] + cfg["weekly_amplitude_weight"], 1):
        raise ValueError("周度两预测权重之和必须为1")
    if not 0 < cfg["base_fraction"] < 1 or cfg["top_n"] <= 0:
        raise ValueError("底仓权重须在(0,1)并预留现金；top_n须正数")
    if cfg["smoke_weeks"]:
        for key in ("weekly_model", "t_model"):
            cfg[key]["n_estimators"] = 20
            cfg[key]["min_child_samples"] = 20
        cfg["min_avg_daily_amount"] = 0
        cfg["min_price"] = 0
        cfg["t_max_train_rows"] = 30000
        cfg["candidate_shortlist_size"] = 2
        cfg["candidate_workers"] = 2
        cfg["min_prior_close_auction_amount"] = 0
    return cfg


def select_weekly_universe(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """只用第N周已知的价格、成交额、历史天数约束候选；不读取下一周标签。"""
    names = {"history": ["f_history_days", "history_days"],
             "price": ["f_last_close", "f_week_close", "f_close"],
             "amount": ["f_avg_daily_amount", "f_daily_amount_mean"],
             "total_amount": ["f_week_amount", "f_amount"],
             "days": ["f_week_days", "week_days"]}

    def get_column(group: str) -> pd.Series | None:
        """查找数据契约中同义字段，无法找到时返回None由主函数失败关闭。"""
        for name in names[group]:
            if name in weekly:
                return weekly[name]
        return None

    history, price, amount = get_column("history"), get_column("price"), get_column("amount")
    if amount is None and get_column("total_amount") is not None and get_column("days") is not None:
        amount = get_column("total_amount") / get_column("days")
    if any(x is None for x in (history, price, amount)):
        raise ValueError(f"缺少因果股票池字段，实际字段={list(weekly.columns)}")
    mask = ((history >= cfg["min_history_days"]) & (price >= cfg["min_price"])
            & (amount >= cfg["min_avg_daily_amount"]))
    if "feature_week_complete" in weekly:
        mask &= weekly.feature_week_complete.fillna(False)
    if cfg.get("min_prior_close_auction_amount", 0) > 0:
        if "f_prior_close_auction_amount_min" not in weekly:
            raise ValueError("启用收盘流动性门槛前必须生成已完成周执行特征")
        mask &= weekly.f_prior_close_auction_amount_min >= cfg["min_prior_close_auction_amount"]
    result = weekly.loc[mask].copy()
    result["universe_previous_close"] = price.loc[mask]
    result["universe_avg_daily_amount"] = amount.loc[mask]
    return result


def select_year_rows(weekly: pd.DataFrame, year: int, cfg: dict) -> pd.DataFrame:
    """选完整持仓周且开平仓都在year；一年只有一天的交易周无法T+1周末退出。"""
    mask = ((weekly.entry_date.dt.year == year) & (weekly.exit_date.dt.year == year)
            & (weekly.entry_date < weekly.exit_date))
    if "trade_week_complete" in weekly:
        mask &= weekly.trade_week_complete.fillna(False)
    result = weekly.loc[mask].copy()
    if cfg["smoke_weeks"]:
        weeks = sorted(result.trade_week.unique())[:cfg["smoke_weeks"]]
        result = result.loc[result.trade_week.isin(weeks)].copy()
    return result


def create_selections(bundle: dict, frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """模型按当周候选横截面70/30排序，保留全部预测及top100周度委托清单。"""
    predictions = predict_weekly(bundle, frame, cfg)
    keys = ["trade_week", "symbol"]
    required = [c for c in ("entry_date", "exit_date", "signal_datetime", "universe_previous_close")
                if c not in predictions]
    if required:
        predictions = predictions.merge(frame[keys + required], on=keys, how="left", validate="one_to_one")
    selected = predictions.loc[predictions.selected].copy()
    return selected, predictions


def select_execution_data(selections: pd.DataFrame, daily: pd.DataFrame,
                          intraday: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """取相同日历的实际行情；盘中信号只保留当周已选股的中间交易日。"""
    if selections.empty:
        raise ValueError("本年没有符合历史样本和完整周规则的候选")
    start = selections.entry_date.min()
    # 当年剩余交易日仍需估值、重试受阻退出；不能在最后计划持仓周结束就丢掉债务。
    year_end = pd.Timestamp(year=start.year, month=12, day=31)
    end = min(year_end, daily.date.max())
    symbols = selections.symbol.unique()
    d = daily.loc[daily.symbol.isin(symbols) & daily.date.between(start, end)].copy()
    columns = ["trade_week", "symbol", "entry_date", "exit_date"]
    i = intraday.loc[intraday.date.between(start, end)].merge(
        selections[columns], on=["trade_week", "symbol"], how="inner", validate="many_to_one")
    i = i.loc[(i.date > i.entry_date) & (i.date < i.exit_date)].copy()
    return d, i


def calculate_prediction_metrics(predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """预测和选股冻结后评价留出周排名相关及前100标签均值；只作报告，不能回流选参。"""
    merged = predictions.merge(frame[["trade_week", "symbol", "target_return", "target_amplitude"]],
                                on=["trade_week", "symbol"], validate="one_to_one")
    rows = []
    for week, group in merged.groupby("trade_week", sort=True):
        picked = group.loc[group.selected]
        rows.append({"trade_week": week, "candidate_count": len(group), "selected_count": len(picked),
                     "return_rank_ic": group.pred_return.corr(group.target_return, method="spearman"),
                     "amplitude_rank_ic": group.pred_amplitude.corr(group.target_amplitude, method="spearman"),
                     "selected_mean_return": picked.target_return.mean(),
                     "universe_mean_return": group.target_return.mean(),
                     "selected_mean_amplitude": picked.target_amplitude.mean(),
                     "universe_mean_amplitude": group.target_amplitude.mean(),
                     "selected_missing_return_labels": int(picked.target_return.isna().sum())})
    return pd.DataFrame(rows)


def create_fold_recovery_store(selections: pd.DataFrame, recovery: pd.DataFrame | None,
                               daily: pd.DataFrame | None = None):
    """为整折股票并集保留整个日期范围的恢复行情，不能丢掉跨周挂起股票的真实价格。"""
    if recovery is None:
        return None
    start = daily.date.min() if daily is not None else selections.entry_date.min()
    end = (daily.date.max() if daily is not None else selections.exit_date.max()) + pd.Timedelta(days=1)
    subset = recovery.loc[recovery.symbol.isin(selections.symbol.unique())
                          & recovery.datetime.between(start, end, inclusive="left")]
    return build_recovery_store(subset)


def create_decisions(predictions: pd.DataFrame, rule: dict, cfg: dict | None = None) -> pd.DataFrame:
    """当前行情风险门控后取最早满足阈值信号；同时刻资金顺序可使用预测强度。"""
    cfg = cfg or {}
    signals = WeeklyLearnedTStrategy(rule).generate_signals(predictions)
    permitted = select_recoverable_signal_sides(predictions, signals.t_action, cfg)
    decisions = predictions[["datetime", "symbol"]].copy()
    decisions["action"] = signals.t_action
    decisions["fraction"] = signals.fraction
    decisions = decisions.loc[(decisions.action != 0) & permitted].copy()
    if cfg.get("t_signal_priority") == "predicted_edge":
        decisions["priority"] = predictions.prediction.abs().reindex(decisions.index)
    decisions = decisions.sort_values(["datetime", "symbol"])
    decisions["date"] = decisions.datetime.dt.normalize()
    return decisions.drop_duplicates(["date", "symbol"], keep="first").reset_index(drop=True)


def calculate_proxy_candidates(predictions: pd.DataFrame, intraday: pd.DataFrame,
                               profile: str, cfg: dict) -> list[dict]:
    """仅在内部验证年用每股等金额配对近似收益缩小候选，实际选择再跑完整资金引擎。

    此近似不代替成交测试；恢复缺价算损失，不能据未来可成交性删除当日信号。
    """
    rows = []
    for threshold in cfg["t_thresholds"]:
        for direction in cfg["t_directions"]:
            rule = {"enabled": True, "profile": profile, "threshold": threshold,
                    "direction": direction, "fraction": 1.0}
            decisions = create_decisions(predictions, rule, cfg)
            matched = decisions.merge(intraday[["datetime", "symbol", "execution_price", "recovery_price"]],
                                      on=["datetime", "symbol"], validate="one_to_one")
            future = matched.recovery_price / matched.execution_price - 1
            stamp = np.where(matched.datetime < pd.Timestamp("2023-08-28"),
                             cfg["stamp_rate_before_20230828"], cfg["stamp_rate_from_20230828"])
            cost = 2 * (cfg["commission_rate"] + cfg["transfer_rate"] + cfg["slippage_rate"]) + stamp
            net = (matched.action * future - cost).where(future.notna(), -0.10)
            for fraction in cfg["t_trade_fractions"]:
                rows.append({**rule, "fraction": fraction, "proxy_pairs": len(matched),
                             "proxy_net_return_sum": float(net.sum()) * fraction,
                             "proxy_win_rate": float((net > 0).mean()) if len(net) else 0.0})
    return rows


def calculate_weekly_increment(curve: pd.DataFrame, initial: float, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """把权益差按自然周归因；固定随机种子对周增量做成对bootstrap，不把各年拼净值。"""
    frame = curve[["datetime", "equity", "baseline_equity"]].copy()
    frame["trade_week"] = frame.datetime.dt.to_period("W-SUN").dt.start_time
    weekly = frame.groupby("trade_week", sort=True).last().reset_index()
    weekly["t_added_equity"] = weekly.equity - weekly.baseline_equity
    weekly["t_increment"] = weekly.t_added_equity.diff().fillna(weekly.t_added_equity)
    weekly["increment_return"] = weekly.t_increment / initial
    values = weekly.increment_return.to_numpy()
    if not len(values):
        return weekly, {"increment_ci_low": 0., "increment_ci_high": 0.}
    rng = np.random.default_rng(cfg["random_state"])
    samples = rng.choice(values, size=(cfg["bootstrap_samples"], len(values)), replace=True).sum(axis=1)
    return weekly, {"increment_ci_low": float(np.quantile(samples, .025)),
                    "increment_ci_high": float(np.quantile(samples, .975)),
                    "positive_t_weeks": int((values > 0).sum()), "evaluated_weeks": len(values)}


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    """用UTF-8 BOM导出中文友好CSV，不输出DataFrame索引。"""
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def validate_execution(result: dict, cfg: dict) -> dict:
    """共用候选与外层结果的独立成交核对器，合法受阻保留，实际违规立即报错。"""
    return {**validate_candidate_result(result, cfg), "metrics": result["metrics"]}


def save_result(result: dict, directory: Path, cfg: dict) -> dict:
    """保存逐日组合、成交、库存和周增量，并计算统一净收益差与置信区间。"""
    mapping = {"equity_curve": "test_equity_curve.csv", "trade_events": "test_trade_events.csv",
               "round_trips": "test_round_trips.csv", "open_positions": "test_open_positions.csv",
               "daily_inventory_audit": "daily_inventory_audit.csv", "order_attempts": "blocked_orders.csv",
               "baseline_events": "baseline_trade_events.csv", "weekly_entry_plans": "weekly_entry_plans.csv",
               "base_plan_mismatches": "base_plan_mismatches.csv"}
    for key, filename in mapping.items():
        if key in result:
            write_frame(result[key], directory / filename)
    curve = result["equity_curve"]
    weekly, uncertainty = calculate_weekly_increment(curve, cfg["initial_cash"], cfg)
    write_frame(weekly, directory / "weekly_increment.csv")
    metrics = {**result["metrics"], **uncertainty,
               "total_return": float(curve.equity.iloc[-1] / cfg["initial_cash"] - 1),
               "baseline_total_return": float(curve.baseline_equity.iloc[-1] / cfg["initial_cash"] - 1),
               "t_added_equity": float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1]),
               "increment_return": float((curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1]) / cfg["initial_cash"])}
    write_frame(pd.DataFrame([metrics]), directory / "test_metrics.csv")
    write_json(directory / "execution_audit.json", validate_execution(result, cfg))
    return metrics


def plot_results(results: list[tuple[int, dict]], path: Path, initial: float) -> None:
    """导出可分享的静态年度对照图；每年独立初始资金，中文字体按本机可用项选择。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = list(Path("/usr/share/fonts/opentype/noto").glob("*CJK*"))
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 3.2 * len(results)), squeeze=False)
    for row, (year, result) in enumerate(results):
        c = result["equity_curve"]
        axes[row, 0].plot(c.datetime, (c.equity / initial - 1) * 100, label="周度底仓＋学习型做T")
        axes[row, 0].plot(c.datetime, (c.baseline_equity / initial - 1) * 100, label="相同周度底仓持有", alpha=.8)
        axes[row, 0].set_title(f"{year} 留出年收益（%）")
        axes[row, 0].legend()
        axes[row, 1].plot(c.datetime, (c.equity - c.baseline_equity) / initial * 100, color="#1e746c")
        axes[row, 1].axhline(0, color="gray", linewidth=.7)
        axes[row, 1].set_title(f"{year} 做T净增量（百分点）")
        for ax in axes[row]:
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_report(summary: pd.DataFrame, cfg: dict, status: dict) -> str:
    """生成包括未达到目标时的明确结论、定义、成本、数据边界的完整报告。"""
    recovery_description = (
        "14:55至15:00逐个真实分钟按当时容量、现金和可卖库存分批恢复；每片段分别计费。"
        if cfg.get("use_minute_recovery") else "14:55及15:00按当时行情尝试恢复。")
    lines = ["# 周度前100与学习型做T年度交叉验证", "",
             f"净增益检验：**{'达到预声明检验标准' if status['gain_criterion_met'] else '尚未达到预声明检验标准'}**。",
             "做T增益标准为累计净增量为正、严格多数留出年增益为正、周增量bootstrap区间下界为正，"
             "并且没有T+1、库存恢复或共同底仓差异。完整执行目标还要求周度平仓无失败。"
             "这些历史结果不等于未来收益保证。", "",
             "|留出年|周度持有收益|加入T收益|净增益百分点|净增益元|周bootstrap 95%区间（百分点）|",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"|{r.test_year}|{r.baseline_total_return:.2%}|{r.total_return:.2%}|"
                     f"{r.increment_return*100:+.3f}|{r.t_added_equity:+,.2f}|"
                     f"[{r.increment_ci_low*100:+.3f}, {r.increment_ci_high*100:+.3f}]|")
    lines += ["", f"做T净增益检验通过：{status['t_gain_criterion_met']}；周度开平仓执行完整："
              f"{status['weekly_execution_complete']}；周度退出受阻合计{status['weekly_exit_failure_days']}个年度交易日。"
              "周度退出受阻会保留持仓，不能把该部分描述为已按周末收盘卖出。", ""]
    lines += ["", "## 策略和时间口径", "",
              "第N周全部有效分钟参与日/周量价聚合，另含更早的历史统计。只使用当周末已知特征。"
              "双模型分别预测第N+1周首日开盘至末日收盘的收益排名，及日振幅均值排名；"
              "日振幅定义为绝对非负的(日最高−日最低)/当日开盘。"
              "两个预测各自在当周股票截面重新排名，按70%和30%加权，稳定排序取前100。", "",
              f"股票池为历史文件并集中的主板，至少{cfg['min_history_days']}日历史、上周平均日成交额"
              f"至少{cfg['min_avg_daily_amount']:,.0f}元、上周末价格至少{cfg['min_price']:g}元。"
              "这些筛选仅用已知过去行情，不使用2026名称快照排除过去的ST或退市股票。", "",
              f"每周以纯持有基准权益的{cfg['base_fraction']:.0%}除以{cfg['top_n']}等金额预算。"
              "首日09:30预定买单用09:31分钟的open作为开盘价代理；末日15:00预定卖单用收盘价。"
              "100股整手、容量、现金和涨跌停约束可能使实际成交少于100只；逐周选择与实际成交均保存。"
              "不足整手不会改选未来表现较好的股票，未用预算保持现金。", "",
              "做T模型预测当前信号后一交易分钟至14:55的回报，仅满足训练年内冻结阈值时换手。"
              "负预测可先卖后买；正预测可先买后卖昨日库存。每股每天最多一组。"
              "首日新仓不可卖；末日为保证T+1周末清仓而禁用T。两天交易周仅周度持有，"
              "仅一天交易周无法同时满足首日买入与当周清仓，事前跳过。", "",
              recovery_description + "当日仍失败则跨日真实保留并审计；不按后来是否成功来删除首腿。"
              "周末底仓退出失败也继续真实估值和重试，不能假定已清仓。"
              "每日卖出消耗开盘可卖库存，任何当日买入均不增加可卖额度。", "",
              f"费用：单边佣金{cfg['commission_rate']:.4%}、最低{cfg['min_commission']:g}元，"
              f"双边过户费{cfg['transfer_rate']:.4%}，印花税按2023-08-28前{cfg['stamp_rate_before_20230828']:.2%}、"
              f"之后{cfg['stamp_rate_from_20230828']:.2%}单边卖出征收；单边滑点{cfg['slippage_rate']:.2%}，"
              f"单分钟成交金额参与上限{cfg['max_bar_amount_fraction']:.0%}。", "",
              "## 训练、交叉验证和限制", "",
              "每折外层留一年，其余年份训练。内部最新训练年选模型和阈值，核心训练用另两个年份。"
              "内部只用模型样本外预测筛候选，再用完整成交引擎选择净增量；测试前冻结。"
              "最终模型在全部训练年重拟合，固定阈值不因外层预测重新校准。特征筛选只用相应训练数据；"
              "训练特征时间和标签起止均须属于训练年，并对留出边界施加7日隔离。", "",
              "2023–2025留一年折的训练含后续年份，是回顾性年度迁移检验；2026折训练止于2025。"
              "原始数据曾用于其他研究，2026不声称是从未查看过的全新测试集。"
              "每年独立重置资金，跨年表不应连乘为实盘连续净值。周bootstrap按周独立抽样，"
              "未完全建模跨周序列相关，也不校正历次研究的选择偏差。", "",
              "数据截至2026-04-30，该周5月1日休市，因此4月30日是当周最后交易日。"
              "原始分钟数据缺完整历史ST状态、精确涨跌停价、盘口队列和公司行动现金分红/复权校准；"
              "这些限制可能影响收益及可成交性。正常收盘集合竞价的单一价不等同一字涨跌停。", "",
              "制度依据：[上交所2026交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)、"
              "[2026劳动节休市公告](https://www.sse.com.cn/disclosure/announcement/general/c/c_20260423_10816345.shtml)、"
              "[印花税减半公告](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm)。", ""]
    if cfg.get("research_version") == "weekly_t_v3":
        lines += ["## 本轮研究变更", "",
                  f"本轮版本{cfg['research_version']}增加第N周所有交易日收盘分钟成交额最低"
                  f"{cfg.get('min_prior_close_auction_amount', 0):,.0f}元的事前股票池门槛，"
                  "缺少过去有效收盘分钟的股票不满足该门槛。它无法保证未来收盘必能成交。"
                  "盘中增加已知隔夜缺口、当日涨跌、相对同期市场及前日振幅尺度特征；"
                  "信号当前已在恢复方向涨跌停代理区域时不启动该方向，稍后解除时可以产生信号。"
                  "同一执行分钟按预测回报绝对值分配有限现金，平手按股票代码。", "",
                  "v2外层结果已被查看，其执行失败分析促成了本轮变更。每一折的模型和阈值仍只在训练年"
                  "选择，但这次复测不是未经研究选择的全新独立测试。所有旧结果保留；"
                  "最终证明增益仍需要未参与研究的新数据。", ""]
    return "\n".join(lines)


def run_cross_validation(weekly: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                         manifest: dict, cfg: dict, directory: Path,
                         recovery: pd.DataFrame | None = None) -> None:
    """所有模型和阈值仅从训练年产生，每折冻结后才运行外层年份，保留所有候选结果。"""
    if cfg.get("use_minute_recovery") and recovery is None:
        raise ValueError("use_minute_recovery开启时必须提供真实尾盘分钟缓存")
    if cfg.get("use_weekly_execution_features", False):
        weekly = add_weekly_execution_features(weekly, daily)
    if cfg.get("add_session_context", False):
        intraday = add_session_context_features(intraday)
    manifest = {**manifest, "runtime_weekly_features": [c for c in weekly if c.startswith("f_")],
                "runtime_intraday_features": [c for c in intraday if c.startswith("f_")],
                "runtime_feature_config": {key: cfg.get(key) for key in ("use_weekly_execution_features", "add_session_context",
                    "min_prior_close_auction_amount", "t_avoid_locked_exit_side", "t_signal_priority")}}
    universe = select_weekly_universe(weekly, cfg)
    t_training = intraday.merge(universe[["trade_week", "symbol"]],
                                 on=["trade_week", "symbol"], how="inner", validate="many_to_one")
    write_json(directory / "data_manifest.json", manifest)
    write_json(directory / "research_protocol.json", {
        "created_before_this_run_outer_test": True, "annual_folds": cfg["test_years"],
        "research_version": cfg.get("research_version", "weekly_t_v2"),
        "previous_outer_results_seen": cfg.get("research_version") == "weekly_t_v3",
        "minute_recovery_enabled": bool(cfg.get("use_minute_recovery")),
        "runtime_feature_config": manifest["runtime_feature_config"],
        "profiles": cfg["t_model_profiles"], "thresholds": cfg["t_thresholds"],
        "directions": cfg["t_directions"], "fractions": cfg["t_trade_fractions"],
        "selection": "all_inner_candidates_incremental_net_pnl_with_zero_t_inventory_failures",
        "gain_criterion": "aggregate_net_positive_and_majority_positive_and_bootstrap_low_positive_and_inventory_legal_and_weekly_exits_complete",
        "weekly_target": "0.70_return_prediction_rank+0.30_amplitude_prediction_rank"})
    # CLI --test-years只缩减要执行的外层折，不应该缩减可训练年份。
    all_years = sorted(set(weekly.entry_date.dropna().dt.year))
    summaries, results, validations, splits = [], [], [], []
    for year in cfg["test_years"]:
        train_years = [y for y in all_years if y != year]
        inner_year = max(train_years)
        core_years = [y for y in train_years if y != inner_year]
        fold = {"test_year": year, "train_years": train_years, "inner_year": inner_year,
                "core_years": core_years, "forward_only": max(train_years) < year}
        splits.append(fold)
        out = directory / f"year_{year}"
        out.mkdir()
        write_json(out / "split.json", fold)
        print(f"[{year}] 核心训练{core_years}，内部选择{inner_year}，外层训练{train_years}", flush=True)
        inner_weekly = fit_weekly_models(universe, core_years, cfg, out / "inner_weekly_model",
                                         excluded_years=[year, inner_year])
        inner_selections, inner_week_predictions = create_selections(
            inner_weekly, select_year_rows(universe, inner_year, cfg), cfg)
        write_frame(inner_selections, out / "validation_weekly_selections.csv")
        inner_daily, inner_intraday = select_execution_data(inner_selections, daily, intraday)
        inner_recovery = create_fold_recovery_store(inner_selections, recovery, inner_daily)
        prediction_sets, proxy_rows = {}, []
        for profile in cfg["t_model_profiles"]:
            profile_cfg = {**cfg, "t_model_profile": profile}
            model = fit_t_model(t_training, core_years, profile_cfg, out / f"inner_t_{profile}",
                                excluded_years=[year, inner_year])
            preds = predict_t(model, inner_intraday, profile_cfg)
            prediction_sets[profile] = preds
            proxy_rows.extend(calculate_proxy_candidates(preds, inner_intraday, profile, cfg))
        proxy = pd.DataFrame(proxy_rows).sort_values("proxy_net_return_sum", ascending=False)
        write_frame(proxy, out / "validation_proxy_candidates.csv")
        shortlist = proxy.head(cfg["candidate_shortlist_size"]) if cfg["candidate_shortlist_size"] else proxy
        candidates, eligible, tasks, rule_tasks, seen = [], [], [], [], set()
        candidate_audits = out / "validation_candidate_audits"
        candidate_audits.mkdir()
        for index, candidate in enumerate(shortlist.to_dict("records")):
            rule = {k: candidate[k] for k in ("enabled", "profile", "threshold", "direction", "fraction")}
            decisions = create_decisions(prediction_sets[rule["profile"]], rule, cfg)
            decision_hash = hashlib.sha256(pd.util.hash_pandas_object(decisions, index=False).to_numpy().tobytes()).hexdigest()
            rule_tasks.append((index, rule, decision_hash))
            if decision_hash not in seen:
                tasks.append({"candidate_index": index, "decision_sha256": decision_hash, "decisions": decisions})
                seen.add(decision_hash)
        evaluated = run_candidate_evaluations(tasks, inner_selections, inner_daily, inner_intraday,
                                              cfg, candidate_audits, recovery_store=inner_recovery)
        for index, rule, decision_hash in rule_tasks:
            evaluation = evaluated[decision_hash]
            increment, metrics = evaluation["incremental_net_pnl"], evaluation["metrics"]
            original_index = evaluation["original_index"]
            failures = sum(int(metrics.get(k, 0)) for k in (
                "inventory_mismatch_days", "recovery_failure_days", "pending_pairs", "base_plan_mismatch_count", "t1_violation_count"))
            row = {"test_year": year, "inner_year": inner_year, **rule, **metrics,
                   "incremental_net_pnl": increment, "strict_eligible": failures == 0,
                   "selected": False, "candidate_index": index, "decision_sha256": decision_hash,
                   "same_decisions_as_candidate": original_index}
            candidates.append(row)
            if failures == 0 and increment > 0:
                eligible.append((increment, index, rule))
            print(f"[{year}] 内部候选 {rule}: 增量{increment:.2f},约束失败{failures}", flush=True)
        if eligible:
            _, chosen_index, rule = max(eligible, key=lambda item: (item[0], -item[1]))
            candidates[chosen_index]["selected"] = True
        else:
            rule = {"enabled": False, "profile": str(shortlist.iloc[0].profile),
                    "threshold": float(shortlist.iloc[0].threshold), "direction": "sell_first", "fraction": 1.0,
                    "training_failure": "内部候选无同时盈利和库存恢复合格者"}
        validations.extend(candidates)
        write_frame(pd.DataFrame(candidates), out / "validation_rule_metrics.csv")
        write_json(out / "selected_rule.json", {**rule, "frozen_before_outer_test": True, "fold": fold})
        print(f"[{year}] 已冻结规则 {rule}", flush=True)
        outer_weekly = fit_weekly_models(universe, train_years, cfg, out / "weekly_model", excluded_years=[year])
        outer_selections, outer_week_predictions = create_selections(
            outer_weekly, select_year_rows(universe, year, cfg), cfg)
        write_frame(outer_week_predictions, out / "weekly_predictions.csv")
        write_frame(outer_selections, out / "weekly_selections.csv")
        outer_daily, outer_intraday = select_execution_data(outer_selections, daily, intraday)
        outer_recovery = create_fold_recovery_store(outer_selections, recovery, outer_daily)
        profile_cfg = {**cfg, "t_model_profile": rule["profile"]}
        outer_t = fit_t_model(t_training, train_years, profile_cfg, out / "t_model", excluded_years=[year])
        t_predictions = predict_t(outer_t, outer_intraday, profile_cfg)
        write_frame(t_predictions, out / "t_predictions.csv")
        decisions = create_decisions(t_predictions, rule, cfg)
        write_frame(decisions, out / "t_decisions.csv")
        result = run_weekly_portfolio(outer_selections, outer_daily, outer_intraday, decisions, cfg,
                                      recovery_bars=outer_recovery)
        metrics = save_result(result, out, cfg)
        prediction_metrics = calculate_prediction_metrics(outer_week_predictions, select_year_rows(universe, year, cfg))
        write_frame(prediction_metrics, out / "weekly_prediction_metrics.csv")
        metrics.update({"weekly_return_rank_ic": float(prediction_metrics.return_rank_ic.mean()),
                        "weekly_amplitude_rank_ic": float(prediction_metrics.amplitude_rank_ic.mean()),
                        "selected_mean_weekly_return": float(prediction_metrics.selected_mean_return.mean()),
                        "selected_mean_daily_amplitude": float(prediction_metrics.selected_mean_amplitude.mean()),
                        "mean_selected_stocks": float(prediction_metrics.selected_count.mean())})
        write_frame(pd.DataFrame([metrics]), out / "test_metrics.csv")
        summaries.append({"test_year": year, "inner_year": inner_year, **metrics, "t_enabled": rule["enabled"]})
        results.append((year, result))
        print(f"[{year}] 留出年持有{metrics['baseline_total_return']:.3%}, 加T{metrics['total_return']:.3%}, 净增量{metrics['t_added_equity']:.2f}", flush=True)
    summary = pd.DataFrame(summaries)
    write_frame(summary, directory / "annual_summary.csv")
    write_frame(summary, directory / "test_metrics.csv")
    write_frame(pd.DataFrame(validations), directory / "validation_rule_metrics.csv")
    write_frame(pd.DataFrame(splits), directory / "split_audit.csv")
    failures = sum(summary.get(key, pd.Series(0)).fillna(0).sum() for key in (
        "inventory_mismatch_days", "recovery_failure_days", "pending_pairs", "base_plan_mismatch_count", "t1_violation_count"))
    increments = np.concatenate([calculate_weekly_increment(r["equity_curve"], cfg["initial_cash"], cfg)[0].t_increment.to_numpy()
                                 for _, r in results])
    rng = np.random.default_rng(cfg["random_state"])
    bootstrap = rng.choice(increments, size=(cfg["bootstrap_samples"], len(increments)), replace=True).sum(axis=1)
    aggregate_low, aggregate_high = np.quantile(bootstrap, [.025, .975])
    t_gain_met = bool(summary.t_added_equity.sum() > 0 and (summary.increment_return > 0).sum() > len(summary) / 2
                      and failures == 0 and aggregate_low > 0)
    weekly_failures = int(summary.get("weekly_liquidation_failure_days", pd.Series(0)).sum())
    weekly_complete = bool(weekly_failures == 0 and summary.get("weekly_unclosed_positions", pd.Series(0)).sum() == 0
                           and summary.get("baseline_weekly_unclosed_positions", pd.Series(0)).sum() == 0)
    status = {"gain_criterion_met": bool(t_gain_met and weekly_complete), "t_gain_criterion_met": t_gain_met,
              "weekly_execution_complete": weekly_complete, "weekly_exit_failure_days": weekly_failures,
              "aggregate_increment_rmb": float(summary.t_added_equity.sum()),
              "aggregate_increment_ci_low_rmb": float(aggregate_low), "aggregate_increment_ci_high_rmb": float(aggregate_high),
              "positive_years": int((summary.increment_return > 0).sum()), "constraint_failures": int(failures)}
    write_json(directory / "goal_evaluation.json", status)
    for key, filename in (("equity_curve", "test_equity_curve.csv"), ("trade_events", "test_trade_events.csv"),
                          ("round_trips", "test_round_trips.csv"), ("open_positions", "test_open_positions.csv")):
        write_frame(pd.concat([r[key].assign(test_year=y) for y, r in results], ignore_index=True), directory / filename)
    plot_results(results, directory / "test_return_curve.png", cfg["initial_cash"])
    shutil.copyfile(directory / "test_return_curve.png", directory / "测试集收益曲线.png")
    (directory / "final_report.md").write_text(create_report(summary, cfg, status), encoding="utf-8")
    (directory / "输出文件说明.md").write_text(
        "# 输出文件说明\n\nfinal_report.md是主报告；annual_summary.csv/test_metrics.csv是年度比较；"
        "test_equity_curve.csv、test_trade_events.csv、test_round_trips.csv、test_open_positions.csv为各年独立结果带test_year汇总。"
        "test_return_curve.png/测试集收益曲线.png是图；validation_rule_metrics.csv是内部选择。\n\n"
        "每个year目录保存weekly_predictions.csv全部候选预测、weekly_selections.csv前100、t_predictions.csv和t_decisions.csv盘中预测/请求、"
        "baseline_trade_events.csv共同基准成交、daily_inventory_audit.csv库存、blocked_orders.csv受阻单、weekly_increment.csv周增量、"
        "execution_audit.json成交检查、selected_rule.json事前冻结规则、split.json年度协议和标准结果CSV。"
        "validation_candidate_audits保留每组不同决策的真实受阻单和库存失败；"
        "分批恢复在test_round_trips.csv中以相同pair_id的多个fragment_id记录，不应把片段数当完整做T次数。"
        "各model目录保存模型、特征和训练来源；inner开头表示只用于内部参数选择。\n\n"
        "resolved_config.json、data_manifest.json、recovery_manifest.json（启用尾盘分钟时）、research_protocol.json、"
        "split_audit.csv、source_hashes.json和source_snapshot/用于复现；"
        "goal_evaluation.json记录事前净增益标准是否满足。所有旧报告保留。\n", encoding="utf-8")


def main() -> None:
    """创建唯一运行目录，先保存配置/源码再读取数据；cache-only也记录数据指纹。"""
    args = parse_arguments()
    cfg = load_configuration(args)
    directory = Path(cfg["output_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True)
    write_json(directory / "resolved_config.json", cfg)
    sources = [Path(__file__), *[PROJECT_ROOT / "quant_backtest" / name for name in
                               ("weekly_t_data.py", "weekly_t_models.py", "weekly_t_engine.py", "weekly_t_features.py",
                                "weekly_t_recovery_data.py", "weekly_t_validation.py", "annual_portfolio.py", "inventory_t.py")]]
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
        hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(directory / "source_hashes.json", hashes)
    print(f"输出目录：{directory}", flush=True)
    weekly, intraday, daily, manifest = load_weekly_t_data(cfg)
    recovery = None
    if cfg.get("use_minute_recovery"):
        recovery, recovery_manifest = load_recovery_data(cfg)
        write_json(directory / "recovery_manifest.json", recovery_manifest)
        print(f"尾盘分钟缓存：{len(recovery):,}条真实行情", flush=True)
    if args.cache_only:
        write_json(directory / "data_manifest.json", manifest)
        print(f"缓存完成：weekly={len(weekly):,}, intraday={len(intraday):,}, daily={len(daily):,}", flush=True)
        return
    run_cross_validation(weekly, intraday, daily, manifest, cfg, directory, recovery=recovery)
    print(f"报告完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
