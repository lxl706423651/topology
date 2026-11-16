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
from quant_backtest.weekly_t_engine import run_weekly_portfolio
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
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/weekly_t_cross_validation.yaml")
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
    start, end = selections.entry_date.min(), selections.exit_date.max()
    symbols = selections.symbol.unique()
    d = daily.loc[daily.symbol.isin(symbols) & daily.date.between(start, end)].copy()
    columns = ["trade_week", "symbol", "entry_date", "exit_date"]
    i = intraday.loc[intraday.date.between(start, end)].merge(
        selections[columns], on=["trade_week", "symbol"], how="inner", validate="many_to_one")
    i = i.loc[(i.date > i.entry_date) & (i.date < i.exit_date)].copy()
    return d, i


def create_decisions(predictions: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """把冻结模型预测变成请求；保留每股每日最早满足阈值的一次信号。"""
    signals = WeeklyLearnedTStrategy(rule).generate_signals(predictions)
    decisions = predictions[["datetime", "symbol"]].copy()
    decisions["action"] = signals.t_action
    decisions["fraction"] = signals.fraction
    decisions = decisions.loc[decisions.action != 0].sort_values(["datetime", "symbol"])
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
                    "direction": direction, "fraction": cfg["t_fraction"]}
            decisions = create_decisions(predictions, rule)
            matched = decisions.merge(intraday[["datetime", "symbol", "execution_price", "recovery_price"]],
                                      on=["datetime", "symbol"], validate="one_to_one")
            future = matched.recovery_price / matched.execution_price - 1
            stamp = np.where(matched.datetime < pd.Timestamp("2023-08-28"),
                             cfg["stamp_rate_before_20230828"], cfg["stamp_rate_from_20230828"])
            cost = 2 * (cfg["commission_rate"] + cfg["transfer_rate"] + cfg["slippage_rate"]) + stamp
            net = (matched.action * future - cost).where(future.notna(), -0.10)
            rows.append({**rule, "proxy_pairs": len(matched), "proxy_net_return_sum": float(net.sum()),
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
    """核查已成交事件的整手、现金、T+1预算及分钟延迟；失败立即停止报告发布。"""
    events = result["trade_events"]
    audit = {"integer_lots": True, "nonnegative_cash": True, "t1_legal": True}
    if len(events):
        audit["integer_lots"] = bool((events.shares % cfg["lot_size"] == 0).all())
        if "cash_after" in events:
            audit["nonnegative_cash"] = bool((events.cash_after >= -1e-6).all())
        if "sellable_before" in events:
            sales = events.loc[events.side == "sell"]
            audit["t1_legal"] = bool((sales.shares <= sales.sellable_before).all())
    if not all(audit.values()):
        raise AssertionError(f"成交不变量失败: {audit}")
    audit["metrics"] = result["metrics"]
    return audit


def save_result(result: dict, directory: Path, cfg: dict) -> dict:
    """保存逐日组合、成交、库存和周增量，并计算统一净收益差与置信区间。"""
    mapping = {"equity_curve": "test_equity_curve.csv", "trade_events": "test_trade_events.csv",
               "round_trips": "test_round_trips.csv", "open_positions": "test_open_positions.csv",
               "daily_inventory_audit": "daily_inventory_audit.csv", "order_attempts": "blocked_orders.csv",
               "baseline_events": "baseline_trade_events.csv"}
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
    lines = ["# 周度前100与学习型做T年度交叉验证", "",
             f"净增益检验：**{'达到预声明检验标准' if status['gain_criterion_met'] else '尚未达到预声明检验标准'}**。",
             "标准为累计净增量为正、严格多数留出年增益为正，并且没有T+1、库存恢复或共同底仓差异。"
             "统计置信区间另行报告；正收益或多数年份改善不等于未来收益保证。", "",
             "|留出年|周度持有收益|加入T收益|净增益百分点|净增益元|周bootstrap 95%区间（百分点）|",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"|{r.test_year}|{r.baseline_total_return:.2%}|{r.total_return:.2%}|"
                     f"{r.increment_return*100:+.3f}|{r.t_added_equity:+,.2f}|"
                     f"[{r.increment_ci_low*100:+.3f}, {r.increment_ci_high*100:+.3f}]|")
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
              "14:55恢复失败尝试收盘，仍失败则跨日真实保留并审计；不按后来是否成功来删除首腿。"
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
    return "\n".join(lines)


def run_cross_validation(weekly: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                         manifest: dict, cfg: dict, directory: Path) -> None:
    """所有模型和阈值仅从训练年产生，每折冻结后才运行外层年份，保留所有候选结果。"""
    universe = select_weekly_universe(weekly, cfg)
    t_training = intraday.merge(universe[["trade_week", "symbol"]],
                                 on=["trade_week", "symbol"], how="inner", validate="many_to_one")
    write_json(directory / "data_manifest.json", manifest)
    write_json(directory / "research_protocol.json", {
        "created_before_outer_test": True, "annual_folds": cfg["test_years"],
        "profiles": cfg["t_model_profiles"], "thresholds": cfg["t_thresholds"],
        "directions": cfg["t_directions"], "selection": "inner_incremental_net_pnl_with_zero_inventory_failures",
        "gain_criterion": "aggregate_net_positive_and_strict_majority_positive_and_inventory_legal",
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
        shortlist = proxy.head(cfg["candidate_shortlist_size"])
        candidates, eligible = [], []
        for index, candidate in enumerate(shortlist.to_dict("records")):
            rule = {k: candidate[k] for k in ("enabled", "profile", "threshold", "direction", "fraction")}
            decisions = create_decisions(prediction_sets[rule["profile"]], rule)
            result = run_weekly_portfolio(inner_selections, inner_daily, inner_intraday, decisions, cfg)
            validate_execution(result, cfg)
            curve = result["equity_curve"]
            increment = float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1])
            metrics = result["metrics"]
            failures = sum(int(metrics.get(k, 0)) for k in (
                "inventory_mismatch_days", "recovery_failure_days", "pending_pairs", "base_plan_mismatch_count", "t1_violation_count"))
            row = {"test_year": year, "inner_year": inner_year, **rule, **metrics,
                   "incremental_net_pnl": increment, "strict_eligible": failures == 0,
                   "selected": False, "candidate_index": index}
            candidates.append(row)
            if failures == 0 and increment > 0:
                eligible.append((increment, index, rule))
            print(f"[{year}] 内部候选 {rule}: 增量{increment:.2f},约束失败{failures}", flush=True)
        if eligible:
            _, chosen_index, rule = max(eligible, key=lambda item: (item[0], -item[1]))
            candidates[chosen_index]["selected"] = True
        else:
            rule = {"enabled": False, "profile": str(shortlist.iloc[0].profile),
                    "threshold": float(shortlist.iloc[0].threshold), "direction": "sell_first", "fraction": cfg["t_fraction"],
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
        profile_cfg = {**cfg, "t_model_profile": rule["profile"]}
        outer_t = fit_t_model(t_training, train_years, profile_cfg, out / "t_model", excluded_years=[year])
        t_predictions = predict_t(outer_t, outer_intraday, profile_cfg)
        write_frame(t_predictions, out / "t_predictions.csv")
        decisions = create_decisions(t_predictions, rule)
        write_frame(decisions, out / "t_decisions.csv")
        result = run_weekly_portfolio(outer_selections, outer_daily, outer_intraday, decisions, cfg)
        metrics = save_result(result, out, cfg)
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
    status = {"gain_criterion_met": bool(summary.t_added_equity.sum() > 0 and
              (summary.increment_return > 0).sum() > len(summary) / 2 and failures == 0),
              "aggregate_increment_rmb": float(summary.t_added_equity.sum()),
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
        "各model目录保存模型、特征和训练来源；inner开头表示只用于内部参数选择。\n\n"
        "resolved_config.json、data_manifest.json、research_protocol.json、split_audit.csv、source_snapshot/用于复现；"
        "goal_evaluation.json记录事前净增益标准是否满足。所有旧报告保留。\n", encoding="utf-8")


def main() -> None:
    """创建唯一运行目录，先保存配置/源码再读取数据；cache-only也记录数据指纹。"""
    args = parse_arguments()
    cfg = load_configuration(args)
    directory = Path(cfg["output_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True)
    write_json(directory / "resolved_config.json", cfg)
    sources = [Path(__file__), *[PROJECT_ROOT / "quant_backtest" / name for name in
                               ("weekly_t_data.py", "weekly_t_models.py", "weekly_t_engine.py", "annual_portfolio.py")]]
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
        hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(directory / "source_hashes.json", hashes)
    print(f"输出目录：{directory}", flush=True)
    weekly, intraday, daily, manifest = load_weekly_t_data(cfg)
    if args.cache_only:
        write_json(directory / "data_manifest.json", manifest)
        print(f"缓存完成：weekly={len(weekly):,}, intraday={len(intraday):,}, daily={len(daily):,}", flush=True)
        return
    run_cross_validation(weekly, intraday, daily, manifest, cfg, directory)
    print(f"报告完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
