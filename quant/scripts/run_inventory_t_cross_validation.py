"""运行固定底仓做T的嵌套年度回测，输入本地分钟行情及YAML，输出独立年度报告。

从quant项目目录运行；复用年度因子缓存，训练各折XGBoost及做T参数，仅在最后
评估留出年。原始数据不变，不接券商，不清空底仓，不将各年重置组合拼成实盘净值。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.annual_panel import load_annual_panel, load_symbol_bars, write_json
from quant_backtest.annual_portfolio import calculate_fee
from quant_backtest.inventory_t import calculate_next_trading_minute, run_inventory_t
from quant_backtest.inventory_t_data import apply_signal_eligibility, load_signal_eligibility
from quant_backtest.inventory_t_models import (
    build_year_folds, calibrate_entry_gate, fit_entry_model, predict_entry_scores,
    select_entry_candidates,
)


def parse_arguments() -> argparse.Namespace:
    """读取YAML位置和一次性实验覆盖项；smoke只缩短回测、模型规模和特征数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/inventory_t_cross_validation.yaml")
    for name in ("workers", "max-symbols", "universe-size", "min-cross-section", "smoke-days"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-years", help="只运行指定留出年，例如2026或2023,2024,2025,2026")
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> dict:
    """加载完整YAML并应用显式CLI覆盖；校验资金、候选与路径，保存时保留最终值。"""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("workers", "max_symbols", "universe_size", "min_cross_section", "smoke_days", "output_dir"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    if args.test_years:
        cfg["test_years"] = [int(year) for year in args.test_years.split(",")]
    for key in ("data_root", "cache_dir", "signal_cache_dir", "name_snapshot", "output_dir", "manual_buys_csv"):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str(PROJECT_ROOT / cfg[key])
    if not 0 < cfg["entry_equity_fraction"] <= 1 or cfg["max_base_symbols"] < 1:
        raise ValueError("开仓占比或底仓股票数非法")
    if cfg["entry_equity_fraction"] * cfg["max_base_symbols"] >= 1:
        raise ValueError("此策略要求预留现金，entry_equity_fraction * max_base_symbols须小于1")
    if cfg["selection_metric"] != "incremental_net_pnl":
        raise ValueError("此版本仅支持预先声明的incremental_net_pnl选择准则")
    if cfg["smoke_days"]:
        cfg["model"]["n_estimators"] = min(cfg["model"]["n_estimators"], 20)
        cfg["max_features"] = min(cfg["max_features"], 12)
    return cfg


class MinuteBarStore:
    """只加载实际底仓候选的完整分钟行情；复用内存且不重建全市场分钟因子。"""

    def __init__(self, cfg: dict):
        """cfg提供只读原始路径、起止日期；audit保存每只加载股票的去重和非法行统计。"""
        self.cfg = cfg
        self.frames: dict[str, pd.DataFrame] = {}
        self.audit: dict[str, dict] = {}

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """加载单只股票；时间索引用于实际成交与历史估值，不向未来填价。"""
        if symbol not in self.frames:
            bars, audit = load_symbol_bars(Path(self.cfg["data_root"]), symbol,
                                            self.cfg["start_date"], self.cfg["end_date"])
            self.frames[symbol] = bars.set_index("datetime", drop=False)
            self.audit[symbol] = audit
        return self.frames[symbol]

    def load_slice(self, symbols: list[str], dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        """返回symbols在dates内的真实分钟行；没有行情的日期仍由引擎交易日时钟保留。"""
        return {symbol: self.load_symbol(symbol).loc[
            self.load_symbol(symbol).index.normalize().isin(dates)].reset_index(drop=True)
                for symbol in symbols}


def select_year_dates(market: pd.DataFrame, year: int, cfg: dict) -> pd.DatetimeIndex:
    """从共同时钟行情提取year所有交易日；smoke可只取开头N天，不能按未来收益筛日。"""
    dates = pd.DatetimeIndex(market.loc[market.datetime.dt.year == year, "datetime"].dt.normalize().unique()).sort_values()
    return dates[:cfg["smoke_days"]] if cfg["smoke_days"] else dates


def create_base_entries(scores: pd.DataFrame, gate: dict, dates: pd.DatetimeIndex,
                        cfg: dict, store: MinuteBarStore) -> pd.DataFrame:
    """按旧top5结构建立固定底仓，冻结同一买单用于所有T参数与持有基准。

单票仅首次买入，按当时纯底仓组合权益10%分配，最多5只；不因退出周池卖出。
人工CSV是指定时刻从原现金买入的额外交易，不是无成本股份或外部现金注入。
"""
    columns = ["datetime", "signal_datetime", "symbol", "shares", "price", "reason", "entry_score"]
    candidates = select_entry_candidates(scores.loc[scores.datetime.dt.normalize().isin(dates)], gate, cfg)
    cash, positions, rows = float(cfg["initial_cash"]), {}, []
    lot = int(cfg["lot_size"])
    for row in candidates.itertuples(index=False):
        if len(positions) >= cfg["max_base_symbols"]:
            break
        symbol, signal = str(row.symbol), pd.Timestamp(row.datetime)
        if symbol in positions:
            continue
        timestamp = calculate_next_trading_minute(signal)
        if timestamp is None or timestamp.normalize() not in dates:
            continue
        data = store.load_symbol(symbol)
        if timestamp not in data.index:
            continue
        execution = data.loc[timestamp]
        if execution.volume <= 0 or execution.high == execution.low:
            continue
        # sizing只看信号时刻的已完成价格；实际成交容量/价格在下一分钟校验。
        equity = cash
        for held, shares in positions.items():
            history = store.load_symbol(held).loc[:signal]
            equity += shares * float(history.close.iloc[-1])
        signal_history = data.loc[:signal]
        if signal_history.empty:
            continue
        reference = float(execution.close)
        price = reference * (1 + cfg["slippage_rate"])
        signal_price = float(signal_history.close.iloc[-1]) * (1 + cfg["slippage_rate"])
        requested = int(equity * cfg["entry_equity_fraction"] / signal_price / lot) * lot
        capacity = int(float(execution.amount) * cfg["max_bar_amount_fraction"] / price / lot) * lot
        quantity = min(requested, capacity)
        while quantity > 0 and quantity * price + calculate_fee(quantity * price, "buy", timestamp, cfg) > cash:
            quantity -= lot
        if quantity <= 0:
            continue
        cash -= quantity * price + calculate_fee(quantity * price, "buy", timestamp, cfg)
        positions[symbol] = quantity
        rows.append({"datetime": timestamp, "signal_datetime": signal, "symbol": symbol,
                     "shares": quantity, "price": reference, "reason": "model_base_entry",
                     "entry_score": float(row.score)})
    if cfg.get("manual_buys_csv"):
        manual = pd.read_csv(cfg["manual_buys_csv"], dtype={"symbol": str})
        if not {"datetime", "symbol", "shares"}.issubset(manual):
            raise ValueError("manual_buys_csv须有datetime,symbol,shares列")
        manual["datetime"] = pd.to_datetime(manual.datetime)
        manual = manual.loc[manual.datetime.dt.normalize().isin(dates)]
        for row in manual.itertuples(index=False):
            quantity = float(row.shares)
            if not np.isfinite(quantity) or quantity <= 0 or quantity % lot:
                raise ValueError("人工加仓股数须为正整手")
            data = store.load_symbol(row.symbol)
            if row.datetime not in data.index:
                raise ValueError(f"人工买单缺少实际行情: {row.symbol} {row.datetime}")
            rows.append({"datetime": row.datetime, "signal_datetime": row.datetime - pd.Timedelta(minutes=1),
                         "symbol": row.symbol, "shares": int(quantity), "price": float(data.loc[row.datetime, "close"]),
                         "reason": "manual_base_addition", "entry_score": np.nan})
    schedule = pd.DataFrame(rows, columns=columns).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    baseline_budget = float(cfg["initial_cash"])
    for row in schedule.itertuples(index=False):
        value = row.shares * row.price * (1 + cfg["slippage_rate"])
        baseline_budget -= value + calculate_fee(value, "buy", row.datetime, cfg)
        if baseline_budget < -1e-6:
            raise ValueError("底仓及人工加仓计划超过持有基准现金；请减少加仓股数，不能依靠未来做T利润或借款建立基准")
    return schedule


def create_t_rules(cfg: dict) -> list[dict]:
    """由预先声明的偏离率、交易比例网格生成候选，测试收益不能改变候选集合。"""
    rules = [{"name": f"vwap_{deviation:g}_fraction_{fraction:g}", "enabled": True,
              "entry_deviation": float(deviation), "trade_fraction": float(fraction),
              "take_profit": cfg["t_take_profit"], "direction": cfg["t_direction"]}
             for deviation in cfg["t_entry_deviations"] for fraction in cfg["t_trade_fractions"]]
    if cfg["include_hold_candidate"]:
        rules.append({"name": "hold_only", "enabled": False})
    if not rules:
        raise ValueError("做T候选不能为空")
    return rules


def create_baseline_curve(curve: pd.DataFrame, initial: float) -> pd.DataFrame:
    """从并行记账得到相同初始买单的纯持有净值，保留全部日历与首笔费用。"""
    baseline = curve[["datetime", "baseline_equity"]].rename(columns={"baseline_equity": "equity"}).copy()
    baseline["return"] = baseline.equity.pct_change(fill_method=None)
    baseline.loc[0, "return"] = baseline.equity.iloc[0] / initial - 1
    baseline["net_value"] = baseline.equity / initial
    baseline["cumulative_return"] = baseline.net_value - 1
    baseline["drawdown"] = baseline.equity / np.maximum.accumulate(np.r_[initial, baseline.equity])[1:] - 1
    return baseline


def audit_result(result: dict, entries: pd.DataFrame, cfg: dict) -> dict:
    """检查T+1预算、现金、整手、执行延迟和配对现金损益，恢复失败是显式结果而非删日。"""
    events, inventory, curve = result["trade_events"], result["daily_inventory_audit"], result["equity_curve"]
    if len(events):
        assert (events.shares % cfg["lot_size"] == 0).all(), "非整手交易"
        assert (pd.to_datetime(events.datetime) > pd.to_datetime(events.signal_datetime)).all(), "同bar成交"
        assert events.cash_after.min() >= -1e-6, "交易后现金为负"
        sales = events.loc[events.side == "sell"]
        assert (sales.sellable_after >= 0).all(), "超出可卖额度"
    if len(inventory):
        assert (inventory.day_sold <= inventory.opening_sellable).all(), "当日新买股票被重复卖出"
        assert inventory.t1_violation_count.sum() == 0, "T+1违规"
    assert curve.cash.min() >= -1e-6
    metrics = result["metrics"]
    if not metrics["pending_pairs"]:
        assert np.isclose(metrics["total_t_net_profit"], metrics["total_t_marked_pnl"], atol=1e-5), "做T现金损益和同底仓增量不一致"
    return {"t1_budget_passed": True, "lot_size_passed": True, "positive_cash_passed": True,
            "delayed_execution_passed": True, "same_base_schedule": True,
            "base_entry_count": len(entries), "inventory_mismatch_days": metrics["inventory_mismatch_days"],
            "recovery_failure_days": metrics["recovery_failure_days"], "pending_pairs": metrics["pending_pairs"],
            "interpretation": "T配对可同日，合法性以当日全部卖出<=日初可卖股数审计；不是卖出当天新买股票"}


def plot_return_curves(results: list[tuple[int, dict]], path: Path, initial: float) -> None:
    """绘制各留出年独立净值和做T增量，不把重置底仓的年度连接成连续策略。"""
    from matplotlib import font_manager
    # 本机字体可能在matplotlib缓存生成后才安装，显式注册系统字体避免中文方框。
    for font_path in font_manager.findSystemFonts():
        if any(name in Path(font_path).name.lower() for name in ("notosanscjk", "droidsansfallback", "wqy")):
            font_manager.fontManager.addfont(font_path)
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Droid Sans Fallback", "WenQuanYi Zen Hei", "DejaVu Sans"):
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(len(results), 2, figsize=(15, 4 * len(results)), squeeze=False)
    for axis_row, (year, result) in zip(axes, results):
        curve, metrics = result["equity_curve"], result["metrics"]
        axis_row[0].plot(curve.datetime, curve.equity / initial, label="底仓＋做T")
        axis_row[0].plot(curve.datetime, curve.baseline_equity / initial, label="相同底仓只持有", alpha=.8)
        axis_row[0].set_title(f"{year} 独立留出年 | 收益 {metrics['total_return']:.2%} | 年化 {metrics['annual_return']:.2%}\n"
                              f"Sharpe {metrics['sharpe']:.2f} | 回撤 {metrics['max_drawdown']:.2%} | 波动 {metrics['annual_volatility']:.2%}")
        axis_row[0].set_ylabel("净值（初始现金=1）")
        axis_row[0].legend()
        axis_row[1].plot(curve.datetime, curve.t_added_equity, color="tab:orange")
        axis_row[1].axhline(0, color="gray", lw=.7)
        axis_row[1].set_title(f"做T净增量 {metrics['total_t_marked_pnl']:,.0f}元 | 配对 {metrics['t_pair_count']}\n"
                              f"胜率 {metrics['win_rate']:.1%} | 交易 {metrics['trade_lots']:,.0f}手 | 未复原日 {metrics['recovery_failure_days']}")
        axis_row[1].set_ylabel("相对纯持有净增量（元）")
        for axis in axis_row:
            axis.grid(alpha=.2)
            axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    """写CSV并为全空无字段诊断表给出status列，避免生成无法读取的空文件。"""
    if not len(frame.columns):
        frame = pd.DataFrame(columns=["status"])
    frame.to_csv(path, index=False)


def save_fold_report(directory: Path, year: int, result: dict, entries: pd.DataFrame,
                     validation: pd.DataFrame, fold: dict, rule: dict, gate: dict, cfg: dict) -> None:
    """保存单年完整报告、逐笔T库存审计及模型选择上下文；测试仅使用已冻结的rule/gate。"""
    mapping = {"equity_curve": "test_equity_curve.csv", "trade_events": "test_trade_events.csv",
               "round_trips": "test_round_trips.csv", "open_positions": "test_open_positions.csv",
               "daily_inventory_audit": "daily_inventory_audit.csv", "order_attempts": "blocked_orders.csv"}
    for key, filename in mapping.items():
        write_frame(result[key], directory / filename)
    write_frame(create_baseline_curve(result["equity_curve"], cfg["initial_cash"]), directory / "baseline_equity_curve.csv")
    write_frame(entries, directory / "base_entry_schedule.csv")
    write_frame(validation, directory / "validation_rule_metrics.csv")
    write_frame(pd.DataFrame([result["metrics"]]), directory / "test_metrics.csv")
    audit = audit_result(result, entries, cfg)
    write_json(directory / "execution_audit.json", audit)
    write_json(directory / "resolved_config.json", cfg)
    write_json(directory / "selected_rule.json", {"fold": fold, "rule": rule, "gate": gate})
    plot_return_curves([(year, result)], directory / "测试集收益曲线.png", cfg["initial_cash"])
    shutil.copyfile(directory / "测试集收益曲线.png", directory / "test_return_curve.png")
    metrics = result["metrics"]
    text = (f"# {year}年固定底仓做T回测\n\n"
            f"训练年份：{fold['train_years']}；内部参数验证年：{fold['inner_validation_year']}。协议：{fold['protocol']}。\n\n"
            f"初始现金{cfg['initial_cash']:,.0f}元；实际区间{metrics['start']}至{metrics['end']}，{metrics['trading_days']}交易日。\n\n"
            f"只持有收益 **{metrics['baseline_total_return']:.2%}**；加做T后 **{metrics['total_return']:.2%}**；"
            f"做T净增量 **{metrics['total_t_marked_pnl']:,.2f}元**（初始资金的{metrics['total_t_marked_pnl']/cfg['initial_cash']:.2%}）。\n\n"
            f"已完成{metrics['t_pair_count']}组T，配对胜率{metrics['win_rate']:.2%}；"
            f"未恢复日{metrics['recovery_failure_days']}，股数变化日{metrics['inventory_mismatch_days']}，期末未完成组{metrics['pending_pairs']}。\n\n"
            f"冻结规则：`{rule['name']}`；训练校准开仓门槛{gate['threshold']:.8f}。"
            "若内部候选全部亏损，默认仍评估内部选出的实际做T规则并保留负结果。\n\n"
            + create_semantics_text(cfg))
    (directory / "final_report.md").write_text(text, encoding="utf-8")
    (directory / "输出文件说明.md").write_text(create_file_description(), encoding="utf-8")


def create_semantics_text(cfg: dict) -> str:
    """生成每份报告必须保留的T+1、收益口径及年度验证限制说明。"""
    return ("## 交易及验证口径\n\n"
            f"- 上一自然周高波动前{cfg['universe_size']}股票池；沿用旧CTA/XGBoost评分门控与top{cfg['entry_top_n']}开仓结构。"
            "每折重新训练，仅核心训练期选特征，用内部留出训练年的样本外预测校准门槛；不复用包含留出年训练信息的旧模型。"
            "新模型标签是次交易日相同时刻收益排名，旧模型标签为24分钟收益，因此本次保留开仓结构而非原固定模型数值。\n"
            f"- 信号时刻一字线过滤：{cfg['exclude_flat_signal_bars']}。用原始单股缓存的当前hl_range>0识别；"
            "过滤在训练和推断前完成，截面数量重新计算，不使用下一分钟行情挑选候选。"
            "因子截面秩与标签秩仍来自原周池。此保守规则也会排除部分实际可成交的普通平价分钟。\n"
            f"- 最多{cfg['max_base_symbols']}只初始底仓，每次按纯底仓组合当时权益{cfg['entry_equity_fraction']:.0%}配仓；"
            "之后不自动加仓、不清仓、不因出池卖出。人工买单从同一现金余额成交，另记新增库存，无外部注资。\n"
            "- 当日所有卖单合计不超过日初持股；买回不恢复当天可卖额度。先买后卖的卖出对象是昨日底仓。"
            "新开仓/人工加仓日目标为开盘股数＋新增股数，其他日目标为开盘股数。\n"
            "- 每个信号只读取已完成分钟，下一交易分钟收盘加滑点成交。"
            f"{cfg['last_entry_time']}后不发起第一腿；{cfg['flatten_signal_time']}发出恢复单，下一分钟起尝试恢复，未成交到收盘如实记录、跨日修复。\n"
            f"- VWAP采用累计close×volume/累计volume的分钟近似。最多每日每股{cfg['max_cycles_per_day']}组T，资金不足缩量或跳过；"
            "交易首腿的预算不会因买回而重置。单根成交金额容量限制，未满足容量的配对腿整组延后。\n"
            f"- 佣金双边{cfg['commission_rate']:.4%}，最低{cfg['min_commission']:g}元；过户费双边{cfg['transfer_rate']:.4%}；"
            f"卖出印花税2023-08-28前{cfg['stamp_rate_before_20230828']:.2%}、起{cfg['stamp_rate_from_20230828']:.2%}；"
            f"单边滑点{cfg['slippage_rate']:.2%}，单分钟成交额上限{cfg['max_bar_amount_fraction']:.0%}。\n"
            "- 做T增量=相同初始买单的期末权益差；库存已全部恢复时等于配对净现金损益。"
            "未完成配对时采用实际股数市值并单列，不当成已实现利润。总收益包含底仓浮动盈亏，底仓期末不卖出。\n"
            f"- 每年独立{cfg['initial_cash']:,.0f}元起步，模型和底仓分别重置；不能将年度收益相乘冒充连续不平仓组合。"
            f"日收益Sharpe/波动按252日，年化收益按实际日历跨度。数据区间{cfg['start_date']}至{cfg['end_date']}，实际天数以各年表为准。\n"
            "- 留一年法2023/2024/2025训练含后续年份，仅供回顾性跨年检验；2026折训练只到2025年。"
            f"标签结束年份必须在训练年内，留出年后{cfg['embargo_days']}日隔离；EMA长记忆未逐年重置。"
            "内部参数选择和门槛校准在最新训练年完成，因此内部收益属于参数选择结果，不是独立测试。"
            "最终外层模型重训后沿用已冻结的内部样本外门槛，不再使用拟合样本分数或外层测试分数重估门槛。\n"
            "- 当前名称快照排除ST/退存在前视筛选及幸存者偏差；缺少完整历史涨跌停价/盘口排队、公司行动现金分红和复权校准。"
            "一字线与无量/缺失行拒绝成交只是保守代理，不能保证实盘收盘一定恢复。现有数据也曾被历史研究查看，2026不称全新未触碰测试集。\n\n"
            "制度来源：[深交所2023交易规则](https://docs.static.szse.cn/www/lawrules/rule/allrules/bussiness/W020230217564423808793.pdf)、"
            "[财政部印花税减半公告](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm)。\n")


def create_file_description() -> str:
    """返回报告文件映射；根目录跨年表带test_year，各年度目录保存完整模型链。"""
    return """# 输出文件说明

| 文件 | 用途 |
|---|---|
| final_report.md | 中文结论、收益比较、参数及局限 |
| test_metrics.csv / annual_summary.csv | 单年指标或带test_year的跨年比较 |
| test_equity_curve.csv / baseline_equity_curve.csv | 日终真实权益与相同底仓持有基准 |
| 测试集收益曲线.png / test_return_curve.png | 各年独立收益及T净增量图 |
| test_trade_events.csv | 底仓买入和T成交，含信号时间、费用、可卖额度 |
| test_round_trips.csv | T两腿配对现金损益，不代表卖出当天新买股票 |
| test_open_positions.csv | 期末底仓、未恢复配对、真实股数与目标差 |
| daily_inventory_audit.csv | 每日每股开收盘数量、新增、卖出预算与恢复失败 |
| blocked_orders.csv | 缺失行情/容量/现金等拒绝或延迟的委托 |
| base_entry_schedule.csv | 所有候选及持有基准共用的底仓买单 |
| validation_rule_metrics.csv | 仅内部验证年候选结果及选择标记 |
| selected_rule.json / selected_rules.json | 测试之前冻结的T参数、开仓门槛和年度协议 |
| execution_audit.json | T+1、整手、延迟与恢复失败核查 |
| split_audit.csv | 外层与内层年份及回顾/向前协议 |
| inner_model/ / model/ | 核心训练模型和全部训练年重拟合模型、特征清单及筛选审计 |
| model_metadata.json / resolved_config.json / run_context.json | 复现路径、模型索引、数据指纹及完整配置 |
| minute_data_quality.csv | 实际加载底仓股票的行情去重及非法行统计 |
| signal_eligibility_metadata.json | 仅用当前分钟一字线状态过滤的辅助缓存指纹及样本数量 |
| validation_base_entry_schedule.csv | 内部验证组合底仓输入 |

根目录跨年表是分别重置组合后的纵向汇总，不是一个连续持仓回测。每年模型目录内
保存 model.joblib、model.json、used_feature_manifest.csv、feature_list.txt、
feature_selection_audit.csv 和 model_metadata.json。未主动删除或覆盖旧文件。
"""


def run_cross_validation(cfg: dict, directory: Path) -> None:
    """依次完成核心训练/内部选T参数/全训练年重拟合/一次留出测试，写完整审计产物。"""
    panel, market, membership, manifest = load_annual_panel(cfg)
    panel["datetime"] = pd.to_datetime(panel.datetime)
    market["datetime"] = pd.to_datetime(market.datetime)
    if cfg["exclude_flat_signal_bars"]:
        flags, eligibility_metadata = load_signal_eligibility(manifest, cfg)
        before_count = len(panel)
        panel = apply_signal_eligibility(panel, flags, cfg)
        eligibility_metadata.update({"input_panel_rows": before_count, "eligible_panel_rows": len(panel)})
        write_json(directory / "signal_eligibility_metadata.json", eligibility_metadata)
        print(f"仅用信号当时行情过滤一字线：{before_count:,} -> {len(panel):,}行", flush=True)
        del flags
    folds = build_year_folds(panel, cfg["test_years"], cfg["embargo_days"])
    write_frame(pd.DataFrame(folds), directory / "split_audit.csv")
    write_json(directory / "data_manifest.json", manifest)
    store, results, summaries, validations, selections = MinuteBarStore(cfg), [], [], [], []
    for fold in folds:
        year, inner_year = fold["test_year"], fold["inner_validation_year"]
        output = directory / f"year_{year}"
        output.mkdir()
        print(f"[{year}] 核心训练{fold['core_train_years']}，内部参数验证{inner_year}，外层留出{year}", flush=True)
        core_model, core_features, _ = fit_entry_model(panel, fold["core_train_years"], cfg,
                                                      output / "inner_model", [year, inner_year])
        dates = select_year_dates(market, inner_year, cfg)
        inner_scores = predict_entry_scores(core_model, core_features, panel, [inner_year], cfg)
        # 门槛从未参与核心拟合的内部训练年预测校准；不使用拟合样本偏高的分数。
        inner_gate = calibrate_entry_gate(inner_scores, cfg)
        inner_gate.update({"calibration_year": inner_year, "source_model_train_years": fold["core_train_years"],
                           "calibration_scope": "外层训练年份内部的样本外预测；此内部年同时用于参数选择，非独立测试",
                           "frozen_after_final_refit": True})
        entries = create_base_entries(inner_scores, inner_gate, dates, cfg, store)
        write_frame(entries, output / "validation_base_entry_schedule.csv")
        bars = store.load_slice(sorted(entries.symbol.unique()), dates)
        candidates, rules = [], create_t_rules(cfg)
        for rule in rules:
            outcome = run_inventory_t(bars, entries, dates, cfg, rule)
            audit_result(outcome, entries, cfg)
            metrics = outcome["metrics"]
            candidates.append({"test_year": year, "inner_validation_year": inner_year,
                               "rule_name": rule["name"], "incremental_net_pnl": metrics["total_t_marked_pnl"],
                               "pair_count": metrics["t_pair_count"], "t_fee": metrics["t_fee"],
                               "inventory_mismatch_days": metrics["inventory_mismatch_days"],
                               "recovery_failure_days": metrics["recovery_failure_days"],
                               "pending_pairs": metrics["pending_pairs"], "total_return": metrics["total_return"]})
            print(f"[{year}] 内部候选{rule['name']}：净增量{metrics['total_t_marked_pnl']:,.2f}元，配对{metrics['t_pair_count']}", flush=True)
        validation = pd.DataFrame(candidates)
        # 股数约束优先：只在内部验证中无库存偏差且期末已恢复的规则中比较净增量。
        eligible = validation.loc[(validation.inventory_mismatch_days == 0) & (validation.pending_pairs == 0)]
        if eligible.empty:
            chosen_rule = {"name": "hold_due_to_training_recovery_failures", "enabled": False}
            chosen_name = chosen_rule["name"]
        else:
            chosen_name = eligible.sort_values(["incremental_net_pnl", "rule_name"], ascending=[False, True]).iloc[0].rule_name
            chosen_rule = next(rule for rule in rules if rule["name"] == chosen_name)
        validation["selected"] = validation.rule_name == chosen_name
        write_frame(validation, output / "validation_rule_metrics.csv")
        selections.append({**fold, "selected_rule": chosen_rule, "inner_gate": inner_gate})
        write_json(output / "frozen_t_rule_before_test.json", selections[-1])
        print(f"[{year}] 冻结{chosen_name}；用全部训练年份{fold['train_years']}重新训练开仓模型", flush=True)
        model, features, metadata = fit_entry_model(panel, fold["train_years"], cfg, output / "model", [year])
        gate = dict(inner_gate)
        dates = select_year_dates(market, year, cfg)
        scores = predict_entry_scores(model, features, panel, [year], cfg)
        entries = create_base_entries(scores, gate, dates, cfg, store)
        bars = store.load_slice(sorted(entries.symbol.unique()), dates)
        result = run_inventory_t(bars, entries, dates, cfg, chosen_rule)
        save_fold_report(output, year, result, entries, validation, fold, chosen_rule, gate, cfg)
        write_json(output / "model_metadata.json", metadata)
        metrics = result["metrics"]
        summaries.append({"test_year": year, "protocol": fold["protocol"], "train_years": str(fold["train_years"]),
                          "selected_rule": chosen_name, "entry_symbol_count": entries.symbol.nunique(),
                          "incremental_return_points": metrics["total_t_marked_pnl"] / cfg["initial_cash"] * 100,
                          **metrics})
        results.append((year, result))
        validations.append(validation)
        print(f"[{year}] 留出结果：持有{metrics['baseline_total_return']:.2%}，做T后{metrics['total_return']:.2%}，净增量{metrics['total_t_marked_pnl']:,.2f}元", flush=True)
    summary = pd.DataFrame(summaries)
    for filename in ("annual_summary.csv", "test_metrics.csv"):
        write_frame(summary, directory / filename)
    for key, filename in (("equity_curve", "test_equity_curve.csv"), ("trade_events", "test_trade_events.csv"),
                          ("round_trips", "test_round_trips.csv"), ("open_positions", "test_open_positions.csv"),
                          ("daily_inventory_audit", "daily_inventory_audit.csv"), ("order_attempts", "blocked_orders.csv")):
        write_frame(pd.concat([result[key].assign(test_year=year) for year, result in results], ignore_index=True), directory / filename)
    write_frame(pd.concat(validations, ignore_index=True), directory / "validation_rule_metrics.csv")
    write_frame(pd.DataFrame(store.audit.values()), directory / "minute_data_quality.csv")
    write_json(directory / "selected_rules.json", selections)
    write_json(directory / "model_metadata.json", {"folds": folds, "models": [f"year_{fold['test_year']}/model/model_metadata.json" for fold in folds],
                                                   "retrained_per_fold": True, "existing_fixed_model_reused": False})
    plot_return_curves(results, directory / "测试集收益曲线.png", cfg["initial_cash"])
    shutil.copyfile(directory / "测试集收益曲线.png", directory / "test_return_curve.png")
    lines = ["# 固定底仓做T：年度交叉验证\n", f"初始现金：{cfg['initial_cash']:,.0f}元/独立年度；开仓规则：{cfg['entry_method']}。\n",
             "| 留出年 | 训练年份 | 持有收益 | 加做T收益 | 做T净增量(元) | 增量(百分点) | T组数 | 未恢复日 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.test_year} | {row.train_years} | {row.baseline_total_return:.2%} | {row.total_return:.2%} | "
                     f"{row.total_t_marked_pnl:,.2f} | {row.incremental_return_points:+.2f} | {row.t_pair_count} | {row.recovery_failure_days} |")
    gains = int((summary.total_t_marked_pnl > 0).sum())
    lines += [f"\n{len(summary)}个留出年中，{gains}年做T净增量为正。总收益高低不能代替做T增量判断；详细规则及股票在各year目录。\n",
              "做T规则只按内部验证选择；若没有任何内部候选满足股数恢复要求，则该折事前退回只持有。\n",
              create_semantics_text(cfg)]
    (directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "输出文件说明.md").write_text(create_file_description(), encoding="utf-8")


def main() -> None:
    """新建不覆盖的时间戳报告目录，记录代码版本/配置后执行完整实验。"""
    args = parse_arguments()
    cfg = load_configuration(args)
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "resolved_config.json", cfg)
    code_files = [Path(__file__)] + [PROJECT_ROOT / filename for filename in (
        "quant_backtest/inventory_t.py", "quant_backtest/inventory_t_models.py", "quant_backtest/inventory_t_data.py", "quant_backtest/annual_panel.py",
        "quant_backtest/annual_portfolio.py", "quant_backtest/factor_library.py", "scripts/run_multi_factor_cta_backtests.py")]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    write_json(directory / "run_context.json", {"command": sys.argv, "git_revision": revision,
               "code_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
               "generated_at": datetime.now().isoformat(), "smoke_only": bool(cfg["smoke_days"])})
    print(f"报告目录：{directory}", flush=True)
    run_cross_validation(cfg, directory)
    print(f"完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
