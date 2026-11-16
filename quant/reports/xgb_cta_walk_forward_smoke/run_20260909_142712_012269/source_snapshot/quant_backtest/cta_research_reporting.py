"""CTA年度研究的中文报告与独立账务复核；只读取已冻结结果，不据测试表现选规则。

输入为模型回测账本/行情缓存，生成收益曲线、逐笔交易、年度/市况分层及文件说明。
原始分钟权益保存在PKL，标准CSV为日终权益，避免报告目录重复保存巨大分钟表。
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .annual_panel import write_json
from .annual_portfolio import calculate_fee
from .cta_research_data import classify_board, save_pickle, read_history

LIMITATIONS = """- 本轮严格按时间向前拟合，但历史行情已被其他研究查看，不能称为全新独立测试。
- 历史ST来自BaoStock，缺失禁止新买；未使用当前名称快照。第三方日线不等于完整交易所状态档案，退市整理/特殊复牌/临时停牌及公司行动仍可能不完整。
- 本地分钟收盘与第三方未复权日收盘作对照。现金分红、送转股和红利税尚未完整入账；不能因此宣称已达到实盘级总回报核算。日线历史参考价并不保证涵盖所有特殊交易日。
- 主板10%、创业板/科创板20%正常日价格限制按参考前收盘四舍五入到分；至少60个历史交易日后入池，排除新上市前五日。特殊复牌无涨跌幅限制、盘口排队及完整价格笼子未精确模拟。
- 成交用信号后下一分钟收盘加滑点并按分取整，信号参考价上下2%保护，成交额最多占本分钟5%；这不是逐笔盘口重放。部分卖出会继续等待，不能按理想止损价保证成交。
- ATR取上一已完成日的14日简单平均真实波幅；每分钟先检查上根已知保护线是否被本分钟低价触及，再更新高点，更新后的线只与当前收盘比较，避免假设分钟高低点先后。
- 每批次T+1、ATR/趋势/到期退出独立；跨年连续账户保留旧批次原退出参数。独立年度账户重置初始现金，不能与连续账户收益混称。
- 训练标签是单笔固定10%初始资金预算下的净收益。共享账户的资金竞争、动态总资产和单股上限另外模拟；标签不是完整组合收益。
- 科创板至少200股、逐股递增。成交手数是成交股数/100的统计单位，出现小数不是违规；买入不足200股才是违规。
- 无成交或日收益无波动时夏普未定义。程序内部0为数值占位，报告显示未定义，不能据此计算成功比例。
"""


def get_last_mark(market, index: int, asset: int) -> tuple[float, int]:
    """取得market中asset截至index的最后实际价格及索引，不用未来数据补估值。"""
    while index >= 0:
        price = market.quotes[index, asset, 0]
        if np.isfinite(price) and price > 0:
            return float(price), index
        index -= 1
    raise ValueError("持仓没有任何已知估值价格")


def audit_result(result: dict, market, cfg: dict) -> dict:
    """独立按result实际事件重算现金/费用/股数/T+1/限价/入场预算，market只提供当时价格。

    检查通过代表这些工程约束成立，不代表历史状态、公司行动齐全或策略会盈利。
    """
    events, trips, curve = result["events"], result["trips"], result["curve"]
    cash, lots, maximum_cash_error, maximum_fee_error = cfg["initial_cash"], {}, 0.0, 0.0
    used = {}
    for row in events.itertuples(index=False):
        if row.signal_datetime >= row.datetime:
            raise AssertionError("成交未严格滞后信号")
        asset, time_index = market.symbol_index[row.symbol], row.time_index
        quote, flags = market.quotes[time_index, asset], market.flags[time_index, asset]
        if not (quote[7] < row.price < quote[6]) or not (quote[7] < row.reference_price < quote[6]):
            raise AssertionError("涨跌停价格发生交易")
        if row.side == "buy" and not flags[0] or row.side == "sell" and not flags[1]:
            raise AssertionError("没有交易资格时成交")
        fee = calculate_fee(row.notional, row.side, row.datetime, cfg)
        maximum_fee_error = max(maximum_fee_error, abs(row.fee - fee))
        if abs(row.notional - row.shares * row.price) > 1e-6:
            raise AssertionError("成交金额不等于股数乘单价")
        key = (row.time_index, row.asset)
        used[key] = used.get(key, 0.0) + row.notional
        if used[key] > quote[3] * cfg["max_bar_amount_fraction"] + 1e-5:
            raise AssertionError("同分钟同股票累计成交超过容量")
        board = classify_board(row.symbol)
        if row.side == "buy":
            if (board == 2 and row.shares < 200) or (board != 2 and row.shares % 100):
                raise AssertionError("买入股数违反板块规则")
            if row.notional > row.signal_equity * cfg["entry_equity_fraction"] + 1e-5:
                raise AssertionError("单笔超过决策时总资产10%")
            lots[row.lot_id] = {"asset": asset, "shares": row.shares, "buy_date": row.datetime.normalize()}
            cash -= row.notional + fee
            held, stock = 0.0, 0.0
            for lot in lots.values():
                mark, _ = get_last_mark(market, time_index, lot["asset"])
                value = lot["shares"] * mark
                held += value
                if lot["asset"] == asset:
                    stock += value
            equity = cash + held
            if stock > cfg["max_symbol_exposure"] * equity + 1e-5 or held > cfg["max_total_exposure"] * equity + 1e-5:
                raise AssertionError("新买入后超出单股或总仓位上限")
        else:
            lot = lots[row.lot_id]
            if row.datetime.normalize() <= lot["buy_date"] or row.shares > lot["shares"]:
                raise AssertionError("T+1违规或超卖")
            if board == 2 and row.shares < 200 and row.shares != lot["shares"]:
                raise AssertionError("科创板不足200股余额未一次卖出")
            lot["shares"] -= row.shares
            cash += row.notional - fee
            if not lot["shares"]:
                del lots[row.lot_id]
        maximum_cash_error = max(maximum_cash_error, abs(cash - row.cash_after))
        if cash < -1e-5:
            raise AssertionError("现金透支")
    opened = result["open"]
    unrealized = float((opened.shares * (opened.mark_price - opened.buy_price) - opened.remaining_buy_fee).sum())
    pnl_identity_error = abs(float(curve.equity.iloc[-1]) - cfg["initial_cash"] - float(trips.net_pnl.sum()) - unrealized)
    if maximum_cash_error > 1e-5 or maximum_fee_error > 1e-6 or pnl_identity_error > 1e-5:
        raise AssertionError("独立现金/费用/权益恒等式失败")
    if set(lots) != set(opened.lot_id):
        raise AssertionError("未平仓批次与事件重建不一致")
    for row in opened.itertuples(index=False):
        if lots[row.lot_id]["shares"] != row.shares:
            raise AssertionError("未平仓股数错误")
    # 对照所有有成交分钟及所有日终现金，防止只核对最终总数。
    cash_events = events.groupby("datetime").cash_after.last()
    reconstructed = cash_events.reindex(pd.DatetimeIndex(curve.datetime), method="ffill").fillna(cfg["initial_cash"])
    curve_cash_error = float(np.max(np.abs(reconstructed.to_numpy() - curve.cash.to_numpy())))
    if curve_cash_error > 1e-5:
        raise AssertionError("分钟净值表现金与事件账本不一致")
    daily = curve.groupby(curve.datetime.dt.normalize(), sort=True).last()
    event_rows = list(events.itertuples(index=False))
    pointer, day_cash, inventory, holding_rows, daily_equity_error = 0, cfg["initial_cash"], {}, [], 0.0
    for day in daily.itertuples():
        while pointer < len(event_rows) and event_rows[pointer].datetime <= day.datetime:
            row = event_rows[pointer]
            sign = 1 if row.side == "buy" else -1
            inventory[row.asset] = inventory.get(row.asset, 0) + sign * row.shares
            day_cash -= sign * row.notional + calculate_fee(row.notional, row.side, row.datetime, cfg)
            pointer += 1
        value = 0.0
        for asset, shares in inventory.items():
            if shares > 0:
                mark, index = get_last_mark(market, int(day.time_index), asset)
                value += shares * mark
                holding_rows.append({"datetime": day.datetime, "symbol": market.symbols[asset], "shares": shares,
                                     "mark_price": mark, "mark_datetime": market.times[index], "market_value": shares * mark})
        daily_equity_error = max(daily_equity_error, abs(day_cash + value - day.equity))
    if daily_equity_error > 1e-5:
        raise AssertionError("日终净值与独立重建的现金和持仓不一致")
    result["holdings"] = pd.DataFrame(holding_rows, columns=["datetime", "symbol", "shares", "mark_price", "mark_datetime", "market_value"])
    return {"passed": True, "events": len(events), "round_trip_fragments": len(trips), "cash_error": maximum_cash_error,
        "curve_cash_error": curve_cash_error, "fee_error": maximum_fee_error, "pnl_identity_error": pnl_identity_error,
        "daily_equity_error": daily_equity_error,
        "t1_violations": 0, "board_quantity_violations": 0, "price_limit_fills": 0,
        "history_st_buy_violations": 0, "entry_budget_violations": 0, "cash_overdrafts": 0,
        "corporate_actions_complete": False, "independent_new_test": False}


def calculate_daily_table(result: dict) -> pd.DataFrame:
    """把result分钟账本转成日终净值及包含首日费用的日收益，保留现金日。"""
    daily = result["curve"].groupby(result["curve"].datetime.dt.normalize(), sort=True).last().reset_index(drop=True)
    daily["daily_return"] = daily.equity.pct_change(fill_method=None)
    daily.loc[daily.index[0], "daily_return"] = daily.equity.iloc[0] / result["initial_cash"] - 1
    daily["cumulative_return"] = daily.equity / result["initial_cash"] - 1
    daily["drawdown"] = daily.equity / np.maximum.accumulate(np.r_[result["initial_cash"], daily.equity])[1:] - 1
    return daily


def save_result(directory: Path, result: dict, selected: dict, cfg: dict, title: str) -> None:
    """将result写入directory的标准中文报告结构，selected/cfg记录冻结规则及配置。"""
    from scripts.run_annual_method_validation import plot_result
    directory.mkdir(parents=True, exist_ok=True)
    daily = calculate_daily_table(result)
    daily.to_csv(directory / "test_equity_curve.csv", index=False, encoding="utf-8-sig")
    save_pickle(directory / "minute_equity_curve.pkl", result["curve"])
    for key, filename in [("events", "test_trade_events.csv"), ("trips", "test_round_trips.csv"),
                          ("open", "test_open_positions.csv"), ("blocked_events", "test_blocked_orders.csv"),
                          ("holdings", "test_holdings.csv")]:
        result[key].to_csv(directory / filename, index=False, encoding="utf-8-sig")
    pd.DataFrame([result["metrics"]]).to_csv(directory / "test_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "strategy_metadata.json", {"rule": selected, "blocked": result["blocked"], "title": title})
    write_json(directory / "execution_audit.json", result["audit"])
    prefix = "流程烟测，非正式策略成绩；" if cfg.get("is_smoke") else "历史回顾、非实盘保证；"
    result["report_note"] = prefix + "含T+1、历史ST、交易成本；公司行动/盘口模型不完整。"
    plot_result(directory, result, title)
    metrics = result["metrics"]
    rows = [("总收益率", f"{metrics['total_return']:.2%}"), ("年化收益率", f"{metrics['annual_return']:.2%}"),
            ("夏普", f"{metrics['sharpe']:.4f}" if metrics["annual_volatility"] else "未定义（无波动）"),
            ("最大回撤", f"{metrics['max_drawdown']:.2%}"), ("年化波动", f"{metrics['annual_volatility']:.2%}"),
            ("买入次数", metrics["buy_count"]), ("卖出次数（含部分成交）", metrics["sell_count"]),
            ("完全平仓批次", metrics["closed_lot_count"]), ("科创板买入次数", metrics["star_buy_count"]),
            ("创业板买入次数", metrics["chinext_buy_count"]), ("完全平仓胜率", f"{metrics['win_rate']:.2%}" if metrics["closed_lot_count"] else "未定义"),
            ("平均持仓自然日", f"{metrics['mean_holding_days']:.2f}"), ("双边换手倍数", f"{metrics['turnover']:.2f}"),
            ("期末未平仓批次", metrics["open_lots"])]
    text = f"# {title}\n\n{prefix}\n\n规则：`{selected['rule_id']}`。\n\n" + pd.DataFrame(rows, columns=["指标", "结果"]).to_markdown(index=False)
    text += "\n\n![收益与回撤](测试集收益曲线.png)\n\n## 成交审计\n\n"
    text += "独立现金、费用、T+1、板块数量、日涨跌停价、容量和每笔10%预算检查通过。通过不等于数据完整或可实盘。\n\n"
    text += "## 已知限制\n\n" + LIMITATIONS
    (directory / "final_report.md").write_text(text, encoding="utf-8")
    (directory / "输出文件说明.md").write_text("""# 输出文件说明

| 文件 | 角色与用途 |
|---|---|
| final_report.md | 人工阅读入口：指标、曲线、真实执行及数据限制 |
| 测试集收益曲线.png / test_return_curve.png | 同一张含策略名、夏普、回撤、收益、年化、波动、手数、胜率的图 |
| test_equity_curve.csv | 日终净值；equity总资产、cash现金、daily_return日收益、exposure多头仓位、drawdown回撤 |
| minute_equity_curve.pkl | 可复用的分钟账本缓存，非用户配置，不推送大二进制 |
| test_trade_events.csv | 每次实际成交；datetime成交时间、signal_datetime委托依据时间、symbol代码、side买卖、shares股数、price成交价、fee费用、reason原因、lot_id独立批次 |
| test_round_trips.csv | 买卖配对片段；buy_datetime开仓、sell_datetime平仓、net_pnl扣费盈亏、sell_signal_datetime最初退出触发时刻；部分卖出多行 |
| test_open_positions.csv | 未平仓批次、估值、剩余成本、stop_price保护线、pending_reason待退出原因 |
| test_holdings.csv | 由成交独立重建的逐日股票持仓和末次真实报价时间，不能把陈旧报价当成当日成交 |
| test_blocked_orders.csv | 被阻挡的请求；同批次同日连续相同卖出原因合并，attempt_count是尝试次数，不是成交笔数 |
| test_metrics.csv | 机器可读指标；无波动夏普0仅作程序占位，人类报告写未定义 |
| strategy_metadata.json | 冻结规则、中文策略名和拒单次数 |
| execution_audit.json | 独立重算的现金/费用/权益恒等式和T+1、数量、容量约束 |

持仓时间分为自然日统计和规则使用的交易分钟/交易日，二者不能混淆。买入/卖出次数
是成交事件数；完整交易是lot_id批次全部退出。科创板股数可以不整百，因此手数可有小数。
""", encoding="utf-8")


def calculate_regimes(daily: pd.DataFrame, manifest: dict, cfg: dict) -> pd.DataFrame:
    """用manifest基准上一完成日的固定60日收益标注daily牛熊震荡，不按策略收益分段。"""
    benchmark = read_history(Path(manifest["history_path"]), cfg["benchmark_symbol"].upper()).set_index("date")
    changes = benchmark.close.pct_change(cfg["regime_lookback_days"], fill_method=None).shift(1)
    current = changes.reindex(pd.DatetimeIndex(daily.datetime).normalize()).to_numpy()
    cutoff = cfg["regime_return_threshold"]
    labels = np.where(current > cutoff, "牛市趋势", np.where(current < -cutoff, "熊市趋势", "震荡"))
    labels[~np.isfinite(current)] = "历史不足"
    out = daily.copy()
    out["regime"] = labels
    out["benchmark_past_return"] = current
    out["benchmark_return"] = benchmark.close.pct_change(fill_method=None).reindex(pd.DatetimeIndex(daily.datetime).normalize()).to_numpy()
    return out


def calculate_bootstrap(returns: np.ndarray, seed: int) -> dict:
    """对returns做固定20交易日连续块抽样1000次，seed复现区间；不是独立测试证明。"""
    generator = np.random.default_rng(seed)
    block = min(20, len(returns))
    values = []
    for _ in range(1000):
        starts = generator.integers(0, len(returns), size=int(np.ceil(len(returns) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(returns)).ravel()[:len(returns)]
        draw = returns[indices]
        std = draw.std(ddof=1)
        if std > 0:
            values.append(float(draw.mean() / std * np.sqrt(252)))
    interval = np.quantile(values, [0.025, 0.975]).tolist() if values else [None, None]
    return {"method": "20日连续块自助法", "samples": 1000, "sharpe_95pct_interval": interval,
            "limitation": "不消除历史选择偏差，不保证未来夏普超过区间下界"}


def save_summary(directory: Path, annual: pd.DataFrame, continuous: dict, stress: dict, manifest: dict,
                 cfg: dict, folds: list[dict]) -> None:
    """汇总各年度、连续账户、固定市况分组和压力测试；所有结论来自已保存的真实结果。"""
    annual.to_csv(directory / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    validation = []
    for fold in folds:
        frame = pd.read_csv(directory / f"year_{fold['year']}" / "validation_rule_metrics.csv")
        frame["test_year"] = fold["year"]
        validation.append(frame)
    pd.concat(validation, ignore_index=True).to_csv(directory / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    daily = calculate_regimes(calculate_daily_table(continuous), manifest, cfg)
    daily.to_csv(directory / "market_regime_daily.csv", index=False, encoding="utf-8-sig")
    groups = []
    for label, group in daily.groupby("regime"):
        returns = group.daily_return
        std = returns.std(ddof=1)
        groups.append({"市场阶段": label, "交易日数": len(group), "平均日收益": returns.mean(),
                       "条件日收益夏普": float(returns.mean() / std * np.sqrt(252)) if std > 0 else np.nan,
                       "该状态日收益连乘": float((1 + returns).prod() - 1), "平均持仓": group.exposure.mean()})
    regimes = pd.DataFrame(groups)
    regimes.to_csv(directory / "market_regime_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap = calculate_bootstrap(daily.daily_return.to_numpy(), cfg["random_seed"])
    write_json(directory / "sharpe_uncertainty.json", bootstrap)
    metrics = continuous["metrics"]
    target = bool(not cfg.get("is_smoke") and metrics["annual_volatility"] > 0 and metrics["sharpe"] > cfg["sharpe_target"])
    summary = {"historical_point_sharpe_above_target": target, "independent_goal_achieved": False, "live_approved": False,
               "continuous": metrics, "cost_stress": stress["metrics"], "bootstrap": bootstrap,
               "reason": "历史数据已有暴露，且公司行动/特殊状态和盘口尚不完整；无论点估计如何都不冒充独立实盘验收"}
    write_json(directory / "summary.json", summary)
    # 根目录保留标准入口；所有复制来源固定为连续账户，不挑选收益最好的年度。
    for name in ["test_equity_curve.csv", "test_trade_events.csv", "test_round_trips.csv", "test_open_positions.csv",
                 "test_metrics.csv", "测试集收益曲线.png", "test_return_curve.png", "test_blocked_orders.csv", "test_holdings.csv"]:
        shutil.copyfile(directory / "continuous" / name, directory / name)
    displayed = annual[["year", "coverage", "total_return", "sharpe", "max_drawdown", "buy_count", "closed_lot_count"]].copy()
    displayed.columns = ["年度", "实际测试范围", "净收益", "夏普", "最大回撤", "买入次数", "完整平仓批次"]
    displayed["净收益"] = displayed["净收益"].map(lambda x: f"{x:.2%}")
    displayed["最大回撤"] = displayed["最大回撤"].map(lambda x: f"{x:.2%}")
    displayed["夏普"] = [f"{x:.3f}" if vol > 0 else "未定义" for x, vol in zip(annual.sharpe, annual.annual_volatility)]
    active = annual.loc[annual.annual_volatility > 0]
    average = active.sharpe.mean() if len(active) == len(annual) else np.nan
    conclusion = "历史点估计超过3，但不构成独立或实盘验收" if target else "本轮未达到夏普大于3目标"
    if cfg.get("is_smoke"):
        conclusion = "仅为三板块小样本流程烟测，不是完整股票池训练或正式年度成绩"
    text = f"# 三板块XGBoost CTA向前验证\n\n**{conclusion}。**\n\n"
    text += f"扩展特征{manifest['feature_count']}列，{manifest['rows']:,}个信号样本，周池累计{len(manifest['symbols'])}只股票。"
    text += f"复用{manifest['reused_factor_symbols']}只股票原因子，其余只补缺少的板块/时间覆盖。历史日收盘不一致记录{manifest['daily_close_mismatch_count']}条。\n\n"
    text += "## 各年度独立账户\n\n" + displayed.to_markdown(index=False)
    text += f"\n\n各年度夏普简单均值：{average:.3f}（有无波动年则未定义，不删除该年）。2023不是全年、2026截至4月。\n\n"
    text += "## 跨年连续账户\n\n"
    text += (f"净收益{metrics['total_return']:.2%}，夏普{metrics['sharpe']:.3f}，最大回撤{metrics['max_drawdown']:.2%}，"
             f"买入{metrics['buy_count']}次、完整平仓{metrics['closed_lot_count']}批。"
             f"双倍滑点夏普{stress['metrics']['sharpe']:.3f}、净收益{stress['metrics']['total_return']:.2%}。\n\n")
    text += f"日收益块自助法夏普95%区间：{bootstrap['sharpe_95pct_interval']}；不能消除选择偏差。\n\n"
    text += "![连续账户收益](测试集收益曲线.png)\n\n## 牛熊震荡\n\n" + regimes.to_markdown(index=False)
    text += "\n\n阶段取沪深300上一已完成日60日涨跌幅，超过±5%分别为牛/熊趋势，其余震荡。"
    text += "这是条件日收益诊断，非连续可交易的独立子策略；不会根据该表再选模型。\n\n## 全流程\n\n"
    text += "历史状态审计 → 上周高波动周池 → 复用/扩展特征PKL → T+1退出路径标签 → "
    text += "逐折过去特征筛选和XGBoost拟合 → 内部验证选CTA/频率/门槛 → 全部冻结 → "
    text += "逐年独立账户 → 连续账户/固定成本压力测试 → 独立账本审计 → 中文报告。\n\n"
    text += "## 限制与实盘边界\n\n" + LIMITATIONS
    (directory / "final_report.md").write_text(text, encoding="utf-8")
    (directory / "输出文件说明.md").write_text("""# 本轮输出文件说明

## 架构

year_年度保留该年开始前拟合/选定的模型；其backtest目录是年度独立账户。
continuous是跨年不清仓的连续账户，根目录标准图表和交易表固定复制自这里。
cost_stress只增加滑点，不重新选规则；deployment_candidate是全量重训候选，尚无独立成绩。

## 文件与角色

| 路径 | 角色与用途 |
|---|---|
| final_report.md | 生成的中文总报告，所有结论的入口 |
| 输出文件说明.md | 本说明，人工可读的产物索引 |
| annual_metrics.csv | 生成的逐年度净收益/夏普/回撤/交易数量与真实覆盖范围 |
| validation_rule_metrics.csv | 全部内部候选的成绩，失败候选不删除 |
| test_*.csv / 测试集收益曲线.png / test_return_curve.png | 连续账户标准报告；字段解释见continuous/输出文件说明.md |
| market_regime_daily.csv / market_regime_metrics.csv | 固定规则的逐日牛熊震荡标记与分层成绩 |
| sharpe_uncertainty.json | 连续块抽样区间，不能当成未来盈利保证 |
| summary.json | 机器可读总指标、目标是否达到及不准实盘状态 |
| resolved_config.yaml | 用户配置的本次解析快照，不是新调参入口 |
| protocol.json / frozen_before_test.json | 实验时间切分、源码指纹、测试前权重与规则哈希 |
| data_manifest.json / label_manifest.json | 输入缓存来源与监督标签定义，便于复现 |
| dynamic_universe_membership.csv | 按上周数据生成的每日股票池 |
| frozen_test_signals.pkl | 冻结模型生成的测试开仓信号缓存，不可反过来选参 |
| source_snapshot/ | 当次运行的源码/配置快照，不修改生产文件 |
| year_*/models/*/xgboost_model.json | 原生XGBoost模型，不能用另一折的特征顺序调用 |
| year_*/models/*/model_metadata.json | 拟合时间、标签结束、固定特征列序、模型参数与输入指纹 |
| year_*/models/*/model_feature_manifest.csv | 每个候选特征中文解释、训练相关性、冗余及入选状态 |
| year_*/models/*/training_rows.csv | 实际拟合样本及标签结束时间，可核验没有测试年份 |
| year_*/selected_rule.json | 内部验证选定的CTA条件、开仓频率、净收益门槛与退出方案 |
| year_*/backtest/ | 对应该年度的独立回测；文件角色见该目录说明 |
| continuous/ / cost_stress/ | 连续账户和预先固定的双倍滑点诊断，分别含完整账本 |
| deployment_candidate/ | 含2026训练数据的候选，不能继承年度测试成绩或直接实盘 |

## 顺序

先看final_report，再看annual_metrics及selected_rule；复盘交易使用成交事件、批次
配对和未平仓表，不把卖出片段数当成完整交易次数。复现命令见6method_xgb_cta/README.md。
""", encoding="utf-8")
