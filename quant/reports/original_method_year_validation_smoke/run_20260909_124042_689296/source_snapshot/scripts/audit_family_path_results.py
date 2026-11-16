"""在quant根目录独立复核模型族策略的四年CSV账本，不训练或重跑测试。

输入--run-dir内冻结配置、成交、配对、持仓、净值和指标；输出独立时间戳审计目录。
仅读取已有结果，不改父轮/冻结文件，不连接券商，不核验原始逐分钟撮合价格。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


FILES = {"curve": "test_equity_curve.csv", "events": "test_trade_events.csv",
         "trips": "test_round_trips.csv", "open": "test_open_positions.csv",
         "holdings": "test_holdings.csv", "metrics": "test_metrics.csv"}
DATES = {"curve": ["datetime"], "events": ["datetime", "signal_datetime"],
         "trips": ["buy_datetime", "sell_datetime", "buy_signal_datetime", "sell_signal_datetime"],
         "open": ["buy_datetime"], "holdings": ["datetime"]}


def validate_condition(condition: bool, message: str) -> None:
    """condition为已计算审计条件；失败抛出message，不将失败静默改成通过。"""
    if not condition:
        raise AssertionError(message)


def compare_values(actual, expected, name: str, tolerance: float = 1e-7) -> float:
    """比较actual/expected数值，name标识口径；绝对tolerance默认1e-7，返回最大误差。"""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, err_msg=name, equal_nan=False)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def audit_account_frames(frames: dict, cfg: dict) -> dict:
    """独立核对frames六份标准表，cfg提供成本/总资产比例/初始现金；不调用引擎指标函数。

    日期显式解析兼容空成交CSV，返回逐项误差与补充持仓口径；非完整日终区间拒绝本审计。
    证明的是CSV内部账务一致性，不证明原行情、信号来源或成交代理正确。
    """
    data = {key: value.copy() for key, value in frames.items()}
    for key, columns in DATES.items():
        for column in columns:
            data[key][column] = pd.to_datetime(data[key][column], errors="raise")
            validate_condition(data[key][column].notna().all(), f"{key}.{column}日期缺失")
    curve, events, trips, opened, holdings = [data[k] for k in ["curve", "events", "trips", "open", "holdings"]]
    validate_condition(len(data["metrics"]) == 1 and len(curve) > 0, "指标须一行，净值不能空")
    metrics, initial = data["metrics"].iloc[0], float(cfg["initial_cash"])
    validate_condition(curve.datetime.is_monotonic_increasing and not curve.datetime.duplicated().any(), "净值时刻重复或倒序")
    validate_condition(np.isfinite(curve[["equity", "cash", "exposure"]]).all().all(), "净值出现非有限值")
    validate_condition((curve.equity > 0).all() and (curve.cash >= -1e-6).all(), "权益/现金非法")
    validate_condition(curve.exposure.between(0, 1 + 1e-8).all(), "发现空头或杠杆")
    validate_condition(set(events.side).issubset({"buy", "sell"}), "未知交易方向")
    validate_condition(events.datetime.is_monotonic_increasing, "成交顺序倒置")
    validate_condition(events.datetime.isin(curve.datetime).all(), "成交不在净值时钟")
    validate_condition(((events.datetime - events.signal_datetime) == pd.Timedelta(minutes=1)).all(), "不满足精确下一分钟成交")
    numeric = events[["shares", "price", "notional", "fee", "signal_equity"]].to_numpy(dtype=float)
    validate_condition(np.isfinite(numeric).all(), "成交包含非有限值")
    validate_condition((events.shares > 0).all() and (events.shares % cfg["lot_size"] == 0).all(), "非正数量或非整手")
    validate_condition((events.price > 0).all() and (events.fee >= 0).all(), "成交价格或费用非法")
    compare_values(events.notional, events.shares * events.price, "成交额")
    buys, sells = events.loc[events.side == "buy"].copy(), events.loc[events.side == "sell"].copy()
    validate_condition(not buys.lot_id.duplicated().any() and not opened.lot_id.duplicated().any(), "批次重复")
    validate_condition(not buys.assign(day=buys.datetime.dt.normalize()).duplicated(["day", "symbol"]).any(), "同股同日重复买入")
    validate_condition((buys.notional <= buys.signal_equity * cfg["entry_equity_fraction"] + 1e-5).all(), "买入超过信号总资产预算")
    stamp = np.where(events.datetime >= pd.Timestamp("2023-08-28"), cfg["stamp_rate_from_20230828"], cfg["stamp_rate_before_20230828"])
    expected_fees = np.maximum(events.notional.to_numpy(dtype=float) * cfg["commission_rate"], cfg["min_commission"])
    expected_fees += events.notional.to_numpy(dtype=float) * (cfg["transfer_rate"] + np.where(events.side == "sell", stamp, 0))
    fee_error = compare_values(events.fee, expected_fees, "历史税费")
    flow = np.where(events.side == "buy", -events.notional - events.fee, events.notional - events.fee)
    cash_flow = pd.Series(np.asarray(flow, dtype=float), index=events.datetime).groupby(level=0).sum()
    expected_cash = cash_flow.reindex(curve.datetime, fill_value=0).cumsum().to_numpy() + initial
    cash_error = compare_values(curve.cash, expected_cash, "全路径现金重建", 1e-5)
    compare_values(curve.equity * (1 - curve.exposure), curve.cash, "净值/现金/多头暴露", 1e-5)

    buy_index = buys.set_index("lot_id")
    validate_condition(set(trips.lot_id).issubset(buy_index.index) and set(opened.lot_id).issubset(buy_index.index), "配对或剩余持仓缺少买入")
    validate_condition(set(sells.lot_id).issubset(buy_index.index), "卖出缺少买入")
    validate_condition(not trips.duplicated(["lot_id", "sell_datetime"]).any(), "同批次同刻重复配对")
    validate_condition((trips.sell_datetime.dt.normalize() > trips.buy_datetime.dt.normalize()).all(), "违反T+1")
    for source, target in [("buy_datetime", "datetime"), ("buy_signal_datetime", "signal_datetime"), ("symbol", "symbol")]:
        validate_condition((trips[source].to_numpy() == trips.lot_id.map(buy_index[target]).to_numpy()).all(), f"配对{source}不一致")
    compare_values(trips.buy_price, trips.lot_id.map(buy_index.price), "配对买入价")
    paired = trips.sort_values(["sell_datetime", "lot_id"]).reset_index(drop=True)
    sold = sells.sort_values(["datetime", "lot_id"]).reset_index(drop=True)
    validate_condition(len(paired) == len(sold), "卖出与配对行数不同")
    for left, right in [("sell_datetime", "datetime"), ("sell_signal_datetime", "signal_datetime"), ("lot_id", "lot_id"), ("symbol", "symbol")]:
        validate_condition((paired[left].to_numpy() == sold[right].to_numpy()).all(), f"卖出配对{left}不一致")
    for left, right in [("shares", "shares"), ("sell_price", "price"), ("sell_fee", "fee")]:
        compare_values(paired[left], sold[right], f"卖出配对{left}")
    compare_values(trips.net_pnl, trips.shares * (trips.sell_price - trips.buy_price) - trips.buy_fee - trips.sell_fee, "配对净盈亏")
    compare_values(trips.holding_days, (trips.sell_datetime - trips.buy_datetime).dt.total_seconds() / 86400, "片段持仓日数")
    remaining = buy_index.shares - sells.groupby("lot_id").shares.sum().reindex(buy_index.index, fill_value=0)
    validate_condition((remaining >= 0).all(), "卖空或超量卖出")
    compare_values(remaining, opened.set_index("lot_id").shares.reindex(buy_index.index, fill_value=0), "剩余股数", 0)
    compare_values(buy_index.fee, trips.groupby("lot_id").buy_fee.sum().reindex(buy_index.index, fill_value=0)
                   + opened.set_index("lot_id").buy_fee.reindex(buy_index.index, fill_value=0), "买入费分摊")
    compare_values(opened.buy_price, opened.lot_id.map(buy_index.price), "剩余买入价")
    pnl = float(trips.net_pnl.sum() + (opened.shares * (opened.mark_price - opened.buy_price) - opened.buy_fee).sum())
    pnl_error = compare_values(curve.equity.iloc[-1] - initial, pnl, "已实现加未实现盈亏", 1e-5)

    daily = curve.groupby(curve.datetime.dt.normalize()).last()
    validate_condition(((daily.datetime.dt.hour == 15) & (daily.datetime.dt.minute == 0)).all(), "非完整日终区间不适用此审计")
    held_values = (holdings.shares * holdings.mark_price).groupby(holdings.datetime.dt.normalize()).sum()
    compare_values(daily.equity - daily.cash, held_values.reindex(daily.index, fill_value=0), "每日持仓估值", 1e-5)
    returns = daily.equity.pct_change(fill_method=None)
    returns.iloc[0] = daily.equity.iloc[0] / initial - 1
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else None
    total = float(daily.equity.iloc[-1] / initial - 1)
    peaks = np.maximum.accumulate(np.r_[initial, daily.equity.to_numpy()])[1:]
    drawdown = float(np.min(daily.equity.to_numpy() / peaks - 1))
    days = (daily.index[-1] - daily.index[0]).days + 1
    annual = (1 + total) ** (365.25 / days) - 1
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
    completed = trips.loc[~trips.lot_id.isin(opened.lot_id)]
    per_lot = completed.groupby("lot_id").agg(pnl=("net_pnl", "sum"), buy=("buy_datetime", "first"), sell=("sell_datetime", "max"))
    expected = {"sharpe": sharpe if sharpe is not None else 0.0, "total_return": total, "annual_return": annual,
        "annual_volatility": std * np.sqrt(252), "max_drawdown": drawdown,
        "sortino": float(returns.mean() / downside * np.sqrt(252)) if downside else 0., "calmar": annual / abs(drawdown) if drawdown else 0.,
        "buy_count": len(buys), "sell_count": len(sells), "closed_lot_count": len(per_lot), "open_lots": len(opened),
        "fees": float(events.fee.sum()), "trade_lots": float(events.shares.sum() / 100),
        "win_rate": float((per_lot.pnl > 0).mean()) if len(per_lot) else 0.,
        "mean_holding_days": float(completed.holding_days.mean()) if len(completed) else 0.,
        "turnover": float(events.notional.sum() / daily.equity.mean()), "mean_exposure": float(curve.exposure.mean()),
        "short_exposure": 0, "trading_days": len(daily), "signal_lag_bars": 1}
    for name, value in expected.items():
        compare_values(metrics[name], value, f"指标{name}", 1e-8)
    return {"passed": True, "cash_error": cash_error, "pnl_error": pnl_error, "fee_error": fee_error,
        **expected, "sharpe": sharpe, "pending_lots": int(opened.pending_reason.fillna("").ne("").sum()),
        "mean_final_close_days": float((per_lot.sell - per_lot.buy).dt.total_seconds().mean() / 86400) if len(per_lot) else None}


def audit_run(directory: Path) -> Path:
    """只读directory四折标准文件与冻结cfg，核对年度汇总，写新审计目录及所有输入指纹。"""
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    years = json.loads((directory / "folds.json").read_text())
    rows, sources = [], [directory / "resolved_config.yaml", directory / "folds.json"]
    for fold in years:
        path = directory / str(fold["year"]) / "family_selected"
        frames = {key: pd.read_csv(path / file, float_precision="round_trip") for key, file in FILES.items()}
        rows.append({"year": fold["year"], **audit_account_frames(frames, cfg)})
        sources.extend(path / file for file in FILES.values())
    validate_condition(len(rows) == 4 and len({r["year"] for r in rows}) == 4, "须完整四年且不能重复")
    annual_table = pd.read_csv(directory / "test_year_metrics.csv", float_precision="round_trip").set_index("year")
    validate_condition(set(annual_table.index) == {r["year"] for r in rows} and len(annual_table) == 4, "年度汇总缺失或重复")
    for row in rows:
        for key in ["total_return", "annual_return", "annual_volatility", "max_drawdown", "buy_count", "closed_lot_count", "win_rate", "fees"]:
            compare_values(annual_table.loc[row["year"], key], row[key], f"年度汇总{key}", 1e-8)
        compare_values(annual_table.loc[row["year"], "sharpe"], row["sharpe"] or 0., "年度汇总夏普", 1e-8)
    valid = all(r["sharpe"] is not None for r in rows)
    mean = float(np.mean([r["sharpe"] for r in rows])) if valid else None
    summary = pd.read_csv(directory / "test_summary.csv", float_precision="round_trip")
    validate_condition(len(summary) == 1, "策略汇总须一行")
    if valid:
        compare_values(summary.iloc[0]["完整四年平均夏普"], mean, "四年等权均夏普", 1e-8)
    else:
        validate_condition(pd.isna(summary.iloc[0]["完整四年平均夏普"]), "现金年不得删除后平均")
    passed_target = bool(valid and mean > cfg["sharpe_target"] and not cfg["is_smoke"])
    validate_condition(bool(summary.iloc[0]["超过3"]) == passed_target, "目标判定不一致")
    sources += [directory / "test_year_metrics.csv", directory / "test_summary.csv"]
    out = directory / ("result_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "account_audit.csv", index=False, encoding="utf-8-sig")
    report = {"created": datetime.now().isoformat(), "passed": True, "mean_sharpe": mean, "target_reached": passed_target,
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "years": rows}
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    table = pd.DataFrame(rows)[["year", "passed", "cash_error", "pnl_error", "sharpe", "mean_final_close_days", "pending_lots"]].rename(columns={
        "year": "留出年", "passed": "账务通过", "cash_error": "最大现金误差元", "pnl_error": "期末盈亏误差元",
        "sharpe": "独立重算夏普", "mean_final_close_days": "按批次最后卖出平均自然日", "pending_lots": "期末待退出批次"})
    text = "# 独立CSV账本审计\n\n" + table.to_markdown(index=False)
    text += f"\n\n完整四年均夏普：{mean}；数值目标达到：{passed_target}。\n\n"
    text += "精确下一分钟、T+1、整手、10%总资产预算、费用、现金流、配对、剩余股数、日终持仓和指标重算通过。\n"
    text += "不读取模型预测调参，不重新撮合，也不证明原始行情、信号来源或价格限制代理正确。现金年夏普未定义；不删除年度。\n\n"
    text += "account_audit.csv为逐年结果，audit.json保存全部输入指纹和数值，本文为中文说明。\n"
    text += "mean_final_close_days按每个完全平仓批次最后卖出计自然日；原mean_holding_days按这些批次的部分卖出片段等权平均，两者不同。\n"
    text += "\n## 字段说明\n\n"
    text += "- year年份；passed账务校验通过；cash_error现金误差、pnl_error期末盈亏误差、fee_error单笔费用误差，均为元。\n"
    text += "- sharpe夏普；total_return总收益率；annual_return年化收益率；annual_volatility年化波动；max_drawdown最大回撤；sortino下行风险调整收益；calmar年化收益/最大回撤绝对值。\n"
    text += "- buy_count买入成交次数；sell_count卖出成交次数（含部分卖出）；closed_lot_count完全平仓批次；open_lots期末未平仓批次；pending_lots期末待退出批次。\n"
    text += "- fees佣金税费元（不重复计入价格内滑点）；trade_lots双边成交手数；win_rate完全平仓批次胜率；turnover双边成交额/平均日终权益。\n"
    text += "- mean_exposure时钟平均多头比例；short_exposure空头比例；trading_days交易日数；signal_lag_bars执行滞后，这里是精确下一分钟而非下一24分钟信号。\n"
    text += "- input_sha256输入文件指纹；auditor_sha256本审计脚本指纹；mean_sharpe完整四年等权均夏普；target_reached数值目标判定，不代表可实盘。\n"
    (out / "审计说明.md").write_text(text, encoding="utf-8")
    print(f"四年独立CSV账本审计通过：{out}；完整均夏普={mean}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读模型族策略四年CSV并重建账务，不重新回测")
    parser.add_argument("--run-dir", type=Path, required=True)
    audit_run(parser.parse_args().run_dir.resolve())
