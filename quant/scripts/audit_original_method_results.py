"""在quant根目录只读核对原XGB/CTA四年双口径CSV，输入--run-dir已完成报告。

输出新时间戳审计目录、中文账务表及交易重合表；不读预测调参，不训练或重新撮合。
约束版采用独立账本审计，旧式仅核对已声明的有偏账务，不把同刻卖出伪装成下一分钟。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.original_method_data import calculate_file_hash
from scripts.audit_family_path_results import FILES, DATES, compare_values, validate_condition
from scripts.run_original_method_year_validation import audit_original_result, METHODS, MODES


def compare_method_trades(first: dict, second: dict, initial: float) -> dict:
    """比较first/second两套账本买入键和日收益，initial为初始权益；仅解释重合，不选优。"""
    keys, returns = [], []
    for frames in [first, second]:
        events = frames["events"]
        buys = events.loc[events.side == "buy", ["datetime", "symbol"]]
        keys.append(set(buys.itertuples(index=False, name=None)))
        curve = frames["curve"]
        daily = curve.groupby(curve.datetime.dt.normalize()).equity.last()
        change = daily.pct_change(fill_method=None)
        change.iloc[0] = daily.iloc[0] / initial - 1
        returns.append(change)
    validate_condition(returns[0].index.equals(returns[1].index), "两方法日终日历不同")
    common, union = len(keys[0] & keys[1]), len(keys[0] | keys[1])
    correlation = float(returns[0].corr(returns[1])) if returns[0].std() > 0 and returns[1].std() > 0 else None
    return {"方法一买入键数": len(keys[0]), "方法二买入键数": len(keys[1]), "共同买入键数": common,
        "买入交并比": common / union if union else None, "日收益相关系数": correlation,
        "净值逐点完全相同": first["curve"]["equity"].equals(second["curve"]["equity"]),
        "最大日收益差": float((returns[0] - returns[1]).abs().max())}


def audit_run(directory: Path) -> Path:
    """读取directory全部16组CSV与汇总，重算账务/均值；成功后写新审计及输入/脚本指纹。"""
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    annual = pd.read_csv(directory / "test_year_metrics.csv", float_precision="round_trip")
    summary = pd.read_csv(directory / "test_summary.csv", float_precision="round_trip")
    expected = {(y, mode, method) for y in cfg["test_years"] for mode in MODES for method in METHODS}
    actual = set(annual[["year", "mode", "method"]].itertuples(index=False, name=None))
    validate_condition(len(annual) == 16 and actual == expected, "须完整四年双方法双口径，不得重复或少报")
    validate_condition(len(summary) == 4 and not summary.duplicated(["mode", "method"]).any(), "须四组完整汇总")
    sources = [directory / name for name in ["resolved_config.yaml", "test_year_metrics.csv", "test_summary.csv"]]
    rows, comparisons = [], []
    for year in cfg["test_years"]:
        for mode in MODES:
            method_frames = {}
            for method in METHODS:
                path = directory / str(year) / mode / method
                frames = {key: pd.read_csv(path / file, float_precision="round_trip") for key, file in FILES.items()}
                for key, columns in DATES.items():
                    for column in columns:
                        frames[key][column] = pd.to_datetime(frames[key][column], errors="raise")
                sources.extend(path / file for file in FILES.values())
                result = {**frames, "metrics": frames["metrics"].iloc[0].to_dict()}
                checked = audit_original_result(result, mode, cfg)
                row = annual.loc[(annual.year == year) & (annual["mode"] == mode) & (annual.method == method)].iloc[0]
                for column, value in result["metrics"].items():
                    if isinstance(value, str):
                        validate_condition(row[column] == value, f"年度汇总文本{column}不一致")
                    else:
                        compare_values(row[column], value, f"年度汇总{year}/{mode}/{method}/{column}", 1e-8)
                rows.append({"year": year, "mode": mode, "method": method, **checked})
                method_frames[method] = frames
            comparisons.append({"年份": year, "执行口径": MODES[mode],
                **compare_method_trades(method_frames["method1"], method_frames["method2"], cfg["initial_cash"])})
    for mode in MODES:
        for method in METHODS:
            selected = annual.loc[(annual["mode"] == mode) & (annual.method == method)]
            row = summary.loc[(summary["mode"] == mode) & (summary.method == method)]
            validate_condition(len(row) == 1, "缺少固定策略汇总")
            row = row.iloc[0]
            valid = (selected.annual_volatility > 0).all()
            mean = float(selected.sharpe.mean()) if valid else None
            if valid:
                compare_values(row.mean_sharpe, mean, "四年等权平均夏普", 1e-8)
            else:
                validate_condition(pd.isna(row.mean_sharpe), "现金年不得丢弃后求均值")
            validate_condition(row.numeric_above3 == bool(valid and mean > cfg["sharpe_target"]), "达标标记不一致")
            validate_condition(row.valid_years == int((selected.annual_volatility > 0).sum()), "有波动年数错误")
            validate_condition(not row.live_ready, "当前研究不能标记实盘就绪")
    out = directory / datetime.now().strftime("csv_audit_%Y%m%d_%H%M%S_%f")
    out.mkdir(exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "account_audit.csv", index=False, encoding="utf-8-sig")
    overlap = pd.DataFrame(comparisons)
    overlap.to_csv(out / "交易重合比较.csv", index=False, encoding="utf-8-sig")
    code = [Path(__file__), ROOT / "scripts/audit_family_path_results.py", ROOT / "scripts/run_original_method_year_validation.py"]
    write_json(out / "audit.json", {"passed": True, "created": datetime.now().isoformat(),
        "input_sha256": {str(p.relative_to(directory)): calculate_file_hash(p) for p in sources},
        "audit_code_sha256": {str(p.relative_to(ROOT)): calculate_file_hash(p) for p in code},
        "groups": len(rows), "max_cash_error": float(table.cash_error.max()), "max_pnl_error": float(table.pnl_error.max())})
    text = "# 原两方法落盘账务与重合核对\n\n16组落盘CSV与年度汇总、四年等权均值核对通过。\n\n"
    text += overlap.to_markdown(index=False) + "\n\n"
    text += "买入键为成交时间与股票，不要求股数相同；交并比=共同键数/全部不同键数。日收益相关包含现金日。\n"
    text += "净值逐点完全相同直接比较账本，不比较标题不同的图片字节。此表仅解释相似度，不据此回选方法。\n\n"
    text += "account_audit.csv记录账务误差及指标：year年份、mode执行口径、method方法、passed通过、cash_error现金误差元、pnl_error期末盈亏误差元、fee_error费用误差元、sharpe夏普。其余交易字段见年度输出文件说明。\n"
    text += "mean_final_close_days为每个完全平仓批次到最后一次卖出的平均自然日；pending_lots为期末待退出批次。\n"
    text += "旧式same_bar_exit_biased=True明确标记同刻卖出，budget_basis记录卖出后成交权益预算；通过账务检查不代表满足信号滞后。\n"
    text += "约束版核验精确下一分钟、T+1、100股、10%总资产预算、税费、现金、配对、库存、日终持仓及指标；不重验原行情撮合，也不证明研究样本独立。\n"
    text += "audit.json保留本次输入和审计代码指纹。审计不修改模型、规则或原报告，也不重新回测。\n"
    (out / "审计说明.md").write_text(text, encoding="utf-8")
    print(f"16组落盘账务核对通过：{out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    audit_run(parser.parse_args().run_dir.resolve())
