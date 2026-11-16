"""验收已完成的CTA内部验证搜索，不训练、不选参、不读取2026测试结果。

在quant根目录传入最终报告、前一完整阶段和中断检查点；读取现有指标/账本/审计，
生成工程验收JSON、源码快照及中文完整批次表。原模型、行情与搜索输出不修改。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.cta_research_data import calculate_sha256
from quant_backtest.annual_panel import write_json


def read_metrics(directory: Path) -> pd.DataFrame:
    """读取directory完整候选CSV，保留浮点往返精度以检查性能优化是否改变数值。"""
    return pd.read_csv(directory / "validation_rule_metrics.csv", float_precision="round_trip")


def verify_completion(directory: Path, previous: Path, interrupted: Path) -> tuple[dict, pd.DataFrame]:
    """检查directory三折目标、previous完整阶段和interrupted旧检查点，返回证据与完整批次表。"""
    old = read_metrics(interrupted).set_index(["rule_id", "year"]).sort_index()
    first = read_metrics(previous)
    pd.testing.assert_frame_equal(old, first.set_index(["rule_id", "year"]).sort_index().loc[old.index], check_exact=True)
    stages, candidate_ids = [], set()
    for stage in [previous, directory]:
        table = read_metrics(stage)
        protocol = json.loads((stage / "protocol.json").read_text())
        summary = json.loads((stage / "summary.json").read_text())
        expected_ids = {rule["rule_id"] for rule in protocol["rules"]}
        assert set(table.rule_id) == expected_ids and len(table) == 3 * len(expected_ids)
        assert not table.duplicated(["rule_id", "year"]).any()
        assert table.groupby("rule_id").year.apply(lambda years: set(years) == {2023, 2024, 2025}).all()
        assert summary["candidate_count"] == len(expected_ids) and summary["account_count"] == len(table)
        assert summary["test_2026_evaluated"] is False
        candidate_ids.update(expected_ids)
        audits = sorted(stage.glob("independent_audit_*.json"))
        assert audits, "完整阶段缺少独立审计"
        audit = json.loads(audits[-1].read_text())
        assert audit["passed"] and len(audit["prediction_checks"]) == 24
        assert audit["candidate_account_count"] == len(table)
        stages.append({"report": str(stage), "candidates": len(expected_ids), "accounts": len(table),
                       "independent_audit": str(audits[-1]), "audit_sha256": calculate_sha256(audits[-1])})
        for path, digest in json.loads((stage / "source_hashes.json").read_text()).items():
            assert calculate_sha256(Path(path)) == digest, "实际执行源码与当前版本不一致"

    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    execution = yaml.safe_load((directory / "execution_config.yaml").read_text())
    chosen = json.loads((directory / "selected_rule.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    selected = pd.read_csv(directory / "selected_validation_metrics.csv", float_precision="round_trip").set_index("year")
    protocol = json.loads((directory / "protocol.json").read_text())
    assert list(selected.index) == [2023, 2024, 2025]
    assert summary["all_validation_folds_pass"] is True and summary["independent_generalization_proven"] is False
    assert not cfg["is_smoke"] and cfg["sharpe_target"] == 2.0 and cfg["minimum_closed_lots_per_fold"] >= 20
    assert execution["entry_equity_fraction"] == 0.1 and execution["commission_rate"] == 0.0008
    assert execution["min_commission"] == 0 and execution["entry_top_n"] == 5
    folds, reviews = [], []
    for fold in protocol["folds"]:
        year = int(fold["year"])
        start, end = pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"])
        assert (str(start.date()), str(end.date())) == {
            2023: ("2023-07-01", "2023-10-01"), 2024: ("2024-01-01", "2025-01-01"),
            2025: ("2025-01-01", "2026-01-01")}[year]
        folder = directory / f"validation_{year}"
        events = pd.read_csv(folder / "test_trade_events.csv", float_precision="round_trip",
                             parse_dates=["datetime", "signal_datetime"])
        trips = pd.read_csv(folder / "test_round_trips.csv", float_precision="round_trip",
                            parse_dates=["buy_datetime", "sell_datetime", "sell_signal_datetime"])
        opened = pd.read_csv(folder / "test_open_positions.csv", float_precision="round_trip")
        curve = pd.read_pickle(folder / "minute_equity_curve.pkl")
        daily = pd.read_csv(folder / "test_equity_curve.csv", float_precision="round_trip", parse_dates=["datetime"])
        expected_daily = curve.groupby(curve.datetime.dt.normalize(), sort=True).last()
        assert ((curve.datetime >= start) & (curve.datetime < end)).all()
        np.testing.assert_array_equal(daily.datetime, expected_daily.datetime)
        np.testing.assert_array_equal(daily.equity, expected_daily.equity)
        equity = daily.equity.to_numpy()
        returns = np.r_[equity[0] / execution["initial_cash"] - 1, equity[1:] / equity[:-1] - 1]
        np.testing.assert_allclose(daily.daily_return, returns, rtol=0, atol=1e-15)
        sd = returns.std(ddof=1)
        assert sd > 0
        sharpe = float(returns.mean() / sd * np.sqrt(252))
        closed = trips.loc[~trips.lot_id.isin(opened.lot_id)]
        assert sharpe > cfg["sharpe_target"] and closed.lot_id.nunique() >= cfg["minimum_closed_lots_per_fold"]
        assert abs(sharpe - selected.loc[year, "sharpe"]) < 1e-10
        assert closed.lot_id.nunique() == selected.loc[year, "closed_lot_count"]
        assert selected.loc[year, "signal_lag_bars"] == 1 and selected.loc[year, "short_exposure"] == 0
        buys = events.loc[events.side.eq("buy")]
        assert (buys.time_index == buys.signal_index + 1).all()
        assert (buys.prediction > chosen["gate"]).all()
        assert (trips.sell_datetime.dt.normalize() > trips.buy_datetime.dt.normalize()).all()
        if chosen["frequency"] == 240:
            assert buys.signal_datetime.dt.strftime("%H:%M").eq("09:31").all()
            assert buys.datetime.dt.strftime("%H:%M").eq("09:32").all()
        style = {"breakout": "十日突破", "trend": "趋势", "pullback": "回调",
                 "recovery": "回撤恢复", "model_only": "仅模型"}[chosen["style"]]
        objective = "盈利概率分数" if chosen["model_name"].endswith("probability") else "预测净收益"
        for lot_id, fragments in closed.groupby("lot_id"):
            buy = buys.loc[buys.lot_id.eq(lot_id)].iloc[0]
            assert fragments.shares.sum() == buy.shares
            assert abs(float(fragments.buy_fee.sum()) - buy.fee) < 1e-7
            final_exit = fragments.sell_datetime.max()
            reviews.append({"验证年份": year, "独立批次号": int(lot_id), "股票代码": buy.symbol,
                "买入时间": buy.datetime, "买入价": buy.price, "买入股数": int(buy.shares),
                "买入时模型分数": buy.prediction, "买入条件": f"{style}且{objective}>{chosen['gate']}",
                "首次退出触发时间": fragments.sell_signal_datetime.min(), "最终平仓时间": final_exit,
                "平仓原因": "；".join(fragments.sell_reason.drop_duplicates()),
                "卖出片段数": len(fragments), "卖出加权均价": float((fragments.sell_price * fragments.shares).sum() / buy.shares),
                "双边费用元": float(fragments.buy_fee.sum() + fragments.sell_fee.sum()),
                "扣费净盈亏元": float(fragments.net_pnl.sum()),
                "完整持仓自然日": (final_exit - buy.datetime).total_seconds() / 86400})
        folds.append({"year": year, "start": str(start), "end_exclusive": str(end), "sharpe": sharpe,
                      "closed_lots": int(closed.lot_id.nunique()), "buy_count": len(buys), "open_lots": len(opened),
                      "source_metrics_sha256": calculate_sha256(folder / "test_metrics.csv")})

    worst = directory / f"validation_{int(selected.sharpe.idxmin())}"
    copied = []
    for path in worst.iterdir():
        if path.name.startswith("test_") or path.name == "测试集收益曲线.png":
            assert calculate_sha256(path) == calculate_sha256(directory / path.name)
            copied.append(path.name)
    result = {"passed": True, "scope": "原协议三段反复优化过的内部验证，不是独立测试或实盘保证",
        "stages": stages, "unique_common_rules_across_stages": len(candidate_ids),
        "performance_optimization_parity": {"passed": True, "old_rules": int(old.index.get_level_values(0).nunique()),
            "old_accounts": len(old), "exact_numeric_equality": True, "old_report": str(interrupted)},
        "selected_rule": chosen, "folds": folds, "compatible_files_equal_weakest_fold": copied,
        "test_2026_evaluated": False, "live_approved": False, "verifier_sha256": calculate_sha256(Path(__file__))}
    return result, pd.DataFrame(reviews)


def run_verification() -> None:
    """读取CLI三个已有目录，生成带时间戳验收及中文表；输入文件不覆盖。"""
    parser = argparse.ArgumentParser(description="验收三折CTA研究和性能优化一致性，不重新回测")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--previous-stage", type=Path, required=True)
    parser.add_argument("--interrupted-stage", type=Path, required=True)
    args = parser.parse_args()
    result, review = verify_completion(args.report_dir, args.previous_stage, args.interrupted_stage)
    output = args.report_dir / "engineering_validation"
    output.mkdir(exist_ok=True)
    review_path = output / "closed_batch_review.csv"
    encoded = review.to_csv(index=False).encode("utf-8-sig")
    if review_path.exists() and review_path.read_bytes() != encoded:
        raise ValueError("中文批次表已存在且内容不同，拒绝覆盖")
    review_path.write_bytes(encoded)
    result["closed_batch_review"] = {"rows": len(review), "sha256": calculate_sha256(review_path)}
    filename = output / f"completion_checks_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(filename, result)
    shutil.copy2(Path(__file__), filename.with_suffix(".py"))
    print(f"最后验收通过：{filename}；完整批次{len(review)}；最低折夏普{min(f['sharpe'] for f in result['folds']):.6f}")


if __name__ == "__main__":
    run_verification()
