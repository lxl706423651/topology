"""独立复核CTA已保存报告的训练边界、模型身份、滑点及每日账务，不重训或重新选参。

输入--report-dir；只读该轮CSV/PKL/原生模型和来源缓存，新增带时间戳的审计JSON。
不改变原始报告、规则、预测或行情，审计通过也不能弥补历史数据覆盖与公司行动缺口。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.cta_research_data import MinuteMarket, calculate_sha256, calculate_identity, read_history, find_history_path
from quant_backtest.cta_research_reporting import audit_result


def audit_market_status(market, manifest: dict) -> dict:
    """独立对照manifest原始历史状态与market可执行标记，拒绝非正常日或ST买入资格。"""
    days = market.times[::240].normalize()
    blocked_active = 0
    for asset, symbol in enumerate(market.symbols):
        history = read_history(Path(manifest["history_path"]), symbol).set_index("date").reindex(days)
        normal = np.repeat((history.tradestatus.eq(1) & history.isST.isin([0, 1])).to_numpy(), 240)
        allowed_buy = np.repeat((history.tradestatus.eq(1) & history.isST.eq(0)).to_numpy(), 240)
        if market.flags[~normal, asset, :2].any() or market.flags[~allowed_buy, asset, 0].any():
            raise AssertionError(f"{symbol}在非正常/未知状态或ST买入时仍有交易资格")
        blocked_active += int((~normal & (market.quotes[:, asset, 3] > 0)).sum())
    return {"passed": True, "symbols_checked": len(market.symbols), "blocked_active_minutes": blocked_active}


def audit_saved_report(directory: Path) -> dict:
    """从directory落盘文件独立重建训练输入及六个账户账本，失败时抛异常，不发布成功标记。"""
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    manifest = json.loads((directory / "data_manifest.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    frozen = json.loads((directory / "frozen_before_test.json").read_text())
    for symbol, expected in manifest["identity"]["history_files"].items():
        assert calculate_sha256(find_history_path(Path(manifest["history_path"]), symbol)) == expected, symbol
    for relative, size, modified in manifest["identity"]["files"]:
        current = (Path(cfg["data_root"]) / relative).stat()
        assert (current.st_size, current.st_mtime_ns) == (size, modified), relative
    for relative, expected in frozen["files"].items():
        assert calculate_sha256(directory / relative) == expected, relative
    for relative, expected in protocol["sources"].items():
        assert calculate_sha256(directory / "source_snapshot" / relative.lstrip("/")) == expected, relative
    cache = Path(manifest["cache_path"])
    assert calculate_sha256(cache / "panel.pkl") == manifest["panel_sha256"]
    panel = pd.read_pickle(cache / "panel.pkl")
    label_identity = json.loads((directory / "label_manifest.json").read_text())
    labels = pd.read_pickle(cache / f"path_labels_{calculate_identity(label_identity)}.pkl")
    models = []
    for path in sorted(directory.glob("**/model_metadata.json")):
        meta = json.loads(path.read_text())
        folder = path.parent
        rows = pd.read_csv(folder / "training_rows.csv", index_col="row_id", parse_dates=["datetime", "label_end"])
        cutoff = pd.Timestamp(meta["fold"]["train_end"])
        assert (rows.datetime < cutoff).all() and (rows.label_end < cutoff).all()
        assert len(rows) == meta["fit_rows"]
        assert calculate_sha256(folder / "training_rows.csv") == meta["training_input_sha256"]
        assert calculate_sha256(folder / "xgboost_model.json") == meta["model_sha256"]
        core = panel.loc[rows.index]
        assert core.datetime.tolist() == rows.datetime.tolist() and core.symbol.tolist() == rows.symbol.tolist()
        values_hash = calculate_identity(pd.util.hash_pandas_object(core[meta["features"]], index=True).tolist())
        assert values_hash == meta["feature_values_sha256"]
        profile = meta["exit_profile"]
        reference = labels.loc[rows.index]
        assert reference[f"label_end_{profile}"].tolist() == rows.label_end.tolist()
        assert np.allclose(reference[f"net_return_{profile}"], rows.target_net_return, atol=1e-12, rtol=1e-12)
        expected_weights = 1 / rows.groupby(rows.datetime.dt.normalize()).symbol.transform("size")
        expected_weights /= expected_weights.mean()
        assert np.allclose(expected_weights, rows.sample_weight)
        model = xgb.XGBRegressor()
        model.load_model(folder / "xgboost_model.json")
        assert model.get_booster().feature_names == meta["features"]
        assert model.get_booster().num_boosted_rounds() == meta["model_profile"]["n_estimators"]
        models.append({"path": str(folder.relative_to(directory)), "rows": len(rows), "features": len(meta["features"]),
                       "last_label": str(rows.label_end.max()), "fit_cutoff": str(cutoff)})
    market = MinuteMarket(cache / "minute_market")
    status_audit = audit_market_status(market, manifest)
    accounts = []
    folders = sorted(directory.glob("year_*/backtest")) + [directory / "continuous", directory / "cost_stress"]
    for folder in folders:
        events = pd.read_csv(folder / "test_trade_events.csv", parse_dates=["datetime", "signal_datetime"])
        trips = pd.read_csv(folder / "test_round_trips.csv", parse_dates=["buy_datetime", "sell_datetime", "buy_signal_datetime", "sell_signal_datetime"])
        opened = pd.read_csv(folder / "test_open_positions.csv", parse_dates=["buy_datetime"])
        curve = pd.read_pickle(folder / "minute_equity_curve.pkl")
        result = {"events": events, "trips": trips, "open": opened, "curve": curve}
        local_cfg = {**cfg, "slippage_rate": cfg["slippage_rate"] * (2 if folder.name == "cost_stress" else 1)}
        report = audit_result(result, market, local_cfg)
        signs = np.where(events.side.eq("buy"), 1, -1)
        unrounded = events.reference_price * (1 + signs * local_cfg["slippage_rate"])
        expected = np.where(signs == 1, np.ceil(unrounded * 100 - 1e-8), np.floor(unrounded * 100 + 1e-8)) / 100
        assert np.allclose(expected, events.price, atol=1e-9, rtol=0)
        daily = curve.groupby(curve.datetime.dt.normalize()).last()
        saved = pd.read_csv(folder / "test_equity_curve.csv")
        assert np.allclose(daily.equity, saved.equity, atol=1e-7, rtol=0)
        returns = np.r_[daily.equity.iloc[0] / cfg["initial_cash"] - 1, daily.equity.to_numpy()[1:] / daily.equity.to_numpy()[:-1] - 1]
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252) if returns.std(ddof=1) > 0 else 0.0
        metrics = pd.read_csv(folder / "test_metrics.csv").iloc[0]
        assert abs(sharpe - metrics.sharpe) < 1e-8
        accounts.append({"path": str(folder.relative_to(directory)), **report, "daily_sharpe_recomputed": sharpe})
    return {"passed": True, "models": models, "accounts": accounts, "frozen_files": len(frozen["files"]),
            "source_snapshots": len(protocol["sources"]), "status_audit": status_audit,
            "audit_code_sha256": calculate_sha256(Path(__file__)),
            "is_smoke": cfg["is_smoke"], "strategy_goal_achieved": False}


def run_audit() -> None:
    """解析只读报告目录并保存唯一审计文件，已有模型或报告不覆盖。"""
    parser = argparse.ArgumentParser(description="独立复核XGBoost CTA训练与落盘账本")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit_saved_report(args.report_dir)
    path = args.report_dir / datetime.now().strftime("independent_audit_%Y%m%d_%H%M%S.json")
    write_json(path, result)
    print(f"独立复核通过：{len(result['models'])}模型、{len(result['accounts'])}账户；{path}")


if __name__ == "__main__":
    run_audit()
