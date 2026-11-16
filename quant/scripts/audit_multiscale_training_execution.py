"""从已训练多尺度实验诊断训练内部预测与成交约束的差异，不读取外层测试预测或收益。

在quant根目录传入--run-dir；只对各折calibration/validation窗口生成诊断预测，
写入新的training_execution_audit子目录，不改模型、规则、原缓存或测试文件，不连接券商。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.learned_exit_models import calculate_entry_target, select_window_rows
from scripts.run_multiscale_year_validation import load_multiscale_data, select_outer_feature_rows
from scripts.run_learned_exit_year_validation import generate_scores, calculate_sha256


def calculate_execution_diagnostics(frame: pd.DataFrame, market, cfg: dict, horizon: int, end: pd.Timestamp) -> dict:
    """对frame当期前5候选计算下一分钟独立买单容量及horizon标签覆盖，end限制标签不跨内部窗口。

    只用于训练标签/可成交性诊断：固定初始资产10%请求，忽略其他持仓、共享容量与退出端容量，
    所以容量加权收益是监督目标代理，绝不是可报告的组合回测收益。future字段只在此诊断标签使用。
    """
    if len(frame) == 0:
        return {"candidate_count": 0, "buyable_count": 0, "buyable_fraction": np.nan,
                "entry_price_limit_count": 0, "missing_entry_count": 0, "insufficient_board_lot_count": 0,
                "mean_budget_fill_fraction": np.nan, "label_covered_count": 0, "mean_prediction": np.nan,
                "mean_label_return": np.nan, "buyable_mean_label_return": np.nan, "capacity_weighted_label_proxy": np.nan}
    idx = market.times.get_indexer(frame.datetime)
    asset = np.array([market.asset_index[s] for s in frame.symbol])
    raw = market.values
    signal_price = raw["close"][idx, asset]
    execution = raw["execution_close"][idx, asset]
    previous = raw["previous_close"][idx, asset]
    budget = cfg["initial_cash"] * cfg["entry_equity_fraction"]
    capacity = np.nan_to_num(raw["execution_amount"][idx, asset], nan=0.) * cfg["max_bar_amount_fraction"]
    price = execution * (1 + cfg["slippage_rate"])
    valid = np.isfinite(price) & (price > 0) & (raw["execution_volume"][idx, asset] > 0)
    limit = execution / previous >= 1.095
    request = np.floor(budget / signal_price / cfg["lot_size"]) * cfg["lot_size"]
    available = np.floor(np.divide(np.minimum(budget, capacity), price,
                                  out=np.zeros(len(frame)), where=np.isfinite(price) & (price > 0)) / cfg["lot_size"]) * cfg["lot_size"]
    shares = np.minimum(request, available)
    filled = valid & ~limit & (shares >= cfg["lot_size"])
    ratio = np.where(filled, shares * price / budget, 0.)
    labelled = frame.assign(forward_return=frame[f"future_return_{horizon}d"], label_end=frame[f"future_end_{horizon}d"])
    covered = (labelled.label_end < end).to_numpy() & np.isfinite(labelled.forward_return)
    net = calculate_entry_target(labelled, cfg) / 10000
    feasible = covered & filled
    return {"candidate_count": len(frame), "buyable_count": int(filled.sum()), "buyable_fraction": float(filled.mean()),
            "entry_price_limit_count": int((valid & limit).sum()), "missing_entry_count": int((~valid).sum()),
            "insufficient_board_lot_count": int((valid & ~limit & ~filled).sum()),
            "mean_budget_fill_fraction": float(np.mean(ratio)), "label_covered_count": int(covered.sum()),
            "mean_prediction": float(frame.prediction.mean()),
            "mean_label_return": float(np.mean(net[covered])) if covered.any() else np.nan,
            "buyable_mean_label_return": float(np.mean(net[feasible])) if feasible.any() else np.nan,
            "capacity_weighted_label_proxy": float(np.mean(net[covered] * ratio[covered])) if covered.any() else np.nan}


def run_audit(directory: Path) -> Path:
    """核对directory冻结模型后，只检查每折内部窗口；输出新审计目录，不接受外层测试年作为诊断窗口。"""
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    frozen = json.loads((directory / "frozen_before_test.json").read_text())
    for relative, expected in frozen["source_hashes"].items():
        if calculate_sha256(ROOT / relative) != expected:
            raise ValueError(f"当前代码不是该实验冻结版本：{relative}")
    out = directory / "training_execution_audit" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    panel, market, matrices, samples, manifest, cache = load_multiscale_data(cfg)
    del matrices, samples
    rows = []
    for year, item in frozen["folds"].items():
        outer = select_outer_feature_rows(panel, market.times, int(year), cfg["feature_embargo_trading_days"])
        features = json.loads((directory / year / "entry_feature_schema.json").read_text())
        for profile in cfg["model_profiles"]:
            model_path = directory / year / f"entry_{profile['name']}.json"
            if calculate_sha256(model_path) != item["hashes"][model_path.name]:
                raise ValueError("模型哈希不符")
            model = XGBRegressor()
            model.load_model(model_path)
            model.set_params(n_jobs=cfg["workers"])
            for role in ["calibration", "validation"]:
                for window in item["fold"][role]:
                    if int(window["year"]) == int(year):
                        raise AssertionError("诊断窗口包含该折外层测试年")
                    bounds = {**window, "start": pd.Timestamp(window["start"]), "end": pd.Timestamp(window["end"])}
                    frame = select_window_rows(outer, [bounds], False)
                    frame = frame.loc[frame.cross_section_count >= cfg["min_cross_section"]]
                    if frame.empty:
                        continue
                    scores = generate_scores(model, frame, features, cfg)
                    top = scores.sort_values(["datetime", "prediction", "symbol"], ascending=[True, False, True]).groupby("datetime", sort=False).head(cfg["entry_top_n"])
                    top = top.loc[top.prediction >= cfg["entry_min_net_return"]]
                    candidates = top[["datetime", "symbol", "prediction"]].merge(frame[["datetime", "symbol", f"future_return_{profile['horizon']}d", f"future_end_{profile['horizon']}d"]],
                                                                                 on=["datetime", "symbol"], validate="one_to_one")
                    stats = calculate_execution_diagnostics(candidates, market, cfg, profile["horizon"], bounds["end"])
                    rows.append({"heldout_year": int(year), "role": role, "window_year": window["year"], "profile": profile["name"], **stats})
            print(f"完成仅内部成交诊断：{year}/{profile['name']}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "training_execution_diagnostics.csv", index=False, encoding="utf-8-sig")
    write_json(out / "audit_metadata.json", {"source_run": str(directory), "script_sha256": calculate_sha256(Path(__file__)),
        "frozen_sha256": calculate_sha256(directory / "frozen_before_test.json"), "outer_test_read": False,
        "specification": "所有非负预测的当期前5，不使用测试或实际资金持仓筛选；逐独立买意图容量估算，不是组合回测"})
    eligible = table.loc[table.candidate_count > 0]
    grouped = eligible.groupby(["heldout_year", "role"]).agg(candidates=("candidate_count", "sum"), buyable=("buyable_count", "sum"),
        price_limit=("entry_price_limit_count", "sum"), missing=("missing_entry_count", "sum"), insufficient_lot=("insufficient_board_lot_count", "sum"))
    grouped["buyable_fraction"] = grouped.buyable / grouped.candidates
    grouped = grouped.reset_index().rename(columns={"heldout_year": "留出年", "role": "内部阶段", "candidates": "候选意图数",
        "buyable": "可买意图数", "price_limit": "涨停代理阻挡", "missing": "缺失或零量", "insufficient_lot": "不足一手", "buyable_fraction": "可买比例"})
    grouped["内部阶段"] = grouped["内部阶段"].map({"calibration": "校准", "validation": "验证"})
    (out / "final_report.md").write_text("# 训练内部预测与执行一致性诊断\n\n没有读取外层测试预测或结果。候选按当期前5且基础预测非负，未应用校准门槛，多个模型可重复提出同一意图。\n\n"
        + grouped.to_markdown(index=False) + "\n\n候选不是实际买入，不能用此表宣称策略收益或夏普。每笔独立假设初始总资产10%请求，容量、涨停代理、整手和零量按当前引擎规则计算；未模拟真实持仓、共享容量、现金竞争和退出端容量。\n\n"
        "CSV的mean_prediction为预测收益比例；mean_label_return为完整内部标签均值，buyable_mean_label_return只针对入口可成交的诊断子集，capacity_weighted_label_proxy是容量缩量后的监督目标代理，均不是组合回测收益。该未来入口资格只能用于训练标签/诊断，不能加入预测特征或提前删除测试候选。\n", encoding="utf-8")
    (out / "输出文件说明.md").write_text("# 文件说明\n\n- final_report.md：内部候选成交覆盖与限制。\n- training_execution_diagnostics.csv：heldout_year留出年、role内部阶段、window_year实际训练年份、profile模型档，以及意图数、入口阻挡、容量比例和标签代理。\n- audit_metadata.json：冻结来源、诊断代码哈希、未读取外层声明与口径。\n", encoding="utf-8")
    print(f"仅训练内部审计完成：{out}", flush=True)
    return out


def run_cli() -> None:
    """解析已冻结实验的--run-dir，只生成内部诊断，不能指定外层测试时间。"""
    parser = argparse.ArgumentParser(description="仅训练内部的多尺度预测成交诊断")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.run_dir.resolve())


if __name__ == "__main__":
    run_cli()
