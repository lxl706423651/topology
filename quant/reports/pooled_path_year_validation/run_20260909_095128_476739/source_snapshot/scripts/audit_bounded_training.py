"""核对有界重训的真实拟合行及来源，输入--run-dir冻结模型和对应可信pkl。

复用通用完整性审计，并从缓存重建每个模型的拟合行，检查标签/来源与年份。
生成新中文审计子目录；不拟合、不计算预测、不读取外层收益，不修改旧模型或缓存。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.bounded_training import create_finite_factor_names, select_bounded_rows
from quant_backtest.learned_exit_models import select_window_rows
from quant_backtest.multiscale_training import CONTEXT_NAMES, select_exit_samples
from scripts.audit_multiscale_frozen_training import audit_frozen_training
from scripts.run_multiscale_year_validation import build_horizon_training
from scripts.run_learned_exit_year_validation import calculate_sha256


def validate_core_sources(frame: pd.DataFrame, heldout_year: int) -> None:
    """验证frame的datetime/feature_source_start/label_end：来源在同年、标签同年且严格晚于信号，排除heldout_year。"""
    year_start = pd.to_datetime(frame.datetime.dt.year.astype(str) + "-01-01")
    if frame.feature_source_start.isna().any() or (frame.feature_source_start < year_start).any():
        raise ValueError("核心拟合来源跨入上年内部验证或未知")
    if (frame.feature_source_start > frame.datetime).any():
        raise ValueError("核心来源晚于当前信号")
    if (frame.datetime.dt.year == heldout_year).any() or (frame.label_end.dt.year != frame.datetime.dt.year).any():
        raise ValueError("核心拟合或标签进入留出年/跨年")
    if not (frame.label_end > frame.datetime).all():
        raise ValueError("核心标签终点不晚于信号")


def audit_bounded_training(directory: Path) -> Path:
    """重建directory冻结四折的每档实际拟合行及入口来源，与保存split_audit逐项对照，只作来源审计。"""
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    if cfg.get("data_contract") != "bounded_observation_v1":
        raise ValueError("仅接受有界数据契约")
    integrity = audit_frozen_training(directory)
    frozen = json.loads((directory / "frozen_before_test.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "data_metadata.json").read_text(encoding="utf-8"))
    cache = metadata["exit_cache"]
    panel = pd.read_pickle(cache["bounded_cache_path"])
    samples = pd.read_pickle(cache["exit_cache_path"])["samples"]
    symbol_path = next(iter(sorted(Path(metadata["annual_cache"]).glob("S[HZ].*.pkl"))))
    times = pd.DatetimeIndex(pd.read_pickle(symbol_path)["market"].datetime)
    recorded = pd.read_csv(directory / "split_audit.csv")
    allowed = {f"cs_rank_{c}" for c in create_finite_factor_names()} | (set(CONTEXT_NAMES) - {
        "cs_rank_context_momentum_60d", "cs_rank_context_drawdown_60d"})
    rows = []
    for year, item in frozen["folds"].items():
        fold = item["fold"]
        windows = [{"year": w["year"], "start": pd.Timestamp(w["start"]), "end": pd.Timestamp(w["end"])} for w in fold["fit"]]
        outer = select_bounded_rows(panel, times, int(year), cfg["feature_embargo_trading_days"])
        outer = outer.loc[(outer.cross_section_count >= cfg["min_cross_section"]) & outer.fit_source_allowed]
        schema = json.loads((directory / year / "entry_feature_schema.json").read_text(encoding="utf-8"))
        if not set(schema).issubset(allowed):
            raise ValueError("模型仍有白名单之外的递推/来源/标签特征")
        sources = outer.set_index(["datetime", "symbol"]).feature_source_start
        exit_train = select_window_rows(select_exit_samples(samples, outer), windows)
        if len(exit_train) > cfg["max_exit_training_rows"]:
            exit_train = exit_train.sample(cfg["max_exit_training_rows"], random_state=cfg["model"]["random_state"]).sort_index()
        entry_keys = pd.MultiIndex.from_arrays([exit_train.entry_datetime - pd.Timedelta(minutes=1), exit_train.symbol])
        exit_train = exit_train.assign(feature_source_start=sources.reindex(entry_keys).to_numpy())
        frames = [(p["name"], build_horizon_training(outer, windows, p["horizon"])) for p in cfg["model_profiles"]]
        frames.append(("exit_model", exit_train))
        for model, frame in frames:
            validate_core_sources(frame, int(year))
            known = recorded.loc[(recorded.test_year == int(year)) & (recorded.model == model)]
            if int(known.rows.sum()) != len(frame) or set(known.fit_year) != set(frame.datetime.dt.year):
                raise ValueError(f"拟合行数或年份无法重建：{year}/{model}")
            for y, part in frame.groupby(frame.datetime.dt.year):
                expected = known.loc[known.fit_year == y].iloc[0]
                if expected.rows != len(part) or pd.Timestamp(expected.first_signal) != part.datetime.min() or pd.Timestamp(expected.last_signal) != part.datetime.max() or pd.Timestamp(expected.last_label) != part.label_end.max():
                    raise ValueError("拟合逐年首末信号/标签摘要无法重建")
                rows.append({"留出年": int(year), "模型代码": model, "实际拟合年": int(y), "拟合行数": len(part),
                             "最早量价来源": part.feature_source_start.min(), "首个信号": part.datetime.min(),
                             "最晚标签": part.label_end.max(), "来源与年份检查": "通过"})
    out = directory / "bounded_source_audit" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "真实拟合来源核对.csv", index=False, encoding="utf-8-sig")
    write_json(out / "audit_metadata.json", {"source_run": str(directory), "integrity_audit": str(integrity),
        "audit_script_sha256": calculate_sha256(Path(__file__)), "model_count": len(frozen["folds"]) * (len(cfg["model_profiles"]) + 1),
        "fitting_year_records": len(rows), "outer_results_read": False, "model_fitting_performed": False,
        "exit_source_note": "持仓列来源以已核验的入场来源包络为起点；其后有限ATR/前收与格点窗口单调向前，同一拟合年标签不越界"})
    (out / "final_report.md").write_text("# 有界拟合来源审计\n\n从真实缓存重建全部开仓与持仓拟合行，与保存年份、行数、首末信号及标签逐项一致。"
        "所有开仓入模列在有限白名单内；核心来源不跨上年，标签同年且不进入留出年。没有读取外层收益，也没有重训。\n\n"
        + table.to_markdown(index=False)
        + "\n\n同一原始样本可能被多个模型重复使用，不能把表内行数相加当独立样本数。持仓来源检查以虚拟入场包络为基础，"
          "不是重算所有持仓特征；其后有限窗口/前收来源不早于入口包络的性质由合成测试和数据构造契约保证。"
          "这不消除非历史ST、复权、未来年份训练、反复历史研究及成交代理偏差，不证明收益目标。\n", encoding="utf-8")
    (out / "输出文件说明.md").write_text("# 文件角色\n\n- final_report.md：来源及拟合重建结论与限制。\n"
        "- 真实拟合来源核对.csv：每折、每模型、每实际拟合年的行数和最早量价来源。\n"
        "- audit_metadata.json：通用冻结审计路径、本脚本指纹及未读取外层结果声明。\n", encoding="utf-8")
    print(f"有界拟合来源审计通过：{out}", flush=True)
    return out


def run_cli() -> None:
    """解析有界冻结实验的--run-dir，执行不预测外层的来源审计。"""
    parser = argparse.ArgumentParser(description="从真实pkl重建有界模型拟合来源")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_bounded_training(args.run_dir.resolve())


if __name__ == "__main__":
    run_cli()
