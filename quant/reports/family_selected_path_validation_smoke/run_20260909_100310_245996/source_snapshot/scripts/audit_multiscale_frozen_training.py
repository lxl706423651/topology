"""在quant根目录审计--run-dir的多尺度冻结模型、门槛和年度边界，不评价策略收益。

输入可信本机模型、配置、校准/验证表及单股pkl的共同时钟；生成新的中文审计子目录。
只以全零人工输入检查模型API，不读取测试预测/收益，不修改模型或缓存，不连接券商。
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
from quant_backtest.learned_exit_models import EXIT_FEATURES
from scripts.run_learned_exit_year_validation import calculate_sha256


def validate_threshold_contract(training: pd.DataFrame, validation: pd.DataFrame, rules: dict) -> int:
    """核对training/validation按方法和规则唯一，以及rules冻结门槛；返回实际检查的入围候选数。

    CSV调用者应使用float_precision='round_trip'保留浮点序列化值。现金保护不对应入围赢家，
    但校准与验证表中的所有实际候选仍要逐一检查，不能只检查最后选择。
    """
    keys = ["method", "rule_id"]
    if training.duplicated(keys).any() or validation.duplicated(keys).any():
        raise ValueError("方法与规则键不唯一")
    joined = validation.merge(training[keys + ["threshold"]], on=keys, how="left", validate="one_to_one",
                              suffixes=("_validation", "_calibration"), indicator=True)
    if not (joined._merge == "both").all():
        raise ValueError("验证候选没有对应校准记录")
    if not np.array_equal(joined.threshold_validation.to_numpy(), joined.threshold_calibration.to_numpy()):
        raise ValueError("校准与验证门槛不一致")
    for method, rule in rules.items():
        if rule["rule_id"] == "cash_validation_failed":
            if rule["threshold"] != 1e9:
                raise ValueError("现金保护门槛不符合固定协议")
            continue
        row = validation.loc[(validation.method == method) & (validation.rule_id == rule["rule_id"])]
        if len(row) != 1 or row.threshold.iloc[0] != rule["threshold"]:
            raise ValueError("冻结门槛与本方法验证记录不一致")
    return len(joined)


def audit_frozen_training(directory: Path) -> Path:
    """读取directory冻结记录与拟合审计，只验证工程契约，输出新目录；发现不一致即抛错，不改旧文件。"""
    frozen = json.loads((directory / "frozen_before_test.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    for filename, key in [("resolved_config.yaml", "config_sha256"), ("research_protocol.json", "protocol_sha256")]:
        if calculate_sha256(directory / filename) != frozen[key]:
            raise ValueError(f"冻结配置或协议已改变：{filename}")
    for relative, expected in frozen["source_hashes"].items():
        if calculate_sha256(ROOT / relative) != expected or calculate_sha256(directory / "source_snapshot" / relative) != expected:
            raise ValueError(f"当前源码或快照不符：{relative}")
    years = set(map(int, frozen["folds"]))
    if years != set(cfg["test_years"]):
        raise ValueError("冻结记录缺少年度")
    metadata = json.loads((directory / "data_metadata.json").read_text(encoding="utf-8"))
    # 单股缓存的market已经按全市场共同日历重建索引，无须加载全量行情或重建因子。
    symbol_path = next(iter(sorted(Path(metadata["annual_cache"]).glob("S[HZ].*.pkl"))))
    times = pd.DatetimeIndex(pd.read_pickle(symbol_path)["market"].datetime)
    days = pd.DatetimeIndex(sorted(times.normalize().unique()))
    if len(days) != metadata["days"]:
        raise ValueError("共同交易日数与训练来源不符")
    audit = pd.read_csv(directory / "split_audit.csv")
    model_rows, boundary_rows = [], []
    threshold_count = 0
    for year, item in frozen["folds"].items():
        folder, fold = directory / year, item["fold"]
        for filename, expected in item["hashes"].items():
            if calculate_sha256(folder / filename) != expected:
                raise ValueError(f"冻结产物已改变：{year}/{filename}")
        rules = json.loads((folder / "selected_rules.json").read_text(encoding="utf-8"))
        if rules != item["rules"]:
            raise ValueError("冻结索引与规则文件不同")
        training = pd.read_csv(folder / "training_rule_metrics.csv", float_precision="round_trip")
        validation = pd.read_csv(folder / "validation_rule_metrics.csv", float_precision="round_trip")
        threshold_count += validate_threshold_contract(training, validation, rules)
        features = json.loads((folder / "entry_feature_schema.json").read_text(encoding="utf-8"))
        if len(features) != len(set(features)) or len(features) > cfg["max_features"]:
            raise ValueError("开仓特征重复或超过上限")
        if not all(c.startswith(("cs_rank_", "context_")) for c in features):
            raise ValueError("开仓特征出现契约外字段")
        exit_schema = json.loads((folder / "exit_feature_schema.json").read_text(encoding="utf-8"))
        if list(exit_schema) != EXIT_FEATURES:
            raise ValueError("持仓模型列序错误")
        for filename, columns in [(f"entry_{p['name']}.json", features) for p in cfg["model_profiles"]] + [("exit_model.json", EXIT_FEATURES)]:
            model = XGBRegressor()
            model.load_model(folder / filename)
            model.set_params(n_jobs=1)
            if model.get_booster().feature_names != columns:
                raise ValueError(f"模型与特征列序不符：{year}/{filename}")
            prediction = model.predict(pd.DataFrame(np.zeros((1, len(columns)), dtype="float32"), columns=columns))
            if not np.isfinite(prediction).all():
                raise ValueError("人工输入预测非有限值")
            model_rows.append({"留出年": int(year), "模型文件": filename, "输入列数": len(columns), "重载与列序检查": "通过"})
        fitted = audit.loc[audit.test_year == int(year)]
        if fitted.empty or (fitted.fit_year == int(year)).any():
            raise ValueError("拟合年份记录缺失或包含留出年")
        excluded = days[days >= pd.Timestamp(f"{int(year) + 1}-01-01")][:cfg["feature_embargo_trading_days"]]
        for row in fitted.itertuples():
            window = next(w for w in fold["fit"] if w["year"] == row.fit_year)
            first, last, label = map(pd.Timestamp, [row.first_signal, row.last_signal, row.last_label])
            if not pd.Timestamp(window["start"]) <= first <= last < label < pd.Timestamp(window["end"]):
                raise ValueError("拟合摘要的信号或标签越过本年内部窗口")
            if len(excluded) and first.normalize() <= excluded[-1] and last.normalize() >= excluded[0]:
                raise ValueError("拟合摘要落入留出年后隔离区间")
        future_fit = bool((fitted.fit_year > int(year)).any())
        if future_fit != fold["future_trained"]:
            raise ValueError("实际拟合年与未来拟合标记不一致")
        for role in ["fit", "calibration", "validation"]:
            if any(w["year"] == int(year) for w in fold[role]):
                raise ValueError("内部窗口包含外层年份")
        boundary_rows.append({"留出年": int(year), "实际核心拟合年份": ",".join(map(str, sorted(fitted.fit_year.unique()))),
                              "未来核心拟合": future_fit, "未来内部选参安排": fold["future_selection"],
                              "拟合记录行数": int(fitted.rows.sum())})
    out = directory / "training_integrity_audit" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(model_rows).to_csv(out / "model_reload_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boundary_rows).to_csv(out / "training_boundary_audit.csv", index=False, encoding="utf-8-sig")
    write_json(out / "audit_metadata.json", {"source_run": str(directory), "audit_script_sha256": calculate_sha256(Path(__file__)),
        "frozen_sha256": calculate_sha256(directory / "frozen_before_test.json"), "model_count": len(model_rows),
        "threshold_candidate_count": threshold_count, "outer_predictions_or_returns_read": False,
        "limitations": "按保存摘要检查边界，不替代原始特征逐行依赖审计；未证明收益目标或无统计偏差"})
    (out / "final_report.md").write_text("# 冻结训练完整性审计\n\n"
        f"{len(model_rows)}个模型重载/列序及{threshold_count}个入围候选的跨阶段门槛检查通过；配置、协议、源码与模型哈希一致。没有读取外层测试预测或收益。\n\n"
        + pd.DataFrame(boundary_rows).to_markdown(index=False)
        + "\n\n拟合记录行数为各模型行数相加，同一训练样本会重复计入，不是去重样本量。未来内部选参列表示窗口安排；不代表每个窗口都有合格交易。\n\n"
        "这只证明所检查的文件、门槛和日期摘要一致；不等于所有原始特征依赖完全隔离，不证明策略盈利。仍须披露递推指标、当前名称筛选、复权和成交代理等限制。\n", encoding="utf-8")
    (out / "输出文件说明.md").write_text("# 文件说明\n\n- final_report.md：审计结果及范围限制。\n"
        "- model_reload_audit.csv：逐模型列数、重载及列序检查。\n- training_boundary_audit.csv：实际拟合年份、未来信息标记及行数摘要。\n"
        "- audit_metadata.json：来源冻结哈希、审计代码哈希、检查数量及未读取外层结果声明。\n", encoding="utf-8")
    print(f"冻结训练完整性审计通过：{out}", flush=True)
    return out


def run_cli() -> None:
    """解析已冻结实验的run-dir路径，运行工程审计并写入新子目录。"""
    parser = argparse.ArgumentParser(description="多尺度模型冻结契约审计，不评价外层收益")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_frozen_training(args.run_dir.resolve())


if __name__ == "__main__":
    run_cli()
