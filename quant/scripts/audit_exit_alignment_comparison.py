"""在quant根目录复查--run-dir期限对照的模型复用、原校准复现及外层晋级条件。

输入本轮和来源实验的冻结产物、内部校准/验证表；生成新的中文审计子目录和逐项CSV。
不拟合、不计算预测、不读取test_*文件内容、不修改原表或缓存，不连接券商。
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from scripts.audit_multiscale_frozen_training import audit_frozen_training
from scripts.run_exit_alignment_year_validation import validate_parent
from scripts.run_learned_exit_year_validation import calculate_sha256

CONTROL_NAMES = {
    "model_control_zero": "continuation0_atr20_40",
    "model_control_negative": "continuation_negative_atr20_40",
}
CONTROL_FIELDS = {
    "threshold": "开仓门槛", "mean_sharpe": "校准平均夏普",
    "worst_drawdown": "最差回撤", "closed_lots": "完整平仓批次",
    "positive_segments": "正收益片段数", "active_segments": "有波动片段数",
    "initial": "初始ATR倍数", "trailing": "移动ATR倍数",
    "max_days": "最长交易日", "exit_threshold": "退出预测门槛",
}


def compare_calibration_controls(parent: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """按方法/模型/分位将current原模型对照与parent旧校准一一配对；缺失、重复或任何数值差异即拒绝。

    两个输入须来自内部校准CSV并用float_precision='round_trip'读取，不能传入测试收益。
    返回带中文差值列的逐候选表；所有待比字段必须有限，不接受NaN等价掩盖缺失。
    """
    keys = ["method", "profile", "quantile", "name"]
    old = parent.loc[parent.name.isin(CONTROL_NAMES.values())].copy()
    new = current.loc[current.name.isin(CONTROL_NAMES)].copy()
    new["name"] = new.name.map(CONTROL_NAMES)
    if old.empty or new.empty or old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("原模型校准对照为空或键重复")
    joined = new.merge(old, on=keys, how="outer", validate="one_to_one",
                       suffixes=("_current", "_parent"), indicator=True)
    if not (joined._merge == "both").all():
        raise ValueError("原模型校准对照缺少配对")
    result = joined[keys + ["rule_id_current", "rule_id_parent"]].copy()
    result.columns = ["方法代码", "开仓模型代码", "门槛分位", "来源退出代码", "本轮规则代码", "来源规则代码"]
    for field, title in CONTROL_FIELDS.items():
        left, right = joined[f"{field}_current"].to_numpy(), joined[f"{field}_parent"].to_numpy()
        if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.array_equal(left, right):
            raise ValueError(f"原模型对照未精确复现：{field}（{title}）")
        result[f"{title}差值"] = left - right
    return result


def calculate_promotion_status(folds: dict, years: list[int], methods: list[str]) -> pd.DataFrame:
    """按folds已冻结规则计算years/methods的完整启用状态；现金年不能删除，缺年或缺方法即拒绝。

    返回逐方法现金年份及是否满足全部年份内部启用，仅是人工流程的必要条件，不证明测试或盈利。
    """
    if len(years) != len(set(years)) or set(folds) != set(map(str, years)):
        raise ValueError("晋级核对缺年或年份重复")
    rows = []
    for method in methods:
        cash = []
        for year in years:
            rules = folds[str(year)]["rules"]
            if set(rules) != set(methods):
                raise ValueError("晋级核对缺少方法或含未知方法")
            if rules[method]["rule_id"] == "cash_validation_failed":
                cash.append(year)
        rows.append({"方法代码": method, "需核对折数": len(years), "内部启用折数": len(years) - len(cash),
                     "现金保护年份": ",".join(map(str, cash)), "全部折内部启用": not cash})
    return pd.DataFrame(rows)


def audit_alignment_comparison(directory: Path) -> Path:
    """对directory先做冻结完整性审计，再核对来源副本/原校准；只写新审计目录，不决定或执行外层评价。"""
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    parent = Path(cfg["source_run"])
    _, original = validate_parent(parent)
    integrity = audit_frozen_training(directory)
    frozen = json.loads((directory / "frozen_before_test.json").read_text(encoding="utf-8"))
    if calculate_sha256(directory / "parent_frozen_before_test.json") != calculate_sha256(parent / "frozen_before_test.json"):
        raise ValueError("来源冻结索引副本不同")
    if set(original["folds"]) != set(frozen["folds"]):
        raise ValueError("来源与本轮折数不同")
    artifact_rows, tables, sources, summaries = [], [], {}, []
    for year, item in frozen["folds"].items():
        if item["fold"] != original["folds"][year]["fold"]:
            raise ValueError("本轮改变了内部/外层窗口")
        names = [f"entry_{p['name']}.json" for p in cfg["model_profiles"]]
        names += ["exit_model.json", "entry_feature_schema.json", "exit_feature_schema.json", "model_feature_manifest.csv"]
        for name in names:
            a, b = directory / year / name, parent / year / name
            digest = calculate_sha256(a)
            if digest != calculate_sha256(b):
                raise ValueError(f"模型或列序副本不同：{year}/{name}")
            artifact_rows.append({"留出年": int(year), "文件": name, "SHA256（内容指纹）": digest, "与来源逐字节一致": True})
        paths = [parent / year / "training_rule_metrics.csv", directory / year / "training_rule_metrics.csv"]
        old, new = [pd.read_csv(p, float_precision="round_trip") for p in paths]
        sources.update({str(p): calculate_sha256(p) for p in paths})
        comparison = compare_calibration_controls(old, new)
        expected = len(cfg["model_profiles"]) * len(cfg["gate_quantiles"]) * len(item["rules"]) * len(CONTROL_NAMES)
        if len(comparison) != expected:
            raise ValueError("原模型对照数量与完整网格不一致")
        comparison.insert(0, "留出年", int(year))
        tables.append(comparison)
        validation = pd.read_csv(directory / year / "validation_rule_metrics.csv", float_precision="round_trip")
        for method, rule in item["rules"].items():
            rows = validation.loc[validation.method == method]
            selected = rows.loc[rows.rule_id == rule["rule_id"]]
            summaries.append({"留出年": int(year), "方法代码": method, "冻结规则代码": rule["rule_id"],
                              "内部选中夏普": float(selected.mean_sharpe.iloc[0]) if len(selected) else np.nan,
                              "入围最高内部夏普": float(rows.mean_sharpe.max()) if len(rows) else np.nan,
                              "内部启用": rule["rule_id"] != "cash_validation_failed"})
    for name in ["data_metadata.json", "split_audit.csv"]:
        if calculate_sha256(directory / name) != calculate_sha256(parent / name):
            raise ValueError(f"来源数据或拟合摘要不同：{name}")
    status = calculate_promotion_status(frozen["folds"], cfg["test_years"], ["method1", "method2"])
    # 只检查本轮外层产物是否存在，不打开收益、预测或交易文件。
    has_outer_artifacts = (directory / "test_started.json").exists() or bool(list(directory.glob("*/method*/test_metrics.csv")))
    out = directory / "alignment_comparison_audit" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    controls = pd.concat(tables, ignore_index=True)
    for name, table in [("原模型校准复现.csv", controls), ("模型与列序副本.csv", pd.DataFrame(artifact_rows)),
                        ("外层晋级状态.csv", status), ("内部选参摘要.csv", pd.DataFrame(summaries))]:
        table.to_csv(out / name, index=False, encoding="utf-8-sig")
    model_count = len(cfg["test_years"]) * (len(cfg["model_profiles"]) + 1)
    write_json(out / "audit_metadata.json", {"source_run": str(directory), "parent_run": str(parent),
        "audit_script_sha256": calculate_sha256(Path(__file__)), "source_table_hashes": sources,
        "frozen_sha256": calculate_sha256(directory / "frozen_before_test.json"), "integrity_audit": str(integrity),
        "model_count": model_count, "control_count": len(controls), "outer_results_read": False,
        "outer_artifacts_present": has_outer_artifacts, "necessary_promotion_condition": bool(status["全部折内部启用"].any()),
        "is_smoke": cfg["is_smoke"], "profitability_assessed": False})
    display_status = status.copy()
    display_status["全部折内部启用"] = display_status["全部折内部启用"].map({True: "是", False: "否"})
    display_rules = pd.DataFrame(summaries)
    display_rules["内部启用"] = display_rules["内部启用"].map({True: "是", False: "否"})
    for column in ["内部选中夏普", "入围最高内部夏普"]:
        display_rules[column] = display_rules[column].map(lambda value: f"{value:.4f}" if np.isfinite(value) else "不适用")
    text = ["# 退出配对复现审计", "", f"{model_count}个模型与来源逐字节相同，列序、因子清单、缓存来源和原拟合摘要也一致。",
            f"{len(controls)}组原模型校准对照的开仓门槛、退出参数、平均夏普、回撤和交易计数全部精确复现。",
            "只比较同方法、同开仓模型、同分位、同ATR20/40的两个原模型退出对照，没有读取外层收益。", "",
            "## 外层必要条件", "", display_status.to_markdown(index=False), "",
            f"至少一种方法全部折内部启用：{'是' if status['全部折内部启用'].any() else '否'}。",
            f"本轮外层产物存在：{'是（仅检查文件存在，不读取内容）' if has_outer_artifacts else '否'}。",
            "晋级状态只是事前人工流程必要条件；不生成外层预测，不自动运行外层，也不宣称夏普目标完成。", "",
            "## 内部规则", "", display_rules.to_markdown(index=False), "",
            "method1为净收益模型基础方法，method2为相同模型加微量趋势叠加；cash_validation_failed表示内部验证失败、保持现金。",
            "代码含义：horizon3/5为3/5交易日预测，shallow/deeper为浅/深树，minimum_horizon为到期后才启用持仓模型，",
            "matched_horizon为到期同刻退出，model_control为原模型退出，negative/zero为负0.10%/零门槛，q为开仓校准分位。",
            "", "## 审计边界", "", "校准对照相同只能支持实现复现，不能证明期限策略有优势或样本外有效。",
            "独立完整性审计目录见audit_metadata.json；原始递推特征状态、非历史ST、公司行动和撮合代理限制仍存在。",
            "留一年协议的前三折可含未来年份拟合或选参；历史数据反复使用，不能称为未见过的前瞻测试。"]
    (out / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "输出文件说明.md").write_text(
        "# 输出文件说明\n\n- final_report.md：中文复现结论、内部晋级条件与限制。\n"
        "- 原模型校准复现.csv：逐对照规则的10个数值差值，全部应为0；不是外层测试。\n"
        "- 模型与列序副本.csv：逐模型、输入列序、因子清单的来源一致性与内容指纹。\n"
        "- 外层晋级状态.csv：两种方法是否完整启用四折，现金年不能跳过。\n"
        "- 内部选参摘要.csv：每折最终规则、选中及入围最高内部夏普，现金规则的选中值留空。\n"
        "- audit_metadata.json：本审计及原校准表指纹、独立完整性审计路径、检查数量；不含测试收益。\n",
        encoding="utf-8")
    print(f"退出配对复现审计通过：{out}", flush=True)
    return out


def run_cli() -> None:
    """解析已冻结实验的--run-dir，执行无外层收益输入的独立审计。"""
    parser = argparse.ArgumentParser(description="核对退出配对模型复用、原校准复现和晋级状态")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_alignment_comparison(args.run_dir.resolve())


if __name__ == "__main__":
    run_cli()
