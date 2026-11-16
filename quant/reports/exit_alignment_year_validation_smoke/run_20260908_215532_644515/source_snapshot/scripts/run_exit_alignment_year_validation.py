"""在quant根目录复用冻结模型，研究预测期限与独立批次退出的配对差异。

输入YAML及来源实验的训练期产物；train只重新校准/验证退出，生成新协议/规则/中文报告，
不重训模型、不读取来源外层预测或收益、不改缓存或券商。evaluate仅消费本轮冻结规则一次。
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.exit_alignment import create_alignment_policy
from scripts.run_multiscale_year_validation import load_multiscale_data, select_outer_feature_rows
from scripts.run_learned_exit_year_validation import calculate_sha256, select_rules, run_evaluation
from scripts.audit_multiscale_frozen_training import validate_threshold_contract


def validate_runner_compatibility(old: str, current: str) -> bool:
    """比较old/current组织器AST，只允许退出工厂接入及报告函数变化；拟合/打分/数据函数必须一致。"""
    allowed = {"create_learned_policy", "select_rules", "run_evaluation", "write_root_report"}
    def filter_tree(source: str) -> str:
        """从source移除允许扩展的顶层函数再比较结构，忽略空白，不执行来源代码。"""
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name not in allowed]
        return ast.dump(tree, include_attributes=False)
    return filter_tree(old) == filter_tree(current)


def validate_parent(directory: Path) -> tuple[dict, dict]:
    """核对directory的冻结配置、模型与源码快照；不打开任何test_*文件，不要求旧选参器与新扩展字节相同。"""
    frozen = json.loads((directory / "frozen_before_test.json").read_text(encoding="utf-8"))
    for name, key in [("resolved_config.yaml", "config_sha256"), ("research_protocol.json", "protocol_sha256")]:
        if calculate_sha256(directory / name) != frozen[key]:
            raise ValueError(f"来源配置或协议已修改：{name}")
    for relative, digest in frozen["source_hashes"].items():
        saved, current = directory / "source_snapshot" / relative, ROOT / relative
        if calculate_sha256(saved) != digest:
            raise ValueError(f"来源源码快照不匹配：{relative}")
        if calculate_sha256(current) == digest:
            continue
        if relative != "scripts/run_learned_exit_year_validation.py" or not validate_runner_compatibility(saved.read_text(encoding="utf-8"), current.read_text(encoding="utf-8")):
            raise ValueError(f"影响模型来源或数据的代码已变化，拒绝复用：{relative}")
    for year, item in frozen["folds"].items():
        for name, digest in item["hashes"].items():
            if calculate_sha256(directory / year / name) != digest:
                raise ValueError(f"来源冻结产物已变化：{year}/{name}")
    return yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8")), frozen


def load_config(args: argparse.Namespace) -> tuple[dict, dict]:
    """args指定研究YAML/烟测及工作线程；只允许覆盖退出候选和输出，模型/成本/门槛网格来自冻结父实验。"""
    extra = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if set(extra) != {"source_run", "source_run_smoke", "output_dir", "exit_candidates"}:
        raise ValueError("配置只能指定模型来源、输出和退出候选，不能暗改数据/费用/开仓网格")
    source = (ROOT / extra["source_run_smoke" if args.smoke else "source_run"]).resolve()
    cfg, parent = validate_parent(source)
    if bool(cfg["is_smoke"]) != args.smoke:
        raise ValueError("烟测与正式来源不能混用")
    cfg.update(source_run=str(source), output_dir=str((ROOT / extra["output_dir"]).resolve()) + ("_smoke" if args.smoke else ""),
               exit_candidates=extra["exit_candidates"])
    if args.workers:
        cfg["workers"] = args.workers
    for rule in cfg["exit_candidates"]:
        if rule["initial"] != 20 or rule["trailing"] != 40:
            raise ValueError("配对研究必须保持相同ATR20/40保护")
        expected = {"model_control": 20, "minimum_horizon_model": 20, "matched_horizon": 0, "next_day_control": 1}
        if rule["alignment_mode"] not in expected or rule["max_days"] != expected[rule["alignment_mode"]]:
            raise ValueError("退出期限保护不符合预设配对协议")
    cfg["report_note"] = "退出期限配对回顾研究；复用冻结模型，前三折含未来信息；非实盘业绩。"
    cfg["exit_strategy_description"] = "退出候选比较原持仓模型、预测期限同刻退出、到期后启用持仓模型和最早T+1对照；全部保留ATR20/40保护。"
    cfg["report_limitations"] = "- 本轮没有重训或选用父实验外层胜者：全部开仓档位与持仓模型原样复用，只在内部校准/验证选择退出。\n" + cfg["report_limitations"]
    cfg["report_limitations"] += "\n- 期限按实际交易日及各批次买入信号时刻计算；ATR可提前退出，缺失行情、T+1及容量可延迟成交。固定预测期限不保证固定实际持仓时长。\n"
    cfg["report_limitations"] += "- 各对照的拟议开仓分数和门槛相同，但退出改变现金和存活持仓，因此实际买单可以不同；这是组合政策比较，不是强行固定每笔成交的反事实。\n"
    return cfg, parent


def write_paired_comparison(folder: Path) -> None:
    """读取folder校准表，按方法/模型/分位与model_control_zero配对；不读取任何测试结果。"""
    table = pd.read_csv(folder / "training_rule_metrics.csv")
    keys = ["method", "profile", "quantile"]
    controls = table.loc[table.name == "model_control_zero", keys + ["mean_sharpe", "worst_drawdown", "closed_lots"]]
    paired = table.merge(controls, on=keys, how="left", validate="many_to_one", suffixes=("", "_control"))
    if paired.mean_sharpe_control.isna().any():
        raise ValueError("配对候选缺少相同入口的基准")
    paired["mean_sharpe_difference"] = paired.mean_sharpe - paired.mean_sharpe_control
    paired.to_csv(folder / "paired_calibration_comparison.csv", index=False, encoding="utf-8-sig")


def run_training(cfg: dict, parent: dict) -> Path:
    """按cfg与parent协议复用全部模型，对四折内部选参后冻结；不生成外层预测，不覆盖父目录。"""
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(cfg["source_run"])
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    sources = list(parent["source_hashes"]) + ["quant_backtest/exit_alignment.py", "scripts/run_exit_alignment_year_validation.py", "scripts/audit_multiscale_frozen_training.py"]
    hashes = {}
    for relative in dict.fromkeys(sources):
        dest = directory / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        hashes[relative] = calculate_sha256(dest)
    shutil.copyfile(source / "frozen_before_test.json", directory / "parent_frozen_before_test.json")
    write_json(directory / "research_protocol.json", {"created": datetime.now().isoformat(), "source_run": str(source),
        "parent_frozen_sha256": calculate_sha256(directory / "parent_frozen_before_test.json"), "source_hashes": hashes,
        "model_fitting_performed": False, "parent_test_results_read": False, "test_predictions_generated": False,
        "criterion": "两方法各自完整四折平均夏普>3，现金年不能删除；内部规则仍先校准后验证",
        "hypothesis": "检验事先提出的3/5日开仓目标与一日持仓判断是否不匹配；不根据父实验外层成绩选模型或参数",
        "reference": "https://www.nber.org/papers/w15205",
        "parent_compatibility": "原快照/模型校验，数据/特征源码不变；组织器仅允许退出工厂、选参/评价和报告函数扩展"})
    print(f"退出期限内部研究目录：{directory}", flush=True)
    panel, market, matrices, samples, manifest, cache = load_multiscale_data(cfg)
    del samples
    original_data = json.loads((source / "data_metadata.json").read_text(encoding="utf-8"))
    if original_data["exit_cache"] != cache or original_data["annual_cache"] != manifest["cache_path"]:
        raise ValueError("缓存来源与冻结父模型不符")
    shutil.copyfile(source / "data_metadata.json", directory / "data_metadata.json")
    shutil.copyfile(source / "split_audit.csv", directory / "split_audit.csv")
    frozen = {}
    for year, item in parent["folds"].items():
        folder = directory / year
        folder.mkdir()
        fold = json.loads(json.dumps(item["fold"]))
        for role in ["fit", "calibration", "validation"]:
            for window in fold[role]:
                window["start"], window["end"] = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        models = {}
        for name in ["entry_feature_schema.json", "exit_feature_schema.json", "model_feature_manifest.csv"]:
            shutil.copyfile(source / year / name, folder / name)
        for profile in cfg["model_profiles"]:
            name = f"entry_{profile['name']}.json"
            shutil.copyfile(source / year / name, folder / name)
            model = XGBRegressor()
            model.load_model(folder / name)
            model.set_params(n_jobs=cfg["workers"])
            models[profile["name"]] = model
        shutil.copyfile(source / year / "exit_model.json", folder / "exit_model.json")
        exit_model = XGBRegressor()
        exit_model.load_model(folder / "exit_model.json")
        exit_model.set_params(n_jobs=cfg["workers"])
        features = json.loads((folder / "entry_feature_schema.json").read_text(encoding="utf-8"))
        outer = select_outer_feature_rows(panel, market.times, int(year), cfg["feature_embargo_trading_days"])
        outer = outer.loc[outer.cross_section_count >= cfg["min_cross_section"]]
        rules = select_rules(models, exit_model, features, outer, fold, market, matrices, cfg, folder, create_alignment_policy)
        validate_threshold_contract(pd.read_csv(folder / "training_rule_metrics.csv", float_precision="round_trip"),
                                    pd.read_csv(folder / "validation_rule_metrics.csv", float_precision="round_trip"), rules)
        write_paired_comparison(folder)
        write_json(folder / "selected_rules.json", rules)
        metadata = json.loads((source / year / "model_metadata.json").read_text(encoding="utf-8"))
        metadata.update(rules=rules, model_fitting_performed=False, parent_model_metadata_sha256=calculate_sha256(source / year / "model_metadata.json"),
                        exit_timing_definition=cfg["exit_strategy_description"], parent_test_results_used=False)
        write_json(folder / "model_metadata.json", metadata)
        frozen[year] = {"fold": fold, "rules": rules, "hashes": {p.name: calculate_sha256(p) for p in folder.iterdir() if p.is_file()}}
        print(f"{year}折完成内部选择：{json.dumps(rules, ensure_ascii=False)}", flush=True)
    write_json(directory / "folds.json", [v["fold"] for v in frozen.values()])
    write_json(directory / "frozen_before_test.json", {"created": datetime.now().isoformat(), "folds": frozen,
        "protocol_sha256": calculate_sha256(directory / "research_protocol.json"), "config_sha256": calculate_sha256(directory / "resolved_config.yaml"),
        "source_hashes": hashes})
    lines = ["# 退出期限训练内部研究", "", "已冻结四折。没有计算本轮外层预测或收益，也没有重训模型。", "", cfg["exit_strategy_description"], ""]
    for year in frozen:
        table = pd.read_csv(directory / year / "validation_rule_metrics.csv")
        display = table[["method", "rule_id", "mean_sharpe", "closed_lots", "positive_segments"]].copy()
        display.columns = ["方法代码（method1基础、method2趋势叠加）", "规则代码", "内部平均夏普", "完整平仓批次", "正收益片段"]
        lines += [f"## {year}留出折的内部验证", "", display.to_markdown(index=False), ""]
        lines.extend(f"- {method}：{r['selection']}，{r['rule_id']}。" for method, r in frozen[year]["rules"].items())
    lines += ["", "## 代码说明", "", "model_control：原模型退出；matched_horizon：预测期限同刻退出；minimum_horizon：到期后启用模型；"
              "next_day_control：最早T+1对照；zero/negative：继续收益门槛为0/-0.10%；horizon3/5：预测3/5交易日；shallow/deeper：浅树/深树；q：校准分位。",
              "", "## 限制", "", cfg["report_limitations"], "", "[信号期限与交易成本的研究依据](https://www.nber.org/papers/w15205)并不证明本A股退出规则有效。"]
    (directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.copyfile(directory / "final_report.md", directory / "training_report.md")
    (directory / "输出文件说明.md").write_text("# 训练阶段文件说明\n\n"
        "- final_report.md、training_report.md：内部验证报告及归档，没有外层收益图。\n"
        "- resolved_config.yaml、research_protocol.json：退出候选、来源模型、原成本和开仓网格、研究依据与哈希。\n"
        "- source_snapshot/：本轮执行源码；parent_frozen_before_test.json：原模型冻结索引，不是原外层结果。\n"
        "- data_metadata.json、split_audit.csv、folds.json：复用缓存、原拟合边界和本轮内部/外层年份。\n"
        "- frozen_before_test.json：本轮四折冻结规则及全部模型/表格哈希，evaluate消费。\n"
        "- 年份/entry_*.json、exit_model.json：原样复制的模型；entry_feature_schema.json、exit_feature_schema.json：输入列序。\n"
        "- 年份/model_feature_manifest.csv、model_metadata.json：因子中文解释、原拟合来源和本轮退出策略。\n"
        "- 年份/training_rule_metrics.csv、validation_rule_metrics.csv、selection_segment_metrics.csv：校准、验证与每片段指标。\n"
        "- 年份/paired_calibration_comparison.csv：同入口网格下相对原模型退出的校准差异，不是测试结果。\n"
        "- 年份/selected_rules.json：各方法独立冻结规则或现金保护原因。\n", encoding="utf-8")
    print(f"四折退出期限研究已冻结，尚未测试：{directory}", flush=True)
    return directory


def run_cli() -> None:
    """解析YAML、阶段、烟测和线程；evaluate必须指定本轮冻结目录，只用本模块退出工厂。"""
    parser = argparse.ArgumentParser(description="复用冻结模型，研究预测期限与独立退出")
    parser.add_argument("--config", type=Path, default=ROOT / "config/exit_alignment_year_validation.yaml")
    parser.add_argument("--phase", choices=["train", "evaluate"], default="train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.phase == "evaluate":
        if args.run_dir is None:
            parser.error("evaluate需要--run-dir")
        run_evaluation(args.run_dir.resolve(), load_multiscale_data, create_alignment_policy)
    else:
        cfg, parent = load_config(args)
        run_training(cfg, parent)


if __name__ == "__main__":
    run_cli()

