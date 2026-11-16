"""在quant根目录研究执行路径净收益模型，输入YAML、既有有界pkl与只读分钟执行行情。

build-only生成并复用独立交易标签缓存，按核心训练样本与原引擎比对。
train拟合回归/概率模型并只在内部选择；audit验证真实训练行和冻结文件；
evaluate晋级后仅一次生成年度回顾收益、逐笔记录和中文说明，不连接券商。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.annual_portfolio import run_lot_portfolio
from quant_backtest.bounded_training import load_bounded_data, select_bounded_rows
from quant_backtest.learned_exit_models import build_research_folds
from quant_backtest.simple_signals import create_exit_policy
from quant_backtest.pathwise_labels import (PATH_FEATURES, build_path_panel, load_path_labels,
    create_path_arrays, validate_path_case)
from quant_backtest.pathwise_models import (METHOD_NAMES, create_path_profiles, select_path_rows,
    fit_path_model, generate_path_scores, create_path_exits, calculate_path_diagnostics)
from scripts.run_bounded_year_validation import load_config as load_bounded_config
from scripts.run_simple_signal_year_validation import (SOURCES, EXTRA_SOURCES, select_rules, write_table, validate_frozen)
from scripts.run_learned_exit_year_validation import calculate_sha256
from scripts.run_annual_method_validation import save_result, validate_result, LIMITATIONS
from scripts.audit_bounded_training import validate_core_sources
from scripts.audit_multiscale_frozen_training import validate_threshold_contract

PATH_SOURCES = ["quant_backtest/pathwise_labels.py", "quant_backtest/pathwise_models.py",
    "scripts/run_pathwise_year_validation.py", "tests/test_pathwise_labels.py", "tests/test_pathwise_models.py",
    "tests/test_annual_validation.py"]
NOTE = "执行路径标签留一年回顾；前三折包含未来年份信息，2026仅1至4月，不是实盘证明。"


def load_config(args: argparse.Namespace) -> dict:
    """合并args指定YAML，解析新缓存路径及烟测参数，成本与资金限制仍由原有界配置验证。"""
    cfg = load_bounded_config(args)
    cfg["path_cache_dir"] = str((ROOT / cfg["path_cache_dir"]).resolve()) + ("_smoke" if args.smoke else "")
    if args.smoke:
        cfg.update(path_horizons=[1, 3], path_audit_cases=12, simple_min_training_rows=10,
                   simple_min_positive_segments=1, simple_shortlist=1,
                   path_model_profiles=[{"name": "smoke", "num_leaves": 4, "max_depth": 2,
                                         "min_child_samples": 10, "n_estimators": 8, "reg_lambda": 10.}])
    if any(h < 1 or int(h) != h for h in cfg["path_horizons"]) or cfg["path_exit_grace_days"] < 1:
        raise ValueError("路径期限和宽限期必须为正交易日数")
    if cfg["path_initial_atr"] <= 0 or cfg["path_trailing_atr"] <= 0:
        raise ValueError("ATR倍数必须为正")
    if cfg["path_training_phase"] not in {"per_symbol", "synchronized"}:
        raise ValueError("抽样相位必须是per_symbol或synchronized")
    cfg["simple_exits"] = create_path_exits({"horizon": cfg["path_horizons"][0]}, cfg)
    cfg["report_note"] = NOTE
    cfg["report_limitations"] = "\n".join(line for line in LIMITATIONS.splitlines()
        if not any(text in line for text in ["方法二沿用方法一", "最终全量模型"])) + "\n- " + NOTE + """
- 原固定期限价格标签与实际ATR/部分卖出不同。本轮按独立初始账户模拟一次10%预算入场，并在ATR或H交易日到期后尝试退出，观察至H+5交易日同刻。没有买入的意图收益为0，尚未退出者按窗口末市值记账，不伪造强平。
- 独立标签不共享不同意图之间的资金、卖出容量和后续加仓，不能当真实组合收益。实际组合仍由共同账户引擎模拟，可能比独立意图更受资金/容量约束。
- 核心样本一股一天预定一个信号钟，再每H+宽限+1交易日抽样；固定潜在结束窗口必须完整落在核心片段，提前盈利退出不放宽边界。标签不是入场筛选，推断保留没有未来标签的最新行。
- 正式默认按股票代码的固定相位错开训练日期，避免全体同步稀疏抽样丢失短年份全部合格入口。相位不读取收益，同股潜在标签仍不得重叠；synchronized仅保留历史方案兼容。
- 方法一预测截断后的独立账户净收益；方法二预测意图产生正收益的概率，并用核心正/非正类的平均幅度转为期望账户收益。概率不是已成交批次胜率，条件幅度简化也不能代替精确收益分布。
- 固定29个当时特征，不进行外层收益筛选。LightGBM原生处理缺失，不从验证/测试拟合填补器、标准化或收益幅度。训练/内部校准/验证分段、7自然日隔离、留出年后60交易日隔离及实际来源审计仍保留。
- 2026是否实际参与核心拟合以split_audit.csv为准；2025留出折2026核心可能为空但后段用于内部选参。年份轮换不是四个独立实验，历史也已被反复研究。
- 修复了成交量NaN、无效前收或非正/无限执行价格和成交额可绕过比较的保护。旧实验保留源码快照，不能修改旧指纹绕过冻结审计；原始特征缓存未重建。
- 至少一种方法四折内部启用才允许正式外层。每方法完整四年平均夏普严格大于3才达标，现金无波动年不许删除，内部夏普不能当年度外层成绩。
"""
    return cfg


def load_data(cfg: dict) -> tuple:
    """从cfg加载有界数据和路径标签，返回固定特征面板、行情、缓存身份；未来标签仅附加给训练选择器。"""
    panel, market, matrices, samples, manifest, bounded = load_bounded_data(cfg)
    panel = build_path_panel(panel, market, matrices)
    del matrices, samples
    labels, label_meta = load_path_labels(panel, market, bounded, cfg)
    panel = panel.merge(labels, on=["datetime", "symbol"], how="left", validate="one_to_one")
    metadata = {"annual_cache": manifest["cache_path"], "bounded_cache": bounded, "path_cache": label_meta,
                "panel_rows": len(panel), "symbols": len(market.symbols), "days": len(set(market.days))}
    return panel, market, metadata


def audit_reference_paths(panel: pd.DataFrame, market, cfg: dict) -> list[dict]:
    """只抽留出2026折的实际核心训练行，与原引擎比较cfg.path_audit_cases个独立路径，不读取测试成绩。"""
    fold = next(f for f in build_research_folds(market.times, cfg) if f["year"] == 2026)
    outer = select_bounded_rows(panel, market.times, 2026, cfg["feature_embargo_trading_days"])
    outer = outer.loc[outer.fit_source_allowed & (outer.cross_section_count >= cfg["min_cross_section"])]
    arrays = create_path_arrays(market, cfg)
    reports = []
    count = max(1, cfg["path_audit_cases"] // len(cfg["path_horizons"]))
    for horizon in cfg["path_horizons"]:
        train = select_path_rows(outer, fold["fit"], horizon, cfg, market.times, True)
        selected = train.sample(min(count, len(train)), random_state=cfg["model"]["random_state"])
        validate_core_sources(selected, 2026)
        for row in selected.itertuples():
            start = market.time_index[row.datetime]
            end = market.time_index[row.label_end - pd.Timedelta(minutes=1)]
            check = validate_path_case(market, cfg, arrays, market.asset_index[row.symbol], start, end, horizon)
            if not np.isclose(check["net_return"], row.path_target, atol=1e-10, rtol=1e-8):
                raise AssertionError("缓存标签与原引擎不一致")
            reports.append(check)
    print(f"真实核心路径逐例比对通过：{len(reports)}例", flush=True)
    return reports


def write_data_audit(cfg: dict) -> Path:
    """按cfg先构建/复用pkl，再与原引擎比对核心路径并记录全量异常执行字段计数，不训练或预测外层。"""
    panel, market, metadata = load_data(cfg)
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("data_audit_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    rows = audit_reference_paths(panel, market, cfg)
    write_table(directory / "核心路径对照.csv", rows)
    price = market.values["execution_close"]
    visible = np.isfinite(price) & (price > 0)
    quality = {}
    for field in ["execution_volume", "execution_amount", "previous_close"]:
        values = market.values[field]
        quality[field + "_invalid_with_price"] = int((visible & (~np.isfinite(values) | (values <= 0))).sum())
        quality[field + "_nan_with_price"] = int((visible & np.isnan(values)).sum())
    write_json(directory / "data_metadata.json", {**metadata, "execution_quality": quality, "reference_cases": len(rows)})
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (directory / "final_report.md").write_text("# 执行路径标签数据审计\n\n"
        f"复用原因子，路径标签{metadata['path_cache']['rows']:,}个预定入口；{len(rows)}个真实核心训练路径与原引擎一致。\n\n"
        "没有训练或生成外层预测。缺失成交量/前收等异常执行输入现在失败关闭，完整质量计数见data_metadata.json。\n\n"
        + cfg["report_limitations"], encoding="utf-8")
    (directory / "输出文件说明.md").write_text("# 文件角色\n\n"
        "- final_report.md：数据契约和限制，不是收益报告。\n- 核心路径对照.csv：单次意图净收益、买股、余股、卖出次数与原引擎误差。\n"
        "- data_metadata.json：缓存身份、实际样本数和异常执行字段数量。\n- resolved_config.yaml：完整配置；大型标签pkl仍在cache目录。\n",
        encoding="utf-8")
    print(f"路径数据审计：{directory}", flush=True)
    return directory


def write_report(directory: Path, cfg: dict, summary: list, promoted: bool) -> None:
    """将directory训练summary写成中文报告及文件说明；promoted只表示可否安排正式外层，cfg给契约限制。"""
    text = "# 执行路径模型内部训练报告\n\n没有外层测试收益，不能据此认定平均夏普超过3。\n\n"
    text += pd.DataFrame(summary).to_markdown(index=False) + "\n\n"
    text += f"至少一种方法四折全部内部启用：{'是' if promoted else '否'}。\n\n"
    text += """## 方法与依据

方法一直接预测独立入场意图的账户净收益；方法二学习意图盈利概率，再乘核心训练的正/非正收益幅度。
两者使用同一执行路径标签、固定29列输入和同一账户引擎，因此不能视为独立收益来源。
策略最多买前5个超过门槛的标的，每批最多总资产10%，隔日可新建同股独立批次，离池不退出。
平仓为初始20倍/吊顶40倍简单分钟ATR，或3/5/10交易日同刻到期，满足T+1后下一分钟尝试成交。
标签观察宽限期不是策略保证成交日，真实回测保留未成交的待退出批次。

本轮借鉴按退出路径定义标签的思路，不把通用三重障碍默认实现直接套到A股，也不声称新增固定百分比止盈。
[标签方法文档](https://mlfinpy.readthedocs.io/en/stable/Labelling.html)与
[LightGBM回归接口](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
仅提供方法背景，不证明本地long-only（仅做多）净收益或夏普。

## 限制

"""
    text += cfg["report_limitations"]
    for name in ["final_report.md", "training_report.md"]:
        (directory / name).write_text(text, encoding="utf-8")
    (directory / "输出文件说明.md").write_text("""# 文件角色与字段

| 文件/目录 | 用途 |
|---|---|
| final_report.md / training_report.md | 当前主报告与冻结训练原文；内部不能冒充外层 |
| resolved_config.yaml / research_protocol.json | 有效参数、事前网格/晋级/验收、源码和软件版本 |
| folds.json | 年度留出与核心/校准/验证边界 |
| data_metadata.json | 原年度、有界特征及执行路径标签pkl身份；不复制大数据 |
| source_snapshot/ | 本轮源码快照；旧研究需用其对应快照，不修改旧指纹 |
| core_path_reference.csv | 只在核心训练样本上的加速标签与原引擎逐例对照 |
| 年份/model_*.joblib | 冻结LightGBM及期望收益转换信息，只加载可信本地文件 |
| 年份/model_*.txt | LightGBM原生文本模型，配合JSON的特征顺序和幅度定义使用 |
| 年份/model_*.json | 期限、超参、核心样本数、目标单位及正/非正收益幅度 |
| 年份/explanation_*.csv | 特征代码、中文含义、训练覆盖率及分裂重要性，不是收益保证 |
| 年份/model_unavailable.csv | 核心样本/类别不足的配置及原因，不按测试剔除 |
| 年份/model_feature_manifest.csv | 29个固定可见输入；标签与执行未来价格不在其中 |
| 年份/training_rule_metrics.csv | 全部内部校准候选的组合收益、回撤、交易数和门槛 |
| 年份/validation_rule_metrics.csv | 每方法去重模型后的短名单验证，不重新拟合门槛 |
| 年份/selection_segment_metrics.csv | 每个内部年份片段的独立现金账户指标 |
| 年份/internal_forecast_diagnostics.csv | 仅内部的意图账户收益标签分箱，不等于实际共享资金组合收益 |
| 年份/frozen_rules.json | 分方法冻结模型、开仓门槛和对应退出；现金规则不跳过年度 |
| split_audit.csv | 各模型实际核心拟合年份、数量、来源及潜在标签终点 |
| internal_summary.csv | 八组内部启用状态，非外层成绩 |
| frozen_manifest.json | 所有模型、规则、内部表、源码和配置指纹 |
| training_audit.json | 真实训练行重建、模型重载、无标签推断及门槛审计 |
| test_started.json | 仅正式晋级或烟测后创建的一次性外层标记 |
| test_year_metrics.csv / test_summary.csv | 仅外层后生成的八组指标与完整四年均值 |
| 年份/方法/ | 仅外层后生成，标准中文收益图、估值、成交、批次配对、期末持仓和执行审计 |

模型代码path_h3_shallow_regression表示3日退出路径、较浅回归模型；classification表示盈利概率模型。
threshold为期望独立账户净收益门槛，quantile为内部校准分位，horizon为交易日期限，
initial/trailing为分钟ATR倍数。cash_validation_failed表示内部标准不满足而现金保护。
rows是样本数，predicted是平均预测账户收益，realized_account_label是实际独立路径账户标签均值。
校准/验证表中的closed_lots是完全平仓批次，positive_segments是正收益片段数，active_segments是有波动片段数。
未正式晋级时不生成空壳测试图，也不复制旧图或烟测图。
""", encoding="utf-8")


def run_training(cfg: dict) -> Path:
    """按cfg先写事前协议，再拟合四折路径模型与内部规则，全部冻结后退出，不评价外层。"""
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    hashes = {}
    for name in dict.fromkeys(SOURCES + EXTRA_SOURCES + PATH_SOURCES):
        dest = directory / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        hashes[name] = calculate_sha256(dest)
    write_json(directory / "research_protocol.json", {"created": datetime.now().isoformat(), "note": NOTE,
        "source_hashes": hashes, "config_sha256": calculate_sha256(directory / "resolved_config.yaml"),
        "profiles": create_path_profiles(cfg), "features": PATH_FEATURES,
        "candidate_basis": "固定期限价格标签未计实际ATR/待卖/容量；修正同步日期抽样把2026的10日核心合格入口全部漏掉的问题；不从旧外层成绩改参数",
        "promotion": "至少一种方法四折内部全部启用，再允许单次正式外层",
        "criterion": "每方法完整四年有效夏普等权平均严格>3；现金或无波动年不得删掉",
        "internal_selection": "校准均夏普>0、全部片段有波动、足够完全平仓/正收益片段；去重模型取前三验证，原门槛不变",
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "numba", "lightgbm", "scikit-learn"]}})
    print(f"执行路径训练目录：{directory}", flush=True)
    panel, market, metadata = load_data(cfg)
    write_json(directory / "data_metadata.json", metadata)
    write_table(directory / "core_path_reference.csv", audit_reference_paths(panel, market, cfg))
    folds = build_research_folds(market.times, cfg)
    write_json(directory / "folds.json", folds)
    split, summary, active = [], [], {}
    for fold in folds:
        year = fold["year"]
        path = directory / str(year)
        path.mkdir()
        write_table(path / "model_feature_manifest.csv", [{"特征代码": c, "中文含义": v} for c, v in PATH_FEATURES.items()])
        outer = select_bounded_rows(panel, market.times, year, cfg["feature_embargo_trading_days"])
        outer = outer.loc[outer.cross_section_count >= cfg["min_cross_section"]]
        core = outer.loc[outer.fit_source_allowed]
        models, unavailable = {}, []
        for profile in create_path_profiles(cfg):
            train = select_path_rows(core, fold["fit"], profile["horizon"], cfg, market.times, True)
            name = profile["name"]
            if len(train) < cfg["simple_min_training_rows"] or (
                    profile["algorithm"] == "classification" and (train.path_target > 0).nunique() < 2):
                unavailable.append({"profile": name, "rows": len(train), "reason": "核心样本数或正/非正类别不足"})
                continue
            validate_core_sources(train, year)
            bundle, details = fit_path_model(train, profile, cfg)
            models[name] = bundle
            joblib.dump(bundle, path / f"model_{name}.joblib", compress=3)
            bundle["model"].booster_.save_model(str(path / f"model_{name}.txt"))
            write_json(path / f"model_{name}.json", {k: v for k, v in bundle.items() if k != "model"})
            details.to_csv(path / f"explanation_{name}.csv", index=False, encoding="utf-8-sig")
            for y, group in train.groupby(train.datetime.dt.year):
                split.append({"test_year": year, "profile": name, "fit_year": int(y), "rows": len(group),
                    "first_source": group.feature_source_start.min(), "first_signal": group.datetime.min(),
                    "last_signal": group.datetime.max(), "last_label": group.label_end.max()})
            print(f"{year}/{name}：核心{len(train):,}条，正意图比例{(train.path_target > 0).mean():.1%}", flush=True)
        write_table(path / "model_unavailable.csv", unavailable, ["profile", "rows", "reason"])
        rules = select_rules(models, outer, fold, market, cfg, path, generate_path_scores,
                             calculate_path_diagnostics, create_path_exits)
        write_json(path / "frozen_rules.json", rules)
        val = pd.read_csv(path / "validation_rule_metrics.csv", float_precision="round_trip")
        for method, rule in rules.items():
            enabled = rule["threshold"] < 1e9
            active[year, method] = enabled
            row = val.loc[(val.method == method) & (val.rule_id == rule["rule_id"])]
            summary.append({"留出年": year, "方法": METHOD_NAMES[method], "内部启用": enabled,
                "选中规则": rule["rule_id"], "内部验证均夏普": float(row.mean_sharpe.iloc[0]) if len(row) else None,
                "完全平仓批次": int(row.closed_lots.iloc[0]) if len(row) else 0,
                "核心拟合含2026": any(r["test_year"] == year and r["profile"] == rule["profile"] and r["fit_year"] == 2026 for r in split)})
        print(f"{year}折冻结：{ {m: r['rule_id'] for m, r in rules.items()} }", flush=True)
    promoted = any(all(active[f["year"], method] for f in folds) for method in METHOD_NAMES)
    write_table(directory / "split_audit.csv", split)
    write_table(directory / "internal_summary.csv", summary)
    write_report(directory, cfg, summary, promoted)
    protected = {str(p.relative_to(directory)): calculate_sha256(p) for p in directory.rglob("*") if p.is_file()}
    write_json(directory / "frozen_manifest.json", {"frozen_at": datetime.now().isoformat(), "promoted": promoted, "files": protected})
    print(f"四折已冻结；正式外层晋级={promoted}：{directory}", flush=True)
    return directory


def load_frozen_data(cfg: dict, directory: Path) -> tuple:
    """按cfg复用缓存并核对directory记录的完整身份，拒绝缓存或定义漂移，不重训。"""
    panel, market, metadata = load_data(cfg)
    if metadata != json.loads((directory / "data_metadata.json").read_text()):
        raise ValueError("路径或有界缓存身份变化")
    return panel, market


def run_audit(directory: Path) -> None:
    """重建directory每模型核心行，核对来源、潜在标签边界、重载无标签推断和全部入围门槛。"""
    cfg, frozen, folds = validate_frozen(directory)
    panel, market = load_frozen_data(cfg, directory)
    splits = pd.read_csv(directory / "split_audit.csv")
    checks = []
    for fold in folds:
        year, path = fold["year"], directory / str(fold["year"])
        outer = select_bounded_rows(panel, market.times, year, cfg["feature_embargo_trading_days"])
        outer = outer.loc[outer.fit_source_allowed & (outer.cross_section_count >= cfg["min_cross_section"])]
        for file in sorted(path.glob("model_*.joblib")):
            bundle = joblib.load(file)
            profile = bundle["profile"]
            train = select_path_rows(outer, fold["fit"], profile["horizon"], cfg, market.times, True)
            validate_core_sources(train, year)
            expected = splits.loc[(splits.test_year == year) & (splits.profile == profile["name"])]
            if len(train) != bundle["training_rows"] or len(train) != expected.rows.sum():
                raise ValueError("实际核心样本数量不同")
            if train.groupby(train.datetime.dt.year).size().to_dict() != dict(zip(expected.fit_year, expected.rows)):
                raise ValueError("实际核心年份不一致")
            for row in expected.itertuples():
                group = train.loc[train.datetime.dt.year == row.fit_year]
                observed = [group.feature_source_start.min(), group.datetime.min(), group.datetime.max(), group.label_end.max()]
                if observed != [pd.Timestamp(v) for v in [row.first_source, row.first_signal, row.last_signal, row.last_label]]:
                    raise ValueError("实际核心来源或标签边界不一致")
            current = train.head(32)
            stripped = current[["datetime", "symbol"] + list(PATH_FEATURES)]
            original = generate_path_scores(bundle, current, cfg)
            reloaded = generate_path_scores(joblib.load(file), stripped, cfg)
            if not original.equals(reloaded):
                raise ValueError("重载或删除未来标签后预测改变")
            if bundle["model"].booster_.feature_name() != list(PATH_FEATURES):
                raise ValueError("模型特征名或顺序不一致")
            checks.append({"year": year, "profile": profile["name"], "rows": len(train), "passed": True})
        validate_threshold_contract(pd.read_csv(path / "training_rule_metrics.csv", float_precision="round_trip"),
            pd.read_csv(path / "validation_rule_metrics.csv", float_precision="round_trip"),
            json.loads((path / "frozen_rules.json").read_text()))
    write_json(directory / "training_audit.json", {"created": datetime.now().isoformat(), "passed": True,
        "models": checks, "promoted": frozen["promoted"], "outer_predictions_read": False})
    print(f"实际训练与重载审计通过：{len(checks)}模型；{directory}", flush=True)


def run_evaluation(directory: Path) -> None:
    """校验directory冻结与晋级后仅一次输出八组真实年度组合；无波动年度不参与有效平均冒充。"""
    cfg, frozen, folds = validate_frozen(directory)
    if (directory / "test_started.json").exists():
        raise ValueError("外层已经启动过，不能重复挑选")
    if not cfg["is_smoke"] and not frozen["promoted"]:
        raise ValueError("没有一种方法四折内部全启用，禁止正式外层")
    panel, market = load_frozen_data(cfg, directory)
    write_json(directory / "test_started.json", {"started": datetime.now().isoformat(), "is_smoke": cfg["is_smoke"]})
    rows = []
    for fold in folds:
        year, path = fold["year"], directory / str(fold["year"])
        frame = panel.loc[(panel.datetime >= fold["test_start"]) & (panel.datetime < fold["test_end"])
            & (panel.cross_section_count >= cfg["min_cross_section"])]
        rules = json.loads((path / "frozen_rules.json").read_text())
        for method, title in METHOD_NAMES.items():
            rule = rules[method]
            bundle = joblib.load(path / f"model_{rule['profile']}.joblib")
            scores = generate_path_scores(bundle, frame, cfg)
            result = run_lot_portfolio(scores, market, fold["test_start"], fold["test_end"], "prediction",
                rule, cfg, create_exit_policy(market, rule))
            audit = validate_result(result, cfg)
            result.update(report_note=NOTE, limitations=cfg["report_limitations"])
            save_result(path / method, result, f"{year}年 {'烟测：' if cfg['is_smoke'] else ''}{title}", rule, cfg)
            write_json(path / method / "execution_audit.json", audit)
            rows.append({"year": year, "method": method, **result["metrics"]})
    write_table(directory / "test_year_metrics.csv", rows)
    table, summary = pd.DataFrame(rows), []
    for method in METHOD_NAMES:
        part = table.loc[table.method == method]
        valid = len(part) == 4 and (part.annual_volatility > 0).all()
        mean = float(part.sharpe.mean()) if valid else None
        summary.append({"方法": METHOD_NAMES[method], "有效年数": int((part.annual_volatility > 0).sum()),
            "完整四年平均夏普": mean, "超过3": bool(valid and mean > cfg["sharpe_target"] and not cfg["is_smoke"])})
    write_table(directory / "test_summary.csv", summary)
    text = "# 执行路径模型年度回顾测试\n\n" + NOTE + "\n\n"
    if cfg["is_smoke"]:
        text += "**8股流程烟测，不用于盈利验收。**\n\n"
    text += pd.DataFrame(summary).to_markdown(index=False) + "\n\n"
    text += table[["year", "method", "total_return", "sharpe", "max_drawdown", "buy_count", "closed_lot_count"]].rename(columns={
        "year": "留出年", "method": "方法代码", "total_return": "收益率", "sharpe": "夏普",
        "max_drawdown": "最大回撤", "buy_count": "买入次数", "closed_lot_count": "完全平仓批次"}).to_markdown(index=False)
    text += "\n\n空仓年度CSV的夏普0只是占位，统计上未定义，不能丢弃年度计算高均值。\n\n"
    for fold in folds:
        for method in METHOD_NAMES:
            text += f"- [{fold['year']}年{METHOD_NAMES[method]}]({fold['year']}/{method}/final_report.md)\n"
    text += "\n## 限制\n\n" + cfg["report_limitations"]
    (directory / "final_report.md").write_text(text, encoding="utf-8")
    print(f"一次性外层完成：{directory}", flush=True)


def run_cli() -> None:
    """解析config/phase/run-dir/build-only/smoke/workers；默认仅训练，审计和测试必须指定冻结目录。"""
    parser = argparse.ArgumentParser(description="执行路径净收益模型研究")
    parser.add_argument("--config", type=Path, default=ROOT / "config/pathwise_year_validation.yaml")
    parser.add_argument("--phase", choices=["train", "audit", "evaluate"], default="train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.phase != "train":
        if args.run_dir is None or args.build_only:
            parser.error("audit/evaluate必须指定run-dir，且不能同时build-only")
        (run_audit if args.phase == "audit" else run_evaluation)(args.run_dir.resolve())
        return
    cfg = load_config(args)
    (write_data_audit if args.build_only else run_training)(cfg)


if __name__ == "__main__":
    run_cli()
