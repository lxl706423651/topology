"""在quant根目录研究完整核心加权路径模型，输入YAML与既有只读特征/标签pkl。

train重新拟合直接模型及环境/个股分解模型，只在内部选择；audit重建实际输入、
目标、权重和各原生模型；evaluate通过事前晋级与审计后仅一次输出年度回顾报告。
创建新模型/报告，不重建原因子、不覆盖旧快照、不连接券商。
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
from lightgbm import Booster

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.annual_portfolio import run_lot_portfolio
from quant_backtest.bounded_training import select_bounded_rows
from quant_backtest.learned_exit_models import build_research_folds
from quant_backtest.multiscale_training import MACRO_COLUMNS
from quant_backtest.pathwise_labels import PATH_FEATURES
from quant_backtest.pathwise_models import create_path_exits
from quant_backtest.pooled_path_models import (METHOD_NAMES, STOCK_FEATURES, select_pooled_training,
    create_pooled_profiles, fit_pooled_model, generate_pooled_scores, calculate_pooled_diagnostics,
    summarize_pooled_training)
from quant_backtest.simple_signals import create_exit_policy
from scripts.run_pathwise_year_validation import (load_config as load_path_config, load_data,
    load_frozen_data, audit_reference_paths, PATH_SOURCES)
from scripts.run_simple_signal_year_validation import (SOURCES, EXTRA_SOURCES, write_table,
    select_rules, validate_frozen)
from scripts.run_learned_exit_year_validation import calculate_sha256
from scripts.run_annual_method_validation import save_result, validate_result
from scripts.audit_bounded_training import validate_core_sources
from scripts.audit_multiscale_frozen_training import validate_threshold_contract

NEW_SOURCES = ["quant_backtest/pooled_path_models.py", "scripts/run_pooled_path_year_validation.py",
               "tests/test_pooled_path_models.py"]
NOTE = "完整核心加权留一年回顾；前三折含未来年份训练，2026仅1至4月，不是实盘证明。"
PACKAGES = ["numpy", "pandas", "numba", "lightgbm", "scikit-learn"]


def load_config(args: argparse.Namespace) -> dict:
    """合并args的完整核心YAML，沿用路径缓存与交易限制，烟测仅缩模型和标签组下限。"""
    cfg = load_path_config(args)
    if args.smoke:
        cfg.update(pooled_min_label_group=1,
            pooled_model_profiles=[{"name": "smoke", "num_leaves": 4, "max_depth": 2,
                                    "min_child_samples": 10, "n_estimators": 8, "reg_lambda": 10.}],
            pooled_environment_model={"num_leaves": 3, "max_depth": 2, "min_child_samples": 5,
                                      "n_estimators": 8, "reg_lambda": 25.})
    if not isinstance(cfg["pooled_min_label_group"], int) or cfg["pooled_min_label_group"] < 1:
        raise ValueError("核心同刻标签数必须为正整数")
    cfg["report_note"] = NOTE
    excluded = ["核心样本一股一天", "正式默认按股票代码", "方法一预测截断后的", "固定29个当时特征"]
    cfg["report_limitations"] = "\n".join(line for line in cfg["report_limitations"].splitlines()
                                         if not any(s in line for s in excluded))
    cfg["report_limitations"] += """
- 本轮使用同股每天预定一个钟的全部完整核心入口，训练内部潜在窗口可以重叠；按同股同年潜在区间的平均1/并发数降权，再令每日总权重相同。权重不能消除时间/跨股票依赖，也不等于独立样本量。
- 跨核心、校准、验证、留出年的完整潜在标签边界与实际量价来源隔离不放宽；即使提前平仓仍按H+5窗口过滤，不按盈利与否放宽。
- 方法一用固定29个当时输入预测账户净收益；方法二用5个当时环境量预测核心同刻共同净收益、24个个股量预测相对残差，推断相加。不从验证/测试标签计算环境均值、标准化或分数校准，正相对收益不意味着绝对盈利。
- 共同目标是核心预定入口样本的加权同刻平均，不是整个股票池未来收益或可交易基准；这些入口不共享现金/容量。分解模型不是市场中性策略，不存在空头或可获得的对冲收益。
- 本轮不强制推断残差在当前截面归零；每行的两个冻结分量直接相加，支持等价分批推断。旧path_training_phase仅用于抽90个路径参考案例，不再用于实际核心拟合。
"""
    return cfg


def write_report(directory: Path, cfg: dict, summary: list, promoted: bool) -> None:
    """把directory的内部summary、cfg契约及promoted状态写成中文报告，不伪造外层结果。"""
    text = "# 完整核心加权与分解模型：内部研究\n\n尚未运行正式外层，内部夏普不能当年度测试夏普。\n\n"
    text += pd.DataFrame(summary).to_markdown(index=False) + "\n\n"
    text += f"至少一种方法四折内部全部启用：{'是' if promoted else '否'}。\n\n"
    text += """## 方法

方法一直接拟合扣费后的独立账户净收益，方法二分开拟合同刻环境和个股相对收益再相加。
两方法共享完整核心样本、路径标签、费用和退出，比较的是监督结构，而不是调整仓位或删掉年份。
前5门控、每批总资产10%、隔日可独立加仓、T+1、万八免5、ATR和对应期限退出均保留。
训练允许核心内部标签重叠但降权，跨角色和留出年的完整边界不放宽，不将样本数增加解释为独立信息等量增加。

依据为[Qlib截面处理源码](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py)、
[标签并发与唯一性](https://mlfinpy.readthedocs.io/en/latest/Sampling.html)及
[LightGBM 4.6.0接口](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html)。
本实现用净收益均值分解，不是照搬Qlib秩标签；唯一性按固定潜在窗口，不使用提前退出收益挑权重。资料不证明盈利。

## 限制

"""
    for name in ["final_report.md", "training_report.md"]:
        (directory/name).write_text(text+cfg["report_limitations"], encoding="utf-8")
    (directory/"输出文件说明.md").write_text("""# 输出文件角色

| 文件/目录 | 角色与用途 |
|---|---|
| final_report.md / training_report.md | 当前主报告和冻结内部原文；未评价时没有测试收益 |
| research_protocol.json / resolved_config.yaml | 事前候选、晋级/验收规则、源码/版本以及所有有效配置 |
| folds.json | 四个留出年和每个内部年份核心/校准/验证的时间边界 |
| data_metadata.json | 年度、有界特征和执行路径标签pkl身份，不复制或重建原因子 |
| source_snapshot/ | 本轮源码快照，不与其他轮次混用 |
| core_path_reference.csv | 仅核心案例的加速路径标签与原账户引擎对照 |
| 年份/model_*.joblib | 模型包：direct直接分量，或stock个股残差+environment环境分量；只加载可信本地文件 |
| 年份/model_*.json | 算法/期限、真实核心数量、特征顺序及输入/标签/权重/环境目标指纹 |
| 年份/booster_*.txt | 各个LightGBM原生分量，必须配合JSON，原始输出是账户收益基点 |
| 年份/explanation_*.csv | 各分量的特征代码、中文含义和分裂重要性，不是收益贡献保证 |
| 年份/model_feature_manifest.csv | 固定29列的中文含义及环境/个股分工 |
| 年份/model_unavailable.csv | 因核心样本不足而未拟合的候选，不按测试成绩剔除 |
| 年份/training_rule_metrics.csv | 全部校准候选，mean_sharpe内部平均夏普、closed_lots完全平仓批次 |
| 年份/validation_rule_metrics.csv | 校准去重后短名单的验证，冻结原门槛，不重新拟合 |
| 年份/selection_segment_metrics.csv | 各内部年份片段真实组合收益、回撤、交易及费用 |
| 年份/internal_forecast_diagnostics.csv | 内部预测账户收益分箱与独立路径标签均值，不是组合收益 |
| 年份/frozen_rules.json | 分方法的模型、开仓门槛、ATR/到期规则或现金保护 |
| split_audit.csv | 每模型实际核心年份、样本数、日期数、唯一性和标签边界 |
| internal_summary.csv | 八组内部选择结果，不能代替留出年收益 |
| frozen_manifest.json | 模型、规则、内部表、源码和配置的冻结指纹 |
| training_audit.json | 真实核心特征/标签/权重/环境目标、原生模型重载和门槛审计 |
| test_started.json | 晋级和审计通过后唯一一次启动外层的标记 |
| test_year_metrics.csv / test_summary.csv | 仅外层后存在，年度指标及完整四年平均夏普 |
| 年份/method1或method2/ | 仅外层后存在：中文收益图、估值、成交/平仓/未平仓、指标和逐文件说明 |

训练/验证中的权重不能当独立样本量。正式未晋级时不创建测试曲线，不复制烟测图或旧图。
""", encoding="utf-8")


def run_training(cfg: dict) -> Path:
    """先按cfg写事前协议，再拟合和内部选择，保存所有分量和真实输入指纹，冻结后退出。"""
    directory = Path(cfg["output_dir"])/datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    hashes = {}
    for name in dict.fromkeys(SOURCES+EXTRA_SOURCES+PATH_SOURCES+NEW_SOURCES):
        target = directory/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        hashes[name] = calculate_sha256(target)
    write_json(directory/"research_protocol.json", {"created": datetime.now().isoformat(),
        "source_hashes": hashes, "config_sha256": calculate_sha256(directory/"resolved_config.yaml"),
        "profiles": create_pooled_profiles(cfg), "features": PATH_FEATURES,
        "packages": {p:importlib.metadata.version(p) for p in PACKAGES}, "note": NOTE,
        "basis": "核心样本稀疏及同刻标签含共同收益；完整核心降重与环境/个股分解；不从旧外层成绩改参数",
        "promotion": "至少一种方法四折内部全部启用，且真实输入与分量重载审计通过，才单次正式外层",
        "criterion": "同一方法完整四个有效年度夏普的等权均值严格大于3，不删除现金或亏损年",
        "training_weights": "同股同年固定潜在闭区间平均1/并发数，归一使每日总权重相同；不是独立样本证明"})
    print(f"完整核心训练目录：{directory}",flush=True)
    panel,market,metadata = load_data(cfg)
    write_json(directory/"data_metadata.json",metadata)
    write_table(directory/"core_path_reference.csv",audit_reference_paths(panel,market,cfg))
    folds = build_research_folds(market.times,cfg)
    write_json(directory/"folds.json",folds)
    split,summary,active = [],[],{}
    for fold in folds:
        year = fold["year"]
        path = directory/str(year)
        path.mkdir()
        write_table(path/"model_feature_manifest.csv",[{"特征代码":c,"中文含义":v,
            "分解模型用途":"环境" if c in MACRO_COLUMNS else "个股"} for c,v in PATH_FEATURES.items()])
        outer = select_bounded_rows(panel,market.times,year,cfg["feature_embargo_trading_days"])
        outer = outer.loc[outer.cross_section_count>=cfg["min_cross_section"]]
        core = outer.loc[outer.fit_source_allowed]
        models,unavailable = {},[]
        for profile in create_pooled_profiles(cfg):
            train = select_pooled_training(core,fold["fit"],profile["horizon"],cfg,market.times)
            name = profile["name"]
            if len(train)<cfg["simple_min_training_rows"]:
                unavailable.append({"profile":name,"rows":len(train),"reason":"完整核心样本不足"})
                continue
            validate_core_sources(train,year)
            bundle,details = fit_pooled_model(train,profile,cfg)
            models[name] = bundle
            joblib.dump(bundle,path/f"model_{name}.joblib",compress=3)
            write_json(path/f"model_{name}.json",{k:v for k,v in bundle.items() if k!="components"})
            for component,model in bundle["components"].items():
                model.booster_.save_model(str(path/f"booster_{name}_{component}.txt"))
            details.to_csv(path/f"explanation_{name}.csv",index=False,encoding="utf-8-sig")
            for y,g in train.groupby(train.datetime.dt.year):
                split.append({"test_year":year,"profile":name,"fit_year":int(y),"rows":len(g),
                    "days":int(g.datetime.dt.normalize().nunique()),"mean_uniqueness":float(g.sample_uniqueness.mean()),
                    "first_source":g.feature_source_start.min(),"first_signal":g.datetime.min(),
                    "last_signal":g.datetime.max(),"last_label":g.label_end.max()})
            print(f"{year}/{name}：核心{len(train):,}条，{bundle['training_days']}日，唯一性均值{bundle['mean_uniqueness']:.3f}",flush=True)
        write_table(path/"model_unavailable.csv",unavailable,["profile","rows","reason"])
        rules = select_rules(models,outer,fold,market,cfg,path,generate_pooled_scores,
                             calculate_pooled_diagnostics,create_path_exits)
        write_json(path/"frozen_rules.json",rules)
        val = pd.read_csv(path/"validation_rule_metrics.csv",float_precision="round_trip")
        for method,rule in rules.items():
            enabled = rule["threshold"]<1e9
            active[year,method] = enabled
            row = val.loc[(val.method==method)&(val.rule_id==rule["rule_id"])]
            summary.append({"留出年":year,"方法":METHOD_NAMES[method],"内部启用":enabled,
                "选中规则":rule["rule_id"],"内部验证均夏普":float(row.mean_sharpe.iloc[0]) if len(row) else None,
                "完全平仓批次":int(row.closed_lots.iloc[0]) if len(row) else 0,
                "核心拟合含2026":any(r["test_year"]==year and r["profile"]==rule["profile"] and r["fit_year"]==2026 for r in split)})
        print(f"{year}折冻结：{ {m:r['rule_id'] for m,r in rules.items()} }",flush=True)
    promoted = any(all(active[f["year"],m] for f in folds) for m in METHOD_NAMES)
    write_table(directory/"split_audit.csv",split)
    write_table(directory/"internal_summary.csv",summary)
    write_report(directory,cfg,summary,promoted)
    protected = {str(p.relative_to(directory)):calculate_sha256(p) for p in directory.rglob("*") if p.is_file()}
    write_json(directory/"frozen_manifest.json",{"frozen_at":datetime.now().isoformat(),"promoted":promoted,"files":protected})
    print(f"四折完整核心训练冻结；外层晋级={promoted}：{directory}",flush=True)
    return directory


def run_audit(directory: Path) -> None:
    """从directory可信缓存重建每模型实际核心及分解监督，核对指纹/原生分量/无标签推断，不评价外层。"""
    cfg,frozen,folds = validate_frozen(directory)
    protocol = json.loads((directory/"research_protocol.json").read_text())
    if any(importlib.metadata.version(p)!=v for p,v in protocol["packages"].items()):
        raise ValueError("软件版本改变，需使用冻结环境")
    panel,market = load_frozen_data(cfg,directory)
    splits = pd.read_csv(directory/"split_audit.csv")
    checks = []
    for fold in folds:
        year,path = fold["year"],directory/str(fold["year"])
        outer = select_bounded_rows(panel,market.times,year,cfg["feature_embargo_trading_days"])
        core = outer.loc[outer.fit_source_allowed & (outer.cross_section_count>=cfg["min_cross_section"])]
        for file in sorted(path.glob("model_*.joblib")):
            bundle = joblib.load(file)
            profile = bundle["profile"]
            train = select_pooled_training(core,fold["fit"],profile["horizon"],cfg,market.times)
            validate_core_sources(train,year)
            actual = summarize_pooled_training(train,cfg)
            if any(bundle[k]!=v for k,v in actual.items()):
                raise ValueError(f"真实特征/目标/权重/环境目标不能重建：{file.name}")
            expected = splits.loc[(splits.test_year==year)&(splits.profile==profile["name"])]
            if train.groupby(train.datetime.dt.year).size().to_dict()!=dict(zip(expected.fit_year,expected.rows)):
                raise ValueError("真实拟合年份或数量变化")
            for row in expected.itertuples():
                g = train.loc[train.datetime.dt.year==row.fit_year]
                if [g.feature_source_start.min(),g.datetime.min(),g.datetime.max(),g.label_end.max()] != [
                        pd.Timestamp(v) for v in [row.first_source,row.first_signal,row.last_signal,row.last_label]]:
                    raise ValueError("真实核心来源边界变化")
            visible = train.head(32)[["datetime","symbol"]+list(PATH_FEATURES)]
            score = generate_pooled_scores(bundle,visible,cfg)
            if not score.equals(generate_pooled_scores(joblib.load(file),train.head(32),cfg)):
                raise ValueError("删除标签或模型重载后预测变化")
            expected_schema = {"direct":list(PATH_FEATURES)} if profile["algorithm"]=="direct" else {
                "stock":STOCK_FEATURES,"environment":MACRO_COLUMNS}
            if set(bundle["components"])!=set(expected_schema):
                raise ValueError("模型分量缺失或额外增加")
            for component,model in bundle["components"].items():
                native = Booster(model_file=str(path/f"booster_{profile['name']}_{component}.txt"))
                if model.booster_.feature_name()!=expected_schema[component] or native.feature_name()!=expected_schema[component]:
                    raise ValueError("原生分量特征顺序不一致")
                x = visible[expected_schema[component]].replace([np.inf,-np.inf],np.nan)
                np.testing.assert_allclose(model.predict(x),native.predict(x,num_threads=cfg["workers"]),rtol=0,atol=0)
            checks.append({"year":year,"profile":profile["name"],"rows":len(train),
                           "components":len(bundle["components"]),"passed":True})
        validate_threshold_contract(pd.read_csv(path/"training_rule_metrics.csv",float_precision="round_trip"),
            pd.read_csv(path/"validation_rule_metrics.csv",float_precision="round_trip"),
            json.loads((path/"frozen_rules.json").read_text()))
    write_json(directory/"training_audit.json",{"created":datetime.now().isoformat(),"passed":True,
        "models":checks,"promoted":frozen["promoted"],"outer_predictions_read":False,
        "frozen_manifest_sha256":calculate_sha256(directory/"frozen_manifest.json")})
    print(f"实际核心/监督/权重及原生分量重载通过：{len(checks)}模型，{sum(r['components'] for r in checks)}分量",flush=True)


def run_evaluation(directory: Path) -> None:
    """只有directory冻结/审计/四折晋级全部通过后单次外层回顾，生成真实仅做多组合和完整四年汇总。"""
    cfg,frozen,folds = validate_frozen(directory)
    if (directory/"test_started.json").exists():
        raise ValueError("本轮外层已启动，禁止重复评价或选参")
    if not cfg["is_smoke"] and not frozen["promoted"]:
        raise ValueError("没有一种方法四折内部全部启用，禁止正式外层")
    audit = json.loads((directory/"training_audit.json").read_text())
    if not audit["passed"] or audit["frozen_manifest_sha256"]!=calculate_sha256(directory/"frozen_manifest.json"):
        raise ValueError("缺少本轮实际核心和原生模型审计")
    panel,market = load_frozen_data(cfg,directory)
    write_json(directory/"test_started.json",{"started":datetime.now().isoformat(),"is_smoke":cfg["is_smoke"]})
    rows = []
    for fold in folds:
        year,path = fold["year"],directory/str(fold["year"])
        frame = panel.loc[(panel.datetime>=fold["test_start"])&(panel.datetime<fold["test_end"])
                          &(panel.cross_section_count>=cfg["min_cross_section"])]
        rules = json.loads((path/"frozen_rules.json").read_text())
        for method,title in METHOD_NAMES.items():
            rule = rules[method]
            scores = generate_pooled_scores(joblib.load(path/f"model_{rule['profile']}.joblib"),frame,cfg)
            result = run_lot_portfolio(scores,market,fold["test_start"],fold["test_end"],"prediction",
                                       rule,cfg,create_exit_policy(market,rule))
            check = validate_result(result,cfg)
            result.update(report_note=NOTE,limitations=cfg["report_limitations"])
            save_result(path/method,result,f"{year}年 {'烟测：' if cfg['is_smoke'] else ''}{title}",rule,cfg)
            write_json(path/method/"execution_audit.json",check)
            rows.append({"year":year,"method":method,**result["metrics"]})
    table,summary = pd.DataFrame(rows),[]
    write_table(directory/"test_year_metrics.csv",rows)
    for method in METHOD_NAMES:
        part = table.loc[table.method==method]
        valid = len(part)==4 and (part.annual_volatility>0).all()
        mean = float(part.sharpe.mean()) if valid else None
        summary.append({"方法":METHOD_NAMES[method],"有效年数":int((part.annual_volatility>0).sum()),
                        "完整四年平均夏普":mean,"超过3":bool(valid and mean>cfg["sharpe_target"] and not cfg["is_smoke"])})
    write_table(directory/"test_summary.csv",summary)
    text = "# 完整核心加权模型：年度回顾测试\n\n"+NOTE+"\n\n"
    if cfg["is_smoke"]:
        text += "**8股流程烟测，不用于盈利验收。**\n\n"
    text += pd.DataFrame(summary).to_markdown(index=False)+"\n\n"
    text += table[["year","method","total_return","sharpe","max_drawdown","buy_count","closed_lot_count"]].rename(columns={
        "year":"留出年","method":"方法代码","total_return":"收益率","sharpe":"夏普",
        "max_drawdown":"最大回撤","buy_count":"买入次数","closed_lot_count":"完全平仓批次"}).to_markdown(index=False)
    text += "\n\n现金无波动年度夏普未定义，不删除年份求高均值；CSV的0只是占位。\n\n"
    for fold in folds:
        for method in METHOD_NAMES:
            text += f"- [{fold['year']}年{METHOD_NAMES[method]}]({fold['year']}/{method}/final_report.md)\n"
    (directory/"final_report.md").write_text(text+"\n## 限制\n\n"+cfg["report_limitations"],encoding="utf-8")
    print(f"完整核心模型单次外层完成：{directory}",flush=True)


def run_cli() -> None:
    """解析config/phase/run-dir/smoke/workers；默认仅内部训练，审计/外层必须显式指定已冻结目录。"""
    parser = argparse.ArgumentParser(description="完整核心加权路径净收益与环境分解")
    parser.add_argument("--config",type=Path,default=ROOT/"config/pooled_path_year_validation.yaml")
    parser.add_argument("--phase",choices=["train","audit","evaluate"],default="train")
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--workers",type=int)
    args = parser.parse_args()
    if args.phase=="train":
        if args.run_dir:
            parser.error("训练必须创建新目录，不能提供run-dir")
        run_training(load_config(args))
    else:
        if args.run_dir is None:
            parser.error("audit/evaluate需要run-dir")
        (run_audit if args.phase=="audit" else run_evaluation)(args.run_dir.resolve())


if __name__ == "__main__":
    run_cli()
