"""在quant根目录运行四组留一年双XGBoost研究；输入YAML、既有年度pkl和只读行情。

train阶段仅拟合及训练内部选参，生成模型、冻结协议和中文说明；evaluate阶段
校验冻结哈希后一次性输出四年测试收益图、逐笔交易、年均指标。长实验写入新目录，
不覆盖旧报告、不访问券商。2023至2025允许未来年份训练，不能称为因果向前测试。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import load_annual_panel, write_json
from quant_backtest.annual_portfolio import MarketStore, run_lot_portfolio
from quant_backtest.annual_training import select_signal_tradable_panel, select_training_features
from quant_backtest.learned_exit_models import (BASE_FEATURES, STATE_FEATURES, EXIT_FEATURES, LearnedExitPolicy,
    build_research_folds, select_window_rows, calculate_entry_target, fit_return_model, load_exit_cache)
from scripts.run_multi_factor_cta_backtests import add_cta_factor_scores
from scripts.run_annual_method_validation import save_result, validate_result, LIMITATIONS

METHODS = {"method1": ("方法一：净收益开仓与持仓双模型", "prediction"),
           "method2": ("方法二：净收益开仓加趋势叠加与持仓模型", "cta_trend_breakout_score")}
SOURCES = ["quant_backtest/annual_panel.py", "quant_backtest/annual_training.py", "quant_backtest/annual_portfolio.py",
           "quant_backtest/learned_exit_models.py", "quant_backtest/factor_library.py",
           "scripts/run_learned_exit_year_validation.py", "scripts/run_annual_method_validation.py",
           "scripts/run_multi_factor_cta_backtests.py"]
RESEARCH_NOTE = "留一年回顾检验；2023至2025含未来年份训练，不是因果向前测试；2026仅1至4月。"
RESEARCH_LIMITATIONS = ("- " + RESEARCH_NOTE + "\n"
    "- 每轮外层测试年没有进入该轮拟合/校准/验证，但前三轮的训练数据在测试年之后；年份之间角色轮换不是四个完全独立实验。\n"
    "- 每个训练年按交易日分为60%拟合、20%校准、20%验证，分段起点隔离7自然日；标签和虚拟批次入场不得跨越拟合边界。模型冻结后不再重拟合，避免分数阈值漂移。\n"
    "- 持仓模型预测继续一天相对立即卖出的收益，不是最优停止证明；虚拟批次不模拟入场资金和止损存活，存在训练与真实持仓分布差异。\n"
    "- 主模型预测次日收益，保护规则允许最多10或20交易日；滚动判断不保证以预测期限或止损价成交。ATR是240根原始分钟的平均真实波幅，候选10/20/40倍是风险保护，不是日ATR。\n"
    "- 任一年收益无波动则四年平均夏普未定义且目标不达标；不能删除空仓/亏损年后声称平均大于3。全部年份都有有效夏普时按四年等权算术均值评价。\n"
    + "\n".join(line for line in LIMITATIONS.splitlines() if "最终全量模型" not in line) + "\n")


def calculate_sha256(path: Path) -> str:
    """计算path小型配置/源码/模型文件的校验和，用于冻结与重载审计。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(args: argparse.Namespace) -> dict:
    """合并args.config与base_config；相对路径以ROOT解释，烟测只覆盖规模，不改变成本/T+1。"""
    extra = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / extra["base_config"]).read_text(encoding="utf-8"))
    cfg.update(extra)
    for key in ["data_root", "output_dir", "cache_dir", "name_snapshot", "aux_cache_dir"]:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if args.workers:
        cfg["workers"] = args.workers
    if args.smoke:
        cfg.update(max_symbols=8, universe_size=8, min_cross_section=4, min_closed_lots=1, max_features=12,
                   gate_quantiles=[0.5], train_shortlist=1, exit_entry_stride=20, max_exit_training_rows=3000)
        cfg["output_dir"] += "_smoke"
        cfg["aux_cache_dir"] = cfg["output_dir"] + "/cache"
        cfg["model_profiles"] = [{"name": "smoke", "n_estimators": 5, "max_depth": 2}]
        cfg["exit_model"] = {"n_estimators": 5, "max_depth": 2}
        cfg["exit_candidates"] = cfg["exit_candidates"][:1]
    cfg["is_smoke"] = args.smoke
    if cfg["sample_minutes"] != 24 or cfg["lot_size"] != 100 or cfg["min_commission"] != 0:
        raise ValueError("本轮仅验证24分钟时钟、100股与免5；标签费用近似不支持最低佣金")
    if cfg["entry_top_n"] > 5 or not 0 < cfg["entry_equity_fraction"] <= 0.1:
        raise ValueError("必须维持top5及每批最多总资产10%")
    if cfg["fit_fraction"] <= 0 or cfg["calibration_fraction"] <= 0 or cfg["fit_fraction"] + cfg["calibration_fraction"] >= 1:
        raise ValueError("训练内部比例不合法")
    return cfg


def generate_scores(model: XGBRegressor, frame: pd.DataFrame, features: list[str], cfg: dict) -> pd.DataFrame:
    """把冻结model对frame过去特征的净收益基点预测还原为收益单位，并依cfg加CTA；输出不含标签。"""
    out = frame[["datetime", "symbol"] + [c for c in frame if c.startswith("cs_rank_")]].copy()
    out["prediction"] = model.predict(out[features]) / 10000
    out = add_cta_factor_scores(out, cfg["cta_blend"])
    # 两方法都必须满足原始模型净收益门槛；CTA不能把预期净亏损标的提升为准许买入。
    rejected = out.prediction < cfg["entry_min_net_return"]
    out.loc[rejected, ["prediction", "cta_trend_breakout_score"]] = -1e9
    return out[["datetime", "symbol", "prediction", "cta_trend_breakout_score"]]


def evaluate_segments(scores: pd.DataFrame, windows: list[dict], market: MarketStore, policy: LearnedExitPolicy,
                       method: str, rule: dict, cfg: dict) -> tuple[dict, list[dict]]:
    """按windows各自现金启动模拟scores，不跨空缺窗口拼仓；market/policy/rule/cfg定义独立批次执行。"""
    metrics = []
    for window in windows:
        part = scores.loc[(scores.datetime >= window["start"]) & (scores.datetime < window["end"])]
        result = run_lot_portfolio(part, market, window["start"], window["end"], METHODS[method][1], rule, cfg, policy)
        validate_result(result, cfg)
        metrics.append({"segment_year": window["year"], **result["metrics"]})
    table = pd.DataFrame(metrics)
    return {"mean_sharpe": float(table.sharpe.mean()), "worst_drawdown": float(table.max_drawdown.min()),
            "closed_lots": int(table.closed_lot_count.sum()), "positive_segments": int((table.total_return > 0).sum()),
            "active_segments": int((table.annual_volatility > 0).sum())}, metrics


def select_rules(models: dict, exit_model: XGBRegressor, features: list[str], panel: pd.DataFrame, fold: dict,
                 market: MarketStore, matrices: dict, cfg: dict, directory: Path) -> dict:
    """仅接收外层训练池；先校准筛规则再验证选择，阈值从校准预测确定，所有候选和现金条件事前固定。"""
    candidates, calibration_rows, detail_rows = {}, [], []
    frames = {role: select_window_rows(panel, fold[role], False) for role in ["calibration", "validation"]}
    scored = {}
    for profile, model in models.items():
        scored[profile] = {role: generate_scores(model, frame, features, cfg) for role, frame in frames.items()}
        for method, (_, col) in METHODS.items():
            top = scored[profile]["calibration"].groupby("datetime")[col].max()
            for q in cfg["gate_quantiles"]:
                gate = max(cfg["entry_min_net_return"], float(top.quantile(q))) if len(top) else 1e9
                for exit_rule in cfg["exit_candidates"]:
                    key = f"{profile}_{exit_rule['name']}_q{q:g}"
                    rule = {**exit_rule, "threshold": gate, "quantile": q, "profile": profile, "rule_id": key}
                    candidates[key] = rule
                    policy = LearnedExitPolicy(exit_model, matrices, market, rule["exit_threshold"])
                    summary, details = evaluate_segments(scored[profile]["calibration"], fold["calibration"], market, policy, method, rule, cfg)
                    calibration_rows.append({"method": method, "rule_id": key, **summary, **rule})
                    detail_rows.extend({"role": "calibration", "method": method, "rule_id": key, **d} for d in details)
    training = pd.DataFrame(calibration_rows)
    training.to_csv(directory / "training_rule_metrics.csv", index=False, encoding="utf-8-sig")
    winners, validation_rows = {}, []
    for method in METHODS:
        shortlist = training.loc[(training.method == method) & (training.closed_lots >= cfg["min_closed_lots"])]
        shortlist = shortlist.sort_values(["mean_sharpe", "worst_drawdown", "rule_id"], ascending=[False, False, True]).head(cfg["train_shortlist"])
        for row in shortlist.itertuples():
            rule = candidates[row.rule_id]
            policy = LearnedExitPolicy(exit_model, matrices, market, rule["exit_threshold"])
            summary, details = evaluate_segments(scored[rule["profile"]]["validation"], fold["validation"], market, policy, method, rule, cfg)
            validation_rows.append({"method": method, "rule_id": row.rule_id, **summary, **rule})
            detail_rows.extend({"role": "validation", "method": method, "rule_id": row.rule_id, **d} for d in details)
        valid = [r for r in validation_rows if r["method"] == method and r["mean_sharpe"] > 0
                 and r["closed_lots"] >= cfg["min_closed_lots"]
                 and r["positive_segments"] >= cfg["minimum_positive_validation_segments"]]
        if valid:
            selected = sorted(valid, key=lambda r: (-r["mean_sharpe"], -r["worst_drawdown"], r["rule_id"]))[0]
            winners[method] = {**candidates[selected["rule_id"]], "selection": "内部验证平均正夏普、交易数及正收益片段数达标"}
        else:
            winners[method] = {**cfg["exit_candidates"][0], "threshold": 1e9, "quantile": None,
                               "profile": sorted(models)[0], "rule_id": "cash_validation_failed",
                               "selection": "内部验证不达标，按事前协议保持现金，非漏跑测试"}
    pd.DataFrame(validation_rows, columns=training.columns).to_csv(directory / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(directory / "selection_segment_metrics.csv", index=False, encoding="utf-8-sig")
    return winners


def load_research_data(cfg: dict) -> tuple:
    """按cfg复用原始年度pkl，只依信号时可见行情过滤买点，再复用独立持仓缓存；不计算测试预测。"""
    panel, raw, membership, manifest = load_annual_panel(cfg)
    panel = select_signal_tradable_panel(panel, raw, cfg)
    market = MarketStore(raw)
    del raw, membership
    matrices, samples, cache = load_exit_cache(panel, market, manifest, cfg)
    return panel, market, matrices, samples, manifest, cache


def run_training(cfg: dict) -> Path:
    """按cfg启动新目录，仅训练/校准/验证四折；快照代码并冻结模型、规则及配置，不预测任何外层测试。"""
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    source_hashes = {}
    for relative in SOURCES:
        dest = directory / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        source_hashes[relative] = calculate_sha256(dest)
    write_json(directory / "research_protocol.json", {"created": datetime.now().isoformat(), "note": RESEARCH_NOTE,
        "config_sha256": calculate_sha256(directory / "resolved_config.yaml"), "source_hashes": source_hashes,
        "criterion": "两方法各自四个有效年度夏普等权均值严格大于3；无波动年存在即不达标；不以测试结果选参",
        "test_predictions_generated": False,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "xgboost", "scikit-learn"]}})
    print(f"训练目录：{directory}", flush=True)
    panel, market, matrices, samples, manifest, cache = load_research_data(cfg)
    write_json(directory / "data_metadata.json", {"annual_cache": manifest["cache_path"], "exit_cache": cache,
        "panel_rows": len(panel), "symbols": len(market.symbols), "days": int(len(set(market.days)))})
    folds = build_research_folds(market.times, cfg)
    write_json(directory / "folds.json", folds)
    frozen, split_rows = {}, []
    for fold in folds:
        year = fold["year"]
        path = directory / str(year)
        path.mkdir()
        outer = panel.loc[panel.datetime.dt.year != year]
        train = select_window_rows(outer, fold["fit"])
        train = train.loc[train.target_rank.notna() & (train.cross_section_count >= cfg["min_cross_section"])]
        target = calculate_entry_target(train, cfg)
        finite = np.isfinite(target)
        train, target = train.loc[finite], target[finite]
        selection = train.assign(target_rank=target / 10000)
        features, feature_manifest = select_training_features(selection, cfg)
        del selection
        feature_manifest.to_csv(path / "model_feature_manifest.csv", index=False, encoding="utf-8-sig")
        write_json(path / "entry_feature_schema.json", features)
        write_json(path / "exit_feature_schema.json", {**BASE_FEATURES, **STATE_FEATURES})
        exit_train = select_window_rows(samples.loc[samples.datetime.dt.year != year], fold["fit"])
        if len(exit_train) > cfg["max_exit_training_rows"]:
            exit_train = exit_train.sample(n=cfg["max_exit_training_rows"], random_state=cfg["model"]["random_state"]).sort_index()
        if len(exit_train) < 100 or len(train) < 100:
            raise ValueError(f"{year}折训练不足；拒绝跳过年度")
        print(f"{year}折：拟合开仓{len(train):,}行/{len(features)}特征；持仓{len(exit_train):,}行；训练年份{fold['training_years']}", flush=True)
        exit_model = fit_return_model(exit_train, EXIT_FEATURES, exit_train.continuation_target.to_numpy(), cfg, cfg["exit_model"])
        exit_model.save_model(path / "exit_model.json")
        models = {}
        for profile in cfg["model_profiles"]:
            print(f"{year}折开仓模型：{profile['name']}", flush=True)
            model = fit_return_model(train, features, target, cfg, profile)
            model.save_model(path / f"entry_{profile['name']}.json")
            models[profile["name"]] = model
        for role, frame in [("entry_fit", train), ("exit_fit", exit_train)]:
            for y, group in frame.groupby(frame.datetime.dt.year):
                split_rows.append({"test_year": year, "role": role, "fit_year": int(y), "rows": len(group),
                                   "first_signal": group.datetime.min(), "last_signal": group.datetime.max(),
                                   "last_label": group.label_end.max()})
        print(f"{year}折仅训练内部校准/验证选参，未计算外层测试预测", flush=True)
        rules = select_rules(models, exit_model, features, outer, fold, market, matrices, cfg, path)
        write_json(path / "selected_rules.json", rules)
        hashes = {p.name: calculate_sha256(p) for p in path.glob("*.json")}
        frozen[str(year)] = {"fold": fold, "rules": rules, "hashes": hashes}
        write_json(path / "model_metadata.json", {"fold": fold, "entry_features": features, "exit_features": EXIT_FEATURES,
            "entry_target": "下一交易日同刻下一分钟买卖净收益，扣历史税费滑点，截断正负20%，以基点拟合",
            "exit_target": "次日同刻卖出净所得/立即卖出净所得-1；不重复扣已沉没买费；截断正负20%，以基点拟合",
            "entry_parameters": [{**cfg["model"], **p} for p in cfg["model_profiles"]],
            "exit_parameters": {**cfg["model"], **cfg["exit_model"]}, "rules": rules,
            "test_used_for_fit_or_selection": False, "refit_after_validation": False})
        frozen[str(year)]["hashes"]["model_metadata.json"] = calculate_sha256(path / "model_metadata.json")
        print(f"{year}折规则已定：{json.dumps(rules, ensure_ascii=False)}", flush=True)
    pd.DataFrame(split_rows).to_csv(directory / "split_audit.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "frozen_before_test.json", {"created": datetime.now().isoformat(), "folds": frozen,
        "protocol_sha256": calculate_sha256(directory / "research_protocol.json"),
        "config_sha256": calculate_sha256(directory / "resolved_config.yaml"), "source_hashes": source_hashes})
    (directory / "final_report.md").write_text("# 四组留一年双模型研究\n\n已完成训练与内部选参，模型已冻结。尚未执行外层测试，暂无测试收益结论。\n\n" + RESEARCH_LIMITATIONS, encoding="utf-8")
    print(f"四折全部冻结，外层测试尚未运行：{directory}", flush=True)
    return directory


def calculate_four_year_summary(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """对table逐方法计算完整四折均值；cfg指定年度及目标，缺年/零波动时平均值为NaN且不达标。"""
    rows = []
    for method in METHODS:
        group = table.loc[table.method == method]
        complete = len(group) == len(cfg["test_years"]) and set(group.year) == set(cfg["test_years"])
        valid = complete and bool(((group.annual_volatility > 0) & np.isfinite(group.sharpe)).all())
        mean = float(group.sharpe.mean()) if valid else np.nan
        rows.append({"method": method, "complete_years": complete, "active_years": int((group.annual_volatility > 0).sum()),
                     "four_year_mean_sharpe": mean, "target_passed": bool(valid and mean > cfg["sharpe_target"]),
                     "positive_years": int((group.total_return > 0).sum())})
    return pd.DataFrame(rows)


def write_root_report(directory: Path, table: pd.DataFrame, summary: pd.DataFrame) -> None:
    """把directory的四年table和summary写成中文索引、限制和文件用途；不按外层结果推荐参数。"""
    display = table[["year", "method", "total_return", "annual_return", "sharpe", "max_drawdown", "buy_count", "closed_lot_count"]].copy()
    display["method"] = display.method.map({m: t[0] for m, t in METHODS.items()})
    for col in ["total_return", "annual_return", "max_drawdown"]:
        display[col] = display[col].map(lambda x: f"{x:.2%}")
    display["sharpe"] = [f"{r.sharpe:.3f}" if r.annual_volatility > 0 else "未定义（空仓/无波动）" for r in table.itertuples()]
    display.columns = ["测试年", "方法", "收益率", "年化收益", "夏普", "最大回撤", "买入次数", "完整平仓批次"]
    lines = ["# 四组留一年双模型研究结果", "", RESEARCH_NOTE, "", "## 测试结果", "", display.to_markdown(index=False), ""]
    for r in summary.itertuples():
        mean = f"{r.four_year_mean_sharpe:.4f}" if np.isfinite(r.four_year_mean_sharpe) else "未定义（不能删除无波动年度求均值）"
        lines.append(f"- {METHODS[r.method][0]}：四年平均夏普{mean}；目标{'达到' if r.target_passed else '未达到'}。")
    lines += ["", "## 策略与调用", "", "先调用train仅拟合和内部选参，冻结全部四折后调用evaluate。详见项目1method/双模型留一年验证说明.md。",
              "开仓按预测次日净收益排序，方法二加微量CTA趋势分；每次最多前5且过门槛，每批最多当前总资产10%。",
              "持仓模型逐批判断继续一天的增益，预测不高于内部选定阈值则退出；独立ATR与最长交易日为风险保护。",
              "换出选股池不是平仓条件；资金足够时可买新目标，已持有股票隔日可加新批次。T+1及下一分钟执行始终生效。",
              "", "## 图与逐笔记录", ""]
    for r in table.itertuples():
        lines.append(f"- [{r.year}年{METHODS[r.method][0]}]({r.year}/{r.method}/final_report.md)：含收益图、成交、平仓片段与未平仓。")
    lines += ["", "## 审计与限制", "", "各折配置/模型哈希在测试前冻结，test_execution_audit.csv记录T+1、整手、现金、逐笔对账；未触发实盘。", "", RESEARCH_LIMITATIONS,
              "## 资料", "", "时间序列应避免未来训练冒充向前预测，见[scikit-learn官方时间序列切分说明](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。",
              "回归目标和正则化参数含义见[XGBoost官方参数说明](https://xgboost.readthedocs.io/en/stable/parameter.html)。"]
    (directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "输出文件说明.md").write_text("""# 输出文件说明

架构：公共行情缓存 -> 每折内部拟合/校准/验证 -> 冻结 -> 留出年测试。每年各方法独立初始资金，不跨年拼仓。

| 文件或目录 | 类型 | 用途及后续消费者 |
|---|---|---|
| final_report.md | 中文报告 | 四年结果、限制与每年报告入口 |
| 输出文件说明.md | 中文说明 | 本文件，各文件用途 |
| resolved_config.yaml | 冻结配置 | 合并的行情、费用、窗口、模型参数；evaluate读取 |
| research_protocol.json | 事前协议 | 代码哈希、包版本、目标定义，不含测试收益 |
| source_snapshot/ | 源码快照 | 启动时源码，供复现；路径保持原项目结构 |
| data_metadata.json | 来源审计 | 原面板与新持仓pkl路径/行数，pkl不进Git |
| folds.json / split_audit.csv | 切片审计 | 日期边界、每模型拟合年、首末信号和标签 |
| frozen_before_test.json | 冻结凭据 | 四折规则、文件哈希、测试前时间；evaluate校验 |
| test_started.json / test_completed.json | 执行标记 | 防止重复评价；失败需新目录审计，不能静默覆盖 |
| annual_test_metrics.csv | 机器表格 | year年、method方法、total_return收益、annual_return年化、sharpe夏普等 |
| four_year_summary.csv | 机器表格 | 完整四折均值与target_passed达标标志，无波动年不剔除 |
| test_execution_audit.csv | 执行审计 | T+1、整手、预算、非负现金、严格延迟与权益重建 |
| method_overlap.csv | 方法重合审计 | 同时同股票买入占比、日收益相关性；不可视为独立收益源 |
| 年份/entry_*.json | 模型 | 各内部候选开仓回归器，输出净收益基点，除10000得到收益比例 |
| 年份/exit_model.json | 模型 | 持仓继续收益回归器，输出基点；不是买入分类器 |
| 年份/entry_feature_schema.json | 输入契约 | 开仓特征列序，持仓模型不可复用此列序 |
| 年份/exit_feature_schema.json | 输入契约 | 持仓特征列序及逐列中文含义 |
| 年份/model_feature_manifest.csv | 因子说明 | 训练覆盖率、相关性与selected是否选用，附中文解释 |
| 年份/model_metadata.json | 模型说明 | 标签、窗口、参数、未来训练标志、选参规则 |
| 年份/selected_rules.json | 冻结规则 | 阈值、模型档、ATR保护、最长持仓，未通过时现金 |
| 年份/training_rule_metrics.csv | 校准报告 | 预定候选在训练内部校准片段的均值，不是外层测试 |
| 年份/validation_rule_metrics.csv | 验证报告 | 校准入围规则在内部验证片段的结果 |
| 年份/selection_segment_metrics.csv | 片段明细 | 每候选、每内部年、每阶段成绩，避免只看均值 |
| 年份/方法/ | 单策略报告 | final_report、输出说明、两种文件名收益图、净值/成交/平仓/未平仓/日终持仓/指标/策略元数据 |

单策略CSV字段含义见各目录输出文件说明。夏普使用日终净收益、含所有现金日、252交易日、无风险0；无波动CSV的sharpe=0只是兼容占位，四年均值不把它当有效夏普。
本轮不生成测试预测pkl，以免形成日常探索入口。特征缓存复用；报告CSV/图片可入Git，大型缓存不可入Git。
""", encoding="utf-8")


def run_evaluation(directory: Path) -> None:
    """对directory已冻结四折执行一次外层测试，先校验所有哈希；任何已开始的测试目录都拒绝重复运行。"""
    frozen = json.loads((directory / "frozen_before_test.json").read_text())
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    if (directory / "test_started.json").exists():
        raise ValueError("该目录已开始测试，禁止重复观察或覆盖")
    for rel, expected in frozen["source_hashes"].items():
        if calculate_sha256(ROOT / rel) != expected:
            raise ValueError(f"源码与冻结不一致：{rel}；不可沿用旧冻结文件")
    for name, key in [("resolved_config.yaml", "config_sha256"), ("research_protocol.json", "protocol_sha256")]:
        if calculate_sha256(directory / name) != frozen[key]:
            raise ValueError(f"冻结配置/协议被修改：{name}")
    for year, item in frozen["folds"].items():
        for name, digest in item["hashes"].items():
            if calculate_sha256(directory / year / name) != digest:
                raise ValueError(f"模型/特征/规则哈希不符：{year}/{name}")
    panel, market, matrices, samples, manifest, cache = load_research_data(cfg)
    expected = json.loads((directory / "data_metadata.json").read_text())
    if cache != expected["exit_cache"] or manifest["cache_path"] != expected["annual_cache"]:
        raise ValueError("数据指纹发生变化，拒绝使用旧模型评价")
    del samples
    write_json(directory / "test_started.json", {"created": datetime.now().isoformat(), "frozen_sha256": calculate_sha256(directory / "frozen_before_test.json")})
    rows, audits, overlaps = [], [], []
    for year, item in frozen["folds"].items():
        path, fold = directory / year, item["fold"]
        features = json.loads((path / "entry_feature_schema.json").read_text())
        exit_model = XGBRegressor()
        exit_model.load_model(path / "exit_model.json")
        frame = panel.loc[(panel.datetime >= pd.Timestamp(fold["test_start"])) & (panel.datetime < pd.Timestamp(fold["test_end"]))]
        results = {}
        for method, rule in item["rules"].items():
            model = XGBRegressor()
            model.load_model(path / f"entry_{rule['profile']}.json")
            scores = generate_scores(model, frame, features, cfg)
            policy = LearnedExitPolicy(exit_model, matrices, market, rule["exit_threshold"])
            result = run_lot_portfolio(scores, market, pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]), METHODS[method][1], rule, cfg, policy)
            checks = validate_result(result, cfg)
            events, opened = result["events"], result["open"]
            unrealized = float((opened.shares * (opened.mark_price - opened.buy_price) - opened.buy_fee).sum()) if len(opened) else 0.0
            error = abs(float(result["curve"].equity.iloc[-1]) - (cfg["initial_cash"] + float(result["trips"].net_pnl.sum()) + unrealized))
            if error > 1e-4:
                raise AssertionError(f"逐笔权益对账失败{error}")
            lag_ok = bool(((events.datetime - events.signal_datetime) == pd.Timedelta(minutes=1)).all())
            if not lag_ok:
                raise AssertionError("并非下一分钟成交")
            result.update(report_note=RESEARCH_NOTE, limitations=RESEARCH_LIMITATIONS)
            save_result(path / method, result, f"{year}年{METHODS[method][0]}", rule, cfg)
            rows.append({"year": int(year), "method": method, "future_trained": fold["future_trained"], **result["metrics"]})
            audits.append({"year": int(year), "method": method, **checks, "next_minute_exact": lag_ok, "ledger_error": error,
                           "model_exit_fills": int(events.reason.str.startswith("持仓模型").sum())})
            results[method] = result
            print(f"留出年{year}/{method}已评价：夏普{result['metrics']['sharpe']:.4f}，收益{result['metrics']['total_return']:.4%}，买入{result['metrics']['buy_count']}次", flush=True)
        sets = {m: set(zip(r["events"].loc[r["events"].side == "buy", "datetime"], r["events"].loc[r["events"].side == "buy", "symbol"])) for m, r in results.items()}
        returns = {m: r["curve"].groupby(r["curve"].datetime.dt.normalize()).equity.last().pct_change(fill_method=None) for m, r in results.items()}
        correlation = returns["method1"].corr(returns["method2"]) if all(s.std() > 0 for s in returns.values()) else np.nan
        common = len(sets["method1"] & sets["method2"])
        overlaps.append({"year": int(year), "method1_buy_overlap": common / len(sets["method1"]) if sets["method1"] else np.nan,
                         "method2_buy_overlap": common / len(sets["method2"]) if sets["method2"] else np.nan, "daily_return_correlation": correlation})
    table = pd.DataFrame(rows)
    summary = calculate_four_year_summary(table, cfg)
    table.to_csv(directory / "annual_test_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(directory / "four_year_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(directory / "test_execution_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlaps).to_csv(directory / "method_overlap.csv", index=False, encoding="utf-8-sig")
    write_root_report(directory, table, summary)
    write_json(directory / "test_completed.json", {"created": datetime.now().isoformat(), "target_passed": bool(summary.target_passed.all())})
    print(f"报告已完成：{directory / 'final_report.md'}", flush=True)


def run_cli() -> None:
    """解析命令行，分离train与evaluate；后者只使用run-dir内冻结配置，不能临时改费率或策略。"""
    parser = argparse.ArgumentParser(description="四组留一年双XGBoost研究（回顾性，不是未来实盘证明）")
    parser.add_argument("--config", type=Path, default=ROOT / "config/learned_exit_year_validation.yaml")
    parser.add_argument("--phase", choices=["train", "evaluate"], default="train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.phase == "evaluate":
        if args.run_dir is None:
            parser.error("evaluate必须指定--run-dir")
        run_evaluation(args.run_dir.resolve())
    else:
        run_training(load_config(args))


if __name__ == "__main__":
    run_cli()
