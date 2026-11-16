"""在quant根目录研究3/5日开仓模型与独立退出策略，输入多尺度YAML及已有缓存。

train仅拟合、内部校准/验证并冻结；evaluate校验后一次性评价四个留出年。
生成新模型/缓存/中文报告，不覆盖前轮、不改行情、不连接券商；默认不评价外层测试。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.learned_exit_models import (BASE_FEATURES, STATE_FEATURES, EXIT_FEATURES,
    build_research_folds, select_window_rows, calculate_entry_target, fit_return_model)
from quant_backtest.multiscale_training import load_multiscale_panel, select_context_features, select_exit_samples
from scripts.run_learned_exit_year_validation import (SOURCES, RESEARCH_LIMITATIONS, load_research_data,
    calculate_sha256, select_rules, run_evaluation)


def load_layered_config(path: Path) -> dict:
    """递归读取path的base_config（相对项目根），由本层覆盖；不引入环境变量配置。"""
    current = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_layered_config(ROOT / current["base_config"]) if current.get("base_config") else {}
    cfg.update(current)
    return cfg


def load_config(args: argparse.Namespace) -> dict:
    """合并args指定YAML并验证原资金约束；smoke只缩规模，保存所有有效覆盖值。"""
    cfg = load_layered_config(args.config)
    for key in ["data_root", "output_dir", "cache_dir", "name_snapshot", "aux_cache_dir", "multiscale_cache_dir", "signal_cache_dir"]:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if args.workers:
        cfg["workers"] = args.workers
    if args.smoke:
        cfg.update(max_symbols=8, universe_size=8, min_cross_section=4, min_closed_lots=1,
                   max_features=16, gate_quantiles=[.5], train_shortlist=1, max_exit_training_rows=3000)
        cfg["output_dir"] += "_smoke"
        cfg["multiscale_cache_dir"] = cfg["output_dir"] + "/cache"
        cfg["model_profiles"] = [{"name": "horizon3_smoke", "horizon": 3, "n_estimators": 5, "max_depth": 2}]
        cfg["exit_model"] = {"n_estimators": 5, "max_depth": 2}
        cfg["exit_candidates"] = cfg["exit_candidates"][:2]
    cfg["is_smoke"] = args.smoke
    if cfg["sample_minutes"] != 24 or cfg["lot_size"] != 100 or cfg["min_commission"] != 0:
        raise ValueError("本轮标签/交易仅验证24分钟、100股和免5")
    if not 0 < cfg["entry_equity_fraction"] <= .1 or cfg["entry_top_n"] > 5:
        raise ValueError("必须保持前5及每批最多总资产10%")
    if not all(p["horizon"] in cfg["entry_horizons"] for p in cfg["model_profiles"]):
        raise ValueError("模型期限没有对应缓存标签")
    if cfg["feature_embargo_trading_days"] < 60:
        raise ValueError("60交易日特征至少需要隔离留出年后的60个交易日")
    cfg["report_note"] = "多尺度3/5日净收益模型；留一年回顾验证，前三折使用未来信息；非实盘业绩。"
    cfg["entry_strategy_description"] = "开仓模型比较3/5个交易日净收益目标，加入过去多尺度和周池环境；两方法最多前5过门控，每批最多总资产10%，方法二加CTA。"
    cfg["report_limitations"] = f"- 当前信号另要求非一字线且成交额至少{cfg['minimum_signal_amount']:g}元；这是当前可见资格，不保证下一分钟可成交。\n"
    cfg["report_limitations"] += "- 多日开仓目标为3或5交易日净收益，持仓模型仍预测继续一天增益；比较最早满足T+1的时间退出对照与动态退出，不是固定价位止盈。\n"
    cfg["report_limitations"] += "- 增加的多日因子由过去共同时钟收盘计算；60交易日窗口不是60分钟。当期环境仅统计当时合资格周池，不使用未来入池股票。\n"
    cfg["report_limitations"] += "- 留出年后60个交易日不进入拟合、校准或验证，避免长窗口特征借用留出年价格；因此2025留出折的2026核心拟合可能为空，但2026仍参与其他折拟合及本折后段选参。\n"
    cfg["report_limitations"] += "- future_trained记录核心模型是否拟合未来年份；future_selection记录是否使用未来年份校准/验证。2025折即使没有未来核心拟合，仍因未来选参属于回顾研究。\n"
    cfg["report_limitations"] += "- 60日隔离覆盖新增有界窗口；旧分钟因子含EMA/RSI等递推状态，长期缺行情股票的跨段记忆未逐一重置，不能声称全部特征与留出年行情严格独立。\n"
    cfg["report_limitations"] += "\n".join(s for s in RESEARCH_LIMITATIONS.splitlines()
        if not any(word in s for word in ["主模型预测次日", "未来年份训练", "前三轮的训练"]))
    return cfg


def load_multiscale_data(cfg: dict) -> tuple:
    """先复用旧因子/退出缓存，再扩展多尺度面板和虚拟入口资格；返回与冻结测试器相同的六项契约。"""
    panel, market, matrices, samples, manifest, base_cache = load_research_data(cfg)
    enriched, new_cache = load_multiscale_panel(panel, market, matrices, manifest, cfg)
    samples = select_exit_samples(samples, enriched)
    return enriched, market, matrices, samples, manifest, {"base_exit_cache": base_cache, "multiscale_cache": new_cache}


def build_horizon_training(frame: pd.DataFrame, windows: list[dict], horizon: int) -> pd.DataFrame:
    """把frame指定horizon标签映射为统一训练列，按windows剔除跨边界标签，不改变用于推断的原始frame。"""
    out = frame.assign(forward_return=frame[f"future_return_{horizon}d"], label_end=frame[f"future_end_{horizon}d"])
    out = select_window_rows(out, windows)
    return out.loc[np.isfinite(out.forward_return)].copy()


def select_outer_feature_rows(frame: pd.DataFrame, times: pd.DatetimeIndex, heldout_year: int, embargo_days: int) -> pd.DataFrame:
    """从frame排除heldout_year及之后embargo_days个times实际交易日；所有内部阶段共用，不能只隔离标签。

    当前最长动量读取60交易日前同刻，排除随后前60交易日后，第一行的最长回看始于
    非留出年。这里只过滤学习/选参行，不删除连续行情或重建特征，也不过滤该折测试输入。
    """
    if embargo_days < 60:
        raise ValueError("多尺度特征隔离不得小于60交易日")
    days = pd.DatetimeIndex(sorted(times.normalize().unique()))
    boundary = pd.Timestamp(f"{heldout_year + 1}-01-01")
    excluded_days = days[days >= boundary][:embargo_days]
    mask = (frame.datetime.dt.year != heldout_year) & ~frame.datetime.dt.normalize().isin(excluded_days)
    return frame.loc[mask].copy()


def run_training(cfg: dict, data_loader=None, row_selector=None, extra_sources: tuple = ()) -> Path:
    """按cfg进行四折内部重训；data_loader/row_selector可注入有界数据/隔离，extra_sources进入源码冻结。

    缺省保持旧多尺度行为；模型期限、环境特征和虚拟入口资格记录在新目录，不预测外层。
    """
    if cfg.get("data_contract") == "bounded_observation_v1" and (data_loader is None or row_selector is None):
        raise ValueError("有界来源配置必须通过对应数据与来源筛选入口，不能退回旧缓存")
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    sources = list(dict.fromkeys(SOURCES + ["quant_backtest/multiscale_training.py", "quant_backtest/inventory_t_data.py",
                                          "scripts/run_multiscale_year_validation.py"] + list(extra_sources)))
    hashes = {}
    for relative in sources:
        dest = directory / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        hashes[relative] = calculate_sha256(dest)
    write_json(directory / "research_protocol.json", {"created": datetime.now().isoformat(), "note": cfg["report_note"],
        "source_hashes": hashes, "config_sha256": calculate_sha256(directory / "resolved_config.yaml"),
        "candidate_basis": cfg.get("candidate_basis", "只基于原训练标签期限与持仓期限不一致、已知当前一字线资格和多尺度资产定价研究扩展；不从原年度测试选择参数"),
        "candidate_source": "https://www.nber.org/papers/w25398",
        "criterion": "每方法四个有效年度夏普等权均值>3，无波动年不能剔除；测试不参与选参",
        "test_predictions_generated": False,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "xgboost", "scikit-learn"]}})
    print(f"多尺度训练目录：{directory}", flush=True)
    panel, market, matrices, samples, manifest, cache = (data_loader or load_multiscale_data)(cfg)
    write_json(directory / "data_metadata.json", {"annual_cache": manifest["cache_path"], "exit_cache": cache,
        "panel_rows": len(panel), "symbols": len(market.symbols), "days": len(set(market.days)), "eligible_exit_samples": len(samples)})
    folds = build_research_folds(market.times, cfg)
    write_json(directory / "folds.json", folds)
    frozen, audit_rows = {}, []
    for fold in folds:
        year = fold["year"]
        path = directory / str(year)
        path.mkdir()
        outer = (row_selector or select_outer_feature_rows)(panel, market.times, year, cfg["feature_embargo_trading_days"])
        outer = outer.loc[outer.cross_section_count >= cfg["min_cross_section"]]
        fit_outer = outer.loc[outer.fit_source_allowed].copy() if "fit_source_allowed" in outer else outer
        selection = build_horizon_training(fit_outer, fold["fit"], max(cfg["entry_horizons"]))
        selection["target_rank"] = calculate_entry_target(selection, cfg) / 10000
        features, feature_manifest = select_context_features(selection, cfg)
        feature_manifest.to_csv(path / "model_feature_manifest.csv", index=False, encoding="utf-8-sig")
        write_json(path / "entry_feature_schema.json", features)
        exit_schema = {**BASE_FEATURES, **STATE_FEATURES}
        if cfg.get("atr_definition"):
            exit_schema["atr_ratio"] = cfg["atr_definition"] + "除以当前价"
            exit_schema["lot_return_atr"] = "本批次浮盈除以当前有界ATR"
        write_json(path / "exit_feature_schema.json", exit_schema)
        if "feature_source_start" in outer:
            provenance = outer.assign(signal_year=outer.datetime.dt.year).groupby("signal_year").agg(
                rows=("symbol", "size"), first_source=("feature_source_start", "min"), first_signal=("datetime", "min"), last_signal=("datetime", "max"))
            provenance.to_csv(path / "feature_source_audit.csv", encoding="utf-8-sig")
        eligible_exit = select_exit_samples(samples, fit_outer)
        exit_train = select_window_rows(eligible_exit, fold["fit"])
        if len(exit_train) > cfg["max_exit_training_rows"]:
            exit_train = exit_train.sample(cfg["max_exit_training_rows"], random_state=cfg["model"]["random_state"]).sort_index()
        if len(exit_train) < 100:
            raise ValueError("持仓拟合不足100行，拒绝跳折")
        print(f"{year}折：{len(features)}开仓特征；持仓拟合{len(exit_train):,}行", flush=True)
        exit_model = fit_return_model(exit_train, EXIT_FEATURES, exit_train.continuation_target.to_numpy(), cfg, cfg["exit_model"])
        exit_model.save_model(path / "exit_model.json")
        models, model_rows = {}, []
        for profile in cfg["model_profiles"]:
            horizon = profile["horizon"]
            train = build_horizon_training(fit_outer, fold["fit"], horizon)
            target = calculate_entry_target(train, cfg)
            if len(train) < 100:
                raise ValueError("开仓拟合不足100行，拒绝跳折")
            print(f"{year}/{profile['name']}：拟合{len(train):,}行，目标{horizon}交易日", flush=True)
            clean_profile = {k: v for k, v in profile.items() if k != "horizon"}
            model = fit_return_model(train, features, target, cfg, clean_profile)
            model.save_model(path / f"entry_{profile['name']}.json")
            models[profile["name"]] = model
            model_rows.append({"profile": profile, "rows": len(train), "first_signal": train.datetime.min(), "last_label": train.label_end.max()})
            for y, group in train.groupby(train.datetime.dt.year):
                audit_rows.append({"test_year": year, "model": profile["name"], "fit_year": int(y), "rows": len(group),
                                   "first_signal": group.datetime.min(), "last_signal": group.datetime.max(), "last_label": group.label_end.max()})
        for y, group in exit_train.groupby(exit_train.datetime.dt.year):
            audit_rows.append({"test_year": year, "model": "exit_model", "fit_year": int(y), "rows": len(group),
                               "first_signal": group.datetime.min(), "last_signal": group.datetime.max(), "last_label": group.label_end.max()})
        fitted_years = {r["fit_year"] for r in audit_rows if r["test_year"] == year}
        fold["future_trained"] = any(y > year for y in fitted_years)
        fold["future_selection"] = any(w["year"] > year for w in fold["calibration"] + fold["validation"])
        rules = select_rules(models, exit_model, features, outer, fold, market, matrices, cfg, path)
        write_json(path / "selected_rules.json", rules)
        write_json(path / "model_metadata.json", {"fold": fold, "entry_features": features, "exit_features": EXIT_FEATURES,
            "models": model_rows, "shared_entry_parameters": cfg["model"], "exit_parameters": {**cfg["model"], **cfg["exit_model"]},
            "entry_target": "profile.horizon交易日同刻下一分钟净收益；税费滑点、正负label_clip截断，以基点训练",
            "exit_target": "继续一天相对立即卖出的净所得增益基点，不重扣沉没买费",
            "rules": rules, "test_used_for_fit_or_selection": False, "refit_after_validation": False})
        frozen[str(year)] = {"fold": fold, "rules": rules, "hashes": {p.name: calculate_sha256(p) for p in path.iterdir() if p.is_file()}}
        print(f"{year}折选择完成：{json.dumps(rules, ensure_ascii=False)}", flush=True)
    pd.DataFrame(audit_rows).to_csv(directory / "split_audit.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "folds.json", folds)
    write_json(directory / "frozen_before_test.json", {"created": datetime.now().isoformat(), "folds": frozen,
        "protocol_sha256": calculate_sha256(directory / "research_protocol.json"),
        "config_sha256": calculate_sha256(directory / "resolved_config.yaml"), "source_hashes": hashes})
    lines = ["# 多尺度模型训练内部研究", "", "已冻结四折，尚未评价外层测试，不能报告测试夏普。", "", cfg["report_note"], ""]
    for year in frozen:
        lines += [f"## {year}留出折的内部验证", ""]
        table = pd.read_csv(directory / year / "validation_rule_metrics.csv")
        columns = ["method", "rule_id", "mean_sharpe", "closed_lots", "positive_segments"]
        display = table[columns].copy()
        display["method"] = display.method.map({"method1": "方法一：基础模型", "method2": "方法二：趋势叠加"})
        display.columns = ["方法", "规则代码（构成见下）", "内部片段平均夏普", "完整平仓批次", "正收益片段数"]
        lines += [display.to_markdown(index=False) if len(table) else "没有候选通过校准入围条件。", ""]
        for method, rule in frozen[year]["rules"].items():
            lines.append(f"- {method}（{'基础模型' if method == 'method1' else '趋势叠加模型'}）：{rule['selection']}，规则{rule['rule_id']}。")
    lines += ["", "## 规则代码说明", "",
              "horizon3/5（预测3/5交易日）；shallow/deeper（浅树/深树）；q（校准分位门槛）；next_day_control（最早T+1退出对照）；"
              "continuation0（继续一天预测收益不高于0退出）；continuation_negative（不高于-0.10%退出）；"
              "atr20_40/atr30_60（初始/移动保护线的分钟ATR倍数）。完整配置见resolved_config.yaml（合并配置）。",
              "", "## 限制", "", cfg["report_limitations"], "", "## 研究依据", "",
              "[Gu、Kelly与Xiu的机器学习资产定价研究](https://www.nber.org/papers/w25398)支持考察动量、流动性与波动状态，但不证明本策略在A股有效或能达到夏普3。"]
    (directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "输出文件说明.md").write_text("# 训练阶段文件说明\n\n先看final_report.md（内部验证结果），不能当成外层测试成绩。\n\n"
        "- resolved_config.yaml：多层合并配置、标签期限及费用，测试只能读取本文件。\n"
        "- research_protocol.json：候选设计、目标、源码校验、包版本和外层未使用声明。\n"
        "- source_snapshot/：冻结前源码快照；data_metadata.json：三种缓存路径与行数。\n"
        "- folds.json、split_audit.csv：日期切片、每模型拟合年份与标签终点。\n"
        "- frozen_before_test.json：四折模型/规则校验和及时间，evaluate的输入。\n"
        "- 年份/entry_*.json、exit_model.json：开仓/继续持仓模型；输出单位为基点。\n"
        "- 年份/entry_feature_schema.json、exit_feature_schema.json：模型输入列序及退出中文定义。\n"
        "- 年份/model_feature_manifest.csv：拟合阶段覆盖率、相关性、是否入模和中文因子解释。\n"
        "- 年份/model_metadata.json、selected_rules.json：模型参数、训练行数、3/5日标签期限、规则及选参原因。\n"
        "- 年份/training_rule_metrics.csv：校准候选；validation_rule_metrics.csv：内部验证入围候选；selection_segment_metrics.csv：每片段明细。\n"
        "\n尚无test_*文件或测试图。只有执行冻结evaluate后才生成真实外层曲线和逐笔成交，不能用训练图冒充。\n", encoding="utf-8")
    print(f"四折内部研究完成、已冻结，尚未测试：{directory}", flush=True)
    return directory


def run_cli() -> None:
    """解析config/phase/run-dir；evaluate消费冻结配置和扩展数据加载器，train写新目录。"""
    parser = argparse.ArgumentParser(description="多尺度开仓与独立退出四折研究")
    parser.add_argument("--config", type=Path, default=ROOT / "config/multiscale_year_validation.yaml")
    parser.add_argument("--phase", choices=["train", "evaluate"], default="train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.phase == "evaluate":
        if args.run_dir is None:
            parser.error("evaluate必须提供--run-dir")
        run_evaluation(args.run_dir.resolve(), load_multiscale_data)
    else:
        run_training(load_config(args))


if __name__ == "__main__":
    run_cli()
