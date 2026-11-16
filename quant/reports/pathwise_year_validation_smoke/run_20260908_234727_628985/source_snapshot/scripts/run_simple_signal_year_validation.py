"""在quant根目录训练可解释模型并做留一年回顾研究，输入YAML及既有有界pkl。

train只使用各折核心拟合/内部校准/验证，生成模型、规则、中文报告及冻结指纹。
evaluate先校验并一次性输出留出年曲线和交易；audit仅重建拟合行与重载模型。
不覆盖旧报告，不改原始行情，不连接券商；回顾研究不能视为实盘收益承诺。
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
from quant_backtest.learned_exit_models import build_research_folds, select_window_rows, calculate_entry_target
from quant_backtest.simple_signals import (FEATURE_NAMES, METHOD_NAMES, FAMILY_NAMES, build_simple_panel,
    select_horizon_rows, select_spaced_training, select_profile_training, create_profiles,
    fit_simple_model, generate_simple_scores, calculate_gate, create_exit_policy)
from scripts.run_bounded_year_validation import load_config as load_bounded_config
from scripts.run_learned_exit_year_validation import SOURCES, calculate_sha256
from scripts.run_annual_method_validation import save_result, validate_result, LIMITATIONS
from scripts.audit_bounded_training import validate_core_sources
from scripts.audit_multiscale_frozen_training import validate_threshold_contract

EXTRA_SOURCES = [
    "quant_backtest/bounded_training.py", "quant_backtest/multiscale_training.py",
    "quant_backtest/inventory_t_data.py", "quant_backtest/exit_alignment.py",
    "quant_backtest/simple_signals.py", "scripts/run_multiscale_year_validation.py",
    "scripts/run_bounded_year_validation.py", "scripts/run_simple_signal_year_validation.py",
    "scripts/audit_bounded_training.py", "scripts/audit_multiscale_frozen_training.py",
    "tests/test_simple_signals.py",
]
NOTE = "留一年回顾研究，部分折含未来年份训练/选参；2026仅1至4月，非实盘证明。"


def load_config(args: argparse.Namespace) -> dict:
    """合并args指定继承配置，烟测只缩模型/样本规模；仍沿用原有界缓存键和费用。"""
    cfg = load_bounded_config(args)
    if args.smoke:
        cfg.update(simple_horizons=[1, 3], ridge_penalties=[.1],
                   isotonic_families=["oversold_reversal"], simple_min_training_rows=10,
                   simple_shortlist=1, simple_min_positive_segments=1)
    if not set(cfg["simple_horizons"]).issubset({1, *cfg["entry_horizons"]}):
        raise ValueError("简单模型期限缺少已有标签")
    if any(h < 1 or int(h) != h for h in cfg["simple_horizons"]):
        raise ValueError("期限必须为正整数")
    if any(p <= 0 for p in cfg["ridge_penalties"]) or cfg["isotonic_bins"] < 2:
        raise ValueError("岭惩罚必须为正，分箱至少2个")
    if not set(cfg["isotonic_families"]).issubset(FAMILY_NAMES):
        raise ValueError("未知技术族")
    cfg["report_note"] = NOTE
    cfg["report_limitations"] = "\n".join(line for line in LIMITATIONS.splitlines()
        if not any(text in line for text in ["方法二沿用方法一", "最终全量模型"])) + """
- 本轮算法已改为岭回归与技术信号保序校准，是独立研究对照，不能当作旧XGBoost/CTA模型复现或静默替换实盘。
- """ + NOTE + """
- 核心拟合、校准、验证各为非留出年的60%/20%/20%交易日片段；每段起点隔离7自然日，标签不得跨段。留出年后另隔离60交易日，逐行审计真实来源包络；核心来源不得借用上年内部验证行情。
- 固定26个输入，不进行收益择优的特征筛选。删除递推及累计因子；ATR为240条有效分钟真实波幅简单均值，另需一条前收。20/40倍分钟ATR不是20/40倍日ATR。
- 核心拟合每股每天仅一个预定钟点，每H+1交易日取样，剔除同股重叠标签；跨股票及年度轮换仍相关，不是独立同分布实验。
- 岭回归填补和标准化只拟合核心数据；保序模型仅在事前技术资格区域拟合10个等宽强度箱。模型不在内部验证后重训，避免门槛漂移。
- 标签为H个交易日同刻下一分钟交易的扣费收益，截断至正负20%。预测不是扣容量/资金约束后的实际组合收益；ATR可提前退出，待卖出可能推迟，吊顶模式也可超过预测期限。
- 当前ST/退市名称快照不是历史状态，前收和涨跌停为代理，复权/公司行动未完整处理；这些偏差及已反复研究历史仍存在。不能保证收益或据此直接实盘。
- 每个内部片段现金重启，期末持仓按市值估值、不强行平仓。年度测试也各自现金重启，四年不是拼接的连续账户。
- 外层晋级须至少一种方法四折全部内部启用。最终每方法完整四折等权平均夏普须严格大于3；空仓无波动年不能删掉，不能把内部夏普当外层成绩。
"""
    return cfg


def write_table(path: Path, rows, columns=None) -> None:
    """把rows以UTF-8签名CSV写到path，columns允许空结果仍保留列契约。"""
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def evaluate_segments(scores: pd.DataFrame, windows: list[dict], market, rule: dict, cfg: dict) -> tuple[dict, list]:
    """按windows分别模拟scores；market/rule/cfg定义执行，逐段审计，现金片段不得伪装有效夏普。"""
    details = []
    for window in windows:
        part = scores.loc[(scores.datetime >= window["start"]) & (scores.datetime < window["end"])]
        result = run_lot_portfolio(part, market, window["start"], window["end"], "prediction",
                                   rule, cfg, create_exit_policy(market, rule))
        validate_result(result, cfg)
        details.append({"segment_year": window["year"], **result["metrics"]})
    table = pd.DataFrame(details)
    all_active = bool((table.annual_volatility > 0).all())
    return {"mean_sharpe": float(table.sharpe.mean()) if all_active else np.nan,
            "worst_drawdown": float(table.max_drawdown.min()),
            "closed_lots": int(table.closed_lot_count.sum()),
            "positive_segments": int((table.total_return > 0).sum()),
            "active_segments": int((table.annual_volatility > 0).sum())}, details


def calculate_forecast_diagnostics(bundle: dict, outer: pd.DataFrame, windows: list, cfg: dict) -> list[dict]:
    """仅对outer内部windows完整标签计算预测分箱；报告标签代理而非实际组合，绝不用于外层选参。"""
    frame = select_horizon_rows(outer, windows, bundle["profile"]["horizon"])
    scores = generate_simple_scores(bundle, frame, cfg)
    keep = scores.eligible.to_numpy()
    if not keep.any():
        return []
    target = calculate_entry_target(frame, cfg)[keep] / 10000
    predictions = scores.prediction.to_numpy()[keep]
    # 固定收益区间，不读取验证/测试分位数定义分组。
    bins = [-np.inf, 0, .002, .005, .01, .02, np.inf]
    groups = pd.cut(predictions, bins=bins)
    table = pd.DataFrame({"group": groups, "prediction": predictions, "target": target,
                          "positive": target > 0})
    out = table.groupby("group", observed=True).agg(rows=("target", "size"), predicted=("prediction", "mean"),
        realized_net_label=("target", "mean"), positive_label_rate=("positive", "mean")).reset_index()
    out["group"] = out["group"].astype(str)
    return out.to_dict("records")


def select_rules(models: dict, outer: pd.DataFrame, fold: dict, market, cfg: dict, directory: Path,
                 score_generator=None, diagnostics_generator=None, exit_selector=None) -> dict:
    """仅用outer内部片段为models校准/验证，directory保存全部候选；market/cfg给执行契约。

    可选score_generator替换冻结模型推断，diagnostics_generator替换内部标签诊断，
    exit_selector(profile,cfg)只返回与该模型标签一致的退出规则；省略时保持原低维行为。
    """
    predict = score_generator or generate_simple_scores
    diagnose = diagnostics_generator or calculate_forecast_diagnostics
    frames = {role: select_window_rows(outer, fold[role], False) for role in ["calibration", "validation"]}
    candidates, calibration, validation, details, forecasts, scored = {}, [], [], [], [], {}
    for name, bundle in models.items():
        profile = bundle["profile"]
        method = profile["method"]
        scored[name] = {role: predict(bundle, frame, cfg) for role, frame in frames.items()}
        print(f"  校准：{fold['year']}/{name}", flush=True)
        forecasts.extend({"role": "calibration", "method": method, "profile": name, **row}
            for row in diagnose(bundle, outer, fold["calibration"], cfg))
        for q in cfg["gate_quantiles"]:
            gate = calculate_gate(scored[name]["calibration"], q, cfg["entry_min_net_return"])
            for exit_rule in (exit_selector(profile, cfg) if exit_selector else cfg["simple_exits"]):
                key = f"{name}_{exit_rule['name']}_q{q:g}"
                rule = {**exit_rule, "rule_id": key, "profile": name, "horizon": profile["horizon"],
                        "threshold": gate, "quantile": q, "selection": "内部校准候选"}
                candidates[method, key] = rule
                summary, segments = evaluate_segments(scored[name]["calibration"], fold["calibration"], market, rule, cfg)
                calibration.append({"method": method, **rule, **summary})
                details.extend({"role": "calibration", "method": method, "rule_id": key, **d} for d in segments)
    train = pd.DataFrame(calibration)
    winners = {}
    for method in METHOD_NAMES:
        shortlist = train.loc[(train.method == method) & (train.closed_lots >= cfg["min_closed_lots"])
            & (train.positive_segments >= cfg["simple_min_positive_segments"]) & (train.mean_sharpe > 0)]
        shortlist = shortlist.sort_values(["mean_sharpe", "worst_drawdown", "rule_id"],
            ascending=[False, False, True]).drop_duplicates("profile").head(cfg["simple_shortlist"])
        for row in shortlist.itertuples():
            rule = candidates[method, row.rule_id]
            forecasts.extend({"role": "validation", "method": method, "profile": rule["profile"], **item}
                for item in diagnose(models[rule["profile"]], outer, fold["validation"], cfg))
            print(f"  验证：{fold['year']}/{method}/{row.rule_id}", flush=True)
            summary, segments = evaluate_segments(scored[rule["profile"]]["validation"], fold["validation"], market, rule, cfg)
            validation.append({"method": method, **rule, **summary})
            details.extend({"role": "validation", "method": method, "rule_id": rule["rule_id"], **d} for d in segments)
        valid = [r for r in validation if r["method"] == method and r["mean_sharpe"] > 0
            and r["closed_lots"] >= cfg["min_closed_lots"]
            and r["positive_segments"] >= cfg["simple_min_positive_segments"]]
        if valid:
            chosen = sorted(valid, key=lambda r: (-r["mean_sharpe"], -r["worst_drawdown"], r["rule_id"]))[0]
            winners[method] = {**candidates[method, chosen["rule_id"]], "selection": "核心拟合后，内部校准和验证全部片段有波动且平均夏普为正，交易数和正收益片段数达标"}
        else:
            matching = sorted(n for n, b in models.items() if b["profile"]["method"] == method)
            if not matching:
                raise ValueError(f"{method}无可用模型，不能悄悄跳过")
            winners[method] = {**cfg["simple_exits"][0], "profile": matching[0],
                "horizon": models[matching[0]]["profile"]["horizon"], "threshold": 1e9, "quantile": None,
                "rule_id": "cash_validation_failed", "selection": "内部未达事前启用标准，现金保护"}
    write_table(directory / "training_rule_metrics.csv", calibration)
    write_table(directory / "validation_rule_metrics.csv", validation, train.columns)
    write_table(directory / "selection_segment_metrics.csv", details)
    write_table(directory / "internal_forecast_diagnostics.csv", forecasts)
    validate_threshold_contract(train, pd.DataFrame(validation, columns=train.columns), winners)
    return winners


def write_report(directory: Path, cfg: dict, summary: list, promoted: bool) -> None:
    """在directory生成训练中文说明；cfg给限制，summary为内部结果，promoted只是外层必要条件。"""
    table = pd.DataFrame(summary)
    text = "# 低维与技术信号模型：内部训练报告\n\n"
    text += "本报告没有外层测试收益，不能据此宣称四年平均夏普超过3。\n\n"
    text += table.to_markdown(index=False) + "\n\n"
    text += f"至少一种方法四折全部内部启用：{'是，允许单次外层评估' if promoted else '否，正式外层禁止运行'}。\n\n"
    text += """## 开平仓规则

- 方法一用固定26列岭回归预测1/3/5交易日净收益；方法二把超跌反转、趋势回调或趋势突破强度校准为净收益。方法名称仅是本轮分组，不替换历史实盘模型。
- 每24交易分钟检查当时周池，分数超过内部冻结门槛才可买入前5。每笔最多信号时总资产10%，不足现金或不满足成交约束则取消/缩量；不会为了凑足5只而放宽门槛。
- 同股同日不重复买，隔日可新建独立批次。离开周池或目标前5不卖；批次按各自入场价、最高价、ATR及到期钟点退出。
- 比较两种退出：初始20倍、吊顶40倍分钟ATR加预测期限到期；或相同ATR加20交易日最长持仓。满足T+1后下一分钟尝试执行；跌停、缺报价、容量不足可推迟或部分成交。

## 研究依据

岭回归用于检验较低复杂度和收缩后的收益信号，保序回归只校准事前确定的技术方向，不把“单调”假设当盈利事实。
实现依据：[scikit-learn岭回归](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)、
[保序回归](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)。
没有用旧外层预测/逐笔盈亏来挑技术族或阈值。

## 指标解释

内部均夏普对每个非留出年的验证片段分别计算，再等权平均；某片段无波动则均值未定义，不能跳掉。
正收益片段数只计净值盈利片段，完全平仓批次数不含期末未平仓批次。
internal_forecast_diagnostics.csv的实际值是完整持有期限的扣费标签代理，不是资金、T+1待卖和容量约束后的组合收益。

## 限制

"""
    text += cfg["report_limitations"]
    (directory / "training_report.md").write_text(text, encoding="utf-8")
    (directory / "final_report.md").write_text(text, encoding="utf-8")
    (directory / "输出文件说明.md").write_text("""# 文件角色

| 文件/目录 | 角色与含义 |
|---|---|
| final_report.md | 当前中文结论；外层完成后更新，训练原文保留 |
| training_report.md | 冻结训练阶段中文结论，不把验证当测试 |
| resolved_config.yaml | 本轮完整配置，含继承费用、窗口和新网格 |
| research_protocol.json | 运行前的候选、晋级及验收规则、源码和软件版本 |
| folds.json | 各留出年及非留出年的核心拟合/校准/验证边界 |
| data_metadata.json | 复用的年度及有界缓存位置和指纹；大pkl不复制 |
| source_snapshot/ | 本轮源码快照，不是可以脱离环境直接运行的安装包 |
| 年份/model_*.joblib | 包含冻结模型、预处理、特征及期限；只可加载可信本地产物 |
| 年份/model_*.json | 该模型类型、参数、核心样本数与净收益基点单位 |
| 年份/explanation_*.csv | 岭回归训练填补/标准化/系数，或技术强度分箱与保序校准值 |
| 年份/model_feature_manifest.csv | 固定输入代码及中文含义，非测试收益筛选名单 |
| 年份/model_unavailable.csv | 核心样本不足而未拟合的候选及原因；不是按测试剔除 |
| 年份/training_rule_metrics.csv | 全部校准候选，mean_sharpe内部平均夏普、closed_lots完全平仓批次、positive_segments正收益片段数 |
| 年份/validation_rule_metrics.csv | 仅按校准选出的不同模型短名单，门槛原样冻结到验证 |
| 年份/selection_segment_metrics.csv | 校准/验证各年的明细；不是外层测试 |
| 年份/internal_forecast_diagnostics.csv | 内部预测收益固定区间、样本数和扣费标签均值；非实际组合回报 |
| 年份/frozen_rules.json | 分方法保存的最终开仓门槛、模型、到期及ATR退出规则；现金保护不会跳过年度 |
| split_audit.csv | 每模型实际拟合年份、样本数、最早来源、首末信号和标签 |
| internal_summary.csv | 八组内部启用状态和选中验证指标 |
| frozen_manifest.json | 全部模型/规则/选参表/配置/协议/源码哈希及冻结时刻 |
| training_audit.json | audit阶段的真实拟合行重建、标签来源、阈值和模型重载审计结果 |
| test_started.json | 仅允许正式晋级或烟测后创建的一次性外层执行标记 |
| test_year_metrics.csv / test_summary.csv | 仅外层执行后产生；八组年度指标及完整四年均值 |
| 年份/method1或method2/ | 仅外层执行后产生，标准收益图、估值、成交、配对交易、期末持仓及各文件中文说明 |

未晋级时不生成空壳测试图，不复制旧报告或烟测图片。CSV的profile是模型配置，threshold是预测净收益门槛，
quantile是内部校准分位数，horizon是交易日期限，initial/trailing是分钟ATR倍数，max_days是时间退出上限。
""", encoding="utf-8")


def run_training(cfg: dict) -> Path:
    """按cfg创建新目录、先写协议再拟合全部年度，冻结所有内部产物；不产生外层预测。"""
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    hashes = {}
    for relative in dict.fromkeys(SOURCES + EXTRA_SOURCES):
        dest = directory / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        hashes[relative] = calculate_sha256(dest)
    write_json(directory / "research_protocol.json", {
        "created": datetime.now().isoformat(), "note": NOTE, "source_hashes": hashes,
        "config_sha256": calculate_sha256(directory / "resolved_config.yaml"),
        "profiles": create_profiles(cfg), "feature_names": FEATURE_NAMES,
        "training_sampling": "每年交易日序号模H+1为0；股/自然日稳定键哈希选10个可交易钟之一；同股标签不得重叠",
        "shortlist": f"校准全部片段有波动、均夏普>0、足够交易和正收益片段；按夏普/回撤/规则名排序，去重模型后取前{cfg['simple_shortlist']}个",
        "promotion": "至少一种方法四折全部内部启用才能正式外层；烟测仅检查输出",
        "criterion": "每方法完整四折等权平均夏普严格>3，存在现金无波动年则未定义；不能跳年或拼方法",
        "test_predictions_generated": False,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]}})
    print(f"低维训练目录：{directory}", flush=True)
    panel, market, matrices, samples, manifest, cache = load_bounded_data(cfg)
    panel = build_simple_panel(panel, market, matrices)
    del matrices, samples
    write_json(directory / "data_metadata.json", {"annual_cache": manifest["cache_path"], "bounded_cache": cache,
        "rows": len(panel), "symbols": len(market.symbols), "days": len(set(market.days))})
    folds = build_research_folds(market.times, cfg)
    write_json(directory / "folds.json", folds)
    split_rows, summary, active = [], [], {}
    for fold in folds:
        year = fold["year"]
        path = directory / str(year)
        path.mkdir()
        write_table(path / "model_feature_manifest.csv", [{"特征代码": c, "中文含义": v} for c, v in FEATURE_NAMES.items()])
        outer = select_bounded_rows(panel, market.times, year, cfg["feature_embargo_trading_days"])
        outer = outer.loc[outer.cross_section_count >= cfg["min_cross_section"]]
        fit_outer = outer.loc[outer.fit_source_allowed]
        models, unavailable = {}, []
        for profile in create_profiles(cfg):
            train = select_horizon_rows(fit_outer, fold["fit"], profile["horizon"])
            train = select_spaced_training(train, market.times, profile["horizon"])
            train = select_profile_training(train, profile)
            name = profile["name"]
            if len(train) < cfg["simple_min_training_rows"]:
                unavailable.append({"profile": name, "rows": len(train), "reason": "核心训练样本不足，不拟合"})
                continue
            validate_core_sources(train, year)
            bundle, details = fit_simple_model(train, profile, cfg)
            models[name] = bundle
            joblib.dump(bundle, path / f"model_{name}.joblib", compress=3)
            write_json(path / f"model_{name}.json", {k: v for k, v in bundle.items() if k != "model"})
            details.to_csv(path / f"explanation_{name}.csv", index=False, encoding="utf-8-sig")
            for y, group in train.groupby(train.datetime.dt.year):
                split_rows.append({"test_year": year, "profile": name, "fit_year": int(y), "rows": len(group),
                    "first_source": group.feature_source_start.min(), "first_signal": group.datetime.min(),
                    "last_signal": group.datetime.max(), "last_label": group.label_end.max()})
            print(f"{year}/{name}：核心拟合{len(train):,}行", flush=True)
        write_table(path / "model_unavailable.csv", unavailable, ["profile", "rows", "reason"])
        rules = select_rules(models, outer, fold, market, cfg, path)
        write_json(path / "frozen_rules.json", rules)
        val = pd.read_csv(path / "validation_rule_metrics.csv", float_precision="round_trip")
        for method, rule in rules.items():
            enabled = rule["threshold"] < 1e9
            active[year, method] = enabled
            row = val.loc[(val.method == method) & (val.rule_id == rule["rule_id"])]
            summary.append({"留出年": year, "方法": METHOD_NAMES[method], "内部启用": enabled,
                "选中规则": rule["rule_id"], "内部验证均夏普": float(row.mean_sharpe.iloc[0]) if len(row) else None,
                "验证完全平仓批次": int(row.closed_lots.iloc[0]) if len(row) else 0,
                "实际拟合含2026": any(r["test_year"] == year and r["profile"] == rule["profile"] and r["fit_year"] == 2026 for r in split_rows)})
        print(f"{year}折冻结：{ {m: r['rule_id'] for m, r in rules.items()} }", flush=True)
    promoted = any(all(active[f["year"], method] for f in folds) for method in METHOD_NAMES)
    write_table(directory / "split_audit.csv", split_rows)
    write_table(directory / "internal_summary.csv", summary)
    write_report(directory, cfg, summary, promoted)
    protected = {str(p.relative_to(directory)): calculate_sha256(p) for p in directory.rglob("*") if p.is_file()}
    write_json(directory / "frozen_manifest.json", {"frozen_at": datetime.now().isoformat(),
        "promoted": promoted, "files": protected})
    print(f"全部训练冻结；正式外层晋级={promoted}：{directory}", flush=True)
    return directory


def validate_frozen(directory: Path) -> tuple[dict, dict, list]:
    """校验directory可信本地实验全部冻结文件及当前源码后读取配置；最终报告允许外层后更新。"""
    frozen = json.loads((directory / "frozen_manifest.json").read_text())
    evaluated = (directory / "test_started.json").exists()
    for relative, expected in frozen["files"].items():
        if evaluated and relative == "final_report.md":
            continue
        if calculate_sha256(directory / relative) != expected:
            raise ValueError(f"冻结文件改变：{relative}")
    protocol = json.loads((directory / "research_protocol.json").read_text())
    for relative, expected in protocol["source_hashes"].items():
        if calculate_sha256(ROOT / relative) != expected:
            raise ValueError(f"当前源码改变，须使用快照对应环境：{relative}")
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    if calculate_sha256(directory / "resolved_config.yaml") != protocol["config_sha256"]:
        raise ValueError("配置指纹不一致")
    folds = json.loads((directory / "folds.json").read_text())
    for fold in folds:
        fold["test_start"], fold["test_end"] = pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
        for role in ["fit", "calibration", "validation"]:
            for window in fold[role]:
                window["start"], window["end"] = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
    return cfg, frozen, folds


def load_frozen_data(cfg: dict, directory: Path) -> tuple:
    """复用cfg有界pkl并核对directory记录的缓存键，返回固定低维面板和执行行情，不拟合。"""
    panel, market, matrices, samples, manifest, cache = load_bounded_data(cfg)
    metadata = json.loads((directory / "data_metadata.json").read_text())
    if cache != metadata["bounded_cache"] or manifest["cache_path"] != metadata["annual_cache"]:
        raise ValueError("本轮缓存身份变化，拒绝继续")
    panel = build_simple_panel(panel, market, matrices)
    return panel, market


def run_audit(directory: Path) -> None:
    """从directory冻结配置重新构建每个真实核心样本，验证来源/标签/重载预测/门槛；不读测试收益。"""
    cfg, frozen, folds = validate_frozen(directory)
    panel, market = load_frozen_data(cfg, directory)
    splits = pd.read_csv(directory / "split_audit.csv")
    checks = []
    for fold in folds:
        year = fold["year"]
        path = directory / str(year)
        outer = select_bounded_rows(panel, market.times, year, cfg["feature_embargo_trading_days"])
        outer = outer.loc[outer.fit_source_allowed & (outer.cross_section_count >= cfg["min_cross_section"])]
        for file in sorted(path.glob("model_*.joblib")):
            bundle = joblib.load(file)
            profile = bundle["profile"]
            train = select_horizon_rows(outer, fold["fit"], profile["horizon"])
            train = select_spaced_training(train, market.times, profile["horizon"])
            train = select_profile_training(train, profile)
            validate_core_sources(train, year)
            expected = splits.loc[(splits.test_year == year) & (splits.profile == profile["name"])]
            if len(train) != bundle["training_rows"] or len(train) != expected.rows.sum():
                raise ValueError("重建核心样本数不同")
            actual = train.groupby(train.datetime.dt.year).size().to_dict()
            if actual != dict(zip(expected.fit_year, expected.rows)):
                raise ValueError("实际拟合年份不同")
            for row in expected.itertuples():
                group = train.loc[train.datetime.dt.year == row.fit_year]
                observed = [group.feature_source_start.min(), group.datetime.min(), group.datetime.max(), group.label_end.max()]
                recorded = [row.first_source, row.first_signal, row.last_signal, row.last_label]
                if observed != [pd.Timestamp(v) for v in recorded]:
                    raise ValueError("拟合时间/来源审计不一致")
            sample = train.head(32)
            original = generate_simple_scores(bundle, sample, cfg)
            poisoned = sample.assign(forward_return=999, label_end=pd.Timestamp("2099-01-01"))
            reloaded = generate_simple_scores(joblib.load(file), poisoned, cfg)
            if not original.equals(reloaded):
                raise ValueError("模型重载或标签污染改变推断")
            checks.append({"year": year, "profile": profile["name"], "rows": len(train), "passed": True})
        validate_threshold_contract(
            pd.read_csv(path / "training_rule_metrics.csv", float_precision="round_trip"),
            pd.read_csv(path / "validation_rule_metrics.csv", float_precision="round_trip"),
            json.loads((path / "frozen_rules.json").read_text()))
    write_json(directory / "training_audit.json", {"created": datetime.now().isoformat(), "models": checks,
        "passed": True, "test_predictions_read": False, "promoted": frozen["promoted"]})
    print(f"真实样本及重载审计通过：{len(checks)}模型；{directory}", flush=True)


def run_evaluation(directory: Path) -> None:
    """只对directory已冻结且晋级的四折执行一次外层；烟测可绕过晋级但不能作为目标证明。"""
    cfg, frozen, folds = validate_frozen(directory)
    if (directory / "test_started.json").exists():
        raise ValueError("外层已启动过，禁止重复挑选")
    if not cfg["is_smoke"] and not frozen["promoted"]:
        raise ValueError("没有方法四折内部全部启用，按事前协议禁止正式外层")
    panel, market = load_frozen_data(cfg, directory)
    write_json(directory / "test_started.json", {"started": datetime.now().isoformat(), "is_smoke": cfg["is_smoke"]})
    rows = []
    for fold in folds:
        year = fold["year"]
        frame = panel.loc[(panel.datetime >= fold["test_start"]) & (panel.datetime < fold["test_end"])
                          & (panel.cross_section_count >= cfg["min_cross_section"])]
        path = directory / str(year)
        rules = json.loads((path / "frozen_rules.json").read_text())
        for method, title in METHOD_NAMES.items():
            rule = rules[method]
            model = joblib.load(path / f"model_{rule['profile']}.joblib")
            scores = generate_simple_scores(model, frame, cfg)
            result = run_lot_portfolio(scores, market, fold["test_start"], fold["test_end"], "prediction",
                                       rule, cfg, create_exit_policy(market, rule))
            checks = validate_result(result, cfg)
            result.update(report_note=NOTE, limitations=cfg["report_limitations"])
            save_result(path / method, result, f"{year}年 {'烟测：' if cfg['is_smoke'] else ''}{title}", rule, cfg)
            write_json(path / method / "execution_audit.json", checks)
            rows.append({"year": year, "method": method, **result["metrics"]})
    write_table(directory / "test_year_metrics.csv", rows)
    table = pd.DataFrame(rows)
    summary = []
    for method in METHOD_NAMES:
        part = table.loc[table.method == method]
        valid = len(part) == 4 and (part.annual_volatility > 0).all()
        mean = float(part.sharpe.mean()) if valid else None
        summary.append({"方法": METHOD_NAMES[method], "有效年数": int((part.annual_volatility > 0).sum()),
            "完整四年平均夏普": mean, "超过3": bool(valid and mean > cfg["sharpe_target"] and not cfg["is_smoke"])})
    write_table(directory / "test_summary.csv", summary)
    report = "# 可解释模型留一年回顾测试\n\n" + NOTE + "\n\n"
    if cfg["is_smoke"]:
        report += "**此为8股流程烟测，任何收益不计入目标。**\n\n"
    report += pd.DataFrame(summary).to_markdown(index=False) + "\n\n"
    report += table[["year", "method", "total_return", "sharpe", "max_drawdown", "buy_count", "closed_lot_count"]].rename(columns={
        "year": "留出年", "method": "方法代码", "total_return": "收益率", "sharpe": "夏普",
        "max_drawdown": "最大回撤", "buy_count": "买入成交次数", "closed_lot_count": "完全平仓批次"}).to_markdown(index=False)
    report += "\n\n空仓无波动的年度CSV夏普0只是占位，统计上未定义，不纳入有效年数。\n\n"
    for fold in folds:
        for method in METHOD_NAMES:
            report += f"- [{fold['year']}年{METHOD_NAMES[method]}]({fold['year']}/{method}/final_report.md)\n"
    report += "\n## 限制\n\n" + cfg["report_limitations"]
    (directory / "final_report.md").write_text(report, encoding="utf-8")
    print(f"外层完成：{directory}", flush=True)


def run_cli() -> None:
    """解析config/phase/run-dir/smoke/workers；训练默认不触碰外层，审计与评价必须指定冻结目录。"""
    parser = argparse.ArgumentParser(description="可解释模型留一年研究")
    parser.add_argument("--config", type=Path, default=ROOT / "config/simple_signal_year_validation.yaml")
    parser.add_argument("--phase", choices=["train", "audit", "evaluate"], default="train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.phase != "train":
        if args.run_dir is None:
            parser.error("audit/evaluate必须指定--run-dir")
        (run_audit if args.phase == "audit" else run_evaluation)(args.run_dir.resolve())
        return
    run_training(load_config(args))


if __name__ == "__main__":
    run_cli()
