"""按本折核心年度内OOF周选股队列训练库存T，再验证同一冻结周底仓的净增量。

从quant根目录运行，默认读取v6 YAML。复用v5时钟、成本、27候选和全折先冻结
协议，只改变T训练队列和已知周预测上下文；输出独立模型/报告，不覆盖旧实验。
"""

from __future__ import annotations

from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.weekly_t_cohort import attach_weekly_context, build_core_t_cohort, normalize_source_year_trace
from quant_backtest.weekly_t_data import load_weekly_t_data
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_features import add_session_context_features, add_weekly_execution_features
from quant_backtest.weekly_t_horizon_data import load_horizon_data
from quant_backtest.weekly_t_models import fit_t_model, predict_t, select_training_rows
from quant_backtest.weekly_t_provenance import add_feature_provenance
from quant_backtest.weekly_t_recovery_data import load_recovery_data
from quant_backtest.weekly_t_short_horizon import (
    attach_horizon_labels, candidate_passes_support, create_short_decisions, fit_feature_support,
)
from quant_backtest.weekly_t_validation import run_candidate_evaluations
from scripts.run_weekly_t_cross_validation import save_result, select_execution_data, select_weekly_universe, write_frame
from scripts.run_weekly_t_short_horizon import (
    load_configuration, read_frozen_selections, snapshot_sources, build_year_stores, write_final_report,
)


def snapshot_cohort_sources(directory: Path) -> None:
    """复用共同依赖快照并追加实际v6入口与OOF模块，保持v5旧快照不变。"""
    snapshot_sources(directory)
    hashes = json.loads((directory / "source_hashes.json").read_text())
    for path in (Path(__file__), PROJECT_ROOT / "quant_backtest/weekly_t_cohort.py"):
        shutil.copyfile(path, directory / "source_snapshot" / path.name)
        hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(directory / "source_hashes.json", hashes)


def attach_frozen_context(intraday: pd.DataFrame, predictions: pd.DataFrame, universe: pd.DataFrame,
                           cfg: dict, model_dir: Path, expected_years: list) -> pd.DataFrame:
    """只接本折已冻结周模型上下文，验证训练年与模型身份，追踪真实周量价来源。"""
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    if sorted(metadata["train_years"]) != sorted(expected_years):
        raise ValueError("冻结周上下文模型的训练年与本折不符")
    available_years = set(pd.to_datetime(universe.trade_week).dt.year)
    if "entry_date" in universe:
        available_years |= set(pd.to_datetime(universe.entry_date).dt.year.dropna())
    expected_excluded = sorted(available_years.difference(expected_years))
    if sorted(metadata.get("excluded_years", [])) != expected_excluded:
        raise ValueError("冻结周上下文模型的排除年与本折不符")
    digest = hashlib.sha256((model_dir / "model_bundle.joblib").read_bytes()).hexdigest()
    expected_trace = normalize_source_year_trace(expected_years)
    if ("source_model_training_years" in predictions
            and not predictions.source_model_training_years.map(normalize_source_year_trace).eq(expected_trace).all()):
        raise ValueError("预测表的模型训练年与实际冻结模型冲突")
    if "source_model_sha256" in predictions and not predictions.source_model_sha256.eq(digest).all():
        raise ValueError("预测表的模型SHA与实际冻结模型冲突")
    context_cfg = {**cfg, "weekly_context_model_training_years": sorted(expected_years),
                   "weekly_context_model_sha256": digest}
    return attach_weekly_context(intraday, predictions, universe, context_cfg)


def append_cohort_report(directory: Path) -> None:
    """追加本次实际队列与模型上下文的审计说明，不改任何数值输出。"""
    report = directory / "final_report.md"
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n## v6：核心年度内OOF周选股训练队列\n\n"
                     "本轮与v5保留相同费用、27候选、库存比例、恢复时钟及选参门槛。T模型仅在核心年内OOF周top100的周中信号上拟合，"
                     "每个核心标签年的周selector只用另一个核心年训练，排除自身标签年、内部验证年及外层留出年。"
                     "选定股票的周预测排名/分数组成新增f_*上下文；训练来源时间取旧盘中来源和周预测输入四周来源的最早值，"
                     "周模型的参数训练年份、模型SHA另有明确记录。此为回顾性交叉拟合特征，不把未来年度参数称为历史当时可用。\n\n"
                     "内外层选股名单仍精确复用本折v4文件，接入的也是该折对应周模型的已知预测。"
                     "year_*/oof_weekly_cohort保存各核心标签年的selector、预测、队列和来源；"
                     "t_training_cohort_manifest.json记录训练队列身份。新的股票队列与上下文是对训练内分布诊断的研究假设，"
                     "此前历史已经查看，不能视为全新测试，也不预设能取得净利润。\n")
    with (directory / "输出文件说明.md").open("a", encoding="utf-8") as stream:
        stream.write("\n新增oof_weekly_cohort/为本折核心标签年对应的OOF周模型、预测和队列；"
                     "t_training_cohort_manifest.json为队列和模型来源。v6入口与cohort源码另存source_snapshot。\n")


def train_all_rules(universe: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                    labels: pd.DataFrame, stores: dict, cfg: dict, directory: Path) -> tuple[list, list]:
    """先完成四折全部内部训练和27候选完整成交，任何外层收益都尚未计算。"""
    all_years = sorted(set(daily.date.dt.year))
    source = Path(cfg["frozen_selection_report_dir"])
    symbols = set(daily.symbol)
    folds, validation_rows = [], []
    for year in cfg["test_years"]:
        train_years = [int(y) for y in all_years if y != year]
        inner_year = max(train_years)
        core_years = [y for y in train_years if y != inner_year]
        fold = {"test_year": year, "train_years": train_years, "inner_year": inner_year,
                "core_years": core_years, "forward_only": max(train_years) < year,
                "outer_t_refit": False, "t_fit_scope": "core_years拟合，inner_year选择，outer保持同一权重"}
        out = directory / f"year_{year}"
        out.mkdir()
        write_json(out / "split.json", fold)
        training, cohort_metadata = build_core_t_cohort(
            universe, intraday, core_years, [year, inner_year], cfg, out / "oof_weekly_cohort")
        write_json(out / "t_training_cohort_manifest.json", cohort_metadata)
        selected = read_frozen_selections(source / f"year_{year}/validation_weekly_selections.csv", symbols, cfg)
        write_frame(selected, out / "validation_weekly_selections.csv")
        inner_daily, inner_intraday = select_execution_data(selected, daily, intraday)
        inner_intraday = attach_frozen_context(inner_intraday, selected, universe, cfg,
                                                source / f"year_{year}/inner_weekly_model", core_years)
        tasks, mappings, seen = [], [], set()
        supports = {}
        print(f"[{year}] 核心训练{core_years}，内部选择{inner_year}；冻结周股池", flush=True)
        for horizon in cfg["horizons"]:
            horizon_training = attach_horizon_labels(training, labels, horizon)
            core, _, _ = select_training_rows(horizon_training, core_years, cfg, "intraday", [year, inner_year])
            support = fit_feature_support(core, cfg)
            del core
            supports[int(horizon)] = support
            write_json(out / f"support_h{horizon}.json", support)
            for profile in cfg["t_model_profiles"]:
                model_cfg = {**cfg, "t_model_profile": profile, "t_label_horizon_minutes": horizon}
                model_name = f"inner_t_h{horizon}_{profile}"
                bundle = fit_t_model(horizon_training, core_years, model_cfg, out / model_name, [year, inner_year])
                predictions = predict_t(bundle, inner_intraday, model_cfg)
                write_frame(predictions[["datetime", "symbol", "prediction"]], out / f"validation_predictions_h{horizon}_{profile}.csv")
                for margin in cfg["short_t_net_margins"]:
                    rule = {"enabled": True, "horizon_minutes": horizon, "profile": profile,
                            "net_margin": float(margin), "fraction": cfg["short_t_fraction"], "direction": "both"}
                    decisions = create_short_decisions(predictions, rule, cfg, support)
                    index = len(mappings)
                    signature = hashlib.sha256(pd.util.hash_pandas_object(decisions, index=False).to_numpy().tobytes()).hexdigest()
                    mappings.append((index, signature, rule, model_name))
                    if signature not in seen:
                        tasks.append({"candidate_index": index, "decision_sha256": signature, "decisions": decisions})
                        seen.add(signature)
                del bundle, predictions
            del horizon_training
            gc.collect()
        evaluated = run_candidate_evaluations(tasks, selected, inner_daily, inner_intraday, cfg,
                                               out / "validation_candidate_audits", stores[inner_year])
        rows, eligible = [], []
        for index, signature, rule, model_name in mappings:
            evaluation = evaluated[signature]
            passed, reasons = candidate_passes_support(evaluation, cfg)
            row = {"test_year": year, "inner_year": inner_year, "candidate_index": index, **rule,
                   **evaluation["metrics"], **evaluation["support"],
                   "incremental_net_pnl": evaluation["incremental_net_pnl"], "strict_eligible": passed,
                   "rejection_reasons": "|".join(reasons), "decision_sha256": signature,
                   "same_decisions_as_candidate": evaluation["original_index"], "selected": False}
            rows.append(row)
            if passed:
                eligible.append((evaluation["incremental_net_pnl"], index, rule, model_name))
            print(f"[{year}] h{rule['horizon_minutes']}/{rule['profile']}/{rule['net_margin']:.4f}: "
                  f"增量{evaluation['incremental_net_pnl']:.2f}，完整{evaluation['metrics']['t_pair_count']}组，"
                  f"活跃{evaluation['support']['active_days']}日/{evaluation['support']['active_weeks']}周，合格{passed}", flush=True)
        if eligible:
            _, chosen, rule, model_name = max(eligible, key=lambda item: (item[0], -item[1]))
            rows[chosen]["selected"] = True
        else:
            _, _, rule, model_name = mappings[0]
            rule = {**rule, "enabled": False, "training_failure": "27组候选无同时通过收益/周CI/样本支持/库存要求者"}
        write_frame(pd.DataFrame(rows), out / "validation_rule_metrics.csv")
        validation_rows.extend(rows)
        shutil.copytree(out / model_name, out / "t_model")
        selected_model_sha = hashlib.sha256((out / "t_model/model_bundle.joblib").read_bytes()).hexdigest()
        write_json(out / "selected_rule.json", {**rule, "fold": fold, "source_model": model_name,
                                                "model_bundle_sha256": selected_model_sha, "frozen_before_any_outer_test": True})
        write_json(out / "selected_support.json", supports[int(rule["horizon_minutes"])])
        folds.append(fold)
        print(f"[{year}] 内部规则已冻结：{rule}", flush=True)
        del training
        gc.collect()
    write_frame(pd.DataFrame(validation_rows), directory / "validation_rule_metrics.csv")
    write_json(directory / "all_rules_frozen_before_outer_test.json", {
        "created_at": datetime.now().isoformat(), "test_years": cfg["test_years"],
        "selected_rule_sha256": {str(f["test_year"]): hashlib.sha256(
            (directory / f"year_{f['test_year']}/selected_rule.json").read_bytes()).hexdigest() for f in folds}})
    return folds, validation_rows


def evaluate_all_folds(folds: list, universe: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                       stores: dict, cfg: dict, directory: Path) -> tuple[pd.DataFrame, list]:
    """仅在所有规则冻结后计算留出收益；核对持有基准与原冻结实验逐日一致。"""
    source = Path(cfg["frozen_selection_report_dir"])
    summaries, results = [], []
    for fold in folds:
        year = fold["test_year"]
        out = directory / f"year_{year}"
        rule = json.loads((out / "selected_rule.json").read_text())
        support = json.loads((out / "selected_support.json").read_text())
        selected = read_frozen_selections(source / f"year_{year}/weekly_selections.csv", set(daily.symbol), cfg)
        write_frame(selected, out / "weekly_selections.csv")
        for filename in ("weekly_predictions.csv", "weekly_prediction_metrics.csv"):
            shutil.copyfile(source / f"year_{year}" / filename, out / filename)
        outer_daily, outer_intraday = select_execution_data(selected, daily, intraday)
        outer_intraday = attach_frozen_context(outer_intraday, selected, universe, cfg,
                                                source / f"year_{year}/weekly_model", fold["train_years"])
        bundle = joblib.load(out / "t_model/model_bundle.joblib")
        predictions = predict_t(bundle, outer_intraday, cfg)
        decisions = create_short_decisions(predictions, rule, cfg, support)
        write_frame(predictions[["datetime", "symbol", "prediction"]], out / "t_predictions.csv")
        write_frame(decisions, out / "t_decisions.csv")
        result = run_weekly_portfolio(selected, outer_daily, outer_intraday, decisions, cfg, recovery_bars=stores[year])
        metrics = save_result(result, out, cfg)
        if not cfg["smoke_weeks"] and not cfg["max_symbols"]:
            frozen_curve = pd.read_csv(source / f"year_{year}/test_equity_curve.csv", parse_dates=["datetime"])
            curve = result["equity_curve"]
            if not curve.datetime.reset_index(drop=True).equals(frozen_curve.datetime.reset_index(drop=True)):
                raise AssertionError("新旧持有基准日期不一致")
            error = float(np.max(np.abs(curve.baseline_equity.to_numpy() - frozen_curve.baseline_equity.to_numpy())))
            if error > 1e-6:
                raise AssertionError(f"冻结持有基准改变：最大误差{error}")
            write_json(out / "frozen_baseline_audit.json", {"max_equity_error_rmb": error, "same_dates": True})
        summaries.append({"test_year": year, "inner_year": fold["inner_year"], **metrics, "t_enabled": rule["enabled"]})
        results.append((year, result))
        print(f"[{year}] 留出年持有{metrics['baseline_total_return']:.3%}，加T{metrics['total_return']:.3%}，"
              f"净增量{metrics['t_added_equity']:.2f}", flush=True)
    return pd.DataFrame(summaries), results


def main() -> None:
    """创建独立输出，先写事前研究协议，构建只读报价后执行训练与统一外层评价。"""
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        sys.argv.extend(["--config", str(PROJECT_ROOT / "config/weekly_t_cross_validation_v6.yaml")])
    cfg = load_configuration()
    if cfg.get("t_training_cohort") != "core_year_oof_weekly_top100_middle_days":
        raise ValueError("v6要求核心年度内OOF周top100周中训练队列")
    directory = Path(cfg["output_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True)
    snapshot_cohort_sources(directory)
    write_json(directory / "resolved_config.json", cfg)
    write_json(directory / "research_protocol.json", {
        "created_before_this_run_outer_test": True, "previous_outer_results_seen": True,
        "research_version": cfg["research_version"], "profiles": cfg["t_model_profiles"],
        "horizons": cfg["horizons"], "net_margins": cfg["short_t_net_margins"],
        "fraction": cfg["short_t_fraction"], "direction": "both", "outer_t_refit": False,
        "scheduled_recovery_times": cfg["scheduled_recovery_times"],
        "candidate_support_requirements": cfg["candidate_support_requirements"],
        "outer_execution_order": "所有年度内部规则冻结后统一评价外层",
        "weekly_selection": "精确复用v4各折冻结内外层选股；T训练队列由本折核心年内OOF周selector产生",
        "t_training_cohort": cfg["t_training_cohort"],
        "weekly_context_lineage": "周预测的原始4周来源并入feature_source_start；参数训练年和模型SHA另行记录",
        "gain_criterion": "aggregate_net_positive_and_majority_positive_and_bootstrap_low_positive_and_inventory_legal_and_weekly_exits_complete"})
    print(f"输出目录：{directory}", flush=True)
    weekly, intraday, daily, manifest = load_weekly_t_data(cfg)
    write_json(directory / "data_manifest.json", manifest)
    tail, recovery_manifest = load_recovery_data(cfg)
    write_json(directory / "recovery_manifest.json", recovery_manifest)
    labels, early, horizon_manifest = load_horizon_data(cfg)
    write_json(directory / "horizon_manifest.json", horizon_manifest)
    write_json(directory / "frozen_selection_sources.json", horizon_manifest["frozen_selection_sources"])
    stores = build_year_stores(early, tail, cfg)
    del early, tail
    gc.collect()
    weekly = add_weekly_execution_features(weekly, daily)
    intraday = add_session_context_features(intraday)
    weekly, intraday, provenance = add_feature_provenance(weekly, intraday, daily)
    write_json(directory / "feature_provenance_manifest.json", provenance)
    universe = select_weekly_universe(weekly, cfg)
    folds, _ = train_all_rules(universe, intraday, daily, labels, stores, cfg, directory)
    write_frame(pd.DataFrame(folds), directory / "split_audit.csv")
    del labels
    gc.collect()
    summary, results = evaluate_all_folds(folds, universe, intraday, daily, stores, cfg, directory)
    write_final_report(summary, results, cfg, directory)
    append_cohort_report(directory)
    print(f"报告完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
