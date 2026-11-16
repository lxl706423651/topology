"""从quant根目录运行冻结周top100上的短周期库存T年度交叉验证。

读取v5 YAML、只读分钟缓存及v4冻结选股；先完成所有年度的训练内部选择，再
统一评价留出年。模型只拟合核心训练年，内部年用于规则选择，不重拟合改变预测
刻度。输出独立模型、CSV、图及中文报告，旧实验与原始行情保持可追溯。
"""

from __future__ import annotations

import argparse
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.weekly_t_data import load_weekly_t_data
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_features import add_session_context_features, add_weekly_execution_features
from quant_backtest.weekly_t_horizon_data import load_horizon_data, validate_horizon_config
from quant_backtest.weekly_t_models import fit_t_model, predict_t, select_training_rows
from quant_backtest.weekly_t_provenance import add_feature_provenance
from quant_backtest.weekly_t_quote_store import build_dense_recovery_store
from quant_backtest.weekly_t_recovery_data import load_recovery_data
from quant_backtest.weekly_t_short_horizon import (
    attach_horizon_labels, candidate_passes_support, create_short_decisions, fit_feature_support,
)
from quant_backtest.weekly_t_validation import run_candidate_evaluations
from scripts.run_weekly_t_cross_validation import (
    calculate_weekly_increment, save_result, select_execution_data, select_weekly_universe, write_frame,
)


def load_configuration() -> dict:
    """YAML为唯一参数来源；CLI只覆盖资源、留出折和明确标记的小样本验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/weekly_t_cross_validation_v5.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--smoke-weeks", type=int)
    parser.add_argument("--test-years")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("workers", "max_symbols", "smoke_weeks", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    if args.test_years:
        cfg["test_years"] = list(map(int, args.test_years.split(",")))
    for key in ("data_root", "cache_dir", "output_dir", "recovery_cache_dir", "horizon_cache_dir",
                "dense_quote_cache_dir", "frozen_selection_report_dir"):
        cfg[key] = str((PROJECT_ROOT / cfg[key]).resolve())
    if cfg.get("outer_t_refit") is not False:
        raise ValueError("v5预声明保留核心训练权重，outer_t_refit必须为false")
    if not cfg.get("strict_feature_provenance") or "f_history_days" not in cfg["excluded_model_features"]:
        raise ValueError("v5要求真实历史来源隔离并排除累计history计数")
    if not 0 < cfg["short_t_fraction"] <= 1 or not 0 < cfg["max_pending_t_notional_fraction"] <= 1:
        raise ValueError("库存比例和未恢复名义额比例须在(0,1]")
    if cfg.get("scheduled_recovery_mode") != "fixed_horizon_clock_union":
        raise ValueError("v5恢复时钟必须由事前signal_times×horizons生成")
    _, _, fixed_minutes = validate_horizon_config(cfg)
    cfg["scheduled_recovery_times"] = [f"{minute // 60:02d}:{minute % 60:02d}" for minute in fixed_minutes]
    if cfg["smoke_weeks"]:
        cfg["t_model"] = {**cfg["t_model"], "n_estimators": 20, "min_child_samples": 20}
        cfg["t_max_train_rows"] = 30000
        cfg["candidate_workers"] = 2
    return cfg


def read_frozen_selections(path: Path, symbols: set, cfg: dict) -> pd.DataFrame:
    """精确读取同一外层折的冻结内/外层选股，不能借其他折训练模型的选股作标签。"""
    frame = pd.read_csv(path)
    for column in ("feature_week", "signal_datetime", "trade_week", "entry_date", "exit_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    if cfg["max_symbols"]:
        frame = frame.loc[frame.symbol.isin(symbols)].copy()
    if cfg["smoke_weeks"]:
        weeks = sorted(frame.trade_week.unique())[:cfg["smoke_weeks"]]
        frame = frame.loc[frame.trade_week.isin(weeks)].copy()
    if frame.empty or frame.duplicated(["trade_week", "symbol"]).any():
        raise ValueError(f"冻结选股为空或键不唯一：{path}")
    return frame.reset_index(drop=True)


def snapshot_sources(directory: Path) -> None:
    """执行前复制实际依赖源码并保存逐文件SHA，避免后来编辑改变该次实验解释。"""
    sources = [Path(__file__), PROJECT_ROOT / "scripts/run_weekly_t_cross_validation.py"]
    sources += [PROJECT_ROOT / "quant_backtest" / (name + ".py") for name in (
        "weekly_t_data", "weekly_t_models", "weekly_t_engine", "weekly_t_features",
        "weekly_t_horizon_data", "weekly_t_quote_store", "weekly_t_recovery_data",
        "weekly_t_provenance", "weekly_t_short_horizon", "weekly_t_validation", "inventory_t", "annual_portfolio")]
    destination = directory / "source_snapshot"
    destination.mkdir()
    hashes = {}
    for path in sources:
        shutil.copyfile(path, destination / path.name)
        hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(directory / "source_hashes.json", hashes)


def build_year_stores(early: pd.DataFrame, tail: pd.DataFrame, cfg: dict) -> dict:
    """每实际年度建立一次只读内存映射；包含该年度冻结股票并集所有真实日期。"""
    stores = {}
    early_year = early.datetime.dt.year
    tail_year = tail.datetime.dt.year
    for year in sorted(tail_year.unique()):
        early_part = early.loc[early_year.eq(year)]
        # 增量报价已按全折选股并集限定；尾盘可保留主板全池以支持遗留库存。
        bars = pd.concat([early_part, tail.loc[tail_year.eq(year)]], ignore_index=True)
        print(f"[{year}] 建立/复用只读报价索引，{len(bars):,}条", flush=True)
        stores[int(year)] = build_dense_recovery_store(bars, Path(cfg["dense_quote_cache_dir"]) / str(year))
        del bars
        gc.collect()
    return stores


def train_all_rules(universe: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                    labels: pd.DataFrame, stores: dict, cfg: dict, directory: Path) -> tuple[list, list]:
    """先完成四折全部内部训练和27候选完整成交，任何外层收益都尚未计算。"""
    training = intraday.merge(universe[["trade_week", "symbol"]], on=["trade_week", "symbol"],
                              how="inner", validate="many_to_one")
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
        selected = read_frozen_selections(source / f"year_{year}/validation_weekly_selections.csv", symbols, cfg)
        write_frame(selected, out / "validation_weekly_selections.csv")
        inner_daily, inner_intraday = select_execution_data(selected, daily, intraday)
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
    write_frame(pd.DataFrame(validation_rows), directory / "validation_rule_metrics.csv")
    write_json(directory / "all_rules_frozen_before_outer_test.json", {
        "created_at": datetime.now().isoformat(), "test_years": cfg["test_years"],
        "selected_rule_sha256": {str(f["test_year"]): hashlib.sha256(
            (directory / f"year_{f['test_year']}/selected_rule.json").read_bytes()).hexdigest() for f in folds}})
    return folds, validation_rows


def evaluate_all_folds(folds: list, intraday: pd.DataFrame, daily: pd.DataFrame,
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


def write_final_report(summary: pd.DataFrame, results: list, cfg: dict, directory: Path) -> None:
    """保存统一年度表、周块不确定区间、清晰曲线与目标是否达到的原始结论。"""
    write_frame(summary, directory / "annual_summary.csv")
    write_frame(summary, directory / "test_metrics.csv")
    failure_keys = ["inventory_mismatch_days", "recovery_failure_days", "pending_pairs", "base_plan_mismatch_count", "t1_violation_count"]
    failures = int(sum(summary.get(key, pd.Series(0)).fillna(0).sum() for key in failure_keys))
    increments = np.concatenate([calculate_weekly_increment(result["equity_curve"], cfg["initial_cash"], cfg)[0].t_increment.to_numpy()
                                 for _, result in results])
    samples = np.random.default_rng(cfg["random_state"]).choice(
        increments, size=(cfg["bootstrap_samples"], len(increments)), replace=True).sum(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    t_pass = bool(summary.t_added_equity.sum() > 0 and (summary.increment_return > 0).sum() > len(summary) / 2
                  and failures == 0 and low > 0)
    weekly_complete = bool(summary.weekly_liquidation_failure_days.sum() == 0
                           and summary.weekly_unclosed_positions.sum() == 0
                           and summary.baseline_weekly_unclosed_positions.sum() == 0)
    status = {"gain_criterion_met": t_pass and weekly_complete, "t_gain_criterion_met": t_pass,
              "weekly_execution_complete": weekly_complete, "constraint_failures": failures,
              "aggregate_increment_rmb": float(summary.t_added_equity.sum()),
              "aggregate_increment_ci_low_rmb": float(low), "aggregate_increment_ci_high_rmb": float(high),
              "positive_years": int((summary.increment_return > 0).sum())}
    write_json(directory / "goal_evaluation.json", status)
    for key, filename in (("equity_curve", "test_equity_curve.csv"), ("trade_events", "test_trade_events.csv"),
                          ("round_trips", "test_round_trips.csv"), ("open_positions", "test_open_positions.csv")):
        write_frame(pd.concat([r[key].assign(test_year=y) for y, r in results], ignore_index=True), directory / filename)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.6 * len(results)), squeeze=False)
    for ax, (year, result) in zip(axes.flat, results):
        curve = result["equity_curve"]
        ax.plot(curve.datetime, 100 * (curve.baseline_equity / cfg["initial_cash"] - 1), label="Weekly hold")
        ax.plot(curve.datetime, 100 * (curve.equity / cfg["initial_cash"] - 1), label="Weekly hold + short T", linestyle="--")
        ax.set_title(str(year)); ax.set_ylabel("Return (%)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "test_return_curve.png", dpi=150)
    plt.close(fig)
    shutil.copyfile(directory / "test_return_curve.png", directory / "测试集收益曲线.png")
    table = ["|留出年|持有净收益|加入T净收益|T净增量(元)|完整T组数|库存差异日|", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        table.append(f"|{row.test_year}|{row.baseline_total_return:.2%}|{row.total_return:.2%}|{row.t_added_equity:,.2f}|{row.t_pair_count}|{row.inventory_mismatch_days}|")
    content = f"""# 冻结周度选股与短周期库存T年度验证

明显净增益标准达到：**{status['gain_criterion_met']}**。各年独立投入{cfg['initial_cash']:,.0f}元，不能把年度收益率当连续组合相加。T净增量合计{status['aggregate_increment_rmb']:,.2f}元；按年度账户周增量汇总的95%自助区间为[{low:,.2f}, {high:,.2f}]元。

{chr(10).join(table)}

## 固定协议

周度选股精确复用v4每折内外层文件，70%下一周涨幅预测排名、30%下一周日均振幅预测排名，周首开盘买入、周末收盘退出，选100只。T比较20/40/60交易分钟、3种既有回归模型、费用后10/20/30基点边际，共27组；比例固定25%，未恢复参考名义额上限初始资金10%，计划恢复不晚于14:30，此后按计划随后五分钟、事前固定的130个盘中时钟及尾盘重试。时钟只由信号时间与候选周期推导，缺未来行情不删除首腿信号。

每折另外三年属于训练流程，其中最新年只用于内部选参，另外两年拟合T权重与输入支持域。T模型直接保留核心训练权重，避免重拟合改变阈值刻度；四折全部规则冻结后才统一计算留出收益。特征真实历史来源到标签末端不得跨入留出年及其前后7天。内部至少100完整组、20活跃日、8活跃周、20只股票，收益集中度受限、周块置信下界为正且库存/恢复/底仓计划无失败才启用T。

成本沿用项目：双边佣金各0.08%、过户各0.001%、滑点各0.05%，卖出印花税按日期0.1%/0.05%；整手100股，成交额占分钟5%，仅昨日库存可卖，零最低佣金。不通过内部门槛时保留持有基准；关闭T不算取得明显增益。

## 结果边界

包含未来年份作为训练的留一年验证是回顾性研究，2026数据仅到4月30日。此前v2/v3/v4外层历史结果已经查看，本轮是后续研究，不能称作未接触过的独立证据。原始文件并非完备历史可交易证券主表，也没有完整历史ST、精确涨跌停/盘口或复权公司行动；执行采用现有保守代理和实际分钟报价。停牌、零成交或受阻退出仍按真实约束记录，不补造当日恢复或周末成交。单分钟09:31.open作为周首开盘代理。

T约束失败计数合计{failures}；周末执行完全达成：{weekly_complete}。收益区间只是该历史研究的不确定性度量，不证明后续实盘收益。
"""
    (directory / "final_report.md").write_text(content, encoding="utf-8")
    (directory / "输出文件说明.md").write_text(
        "# 输出文件说明\n\nfinal_report.md为主报告；annual_summary.csv/test_metrics.csv为年度比较；"
        "test_equity_curve.csv、test_trade_events.csv、test_round_trips.csv、test_open_positions.csv带test_year汇总独立账户；"
        "test_return_curve.png/测试集收益曲线.png为持有及T对比图；goal_evaluation.json为统一判定。\n\n"
        "year_*保存冻结weekly_selections.csv、实际t_predictions/t_decisions、标准成交/库存/失败/周增量CSV、"
        "execution_audit.json与frozen_baseline_audit.json。validation_rule_metrics.csv含全部27组真实执行和支持门槛；"
        "validation_candidate_audits保留候选受阻及库存失败。inner_t_h*保存所有核心训练模型、原生权重、特征、"
        "年份及来源SHA；t_model精确复制被选核心模型；support_h*.json和selected_support.json记录训练支持域；"
        "selected_rule.json以及all_rules_frozen_before_outer_test.json证明本轮先冻结规则再评价外层。\n\n"
        "resolved_config.json、research_protocol.json、各data/recovery/horizon/provenance_manifest.json、"
        "source_hashes.json/source_snapshot和frozen_selection_sources.json记录参数、数据及源码身份。\n", encoding="utf-8")


def main() -> None:
    """创建独立输出，先写事前研究协议，构建只读报价后执行训练与统一外层评价。"""
    cfg = load_configuration()
    directory = Path(cfg["output_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True)
    snapshot_sources(directory)
    write_json(directory / "resolved_config.json", cfg)
    write_json(directory / "research_protocol.json", {
        "created_before_this_run_outer_test": True, "previous_outer_results_seen": True,
        "research_version": cfg["research_version"], "profiles": cfg["t_model_profiles"],
        "horizons": cfg["horizons"], "net_margins": cfg["short_t_net_margins"],
        "fraction": cfg["short_t_fraction"], "direction": "both", "outer_t_refit": False,
        "scheduled_recovery_times": cfg["scheduled_recovery_times"],
        "candidate_support_requirements": cfg["candidate_support_requirements"],
        "outer_execution_order": "所有年度内部规则冻结后统一评价外层",
        "weekly_selection": "精确复用v4各折冻结选股，T训练用当前因果周候选池，不用其他外层模型选股",
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
    summary, results = evaluate_all_folds(folds, intraday, daily, stores, cfg, directory)
    write_final_report(summary, results, cfg, directory)
    print(f"报告完成：{directory / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
