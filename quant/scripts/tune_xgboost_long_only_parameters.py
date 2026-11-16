"""验证集内调优 XGBoost 参数和 A 股 long-only 策略参数。

输入为稳定沪深 300 股票池、训练期因子清单和可复用特征面板 pkl 缓存。
脚本只用核心训练集拟合候选模型，只用训练期内部验证集选择模型参数、
top-N 规则和持仓门控阈值；测试集只在选定组合后执行一次最终评估。
输出包括候选参数表、参数敏感性分析、中文报告、最终模型、测试收益曲线
和逐笔交易记录。不会修改原始行情数据。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_xgboost_long_only_backtest as xb


TUNING_OUTPUT_DIR = PROJECT_ROOT / "reports" / "xgboost_parameter_tuning"


@dataclass(frozen=True)
class ModelCandidate:
    """单个 XGBoost 参数候选。

    name: 稳定英文标识，用于文件名和候选表。
    params: 相对基准模型的参数覆盖项。
    rationale: 中文说明，解释该候选主要检验哪个参数方向。
    """

    name: str
    params: dict[str, Any]
    rationale: str


def parse_args() -> argparse.Namespace:
    """解析调优脚本参数，字段与主回测脚本保持一致。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=xb.DATA_ROOT)
    parser.add_argument("--stable-symbols-file", type=Path, default=xb.STABLE_SYMBOLS_FILE)
    parser.add_argument("--feature-manifest", type=Path, default=xb.MODEL_FEATURE_MANIFEST)
    parser.add_argument("--feature-panel-cache-dir", type=Path, default=xb.FEATURE_PANEL_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=TUNING_OUTPUT_DIR)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--commission-rate", type=float, default=xb.COMMISSION_RATE)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--top-n-candidates", default="1,2,3")
    parser.add_argument("--top-quantile-candidates", default="0.02")
    parser.add_argument("--min-assets", type=int, default=xb.MIN_ASSETS_PER_REBALANCE)
    parser.add_argument("--min-holding-count", type=int, default=1)
    parser.add_argument("--hold-steps-candidates", default="1")
    parser.add_argument("--target-exposure-candidates", default="1.0")
    parser.add_argument("--confidence-quantile-candidates", default="none,0.85,0.9,0.92")
    parser.add_argument("--spread-quantile-candidates", default="none,0.6,0.7,0.8")
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--sharpe-target", type=float, default=2.0)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--workers", type=int, default=xb.WORKERS)
    parser.add_argument("--seed", type=int, default=xb.RANDOM_SEED)
    parser.add_argument("--candidate-profile", choices=["smoke", "standard", "extended"], default="standard")
    parser.add_argument("--no-final-test", action="store_true", help="只做验证集调优，不执行最终测试集评估。")
    parser.add_argument("--no-feature-panel-cache", action="store_true", help="禁用特征面板缓存，仅用于排查。")
    return parser.parse_args()


def create_backtest_config(args: argparse.Namespace) -> xb.ExperimentConfig:
    """把调优脚本参数转换为主回测脚本的 ExperimentConfig。"""

    namespace = argparse.Namespace(
        data_root=args.data_root,
        stable_symbols_file=args.stable_symbols_file,
        feature_manifest=args.feature_manifest,
        output_dir=args.output_dir,
        feature_panel_cache_dir=args.feature_panel_cache_dir,
        split_date=args.split_date,
        validation_start=args.validation_start,
        horizon=args.horizon,
        eval_step=args.eval_step,
        commission_rate=args.commission_rate,
        initial_cash=args.initial_cash,
        top_n_candidates=args.top_n_candidates,
        top_quantile_candidates=args.top_quantile_candidates,
        min_assets=args.min_assets,
        min_holding_count=args.min_holding_count,
        hold_steps_candidates=args.hold_steps_candidates,
        target_exposure_candidates=args.target_exposure_candidates,
        confidence_quantile_candidates=args.confidence_quantile_candidates,
        spread_quantile_candidates=args.spread_quantile_candidates,
        lot_size=args.lot_size,
        sharpe_target=args.sharpe_target,
        max_symbols=args.max_symbols,
        workers=args.workers,
        seed=args.seed,
        rebuild_feature_panel=False,
        no_feature_panel_cache=args.no_feature_panel_cache,
        build_feature_panel_only=False,
    )
    return xb.create_config(namespace)


def create_model_candidates(profile: str) -> list[ModelCandidate]:
    """返回按知识和当前基准模型构造的参数候选集。"""

    candidates = [
        ModelCandidate(
            "baseline_current",
            {},
            "当前基准参数，用作所有调优方向的对照。",
        ),
        ModelCandidate(
            "lower_lr_more_trees",
            {"learning_rate": 0.025, "n_estimators": 360},
            "降低学习率并增加树数，通常能降低单棵树噪声、提高排序稳定性，但训练更慢。",
        ),
        ModelCandidate(
            "higher_lr_fewer_trees",
            {"learning_rate": 0.06, "n_estimators": 160},
            "提高学习率并减少树数，检验当前模型是否过慢或过平滑。",
        ),
        ModelCandidate(
            "shallower_regularized",
            {"max_depth": 3, "min_child_weight": 30},
            "降低树深并提高叶子最小权重，减少分钟级噪声下的过拟合。",
        ),
        ModelCandidate(
            "deeper_more_regularized",
            {"max_depth": 5, "min_child_weight": 40, "reg_lambda": 8.0},
            "提高树深以捕捉非线性，同时加大叶子约束和 L2 正则防止过拟合。",
        ),
        ModelCandidate(
            "lower_min_child_weight",
            {"min_child_weight": 10},
            "降低叶子最小权重，让模型更容易学习小样本局部结构。",
        ),
        ModelCandidate(
            "stronger_regularization",
            {"reg_lambda": 10.0, "reg_alpha": 0.5},
            "加强 L2/L1 正则，降低高维因子噪声和共线性带来的过拟合。",
        ),
        ModelCandidate(
            "weaker_regularization",
            {"reg_lambda": 3.0, "reg_alpha": 0.0},
            "减弱正则，检验当前参数是否过度压制有效 alpha。",
        ),
        ModelCandidate(
            "lower_sampling",
            {"subsample": 0.75, "colsample_bytree": 0.75},
            "降低行采样和列采样，提高树之间差异，通常有助于泛化。",
        ),
        ModelCandidate(
            "higher_sampling",
            {"subsample": 0.95, "colsample_bytree": 0.95},
            "提高行采样和列采样，检验当前随机性是否过强。",
        ),
        ModelCandidate(
            "combined_conservative",
            {
                "learning_rate": 0.025,
                "n_estimators": 360,
                "max_depth": 3,
                "min_child_weight": 30,
                "colsample_bytree": 0.75,
                "reg_lambda": 8.0,
                "reg_alpha": 0.3,
            },
            "把低学习率、浅树、列采样和较强正则组合成保守泛化版本。",
        ),
        ModelCandidate(
            "combined_expressive",
            {
                "learning_rate": 0.03,
                "n_estimators": 300,
                "max_depth": 5,
                "min_child_weight": 20,
                "reg_lambda": 5.0,
                "reg_alpha": 0.1,
            },
            "在控制正则的前提下提高表达能力，检验更复杂非线性是否有用。",
        ),
    ]
    if profile == "smoke":
        return [candidates[0], candidates[1]]
    if profile == "standard":
        return candidates
    return candidates + [
        ModelCandidate(
            "very_low_lr_many_trees",
            {"learning_rate": 0.018, "n_estimators": 520, "max_depth": 3, "min_child_weight": 30},
            "更细学习率和更多树，用于检验排序信号是否受提升步长影响。",
        ),
        ModelCandidate(
            "very_strong_child_weight",
            {"min_child_weight": 60, "max_depth": 4, "reg_lambda": 8.0},
            "极强叶子样本约束，用于压制微结构噪声。",
        ),
    ]


def create_xgboost_model(config: xb.ExperimentConfig, candidate: ModelCandidate) -> Any:
    """基于主脚本基准模型创建一个参数候选模型。"""

    model = xb.create_xgboost_model(config)
    if candidate.params:
        model.set_params(**candidate.params)
    return model


def create_selected_rule(row: pd.Series) -> xb.PortfolioRule:
    """从验证结果表的一行恢复 PortfolioRule。"""

    return xb.PortfolioRule(
        str(row["rule_type"]),
        float(row["rule_value"]),
        int(row["hold_steps"]),
        float(row["target_exposure"]),
        None if pd.isna(row.get("confidence_quantile")) else float(row.get("confidence_quantile")),
        None if pd.isna(row.get("confidence_threshold")) else float(row.get("confidence_threshold")),
        None if pd.isna(row.get("spread_quantile")) else float(row.get("spread_quantile")),
        None if pd.isna(row.get("spread_threshold")) else float(row.get("spread_threshold")),
    )


def evaluate_model_candidate(
    candidate: ModelCandidate,
    config: xb.ExperimentConfig,
    feature_cols: list[str],
    x_core: pd.DataFrame,
    y_core: pd.Series,
    x_validation: pd.DataFrame,
    validation_eval_base: pd.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    """训练单个候选模型，并只在验证集上选择策略规则。"""

    start = perf_counter()
    model = create_xgboost_model(config, candidate)
    model.fit(x_core, y_core)
    validation_eval = validation_eval_base.copy()
    validation_eval["prediction"] = model.predict(x_validation).astype("float32")
    validation_table = xb.evaluate_rules(validation_eval, config)
    validation_table.to_csv(run_dir / f"validation_rules_{candidate.name}.csv", index=False, encoding="utf-8-sig")
    best = validation_table.iloc[0].copy()
    elapsed = perf_counter() - start
    row: dict[str, Any] = {
        "candidate": candidate.name,
        "rationale": candidate.rationale,
        "changed_params": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        "elapsed_seconds": elapsed,
        "best_rule": best["rule"],
    }
    for key in [
        "sharpe",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "total_return",
        "avg_turnover",
        "avg_gross_exposure",
        "round_trip_win_rate",
        "trade_count",
        "confidence_quantile",
        "confidence_threshold",
        "spread_quantile",
        "spread_threshold",
    ]:
        row[f"validation_{key}"] = best.get(key)
    row["model_params"] = json.dumps({**model.get_params(), **candidate.params}, ensure_ascii=False, sort_keys=True)
    return row


def write_split_audit(
    run_dir: Path,
    config: xb.ExperimentConfig,
    core_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """写出训练、验证、测试切分审计，证明调参边界没有读取测试期标签。"""

    rows = []
    for name, frame in [("核心训练集", core_train), ("内部验证集", validation), ("最终测试集", test)]:
        label_exit = pd.to_datetime(frame["label_exit_datetime"]) if "label_exit_datetime" in frame else pd.Series(dtype="datetime64[ns]")
        rows.append(
            {
                "数据段(split)": name,
                "行数(rows)": int(len(frame)),
                "股票数(symbol_count)": int(frame["symbol"].nunique()) if "symbol" in frame else 0,
                "开始时间(start_datetime)": str(frame["datetime"].min()) if not frame.empty else "",
                "结束时间(end_datetime)": str(frame["datetime"].max()) if not frame.empty else "",
                "标签退出最晚时间(max_label_exit_datetime)": str(label_exit.max()) if not label_exit.empty else "",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")

    text = f"""# 数据切分审计

- 核心训练集用于拟合候选模型：`datetime < {config.validation_start.date()}`，并且 `label_exit_datetime < {config.validation_start.date()}`。
- 内部验证集用于选择 XGBoost 参数和交易规则：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`，并且 `label_exit_datetime < {config.split_date.date()}`。
- 最终测试集只在选中模型参数和交易规则后执行一次预测与回测：`datetime >= {config.split_date.date()}`。
- 本调优脚本的候选排序字段为验证集 `validation_sharpe` 和 `validation_annual_return`，没有使用测试集指标参与选择。

对应行数见 `split_audit.csv`。
"""
    (run_dir / "数据切分审计.md").write_text(text, encoding="utf-8")


def calculate_parameter_sensitivity(summary: pd.DataFrame, candidates: list[ModelCandidate]) -> pd.DataFrame:
    """按被改动参数汇总验证集 Sharpe，用于参数重要性分析。"""

    rows: list[dict[str, Any]] = []
    candidate_map = {candidate.name: candidate for candidate in candidates}
    baseline = summary.loc[summary["candidate"] == "baseline_current"]
    baseline_sharpe = float(baseline.iloc[0]["validation_sharpe"]) if not baseline.empty else np.nan
    tracked = [
        "learning_rate",
        "n_estimators",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
    ]
    for param in tracked:
        affected = summary[
            summary["candidate"].map(lambda name: param in candidate_map[str(name)].params)
        ].copy()
        if affected.empty:
            continue
        best = affected.sort_values(["validation_sharpe", "validation_annual_return"], ascending=False).iloc[0]
        rows.append(
            {
                "参数(parameter)": param,
                "候选数量(candidate_count)": int(len(affected)),
                "基准验证Sharpe(baseline_validation_sharpe)": baseline_sharpe,
                "该参数最佳验证Sharpe(best_validation_sharpe)": float(best["validation_sharpe"]),
                "相对基准提升(delta_vs_baseline)": float(best["validation_sharpe"] - baseline_sharpe),
                "最佳候选(best_candidate)": best["candidate"],
                "说明(note)": describe_parameter(param),
            }
        )
    return pd.DataFrame(rows).sort_values("该参数最佳验证Sharpe(best_validation_sharpe)", ascending=False)


def describe_parameter(param: str) -> str:
    """返回关键 XGBoost 参数的中文含义和重要性。"""

    descriptions = {
        "learning_rate": "每棵树的步长，和 n_estimators 共同决定拟合速度；低步长通常更稳但需要更多树。",
        "n_estimators": "树的数量，控制 boosting 轮数；太少欠拟合，太多在噪声分钟数据上可能过拟合。",
        "max_depth": "单棵树深度，控制非线性复杂度；越深越能捕捉交互，也越容易过拟合。",
        "min_child_weight": "叶子节点最小样本权重，越大越保守；对分钟级噪声和小样本局部规律很关键。",
        "subsample": "行采样比例，降低过拟合并增加树之间差异。",
        "colsample_bytree": "列采样比例，降低高维因子共线性风险。",
        "reg_lambda": "L2 正则，平滑叶子权重，降低极端拟合。",
        "reg_alpha": "L1 正则，鼓励稀疏和更强约束，可压制弱噪声特征。",
    }
    return descriptions[param]


def write_parameter_report(
    run_dir: Path,
    config: xb.ExperimentConfig,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    selected_candidate: ModelCandidate,
    selected_rule: xb.PortfolioRule | None,
    test_metrics: dict[str, Any] | None,
) -> None:
    """写出中文参数分析和调优结论。"""

    top_rows = summary.sort_values(["validation_sharpe", "validation_annual_return"], ascending=False).head(8)
    lines = [
        "# XGBoost 参数与策略门控调优报告",
        "",
        "## 调优边界",
        "",
        f"- 核心训练集：`datetime < {config.validation_start.date()}`，且标签窗口不跨验证集。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`，且标签窗口不跨测试集。",
        f"- 测试集：`datetime >= {config.split_date.date()}`，只在模型参数和策略规则选定后评估一次。",
        "- 特征筛选、XGBoost 参数选择、top-N、置信度门控和分差门控都不使用测试集。",
        "",
        "## 参数重要性说明",
        "",
    ]
    for _, row in sensitivity.iterrows():
        lines.append(
            f"- `{row['参数(parameter)']}`：{row['说明(note)']} "
            f"本轮该参数最佳候选 `{row['最佳候选(best_candidate)']}`，"
            f"验证 Sharpe {float(row['该参数最佳验证Sharpe(best_validation_sharpe)']):.6f}。"
        )
    lines.extend(
        [
            "",
            "## 验证集候选排名",
            "",
            "| 排名 | 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 | 改动参数 |",
            "|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['candidate']}` | {float(row['validation_sharpe']):.6f} | "
            f"{float(row['validation_annual_return']):.6f} | {float(row['validation_max_drawdown']):.6f} | "
            f"`{row['best_rule']}` | `{row['changed_params']}` |"
        )
    lines.extend(
        [
            "",
            "## 选中组合",
            "",
            f"- 选中模型参数候选：`{selected_candidate.name}`。",
            f"- 候选含义：{selected_candidate.rationale}",
            f"- 参数覆盖：`{json.dumps(selected_candidate.params, ensure_ascii=False, sort_keys=True)}`。",
        ]
    )
    if selected_rule is not None:
        lines.extend(
            [
                f"- 选中策略规则：`{selected_rule.name}`。",
                f"- 置信度门控阈值：`{selected_rule.confidence_threshold}`，来自验证集分位数 `{selected_rule.confidence_quantile}`。",
                f"- top1-top2 分差门控阈值：`{selected_rule.spread_threshold}`，来自验证集分位数 `{selected_rule.spread_quantile}`。",
            ]
        )
    if test_metrics is not None:
        reached_target = float(test_metrics.get("sharpe", 0.0)) >= config.sharpe_target
        lines.extend(
            [
                "",
                "## 最终测试集验收",
                "",
                f"- Sharpe：{float(test_metrics.get('sharpe', 0.0)):.6f}",
                f"- 年化收益：{float(test_metrics.get('annual_return', 0.0)):.6f}",
                f"- 年化波动率：{float(test_metrics.get('annual_volatility', 0.0)):.6f}",
                f"- 最大回撤：{float(test_metrics.get('max_drawdown', 0.0)):.6f}",
                f"- 成本后总收益：{float(test_metrics.get('total_return', 0.0)):.6f}",
                f"- 平均多头暴露：{float(test_metrics.get('avg_gross_exposure', 0.0)):.6f}",
                f"- short_exposure：{float(test_metrics.get('short_exposure', 0.0)):.6f}",
                f"- Sharpe 目标 `{config.sharpe_target:g}` 是否达成：`{reached_target}`。",
                "- 如果最终测试未达目标，也不能根据测试集结果回头修改参数；下一步只能使用新的未来数据、"
                "重新定义训练期内部 walk-forward 验证，或在明确标注已触碰测试集的前提下另开研究实验。",
            ]
        )
    lines.extend(
        [
            "",
            "## 产物说明",
            "",
            "- `model_parameter_candidates.csv`：每个模型候选在验证集上最优策略规则和指标。",
            "- `parameter_sensitivity.csv`：按参数维度汇总的验证集敏感性。",
            "- `validation_rules_<candidate>.csv`：单个候选模型的全部验证集策略规则表现。",
            "- 最终模型、预测、交易记录、收益曲线和报告复用主回测脚本的输出格式。",
        ]
    )
    (run_dir / "parameter_tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tuning(args: argparse.Namespace) -> Path:
    """执行参数调优，并在选中组合后可选执行一次最终测试集回测。"""

    config = create_backtest_config(args)
    feature_manifest = xb.read_feature_manifest(config.feature_manifest)
    feature_cols = feature_manifest["model_feature"].astype(str).tolist()
    symbols = xb.read_stable_symbols(config.stable_symbols_file, config.max_symbols)
    panel_base, feature_panel_cache_path, feature_panel_cache_loaded = xb.load_or_build_feature_panel(
        config,
        symbols,
        feature_manifest,
    )
    panel = xb.add_target_rank(panel_base)
    core_train, validation, test = xb.split_panel(panel, config)
    if core_train.empty or validation.empty or test.empty:
        raise ValueError(
            f"切分后样本量无效：core_train={len(core_train)} validation={len(validation)} test={len(test)}"
        )

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_split_audit(run_dir, config, core_train, validation, test)
    (run_dir / "resolved_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x_core, y_core = xb.build_training_matrix(core_train, feature_cols)
    x_validation, _ = xb.build_training_matrix(validation, feature_cols)
    validation_eval_base = validation.dropna(subset=["target_rank"]).copy()
    candidates = create_model_candidates(args.candidate_profile)

    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"正在验证候选模型 {idx}/{len(candidates)}：{candidate.name}", flush=True)
        rows.append(
            evaluate_model_candidate(
                candidate,
                config,
                feature_cols,
                x_core,
                y_core,
                x_validation,
                validation_eval_base,
                run_dir,
            )
        )
    summary = pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_annual_return"], ascending=False)
    summary.to_csv(run_dir / "model_parameter_candidates.csv", index=False, encoding="utf-8-sig")
    sensitivity = calculate_parameter_sensitivity(summary, candidates)
    sensitivity.to_csv(run_dir / "parameter_sensitivity.csv", index=False, encoding="utf-8-sig")

    selected_name = str(summary.iloc[0]["candidate"])
    selected_candidate = next(candidate for candidate in candidates if candidate.name == selected_name)
    (run_dir / "selected_model_candidate.json").write_text(
        json.dumps(
            {
                "candidate": selected_candidate.name,
                "changed_params": selected_candidate.params,
                "rationale": selected_candidate.rationale,
                "selection_metric": "validation_sharpe_then_validation_annual_return",
                "test_set_used_for_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selected_rule: xb.PortfolioRule | None = None
    test_metrics: dict[str, Any] | None = None

    if not args.no_final_test:
        print(f"正在用选中候选重训并最终测试：{selected_candidate.name}", flush=True)
        validation_model = create_xgboost_model(config, selected_candidate)
        validation_model.fit(x_core, y_core)
        validation_eval = validation_eval_base.copy()
        validation_eval["prediction"] = validation_model.predict(x_validation).astype("float32")
        validation_table = xb.evaluate_rules(validation_eval, config)
        selected_rule = create_selected_rule(validation_table.iloc[0])

        train_eval = core_train.dropna(subset=["target_rank"]).copy()
        x_train_core, _ = xb.build_training_matrix(core_train, feature_cols)
        train_eval["prediction"] = validation_model.predict(x_train_core).astype("float32")
        train_metrics = {"mean_spearman_ic": xb.mean_spearman_ic(train_eval)}
        validation_metrics = {"mean_spearman_ic": xb.mean_spearman_ic(validation_eval)}

        train_full = panel.loc[
            (panel["datetime"] < config.split_date)
            & (pd.to_datetime(panel["label_exit_datetime"]) < config.split_date)
        ].copy()
        x_train_full, y_train_full = xb.build_training_matrix(train_full, feature_cols)
        final_model = create_xgboost_model(config, selected_candidate)
        final_model.fit(x_train_full, y_train_full)

        test_eval = test.dropna(subset=["forward_return"]).copy()
        test_eval["prediction"] = final_model.predict(test_eval[feature_cols]).astype("float32")
        test_curve, test_holdings, test_trade_events, test_round_trips, test_open_positions, test_metrics = xb.run_long_only_backtest(
            test_eval,
            selected_rule,
            config,
        )
        test_metrics["mean_spearman_ic"] = xb.mean_spearman_ic(test_eval)
        test_metrics["test_start"] = str(test_eval["datetime"].min())
        test_metrics["test_end"] = str(test_eval["datetime"].max())
        test_metrics["test_rows"] = int(len(test_eval))
        test_metrics["test_symbol_count"] = int(test_eval["symbol"].nunique())
        test_metrics["selected_model_candidate"] = selected_candidate.name

        xb.save_outputs(
            config,
            run_dir,
            final_model,
            feature_cols,
            feature_manifest,
            validation_table,
            selected_rule,
            train_eval[
                [
                    "datetime",
                    "symbol",
                    "close",
                    "forward_return",
                    "entry_datetime",
                    "entry_price",
                    "exit_price",
                    "target_rank",
                    "prediction",
                ]
            ].head(200_000),
            validation_eval[
                [
                    "datetime",
                    "symbol",
                    "close",
                    "forward_return",
                    "entry_datetime",
                    "entry_price",
                    "exit_price",
                    "target_rank",
                    "prediction",
                ]
            ],
            test_eval[
                [
                    "datetime",
                    "symbol",
                    "close",
                    "forward_return",
                    "entry_datetime",
                    "entry_price",
                    "exit_price",
                    "target_rank",
                    "prediction",
                ]
            ],
            test_curve,
            test_holdings,
            test_trade_events,
            test_round_trips,
            test_open_positions,
            test_metrics,
            train_metrics,
            validation_metrics,
            symbols,
            feature_panel_cache_path,
            feature_panel_cache_loaded,
        )

    write_parameter_report(run_dir, config, summary, sensitivity, selected_candidate, selected_rule, test_metrics)
    print(f"参数调优结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行验证集内参数调优流程。"""

    run_tuning(parse_args())


if __name__ == "__main__":
    main()
