"""训练并回测全 A 小市值机器学习叠加策略。

输入为 `run_all_a_small_cap_backtest.py` 生成的全 A 采样面板 pkl。脚本只用
测试集之前的数据训练 LightGBM，并只在验证集上选择小市值叠加候选和组合
规则；测试集只做一次最终验收。输出包括模型文件、验证候选表、测试收益
曲线、逐笔交易记录和中文报告。脚本不修改原始分钟行情，默认仍执行 A 股
现货 long-only、signal_lag=1、T+1、100 股整数手和交易成本约束。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - 预期在 quant conda 环境中可用
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_xgboost_long_only_backtest as xb
from scripts.tune_xgboost_long_only_parameters import create_selected_rule


PANEL_CACHE = PROJECT_ROOT / "reports" / "all_a_small_cap_backtest" / "panel_cache" / "all_a_small_cap_panel_v2_proxy_20241101_20260430_h24_s24_m0.pkl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "all_a_small_cap_model_backtest"
FEATURE_COLUMNS = [
    "small_cap_score",
    "small_cap_liquidity_score",
    "small_cap_reversal_score",
    "small_cap_momentum_quality_score",
    "low_price_score",
    "low_amount_score",
    "illiquidity_score",
    "reversal_24_score",
    "momentum_120_score",
    "low_volatility_score",
    "return_24",
    "return_120",
    "volatility_120",
]


@dataclass(frozen=True)
class ModelCandidate:
    """小市值机器学习叠加候选。

    name: 稳定英文标识，用于文件名。
    score_col: 候选打分列，后续映射为组合回测的 prediction。
    rationale: 中文策略逻辑说明。
    """

    name: str
    score_col: str
    rationale: str


@dataclass(frozen=True)
class SmallCapModelConfig:
    """全 A 小市值机器学习策略配置。"""

    panel_cache: Path
    output_dir: Path
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    split_date: pd.Timestamp
    forward_horizon_bars: int
    evaluation_step_bars: int
    commission_rate: float
    initial_cash: float
    top_n_candidates: tuple[int, ...]
    top_quantile_candidates: tuple[float, ...]
    min_assets_per_rebalance: int
    min_holding_count: int
    hold_steps_candidates: tuple[int, ...]
    target_exposure_candidates: tuple[float, ...]
    confidence_quantile_candidates: tuple[float | None, ...]
    spread_quantile_candidates: tuple[float | None, ...]
    lot_size: int
    sharpe_target: float
    random_seed: int
    max_train_rows: int
    prediction_chunk_rows: int


def parse_args() -> argparse.Namespace:
    """解析全 A 小市值机器学习策略参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-cache", type=Path, default=PANEL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-start", default="2024-11-01")
    parser.add_argument("--validation-start", default="2025-03-01")
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--commission-rate", type=float, default=xb.COMMISSION_RATE)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--top-n-candidates", default="20,50,100")
    parser.add_argument("--top-quantile-candidates", default="0.005,0.01")
    parser.add_argument("--min-assets", type=int, default=1000)
    parser.add_argument("--min-holding-count", type=int, default=1)
    parser.add_argument("--hold-steps-candidates", default="1")
    parser.add_argument("--target-exposure-candidates", default="1.0")
    parser.add_argument("--confidence-quantile-candidates", default="none")
    parser.add_argument("--spread-quantile-candidates", default="none")
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--sharpe-target", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=xb.RANDOM_SEED)
    parser.add_argument("--max-train-rows", type=int, default=2_500_000)
    parser.add_argument("--prediction-chunk-rows", type=int, default=2_000_000)
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> SmallCapModelConfig:
    """校验参数并生成配置对象。"""

    if lgb is None:
        raise RuntimeError("当前 Python 环境未安装 lightgbm。")
    return SmallCapModelConfig(
        panel_cache=args.panel_cache,
        output_dir=args.output_dir,
        train_start=pd.Timestamp(args.train_start),
        validation_start=pd.Timestamp(args.validation_start),
        split_date=pd.Timestamp(args.split_date),
        forward_horizon_bars=int(args.horizon),
        evaluation_step_bars=int(args.eval_step),
        commission_rate=float(args.commission_rate),
        initial_cash=float(args.initial_cash),
        top_n_candidates=xb.parse_int_tuple(args.top_n_candidates),
        top_quantile_candidates=xb.parse_float_tuple(args.top_quantile_candidates),
        min_assets_per_rebalance=int(args.min_assets),
        min_holding_count=int(args.min_holding_count),
        hold_steps_candidates=xb.parse_int_tuple(args.hold_steps_candidates),
        target_exposure_candidates=xb.parse_float_tuple(args.target_exposure_candidates),
        confidence_quantile_candidates=xb.parse_optional_float_tuple(args.confidence_quantile_candidates),
        spread_quantile_candidates=xb.parse_optional_float_tuple(args.spread_quantile_candidates),
        lot_size=int(args.lot_size),
        sharpe_target=float(args.sharpe_target),
        random_seed=int(args.seed),
        max_train_rows=max(int(args.max_train_rows), 0),
        prediction_chunk_rows=max(int(args.prediction_chunk_rows), 100_000),
    )


def create_xb_config(config: SmallCapModelConfig) -> xb.ExperimentConfig:
    """把当前配置转换为主回测引擎兼容配置。"""

    return xb.ExperimentConfig(
        data_root=xb.DATA_ROOT,
        stable_symbols_file=Path(""),
        feature_manifest=Path(""),
        output_dir=config.output_dir,
        feature_panel_cache_dir=config.output_dir / "panel_cache",
        split_date=config.split_date,
        validation_start=config.validation_start,
        forward_horizon_bars=config.forward_horizon_bars,
        evaluation_step_bars=config.evaluation_step_bars,
        commission_rate=config.commission_rate,
        initial_cash=config.initial_cash,
        top_n_candidates=config.top_n_candidates,
        top_quantile_candidates=config.top_quantile_candidates,
        min_assets_per_rebalance=config.min_assets_per_rebalance,
        min_holding_count=config.min_holding_count,
        hold_steps_candidates=config.hold_steps_candidates,
        target_exposure_candidates=config.target_exposure_candidates,
        confidence_quantile_candidates=config.confidence_quantile_candidates,
        spread_quantile_candidates=config.spread_quantile_candidates,
        lot_size=config.lot_size,
        sharpe_target=config.sharpe_target,
        max_symbols=0,
        workers=xb.WORKERS,
        random_seed=config.random_seed,
        rebuild_feature_panel=False,
        no_feature_panel_cache=False,
        build_feature_panel_only=False,
    )


def json_default(value: object) -> object:
    """把 Path、Timestamp 和 numpy 标量转换为 JSON 可写格式。"""

    if isinstance(value, Path | pd.Timestamp):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_panel(config: SmallCapModelConfig) -> pd.DataFrame:
    """读取采样面板并保留训练、预测和回测所需列。"""

    needed = [
        "datetime",
        "symbol",
        "entry_datetime",
        "entry_price",
        "exit_price",
        "label_exit_datetime",
        "forward_return",
        *FEATURE_COLUMNS,
    ]
    frame = pd.read_pickle(config.panel_cache)
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise ValueError(f"面板缺少必要列：{missing}")
    frame = frame.loc[:, needed].copy()
    for col in ["datetime", "entry_datetime", "label_exit_datetime"]:
        frame[col] = pd.to_datetime(frame[col])
    frame = frame.loc[frame["datetime"] >= config.train_start].replace([np.inf, -np.inf], np.nan)
    return frame.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def fit_lightgbm_model(train: pd.DataFrame, config: SmallCapModelConfig) -> tuple[Any, pd.Series, int]:
    """只用训练集拟合 LightGBM，并返回模型、训练集中位数和实际训练行数。"""

    fit_frame = train.dropna(subset=["forward_return"]).copy()
    if config.max_train_rows and len(fit_frame) > config.max_train_rows:
        fit_frame = fit_frame.sample(n=config.max_train_rows, random_state=config.random_seed)
    medians = fit_frame[FEATURE_COLUMNS].median(numeric_only=True).fillna(0.0)
    x_train = fit_frame[FEATURE_COLUMNS].fillna(medians).astype("float32")
    y_train = fit_frame["forward_return"].astype("float32")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=450,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=250,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=8.0,
        random_state=config.random_seed,
        n_jobs=max(xb.WORKERS, 1),
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model, medians, int(len(fit_frame))


def predict_in_chunks(frame: pd.DataFrame, model: Any, medians: pd.Series, chunk_rows: int) -> np.ndarray:
    """按块预测，避免一次性复制完整测试特征矩阵。"""

    predictions = np.empty(len(frame), dtype="float32")
    for start in range(0, len(frame), chunk_rows):
        end = min(start + chunk_rows, len(frame))
        x_chunk = frame.iloc[start:end][FEATURE_COLUMNS].fillna(medians).astype("float32")
        predictions[start:end] = model.predict(x_chunk).astype("float32")
    return predictions


def add_candidate_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """构造机器学习预测和小市值暴露的候选组合打分。"""

    out = frame.copy()
    out["model_rank_score"] = out.groupby("datetime")["model_prediction"].rank(method="average", pct=True).astype("float32")
    out["model_small_cap_score"] = (
        0.70 * out["model_rank_score"].fillna(0.5)
        + 0.20 * out["small_cap_score"].fillna(0.5)
        + 0.10 * out["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    out["model_small_cap_reversal_score"] = (
        0.60 * out["model_rank_score"].fillna(0.5)
        + 0.25 * out["small_cap_reversal_score"].fillna(0.5)
        + 0.15 * out["illiquidity_score"].fillna(0.5)
    ).astype("float32")
    return out


def create_candidates() -> list[ModelCandidate]:
    """返回固定的小市值机器学习叠加候选集合。"""

    return [
        ModelCandidate("model_only", "model_rank_score", "只使用训练期 LightGBM 预测的截面排名。"),
        ModelCandidate("model_small_cap", "model_small_cap_score", "以模型预测为主，叠加小市值和低波动暴露。"),
        ModelCandidate("model_small_cap_reversal", "model_small_cap_reversal_score", "以模型预测为主，叠加小市值、短期反转和非流动性暴露。"),
    ]


def evaluate_candidate(candidate: ModelCandidate, validation: pd.DataFrame, xb_config: xb.ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """只用验证集评估一个模型叠加候选。"""

    frame = validation.copy()
    frame["prediction"] = frame[candidate.score_col].astype("float32")
    table = xb.evaluate_rules(frame, xb_config)
    table.to_csv(run_dir / f"validation_rules_{candidate.name}.csv", index=False, encoding="utf-8-sig")
    best = table.iloc[0]
    row: dict[str, Any] = {
        "candidate": candidate.name,
        "score_col": candidate.score_col,
        "rationale": candidate.rationale,
        "best_rule": best["rule"],
    }
    for key in ["sharpe", "annual_return", "annual_volatility", "max_drawdown", "total_return", "avg_turnover", "avg_gross_exposure", "round_trip_win_rate", "trade_count"]:
        row[f"validation_{key}"] = best.get(key)
    return row


def save_test_outputs(run_dir: Path, candidate: ModelCandidate, selected_rule: xb.PortfolioRule, test: pd.DataFrame, xb_config: xb.ExperimentConfig) -> dict[str, Any]:
    """保存验证集选中候选在测试集上的最终回测产物。"""

    frame = test.copy()
    frame["prediction"] = frame[candidate.score_col].astype("float32")
    curve, holdings, trades, round_trips, open_positions, metrics = xb.run_long_only_backtest(frame, selected_rule, xb_config)
    metrics["strategy_candidate"] = candidate.name
    metrics["strategy_rationale"] = candidate.rationale
    metrics["test_start"] = str(frame["datetime"].min())
    metrics["test_end"] = str(frame["datetime"].max())
    metrics["test_rows"] = int(len(frame))
    metrics["test_symbol_count"] = int(frame["symbol"].nunique())
    curve.to_csv(run_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    holdings.to_csv(run_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(run_dir / "test_trade_events.csv", index=False, encoding="utf-8-sig")
    round_trips.to_csv(run_dir / "test_round_trips.csv", index=False, encoding="utf-8-sig")
    open_positions.to_csv(run_dir / "test_open_positions.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(run_dir / "test_metrics.csv", header=["value"])
    xb.plot_test_return_curve(run_dir, curve, metrics)
    xb.write_output_guides(run_dir)
    return metrics


def write_report(
    run_dir: Path,
    config: SmallCapModelConfig,
    split_audit: pd.DataFrame,
    summary: pd.DataFrame,
    selected: ModelCandidate,
    selected_rule: xb.PortfolioRule,
    metrics: dict[str, Any],
    train_rows_used: int,
) -> None:
    """写出中文小市值机器学习策略报告。"""

    reached = float(metrics.get("sharpe", 0.0)) >= config.sharpe_target
    lines = [
        "# 全 A 小市值机器学习策略回测报告",
        "",
        "## 边界",
        "",
        "- 股票池来自本地全 A 采样面板。",
        "- 模型只使用测试集之前的数据训练；候选和规则只用验证集选择；测试集只做最终验收。",
        "- 策略是 A 股现货 long-only，不做空，不加杠杆，执行 signal_lag=1、T+1、100 股整数手和交易成本。",
        f"- 训练集：`{config.train_start.date()} <= datetime < {config.validation_start.date()}`。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`。",
        f"- 测试集：`datetime >= {config.split_date.date()}`。",
        f"- 实际训练行数：{train_rows_used}",
        "",
        "## 数据切分",
        "",
        split_audit.to_markdown(index=False),
        "",
        "## 验证集候选",
        "",
        "| 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['candidate']}` | {float(row['validation_sharpe']):.6f} | "
            f"{float(row['validation_annual_return']):.6f} | {float(row['validation_max_drawdown']):.6f} | `{row['best_rule']}` |"
        )
    lines.extend(
        [
            "",
            "## 选中策略",
            "",
            f"- 候选：`{selected.name}`。",
            f"- 规则：`{selected_rule.name}`。",
            f"- 说明：{selected.rationale}",
            "",
            "## 测试集验收",
            "",
            f"- Sharpe：{float(metrics.get('sharpe', 0.0)):.6f}",
            f"- 年化收益：{float(metrics.get('annual_return', 0.0)):.6f}",
            f"- 年化波动率：{float(metrics.get('annual_volatility', 0.0)):.6f}",
            f"- 最大回撤：{float(metrics.get('max_drawdown', 0.0)):.6f}",
            f"- 成本后总收益：{float(metrics.get('total_return', 0.0)):.6f}",
            f"- short_exposure：{float(metrics.get('short_exposure', 0.0)):.6f}",
            f"- Sharpe 目标 `{config.sharpe_target:g}` 是否达成：`{reached}`",
        ]
    )
    (run_dir / "all_a_small_cap_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_a_small_cap_model(args: argparse.Namespace) -> Path:
    """执行全 A 小市值机器学习叠加策略训练、验证和测试。"""

    config = create_config(args)
    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    panel = read_panel(config)
    train = panel.loc[(panel["datetime"] >= config.train_start) & (panel["datetime"] < config.validation_start)].copy()
    validation = panel.loc[(panel["datetime"] >= config.validation_start) & (panel["datetime"] < config.split_date)].copy()
    test = panel.loc[panel["datetime"] >= config.split_date].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"切分后样本为空：train={len(train)} validation={len(validation)} test={len(test)}")
    split_audit = pd.DataFrame(
        [
            {"split": "train", "rows": len(train), "symbols": train["symbol"].nunique(), "start": train["datetime"].min(), "end": train["datetime"].max()},
            {"split": "validation", "rows": len(validation), "symbols": validation["symbol"].nunique(), "start": validation["datetime"].min(), "end": validation["datetime"].max()},
            {"split": "test", "rows": len(test), "symbols": test["symbol"].nunique(), "start": test["datetime"].min(), "end": test["datetime"].max()},
        ]
    )
    split_audit.to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    print("正在训练 LightGBM 小市值叠加模型", flush=True)
    model, medians, train_rows_used = fit_lightgbm_model(train, config)
    model.booster_.save_model(run_dir / "lightgbm_small_cap_model.txt")
    with (run_dir / "feature_medians.pkl").open("wb") as file:
        pickle.dump(medians, file)
    pd.Series(FEATURE_COLUMNS, name="feature").to_csv(run_dir / "model_features.csv", index=False, encoding="utf-8-sig")
    print("正在预测验证集和测试集", flush=True)
    validation["model_prediction"] = predict_in_chunks(validation, model, medians, config.prediction_chunk_rows)
    test["model_prediction"] = predict_in_chunks(test, model, medians, config.prediction_chunk_rows)
    validation = add_candidate_scores(validation)
    test = add_candidate_scores(test)
    xb_config = create_xb_config(config)
    rows = []
    candidates = create_candidates()
    for candidate in candidates:
        print(f"正在验证小市值模型候选：{candidate.name}", flush=True)
        rows.append(evaluate_candidate(candidate, validation, xb_config, run_dir))
    summary = pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_annual_return"], ascending=False)
    summary.to_csv(run_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected = next(candidate for candidate in candidates if candidate.name == str(summary.iloc[0]["candidate"]))
    validation_table = pd.read_csv(run_dir / f"validation_rules_{selected.name}.csv")
    selected_rule = create_selected_rule(validation_table.iloc[0])
    metrics = save_test_outputs(run_dir, selected, selected_rule, test, xb_config)
    write_report(run_dir, config, split_audit, summary, selected, selected_rule, metrics, train_rows_used)
    print(f"全 A 小市值机器学习策略结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行全 A 小市值机器学习叠加策略。"""

    run_all_a_small_cap_model(parse_args())


if __name__ == "__main__":
    main()
