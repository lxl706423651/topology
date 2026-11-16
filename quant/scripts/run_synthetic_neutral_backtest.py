"""运行合成中性模型策略回测。

输入为既有 XGBoost 验证集和测试集预测文件。脚本只在验证集上选择 top-N、
置信度门控和分差门控规则；测试集只做一次最终验收。策略收益按“多头模型
组合收益 - 同期股票池等权基准收益 - 交易成本”计算，因此这是需要指数、
ETF、股指期货或其他对冲腿的研究口径，不是 A 股现货纯 long-only 结果。
输出包括验证规则表、测试主动收益曲线、指标和中文报告；不修改原始行情。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_xgboost_long_only_backtest as xb
from quant_backtest.metrics import performance_metrics


PREDICTION_RUN_DIR = PROJECT_ROOT / "reports" / "xgboost_long_only_backtest" / "run_20260526_202355"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "synthetic_neutral_backtest"


@dataclass(frozen=True)
class SyntheticNeutralConfig:
    """合成中性策略配置。"""

    prediction_run_dir: Path
    output_dir: Path
    split_date: pd.Timestamp
    validation_start: pd.Timestamp
    forward_horizon_bars: int
    evaluation_step_bars: int
    commission_rate: float
    hedge_cost_rate: float
    hedge_ratio: float
    top_n_candidates: tuple[int, ...]
    top_quantile_candidates: tuple[float, ...]
    min_assets_per_rebalance: int
    min_holding_count: int
    hold_steps_candidates: tuple[int, ...]
    target_exposure_candidates: tuple[float, ...]
    confidence_quantile_candidates: tuple[float | None, ...]
    spread_quantile_candidates: tuple[float | None, ...]
    sharpe_target: float


def parse_args() -> argparse.Namespace:
    """解析合成中性策略参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-run-dir", type=Path, default=PREDICTION_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--commission-rate", type=float, default=xb.COMMISSION_RATE)
    parser.add_argument("--hedge-cost-rate", type=float, default=0.0001)
    parser.add_argument("--hedge-ratio", type=float, default=1.0)
    parser.add_argument("--top-n-candidates", default="1,2,3")
    parser.add_argument("--top-quantile-candidates", default="")
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--min-holding-count", type=int, default=1)
    parser.add_argument("--hold-steps-candidates", default="1")
    parser.add_argument("--target-exposure-candidates", default="1.0")
    parser.add_argument("--confidence-quantile-candidates", default="none,0.85,0.9")
    parser.add_argument("--spread-quantile-candidates", default="none,0.6,0.7")
    parser.add_argument("--sharpe-target", type=float, default=3.0)
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> SyntheticNeutralConfig:
    """校验参数并生成配置对象。"""

    top_quantile_text = str(args.top_quantile_candidates).strip()
    return SyntheticNeutralConfig(
        prediction_run_dir=args.prediction_run_dir,
        output_dir=args.output_dir,
        split_date=pd.Timestamp(args.split_date),
        validation_start=pd.Timestamp(args.validation_start),
        forward_horizon_bars=int(args.horizon),
        evaluation_step_bars=int(args.eval_step),
        commission_rate=float(args.commission_rate),
        hedge_cost_rate=float(args.hedge_cost_rate),
        hedge_ratio=float(args.hedge_ratio),
        top_n_candidates=xb.parse_int_tuple(args.top_n_candidates),
        top_quantile_candidates=xb.parse_float_tuple(top_quantile_text) if top_quantile_text else tuple(),
        min_assets_per_rebalance=int(args.min_assets),
        min_holding_count=int(args.min_holding_count),
        hold_steps_candidates=xb.parse_int_tuple(args.hold_steps_candidates),
        target_exposure_candidates=xb.parse_float_tuple(args.target_exposure_candidates),
        confidence_quantile_candidates=xb.parse_optional_float_tuple(args.confidence_quantile_candidates),
        spread_quantile_candidates=xb.parse_optional_float_tuple(args.spread_quantile_candidates),
        sharpe_target=float(args.sharpe_target),
    )


def create_xb_config(config: SyntheticNeutralConfig) -> xb.ExperimentConfig:
    """把中性策略配置转换为规则选择函数所需配置。"""

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
        initial_cash=1.0,
        top_n_candidates=config.top_n_candidates,
        top_quantile_candidates=config.top_quantile_candidates,
        min_assets_per_rebalance=config.min_assets_per_rebalance,
        min_holding_count=config.min_holding_count,
        hold_steps_candidates=config.hold_steps_candidates,
        target_exposure_candidates=config.target_exposure_candidates,
        confidence_quantile_candidates=config.confidence_quantile_candidates,
        spread_quantile_candidates=config.spread_quantile_candidates,
        lot_size=100,
        sharpe_target=config.sharpe_target,
        max_symbols=0,
        workers=xb.WORKERS,
        random_seed=xb.RANDOM_SEED,
        rebuild_feature_panel=False,
        no_feature_panel_cache=False,
        build_feature_panel_only=False,
    )


def read_prediction_file(path: Path) -> pd.DataFrame:
    """读取预测文件并标准化时间列。"""

    frame = pd.read_csv(path)
    for col in ["datetime", "entry_datetime"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def calculate_synthetic_curve(frame: pd.DataFrame, rule: xb.PortfolioRule, xb_config: xb.ExperimentConfig, config: SyntheticNeutralConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算合成中性主动收益曲线和每期多头选择明细。"""

    rows: list[dict[str, Any]] = []
    holdings: list[pd.DataFrame] = []
    previous_symbols: set[str] = set()
    for dt, group in frame.groupby("datetime", sort=True):
        tradable = group.dropna(subset=["prediction", "forward_return", "entry_price", "exit_price"]).copy()
        if len(tradable) < config.min_assets_per_rebalance:
            rows.append(
                {
                    "datetime": pd.Timestamp(dt),
                    "long_return": 0.0,
                    "hedge_return": 0.0,
                    "cost_return": 0.0,
                    "return": 0.0,
                    "turnover": 0.0,
                    "holding_count": 0,
                }
            )
            previous_symbols = set()
            continue
        selected = xb.select_symbols_for_rule(tradable, rule, xb_config)
        if selected.empty:
            rows.append(
                {
                    "datetime": pd.Timestamp(dt),
                    "long_return": 0.0,
                    "hedge_return": 0.0,
                    "cost_return": 0.0,
                    "return": 0.0,
                    "turnover": 0.0,
                    "holding_count": 0,
                }
            )
            previous_symbols = set()
            continue
        current_symbols = set(selected["symbol"].astype(str))
        turnover = 1.0 if not previous_symbols else len(current_symbols.symmetric_difference(previous_symbols)) / max(len(current_symbols) + len(previous_symbols), 1)
        previous_symbols = current_symbols
        long_return = float(selected["forward_return"].mean())
        hedge_return = float(tradable["forward_return"].mean()) * config.hedge_ratio
        cost_return = turnover * (config.commission_rate + config.hedge_cost_rate)
        active_return = long_return - hedge_return - cost_return
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "long_return": long_return,
                "hedge_return": hedge_return,
                "cost_return": cost_return,
                "return": active_return,
                "turnover": turnover,
                "holding_count": int(len(selected)),
            }
        )
        held = selected.loc[:, ["datetime", "symbol", "prediction", "forward_return", "entry_price", "exit_price"]].copy()
        held["rule"] = rule.name
        holdings.append(held)
    curve = pd.DataFrame(rows).sort_values("datetime").set_index("datetime")
    curve["equity"] = (1.0 + curve["return"].fillna(0.0)).cumprod()
    curve["cumulative_return"] = curve["equity"] - 1.0
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    holdings_frame = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    return curve, holdings_frame


def calculate_metrics(curve: pd.DataFrame, rule: xb.PortfolioRule, config: SyntheticNeutralConfig) -> dict[str, Any]:
    """根据主动收益曲线计算中性策略指标。"""

    annualization = 252 * 240 / config.evaluation_step_bars
    metrics = performance_metrics(curve["equity"], curve["return"], annualization=annualization)
    metrics.update(
        {
            "rule": rule.name,
            "avg_turnover": float(curve["turnover"].mean()) if not curve.empty else 0.0,
            "total_turnover": float(curve["turnover"].sum()) if not curve.empty else 0.0,
            "avg_holding_count": float(curve["holding_count"].mean()) if not curve.empty else 0.0,
            "avg_gross_exposure": float(1.0 + config.hedge_ratio),
            "short_exposure": float(config.hedge_ratio),
            "commission_rate": float(config.commission_rate),
            "hedge_cost_rate": float(config.hedge_cost_rate),
            "hedge_ratio": float(config.hedge_ratio),
            "signal_lag_bars": 1,
            "rebalance_horizon_bars": int(config.forward_horizon_bars),
            "strategy_type": "synthetic_long_minus_equal_weight_hedge",
        }
    )
    return metrics


def evaluate_rules(validation: pd.DataFrame, xb_config: xb.ExperimentConfig, config: SyntheticNeutralConfig) -> pd.DataFrame:
    """只使用验证集合成主动收益 Sharpe 选择规则。"""

    rows: list[dict[str, Any]] = []
    for rule in xb.create_candidate_rules(xb_config, validation):
        curve, _ = calculate_synthetic_curve(validation, rule, xb_config, config)
        metrics = calculate_metrics(curve, rule, config)
        row = {
            "rule": rule.name,
            "rule_type": rule.rule_type,
            "rule_value": rule.value,
            "hold_steps": rule.hold_steps,
            "target_exposure": rule.target_exposure,
            "confidence_quantile": rule.confidence_quantile,
            "confidence_threshold": rule.confidence_threshold,
            "spread_quantile": rule.spread_quantile,
            "spread_threshold": rule.spread_threshold,
            "periods": int(len(curve)),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)


def write_report(run_dir: Path, config: SyntheticNeutralConfig, selected_rule: xb.PortfolioRule, metrics: dict[str, Any]) -> None:
    """写出中文合成中性策略报告。"""

    reached = float(metrics.get("sharpe", 0.0)) >= config.sharpe_target
    lines = [
        "# 合成中性模型策略回测报告",
        "",
        "## 边界",
        "",
        "- 这是研究口径的中性策略：多头模型组合收益减去同期股票池等权基准收益。",
        "- 它需要指数、ETF、股指期货或其他对冲腿；不是 A 股现货纯 long-only 结果。",
        "- 规则只在验证集选择，测试集只做最终验收。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`。",
        f"- 测试集：`datetime >= {config.split_date.date()}`。",
        "",
        "## 选中规则",
        "",
        f"- 规则：`{selected_rule.name}`。",
        f"- 对冲比例：{config.hedge_ratio:.2f}",
        "",
        "## 测试集验收",
        "",
        f"- Sharpe：{float(metrics.get('sharpe', 0.0)):.6f}",
        f"- 年化收益：{float(metrics.get('annual_return', 0.0)):.6f}",
        f"- 年化波动率：{float(metrics.get('annual_volatility', 0.0)):.6f}",
        f"- 最大回撤：{float(metrics.get('max_drawdown', 0.0)):.6f}",
        f"- 主动总收益：{float(metrics.get('total_return', 0.0)):.6f}",
        f"- short_exposure：{float(metrics.get('short_exposure', 0.0)):.6f}",
        f"- Sharpe 目标 `{config.sharpe_target:g}` 是否达成：`{reached}`",
    ]
    (run_dir / "synthetic_neutral_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_neutral(args: argparse.Namespace) -> Path:
    """执行合成中性策略验证和测试。"""

    config = create_config(args)
    xb_config = create_xb_config(config)
    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    validation = read_prediction_file(config.prediction_run_dir / "validation_predictions.csv")
    test = read_prediction_file(config.prediction_run_dir / "test_predictions.csv")
    validation = validation.loc[(validation["datetime"] >= config.validation_start) & (validation["datetime"] < config.split_date)].copy()
    test = test.loc[test["datetime"] >= config.split_date].copy()
    table = evaluate_rules(validation, xb_config, config)
    table.to_csv(run_dir / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    selected_row = table.iloc[0]
    selected_rule = xb.PortfolioRule(
        rule_type=str(selected_row["rule_type"]),
        value=float(selected_row["rule_value"]),
        hold_steps=int(selected_row["hold_steps"]),
        target_exposure=float(selected_row["target_exposure"]),
        confidence_quantile=None if pd.isna(selected_row.get("confidence_quantile")) else float(selected_row["confidence_quantile"]),
        confidence_threshold=None if pd.isna(selected_row.get("confidence_threshold")) else float(selected_row["confidence_threshold"]),
        spread_quantile=None if pd.isna(selected_row.get("spread_quantile")) else float(selected_row["spread_quantile"]),
        spread_threshold=None if pd.isna(selected_row.get("spread_threshold")) else float(selected_row["spread_threshold"]),
    )
    curve, holdings = calculate_synthetic_curve(test, selected_rule, xb_config, config)
    metrics = calculate_metrics(curve, selected_rule, config)
    curve.to_csv(run_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    holdings.to_csv(run_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(run_dir / "test_metrics.csv", header=["value"])
    xb.plot_test_return_curve(run_dir, curve, metrics)
    write_report(run_dir, config, selected_rule, metrics)
    print(f"合成中性策略结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行合成中性策略。"""

    run_synthetic_neutral(parse_args())


if __name__ == "__main__":
    main()
