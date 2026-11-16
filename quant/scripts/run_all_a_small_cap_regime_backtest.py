"""运行全 A 小市值状态过滤策略回测。

输入为全 A 小市值代理采样面板 pkl。脚本根据 A 股异常研究中更稳定的
短期反转、低波动、流动性和市场状态线索，构造“小市值 + 反转/低波动 +
状态过滤”候选。所有市场状态阈值只用验证集估计，候选和组合规则只在
验证集选择，测试集只做一次最终验收。输出包括状态阈值、验证候选表、
测试收益曲线、逐笔交易记录和中文报告；不修改原始分钟行情。
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
from scripts.tune_xgboost_long_only_parameters import create_selected_rule


PANEL_CACHE = PROJECT_ROOT / "reports" / "all_a_small_cap_backtest" / "panel_cache" / "all_a_small_cap_panel_v2_proxy_20241101_20260430_h24_s24_m0.pkl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "all_a_small_cap_regime_backtest"


@dataclass(frozen=True)
class RegimeCandidate:
    """小市值状态过滤候选。

    name: 稳定英文标识，用于输出文件名。
    score_col: 候选打分列，后续映射为组合回测的 prediction。
    regime_col: 候选状态门控列；为 True 的时间点才允许交易。
    rationale: 中文策略逻辑说明。
    """

    name: str
    score_col: str
    regime_col: str
    rationale: str


@dataclass(frozen=True)
class SmallCapRegimeConfig:
    """全 A 小市值状态过滤策略配置。"""

    panel_cache: Path
    output_dir: Path
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


def parse_args() -> argparse.Namespace:
    """解析小市值状态过滤策略参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-cache", type=Path, default=PANEL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
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
    parser.add_argument("--target-exposure-candidates", default="0.5,1.0")
    parser.add_argument("--confidence-quantile-candidates", default="none")
    parser.add_argument("--spread-quantile-candidates", default="none")
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--sharpe-target", type=float, default=3.0)
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> SmallCapRegimeConfig:
    """校验参数并生成配置对象。"""

    return SmallCapRegimeConfig(
        panel_cache=args.panel_cache,
        output_dir=args.output_dir,
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
    )


def create_xb_config(config: SmallCapRegimeConfig) -> xb.ExperimentConfig:
    """把状态过滤配置转换为主组合回测引擎兼容配置。"""

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
        random_seed=xb.RANDOM_SEED,
        rebuild_feature_panel=False,
        no_feature_panel_cache=False,
        build_feature_panel_only=False,
    )


def read_panel(config: SmallCapRegimeConfig) -> pd.DataFrame:
    """读取全 A 小市值采样面板，并保留策略所需列。"""

    needed = [
        "datetime",
        "symbol",
        "close",
        "amount",
        "amount_ma_120",
        "return_24",
        "return_120",
        "volatility_120",
        "entry_datetime",
        "entry_price",
        "exit_price",
        "label_exit_datetime",
        "forward_return",
        "low_price_score",
        "low_amount_score",
        "illiquidity_score",
        "reversal_24_score",
        "momentum_120_score",
        "low_volatility_score",
    ]
    frame = pd.read_pickle(config.panel_cache)
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise ValueError(f"面板缺少必要列：{missing}")
    frame = frame.loc[:, needed].copy().replace([np.inf, -np.inf], np.nan)
    for col in ["datetime", "entry_datetime", "label_exit_datetime"]:
        frame[col] = pd.to_datetime(frame[col])
    return frame.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def add_market_state_features(frame: pd.DataFrame, config: SmallCapRegimeConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """添加市场状态和候选分数；状态阈值只从验证集估计。"""

    out = frame.copy()
    price_rank = (1.0 - out["low_price_score"]).clip(0.0, 1.0)
    out["price_health_score"] = (1.0 - ((price_rank - 0.35).abs() / 0.35)).clip(0.0, 1.0).astype("float32")
    out["liquidity_size_score"] = (
        0.65 * out["low_amount_score"].fillna(0.5)
        + 0.35 * out["illiquidity_score"].fillna(0.5)
    ).astype("float32")
    out["tradable_small_score"] = (
        0.55 * out["liquidity_size_score"].fillna(0.5)
        + 0.25 * out["price_health_score"].fillna(0.5)
        + 0.20 * out["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    out["small_reversal_low_vol_score"] = (
        0.45 * out["liquidity_size_score"].fillna(0.5)
        + 0.35 * out["reversal_24_score"].fillna(0.5)
        + 0.20 * out["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    out["small_momentum_low_vol_score"] = (
        0.45 * out["liquidity_size_score"].fillna(0.5)
        + 0.35 * out["momentum_120_score"].fillna(0.5)
        + 0.20 * out["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    state = (
        out.groupby("datetime", sort=False)
        .agg(
            market_return_24=("return_24", "mean"),
            market_return_120=("return_120", "mean"),
            breadth_24=("return_24", lambda values: float((values > 0).mean())),
            breadth_120=("return_120", lambda values: float((values > 0).mean())),
            market_volatility_120=("volatility_120", "mean"),
            market_liquidity=("amount_ma_120", "median"),
        )
        .reset_index()
    )
    validation_state = state.loc[(state["datetime"] >= config.validation_start) & (state["datetime"] < config.split_date)]
    thresholds = {
        "market_return_120_q40": float(validation_state["market_return_120"].quantile(0.40)),
        "market_return_120_q50": float(validation_state["market_return_120"].quantile(0.50)),
        "market_return_120_q60": float(validation_state["market_return_120"].quantile(0.60)),
        "breadth_24_q45": float(validation_state["breadth_24"].quantile(0.45)),
        "breadth_120_q50": float(validation_state["breadth_120"].quantile(0.50)),
        "market_volatility_120_q60": float(validation_state["market_volatility_120"].quantile(0.60)),
        "market_volatility_120_q70": float(validation_state["market_volatility_120"].quantile(0.70)),
        "market_liquidity_q40": float(validation_state["market_liquidity"].quantile(0.40)),
    }
    state["risk_on_regime"] = (
        (state["market_return_120"] >= thresholds["market_return_120_q50"])
        & (state["breadth_120"] >= thresholds["breadth_120_q50"])
    )
    state["reversal_low_vol_regime"] = (
        (state["market_return_120"] >= thresholds["market_return_120_q40"])
        & (state["breadth_24"] >= thresholds["breadth_24_q45"])
        & (state["market_volatility_120"] <= thresholds["market_volatility_120_q70"])
    )
    state["liquidity_supported_regime"] = (
        (state["market_liquidity"] >= thresholds["market_liquidity_q40"])
        & (state["market_volatility_120"] <= thresholds["market_volatility_120_q70"])
    )
    state["defensive_low_vol_regime"] = state["market_volatility_120"] <= thresholds["market_volatility_120_q60"]
    state["momentum_upside_regime"] = state["market_return_120"] >= thresholds["market_return_120_q60"]
    out = out.merge(
        state[
            [
                "datetime",
                "market_return_24",
                "market_return_120",
                "breadth_24",
                "breadth_120",
                "market_volatility_120",
                "market_liquidity",
                "risk_on_regime",
                "reversal_low_vol_regime",
                "liquidity_supported_regime",
                "defensive_low_vol_regime",
                "momentum_upside_regime",
            ]
        ],
        on="datetime",
        how="left",
        validate="many_to_one",
    )
    return out, thresholds


def create_candidates() -> list[RegimeCandidate]:
    """返回文献驱动的小市值状态过滤候选。"""

    return [
        RegimeCandidate("tradable_small_defensive", "tradable_small_score", "defensive_low_vol_regime", "小市值代理去掉极端低价暴露，并只在市场波动较低时交易。"),
        RegimeCandidate("small_reversal_low_vol", "small_reversal_low_vol_score", "reversal_low_vol_regime", "小市值代理叠加短期反转和低波动，并要求市场反转/低波环境。"),
        RegimeCandidate("small_liquidity_supported", "small_reversal_low_vol_score", "liquidity_supported_regime", "小市值反转低波策略只在全市场流动性不差且波动不过高时交易。"),
        RegimeCandidate("small_momentum_upside", "small_momentum_low_vol_score", "momentum_upside_regime", "小市值代理叠加中短期动量和低波动，并只在市场上行状态交易。"),
        RegimeCandidate("small_risk_on", "liquidity_size_score", "risk_on_regime", "纯流动性小市值代理只在市场收益和广度同时偏强时交易。"),
    ]


def prepare_candidate_frame(frame: pd.DataFrame, candidate: RegimeCandidate) -> pd.DataFrame:
    """按候选状态门控生成 prediction，状态不满足时不交易。"""

    out = frame.copy()
    active = out[candidate.regime_col].fillna(False).astype(bool)
    out["prediction"] = out[candidate.score_col].where(active).astype("float32")
    return out


def evaluate_candidate(candidate: RegimeCandidate, validation: pd.DataFrame, xb_config: xb.ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """只用验证集评估一个小市值状态过滤候选。"""

    table = xb.evaluate_rules(prepare_candidate_frame(validation, candidate), xb_config)
    table.to_csv(run_dir / f"validation_rules_{candidate.name}.csv", index=False, encoding="utf-8-sig")
    best = table.iloc[0]
    row: dict[str, Any] = {
        "candidate": candidate.name,
        "score_col": candidate.score_col,
        "regime_col": candidate.regime_col,
        "rationale": candidate.rationale,
        "best_rule": best["rule"],
    }
    for key in ["sharpe", "annual_return", "annual_volatility", "max_drawdown", "total_return", "avg_turnover", "avg_gross_exposure", "round_trip_win_rate", "trade_count"]:
        row[f"validation_{key}"] = best.get(key)
    return row


def save_test_outputs(run_dir: Path, candidate: RegimeCandidate, selected_rule: xb.PortfolioRule, test: pd.DataFrame, xb_config: xb.ExperimentConfig) -> dict[str, Any]:
    """保存验证集选中候选的最终测试回测产物。"""

    frame = prepare_candidate_frame(test, candidate)
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
    config: SmallCapRegimeConfig,
    thresholds: dict[str, float],
    summary: pd.DataFrame,
    selected: RegimeCandidate,
    selected_rule: xb.PortfolioRule,
    metrics: dict[str, Any],
) -> None:
    """写出中文小市值状态过滤策略报告。"""

    reached = float(metrics.get("sharpe", 0.0)) >= config.sharpe_target
    threshold_lines = [f"- `{key}`：{value:.8f}" for key, value in thresholds.items()]
    lines = [
        "# 全 A 小市值状态过滤策略回测报告",
        "",
        "## 研究边界",
        "",
        "- 股票池来自本地全 A 小市值代理面板。",
        "- 状态阈值只从验证集估计；候选和组合规则只用验证集选择；测试集只做最终验收。",
        "- 策略是 A 股现货 long-only，不做空，不加杠杆，执行 signal_lag=1、T+1、100 股整数手和交易成本。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`。",
        f"- 测试集：`datetime >= {config.split_date.date()}`。",
        "",
        "## 验证集状态阈值",
        "",
        *threshold_lines,
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
    (run_dir / "all_a_small_cap_regime_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_a_small_cap_regime(args: argparse.Namespace) -> Path:
    """执行全 A 小市值状态过滤策略验证和测试。"""

    config = create_config(args)
    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    panel, thresholds = add_market_state_features(read_panel(config), config)
    (run_dir / "validation_state_thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = panel.loc[(panel["datetime"] >= config.validation_start) & (panel["datetime"] < config.split_date)].copy()
    test = panel.loc[panel["datetime"] >= config.split_date].copy()
    if validation.empty or test.empty:
        raise ValueError(f"切分后样本为空：validation={len(validation)} test={len(test)}")
    pd.DataFrame(
        [
            {"split": "validation", "rows": len(validation), "symbols": validation["symbol"].nunique(), "start": validation["datetime"].min(), "end": validation["datetime"].max()},
            {"split": "test", "rows": len(test), "symbols": test["symbol"].nunique(), "start": test["datetime"].min(), "end": test["datetime"].max()},
        ]
    ).to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    xb_config = create_xb_config(config)
    rows = []
    candidates = create_candidates()
    for candidate in candidates:
        print(f"正在验证小市值状态候选：{candidate.name}", flush=True)
        rows.append(evaluate_candidate(candidate, validation, xb_config, run_dir))
    summary = pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_annual_return"], ascending=False)
    summary.to_csv(run_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected = next(candidate for candidate in candidates if candidate.name == str(summary.iloc[0]["candidate"]))
    validation_table = pd.read_csv(run_dir / f"validation_rules_{selected.name}.csv")
    selected_rule = create_selected_rule(validation_table.iloc[0])
    metrics = save_test_outputs(run_dir, selected, selected_rule, test, xb_config)
    write_report(run_dir, config, thresholds, summary, selected, selected_rule, metrics)
    print(f"全 A 小市值状态过滤策略结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行全 A 小市值状态过滤策略。"""

    run_all_a_small_cap_regime(parse_args())


if __name__ == "__main__":
    main()
