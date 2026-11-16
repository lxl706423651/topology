"""运行全 A 小市值开盘反转策略回测。

输入为全 A 小市值代理采样面板 pkl。脚本只取每天 09:55 附近的开盘后
短期信号，用下一分钟价格入场，并用下一交易日同一入场时点价格作为退出
估值，避免把 A 股 T+1 场景简化成日内平仓。候选策略来自 A 股日内反转、
短期反转、小市值、低波动和流动性状态线索；状态阈值、候选和组合规则只
在验证集选择，测试集只做一次最终验收。输出验证表、测试收益曲线、逐笔
交易和中文报告，不修改原始分钟行情。
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
OUTPUT_DIR = PROJECT_ROOT / "reports" / "all_a_small_cap_open_reversal_backtest"


@dataclass(frozen=True)
class OpenReversalCandidate:
    """小市值开盘反转候选。"""

    name: str
    score_col: str
    regime_col: str
    rationale: str


@dataclass(frozen=True)
class OpenReversalConfig:
    """全 A 小市值开盘反转策略配置。"""

    panel_cache: Path
    output_dir: Path
    signal_time: str
    validation_start: pd.Timestamp
    split_date: pd.Timestamp
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
    """解析小市值开盘反转策略参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-cache", type=Path, default=PANEL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--signal-time", default="09:55")
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
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


def create_config(args: argparse.Namespace) -> OpenReversalConfig:
    """校验参数并生成配置对象。"""

    return OpenReversalConfig(
        panel_cache=args.panel_cache,
        output_dir=args.output_dir,
        signal_time=str(args.signal_time),
        validation_start=pd.Timestamp(args.validation_start),
        split_date=pd.Timestamp(args.split_date),
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


def create_xb_config(config: OpenReversalConfig) -> xb.ExperimentConfig:
    """把开盘反转配置转换为主组合回测引擎兼容配置。"""

    return xb.ExperimentConfig(
        data_root=xb.DATA_ROOT,
        stable_symbols_file=Path(""),
        feature_manifest=Path(""),
        output_dir=config.output_dir,
        feature_panel_cache_dir=config.output_dir / "panel_cache",
        split_date=config.split_date,
        validation_start=config.validation_start,
        forward_horizon_bars=240,
        evaluation_step_bars=240,
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


def read_open_panel(config: OpenReversalConfig) -> pd.DataFrame:
    """读取采样面板并构造次交易日退出标签。"""

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
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["entry_datetime"] = pd.to_datetime(frame["entry_datetime"])
    frame = frame.loc[frame["datetime"].dt.strftime("%H:%M") == config.signal_time].copy()
    frame = frame.sort_values(["symbol", "datetime"])
    frame["exit_price"] = frame.groupby("symbol", sort=False)["entry_price"].shift(-1)
    frame["label_exit_datetime"] = frame.groupby("symbol", sort=False)["entry_datetime"].shift(-1)
    frame["forward_return"] = (frame["exit_price"] / frame["entry_price"] - 1.0).astype("float32")
    frame = frame.dropna(subset=["entry_price", "exit_price", "label_exit_datetime", "forward_return"])
    return frame.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def add_open_reversal_features(frame: pd.DataFrame, config: OpenReversalConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """添加开盘反转候选分数和状态门控，状态阈值只用验证集估计。"""

    out = frame.copy()
    price_rank = (1.0 - out["low_price_score"]).clip(0.0, 1.0)
    out["price_health_score"] = (1.0 - ((price_rank - 0.35).abs() / 0.35)).clip(0.0, 1.0).astype("float32")
    out["liquidity_size_score"] = (
        0.65 * out["low_amount_score"].fillna(0.5)
        + 0.35 * out["illiquidity_score"].fillna(0.5)
    ).astype("float32")
    out["morning_momentum_score"] = out.groupby("datetime")["return_24"].rank(method="average", pct=True).astype("float32")
    out["open_reversal_small_score"] = (
        0.55 * out["reversal_24_score"].fillna(0.5)
        + 0.30 * out["liquidity_size_score"].fillna(0.5)
        + 0.15 * out["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    out["open_tradable_reversal_score"] = (
        0.45 * out["reversal_24_score"].fillna(0.5)
        + 0.25 * out["liquidity_size_score"].fillna(0.5)
        + 0.20 * out["price_health_score"].fillna(0.5)
        + 0.10 * out["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    out["open_momentum_small_score"] = (
        0.50 * out["morning_momentum_score"].fillna(0.5)
        + 0.30 * out["liquidity_size_score"].fillna(0.5)
        + 0.20 * out["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    state = (
        out.groupby("datetime", sort=False)
        .agg(
            open_market_return=("return_24", "mean"),
            open_breadth=("return_24", lambda values: float((values > 0).mean())),
            open_volatility=("volatility_120", "mean"),
            market_liquidity=("amount_ma_120", "median"),
        )
        .reset_index()
    )
    validation_state = state.loc[(state["datetime"] >= config.validation_start) & (state["datetime"] < config.split_date)]
    thresholds = {
        "open_market_return_q40": float(validation_state["open_market_return"].quantile(0.40)),
        "open_market_return_q50": float(validation_state["open_market_return"].quantile(0.50)),
        "open_market_return_q60": float(validation_state["open_market_return"].quantile(0.60)),
        "open_breadth_q45": float(validation_state["open_breadth"].quantile(0.45)),
        "open_breadth_q55": float(validation_state["open_breadth"].quantile(0.55)),
        "open_volatility_q70": float(validation_state["open_volatility"].quantile(0.70)),
        "market_liquidity_q40": float(validation_state["market_liquidity"].quantile(0.40)),
    }
    state["always_regime"] = True
    state["opening_selloff_regime"] = state["open_market_return"] <= thresholds["open_market_return_q40"]
    state["opening_stable_reversal_regime"] = (
        (state["open_volatility"] <= thresholds["open_volatility_q70"])
        & (state["market_liquidity"] >= thresholds["market_liquidity_q40"])
    )
    state["opening_risk_on_regime"] = (
        (state["open_market_return"] >= thresholds["open_market_return_q50"])
        & (state["open_breadth"] >= thresholds["open_breadth_q55"])
    )
    state["opening_rebound_regime"] = (
        (state["open_market_return"] <= thresholds["open_market_return_q50"])
        & (state["open_breadth"] >= thresholds["open_breadth_q45"])
        & (state["open_volatility"] <= thresholds["open_volatility_q70"])
    )
    out = out.merge(
        state[
            [
                "datetime",
                "open_market_return",
                "open_breadth",
                "open_volatility",
                "market_liquidity",
                "always_regime",
                "opening_selloff_regime",
                "opening_stable_reversal_regime",
                "opening_risk_on_regime",
                "opening_rebound_regime",
            ]
        ],
        on="datetime",
        how="left",
        validate="many_to_one",
    )
    return out, thresholds


def create_candidates() -> list[OpenReversalCandidate]:
    """返回小市值开盘反转候选集合。"""

    return [
        OpenReversalCandidate("open_reversal_small", "open_reversal_small_score", "always_regime", "每日 09:55 买入开盘后弱势的小市值代理股票，次交易日同一时点退出。"),
        OpenReversalCandidate("open_reversal_stable", "open_reversal_small_score", "opening_stable_reversal_regime", "开盘反转小市值策略只在市场波动不过高且流动性不差时交易。"),
        OpenReversalCandidate("open_tradable_reversal", "open_tradable_reversal_score", "opening_stable_reversal_regime", "开盘反转小市值策略加入价格健康度，避免极端低价股票。"),
        OpenReversalCandidate("open_selloff_reversal", "open_reversal_small_score", "opening_selloff_regime", "只在全市场开盘后偏弱时买入小市值短期反转组合。"),
        OpenReversalCandidate("open_rebound_reversal", "open_reversal_small_score", "opening_rebound_regime", "只在市场开盘偏弱但广度和波动允许反弹时做小市值反转。"),
        OpenReversalCandidate("open_momentum_risk_on", "open_momentum_small_score", "opening_risk_on_regime", "市场开盘强势时做小市值开盘动量，而不是反转。"),
    ]


def prepare_candidate_frame(frame: pd.DataFrame, candidate: OpenReversalCandidate) -> pd.DataFrame:
    """按候选状态门控生成 prediction，状态不满足时不交易。"""

    out = frame.copy()
    active = out[candidate.regime_col].fillna(False).astype(bool)
    out["prediction"] = out[candidate.score_col].where(active).astype("float32")
    return out


def evaluate_candidate(candidate: OpenReversalCandidate, validation: pd.DataFrame, xb_config: xb.ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """只用验证集评估一个开盘反转候选。"""

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


def save_test_outputs(run_dir: Path, candidate: OpenReversalCandidate, selected_rule: xb.PortfolioRule, test: pd.DataFrame, xb_config: xb.ExperimentConfig) -> dict[str, Any]:
    """保存验证集选中候选在测试集上的最终回测产物。"""

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
    config: OpenReversalConfig,
    thresholds: dict[str, float],
    summary: pd.DataFrame,
    selected: OpenReversalCandidate,
    selected_rule: xb.PortfolioRule,
    metrics: dict[str, Any],
) -> None:
    """写出中文小市值开盘反转策略报告。"""

    reached = float(metrics.get("sharpe", 0.0)) >= config.sharpe_target
    threshold_lines = [f"- `{key}`：{value:.8f}" for key, value in thresholds.items()]
    lines = [
        "# 全 A 小市值开盘反转策略回测报告",
        "",
        "## 研究边界",
        "",
        f"- 信号时间：`{config.signal_time}`。",
        "- 入场价格使用下一分钟价格，退出价格使用下一交易日同一入场时点价格。",
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
    (run_dir / "all_a_small_cap_open_reversal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_a_small_cap_open_reversal(args: argparse.Namespace) -> Path:
    """执行全 A 小市值开盘反转策略验证和测试。"""

    config = create_config(args)
    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    panel, thresholds = add_open_reversal_features(read_open_panel(config), config)
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
        print(f"正在验证小市值开盘候选：{candidate.name}", flush=True)
        rows.append(evaluate_candidate(candidate, validation, xb_config, run_dir))
    summary = pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_annual_return"], ascending=False)
    summary.to_csv(run_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected = next(candidate for candidate in candidates if candidate.name == str(summary.iloc[0]["candidate"]))
    validation_table = pd.read_csv(run_dir / f"validation_rules_{selected.name}.csv")
    selected_rule = create_selected_rule(validation_table.iloc[0])
    metrics = save_test_outputs(run_dir, selected, selected_rule, test, xb_config)
    write_report(run_dir, config, thresholds, summary, selected, selected_rule, metrics)
    print(f"全 A 小市值开盘反转策略结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行全 A 小市值开盘反转策略。"""

    run_all_a_small_cap_open_reversal(parse_args())


if __name__ == "__main__":
    main()
