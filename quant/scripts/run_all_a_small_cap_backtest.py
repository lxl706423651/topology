"""运行全 A 股票池小市值策略验证和测试回测。

输入为本地全 A 分钟 OHLCV 数据、AKShare 日频流通市值特征缓存和可配置的
时间切分。脚本读取验证期和测试期分钟数据，按 signal_lag=1 生成未来收益，
并把分钟信号对齐到上一交易日可知的流通市值。策略只在内部验证集上选择
持仓数量、持仓门控和小市值/流动性/反转候选，测试集只做一次最终验收。
输出包括采样面板缓存、验证候选表、测试收益曲线和逐笔交易记录。不会修改
原始分钟行情。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_xgboost_long_only_backtest as xb
from scripts.tune_xgboost_long_only_parameters import create_selected_rule


OUTPUT_DIR = PROJECT_ROOT / "reports" / "all_a_small_cap_backtest"
DAILY_SIZE_FEATURE_CACHE = PROJECT_ROOT / "reports" / "daily_size_features_all_a" / "latest" / "daily_size_features.pkl"


@dataclass(frozen=True)
class SmallCapCandidate:
    """全 A 小市值候选策略。"""

    name: str
    score_col: str
    rationale: str


@dataclass(frozen=True)
class AllASmallCapConfig:
    """全 A 小市值回测配置。"""

    data_root: Path
    daily_size_feature_cache: Path
    output_dir: Path
    size_feature_mode: str
    validation_start: pd.Timestamp
    split_date: pd.Timestamp
    end_date: pd.Timestamp
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
    max_symbols: int
    workers: int
    random_seed: int
    rebuild_panel: bool
    panel_cache_dir: Path


def parse_args() -> argparse.Namespace:
    """解析全 A 小市值策略参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=xb.DATA_ROOT)
    parser.add_argument("--daily-size-feature-cache", type=Path, default=DAILY_SIZE_FEATURE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--panel-cache-dir", type=Path, default=OUTPUT_DIR / "panel_cache")
    parser.add_argument("--size-feature-mode", choices=["real", "proxy", "auto"], default="auto", help="real 使用日频流通市值，proxy 只用本地分钟代理，auto 在真实市值覆盖不足时降级。")
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--commission-rate", type=float, default=xb.COMMISSION_RATE)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--top-n-candidates", default="1,2,3,5,10,20")
    parser.add_argument("--top-quantile-candidates", default="0.002,0.005,0.01")
    parser.add_argument("--min-assets", type=int, default=200)
    parser.add_argument("--min-holding-count", type=int, default=1)
    parser.add_argument("--hold-steps-candidates", default="1")
    parser.add_argument("--target-exposure-candidates", default="1.0")
    parser.add_argument("--confidence-quantile-candidates", default="none,0.85,0.9")
    parser.add_argument("--spread-quantile-candidates", default="none,0.6,0.7")
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--sharpe-target", type=float, default=3.0)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--workers", type=int, default=xb.WORKERS)
    parser.add_argument("--seed", type=int, default=xb.RANDOM_SEED)
    parser.add_argument("--rebuild-panel", action="store_true")
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> AllASmallCapConfig:
    """校验参数并生成配置对象。"""

    return AllASmallCapConfig(
        data_root=args.data_root,
        daily_size_feature_cache=args.daily_size_feature_cache,
        output_dir=args.output_dir,
        size_feature_mode=str(args.size_feature_mode),
        validation_start=pd.Timestamp(args.validation_start),
        split_date=pd.Timestamp(args.split_date),
        end_date=pd.Timestamp(args.end_date),
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
        max_symbols=max(int(args.max_symbols), 0),
        workers=max(int(args.workers), 1),
        random_seed=int(args.seed),
        rebuild_panel=bool(args.rebuild_panel),
        panel_cache_dir=args.panel_cache_dir,
    )


def create_xb_config(config: AllASmallCapConfig) -> xb.ExperimentConfig:
    """把全 A 小市值配置转换为主回测引擎兼容配置。"""

    return xb.ExperimentConfig(
        data_root=config.data_root,
        stable_symbols_file=Path(""),
        feature_manifest=Path(""),
        output_dir=config.output_dir,
        feature_panel_cache_dir=config.panel_cache_dir,
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
        max_symbols=config.max_symbols,
        workers=config.workers,
        random_seed=config.random_seed,
        rebuild_feature_panel=False,
        no_feature_panel_cache=False,
        build_feature_panel_only=False,
    )


def discover_local_symbols(data_root: Path, max_symbols: int) -> list[str]:
    """读取本地 2025/2026 都存在的股票代码。"""

    year_sets = []
    for year in ["2025", "2026"]:
        year_dir = data_root / year
        if not year_dir.exists():
            raise FileNotFoundError(f"找不到年份目录：{year_dir}")
        year_sets.append({path.stem for path in year_dir.glob("*.csv")})
    symbols = sorted(set.intersection(*year_sets))
    if max_symbols:
        symbols = symbols[:max_symbols]
    return symbols


def read_daily_size(path: Path) -> pd.DataFrame:
    """读取日频流通市值特征，并保留对齐所需列。"""

    frame = pd.read_pickle(path)
    frame = frame.loc[:, ["symbol", "date", "log_float_market_cap", "derived_float_market_cap"]].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["symbol", "date_ts"])


def read_symbol_sample(symbol: str, config: AllASmallCapConfig) -> pd.DataFrame:
    """读取单只股票验证/测试期分钟数据并生成 signal-lag 标签。"""

    frames = []
    for year in sorted({str(config.validation_start.year), str(config.split_date.year), str(config.end_date.year)}):
        path = config.data_root / year / f"{symbol}.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path, usecols=["datetime", "close", "volume", "amount"])
        data["datetime"] = pd.to_datetime(data["datetime"])
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True).drop_duplicates("datetime").sort_values("datetime")
    data = data.loc[
        (data["datetime"] >= config.validation_start - pd.Timedelta(days=3))
        & (data["datetime"] <= config.end_date + pd.Timedelta(days=1))
    ].copy()
    if len(data) <= config.forward_horizon_bars + 2:
        return pd.DataFrame()
    close = data["close"].astype(float)
    amount = pd.to_numeric(data["amount"], errors="coerce").astype(float)
    volume = pd.to_numeric(data["volume"], errors="coerce").astype(float)
    minute_return = close.pct_change()
    data["amount_sum_24"] = amount.rolling(24, min_periods=12).sum().astype("float32")
    data["amount_ma_120"] = amount.rolling(120, min_periods=60).mean().astype("float32")
    data["volume_ma_120"] = volume.rolling(120, min_periods=60).mean().astype("float32")
    data["return_24"] = close.pct_change(24).astype("float32")
    data["return_120"] = close.pct_change(120).astype("float32")
    data["volatility_120"] = minute_return.rolling(120, min_periods=60).std().astype("float32")
    entry_price = close.shift(-1)
    exit_price = close.shift(-(config.forward_horizon_bars + 1))
    entry_datetime = data["datetime"].shift(-1)
    label_exit_datetime = data["datetime"].shift(-(config.forward_horizon_bars + 1))
    keep = (
        entry_price.notna()
        & exit_price.notna()
        & entry_datetime.notna()
        & label_exit_datetime.notna()
        & (entry_price > 0)
    )
    base = data.loc[
        keep,
        [
            "datetime",
            "close",
            "volume",
            "amount",
            "amount_sum_24",
            "amount_ma_120",
            "volume_ma_120",
            "return_24",
            "return_120",
            "volatility_120",
        ],
    ].copy()
    base["symbol"] = symbol
    base["entry_datetime"] = pd.to_datetime(entry_datetime.loc[keep]).to_numpy()
    base["entry_price"] = entry_price.loc[keep].to_numpy(dtype="float32")
    base["exit_price"] = exit_price.loc[keep].to_numpy(dtype="float32")
    base["label_exit_datetime"] = pd.to_datetime(label_exit_datetime.loc[keep]).to_numpy()
    base["forward_return"] = (base["exit_price"] / base["entry_price"] - 1.0).astype("float32")
    base = base.loc[base["datetime"] >= config.validation_start].copy()
    if base.empty:
        return base
    sampled = base.iloc[:: config.evaluation_step_bars].copy()
    return sampled


def build_or_load_panel(config: AllASmallCapConfig) -> pd.DataFrame:
    """构建或读取全 A 小市值采样面板缓存。"""

    cache_name = (
        f"all_a_small_cap_panel_v2_{config.size_feature_mode}_{config.validation_start:%Y%m%d}_{config.end_date:%Y%m%d}_"
        f"h{config.forward_horizon_bars}_s{config.evaluation_step_bars}_m{config.max_symbols}.pkl"
    )
    config.panel_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.panel_cache_dir / cache_name
    if cache_path.exists() and not config.rebuild_panel:
        print(f"正在读取全 A 小市值面板缓存：{cache_path}", flush=True)
        return pd.read_pickle(cache_path)

    symbols = discover_local_symbols(config.data_root, config.max_symbols)
    print(f"正在构建全 A 小市值面板：symbols={len(symbols)}", flush=True)
    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(read_symbol_sample, symbol, config): symbol for symbol in symbols}
        for idx, future in enumerate(as_completed(futures), start=1):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
            if idx % 250 == 0 or idx == len(symbols):
                print(f"已处理分钟采样 {idx}/{len(symbols)}", flush=True)
    if not frames:
        raise RuntimeError("没有构建出任何全 A 小市值样本。")
    panel = pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"])
    used_real_size = False
    panel["date_ts"] = panel["datetime"].dt.normalize()
    if config.size_feature_mode != "proxy":
        if not config.daily_size_feature_cache.exists():
            if config.size_feature_mode == "real":
                raise FileNotFoundError(f"找不到日频市值缓存：{config.daily_size_feature_cache}")
            print(f"未找到日频市值缓存，自动降级为本地代理：{config.daily_size_feature_cache}", flush=True)
        else:
            size = read_daily_size(config.daily_size_feature_cache)
            merged_frames = []
            for symbol, group in panel.groupby("symbol", sort=False):
                daily = size.loc[size["symbol"] == symbol]
                if daily.empty:
                    continue
                merged = pd.merge_asof(
                    group.sort_values("date_ts"),
                    daily[["date_ts", "log_float_market_cap", "derived_float_market_cap"]].sort_values("date_ts"),
                    on="date_ts",
                    direction="backward",
                    allow_exact_matches=False,
                )
                merged_frames.append(merged)
            if merged_frames:
                real_panel = pd.concat(merged_frames, ignore_index=True).dropna(subset=["log_float_market_cap"])
                real_symbol_count = int(real_panel["symbol"].nunique())
                if real_symbol_count >= config.min_assets_per_rebalance:
                    panel = real_panel
                    used_real_size = True
                elif config.size_feature_mode == "real":
                    raise ValueError(
                        f"日频市值覆盖股票数 {real_symbol_count} 小于 min_assets={config.min_assets_per_rebalance}，"
                        "不能进行真实小市值回测。"
                    )
                else:
                    print(
                        f"日频市值覆盖股票数 {real_symbol_count} 不足，自动降级为本地代理。",
                        flush=True,
                    )
            elif config.size_feature_mode == "real":
                raise ValueError("日频市值缓存没有覆盖任何当前股票。")

    panel["low_price_score"] = (1.0 - panel.groupby("datetime")["close"].rank(method="average", pct=True)).astype("float32")
    panel["low_amount_score"] = (
        1.0 - panel.groupby("datetime")["amount_ma_120"].rank(method="average", pct=True)
    ).astype("float32")
    illiquidity_raw = panel["return_24"].abs() / np.log1p(panel["amount_sum_24"].clip(lower=0))
    panel["_illiquidity_raw"] = illiquidity_raw.replace([np.inf, -np.inf], np.nan).astype("float32")
    panel["illiquidity_score"] = panel.groupby("datetime")["_illiquidity_raw"].rank(method="average", pct=True).astype("float32")
    panel["reversal_24_score"] = (1.0 - panel.groupby("datetime")["return_24"].rank(method="average", pct=True)).astype("float32")
    panel["momentum_120_score"] = panel.groupby("datetime")["return_120"].rank(method="average", pct=True).astype("float32")
    panel["low_volatility_score"] = (
        1.0 - panel.groupby("datetime")["volatility_120"].rank(method="average", pct=True)
    ).astype("float32")
    if used_real_size:
        panel["small_cap_score"] = (
            1.0 - panel.groupby("datetime")["log_float_market_cap"].rank(method="average", pct=True)
        ).astype("float32")
    else:
        panel["log_float_market_cap"] = np.nan
        panel["derived_float_market_cap"] = np.nan
        panel["small_cap_score"] = (
            0.50 * panel["low_price_score"].fillna(0.5)
            + 0.35 * panel["low_amount_score"].fillna(0.5)
            + 0.15 * panel["illiquidity_score"].fillna(0.5)
        ).astype("float32")
    panel["low_liquidity_score"] = panel["low_amount_score"].astype("float32")
    panel["small_cap_liquidity_score"] = (
        0.70 * panel["small_cap_score"].fillna(0.5)
        + 0.30 * panel["illiquidity_score"].fillna(0.5)
    ).astype("float32")
    panel["small_cap_reversal_score"] = (
        0.65 * panel["small_cap_score"].fillna(0.5)
        + 0.25 * panel["reversal_24_score"].fillna(0.5)
        + 0.10 * panel["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    panel["small_cap_momentum_quality_score"] = (
        0.55 * panel["small_cap_score"].fillna(0.5)
        + 0.25 * panel["momentum_120_score"].fillna(0.5)
        + 0.20 * panel["low_volatility_score"].fillna(0.5)
    ).astype("float32")
    panel.to_pickle(cache_path)
    metadata = {
        "rows": int(len(panel)),
        "symbol_count": int(panel["symbol"].nunique()),
        "start": str(panel["datetime"].min()),
        "end": str(panel["datetime"].max()),
        "cache_path": str(cache_path),
        "daily_size_feature_cache": str(config.daily_size_feature_cache),
        "size_feature_mode": config.size_feature_mode,
        "used_real_size": used_real_size,
    }
    (cache_path.with_suffix(".metadata.json")).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return panel


def create_candidates() -> list[SmallCapCandidate]:
    """返回全 A 小市值候选策略。"""

    return [
        SmallCapCandidate("small_cap_pure", "small_cap_score", "优先按上一交易日流通市值从小到大排序；真实市值不可用时使用低价格、低成交额和非流动性代理。"),
        SmallCapCandidate("small_cap_low_liquidity", "small_cap_liquidity_score", "小市值暴露为主，叠加非流动性暴露。"),
        SmallCapCandidate("small_cap_reversal", "small_cap_reversal_score", "小市值暴露为主，叠加短期反转和低波动过滤。"),
        SmallCapCandidate("small_cap_momentum_quality", "small_cap_momentum_quality_score", "小市值暴露为主，叠加中短期动量和低波动过滤。"),
    ]


def evaluate_candidate(candidate: SmallCapCandidate, validation: pd.DataFrame, xb_config: xb.ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """只用验证集评估一个全 A 小市值候选。"""

    frame = validation.copy()
    frame["prediction"] = frame[candidate.score_col].astype("float32")
    table = xb.evaluate_rules(frame, xb_config)
    table.to_csv(run_dir / f"validation_rules_{candidate.name}.csv", index=False, encoding="utf-8-sig")
    best = table.iloc[0]
    row = {
        "candidate": candidate.name,
        "score_col": candidate.score_col,
        "rationale": candidate.rationale,
        "best_rule": best["rule"],
    }
    for key in ["sharpe", "annual_return", "annual_volatility", "max_drawdown", "total_return", "avg_turnover", "avg_gross_exposure", "round_trip_win_rate", "trade_count"]:
        row[f"validation_{key}"] = best.get(key)
    return row


def save_test_outputs(run_dir: Path, candidate: SmallCapCandidate, selected_rule: xb.PortfolioRule, test: pd.DataFrame, xb_config: xb.ExperimentConfig) -> dict[str, Any]:
    """保存验证选中候选的最终测试回测产物。"""

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


def write_report(run_dir: Path, config: AllASmallCapConfig, summary: pd.DataFrame, selected: SmallCapCandidate, selected_rule: xb.PortfolioRule, metrics: dict[str, Any]) -> None:
    """写出中文全 A 小市值回测报告。"""

    reached = float(metrics.get("sharpe", 0.0)) >= config.sharpe_target
    lines = [
        "# 全 A 小市值策略回测报告",
        "",
        "## 边界",
        "",
        "- 股票池来自本地全 A 分钟数据中 2025/2026 都存在的股票。",
        f"- 小市值特征模式：`{config.size_feature_mode}`。`real` 使用上一交易日 AKShare 日频行情反推流通市值；`proxy` 只使用本地分钟数据代理；`auto` 在真实市值覆盖不足时自动降级为代理。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`。",
        f"- 测试集：`datetime >= {config.split_date.date()}`，只用于最终验收。",
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
    (run_dir / "all_a_small_cap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_a_small_cap(args: argparse.Namespace) -> Path:
    """执行全 A 小市值策略验证和测试。"""

    config = create_config(args)
    panel = build_or_load_panel(config)
    validation = panel.loc[(panel["datetime"] >= config.validation_start) & (panel["datetime"] < config.split_date)].copy()
    test = panel.loc[panel["datetime"] >= config.split_date].copy()
    if validation.empty or test.empty:
        raise ValueError(f"切分后样本为空：validation={len(validation)} test={len(test)}")
    xb_config = create_xb_config(config)
    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"split": "validation", "rows": len(validation), "symbols": validation["symbol"].nunique(), "start": validation["datetime"].min(), "end": validation["datetime"].max()},
            {"split": "test", "rows": len(test), "symbols": test["symbol"].nunique(), "start": test["datetime"].min(), "end": test["datetime"].max()},
        ]
    ).to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    candidates = create_candidates()
    rows = []
    for candidate in candidates:
        print(f"正在验证全 A 小市值候选：{candidate.name}", flush=True)
        rows.append(evaluate_candidate(candidate, validation, xb_config, run_dir))
    summary = pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_annual_return"], ascending=False)
    summary.to_csv(run_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected = next(candidate for candidate in candidates if candidate.name == str(summary.iloc[0]["candidate"]))
    validation_table = pd.read_csv(run_dir / f"validation_rules_{selected.name}.csv")
    selected_rule = create_selected_rule(validation_table.iloc[0])
    metrics = save_test_outputs(run_dir, selected, selected_rule, test, xb_config)
    write_report(run_dir, config, summary, selected, selected_rule, metrics)
    print(f"全 A 小市值策略结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行全 A 小市值策略。"""

    run_all_a_small_cap(parse_args())


if __name__ == "__main__":
    main()
