"""Audit and select no-lookahead factor features on a training window.

Required inputs are stable HS300 symbols, extracted minute OHLCV CSV files, and
the previous effective-factor report. Generated outputs are CSV/JSON/Markdown
reports under ``reports/factor_feature_audit``. The script evaluates features
only on the configured training window; validation/test periods are not used for
feature selection. It reads raw data and writes reports only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.factor_library import (
    build_factor_library,
    calculate_forward_return,
    infer_factor_metadata,
    validate_factor_frame,
    validate_no_future_leakage,
)


DATA_ROOT = PROJECT_ROOT / "ashare_minute_data_2023_2026_04" / "extracted"
STABLE_SYMBOLS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "stable_hs300_factor_correlation"
    / "run_20260518_223759"
    / "stable_hs300_symbols.csv"
)
PREVIOUS_FACTOR_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "stable_hs300_factor_correlation"
    / "run_20260518_223759"
    / "factor_mean_metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "factor_feature_audit"

TRAIN_START = "2023-01-03"
TRAIN_END = "2025-04-30"
FORWARD_HORIZON_BARS = 24
EVALUATION_STEP_BARS = 24
MIN_ASSETS_PER_TIME = 20
MIN_RANK_IC_PERIODS = 120
MAX_MISSING_RATIO = 0.40
MIN_ABS_RANK_IC = 0.001
MIN_MONTHLY_STABILITY = 0.52
CORRELATION_THRESHOLD = 0.95
CORRELATION_SAMPLE_ROWS = 100_000
WORKERS = min(4, os.cpu_count() or 1)


@dataclass(frozen=True)
class AuditConfig:
    """Resolved configuration for training-only feature audit."""

    data_root: Path
    stable_symbols_file: Path
    previous_factor_report: Path
    output_dir: Path
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forward_horizon_bars: int
    evaluation_step_bars: int
    min_assets_per_time: int
    min_rank_ic_periods: int
    max_missing_ratio: float
    min_abs_rank_ic: float
    min_monthly_stability: float
    correlation_threshold: float
    correlation_sample_rows: int
    max_symbols: int
    max_selected_features: int
    workers: int
    model_feature_transform: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the factor-feature audit run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--stable-symbols-file", type=Path, default=STABLE_SYMBOLS_FILE)
    parser.add_argument("--previous-factor-report", type=Path, default=PREVIOUS_FACTOR_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--horizon", type=int, default=FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=EVALUATION_STEP_BARS)
    parser.add_argument("--min-assets", type=int, default=MIN_ASSETS_PER_TIME)
    parser.add_argument("--min-rank-ic-periods", type=int, default=MIN_RANK_IC_PERIODS)
    parser.add_argument("--max-missing-ratio", type=float, default=MAX_MISSING_RATIO)
    parser.add_argument("--min-abs-rank-ic", type=float, default=MIN_ABS_RANK_IC)
    parser.add_argument("--min-monthly-stability", type=float, default=MIN_MONTHLY_STABILITY)
    parser.add_argument("--corr-threshold", type=float, default=CORRELATION_THRESHOLD)
    parser.add_argument("--corr-sample-rows", type=int, default=CORRELATION_SAMPLE_ROWS)
    parser.add_argument("--max-symbols", type=int, default=0, help="Debug cap. 0 means all stable symbols.")
    parser.add_argument("--max-selected-features", type=int, default=0, help="0 keeps all features passing filters.")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--model-feature-transform",
        choices=("cs_rank", "cs_zscore", "raw"),
        default="cs_rank",
        help="Feature-name transform used in the final model feature list.",
    )
    return parser.parse_args()


def parse_inclusive_end(value: str) -> pd.Timestamp:
    """Parse a date or timestamp, making date-only end bounds full-day inclusive."""

    parsed = pd.Timestamp(value)
    if len(str(value)) <= 10:
        return parsed + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return parsed


def create_config(args: argparse.Namespace) -> AuditConfig:
    """Validate CLI arguments and create an immutable config."""

    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.eval_step < 1:
        raise ValueError("--eval-step must be positive.")
    if not 0 <= args.max_missing_ratio < 1:
        raise ValueError("--max-missing-ratio must be in [0, 1).")
    if not 0 < args.corr_threshold <= 1:
        raise ValueError("--corr-threshold must be in (0, 1].")
    return AuditConfig(
        data_root=args.data_root,
        stable_symbols_file=args.stable_symbols_file,
        previous_factor_report=args.previous_factor_report,
        output_dir=args.output_dir,
        train_start=pd.Timestamp(args.train_start),
        train_end=parse_inclusive_end(args.train_end),
        forward_horizon_bars=args.horizon,
        evaluation_step_bars=args.eval_step,
        min_assets_per_time=args.min_assets,
        min_rank_ic_periods=args.min_rank_ic_periods,
        max_missing_ratio=args.max_missing_ratio,
        min_abs_rank_ic=args.min_abs_rank_ic,
        min_monthly_stability=args.min_monthly_stability,
        correlation_threshold=args.corr_threshold,
        correlation_sample_rows=max(int(args.corr_sample_rows), 0),
        max_symbols=max(int(args.max_symbols), 0),
        max_selected_features=max(int(args.max_selected_features), 0),
        workers=max(int(args.workers), 1),
        model_feature_transform=args.model_feature_transform,
    )


def discover_year_dirs(data_root: Path) -> list[Path]:
    """Return extracted year directories sorted by name."""

    dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not dirs:
        raise FileNotFoundError(f"No extracted year directories found under {data_root}")
    return dirs


def read_stable_symbols(path: Path, max_symbols: int) -> list[str]:
    """Read stable symbols from the previous stable-universe report."""

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "symbol" not in frame.columns:
        raise ValueError(f"Stable symbols file must contain a symbol column: {path}")
    symbols = frame["symbol"].astype(str).drop_duplicates().tolist()
    return symbols[:max_symbols] if max_symbols else symbols


def read_previous_factor_sets(path: Path) -> tuple[set[str], set[str], pd.DataFrame]:
    """Load previous all-factor and effective-factor sets for reuse checks."""

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if not {"factor", "is_effective_mean"}.issubset(frame.columns):
        raise ValueError(f"Previous factor report lacks required columns: {path}")
    all_factors = set(frame["factor"].astype(str))
    effective = set(frame.loc[frame["is_effective_mean"].astype(bool), "factor"].astype(str))
    return all_factors, effective, frame


def read_symbol_data(symbol: str, year_dirs: Sequence[Path], train_end: pd.Timestamp) -> pd.DataFrame:
    """Load one symbol's raw data up to train_end, preserving warm-up history."""

    parts: list[pd.DataFrame] = []
    for year_dir in year_dirs:
        path = year_dir / f"{symbol}.csv"
        if path.exists():
            part = pd.read_csv(path)
            part["symbol"] = symbol
            parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No files found for stable symbol {symbol}")
    frame = pd.concat(parts, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.loc[frame["datetime"] <= train_end].copy()
    numeric_cols = [col for col in ["open", "high", "low", "close", "volume", "amount"] if col in frame.columns]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("datetime").drop_duplicates(["symbol", "datetime"], keep="last").reset_index(drop=True)


def build_symbol_training_panel(args: tuple[str, list[Path], AuditConfig]) -> pd.DataFrame:
    """Build sampled training features and labels for one symbol."""

    symbol, year_dirs, config = args
    data = read_symbol_data(symbol, year_dirs, config.train_end)
    factors = build_factor_library(data)
    forward_return = calculate_forward_return(data, config.forward_horizon_bars)
    exit_datetime = data["datetime"].shift(-(config.forward_horizon_bars + 1))
    sampled = np.arange(len(data)) % config.evaluation_step_bars == 0
    keep = (
        sampled
        & data["datetime"].ge(config.train_start)
        & data["datetime"].le(config.train_end)
        & exit_datetime.le(config.train_end)
        & forward_return.notna()
    )
    base = data.loc[keep, ["datetime", "symbol"]].copy()
    base["forward_return"] = forward_return.loc[keep].to_numpy(dtype=float)
    feature_part = factors.loc[keep].astype("float32")
    return pd.concat([base.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1)


def calculate_rank_ic_series(data: pd.DataFrame, factor_col: str, min_assets: int) -> pd.Series:
    """Calculate cross-sectional Spearman RankIC by timestamp."""

    work = data[["datetime", factor_col, "forward_return"]].dropna().copy()
    if work.empty:
        return pd.Series(dtype=float)
    counts = work.groupby("datetime")[factor_col].transform("count")
    work = work.loc[counts >= min_assets].copy()
    if work.empty:
        return pd.Series(dtype=float)
    work["x"] = work.groupby("datetime")[factor_col].rank(method="average")
    work["y"] = work.groupby("datetime")["forward_return"].rank(method="average")
    work["xy"] = work["x"] * work["y"]
    work["x2"] = work["x"] * work["x"]
    work["y2"] = work["y"] * work["y"]
    stats = work.groupby("datetime").agg(
        n=("x", "count"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxy=("xy", "sum"),
        sx2=("x2", "sum"),
        sy2=("y2", "sum"),
        ux=("x", "nunique"),
        uy=("y", "nunique"),
    )
    numerator = stats["n"] * stats["sxy"] - stats["sx"] * stats["sy"]
    denom_x = stats["n"] * stats["sx2"] - stats["sx"] * stats["sx"]
    denom_y = stats["n"] * stats["sy2"] - stats["sy"] * stats["sy"]
    corr = numerator / np.sqrt(denom_x * denom_y).replace(0, np.nan)
    corr = corr.loc[(stats["ux"] >= 2) & (stats["uy"] >= 2)]
    return corr.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def calculate_feature_metrics(panel: pd.DataFrame, feature_cols: Sequence[str], config: AuditConfig) -> pd.DataFrame:
    """Calculate training-only RankIC, missing ratio, and stability metrics."""

    rows: list[dict[str, object]] = []
    total_rows = len(panel)
    for idx, factor in enumerate(feature_cols, start=1):
        series = panel[factor]
        missing_ratio = float(series.isna().mean()) if total_rows else 1.0
        ic = calculate_rank_ic_series(panel, factor, config.min_assets_per_time)
        ic_mean = float(ic.mean()) if len(ic) else np.nan
        ic_std = float(ic.std(ddof=0)) if len(ic) else np.nan
        ic_ir = float(ic_mean / ic_std * math.sqrt(len(ic))) if ic_std and not math.isnan(ic_std) else np.nan
        month_ic = ic.groupby(ic.index.to_period("M")).mean() if len(ic) else pd.Series(dtype=float)
        sign = np.sign(ic_mean) if pd.notna(ic_mean) else 0
        stability = float((np.sign(month_ic) == sign).mean()) if sign and len(month_ic) else np.nan
        rows.append(
            {
                "factor": factor,
                "train_missing_ratio": missing_ratio,
                "rank_ic_mean": ic_mean,
                "rank_ic_std": ic_std,
                "rank_ic_ir_like": ic_ir,
                "rank_ic_periods": int(len(ic)),
                "monthly_ic_periods": int(len(month_ic)),
                "monthly_sign_stability": stability,
                "selection_score": float(abs(ic_mean) * math.sqrt(max(len(ic), 1)) * (1 - missing_ratio) * (0.5 + (stability if pd.notna(stability) else 0)))
                if pd.notna(ic_mean)
                else 0.0,
            }
        )
        if idx % 50 == 0 or idx == len(feature_cols):
            print(f"calculated train RankIC for {idx}/{len(feature_cols)} raw factors", flush=True)
    return pd.DataFrame(rows)


def mark_prefilter_reasons(metrics: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Mark training-only prefilter pass/fail reasons."""

    frame = metrics.copy()
    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        reason_parts: list[str] = []
        if row.train_missing_ratio > config.max_missing_ratio:
            reason_parts.append("missing_ratio")
        if row.rank_ic_periods < config.min_rank_ic_periods:
            reason_parts.append("too_few_rank_ic_periods")
        if pd.isna(row.rank_ic_mean) or abs(row.rank_ic_mean) < config.min_abs_rank_ic:
            reason_parts.append("weak_abs_rank_ic")
        if pd.isna(row.monthly_sign_stability) or row.monthly_sign_stability < config.min_monthly_stability:
            reason_parts.append("unstable_monthly_ic_sign")
        reasons.append(";".join(reason_parts))
    frame["prefilter_pass"] = [not reason for reason in reasons]
    frame["removal_reason"] = reasons
    return frame


def apply_correlation_filter(panel: pd.DataFrame, metrics: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Select features using only training-set correlations."""

    frame = metrics.copy().sort_values("selection_score", ascending=False).reset_index(drop=True)
    eligible = frame.loc[frame["prefilter_pass"], "factor"].astype(str).tolist()
    if not eligible:
        frame["selected"] = False
        frame.loc[frame["removal_reason"].eq(""), "removal_reason"] = "no_prefilter_pass"
        return frame

    sample = panel[eligible].copy()
    if config.correlation_sample_rows and len(sample) > config.correlation_sample_rows:
        sample = sample.sample(config.correlation_sample_rows, random_state=42)
    ranked_sample = sample.rank(method="average", pct=True).astype("float32")
    corr = ranked_sample.corr(method="pearson", min_periods=max(config.min_assets_per_time, 30)).abs()

    selected: list[str] = []
    selected_set: set[str] = set()
    corr_reasons: dict[str, str] = {}
    for factor in eligible:
        if config.max_selected_features and len(selected) >= config.max_selected_features:
            corr_reasons[factor] = "max_selected_features"
            continue
        if selected:
            values = corr.loc[factor, selected].dropna()
            if not values.empty and float(values.max()) > config.correlation_threshold:
                blocker = str(values.idxmax())
                corr_reasons[factor] = f"high_train_corr_with:{blocker}:{float(values.max()):.4f}"
                continue
        selected.append(factor)
        selected_set.add(factor)

    frame["selected"] = frame["factor"].isin(selected_set)
    for idx, row in frame.iterrows():
        if row["selected"]:
            frame.at[idx, "removal_reason"] = ""
        elif not row["removal_reason"] and row["factor"] in corr_reasons:
            frame.at[idx, "removal_reason"] = corr_reasons[str(row["factor"])]
        elif not row["removal_reason"]:
            frame.at[idx, "removal_reason"] = "correlation_filter"
    return frame


def create_model_feature_name(base_factor: str, transform: str) -> str:
    """Map a selected base factor to the final model feature name."""

    if transform == "cs_rank":
        return f"cs_rank_{base_factor}"
    if transform == "cs_zscore":
        return f"cs_zscore_{base_factor}"
    return base_factor


def build_leakage_report(panel: pd.DataFrame, feature_cols: Sequence[str], symbols: Sequence[str], year_dirs: list[Path], config: AuditConfig) -> pd.DataFrame:
    """Run static and dynamic no-future checks and label-alignment checks."""

    rows = validate_factor_frame(panel.sort_values(["symbol", "datetime"]).reset_index(drop=True), feature_cols, label_cols=["forward_return"]).to_dict(orient="records")
    max_feature_dt = panel["datetime"].max()
    rows.append(
        {
            "check": "train_label_window_inside_train_period",
            "passed": bool(max_feature_dt <= config.train_end),
            "detail": f"max_feature_datetime={max_feature_dt};train_end={config.train_end}",
        }
    )
    for symbol in list(symbols)[:3]:
        data = read_symbol_data(symbol, year_dirs, config.train_end).head(1_000)
        result = validate_no_future_leakage(data)
        result["symbol"] = symbol
        rows.append(result)
    return pd.DataFrame(rows)


def write_source_references(path: Path) -> None:
    """Write external inspiration sources used by the expanded factor library."""

    text = """# Factor Source Notes

- Momentum and relative-strength inspiration: Jegadeesh and Titman (1993), "Returns to Buying Winners and Selling Losers", DOI `10.1111/j.1540-6261.1993.tb04702.x`.
- Low-volatility and downside-risk inspiration: Ang, Hodrick, Xing, and Zhang (2006), "The Cross-Section of Volatility and Expected Returns", DOI `10.1111/j.1540-6261.2006.00836.x`.
- Amihud illiquidity inspiration: Amihud (2002), "Illiquidity and Stock Returns", DOI `10.1016/S1386-4181(01)00024-6`.
- Parkinson range volatility inspiration: Parkinson (1980), "The Extreme Value Method for Estimating the Variance of the Rate of Return", DOI `10.1086/296071`.
- Garman-Klass range volatility inspiration: Garman and Klass (1980), "On the Estimation of Security Price Volatilities from Historical Data", DOI `10.1086/296072`.
- Formulaic price-volume transformations and correlation-aware alpha construction inspiration: Kakushadze (2016), "101 Formulaic Alphas", arXiv `1601.00991`.

All implemented features are rewritten against the local OHLCV/amount schema and use only current or historical completed bars.
"""
    path.write_text(text, encoding="utf-8")


def write_no_future_check_report(path: Path, config: AuditConfig, leakage_report: pd.DataFrame) -> None:
    """Write a human-readable no-future-function check summary."""

    status = "PASS" if bool(leakage_report["passed"].all()) else "FAIL"
    text = f"""# No-Future-Function Check

Status: {status}

Training window: {config.train_start} to {config.train_end}.
Label: signal-lagged forward return with entry at next bar close and exit after {config.forward_horizon_bars} bars.
Feature timestamps whose label exit would pass train_end are excluded from training metrics.

Checks performed:

- Feature names exclude `future_return`, `forward_return`, `target`, and `label` columns.
- All selected feature columns exist in the training panel.
- No duplicate `(symbol, datetime)` rows in the training panel.
- No infinite feature values after factor generation.
- Training panel sorted by `symbol, datetime` for validation.
- Label windows remain inside the training period.
- Dynamic future perturbation check: for three sample stable symbols, OHLCV rows strictly after a cutoff were changed and generated factor values up to the cutoff remained unchanged.

Alignment policy:

- Per-symbol rolling, lag, pct-change, trend, volatility, liquidity, and technical features use the current and historical completed bars only.
- Cross-sectional rank/z-score features group only by the same `datetime` and do not use future dates.
- Any generated feature based on bar t is marked as usable no earlier than t+1 trading/execution.
- Feature selection uses only training-window RankIC, missing ratio, monthly stability, and training-set correlation filtering. Test-set IC, Sharpe, or backtest results are not used.
"""
    path.write_text(text, encoding="utf-8")


def save_outputs(
    config: AuditConfig,
    panel: pd.DataFrame,
    manifest: pd.DataFrame,
    model_manifest: pd.DataFrame,
    leakage_report: pd.DataFrame,
    previous_effective: set[str],
    previous_all: set[str],
) -> Path:
    """Write audit outputs and resolved run context."""

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(run_dir / "factor_review_manifest.csv", index=False, encoding="utf-8-sig")
    manifest.loc[manifest["selected"]].to_csv(run_dir / "selected_base_factors.csv", index=False, encoding="utf-8-sig")
    model_manifest.to_csv(run_dir / "model_feature_manifest.csv", index=False, encoding="utf-8-sig")
    leakage_report.to_csv(run_dir / "leakage_check_report.csv", index=False, encoding="utf-8-sig")
    category_summary = manifest.groupby("category").agg(
        total=("factor", "count"),
        selected=("selected", "sum"),
        previous_effective_147=("was_previous_effective_147", "sum"),
        new_base_factors=("is_new_base_factor", "sum"),
    ).reset_index().sort_values("total", ascending=False)
    category_summary.to_csv(run_dir / "factor_category_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "feature_list.txt").write_text("\n".join(model_manifest["model_feature"].astype(str)) + "\n", encoding="utf-8")
    write_source_references(run_dir / "source_references.md")
    write_no_future_check_report(run_dir / "no_future_function_check.md", config, leakage_report)

    selected_base = set(manifest.loc[manifest["selected"], "factor"].astype(str))
    previous_retained = selected_base & previous_effective
    previous_removed = previous_effective - selected_base
    new_base = set(manifest.loc[manifest["is_new_base_factor"], "factor"].astype(str))
    new_selected = selected_base & new_base
    context = {
        "data_root": str(config.data_root),
        "stable_symbols_file": str(config.stable_symbols_file),
        "previous_factor_report": str(config.previous_factor_report),
        "train_start": str(config.train_start),
        "train_end": str(config.train_end),
        "forward_horizon_bars": int(config.forward_horizon_bars),
        "evaluation_step_bars": int(config.evaluation_step_bars),
        "train_rows": int(len(panel)),
        "symbol_count": int(panel["symbol"].nunique()),
        "previous_all_factor_count": int(len(previous_all)),
        "previous_effective_factor_count": int(len(previous_effective)),
        "retained_previous_effective_count": int(len(previous_retained)),
        "removed_previous_effective_count": int(len(previous_removed)),
        "new_base_factor_count": int(len(new_base)),
        "selected_new_base_factor_count": int(len(new_selected)),
        "raw_base_factor_count_after_expansion": int(len(manifest)),
        "selected_base_factor_count": int(len(selected_base)),
        "model_feature_count": int(len(model_manifest)),
        "selection_filters": {
            "max_missing_ratio": float(config.max_missing_ratio),
            "min_rank_ic_periods": int(config.min_rank_ic_periods),
            "min_abs_rank_ic": float(config.min_abs_rank_ic),
            "min_monthly_stability": float(config.min_monthly_stability),
            "correlation_threshold": float(config.correlation_threshold),
            "correlation_sample_rows": int(config.correlation_sample_rows),
        },
        "method_notes": [
            "Feature selection uses only the configured training window.",
            "Labels enter at the next bar close and exit after the configured horizon.",
            "Rows whose label exit would pass train_end are excluded from training metrics.",
            "Cross-sectional model feature names are monotonic same-timestamp transforms of selected base factors.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "feature_selection_summary.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def run_audit(config: AuditConfig) -> Path:
    """Run training-only factor audit and write output reports."""

    year_dirs = discover_year_dirs(config.data_root)
    symbols = read_stable_symbols(config.stable_symbols_file, config.max_symbols)
    previous_all, previous_effective, previous_report = read_previous_factor_sets(config.previous_factor_report)

    parts: list[pd.DataFrame] = []
    print(f"building sampled training panel for {len(symbols)} symbols with workers={config.workers}", flush=True)
    if config.workers == 1:
        for idx, symbol in enumerate(symbols, start=1):
            parts.append(build_symbol_training_panel((symbol, year_dirs, config)))
            if idx % 10 == 0 or idx == len(symbols):
                print(f"built panel for {idx}/{len(symbols)} symbols", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {executor.submit(build_symbol_training_panel, (symbol, year_dirs, config)): symbol for symbol in symbols}
            for idx, future in enumerate(as_completed(futures), start=1):
                parts.append(future.result())
                if idx % 10 == 0 or idx == len(symbols):
                    print(f"built panel for {idx}/{len(symbols)} symbols", flush=True)

    panel = pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    feature_cols = [col for col in panel.select_dtypes(include=[np.number]).columns if col != "forward_return"]
    metrics = calculate_feature_metrics(panel, feature_cols, config)
    metrics = mark_prefilter_reasons(metrics, config)
    metrics = apply_correlation_filter(panel, metrics, config)

    metadata = infer_factor_metadata(metrics["factor"].astype(str).tolist())
    manifest = metadata.merge(metrics, on="factor", how="left")
    manifest["existed_in_previous_library"] = manifest["factor"].isin(previous_all)
    manifest["was_previous_effective_147"] = manifest["factor"].isin(previous_effective)
    manifest["is_new_base_factor"] = ~manifest["existed_in_previous_library"]
    previous_cols = [
        "factor",
        "mean_directional_rank_ic",
        "mean_excess_return",
        "positive_excess_return_rate",
        "is_effective_mean",
    ]
    manifest = manifest.merge(previous_report[previous_cols], on="factor", how="left", suffixes=("", "_previous_full_sample"))

    selected = manifest.loc[manifest["selected"]].copy()
    selected["model_feature"] = selected["factor"].map(lambda name: create_model_feature_name(str(name), config.model_feature_transform))
    model_manifest = infer_factor_metadata(selected["model_feature"].astype(str).tolist())
    model_manifest = model_manifest.merge(
        selected[
            [
                "factor",
                "model_feature",
                "rank_ic_mean",
                "rank_ic_ir_like",
                "train_missing_ratio",
                "monthly_sign_stability",
                "selection_score",
                "was_previous_effective_147",
                "is_new_base_factor",
            ]
        ],
        left_on="factor",
        right_on="model_feature",
        how="left",
        suffixes=("", "_base"),
    )
    leakage_report = build_leakage_report(panel, feature_cols, symbols, year_dirs, config)
    return save_outputs(config, panel, manifest, model_manifest, leakage_report, previous_effective, previous_all)


def main() -> None:
    """Run the factor-feature audit from CLI arguments."""

    config = create_config(parse_args())
    run_dir = run_audit(config)
    summary = json.loads((run_dir / "feature_selection_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved factor-feature audit report to {run_dir}")


if __name__ == "__main__":
    main()
