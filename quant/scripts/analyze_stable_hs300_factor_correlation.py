"""Analyze factors on stable HS300 stocks and report factor correlations.

Required inputs are extracted A-share minute OHLCV CSV files and an HS300
universe CSV with a ``symbol`` column. Generated outputs are CSV/JSON reports
under ``reports/stable_hs300_factor_correlation``. The script reads raw market
data and writes reports only; it does not mutate data, train models, or place
trades. Run it from the project root, preferably inside the ``quant`` conda
environment.
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
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.factor_library import build_factor_library, calculate_forward_return, has_talib


DATA_ROOT = PROJECT_ROOT / "ashare_minute_data_2023_2026_04" / "extracted"
UNIVERSE_FILE = PROJECT_ROOT / "config" / "hs300_current_constituents.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "stable_hs300_factor_correlation"

START_DATE: str | None = None
END_DATE: str | None = None
FORWARD_HORIZON_BARS = 24
EVALUATION_STEP_BARS = 24
TOP_QUANTILE = 0.2
HIGH_CORRELATION_THRESHOLD = 0.80
MIN_FACTOR_OBSERVATIONS = 120
MIN_CORRELATION_OBSERVATIONS = 120
MIN_SYMBOLS_PER_FACTOR = 5
STABLE_MODE = "modal_grid"
WORKERS = min(4, os.cpu_count() or 1)
CSV_ENCODING: str | None = None

KEY_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class StableFactorConfig:
    """Resolved configuration for stable-universe factor analysis.

    Parameters map to top-level defaults and CLI overrides. Date bounds are
    optional; when omitted, every available extracted year under ``data_root``
    is included.
    """

    data_root: Path
    universe_file: Path
    output_dir: Path
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    forward_horizon_bars: int
    evaluation_step_bars: int
    top_quantile: float
    high_correlation_threshold: float
    min_factor_observations: int
    min_correlation_observations: int
    min_symbols_per_factor: int
    stable_mode: str
    workers: int
    max_symbols: int


@dataclass
class SymbolScan:
    """Data quality scan result for one candidate symbol."""

    symbol: str
    row_count: int
    unique_datetime_count: int
    first_datetime: str | None
    last_datetime: str | None
    missing_file_count: int
    missing_files: str
    null_key_row_count: int
    duplicate_datetime_count: int
    invalid_ohlc_count: int
    non_positive_price_count: int
    negative_volume_count: int
    missing_bar_count: int = 0
    extra_bar_count: int = 0
    is_stable: bool = False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the stable HS300 factor-correlation run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--universe-file", type=Path, default=UNIVERSE_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--horizon", type=int, default=FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=EVALUATION_STEP_BARS)
    parser.add_argument("--top-quantile", type=float, default=TOP_QUANTILE)
    parser.add_argument("--high-corr", type=float, default=HIGH_CORRELATION_THRESHOLD)
    parser.add_argument("--min-factor-observations", type=int, default=MIN_FACTOR_OBSERVATIONS)
    parser.add_argument("--min-corr-observations", type=int, default=MIN_CORRELATION_OBSERVATIONS)
    parser.add_argument("--min-symbols", type=int, default=MIN_SYMBOLS_PER_FACTOR)
    parser.add_argument(
        "--stable-mode",
        choices=("modal_grid", "no_missing_values", "complete_grid"),
        default=STABLE_MODE,
        help=(
            "modal_grid uses the most common clean timestamp count; "
            "no_missing_values requires all year files and clean OHLCV rows; "
            "complete_grid also requires every timestamp in the HS300 union grid."
        ),
    )
    parser.add_argument("--workers", type=int, default=WORKERS, help="Parallel workers for per-symbol factor analysis.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional debug cap. 0 means all stable symbols.")
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> StableFactorConfig:
    """Validate CLI values and return an immutable config object."""

    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.eval_step < 1:
        raise ValueError("--eval-step must be positive.")
    if not 0 < args.top_quantile <= 1:
        raise ValueError("--top-quantile must be in (0, 1].")
    if not 0 < args.high_corr <= 1:
        raise ValueError("--high-corr must be in (0, 1].")
    return StableFactorConfig(
        data_root=args.data_root,
        universe_file=args.universe_file,
        output_dir=args.output_dir,
        start=pd.Timestamp(args.start) if args.start else None,
        end=pd.Timestamp(args.end) if args.end else None,
        forward_horizon_bars=args.horizon,
        evaluation_step_bars=args.eval_step,
        top_quantile=args.top_quantile,
        high_correlation_threshold=args.high_corr,
        min_factor_observations=args.min_factor_observations,
        min_correlation_observations=args.min_corr_observations,
        min_symbols_per_factor=args.min_symbols,
        stable_mode=args.stable_mode,
        workers=max(int(args.workers), 1),
        max_symbols=max(int(args.max_symbols), 0),
    )


def read_universe_symbols(path: Path) -> list[str]:
    """Read ordered unique symbols from a universe CSV with a symbol column."""

    universe = pd.read_csv(path, encoding="utf-8-sig")
    if "symbol" not in universe.columns:
        raise ValueError(f"Universe file must contain a symbol column: {path}")
    return universe["symbol"].astype(str).drop_duplicates().tolist()


def discover_year_dirs(data_root: Path) -> list[Path]:
    """Return extracted year directories sorted by name."""

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not dirs:
        raise FileNotFoundError(f"No year directories found under {data_root}")
    return dirs


def filter_by_date(frame: pd.DataFrame, config: StableFactorConfig) -> pd.DataFrame:
    """Filter a frame by optional inclusive datetime bounds."""

    if config.start is not None:
        frame = frame.loc[frame["datetime"] >= config.start].copy()
    if config.end is not None:
        frame = frame.loc[frame["datetime"] <= config.end].copy()
    return frame


def read_symbol_file(path: Path) -> pd.DataFrame:
    """Read one symbol-year CSV and preserve required OHLCV fields."""

    raw = pd.read_csv(path, encoding=CSV_ENCODING)
    missing = [column for column in KEY_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    keep = [column for column in (*KEY_COLUMNS, "amount") if column in raw.columns]
    frame = raw[keep].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def format_report_path(path: Path) -> str:
    """Return a readable project-relative path when possible."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_symbol_data(symbol: str, year_dirs: Iterable[Path], config: StableFactorConfig) -> pd.DataFrame:
    """Load all available year files for one symbol and attach the symbol code."""

    parts: list[pd.DataFrame] = []
    for year_dir in year_dirs:
        path = year_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        parts.append(read_symbol_file(path))
    if not parts:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "amount", "symbol"])
    frame = pd.concat(parts, ignore_index=True)
    frame = filter_by_date(frame, config)
    frame["symbol"] = symbol
    return frame.sort_values("datetime").reset_index(drop=True)


def scan_symbol(symbol: str, year_dirs: list[Path], config: StableFactorConfig) -> tuple[SymbolScan, set[pd.Timestamp]]:
    """Scan one symbol for missing files, missing bars, and OHLCV defects."""

    missing_files: list[str] = []
    parts: list[pd.DataFrame] = []
    for year_dir in year_dirs:
        path = year_dir / f"{symbol}.csv"
        if not path.exists():
            missing_files.append(format_report_path(path))
            continue
        parts.append(read_symbol_file(path))
    if parts:
        frame = filter_by_date(pd.concat(parts, ignore_index=True), config)
    else:
        frame = pd.DataFrame(columns=KEY_COLUMNS)

    if frame.empty:
        scan = SymbolScan(
            symbol=symbol,
            row_count=0,
            unique_datetime_count=0,
            first_datetime=None,
            last_datetime=None,
            missing_file_count=len(missing_files),
            missing_files=";".join(missing_files),
            null_key_row_count=0,
            duplicate_datetime_count=0,
            invalid_ohlc_count=0,
            non_positive_price_count=0,
            negative_volume_count=0,
        )
        return scan, set()

    date_series = frame["datetime"]
    datetime_set = set(pd.Timestamp(value) for value in date_series.dropna().unique())
    key_cols = [column for column in KEY_COLUMNS if column in frame.columns]
    null_key_rows = frame[key_cols].isna().any(axis=1)
    price_cols = ["open", "high", "low", "close"]
    non_positive_price = frame[price_cols].le(0).any(axis=1)
    invalid_ohlc = (frame["high"] < frame[["open", "close"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close"]].min(axis=1)
    )
    negative_volume = frame["volume"].lt(0)
    scan = SymbolScan(
        symbol=symbol,
        row_count=int(len(frame)),
        unique_datetime_count=int(len(datetime_set)),
        first_datetime=str(date_series.min()) if date_series.notna().any() else None,
        last_datetime=str(date_series.max()) if date_series.notna().any() else None,
        missing_file_count=len(missing_files),
        missing_files=";".join(missing_files),
        null_key_row_count=int(null_key_rows.sum()),
        duplicate_datetime_count=int(date_series.duplicated().sum()),
        invalid_ohlc_count=int(invalid_ohlc.fillna(False).sum()),
        non_positive_price_count=int(non_positive_price.fillna(False).sum()),
        negative_volume_count=int(negative_volume.fillna(False).sum()),
    )
    return scan, datetime_set


def is_clean_existing_scan(scan: SymbolScan, expected_count: int) -> bool:
    """Return whether one symbol has all files and clean existing OHLCV rows."""

    return (
        expected_count > 0
        and scan.missing_file_count == 0
        and scan.row_count > 0
        and scan.null_key_row_count == 0
        and scan.duplicate_datetime_count == 0
        and scan.invalid_ohlc_count == 0
        and scan.non_positive_price_count == 0
        and scan.negative_volume_count == 0
    )


def scan_stable_symbols(
    symbols: list[str],
    year_dirs: list[Path],
    config: StableFactorConfig,
) -> tuple[pd.DataFrame, list[str], set[pd.Timestamp]]:
    """Select stable symbols according to the configured data-quality mode."""

    scans: list[SymbolScan] = []
    expected_datetimes: set[pd.Timestamp] = set()
    print(f"scanning {len(symbols)} HS300 symbols for the expected datetime grid", flush=True)
    for idx, symbol in enumerate(symbols, start=1):
        scan, dates = scan_symbol(symbol, year_dirs, config)
        scans.append(scan)
        expected_datetimes.update(dates)
        if idx % 50 == 0 or idx == len(symbols):
            print(f"first pass scanned {idx}/{len(symbols)} symbols", flush=True)

    expected_count = len(expected_datetimes)
    clean_counts = [
        scan.unique_datetime_count
        for scan in scans
        if is_clean_existing_scan(scan, expected_count)
    ]
    modal_datetime_count = 0
    if clean_counts:
        count_frequency = pd.Series(clean_counts).value_counts()
        max_frequency = count_frequency.max()
        modal_datetime_count = int(count_frequency.loc[count_frequency == max_frequency].index.max())
    stable_symbols: list[str] = []
    rescanned: list[SymbolScan] = []
    print(
        f"expected datetime count from union={expected_count}; "
        f"modal clean datetime count={modal_datetime_count}; verifying stable symbols",
        flush=True,
    )
    for idx, symbol in enumerate(symbols, start=1):
        scan, dates = scan_symbol(symbol, year_dirs, config)
        scan.missing_bar_count = int(len(expected_datetimes - dates))
        scan.extra_bar_count = int(len(dates - expected_datetimes))
        clean_existing_rows = is_clean_existing_scan(scan, expected_count)
        modal_grid = clean_existing_rows and scan.unique_datetime_count == modal_datetime_count
        complete_grid = clean_existing_rows and scan.unique_datetime_count == expected_count and scan.missing_bar_count == 0
        if config.stable_mode == "complete_grid":
            scan.is_stable = complete_grid
        elif config.stable_mode == "no_missing_values":
            scan.is_stable = clean_existing_rows
        else:
            scan.is_stable = modal_grid
        if scan.is_stable:
            stable_symbols.append(symbol)
        rescanned.append(scan)
        if idx % 50 == 0 or idx == len(symbols):
            print(f"second pass verified {idx}/{len(symbols)} symbols; stable={len(stable_symbols)}", flush=True)

    scan_frame = pd.DataFrame([scan.__dict__ for scan in rescanned])
    return scan_frame, stable_symbols, expected_datetimes


def calculate_long_only_time_metrics(
    factor: pd.Series,
    future_return: pd.Series,
    top_quantile: float,
    min_observations: int,
) -> dict[str, float | int]:
    """Evaluate one factor as a long-only timing signal for one stock.

    High-direction buys the highest factor quantile; low-direction buys the
    lowest factor quantile. The better excess-return direction is retained.
    """

    data = pd.DataFrame({"factor": factor, "future_return": future_return}).replace([np.inf, -np.inf], np.nan).dropna()
    observations = int(len(data))
    if observations < min_observations or data["factor"].nunique() < 2 or data["future_return"].nunique() < 2:
        return {
            "observations": observations,
            "selected_observations": 0,
            "coverage": float(observations / len(factor)) if len(factor) else 0.0,
            "direction": 0,
            "rank_ic": np.nan,
            "directional_rank_ic": np.nan,
            "long_mean_return": np.nan,
            "benchmark_mean_return": np.nan,
            "excess_mean_return": np.nan,
            "long_win_rate": np.nan,
            "excess_win_rate": np.nan,
            "long_ir_like": np.nan,
        }

    rank_ic = float(data["factor"].corr(data["future_return"], method="spearman"))
    benchmark_mean = float(data["future_return"].mean())

    high_threshold = data["factor"].quantile(1 - top_quantile)
    low_threshold = data["factor"].quantile(top_quantile)
    high_returns = data.loc[data["factor"] >= high_threshold, "future_return"]
    low_returns = data.loc[data["factor"] <= low_threshold, "future_return"]

    high_excess = high_returns - benchmark_mean
    low_excess = low_returns - benchmark_mean
    high_score = float(high_excess.mean()) * max(float((high_excess > 0).mean()), 0.01)
    low_score = float(low_excess.mean()) * max(float((low_excess > 0).mean()), 0.01)
    direction = -1 if low_score > high_score else 1
    selected = low_returns if direction < 0 else high_returns
    selected_excess = selected - benchmark_mean
    selected_std = float(selected.std(ddof=0))

    return {
        "observations": observations,
        "selected_observations": int(len(selected)),
        "coverage": float(observations / len(factor)) if len(factor) else 0.0,
        "direction": direction,
        "rank_ic": rank_ic,
        "directional_rank_ic": float(direction * rank_ic),
        "long_mean_return": float(selected.mean()),
        "benchmark_mean_return": benchmark_mean,
        "excess_mean_return": float(selected_excess.mean()),
        "long_win_rate": float((selected > 0).mean()),
        "excess_win_rate": float((selected_excess > 0).mean()),
        "long_ir_like": float(selected.mean() / selected_std * math.sqrt(len(selected))) if selected_std else 0.0,
    }


def calculate_symbol_factor_metrics(
    symbol: str,
    symbol_data: pd.DataFrame,
    config: StableFactorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build all factors for one symbol and return metrics plus correlations."""

    symbol_data = symbol_data.sort_values("datetime").reset_index(drop=True)
    factors = build_factor_library(symbol_data)
    future_return = calculate_forward_return(symbol_data, config.forward_horizon_bars)
    sampled_index = np.arange(len(symbol_data)) % config.evaluation_step_bars == 0
    sampled_factors = factors.loc[sampled_index].copy()
    sampled_future = future_return.loc[sampled_index].copy()

    rows: list[dict[str, object]] = []
    factor_cols = list(sampled_factors.columns)
    for factor_name in factor_cols:
        metrics = calculate_long_only_time_metrics(
            sampled_factors[factor_name],
            sampled_future,
            config.top_quantile,
            config.min_factor_observations,
        )
        rows.append({"symbol": symbol, "factor": factor_name, **metrics})

    corr = sampled_factors.corr(method="spearman", min_periods=config.min_correlation_observations)
    return pd.DataFrame(rows), corr, factor_cols


def analyze_symbol_worker(args: tuple[str, list[Path], StableFactorConfig]) -> tuple[str, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load one stable symbol, build factors, and return metrics for merging."""

    symbol, year_dirs, config = args
    symbol_data = read_symbol_data(symbol, year_dirs, config)
    metrics, corr, factor_cols = calculate_symbol_factor_metrics(symbol, symbol_data, config)
    return symbol, metrics, corr, factor_cols


def initialize_corr_accumulators(factor_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create zero-filled arrays used to average per-symbol correlations."""

    size = len(factor_cols)
    return np.zeros((size, size), dtype=float), np.zeros((size, size), dtype=float), np.zeros((size, size), dtype=int)


def update_corr_accumulators(
    corr: pd.DataFrame,
    factor_cols: list[str],
    sum_corr: np.ndarray,
    sum_abs_corr: np.ndarray,
    count_corr: np.ndarray,
) -> None:
    """Add one per-symbol factor-correlation matrix into aggregate arrays."""

    aligned = corr.reindex(index=factor_cols, columns=factor_cols)
    values = aligned.to_numpy(dtype=float)
    valid = np.isfinite(values)
    sum_corr[valid] += values[valid]
    sum_abs_corr[valid] += np.abs(values[valid])
    count_corr[valid] += 1


def aggregate_factor_metrics(symbol_metrics: pd.DataFrame, config: StableFactorConfig) -> pd.DataFrame:
    """Average per-symbol factor test metrics into one row per factor."""

    valid = symbol_metrics.loc[symbol_metrics["direction"].ne(0)].copy()
    grouped = valid.groupby("factor", dropna=False)
    summary = grouped.agg(
        tested_symbol_count=("symbol", "nunique"),
        mean_direction=("direction", "mean"),
        positive_direction_count=("direction", lambda values: int((values > 0).sum())),
        negative_direction_count=("direction", lambda values: int((values < 0).sum())),
        mean_observations=("observations", "mean"),
        mean_selected_observations=("selected_observations", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_rank_ic=("rank_ic", "mean"),
        mean_directional_rank_ic=("directional_rank_ic", "mean"),
        median_directional_rank_ic=("directional_rank_ic", "median"),
        positive_directional_ic_rate=("directional_rank_ic", lambda values: float((values > 0).mean())),
        mean_long_return=("long_mean_return", "mean"),
        mean_benchmark_return=("benchmark_mean_return", "mean"),
        mean_excess_return=("excess_mean_return", "mean"),
        median_excess_return=("excess_mean_return", "median"),
        positive_excess_return_rate=("excess_mean_return", lambda values: float((values > 0).mean())),
        mean_long_win_rate=("long_win_rate", "mean"),
        mean_excess_win_rate=("excess_win_rate", "mean"),
        mean_long_ir_like=("long_ir_like", "mean"),
    ).reset_index()
    summary["majority_direction"] = np.where(summary["mean_direction"] >= 0, 1, -1)
    summary["selected_factor"] = np.where(
        summary["majority_direction"] > 0,
        summary["factor"],
        "neg_" + summary["factor"].astype(str),
    )
    summary["combined_score"] = (
        summary["mean_directional_rank_ic"].fillna(0) * np.sqrt(summary["tested_symbol_count"].clip(lower=1))
        + summary["mean_excess_return"].fillna(0) * 10_000
        + summary["positive_excess_return_rate"].fillna(0)
    )
    summary["is_effective_mean"] = (
        summary["tested_symbol_count"].ge(config.min_symbols_per_factor)
        & summary["mean_directional_rank_ic"].gt(0)
        & summary["mean_excess_return"].gt(0)
        & summary["positive_excess_return_rate"].ge(0.55)
    )
    return summary.sort_values("combined_score", ascending=False).reset_index(drop=True)


def build_correlation_outputs(
    factor_cols: list[str],
    sum_corr: np.ndarray,
    sum_abs_corr: np.ndarray,
    count_corr: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create mean-correlation matrices and high-correlation pair table."""

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_corr = sum_corr / count_corr
        mean_abs_corr = sum_abs_corr / count_corr
    mean_corr_frame = pd.DataFrame(mean_corr, index=factor_cols, columns=factor_cols)
    mean_abs_corr_frame = pd.DataFrame(mean_abs_corr, index=factor_cols, columns=factor_cols)

    rows: list[dict[str, object]] = []
    for i, factor_a in enumerate(factor_cols):
        for j in range(i + 1, len(factor_cols)):
            factor_b = factor_cols[j]
            if count_corr[i, j] <= 0:
                continue
            abs_corr = float(mean_abs_corr[i, j])
            if abs_corr >= threshold:
                rows.append(
                    {
                        "factor_a": factor_a,
                        "factor_b": factor_b,
                        "mean_correlation": float(mean_corr[i, j]),
                        "mean_abs_correlation": abs_corr,
                        "symbols_used": int(count_corr[i, j]),
                    }
                )
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["mean_abs_correlation", "symbols_used"], ascending=[False, False]).reset_index(drop=True)
    return mean_corr_frame, mean_abs_corr_frame, pairs


def find_components(factor_cols: list[str], high_corr_pairs: pd.DataFrame) -> list[list[str]]:
    """Find connected factor clusters from high-correlation edges."""

    adjacency: dict[str, set[str]] = {factor: set() for factor in factor_cols}
    if not high_corr_pairs.empty:
        for row in high_corr_pairs.itertuples(index=False):
            adjacency[str(row.factor_a)].add(str(row.factor_b))
            adjacency[str(row.factor_b)].add(str(row.factor_a))

    seen: set[str] = set()
    components: list[list[str]] = []
    for factor in factor_cols:
        if factor in seen:
            continue
        stack = [factor]
        component: list[str] = []
        seen.add(factor)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def build_cluster_outputs(
    factor_cols: list[str],
    mean_abs_corr: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    factor_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize correlated factor clusters and pick low-correlation candidates."""

    score_map = factor_summary.set_index("factor")["combined_score"].to_dict()
    metric_map = factor_summary.set_index("factor").to_dict(orient="index")
    cluster_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for cluster_id, members in enumerate(find_components(factor_cols, high_corr_pairs), start=1):
        representative = max(members, key=lambda factor: float(score_map.get(factor, -np.inf)))
        pair_values: list[float] = []
        for i, factor_a in enumerate(members):
            for factor_b in members[i + 1 :]:
                value = mean_abs_corr.loc[factor_a, factor_b]
                if pd.notna(value):
                    pair_values.append(float(value))
        row = {
            "cluster_id": cluster_id,
            "representative_factor": representative,
            "member_count": len(members),
            "cluster_mean_abs_correlation": float(np.mean(pair_values)) if pair_values else np.nan,
            "cluster_max_abs_correlation": float(np.max(pair_values)) if pair_values else np.nan,
            "members": ",".join(members),
        }
        if len(members) > 1:
            cluster_rows.append(row)
        metrics = metric_map.get(representative, {})
        candidate_rows.append(
            {
                "cluster_id": cluster_id,
                "factor": representative,
                "member_count": len(members),
                "selected_factor": metrics.get("selected_factor", representative),
                "combined_score": metrics.get("combined_score", np.nan),
                "mean_directional_rank_ic": metrics.get("mean_directional_rank_ic", np.nan),
                "mean_excess_return": metrics.get("mean_excess_return", np.nan),
                "positive_excess_return_rate": metrics.get("positive_excess_return_rate", np.nan),
                "is_effective_mean": metrics.get("is_effective_mean", False),
            }
        )

    clusters = pd.DataFrame(cluster_rows)
    candidates = pd.DataFrame(candidate_rows)
    if not clusters.empty:
        clusters = clusters.sort_values(["member_count", "cluster_max_abs_correlation"], ascending=[False, False])
    if not candidates.empty:
        candidates = candidates.sort_values("combined_score", ascending=False)
    return clusters.reset_index(drop=True), candidates.reset_index(drop=True)


def save_outputs(
    config: StableFactorConfig,
    year_dirs: list[Path],
    scan_report: pd.DataFrame,
    stable_symbols: list[str],
    expected_datetimes: set[pd.Timestamp],
    symbol_metrics: pd.DataFrame,
    factor_summary: pd.DataFrame,
    mean_corr: pd.DataFrame,
    mean_abs_corr: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    clusters: pd.DataFrame,
    candidates: pd.DataFrame,
) -> Path:
    """Write CSV/JSON report files and resolved run context."""

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scan_report.to_csv(run_dir / "symbol_stability_report.csv", index=False, encoding="utf-8-sig")
    scan_report.loc[scan_report["is_stable"]].to_csv(run_dir / "stable_hs300_symbols.csv", index=False, encoding="utf-8-sig")
    symbol_metrics.to_csv(run_dir / "factor_symbol_metrics.csv", index=False, encoding="utf-8-sig")
    factor_summary.to_csv(run_dir / "factor_mean_metrics.csv", index=False, encoding="utf-8-sig")
    mean_corr.to_csv(run_dir / "factor_correlation_mean.csv", encoding="utf-8-sig")
    mean_abs_corr.to_csv(run_dir / "factor_correlation_abs_mean.csv", encoding="utf-8-sig")
    high_corr_pairs.to_csv(run_dir / "high_correlation_factor_pairs.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(run_dir / "factor_correlation_clusters.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(run_dir / "low_correlation_factor_candidates.csv", index=False, encoding="utf-8-sig")

    context = {
        "data_root": str(config.data_root),
        "universe_file": str(config.universe_file),
        "year_dirs": [path.name for path in year_dirs],
        "start": str(config.start) if config.start is not None else None,
        "end": str(config.end) if config.end is not None else None,
        "expected_datetime_count": len(expected_datetimes),
        "expected_datetime_start": str(min(expected_datetimes)) if expected_datetimes else None,
        "expected_datetime_end": str(max(expected_datetimes)) if expected_datetimes else None,
        "candidate_symbol_count": int(len(scan_report)),
        "stable_symbol_count": int(len(stable_symbols)),
        "stable_mode": config.stable_mode,
        "stable_unique_datetime_count_min": int(scan_report.loc[scan_report["is_stable"], "unique_datetime_count"].min())
        if stable_symbols
        else None,
        "stable_unique_datetime_count_max": int(scan_report.loc[scan_report["is_stable"], "unique_datetime_count"].max())
        if stable_symbols
        else None,
        "stable_symbols": stable_symbols,
        "factor_count": int(len(factor_summary)),
        "effective_mean_factor_count": int(factor_summary["is_effective_mean"].sum()) if not factor_summary.empty else 0,
        "high_correlation_threshold": float(config.high_correlation_threshold),
        "high_correlation_pair_count": int(len(high_corr_pairs)),
        "correlated_cluster_count": int(len(clusters)),
        "forward_horizon_bars": int(config.forward_horizon_bars),
        "evaluation_step_bars": int(config.evaluation_step_bars),
        "top_quantile": float(config.top_quantile),
        "min_factor_observations": int(config.min_factor_observations),
        "min_correlation_observations": int(config.min_correlation_observations),
        "workers": int(config.workers),
        "talib_available": bool(has_talib()),
        "method_notes": [
            "Default modal_grid stable stocks must have all extracted year files, clean OHLCV rows, and the most common clean timestamp count.",
            "Use --stable-mode no_missing_values to ignore missing bars, or complete_grid to require every timestamp in the HS300 union grid.",
            "Factor tests are long-only timing tests per symbol; high and low quantile directions are compared without creating short legs.",
            "Reported factor metrics are means across stable symbols.",
            "Factor correlations are Spearman correlations computed per stable symbol and then averaged by factor pair.",
            "Evaluation samples every configured number of bars to reduce overlapping-label dependence and runtime.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run_context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def run_analysis(config: StableFactorConfig) -> Path:
    """Execute stable-universe selection, factor tests, and correlation reports."""

    symbols = read_universe_symbols(config.universe_file)
    year_dirs = discover_year_dirs(config.data_root)
    scan_report, stable_symbols, expected_datetimes = scan_stable_symbols(symbols, year_dirs, config)
    if config.max_symbols:
        stable_symbols = stable_symbols[: config.max_symbols]
    if not stable_symbols:
        raise ValueError("No stable HS300 symbols met the complete-data criteria.")

    all_symbol_metrics: list[pd.DataFrame] = []
    factor_cols: list[str] | None = None
    sum_corr: np.ndarray | None = None
    sum_abs_corr: np.ndarray | None = None
    count_corr: np.ndarray | None = None

    print(f"building factors for {len(stable_symbols)} stable symbols with workers={config.workers}", flush=True)

    def merge_symbol_result(idx: int, symbol: str, metrics: pd.DataFrame, corr: pd.DataFrame, current_factor_cols: list[str]) -> None:
        """Merge one completed symbol result into aggregate metrics."""

        nonlocal factor_cols, sum_corr, sum_abs_corr, count_corr
        all_symbol_metrics.append(metrics)
        if factor_cols is None:
            factor_cols = current_factor_cols
            sum_corr, sum_abs_corr, count_corr = initialize_corr_accumulators(factor_cols)
        update_corr_accumulators(corr, factor_cols, sum_corr, sum_abs_corr, count_corr)
        if idx % 10 == 0 or idx == len(stable_symbols):
            print(f"evaluated factors for {idx}/{len(stable_symbols)} stable symbols; latest={symbol}", flush=True)

    if config.workers == 1:
        for idx, symbol in enumerate(stable_symbols, start=1):
            symbol, metrics, corr, current_factor_cols = analyze_symbol_worker((symbol, year_dirs, config))
            merge_symbol_result(idx, symbol, metrics, corr, current_factor_cols)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(analyze_symbol_worker, (symbol, year_dirs, config)): symbol for symbol in stable_symbols
            }
            for future in as_completed(futures):
                symbol, metrics, corr, current_factor_cols = future.result()
                completed += 1
                merge_symbol_result(completed, symbol, metrics, corr, current_factor_cols)

    if factor_cols is None or sum_corr is None or sum_abs_corr is None or count_corr is None:
        raise ValueError("No factor columns were generated.")

    symbol_metrics = pd.concat(all_symbol_metrics, ignore_index=True)
    factor_summary = aggregate_factor_metrics(symbol_metrics, config)
    mean_corr, mean_abs_corr, high_corr_pairs = build_correlation_outputs(
        factor_cols,
        sum_corr,
        sum_abs_corr,
        count_corr,
        config.high_correlation_threshold,
    )
    clusters, candidates = build_cluster_outputs(factor_cols, mean_abs_corr, high_corr_pairs, factor_summary)
    return save_outputs(
        config,
        year_dirs,
        scan_report,
        stable_symbols,
        expected_datetimes,
        symbol_metrics,
        factor_summary,
        mean_corr,
        mean_abs_corr,
        high_corr_pairs,
        clusters,
        candidates,
    )


def main() -> None:
    """Run the stable HS300 factor-correlation experiment."""

    config = create_config(parse_args())
    run_dir = run_analysis(config)
    summary = pd.read_csv(run_dir / "factor_mean_metrics.csv", encoding="utf-8-sig")
    high_corr = pd.read_csv(run_dir / "high_correlation_factor_pairs.csv", encoding="utf-8-sig")
    stable = pd.read_csv(run_dir / "stable_hs300_symbols.csv", encoding="utf-8-sig")
    print(
        f"saved stable HS300 factor-correlation report to {run_dir}\n"
        f"stable_symbols={len(stable)} factors={len(summary)} high_corr_pairs={len(high_corr)}"
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
