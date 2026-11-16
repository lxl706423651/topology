"""Build reusable factor-feature files from A-share OHLCV data.

This script is the one-click feature-construction entrypoint for the factors
defined in ``quant_backtest.factor_library``. It reads CSV/PKL/Parquet market
data, optionally filters by a universe file, builds factors per symbol using
only current and historical bars, and writes a generated feature table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.data import OHLCV_ALIASES, standardize_ohlcv
from quant_backtest.factor_library import (
    build_cross_sectional_factor_library,
    build_factor_library,
    has_talib,
    infer_factor_metadata,
    validate_factor_frame,
    validate_no_future_leakage,
)


DEFAULT_DATA_PATH = PROJECT_ROOT / "ashare_minute_data_2023_2026_04" / "extracted" / "2023"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "factor_features" / "factor_features.pkl"
DEFAULT_UNIVERSE_FILE = PROJECT_ROOT / "config" / "hs300_current_constituents.csv"


def parse_args() -> argparse.Namespace:
    """Parse data, universe, date-range, and output settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-files", type=int, default=0, help="Directory debug limit. 0 means no limit.")
    parser.add_argument("--format", choices=["pkl", "csv", "parquet"], default="pkl")
    parser.add_argument("--include-cross-sectional", action="store_true", help="Add same-timestamp rank/z-score features.")
    parser.add_argument("--include-winsorized-zscore", action="store_true", help="Add winsorized cross-sectional z-scores.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional CSV path for generated factor metadata.")
    parser.add_argument("--leakage-report-path", type=Path, default=None, help="Optional CSV path for leakage-check results.")
    return parser.parse_args()


def parse_inclusive_end(value: str) -> pd.Timestamp:
    """Parse date-only end bounds as inclusive through the whole day."""

    parsed = pd.Timestamp(value)
    if len(str(value)) <= 10:
        return parsed + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return parsed


def has_any_column(columns: set[str], target: str) -> bool:
    """Return whether input columns contain a target or one of its aliases."""

    candidates = {target, *OHLCV_ALIASES.get(target, [])}
    lower_columns = {col.lower() for col in columns}
    return any(candidate in columns or candidate.lower() in lower_columns for candidate in candidates)


def read_market_file(path: Path) -> pd.DataFrame:
    """Read one market-data file and infer symbol from file name if needed."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path)
    elif suffix in {".pkl", ".pickle"}:
        raw = pd.read_pickle(path)
    elif suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported data file type: {path.suffix}")
    if not has_any_column(set(map(str, raw.columns)), "symbol"):
        raw["symbol"] = path.stem
    return standardize_ohlcv(raw)


def load_universe(path: Path | None) -> set[str] | None:
    """Load a symbol universe CSV with a required ``symbol`` column."""

    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if "symbol" not in frame.columns:
        raise ValueError(f"Universe file must contain a symbol column: {path}")
    return set(frame["symbol"].astype(str))


def select_data_files(path: Path, universe: set[str] | None, max_files: int) -> list[Path]:
    """Select files from a file or directory, using file stems for universe matching."""

    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Data path does not exist: {path}")
    files = sorted(
        candidate
        for suffix in ("*.csv", "*.pkl", "*.pickle", "*.parquet")
        for candidate in path.rglob(suffix)
    )
    if universe:
        files = [file for file in files if file.stem in universe]
    return files[:max_files] if max_files else files


def get_factor_columns(frame: pd.DataFrame) -> list[str]:
    """Return generated factor columns, excluding identifiers and OHLCV fields."""

    excluded = {"datetime", "symbol", "open", "high", "low", "close", "volume", "amount"}
    excluded.update(col for col in frame.columns if col.startswith(("future_return", "forward_return", "label", "target")))
    return [col for col in frame.select_dtypes(include="number").columns if col not in excluded]


def build_features(args: argparse.Namespace) -> pd.DataFrame:
    """Load market data and build factor features per symbol."""

    universe = load_universe(args.universe_file)
    files = select_data_files(args.data_path, universe, args.max_files)
    if not files:
        raise FileNotFoundError(f"No input files matched data path/universe: {args.data_path}")

    parts: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, object]] = []
    for idx, path in enumerate(files, start=1):
        data = read_market_file(path)
        if universe:
            data = data.loc[data["symbol"].astype(str).isin(universe)].copy()
        if args.start:
            data = data.loc[data["datetime"] >= pd.Timestamp(args.start)].copy()
        if args.end:
            data = data.loc[data["datetime"] <= parse_inclusive_end(args.end)].copy()
        for symbol, group in data.groupby("symbol", sort=False):
            group = group.sort_values("datetime").copy()
            factors = build_factor_library(group)
            base_cols = ["datetime", "symbol", "open", "high", "low", "close", "volume"]
            if "amount" in group.columns:
                base_cols.append("amount")
            part = pd.concat([group[base_cols], factors], axis=1)
            parts.append(part)
            if len(leakage_rows) < 3:
                leakage = validate_no_future_leakage(group)
                leakage["symbol"] = str(symbol)
                leakage["source_file"] = str(path)
                leakage_rows.append(leakage)
        if idx % 25 == 0 or idx == len(files):
            print(f"built factor features for {idx}/{len(files)} files")

    features = pd.concat(parts, ignore_index=True).sort_values(["symbol", "datetime"]).reset_index(drop=True)
    factor_cols = get_factor_columns(features)
    if args.include_cross_sectional:
        cross_sectional = build_cross_sectional_factor_library(
            features,
            factor_cols=factor_cols,
            include_winsorized_zscore=args.include_winsorized_zscore,
        )
        features = pd.concat([features, cross_sectional], axis=1)
        factor_cols = get_factor_columns(features)
    validation = validate_factor_frame(features, factor_cols)
    for row in validation.to_dict(orient="records"):
        row["symbol"] = "all"
        row["source_file"] = "feature_frame"
        leakage_rows.append(row)
    if args.leakage_report_path:
        args.leakage_report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(leakage_rows).to_csv(args.leakage_report_path, index=False, encoding="utf-8-sig")
    if args.metadata_path:
        args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        infer_factor_metadata(factor_cols).to_csv(args.metadata_path, index=False, encoding="utf-8-sig")
    return features


def save_features(frame: pd.DataFrame, path: Path, output_format: str) -> None:
    """Save generated factor features in pkl, csv, or parquet format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    elif output_format == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_pickle(path)


def main() -> None:
    """Build factor features and write the generated table."""

    args = parse_args()
    features = build_features(args)
    save_features(features, args.output_path, args.format)
    print(
        f"saved rows={len(features)} symbols={features['symbol'].nunique()} "
        f"columns={len(features.columns)} talib={has_talib()} -> {args.output_path}"
    )


if __name__ == "__main__":
    main()
