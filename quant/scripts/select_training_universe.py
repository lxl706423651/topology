"""Select a liquid stock universe from processed pkl slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "ashare_minute_data_2023_2026_04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--year", type=str, default="2026")
    parser.add_argument("--top-n", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    input_dir = args.input_dir or data_dir / "processed" / "pkl"
    analysis_dir = data_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((input_dir / f"year={args.year}").rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pkl files found for year={args.year} under {input_dir}")

    rows: list[pd.DataFrame] = []
    for idx, path in enumerate(files):
        frame = pd.read_pickle(path)
        frame["symbol"] = frame["symbol"].astype(str)
        grouped = frame.groupby("symbol", observed=True).agg(
            rows=("datetime", "size"),
            amount_sum=("amount", "sum"),
            volume_sum=("volume", "sum"),
            avg_close=("close", "mean"),
            start=("datetime", "min"),
            end=("datetime", "max"),
        )
        rows.append(grouped.reset_index())
        print(f"universe scan {idx + 1}/{len(files)} {path.name}")

    stats = pd.concat(rows, ignore_index=True).groupby("symbol", observed=True).agg(
        rows=("rows", "sum"),
        amount_sum=("amount_sum", "sum"),
        volume_sum=("volume_sum", "sum"),
        avg_close=("avg_close", "mean"),
        start=("start", "min"),
        end=("end", "max"),
    )
    stats["avg_amount_per_bar"] = stats["amount_sum"] / stats["rows"].replace(0, pd.NA)
    stats = stats.sort_values(["amount_sum", "rows"], ascending=False).reset_index()
    all_path = analysis_dir / f"universe_liquidity_{args.year}.csv"
    top_path = analysis_dir / f"training_universe_top{args.top_n}_{args.year}.csv"
    stats.to_csv(all_path, index=False, encoding="utf-8-sig")
    stats.head(args.top_n).to_csv(top_path, index=False, encoding="utf-8-sig")
    print(f"Saved full liquidity stats to {all_path}")
    print(f"Saved top universe to {top_path}")


if __name__ == "__main__":
    main()
