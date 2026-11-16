"""Clean extracted one-minute stock CSV files and save pkl slices."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "ashare_minute_data_2023_2026_04"
BASE_COLUMNS = ["datetime", "symbol", "open", "high", "low", "close", "volume", "amount"]


def infer_symbol(csv_path: Path) -> str:
    return csv_path.stem


def trading_session_mask(dt: pd.Series) -> pd.Series:
    minute = dt.dt.hour * 60 + dt.dt.minute
    morning = (minute >= 9 * 60 + 30) & (minute <= 11 * 60 + 30)
    afternoon = (minute >= 13 * 60) & (minute <= 15 * 60)
    return morning | afternoon


def half_name(dt: pd.Series) -> pd.Series:
    return np.where(dt.dt.month <= 6, "H1", "H2")


def read_and_clean(csv_path: Path, year: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw_columns = list(raw.columns)
    raw_rows = len(raw)

    rename = {c: c.lower() for c in raw.columns}
    frame = raw.rename(columns=rename)
    if "datetime" not in frame.columns:
        raise ValueError(f"{csv_path} has no datetime column")
    for col in ["open", "high", "low", "close"]:
        if col not in frame.columns:
            raise ValueError(f"{csv_path} missing required column: {col}")
    if "volume" not in frame.columns:
        frame["volume"] = 0
    if "amount" not in frame.columns:
        frame["amount"] = np.nan

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["symbol"] = infer_symbol(csv_path)

    duplicate_rows = int(frame.duplicated(subset=["datetime"]).sum())
    null_counts = {f"null_{col}": int(frame[col].isna().sum()) for col in ["datetime", "open", "high", "low", "close", "volume", "amount"]}
    valid_time = frame["datetime"].notna()
    session = trading_session_mask(frame.loc[valid_time, "datetime"])
    non_trading_rows = int(valid_time.sum() - session.sum())

    invalid_ohlc = (
        frame["high"].lt(frame["low"])
        | frame["high"].lt(frame[["open", "close"]].max(axis=1))
        | frame["low"].gt(frame[["open", "close"]].min(axis=1))
        | frame[["open", "high", "low", "close"]].le(0).any(axis=1)
    )
    invalid_ohlc_count = int(invalid_ohlc.fillna(True).sum())

    clean = frame[BASE_COLUMNS].copy()
    clean = clean.dropna(subset=["datetime", "open", "high", "low", "close"])
    clean = clean.loc[trading_session_mask(clean["datetime"])]
    clean = clean.loc[
        clean["high"].ge(clean["low"])
        & clean["high"].ge(clean[["open", "close"]].max(axis=1))
        & clean["low"].le(clean[["open", "close"]].min(axis=1))
        & clean[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ]
    clean = clean.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    clean["symbol"] = clean["symbol"].astype("category")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        clean[col] = clean[col].astype("float32")

    stats: dict[str, object] = {
        "year": year,
        "file": str(csv_path),
        "symbol": infer_symbol(csv_path),
        "raw_rows": raw_rows,
        "clean_rows": len(clean),
        "dropped_rows": raw_rows - len(clean),
        "duplicate_rows": duplicate_rows,
        "non_trading_rows": non_trading_rows,
        "invalid_ohlc_rows": invalid_ohlc_count,
        "zero_volume_rows": int((frame["volume"].fillna(0) == 0).sum()),
        "zero_amount_rows": int((frame["amount"].fillna(0) == 0).sum()),
        "start": str(clean["datetime"].min()) if not clean.empty else "",
        "end": str(clean["datetime"].max()) if not clean.empty else "",
        "columns": ",".join(raw_columns),
    }
    stats.update(null_counts)
    return clean, stats


def save_batch(
    batch: pd.DataFrame,
    output_dir: Path,
    year: str,
    batch_id: int,
    slices_dir: Path,
    slice_state: dict[str, bool],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if batch.empty:
        return rows
    batch = batch.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    batch["half"] = half_name(batch["datetime"])

    for half, part in batch.groupby("half", sort=True):
        part = part.drop(columns=["half"]).reset_index(drop=True)
        target_dir = output_dir / f"year={year}" / f"half={half}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"part_{batch_id:04d}.pkl"
        part.to_pickle(target)
        rows.append(
            {
                "year": year,
                "half": half,
                "part": target.name,
                "path": str(target),
                "rows": len(part),
                "symbols": int(part["symbol"].nunique()),
                "start": str(part["datetime"].min()),
                "end": str(part["datetime"].max()),
            }
        )

        if not slice_state.get("head"):
            part.head(1000).to_pickle(slices_dir / "slice_head_1000.pkl")
            slice_state["head"] = True
        if not slice_state.get("single_symbol"):
            symbol = part["symbol"].astype(str).iloc[0]
            part.loc[part["symbol"].astype(str) == symbol].head(1500).to_pickle(slices_dir / f"slice_one_symbol_{symbol}.pkl")
            slice_state["single_symbol"] = True
        if not slice_state.get("multi_symbol"):
            symbols = part["symbol"].astype(str).drop_duplicates().head(5).tolist()
            sample = part.loc[part["symbol"].astype(str).isin(symbols)].head(5000)
            sample.to_pickle(slices_dir / "slice_5_symbols_5000_rows.pkl")
            slice_state["multi_symbol"] = True
    return rows


def write_markdown(data_dir: Path, profile: pd.DataFrame, manifest: pd.DataFrame, output_dir: Path, slices_dir: Path) -> None:
    readme = data_dir / "README_DATA_ANALYSIS.md"
    total_raw = int(profile["raw_rows"].sum()) if not profile.empty else 0
    total_clean = int(profile["clean_rows"].sum()) if not profile.empty else 0
    symbols = int(profile["symbol"].nunique()) if not profile.empty else 0
    years = ", ".join(map(str, sorted(profile["year"].unique()))) if not profile.empty else ""
    start = profile.loc[profile["start"] != "", "start"].min() if not profile.empty else ""
    end = profile.loc[profile["end"] != "", "end"].max() if not profile.empty else ""
    total_size_gb = sum(p.stat().st_size for p in output_dir.rglob("*.pkl")) / 1024**3 if output_dir.exists() else 0

    text = f"""# A-Share Minute Data Analysis

生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}

## 数据概览

- 原始目录：`{data_dir}`
- 解压目录：`{data_dir / "extracted"}`
- 清洗后 pkl：`{output_dir}`
- slice 示例：`{slices_dir}`
- 覆盖年份：{years}
- 股票文件数/代码数：{symbols}
- 原始行数：{total_raw:,}
- 清洗后行数：{total_clean:,}
- 行数保留率：{(total_clean / total_raw if total_raw else 0):.2%}
- 时间范围：{start} 到 {end}
- 当前 pkl 总大小：{total_size_gb:.2f} GB

## 清洗规则

1. 每个 ZIP 按年份解压到 `extracted/<year>/`，避免不同年份同名股票 CSV 互相覆盖。
2. 标准字段统一为 `datetime, symbol, open, high, low, close, volume, amount`。
3. `datetime` 转成 pandas 时间类型；价格、成交量、成交额转成数值。
4. 过滤非 A 股常规分钟时段，只保留 09:30-11:30 与 13:00-15:00。
5. 删除缺失 OHLC、非正价格、`high < low`、`high` 小于开/收、`low` 大于开/收的异常行。
6. 同一股票同一分钟重复时保留最后一条。
7. 为了照顾 32GB 内存，pkl 按 `year=<年>/half=H1|H2/part_xxxx.pkl` 切片保存。

## 质量统计文件

- `analysis/data_profile_by_symbol.csv`：每个股票 CSV 的行数、缺失、重复、异常、时间范围。
- `analysis/pkl_manifest.csv`：每个 pkl 切片的行数、股票数、起止时间。

## 如何读取一个切片

```python
import pandas as pd

df = pd.read_pickle(r"{manifest['path'].iloc[0] if not manifest.empty else output_dir}")
print(df.head())
print(df.dtypes)
```

## slice 示例用途

- `slice_head_1000.pkl`：快速看字段和类型。
- `slice_one_symbol_*.pkl`：看单只股票的分钟序列。
- `slice_5_symbols_5000_rows.pkl`：看多股票横截面与时间排序。

后续特征工程会基于这些清洗后的 pkl 继续生成 100-200 个技术和统计特征。
"""
    readme.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--max-files", type=int, default=0, help="Debug limit. 0 means all files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    input_dir = args.input_dir or data_dir / "extracted"
    output_dir = args.output_dir or data_dir / "processed" / "pkl"
    analysis_dir = data_dir / "analysis"
    slices_dir = data_dir / "slices"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    profile_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    slice_state: dict[str, bool] = {}
    global_batch_id = 0

    year_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    for year_dir in year_dirs:
        year = year_dir.name
        files = sorted(year_dir.rglob("*.csv"))
        if args.max_files:
            files = files[: args.max_files]
        print(f"Processing year={year} files={len(files)}")
        for start in range(0, len(files), args.batch_size):
            batch_files = files[start : start + args.batch_size]
            parts: list[pd.DataFrame] = []
            for csv_path in batch_files:
                try:
                    clean, stats = read_and_clean(csv_path, year)
                except Exception as exc:
                    stats = {
                        "year": year,
                        "file": str(csv_path),
                        "symbol": infer_symbol(csv_path),
                        "raw_rows": 0,
                        "clean_rows": 0,
                        "dropped_rows": 0,
                        "error": repr(exc),
                    }
                    clean = pd.DataFrame(columns=BASE_COLUMNS)
                profile_rows.append(stats)
                if not clean.empty:
                    parts.append(clean)
            if parts:
                batch = pd.concat(parts, ignore_index=True)
                manifest_rows.extend(save_batch(batch, output_dir, year, global_batch_id, slices_dir, slice_state))
                del batch
            del parts
            gc.collect()
            print(f"  batch={global_batch_id:04d} done files={len(batch_files)}")
            global_batch_id += 1

    profile = pd.DataFrame(profile_rows)
    manifest = pd.DataFrame(manifest_rows)
    profile.to_csv(analysis_dir / "data_profile_by_symbol.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(analysis_dir / "pkl_manifest.csv", index=False, encoding="utf-8-sig")
    write_markdown(data_dir, profile, manifest, output_dir, slices_dir)
    print(f"Profile saved to {analysis_dir / 'data_profile_by_symbol.csv'}")
    print(f"PKL manifest saved to {analysis_dir / 'pkl_manifest.csv'}")
    print(f"README saved to {data_dir / 'README_DATA_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
