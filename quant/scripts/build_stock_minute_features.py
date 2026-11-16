"""Build technical/statistical feature pkl files from cleaned minute pkl slices."""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATA_DIR = PROJECT_ROOT / "ashare_minute_data_2023_2026_04"
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

from quant_backtest.features import build_minute_features, feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-parts", type=int, default=0, help="Debug/resource limit. 0 means all pkl parts.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional symbol universe limit per run.")
    parser.add_argument("--universe-file", type=Path, default=None, help="CSV with a symbol column.")
    parser.add_argument("--min-rows", type=int, default=260)
    return parser.parse_args()


def write_feature_readme(data_dir: Path, output_dir: Path, manifest: pd.DataFrame, columns: list[str]) -> None:
    path = data_dir / "README_FEATURES.md"
    rows = int(manifest["rows"].sum()) if not manifest.empty else 0
    symbols = int(manifest["symbols"].max()) if not manifest.empty else 0
    text = f"""# 分钟数据特征工程说明

生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}

## 输出位置

- 特征 pkl：`{output_dir}`
- 特征清单：`{data_dir / "analysis" / "feature_columns.txt"}`
- 特征 manifest：`{data_dir / "analysis" / "feature_manifest.csv"}`
- 特征 slice：`{data_dir / "slices" / "feature_slice_2000.pkl"}`

## 当前结果

- 特征列数量：{len(columns)}
- 已生成行数：{rows:,}
- 单个切片最大股票数：{symbols}

## 特征类别

1. 收益与动量：`ret_*`, `logret_*`, `roc_*`
2. 波动率与极值：`ret_std_*`, `atr_ratio_*`, `range_*`, `dist_high_*`, `dist_low_*`
3. 均线与趋势：`sma_gap_*`, `ema_gap_*`, `ma_cross_*`
4. 价量关系：`volume_ratio_*`, `amount_ratio_*`, `vwap_gap_*`, `obv_z_*`
5. 技术指标：`rsi_*`, `cci_*`, `willr_*`, `mfi_*`, `adx_*`, `macd_*`, `boll_*`, `stoch_*`
6. 日内时间特征：`minute_sin`, `minute_cos`, `weekday_sin`, `weekday_cos`, `is_morning`

## 标签

- `future_return_1`：未来 1 分钟收益
- `future_return_5`：未来 5 分钟收益
- `future_return_24` / `Return_24`：未来 24 分钟收益，兼容 `legacy_bangsheng_quant_strategies` 里的旧脚本命名

特征只使用当前和历史 K 线，未来收益只作为训练标签，回测时不能作为模型输入。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    input_dir = args.input_dir or data_dir / "processed" / "pkl"
    output_dir = args.output_dir or data_dir / "processed" / "features_pkl"
    analysis_dir = data_dir / "analysis"
    slices_dir = data_dir / "slices"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(input_dir.rglob("*.pkl"))
    if args.max_parts:
        pkl_files = pkl_files[: args.max_parts]
    selected_symbols: set[str] | None = None
    manifest_rows: list[dict[str, object]] = []
    all_feature_columns: list[str] = []
    wrote_slice = False
    if args.universe_file:
        universe = pd.read_csv(args.universe_file)["symbol"].astype(str).tolist()
        selected_symbols: set[str] | None = set(universe[: args.max_symbols] if args.max_symbols else universe)
    else:
        selected_symbols = None

    for idx, pkl_path in enumerate(pkl_files):
        rel = pkl_path.relative_to(input_dir)
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_pickle(pkl_path)
        frame["symbol"] = frame["symbol"].astype(str)

        if args.max_symbols and selected_symbols is None:
            if selected_symbols is None:
                selected_symbols = set(frame["symbol"].drop_duplicates().head(args.max_symbols).tolist())
            frame = frame.loc[frame["symbol"].isin(selected_symbols)].copy()
        elif selected_symbols is not None:
            frame = frame.loc[frame["symbol"].isin(selected_symbols)].copy()
        if frame.empty:
            continue

        parts: list[pd.DataFrame] = []
        for symbol, group in frame.groupby("symbol", sort=False):
            if len(group) < args.min_rows:
                continue
            parts.append(build_minute_features(group))
        if not parts:
            continue

        features = pd.concat(parts, ignore_index=True)
        features["symbol"] = features["symbol"].astype("category")
        cols = feature_columns(features)
        if not all_feature_columns:
            all_feature_columns = cols
        features.to_pickle(target)
        manifest_rows.append(
            {
                "source": str(pkl_path),
                "path": str(target),
                "rows": len(features),
                "symbols": int(features["symbol"].nunique()),
                "feature_count": len(cols),
                "start": str(features["datetime"].min()),
                "end": str(features["datetime"].max()),
            }
        )
        if not wrote_slice:
            features.head(2000).to_pickle(slices_dir / "feature_slice_2000.pkl")
            wrote_slice = True
        print(f"features part={idx + 1}/{len(pkl_files)} rows={len(features)} cols={len(cols)} -> {target}")
        del frame, parts, features
        gc.collect()

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(analysis_dir / "feature_manifest.csv", index=False, encoding="utf-8-sig")
    (analysis_dir / "feature_columns.txt").write_text("\n".join(all_feature_columns), encoding="utf-8")
    write_feature_readme(data_dir, output_dir, manifest, all_feature_columns)
    print(f"Feature manifest saved to {analysis_dir / 'feature_manifest.csv'}")
    print(f"Feature columns saved to {analysis_dir / 'feature_columns.txt'}")


if __name__ == "__main__":
    main()
