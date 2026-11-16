"""Data loading and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OHLCV_ALIASES = {
    "datetime": ["datetime", "date", "time", "trade_time", "timestamp"],
    "symbol": ["symbol", "underlying_symbol", "code", "ticker", "instrument"],
    "open": ["open", "Open", "OPEN"],
    "high": ["high", "High", "HIGH"],
    "low": ["low", "Low", "LOW"],
    "close": ["close", "Close", "CLOSE", "last"],
    "volume": ["volume", "vol", "Volume", "turnover_volume"],
    "amount": ["amount", "turnover", "turnover_amount", "money", "value"],
}


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def standardize_ohlcv(
    data: pd.DataFrame,
    datetime_col: str | None = None,
    symbol_col: str | None = None,
) -> pd.DataFrame:
    """Return a normalized OHLCV frame sorted by symbol and datetime.

    Required output columns are ``datetime``, ``symbol``, ``open``, ``high``,
    ``low``, ``close`` and ``volume``. Missing volume is filled with 0 and a
    single-symbol file receives symbol ``SINGLE``. ``amount`` is optional.
    """

    frame = data.copy()
    columns = list(frame.columns)

    rename: dict[str, str] = {}
    for target, aliases in OHLCV_ALIASES.items():
        if target == "datetime" and datetime_col:
            source = datetime_col
        elif target == "symbol" and symbol_col:
            source = symbol_col
        else:
            source = _find_column(columns, aliases)
        if source and source != target:
            rename[source] = target

    frame = frame.rename(columns=rename)
    if "datetime" not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "datetime"})
        else:
            raise ValueError("Data must contain a datetime column or DatetimeIndex.")

    if "symbol" not in frame.columns:
        frame["symbol"] = "SINGLE"

    if "volume" not in frame.columns:
        frame["volume"] = 0

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    frame["datetime"] = pd.to_datetime(frame["datetime"])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    if "amount" in frame.columns:
        numeric_cols.append("amount")
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["datetime", "open", "high", "low", "close"])
    keep = ["datetime", "symbol", "open", "high", "low", "close", "volume"]
    if "amount" in frame.columns:
        keep.append("amount")
    extras = [c for c in frame.columns if c not in keep]
    frame = frame[keep + extras]
    frame = frame.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return frame


def load_csv(
    path: str | Path,
    datetime_col: str | None = None,
    symbol_col: str | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding=encoding)
    return standardize_ohlcv(frame, datetime_col=datetime_col, symbol_col=symbol_col)


def load_pickle(path: str | Path) -> pd.DataFrame:
    """Load a pickle file and normalize it as OHLCV data."""

    frame = pd.read_pickle(path)
    return standardize_ohlcv(frame)


def load_ohlcv(
    path: str | Path,
    datetime_col: str | None = None,
    symbol_col: str | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from CSV, pickle or parquet."""

    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return load_csv(data_path, datetime_col=datetime_col, symbol_col=symbol_col, encoding=encoding)
    if suffix in {".pkl", ".pickle"}:
        return load_pickle(data_path)
    if suffix == ".parquet":
        return standardize_ohlcv(pd.read_parquet(data_path), datetime_col=datetime_col, symbol_col=symbol_col)
    raise ValueError(f"Unsupported data file type: {data_path.suffix}")


def resample_ohlcv(data: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """Resample each symbol using OHLCV aggregation."""

    frame = standardize_ohlcv(data)
    parts = []
    for symbol, group in frame.groupby("symbol", sort=False):
        group = group.set_index("datetime").sort_index()
        agg = group.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                **({"amount": "sum"} if "amount" in group.columns else {}),
            }
        )
        agg = agg.dropna(subset=["open", "high", "low", "close"])
        agg["symbol"] = symbol
        parts.append(agg.reset_index())
    if not parts:
        return frame.iloc[0:0]
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "datetime"])
