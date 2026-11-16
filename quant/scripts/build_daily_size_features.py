"""拉取 A 股日频行情并派生历史流通市值特征。

输入为稳定股票池和 AKShare 东方财富日行情接口。脚本读取日收盘价、成交量
和换手率，用 ``成交量(手) * 100 / (换手率 / 100)`` 反推流通股本，再乘
日收盘价得到派生流通市值。输出 pkl、csv 和元数据，供小市值策略按分钟
信号日期对齐最近同日的市值特征。脚本只写入 reports/daily_size_features，
不会修改原始分钟行情。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - 预期在 quant conda 环境中可用
    import akshare as ak
except Exception:  # pragma: no cover
    ak = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_xgboost_long_only_backtest as xb


OUTPUT_DIR = PROJECT_ROOT / "reports" / "daily_size_features"


@dataclass(frozen=True)
class SizeFeatureConfig:
    """日频市值特征构建配置。"""

    stable_symbols_file: Path
    local_data_root: Path
    use_local_symbols: bool
    require_years: tuple[str, ...]
    output_dir: Path
    start_date: str
    end_date: str
    include_prefixes: tuple[str, ...]
    max_symbols: int
    workers: int
    timeout: float
    retries: int
    retry_sleep: float


def parse_args() -> argparse.Namespace:
    """解析日频市值特征构建参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-symbols-file", type=Path, default=xb.STABLE_SYMBOLS_FILE)
    parser.add_argument("--local-data-root", type=Path, default=xb.DATA_ROOT)
    parser.add_argument("--use-local-symbols", action="store_true", help="从本地分钟数据目录读取全 A 股票池。")
    parser.add_argument("--require-years", default="", help="使用本地股票池时要求同时存在的年份，例如 2025,2026。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-date", default="20230101")
    parser.add_argument("--end-date", default="20260430")
    parser.add_argument("--include-prefixes", default="", help="只拉取指定本地代码前缀，例如 SH,SZ；留空表示不限制。")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=0.5)
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> SizeFeatureConfig:
    """校验参数并生成配置对象。"""

    if ak is None:
        raise RuntimeError("当前 Python 环境未安装 akshare。")
    require_years = tuple(item.strip() for item in str(args.require_years).split(",") if item.strip())
    include_prefixes = tuple(item.strip().upper() for item in str(args.include_prefixes).split(",") if item.strip())
    return SizeFeatureConfig(
        stable_symbols_file=args.stable_symbols_file,
        local_data_root=args.local_data_root,
        use_local_symbols=bool(args.use_local_symbols),
        require_years=require_years,
        output_dir=args.output_dir,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        include_prefixes=include_prefixes,
        max_symbols=max(int(args.max_symbols), 0),
        workers=max(int(args.workers), 1),
        timeout=float(args.timeout),
        retries=max(int(args.retries), 0),
        retry_sleep=max(float(args.retry_sleep), 0.0),
    )


def convert_symbol_to_akshare(symbol: str) -> str:
    """把本地 `SH.600000` / `SZ.000001` 代码转换为 AKShare 六位代码。"""

    return symbol.split(".")[-1]


def read_symbols(path: Path, max_symbols: int) -> list[str]:
    """读取稳定股票池，并按需截断用于 smoke test。"""

    symbols = xb.read_stable_symbols(path, max_symbols)
    if not symbols:
        raise ValueError(f"没有读取到股票代码：{path}")
    return symbols


def read_local_symbols(data_root: Path, require_years: tuple[str, ...], max_symbols: int) -> list[str]:
    """从本地分钟数据目录读取股票池，并可要求多个年份都存在。"""

    if require_years:
        sets: list[set[str]] = []
        for year in require_years:
            year_dir = data_root / year
            if not year_dir.exists():
                raise FileNotFoundError(f"找不到年份目录：{year_dir}")
            sets.append({path.stem for path in year_dir.glob("*.csv")})
        symbols = sorted(set.intersection(*sets)) if sets else []
    else:
        symbols = sorted({path.stem for path in data_root.glob("*/*.csv")})
    if max_symbols:
        symbols = symbols[:max_symbols]
    if not symbols:
        raise ValueError(f"没有从本地数据目录读取到股票代码：{data_root}")
    return symbols


def filter_symbols_by_prefix(symbols: list[str], include_prefixes: tuple[str, ...]) -> list[str]:
    """按本地代码前缀过滤股票池，供全 A 拉取时排除暂不研究的板块。"""

    if not include_prefixes:
        return symbols
    allowed = tuple(f"{prefix}." for prefix in include_prefixes)
    filtered = [symbol for symbol in symbols if symbol.upper().startswith(allowed)]
    if not filtered:
        raise ValueError(f"按前缀 {include_prefixes} 过滤后没有股票代码。")
    return filtered


def fetch_one_symbol(symbol: str, config: SizeFeatureConfig) -> pd.DataFrame:
    """拉取单只股票日行情，并派生流通市值。

    symbol: 本地股票代码，例如 `SZ.000001`。
    config: 日频市值特征构建配置，提供日期范围和请求超时。
    """

    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=convert_symbol_to_akshare(symbol),
                period="daily",
                start_date=config.start_date,
                end_date=config.end_date,
                adjust="",
                timeout=config.timeout,
            )
            break
        except Exception as exc:  # pragma: no cover - 网络异常路径
            last_error = exc
            if attempt >= config.retries:
                raise
            time.sleep(config.retry_sleep * (attempt + 1))
    else:  # pragma: no cover - 仅为类型检查保底
        raise RuntimeError(f"{symbol} 日频数据拉取失败：{last_error!r}")
    if raw.empty:
        return pd.DataFrame()
    frame = raw.rename(
        columns={
            "日期": "date",
            "收盘": "close",
            "成交量": "volume_hands",
            "成交额": "amount",
            "换手率": "turnover_rate",
        }
    ).copy()
    frame["symbol"] = symbol
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    for col in ["close", "volume_hands", "amount", "turnover_rate"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid_turnover = frame["turnover_rate"].where(frame["turnover_rate"] > 0)
    frame["derived_float_shares"] = frame["volume_hands"] * 100.0 / (valid_turnover / 100.0)
    frame["derived_float_market_cap"] = frame["derived_float_shares"] * frame["close"]
    frame["log_float_market_cap"] = np.log(frame["derived_float_market_cap"].where(frame["derived_float_market_cap"] > 0))
    return frame[
        [
            "symbol",
            "date",
            "close",
            "volume_hands",
            "amount",
            "turnover_rate",
            "derived_float_shares",
            "derived_float_market_cap",
            "log_float_market_cap",
        ]
    ].copy()


def build_daily_size_features(config: SizeFeatureConfig) -> Path:
    """并发拉取股票池日频市值特征并写出缓存。"""

    if config.use_local_symbols:
        symbols = read_local_symbols(config.local_data_root, config.require_years, 0)
    else:
        symbols = read_symbols(config.stable_symbols_file, 0)
    symbols = filter_symbols_by_prefix(symbols, config.include_prefixes)
    if config.max_symbols:
        symbols = symbols[: config.max_symbols]
    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(fetch_one_symbol, symbol, config): symbol for symbol in symbols}
        for idx, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                frame = future.result()
                if frame.empty:
                    failures.append({"symbol": symbol, "error": "empty"})
                else:
                    frames.append(frame)
            except Exception as exc:  # pragma: no cover - 网络异常路径
                failures.append({"symbol": symbol, "error": repr(exc)})
            if idx % 25 == 0 or idx == len(symbols):
                print(f"已处理日频市值特征 {idx}/{len(symbols)}", flush=True)
    if not frames:
        raise RuntimeError(f"没有成功拉取任何日频市值数据：{failures[:5]}")
    result = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    pkl_path = run_dir / "daily_size_features.pkl"
    csv_path = run_dir / "daily_size_features.csv"
    result.to_pickle(pkl_path)
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(run_dir / "fetch_failures.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "symbol_count_requested": len(symbols),
        "symbol_count_success": int(result["symbol"].nunique()),
        "rows": int(len(result)),
        "start_date": str(result["date"].min()),
        "end_date": str(result["date"].max()),
        "failure_count": len(failures),
        "output_pkl": str(pkl_path),
        "calculation": "derived_float_market_cap = close * volume_hands * 100 / (turnover_rate / 100)",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 日频小市值特征缓存

- 成功股票数：{metadata['symbol_count_success']} / {metadata['symbol_count_requested']}
- 行数：{metadata['rows']}
- 日期范围：{metadata['start_date']} 至 {metadata['end_date']}
- pkl：`daily_size_features.pkl`
- csv：`daily_size_features.csv`
- 计算方法：`derived_float_market_cap = close * volume_hands * 100 / (turnover_rate / 100)`

注意：这是从换手率反推的流通市值，不是交易所原始披露字段。若换手率为 0 或缺失，该日市值为 NaN。
"""
    (run_dir / "README.md").write_text(report, encoding="utf-8")
    latest_path = config.output_dir / "latest"
    if latest_path.is_symlink() or not latest_path.exists():
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(run_dir.name, target_is_directory=True)
    else:
        print(f"latest 路径已存在且不是符号链接，跳过更新：{latest_path}", flush=True)
    print(f"日频市值特征已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行日频市值特征构建。"""

    build_daily_size_features(create_config(parse_args()))


if __name__ == "__main__":
    main()
