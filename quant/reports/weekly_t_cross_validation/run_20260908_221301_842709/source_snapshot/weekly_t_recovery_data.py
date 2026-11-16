"""独立的14:55–15:00恢复成交数据缓存，不改变周预测或日内信号的冻结缓存。

只读主板分钟CSV并复用weekly_t_data的主板范围、OHLCV校验和原子写文件逻辑。
输出只有真实成交行情，没有预测特征；缺失分钟不填充，零量分钟保留供执行器
拒绝成交。恢复行情仅在到达其时间后可执行，禁止用于筛掉不能恢复的首腿。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd

from .weekly_t_data import MAINBOARD_PREFIXES, load_weekly_symbol_bars, write_weekly_cache


SCHEMA_VERSION = 1
RECOVERY_COLUMNS = ["datetime", "symbol", "open", "high", "low", "close", "volume", "amount"]
RECOVERY_MINUTES = ["14:55", "14:56", "14:57", "14:58", "14:59", "15:00"]


def select_recovery_bars(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """从已校验单股raw中提取六个精确分钟，不补缺失或剔除零量，返回固定八列。"""
    timestamp = pd.to_datetime(raw.datetime, errors="raise")
    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    exact = (minute.between(895, 900) & timestamp.dt.second.eq(0)
             & timestamp.dt.microsecond.eq(0) & timestamp.dt.nanosecond.eq(0))
    selected = raw.loc[exact].copy()
    selected["symbol"] = symbol
    return selected[RECOVERY_COLUMNS].sort_values("datetime").reset_index(drop=True)


def build_recovery_symbol_cache(task: tuple) -> str:
    """执行(symbol,源文件列表,start,end,缓存目录)任务，分股pkl支持中断后复用。"""
    symbol, paths, start, end, directory = task
    path = Path(directory) / f"{symbol}.pkl"
    if path.exists():
        return str(path)
    raw, audit = load_weekly_symbol_bars(paths, start, end)
    bars = select_recovery_bars(raw, symbol)
    audit.update({"symbol": symbol, "recovery_rows": len(bars),
                  "recovery_zero_volume_rows": int(bars.volume.eq(0).sum()),
                  "recovery_zero_amount_rows": int(bars.amount.eq(0).sum())})
    write_weekly_cache({"bars": bars, "audit": audit}, path)
    return str(path)


def load_recovery_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """建立/复用全主板六分钟恢复表，返回bars和包含输入指纹、质量统计的manifest。

    cfg读取data_root/start_date/end_date/max_symbols/workers/recovery_cache_dir。
    独立缓存指纹涵盖源CSV路径、大小、mtime、日期范围、固定六时刻及本模块和
    被复用读取模块的代码哈希。不会训练模型、改写原始数据或重建周/信号缓存。
    """
    project_root = Path(__file__).resolve().parents[1]
    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()
    start, end = str(cfg.get("start_date", "2023-01-01")), str(cfg.get("end_date", "2026-04-30"))
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start_date不能晚于end_date")
    maximum = int(cfg.get("max_symbols", 0))
    workers = int(cfg.get("workers", 4))
    if maximum < 0 or workers < 1:
        raise ValueError("max_symbols须非负，workers须至少为1")
    paths = sorted(path for path in root.glob("[0-9][0-9][0-9][0-9]/*.csv")
                   if path.stem.startswith(MAINBOARD_PREFIXES) and pd.Timestamp(start).year <= int(path.parent.name) <= pd.Timestamp(end).year)
    symbols = sorted({path.stem for path in paths})
    if maximum:
        symbols = symbols[:maximum]
    allowed = set(symbols)
    paths = [path for path in paths if path.stem in allowed]
    if not paths:
        raise ValueError("指定目录及日期范围没有沪深主板CSV")
    source_stats = [[str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]
    fingerprint = {"schema": SCHEMA_VERSION, "data_root": str(root), "files": source_stats,
                   "symbols": symbols, "start_date": start, "end_date": end,
                   "recovery_minutes": RECOVERY_MINUTES,
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "reader_code_sha256": hashlib.sha256(Path(__file__).with_name("weekly_t_data.py").read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    cache_root = Path(cfg.get("recovery_cache_dir") or project_root / "reports/weekly_t_cross_validation/recovery_cache")
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    cache = cache_root / f"recovery_{key}"
    cache.mkdir(parents=True, exist_ok=True)
    bars_path, metadata_path = cache / "recovery_bars.pkl", cache / "metadata.json"
    if bars_path.exists() and metadata_path.exists():
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
        bars = pd.read_pickle(bars_path)
        if (manifest.get("fingerprint") != key or list(bars.columns) != RECOVERY_COLUMNS
                or len(bars) != manifest["rows"]):
            raise ValueError("恢复缓存指纹、列或行数与元数据不一致")
        print(f"复用独立恢复缓存：{cache}；{len(bars):,}条真实分钟", flush=True)
        return bars, {**manifest, "cache_hit": True}
    files_by_symbol = {symbol: [] for symbol in symbols}
    for path in paths:
        files_by_symbol[path.stem].append(str(path))
    tasks = [(symbol, files_by_symbol[symbol], start, end, str(cache)) for symbol in symbols]
    print(f"构建独立恢复缓存：{len(tasks)}股/{workers}进程，仅提取14:55–15:00实际分钟", flush=True)
    fragments = []
    if workers == 1:
        for index, path in enumerate(map(build_recovery_symbol_cache, tasks), 1):
            fragments.append(path)
            if index % 250 == 0 or index == len(tasks):
                print(f"恢复行情分股缓存 {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, path in enumerate(pool.map(build_recovery_symbol_cache, tasks), 1):
                fragments.append(path)
                if index % 250 == 0 or index == len(tasks):
                    print(f"恢复行情分股缓存 {index}/{len(tasks)}", flush=True)
    parts, audits = [], []
    for path in fragments:
        payload = pd.read_pickle(path)
        parts.append(payload["bars"])
        audits.append(payload["audit"])
    bars = pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    del parts
    if bars.duplicated(["datetime", "symbol"]).any():
        raise ValueError("汇总恢复行情出现重复(datetime,symbol)键")
    quality = pd.DataFrame(audits)
    manifest = {**fingerprint, "fingerprint": key, "cache_path": str(cache), "bars_path": str(bars_path),
                "cache_hit": False, "rows": len(bars), "source_symbol_count": len(symbols),
                "symbol_count": int(bars.symbol.nunique()),
                "first_datetime": str(bars.datetime.min()), "last_datetime": str(bars.datetime.max()),
                "zero_volume_rows": int(bars.volume.eq(0).sum()), "zero_amount_rows": int(bars.amount.eq(0).sum()),
                "source_duplicates": int(quality.duplicates.sum()), "source_invalid_bars": int(quality.invalid_bars.sum()),
                "columns": RECOVERY_COLUMNS, "universe_rule": "所有历史主板文件并集；不使用当前名称/ST或未来存续筛选",
                "missing_data_rule": "缺失或非法行情不生成bar；合法零量/零成交额和一字线原样保留，不补成交价格",
                "execution_contract": "本表仅提供到达该datetime时的恢复执行行情，不能作为模型特征或首腿准入条件"}
    write_weekly_cache(bars, bars_path)
    quality.to_csv(cache / "data_quality.csv", index=False)
    write_weekly_cache(manifest, metadata_path)
    print(f"独立恢复缓存完成：{len(bars):,}行，零量{manifest['zero_volume_rows']:,}行；{cache}", flush=True)
    return bars, manifest
