"""从已有年度单股因子缓存提取当前分钟交易资格，不重读原始行情或更改年度缓存。

输入manifest.cache_path内的SH/SZ单股票pkl及cfg；输出带指纹的独立资格缓存。
仅使用信号时刻hl_range=(high-low)/close，有限且大于零才有资格；不读取下一分钟
成交字段。沿用执行器拒绝一字线的保守假设，并非真实涨跌停价或排队成交模型。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
KEY_COLUMNS = ["datetime", "symbol"]
FLAG_COLUMNS = [*KEY_COLUMNS, "signal_bar_tradable", "signal_hl_range"]


def validate_signal_flags(flags: pd.DataFrame) -> pd.DataFrame:
    """验证flags的唯一键及必需列；返回日期规范化后的副本，不容许重复资格记录。"""
    missing = set(FLAG_COLUMNS).difference(flags.columns)
    if missing:
        raise ValueError(f"信号资格缺少列：{sorted(missing)}")
    result = flags[FLAG_COLUMNS].copy()
    result["datetime"] = pd.to_datetime(result.datetime, errors="raise")
    if result[KEY_COLUMNS].isna().any().any() or result.duplicated(KEY_COLUMNS).any():
        raise ValueError("信号资格存在缺失或重复的(datetime,symbol)键")
    result["signal_hl_range"] = pd.to_numeric(result.signal_hl_range, errors="coerce")
    # 布尔资格为True且当前价差为正才放行；NaN/错误字符串不能按Python真值放行。
    permitted = result.signal_bar_tradable.eq(True)
    permitted &= np.isfinite(result.signal_hl_range) & (result.signal_hl_range > 0)
    result["signal_bar_tradable"] = permitted
    return result


def write_signal_cache(flags: pd.DataFrame, metadata: dict, path: Path, metadata_path: Path) -> None:
    """原子保存独立资格pkl与JSON元数据；中途失败不留下可误读的半个pkl。"""
    temporary_paths = []
    try:
        for final_path, payload in ((path, flags), (metadata_path, metadata)):
            with tempfile.NamedTemporaryFile(prefix=final_path.name + ".", suffix=".tmp", dir=final_path.parent,
                                             delete=False) as temporary:
                temporary_path = Path(temporary.name)
            temporary_paths.append(temporary_path)
            if isinstance(payload, pd.DataFrame):
                payload.to_pickle(temporary_path)
            else:
                temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            temporary_path.replace(final_path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def load_signal_eligibility(manifest: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """从manifest指定的年度单股缓存建立/加载当前信号资格表及来源元数据。

    cfg.signal_cache_dir为独立输出目录，默认reports/inventory_t_cross_validation/cache。
    指纹包括源目录、年度代码/数据指纹、单股文件大小及mtime、本模块代码；不修改
    manifest或任一年度文件。缓存只保留键和当前hl_range，不保留标签或执行价格。
    """
    project_root = Path(__file__).resolve().parents[1]
    source = Path(manifest["cache_path"])
    if not source.is_absolute():
        source = project_root / source
    source = source.resolve()
    files = sorted([*source.glob("SH.*.pkl"), *source.glob("SZ.*.pkl")])
    if not files:
        raise ValueError(f"年度缓存没有可提取hl_range的SH/SZ单股文件：{source}")
    directory = Path(cfg.get("signal_cache_dir") or project_root / "reports/inventory_t_cross_validation/cache")
    if not directory.is_absolute():
        directory = project_root / directory
    directory.mkdir(parents=True, exist_ok=True)
    file_stats = [[path.name, path.stat().st_size, path.stat().st_mtime_ns] for path in files]
    fingerprint = {"schema": SCHEMA_VERSION, "source_cache_path": str(source),
                   "annual_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest(),
                   "annual_factor_code_sha256": manifest.get("factor_code_sha256"),
                   "annual_panel_code_sha256": manifest.get("panel_code_sha256"),
                   "source_files": file_stats,
                   "signal_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    path = directory / f"signal_eligibility_{key}.pkl"
    metadata_path = directory / f"signal_eligibility_{key}.metadata.json"
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        flags = validate_signal_flags(pd.read_pickle(path))
        if len(flags) != metadata["rows"] or int(flags.signal_bar_tradable.sum()) != metadata["tradable_rows"]:
            raise ValueError("信号资格缓存行数与元数据不一致")
        print(f"复用当前分钟资格缓存：{path}；阻止{metadata['blocked_rows']:,}/{len(flags):,}行", flush=True)
        return flags, {**metadata, "cache_hit": True}
    print(f"提取当前分钟资格：{len(files)}个已有单股缓存；不读取未来成交字段", flush=True)
    parts = []
    for index, source_path in enumerate(files, 1):
        payload = pd.read_pickle(source_path)
        raw = payload["panel"]
        required = {*KEY_COLUMNS, "hl_range"}
        if not required.issubset(raw.columns):
            raise ValueError(f"单股缓存缺少信号字段：{source_path}")
        part = raw[[*KEY_COLUMNS, "hl_range"]].rename(columns={"hl_range": "signal_hl_range"}).copy()
        if not part.symbol.eq(source_path.stem).all():
            raise ValueError(f"单股缓存symbol与文件名不一致：{source_path}")
        part["signal_hl_range"] = pd.to_numeric(part.signal_hl_range, errors="coerce")
        part["signal_bar_tradable"] = np.isfinite(part.signal_hl_range) & (part.signal_hl_range > 0)
        parts.append(part[FLAG_COLUMNS])
        if index % 500 == 0 or index == len(files):
            print(f"当前分钟资格提取 {index}/{len(files)}", flush=True)
    flags = validate_signal_flags(pd.concat(parts, ignore_index=True))
    flags = flags.sort_values(KEY_COLUMNS).reset_index(drop=True)
    positive = int(flags.signal_bar_tradable.sum())
    metadata = {**fingerprint, "fingerprint": key, "cache_path": str(path), "metadata_path": str(metadata_path),
                "source_stock_count": len(files), "rows": len(flags), "tradable_rows": positive,
                "blocked_rows": len(flags) - positive, "cache_hit": False,
                "signal_time_rule": "当前信号分钟hl_range有限且>0；不读取下一分钟成交、未来标签或是否能完成T",
                "missing_flag_rule": "没有唯一信号资格记录的行不允许开仓或训练",
                "rank_definition": "过滤后重新计数cross_section_count；因子cs_rank和target_rank保留原周池口径，不重排",
                "execution_limitation": "当前非一字线不保证下一分钟可成交，执行器仍检查真实成交分钟"}
    write_signal_cache(flags, metadata, path, metadata_path)
    print(f"当前分钟资格缓存完成：允许{positive:,}行，阻止{len(flags)-positive:,}行；{path}", flush=True)
    return flags, metadata


def apply_signal_eligibility(panel: pd.DataFrame, flags: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """用当前flags同时过滤训练/推断panel，缺失匹配关闭资格；保留原因子及标签排名。

    返回新的面板并按过滤后股票数重新计算cross_section_count，不提前删除小截面；
    cfg.min_cross_section由后续训练及开仓排序应用，避免以未来标签筛推断股票。
    """
    if not set(KEY_COLUMNS).issubset(panel.columns):
        raise ValueError("面板缺少datetime/symbol键")
    if panel[KEY_COLUMNS].isna().any().any() or panel.duplicated(KEY_COLUMNS).any():
        raise ValueError("面板存在缺失或重复的(datetime,symbol)键")
    current = validate_signal_flags(flags)
    keys = panel[KEY_COLUMNS].copy()
    keys["datetime"] = pd.to_datetime(keys.datetime, errors="raise")
    positions = pd.MultiIndex.from_frame(current[KEY_COLUMNS]).get_indexer(pd.MultiIndex.from_frame(keys))
    matched = positions >= 0
    eligible = np.zeros(len(panel), dtype=bool)
    eligible[matched] = current.signal_bar_tradable.to_numpy(dtype=bool)[positions[matched]]
    result = panel.loc[eligible].copy()
    result["signal_bar_tradable"] = True
    result["signal_hl_range"] = current.signal_hl_range.to_numpy()[positions[eligible]]
    result["cross_section_count"] = result.groupby("datetime")["symbol"].transform("size")
    print(f"当前分钟资格过滤：{len(panel):,} -> {len(result):,}行；缺失资格{int((~matched).sum()):,}行；"
          f"后续最小截面{cfg.get('min_cross_section', 30)}只", flush=True)
    return result
