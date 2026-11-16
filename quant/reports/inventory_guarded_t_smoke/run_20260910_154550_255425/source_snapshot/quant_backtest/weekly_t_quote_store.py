"""将真实恢复分钟写成只读float64 mmap，供spawn候选进程共享，不改输入行情。

只接受中国本地无时区的整分钟09:31–11:30/13:01–15:00，不补缺行、不查未来。
数值沿用RecoveryBarStore语义：NaN和真实0保留，负数及无穷值拒绝；成交资格仍由
引擎判断。present.npy独立区分缺失行与包含NaN/0的实际行。缓存以输入及源码哈希
寻址，先在同目录临时位置完整写出再原子发布；pickle仅保存路径和身份摘要。
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType

import numpy as np
import pandas as pd

from . import weekly_t_engine
from .weekly_t_engine import RecoveryBarStore


QUOTE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
SCHEMA_VERSION = 1
MINUTE_NS = 60_000_000_000


def calculate_file_sha256(path: Path) -> str:
    """流式读取本地缓存/源码并计算SHA256，不将整个数组文件复制进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_source_hashes() -> dict:
    """缓存身份包含当前存储实现及继承的引擎源码，禁止跨实现版本误复用。"""
    paths = (Path(__file__), Path(weekly_t_engine.__file__))
    return {path.name: calculate_file_sha256(path) for path in paths}


def encode_metadata(value: dict | list) -> bytes:
    """用确定的JSON字节表示元数据，保证输入行排序不改变缓存身份。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def validate_session_times(timestamps: pd.DatetimeIndex) -> None:
    """严格校验本地整分钟时钟；日期交易资格由上游日历控制，不推断未来交易日。"""
    if timestamps.tz is not None or timestamps.hasnans:
        raise ValueError("quote datetime必须为无时区、非缺失的中国本地时间")
    if (timestamps.asi8 % MINUTE_NS != 0).any():
        raise ValueError("quote datetime必须使用整分钟时间标签")
    clock = timestamps.hour * 60 + timestamps.minute
    if not (((clock >= 571) & (clock <= 690)) | ((clock >= 781) & (clock <= 900))).all():
        raise ValueError("quote datetime必须在09:31–11:30或13:01–15:00，午休不合法")


def normalize_quote_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """复制并规范8列行情，保留float64/NaN/0；额外列不进入报价或缓存身份。"""
    required = ["datetime", "symbol", *QUOTE_COLUMNS]
    if not isinstance(bars, pd.DataFrame) or not set(required).issubset(bars.columns):
        raise ValueError(f"quotes须含列{required}")
    frame = bars[required].copy(deep=True)
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
    validate_session_times(pd.DatetimeIndex(frame.datetime))
    if frame.symbol.isna().any():
        raise ValueError("quote symbol不能缺失")
    frame["symbol"] = frame.symbol.astype(str)
    if frame.symbol.str.strip().eq("").any():
        raise ValueError("quote symbol不能为空")
    if frame.duplicated(["datetime", "symbol"]).any():
        raise ValueError("quote datetime/symbol必须唯一")
    for column in QUOTE_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if pd.api.types.is_complex_dtype(numeric.dtype):
            raise ValueError(f"quote {column}不能为复数")
        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"quote {column}存在负数或无穷值")
        values[np.isnan(values)] = np.nan
        frame[column] = values
    return frame.sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)


def write_array_file(path: Path, values: np.ndarray) -> None:
    """将轻量索引写为禁用pickle的npy并同步到磁盘，供原子目录发布。"""
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def synchronize_file(path: Path) -> None:
    """确保已flush的mmap文件持久化；仅用于构建进程。"""
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


class DenseRecoveryBarStore(RecoveryBarStore):
    """复用引擎接口的只读稠密行情；每个worker只创建轻量索引及只读mmap映射。

cache_path是build_dense_recovery_store生成的完整目录。直接构造默认复核文件
SHA256；spawn反序列化校验元数据身份/源码/尺寸/mtime后只映射，不重新扫描大数组。
不继承父类构造中的尾盘筛选，也不建立其巨大Python行字典。
    """

    def __init__(self, cache_path: str | Path, *, verify_integrity: bool = True):
        """从完整已发布缓存打开数组；verify_integrity=False仅供已校验构建/worker。"""
        self._load_cache(Path(cache_path), verify_integrity=verify_integrity)

    def _load_cache(self, path: Path, *, verify_integrity: bool,
                    expected_fingerprint: str | None = None,
                    expected_metadata_sha256: str | None = None) -> None:
        """校验缓存身份与数组头，再加载只读视图；异常失败关闭，不尝试修补缓存。"""
        path = path.resolve()
        try:
            raw_metadata = (path / "metadata.json").read_bytes()
            metadata = json.loads(raw_metadata)
            identity = metadata["identity"]
            fingerprint = hashlib.sha256(encode_metadata(identity)).hexdigest()
            metadata_sha256 = hashlib.sha256(raw_metadata).hexdigest()
            if (metadata["fingerprint"] != fingerprint
                    or path.name != f"dense_recovery_{fingerprint}"
                    or (expected_fingerprint is not None and fingerprint != expected_fingerprint)
                    or (expected_metadata_sha256 is not None and metadata_sha256 != expected_metadata_sha256)
                    or identity["schema_version"] != SCHEMA_VERSION
                    or identity["quote_columns"] != list(QUOTE_COLUMNS)
                    or identity["source_sha256"] != calculate_source_hashes()):
                raise ValueError("dense quote缓存身份或源码不匹配")
            symbols = metadata["symbols"]
            if (not isinstance(symbols, list) or any(not isinstance(x, str) or not x.strip() for x in symbols)
                    or symbols != sorted(set(symbols))
                    or hashlib.sha256(encode_metadata(symbols)).hexdigest() != identity["symbols_sha256"]):
                raise ValueError("dense quote symbol索引非法")
            time_count = int(metadata["time_count"])
            if time_count != identity["time_count"]:
                raise ValueError("dense quote时间索引长度与身份不匹配")
            shape = (time_count, len(symbols))
            expected_arrays = {"times.npy": ((time_count,), np.dtype("int64")),
                               "quotes.npy": ((*shape, 6), np.dtype("float64")),
                               "present.npy": (shape, np.dtype("uint8"))}
            if time_count < 0 or set(metadata["arrays"]) != set(expected_arrays):
                raise ValueError("dense quote数组清单非法")
            arrays = {}
            for name, (expected_shape, expected_dtype) in expected_arrays.items():
                target, recorded = path / name, metadata["arrays"][name]
                stat = target.stat()
                if stat.st_size != recorded["size_bytes"] or stat.st_mtime_ns != recorded["mtime_ns"]:
                    raise ValueError(f"dense quote {name}尺寸或修改时间不匹配")
                if verify_integrity and calculate_file_sha256(target) != recorded["sha256"]:
                    raise ValueError(f"dense quote {name}内容哈希不匹配")
                mode = "r" if np.prod(expected_shape, dtype=np.int64) else None
                array = np.load(target, mmap_mode=mode, allow_pickle=False)
                if array.shape != expected_shape or array.dtype != expected_dtype:
                    raise ValueError(f"dense quote {name}形状或dtype不匹配")
                array.flags.writeable = False
                arrays[name] = array
            times = arrays["times.npy"]
            if hashlib.sha256(times.astype("<i8", copy=False).tobytes()).hexdigest() != identity["times_sha256"]:
                raise ValueError("dense quote时间索引哈希与身份不匹配")
            validate_session_times(pd.DatetimeIndex(times.view("datetime64[ns]")))
            if len(times) > 1 and not (np.diff(times) > 0).all():
                raise ValueError("dense quote时间索引必须严格递增")
            row_count = int(identity["row_count"])
            if not 0 <= row_count <= time_count * len(symbols):
                raise ValueError("dense quote行数非法")
        except (OSError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError(f"dense quote缓存不完整或非法：{path}") from exc
        self.cache_path = path
        self.fingerprint = fingerprint
        self.row_count = row_count
        self.symbols = frozenset(symbols)
        self.cache_hit = True
        self.metadata = MappingProxyType(metadata)
        self._metadata_sha256 = metadata_sha256
        self._times = times
        self._quotes = arrays["quotes.npy"]
        self._present = arrays["present.npy"]
        self._time_lookup = MappingProxyType({int(value): index for index, value in enumerate(times)})
        self._symbol_lookup = MappingProxyType({value: index for index, value in enumerate(symbols)})

    def get_quote(self, timestamp: pd.Timestamp, symbol: str):
        """仅返回指定真实行的只读OHLCVA映射；缺行/非法查询返回None，不查后续分钟。"""
        try:
            value = pd.Timestamp(timestamp)
            if pd.isna(value) or value.tz is not None:
                return None
            time_index = self._time_lookup.get(value.value)
            symbol_index = self._symbol_lookup.get(symbol)
        except (TypeError, ValueError, OverflowError):
            return None
        if time_index is None or symbol_index is None or not self._present[time_index, symbol_index]:
            return None
        row = self._quotes[time_index, symbol_index]
        return MappingProxyType({key: float(row[index]) for index, key in enumerate(QUOTE_COLUMNS)})

    def __getstate__(self) -> dict:
        """spawn只传路径及身份摘要；不pickle任何行情数组或Python逐行报价字典。"""
        return {"cache_path": str(self.cache_path), "fingerprint": self.fingerprint,
                "metadata_sha256": self._metadata_sha256}

    def __setstate__(self, state: dict) -> None:
        """worker校验轻量身份/文件stat后只读mmap，避免复制或重复全文件哈希扫描。"""
        self._load_cache(Path(state["cache_path"]), verify_integrity=False,
                         expected_fingerprint=state["fingerprint"],
                         expected_metadata_sha256=state["metadata_sha256"])


def build_dense_recovery_store(bars: pd.DataFrame, cache_dir: str | Path) -> DenseRecoveryBarStore:
    """校验8列真实行情并原子构建/复用内容寻址mmap；只创建cache_dir下的新生成物。

输入应覆盖整折股票并集所需日期/时钟，不能按某周持仓或未来恢复成功筛选。
相同行情不受输入行序/额外列影响；已有缓存损坏时失败关闭，不覆盖历史缓存。
    """
    frame = normalize_quote_frame(bars)
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes()).hexdigest()
    times = np.sort(frame.datetime.unique().astype("datetime64[ns]").view("int64"))
    symbols = sorted(frame.symbol.unique())
    identity = {"schema_version": SCHEMA_VERSION, "quote_columns": list(QUOTE_COLUMNS),
                "input_sha256": input_hash, "row_count": len(frame),
                "time_count": len(times),
                "times_sha256": hashlib.sha256(times.astype("<i8", copy=False).tobytes()).hexdigest(),
                "symbols_sha256": hashlib.sha256(encode_metadata(symbols)).hexdigest(),
                "source_sha256": calculate_source_hashes(), "pandas_version": pd.__version__,
                "numpy_version": np.__version__, "timezone": "Asia/Shanghai naive",
                "session": "09:31–11:30/13:01–15:00; exact minutes; missing rows not filled"}
    fingerprint = hashlib.sha256(encode_metadata(identity)).hexdigest()
    directory = Path(cache_dir).resolve()
    final = directory / f"dense_recovery_{fingerprint}"
    if final.exists():
        store = DenseRecoveryBarStore(final)
        if store.fingerprint != fingerprint:
            raise ValueError("dense quote已有缓存与输入身份不匹配")
        return store
    directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dense_recovery_build_", dir=directory))
    published = False
    try:
        shape = (len(times), len(symbols))
        write_array_file(temporary / "times.npy", times)
        if len(frame):
            quotes = np.lib.format.open_memmap(temporary / "quotes.npy", mode="w+", dtype="float64", shape=(*shape, 6))
            present = np.lib.format.open_memmap(temporary / "present.npy", mode="w+", dtype="uint8", shape=shape)
            quotes[:] = np.nan
            present[:] = 0
            time_indices = np.searchsorted(times, frame.datetime.to_numpy(dtype="datetime64[ns]").view("int64"))
            symbol_indices = pd.Index(symbols).get_indexer(frame.symbol)
            quotes[time_indices, symbol_indices] = frame[list(QUOTE_COLUMNS)].to_numpy(dtype="float64")
            present[time_indices, symbol_indices] = 1
            quotes.flush()
            present.flush()
            del quotes, present
            synchronize_file(temporary / "quotes.npy")
            synchronize_file(temporary / "present.npy")
        else:
            write_array_file(temporary / "quotes.npy", np.empty((*shape, 6), dtype="float64"))
            write_array_file(temporary / "present.npy", np.empty(shape, dtype="uint8"))
        records = {}
        for name in ("times.npy", "quotes.npy", "present.npy"):
            path = temporary / name
            stat = path.stat()
            records[name] = {"sha256": calculate_file_sha256(path), "size_bytes": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns}
        metadata = {"identity": identity, "fingerprint": fingerprint, "time_count": len(times),
                    "symbols": symbols, "arrays": records,
                    "missing_row_encoding": "present=0; actual NaN/zero rows have present=1"}
        with (temporary / "metadata.json").open("wb") as stream:
            stream.write(encode_metadata(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.rename(temporary, final)
            published = True
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not final.is_dir():
                raise
        store = DenseRecoveryBarStore(final, verify_integrity=not published)
        if store.fingerprint != fingerprint:
            raise ValueError("dense quote发布后的缓存身份不匹配")
        store.cache_hit = not published
        return store
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
