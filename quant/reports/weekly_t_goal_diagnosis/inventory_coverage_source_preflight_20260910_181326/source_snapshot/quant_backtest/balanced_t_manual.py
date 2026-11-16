"""人工底仓买单的年度上下文及少量真实分钟报价补充，不修改原缓存或模型输入。

完整人工CSV被视为用户事先给出的外部买入指令。仅按研究年份和声明的时间范围
切片；有效范围内的非法日期/时刻/股数必须拒绝，不能静默丢掉。报价只查请求
当前分钟，缺行保留缺行，不向后寻找可成交价格。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np
import pandas as pd

from .weekly_t_engine import RecoveryBarStore, is_recovery_minute

QUOTE_COLUMNS = ('open', 'high', 'low', 'close', 'volume', 'amount')
MAIN_BOARD_SYMBOL = re.compile(r'(?:SH\.(?:600|601|603|605)|SZ\.(?:000|001|002|003))\d{3}\Z')


def fingerprint_manual_input(path: Path, inputs: dict) -> dict:
    """完整文件字节SHA写入输入清单，扫描期间发生变化则失败关闭。"""
    path = path.resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'人工上下文源文件在读取时发生变化: {path}')
    # 与balanced_t_data的公共inputs合同一致；mtime仅用于本次扫描的并发变化检查。
    record = {'sha256': digest.hexdigest(), 'size_bytes': after.st_size}
    if str(path) in inputs and inputs[str(path)] != record:
        raise ValueError(f'人工上下文源文件与已有输入身份冲突: {path}')
    inputs[str(path)] = record
    return record


class ManualRecoveryStore(RecoveryBarStore):
    """兼容旧RecoveryBarStore的只读当前分钟覆盖；spawn只传少量覆盖和底层store。

    overrides包含(timestamp,symbol,quote或None)。显式None代表已检查但原始分钟
    缺失，不能让未来数据或其他时间的报价补上。其他键原样交给base_store。
    """
    def __init__(self, base_store: RecoveryBarStore | None, overrides: tuple | list):
        self._manual_base_store = base_store
        mapping = {}
        for timestamp, symbol, quote in overrides:
            key = (pd.Timestamp(timestamp), str(symbol))
            if key in mapping:
                raise ValueError('人工报价覆盖键不能重复')
            mapping[key] = None if quote is None else MappingProxyType(dict(quote))
        self._manual_quotes = MappingProxyType(mapping)

    def get_quote(self, timestamp: pd.Timestamp, symbol: str) -> dict | None:
        """只读取请求的当前键，不扫描其他报价或重置底层缓存。"""
        key = (pd.Timestamp(timestamp), str(symbol))
        if key in self._manual_quotes:
            value = self._manual_quotes[key]
            return None if value is None else dict(value)
        return self._manual_base_store.get_quote(*key) if self._manual_base_store is not None else None

    def __reduce__(self):
        """保留底层store的原pickle协议，稀疏人工覆盖只复制几条普通记录。"""
        rows = tuple((timestamp, symbol, None if value is None else dict(value))
                     for (timestamp, symbol), value in self._manual_quotes.items())
        return type(self), (self._manual_base_store, rows)


def prepare_manual_context(cfg: dict, evaluation_year: int, days: pd.DatetimeIndex,
                           base_store: RecoveryBarStore | None, inputs: dict) -> tuple:
    """返回local_cfg/本折人工买单/报价store/审计；null配置不读CSV或raw。

    days为本折实际市场日历，包括无仓日。主板代码先校验；可通过配置
    manual_supported_symbols额外核对当前日行情支持的股票池，否则runner必须核对
    audit.market_membership_check_required中的股票。09:30/15:00由引擎使用daily。
    """
    local_cfg = {**cfg, 'manual_buys_csv': None}
    audit = {'classification': 'external_user_instructions_not_model_inputs', 'evaluation_year': int(evaluation_year),
             'configured': bool(cfg.get('manual_buys_csv')), 'rows_in_file': 0, 'rows_in_scope': 0,
             'future_prices_used_to_filter_requests': False, 'raw_sources': {}, 'missing_raw_files': [],
             'requested_raw_minutes': 0, 'raw_minutes_found': 0, 'raw_minutes_missing': 0,
             'existing_current_cache_quotes_used': 0, 'daily_quote_requests': 0,
             'market_membership_check_required': [], 'manual_amount_band_exception': True}
    if not cfg.get('manual_buys_csv'):
        return local_cfg, None, base_store, audit
    csv_path = Path(cfg['manual_buys_csv']).expanduser().resolve()
    audit['manual_csv_path'] = str(csv_path)
    audit['manual_csv_identity'] = fingerprint_manual_input(csv_path, inputs)
    frame = pd.read_csv(csv_path, float_precision='round_trip')
    if not {'datetime', 'signal_datetime', 'symbol', 'shares'}.issubset(frame):
        raise ValueError('manual CSV须包含datetime/signal_datetime/symbol/shares')
    frame['datetime'] = pd.to_datetime(frame.datetime, format='mixed', errors='raise')
    frame['signal_datetime'] = pd.to_datetime(frame.signal_datetime, format='mixed', errors='raise')
    frame['shares'] = pd.to_numeric(frame.shares, errors='raise')
    lot = int(cfg.get('lot_size', 100))
    if (frame.datetime.isna().any() or frame.signal_datetime.isna().any()
            or frame.datetime.dt.tz is not None or frame.signal_datetime.dt.tz is not None
            or not frame.datetime.gt(frame.signal_datetime).all()
            or not np.isfinite(frame.shares).all() or not (frame.shares.gt(0) & frame.shares.mod(lot).eq(0)).all()
            or frame.symbol.isna().any() or frame.duplicated(['datetime', 'symbol']).any()):
        raise ValueError('manual CSV时间、正整手、股票或重复键不合法')
    frame['symbol'] = frame.symbol.astype(str)
    if not frame.symbol.map(lambda value: bool(MAIN_BOARD_SYMBOL.fullmatch(value))).all():
        raise ValueError('manual CSV只支持原A股主板代码，股票须以SH./SZ.表示')
    if 'side' in frame and not frame.side.eq('buy').all():
        raise ValueError('人工上下文仅支持基础买入，不允许暗含卖出')
    calendar = pd.DatetimeIndex(days)
    if calendar.empty or calendar.hasnans or calendar.tz is not None:
        raise ValueError('人工上下文须有非空、无时区的实际市场日历')
    calendar = calendar.normalize().unique().sort_values()
    if not (calendar.year == int(evaluation_year)).all():
        raise ValueError('人工上下文days只能包含本评价年度')
    start = pd.Timestamp(year=int(evaluation_year), month=1, day=1)
    end = min(pd.Timestamp(year=int(evaluation_year), month=12, day=31), calendar.max())
    if cfg.get('start_date'):
        start = max(start, pd.Timestamp(cfg['start_date']).normalize())
    if cfg.get('end_date'):
        end = min(end, pd.Timestamp(cfg['end_date']).normalize())
    normalized = frame.datetime.dt.normalize()
    mask = normalized.between(start, end)
    selected = frame.loc[mask].copy().sort_values(['datetime', 'symbol'], kind='stable').reset_index(drop=True)
    audit.update({'rows_in_file': len(frame), 'rows_in_scope': len(selected),
                  'rows_outside_scope': int((~mask).sum()), 'scope_start': str(start.date()), 'scope_end': str(end.date()),
                  'outside_scope_policy': '仅按已声明年份/配置起止/实际评价截止切片；不按可成交性筛选'})
    # 必须在范围切片之后明确拒绝周末，不能用isin(calendar)把坏指令悄悄删掉。
    if not selected.datetime.dt.normalize().isin(calendar).all():
        raise ValueError('研究范围内人工加仓日期不属于实际交易日，不能静默过滤')
    for timestamp in selected.datetime:
        if timestamp.time() != pd.Timestamp('09:30').time() and not is_recovery_minute(timestamp):
            raise ValueError('研究范围内人工加仓时刻非法，仅支持09:30或会话内整分钟')
    symbols = sorted(selected.symbol.unique())
    if cfg.get('manual_supported_symbols') is not None:
        if not set(symbols).issubset(set(cfg['manual_supported_symbols'])):
            raise ValueError('人工股票不在本年真实日行情支持的主板池中')
        audit['market_membership_verified'] = True
    else:
        audit['market_membership_verified'] = False
        audit['market_membership_check_required'] = symbols
    wanted = defaultdict(list)
    for row in selected.itertuples(index=False):
        timestamp = row.datetime
        if timestamp.time() in {pd.Timestamp('09:30').time(), pd.Timestamp('15:00').time()}:
            audit['daily_quote_requests'] += 1
        elif base_store is not None and base_store.get_quote(timestamp, row.symbol) is not None:
            audit['existing_current_cache_quotes_used'] += 1
        else:
            wanted[(timestamp.year, row.symbol)].append(timestamp)
    overrides = []
    root = Path(cfg.get('data_root', '.')).expanduser().resolve()
    for (year, symbol), timestamps in sorted(wanted.items()):
        path = root / str(year) / f'{symbol}.csv'
        audit['requested_raw_minutes'] += len(timestamps)
        if not path.is_file():
            audit['missing_raw_files'].append(str(path))
            overrides.extend((timestamp, symbol, None) for timestamp in timestamps)
            audit['raw_minutes_missing'] += len(timestamps)
            continue
        identity = fingerprint_manual_input(path, inputs)
        audit['raw_sources'][str(path)] = identity
        raw = pd.read_csv(path, usecols=['datetime', *QUOTE_COLUMNS], float_precision='round_trip')
        raw['datetime'] = pd.to_datetime(raw.datetime, format='mixed', errors='raise')
        current = raw.loc[raw.datetime.isin(timestamps)].copy()
        if current.datetime.duplicated().any():
            raise ValueError('人工请求当前原始分钟存在重复，拒绝选择任意价格')
        for column in QUOTE_COLUMNS:
            current[column] = pd.to_numeric(current[column], errors='raise')
        mapping = current.set_index('datetime').to_dict('index')
        for timestamp in timestamps:
            value = mapping.get(timestamp)
            overrides.append((timestamp, symbol, value))
            audit['raw_minutes_found' if value is not None else 'raw_minutes_missing'] += 1
    store = ManualRecoveryStore(base_store, overrides) if overrides else base_store
    audit['overlay_keys'] = len(overrides)
    return local_cfg, selected, store, audit
