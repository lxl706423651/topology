"""扩展CTA特征并持久缓存，不拟合参数、不读取标签或旧预测。

输入为已审计分股原因子缓存与历史日线；输出独立PKL、定义表和输入指纹。
日线通过严格早于信号日期的连接提供，禁止把当日收盘资料用于日内决策。
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_panel import write_json
from .cta_research_data import calculate_identity, calculate_sha256, read_history, save_pickle


def build_daily_features(history: pd.DataFrame, lags: int, windows: list[int]) -> pd.DataFrame:
    """按history已完成日OHLCV构造lags期价量序列和windows滚动统计；调用方必须严格滞后连接。"""
    h = history.sort_values('date').reset_index(drop=True)
    close, volume = h.close.where(h.close > 0), h.volume.where(h.volume > 0)
    returns = close.pct_change(fill_method=None)
    vwap = h.amount / volume
    values = {}
    for lag in range(lags):
        for name in ['open', 'high', 'low', 'close']:
            values[f'ctx_daily_{name}_lag{lag}'] = h[name].shift(lag) / close - 1
        values[f'ctx_daily_vwap_lag{lag}'] = vwap.shift(lag) / close - 1
        values[f'ctx_daily_volume_lag{lag}'] = volume.shift(lag) / volume
    tr = pd.concat([h.high-h.low, (h.high-close.shift()).abs(), (h.low-close.shift()).abs()], axis=1).max(axis=1)
    for window in windows:
        roll = close.rolling(window)
        values[f'ctx_daily_momentum_{window}'] = close / close.shift(window) - 1
        values[f'ctx_daily_mean_gap_{window}'] = close / roll.mean() - 1
        values[f'ctx_daily_volatility_{window}'] = returns.rolling(window).std()
        values[f'ctx_daily_atr_{window}'] = tr.rolling(window).mean() / close
        values[f'ctx_daily_breakout_{window}'] = close / h.high.rolling(window).max() - 1
        values[f'ctx_daily_position_{window}'] = (close-h.low.rolling(window).min()) / (h.high.rolling(window).max()-h.low.rolling(window).min()).replace(0, np.nan)
        values[f'ctx_daily_volume_ratio_{window}'] = volume / volume.rolling(window).mean()
        values[f'ctx_daily_up_fraction_{window}'] = returns.gt(0).astype(float).where(returns.notna()).rolling(window).mean()
        values[f'ctx_daily_illiquidity_{window}'] = (returns.abs()/h.amount.where(h.amount > 0)*1e8).rolling(window).mean()
    result = pd.DataFrame(values).replace([np.inf, -np.inf], np.nan).astype('float32')
    result.insert(0, 'source_date', h.date)
    return result


def align_daily_features(signals: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """把daily严格早于signals自然日的数据连接到信号，保留原行顺序供审计和推断。"""
    keys = pd.DataFrame({'date': pd.to_datetime(signals.datetime).dt.normalize(), '_row': np.arange(len(signals))})
    joined = pd.merge_asof(keys.sort_values('date'), daily.sort_values('source_date'), left_on='date',
                           right_on='source_date', direction='backward', allow_exact_matches=False)
    if not (joined.source_date.isna() | (joined.source_date < joined.date)).all():
        raise AssertionError('日线特征读取了当前或未来交易日')
    return joined.sort_values('_row').drop(columns=['date', '_row', 'source_date']).reset_index(drop=True)


def build_symbol_features(symbol: str, source: Path, history_dir: Path, cfg: dict) -> pd.DataFrame:
    """复用source的symbol原因子并连接history_dir的过去日线；cfg指定滞后和滚动窗口。"""
    raw = pd.read_pickle(source / 'symbols' / f'{symbol}.pkl')['features'].sort_values('datetime').reset_index(drop=True)
    columns = [c for c in raw if c not in ['datetime', 'symbol'] and not c.startswith('ctx_')]
    original = raw[columns].rename(columns={c: f'ctx_raw_{c}' for c in columns}).astype('float32')
    daily = build_daily_features(read_history(history_dir, symbol), cfg['daily_lags'], cfg['daily_windows'])
    return pd.concat([raw[['datetime', 'symbol']], original, align_daily_features(raw, daily)], axis=1)


def load_expanded_panel(cfg: dict, max_symbols: int = 0) -> tuple[pd.DataFrame, dict]:
    """按cfg来源和代码指纹加载/构建扩展面板；max_symbols仅限制工程烟测，不按收益选股。"""
    source, history_dir = Path(cfg['source_cache']), Path(cfg['history_data_dir'])
    old = json.loads((source / 'complete.json').read_text())
    symbols = old['symbols'][:max_symbols] if max_symbols else old['symbols']
    identity = {'source_manifest': calculate_sha256(source / 'complete.json'),
                'feature_code': calculate_sha256(Path(__file__)), 'lags': cfg['daily_lags'],
                'windows': cfg['daily_windows'], 'symbols': symbols,
                'inputs': {str(p): calculate_sha256(p) for s in symbols for p in
                           [source / 'symbols' / f'{s}.pkl', history_dir / f'{s}.pkl']}}
    directory = Path(cfg['expanded_cache']) / calculate_identity(identity)[:20]
    metadata = directory / 'complete.json'
    if metadata.exists():
        manifest = json.loads(metadata.read_text())
        if calculate_sha256(directory / 'panel.pkl') != manifest['panel_sha256']:
            raise AssertionError('扩展缓存内容指纹不一致')
        print(f'复用扩展特征PKL：{directory}', flush=True)
        return pd.read_pickle(directory / 'panel.pkl'), manifest
    if calculate_sha256(source / 'panel.pkl') != old['panel_sha256']:
        raise AssertionError('来源面板指纹不一致')
    original = pd.read_pickle(source / 'panel.pkl')
    original = original.loc[original.symbol.isin(symbols)].reset_index(drop=True)
    pieces = []
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        for i, frame in enumerate(pool.map(lambda s: build_symbol_features(s, source, history_dir, cfg), symbols)):
            pieces.append(frame)
            if i % 50 == 0:
                print(f'扩展特征 {i+1}/{len(symbols)}只', flush=True)
    additions = pd.concat(pieces, ignore_index=True)
    panel = original.merge(additions, on=['datetime', 'symbol'], how='left', validate='one_to_one', sort=False)
    panel = panel.sort_values(['datetime', 'symbol']).reset_index(drop=True)
    numeric = panel.columns.difference(['datetime', 'symbol'])
    panel[numeric] = panel[numeric].replace([np.inf, -np.inf], np.nan).astype('float32')
    save_pickle(directory / 'panel.pkl', panel)
    manifest = {**old, 'identity': identity, 'cache_path': str(directory), 'source_cache': str(source),
                'symbols': symbols, 'panel_sha256': calculate_sha256(directory / 'panel.pkl'),
                'feature_count': len(numeric), 'rows': len(panel)}
    write_json(metadata, manifest)
    descriptions = []
    for c in numeric:
        kind = '上一完成日价量滞后/滚动统计' if c.startswith('ctx_daily_') else ('原始分钟因子值（不做截面排名）' if c.startswith('ctx_raw_') else '原有合格股票池截面排名/当时上下文')
        descriptions.append({'feature': c, '中文说明': kind, '构造实现': 'cta_expanded_features.py / factor_library.py'})
    pd.DataFrame(descriptions).to_csv(directory / 'feature_definitions.csv', index=False, encoding='utf-8-sig')
    print(f'扩展缓存完成：{len(panel):,}行/{len(numeric)}特征', flush=True)
    return panel, manifest
