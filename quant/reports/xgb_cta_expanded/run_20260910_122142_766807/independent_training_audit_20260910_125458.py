"""独立复核扩展CTA模型训练时间/目标/输入指纹和冻结文件，不调参、不重新选择策略。

输入为--report-dir现有报告；读取特征PKL、训练记录和原生模型，输出新增审计JSON。
运行会读取较大缓存，但不会重训、改模型、连接网络或访问实盘账户。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import shutil

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.cta_research_data import calculate_identity, calculate_sha256
from quant_backtest.cta_research_data import MinuteMarket, QUOTE_FIELDS
from quant_backtest.cta_research_reporting import audit_result
from scripts.audit_xgb_cta_results import audit_market_status


def audit_models(directory: Path) -> dict:
    """独立重建directory各模型实际输入指纹/目标，验证无越过拟合截止或进入2026的行。"""
    manifest = json.loads((directory / 'data_manifest.json').read_text())
    for name, digest in manifest['identity']['inputs'].items():
        assert calculate_sha256(Path(name)) == digest, f'原因子或历史日线变更：{name}'
    panel_path = Path(manifest['cache_path']) / 'panel.pkl'
    assert calculate_sha256(panel_path) == manifest['panel_sha256'], '特征PKL指纹不一致'
    panel = pd.read_pickle(panel_path)
    frozen = json.loads((directory / 'frozen_before_test.json').read_text())['files']
    for name, digest in frozen.items():
        assert calculate_sha256(directory / name) == digest, f'冻结文件变更：{name}'
    label_identity = json.loads((directory / 'label_manifest.json').read_text())
    labels = pd.read_pickle(Path(manifest['cache_path']) / f'path_labels_{calculate_identity(label_identity)}.pkl')
    checked = []
    for path in sorted(directory.rglob('model_metadata.json')):
        if 'source_snapshot' in path.parts:
            continue
        meta = json.loads(path.read_text())
        rows_path = path.parent / 'training_rows.csv'
        assert calculate_sha256(rows_path) == meta['training_input_sha256']
        rows = pd.read_csv(rows_path, float_precision='round_trip', index_col='row_id', parse_dates=['datetime', 'label_end'])
        start, end = pd.Timestamp(meta['fold']['train_start']), pd.Timestamp(meta['fold']['train_end'])
        assert end <= pd.Timestamp('2026-01-01')
        assert ((rows.datetime >= start) & (rows.datetime < end) & (rows.label_end < end)).all()
        feature_rows = panel.loc[rows.index]
        pd.testing.assert_series_equal(feature_rows.datetime, rows.datetime, check_names=False, check_index_type=False)
        assert feature_rows.symbol.tolist() == rows.symbol.tolist()
        features = meta['features']
        assert all(c.startswith(('cs_rank_', 'ctx_')) for c in features)
        identity = calculate_identity(pd.util.hash_pandas_object(feature_rows[features], index=True).tolist())
        assert identity == meta['feature_values_sha256'], '实际训练特征值不一致'
        target_key, end_key = f"net_return_{meta['exit_profile']}", f"label_end_{meta['exit_profile']}"
        expected = labels.join(panel[['datetime']])
        expected = expected.loc[(expected.datetime >= start) & (expected.datetime < end) &
                                (expected[end_key] < end) & expected[target_key].notna()]
        np.testing.assert_array_equal(expected.index, rows.index)
        np.testing.assert_array_equal(expected[target_key], rows.target_net_return)
        np.testing.assert_array_equal(expected[end_key].to_numpy(), rows.label_end.to_numpy())
        target = (rows.target_net_return > 0).astype(float) if meta['objective'] == 'probability' else rows.target_net_return.clip(-0.5, 0.5)*100
        np.testing.assert_array_equal(target, rows.actual_target)
        weights = 1 / feature_rows.groupby(feature_rows.datetime.dt.normalize()).symbol.transform('size')
        weights /= weights.mean()
        np.testing.assert_array_equal(weights, rows.sample_weight)
        assert calculate_sha256(path.parent / 'xgboost_model.json') == meta['model_sha256']
        model = xgb.XGBRegressor()
        model.load_model(path.parent / 'xgboost_model.json')
        prediction = model.predict(feature_rows[features].iloc[:100])
        assert np.isfinite(prediction).all()
        checked.append({'model': str(path.parent.relative_to(directory)), 'fit_rows': len(rows),
                        'last_feature': rows.datetime.max(), 'last_label': rows.label_end.max(), 'feature_count': len(features)})
    market = MinuteMarket(Path(manifest['source_cache']) / 'minute_market')
    config = yaml.safe_load((directory / 'resolved_config.yaml').read_text())
    summaries = list(directory.rglob('execution_audit.json'))
    accounts = []
    for path in summaries:
        audit = json.loads(path.read_text())
        assert audit.get('passed'), f'账户审计未通过：{path}'
        folder = path.parent
        events = pd.read_csv(folder / 'test_trade_events.csv', float_precision='round_trip', parse_dates=['datetime', 'signal_datetime'])
        trips = pd.read_csv(folder / 'test_round_trips.csv', float_precision='round_trip',
                            parse_dates=['buy_datetime', 'sell_datetime', 'buy_signal_datetime', 'sell_signal_datetime'])
        opened = pd.read_csv(folder / 'test_open_positions.csv', float_precision='round_trip', parse_dates=['buy_datetime'])
        curve = pd.read_pickle(folder / 'minute_equity_curve.pkl')
        local_cfg = {**config, 'slippage_rate': config['slippage_rate'] * (2 if folder.name == 'double_slippage' else 1)}
        account = audit_result({'events': events, 'trips': trips, 'open': opened, 'curve': curve}, market, local_cfg)
        signs = np.where(events.side.eq('buy'), 1, -1)
        unrounded = events.reference_price.to_numpy(dtype=float) * (1 + signs * local_cfg['slippage_rate'])
        expected_prices = np.where(signs == 1, np.ceil(unrounded*100-1e-8), np.floor(unrounded*100+1e-8))/100
        np.testing.assert_allclose(expected_prices, events.price.to_numpy(dtype=float), atol=1e-9, rtol=0)
        daily = pd.read_csv(path.parent / 'test_equity_curve.csv', float_precision='round_trip')
        metrics = pd.read_csv(path.parent / 'test_metrics.csv', float_precision='round_trip').iloc[0]
        reconstructed = curve.groupby(curve.datetime.dt.normalize()).last()
        np.testing.assert_allclose(reconstructed.equity, daily.equity, atol=1e-7, rtol=0)
        equity = reconstructed.equity.to_numpy()
        returns = np.r_[equity[0] / config['initial_cash'] - 1, equity[1:] / equity[:-1] - 1]
        np.testing.assert_allclose(returns, daily.daily_return, atol=1e-12, rtol=0)
        volatility = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / volatility * np.sqrt(252)) if volatility > 0 else 0.0
        assert abs(sharpe - metrics.sharpe) < 1e-8, '报告夏普与日收益不一致'
        assert len(daily) == len(pd.to_datetime(daily.datetime).dt.normalize().unique()), '日收益日期重复'
        accounts.append({'account': str(folder.relative_to(directory)), 'sharpe': sharpe, **account})
    source_manifest = json.loads((Path(manifest['source_cache']) / 'complete.json').read_text())
    for name, size, modified in source_manifest['identity']['files']:
        stat = (Path(config['data_root']) / name).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (size, modified), f'原始分钟文件发生变化：{name}'
    assert calculate_sha256(ROOT / 'quant_backtest/factor_library.py') == source_manifest['identity']['factor_code']
    for i, symbol in enumerate(manifest['symbols']):
        shard = pd.read_pickle(Path(manifest['source_cache']) / 'symbols' / f'{symbol}.pkl')['quotes']
        locations = market.times.get_indexer(shard.index)
        valid = locations >= 0
        expected = np.full((len(market.times), len(QUOTE_FIELDS)), np.nan)
        expected[locations[valid]] = shard.loc[valid, QUOTE_FIELDS].to_numpy()
        assert np.array_equal(expected, market.quotes[:, market.symbol_index[symbol], :], equal_nan=True), f'执行报价与原分股缓存不符：{symbol}'
        expected_flags = np.zeros((len(market.times), 3), dtype='uint8')
        executable = shard.volume.gt(0) & shard.amount.gt(0) & shard.status_known.eq(1)
        expected_flags[locations[valid], 0] = (executable & shard.allow_buy.eq(1)).to_numpy(dtype='uint8')[valid]
        expected_flags[locations[valid], 1] = executable.to_numpy(dtype='uint8')[valid]
        expected_flags[locations[valid], 2] = shard.is_st.to_numpy(dtype='uint8')[valid]
        assert np.array_equal(expected_flags, market.flags[:, market.symbol_index[symbol], :]), f'执行资格与原分股缓存不符：{symbol}'
        if i % 100 == 0:
            print(f'执行报价逐值复核 {i+1}/{len(manifest["symbols"])}', flush=True)
    status = audit_market_status(market, source_manifest)
    return {'passed': True, 'models': checked, 'account_audit_files': len(summaries), 'frozen_files': len(frozen),
            'market_status_audit': status,
            'execution_quote_symbols_checked': len(manifest['symbols']),
            'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'xgboost': xgb.__version__},
            'accounts': accounts,
            'boundary': '独立重算训练输入/目标/样本权重及时间，并从落盘成交及分钟权益复核现金/费用/T+1/滑点/日夏普；不等于第三方行情完整',
            'audit_source_sha256': calculate_sha256(Path(__file__))}


def run_audit() -> None:
    """解析现有报告路径，失败抛错不掩盖，成功写入带时间戳的独立审计结果。"""
    parser = argparse.ArgumentParser(description='扩展CTA独立训练审计')
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit_models(args.report_dir)
    output = args.report_dir / f"independent_training_audit_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(output, result)
    shutil.copy2(Path(__file__), output.with_suffix('.py'))
    print(f"通过：{len(result['models'])}模型，{result['account_audit_files']}账户；{output}")


if __name__ == '__main__':
    run_audit()
