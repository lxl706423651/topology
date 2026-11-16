"""独立审核逐验证集搜索：候选齐全、全部折门槛、过去模型、预测缓存和真实账本。

在quant根目录输入--report-dir现有阶段，输出新增带时间戳审计及源码快照。
只读已完成研究，不调参、不改旧模型、不调用2026测试或外部服务。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.cta_research_data import calculate_sha256, MinuteMarket
from quant_backtest.cta_expanded_training import truncate_market
from quant_backtest.cta_research_reporting import audit_result
from scripts.audit_xgb_cta_results import audit_market_status


def audit_stage(directory: Path) -> dict:
    """独立复核directory既有候选/来源/模型预测及逐折账本，失败抛异常，不更改研究选择。"""
    protocol = json.loads((directory/'protocol.json').read_text())
    cfg = yaml.safe_load((directory/'resolved_config.yaml').read_text())
    execution = yaml.safe_load((directory/'execution_config.yaml').read_text())
    table = pd.read_csv(directory/'validation_rule_metrics.csv', float_precision='round_trip')
    candidate = pd.read_csv(directory/'candidate_summary.csv', float_precision='round_trip')
    rules = {r['rule_id']: r for r in protocol['rules']}
    expected_years = [2023, 2024, 2025]
    assert set(table.rule_id)==set(rules) and len(table)==len(rules)*3
    assert not table.duplicated(['rule_id', 'year']).any()
    matrix = table.pivot(index='rule_id', columns='year', values='sharpe')
    assert list(matrix.columns)==expected_years and np.isfinite(matrix.to_numpy()).all()
    closed = table.pivot(index='rule_id', columns='year', values='closed_lot_count')
    volatility = table.pivot(index='rule_id', columns='year', values='annual_volatility')
    sufficient = (closed>=cfg['minimum_closed_lots_per_fold']).all(axis=1) & (volatility>0).all(axis=1)
    expected = pd.DataFrame({'minimum_sharpe': matrix.min(axis=1), 'mean_sharpe': matrix.mean(axis=1),
                            'sufficient': sufficient, 'all_folds_pass': sufficient & matrix.gt(cfg['sharpe_target']).all(axis=1)})
    actual = candidate.set_index('rule_id').reindex(expected.index)
    for key in expected:
        np.testing.assert_array_equal(expected[key].to_numpy(), actual[key].to_numpy())
    ordered = expected.reset_index().sort_values(['all_folds_pass', 'sufficient', 'minimum_sharpe', 'mean_sharpe', 'rule_id'],
                                                 ascending=[False, False, False, False, True])
    chosen = json.loads((directory/'selected_rule.json').read_text())
    assert chosen['rule_id']==ordered.iloc[0].rule_id and chosen==rules[chosen['rule_id']]
    before = json.loads((directory/'selected_before_replay.json').read_text())
    assert before['rule_sha256']==calculate_sha256(directory/'selected_rule.json')
    for path, digest in json.loads((directory/'source_hashes.json').read_text()).items():
        assert calculate_sha256(Path(path))==digest
    cache = json.loads((directory/'prediction_cache_manifest.json').read_text())
    cache_path = Path(cache['cache_path'])
    for name, digest in cache['files'].items():
        assert calculate_sha256(cache_path/name)==digest
    source = Path(cfg['source_report'])
    original_freeze = json.loads((source/'frozen_before_test.json').read_text())['files']
    for path, fingerprints in cache['identity']['models'].items():
        model_dir = Path(path)
        for filename, key in [('xgboost_model.json', 'model'), ('model_metadata.json', 'metadata')]:
            actual_hash = calculate_sha256(model_dir/filename)
            assert actual_hash==fingerprints[key]==original_freeze[str((model_dir/filename).relative_to(source))]
        meta = json.loads((model_dir/'model_metadata.json').read_text())
        ledger = pd.read_csv(model_dir/'training_rows.csv', parse_dates=['datetime', 'label_end'])
        cutoff = pd.Timestamp(meta['fold']['train_end'])
        assert (ledger.datetime<cutoff).all() and (ledger.label_end<cutoff).all() and cutoff<pd.Timestamp('2026-01-01')
        assert calculate_sha256(model_dir/'training_rows.csv')==meta['training_input_sha256']
    # 根据原始特征重新生成过去模型的验证预测，与已缓存分数逐值相等。
    manifest = cache['data_manifest']
    panel_path = Path(manifest['cache_path'])/'panel.pkl'
    assert calculate_sha256(panel_path)==manifest['panel_sha256']
    panel = pd.read_pickle(panel_path)
    panel = panel.loc[panel.datetime<pd.Timestamp('2026-01-01')]
    prediction_checks = []
    for fold in protocol['folds']:
        year = fold['year']
        start, end = pd.Timestamp(fold['validation_start']), pd.Timestamp(fold['validation_end'])
        values = panel.loc[(panel.datetime>=start)&(panel.datetime<end)]
        saved = pd.read_pickle(cache_path/f'validation_{year}.pkl')
        np.testing.assert_array_equal(values.datetime.to_numpy(), saved.datetime.to_numpy())
        np.testing.assert_array_equal(values.symbol.to_numpy(), saved.symbol.to_numpy())
        for column in cache['identity']['context_columns']:
            np.testing.assert_array_equal(values[column].to_numpy(), saved[column].to_numpy())
        for column in [c for c in saved if c.startswith('score_')]:
            name = column[6:]
            model_dir = source/'internal'/str(year)/name
            meta = json.loads((model_dir/'model_metadata.json').read_text())
            assert pd.Timestamp(meta['last_training_label'])<start and pd.Timestamp(meta['last_training_feature'])<start
            model = xgb.XGBRegressor(n_jobs=cfg['workers'])
            model.load_model(model_dir/'xgboost_model.json')
            prediction = model.predict(values[meta['features']]) / (1 if meta['objective']=='probability' else 100)
            np.testing.assert_array_equal(prediction, saved[column].to_numpy())
            prediction_checks.append({'year': year, 'model': name, 'rows': len(prediction), 'maximum_error': 0.0})
        print(f'验证预测独立重算：{year}通过', flush=True)
    market = truncate_market(MinuteMarket(Path(execution['source_cache'])/'minute_market'), pd.Timestamp('2026-01-01'))
    accounts = []
    for year in expected_years:
        folder = directory/f'validation_{year}'
        events = pd.read_csv(folder/'test_trade_events.csv', float_precision='round_trip', parse_dates=['datetime', 'signal_datetime'])
        trips = pd.read_csv(folder/'test_round_trips.csv', float_precision='round_trip', parse_dates=['buy_datetime','sell_datetime','buy_signal_datetime','sell_signal_datetime'])
        opened = pd.read_csv(folder/'test_open_positions.csv', float_precision='round_trip', parse_dates=['buy_datetime'])
        curve = pd.read_pickle(folder/'minute_equity_curve.pkl')
        audited = audit_result({'events': events, 'trips': trips, 'open': opened, 'curve': curve}, market, execution)
        daily = curve.groupby(curve.datetime.dt.normalize()).last()
        equity = daily.equity.to_numpy()
        returns = np.r_[equity[0]/execution['initial_cash']-1, equity[1:]/equity[:-1]-1]
        sd = returns.std(ddof=1)
        sharpe = float(returns.mean()/sd*np.sqrt(252)) if sd else 0.0
        row = table.loc[(table.rule_id==chosen['rule_id']) & (table.year==year)].iloc[0]
        assert abs(sharpe-row.sharpe)<1e-10
        assert int((events.side=='buy').sum())==row.buy_count
        accounts.append({'year': year, 'sharpe': sharpe, **audited})
    status = audit_market_status(market, manifest)
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['all_validation_folds_pass']==bool(not cfg['is_smoke'] and ordered.iloc[0].all_folds_pass)
    assert summary['test_2026_evaluated'] is False
    return {'passed': True, 'candidate_count': len(rules), 'candidate_account_count': len(table), 'selected_accounts': accounts,
        'prediction_checks': prediction_checks, 'market_status': status, 'all_validation_folds_pass': summary['all_validation_folds_pass'],
        'minimum_sharpe': ordered.iloc[0].minimum_sharpe, 'test_2026_evaluated': False,
        'audit_source_sha256': calculate_sha256(Path(__file__))}


def run_audit() -> None:
    """解析报告目录后写入唯一审计JSON/源码，不覆盖旧审计。"""
    parser = argparse.ArgumentParser(description='CTA逐验证集搜索的独立审计')
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit_stage(args.report_dir)
    output = args.report_dir/f'independent_audit_{datetime.now():%Y%m%d_%H%M%S}.json'
    write_json(output, result)
    shutil.copy2(Path(__file__), output.with_suffix('.py'))
    print(f'独立审计通过：{output}', flush=True)


if __name__=='__main__':
    run_audit()
