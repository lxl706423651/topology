"""只读缓存的v13小样本性能/账户检查；不选规则，不训练，不保存盈利比较。"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path.cwd()))
from quant_backtest.weekly_t_quote_store import DenseRecoveryBarStore
from quant_backtest.dynamic_t_engine import run_dynamic_portfolio
from quant_backtest.dynamic_t_policy import build_dynamic_candidates, generate_dynamic_decisions, create_dynamic_rule_config
from quant_backtest.balanced_t_data import INTRADAY_COLUMNS

archive = Path(Path('/tmp/dynamic_t_engine_archive_path.txt').read_text())
source = Path('quant_backtest/dynamic_t_engine.py')
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
metadata_path = Path('reports/dynamic_inventory_t/data_validation/real_12_symbols_15_days/verification.json')
meta = json.loads(metadata_path.read_text())
store = DenseRecoveryBarStore(meta['cache_path'])
cachemeta = json.loads((Path(meta['cache_path'])/'metadata.json').read_text())
dates = pd.DatetimeIndex(np.load(Path(meta['cache_path'])/'times.npy', mmap_mode='r').view('datetime64[ns]')).normalize().unique()
symbols = cachemeta['symbols']
cache = Path('reports/weekly_t_cross_validation/cache/weekly_t_9b6b9bfb3309dd53')
parts = [pd.read_pickle(cache/f'{symbol}.pkl') for symbol in symbols]
daily = pd.concat([part['daily'].loc[part['daily'].date.isin(dates)] for part in parts], ignore_index=True)
intraday = pd.concat([part['intraday'].loc[part['intraday'].datetime.dt.normalize().isin(dates), INTRADAY_COLUMNS]
                      for part in parts], ignore_index=True)
selection_path = Path('reports/balanced_inventory_t_smoke/run_20260910_140418_969900/year_2026/test_context/frozen_base_selections.csv')
selected = pd.read_csv(selection_path, float_precision='round_trip')
cfg_path = Path('config/dynamic_inventory_t_v13.yaml')
cfg = yaml.safe_load(cfg_path.read_text())
cfg['manual_buys_csv'] = None
rules = build_dynamic_candidates(cfg)
results, baseline_curve = [], None
for rule in rules[:4]:
    decisions, _ = generate_dynamic_decisions(intraday, rule, cfg)
    start = time.perf_counter()
    account = run_dynamic_portfolio(selected, daily, intraday, decisions, create_dynamic_rule_config(cfg, rule), recovery_bars=store)
    elapsed = time.perf_counter() - start
    events, trips, inv, curve = [account[k] for k in ('trade_events', 'round_trips', 'daily_inventory_audit', 'equity_curve')]
    assert events.cash_after.ge(-1e-7).all()
    assert inv.day_sold.le(inv.opening_sellable).all()
    assert events.shares.mod(100).eq(0).all()
    assert len(account['exit_decisions']) == account['metrics']['started_pair_count']
    assert not account['exit_decisions'].pair_id.duplicated().any()
    if baseline_curve is None:
        baseline_curve = curve.baseline_equity.copy()
    else:
        pd.testing.assert_series_equal(baseline_curve, curve.baseline_equity, check_exact=True)
    t = events.loc[events.event_type.eq('t')]
    cashflow = float((t.notional * t.side.map({'sell':1., 'buy':-1.}) - t.fee).sum())
    pnl_error = abs(cashflow - trips.net_pnl.sum())
    if account['metrics']['pending_pairs'] == 0 and account['metrics']['base_plan_mismatch_count'] == 0:
        assert pnl_error < 1e-7
        assert abs(curve.t_added_equity.iloc[-1] - trips.net_pnl.sum()) < 1e-7
    results.append({'candidate_index': rule['candidate_index'], 'mode': account['metrics']['dynamic_exit_mode'],
                    'seconds': elapsed, 'requests': len(decisions), 'started_pairs': account['metrics']['started_pair_count'],
                    'completed_pairs': account['metrics']['completed_pair_count'], 'fragments': len(trips),
                    'exit_intents': len(account['exit_decisions']), 'pending_pairs': account['metrics']['pending_pairs'],
                    't1_violations': account['metrics']['t1_violation_count'], 'pair_accounting_error': pnl_error,
                    'common_baseline_equity_exact': True})
    print(results[-1], flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
out = {'passed': True, 'classification': 'real_quote_execution_and_performance_only_not_parameter_selection',
       'engine_sha256': source_sha, 'symbols': len(symbols), 'days': len(dates), 'quote_rows': meta['present_rows'],
       'cache_path': meta['cache_path'], 'runs': results,
       'input_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (metadata_path, cfg_path, selection_path)}}
(archive/'real_profile.json').write_text(json.dumps(out, indent=2)+'\n')
