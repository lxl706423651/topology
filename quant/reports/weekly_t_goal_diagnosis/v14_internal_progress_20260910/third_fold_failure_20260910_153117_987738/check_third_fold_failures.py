"""第三折内部恢复失败只读诊断；不运行账户、模型或读取外层。"""
from pathlib import Path
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'reports/inventory_learning_t/run_20260910_151232_841321'
OUT = Path(__file__).resolve().parent / f'third_fold_failure_{datetime.now():%Y%m%d_%H%M%S_%f}'
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def bind(path):
    p = Path(path).resolve()
    INPUTS.setdefault(str(p), {'sha256': sha(p), 'bytes': p.stat().st_size})
    return p


def read(path): return pd.read_csv(bind(path), float_precision='round_trip')
def js(path): return json.loads(bind(path).read_text())
def save(name, frame): frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')
def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    directory = RUN / 'year_2025'
    cfg = js(RUN / 'resolved_config.json')
    selected = js(directory / 'selected_rule.json')
    table = read(directory / 'validation_rule_metrics.csv')
    active = table.loc[table.enabled].copy()
    failures = ['inventory_mismatch_days', 'recovery_failure_days', 'pending_pairs',
                't1_violation_count', 'base_plan_mismatch_count']
    assert len(active) == 36 and selected['rule']['candidate_index'] == 0
    assert not (active.completed_pair_count.gt(0) & active[failures].eq(0).all(axis=1)).any()
    minimum = active.loc[active.recovery_failure_days.eq(active.recovery_failure_days.min())]
    minimum = minimum.loc[minimum.inventory_mismatch_days.eq(minimum.inventory_mismatch_days.min())]
    write('protocol.json', {'outer_fold_label': 2025, 'actual_inner_year': 2026,
        'all_active_metrics_and_daily_audits': 36,
        'detailed_selection': 'minimum recovery_failure_days, then minimum inventory_mismatch_days; retain all ties',
        'detailed_candidates': minimum.candidate_index.tolist(),
        'no_outer_read': True, 'no_account_or_model_rerun': True, 'no_rule_changes': True})
    save('all_active_metrics.csv', active)
    failed_rows = []
    for r in active.itertuples():
        a = read(Path(r.candidate_directory) / 'daily_inventory_audit.csv')
        assert a.loc[a.inventory_difference.ne(0), 'date'].nunique() == r.inventory_mismatch_days
        assert a.loc[a.recovery_failure, 'date'].nunique() == r.recovery_failure_days
        failed_rows.append(a.loc[a.inventory_difference.ne(0) | a.recovery_failure].assign(
            candidate_index=r.candidate_index, exit_mode=r.exit_mode))
    failed = pd.concat(failed_rows, ignore_index=True)
    save('all_failure_stock_days.csv', failed)
    prevalence = failed.loc[failed.recovery_failure].groupby(['date', 'symbol']).agg(
        candidates=('candidate_index', 'nunique'),
        exit_modes=('exit_mode', lambda x: ','.join(sorted(set(x))))).reset_index()
    save('recovery_failure_prevalence.csv', prevalence.sort_values('candidates', ascending=False))

    symbol, day = 'SH.603949', pd.Timestamp('2026-03-23')
    raw = read(Path(cfg['data_root']) / '2026' / f'{symbol}.csv')
    raw.datetime = pd.to_datetime(raw.datetime)
    raw = raw.set_index('datetime').sort_index()
    previous_close = float(raw.loc[raw.index < day].iloc[-1].close)
    today = raw.loc[raw.index.normalize() == day]
    clocks = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), day + pd.Timedelta(hours=11, minutes=30), freq='min').append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=1), day + pd.Timedelta(hours=15), freq='min'))
    assert today.index.equals(clocks)
    coverage = js(directory / 'validation_context/full_minute_coverage_audit.json')
    cache = Path(coverage['dense_cache_path'])
    metadata = js(cache / 'metadata.json')
    arrays = {n: np.load(bind(cache / f'{n}.npy'), mmap_mode='r', allow_pickle=False)
              for n in ['times', 'quotes', 'present']}
    i = metadata['symbols'].index(symbol)
    j = np.searchsorted(arrays['times'], today.index.asi8)
    assert np.array_equal(arrays['times'][j], today.index.asi8)
    assert arrays['present'][j, i].all()
    assert np.array_equal(arrays['quotes'][j, i], today[metadata['identity']['quote_columns']].to_numpy())
    save('case_raw_quotes.csv', today.reset_index().assign(previous_close=previous_close, symbol=symbol))

    cases, reasons, all_events, quote_checks = [], [], [], []
    for r in minimum.itertuples():
        folder = Path(r.candidate_directory)
        e = read(folder / 'test_trade_events.csv')
        e.datetime = pd.to_datetime(e.datetime, format='mixed')
        e.signal_datetime = pd.to_datetime(e.signal_datetime, format='mixed')
        b = read(folder / 'blocked_orders.csv')
        b.datetime = pd.to_datetime(b.datetime, format='mixed')
        x = read(folder / 'exit_decisions.csv')
        for k in ['first_signal_time', 'first_execution_time', 'original_deadline', 'exit_signal_time', 'exit_earliest_time']:
            x[k] = pd.to_datetime(x[k], format='mixed')
        for keys, f in b.groupby(['event_type', 'reason']):
            reasons.append({'candidate_index': r.candidate_index, 'event_type': keys[0], 'reason': keys[1], 'rows': len(f)})
        a = failed.loc[failed.candidate_index.eq(r.candidate_index) & failed.recovery_failure]
        assert len(a) == 1 and a.iloc[0].symbol == symbol and pd.Timestamp(a.iloc[0].date) == day
        first = e.loc[e.event_type.eq('t') & e.fragment_id.eq(0) & e.symbol.eq(symbol) & e.datetime.dt.normalize().eq(day)]
        assert len(first) == 1
        first = first.iloc[0]
        pair = e.loc[e.event_type.eq('t') & e.pair_id.eq(first.pair_id)]
        rest = pair.loc[pair.fragment_id.gt(0)]
        ex = x.loc[x.pair_id.eq(first.pair_id)].iloc[0]
        assert first.side == 'buy' and rest.side.eq('sell').all()
        assert rest.shares.sum() == first.shares and rest.datetime.min().normalize() > day
        blocks = b.loc[b.event_type.eq('t_recovery') & b.symbol.eq(symbol) &
            b.datetime.between(ex.exit_earliest_time, day + pd.Timedelta(hours=15))]
        expected = clocks[clocks >= ex.exit_earliest_time]
        assert pd.DatetimeIndex(blocks.datetime).equals(expected)
        for q in blocks.itertuples():
            quote = today.loc[q.datetime]
            zero = quote.amount <= 0 or quote.volume <= 0 or quote.close <= 0
            ten = 1 - quote.close / previous_close >= cfg['nonflat_limit_proxy']
            limit5 = np.floor(previous_close * .95 * 100 + .5 + 1e-8) / 100
            five = quote.high == quote.low and abs(quote.close - limit5) <= 1e-6
            reason = 'zero_price_or_volume' if zero else 'conservative_price_limit_proxy' if ten or five else 'OTHER'
            assert reason == q.reason
            quote_checks.append({'candidate_index': r.candidate_index, 'datetime': q.datetime,
                'stored_reason': q.reason, 'independent_reason': reason, 'raw_close': quote.close,
                'raw_volume': quote.volume, 'raw_amount': quote.amount, 'ten_percent_proxy': bool(ten), 'five_percent_proxy': bool(five)})
        trips = read(folder / 'test_round_trips.csv')
        trips = trips.loc[trips.pair_id.eq(first.pair_id)]
        cases.append({'candidate_index': r.candidate_index, 'exit_mode': r.exit_mode, 'fraction': r.fraction,
            'symbol': symbol, 'failure_date': str(day.date()), 'pair_id': int(first.pair_id), 'direction': 'buy_first',
            'first_signal_time': first.signal_datetime, 'signal_close': float(today.loc[first.signal_datetime].close),
            'signal_return_from_previous_close': float(today.loc[first.signal_datetime].close / previous_close - 1),
            'first_execution_time': first.datetime, 'first_reference_price': first.reference_price,
            'first_shares': int(first.shares), 'previous_close': previous_close,
            'original_deadline': ex.original_deadline, 'exit_signal_time': ex.exit_signal_time,
            'exit_earliest_time': ex.exit_earliest_time, 'exit_reason': ex.exit_reason,
            'estimated_net_return_at_latch': ex.estimated_net_return,
            'zero_volume_attempts': int(blocks.reason.eq('zero_price_or_volume').sum()),
            'limit_proxy_attempts': int(blocks.reason.eq('conservative_price_limit_proxy').sum()),
            'expected_retry_minutes': len(expected), 'missing_attempts': 0,
            'final_recovery_time': rest.datetime.max(), 'reference_recovery_price': rest.iloc[-1].reference_price,
            'pair_net_pnl': float(trips.net_pnl.sum()), 'inventory_difference_at_eod': int(a.iloc[0].inventory_difference)})
        all_events.append(pair.assign(candidate_index=r.candidate_index))
    save('minimum_failure_cases.csv', pd.DataFrame(cases))
    save('minimum_sample_order_reasons.csv', pd.DataFrame(reasons))
    save('case_pair_events.csv', pd.concat(all_events, ignore_index=True))
    save('case_retry_quote_checks.csv', pd.DataFrame(quote_checks))
    assert all(sha(p) == v['sha256'] for p, v in INPUTS.items())
    write('inputs_sha256.json', INPUTS)
    summary = {'passed': True, 'scope': 'third fold inner2026, not outer', 'active_candidates': 36,
        'execution_valid_active_candidates': 0, 'positive_net_active_candidates': int(active.incremental_net_pnl.gt(0).sum()),
        'inventory_mismatch_days_range': [int(active.inventory_mismatch_days.min()), int(active.inventory_mismatch_days.max())],
        'recovery_failure_days_range': [int(active.recovery_failure_days.min()), int(active.recovery_failure_days.max())],
        'pending_t1_baseplan_all_zero': bool(active[['pending_pairs', 't1_violation_count', 'base_plan_mismatch_count']].eq(0).all().all()),
        'unique_recovery_stockdays_all_active': len(prevalence), 'minimum_failure_candidates': minimum.candidate_index.tolist(),
        'case_affected_candidates_all36': int(prevalence.loc[prevalence.symbol.eq(symbol) & prevalence.date.eq(str(day.date())), 'candidates'].iloc[0]),
        'case_raw_rows': len(today), 'case_raw_matches_dense': True, 'case_retry_rows_checked': len(quote_checks),
        'no_missing_case_quote_or_retry': True, 'case_not_cash_or_t1_or_capacity_failure': True,
        'case_signal_was_below_proxy_limit': False,
        'case_signal_return': float(today.loc[day + pd.Timedelta(hours=9, minutes=45)].close / previous_close - 1),
        'case_execution_return': float(today.loc[day + pd.Timedelta(hours=9, minutes=46)].close / previous_close - 1),
        'input_files_unchanged': len(INPUTS), 'no_outer_read': True, 'no_runtime_changes': True,
        'no_account_or_model_rerun': True, 'script_sha256': sha(__file__)}
    write('verification.json', summary)
    print(OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
