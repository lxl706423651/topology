"""只读v13内部失败：全体轻量库存表、18个低活动账表、三个共性原行情日。

不导入生产引擎、不运行账户、不读取外层、不按净收益选样或改参数。
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'reports/dynamic_inventory_t/run_20260910_143829_552610'
INPUTS = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve()
    if str(path) not in INPUTS:
        INPUTS[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read_csv(path, **kwargs):
    return pd.read_csv(bind(path), **kwargs)


def read_json(path):
    return json.loads(bind(path).read_text())


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False)+'\n')


def save(name, frame):
    frame.to_csv(OUT/name, index=False, encoding='utf-8-sig')


def clock(day):
    return pd.date_range(day+' 09:31', day+' 11:30', freq='min').append(
        pd.date_range(day+' 13:01', day+' 15:00', freq='min'))


def main():
    write_json('protocol.json', {'scope': 'v13 only year_2024/year_2025 inner2026',
        'account_rerun': False, 'outer_read': False, 'rule_or_stock_filter_changed': False,
        'sample_selection': 'minimum started_pair_count then candidate_index within fold/original_horizon/exit_mode',
        'scope_extension': 'all324 lightweight daily audit files for failure prevalence; only18 detailed accounts',
        'raw_cases': [[2024,'SH.600735','2026-01-28'],[2024,'SH.603398','2026-03-20'],[2025,'SZ.002945','2026-01-06']],
        'raw_case_selection': 'common failure days across all nine least-active horizon/exit cells, not PnL',
        'code_review': 'read saved production source; independent quote/retry checks, no engine import',
        'started_at': datetime.now().isoformat()})
    cfg = read_json(RUN/'resolved_config.json')
    metrics = read_csv(RUN/'validation_rule_metrics.csv', float_precision='round_trip')
    active = metrics.loc[metrics.test_year.isin([2024,2025]) & metrics.enabled].copy()
    failcols = ['inventory_mismatch_days','recovery_failure_days','pending_pairs','t1_violation_count','base_plan_mismatch_count']
    derived = active.started_pair_count.gt(0) & active.t_pair_count.gt(0) & active[failcols].eq(0).all(axis=1)
    assert derived.eq(active.execution_eligible).all() and not derived.any()
    columns = ['test_year','candidate_index','family','entry_deviation','recovery_horizon','fraction','exit_mode',
               'started_pair_count','t_pair_count',*failcols,'execution_eligible','request_count','candidate_directory']
    save('all_active_failure_metrics.csv', active[columns])
    selected = active.sort_values(['started_pair_count','candidate_index']).groupby(
        ['test_year','recovery_horizon','exit_mode'], sort=True).head(1).sort_values(['test_year','candidate_index'])
    save('low_activity_sample.csv', selected[columns])
    all_failures = []
    for row in active.itertuples():
        audit = read_csv(Path(row.candidate_directory)/'daily_inventory_audit.csv', usecols=[
            'date','symbol','recovery_failure','inventory_difference','t1_violation_count'])
        assert audit.loc[audit.inventory_difference.ne(0),'date'].nunique() == row.inventory_mismatch_days
        assert audit.loc[audit.recovery_failure,'date'].nunique() == row.recovery_failure_days
        part = audit.loc[audit.recovery_failure | audit.inventory_difference.ne(0)].copy()
        part['test_year'],part['candidate_index'],part['exit_mode'] = row.test_year,row.candidate_index,row.exit_mode
        all_failures.append(part)
    failures = pd.concat(all_failures,ignore_index=True)
    save('all_inventory_failure_stockdays.csv',failures)
    prevalence = failures.loc[failures.recovery_failure].groupby(['test_year','date','symbol']).agg(
        affected_candidates=('candidate_index','nunique'),exit_modes=('exit_mode',lambda s: ','.join(sorted(set(s))))).reset_index()
    save('failure_stockday_prevalence.csv',prevalence.sort_values(['test_year','affected_candidates'],ascending=[True,False]))
    accounts, reason_counts, detail_failures = {}, [], []
    for row in selected.itertuples():
        folder = Path(row.candidate_directory)
        frames = {}
        for name in ['blocked_orders','test_trade_events','exit_decisions','daily_inventory_audit']:
            frames[name] = read_csv(folder/f'{name}.csv',float_precision='round_trip')
            for col in frames[name]:
                if col in ['datetime','date','signal_datetime','first_execution_time','original_deadline','exit_signal_time','exit_earliest_time']:
                    frames[name][col] = pd.to_datetime(frames[name][col],format='mixed')
        accounts[(row.test_year,row.candidate_index)] = frames
        blocks = frames['blocked_orders']
        for keys,group in blocks.groupby(['event_type','reason']):
            reason_counts.append({'test_year':row.test_year,'candidate_index':row.candidate_index,
                'event_type':keys[0],'reason':keys[1],'attempt_rows':len(group),'days':group.datetime.dt.normalize().nunique()})
        audit = frames['daily_inventory_audit']
        detail_failures.append(audit.loc[audit.recovery_failure].assign(test_year=row.test_year,candidate_index=row.candidate_index))
    save('sample_order_reason_counts.csv',pd.DataFrame(reason_counts))
    save('sample_recovery_failure_stockdays.csv',pd.concat(detail_failures,ignore_index=True))
    stores = {}
    for year in [2024,2025]:
        coverage = read_json(RUN/f'year_{year}/validation_context/full_minute_coverage_audit.json')
        directory = Path(coverage['dense_cache_path'])
        metadata = read_json(directory/'metadata.json')
        arrays = {name:np.load(bind(directory/(name+'.npy')),mmap_mode='r',allow_pickle=False)
                  for name in ['times','quotes','present']}
        stores[year] = (metadata,arrays)
    raw_cache, case_summaries, retry_checks, case_events, case_audits, raw_extracts = {}, [], [], [], [], []
    for year,symbol,day in [(2024,'SH.600735','2026-01-28'),(2024,'SH.603398','2026-03-20'),(2025,'SZ.002945','2026-01-06')]:
        raw = read_csv(ROOT/cfg['data_root']/'2026'/f'{symbol}.csv')
        raw.datetime = pd.to_datetime(raw.datetime)
        raw = raw.set_index('datetime').sort_index()
        today = raw.loc[raw.index.normalize().eq(pd.Timestamp(day))] if hasattr(raw.index.normalize(),'eq') else raw.loc[raw.index.normalize()==pd.Timestamp(day)]
        previous = raw.loc[raw.index<pd.Timestamp(day)].iloc[-1]
        previous_time = raw.loc[raw.index<pd.Timestamp(day)].index[-1]
        previous_close = float(previous.close)
        assert len(today)==240 and today.index.equals(clock(day))
        metadata, arrays = stores[year]
        symbol_index = metadata['symbols'].index(symbol)
        indices = np.searchsorted(arrays['times'],today.index.asi8)
        assert np.array_equal(arrays['times'][indices],today.index.asi8)
        assert arrays['present'][indices,symbol_index].all()
        assert np.array_equal(arrays['quotes'][indices,symbol_index],today[metadata['identity']['quote_columns']].to_numpy())
        raw_extracts.append(today.reset_index().assign(symbol=symbol,test_year=year,previous_close=previous_close))
        for (fold,candidate),frames in accounts.items():
            if fold!=year: continue
            end = pd.Timestamp(day+' 15:00')
            events = frames['test_trade_events'].loc[lambda f:f.symbol.eq(symbol)&f.event_type.eq('t')]
            starts = events.loc[events.fragment_id.eq(0)&events.datetime.le(end)]
            ending = []
            for first in starts.itertuples():
                recovery = events.loc[events.pair_id.eq(first.pair_id)&events.fragment_id.gt(0)]
                if recovery.loc[recovery.datetime.le(end),'shares'].sum()<first.shares: ending.append((first,recovery))
            assert len(ending)==1,(year,candidate,symbol,day,len(ending))
            first,recovery = ending[0]
            exit_row = frames['exit_decisions'].loc[lambda f:f.pair_id.eq(first.pair_id)].iloc[0]
            assert first.datetime.normalize()==pd.Timestamp(day)
            side = 'buy' if first.side=='sell' else 'sell'
            blocks = frames['blocked_orders'].loc[lambda f:f.symbol.eq(symbol)&f.event_type.eq('t_recovery')&
                f.side.eq(side)&f.datetime.ge(exit_row.exit_earliest_time)&f.datetime.le(end)].copy()
            expected = clock(day);expected=expected[expected>=exit_row.exit_earliest_time]
            actual_recovery = recovery.loc[recovery.datetime.ge(exit_row.exit_earliest_time)&recovery.datetime.le(end)]
            attempted = pd.DatetimeIndex(blocks.datetime).union(pd.DatetimeIndex(actual_recovery.datetime))
            assert expected.difference(attempted).empty and attempted.difference(expected).empty
            assert actual_recovery.empty
            window = today.loc[expected]
            direction = 1 if side=='buy' else -1
            st_price = np.floor(previous_close*(1+direction*.05)*100+.5+1e-8)/100
            recomputed = []
            for item in blocks.itertuples():
                q=raw.loc[item.datetime]
                zero=q.close<=0 or q.amount<=0 or q.volume<=0
                ten=(q.close/previous_close-1)*direction>=cfg['nonflat_limit_proxy']
                five=q.high==q.low and abs(q.close-st_price)<=1e-6
                expected_reason='zero_price_or_volume' if zero else 'conservative_price_limit_proxy' if ten or five else 'OTHER'
                assert item.reason==expected_reason,(year,candidate,item.datetime,item.reason,expected_reason)
                recomputed.append({'test_year':year,'candidate_index':candidate,'symbol':symbol,'date':day,
                    'pair_id':first.pair_id,'datetime':item.datetime,'side':side,'stored_reason':item.reason,
                    'independent_reason':expected_reason,'raw_close':q.close,'raw_high':q.high,'raw_low':q.low,
                    'amount':q.amount,'volume':q.volume,'previous_close':previous_close,
                    'five_percent_proxy':bool(five),'ten_percent_proxy':bool(ten),'requested_shares':item.requested_shares,
                    'filled_shares':item.filled_shares})
            retry_checks.extend(recomputed)
            end_audit = frames['daily_inventory_audit'].loc[lambda f:f.symbol.eq(symbol)&f.date.eq(pd.Timestamp(day))].iloc[0]
            case_audits.append(end_audit.to_dict()|{'test_year':year,'candidate_index':candidate})
            pair = events.loc[events.pair_id.eq(first.pair_id)].copy().assign(test_year=year,candidate_index=candidate)
            case_events.append(pair)
            record={'test_year':year,'candidate_index':candidate,'symbol':symbol,'failure_date':day,
                'side_to_recover':side,'pair_id':first.pair_id,'first_signal_time':first.signal_datetime,
                'first_execution_time':first.datetime,'first_reference_price':first.reference_price,'original_shares':first.shares,
                'exit_mode':exit_row.exit_mode,'exit_reason':exit_row.exit_reason,'original_deadline':exit_row.original_deadline,
                'exit_signal_time':exit_row.exit_signal_time,'exit_earliest_time':exit_row.exit_earliest_time,
                'previous_close':previous_close,'previous_close_time':previous_time,
                'raw_rows_entire_day':len(today),'raw_matches_saved_dense_six_columns':True,
                'expected_retry_minutes':len(expected),'actual_retry_minutes':len(attempted),'missing_retry_minutes':0,
                'missing_quote_rows':0,'other_block_reasons':0,
                'zero_volume_or_amount_attempts':int(blocks.reason.eq('zero_price_or_volume').sum()),
                'limit_proxy_attempts':int(blocks.reason.eq('conservative_price_limit_proxy').sum()),
                'five_percent_candidate_price':st_price,'recovery_window_price_min':window.close.min(),
                'recovery_window_price_max':window.close.max(),'remaining_eod_shares':first.shares,
                'inventory_difference':end_audit.inventory_difference,'recovery_failure':bool(end_audit.recovery_failure),
                'final_recovery_time':recovery.datetime.max(),'next_day_recovered_shares':int(recovery.shares.sum())}
            case_summaries.append(record)
    save('case_retry_quote_checks.csv',pd.DataFrame(retry_checks))
    save('case_summary.csv',pd.DataFrame(case_summaries))
    save('case_pair_events.csv',pd.concat(case_events,ignore_index=True))
    save('case_eod_inventory.csv',pd.DataFrame(case_audits))
    save('case_raw_minute_quotes.csv',pd.concat(raw_extracts,ignore_index=True))
    source=RUN/'source_snapshot/quant_backtest/dynamic_t_engine.py'
    if not source.exists(): source=ROOT/'quant_backtest/dynamic_t_engine.py'
    bind(source)
    runtime=read_json(RUN/'source_hashes.json')
    failures_by_year=[]
    for year, group in active.groupby('test_year'):
        item={'test_year':int(year),'active_candidates':len(group),'execution_eligible':int(group.execution_eligible.sum())}
        item.update({name:{'min':int(group[name].min()),'max':int(group[name].max()),'nonzero_candidates':int(group[name].ne(0).sum())} for name in failcols})
        failures_by_year.append(item)
    quote_checks=pd.DataFrame(retry_checks)
    summary={'all_candidate_eligibility_recomputed':True,'all324_daily_metric_counts_match':True,
        'candidate_metrics_scope':len(active),'detailed_accounts':len(accounts),'common_failure_cases':3,
        'case_pair_episodes':len(case_summaries),'raw_stock_days':3,'raw_rows_checked':720,
        'stored_dense_quote_columns_exact':True,'retry_attempts_checked':len(quote_checks),
        'missing_quote_or_retry_in_cases':0,'reasons_in_cases':quote_checks.stored_reason.value_counts().to_dict(),
        'sample_recovery_reason_counts':pd.DataFrame(reason_counts).query("event_type=='t_recovery'").groupby('reason').attempt_rows.sum().to_dict(),
        'folds':failures_by_year,'outer_read':False,'account_or_model_rerun':False,'parameters_changed':False,
        'interpretation_boundary':'three common bottlenecks independently checked; not an exhaustive independent full-account replay',
        'profit_filter_used_to_select_samples':False}
    write_json('summary.json',summary)
    for name,value in INPUTS.items():
        assert sha(name)==value['sha256'],name
    write_json('inputs_sha256.json',INPUTS)
    write_json('verification.json',{'passed':True,'input_files_unchanged':len(INPUTS),'script_sha256':sha(__file__),
        'independent_reason_and_retry_checks_passed':True,'no_production_import':True,'no_account_rerun':True,
        'no_outer_read':True,'runtime_sources_modified':False})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
