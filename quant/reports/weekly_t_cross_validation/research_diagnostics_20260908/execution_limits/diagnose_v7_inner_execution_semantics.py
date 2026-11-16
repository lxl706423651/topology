"""只分析v7首折内部2026的已存拒单和v4冻结validation股周，禁止外层结果和完整重跑。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path('/data/lxl/seed-emulator/topology/quant')
RUN=ROOT/'reports/weekly_t_cross_validation/run_20260908_231721_245847'
FOLD=RUN/'year_2023';OUT=RUN/'inner_execution_semantics_diagnostic';OUT.mkdir(exist_ok=True)
cfg=json.loads((RUN/'resolved_config.json').read_text());manifest=json.loads((RUN/'data_manifest.json').read_text())
attempts=[];failures=[];source_hashes={}
for suffix,container in [('t_attempts',attempts),('t_inventory_failures',failures)]:
    for path in sorted((FOLD/'validation_candidate_audits').glob(f'*_{suffix}.csv')):
        source_hashes[str(path.relative_to(RUN))]=hashlib.sha256(path.read_bytes()).hexdigest()
        f=pd.read_csv(path)
        if len(f):f['candidate_index']=int(path.name.split('_')[1]);container.append(f)
attempts=pd.concat(attempts,ignore_index=True);failures=pd.concat(failures,ignore_index=True)
attempts['datetime']=pd.to_datetime(attempts.datetime);attempts['date']=attempts.datetime.dt.normalize()
failures['date']=pd.to_datetime(failures.date)
assert attempts.datetime.dt.year.eq(2026).all() and failures.date.dt.year.eq(2026).all()
recovery=attempts.loc[attempts.event_type.eq('t_recovery')].copy()
daily=pd.read_pickle(Path(manifest['cache_path'])/'daily.pkl');daily=daily.loc[daily.date.dt.year.eq(2026)].copy()
day_index=daily.set_index(['date','symbol'])
raw={}
for symbol in sorted(recovery.symbol.unique()):
    path=ROOT/cfg['data_root']/'2026'/f'{symbol}.csv';f=pd.read_csv(path,parse_dates=['datetime'])
    f=f.sort_values('datetime',kind='stable').drop_duplicates('datetime',keep='last').set_index('datetime')
    raw[symbol]=f
    source_hashes[str(path.relative_to(ROOT))]={'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}

def classify(price,high,low,amount,volume,previous,side):
    direction=1 if side=='buy' else -1
    signed=(price/previous-1)*direction
    c5=np.floor(previous*(1+direction*.05)*100+.5+1e-8)/100
    c10=np.floor(previous*(1+direction*.10)*100+.5+1e-8)/100
    flat=high==low
    five=flat and abs(price-c5)<=1e-6
    ten=signed>=.098
    valid=all(np.isfinite(v) for v in [price,high,low,amount]) and price>0 and amount>0 and volume>0
    valid &= low-1e-6<=price<=high+1e-6
    capacity=int(max(amount*.05,0)/price/100)*100 if valid else 0
    reason='zero_or_invalid_quote' if not valid else '10pct_threshold' if ten else '5pct_flat_exact' if five else 'capacity' if capacity<100 else 'quote_eligible'
    return {'price':price,'high':high,'low':low,'amount':amount,'volume':volume,'previous_close':previous,
        'signed_return':signed,'flat':flat,'candidate_5':c5,'candidate_10':c10,
        'distance_5_ticks':int(round((price-c5)*direction*100)),'distance_10_ticks':int(round((price-c10)*direction*100)),
        'proxy_5_branch':five,'proxy_10_branch':ten,'classification':reason,'capacity_shares':capacity,
        'flat_between_5_and_9_8':bool(flat and .05<=signed<.098)}

annotations=[]
for r in recovery.itertuples(index=False):
    day=day_index.loc[(r.date,r.symbol)];timestamp=r.datetime
    stamp=timestamp+pd.Timedelta(minutes=1) if timestamp.hour==9 and timestamp.minute==30 else timestamp
    q=raw[r.symbol].loc[stamp]
    price=float(q.open if timestamp!=stamp else q.close)
    annotations.append(classify(price,q.high,q.low,q.amount,q.volume,day.previous_close,r.side))
classified=pd.concat([recovery.reset_index(drop=True),pd.DataFrame(annotations)],axis=1)
proxy=classified.loc[classified.reason.eq('conservative_price_limit_proxy')].copy()
assert proxy.classification.isin(['5pct_flat_exact','10pct_threshold']).all()
classified.to_csv(OUT/'inner_recovery_attempt_quotes.csv',index=False)
unique_proxy=proxy.drop_duplicates(['datetime','symbol','side'])
failures.to_csv(OUT/'inner_inventory_failure_rows.csv',index=False)

potential=[]
for r in failures.loc[failures.recovery_failure].itertuples(index=False):
    a=classified.loc[classified.candidate_index.eq(r.candidate_index)&classified.date.eq(r.date)&classified.symbol.eq(r.symbol)]
    assert len(a)>0
    first=a.datetime.min();side=a.side.iloc[0];needed=int(a.requested_shares.max())
    quotes=raw[r.symbol].loc[(raw[r.symbol].index>=first)&(raw[r.symbol].index.normalize()==r.date)]
    previous=day_index.loc[(r.date,r.symbol),'previous_close'];legal=[]
    for ts,q in quotes.iterrows():
        c=classify(q.close,q.high,q.low,q.amount,q.volume,previous,side)
        if c['classification']=='quote_eligible':legal.append((ts,c['capacity_shares'],c['price']))
    attempted=set(a.datetime)
    missing=[x for x in legal if x[0] not in attempted]
    potential.append({'candidate_index':int(r.candidate_index),'date':str(r.date.date()),'symbol':r.symbol,'side':side,
        'first_recorded_recovery_attempt':str(first),'remaining_request_shares':needed,
        'same_day_recorded_partial_fills':int(a.filled_shares.sum()),'reason_counts':a.reason.value_counts().to_dict(),
        'all_minute_quote_eligible_count':len(legal),'unattempted_quote_eligible_count':len(missing),
        'unattempted_total_capacity':sum(x[1] for x in missing),
        'unattempted_capacity_covers_request':sum(x[1] for x in missing)>=needed,
        'first_unattempted_quote':str(missing[0][0]) if missing else None,
        'largest_unattempted_minute_capacity':max((x[1] for x in missing),default=0)})
pd.DataFrame(potential).to_csv(OUT/'failed_stockday_all_minute_quote_feasibility.csv',index=False)

# 周退出的完整逐笔attempt未被训练worker导出；仅检查本折冻结validation已知退出日的当前报价约束。
selection_path=ROOT/cfg['frozen_selection_report_dir']/'year_2023/validation_weekly_selections.csv'
selected=pd.read_csv(selection_path,parse_dates=['entry_date','exit_date','trade_week'])
assert selected.entry_date.dt.year.eq(2026).all() and selected.exit_date.dt.year.eq(2026).all()
scheduled=[]
for r in selected.itertuples(index=False):
    key=(r.exit_date,r.symbol)
    if key not in day_index.index:
        scheduled.append({'date':r.exit_date,'symbol':r.symbol,'rank':r.rank,'classification':'missing_daily_quote'});continue
    q=day_index.loc[key]
    if not np.isfinite(q.previous_close) or q.previous_close<=0:
        scheduled.append({'date':r.exit_date,'symbol':r.symbol,'rank':r.rank,'classification':'unknown_previous_close'});continue
    c=classify(q.close,q.close_high,q.close_low,q.close_amount,q.close_volume,q.previous_close,'sell')
    scheduled.append({'date':r.exit_date,'symbol':r.symbol,'rank':r.rank,**c})
scheduled=pd.DataFrame(scheduled);scheduled.to_csv(OUT/'validation_scheduled_weekly_exit_quote_constraints.csv',index=False)
metrics=pd.read_csv(FOLD/'validation_rule_metrics.csv')
assert metrics.inner_year.eq(2026).all() and metrics.test_year.eq(2023).all()
distinct=metrics.drop_duplicates('decision_sha256')
failed=failures.loc[failures.recovery_failure]
unique_failed=failed.drop_duplicates(['date','symbol'])
result={
 'scope':{'outer_year':2023,'core_years':[2024,2025],'inner_year':2026,'outer_execution_or_failure_data_read':False,
          'complete_backtest_rerun':False,'current_source_changed':False,'rule_retuning':False},
 'attempt_candidate_files':len(list((FOLD/'validation_candidate_audits').glob('*_t_attempts.csv'))),
 'candidate_rules':len(metrics),'distinct_decision_streams':len(distinct),'inner_calendar_days':int(metrics.trading_days.iloc[0]),
 'attempt_reason_counts':{f'{event}/{reason}':int(n) for (event,reason),n in attempts.groupby(['event_type','reason']).size().items()},
 'proxy_rejection_rows':len(proxy),'unique_proxy_rejection_minutes':len(unique_proxy),
 'proxy_branch_counts':proxy.classification.value_counts().to_dict(),'unique_proxy_branch_counts':unique_proxy.classification.value_counts().to_dict(),
 'flat_5_to_9_8_rejection_rows':int(proxy.flat_between_5_and_9_8.sum()),
 'flat_5_to_9_8_unique_minutes':int(unique_proxy.flat_between_5_and_9_8.sum()),
 'below_9_8_far_from_rounded_5_rejection_rows':int((proxy.signed_return.lt(.098)&proxy.distance_5_ticks.ne(0)).sum()),
 'proxy_tick_distribution':{branch:{key:{str(k):int(v) for k,v in group[key].value_counts().sort_index().items()} for key in ['distance_5_ticks','distance_10_ticks']} for branch,group in unique_proxy.groupby('classification')},
 'proxy_five_examples':unique_proxy.loc[unique_proxy.classification.eq('5pct_flat_exact'),['datetime','symbol','side','price','previous_close','signed_return','candidate_5','candidate_10','distance_5_ticks','distance_10_ticks']].astype({'datetime':str}).to_dict('records'),
 'intraday_failure_rows_across_distinct_candidates':len(failed),'intraday_failure_unique_stockdays':len(unique_failed),
 'mismatch_or_failure_rows':len(failures),'restoration_positive_mismatch_rows':int(failures.inventory_difference.gt(0).sum()),
 'failure_stockday_records':potential,'quote_only_all_minute_potential_rescue_rows':sum(r['unattempted_capacity_covers_request'] for r in potential),
 'recovery_cash_or_t1_block_rows':int(recovery.reason.isin(['cash','t1_budget']).sum()),
 'pending_pairs_at_inner_end_range':[int(distinct.pending_pairs.min()),int(distinct.pending_pairs.max())],
 'baseline_weekly_unclosed_positions_range':[int(distinct.baseline_weekly_unclosed_positions.min()),int(distinct.baseline_weekly_unclosed_positions.max())],
 'weekly_exit_failure_days_range':[int(distinct.weekly_exit_failure_days.min()),int(distinct.weekly_exit_failure_days.max())],
 'weekly_quote_constraint_counts':scheduled.classification.value_counts().to_dict(),
 'scheduled_weekly_stockweeks':len(selected),
 'weekly_causal_attribution_limitation':'训练worker仅保存T拒单及T库存失败，未保存纯baseline周退出逐笔拒单。上述股周报价分类是已预定退出时刻的潜在约束，不等于26/27个实际周退出失败日的完整原因归因。',
 'historical_st_or_official_limit_columns_available':False,
 'interpretation':'当前5%分支仅拦截平价且恰在按分四舍五入的5%候选价；不存在5%到9.8%整个半无限区间的ST拒单。实际>=9.8%分支仍是保守代理，不能等同官方涨停价。',
 'all_minute_study_limitation':'只评价从已记录首个恢复时刻之后每分钟的报价/容量，不重新执行候选，不假定成交或利润；现金、跨计划约束仍须实际执行检验。禁止根据未来恢复好坏删除首腿或选择外层规则。',
 'source_identity':source_hashes,
 'engine_sha256':hashlib.sha256((RUN/'source_snapshot/weekly_t_engine.py').read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['failure_stockday_records','source_identity','proxy_five_examples','proxy_tick_distribution']},ensure_ascii=False,indent=2))
print('five examples',json.dumps(result['proxy_five_examples'],ensure_ascii=False))
print('potential rescues',json.dumps([r for r in potential if r['unattempted_capacity_covers_request']],ensure_ascii=False))
