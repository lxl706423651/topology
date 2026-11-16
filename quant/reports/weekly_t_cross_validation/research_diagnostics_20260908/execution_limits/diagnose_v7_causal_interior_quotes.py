"""只读内部拒单：用严格早于被拒时刻的有效成交证明价格是否处于当日历史极值内部；不回测、不补成交。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('/data/lxl/seed-emulator/topology/quant')
RUN=ROOT/'reports/weekly_t_cross_validation/run_20260908_231721_245847'
OUT=RUN/'inner_execution_semantics_diagnostic'
cfg=json.loads((RUN/'resolved_config.json').read_text())
a=pd.read_csv(OUT/'inner_recovery_attempt_quotes.csv',parse_dates=['datetime','date'])
fail=pd.read_csv(OUT/'failed_stockday_all_minute_quote_feasibility.csv',parse_dates=['date','first_recorded_recovery_attempt'])
weekly=pd.read_csv(OUT/'validation_scheduled_weekly_exit_quote_constraints.csv',parse_dates=['date'])
raw={}
def read_quotes(symbol):
    """读取2026原始分钟，逐日记录严格前一分钟有效收盘极值。"""
    if symbol not in raw:
        f=pd.read_csv(ROOT/cfg['data_root']/'2026'/f'{symbol}.csv',parse_dates=['datetime'])
        f=f.sort_values('datetime',kind='stable').drop_duplicates('datetime',keep='last').set_index('datetime')
        good=(f[['close','open','high','low','amount','volume']].apply(np.isfinite).all(axis=1)&f.close.gt(0)&f.amount.gt(0)&f.volume.gt(0)&f.high.ge(f.close)&f.low.le(f.close))
        days=f.index.normalize()
        clean=f.close.where(good)
        f['prior_max_close']=clean.groupby(days).cummax().groupby(days).shift(1).groupby(days).ffill()
        f['prior_min_close']=clean.groupby(days).cummin().groupby(days).shift(1).groupby(days).ffill()
        f['valid_quote']=good
        f['capacity_shares']=(f.amount*.05/f.close/100).fillna(0).clip(lower=0).astype(int)*100
        raw[symbol]=f
    return raw[symbol]

def describe(ts,symbol,side,price):
    """当日此前成交极值须严格位于当前价外侧；不使用当前bar的high/low。"""
    f=read_quotes(symbol);q=f.loc[ts]
    bound=q.prior_max_close if side=='buy' else q.prior_min_close
    interior=bool(np.isfinite(bound) and ((price<bound-1e-6) if side=='buy' else (price>bound+1e-6)))
    earlier=f.loc[(f.index.normalize()==ts.normalize())&(f.index<ts)&f.valid_quote]
    witness=earlier.loc[earlier.close.eq(bound)]
    return {'prior_extreme_price':float(bound),'earlier_extreme_time':str(witness.index[0]) if len(witness) else None,
            'causally_inside_prior_traded_extreme':interior,'current_capacity_shares':int(q.capacity_shares),
            'current_valid_quote':bool(q.valid_quote)}

proxy=a.loc[a.reason.eq('conservative_price_limit_proxy')].drop_duplicates(['datetime','symbol','side']).copy()
proxy=pd.concat([proxy.reset_index(drop=True),pd.DataFrame([describe(r.datetime,r.symbol,r.side,r.price) for r in proxy.itertuples(index=False)])],axis=1)
proxy.to_csv(OUT/'proxy_prior_trade_extreme_evidence.csv',index=False)
causal=[]
for r in fail.itertuples(index=False):
    q=read_quotes(r.symbol)
    q=q.loc[(q.index.normalize()==r.date)&(q.index>=r.first_recorded_recovery_attempt)]
    evidence=(q.close.lt(q.prior_max_close-1e-6) if r.side=='buy' else q.close.gt(q.prior_min_close+1e-6))&q.valid_quote&q.capacity_shares.ge(100)
    candidates=q.loc[evidence]
    attempts=a.loc[a.candidate_index.eq(r.candidate_index)&a.date.eq(r.date)&a.symbol.eq(r.symbol)]
    sampled=candidates.loc[candidates.index.isin(attempts.datetime)]
    causal.append({'candidate_index':r.candidate_index,'date':r.date,'symbol':r.symbol,'side':r.side,'remaining_shares':r.remaining_request_shares,
        'all_minute_causal_interior_count':len(candidates),'all_minute_capacity':int(candidates.capacity_shares.sum()),
        'all_minute_capacity_covers_request':int(candidates.capacity_shares.sum())>=r.remaining_request_shares,
        'attempted_clock_causal_interior_count':len(sampled),'attempted_clock_capacity':int(sampled.capacity_shares.sum()),
        'attempted_clock_capacity_covers_request':int(sampled.capacity_shares.sum())>=r.remaining_request_shares,
        'first_all_minute_time':str(candidates.index[0]) if len(candidates) else None,
        'first_attempted_clock_time':str(sampled.index[0]) if len(sampled) else None})
causal=pd.DataFrame(causal);causal.to_csv(OUT/'failed_stockday_causal_interior_evidence.csv',index=False)
w=weekly.loc[weekly.classification.eq('5pct_flat_exact')].copy()
w['datetime']=w.date+pd.Timedelta(hours=15)
w=pd.concat([w.reset_index(drop=True),pd.DataFrame([describe(r.datetime,r.symbol,'sell',r.price) for r in w.itertuples(index=False)])],axis=1)
w.to_csv(OUT/'scheduled_five_percent_exit_prior_trade_evidence.csv',index=False)
result={'scope':'v7 year_2023 内部2026；不读外层、不执行反事实成交、不估反事实利润',
 'unique_rejected_minutes':len(proxy),'causal_interior_minutes':int(proxy.causally_inside_prior_traded_extreme.sum()),
 'causal_interior_capacity_at_least_one_lot':int((proxy.causally_inside_prior_traded_extreme&proxy.current_capacity_shares.ge(100)).sum()),
 'interior_10_candidate_nonboundary_rows':proxy.loc[proxy.distance_10_ticks.ne(0),['datetime','symbol','side','price','candidate_10','distance_10_ticks','prior_extreme_price','earlier_extreme_time','causally_inside_prior_traded_extreme','current_capacity_shares']].to_dict('records'),
 'failed_stockday_candidate_rows_with_all_minute_causal_capacity':int(causal.all_minute_capacity_covers_request.sum()),
 'failed_stockday_candidate_rows_with_existing_clock_causal_capacity':int(causal.attempted_clock_capacity_covers_request.sum()),
 'unique_failed_stockdays_with_existing_clock_causal_capacity':len(causal.loc[causal.attempted_clock_capacity_covers_request].drop_duplicates(['date','symbol'])),
 'scheduled_five_percent_exit_causal_interior_count':int(w.causally_inside_prior_traded_extreme.sum()),
 'caveat':'先前有效成交极值仅在同日固定价格边界及数据正确前提下，证明该价格位于可观察边界内部；这不是真实ST/除权基准信息，也不能证明挂单实际成交。仅是事前可得的定向代理改进设计证据，所有现金/T1/容量/共享时钟仍应在未来独立协议中验证。'}
(OUT/'causal_interior_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
