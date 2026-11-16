"""固定一周/代码/双方向的真实提前恢复通路诊断，不用于训练、选参或盈利论证。"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path('/data/lxl/seed-emulator/topology/quant')
sys.path.insert(0,str(ROOT))
from quant_backtest.annual_panel import write_json
from quant_backtest.annual_portfolio import calculate_fee
from quant_backtest.weekly_t_data import load_weekly_symbol_bars
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_quote_store import build_dense_recovery_store
from quant_backtest.weekly_t_validation import validate_candidate_result
from scripts.run_weekly_t_cross_validation import save_result, write_frame

SOURCE=ROOT/'reports/weekly_t_cross_validation_smoke/run_20260908_221134_755324'
OUT=ROOT/'reports/weekly_t_cross_validation_smoke/real_t_v5_early_execution'
OUT.mkdir(exist_ok=False)
shutil.copyfile(__file__,OUT/'run_diagnostic.py')
cfg=json.loads((SOURCE/'resolved_config.json').read_text())
cfg.update(initial_cash=1000000.,top_n=4,weekly_top_n=4,base_fraction=.95,short_t_fraction=.25,
           max_pending_t_notional_fraction=.1,smoke_weeks=0)
manifest=json.loads((SOURCE/'data_manifest.json').read_text())
horizon_manifest=json.loads((SOURCE/'horizon_manifest.json').read_text())
symbols=sorted(manifest['symbols'])[:4]
start,end=pd.Timestamp('2024-01-08'),pd.Timestamp('2024-01-12')
source_files=[ROOT/'quant_backtest/weekly_t_engine.py',ROOT/'quant_backtest/weekly_t_quote_store.py',
              ROOT/'quant_backtest/weekly_t_data.py',ROOT/'quant_backtest/weekly_t_validation.py']
protocol={'classification':'execution_only_not_profit_evidence','created_before_simulation':datetime.now().isoformat(),
          'symbol_selection':'lexicographic first 4 of existing 8-symbol cache; not outcome selected',
          'symbols':symbols,'entry_date':str(start),'exit_date':str(end),'signal_time':'10:00','horizon_minutes':20,
          'planned_recovery_time':'10:21','directions':['sell_first','buy_first'],'fraction':.25,
          'selection_bypassed':True,'model_fitting':False,'candidate_selection':False,'priority':0.,
          'recovery_scope':'all configured fixed clocks plus 14:55..15:00 from fixed stocks/week raw bars',
          'config':cfg,'source_cache':manifest['cache_path'],'short_label_cache':horizon_manifest['cache_path'],
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
write_json(OUT/'frozen_execution_protocol.json',protocol)
write_json(OUT/'resolved_config.json',cfg)
cache=Path(manifest['cache_path'])
daily_all=pd.read_pickle(cache/'daily.pkl')
intraday_all=pd.read_pickle(cache/'intraday.pkl')
assert set(pd.bdate_range(start,end)).issubset(set(daily_all.date)),'fixed calendar week absent'
daily=daily_all.loc[daily_all.symbol.isin(symbols)&daily_all.date.between(start,end)].copy()
intraday=intraday_all.loc[intraday_all.symbol.isin(symbols)&intraday_all.date.between(start,end)].copy()
selections=[]
for rank,symbol in enumerate(symbols,1):
    previous=daily_all.loc[daily_all.symbol.eq(symbol)&daily_all.date.lt(start)].sort_values('date').iloc[-1]
    selections.append({'trade_week':start,'entry_date':start,'exit_date':end,'symbol':symbol,'rank':rank,
                       'universe_previous_close':float(previous.close),'reference_date':previous.date})
selections=pd.DataFrame(selections)
write_frame(selections,OUT/'weekly_selections.csv')
write_frame(daily,OUT/'input_daily.csv')
write_frame(intraday,OUT/'input_intraday.csv')
raws=[];raw_audits=[]
for symbol in symbols:
    path=Path(cfg['data_root'])/'2024'/f'{symbol}.csv'
    raw,audit=load_weekly_symbol_bars([str(path)],str(start.date()),str(end.date()))
    raws.append(raw.assign(symbol=symbol))
    raw_audits.append({'symbol':symbol,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**audit})
raw=pd.concat(raws,ignore_index=True)
lookup=raw.set_index(['datetime','symbol'])
minutes={int(t[:2])*60+int(t[3:]) for t in cfg['scheduled_recovery_times']}|set(range(895,901))
quotes=raw.loc[(raw.datetime.dt.hour*60+raw.datetime.dt.minute).isin(minutes)].copy()
write_frame(quotes,OUT/'input_recovery_quotes.csv')
store=build_dense_recovery_store(quotes,OUT/'dense_recovery_cache')
write_json(OUT/'input_manifest.json',{'raw_sources':raw_audits,'quote_rows':len(quotes),
                                   'store_fingerprint':store.fingerprint,'store_path':str(store.cache_path)})
labels=pd.read_pickle(Path(horizon_manifest['cache_path'])/'labels.pkl')
labels=labels.loc[labels.symbol.isin(symbols)&labels.datetime.dt.normalize().between(start,end)
                  &labels.datetime.dt.strftime('%H:%M').eq('10:00')&labels.horizon_minutes.eq(20)]
write_frame(labels,OUT/'cached_h20_labels.csv')
for row in labels.itertuples(index=False):
    assert row.recovery_time==row.datetime.normalize()+pd.Timedelta(hours=10,minutes=21)
    key=(row.recovery_time,row.symbol)
    if key in lookup.index and np.isfinite(row.recovery_price):
        assert abs(row.recovery_price-lookup.loc[key,'close'])<1e-9


def independent_audit(result,out):
    """从raw和事件独立重放股份/现金/容量/费用，以及配对片段现金流。"""
    replay=[];error=0.;peak=0.;event_checks=0
    for account,key in [('actual','trade_events'),('baseline','baseline_events')]:
        events=result[key]
        assert events.datetime.is_monotonic_increasing
        positions=defaultdict(int);cash=cfg['initial_cash'];marks={};pending={}
        for date in sorted(daily.date.unique()):
            date=pd.Timestamp(date);opening=positions.copy();sold=defaultdict(int);adds=defaultdict(int);exits=defaultdict(int)
            used=defaultdict(float)
            current=events.loc[events.datetime.dt.normalize().eq(date)]
            for event in current.itertuples(index=False):
                q=int(event.shares);assert q>0 and q%100==0
                minute=event.datetime if event.datetime.time()!=pd.Timestamp('09:30').time() else date+pd.Timedelta(hours=9,minutes=31)
                bar=lookup.loc[(minute,event.symbol)]
                price=float(bar.open if event.datetime.time()==pd.Timestamp('09:30').time() else bar.close)
                assert abs(event.reference_price-price)<1e-8
                expected=price*(1+cfg['slippage_rate']*(1 if event.side=='buy' else -1))
                assert abs(event.price-expected)<1e-8 and bar.volume>0 and bar.amount>0
                capacity_key=(event.datetime,event.symbol)
                used[capacity_key]+=q*price
                assert used[capacity_key]<=bar.amount*cfg['max_bar_amount_fraction']+1e-6
                assert abs(event.fee-calculate_fee(q*expected,event.side,event.datetime,cfg))<1e-7
                before=opening[event.symbol]-sold[event.symbol]
                assert event.sellable_before==before
                if event.side=='sell':
                    assert q<=before and q<=positions[event.symbol]
                    sold[event.symbol]+=q;positions[event.symbol]-=q;cash+=event.notional-event.fee
                    if event.event_type=='weekly':exits[event.symbol]+=q
                else:
                    positions[event.symbol]+=q;cash-=event.notional+event.fee
                    if event.event_type=='weekly':adds[event.symbol]+=q
                assert event.sellable_after==opening[event.symbol]-sold[event.symbol]
                error=max(error,abs(cash-event.cash_after));assert error<1e-5 and cash>=-1e-6
                if event.event_type=='t':
                    if event.fragment_id==0:
                        assert event.datetime==date+pd.Timedelta(hours=10,minutes=1)
                        assert event.planned_recovery_time==date+pd.Timedelta(hours=10,minutes=21)
                        pending[event.pair_id]=[q,price]
                    else:
                        assert event.datetime>=event.planned_recovery_time
                        pending[event.pair_id][0]-=q
                        assert pending[event.pair_id][0]>=0
                    notional=sum(n*p for n,p in pending.values());peak=max(peak,notional)
                    assert notional<=cfg['initial_cash']*cfg['max_pending_t_notional_fraction']+1e-6
                event_checks+=1
            for row in daily.loc[daily.date.eq(date)].itertuples(index=False):
                if np.isfinite(row.close) and row.close>0:marks[row.symbol]=row.close
            curve=result['equity_curve'].loc[result['equity_curve'].datetime.dt.normalize().eq(date)].iloc[0]
            equity=cash+sum(q*marks.get(symbol,0.) for symbol,q in positions.items())
            expected_equity=curve.equity if account=='actual' else curve.baseline_equity
            assert abs(equity-expected_equity)<1e-5
            for symbol in set(positions)|set(opening):
                target=opening[symbol]+adds[symbol]-exits[symbol]
                replay.append({'account':account,'date':date,'symbol':symbol,'opening_shares':opening[symbol],
                               'closing_shares':positions[symbol],'day_sold':sold[symbol],'target_shares':target,
                               'inventory_difference':positions[symbol]-target,'cash':cash})
    replay=pd.DataFrame(replay);write_frame(replay,out/'independent_inventory_replay.csv')
    pairs=[];t=result['trade_events'].query("event_type=='t'");trips=result['round_trips']
    for pair_id,events in t.groupby('pair_id',sort=False):
        first=events.iloc[0];second=events.iloc[1:];fragments=trips.loc[trips.pair_id.eq(pair_id)]
        matched=int(second.shares.sum());fraction=matched/first.shares
        firstflow=first.notional-first.fee if first.side=='sell' else -first.notional-first.fee
        net=firstflow*fraction+np.where(second.side.eq('sell'),second.notional-second.fee,-second.notional-second.fee).sum()
        assert abs(net-fragments.net_pnl.sum())<1e-6
        assert abs(first.fee*fraction+second.fee.sum()-(fragments.buy_fee+fragments.sell_fee).sum())<1e-6
        assert abs(first.slippage_cost*fraction+second.slippage_cost.sum()-fragments.slippage_cost.sum())<1e-6
        pairs.append({'pair_id':pair_id,'symbol':first.symbol,'first_datetime':first.datetime,'first_shares':int(first.shares),
                      'restored_shares':matched,'fragments':len(fragments),'first_recovery':second.datetime.min(),
                      'last_recovery':second.datetime.max(),'net_pnl':net,'pnl_error':net-fragments.net_pnl.sum()})
    pairs=pd.DataFrame(pairs);write_frame(pairs,out/'independent_pair_cashflow.csv')
    same=result['metrics']['same_base_plan'] and result['metrics']['pending_pairs']==0
    if same:assert abs(result['metrics']['total_t_marked_pnl']-trips.net_pnl.sum())<1e-5
    plans=result['weekly_entry_plans']
    expected=np.floor(plans.entry_budget/(plans.known_reference_price*(1+cfg['slippage_rate']))/100)*100
    assert expected.eq(plans.planned_shares).all()
    return {'raw_event_checks':event_checks,'inventory_rows':len(replay),'max_cash_error':error,'peak_pending_notional':peak,
            'raw_price_capacity_fees_match':True,'t1_legal':True,'same_base_no_pending_pnl_identity_checked':same,
            'known_price_share_plans_match':True,'first_recovery_1021_pairs':int(pairs.first_recovery.dt.strftime('%H:%M').eq('10:21').sum()) if len(pairs) else 0,
            'partial_pairs':int(pairs.fragments.gt(1).sum()) if len(pairs) else 0}

summaries=[]
for name,action in [('sell_first',-1),('buy_first',1)]:
    out=OUT/name;out.mkdir()
    clocks=pd.bdate_range(start,end)[1:-1]+pd.Timedelta(hours=10)
    decisions=pd.DataFrame([{'datetime':clock,'symbol':symbol,'action':action,'fraction':.25,'priority':0.,
                             'planned_recovery_time':clock+pd.Timedelta(minutes=21)} for clock in clocks for symbol in symbols])
    write_frame(decisions,out/'t_decisions.csv')
    result=run_weekly_portfolio(selections,daily,intraday,decisions,cfg,recovery_bars=store)
    standard=validate_candidate_result(result,cfg)
    metrics=save_result(result,out,cfg)
    independent=independent_audit(result,out)
    write_json(out/'independent_execution_audit.json',{'classification':'execution_only_not_profit_evidence',
                                                     'standard':standard,'independent':independent})
    summaries.append({'direction':name,**metrics,**independent})
summary=pd.DataFrame(summaries);write_frame(summary,OUT/'diagnostic_summary.csv')
rows=['|方向|完整组合|恢复片段|10:21开始恢复|分批组合|T净增量(元)|库存差异日|期末待恢复|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in summaries:
    rows.append(f"|{row['direction']}|{row['t_pair_count']}|{row['fragment_count']}|{row['first_recovery_1021_pairs']}|{row['partial_pairs']}|{row['t_added_equity']:.6f}|{row['inventory_mismatch_days']}|{row['pending_pairs']}|")
(OUT/'README.md').write_text('# 真实行情提前恢复执行诊断\n\n此目录仅验证执行通路及记账，不是策略盈利证据，也不参与模型/阈值/方向选择。\n\n'
    +'2024年1月8日至12日；按代码顺序固定选择已有8股缓存的前4只。两个独立100万元账户分别固定卖先、买先，95%底仓预算按4股均分；实际首分钟容量可缩小底仓。中间三日每股固定10:00信号，10:01首腿，20交易分钟后的10:21起恢复，比例25%，未恢复名义金额上限10万元。\n\n'
    +'原缓存只读。因原早报价限于v4年度入选并集，本诊断预先从固定4股固定一周raw提取全部事前配置时钟和尾盘报价，建立独立DenseRecoveryBarStore；未按后续收益或可恢复性挑股票、方向、日期或首腿。短周期缓存标签与raw的10:21价一致。\n\n'
    +'\n'.join(rows)+'\n\n双边佣金0.08%、过户0.001%、滑点0.05%，卖出印花税0.05%，最低佣金0，分钟成交额参与上限5%。每笔价格、容量、费用及现金从原始bar和成交事件独立复核；每日可卖额由独立重放的开盘旧股推导；片段费用/滑点及配对净利核对通过。具有相同底仓计划且无待恢复组合时，权益净增量等于配对净利。\n\n'
    +'缺价、受阻恢复或周退出失败保留实际持仓；检查通过表示记账合法，并不把失败解释成满足每日股数不变。完整明细在两个方向子目录；frozen_execution_protocol.json记录事前固定协议。\n',encoding='utf-8')
print(json.dumps({'output':str(OUT),'summary':[{k:r[k] for k in ['direction','started_pair_count','t_pair_count','fragment_count','first_recovery_1021_pairs','partial_pairs','t_added_equity','inventory_mismatch_days','recovery_failure_days','pending_pairs','base_plan_mismatch_count','weekly_exit_failure_days','raw_event_checks','max_cash_error','peak_pending_notional']} for r in summaries]},ensure_ascii=False,indent=2))
