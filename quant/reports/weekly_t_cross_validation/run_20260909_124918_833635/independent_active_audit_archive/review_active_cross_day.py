"""Read completed v9 2025 ledgers/raw minutes; no retraining or trading replay."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
p=Path('/data/lxl/seed-emulator/topology/quant/reports/weekly_t_cross_validation/run_20260909_124918_833635')
cfg=json.loads((p/'resolved_config.json').read_text());f=p/'year_2025'
t=pd.read_csv(f/'test_round_trips.csv',parse_dates=['first_datetime','second_datetime'])
e=pd.read_csv(f/'test_trade_events.csv',parse_dates=['datetime'])
a=pd.read_csv(f/'blocked_orders.csv',parse_dates=['datetime'])
i=pd.read_csv(f/'daily_inventory_audit.csv',parse_dates=['date'])
late=t[t.first_datetime.dt.normalize()!=t.second_datetime.dt.normalize()]
output=[];sources=[f/'test_round_trips.csv',f/'test_trade_events.csv',f/'blocked_orders.csv',f/'daily_inventory_audit.csv',p/'resolved_config.json']
for row in late.itertuples():
    rpath=Path(cfg['data_root'])/'2025'/f'{row.symbol}.csv';sources.append(rpath)
    raw=pd.read_csv(rpath,parse_dates=['datetime']).sort_values('datetime').drop_duplicates('datetime',keep='last').set_index('datetime')
    prev=raw.loc[raw.index<row.first_datetime.normalize(),'close'].iloc[-1]
    q=a[a.symbol.eq(row.symbol)&a.event_type.eq('t_recovery')&a.datetime.between(row.first_datetime,row.second_datetime)].copy()
    q=q.join(raw[['close','high','low','volume','amount']],on='datetime',validate='many_to_one')
    blocked=q[q.reason.eq('conservative_price_limit_proxy')]
    event=e[e.event_type.eq('t')&e.pair_id.eq(row.pair_id)]
    fee=0.;slippage=0.;flow=0.
    for v in event.itertuples():
        expected=max(v.notional*cfg['commission_rate'],cfg['min_commission'])+v.notional*cfg['transfer_rate']
        if v.side=='sell':expected+=v.notional*cfg['stamp_rate_from_20230828']
        assert abs(expected-v.fee)<1e-9
        fee+=expected;slippage+=v.slippage_cost
        flow+=v.notional-expected if v.side=='sell' else -v.notional-expected
    assert abs(flow-row.net_pnl)<1e-9
    first_ref=row.sell_price/(1-cfg['slippage_rate']);second_ref=row.buy_price/(1+cfg['slippage_rate'])
    gross_ref=(first_ref-second_ref)*row.shares
    assert abs(gross_ref-slippage-fee-flow)<1e-8
    inv=i[i.symbol.eq(row.symbol)&i.date.between(row.first_datetime.normalize(),row.second_datetime.normalize())]
    output.append({'pair_id':int(row.pair_id),'symbol':row.symbol,'shares':int(row.shares),'direction':row.direction,'first_time':str(row.first_datetime),'planned_recovery_time':row.planned_recovery_time,'actual_recovery_time':str(row.second_datetime),'raw_first_price':first_ref,'raw_recovery_price':second_ref,'reference_gross_pnl_CNY':gross_ref,'fees_CNY':fee,'slippage_cost_CNY':slippage,'net_pnl_CNY':flow,'first_day_previous_close':float(prev),'first_day_rounded_10pct_candidate_price':float(np.floor(prev*1.1*100+.5+1e-8)/100),'recovery_reject_reasons':q.reason.value_counts().to_dict(),'limit_blocked_rows':len(blocked),'limit_blocked_return_min':float((blocked.close/prev-1).min()),'limit_blocked_return_max':float((blocked.close/prev-1).max()),'limit_blocked_distinct_prices':sorted(blocked.close.unique().tolist()),'limit_blocked_flat_rows':int(np.isclose(blocked.high,blocked.low).sum()),'limit_blocked_9_8_percent_rows':int((blocked.close/prev-1>=cfg['nonflat_limit_proxy']).sum()),'daily_inventory':json.loads(inv[['date','opening_shares','closing_shares','inventory_difference','opening_sellable','closing_sellable','day_sold','recovery_failure','t1_violation_count']].to_json(orient='records',date_format='iso'))})
summary={'classification':'v9_completed_2025_cross_day_execution_review','read_only':True,'future_data_used_for_rule_selection':False,'pairs':output,'total_cross_day_net_pnl_CNY':sum(r['net_pnl_CNY'] for r in output),'total_cross_day_fees_CNY':sum(r['fees_CNY'] for r in output),'total_cross_day_slippage_CNY':sum(r['slippage_cost_CNY'] for r in output),'other_same_day_pairs_net_pnl_CNY':float(t.net_pnl.sum()-sum(r['net_pnl_CNY'] for r in output)),'input_sha256':{str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in sources},'limitations':'Rejected recovery observations verify the implemented directional price-limit proxy, not historical exchange-confirmed ST/limit status. No hypothetical fill is added.'}
(p/'independent_cross_day_execution_2025.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# 2025年两次跨日恢复独立核对','','只读已完成的外层执行产物；本核对不用于选择、调整或补填成交。','', '|股票|先卖100股|计划恢复|实际恢复|原价差(元)|滑点(元)|费用(元)|净损益(元)|','|---|---|---|---|---:|---:|---:|---:|']
for r in output:
    lines.append(f"|{r['symbol']}|{r['first_time']}|{r['planned_recovery_time']}|{r['actual_recovery_time']}|{r['reference_gross_pnl_CNY']:.4f}|{r['slippage_cost_CNY']:.4f}|{r['fees_CNY']:.9f}|{r['net_pnl_CNY']:.9f}|")
lines+=['','两组在卖出日都少100股，次日买回后恢复底仓；次日买入未增加当日可卖旧库存。对应4个库存不一致日、2个恢复失败日，不能声称满足每日开收盘股数相同。','']
for r in output:
    lines.append(f"- {r['symbol']}：恢复拒单原因 {r['recovery_reject_reasons']}。原始行情前收 {r['first_day_previous_close']:.2f}，限价代理阻单价格 {r['limit_blocked_distinct_prices']}，涨幅 {r['limit_blocked_return_min']:.6%} 至 {r['limit_blocked_return_max']:.6%}；全部 {r['limit_blocked_9_8_percent_rows']} 个限价阻单满足9.8%方向阈值。")
lines+=['',f"两组净损益合计 {summary['total_cross_day_net_pnl_CNY']:.9f} 元，费用 {summary['total_cross_day_fees_CNY']:.9f} 元，滑点 {summary['total_cross_day_slippage_CNY']:.4f} 元；其余当日恢复组净损益 {summary['other_same_day_pairs_net_pnl_CNY']:.9f} 元。",'',summary['limitations']]
(p/'independent_cross_day_execution_2025.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},ensure_ascii=False,indent=2))
