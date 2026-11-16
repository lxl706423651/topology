"""只读v12完整内部55候选的择时/费用/旧股额度，不读外层、不修改规则。

仅在对应fold的候选指标和选中规则已落盘时分析；主进程终态后才能写终版。
输出位于本脚本目录，每折账户相互重叠，任何跨候选汇总均为实验诊断而非组合收益。
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'reports/balanced_inventory_t/run_20260910_140512_310726'
OUT=Path(__file__).resolve().parent
INPUTS={};TABLES={}
FAILURES=['inventory_mismatch_days','recovery_failure_days','pending_pairs','t1_violation_count','base_plan_mismatch_count']


def sha(path):
    """实际输入逐文件内容SHA。"""
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def record(path):
    """登记仅读取的内部报告文件。"""
    path=Path(path).resolve()
    if str(path) not in INPUTS:INPUTS[str(path)]=sha(path)
    return path


def csv(path):
    """保持保存账簿浮点精度，允许无表体的空拒单文件。"""
    path=record(path)
    try:return pd.read_csv(path,float_precision='round_trip')
    except pd.errors.EmptyDataError:return pd.DataFrame()


def read_json(path):
    """读取并登记JSON身份。"""
    return json.loads(record(path).read_text())


def append(table,values):
    """追加每账户的派生诊断，保留候选身份避免错加不同账户。"""
    TABLES.setdefault(table,[]).append(values)


def to_time(series):
    """兼容日期和完整时间混排。"""
    return pd.to_datetime(series,format='mixed')


def pair_statistics(part):
    """完整T组的实际金额/标准手及方向参考毛价差，不把恢复片段当组。"""
    denominator=float(part.first_reference_notional.sum())
    values={k:float(part[k].sum()) for k in ['reference_gross_pnl','slippage_cost','fees','net_pnl']}
    return {'complete_pairs':len(part),'recovery_fragments':int(part.fragments.sum()),
        'paired_lots':float(part.original_shares.sum()/100),**values,
        'reference_first_notional':denominator,
        **{k+'_bp':value/denominator*1e4 if denominator>0 else None for k,value in values.items()},
        'positive_pair_fraction':float(part.net_pnl.gt(0).mean()) if len(part) else None,
        'stock_days':len(part[['date','symbol']].drop_duplicates()),'symbols':int(part.symbol.nunique())}


def one_candidate(year,inner,folder,rule,selected,cfg):
    """按真实事件和每日旧股快照识别额度耗尽，拒单只作为到达时执行诊断。"""
    identity={'fold':year,'inner_year':inner,**rule,'selected':rule['candidate_index']==selected}
    events=csv(folder/'test_trade_events.csv');trips=csv(folder/'test_round_trips.csv')
    blocked=csv(folder/'blocked_orders.csv');audit=csv(folder/'daily_inventory_audit.csv')
    metrics=read_json(folder/'metrics.json');funnel=read_json(folder/'signal_causality_audit.json')
    events['datetime']=to_time(events.datetime);events['signal_datetime']=to_time(events.signal_datetime)
    events['date']=events.datetime.dt.normalize();audit['date']=to_time(audit.date)
    if len(blocked):
        blocked['datetime']=to_time(blocked.datetime);blocked['date']=blocked.datetime.dt.normalize()
        blocked['clock']=blocked.datetime.dt.strftime('%H:%M')
    first=events.loc[events.event_type.eq('t')&events.fragment_id.eq(0)].copy()
    first['clock']=first.signal_datetime.dt.strftime('%H:%M')
    pair_rows=[]
    groups={key:p for key,p in trips.groupby('pair_id')} if len(trips) else {}
    for event in first.itertuples(index=False):
        parts=groups.get(event.pair_id,pd.DataFrame())
        complete=len(parts)>0 and parts.shares.sum()==event.shares
        if not complete:continue
        fee=float(parts.buy_fee.sum()+parts.sell_fee.sum());slip=float(parts.slippage_cost.sum())
        pair_rows.append({'pair_id':int(event.pair_id),'symbol':event.symbol,'date':event.date,
            'direction':'buy_first' if event.side=='buy' else 'sell_first','clock':event.clock,
            'original_shares':int(event.shares),'first_reference_notional':float(event.shares*event.reference_price),
            'fragments':len(parts),'reference_gross_pnl':float(parts.gross_pnl.sum()+slip),
            'slippage_cost':slip,'fees':fee,'net_pnl':float(parts.net_pnl.sum())})
    pairs=pd.DataFrame(pair_rows,columns=['pair_id','symbol','date','direction','clock','original_shares',
        'first_reference_notional','fragments','reference_gross_pnl','slippage_cost','fees','net_pnl'])
    assert len(pairs)==metrics['t_pair_count']
    complete_stats=pair_statistics(pairs)
    if len(pairs):
        for (direction,clock),part in pairs.groupby(['direction','clock']):
            append('complete_pairs_by_direction_clock',{**identity,'direction':direction,'signal_clock':clock,**pair_statistics(part)})
        for direction,part in pairs.groupby('direction'):
            append('complete_pairs_by_direction',{**identity,'direction':direction,**pair_statistics(part)})
    selling=events.loc[events.event_type.eq('t')&events.side.eq('sell')].copy()
    opens=audit[['date','symbol','opening_shares']]
    selling=selling.merge(opens,on=['date','symbol'],how='left',validate='many_to_one')
    selling['cumulative_sold']=selling.groupby(['date','symbol'],sort=False).shares.cumsum()
    consumed=selling.loc[selling.opening_shares.gt(0)&selling.cumulative_sold.ge(selling.opening_shares)]
    exhausted=consumed.groupby(['date','symbol']).datetime.min().rename('exhausted_at').reset_index()
    exhaustion_before_1031=int(exhausted.exhausted_at.dt.strftime('%H:%M').le('10:31').sum())
    old_days=int(audit.opening_shares.ge(100).sum())
    blocked_old=0;no_opening=0;later_blocked=0;cash_blocked=0;pending_blocked=0
    if len(blocked):
        attempts=blocked.loc[blocked.account.eq('actual')].copy()
        for (kind,reason),part in attempts.groupby(['event_type','reason']):
            append('blocked_reason_counts',{**identity,'event_type':kind,'reason':reason,
                'attempt_rows':len(part),'stock_days':len(part[['date','symbol']].drop_duplicates()),
                'symbols':int(part.symbol.nunique()),'requested_shares':float(part.requested_shares.sum()),
                'filled_shares':float(part.filled_shares.sum())})
        entries=attempts.loc[attempts.event_type.eq('t_entry')].copy()
        old=entries.loc[entries.reason.eq('no_opening_t1_inventory')].merge(opens,on=['date','symbol'],how='left',validate='many_to_one')
        old=old.merge(exhausted,on=['date','symbol'],how='left',validate='many_to_one')
        depleted=old.opening_shares.gt(0)&old.exhausted_at.notna()&old.datetime.ge(old.exhausted_at)
        blocked_old=int(depleted.sum());no_opening=int(old.opening_shares.fillna(0).eq(0).sum())
        later_blocked=int((depleted&old.datetime.dt.strftime('%H:%M').ge('13:00')).sum())
        for clock,part in old.loc[depleted].groupby('clock'):
            append('depleted_old_stock_blocks_by_clock',{**identity,'execution_clock':clock,'attempt_rows':len(part),
                'stock_days':len(part[['date','symbol']].drop_duplicates())})
        cash_blocked=int(entries.reason.isin(['cash_or_repurchase_reserve','pending_t_notional_limit']).sum())
        pending_blocked=int(entries.reason.eq('pending_t_recovery').sum())
    # 未恢复/股数失败按股票日统计，不将连续重复恢复尝试当独立失败机会。
    flags=audit.recovery_failure if audit.recovery_failure.dtype==bool else audit.recovery_failure.map({'True':True,'False':False})
    failed=audit.loc[flags]
    for symbol,part in failed.groupby('symbol'):
        append('recovery_failure_stocks',{**identity,'symbol':symbol,'failure_days':int(part.date.nunique()),
            'first_failure_date':str(part.date.min()),'last_failure_date':str(part.date.max()),
            'max_abs_baseline_share_gap':float(part.baseline_inventory_difference.abs().max())})
    append('candidate_summary',{**identity,**complete_stats,
        'marked_incremental_net_pnl':metrics['incremental_net_pnl'],
        'all_reference_acquisition_cost':metrics['all_reference_acquisition_cost'],
        'holding_cost_reduction_rate':metrics['holding_cost_reduction_rate'],
        **{key:metrics[key] for key in FAILURES},
        'execution_eligible':metrics['t_pair_count']>0 and all(metrics[k]==0 for k in FAILURES),
        'request_count':funnel['request_rows'],'active_days':metrics['active_days'],
        'old_stock_opportunity_stock_days':old_days,'old_stock_exhausted_stock_days':len(exhausted),
        'exhausted_by_1031_stock_days':exhaustion_before_1031,
        'entry_blocks_due_to_consumed_old_stock':blocked_old,'afternoon_entry_blocks_after_depletion':later_blocked,
        'entry_blocks_without_opening_shares':no_opening,'entry_cash_or_pending_limit_blocks':cash_blocked,
        'entry_pending_recovery_blocks':pending_blocked})


def main():
    """主进程完整终态后输出终版，不将运行中的部分折当完整实验。"""
    start=time.monotonic()
    if not (SOURCE/'runtime_integrity_audit.json').exists():
        raise RuntimeError('主进程尚未terminal成功，不写终版诊断')
    integrity=read_json(SOURCE/'runtime_integrity_audit.json');assert integrity['all_unchanged']
    cfg=read_json(SOURCE/'resolved_config.json')
    protocol={'scope':'仅v12完整内部validation_candidates；不读取任何外层预测/账户',
        'groups':'逐候选保留family/threshold/horizon/fraction/direction/signal clock身份；候选账户相互重叠不可加成组合收益',
        'old_stock_exhaustion':'只用日初旧股与当刻之前实际累计卖出定位耗尽；不把当日新底仓不可卖误称额度消耗',
        'recovery_failure_denominator':'股票日/具体股票；重复恢复拒单另列attempt_rows',
        'model_training_or_rule_changes':False}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    for year in cfg['test_years']:
        folder=SOURCE/f'year_{year}';split=read_json(folder/'split.json')
        selected=read_json(folder/'selected_rule.json')['rule']['candidate_index']
        metrics=csv(folder/'validation_rule_metrics.csv');assert len(metrics)==55
        for index in range(55):
            candidate=folder/f'validation_candidates/candidate_{index:03d}'
            rule=read_json(candidate/'rule.json')
            one_candidate(year,split['inner_year'],candidate,rule,selected,cfg)
        print(f'内部诊断已完成：fold={year}',flush=True)
    for name,rows in TABLES.items():pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    changed=[path for path,digest in INPUTS.items() if sha(path)!=digest]
    assert not changed,changed
    result={'complete':True,'scope':protocol,'source_report':str(SOURCE),'inputs_sha256':INPUTS,
        'input_files_unchanged':True,'script_sha256':sha(__file__),
        'tables':{k:len(v) for k,v in TABLES.items()},'elapsed_seconds':time.monotonic()-start,
        'generated_utc':datetime.now(timezone.utc).isoformat()}
    (OUT/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ['complete','tables','elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
