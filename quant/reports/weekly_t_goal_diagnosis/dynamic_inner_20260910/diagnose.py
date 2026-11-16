"""v13完整内部退出模式成对诊断；只读已完成fold，不选规则、不读取外层账户。

--stage仅读取指标和selected_rule均已落盘的fold，并缓存独立派生结果；--final
还要求主运行终态与全部fold齐全，再核实际读取输入SHA后生成终版。不同候选
账户相互重叠，跨候选统计仅描述实验分布，绝不当作一只真实组合累计收益。
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
KEY=['symbol','first_datetime','direction','original_shares']
COARSE=KEY[:3]
FAIL=['inventory_mismatch_days','recovery_failure_days','pending_pairs','t1_violation_count','base_plan_mismatch_count']
TABLE_NAMES=['candidate_summary','paired_exit_contrasts','pair_match_categories','unmatched_entry_causes','trigger_realization']


def sha(path):
    """流式核实际读取文件身份。"""
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def numeric(value):
    """JSON保留缺失为null，不把未知收益变为0。"""
    if isinstance(value,dict):return {k:numeric(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [numeric(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    return value


def stats(frame):
    """完整与未完整分开；未完整已实现片段仍保留，费用含真实已发生首腿。"""
    complete=frame.loc[frame.complete] if len(frame) else frame
    norm=complete.first_reference_notional.sum()
    return {'started_groups':len(frame),'complete_groups':len(complete),'incomplete_groups':len(frame)-len(complete),
        'paired_lots':float(complete.original_shares.sum()/100),
        'complete_reference_gross':float(complete.reference_gross.sum()),
        'complete_fees':float(complete.allocated_fees.sum()),'complete_slippage':float(complete.slippage.sum()),
        'complete_net':float(complete.fragment_net.sum()),'all_fragment_net':float(frame.fragment_net.sum()),
        'actual_t_fee':float(frame.actual_fees.sum()),'reference_notional':float(norm),
        'complete_net_bp':float(complete.fragment_net.sum()/norm*1e4) if norm else None,
        'complete_reference_gross_bp':float(complete.reference_gross.sum()/norm*1e4) if norm else None,
        'complete_win_rate':float(complete.fragment_net.gt(0).mean()) if len(complete) else None}


class Diagnosis:
    """仅接收一个已冻结完整内部折，所有读取路径受白名单限制。"""
    def __init__(self,report,year):
        self.report=report.resolve();self.year=year;self.fold=self.report/f'year_{year}'
        self.inputs={};self.tables={key:[] for key in TABLE_NAMES}
    def record(self,path):
        path=Path(path).resolve();rel=path.relative_to(self.report)
        assert (rel.parts[0]==f'year_{self.year}' and
            (len(rel.parts)==2 and rel.name in ['split.json','validation_rule_metrics.csv','selected_rule.json']
             or len(rel.parts)>2 and rel.parts[1]=='validation_candidates')),f'禁止读取非本折内部账户:{rel}'
        self.inputs.setdefault(str(path),sha(path));return path
    def csv(self,path):
        try:return pd.read_csv(self.record(path),float_precision='round_trip')
        except pd.errors.EmptyDataError:return pd.DataFrame()
    def json(self,path):return json.loads(self.record(path).read_text())
    def add(self,name,value):self.tables[name].append(numeric(value))
    def candidate(self,index):
        """从真实首腿/恢复片段汇总全部组，完整状态由恢复股数确定，不筛收益。"""
        path=self.fold/f'validation_candidates/candidate_{index:03d}'
        rule=self.json(path/'rule.json');metrics=self.json(path/'metrics.json')
        events=self.csv(path/'test_trade_events.csv');trips=self.csv(path/'test_round_trips.csv')
        intents=self.csv(path/'exit_decisions.csv');blocked=self.csv(path/'blocked_orders.csv')
        events['datetime']=pd.to_datetime(events.datetime,format='mixed')
        first=events.loc[events.event_type.eq('t')&events.fragment_id.eq(0)].copy()
        one=first.rename(columns={'datetime':'first_datetime','shares':'original_shares'})
        one['direction']=np.where(one.side.eq('buy'),'buy_first','sell_first')
        one['first_reference_notional']=one.original_shares*one.reference_price
        if len(trips):
            trips['second_datetime']=pd.to_datetime(trips.second_datetime,format='mixed')
            trips['allocated_fees']=trips.buy_fee+trips.sell_fee
            grouped=trips.groupby('pair_id').agg(recovered_shares=('shares','sum'),fragments=('shares','size'),
                fragment_net=('net_pnl','sum'),slipped_gross=('gross_pnl','sum'),slippage=('slippage_cost','sum'),
                allocated_fees=('allocated_fees','sum'),last_recovery=('second_datetime','max'))
        else:grouped=pd.DataFrame(columns=['recovered_shares','fragments','fragment_net','slipped_gross','slippage','allocated_fees','last_recovery'])
        result=one[['pair_id',*KEY,'reference_price','price','fee','first_reference_notional']].merge(grouped,left_on='pair_id',right_index=True,how='left',validate='one_to_one')
        for col in ['recovered_shares','fragments','fragment_net','slipped_gross','slippage','allocated_fees']:result[col]=result[col].fillna(0.)
        result['complete']=result.recovered_shares.eq(result.original_shares)
        result['reference_gross']=result.slipped_gross+result.slippage
        fees=events.loc[events.event_type.eq('t')].groupby('pair_id').fee.sum()
        result['actual_fees']=result.pair_id.map(fees).fillna(0.)
        if len(intents):
            assert not intents.pair_id.duplicated().any()
            for col in ['exit_signal_time','exit_earliest_time']:intents[col]=pd.to_datetime(intents[col],format='mixed')
        result=result.merge(intents[['pair_id','exit_reason','exit_signal_time','exit_earliest_time','estimated_net_pnl','estimated_net_return']],on='pair_id',how='left',validate='one_to_one')
        assert not result.duplicated(KEY).any() and not result.duplicated(COARSE).any()
        result['first_clock']=result.first_datetime.dt.strftime('%H:%M')
        summary=stats(result)
        assert summary['complete_groups']==metrics['t_pair_count']
        assert abs(summary['complete_net']-metrics['realized_complete_pair_pnl'])<1e-5
        identity={'fold':self.year,'inner_year':self.inner,**rule,'selected':index==self.selected}
        self.add('candidate_summary',{**identity,**summary,**{k:metrics[k] for k in FAIL},
            'incremental_marked_net':metrics['incremental_net_pnl'],
            'holding_cost_reduction_rate':metrics['holding_cost_reduction_rate'],
            'original_baseline_cost':metrics['all_reference_acquisition_cost'],
            'execution_eligible':index>0 and summary['complete_groups']>0 and all(metrics[k]==0 for k in FAIL)})
        for trigger,part in result.groupby('exit_reason',dropna=False):
            valid=part.loc[part.complete&np.isfinite(part.estimated_net_pnl)]
            norm=valid.first_reference_notional.sum()
            actual_net=valid.fragment_net.sum();estimated_net=valid.estimated_net_pnl.sum()
            self.add('trigger_realization',{**identity,'trigger':trigger,**stats(part),
                'complete_with_finite_estimate':len(valid),
                'missing_estimate_groups':int(part.estimated_net_pnl.isna().sum()),
                'estimated_net_sum_same_complete_groups':float(estimated_net),
                'actual_net_sum_same_complete_groups':float(actual_net),
                'actual_minus_estimated_net':float(actual_net-estimated_net),
                'estimated_net_bp':float(estimated_net/norm*1e4) if norm else None,
                'actual_net_bp_same_groups':float(actual_net/norm*1e4) if norm else None,
                'realized_loss_despite_positive_estimate':int((valid.estimated_net_pnl.gt(0)&valid.fragment_net.lt(0)).sum()),
                'multiple_recovery_fragments':int(part.fragments.gt(1).sum()),
                'completed_later_than_earliest':int((part.complete&part.last_recovery.gt(part.exit_earliest_time)).sum())})
        if len(blocked):
            blocked=blocked.loc[blocked.event_type.eq('t_entry')&blocked.account.eq('actual')].copy()
            blocked['first_datetime']=pd.to_datetime(blocked.datetime,format='mixed')
            blocked['direction']=np.where(blocked.side.eq('buy'),'buy_first','sell_first')
        return rule,metrics,result,blocked
    def contrast(self,entry,left,right):
        """保留共同/单边/未完整所有组，并以另一账户同刻真实拒单解释新增或缺少组。"""
        lr,lm,l,lb=left;rr,rm,r,rb=right
        merged=l.merge(r,on=KEY,how='outer',suffixes=('_fixed','_dynamic'),indicator=True,validate='one_to_one')
        identity={'fold':self.year,'inner_year':self.inner,'entry_candidate_index':entry,
            'fixed_index':lr['candidate_index'],'dynamic_index':rr['candidate_index'],
            'family':lr['family'],'entry_deviation':lr['entry_deviation'],'recovery_horizon':lr['recovery_horizon'],
            'fraction':lr['fraction'],'dynamic_mode':rr['exit_mode']}
        both=merged._merge.eq('both')
        complete_l=merged.complete_fixed.eq(True);complete_r=merged.complete_dynamic.eq(True)
        category=np.where(~both,np.where(merged._merge.eq('left_only'),'only_fixed','only_dynamic'),
            np.where(complete_l&complete_r,'both_complete',np.where(complete_l,'fixed_complete_dynamic_incomplete',
            np.where(complete_r,'fixed_incomplete_dynamic_complete','both_incomplete'))))
        merged['category']=category
        for kind,part in merged.groupby('category'):
            value={**identity,'category':kind,'keys':len(part)}
            for side in ['fixed','dynamic']:
                present=part['pair_id_'+side].notna();complete=part['complete_'+side].eq(True)
                value.update({side+'_started_groups':int(present.sum()),side+'_complete_groups':int(complete.sum()),
                    side+'_complete_net':float(part.loc[complete,'fragment_net_'+side].sum()),
                    side+'_all_fragment_net':float(part.loc[present,'fragment_net_'+side].sum()),
                    side+'_paired_lots':float(part.loc[complete,'original_shares'].sum()/100),
                    side+'_complete_gross':float(part.loc[complete,'reference_gross_'+side].sum()),
                    side+'_complete_fees':float(part.loc[complete,'allocated_fees_'+side].sum()),
                    side+'_complete_slippage':float(part.loc[complete,'slippage_'+side].sum())})
            value['complete_net_delta']=value['dynamic_complete_net']-value['fixed_complete_net']
            self.add('pair_match_categories',value)
        shared=merged.loc[category=='both_complete']
        if len(shared):
            assert np.allclose(shared.reference_price_fixed,shared.reference_price_dynamic,rtol=0,atol=1e-12)
            assert np.allclose(shared.fee_fixed,shared.fee_dynamic,rtol=0,atol=1e-9)
        common_delta=float((shared.fragment_net_dynamic-shared.fragment_net_fixed).sum())
        total_delta=float(r.loc[r.complete,'fragment_net'].sum()-l.loc[l.complete,'fragment_net'].sum())
        self.add('paired_exit_contrasts',{**identity,'common_exact_first_legs':int(both.sum()),
            'common_both_complete':len(shared),'only_fixed_first_legs':int(merged._merge.eq('left_only').sum()),
            'only_dynamic_first_legs':int(merged._merge.eq('right_only').sum()),
            'same_key_incomplete_at_least_one':int((both&~(complete_l&complete_r)).sum()),
            'common_complete_net_delta':common_delta,'remaining_complete_net_delta':total_delta-common_delta,
            'all_complete_net_delta':total_delta,
            'incremental_marked_net_delta':rm['incremental_net_pnl']-lm['incremental_net_pnl'],
            'complete_group_delta':rm['t_pair_count']-lm['t_pair_count'],
            'paired_lot_delta':rm['completed_t_pair_lots']-lm['completed_t_pair_lots'],
            **{k+'_delta':rm[k]-lm[k] for k in FAIL},
            'fixed_marked_net':lm['incremental_net_pnl'],'dynamic_marked_net':rm['incremental_net_pnl'],
            'common_net_improved_groups':int(shared.fragment_net_dynamic.gt(shared.fragment_net_fixed).sum()),
            'common_net_worsened_groups':int(shared.fragment_net_dynamic.lt(shared.fragment_net_fixed).sum())})
        for observed,absent,part,other,blocks in [('fixed','dynamic',merged.loc[merged._merge.eq('left_only')],r,rb),
                                                ('dynamic','fixed',merged.loc[merged._merge.eq('right_only')],l,lb)]:
            causes=[]
            other_keys={tuple(row[k] for k in COARSE):row['original_shares'] for row in other.to_dict('records')}
            block_keys={}
            if len(blocks):
                for key,rows in blocks.groupby(COARSE):block_keys[key]='|'.join(sorted(set(rows.reason.astype(str))))
            for row in part.to_dict('records'):
                key=tuple(row[k] for k in COARSE)
                cause='same_signal_different_filled_quantity' if key in other_keys else block_keys.get(key,'no_recorded_entry_rejection')
                causes.append({'cause':cause,'complete':row['complete_'+observed],
                    'net':row['fragment_net_'+observed],'lots':row['original_shares']/100})
            if causes:
                for cause,items in pd.DataFrame(causes).groupby('cause'):
                    completed=items.loc[items.complete]
                    self.add('unmatched_entry_causes',{**identity,'present_mode':observed,'absent_mode':absent,
                        'other_account_observed_reason':cause,'first_legs':len(items),'complete_groups':len(completed),
                        'complete_net':float(completed.net.sum()),'paired_lots':float(completed.lots.sum()),
                        'interpretation':'另一真实账户同刻记录；不是用事后收益反推的原因'})
    def run(self):
        """每次只驻留一个入场的三个账户；相同入场的所有退出模式全部比较。"""
        assert (self.fold/'validation_rule_metrics.csv').exists() and (self.fold/'selected_rule.json').exists()
        split=self.json(self.fold/'split.json');self.inner=split['inner_year']
        selected=self.json(self.fold/'selected_rule.json');self.selected=selected['rule']['candidate_index']
        table=self.csv(self.fold/'validation_rule_metrics.csv')
        assert len(table)==163 and set(table.candidate_index)==set(range(163))
        assert self.year not in split['train_years'] and self.inner not in split['core_years']
        self.candidate(0)
        for entry in range(1,55):
            items=[self.candidate(1+(entry-1)*3+offset) for offset in range(3)]
            assert [v[0]['exit_mode'] for v in items]==['fixed','profit_buffer','profit_buffer_stop']
            for alt in items[1:]:self.contrast(entry,items[0],alt)
            if entry%9==0:print(f'[{self.year}] 内部成对诊断完成入场{entry}/54',flush=True)
        return {'fold':self.year,'inner_year':self.inner,'selected_rule':selected['rule'],
            'source_report':str(self.report),'inputs_sha256':self.inputs,'script_sha256':sha(__file__),
            'no_outer_account_read':True,'no_rule_selected_or_modified':True}


def main():
    """分折缓存只读分析；主进程完成后才生成完整年度终版。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--stage',action='store_true');group.add_argument('--final',action='store_true')
    args=parser.parse_args();report=args.report_dir.resolve()
    cfg=json.loads((report/'resolved_config.json').read_text());years=cfg['test_years']
    ready=[y for y in years if (report/f'year_{y}/validation_rule_metrics.csv').exists() and (report/f'year_{y}/selected_rule.json').exists()]
    if args.final:
        assert ready==years and json.loads((report/'runtime_integrity_audit.json').read_text())['all_unchanged']
    analyses=[]
    for year in ready:
        base=OUT/f'fold_{year}';cache=None
        for p in sorted(base.glob('run_*/analysis.json'),reverse=True):
            old=json.loads(p.read_text())
            if old['source_report']==str(report) and old['script_sha256']==sha(__file__):cache=(p.parent,old);break
        if cache:
            analyses.append(cache);continue
        stage=base/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}';stage.mkdir(parents=True,exist_ok=False)
        d=Diagnosis(report,year);start=time.monotonic();summary=d.run()
        for name,rows in d.tables.items():pd.DataFrame(rows).to_csv(stage/(name+'.csv'),index=False,encoding='utf-8-sig')
        for path,value in d.inputs.items():assert sha(path)==value,path
        summary.update(elapsed_seconds=time.monotonic()-start,passed=True,
                       table_sha256={name+'.csv':sha(stage/(name+'.csv')) for name in TABLE_NAMES})
        (stage/'analysis.json').write_text(json.dumps(numeric(summary),ensure_ascii=False,indent=2,allow_nan=False))
        analyses.append((stage,summary));print(f'[{year}] 分折诊断及输入SHA已保存',flush=True)
    if not args.final:
        print(json.dumps({'stage_complete_folds':ready,'waiting_folds':[y for y in years if y not in ready]},ensure_ascii=False));return
    final=OUT/f'final_{datetime.now():%Y%m%d_%H%M%S_%f}';final.mkdir(exist_ok=False)
    shutil.copy2(__file__,final/'diagnose.py');inputs={};summary=[]
    for _,item in analyses:inputs.update(item['inputs_sha256']);summary.append({k:v for k,v in item.items() if k not in ['inputs_sha256','table_sha256']})
    for path,value in inputs.items():assert sha(path)==value,path
    tables={}
    for name in TABLE_NAMES:
        parts=[]
        for stage,item in analyses:
            path=stage/(name+'.csv');assert sha(path)==item['table_sha256'][path.name]
            try:parts.append(pd.read_csv(path,float_precision='round_trip'))
            except pd.errors.EmptyDataError:pass
        frame=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        frame.to_csv(final/(name+'.csv'),index=False,encoding='utf-8-sig');tables[name]=frame
    (final/'diagnosis.json').write_text(json.dumps(numeric({'passed':True,'source_report':str(report),'folds':summary,
        'read_input_count':len(inputs),'inputs_sha256':inputs,'all_read_inputs_unchanged':True,
        'no_outer_account_read':True,'script_sha256':sha(__file__),
        'comparison_contract':'same entry specification and shared full-minute quote coverage; exact first-leg keys; incomplete and one-sided groups retained',
        'cross_candidate_aggregation_is_not_single_portfolio_pnl':True}),ensure_ascii=False,indent=2,allow_nan=False))
    render(final,tables,summary)
    print(json.dumps({'passed':True,'final_directory':str(final),'input_files':len(inputs)},ensure_ascii=False),flush=True)


def render(final,tables,folds):
    """用固定统计口径描述模式差异，不重新推荐或挑选候选。"""
    candidates=tables['candidate_summary'];contrasts=tables['paired_exit_contrasts'];triggers=tables['trigger_realization']
    lines=['# v13内部动态恢复成对诊断','',
        '仅使用四个完整内部年度账户。相同入场参数的三个退出模式共用完整分钟行情、费用和底仓；不读取外层账户，不改变已选规则。跨候选行是重叠实验，金额不能相加当作真实组合收益。','',
        '| 留出年 | 内部年 | 已选规则/退出 | 内部完整组 | 配对手 | 已选净额（元） | 主动执行合格数 | 主动正/负/零候选 |',
        '|---|---:|---|---:|---:|---:|---:|---|']
    for item in folds:
        y=item['fold'];part=candidates.loc[candidates.fold.eq(y)];active=part.loc[part.candidate_index.gt(0)]
        chosen=part.loc[part.selected.eq(True)].iloc[0]
        lines.append(f'| {y} | {item["inner_year"]} | {int(chosen.candidate_index)}/{chosen.exit_mode} | {int(chosen.complete_groups)} | {chosen.paired_lots:.0f} | {chosen.incremental_marked_net:.2f} | {int(active.execution_eligible.sum())} | {int(active.incremental_marked_net.gt(0).sum())}/{int(active.incremental_marked_net.lt(0).sum())}/{int(active.incremental_marked_net.eq(0).sum())} |')
    lines+=['','| 动态模式相对fixed | 比较数 | 整账户净额提高/降低/相等 | 整账户净额差中位数（元） | 共同完整首腿净额差中位数（元） | 恢复失败日增加/减少比较数 |',
        '|---|---:|---|---:|---:|---|']
    for mode,p in contrasts.groupby('dynamic_mode'):
        d=p.incremental_marked_net_delta
        lines.append(f'| {mode} | {len(p)} | {int(d.gt(1e-8).sum())}/{int(d.lt(-1e-8).sum())}/{int(d.abs().le(1e-8).sum())} | {d.median():.2f} | {p.common_complete_net_delta.median():.2f} | {int(p.recovery_failure_days_delta.gt(0).sum())}/{int(p.recovery_failure_days_delta.lt(0).sum())} |')
    lines+=['','共同首腿严格按股票、实际首腿时刻、方向、原始股数匹配。共同但任一侧未完整、仅fixed存在、仅动态存在都单列；共同完整组的差额只解释其自身退出变化，不能代替整账户结果。另一账户同刻若有不同成交量或现金、旧股、待恢复、容量拒单，单列原记录，不根据之后收益猜测原因。','',
        '触发时预计净额只使用当时已知close和费用，实际净额来自后续真实完成片段。达到止盈观察阈值并不保证下一分钟以同价成交；缺估值、部分恢复、未完整组均保留。详见trigger_realization.csv。','',
        '这里的fixed已经使用新完整分钟恢复覆盖，因此本报告不把相对v12的覆盖差额解释成动态择时收益。2023–2025的留一年协议可用以后年份训练旧周模型；这是历史研究复测，不是全新未接触样本。','',
        '文件：candidate_summary.csv为652个内部账户；paired_exit_contrasts.csv为相同入场的432个成对比较；pair_match_categories.csv保留完整/未完整/单边组；unmatched_entry_causes.csv记录另一账户同刻拒单或数量变化；trigger_realization.csv对比首次锁存预计与实际完成净额；diagnosis.json保存所有实际读取输入SHA。']
    (final/'diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
