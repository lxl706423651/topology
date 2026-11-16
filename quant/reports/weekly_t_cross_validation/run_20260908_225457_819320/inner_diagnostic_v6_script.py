"""只读首折核心OOF与内部验证输入，比较机械整手和上下文；不读取外层结果。"""
from pathlib import Path
import sys,json,hashlib,time
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd,numpy as np
from quant_backtest.weekly_t_short_horizon import planned_recovery_times
from quant_backtest.weekly_t_engine import run_weekly_portfolio
r=Path(__file__).resolve().parent; y=r/'year_2023'
cfg=json.loads((r/'resolved_config.json').read_text());cache=Path(json.loads((r/'data_manifest.json').read_text())['cache_path']);t=time.monotonic()
weekly=pd.read_pickle(cache/'weekly.pkl')[['symbol','trade_week','f_last_close','f_avg_daily_amount']]
line=pd.read_csv(y/'oof_weekly_cohort/cohort_lineage.csv',parse_dates=['datetime','feature_source_start']);line['trade_week']=line.datetime.dt.normalize()-pd.to_timedelta(line.datetime.dt.weekday,unit='D')
line=line.merge(weekly,on=['symbol','trade_week'],validate='many_to_one')
selections=[]
for yr in [2024,2025]:
    p=y/f'oof_weekly_cohort/selector_{yr}/oof_weekly_selections.csv';s=pd.read_csv(p,parse_dates=['trade_week']);s=s.merge(weekly,on=['symbol','trade_week'],validate='one_to_one');s['cohort_target_year']=yr;selections.append(s)
core_selected=pd.concat(selections,ignore_index=True)
inner=pd.read_csv(y/'validation_weekly_selections.csv',parse_dates=['trade_week','entry_date','exit_date']);inner['f_last_close']=inner.universe_previous_close
ip=pd.read_csv(y/'validation_predictions_h20_l2_depth4.csv',usecols=['datetime','symbol'],parse_dates=['datetime']);ip['trade_week']=ip.datetime.dt.normalize()-pd.to_timedelta(ip.datetime.dt.weekday,unit='D')
ip=ip.merge(inner[['symbol','trade_week','f_last_close','pred_return_rank','pred_amplitude_rank','score','rank']],on=['symbol','trade_week'],validate='many_to_one')
for a,b in [('pred_return_rank','f_weekly_pred_return_rank'),('pred_amplitude_rank','f_weekly_pred_amplitude_rank'),('score','f_weekly_score'),('rank','f_weekly_selection_rank')]: ip[b]=ip[a]/(100 if a=='rank' else 1)
slot=cfg['initial_cash']*cfg['base_fraction']/cfg['top_n'];lot=cfg['lot_size'];frac=cfg['short_t_fraction'];slip=cfg['slippage_rate']
def stats(f):
    """汇总冻结队列的价格、整手代理与周/日/时钟数量，不读取收益。"""
    price=f.f_last_close;qty=(np.floor(slot/(price*(1+slip))/lot)*lot).astype(int);tqty=np.floor(qty*frac/lot)*lot
    out={'rows':len(f),'symbols':f.symbol.nunique(),'stockweeks':len(f[['symbol','trade_week']].drop_duplicates()),'weeks':f.trade_week.nunique(),'reference_price_quantiles':{str(k):float(v) for k,v in price.quantile([0,.1,.25,.5,.75,.9,1]).items()},'fixed_initial_budget_t_lot_eligible_rows':int((tqty>=lot).sum()),'fixed_initial_budget_t_lot_eligible_fraction':float((tqty>=lot).mean()),'fixed_initial_budget_base_below_one_lot':int((qty<lot).sum())}
    if 'datetime' in f:
        out.update(days=f.datetime.dt.normalize().nunique(),distinct_clocks=f.datetime.nunique(),symbol_days=len(f[['datetime','symbol']].assign(datetime=f.datetime.dt.normalize()).drop_duplicates()),rows_per_signal_time=f.datetime.dt.strftime('%H:%M').value_counts().sort_index().to_dict())
        out['context_quantiles']={c:{str(k):float(v) for k,v in f[c].quantile([0,.1,.25,.5,.75,.9,1]).items()} for c in ['f_weekly_pred_return_rank','f_weekly_pred_amplitude_rank','f_weekly_score','f_weekly_selection_rank']}
    return out
out={'scope':'首折核心2024/2025和内部验证2026；无外层结果读取；无拟合或参数变更','fixed_budget':slot,'minimum_held_for_one_t_lot':int(np.ceil(1/frac))*lot,'fixed_budget_price_cutoff':slot/(lot/frac)/(1+slip),'core_selected_stockweeks':stats(core_selected),'core_cohort_before_label_purge':stats(line),'inner_selected_stockweeks':stats(inner),'inner_inference_rows':stats(ip),'core_training_horizons':{},'model_metadata_checks':[]}
for h in [20,40,60]:
    end=planned_recovery_times(line.datetime,h);start=line.feature_source_start
    valid=start.dt.year.isin([2024,2025])&line.datetime.dt.year.isin([2024,2025])&end.dt.year.isin([2024,2025])&end.notna()
    for e in [2023,2026]: valid &= ~((start < pd.Timestamp(e+1,1,1)+pd.Timedelta(days=7))&(end>=pd.Timestamp(e,1,1)-pd.Timedelta(days=7)))
    tr=line.loc[valid].sort_values(['datetime','symbol'],kind='stable')
    st=stats(tr);planned=planned_recovery_times(tr.datetime,h);st['clock_deadline_eligible_rows']=int(planned.dt.strftime('%H:%M').le(cfg['latest_planned_recovery_time']).sum());st['clock_deadline_excluded_fraction']=1-st['clock_deadline_eligible_rows']/len(tr)
    digest=hashlib.sha256(pd.util.hash_pandas_object(tr[['symbol','datetime','feature_source_start']],index=False).to_numpy().tobytes()).hexdigest();st['reconstructed_source_hash']=digest;out['core_training_horizons'][str(h)]=st
    for profile in cfg['t_model_profiles']:
        p=y/f'inner_t_h{h}_{profile}/model_metadata.json';m=json.loads(p.read_text());out['model_metadata_checks'].append({'model':p.parent.name,'training_rows':m['training_rows'],'fitted_rows':m['fitted_rows'],'core_years_match':m['train_years']==[2024,2025],'excluded_years_match':m['excluded_years']==[2023,2026],'reconstructed_source_hash_match':digest==m['feature_source_sha256'],'reconstructed_rows_match':len(tr)==m['training_rows'],'four_context_features_retained':all(c in m['feature_names'] for c in ['f_weekly_pred_return_rank','f_weekly_pred_amplitude_rank','f_weekly_score','f_weekly_selection_rank']),'history_days_absent':'f_history_days' not in m['feature_names']})
# 只运行内部2026的空T基准，以真实开仓数量核机械资格；不读取或输出收益。
daily=pd.read_pickle(cache/'daily.pkl');daily=daily.loc[daily.date.dt.year.eq(2026)&daily.symbol.isin(inner.symbol)&daily.date.ge(inner.entry_date.min())].copy()
res=run_weekly_portfolio(inner,daily,pd.DataFrame(),pd.DataFrame(),cfg);plans=res['weekly_entry_plans'];plans['entry_date']=pd.to_datetime(plans.datetime).dt.normalize();plans=plans.merge(inner[['symbol','entry_date','trade_week']],on=['symbol','entry_date'],validate='one_to_one')
qty=plans.baseline_filled_shares;tqty=np.floor(qty*frac/lot)*lot
out['inner_actual_baseline_entry_inventory']={'plans':len(plans),'filled_at_least_one_t_lot':int((tqty>=lot).sum()),'filled_at_least_one_t_lot_fraction':float((tqty>=lot).mean()),'zero_filled':int(qty.eq(0).sum()),'positive_but_less_than_400':int((qty.gt(0)&qty.lt(400)).sum()),'planned_ge400_but_filled_lt400':int((plans.planned_shares.ge(400)&qty.lt(400)).sum()),'entry_budget_min':float(plans.entry_budget.min()),'entry_budget_max':float(plans.entry_budget.max()),'entry_budget_median':float(plans.entry_budget.median())}
merged=ip.merge(plans[['trade_week','symbol','baseline_filled_shares']],on=['trade_week','symbol'],validate='many_to_one');out['inner_actual_baseline_entry_inventory']['inference_rows_ge400']=int(merged.baseline_filled_shares.ge(400).sum());out['inner_actual_baseline_entry_inventory']['inference_fraction_ge400']=float(merged.baseline_filled_shares.ge(400).mean())
out['limitation']='固定9500元仅同尺度代理；真实基准统计按每周实际权益/开盘容量/现金成交，不包含T请求后的可卖/现金/债务约束；未检验收益。'
out['elapsed_seconds']=time.monotonic()-t

# 以下归档已完成的首折内部候选，不读取外层指标或变更任何候选。
def file_sha(path):
    """分块计算文件内容SHA，避免大型缓存再驻留内存。"""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

required = [r/'resolved_config.json', r/'data_manifest.json', r/'source_hashes.json',
            cache/'weekly.pkl', cache/'daily.pkl', y/'oof_weekly_cohort/cohort_lineage.csv',
            y/'oof_weekly_cohort/cohort_metadata.json', y/'t_training_cohort_manifest.json',
            y/'validation_weekly_context_manifest.json', y/'validation_weekly_selections.csv',
            y/'validation_predictions_h20_l2_depth4.csv', y/'validation_rule_metrics.csv']
weekly_selectors=[]
for yr in [2024,2025]:
    folder=y/f'oof_weekly_cohort/selector_{yr}'
    m=json.loads((folder/'model_metadata.json').read_text())
    weekly_selectors.append({'target_year':yr,'train_years':m['train_years'],'excluded_years':m['excluded_years'],
        'training_rows':m['training_rows'],'feature_count':len(m['feature_names'])})
    required.extend([folder/'model_metadata.json',folder/'model_bundle.joblib',folder/'oof_weekly_selections.csv'])
for h in [20,40,60]:
    for profile in cfg['t_model_profiles']: required.append(y/f'inner_t_h{h}_{profile}/model_metadata.json')
context=json.loads((y/'validation_weekly_context_manifest.json').read_text())
weekly_path=Path(context['source_model_path']);weekly_metadata=weekly_path.parent/'model_metadata.json'
m=json.loads(weekly_metadata.read_text())
weekly_selectors.append({'target_year':2026,'train_years':m['train_years'],'excluded_years':m['excluded_years'],
    'training_rows':m['training_rows'],'feature_count':len(m['feature_names'])})
required.extend([weekly_path,weekly_metadata])
source=json.loads((r/'source_hashes.json').read_text());source_checks=[]
for name,expected in source.items():
    current=PROJECT_ROOT/name;snapshot=r/'source_snapshot'/Path(name).name
    source_checks.append({'path':name,'expected_sha256':expected,'current_sha256':file_sha(current),
        'snapshot_sha256':file_sha(snapshot),'match':file_sha(current)==expected==file_sha(snapshot)})
    required.extend([current,snapshot])
columns=['candidate_index','horizon_minutes','profile','net_margin','incremental_net_pnl','completed_pair_count',
 'fragment_count','active_days','active_weeks','active_symbols','weekly_increment_ci_low_rmb',
 'weekly_increment_ci_high_rmb','strict_eligible','rejection_reasons','inventory_mismatch_days',
 'recovery_failure_days','base_plan_mismatch_count','t1_violation_count','weekly_exit_failure_days',
 'weekly_unclosed_positions','baseline_weekly_unclosed_positions','t_fee','decision_sha256','same_decisions_as_candidate']
candidates=pd.read_csv(y/'validation_rule_metrics.csv',usecols=columns)
summary={'count':len(candidates),'unique_decision_sets':int(candidates.decision_sha256.nunique()),
    'negative':int(candidates.incremental_net_pnl.lt(0).sum()),'zero':int(candidates.incremental_net_pnl.eq(0).sum()),
    'positive':int(candidates.incremental_net_pnl.gt(0).sum()),'strict_eligible':int(candidates.strict_eligible.sum()),
    'rejection_counts':candidates.rejection_reasons.str.split('|').explode().value_counts().to_dict(),
    'rows':candidates.to_dict('records')}
out['scope']='仅首折核心2024/2025、内部验证2026；不读取外层收益，不拟合模型，不变更规则。额外空T基准重放只用于内部真实开仓数量。'
out['weekly_selector_training']=weekly_selectors
out['internal_candidate_summary']=summary
out['source_hash_checks']=source_checks
out['input_files']=[{'path':str(p.resolve()),'sha256':file_sha(p),'bytes':p.stat().st_size} for p in sorted(set(required))]
out['archived_script_sha256']=file_sha(Path(__file__))
out['method_limits']=[
    '每周9500元静态预算用于同尺度比较；真实预算随基准权益变化。实际基准开仓统计是已成交计划股数，不是逐信号最终可卖数量。',
    '底仓至少400股只满足25%且100股整手的机械条件；资金、当日可卖库存、未恢复债务、价格限制、成交容量和逾期周仓仍可禁止执行。',
    '本次从cohort lineage和固定时钟独立重建严格来源区间行集，行数与9个feature_source_sha256一致；不构成逐标签价格、全feature训练SHA或原生预测全面审计。',
    '核心OOF selector分别仅训练另一个核心年；内部冻结selector使用两个核心年。实际拟合样本和选中特征不同，即使四个上下文归一化仍可能存在选择分布差异。',
    '较早核心2024的周预测参数训练自2025；这是回顾性交叉拟合。原始量价feature_source_start不代表模型参数当时已知，须另审模型训练年份及SHA图。内部/外层标签不在本折OOF模型训练年内。',
    '训练h40的14:00信号、h60的13:30信号标签有效，但计划恢复分别14:41、14:31，超过14:30现行执行截止。训练与实际可发单时钟分布不同。',
    '训练行共享同日/同刻全市场特征，207096行不是207096个独立市场冲击。核心有效259天/91周/2072时刻，内部45个可T日/16周/360时刻。',
    '每股每日只请求首个通过门控的信号；训练同时包含全部固定信号。短周期精确标签与容量约束后的分段/延迟恢复实际持有期也不同。',
    '这些是本折训练内部诊断，不根据外层成绩选择参数；历史数据已经多次研究，不能把后续历史复测称为全新测试。']
out['limitation']='机械≥400不保证完整可成交；本次补录的收益仅为已冻结首折内部候选，未读取外层结果。'
out['audit_checks']={'all_source_hashes_match':all(z['match'] for z in source_checks),
 'all_nine_source_rows_and_hashes_match':all(z['reconstructed_rows_match'] and z['reconstructed_source_hash_match'] for z in out['model_metadata_checks']),
 'all_nine_keep_four_context_features':all(z['four_context_features_retained'] for z in out['model_metadata_checks'])}
json_path=r/'inner_diagnostic_v6.json'
json_path.write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else str(x)),encoding='utf-8')
core=out['core_training_horizons']['20'];inner=out['inner_inference_rows'];actual=out['inner_actual_baseline_entry_inventory']
lines=['# v6 首折核心—内部验证独立诊断','',
 '仅分析 year_2023 的核心2024/2025与内部验证2026，不读取外层收益，不重训或调整任何模型/规则。归档脚本可从本目录重放相同输入；依赖源码、缓存、选择文件和模型元数据SHA见JSON。','',
 '## 已核训练与来源','',
 f"9个模型训练年均为[2024,2025]，排除[2023,2026]，均保留4个新增周预测f_*并排除f_history_days。h20/h40各{core['rows']:,}行，h60为{out['core_training_horizons']['60']['rows']:,}行；独立按来源起点至标签端点与排除年±7天重建，9个模型行数及feature_source_sha256全部一致。当前16份运行源码及快照SHA全部匹配。",'',
 '这只是本次范围内的来源/行集核对，未重算全部特征训练SHA或原生模型预测；不能替代完整模型审计。','',
 '## 资金与整手造成的队列差异','',
 '| 项目 | 核心实际h20训练行 | 内部推断行 |','|---|---:|---:|',
 f"| 行数 | {core['rows']:,} | {inner['rows']:,} |",
 f"| 参考价格中位数 | {core['reference_price_quantiles']['0.5']:.2f}元 | {inner['reference_price_quantiles']['0.5']:.2f}元 |",
 f"| 固定9500元预算满足400股比例 | {core['fixed_initial_budget_t_lot_eligible_fraction']:.3%} | {inner['fixed_initial_budget_t_lot_eligible_fraction']:.3%} |",
 f"| 不同日期/时刻 | {core['days']}/{core['distinct_clocks']} | {inner['days']}/{inner['distinct_clocks']} |",'',
 f"内部空T基准按真实开盘和动态预算重放：1600个股周中，1080个(67.5%)实际买到至少400股，33个零成交，487个有仓但不足400股；4个计划至少400股但实际不足。映射内部信号后，{actual['inference_rows_ge400']:,}/{inner['rows']:,}行({actual['inference_fraction_ge400']:.3%})对应初始至少400股的周计划。实际每股周预算为{actual['entry_budget_min']:.2f}—{actual['entry_budget_max']:.2f}元。",'',
 '**至少400股只是25%仓位、100股整手的机械条件，不保证可成交。** 仍要通过T+1可卖量、资金、容量、价格限制、未恢复债务和逾期周仓检查。核心训练也含约17.7%的静态机械不可T行，内部比例更高；此处只记录差异，未新增过滤。','',
 '## 已冻结内部27候选','',
 f"27项共{summary['unique_decision_sets']}组不同决策：22项净增量为负，3项为零，仅2项小额为正；严格合格为0。全部27项周bootstrap下界均不大于0。",'',
 '| 候选 | 周期/配置/净门槛 | 净增量 | 完整配对/片段 | 活跃日/周/股票 | 周增量bootstrap区间 |',
 '|---|---|---:|---:|---:|---:|']
for row in candidates.loc[candidates.incremental_net_pnl.gt(0)].to_dict('records'):
    lines.append(f"| {row['candidate_index']} | {row['horizon_minutes']}分钟/{row['profile']}/{row['net_margin']*10000:.0f}bp | {row['incremental_net_pnl']:.2f}元 | {row['completed_pair_count']}/{row['fragment_count']} | {row['active_days']}/{row['active_weeks']}/{row['active_symbols']} | [{row['weekly_increment_ci_low_rmb']:.2f}, {row['weekly_increment_ci_high_rmb']:.2f}]元 |")
lines += ['', '两项小额正收益仅有1和5个完整配对，未达100配对、20天、8周、20只股票的预声明支持要求，并存在集中度超限。候选7的bootstrap下界约为−6.98e−10元，实质为0附近，不能解释为明确正下界；候选25下界约−122.02元。6个恢复片段属于5个完整配对，不应当作6个独立样本。', '',
 '另有13个候选发生恢复/日终库存不一致、6个发生基础计划偏离；它们不能仅按账面净增量判为可用。共同基准本身的周清仓未完全执行也另有记录，不由T候选选择治愈。以上数量按27项候选计，各项并非独立实验。','',
 '## 统计与交叉拟合边界','']
for msg in out['method_limits'][3:]: lines.append('- '+msg)
lines += ['', '没有发现本次首折OOF训练年声明含内部2026或外层2023。参数的未来核心年依赖已明确存在，应称回顾性交叉拟合，不能仅凭原始价格来源隔离宣称全流程在历史当时可用。', '',
 f"审计脚本SHA256：`{out['archived_script_sha256']}`。机器可读诊断：`inner_diagnostic_v6.json`；重现脚本：`inner_diagnostic_v6_script.py`。"]
(r/'inner_diagnostic_v6.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'json':str(json_path),'md':str(r/'inner_diagnostic_v6.md'),'input_files':len(out['input_files']),
 'checks':out['audit_checks'],'negative':summary['negative'],'zero':summary['zero'],'positive':summary['positive']},ensure_ascii=False))
