"""从quant根目录独立审计已结束的v10 主动T四折报告，不训练、不选参、不修改原结果。

用法：python scripts/audit_weekly_t_active_execution.py --report-dir reports/.../run_...
只在final_report及全部年度结果齐全后读取外层结果。逐笔费用、现金、旧库存、
真实分钟容量、计划恢复时序、名义额度、片段损益及冻结v4基准均独立重放；输出
independent_execution_audit.json/md。不会调用生产validate_candidate_result代替审计。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AUDITED_ENGINE_SHA256='3a451f3cdbfb4a7249b1684b91267c4769b4cb20266c9a0b237661bf9df78ba5'


def file_sha256(path: Path) -> str:
    """对原模型、源报告与新快照逐文件核对字节身份。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_audit_csv(path: Path) -> pd.DataFrame:
    """精确读取已有CSV，空文件保留为空表，不创建候选或补造结果。"""
    try:return pd.read_csv(path,float_precision='round_trip')
    except pd.errors.EmptyDataError:return pd.DataFrame()


def check_close(actual,expected,label):
    """候选独立核账仅允许CSV/浮点累加误差。"""
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-5,err_msg=label)


def verify_saved_frequency_account(folder: Path,cfg: dict,no_t: bool=False) -> dict:
    """独立逐笔重算所有内部候选的现金/费用/旧股预算及片段，合法失败照实返回。"""
    events=read_audit_csv(folder/'test_trade_events.csv');base=read_audit_csv(folder/'baseline_trade_events.csv')
    curve=read_audit_csv(folder/'test_equity_curve.csv');inventory=read_audit_csv(folder/'daily_inventory_audit.csv')
    trips=read_audit_csv(folder/'test_round_trips.csv')
    metrics=(json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').is_file()
             else read_audit_csv(folder/'test_metrics.csv').iloc[0].to_dict())
    assert len(curve)>0
    curve['datetime']=pd.to_datetime(curve.datetime,format='mixed');inventory['date']=pd.to_datetime(inventory.date,format='mixed')
    first=None;max_cash_error=0.
    for account,table,cash_key in [('actual',events,'cash'),('baseline',base,'baseline_cash')]:
        table['datetime']=pd.to_datetime(table.datetime,format='mixed');table['signal_datetime']=pd.to_datetime(table.signal_datetime,format='mixed')
        if table.empty:
            check_close(curve[cash_key],cfg['initial_cash'],'empty account cash');continue
        table['datetime']=pd.to_datetime(table.datetime,format='mixed');table['signal_datetime']=pd.to_datetime(table.signal_datetime,format='mixed')
        assert table.datetime.is_monotonic_increasing and (table.datetime>table.signal_datetime).all()
        assert (table.shares.gt(0)&table.shares.mod(cfg['lot_size']).eq(0)).all()
        assert np.isfinite(table[['price','reference_price','notional','fee','cash_after']]).all().all()
        assert table.price.gt(0).all() and table.reference_price.gt(0).all()
        selling=table.side.eq('sell');assert (selling|table.side.eq('buy')).all()
        expected_fee=np.maximum(table.notional*cfg['commission_rate'],cfg['min_commission'])+table.notional*cfg['transfer_rate']
        stamp=np.where(table.datetime.ge(pd.Timestamp('2023-08-28')),cfg['stamp_rate_from_20230828'],cfg['stamp_rate_before_20230828'])
        expected_fee+=selling*table.notional*stamp
        check_close(table.notional,table.shares*table.price,'notional')
        check_close(table.fee,expected_fee,'independent fee')
        check_close(table.price,table.reference_price*(1+np.where(selling,-1,1)*cfg['slippage_rate']),'slipped price')
        check_close(table.slippage_cost,table.shares*(table.price-table.reference_price).abs(),'slippage')
        cash=cfg['initial_cash']+np.cumsum(np.where(selling,table.notional,-table.notional)-expected_fee)
        check_close(table.cash_after,cash,'independent running cash');assert (cash>=-1e-6).all()
        max_cash_error=max(max_cash_error,float(np.max(np.abs(cash-table.cash_after))))
        check_close(table.sellable_after,table.sellable_before-np.where(selling,table.shares,0),'T1 per event')
        assert table.sellable_after.ge(0).all() and table.sellable_before.ge(0).all()
        daycash=pd.Series(np.asarray(cash),index=table.datetime.dt.normalize()).groupby(level=0).last()
        aligned=daycash.reindex(curve.datetime.dt.normalize()).ffill().fillna(cfg['initial_cash'])
        check_close(curve[cash_key],aligned.to_numpy(),'daily cash')
        if account=='actual':
            flows=table.assign(date=table.datetime.dt.normalize(),delta=np.where(selling,-table.shares,table.shares),sold=np.where(selling,table.shares,0))
            daily_flows=flows.groupby(['date','symbol'])[['delta','sold']].sum().reset_index()
            joined=inventory.merge(daily_flows,on=['date','symbol'],how='outer',validate='one_to_one',indicator=True)
            assert joined._merge.eq('both').all() or not joined._merge.eq('right_only').any()
            joined[['delta','sold']]=joined[['delta','sold']].fillna(0)
            joined=joined.sort_values(['symbol','date'])
            closing=joined.groupby('symbol').delta.cumsum()
            check_close(joined.opening_shares,closing-joined.delta,'opening inventory independent ledger')
            check_close(joined.closing_shares,closing,'closing inventory independent ledger')
            check_close(joined.day_sold,joined.sold,'daily sold')
            assert (joined.sold<=joined.opening_sellable).all()
    check_close(curve.equity,curve.cash+curve.holdings_value,'daily equity sum')
    t=events.loc[events.event_type.eq('t')]
    first=t.loc[t.fragment_id.eq(0)].copy();rest=t.loc[t.fragment_id.gt(0)]
    assert len(t)==len(first)+len(rest) and first.pair_id.nunique()==len(first)
    assert len(trips)==len(rest)
    completed=0;pending=0
    for row in first.itertuples(index=False):
        fragments=trips.loc[trips.pair_id.eq(row.pair_id)]
        recovered=rest.loc[rest.pair_id.eq(row.pair_id)]
        quantity=int(recovered.shares.sum());assert 0<=quantity<=row.shares
        assert quantity==int(fragments.shares.sum())
        completed+=quantity==row.shares;pending+=quantity<row.shares
        if len(fragments):
            first_fee=fragments.sell_fee if row.side=='sell' else fragments.buy_fee
            check_close(first_fee,fragments.shares/row.shares*row.fee,'first fee allocation')
            assert (pd.to_datetime(fragments.second_datetime,format='mixed')>row.datetime).all()
    if len(trips):
        check_close(trips.gross_pnl,trips.shares*(trips.sell_price-trips.buy_price),'fragment gross')
        check_close(trips.net_pnl,trips.gross_pnl-trips.buy_fee-trips.sell_fee,'fragment net')
    increment=float(curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1])
    checks={'t_pair_count':int(completed),'pending_pairs':int(pending),'started_pair_count':len(first),
            'fragment_count':len(trips),'inventory_mismatch_days':int(inventory.loc[inventory.inventory_difference.ne(0),'date'].nunique()),
            'recovery_failure_days':int(inventory.loc[inventory.recovery_failure.eq(True),'date'].nunique()),
            'weekly_liquidation_failure_days':int(inventory.loc[inventory.weekly_liquidation_failure.eq(True),'date'].nunique()),
            'base_plan_mismatch_count':len(read_audit_csv(folder/'base_plan_mismatches.csv')),'t1_violation_count':0}
    for key,value in checks.items():assert metrics[key]==value,(folder,key,value,metrics[key])
    check_close(metrics['total_t_marked_pnl'],increment,'metric increment')
    check_close(metrics['total_t_net_profit'],trips.net_pnl.sum(),'metric fragments')
    if checks['base_plan_mismatch_count']==0 and pending==0:check_close(increment,trips.net_pnl.sum(),'paired full account identity')
    if no_t:
        assert not len(t) and increment==0
        # 基准记录原始预算请求，实际账户复制基准已成交量；容量缩量时两者意图可不同。
        pd.testing.assert_frame_equal(events.drop(columns=['account','reason','planned_shares']),base.drop(columns=['account','reason','planned_shares']),check_dtype=False)
        buys=events.side.eq('buy')
        check_close(events.loc[buys,'planned_shares'],base.loc[buys,'shares'],'actual copied baseline filled order')
        check_close(events.loc[~buys,'planned_shares'],base.loc[~buys,'planned_shares'],'same no-T sell order')
        plans=read_audit_csv(folder/'weekly_entry_plans.csv')
        assert plans.plan_id.is_unique
        check_close(plans.entry_budget,plans.baseline_entry_equity*cfg['base_fraction']/cfg['top_n'],'fixed baseline entry budget')
        reference=plans.known_reference_price
        requested=np.where(np.isfinite(reference)&reference.gt(0),np.floor(plans.entry_budget/(reference*(1+cfg['slippage_rate']))/cfg['lot_size'])*cfg['lot_size'],0)
        check_close(plans.planned_shares,requested,'prior-price planned shares')
        assert (plans.baseline_filled_shares.le(plans.planned_shares)&plans.baseline_filled_shares.ge(0)).all()
        check_close(plans.actual_filled_shares,plans.baseline_filled_shares,'no-T baseline filled quantity')
        filled=plans.loc[plans.baseline_filled_shares.gt(0)].set_index('plan_id')
        assert set(base.loc[buys,'plan_id'])==set(filled.index)
        matched=filled.loc[base.loc[buys,'plan_id']]
        check_close(base.loc[buys,'planned_shares'],matched.planned_shares,'baseline original requested shares')
        check_close(base.loc[buys,'shares'],matched.baseline_filled_shares,'baseline plan actual fills')
    result={**checks,'incremental_net_pnl':increment,'active_days':int(first.datetime.dt.normalize().nunique()),
            'active_stock_days':len(first.assign(date=first.datetime.dt.normalize())[['date','symbol']].drop_duplicates()),
            'event_count':len(events),'cash_max_abs_error_CNY':max_cash_error,'passed':True}
    if (folder/'summary.json').is_file():
        summary=json.loads((folder/'summary.json').read_text())
        check_close(summary['incremental_net_pnl'],increment,'saved summary increment')
        for key in ['active_days','active_stock_days']:assert summary[key]==result[key]
        for key in checks:assert summary['metrics'][key]==checks[key]
    return result


def verify_frequency_denominator(folder: Path,cfg: dict,keys: pd.DataFrame) -> dict:
    """从原缓存当前信号键、冻结中间交易日与独立基准旧股重建分母，不调用生产helper。"""
    selections=read_audit_csv(folder/'validation_weekly_selections.csv')
    for name in ['trade_week','entry_date','exit_date']:selections[name]=pd.to_datetime(selections[name],format='mixed')
    joined=keys.merge(selections[['trade_week','symbol','entry_date','exit_date']],on=['trade_week','symbol'],how='inner',validate='many_to_one')
    joined=joined.loc[joined.date.gt(joined.entry_date)&joined.date.lt(joined.exit_date)].copy()
    mapping={ts:advance_minutes(advance_minutes(pd.Timestamp(ts),1),20) for ts in joined.datetime.unique()}
    recovery=joined.datetime.map(mapping)
    cutoff=joined.date+pd.to_timedelta(cfg['latest_planned_recovery_time']+':00')
    legal=joined.loc[recovery.notna()&recovery.le(cutoff)]
    clocks=legal.groupby(['date','symbol']).agg(eligible_signal_count=('datetime','size'),first_eligible_signal=('datetime','min'),last_eligible_signal=('datetime','max')).reset_index()
    inv=read_audit_csv(folder/'validation_baseline/daily_inventory_audit.csv');inv['date']=pd.to_datetime(inv.date,format='mixed')
    inv['eligible_t_shares']=(np.floor(inv.opening_sellable*cfg['short_t_fraction']/cfg['lot_size'])*cfg['lot_size']).astype(int)
    expected=inv.loc[inv.eligible_t_shares.ge(cfg['lot_size']),['date','symbol','opening_sellable','eligible_t_shares']].merge(clocks,on=['date','symbol'],validate='one_to_one')
    expected=expected.sort_values(['date','symbol']).reset_index(drop=True)
    saved=read_audit_csv(folder/'validation_eligible_inventory_stock_days.csv')
    for name in ['date','first_eligible_signal','last_eligible_signal']:saved[name]=pd.to_datetime(saved[name],format='mixed')
    saved=saved.sort_values(['date','symbol']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,saved,check_dtype=False)
    summary=json.loads((folder/'validation_frequency_opportunities.json').read_text())
    for key,value in {'eligible_portfolio_days':expected.date.nunique(),'eligible_inventory_stock_days':len(expected),'eligible_signal_rows':int(expected.eligible_signal_count.sum())}.items():assert summary[key]==value
    # 三个模型的完整推断键必须等于来源缓存的当前中间日键，不能预删未来标签缺失行。
    expected_keys=joined[['datetime','symbol']].sort_values(['datetime','symbol']).reset_index(drop=True)
    for profile in ['l2_depth4','l2_depth6','huber_depth4']:
        predicted=read_audit_csv(folder/f'validation_h20_{profile}_predictions.csv')[['datetime','symbol']]
        predicted['datetime']=pd.to_datetime(predicted.datetime,format='mixed')
        pd.testing.assert_frame_equal(predicted.sort_values(['datetime','symbol']).reset_index(drop=True),expected_keys,check_dtype=False)
    return {**summary,'independently_rebuilt_from_original_cache':True,'current_signal_key_count':len(joined)}


def select_independent_frequency_candidate(rows: pd.DataFrame, days: float) -> pd.Series:
    """独立复算固定目标3组/日的内层频率排序，不调用生产selector。"""
    assert np.isfinite(days) and days>0
    failures=['inventory_mismatch_days','recovery_failure_days','pending_pairs','base_plan_mismatch_count','t1_violation_count']
    score=np.minimum(rows.t_pair_count/days,3.)
    check_close(rows.frequency_score,score,'independent frequency score')
    eligible=rows.t_pair_count.gt(0)&rows[failures].eq(0).all(axis=1)
    assert eligible.equals(rows.active_execution_eligible)
    ranked=rows.assign(independent_score=score).loc[eligible].sort_values(['independent_score','incremental_net_pnl','candidate_index'],ascending=[False,False,True],kind='stable')
    assert len(ranked)>0
    return ranked.iloc[0]


def verify_frequency_training_lineage(report: Path,cfg: dict) -> dict:
    """核原训练权重、全部内部真实账户/机会分母和频率排序，不读取外层收益选择规则。"""
    source=Path(cfg['source_training_report_dir']);source=source if source.is_absolute() else ROOT/source
    original_cfg=json.loads((source/'resolved_config.json').read_text())
    assert original_cfg['research_version']=='weekly_t_v8_residual_zero_market'
    for name in ['initial_cash','base_fraction','top_n','lot_size','commission_rate','min_commission','transfer_rate',
                 'stamp_rate_before_20230828','stamp_rate_from_20230828','slippage_rate','max_bar_amount_fraction',
                 'short_t_fraction','max_pending_t_notional_fraction','latest_planned_recovery_time']:
        assert cfg[name]==original_cfg[name],('费用/执行合同改变',name)
    assert cfg['frequency_target_pairs_per_day']==3 and cfg['frequency_horizon_minutes']==20
    old_audit_path=source/'independent_residual_audit.json';previous=json.loads(old_audit_path.read_text())
    assert previous['passed'] and previous['full_configured_scope_complete']
    manifest=json.loads((report/'data_manifest.json').read_text())
    cache=Path(manifest['cache_path'])/'intraday.pkl'
    inputs=json.loads((report/'input_sources.json').read_text());record=inputs[str(cache.resolve())]
    digest=hashlib.sha256()
    with cache.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    assert cache.stat().st_size==record['size_bytes'] and digest.hexdigest()==record['sha256'],'原始当前信号缓存身份变化'
    panel=pd.read_pickle(cache);assert len(panel)==manifest['intraday_rows']
    keys=panel[['datetime','symbol','date','trade_week']].copy();del panel
    results={};candidate_count=0
    for year in cfg['test_years']:
        folder=report/f'year_{year}';old_fold=source/f'year_{year}'
        rule=json.loads((folder/'selected_rule.json').read_text())
        assert rule['enabled'] is True and rule['frozen_before_any_outer_test'] is True
        split=json.loads((folder/'split.json').read_text());assert split==json.loads((old_fold/'split.json').read_text())
        copies={}
        for profile in ['l2_depth4','l2_depth6','huber_depth4']:
            name=f'inner_t_h20_{profile}';canonical=old_fold/name
            hashes={str(p.relative_to(canonical)):file_sha256(p) for p in canonical.rglob('*') if p.is_file()}
            copied={str(p.relative_to(folder/name)):file_sha256(p) for p in (folder/name).rglob('*') if p.is_file()}
            assert hashes and copied==hashes
            verified=previous['models'][f'year_{year}/{name}'];assert verified['passed']
            for key,file in [('model_bundle_sha256','model_bundle.joblib'),('native_model_sha256','return_model.txt'),('metadata_sha256','model_metadata.json')]:assert verified[key]==hashes[file]
            metadata=json.loads((canonical/'model_metadata.json').read_text())
            assert sorted(metadata['train_years'])==sorted(split['core_years'])
            assert sorted(metadata['excluded_years'])==sorted([year,split['inner_year']])
            assert metadata['resolved_config']['t_label_horizon_minutes']==20
            assert metadata['t_target_mode']=='same_clock_cohort_residual_zero_market' and metadata['t_market_prior']==0.
            copies[profile]=hashes
        selected={str(p.relative_to(folder/'t_model')):file_sha256(p) for p in (folder/'t_model').rglob('*') if p.is_file()}
        assert selected==copies[rule['profile']] and selected['model_bundle.joblib']==rule['model_bundle_sha256']
        assert Path(rule['source_model_path']).resolve()==(old_fold/f"inner_t_h20_{rule['profile']}").resolve()
        support=old_fold/'support_h20.json';assert file_sha256(support)==file_sha256(folder/'selected_support.json')
        assert previous['supports'][f'year_{year}/support_h20.json']['support_file_sha256']==file_sha256(support)
        baseline=verify_saved_frequency_account(folder/'validation_baseline',cfg,no_t=True)
        denominator=verify_frequency_denominator(folder,cfg,keys)
        rows=read_audit_csv(folder/'validation_rule_metrics.csv');assert set(rows.candidate_index)==set(range(25)) and len(rows)==25
        protocols=json.loads((folder/'candidate_rules_frozen_before_inner_execution.json').read_text());assert len(protocols)==25
        original=read_audit_csv(old_fold/'validation_rule_metrics.csv')
        failures=['inventory_mismatch_days','recovery_failure_days','pending_pairs','base_plan_mismatch_count','t1_violation_count']
        valid=original.t_pair_count.gt(0)&original[failures].eq(0).all(axis=1)
        old_choice=original.loc[valid].sort_values(['incremental_net_pnl','candidate_index'],ascending=[False,True]).iloc[0]
        expected=[{'signal_mode':'reference','profile':old_choice.profile,'horizon_minutes':int(old_choice.horizon_minutes),'net_margin':float(old_choice.net_margin)}]
        expected += [{'signal_mode':'absolute','profile':profile,'horizon_minutes':20,'net_margin':margin} for profile in copies for margin in [0.,.0005]]
        expected += [{'signal_mode':'vwap_rank','profile':profile,'horizon_minutes':20,'rank_quantile':q,'min_vwap_gap':gap} for profile in copies for q in [.70,.85,.95] for gap in [.0035,.005]]
        candidates={}
        for index,protocol in enumerate(protocols):
            assert protocol['candidate_index']==index and protocol['enabled'] is True
            assert protocol['fraction']==.25 and protocol['direction']=='both'
            for field,value in expected[index].items():assert protocol[field]==value
            candidate_dir=folder/f'validation_candidate_audits/candidate_{index:03d}'
            assert (folder/'candidate_rules_frozen_before_inner_execution.json').stat().st_mtime_ns<=(candidate_dir/'summary.json').stat().st_mtime_ns
            checked=verify_saved_frequency_account(candidate_dir,cfg);candidate_count+=1
            row=rows.loc[rows.candidate_index.eq(index)].iloc[0]
            for field,value in expected[index].items():assert row[field]==value
            for field in ['t_pair_count',*failures,'active_days','active_stock_days']:assert row[field]==checked[field]
            check_close(row.incremental_net_pnl,checked['incremental_net_pnl'],'candidate selected row net')
            for field in ['eligible_portfolio_days','eligible_inventory_stock_days']:assert row[field]==denominator[field]
            saved=json.loads((candidate_dir/'summary.json').read_text())
            assert saved['candidate_index']==index and saved['decision_sha256']==row.decision_sha256
            assert file_sha256(candidate_dir/'decisions.csv')==saved['saved_decisions_sha256']
            assert len(read_audit_csv(candidate_dir/'decisions.csv'))==int(row.submitted_t_requests)
            candidates[str(index)]=checked
        days=float(denominator['eligible_portfolio_days'])
        chosen=select_independent_frequency_candidate(rows,days)
        assert rule['candidate_index']==int(chosen.candidate_index)
        assert rows.loc[rows.selected,'candidate_index'].tolist()==[int(chosen.candidate_index)]
        for field,value in expected[int(chosen.candidate_index)].items():assert rule[field]==value
        check_close(rule['inner_incremental_net_pnl'],chosen.incremental_net_pnl,'frozen inner net')
        assert rule['inner_complete_pairs']==chosen.t_pair_count and rule['inner_eligible_portfolio_days']==days
        assert rule['activation_policy']=='frequency_execution_valid' and rule['statistical_gates_are_informational'] is True
        results[str(year)]={'passed':True,'model_copies':copies,'selected_profile':rule['profile'],'selected_candidate_index':int(chosen.candidate_index),
            'selected_frequency_score':float(chosen.independent_score),'source_inner_year':split['inner_year'],
            'support_sha256':file_sha256(support),'core_years':split['core_years'],'excluded_years':[year,split['inner_year']],
            'baseline_independent_account':baseline,'denominator':denominator,'all_candidates':candidates,
            'active_selection_independently_reproduced':True,'new_model_fit_performed':False}
        print(f'frequency lineage/candidate accounting passed {year}: 25 accounts',flush=True)
    return {'passed':True,'source_training_report_dir':str(source.resolve()),'original_model_audit_sha256':file_sha256(old_audit_path),
        'selected_models':results,'candidate_accounts_independently_checked':candidate_count,'core_model_copy_count':3*len(results),'selected_outer_model_copy_count':len(results),'original_current_signal_cache_sha256':digest.hexdigest(),
        'scope':'原v8严格训练审计引用、指定折全部核心及选中模型copy身份、全部内部候选独立账目/当前信号分母/仅内部频率排序；未重训。'}


def decompose_portfolio_increment(actual: pd.DataFrame,baseline: pd.DataFrame,books: dict,
                                  marks: dict,pair_state: dict) -> dict:
    """始终分解真实增量，包含未恢复T与周底仓成交偏差，不能强迫它们等于已配对利润。"""
    def cashflow(frame):
        return float(np.where(frame.side.eq('sell'),frame.notional-frame.fee,-frame.notional-frame.fee).sum())
    t=actual.loc[actual.event_type.eq('t')]
    t_cash=cashflow(t)
    weekly_cash=cashflow(actual.loc[actual.event_type.eq('weekly')])-cashflow(baseline)
    holding_difference=sum((books['actual']['qty'].get(symbol,0)-books['baseline']['qty'].get(symbol,0))*marks.get(symbol,0.)
                           for symbol in set(books['actual']['qty'])|set(books['baseline']['qty']))
    unmatched_cash=0.;unmatched_market=0.
    for state in pair_state.values():
        first=state['first'];remaining=state['remaining']
        initial_flow=first.notional-first.fee if first.side=='sell' else -first.notional-first.fee
        unmatched_cash+=initial_flow*remaining/first.shares
        unmatched_market+=remaining*marks.get(first.symbol,0.)*(1 if first.side=='buy' else -1)
    return {'T_cashflow_CNY':t_cash,'weekly_cashflow_difference_CNY':weekly_cash,
            'holding_market_value_difference_CNY':holding_difference,
            'independent_total_increment_CNY':t_cash+weekly_cash+holding_difference,
            'paired_T_net_pnl_CNY':t_cash-unmatched_cash,
            'unmatched_T_first_leg_cashflow_CNY':unmatched_cash,
            'pending_T_inventory_market_value_CNY':unmatched_market,
            'direct_T_marked_pnl_CNY':t_cash+unmatched_market,
            'weekly_position_value_difference_excluding_pending_T_CNY':holding_difference-unmatched_market,
            'weekly_plan_effect_CNY':weekly_cash+holding_difference-unmatched_market}


def advance_minutes(timestamp: pd.Timestamp, steps: int) -> pd.Timestamp:
    """独立按240分钟会话推进，跳午休且不跨日；不依赖生产时钟实现。"""
    clock=timestamp.hour*60+timestamp.minute
    minutes=list(range(571,691))+list(range(781,901))
    assert clock in minutes and timestamp.second==timestamp.microsecond==timestamp.nanosecond==0
    index=minutes.index(clock)+steps
    if index>=len(minutes):return pd.NaT
    return timestamp.normalize()+pd.Timedelta(minutes=minutes[index])


def run_audit(report_dir: Path | str) -> dict:
    """完整运行冻结报告的独立重放，任何会计/时间不一致立即失败并保留原结果。"""
    RUN=Path(report_dir).resolve()
    if not (RUN/'final_report.md').is_file():
        raise ValueError('正式四折报告尚未结束，禁止提前读取外层执行结果')
    cfg=json.loads((RUN/'resolved_config.json').read_text())
    if (cfg.get('research_version')!='weekly_t_v10_frequency'
            or cfg.get('candidate_activation_policy')!='frequency_execution_valid'
            or cfg.get('t_training_cohort')!='core_year_oof_weekly_top100_middle_days'
            or cfg.get('t_training_domain')!='known_slot_and_recovery_deadline'
            or cfg.get('t_target_mode')!='same_clock_cohort_residual_zero_market'
            or cfg.get('t_market_prior')!=0.0 or cfg.get('t_residual_min_group_size')!=2
            or cfg.get('strict_feature_provenance') is not True or cfg.get('outer_t_refit') is not False
            or 'f_history_days' not in cfg.get('excluded_model_features',[])):
        raise ValueError('本执行审计仅适用v10主动T协议；冻结v8训练模型身份另行逐文件验证')
    manifest=json.loads((RUN/'data_manifest.json').read_text())
    declared=json.loads((RUN/'source_hashes.json').read_text())
    if not {'scripts/run_weekly_t_frequency.py','quant_backtest/weekly_t_frequency.py','quant_backtest/weekly_t_frequency_validation.py','quant_backtest/weekly_t_frequency_reporting.py','quant_backtest/weekly_t_residual.py'}.issubset(declared):
        raise ValueError('缺少实际v10运行入口/主动规则/报告模块冻结源码身份')
    if declared.get('quant_backtest/weekly_t_engine.py')!=AUDITED_ENGINE_SHA256:
        raise ValueError('冻结执行引擎与独立重放已审计语义不符，需先复核新版本')
    for name,sha in declared.items():
        assert hashlib.sha256((RUN/'source_snapshot'/Path(name).name).read_bytes()).hexdigest()==sha,name

    def close(a,b,label):
        """仅允许CSV浮点序列化量级误差，逻辑数量必须另以整数精确核对。"""
        assert np.isclose(a,b,atol=1e-5,rtol=1e-10),(label,a,b)

    def fee(value,side,timestamp):
        """独立按冻结费用合同重算佣金、过户费和日期适用印花税。"""
        answer=max(value*cfg['commission_rate'],cfg['min_commission'])+value*cfg['transfer_rate']
        if side=='sell':
            answer+=value*cfg['stamp_rate_from_20230828' if timestamp>=pd.Timestamp('2023-08-28') else 'stamp_rate_before_20230828']
        return answer

    years=list(cfg['test_years'])
    assert all((RUN/f'year_{year}/test_metrics.csv').is_file() for year in years),'年度结果不完整'
    marker=json.loads((RUN/'all_rules_frozen_before_outer_test.json').read_text())
    assert marker['test_years']==years
    assert marker.get('all_t_enabled') is True and marker.get('selection_uses_inner_only') is True
    for year in years:
        expected=marker['selected_rule_sha256'][str(year)]
        assert hashlib.sha256((RUN/f'year_{year}/selected_rule.json').read_bytes()).hexdigest()==expected,'冻结规则哈希变化'
    training_lineage=verify_frequency_training_lineage(RUN,cfg)
    tables={}
    for year in years:
        directory=RUN/f'year_{year}'
        values={}
        for key,file in {'actual':'test_trade_events.csv','baseline':'baseline_trade_events.csv','curve':'test_equity_curve.csv',
                         'trips':'test_round_trips.csv','inventory':'daily_inventory_audit.csv','positions':'test_open_positions.csv',
                         'plans':'weekly_entry_plans.csv','mismatches':'base_plan_mismatches.csv','attempts':'blocked_orders.csv',
                         'selections':'weekly_selections.csv','decisions':'t_decisions.csv'}.items():
            try:
                frame=pd.read_csv(directory/file)
            except pd.errors.EmptyDataError:
                frame=pd.DataFrame()
            for col in ['datetime','signal_datetime','date','first_datetime','second_datetime','buy_datetime','sell_datetime',
                        'entry_date','exit_date','trade_week','mark_datetime','planned_recovery_time']:
                if col in frame:
                    frame[col]=pd.to_datetime(frame[col],format='mixed')
            values[key]=frame
        values['metrics']=pd.read_csv(directory/'test_metrics.csv').iloc[0].to_dict()
        values['rule']=json.loads((directory/'selected_rule.json').read_text())
        tables[year]=values
    print('audit completed years',years,flush=True)
    cache=Path(manifest['cache_path'])
    daily=pd.read_pickle(cache/'daily.pkl')
    original_stats={name:(size,mtime) for name,size,mtime in manifest['files']}
    # 恢复可发生于全会话任一配置时钟；按已成交键只读raw，不载入全量未来特征或模型。
    raw_keys=defaultdict(set)
    for values in tables.values():
        for account in ['actual','baseline']:
            for event in values[account].itertuples(index=False):
                raw_time=(event.datetime+pd.Timedelta(minutes=1)
                          if event.datetime.time()==pd.Timestamp('09:30').time() else event.datetime)
                raw_keys[(event.datetime.year,event.symbol)].add(raw_time)
    raw_quotes={};raw_sources=[]
    for (year,symbol),wanted in sorted(raw_keys.items()):
        root=Path(cfg['data_root']);root=root if root.is_absolute() else ROOT/root
        path=root/str(year)/f'{symbol}.csv'
        stat=path.stat()
        assert original_stats[str(path.relative_to(root))]==(stat.st_size,stat.st_mtime_ns),('raw source changed',path)
        frame=pd.read_csv(path,usecols=['datetime','open','high','low','close','volume','amount'])
        frame['datetime']=pd.to_datetime(frame.datetime)
        frame=frame.loc[frame.datetime.isin(wanted)].sort_values('datetime',kind='stable').drop_duplicates('datetime',keep='last')
        assert set(frame.datetime)==wanted,('executed minute absent in raw',year,symbol,wanted-set(frame.datetime))
        raw_quotes.update({(timestamp,symbol):row for timestamp,row in frame.set_index('datetime').to_dict('index').items()})
        raw_sources.append({'year':year,'symbol':symbol,'path':str(path),'size_bytes':stat.st_size,
                            'mtime_ns':stat.st_mtime_ns,'executed_minutes_checked':len(frame)})

    summaries=[]
    for year,v in tables.items():
        print('replay year',year,flush=True)
        metrics=v['metrics'];curve=v['curve'];inventory=v['inventory'];trips=v['trips']
        peak_pending_notional=0.;previous_baseline_equity=float(cfg['initial_cash'])
        frozen_baseline_error=None
        if not cfg.get('smoke_weeks') and not cfg.get('max_symbols'):
            frozen=Path(cfg['frozen_selection_report_dir'])
            frozen=frozen if frozen.is_absolute() else ROOT/frozen
            frozen_curve=pd.read_csv(frozen/f'year_{year}/test_equity_curve.csv',parse_dates=['datetime'])
            assert curve.datetime.equals(frozen_curve.datetime),('frozen baseline dates',year)
            frozen_baseline_error=float(np.max(np.abs(curve.baseline_equity.to_numpy()-frozen_curve.baseline_equity.to_numpy())))
            assert frozen_baseline_error<=1e-6,('frozen v4 baseline changed',year,frozen_baseline_error)
        plans=v['selections'].sort_values(['entry_date','rank','symbol'],kind='stable').reset_index(drop=True)
        plan_exit=dict(zip(range(1,len(plans)+1),plans.exit_date))
        plan_symbol=dict(zip(range(1,len(plans)+1),plans.symbol))
        symbols=set(plan_symbol.values())
        for _,group in v['actual'].groupby('datetime',sort=False):
            # 同刻到期恢复优先，首腿其次，周度开收盘委托最后；仅核对真实事件顺序。
            phase=pd.Series(np.where(group.event_type.eq('weekly'),2,np.where(group.fragment_id.eq(0),1,0)),index=group.index)
            assert phase.is_monotonic_increasing,'same minute event order changed'
        prices=daily.loc[daily.symbol.isin(symbols)&daily.date.between(curve.datetime.min().normalize(),curve.datetime.max().normalize())]
        daily_groups={date:group for date,group in prices.groupby('date',sort=False)}
        event_groups={account:{date:group for date,group in v[account].groupby(v[account].datetime.dt.normalize(),sort=False)} for account in ['actual','baseline']}
        books={name:{'cash':float(cfg['initial_cash']),'qty':defaultdict(int),'plans':defaultdict(int)} for name in ['actual','baseline']}
        marks={};mark_times={};rows=[];cash_max_error=0.;equity_max_error=0.;capacity_ratios=[];fee_max_error=0.;slippage_max_error=0.;stale_dates=[]
        completed_ids=set();pair_state={};weekly_failure_dates=set();t_failure_dates=set();mismatch_dates=set()
        for date,day_curve in zip(curve.datetime.dt.normalize(),curve.itertuples(index=False)):
            today_plans=v['plans'].loc[v['plans'].datetime.dt.normalize().eq(date)]
            for plan in today_plans.itertuples(index=False):
                close(plan.baseline_entry_equity,previous_baseline_equity,'past baseline entry budget')
                close(plan.entry_budget,previous_baseline_equity*cfg['base_fraction']/cfg['top_n'],'fixed top_n budget')
            quotes=daily_groups.get(date,pd.DataFrame())
            for row in quotes.itertuples(index=False):
                if np.isfinite(row.open) and row.open>0:
                    marks[row.symbol]=row.open;mark_times[row.symbol]=date+pd.Timedelta(hours=9,minutes=30)
                if np.isfinite(row.close) and row.close>0:
                    marks[row.symbol]=row.close;mark_times[row.symbol]=date+pd.Timedelta(hours=15)
            day_state={}
            for account,book in books.items():
                opening=book['qty'].copy();old_by_plan=book['plans'].copy();sold=defaultdict(int)
                weekly_buys=defaultdict(int);weekly_sells=defaultdict(int);used=defaultdict(float);caps={}
                events=event_groups[account].get(date,pd.DataFrame())
                previous=date
                for event in events.itertuples(index=False):
                    assert event.datetime>=previous and event.datetime>event.signal_datetime
                    previous=event.datetime
                    quantity=int(event.shares)
                    assert quantity>0 and quantity%cfg['lot_size']==0 and quantity==event.shares
                    close(event.notional,quantity*event.price,'notional')
                    recalculated_fee=fee(event.notional,event.side,event.datetime)
                    fee_max_error=max(fee_max_error,abs(recalculated_fee-event.fee));close(recalculated_fee,event.fee,'fee')
                    direction=1 if event.side=='buy' else -1
                    close(event.price,event.reference_price*(1+direction*cfg['slippage_rate']),'slipped_price')
                    expected_slip=quantity*abs(event.price-event.reference_price)
                    slippage_max_error=max(slippage_max_error,abs(expected_slip-event.slippage_cost));close(expected_slip,event.slippage_cost,'slippage')
                    first=event.event_type=='t' and event.pair_id not in pair_state and account=='actual'
                    if first:
                        assert event.datetime==advance_minutes(event.signal_datetime,1)
                        expected_plan=advance_minutes(event.datetime,int(v['rule']['horizon_minutes']))
                        assert pd.notna(expected_plan) and event.planned_recovery_time==expected_plan
                        assert expected_plan.strftime('%H:%M')<=cfg['latest_planned_recovery_time']
                        pair_state[event.pair_id]={'shares':quantity,'remaining':quantity,'first':event,'allocations':{}}
                    # 跨计划事件未提供每计划份额时，不猜测审计通过。
                    assert '|' not in str(event.plan_id),'multi-plan allocation needs explicit replay'
                    pid=int(event.plan_id)
                    assert plan_symbol[pid]==event.symbol
                    before=opening[event.symbol]-sold[event.symbol]
                    assert before==event.sellable_before
                    if event.side=='sell':
                        assert quantity<=before and quantity<=book['qty'][event.symbol]
                        assert quantity<=old_by_plan[pid] and quantity<=book['plans'][pid]
                        old_by_plan[pid]-=quantity;sold[event.symbol]+=quantity
                        book['qty'][event.symbol]-=quantity;book['plans'][pid]-=quantity
                        book['cash']+=event.notional-event.fee
                        if event.event_type=='weekly':weekly_sells[event.symbol]+=quantity
                    else:
                        book['qty'][event.symbol]+=quantity;book['plans'][pid]+=quantity
                        book['cash']-=event.notional+event.fee
                        if event.event_type=='weekly':weekly_buys[event.symbol]+=quantity
                    assert book['cash']>=-1e-6
                    assert opening[event.symbol]-sold[event.symbol]==event.sellable_after
                    cash_max_error=max(cash_max_error,abs(book['cash']-event.cash_after));close(book['cash'],event.cash_after,'event_cash')
                    if event.event_type=='t' and not first:
                        planned=pair_state[event.pair_id]['first'].planned_recovery_time
                        assert event.planned_recovery_time==planned and event.datetime>=planned
                        retry={advance_minutes(planned,n) for n in range(6)}
                        allowed_clock=set(cfg.get('scheduled_recovery_times',[]))|{'14:55','14:56','14:57','14:58','14:59','15:00'}
                        assert event.datetime.time()==pd.Timestamp('09:30').time() or event.datetime in retry or event.datetime.strftime('%H:%M') in allowed_clock
                        pair_state[event.pair_id]['remaining']-=quantity
                        assert pair_state[event.pair_id]['remaining']>=0
                        if pair_state[event.pair_id]['remaining']==0:completed_ids.add(event.pair_id)
                    if event.event_type=='t':
                        pending_notional=sum(s['remaining']*s['first'].reference_price for s in pair_state.values())
                        peak_pending_notional=max(peak_pending_notional,pending_notional)
                        fraction=cfg.get('max_pending_t_notional_fraction')
                        if fraction is not None:
                            assert pending_notional<=cfg['initial_cash']*fraction+1e-5,('pending cap exceeded',year,event.datetime,pending_notional)
                    key=(event.datetime,event.symbol)
                    raw_time=(event.datetime+pd.Timedelta(minutes=1)
                              if event.datetime.time()==pd.Timestamp('09:30').time() else event.datetime)
                    row=raw_quotes[(raw_time,event.symbol)]
                    reference=row['open'] if raw_time!=event.datetime else row['close']
                    amount,volume=row['amount'],row['volume']
                    assert row['high']>=reference>=row['low'] and np.isfinite(reference)
                    close(event.reference_price,reference,'real_reference_price')
                    assert amount>0 and volume>0
                    used[key]+=quantity*reference
                    if key in caps:close(caps[key],amount*cfg['max_bar_amount_fraction'],'shared_bar_capacity')
                    caps[key]=amount*cfg['max_bar_amount_fraction']
                    assert used[key]<=caps[key]+1e-5,(year,key,used[key],caps[key])
                capacity_ratios.extend(used[key]/caps[key] for key in used)
                valued=book['cash']+sum(qty*marks.get(symbol,0.) for symbol,qty in book['qty'].items())
                recorded=day_curve.equity if account=='actual' else day_curve.baseline_equity
                equity_max_error=max(equity_max_error,abs(valued-recorded));close(valued,recorded,'marked_equity')
                close(book['cash'],day_curve.cash if account=='actual' else day_curve.baseline_cash,'daily_cash')
                day_state[account]={'opening':opening,'qty':book['qty'].copy(),'sold':sold,'buys':weekly_buys,'sells':weekly_sells}
            previous_baseline_equity=day_curve.baseline_equity
            pending_notional=sum(s['remaining']*s['first'].reference_price for s in pair_state.values())
            close(pending_notional,day_curve.pending_t_notional,'daily pending reference notional')
            pending_symbols={s['first'].symbol for s in pair_state.values() if s['remaining']}
            stale_count=sum(qty>0 and mark_times.get(symbol)!=day_curve.datetime for symbol,qty in books['actual']['qty'].items())
            assert stale_count==day_curve.stale_price_count
            if stale_count:stale_dates.append(date)
            overdue_symbols={plan_symbol[pid] for account in books.values() for pid,qty in account['plans'].items() if qty and plan_exit[pid]<=date}
            overdue_symbols|={s['first'].symbol for s in pair_state.values() if s['remaining'] and plan_exit[int(s['first'].plan_id)]<=date}
            if overdue_symbols:weekly_failure_dates.add(date)
            if pending_symbols:t_failure_dates.add(date)
            for symbol in symbols:
                a,b=day_state['actual'],day_state['baseline']
                if not(a['opening'][symbol] or a['qty'][symbol] or b['qty'][symbol] or a['buys'][symbol] or a['sells'][symbol] or symbol in pending_symbols):continue
                target=a['opening'][symbol]+a['buys'][symbol]-a['sells'][symbol]
                difference=a['qty'][symbol]-target
                if difference:mismatch_dates.add(date)
                rows.append({'date':date,'symbol':symbol,'opening_shares':a['opening'][symbol],'closing_shares':a['qty'][symbol],
                             'day_sold':a['sold'][symbol],'target_shares':target,'inventory_difference':difference,
                             'baseline_closing_shares':b['qty'][symbol],'recovery_failure':symbol in pending_symbols,
                             'weekly_liquidation_failure':symbol in overdue_symbols})
        replay=pd.DataFrame(rows).sort_values(['date','symbol']).reset_index(drop=True)
        recorded=inventory.sort_values(['date','symbol']).reset_index(drop=True)
        assert len(replay)==len(recorded)
        for column in replay:
            assert replay[column].equals(recorded[column]),(year,column)
        close(peak_pending_notional,metrics['peak_pending_t_notional'],'peak pending reference notional')
        close(sum(s['remaining']*s['first'].reference_price for s in pair_state.values()),metrics['final_pending_t_notional'],'final pending reference notional')
        pair_rows=[]
        for pair_id,state in pair_state.items():
            events=v['actual'].loc[(v['actual'].event_type=='t')&(v['actual'].pair_id==pair_id)]
            first=events.iloc[0];reverse=events.iloc[1:];fragments=trips.loc[trips.pair_id==pair_id]
            restored=int(reverse.shares.sum());proportion=restored/first.shares
            cashflow=float(np.where(events.side=='sell',events.notional-events.fee,-events.notional-events.fee).sum())
            matched=float(np.where(reverse.side=='sell',reverse.notional-reverse.fee,-reverse.notional-reverse.fee).sum())
            matched+=(first.notional-first.fee if first.side=='sell' else -first.notional-first.fee)*proportion
            close(matched,fragments.net_pnl.sum(),'fragment_net_pnl')
            close(first.fee*proportion+reverse.fee.sum(),(fragments.buy_fee+fragments.sell_fee).sum(),'fragment_fees')
            close(first.slippage_cost*proportion+reverse.slippage_cost.sum(),fragments.slippage_cost.sum(),'fragment_slippage')
            if not state['remaining']:close(cashflow,fragments.net_pnl.sum(),'complete_pair_cashflow')
            assert (reverse.side!=first.side).all() and restored<=first.shares
            running=first.shares
            assert len(fragments)==len(reverse)
            for fragment,event in zip(fragments.itertuples(index=False),reverse.itertuples(index=False)):
                running-=event.shares
                assert fragment.shares==event.shares and fragment.second_datetime==event.datetime
                assert fragment.remaining_shares_after==running and fragment.original_pair_shares==first.shares
                assert fragment.planned_recovery_time==first.planned_recovery_time
                assert fragment.pair_fully_recovered==(state['remaining']==0)
                assert fragment.first_datetime==first.datetime and event.datetime>first.datetime
                buy_event,sell_event=(first,event) if first.side=='buy' else (event,first)
                assert fragment.buy_datetime==buy_event.datetime and fragment.sell_datetime==sell_event.datetime
                close(fragment.buy_price,buy_event.price,'fragment buy event price')
                close(fragment.sell_price,sell_event.price,'fragment sell event price')
                allocated_fee=first.fee*event.shares/first.shares
                close(fragment.buy_fee,allocated_fee if first.side=='buy' else event.fee,'fragment buy fee allocation')
                close(fragment.sell_fee,allocated_fee if first.side=='sell' else event.fee,'fragment sell fee allocation')
                close(fragment.slippage_cost,first.slippage_cost*event.shares/first.shares+event.slippage_cost,'fragment exact slippage allocation')
            for fragment in fragments.itertuples(index=False):
                close(fragment.gross_pnl,fragment.shares*(fragment.sell_price-fragment.buy_price),'fragment_gross')
                close(fragment.net_pnl,fragment.gross_pnl-fragment.buy_fee-fragment.sell_fee,'fragment_net')
            pair_rows.append({'pair_id':pair_id,'net_pnl':matched,'cashflow':cashflow,'pending_shares':state['remaining'],'fragments':len(fragments)})
        actual_weekly=v['actual'].loc[v['actual'].event_type=='weekly'].groupby(['datetime','plan_id','side']).shares.sum()
        baseline_weekly=v['baseline'].loc[v['baseline'].event_type=='weekly'].groupby(['datetime','plan_id','side']).shares.sum()
        combined=pd.concat([actual_weekly.rename('actual'),baseline_weekly.rename('baseline')],axis=1).fillna(0)
        plan_mismatches=int(combined.actual.ne(combined.baseline).sum())
        assert plan_mismatches==metrics['base_plan_mismatch_count']==len(v['mismatches'])
        planned=v['plans'];expected=np.floor(planned.entry_budget/(planned.known_reference_price*(1+cfg['slippage_rate']))/cfg['lot_size'])*cfg['lot_size']
        assert expected.eq(planned.planned_shares).all()
        assert len(completed_ids)==metrics['completed_pair_count']==metrics['t_pair_count']
        assert len(trips)==metrics['fragment_count']
        pending=sum(state['remaining']>0 for state in pair_state.values())
        assert pending==metrics['pending_pairs']
        assert len(mismatch_dates)==metrics['inventory_mismatch_days']
        assert len(t_failure_dates)==metrics['recovery_failure_days']
        assert len(weekly_failure_dates)==metrics['weekly_exit_failure_days']
        assert sum(qty>0 for qty in books['actual']['plans'].values())==metrics['weekly_unclosed_positions']
        assert sum(qty>0 for qty in books['baseline']['plans'].values())==metrics['baseline_weekly_unclosed_positions']
        assert len(stale_dates)==metrics['stale_valuation_days']
        final_positions=[]
        for position in v['positions'].itertuples(index=False):
            pid=int(position.plan_id)
            assert position.shares==books['actual']['plans'][pid]
            assert position.baseline_shares==books['baseline']['plans'][pid]
            close(position.mark_price,marks[position.symbol],'position_mark_price')
            close(position.market_value,position.shares*marks[position.symbol],'position_market_value')
            assert position.mark_datetime==mark_times[position.symbol]
            final_positions.append({'plan_id':pid,'symbol':position.symbol,'shares':int(position.shares),
                                    'baseline_shares':int(position.baseline_shares),'mark_price':float(position.mark_price),
                                    'market_value_CNY':float(position.market_value),'mark_datetime':str(position.mark_datetime),
                                    'stale_at_year_end':bool(position.shares>0 and position.mark_datetime!=curve.datetime.iloc[-1]),
                                    'mark_age_calendar_days':int((curve.datetime.iloc[-1]-position.mark_datetime).days),
                                    'planned_exit_date':str(position.exit_date)})
        t=v['actual'].loc[v['actual'].event_type=='t'];increment=curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1]
        decomposition=decompose_portfolio_increment(v['actual'],v['baseline'],books,marks,pair_state)
        close(decomposition['independent_total_increment_CNY'],increment,'always_valid_full_increment_decomposition')
        close(decomposition['paired_T_net_pnl_CNY'],trips.net_pnl.sum(),'independent_matched_cashflow_fragment_profit')
        close(decomposition['T_cashflow_CNY'],metrics['t_cashflow'],'all_T_cashflow')
        close(increment,metrics['total_t_marked_pnl'],'T_marked_increment')
        close(trips.net_pnl.sum(),metrics['total_t_net_profit'],'T_fragment_profit')
        close(v['actual'].fee.sum(),metrics['fees'],'all_fees')
        close(t.fee.sum(),metrics['t_fee'],'T_fees')
        if plan_mismatches==0 and pending==0:close(increment,trips.net_pnl.sum(),'baseline_increment_pair_identity')
        first_events=t.drop_duplicates('pair_id',keep='first').copy()
        if len(first_events):
            decision_index=v['decisions'].set_index(['datetime','symbol'])
            keys=pd.MultiIndex.from_arrays([first_events.signal_datetime,first_events.symbol])
            np.testing.assert_allclose(first_events.signal_priority.to_numpy(),decision_index.priority.reindex(keys).to_numpy(),rtol=1e-12,atol=1e-15)
            assert not first_events.assign(date=first_events.datetime.dt.normalize()).duplicated(['date','symbol']).any()
            for _,group in first_events.groupby('datetime',sort=False):assert group.signal_priority.is_monotonic_decreasing
            for event in first_events.itertuples(index=False):
                plan=plans.iloc[int(event.plan_id)-1]
                assert plan.entry_date<event.datetime.normalize()<plan.exit_date
                decision=decision_index.loc[(event.signal_datetime,event.symbol)]
                assert event.side==('buy' if decision.action>0 else 'sell')
                assert event.planned_recovery_time==decision.planned_recovery_time
                opening=inventory.loc[inventory.date.eq(event.datetime.normalize())&inventory.symbol.eq(event.symbol),'opening_shares'].iloc[0]
                assert event.shares<=np.floor(opening*decision.fraction/cfg['lot_size'])*cfg['lot_size']
        failures=inventory.loc[inventory.recovery_failure,['date','symbol']]
        affected_attempts=v['attempts'].loc[v['attempts'].event_type.eq('t_recovery')].copy()
        affected_attempts['date']=affected_attempts.datetime.dt.normalize()
        affected_attempts=affected_attempts.merge(failures,on=['date','symbol'],how='inner',validate='many_to_one')
        if len(trips):
            loss_dates=trips.assign(first_date=trips.first_datetime.dt.normalize()).groupby('first_date').agg(
                net_pnl=('net_pnl','sum'),pair_count=('pair_id','nunique')).sort_values('net_pnl').head(3)
        else:loss_dates=pd.DataFrame()
        summary={'year':year,'audit_pass':True,'evaluation_start':str(curve.datetime.iloc[0]),'evaluation_end':str(curve.datetime.iloc[-1]),
                 'trading_days':len(curve),'events':len(v['actual']),'baseline_events':len(v['baseline']),'replayed_inventory_rows':len(replay),
                 't_first_leg_events':int(t.fragment_id.eq(0).sum()),'t_recovery_events':int(t.fragment_id.gt(0).sum()),
                 'completed_pairs':len(completed_ids),'fragments':len(trips),'pending_pairs':pending,'T_increment_CNY':float(increment),
                 'increment_decomposition':decomposition,
                 'total_return':float(metrics['total_return']),'baseline_total_return':float(metrics['baseline_total_return']),
                 'actual_fees_CNY':float(v['actual'].fee.sum()),'T_fees_CNY':float(t.fee.sum()),
                 'actual_slippage_CNY':float(v['actual'].slippage_cost.sum()),'T_slippage_CNY':float(t.slippage_cost.sum()),
                 'inventory_mismatch_days':len(mismatch_dates),'recovery_failure_days':len(t_failure_dates),
                 'weekly_exit_failure_days':len(weekly_failure_dates),'weekly_unclosed_positions':int(metrics['weekly_unclosed_positions']),
                 'baseline_weekly_unclosed_positions':int(metrics['baseline_weekly_unclosed_positions']),
                 'base_plan_mismatches':plan_mismatches,'cash_max_abs_error_CNY':cash_max_error,'equity_max_abs_error_CNY':equity_max_error,
                 'fee_max_abs_error_CNY':fee_max_error,'slippage_max_abs_error_CNY':slippage_max_error,
                 'maximum_used_fraction_of_configured_capacity':max(capacity_ratios,default=0.),
                 'minimum_actual_cash_CNY':float(v['actual'].cash_after.min()),'t1_legal':True,
                 'peak_pending_reference_notional_CNY':peak_pending_notional,
                 'final_pending_reference_notional_CNY':float(metrics['final_pending_t_notional']),
                 'pending_notional_cap_and_partial_release_verified':True,
                 'fixed_horizon_planned_recovery_and_no_early_restore_verified':True,
                 'frozen_v4_baseline_max_abs_error_CNY':frozen_baseline_error,
                 'signal_priority_and_daily_one_pair_and_middle_week_days_verified':True,
                 'stale_valuation_days':len(stale_dates),
                 'recovery_failure_stock_days':len(failures),
                 'inventory_mismatch_stock_days':int(inventory.inventory_difference.ne(0).sum()),
                 'weekly_exit_failure_stock_days':int(inventory.weekly_liquidation_failure.sum()),
                 'recovery_failure_dates':{str(k.date()):int(value) for k,value in failures.groupby('date').size().items()},
                 'recovery_attempt_reasons_on_unrecovered_stock_days':{str(k):int(value) for k,value in affected_attempts.reason.value_counts().items()},
                 'lowest_PnL_first_dates':[{'date':str(date.date()),'net_pnl_CNY':float(row.net_pnl),'pair_count':int(row.pair_count)}
                                          for date,row in loss_dates.iterrows()],
                 'final_stale_position_count':sum(p['stale_at_year_end'] for p in final_positions),
                 'final_stale_market_value_CNY':sum(p['market_value_CNY'] for p in final_positions if p['stale_at_year_end']),
                 'final_cash_CNY':float(curve.cash.iloc[-1]),'final_holdings_value_CNY':float(curve.holdings_value.iloc[-1]),
                 'final_equity_CNY':float(curve.equity.iloc[-1]),
                 'final_open_positions':final_positions,
                 'strict_daily_inventory_contract_met':len(mismatch_dates)==0 and len(t_failure_dates)==0 and pending==0,
                 'full_weekly_exit_contract_met':len(weekly_failure_dates)==0 and metrics['weekly_unclosed_positions']==0}
        summaries.append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)

    complete=years==[2023,2024,2025,2026]
    audit={'passed':True,'classification':'v10_independent_execution_audit','completed_all_four_years':complete,
           'completed_years':years,'snapshot_hashes_verified':declared,'audit_method':'Independent event replay per account and plan, historical fee formula, original raw execution/recovery minute price and shared capacity, planned horizon and pending notional cap, daily marked equity, T fragment allocation and cashflow, daily inventory/weekly exit failures, frozen v4 baseline daily identity.',
           'raw_history_isolation':'No new model is trained. Original v8 model-isolation audit is SHA-referenced and every selected inner model artifact is compared byte-for-byte with the new outer model copy.',
           'model_isolation_audit_source':training_lineage,
           'inherited_execution_replay':'v5 independent outer replay retained; all internal candidates independently checked for saved ledger fees/cash/T1/counts/denominator/frequency selection; raw-minute fill replay applies to the final outer accounts. v10 uses frozen v8 models; execution still uses raw stock prices and unchanged fees, T+1 and capacity. Exact frozen engine SHA required.',
           'audited_engine_sha256':AUDITED_ENGINE_SHA256,
           'source_policy':'Only frozen v10 source semantics and original v10 report/cache/raw artifacts were used. Workspace model/runner not imported. Original result files unchanged.',
           'raw_execution_sources':raw_sources,
           'auditor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'annual':summaries,'sum_independent_annual_T_increment_CNY':sum(row['T_increment_CNY'] for row in summaries),
           'fees_include_slippage':False,'slippage_is_already_in_execution_price_and_must_not_be_subtracted_twice':True}
    lines=['# v10 主动T独立执行审计','',
           '本审计独立核对主动执行和记账。训练隔离引用已完成v8审计的SHA，本次实际选中核心模型、原生权重、元数据和支持域逐文件确认与源产物相同；没有重新训练。执行重放严格核对相同冻结引擎SHA。','',
           f'已审计年度：{years}；四年结果是否齐全：{complete}。按冻结v10源码语义，独立重放实际/基准账户、逐笔旧库存与现金、真实分钟价格和共享容量、费用与滑点、T片段分摊、逐日估值和周度退出状态。未导入当前模型或runner，原结果未修改。','',
           '|年度|T净增量(元)|完成T组/片段|库存不一致日|恢复失败日|周退出失败日|期末周持仓|陈旧估值市值(元)|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"|{row['year']}|{row['T_increment_CNY']:,.6f}|{row['completed_pairs']}/{row['fragments']}|{row['inventory_mismatch_days']}|{row['recovery_failure_days']}|{row['weekly_exit_failure_days']}|{row['weekly_unclosed_positions']}|{row['final_stale_market_value_CNY']:,.2f}|")
    lines+=['','账户增量始终独立分解为T现金流＋周底仓现金流差＋期末持仓市值差；基准计划相同且无pending时才额外要求它等于配对T净利。下表保留未恢复T及底仓偏差造成的影响，不能把全部账户差冒称已完成T收益。','',
            '|年度|已配对T净利(元)|未恢复首腿现金流(元)|pending T库存市值(元)|周底仓计划影响(元)|','|---|---:|---:|---:|---:|']
    for row in summaries:
        part=row['increment_decomposition']
        lines.append(f"|{row['year']}|{part['paired_T_net_pnl_CNY']:.6f}|{part['unmatched_T_first_leg_cashflow_CNY']:.6f}|{part['pending_T_inventory_market_value_CNY']:.6f}|{part['weekly_plan_effect_CNY']:.6f}|")
    lines+=['','所有已审计年度的成交合法性、费用及会计恒等式通过。通过记账审计并不表示每天股数不变或周末能全部退出；表中的真实失败继续保留。不同年度独立重置100万元，不将年度收益串接为连续实盘净值。',
            '佣金、过户费及历史日期对应印花税已独立重算；滑点已计入成交价格，另列滑点成本仅为归因，不能再次扣减。期末持仓及最后真实估值时间见independent_execution_audit.json。陈旧估值保留其最后真实行情，不能假设可以按该价格清仓。','']
    stale_positions=[(row['year'],position) for row in summaries for position in row['final_open_positions'] if position['stale_at_year_end']]
    if stale_positions:
        lines+=['期末陈旧估值持仓：','',
                '|年度|股票|股份|最后价格|最后真实估值时间|陈旧天数|账面市值(元)|',
                '|---|---|---:|---:|---|---:|---:|']
        for year,p in stale_positions:
            lines.append(f"|{year}|{p['symbol']}|{p['shares']}|{p['mark_price']:.2f}|{p['mark_datetime']}|{p['mark_age_calendar_days']}|{p['market_value_CNY']:,.2f}|")
        lines.append('')
    (RUN/'independent_execution_audit.md').write_text('\n'.join(lines))

    # 各年度独立账户的计数求和，不能拼接成年复利净值。
    audit['totals']={key:sum(row[source] for row in summaries) for key,source in {
        'actual_events':'events','baseline_events':'baseline_events','inventory_rows':'replayed_inventory_rows',
        't_first_leg_events':'t_first_leg_events','t_recovery_events':'t_recovery_events',
        'completed_pairs':'completed_pairs','fragments':'fragments','pending_pairs':'pending_pairs',
        'inventory_mismatch_days':'inventory_mismatch_days','inventory_mismatch_stock_days':'inventory_mismatch_stock_days',
        'recovery_failure_days':'recovery_failure_days','recovery_failure_stock_days':'recovery_failure_stock_days',
        'weekly_exit_failure_days':'weekly_exit_failure_days','weekly_exit_failure_stock_days':'weekly_exit_failure_stock_days',
        'weekly_unclosed_positions':'weekly_unclosed_positions','baseline_weekly_unclosed_positions':'baseline_weekly_unclosed_positions',
        'final_stale_position_count':'final_stale_position_count','final_stale_market_value_CNY':'final_stale_market_value_CNY'}.items()}
    audit['totals']['t1_violation_count']=0
    (RUN/'independent_execution_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return audit


def main() -> None:
    """接收已有完整报告路径，只新增独立审计文件。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    audit=run_audit(args.report_dir)
    print(json.dumps({'report_dir':str(args.report_dir.resolve()),'totals':audit['totals'],
                      'T_increment_CNY':audit['sum_independent_annual_T_increment_CNY']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
