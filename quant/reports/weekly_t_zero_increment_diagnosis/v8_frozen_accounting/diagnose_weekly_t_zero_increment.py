"""只读v8四年规则/收益通路及已存非零真实执行，新增独立诊断，不拟合或回测。"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path('/data/lxl/seed-emulator/topology/quant')
RUN=ROOT/'reports/weekly_t_cross_validation/run_20260908_235818_404243'
OUT=ROOT/'reports/weekly_t_zero_increment_diagnosis/v8_frozen_accounting';OUT.mkdir(parents=True,exist_ok=True)
REAL=ROOT/'reports/weekly_t_cross_validation_smoke/real_t_v5_early_execution'
sources={}
def read_frame(path):
    """读取CSV并绑定原始字节SHA，拒绝靠修改历史文件构造非零例证。"""
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)

def read_json(path):
    """读取已冻结JSON并记录身份。"""
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

cfg=read_json(RUN/'resolved_config.json');declared=read_json(RUN/'source_hashes.json')
diagnostic_protocol=read_json(REAL/'frozen_execution_protocol.json')
assert diagnostic_protocol['source_sha256']['quant_backtest/weekly_t_engine.py']==declared['quant_backtest/weekly_t_engine.py']
for key in ['commission_rate','min_commission','transfer_rate','stamp_rate_before_20230828','stamp_rate_from_20230828','slippage_rate','max_bar_amount_fraction']:
    assert diagnostic_protocol['config'][key]==cfg[key]
code_checks={}
for name in ['quant_backtest/weekly_t_engine.py','quant_backtest/weekly_t_validation.py','quant_backtest/weekly_t_short_horizon.py',
             'scripts/run_weekly_t_residual.py','scripts/run_weekly_t_cross_validation.py']:
    current=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    snapshot=hashlib.sha256((RUN/'source_snapshot'/Path(name).name).read_bytes()).hexdigest()
    assert current==snapshot==declared[name]
    code_checks[name]={'passed':True,'sha256':current}

years=[];candidates=[]
for year in [2023,2024,2025,2026]:
    folder=RUN/f'year_{year}';rule=read_json(folder/'selected_rule.json')
    decisions=read_frame(folder/'t_decisions.csv');events=read_frame(folder/'test_trade_events.csv')
    base=read_frame(folder/'baseline_trade_events.csv');trips=read_frame(folder/'test_round_trips.csv')
    curve=read_frame(folder/'test_equity_curve.csv');metrics=read_frame(folder/'test_metrics.csv').iloc[0]
    predictions=read_frame(folder/'t_predictions.csv');validation=read_frame(folder/'validation_rule_metrics.csv')
    validation['audited_outer_fold']=year;candidates.append(validation)
    event_t=events.loc[events.event_type.eq('t')]
    increment=float(curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1])
    daily_difference=float((curve.equity-curve.baseline_equity).abs().max())
    same_weekly=events.loc[events.event_type.eq('weekly'),['datetime','symbol','side','shares','price','fee']].reset_index(drop=True)
    expected_weekly=base[['datetime','symbol','side','shares','price','fee']].reset_index(drop=True)
    pd.testing.assert_frame_equal(same_weekly,expected_weekly)
    assert rule['enabled'] is False and len(decisions)==len(event_t)==len(trips)==0
    assert increment==daily_difference==metrics.t_added_equity==metrics.total_t_net_profit==metrics.t_cashflow==0
    years.append({'year':year,'rule_enabled':rule['enabled'],'decision_rows':len(decisions),'nonzero_action_rows':int(decisions.action.ne(0).sum()),
        'all_actual_events':len(events),'T_events':len(event_t),'round_trip_fragments':len(trips),'prediction_rows':len(predictions),
        'nonzero_prediction_rows':int(predictions.prediction.ne(0).sum()),'max_abs_prediction':float(predictions.prediction.abs().max()),
        'all_weekly_orders_exactly_match_baseline':True,'increment_CNY':increment,'max_daily_actual_minus_baseline_CNY':daily_difference,
        'curve_metrics_cashflow_agree':True,'nonzero_internal_candidate_count':int(validation.incremental_net_pnl.ne(0).sum())})
all_candidates=pd.concat(candidates,ignore_index=True)
all_candidates['increment_minus_paired_pnl']=all_candidates.incremental_net_pnl-all_candidates.total_t_net_profit
all_candidates['cashflow_minus_paired_pnl']=all_candidates.t_cashflow-all_candidates.total_t_net_profit
clean=all_candidates.base_plan_mismatch_count.eq(0)&all_candidates.pending_pairs.eq(0)
assert np.allclose(all_candidates.loc[clean,'increment_minus_paired_pnl'],0,atol=1e-6,rtol=0)
assert np.allclose(all_candidates.loc[all_candidates.pending_pairs.eq(0),'cashflow_minus_paired_pnl'],0,atol=1e-6,rtol=0)
all_candidates.to_csv(OUT/'internal_candidate_accounting_identity.csv',index=False)
pd.DataFrame(years).to_csv(OUT/'outer_rule_decision_event_counts.csv',index=False)
selected=all_candidates.loc[all_candidates.audited_outer_fold.eq(2023)&all_candidates.candidate_index.eq(14)].iloc[0]
attempts=read_frame(RUN/'year_2023/validation_candidate_audits/candidate_014_t_attempts.csv')
failures=read_frame(RUN/'year_2023/validation_candidate_audits/candidate_014_t_inventory_failures.csv')
attempts.to_csv(OUT/'example_inner_candidate_014_saved_attempts.csv',index=False)
failures.to_csv(OUT/'example_inner_candidate_014_saved_inventory_failures.csv',index=False)

def independent_existing_account(direction):
    """由既存完整真实事件重算双边费/滑点/分片和现金恒等式，不调用执行引擎。"""
    folder=REAL/direction;params=read_json(REAL/'resolved_config.json')
    events=read_frame(folder/'test_trade_events.csv');base=read_frame(folder/'baseline_trade_events.csv')
    trips=read_frame(folder/'test_round_trips.csv');curve=read_frame(folder/'test_equity_curve.csv')
    metrics=read_frame(folder/'test_metrics.csv').iloc[0]
    for frame in [events,base]:frame['datetime']=pd.to_datetime(frame.datetime)
    pair_records=[];fees=[]
    def fee(value,side,ts):
        charge=max(value*params['commission_rate'],params['min_commission'])+value*params['transfer_rate']
        if side=='sell':charge+=value*params['stamp_rate_before_20230828' if ts<pd.Timestamp('2023-08-28') else 'stamp_rate_from_20230828']
        return charge
    for frame in [events,base]:
        charges=np.array([fee(row.shares*row.price,row.side,row.datetime) for row in frame.itertuples(index=False)])
        np.testing.assert_allclose(charges,frame.fee,atol=1e-10,rtol=1e-12)
        fees.extend(np.abs(charges-frame.fee))
        sign=np.where(frame.side.eq('buy'),-1,1)
        flows=sign*frame.shares*frame.price-charges
        np.testing.assert_allclose(params['initial_cash']+flows.cumsum(),frame.cash_after,atol=1e-7,rtol=0)
        expected=frame.reference_price*(1+np.where(frame.side.eq('buy'),1,-1)*params['slippage_rate'])
        np.testing.assert_allclose(expected,frame.price,atol=1e-12,rtol=1e-12)
    weekly_columns=['datetime','symbol','side','shares','price','fee']
    pd.testing.assert_frame_equal(events.loc[events.event_type.eq('weekly'),weekly_columns].reset_index(drop=True),base[weekly_columns].reset_index(drop=True))
    active=events.loc[events.event_type.eq('t')]
    fragment_rows=[]
    for pair_id,group in active.groupby('pair_id'):
        first=group.loc[group.fragment_id.eq(0)].iloc[0]
        reverse=group.loc[group.fragment_id.ne(0)]
        assert first.shares==reverse.shares.sum()
        group_pnl=0.0
        for second in reverse.itertuples(index=False):
            quantity=int(second.shares);first_fee=fee(first.shares*first.price,first.side,first.datetime)*quantity/first.shares
            second_fee=fee(quantity*second.price,second.side,second.datetime)
            raw_profit=quantity*((second.price-first.price) if first.side=='buy' else (first.price-second.price))
            net=raw_profit-first_fee-second_fee
            stored=trips.loc[trips.pair_id.eq(pair_id)&trips.fragment_id.eq(second.fragment_id)].iloc[0]
            np.testing.assert_allclose(net,stored.net_pnl,atol=1e-9,rtol=0)
            fragment_rows.append({'account':direction,'pair_id':int(pair_id),'fragment_id':int(second.fragment_id),'symbol':first.symbol,
                'first_datetime':first.datetime,'second_datetime':second.datetime,'first_side':first.side,'shares':quantity,
                'first_execution_price':first.price,'second_execution_price':second.price,'allocated_first_fee':first_fee,
                'second_fee':second_fee,'independent_gross_pnl':raw_profit,'independent_net_pnl':net,'saved_net_pnl':stored.net_pnl})
            group_pnl+=net
        flow=np.where(group.side.eq('sell'),1,-1)*group.shares*group.price-group.fee
        np.testing.assert_allclose(group_pnl,flow.sum(),atol=1e-9,rtol=0)
        pair_records.append({'account':direction,'pair_id':int(pair_id),'symbol':first.symbol,'original_shares':int(first.shares),
                             'fragment_count':len(reverse),'independent_net_pnl':group_pnl,'independent_cashflow':float(flow.sum())})
    cashflow=float((np.where(active.side.eq('sell'),1,-1)*active.shares*active.price-active.fee).sum())
    total=sum(row['independent_net_pnl'] for row in pair_records)
    increment=float(curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1])
    np.testing.assert_allclose([total,cashflow,trips.net_pnl.sum(),metrics.total_t_net_profit],increment,atol=1e-7,rtol=0)
    assert metrics.pending_pairs==metrics.base_plan_mismatch_count==0
    pd.DataFrame(fragment_rows).to_csv(OUT/f'existing_real_{direction}_independent_fragments.csv',index=False)
    pd.DataFrame(pair_records).to_csv(OUT/f'existing_real_{direction}_independent_pairs.csv',index=False)
    return {'account':direction,'classification':'preexisting_fixed_signal_real_market_execution_diagnostic_not_model_profit_evidence',
        'pair_count':len(pair_records),'fragment_count':len(fragment_rows),'independent_paired_net_pnl_CNY':total,
        'independent_T_cashflow_CNY':cashflow,'actual_equity_minus_baseline_CNY':increment,
        'max_fee_error_CNY':float(max(fees)),'same_weekly_baseline_orders':True,'all_cash_after_and_slippage_recomputed':True,
        'max_account_vs_pair_error_CNY':abs(total-increment),'first_fragment_example':fragment_rows[0],
        'first_positive_fragment_example':next((row for row in fragment_rows if row['independent_net_pnl']>0),None)}

accounts=[independent_existing_account(direction) for direction in ['sell_first','buy_first']]
result={'scope':'只读v8冻结代码/产物与既存真实主动T诊断；未重训/回测/重跑全审计，未改正式源或结果。',
 'existing_active_diagnostic_engine_sha_and_costs_equal_v8':True,
 'outer_years':years,'code_snapshot_checks':code_checks,'internal_candidate_count':len(all_candidates),
 'nonzero_internal_candidate_count':int(all_candidates.incremental_net_pnl.ne(0).sum()),
 'positive_internal_candidate_count':int(all_candidates.incremental_net_pnl.gt(0).sum()),
 'negative_internal_candidate_count':int(all_candidates.incremental_net_pnl.lt(0).sum()),
 'clean_baseline_complete_candidates':int(clean.sum()),
 'clean_candidate_max_increment_vs_pair_error_CNY':float(all_candidates.loc[clean,'increment_minus_paired_pnl'].abs().max()),
 'complete_candidate_max_cashflow_vs_pair_error_CNY':float(all_candidates.loc[all_candidates.pending_pairs.eq(0),'cashflow_minus_paired_pnl'].abs().max()),
 'positive_inner_example':{key:selected[key] for key in ['audited_outer_fold','inner_year','candidate_index','profile','horizon_minutes','net_margin',
     't_pair_count','total_t_net_profit','t_cashflow','incremental_net_pnl','base_plan_mismatch_count','pending_pairs','rejection_reasons']},
 'inner_per_trade_evidence_limitation':'worker仅存失败/部分attempts与库存异常，未保存inner完整events/trips/curve。内层指标可验证非零及汇总恒等式，不能凭缺失文件声称已逐笔独立验证候选。逐笔独立非零证据来自既存相同执行引擎的真实固定信号诊断。',
 'preexisting_real_active_accounts':accounts,'source_sha256':sources,
 'conclusion':'未发现收益被常量0覆盖；四个outer规则明确关闭T导致0订单/0T成交/0配对，与纯底仓逐日相同。internal非零及独立真实事件现金重算反证收益函数不是常量0。'}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps({key:value for key,value in result.items() if key not in ['source_sha256','code_snapshot_checks','outer_years']},ensure_ascii=False,indent=2,default=str))
