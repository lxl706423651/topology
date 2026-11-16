"""已完成内部折的只读账表/目标量级核对；不读外层或重跑模型/账户。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'reports/inventory_learning_t/run_20260910_151232_841321'
OUT = Path(__file__).resolve().parent/f'snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}'
INPUTS = {}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def bind(path):
    path=Path(path).resolve()
    if str(path) not in INPUTS:INPUTS[str(path)]={'sha256':sha(path),'size':path.stat().st_size}
    return path


def csv(path,**kw):return pd.read_csv(bind(path),float_precision='round_trip',**kw)
def js(path):return json.loads(bind(path).read_text())
def save(name,frame):frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def main():
    OUT.mkdir(exist_ok=False)
    # 用户本次明确授权的两个已完成内部折，后续完成的折不在运行中偷偷纳入。
    years=[2023,2024]
    write('protocol.json',{'source':str(SOURCE),'completed_inner_folds_in_scope':years,
        'nonfinal_progress_only':True,'outer_read':False,'rerun_accounts_or_models':False,
        'no_candidate_selection_or_parameter_changes':True,
        'identity':'complete and partial fragment costs separated from pending terminal valuation'})
    cfg=js(SOURCE/'resolved_config.json')
    rows,selected_rows,stock_targets=[],[],[]
    for year in years:
        directory=SOURCE/f'year_{year}'
        table=csv(directory/'validation_rule_metrics.csv')
        selected=js(directory/'selected_rule.json')
        split=js(directory/'split.json')
        assert len(table)==37 and table.candidate_index.nunique()==37
        comparison=None
        for row in table.itertuples():
            folder=Path(row.candidate_directory)
            events=csv(folder/'test_trade_events.csv')
            trips=csv(folder/'test_round_trips.csv')
            base=csv(folder/'baseline_trade_events.csv')
            positions=csv(folder/'test_open_positions.csv')
            curve=csv(folder/'test_equity_curve.csv')
            baseline=curve[['datetime','baseline_equity','baseline_cash']]
            if comparison is None:comparison=baseline.copy()
            pd.testing.assert_frame_equal(baseline,comparison,check_exact=True)
            base_buys=base.loc[base.side.eq('buy')]
            denominator=float((base_buys.notional+base_buys.fee).sum())
            assert abs(denominator-row.all_reference_acquisition_cost)<1e-7
            base_columns=['datetime','symbol','side','shares','price','notional','fee','base_order_type']
            actual_base=events.loc[events.event_type.eq('weekly'),base_columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual_base,base[base_columns].reset_index(drop=True),check_exact=True)
            t=events.loc[events.event_type.eq('t')]
            first=t.loc[t.fragment_id.eq(0)]
            recovery=t.loc[t.fragment_id.gt(0)]
            assert not first.pair_id.duplicated().any()
            detail=recovery.merge(first[['pair_id','shares','reference_price','fee','slippage_cost','side']].rename(columns={
                name:'first_'+name for name in ['shares','reference_price','fee','slippage_cost','side']}),on='pair_id',validate='many_to_one')
            direction=np.where(detail.first_side.eq('buy'),1.,-1.)
            part_ref=detail.shares*(detail.reference_price-detail.first_reference_price)*direction
            part_slip=detail.slippage_cost+detail.first_slippage_cost*detail.shares/detail.first_shares
            part_fee=detail.fee+detail.first_fee*detail.shares/detail.first_shares
            part_net=part_ref-part_slip-part_fee
            recorded=trips.set_index(['pair_id','fragment_id']).net_pnl
            reconstructed=pd.Series(part_net.to_numpy(),index=pd.MultiIndex.from_frame(detail[['pair_id','fragment_id']]))
            assert set(recorded.index)==set(reconstructed.index)
            max_fragment_error=float((reconstructed-recorded).abs().max()) if len(recorded) else 0.
            assert max_fragment_error<1e-7
            qty=recovery.groupby('pair_id').shares.sum()
            closed=first.loc[first.pair_id.map(qty).fillna(0).eq(first.shares)]
            complete_net=float(part_net.loc[detail.pair_id.isin(closed.pair_id)].sum())
            fee=float(t.fee.sum());slip=float(t.slippage_cost.sum())
            reference_cash=float((np.where(t.side.eq('sell'),1.,-1.)*t.reference_price*t.shares).sum())
            pending_mark=float(((positions.shares-positions.baseline_shares)*positions.mark_price).sum())
            marked_ref_gross=reference_cash+pending_mark
            reconstructed_marked=marked_ref_gross-fee-slip
            actual_marked=float(curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1])
            error=abs(reconstructed_marked-actual_marked)
            assert error<1e-6
            for actual,expected in [(fee,row.t_fee),(slip,row.t_slippage_cost),(complete_net,row.realized_complete_pair_pnl),
                (part_net.sum(),row.realized_all_fragment_pnl),(part_ref.sum(),row.realized_fragment_reference_gross_pnl),
                (actual_marked,row.incremental_marked_pnl),(len(closed),row.t_pair_count),
                (closed.shares.sum()/100,row.completed_t_pair_lots)]:
                assert abs(actual-expected)<1e-6,(year,row.candidate_index,actual,expected)
            target=denominator*cfg['profit_target_rate']
            item={'test_year':year,'actual_inner_year':split['inner_year'],'candidate_index':int(row.candidate_index),
                'selected':int(row.candidate_index)==selected['rule']['candidate_index'],'profile':row.profile,
                'min_net_edge':row.min_net_edge,'fraction':row.fraction,'exit_mode':row.exit_mode,
                'complete_groups':len(closed),'recovery_fragments':len(trips),'completed_pair_lots':closed.shares.sum()/100,
                'one_way_traded_lots':t.shares.sum()/100,'pending_pairs':int(row.pending_pairs),
                'realized_reference_gross':float(part_ref.sum()),'realized_allocated_slippage':float(part_slip.sum()),
                'realized_allocated_fee':float(part_fee.sum()),'realized_all_fragment_net':float(part_net.sum()),
                'complete_pair_net':complete_net,'all_t_fee':fee,'all_t_slippage':slip,
                'marked_reference_gross':marked_ref_gross,'marked_net':actual_marked,
                'unpaired_marked_component':actual_marked-complete_net,'all_reference_acquisition_cost':denominator,
                'cost_reduction_rate':complete_net/denominator,'initial_cash':cfg['initial_cash'],
                'account_t_increment_rate':actual_marked/cfg['initial_cash'],
                'baseline_total_return':row.baseline_total_return,'combined_total_return':row.total_return,
                'target_net_5pct':target,'shortfall_to_target':target-complete_net,
                'old_inventory_stock_days':row.baseline_old_stock_opportunity_stock_days,
                'target_net_per_old_inventory_stock_day':target/row.baseline_old_stock_opportunity_stock_days,
                'inventory_mismatch_days':row.inventory_mismatch_days,'recovery_failure_days':row.recovery_failure_days,
                't1_violation_count':row.t1_violation_count,'base_plan_mismatch_count':row.base_plan_mismatch_count,
                'baseline_curve_exact_within_fold':True,'marked_net_identity_abs_error':error,
                'max_fragment_net_abs_error':max_fragment_error}
            rows.append(item)
            if item['selected']:
                selected_rows.append(item)
                cost=base_buys.assign(cost=base_buys.notional+base_buys.fee).groupby('symbol').cost.sum()
                audit=csv(folder/'daily_inventory_audit.csv')
                opportunities=audit.loc[audit.baseline_opening_shares.ge(100)].groupby('symbol').date.nunique()
                for symbol,amount in cost.items():
                    days=int(opportunities.get(symbol,0))
                    stock_targets.append({'test_year':year,'actual_inner_year':split['inner_year'],'symbol':symbol,
                        'original_acquisition_cost':amount,'original_cost_weight':amount/denominator,
                        'assigned_5pct_target_net':amount*cfg['profit_target_rate'],'old_inventory_days':days,
                        'arithmetic_daily_target':amount*cfg['profit_target_rate']/days if days else None,
                        'not_profit_forecast':True})
    save('all_completed_candidate_accounting.csv',pd.DataFrame(rows))
    save('completed_inner_progress.csv',pd.DataFrame(selected_rows))
    save('per_stock_arithmetic_target.csv',pd.DataFrame(stock_targets))
    for path,meta in INPUTS.items():assert sha(path)==meta['sha256'],path
    write('inputs_sha256.json',INPUTS)
    check={'passed':True,'nonfinal':True,'completed_folds':years,'candidates_checked':len(rows),
        'all_common_baseline_curves_exact_within_fold':True,'all_baseline_cost_denominators_match':True,
        'max_marked_pnl_identity_error':max(x['marked_net_identity_abs_error'] for x in rows),
        'max_fragment_identity_error':max(x['max_fragment_net_abs_error'] for x in rows),
        'pending_candidate_count':sum(x['pending_pairs']>0 for x in rows),'input_files_unchanged':len(INPUTS),
        'no_outer_read':True,'no_account_or_model_rerun':True,'script_sha256':sha(__file__),
        'full_runtime_terminal_integrity_claimed':False}
    write('verification.json',check)
    print(OUT)
    print(json.dumps(check,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
