"""只读旧底仓金额/股数账表的机械诊断；不读取收益列，不回测或选参。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path('/data/lxl/seed-emulator/topology/quant')
OUT = Path(__file__).resolve().parent
REPORT = ROOT/'reports/weekly_t_cross_validation/run_20260909_182945_388303'
INPUTS = {}

def read(path, columns):
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, usecols=columns, float_precision='round_trip')

def summarize(values):
    values = pd.Series(values)
    return {key: float(value) for key, value in values.quantile([0,.25,.5,.75,1]).rename(index={0:'min',.25:'p25',.5:'median',.75:'p75',1:'max'}).items()}

annual=[]
for year in [2023,2024,2025,2026]:
    folder=REPORT/f'year_{year}'
    plans=read(folder/'weekly_entry_plans.csv', ['datetime','symbol','plan_id','known_reference_price','entry_budget','planned_shares','baseline_filled_shares'])
    events=read(folder/'baseline_trade_events.csv', ['datetime','symbol','plan_id','side','shares','notional','fee'])
    inv=read(folder/'daily_inventory_audit.csv', ['date','symbol','opening_sellable','closing_shares','day_sold','cycles_started','weekly_buys','weekly_sells'])
    selections=read(folder/'weekly_selections.csv', ['trade_week','symbol','entry_date','exit_date'])
    blocks=read(folder/'blocked_orders.csv', ['account','event_type','reason','filled_shares','requested_shares'])
    buys=events[events.side.eq('buy')].copy()
    old=inv[inv.opening_sellable.gt(0)].copy()
    quarter=(old.opening_sellable*.25//100*100).astype(int)
    schedules=selections.assign(entry_date=pd.to_datetime(selections.entry_date),exit_date=pd.to_datetime(selections.exit_date))
    inv['date']=pd.to_datetime(inv.date)
    inv['trade_week']=inv.date-pd.to_timedelta(inv.date.dt.weekday,unit='D')
    schedules['trade_week']=pd.to_datetime(schedules.trade_week)
    joined=inv.merge(schedules,on=['trade_week','symbol'],how='left',validate='many_to_one')
    old_exit=joined.opening_sellable.gt(0)&joined.date.eq(joined.exit_date)
    # 仅用历史入场前收价进行静态整手计划，不模拟新策略成交或收益。
    p=plans.known_reference_price*(1+.0005)
    low=np.ceil(10000/(p*100));high=np.floor(20000/(p*100))
    chosen=np.floor(15000/(p*100)+.5).clip(lower=low,upper=high)
    qty=np.where(np.isfinite(p)&p.gt(0)&high.ge(low)&high.ge(1),chosen*100,0)
    valid=plans.baseline_filled_shares.gt(0)
    block_counts=blocks[blocks.account.eq('actual')&blocks.event_type.eq('t_entry')].groupby('reason').size().to_dict()
    row={'year':year,'baseline_buy_events':len(buys),'entry_plans':len(plans),'base_buy_notional_CNY':summarize(buys.notional),
         'base_buy_count_below_10000':int(buys.notional.lt(10000).sum()),'base_buy_count_10000_to_20000':int(buys.notional.between(10000,20000).sum()),
         'base_buy_count_above_20000':int(buys.notional.gt(20000).sum()),'entry_budget_CNY':summarize(plans.entry_budget),
         'filled_100_to_300_share_entries':int(buys.shares.between(100,300).sum()),
         'zero_fill_entry_plans':int(plans.baseline_filled_shares.eq(0).sum()),
         'partially_filled_entry_plans':int((plans.baseline_filled_shares.gt(0)&plans.baseline_filled_shares.lt(plans.planned_shares)).sum()),
         'opening_old_inventory_stock_days':len(old),'quarter_fraction_zero_T_stock_days':int(quarter.eq(0).sum()),
         'quarter_fraction_positive_T_stock_days':int(quarter.gt(0).sum()),
         'old_sellable_lots_total':float(old.opening_sellable.sum()/100),'quarter_first_leg_lots_capacity_free_total':float(quarter.sum()/100),
         'remaining_old_sellable_lots_after_one_quarter_pair_capacity_free':float((old.opening_sellable-quarter).sum()/100),
         'min_100_share_policy_additional_eligible_stock_days':int(quarter.eq(0).sum()),
         'old_inventory_stock_days_on_weekly_exit_date':int(old_exit.sum()),
         'realized_daily_sell_lots_including_weekly_exits':float(inv.day_sold.sum()/100),
         'maximum_recorded_T_cycles_per_stock_day':int(inv.cycles_started.max()),
         'baseline_weekly_sell_events':int(events.side.eq('sell').sum()),
         'first_entry_day_buy_count':int(buys.datetime.eq(buys.datetime.min()).sum()),
         'first_entry_day_buy_notional_sum_CNY':float(buys.loc[buys.datetime.eq(buys.datetime.min()),'notional'].sum()),
         'actual_T_entry_rejections':{str(k):int(v) for k,v in block_counts.items()},
         'static_15000_sizing_only':{'same_filled_old_entry_keys':int(valid.sum()),'positive_quarter_lot_plans':int((valid&(qty>=400)).sum()),
             'original_positive_quarter_lot_plans':int((valid&plans.baseline_filled_shares.ge(400)).sum()),
             'no_100_share_lot_within_band_plans':int((qty==0).sum()),'not_a_counterfactual_fill_or_portfolio':True}}
    annual.append(row)
source_names=['quant_backtest/weekly_t_engine.py','quant_backtest/inventory_t.py','quant_backtest/weekly_t_quote_store.py','scripts/run_weekly_t_cross_validation.py']
sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
result={'classification':'read_only_holdings_and_T1_mechanical_diagnosis_not_return_selection','source_report':str(REPORT),
        'read_profit_columns':False,'initial_cash':1000000,'old_initial_slot_CNY':1000000*.95/100,
        'new_proposed_target_notional_CNY':15000,'new_proposed_base_fraction':.85,
        'cash_limited_target_position_count_approx_85pct_with_buy_fee':int(850000/(15000*1.00081)),
        'cash_limited_target_position_count_approx_95pct_with_buy_fee':int(950000/(15000*1.00081)),
        'cash_limited_min_position_count_upper_bound_85pct_with_buy_fee':int(850000/(10000*1.00081)),
        'annual':annual,'input_sha256':INPUTS,'source_sha256':sources,
        'caveats':['所有容量/股数上限均不是可成交或盈利预测。','静态15000元股数仅按同一历史请求的已知前收估算，不按后续涨跌选股。','每日不平底仓并不能使持仓市值一直落在1—2万元，区间只能定义在建仓。','年度增量5%以初始资金为分母时每年须净增50000元；按底仓成本为分母需另列，不能混用。']}
(OUT/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,dict)} for row in annual]).to_csv(OUT/'annual_inventory_summary.csv',index=False)
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha for name,sha in INPUTS.items())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
print(json.dumps({'source_unchanged':True,'input_files_unchanged':len(INPUTS),'annual':annual},ensure_ascii=False,indent=2))
