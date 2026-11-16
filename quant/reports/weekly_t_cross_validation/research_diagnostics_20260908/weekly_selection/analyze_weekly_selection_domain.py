"""首折core2024/25、inner2026周选择资金/流动性诊断；不读外层结果、不拟合、不改输入。"""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,'/data/lxl/seed-emulator/topology/quant')
import pandas as pd,numpy as np,joblib,lightgbm
from quant_backtest.weekly_t_features import add_weekly_execution_features
from quant_backtest.weekly_t_models import predict_weekly
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from scripts.run_weekly_t_cross_validation import select_weekly_universe,select_year_rows
R=Path('/data/lxl/seed-emulator/topology/quant');run=R/'reports/weekly_t_cross_validation/run_20260908_231721_245847';fold=run/'year_2023';frozen=R/'reports/weekly_t_cross_validation/run_20260908_212722_188543/year_2023'
inputs=[]
def use(p):
 p=Path(p);inputs.append(p);return p
cfg=json.loads(use(run/'resolved_config.json').read_text());cache=Path(json.loads(use(run/'data_manifest.json').read_text())['cache_path']);started=time.monotonic()
weekly=pd.read_pickle(use(cache/'weekly.pkl'));daily=pd.read_pickle(use(cache/'daily.pkl'))
# 只保留授权target年份；过去已完成周行情是同一因果输入，绝不读取outer2023标签/账户文件。
weekly=weekly.loc[weekly.trade_week.dt.year.isin([2024,2025,2026])].copy()
weekly=add_weekly_execution_features(weekly,daily);universe=select_weekly_universe(weekly,cfg)
columns=['f_last_close','f_avg_daily_amount','f_prior_close_auction_amount_min','f_prior_close_auction_amount_mean','f_daily_amplitude_mean','f_week_return','f_intraday_reversal_fraction','f_amount_concentration','f_daily_trend_strength']
keys=['trade_week','symbol'];parts=[];preds={};modeldirs={}
for y in [2024,2025]:
 d=fold/f'oof_weekly_cohort/selector_{y}';modeldirs[y]=d
 p=pd.read_csv(use(d/'oof_weekly_predictions.csv'),parse_dates=['trade_week','feature_week','signal_datetime']);assert set(p.trade_week.dt.year)=={y}
 p=p.merge(universe[keys+['entry_date','exit_date','universe_previous_close',*columns,'target_return','target_amplitude']],on=keys,validate='one_to_one')
 preds[y]=p;parts.append(p.loc[p.selected].assign(scope='core_oof',year=y))
# 仅本折冻结inner模型重放当时全部周候选预测，核selected keys和rank不变；不重新拟合。
modeldirs[2026]=frozen/'inner_weekly_model';bundle=joblib.load(use(modeldirs[2026]/'model_bundle.joblib'))
assert bundle['metadata']['train_years']==[2024,2025]
target=select_year_rows(universe,2026,cfg);pred=predict_weekly(bundle,target,cfg)
pred=pred.merge(target[keys+['entry_date','exit_date','universe_previous_close',*columns,'target_return','target_amplitude']],on=keys,validate='one_to_one');preds[2026]=pred
saved=pd.read_csv(use(fold/'validation_weekly_selections.csv'),parse_dates=['trade_week','entry_date','exit_date'])
actual=pred.loc[pred.selected].sort_values(keys);expected=saved.sort_values(keys)
assert actual[keys].reset_index(drop=True).equals(expected[keys].reset_index(drop=True))
assert np.max(np.abs(actual.score.to_numpy()-expected.score.to_numpy()))<1e-12
parts.append(actual.assign(scope='inner',year=2026));selected=pd.concat(parts,ignore_index=True)
slot=cfg['initial_cash']*cfg['base_fraction']/cfg['top_n'];lot=cfg['lot_size'];fraction=cfg['short_t_fraction']
def stats(f):
 price=f.f_last_close;qty=(np.floor(slot/(price*(1+cfg['slippage_rate']))/lot)*lot).astype(int)
 return {'stockweeks':len(f),'weeks':f.trade_week.nunique(),'symbols':f.symbol.nunique(),'static_ge400':int((np.floor(qty*fraction/lot)*lot>=lot).sum()),'static_ge400_fraction':float((np.floor(qty*fraction/lot)*lot>=lot).mean()),'static_base_zero':int(qty.eq(0).sum()),'features':{c:{'median':float(f[c].median()),'p10':float(f[c].quantile(.1)),'p90':float(f[c].quantile(.9)),'missing':int(f[c].isna().sum())} for c in columns},'pred_rank_summary':{c:float(f[c].median()) for c in ['pred_return_rank','pred_amplitude_rank','score'] if c in f}}
out={'scope':'firstfold core OOF2024/2025 + frozen inner2026 only; no outer returns read; no fitting or threshold selection','slot_budget':slot,'selected':{},'eligible_pool':{},'actual_entries':{},'support':{},'target_semantics':{},'native_importance':{},'inner_frozen_selection_replay':True}
for y,p in preds.items():
 s=selected.loc[selected.year.eq(y)].copy();out['selected'][str(y)]=stats(s);out['eligible_pool'][str(y)]=stats(p)
 # 只重放授权年份空T底仓，统计实际开仓股数/容量；不输出账户收益，不评价T规则。
 dd=daily.loc[daily.date.dt.year.eq(y)&daily.symbol.isin(s.symbol)&daily.date.ge(s.entry_date.min())].copy()
 res=run_weekly_portfolio(s,dd,pd.DataFrame(),pd.DataFrame(),cfg)
 plan=res['weekly_entry_plans'].copy();q=plan.baseline_filled_shares;tq=np.floor(q*fraction/lot)*lot
 e={'plans':len(plan),'zero_filled':int(q.eq(0).sum()),'positive_below400':int((q.gt(0)&q.lt(400)).sum()),'filled_ge400':int(tq.ge(lot).sum()),'filled_ge400_fraction':float(tq.ge(lot).mean()),'planned_ge400_but_filled_lt400':int((plan.planned_shares.ge(400)&q.lt(400)).sum()),'entry_budget_range':[float(plan.entry_budget.min()),float(plan.entry_budget.max())],'actual_fill_share_quantiles':{str(k):float(q.quantile(k)) for k in [0,.1,.5,.9,1]}}
 events=res['baseline_events'];buy=events.loc[events.side.eq('buy')];e['initial_buy_notional_total']=float(buy.notional.sum())
 blocked=res.get('blocked_orders',res.get('attempts',pd.DataFrame()))
 e['result_keys']=list(res)
 out['actual_entries'][str(y)]=e
 # 目标诊断仅raw下一周标签，不含账户成本/实际成交；等股周均值、等周IC。
 icret=[];icamp=[]
 for _,g in p.groupby('trade_week'):
  icret.append(g.pred_return.corr(g.target_return,method='spearman'));icamp.append(g.pred_amplitude.corr(g.target_amplitude,method='spearman'))
 out['target_semantics'][str(y)]={'mean_weekly_return_rank_ic':float(np.nanmean(icret)),'mean_weekly_amplitude_rank_ic':float(np.nanmean(icamp)),'selected_raw_week_return_mean':float(s.target_return.mean()),'eligible_raw_week_return_mean':float(p.target_return.mean()),'selected_daily_amplitude_mean':float(s.target_amplitude.mean()),'eligible_daily_amplitude_mean':float(p.target_amplitude.mean()),'selected_valid_labels':int((s.target_return.notna()&s.target_amplitude.notna()).sum()),'selected_negative_week_return_fraction':float(s.target_return.lt(0).mean())}
 del res
core=selected.loc[selected.year.isin([2024,2025])];inner=selected.loc[selected.year.eq(2026)]
out['selected']['core_combined']=stats(core)
for c in columns:
 lo,hi=core[c].quantile([.001,.999]);x=inner[c]
 out['support'][c]={'core_q001':float(lo),'core_q999':float(hi),'inner_outside_count':int((~x.between(lo,hi)|x.isna()).sum()),'inner_outside_fraction':float((~x.between(lo,hi)|x.isna()).mean())}
liquidity=['f_avg_daily_amount','f_week_amount','f_week_volume','f_amount_concentration','f_largest_minute_amount_share','f_amount_ratio_prior4w','f_prior_close_auction_amount_min','f_prior_close_auction_amount_mean','f_missing_prior_close_days']
for y,d in modeldirs.items():
 meta=json.loads(use(d/'model_metadata.json').read_text());info={'train_years':meta['train_years'],'excluded_years':meta['excluded_years'],'training_rows':meta['training_rows'],'features':meta['feature_names'],'heads':{}}
 for head in ['return','amplitude']:
  b=lightgbm.Booster(model_file=str(use(d/f'{head}_model.txt')));names=b.feature_name();gain=b.feature_importance('gain');gain=gain/gain.sum();pairs=dict(zip(names,map(float,gain)))
  info['heads'][head]={'price_gain_fraction':pairs.get('f_last_close',0),'liquidity_gain_fraction':sum(pairs.get(c,0) for c in liquidity),'top5':dict(sorted(pairs.items(),key=lambda z:-z[1])[:5]),'all_gain':pairs}
 out['native_importance'][str(y)]=info
for p in [Path(__file__),R/'quant_backtest/weekly_t_data.py',R/'quant_backtest/weekly_t_models.py',R/'quant_backtest/weekly_t_features.py',R/'quant_backtest/weekly_t_engine.py',R/'quant_backtest/annual_portfolio.py',R/'scripts/run_weekly_t_cross_validation.py',run/'runtime_notice.json']:use(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
out['inputs']=[{'path':str(p),'sha256':sha(p)} for p in sorted(set(inputs))];out['elapsed_seconds']=time.monotonic()-started
Path('/tmp/weekly_selection_domain_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['native_importance','inputs']},ensure_ascii=False,indent=2))
