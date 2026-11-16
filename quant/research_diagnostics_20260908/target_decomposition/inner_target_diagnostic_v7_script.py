"""只读v7首折核心2024/2025与内部2026，分解短周期目标共同项和个股残差。

读取冻结分钟/短期标签缓存、OOF训练域、模型元数据与原生重要性；写本目录的
JSON/MD诊断，不拟合模型、不改规则、不读取外层收益。未来市场收益仅为标签诊断。
"""
from __future__ import annotations
import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
PROJECT=Path(__file__).resolve().parents[2]
REPORT=PROJECT/'reports/weekly_t_cross_validation/run_20260908_231721_245847'
OUTPUT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))
from quant_backtest.weekly_t_features import add_session_context_features
from quant_backtest.weekly_t_short_horizon import feature_support_mask,planned_recovery_times


def sha(path):
    """按块摘要文件内容，不让缓存副本重复占据内存。"""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):value.update(part)
    return value.hexdigest()


def correlation(first,second):
    """有限数据Pearson描述相关；不把相关性作为因果或独立统计显著性。"""
    good=np.isfinite(first)&np.isfinite(second)
    if good.sum()<3 or first[good].std()==0 or second[good].std()==0:return None
    return float(first[good].corr(second[good]))


def describe(frame):
    """报告目标均值/尺度、共同波动、当前市场与未来收益及个股残差信号相关。"""
    frame=frame.loc[np.isfinite(frame.target_return)&np.isfinite(frame.market_forward_return)].copy()
    y=frame.target_return
    m=frame.market_forward_return
    residual=y-m
    clock_common=frame.groupby('datetime').target_return.transform('mean')
    centered=float(((y-y.mean())**2).sum())
    energy=float((y*y).sum())
    extreme=y.abs().ge(y.abs().quantile(.99))
    clocks=frame.drop_duplicates('datetime')
    result={'rows':len(frame),'days':int(frame.datetime.dt.normalize().nunique()),
        'weeks':int(frame.trade_week.nunique()),'symbols':int(frame.symbol.nunique()),
        'distinct_clocks':int(frame.datetime.nunique()),'mean_rows_per_clock':float(len(frame)/frame.datetime.nunique()),
        'target_mean_bp':float(y.mean()*10000),'target_std_bp':float(y.std()*10000),
        'market_forward_mean_bp':float(m.mean()*10000),'market_forward_std_bp':float(m.std()*10000),
        'residual_mean_bp':float(residual.mean()*10000),'residual_std_bp':float(residual.std()*10000),
        'corr_target_with_all_mainboard_market_forward':correlation(y,m),
        'selected_clock_common_share_of_centered_target_ss':float(((clock_common-y.mean())**2).sum()/centered) if centered else None,
        'top_abs_one_percent_target_energy_share':float((y[extreme]**2).sum()/energy) if energy else None,
        'abs_target_99pct_bp':float(y.abs().quantile(.99)*10000),
        'corr_current_market_day_to_future_market_by_unique_clock':correlation(clocks.f_market_day_return,clocks.market_forward_return),
        'corr_current_market_open_to_future_market_by_unique_clock':correlation(clocks.f_market_return,clocks.market_forward_return),
        'corr_current_market_day_to_target':correlation(frame.f_market_day_return,y),
        'corr_current_market_open_to_target':correlation(frame.f_market_return,y),
        'corr_own_open_to_target':correlation(frame.f_open_return,y),
        'corr_own_open_to_residual':correlation(frame.f_open_return,residual),
        'corr_vwap_deviation_to_residual':correlation(frame.f_vwap_deviation,residual),
        'corr_own_last5m_to_residual':correlation(frame.f_return_5m,residual)}
    return result


def main():
    """保持预声明三个周期，重建实际核心训练键后只作描述统计和原生权重审阅。"""
    started=time.monotonic()
    fold=REPORT/'year_2023'
    cfg=json.loads((REPORT/'resolved_config.json').read_text())
    data=json.loads((REPORT/'data_manifest.json').read_text())
    horizon_manifest=json.loads((REPORT/'horizon_manifest.json').read_text())
    inputs=[REPORT/'resolved_config.json',REPORT/'data_manifest.json',REPORT/'horizon_manifest.json',
            REPORT/'source_hashes.json',REPORT/'inner_calibration_v7.json',PROJECT/'quant_backtest/weekly_t_models.py',
            PROJECT/'quant_backtest/weekly_t_features.py',PROJECT/'quant_backtest/weekly_t_short_horizon.py']
    output={'scope':'仅首折core2024/2025与inner2026；无外层收益/新模型拟合/参数变更',
            'native_importance':[],'horizons':[],'core_row_checks':[]}
    for horizon in [20,40,60]:
        for profile in cfg['t_model_profiles']:
            directory=fold/f'inner_t_h{horizon}_{profile}'
            meta_path=directory/'model_metadata.json';native_path=directory/'return_model.txt'
            metadata=json.loads(meta_path.read_text());model=lgb.Booster(model_file=str(native_path))
            assert model.feature_name()==metadata['feature_names']
            gain=model.feature_importance(importance_type='gain');split=model.feature_importance(importance_type='split')
            names=model.feature_name();total=float(gain.sum())
            features=[{'feature':name,'gain':float(g),'normalized_gain':float(g/total),'split':int(s)}
                      for name,g,s in zip(names,gain,split)]
            features.sort(key=lambda row:-row['gain'])
            market_share=sum(row['normalized_gain'] for row in features if row['feature'] in ['f_market_return','f_market_day_return'])
            output['native_importance'].append({'horizon':horizon,'profile':profile,'feature_names_match':True,
                'training_rows':metadata['training_rows'],'fitted_rows':metadata['fitted_rows'],
                'market_only_gain_share':market_share,'features':features})
            inputs.extend([meta_path,native_path])
    selected_path=fold/'validation_weekly_selections.csv'
    selected=pd.read_csv(selected_path,parse_dates=['trade_week','entry_date','exit_date'],float_precision='round_trip')
    assert selected.entry_date.dt.year.eq(2026).all()
    inputs.append(selected_path)
    domains={}
    for h in [20,40,60]:
        path=fold/f'training_domain_h{h}.csv'
        domain=pd.read_csv(path,parse_dates=['datetime','trade_week','feature_source_start'],float_precision='round_trip')
        assert domain.datetime.dt.year.isin([2024,2025]).all()
        domains[h]=domain.loc[domain.keep,['datetime','symbol','trade_week','feature_source_start']].copy()
        inputs.extend([path,fold/f'training_domain_h{h}.json',fold/f'support_h{h}.json'])
    inner_keys_path=fold/'validation_predictions_h20_l2_depth4.csv'
    inner_keys=pd.read_csv(inner_keys_path,usecols=['datetime','symbol'],parse_dates=['datetime'])
    inputs.append(inner_keys_path)
    wanted=pd.concat([domains[20][['datetime','symbol']],inner_keys]).drop_duplicates()
    raw_path=Path(data['cache_path'])/'intraday.pkl';labels_path=Path(horizon_manifest['labels_path'])
    inputs.extend([raw_path,labels_path])
    print('读取2024/2025/2026完整同刻截面，计算当前市场特征后只保留核心/内部键。',flush=True)
    raw=pd.read_pickle(raw_path)
    cols=['datetime','symbol','date','trade_week','signal_price','previous_close','f_market_return',
          'f_open_return','f_vwap_deviation','f_previous_day_amplitude','f_return_5m']
    current=raw.loc[raw.datetime.ge('2024-01-01')&raw.datetime.lt('2027-01-01'),cols].copy()
    del raw;gc.collect()
    current=add_session_context_features(current)
    current=current.merge(wanted,on=['datetime','symbol'],how='inner',validate='one_to_one')
    del wanted;gc.collect()
    inner=inner_keys.merge(current,on=['datetime','symbol'],validate='one_to_one').merge(
        selected[['trade_week','symbol','entry_date','exit_date','universe_previous_close']],
        on=['trade_week','symbol'],validate='many_to_one')
    assert inner.date.gt(inner.entry_date).all() and inner.date.lt(inner.exit_date).all()
    slot=cfg['initial_cash']*cfg['base_fraction']/cfg['top_n'];lot=cfg['lot_size']
    base=np.floor(slot/(inner.universe_previous_close*(1+cfg['slippage_rate']))/lot)*lot
    inner['static_slot_eligible']=np.floor(base*cfg['short_t_fraction']/lot)>=1
    print('读取标签；未来全主板等权市场收益仅作统计目标，不加入任何当前f_*。',flush=True)
    labels=pd.read_pickle(labels_path)
    for h in [20,40,60]:
        mask=labels.horizon_minutes.eq(h)&labels.datetime.ge('2024-01-01')&labels.datetime.lt('2027-01-01')
        values=labels.loc[mask,['datetime','symbol','label_start','label_end','target_return']].copy()
        values['target_return']=values.target_return.where(np.isfinite(values.target_return))
        # 所有当刻具有真实未来标签的本地主板股票等权，不使用选股名次加权。
        market=values.groupby('datetime').target_return.agg(market_forward_return='mean',market_contributors='count').reset_index()
        core=domains[h].merge(values,on=['datetime','symbol'],how='left',validate='one_to_one')
        valid=core.feature_source_start.dt.year.isin([2024,2025])&core.label_start.dt.year.isin([2024,2025])
        valid&=core.label_end.dt.year.isin([2024,2025])&np.isfinite(core.target_return)
        valid&=core.feature_source_start.le(core.datetime)&core.datetime.lt(core.label_start)&core.label_start.le(core.label_end)
        for excluded in [2023,2026]:
            valid&=~((core.feature_source_start<pd.Timestamp(excluded+1,1,1)+pd.Timedelta(days=7))&
                     (core.label_end>=pd.Timestamp(excluded,1,1)-pd.Timedelta(days=7)))
        core=core.loc[valid].sort_values(['datetime','symbol'],kind='stable')
        signature=hashlib.sha256(pd.util.hash_pandas_object(core[['symbol','datetime','feature_source_start']],index=False).to_numpy().tobytes()).hexdigest()
        for profile in cfg['t_model_profiles']:
            meta=json.loads((fold/f'inner_t_h{h}_{profile}/model_metadata.json').read_text())
            assert meta['training_rows']==len(core)==meta['fitted_rows']
            assert meta['feature_source_sha256']==signature
            output['core_row_checks'].append({'horizon':h,'profile':profile,'rows':len(core),'source_sha256':signature,'match':True})
        core=core.merge(current.drop(columns=['trade_week']),on=['datetime','symbol'],validate='one_to_one').merge(market,on='datetime',validate='many_to_one')
        support=json.loads((fold/f'support_h{h}.json').read_text())
        planned=planned_recovery_times(inner.datetime,h)
        known=feature_support_mask(inner,support)&planned.notna()&planned.dt.strftime('%H:%M').le(cfg['latest_planned_recovery_time'])
        actual_inner=inner.loc[known].merge(values,on=['datetime','symbol'],how='left',validate='one_to_one').merge(market,on='datetime',validate='many_to_one')
        populations={'core_actual_training':core,'core_2024':core.loc[core.datetime.dt.year.eq(2024)],
                     'core_2025':core.loc[core.datetime.dt.year.eq(2025)],
                     'inner_current_domain':actual_inner,'inner_static_slot_proxy':actual_inner.loc[actual_inner.static_slot_eligible]}
        entry={'horizon':h,'populations':{},'clock_groups':[]}
        for name,frame in populations.items():
            entry['populations'][name]=describe(frame)
            for clock,group in frame.groupby(frame.datetime.dt.strftime('%H:%M')):
                entry['clock_groups'].append({'population':name,'clock':clock,**describe(group)})
        output['horizons'].append(entry)
        print(f'h{h}:真实核心{len(core):,}，内部当前域{len(actual_inner):,}；3模型来源hash通过。',flush=True)
        del values,market,core,actual_inner,populations;gc.collect()
    del labels,current,inner;gc.collect()
    output['target_contract']={'target':'recovery_price/execution_price-1 raw stock return',
        'units':'无量纲收益，乘10000才为bp；特征和目标没有被误乘100/10000的证据',
        'fit_weighting':'fit_model直接model.fit(x,target)，未传sample_weight；每股每固定信号等权',
        'market_forward_mean':'仅诊断标签：同datetime+horizon全部具有有限标签的本地主板股票等权未来收益',
        'selected_clock_variance':'选中队列按datetime目标均值的加权组间平方和/目标总中心化平方和；仅描述同刻共同波动',
        'residual':'个股raw未来收益减本地主板等权未来收益；仅诊断，未作新模型拟合'}
    output['limits']=['只看首折core2024/25与inner2026；不读取任何外层收益，不反选模型或方向。',
        '原生gain为训练目标上的分裂增益，不是因果解释或独立验证贡献。',
        '未来市场均值只构成目标诊断；不能把实际未来市场收益当作推断输入或可交易收益。',
        '原主板均值非市值权重指数，未来标签缺失股票不参与该时刻统计，贡献股数留在聚合输入。',
        '所有相关性和均值为描述统计；同刻股票/重叠未来窗口相关，不能按行数宣称独立显著性。',
        '残差预测不等于持有原股票T的绝对收益；未经验证不能将市场项真实未来值加回或假定已对冲。']
    output['input_files']=[{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(set(inputs))]
    output['script_sha256']=sha(Path(__file__));output['elapsed_seconds']=time.monotonic()-started
    (OUTPUT/'inner_target_diagnostic_v7.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'描述统计完成，耗时{output["elapsed_seconds"]:.2f}秒；缓存已释放。',flush=True)


if __name__=='__main__':main()
