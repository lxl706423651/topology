"""独立审计v8核心OOF、已知训练域、严格purge后残差供体及模型，不读取外层收益。

从quant根目录运行：python scripts/audit_weekly_t_train_domain.py --report-dir reports/...。
--test-years 2023可先完整审首折；--smoke必须与冻结的小样本配置一致。只新增
independent_residual_audit.json/md，不拟合模型，不改原CSV、权重、缓存或旧版审计。
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import joblib
import lightgbm
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.audit_weekly_t_short_horizon import (
    audit_model as audit_raw_model, audit_outer_copy, audit_support, file_digest, frame_digest,
    independent_training_rows, interval_overlap, read_json, verify_cache_inputs, verify_sources as verify_frozen_sources,
)

CONTEXT={"pred_return_rank":"f_weekly_pred_return_rank","pred_amplitude_rank":"f_weekly_pred_amplitude_rank",
         "score":"f_weekly_score","rank":"f_weekly_selection_rank"}
TRACE=["source_model_training_years","source_model_sha256"]
KEYS=["trade_week","symbol"]


def verify_sources(report: Path) -> dict:
    """v8以当前完整源码首次冻结，全部声明核心源及公共接口与快照须严格相同，不沿用旧并发例外。"""
    checked=verify_frozen_sources(report)
    checked['current_full_file_match']=all(item['current_sha256']==item['expected_sha256'] for item in checked['files'].values())
    return checked


def save_progress(report: Path,result: dict) -> None:
    """原子更新自有审计文件；失败也保留已完成模型和隔离证据。"""
    destination=report/'independent_residual_audit.json'
    temporary=destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temporary.replace(destination)
    lines=['# v8 同刻合法队列残差模型独立审计','',
           f"请求范围完成：{result.get('complete',False)}；全部通过：{result.get('passed',False)}。",
           '只读模型、数据和选股来源；不读取外层收益、不训练或重新选参。', '',
           '|周selector|训练年|目标年|样本SHA|来源SHA|OOF排名及选择|', '|---|---|---:|---|---|---|']
    for name,item in result.get('selectors',{}).items():
        lines.append(f"|{name}|{item['train_years']}|{item['target_year']}|{item['sample_hash_matches']}|{item['source_hash_matches']}|{item['oof_predictions_and_selections_match']}|")
    lines+=['','|T模型|训练行|样本SHA|来源SHA|隔离重叠|原生权重一致|','|---|---:|---|---|---:|---|']
    for name,item in result.get('models',{}).items():
        lines.append(f"|{name}|{item['training_rows']}|{item['training_sample_hash_matches']}|{item['feature_source_hash_matches']}|{item['training_overlap_rows']}|{item['native_joblib_predictions_match']}|")
    lines+=['', 'OOF目标年只由另一个核心年训练的selector生成。周预测原始四周来源与盘中原来源取最早值，参数训练年与权重SHA另行逐行核对。',
            '较早年份使用较晚核心年训练的参数仍属于回顾性交叉拟合；不能称为当时可部署或全新未接触测试。',
            '小样本内外层上下文保留v4原全池冻结排名，仅按股票过滤，绝不在小池中重排该排名。',
            f"\n审计脚本SHA：{result.get('audit_script_sha256','')}"]
    lines+=['','|训练域|OOF输入行|保留行|期限排除|整手不足|输入及审计SHA|',
            '|---|---:|---:|---:|---:|---|']
    for name,item in result.get('training_domains',{}).items():
        lines.append(f"|{name}|{item['input_rows']}|{item['kept_rows']}|{item['failed_condition_counts']['recovery_deadline']}|{item['failed_condition_counts']['static_t_below_one_lot']}|{item['passed']}|")
    lines+=['','训练域只根据既有OOF周前收、固定初始槽位预算、滑点、两层整手和确定性同日恢复期限重建；过滤后才连接未来标签并执行严格来源purge。静态整手资格不能保证真实现金/容量或成交。']
    lines+=['','残差均值由严格purge后的完整同刻合法OOF供体计算；不足两股只保留审计。独立重建全部供体/残差/来源SHA、残差特征选择及raw/残差拟合样本SHA，并核对所有内外层预测的市场项精确为0；未拟合模型、未把未来均值加回。']
    if result.get('error'):lines+=['','审计错误：'+result['error']]
    destination=report/'independent_residual_audit.md'
    temporary=destination.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8');temporary.replace(destination)


def parse_frame(path: Path) -> pd.DataFrame:
    """按明确列名恢复CSV时间字段，保持记录顺序，不筛预测或未来标签。"""
    frame=pd.read_csv(path)
    for column in ['datetime','date','trade_week','feature_week','signal_datetime','entry_date','exit_date',
                   'feature_source_start','intraday_feature_source_start','weekly_context_feature_source_start',
                   'weekly_context_signal_datetime','planned_recovery_time','weekly_reference_source_start',
                   'weekly_reference_signal_datetime','planned_recovery_deadline']:
        if column in frame:frame[column]=pd.to_datetime(frame[column],format='mixed')
    return frame


def same_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """允许CSV的末位浮点序列化差异；日期、代码、顺序与行数必须精确一致。"""
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),
                                  check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)


def independent_weekly_training(frame: pd.DataFrame,years: list[int],excluded: list[int],cfg: dict) -> pd.DataFrame:
    """独立重建周标签训练资格和原排序索引；不调用生产select_training_rows。"""
    source=frame.loc[frame.feature_week_complete.eq(True)].reset_index(drop=True)
    times={key:pd.to_datetime(source[key],errors='coerce') for key in
           ['feature_source_start','feature_week','signal_datetime','label_start','label_end']}
    eligible=np.ones(len(source),dtype=bool)
    for values in times.values():eligible &= values.dt.year.isin(years).to_numpy()
    eligible &= ((times['feature_source_start']<=times['signal_datetime'])
                 &(times['signal_datetime']<times['label_start'])&(times['label_start']<=times['label_end'])).to_numpy()
    eligible &= np.isfinite(pd.to_numeric(source.target_return,errors='coerce'))
    amplitude=pd.to_numeric(source.target_amplitude,errors='coerce')
    eligible &= np.isfinite(amplitude)&amplitude.ge(0)
    for column in ['trade_week_complete','label_available']:
        if column in source:eligible &= source[column].eq(True).to_numpy()
    overlaps,_=interval_overlap(source,excluded,int(cfg['embargo_days']))
    return source.loc[eligible&~overlaps].sort_values(['signal_datetime','symbol'],kind='stable').copy()


def audit_parameters(path: Path,cfg: dict,task: str,profile: str) -> bool:
    """独立合成约定超参，并逐个核对实际joblib估计器和持久配置。"""
    parameters={'n_estimators':160,'learning_rate':.04,'max_depth':4,'num_leaves':15,'min_child_samples':50,
                'colsample_bytree':.85,'subsample':.85,'subsample_freq':1,'reg_lambda':8.,'reg_alpha':.1,
                'random_state':int(cfg.get('random_state',42)),'verbosity':-1}
    parameters.update(cfg['weekly_model' if task=='weekly' else 't_model'])
    if task=='weekly':parameters['objective']='regression'
    else:
        parameters.update({'objective':'huber' if profile=='huber_depth4' else 'regression',
                           'max_depth':6 if profile=='l2_depth6' else 4,
                           'num_leaves':31 if profile=='l2_depth6' else 15})
        if profile=='huber_depth4':parameters['alpha']=float(cfg['t_huber_alpha'])
    parameters.update(n_jobs=max(1,int(cfg['workers'])),device_type='cpu',deterministic=True,force_col_wise=True)
    metadata=read_json(path/'model_metadata.json');bundle=joblib.load(path/'model_bundle.joblib')
    assert metadata['model_parameters']==parameters
    assert read_json(path/'resolved_config.json')==metadata['resolved_config']
    for model in bundle['models'].values():
        assert all(model.get_params()[key]==value for key,value in parameters.items())
    for target,model in bundle['models'].items():
        native=lightgbm.Booster(model_file=str(path/f'{target}_model.txt'))
        assert native.model_to_string()==model.booster_.model_to_string(), '全部树/叶权重与原生保存不一致'
    return True


def audit_selector(path: Path,universe: pd.DataFrame,target_year: int,years: list[int],excluded: list[int],cfg: dict) -> tuple:
    """独立训练样本SHA+真实native预测，重推全目标周排名和选择；不查看未来标签资格。"""
    metadata=read_json(path/'model_metadata.json');bundle=joblib.load(path/'model_bundle.joblib')
    assert metadata['train_years']==years and metadata['excluded_years']==excluded
    assert metadata['strict_feature_provenance'] is True and metadata['task']==bundle['task']=='weekly'
    assert metadata==bundle['metadata']
    features=metadata['feature_names']
    assert features==bundle['features']==(path/'feature_list.txt').read_text().splitlines()
    assert all(name.startswith('f_') for name in features) and not set(features)&set(cfg['excluded_model_features'])
    training=independent_weekly_training(universe,years,excluded,cfg)
    assert len(training)==metadata['training_rows']
    cap=int(metadata['training_row_cap']);assert cap==int(cfg['weekly_max_train_rows'])
    fitted=training.sample(n=cap,random_state=int(cfg['random_state'])).sort_index() if cap and len(training)>cap else training
    numeric=fitted[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    targets=pd.DataFrame({'return':training.groupby('trade_week').target_return.rank(pct=True),
                          'amplitude':training.groupby('trade_week').target_amplitude.rank(pct=True)})
    sample_sha=frame_digest(pd.concat([fitted[['symbol','signal_datetime']],numeric,targets.loc[fitted.index]],axis=1))
    source_sha=frame_digest(fitted[['symbol','signal_datetime','feature_source_start']])
    assert sample_sha==metadata['training_sample_sha256'] and source_sha==metadata['feature_source_sha256']
    assert len(fitted)==metadata['fitted_rows']
    overlap,counts=interval_overlap(training,excluded,int(cfg['embargo_days']));assert not overlap.any()
    # 只有已知周日历/已完成特征周参与OOF目标资格，target_return等未来列完全不读取。
    allowed=(universe.entry_date.dt.year.eq(target_year)&universe.exit_date.dt.year.eq(target_year)
             &universe.entry_date.lt(universe.exit_date)&universe.trade_week.dt.year.eq(target_year)
             &universe.feature_week_complete.eq(True))
    inference=universe.loc[allowed].copy()
    x=inference[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    predictions=inference[['symbol','feature_week','signal_datetime','trade_week']].copy()
    native_checks={}
    for target in ['return','amplitude']:
        native=lightgbm.Booster(model_file=str(path/f'{target}_model.txt'))
        model=bundle['models'][target]
        assert native.feature_name()==model.booster_.feature_name()==features
        assert native.num_trees()==model.booster_.num_trees()
        a=np.asarray(model.predict(x,num_threads=1));b=np.asarray(native.predict(x,num_threads=1))
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        predictions['pred_'+target]=a
        native_checks[target]={'rows':len(x),'max_abs_error':float(np.max(np.abs(a-b))) if len(x) else 0.,
                               'native_sha256':file_digest(path/f'{target}_model.txt'),'trees':native.num_trees()}
    predictions['pred_return_rank']=predictions.groupby('trade_week').pred_return.rank(pct=True)
    predictions['pred_amplitude_rank']=predictions.groupby('trade_week').pred_amplitude.rank(pct=True)
    predictions['score']=cfg['weekly_return_weight']*predictions.pred_return_rank+cfg['weekly_amplitude_weight']*predictions.pred_amplitude_rank
    predictions=predictions.sort_values(['trade_week','score','symbol'],ascending=[True,False,True],kind='stable')
    predictions['rank']=predictions.groupby('trade_week').cumcount()+1
    predictions['selected']=predictions['rank']<=cfg['top_n']
    predictions=predictions.reset_index(drop=True)
    same_frame(predictions,parse_frame(path/'oof_weekly_predictions.csv'))
    selected=predictions.loc[predictions.selected].copy()
    model_sha=file_digest(path/'model_bundle.joblib')
    selected[TRACE[0]]=json.dumps(years,separators=(',',':'));selected[TRACE[1]]=model_sha
    same_frame(selected,parse_frame(path/'oof_weekly_selections.csv'))
    result={'passed':True,'target_year':target_year,'train_years':years,'excluded_years':excluded,
            'training_rows':len(training),'fitted_rows':len(fitted),'feature_names':features,
            'sample_sha256':sample_sha,'source_sha256':source_sha,'sample_hash_matches':True,'source_hash_matches':True,
            'training_overlap_rows':int(overlap.sum()),'overlap_by_excluded_year':counts,
            'oof_predictions_and_selections_match':True,'target_rows':len(inference),'selected_stockweeks':len(selected),
            'missing_future_return_labels_still_predicted':int(inference.target_return.isna().sum()),
            'model_bundle_sha256':model_sha,'selected_csv_sha256':file_digest(path/'oof_weekly_selections.csv'),
            'native_checks':native_checks,'model_parameters_match':audit_parameters(path,cfg,'weekly','weekly_dual_rank')}
    return selected,result


def independent_context(minutes: pd.DataFrame,predictions: pd.DataFrame,universe: pd.DataFrame,cfg: dict) -> pd.DataFrame:
    """独立按股周连接已知排名与真实四周来源，保留原minute索引/顺序和所有未来字段。"""
    source=universe[KEYS+['signal_datetime','feature_source_start']].rename(
        columns={'signal_datetime':'weekly_context_signal_datetime','feature_source_start':'weekly_context_feature_source_start'})
    context=predictions[KEYS+list(CONTEXT)+TRACE].merge(source,on=KEYS,validate='one_to_one')
    assert len(context)==len(predictions)
    for name,column in CONTEXT.items():
        context[column]=context[name]/cfg['top_n'] if name=='rank' else context[name]
    context=context.drop(columns=list(CONTEXT))
    result=minutes.assign(_order=np.arange(len(minutes))).merge(context,on=KEYS,how='left',validate='many_to_one',indicator=True)
    assert result._merge.eq('both').all()
    result['intraday_feature_source_start']=result.feature_source_start
    assert result.weekly_context_signal_datetime.lt(result.datetime).all()
    assert result.weekly_context_feature_source_start.le(result.weekly_context_signal_datetime).all()
    result['feature_source_start']=np.minimum(result.intraday_feature_source_start.to_numpy(),result.weekly_context_feature_source_start.to_numpy())
    result=result.sort_values('_order',kind='stable').drop(columns=['_order','_merge'])
    result.index=minutes.index.copy()
    return result


def reconstruct_cohort(universe: pd.DataFrame,intraday: pd.DataFrame,core: list[int],excluded: list[int],
                       cfg: dict,fold_dir: Path,result: dict) -> pd.DataFrame:
    """从实际selector重新生成核心训练队列，并逐值对比持久lineage及上下文摘要。"""
    directory=fold_dir/'oof_weekly_cohort';metadata=read_json(directory/'cohort_metadata.json')
    assert metadata==read_json(fold_dir/'t_training_cohort_manifest.json')
    available=sorted(set(universe.trade_week.dt.year)|set(intraday.datetime.dt.year)|set(core)|set(excluded))
    assert metadata['core_years']==core and metadata['excluded_years']==excluded and metadata['available_years']==available
    assert metadata['selector_count']==2 and metadata['weekly_context_features']==list(CONTEXT.values())
    assert metadata['source_sha256']=={'weekly_t_cohort.py':file_digest(ROOT/'quant_backtest/weekly_t_cohort.py')}
    assert metadata['resolved_config']==cfg
    key_columns=KEYS+['feature_week','signal_datetime','feature_source_start','entry_date','exit_date']
    assert frame_digest(universe[key_columns])==metadata['weekly_source_keys_sha256']
    parts=[]
    for target_year in core:
        train=[year for year in core if year!=target_year];denied=[year for year in available if year not in train]
        path=directory/f'selector_{target_year}'
        selected,check=audit_selector(path,universe,target_year,train,denied,cfg)
        result['selectors'][str(path.relative_to(fold_dir.parent))]=check
        stored=next(row for row in metadata['selectors'] if row['target_year']==target_year)
        assert stored['source_model_training_years']==train and stored['excluded_years']==denied
        assert stored['source_model_sha256']==check['model_bundle_sha256']
        assert stored['selected_source_sha256']==check['selected_csv_sha256']
        assert stored['predicted_rows']==check['target_rows'] and stored['selected_stockweeks']==len(selected)
        dates=universe[KEYS+['entry_date','exit_date']]
        selected_dates=selected[KEYS].merge(dates,on=KEYS,validate='one_to_one')
        positions=intraday[KEYS+['datetime']].assign(_position=np.arange(len(intraday))).merge(selected_dates,on=KEYS,validate='many_to_one')
        day=positions.datetime.dt.normalize();positions=positions.loc[day.gt(positions.entry_date)&day.lt(positions.exit_date)]
        current=intraday.iloc[positions._position.to_numpy()].copy()
        current=independent_context(current,selected,universe,cfg)
        current['cohort_target_year']=target_year
        assert len(current)==stored['cohort_rows_before_t_label_purge']
        assert current[TRACE[0]].eq(json.dumps(train,separators=(',',':'))).all()
        assert current[TRACE[1]].eq(check['model_bundle_sha256']).all()
        parts.append(current)
        print(f"{fold_dir.name}/selector_{target_year}: {len(current):,}分钟行，另一核心年{train}，OOF逐值一致",flush=True)
    training=pd.concat(parts,ignore_index=True).sort_values(['datetime','symbol'],kind='stable').reset_index(drop=True)
    trace=['datetime','symbol','feature_source_start',*CONTEXT.values(),*TRACE]
    same_frame(training[trace+['cohort_target_year']],parse_frame(directory/'cohort_lineage.csv'))
    signature=frame_digest(training[trace])
    assert signature==metadata['cohort_context_sha256'] and len(training)==metadata['training_rows_before_t_label_purge']
    result['cohorts'][fold_dir.name]={'passed':True,'rows':len(training),'cohort_context_sha256':signature,
        'min_raw_source_and_four_week_context_verified':True,'row_lineage_matches':True,'oof_not_filtered_by_future_labels':True}
    return training


def audit_frozen_context(fold_dir: Path,stage: str,universe: pd.DataFrame,intraday: pd.DataFrame,
                         symbols: set,years: list[int],cfg: dict) -> tuple[pd.DataFrame,dict]:
    """按本折v4原文件注入上下文；小样本只过滤股票/周，绝不重新计算原全池排名。"""
    source=Path(cfg['frozen_selection_report_dir'])/fold_dir.name
    model_dir=source/('inner_weekly_model' if stage=='validation' else 'weekly_model')
    selection_name='validation_weekly_selections.csv' if stage=='validation' else 'weekly_selections.csv'
    metadata=read_json(model_dir/'model_metadata.json');bundle=joblib.load(model_dir/'model_bundle.joblib')
    available=sorted(set(universe.trade_week.dt.year)|set(universe.entry_date.dt.year.dropna()))
    excluded=[year for year in available if year not in years]
    assert metadata['train_years']==years and metadata['excluded_years']==excluded
    assert metadata==bundle['metadata'] and metadata['strict_feature_provenance'] is True
    assert bundle['features']==metadata['feature_names']==(model_dir/'feature_list.txt').read_text().splitlines()
    assert all(name.startswith('f_') for name in bundle['features']) and 'f_history_days' not in bundle['features']
    original=parse_frame(source/selection_name)
    selected=original.loc[original.symbol.isin(symbols)].copy() if cfg['max_symbols'] else original.copy()
    if cfg['smoke_weeks']:
        weeks=sorted(selected.trade_week.unique())[:int(cfg['smoke_weeks'])]
        selected=selected.loc[selected.trade_week.isin(weeks)]
    selected=selected.reset_index(drop=True)
    assert len(selected)>0 and not selected.duplicated(KEYS).any()
    if (fold_dir/selection_name).exists():same_frame(selected,parse_frame(fold_dir/selection_name))
    model_sha=file_digest(model_dir/'model_bundle.joblib')
    selected[TRACE[0]]=json.dumps(years,separators=(',',':'));selected[TRACE[1]]=model_sha
    # 只连接同股同周的周中日期。选股来自预先冻结CSV，未来标签不参与行资格。
    minutes=intraday.merge(selected[KEYS+['entry_date','exit_date']],on=KEYS,how='inner',validate='many_to_one')
    minutes=minutes.loc[minutes.date.gt(minutes.entry_date)&minutes.date.lt(minutes.exit_date)].copy()
    context=independent_context(minutes,selected,universe,cfg)
    columns=['datetime','symbol','feature_source_start',*CONTEXT.values(),*TRACE]
    expected={'source_model_path':str((model_dir/'model_bundle.joblib').resolve()),
              'source_model_sha256':model_sha,'source_metadata_sha256':file_digest(model_dir/'model_metadata.json'),
              'source_model_training_years':years,'excluded_years':excluded,'context_rows':len(context),
              'context_features':list(CONTEXT.values()),'context_inputs_sha256':frame_digest(context[columns])}
    manifest_name='validation_weekly_context_manifest.json' if stage=='validation' else 'test_weekly_context_manifest.json'
    stored_manifest_present=(fold_dir/manifest_name).exists()
    if stored_manifest_present:assert expected==read_json(fold_dir/manifest_name)
    elif stage=='validation':raise FileNotFoundError(fold_dir/manifest_name)
    # 核对实际原生和bundle权重，但不在截小的当前股池重新生成冻结v4排名。
    sample=universe.loc[universe.trade_week.isin(selected.trade_week)&universe.symbol.isin(selected.symbol)]
    sample=sample[bundle['features']].head(128).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    native_checks={}
    for target,model in bundle['models'].items():
        native=lightgbm.Booster(model_file=str(model_dir/f'{target}_model.txt'))
        assert native.feature_name()==bundle['features'] and native.num_trees()==model.booster_.num_trees()
        assert native.model_to_string()==model.booster_.model_to_string()
        a=np.asarray(native.predict(sample,num_threads=1));b=np.asarray(model.predict(sample,num_threads=1))
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        native_checks[target]={'native_sha256':file_digest(model_dir/f'{target}_model.txt'),'all_tree_weights_match':True,
                               'rows':len(sample),'max_abs_error':float(np.max(np.abs(a-b))) if len(sample) else 0.}
    return context,{'passed':True,'source_selection_sha256':file_digest(source/selection_name),
        'original_full_pool_selected_stockweeks':len(original),'used_selected_stockweeks':len(selected),
        'small_scope_re_ranking_performed':False,'context':expected,
        'persisted_context_manifest_present':stored_manifest_present,'native_checks':native_checks}


def audit_saved_t_predictions(path: Path,model_dir: Path,context: pd.DataFrame,cfg: dict) -> dict:
    """独立重载残差树，市场项固定0；逐行核对推断未按未来标签或供体数删行。"""
    bundle=joblib.load(model_dir/'model_bundle.joblib')
    eligible=context.signal_bar_tradable.eq(True) if cfg['exclude_flat_signal_bars'] else np.ones(len(context),dtype=bool)
    current=context.loc[eligible]
    x=current[bundle['features']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    predicted=current[['datetime','symbol']].copy()
    predicted['prediction']=bundle['models']['return'].predict(x,num_threads=1) if len(x) else np.array([])
    predicted['residual_prediction']=predicted.prediction
    predicted['market_prediction']=0.0
    predicted=predicted.sort_values(['datetime','symbol'],kind='stable').reset_index(drop=True)
    stored=parse_frame(path);same_frame(predicted,stored)
    return {'passed':True,'rows':len(predicted),'predictions_file_sha256':file_digest(path),
            'max_abs_error':float(np.max(np.abs(predicted.prediction-stored.prediction))) if len(predicted) else 0.,
            'market_prediction_exactly_zero':bool(stored.market_prediction.eq(0).all()),
            'raw_future_or_cohort_mean_added_back':False,
            'all_current_signal_eligible_rows_preserved':len(predicted)==len(current)}


def reconstruct_residual_targets(frame: pd.DataFrame,years: list[int],excluded: list[int],cfg: dict) -> tuple:
    """独立先做完整原始标签purge，再构造同刻供体均值；未来目标永不变成f_*列。"""
    donors=independent_training_rows(frame,years,excluded,cfg)
    assert not donors.duplicated(['datetime','symbol']).any()
    assert not donors.groupby('datetime')[['label_start','label_end']].nunique().gt(1).any().any()
    raw=pd.to_numeric(donors.target_return,errors='raise').astype(float)
    means=raw.groupby(donors.datetime).transform('mean')
    counts=raw.groupby(donors.datetime).transform('size')
    keep=counts.ge(int(cfg['t_residual_min_group_size']))
    residual=raw-means
    decomposition=donors[['datetime','symbol']].copy()
    decomposition['raw_target_return']=raw
    decomposition['cohort_mean_return']=means
    decomposition['residual_target_return']=residual
    decomposition['cohort_size']=counts.astype(np.int64)
    for column in ['feature_source_start','label_start','label_end']:
        decomposition[column]=donors[column]
    decomposition['kept_for_training']=keep
    training=donors.loc[keep].copy()
    training['target_return']=raw.loc[keep]
    training['target_residual_return']=residual.loc[keep]
    assert len(training)>=int(cfg['min_training_rows'])
    assert not interval_overlap(decomposition,excluded,int(cfg['embargo_days']))[0].any()
    groups=decomposition.groupby('datetime').residual_target_return.sum()
    np.testing.assert_allclose(groups.to_numpy(),0.0,atol=1e-12,rtol=0)
    return training,decomposition


def reconstruct_residual_feature_selection(training: pd.DataFrame,cfg: dict) -> tuple:
    """用残差标签独立重算覆盖率、样本秩相关和贪心去相关，防止误用raw标签选特征。"""
    names=sorted(name for name in training if name.startswith('f_') and name not in cfg['excluded_model_features'])
    values=training[names].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    selected_indices=training.sample(n=min(int(cfg['feature_selection_sample_rows']),len(training)),random_state=int(cfg['random_state'])).index
    sample=values.loc[selected_indices]
    coverage=values.notna().mean();diversity=sample.nunique(dropna=True)
    eligible=[name for name in names if coverage[name]>=float(cfg['feature_min_coverage']) and diversity[name]>1]
    target_rank=training.loc[selected_indices,'target_residual_return'].rank(pct=True)
    relevance=pd.Series(0.0,index=names)
    if target_rank.nunique()>1 and eligible:
        relevance.loc[eligible]=sample[eligible].rank(pct=True).corrwith(target_rank).abs().fillna(0)
    ordered=sorted(eligible,key=lambda name:(-float(relevance[name]),name))
    corr=sample[ordered].corr(method='spearman')
    selected=[];records=[]
    for name in ordered:
        prior=corr.loc[name,selected].abs().dropna()
        maximum=float(prior.max()) if len(prior) else 0.0
        if maximum>float(cfg['feature_max_correlation']):reason='correlation_limit'
        elif len(selected)>=int(cfg['t_max_features']):reason='feature_limit'
        else:reason='selected';selected.append(name)
        records.append({'feature':name,'coverage':coverage[name],'target_rank_relevance':relevance[name],
                        'max_selected_correlation':maximum,'selected':reason=='selected','reason':reason})
    for name in names:
        if name not in eligible:
            records.append({'feature':name,'coverage':coverage[name],'target_rank_relevance':relevance[name],
                            'max_selected_correlation':0.0,'selected':False,
                            'reason':'coverage' if coverage[name]<float(cfg['feature_min_coverage']) else 'constant'})
    assert selected
    return selected,pd.DataFrame(records)


def audit_residual_decomposition(path: Path,training: pd.DataFrame,decomposition: pd.DataFrame,cfg: dict) -> dict:
    """逐值核全供体表及标签/来源摘要；singleton必须留审计且不得影响保留训练样本。"""
    metadata=read_json(path/'model_metadata.json')
    stored=pd.read_csv(path/'target_decomposition.csv')
    for column in ['datetime','feature_source_start','label_start','label_end']:
        stored[column]=pd.to_datetime(stored[column])
    same_frame(decomposition,stored)
    keep=decomposition.kept_for_training
    group_sizes=decomposition.groupby('datetime').cohort_size.first()
    hashes={
        'raw_training_target_sha256':frame_digest(decomposition[['symbol','datetime','raw_target_return']]),
        'residual_training_target_sha256':frame_digest(decomposition.loc[keep,['symbol','datetime','residual_target_return']]),
        'target_decomposition_sha256':frame_digest(decomposition),
        'donor_source_sha256':frame_digest(decomposition[['symbol','datetime','feature_source_start','label_start','label_end']]),
        'feature_source_full_training_sha256':frame_digest(training[['symbol','datetime','feature_source_start']]),
        'target_decomposition_file_sha256':file_digest(path/'target_decomposition.csv')}
    values={**hashes,'target_decomposition_columns':list(decomposition),
        't_target_mode':'same_clock_cohort_residual_zero_market','t_market_prior':0.0,'t_residual_min_group_size':2,
        'raw_training_rows':len(decomposition),'training_rows':len(training),'donor_rows':len(decomposition),
        'donor_clock_count':len(group_sizes),'training_clock_count':training.datetime.nunique(),
        'singleton_rows_excluded':int((~keep).sum()),'singleton_clock_count':int(group_sizes.lt(2).sum()),
        'donor_group_size_min':int(group_sizes.min()),'donor_group_size_max':int(group_sizes.max()),
        'donor_group_size_median_over_clocks':float(group_sizes.median()),
        'max_absolute_donor_clock_mean_residual':float(decomposition.groupby('datetime').residual_target_return.mean().abs().max()),
        'training_start':str(training.datetime.min()),'training_end':str(training.datetime.max()),
        'label_start_min':str(training.label_start.min()),'label_end_max':str(training.label_end.max()),
        'feature_source_start_min':str(training.feature_source_start.min()),'raw_label_unchanged':True,
        'source_sha256':{name:file_digest(ROOT/'quant_backtest'/name) for name in ['weekly_t_residual.py','weekly_t_models.py']}}
    for name,value in values.items():
        assert metadata[name]==value,('residual metadata mismatch',name,metadata[name],value)
    split=pd.read_csv(path/'training_split_audit.csv').set_index('reason').excluded_rows
    assert split['retained_training_rows']==len(decomposition)
    assert split['residual_singleton_rows_excluded']==int((~keep).sum())
    assert split['retained_residual_training_rows']==len(training)
    return {'passed':True,**values,'full_donor_values_match':True,'support_and_model_use_same_kept_rows':True,
            'group_means_computed_after_strict_purge_before_sampling':True}


def audit_model(path: Path,training: pd.DataFrame,cfg: dict,years: list[int],excluded: list[int],
                horizon: int,profile: str,decomposition: pd.DataFrame,feature_selection: tuple) -> dict:
    """残差实目标核账后复用独立原生重载核对；另核原raw拟合摘要及残差特征选择。"""
    effective=training.copy();effective['target_return']=effective.target_residual_return
    result=audit_raw_model(path,effective,cfg,years,excluded,horizon,profile)
    result['residual_targets']=audit_residual_decomposition(path,training,decomposition,cfg)
    metadata=read_json(path/'model_metadata.json');features,audit=feature_selection
    assert features==metadata['feature_names']
    same_frame(audit,pd.read_csv(path/'feature_selection_audit.csv'))
    cap=int(cfg['t_max_train_rows'])
    fitted=training.sample(n=cap,random_state=int(cfg['random_state'])).sort_index() if cap and len(training)>cap else training
    numeric=fitted[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    raw_sha=frame_digest(pd.concat([fitted[['symbol','datetime']],numeric,
                                   pd.DataFrame({'return':fitted.target_return})],axis=1))
    assert raw_sha==metadata['raw_training_sample_sha256']
    bundle=joblib.load(path/'model_bundle.joblib')
    assert bundle['target_mode']=='same_clock_cohort_residual_zero_market'
    assert set(bundle['models'])=={'return'}
    result.update(raw_training_sample_sha256=raw_sha,raw_fitted_sample_hash_matches=True,
                  residual_based_feature_selection_matches=True,target_decomposition_matches=True,
                  market_prediction_fixed_zero=True,all_donors_strictly_core=True)
    return result


def audit_training_lineage(training: pd.DataFrame,fold_dir: Path,core: list[int],excluded: list[int]) -> dict:
    """在T真实训练行上另行检查参数年份/SHA与原始四周来源，排除内外年参数泄漏。"""
    targets=sorted(int(year) for year in training.cohort_target_year.unique())
    assert set(targets).issubset(core) and not set(targets)&set(excluded)
    for target in targets:
        part=training.loc[training.cohort_target_year.eq(target)]
        other=[year for year in core if year!=target]
        sha=file_digest(fold_dir/f'oof_weekly_cohort/selector_{target}/model_bundle.joblib')
        assert part.datetime.dt.year.eq(target).all()
        assert part[TRACE[0]].eq(json.dumps(other,separators=(',',':'))).all() and part[TRACE[1]].eq(sha).all()
        assert part.weekly_context_signal_datetime.lt(part.datetime).all()
        assert part.feature_source_start.le(part.weekly_context_feature_source_start).all()
        assert part.feature_source_start.le(part.intraday_feature_source_start).all()
    columns=['datetime','symbol','feature_source_start','cohort_target_year',*CONTEXT.values(),*TRACE]
    return {'passed':True,'training_rows':len(training),'target_years':targets,
            'training_lineage_sha256':frame_digest(training[columns]),
            'own_inner_outer_parameter_year_overlap_rows':0}


def reconstruct_known_domain(frame: pd.DataFrame,universe: pd.DataFrame,horizon: int,cfg: dict) -> tuple:
    """独立重建当前股周价格、静态两层整手及240交易分钟时钟；不调用生产过滤器。"""
    columns=['datetime','symbol','trade_week','feature_source_start']
    minutes=frame[columns].copy().reset_index(drop=True)
    weekly=universe[['trade_week','symbol','f_last_close','signal_datetime','feature_source_start']].copy()
    weekly['f_last_close']=pd.to_numeric(weekly.f_last_close,errors='raise').astype(np.float64)
    weekly=weekly.rename(columns={'f_last_close':'universe_previous_close','signal_datetime':'weekly_reference_signal_datetime',
                                  'feature_source_start':'weekly_reference_source_start'})
    assert weekly.universe_previous_close.gt(0).all() and np.isfinite(weekly.universe_previous_close).all()
    assert weekly.weekly_reference_source_start.le(weekly.weekly_reference_signal_datetime).all()
    assert weekly.weekly_reference_signal_datetime.lt(weekly.trade_week).all()
    joined=minutes.merge(weekly,on=['trade_week','symbol'],how='left',sort=False,validate='many_to_one',indicator=True)
    assert joined._merge.eq('both').all();joined=joined.drop(columns='_merge');joined.index=frame.index.copy()
    assert joined.feature_source_start.le(joined.weekly_reference_source_start).all()
    assert joined.weekly_reference_signal_datetime.lt(joined.datetime).all()
    known_sha=hashlib.sha256(pd.util.hash_pandas_object(joined,index=True).to_numpy().tobytes()).hexdigest()
    session=np.array(list(range(571,691))+list(range(781,901)),dtype=np.int64)
    positions={int(value):index for index,value in enumerate(session)};positions[570]=-1
    clocks=joined.datetime.dt.hour*60+joined.datetime.dt.minute
    assert clocks.isin(positions).all()
    assert joined.datetime.dt.second.eq(0).all() and joined.datetime.dt.microsecond.eq(0).all() and joined.datetime.dt.nanosecond.eq(0).all()
    # 成交为信号下一交易分钟；再加预定horizon，只推进固定会话索引而不看真实未来行。
    planned_positions=clocks.map(positions).to_numpy()+1+int(horizon)
    same_day=planned_positions<len(session)
    nanos=np.full(len(joined),np.iinfo(np.int64).min,dtype=np.int64)
    nanos[same_day]=joined.datetime.dt.normalize().astype('int64').to_numpy()[same_day]+session[planned_positions[same_day]]*60*10**9
    planned=pd.Series(pd.to_datetime(nanos),index=joined.index)
    hour,minute=map(int,cfg['latest_planned_recovery_time'].split(':'))
    deadline=joined.datetime.dt.normalize()+pd.Timedelta(hours=hour,minutes=minute)
    clock_ok=planned.notna()&planned.le(deadline)
    budget=float(cfg['initial_cash'])*float(cfg['base_fraction'])/int(cfg['top_n']);lot=int(cfg['lot_size'])
    base=np.floor(budget/(joined.universe_previous_close*(1+float(cfg['slippage_rate'])))/lot).astype(np.int64)*lot
    t=np.floor(base*float(cfg['short_t_fraction'])/lot).astype(np.int64)*lot
    slot_ok=t.ge(lot);keep=clock_ok&slot_ok
    reasons=np.where(same_day,np.where(clock_ok,'keep','recovery_after_deadline'),'no_same_day_planned_recovery').astype(object)
    for index in np.flatnonzero(~slot_ok.to_numpy()):
        reasons[index]='static_t_below_one_lot' if clock_ok.iloc[index] else reasons[index]+';static_t_below_one_lot'
    audit=joined.copy()
    audit['horizon_minutes']=int(horizon);audit['planned_recovery_time']=planned;audit['planned_recovery_deadline']=deadline
    audit['same_day_planned_recovery']=same_day;audit['recovery_deadline_eligible']=clock_ok
    audit['static_slot_budget']=budget;audit['static_base_shares']=base;audit['static_t_shares']=t
    audit['static_slot_eligible']=slot_ok;audit['keep']=keep;audit['drop_reason']=reasons
    resolved={'t_training_domain':'known_slot_and_recovery_deadline','horizon_minutes':int(horizon),
              **{key:float(cfg[key]) for key in ['initial_cash','base_fraction','short_t_fraction','slippage_rate']},
              **{key:int(cfg[key]) for key in ['top_n','lot_size']},'latest_planned_recovery_time':cfg['latest_planned_recovery_time']}
    metadata={'schema_version':1,'t_training_domain':'known_slot_and_recovery_deadline','config':resolved,
              'input_rows':len(frame),'kept_rows':int(keep.sum()),'dropped_rows':int((~keep).sum()),
              'drop_reason_counts':{str(key):int(value) for key,value in audit.drop_reason.value_counts().sort_index().items()},
              'failed_condition_counts':{'recovery_deadline':int((~clock_ok).sum()),'static_t_below_one_lot':int((~slot_ok).sum()),'both':int((~clock_ok&~slot_ok).sum())},
              'known_inputs_sha256':known_sha,'audit_sha256':hashlib.sha256(pd.util.hash_pandas_object(audit,index=True).to_numpy().tobytes()).hexdigest(),
              'source_sha256':{name:file_digest(ROOT/'quant_backtest'/name) for name in ['weekly_t_train_domain.py','weekly_t_short_horizon.py','weekly_t_horizon_data.py','weekly_t_engine.py']},
              'feature_source_unchanged':True,'future_columns_consumed':[]}
    return frame.iloc[np.flatnonzero(keep.to_numpy())].copy(),audit,metadata


def audit_training_domain(frame: pd.DataFrame,universe: pd.DataFrame,horizon: int,cfg: dict,path: Path) -> tuple:
    """对比全部逐行过滤记录、两类含原索引SHA、各原因和真实消费配置。"""
    kept,audit,metadata=reconstruct_known_domain(frame,universe,horizon,cfg)
    stored=read_json(path.with_suffix('.json'))
    for name,value in metadata.items():assert stored[name]==value, ('training domain metadata mismatch',name)
    same_frame(audit,parse_frame(path.with_suffix('.csv')))
    assert kept.feature_source_start.equals(frame.loc[kept.index,'feature_source_start'])
    return kept,{'passed':True,**metadata,'input_and_audit_sha_match':True,'all_rows_and_drop_reasons_match':True,
                'future_inputs_used_for_filter':False,'domain_csv_sha256':file_digest(path.with_suffix('.csv')),
                'domain_metadata_sha256':file_digest(path.with_suffix('.json'))}


def execute_audit(report: Path,smoke: bool,requested: list[int]|None,result: dict) -> None:
    """只读重建请求折；无需外层收益完成，核心权重和规则复制必须已落盘。"""
    cfg=read_json(report/'resolved_config.json')
    assert smoke==bool(cfg.get('max_symbols',0) or cfg.get('smoke_weeks',0)), '--smoke必须匹配冻结配置'
    assert cfg['strict_feature_provenance'] is True and cfg['outer_t_refit'] is False
    assert 'f_history_days' in cfg['excluded_model_features'] and cfg['top_n']==cfg['weekly_top_n']
    assert cfg['t_training_cohort']=='core_year_oof_weekly_top100_middle_days'
    assert cfg['research_version']=='weekly_t_v8_residual_zero_market' and cfg['t_training_domain']=='known_slot_and_recovery_deadline'
    assert cfg['t_target_mode']=='same_clock_cohort_residual_zero_market' and cfg['t_market_prior']==0.0
    assert cfg['t_residual_min_group_size']==2
    years=sorted(set(requested or cfg['test_years']))
    assert set(years).issubset(cfg['test_years'])
    result.update(requested_years=years,configured_years=cfg['test_years'],small_sample=smoke,
                  reused_independent_auditor_sha256=file_digest(ROOT/'scripts/audit_weekly_t_short_horizon.py'))
    result['sources']=verify_sources(report);assert result['sources']['passed']
    assert {'scripts/run_weekly_t_residual.py','quant_backtest/weekly_t_train_domain.py','quant_backtest/weekly_t_residual.py'}.issubset(result['sources']['files'])
    weekly_manifest,horizon_manifest,result['input_identity']=verify_cache_inputs(report,cfg)
    assert result['input_identity']['passed']
    folds={year:read_json(report/f'year_{year}/split.json') for year in years}
    for year in years:
        fold_dir=report/f'year_{year}'
        for horizon in cfg['horizons']:
            for profile in cfg['t_model_profiles']:
                assert (fold_dir/f'inner_t_h{horizon}_{profile}/model_metadata.json').is_file()
        result['outer_model_copies'][str(year)]=audit_outer_copy(fold_dir,cfg)
    freeze_path=report/'all_rules_frozen_before_outer_test.json'
    result['global_rule_freeze_present']=freeze_path.exists()
    if freeze_path.exists():
        freeze=read_json(freeze_path)
        assert sorted(freeze['test_years'])==sorted(cfg['test_years'])
        assert all(freeze['selected_rule_sha256'][str(year)]==result['outer_model_copies'][str(year)]['selected_rule_sha256'] for year in years)
        timing={}
        for year in years:
            folder=report/f'year_{year}'
            rule_time=(folder/'selected_rule.json').stat().st_mtime_ns
            marker_time=freeze_path.stat().st_mtime_ns
            assert rule_time<=marker_time
            prediction_path=folder/'t_predictions.csv'
            prediction_time=prediction_path.stat().st_mtime_ns if prediction_path.exists() else None
            assert prediction_time is None or marker_time<=prediction_time
            timing[str(year)]={'selected_rule_mtime_ns':rule_time,'global_freeze_mtime_ns':marker_time,
                               'outer_prediction_mtime_ns':prediction_time,'passed':True}
        result['global_freeze_observed_file_order']=timing
    save_progress(report,result)
    from quant_backtest.weekly_t_features import add_session_context_features,add_weekly_execution_features
    from quant_backtest.weekly_t_provenance import add_feature_provenance
    from quant_backtest.weekly_t_short_horizon import attach_horizon_labels

    print('独立审计读取周/日/盘中缓存及标签；不加载恢复报价，不读取外层收益',flush=True)
    cache=Path(weekly_manifest['cache_path'])
    weekly,intraday,daily=[pd.read_pickle(cache/f'{name}.pkl') for name in ['weekly','intraday','daily']]
    labels=pd.read_pickle(Path(horizon_manifest['labels_path']))
    for name,frame in [('weekly',weekly),('intraday',intraday),('daily',daily)]:assert len(frame)==weekly_manifest[name+'_rows']
    assert len(labels)==horizon_manifest['label_rows']
    all_years=sorted(int(year) for year in daily.date.dt.year.unique());symbols=set(daily.symbol)
    weekly=add_weekly_execution_features(weekly,daily);intraday=add_session_context_features(intraday)
    weekly,intraday,provenance=add_feature_provenance(weekly,intraday,daily)
    assert provenance==read_json(report/'feature_provenance_manifest.json')
    result['provenance']={'passed':True,'manifest':provenance}
    mask=(weekly.f_history_days.ge(cfg['min_history_days'])&weekly.f_last_close.ge(cfg['min_price'])
          &weekly.f_avg_daily_amount.ge(cfg['min_avg_daily_amount'])&weekly.feature_week_complete.fillna(False))
    if cfg.get('min_prior_close_auction_amount',0)>0:mask &= weekly.f_prior_close_auction_amount_min.ge(cfg['min_prior_close_auction_amount'])
    universe=weekly.loc[mask].copy()
    result['input_rows']={'weekly_universe':len(universe),'intraday':len(intraday),'labels':len(labels)}
    del weekly,daily;gc.collect()
    for year,fold in folds.items():
        train=[value for value in all_years if value!=year];inner=max(train)
        core=[value for value in train if value!=inner];excluded=sorted([year,inner])
        assert fold['test_year']==year and fold['train_years']==train and fold['inner_year']==inner and fold['core_years']==core
        assert fold['outer_t_refit'] is False and fold['forward_only']==(max(train)<year)
        fold_dir=report/f'year_{year}'
        training_source=reconstruct_cohort(universe,intraday,core,excluded,cfg,fold_dir,result)
        inner_context,context_check=audit_frozen_context(fold_dir,'validation',universe,intraday,symbols,core,cfg)
        result['frozen_contexts'][f'year_{year}/validation']=context_check
        for horizon in cfg['horizons']:
            print(f'year_{year}/h{horizon}: 独立训练资格、支持域、模型、参数来源核对',flush=True)
            domain,domain_check=audit_training_domain(training_source,universe,horizon,cfg,fold_dir/f'training_domain_h{horizon}')
            result['training_domains'][f'year_{year}/h{horizon}']=domain_check
            frame=attach_horizon_labels(domain,labels,horizon)
            del domain
            training,decomposition=reconstruct_residual_targets(frame,core,excluded,cfg)
            feature_selection=reconstruct_residual_feature_selection(training,cfg)
            name=f'year_{year}/support_h{horizon}.json'
            result['supports'][name]=audit_support(report/name,training,cfg,core,excluded)
            lineage=audit_training_lineage(training,fold_dir,core,excluded)
            for profile in cfg['t_model_profiles']:
                name=f'year_{year}/inner_t_h{horizon}_{profile}'
                check=audit_model(report/name,training,cfg,core,excluded,horizon,profile,decomposition,feature_selection)
                check['lineage']=lineage
                check['model_parameters_match']=audit_parameters(report/name,cfg,'intraday',profile)
                check['all_native_tree_weights_match']=True
                path=fold_dir/f'validation_predictions_h{horizon}_{profile}.csv'
                check['saved_inner_predictions']=audit_saved_t_predictions(path,report/name,inner_context,cfg)
                result['models'][name]=check
            save_progress(report,result)
            del frame,training,decomposition,feature_selection;gc.collect()
        outer_context,context_check=audit_frozen_context(fold_dir,'test',universe,intraday,symbols,train,cfg)
        result['frozen_contexts'][f'year_{year}/test']=context_check
        if (fold_dir/'t_predictions.csv').exists():
            result['outer_predictions'][str(year)]=audit_saved_t_predictions(fold_dir/'t_predictions.csv',fold_dir/'t_model',outer_context,cfg)
        del training_source,inner_context,outer_context;gc.collect()
        save_progress(report,result)
    result['source_files_unchanged_during_audit']=verify_sources(report)==result['sources']
    result['audit_script_unchanged']=file_digest(Path(__file__))==result['audit_script_sha256']
    result['reused_auditor_unchanged']=file_digest(ROOT/'scripts/audit_weekly_t_short_horizon.py')==result['reused_independent_auditor_sha256']
    result['complete']=(len(result['selectors'])==2*len(years) and len(result['models'])==len(years)*len(cfg['horizons'])*len(cfg['t_model_profiles'])
                        and len(result['training_domains'])==len(years)*len(cfg['horizons']))
    result['full_configured_scope_complete']=result['complete'] and years==sorted(cfg['test_years'])
    result['passed']=bool(result['complete'] and result['source_files_unchanged_during_audit'] and result['audit_script_unchanged']
        and result['reused_auditor_unchanged'] and all(value['passed'] for group in ['selectors','cohorts','models','supports','outer_model_copies','frozen_contexts','outer_predictions','training_domains'] for value in result[group].values()))
    result['strict_feature_and_parameter_year_isolation_passed']=result['passed']


def main() -> None:
    """异常也保存自有进度，不等外层完成、不修改训练产物或实验入口。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--test-years',type=int,nargs='+')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();report=args.report_dir.resolve()
    if not report.is_dir():parser.error('report-dir必须存在')
    started=time.monotonic()
    result={'report_dir':str(report),'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'audit_script_sha256':file_digest(Path(__file__)),'complete':False,'passed':False,
        'selectors':{},'cohorts':{},'models':{},'supports':{},'outer_model_copies':{},'frozen_contexts':{},'outer_predictions':{},'training_domains':{},
        'outer_performance_read':False,'model_training_performed':False,'historical_retest_not_fresh_holdout':True,
        'method':'冻结helper仅复算因果特征/来源/原始标签；样本隔离、同刻合法供体均值、残差、特征选择、抽样、权重、OOF排名、上下文和参数年份SHA独立重建'}
    try:execute_audit(report,args.smoke,args.test_years,result)
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}';result['error_traceback']=traceback.format_exc()
        print(result['error_traceback'],file=sys.stderr,flush=True)
    result['elapsed_seconds']=time.monotonic()-started;save_progress(report,result)
    print(f"审计完成={result['complete']}，通过={result['passed']}；{report/'independent_residual_audit.json'}",flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
