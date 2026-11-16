"""独立审计v6核心年OOF选股、周上下文来源及短周期T模型，不读取外层收益。

从quant根目录运行：python scripts/audit_weekly_t_cohort.py --report-dir reports/...。
--test-years 2023可先完整审首折；--smoke必须与冻结的小样本配置一致。只新增
independent_cohort_audit.json/md，不拟合模型，不改原CSV、权重、缓存或旧版审计。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
    audit_model, audit_outer_copy, audit_support, file_digest, frame_digest,
    independent_training_rows, interval_overlap, read_json, verify_cache_inputs, verify_sources,
)

CONTEXT={"pred_return_rank":"f_weekly_pred_return_rank","pred_amplitude_rank":"f_weekly_pred_amplitude_rank",
         "score":"f_weekly_score","rank":"f_weekly_selection_rank"}
TRACE=["source_model_training_years","source_model_sha256"]
KEYS=["trade_week","symbol"]


def save_progress(report: Path,result: dict) -> None:
    """原子更新自有审计文件；失败也保留已完成模型和隔离证据。"""
    destination=report/'independent_cohort_audit.json'
    temporary=destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temporary.replace(destination)
    lines=['# v6 OOF队列与模型独立审计','',
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
    if result.get('error'):lines+=['','审计错误：'+result['error']]
    destination=report/'independent_cohort_audit.md'
    temporary=destination.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8');temporary.replace(destination)


def parse_frame(path: Path) -> pd.DataFrame:
    """按明确列名恢复CSV时间字段，保持记录顺序，不筛预测或未来标签。"""
    frame=pd.read_csv(path)
    for column in ['datetime','date','trade_week','feature_week','signal_datetime','entry_date','exit_date',
                   'feature_source_start','intraday_feature_source_start','weekly_context_feature_source_start',
                   'weekly_context_signal_datetime','planned_recovery_time']:
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
    """独立按当前已知信号资格推断，并对比持久预测；不打开标签评价/收益文件。"""
    bundle=joblib.load(model_dir/'model_bundle.joblib')
    eligible=context.signal_bar_tradable.eq(True) if cfg['exclude_flat_signal_bars'] else np.ones(len(context),dtype=bool)
    current=context.loc[eligible]
    x=current[bundle['features']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    predicted=current[['datetime','symbol']].copy()
    predicted['prediction']=bundle['models']['return'].predict(x,num_threads=1) if len(x) else np.array([])
    predicted=predicted.sort_values(['datetime','symbol'],kind='stable').reset_index(drop=True)
    stored=parse_frame(path);same_frame(predicted,stored)
    return {'passed':True,'rows':len(predicted),'predictions_file_sha256':file_digest(path),
            'max_abs_error':float(np.max(np.abs(predicted.prediction-stored.prediction))) if len(predicted) else 0.}


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


def execute_audit(report: Path,smoke: bool,requested: list[int]|None,result: dict) -> None:
    """只读重建请求折；无需外层收益完成，核心权重和规则复制必须已落盘。"""
    cfg=read_json(report/'resolved_config.json')
    assert smoke==bool(cfg.get('max_symbols',0) or cfg.get('smoke_weeks',0)), '--smoke必须匹配冻结配置'
    assert cfg['strict_feature_provenance'] is True and cfg['outer_t_refit'] is False
    assert 'f_history_days' in cfg['excluded_model_features'] and cfg['top_n']==cfg['weekly_top_n']
    assert cfg['t_training_cohort']=='core_year_oof_weekly_top100_middle_days'
    years=sorted(set(requested or cfg['test_years']))
    assert set(years).issubset(cfg['test_years'])
    result.update(requested_years=years,configured_years=cfg['test_years'],small_sample=smoke,
                  reused_independent_auditor_sha256=file_digest(ROOT/'scripts/audit_weekly_t_short_horizon.py'))
    result['sources']=verify_sources(report);assert result['sources']['passed']
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
            frame=attach_horizon_labels(training_source,labels,horizon)
            training=independent_training_rows(frame,core,excluded,cfg)
            name=f'year_{year}/support_h{horizon}.json'
            result['supports'][name]=audit_support(report/name,training,cfg,core,excluded)
            lineage=audit_training_lineage(training,fold_dir,core,excluded)
            for profile in cfg['t_model_profiles']:
                name=f'year_{year}/inner_t_h{horizon}_{profile}'
                check=audit_model(report/name,training,cfg,core,excluded,horizon,profile)
                check['lineage']=lineage
                check['model_parameters_match']=audit_parameters(report/name,cfg,'intraday',profile)
                check['all_native_tree_weights_match']=True
                path=fold_dir/f'validation_predictions_h{horizon}_{profile}.csv'
                check['saved_inner_predictions']=audit_saved_t_predictions(path,report/name,inner_context,cfg)
                result['models'][name]=check
            save_progress(report,result)
            del frame,training;gc.collect()
        outer_context,context_check=audit_frozen_context(fold_dir,'test',universe,intraday,symbols,train,cfg)
        result['frozen_contexts'][f'year_{year}/test']=context_check
        if (fold_dir/'t_predictions.csv').exists():
            result['outer_predictions'][str(year)]=audit_saved_t_predictions(fold_dir/'t_predictions.csv',fold_dir/'t_model',outer_context,cfg)
        del training_source,inner_context,outer_context;gc.collect()
        save_progress(report,result)
    result['source_files_unchanged_during_audit']=verify_sources(report)==result['sources']
    result['audit_script_unchanged']=file_digest(Path(__file__))==result['audit_script_sha256']
    result['reused_auditor_unchanged']=file_digest(ROOT/'scripts/audit_weekly_t_short_horizon.py')==result['reused_independent_auditor_sha256']
    result['complete']=len(result['selectors'])==2*len(years) and len(result['models'])==len(years)*len(cfg['horizons'])*len(cfg['t_model_profiles'])
    result['full_configured_scope_complete']=result['complete'] and years==sorted(cfg['test_years'])
    result['passed']=bool(result['complete'] and result['source_files_unchanged_during_audit'] and result['audit_script_unchanged']
        and result['reused_auditor_unchanged'] and all(value['passed'] for group in ['selectors','cohorts','models','supports','outer_model_copies','frozen_contexts','outer_predictions'] for value in result[group].values()))
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
        'selectors':{},'cohorts':{},'models':{},'supports':{},'outer_model_copies':{},'frozen_contexts':{},'outer_predictions':{},
        'outer_performance_read':False,'model_training_performed':False,'historical_retest_not_fresh_holdout':True,
        'method':'冻结helper仅复算因果特征/来源/标签；样本隔离、抽样、权重、OOF排名、周中队列、上下文和参数年份SHA独立重建'}
    try:execute_audit(report,args.smoke,args.test_years,result)
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}';result['error_traceback']=traceback.format_exc()
        print(result['error_traceback'],file=sys.stderr,flush=True)
    result['elapsed_seconds']=time.monotonic()-started;save_progress(report,result)
    print(f"审计完成={result['complete']}，通过={result['passed']}；{report/'independent_cohort_audit.json'}",flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
