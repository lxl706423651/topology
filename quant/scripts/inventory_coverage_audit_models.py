"""v18独立训练核：完整原始标签、0.70/0.80筛选、真实L2权重与共同列身份。

不导入生产coverage模型/selector；仅复用旧独立来源purge和原始行情工具。
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import lightgbm
import numpy as np
import pandas as pd
from scripts.audit_dynamic_inventory_t import close, digest, next_minute, require
from scripts.audit_inventory_learning_t import exact, frame_digest, recovery_times
from scripts.audit_weekly_t_short_horizon import interval_overlap
from scripts.inventory_robust_audit_models import independent_robust_training_split as independent_training_split

VERSION='inventory_coverage_t_v18'
KEYS=['datetime','symbol']
INPUT_KEYS=KEYS+['feature_source_start','cohort_target_year','baseline_opening_shares']
PROFILES=('l2_depth4','l2_depth6')
CONTROL_PAIRS=(('source_training_inputs.pkl','training_inputs.pkl'),
    ('source_feature_manifest.csv','feature_manifest.csv'),('training_labels.csv','training_labels.csv'),
    ('training_split_audit.csv','training_split_audit.csv'))


def coverage_feature_selection(training, cfg, threshold=None):
    """独立按全训练覆盖率、固定样本秩相关及严格贪心顺序重建选择。"""
    threshold=cfg['feature_min_coverage'] if threshold is None else threshold
    require(threshold in (.7,.8), '仅核事前0.70/0.80，不扫描阈值')
    require(cfg['feature_selection_sample_rows']==20000 and cfg['random_state']==42
        and cfg['feature_max_correlation']==.95 and cfg['t_max_features']==60,'原selector非覆盖率参数改变')
    names=sorted(c for c in training if c.startswith('f_') and c not in cfg['excluded_model_features'])
    values=training[names].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    loc=training.sample(n=min(20000,len(training)),random_state=42).index
    sample=values.loc[loc]
    coverage=values.notna().mean(); diverse=sample.nunique(dropna=True)
    eligible=[name for name in names if coverage[name]>=threshold and diverse[name]>1]
    target=pd.to_numeric(training.target_return).loc[loc].rank(pct=True)
    relevance=pd.Series(0.,index=names)
    if target.nunique()>1 and eligible:
        relevance.loc[eligible]=sample[eligible].rank(pct=True).corrwith(target).abs().fillna(0.)
    order=sorted(eligible,key=lambda name:(-float(relevance[name]),name))
    correlation=sample[order].corr(method='spearman')
    selected=[];rows=[]
    for name in order:
        peers=correlation.loc[name,selected].abs().dropna()
        maximum=float(peers.max()) if len(peers) else 0.
        reason='correlation_limit' if maximum>.95 else 'feature_limit' if len(selected)>=60 else 'selected'
        if reason=='selected':selected.append(name)
        rows.append(dict(feature=name,coverage=coverage[name],target_rank_relevance=relevance[name],
            max_selected_correlation=maximum,selected=reason=='selected',reason=reason))
    for name in names:
        if name not in eligible:
            rows.append(dict(feature=name,coverage=coverage[name],target_rank_relevance=relevance[name],
                max_selected_correlation=0.,selected=False,reason='coverage' if coverage[name]<threshold else 'constant'))
    return selected,pd.DataFrame(rows)


def validate_native_parameters(native, model, parameters, features):
    """实际损失必须为L2；原生通用alpha默认字段不代表使用Huber。"""
    require(parameters.get('objective')=='regression' and 'alpha' not in parameters,'声明须原L2且无新增alpha')
    require(native.params.get('objective')=='regression' and model.booster_.params.get('objective')=='regression','实际原生损失不是L2')
    require(native.feature_name()==features==model.booster_.feature_name(),'原生特征列序不同')
    require(all(model.get_params().get(k)==v for k,v in parameters.items()),'实际拟合器参数不同')
    require(native.model_to_string()==model.booster_.model_to_string(),'原生全部树与joblib不同')


def compare_source_training(auditor,directory,source,meta):
    """.80重建产物与原L2逐值/dtype相同；新.70矩阵只要求共同列与行严格同源。"""
    proof=auditor.source_audits['learning'][1]['inputs_sha256']
    original=auditor.json(source/'model_metadata.json')
    require(proof.get(str((source/'model_metadata.json').resolve()))==digest(source/'model_metadata.json'),'原L2元数据未锚定')
    evidence={}
    for new_name,old_name in CONTROL_PAIRS:
        old,new=source/old_name,directory/new_name
        require(proof.get(str(old.resolve()))==digest(auditor.record(old)), '原训练文件未被完整独审锚定:'+old_name)
        auditor.record(new)
        actual,expected=((pd.read_pickle(new),pd.read_pickle(old)) if old_name.endswith('.pkl') else (auditor.csv(new),auditor.csv(old)))
        exact(actual,expected,'独立0.80原产物/labels/split:'+old_name,dtype=True)
        if old_name.endswith('.pkl'):require(actual.index.equals(expected.index),'0.80控制pickle索引与原训练索引不同')
        evidence[old_name]=dict(current_file=new_name,current_sha256=digest(new),source_sha256=digest(old),values_and_dtype_exact=True)
    require(original['research_version']=='inventory_learning_t_v14' and meta['profile']==original['profile']
        and meta['horizon_minutes']==original['horizon_minutes'],'原L2架构/期限不一致')
    for key in ('train_years','excluded_years','fitted_rows','training_rows','full_training_rows_sha256','training_target_sha256'):
        require(meta[key]==original[key],'原训练行/标签身份改变:'+key)
    require(meta['source_features']==original['features'] and meta['source_training_matrix_sha256']==original['training_matrix_sha256'],'原0.80选择身份错误')
    require(meta['feature_min_coverage']==.7 and meta['source_feature_min_coverage']==.8,'阈值身份错误')
    actual=pd.read_pickle(auditor.record(directory/'training_inputs.pkl'))
    old=pd.read_pickle(auditor.record(source/'training_inputs.pkl'))
    require(list(actual)==INPUT_KEYS+meta['features'] and list(old)==INPUT_KEYS+original['features'],'实际矩阵列序错误')
    common=[name for name in original['features'] if name in meta['features']]
    exact(actual[INPUT_KEYS+common],old[INPUT_KEYS+common],'原实际矩阵共同列及行',dtype=True)
    require(actual.index.equals(old.index),'新.70矩阵索引与原训练索引不同')
    left,right=meta['model_parameters'],original['model_parameters']
    require(left['objective']==right['objective']=='regression' and 'alpha' not in left,'L2损失改变')
    require({k:v for k,v in left.items() if k!='n_jobs'}=={k:v for k,v in right.items() if k!='n_jobs'},'非资源模型参数改变')
    return dict(source_model=str(source),reconstructed_source_selection_labels_and_split_exact=evidence,
        current_features=meta['features'],source_features=original['features'],common_features=common,
        added_features=[name for name in meta['features'] if name not in original['features']],
        removed_features=[name for name in original['features'] if name not in meta['features']],
        common_feature_values_and_dtype_exact=True,source_workers=right['n_jobs'],current_workers=left['n_jobs'])


def audit_coverage_model(auditor,directory,training,features,feature_audit,fold,horizon,profile,*,full_frame):
    """从完整core及raw端点核真实新矩阵、重建控制矩阵和原生权重，不重新拟合。"""
    cfg=auditor.cfg;meta=auditor.json(directory/'model_metadata.json')
    require(meta['research_version']==cfg['research_version']==VERSION and profile in PROFILES
        and meta['profile']==profile and meta['horizon_minutes']==horizon,'新覆盖模型身份错误')
    require(meta['resolved_config']=={**cfg,'t_model_profile':profile,'t_label_horizon_minutes':horizon},'模型完整cfg不一致')
    require(meta['train_years']==sorted(fold['core_years']) and meta['excluded_years']==sorted([fold['test_year'],fold['inner_year']]),'模型年份泄漏')
    require(meta['training_loss']=='regression_l2' and meta['source_l2_profile']==profile
        and meta['feature_min_coverage']==.7 and meta['source_feature_min_coverage']==.8,'原L2/覆盖率身份错误')
    require('t_huber_alpha' not in cfg and 'huber_alpha' not in meta,'Huber配置污染本轮')
    require(meta['only_selector_configuration_change']=={'feature_min_coverage':{'source':.8,'current':.7}},'非唯一覆盖率改变')
    require(all(meta[k] is False for k in ('score_is_calibrated_conditional_mean','score_is_actual_execution_pnl',
        'nan_values_filled','training_targets_clipped','tail_training_rows_removed','target_is_portfolio_execution_net_pnl','probability_gate_used')),'缺失/目标解释错误')
    require(meta['target_definition']=='reference_recovery_price/reference_next_minute_price-1'
        and meta['training_cohort']=='permanent_inventory_core_oof_opening_old_shares','原始目标/队列身份错误')
    require(cfg['t_max_train_rows']==0 and len(training)==meta['training_rows']==meta['fitted_rows'],'原训练行被抽样/删除')
    selected,split,split_meta=independent_training_split(full_frame,fold,cfg)
    exact(selected,training,'完整core训练purge',dtype=True)
    require(selected.index.equals(training.index),'特征抽样索引改变')
    exact(auditor.table(directory/'training_split_audit.csv'),split,'逐原因purge',dtype=True)
    for key,value in split_meta.items():require(meta[key]==value,'隔离统计不同:'+key)
    require(meta['strict_feature_provenance'] is True and meta['embargo_days']==7 and meta['purge_reason_counts_overlap'] is True,'隔离语义错误')
    overlap,counts=interval_overlap(training,meta['excluded_years'],7)
    require(not overlap.any() and training.cohort_target_year.eq(training.datetime.dt.year).all()
        and training.baseline_opening_shares.ge(cfg['lot_size']).all(),'来源或旧股机会错误')
    require(training.label_start.eq(training.datetime.map(next_minute)).all()
        and training.label_end.eq(recovery_times(training.datetime,horizon)).all()
        and training.execution_time.eq(training.label_start).all() and training.recovery_time.eq(training.label_end).all()
        and training.horizon_minutes.eq(horizon).all(),'短期限端点时钟错误')
    target=training.recovery_price/training.execution_price-1
    require(np.isfinite(target).all() and np.array_equal(target.to_numpy(),training.target_return.to_numpy()),'原始目标被修改')
    own_features,own_audit=coverage_feature_selection(training,cfg,.7)
    require(own_features==features==meta['features'],'0.70实际选择/列序错误')
    exact(own_audit,feature_audit,'新0.70选择独立重建')
    exact(auditor.table(directory/'feature_manifest.csv'),own_audit,'新0.70清单')
    source_features,source_audit=coverage_feature_selection(training,cfg,.8)
    require(source_features==meta['source_features'],'重建0.80特征不一致')
    exact(auditor.table(directory/'source_feature_manifest.csv'),source_audit,'原0.80反事实选择清单')
    matrices={}
    for prefix,names in (('',features),('source_',source_features)):
        matrix=training[names].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        expected=training[INPUT_KEYS].copy()
        for name in names:expected[name]=matrix[name].to_numpy()
        actual=pd.read_pickle(auditor.record(directory/(prefix+'training_inputs.pkl')))
        exact(actual,expected,'真实'+prefix+'矩阵全部列/NaN/dtype',dtype=True)
        require(actual.index.equals(expected.index),'真实'+prefix+'pickle索引不同于拟合行索引')
        require(meta[prefix+'training_matrix_sha256']==frame_digest(matrix),'实际矩阵内容SHA不同')
        matrices[prefix]=matrix
    trace=[c for c in training if c in (*INPUT_KEYS,'intraday_feature_source_start','entry_feature_source_start',
        'entry_rank_source_start','portfolio_cash_source_start','base_entry_datetime','source_model_training_years',
        'source_model_sha256','label_start','label_end','target_return','execution_price','recovery_price','horizon_minutes')]
    expected_labels=training[trace].copy();expected_labels['fitted']=True
    exact(auditor.table(directory/'training_labels.csv'),expected_labels,'全部目标与来源trace')
    zeros={'label_start':0,'label_end':0}
    for (year,symbol),part in training.groupby([training.datetime.dt.year,'symbol'],sort=False):
        raw=auditor.full_bars(year,symbol)
        for endpoint,price in (('label_start','execution_price'),('label_end','recovery_price')):
            quotes=raw.reindex(pd.DatetimeIndex(part[endpoint]))
            require(np.isfinite(quotes.close).all() and np.array_equal(quotes.close.to_numpy(),part[price].to_numpy()),'原始精确价格端点不一致')
            zeros[endpoint]+=int(quotes.volume.eq(0).sum())
    for key,value in {'full_training_rows_sha256':frame_digest(training[KEYS+['feature_source_start']]),
        'training_target_sha256':frame_digest(training[['target_return']])}.items():require(meta[key]==value,'训练身份SHA错误:'+key)
    for name,key in (('training_inputs.pkl','training_inputs_file_sha256'),('source_training_inputs.pkl','source_training_inputs_file_sha256'),
        ('source_feature_manifest.csv','source_feature_manifest_file_sha256'),('training_labels.csv','training_labels_file_sha256'),
        ('return_model.txt','native_model_sha256'),('model_bundle.joblib','model_bundle_sha256')):
        require(digest(auditor.record(directory/name))==meta[key],'模型产物SHA不一致:'+name)
    bundle=joblib.load(directory/'model_bundle.joblib')
    require(json.loads(json.dumps(bundle['metadata'],default=str))=={k:v for k,v in meta.items() if k!='model_bundle_sha256'},'bundle实际元数据不同')
    parameters={'n_estimators':160,'learning_rate':.04,'max_depth':4,'num_leaves':15,'min_child_samples':50,
        'colsample_bytree':.85,'subsample':.85,'subsample_freq':1,'reg_lambda':8.,'reg_alpha':.1,
        'random_state':int(cfg['random_state']),'verbosity':-1,**cfg['t_model']}
    parameters.update(objective='regression',max_depth=4 if profile=='l2_depth4' else 6,
        num_leaves=15 if profile=='l2_depth4' else 31,n_jobs=max(1,int(cfg['workers'])),
        device_type='cpu',deterministic=True,force_col_wise=True)
    require(parameters==meta['model_parameters'] and meta['lightgbm_version']==lightgbm.__version__,'完整实际参数/库版本不同')
    native=lightgbm.Booster(model_file=str(directory/'return_model.txt'))
    require(bundle['features']==features,'bundle真实特征序错误')
    validate_native_parameters(native,bundle['model'],parameters,features)
    matrix=matrices[''];values=np.asarray(native.predict(matrix,num_threads=min(4,cfg['workers'])))
    require(values.shape==(len(matrix),) and np.isfinite(values).all(),'原生全训练预测shape/有限性错误')
    indices=np.linspace(0,len(matrix)-1,min(len(matrix),1024),dtype=int)
    other=np.asarray(bundle['model'].predict(matrix.iloc[indices]))
    require(other.shape==(len(indices),) and np.isfinite(other).all(),'joblib重载shape/有限性错误')
    error=float(np.max(np.abs(values[indices]-other)));close(error,0.,'L2原生/joblib全部树重载',1e-12)
    reload=auditor.json(directory/'native_reload_audit.json')
    require(reload['passed'] is True and reload['max_error']<=1e-12 and reload['fitted_rows']==len(training)
        and reload['training_inputs_sha256']==meta['training_inputs_file_sha256'],'重载归档身份错')
    comparison=compare_source_training(auditor,directory,auditor.learning_source/f"year_{fold['test_year']}/models/h{horizon}_{profile}",meta)
    for key in ('added_features','removed_features'):require(meta[key]==comparison[key],'新增/移除因子记录错误')
    recorded=auditor.json(directory.parent.parent/'training_identity'/f'h{horizon}_{profile}.json')
    require(recorded['passed'] is True,'源训练对照未通过')
    for key in ('reconstructed_source_selection_labels_and_split_exact','current_features','source_features',
                'common_features','added_features','removed_features','source_workers','current_workers'):
        require(recorded[key]==comparison[key],'runner对照与独立重算不一致:'+key)
    require(recorded['same_selected_feature_matrix_required'] is False,'错误要求新旧全矩阵不变')
    return dict(path=str(directory.relative_to(auditor.report)),passed=True,rows=len(training),features=features,
        profile=profile,horizon_minutes=horizon,source_features=source_features,candidate_features=len(own_audit),
        training_split=split_meta,source_overlap_counts=counts,original_l2_comparison=comparison,
        independent_both_coverage_thresholds_verified=True,raw_reference_target_unmodified=True,
        actual_nan_values_and_common_dtypes_preserved=True,all_native_trees_equal_joblib=True,
        native_full_training_rows=len(values),native_reload_sample_rows=len(indices),native_reload_max_error=error,
        raw_zero_volume_reference_labels=zeros)
