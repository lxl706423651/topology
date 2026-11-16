"""v17独立训练核：原始短期限标签、完整core隔离、实际矩阵/来源对照与Huber原生参数。

只读辅助，供audit_inventory_robust_t调用；不导入生产robust模型/策略/标签模块。
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
VERSION = "inventory_robust_t_v17"
KEYS = ["datetime", "symbol"]
PROFILES = {"huber_depth4": "l2_depth4", "huber_depth6": "l2_depth6"}
TRAIN_FILES = ("training_inputs.pkl", "training_labels.csv", "feature_manifest.csv", "training_split_audit.csv")

def independent_robust_training_split(frame: pd.DataFrame, fold: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """独立生成完整机会的逐原因隔离统计；不调用生产select_training_rows。

本轮当前资格保留全部8信号，包括平价；只在这里以来源/未来标签做训练purge。
原因计数允许重叠，与保留行数分别报告，不把缺标签行当不存在的持仓机会。
    """
    require(cfg.get("exclude_flat_signal_bars") is False and cfg.get("strict_feature_provenance") is True,
            "本轮当前资格/严格来源配置错误")
    require(cfg.get("embargo_days") == 7, "本轮隔离须7日")
    require("trade_week_complete" not in frame and frame.horizon_minutes.isin([20, 40, 60]).all(), "旧周或错误期限污染训练机会")
    require(not frame[KEYS].isna().any().any() and not frame.duplicated(KEYS).any(), "原完整机会键缺失/重复")
    source = frame.reset_index(drop=True)
    values = {name: pd.to_datetime(source[name], errors="coerce")
              for name in ("datetime", "feature_source_start", "label_start", "label_end")}
    start, signal, first, end = [values[name] for name in ("feature_source_start", "datetime", "label_start", "label_end")]
    years = sorted(fold["core_years"])
    excluded = sorted([fold["inner_year"], fold["test_year"]])
    require(len(set(years)) == 2 and len(set(excluded)) == 2 and not set(years) & set(excluded), "核心/排除年交叉")
    bad = {
        "feature_year_not_training": ~start.dt.year.isin(years),
        "signal_year_not_training": ~signal.dt.year.isin(years),
        "label_start_year_not_training": ~first.dt.year.isin(years),
        "label_end_year_not_training": ~end.dt.year.isin(years),
        "noncausal_time_order": ~(start.le(signal) & signal.lt(first) & first.le(end)),
        "intraday_label_crosses_day": ~(signal.dt.normalize().eq(first.dt.normalize()) & signal.dt.normalize().eq(end.dt.normalize())),
        "invalid_target_return": ~np.isfinite(pd.to_numeric(source.target_return, errors="coerce")),
        "label_unavailable": ~source.label_available.eq(True),
    }
    # 与独立ns区间相交算法并列校验，NaT在前面时序检查拒绝。
    boundary = pd.Series(False, index=source.index)
    for year in excluded:
        left = pd.Timestamp(year, 1, 1)-pd.Timedelta(days=7)
        right = pd.Timestamp(year+1, 1, 1)+pd.Timedelta(days=7)
        boundary |= start.lt(right) & end.ge(left)
    independent_overlap, _ = interval_overlap(source, excluded, 7)
    require(np.array_equal(boundary.to_numpy(), independent_overlap), "隔离区间独立算法不一致")
    bad["excluded_year_or_boundary_embargo"] = boundary
    kept = source.loc[~pd.DataFrame(bad).any(axis=1)].sort_values(KEYS, kind="stable").copy()
    rows = [{"reason": "causal_inference_filter", "excluded_rows": 0}]
    rows += [{"reason": name, "excluded_rows": int(mask.sum())} for name, mask in bad.items()]
    rows.append({"reason": "retained_training_rows", "excluded_rows": len(kept)})
    summary = {"input_rows": len(frame), "causally_eligible_rows": len(frame), "training_rows": len(kept),
               "training_start": str(kept.datetime.min()), "training_end": str(kept.datetime.max()),
               "label_start_min": str(kept.label_start.min()), "label_end_max": str(kept.label_end.max()),
               "feature_source_start_min": str(kept.feature_source_start.min())}
    return kept, pd.DataFrame(rows), summary


def validate_native_parameters(native, model, parameters, features):
    """读取真正Booster/拟合器参数，不接受只改JSON中的Huber声明。"""
    require(parameters.get("objective") == "huber" and parameters.get("alpha") == .01, "声明损失不是固定Huber")
    require(native.params.get("objective") == "huber" and float(native.params.get("alpha", np.nan)) == .01,
            "原生实际损失或alpha不是固定Huber")
    require(model.booster_.params.get("objective") == "huber" and float(model.booster_.params.get("alpha", np.nan)) == .01,
            "joblib实际损失或alpha不是固定Huber")
    require(native.feature_name() == features == model.booster_.feature_name(), "原生实际特征顺序不同")
    require(all(model.get_params().get(k) == v for k, v in parameters.items()), "实际拟合器完整参数不同")
    require(native.model_to_string() == model.booster_.model_to_string(), "原生全部树与joblib权重不同")


def compare_source_training(auditor, directory, source, meta):
    """独立逐值/dtype核四个真实训练产物；来源必须被原完整独审SHA锚定。"""
    proof = auditor.source_audits['learning'][1]['inputs_sha256']
    evidence = {}
    original = auditor.json(source/'model_metadata.json')
    require(proof.get(str((source/'model_metadata.json').resolve())) == digest(source/'model_metadata.json'), "来源模型metadata未被完整独审锚定")
    for name in TRAIN_FILES:
        old, new = source/name, directory/name
        require(proof.get(str(old.resolve())) == digest(auditor.record(old)), f"原训练产物未锚定：{name}")
        auditor.record(new)
        left, right = ((pd.read_pickle(new), pd.read_pickle(old)) if name.endswith('.pkl') else
                       (auditor.csv(new), auditor.csv(old)))
        exact(left, right, f"独立原L2/新Huber训练产物逐值{ name}", dtype=True)
        evidence[name] = {'current_sha256': digest(new), 'source_sha256': digest(old), 'values_and_dtype_exact': True}
    require(original['research_version'] == 'inventory_learning_t_v14' and PROFILES[meta['profile']] == original['profile']
            and meta['horizon_minutes'] == original['horizon_minutes'], "Huber/L2明确期限架构映射错")
    for key in ('train_years', 'excluded_years', 'features', 'fitted_rows', 'training_rows', 'full_training_rows_sha256',
                'training_matrix_sha256', 'training_target_sha256'):
        require(meta[key] == original[key], f"原训练身份改变：{key}")
    old, new = original['model_parameters'], meta['model_parameters']
    require(old.get('objective') == 'regression' and new.get('objective') == 'huber' and new.get('alpha') == .01, '损失对照错误')
    ignored = {'objective', 'alpha', 'n_jobs'}
    require({k:v for k,v in old.items() if k not in ignored} == {k:v for k,v in new.items() if k not in ignored}, '非损失实际训练参数改变')
    return {'source_model': str(source), 'four_training_artifacts': evidence, 'source_profile': original['profile'],
            'source_workers': old['n_jobs'], 'current_workers': new['n_jobs'], 'all_nonloss_parameters_exact': True}


def audit_robust_model(auditor, directory, training, features, feature_audit, fold, horizon, profile, *, full_frame):
    """重建全矩阵/标签/哈希与原生完整预测，不重拟合；再独立检查原L2四产物。"""
    cfg = auditor.cfg
    meta = auditor.json(directory/'model_metadata.json')
    require(meta['research_version'] == cfg['research_version'] == VERSION and profile in PROFILES
            and meta['profile'] == profile and meta['horizon_minutes'] == horizon, '新Huber模型身份错')
    require(meta['train_years'] == sorted(fold['core_years']) and meta['excluded_years'] == sorted([fold['inner_year'], fold['test_year']]), '模型年份泄漏')
    require(meta['resolved_config'] == {**cfg, 't_model_profile': profile, 't_label_horizon_minutes': horizon}, '模型完整配置不同')
    require(meta['features'] == features and all(c.startswith('f_') and c != 'f_history_days' for c in features), '实际特征身份错')
    require(meta['training_loss'] == 'huber' and meta['huber_alpha'] == .01 and meta['source_l2_profile'] == PROFILES[profile], 'Huber训练声明错')
    require(all(meta[k] is False for k in ('score_is_calibrated_conditional_mean', 'score_is_actual_execution_pnl',
            'training_targets_clipped', 'tail_training_rows_removed', 'target_is_portfolio_execution_net_pnl', 'probability_gate_used')), '评分/原标签合同错')
    require(meta['only_parameter_changes_from_source_l2'] == {'objective': 'huber', 'alpha': .01}, '非声明损失变化')
    require(meta['target_definition'] == 'reference_recovery_price/reference_next_minute_price-1'
            and meta['training_cohort'] == 'permanent_inventory_core_oof_opening_old_shares', '原始目标/旧股队列身份错')
    require(cfg['t_max_train_rows'] == 0 and meta['fitted_rows'] == meta['training_rows'] == len(training), '真实拟合行数不同')
    selected, split, split_meta = independent_robust_training_split(full_frame, fold, cfg)
    exact(selected, training, '完整core独立purge', dtype=True)
    require(selected.index.equals(training.index), '特征抽样的训练索引改变')
    exact(auditor.table(directory/'training_split_audit.csv'), split, '逐原因训练purge', dtype=True)
    for key, value in split_meta.items(): require(meta[key] == value, f'训练隔离meta错：{key}')
    require(meta['strict_feature_provenance'] is True and meta['embargo_days'] == 7 and meta['purge_reason_counts_overlap'] is True, '隔离语义错')
    overlaps, counts = interval_overlap(training, meta['excluded_years'], 7)
    require(not overlaps.any() and training.cohort_target_year.eq(training.datetime.dt.year).all()
            and training.baseline_opening_shares.ge(cfg['lot_size']).all(), '核心旧股/来源穿越排除年')
    require(training.label_start.eq(training.datetime.map(next_minute)).all()
            and training.label_end.eq(recovery_times(training.datetime, horizon)).all()
            and training.execution_time.eq(training.label_start).all()
            and training.recovery_time.eq(training.label_end).all()
            and training.horizon_minutes.eq(horizon).all(), '原始短期限标签时钟错误')
    raw_target = training.recovery_price/training.execution_price-1
    require(np.isfinite(raw_target).all() and np.array_equal(raw_target.to_numpy(), training.target_return.to_numpy()), '原标签被居中/缩尾/换净额')
    exact(auditor.table(directory/'feature_manifest.csv'), feature_audit, '完整原收益特征选择')
    matrix = training[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    expected = training[KEYS+['feature_source_start', 'cohort_target_year', 'baseline_opening_shares']].copy()
    for name in features: expected[name] = matrix[name].to_numpy()
    actual = pd.read_pickle(auditor.record(directory/'training_inputs.pkl'))
    exact(actual, expected, '实际拟合input.pkl全部列/dtype', dtype=True)
    trace = [c for c in training if c in ('datetime','symbol','feature_source_start','cohort_target_year','baseline_opening_shares',
        'intraday_feature_source_start','entry_feature_source_start','entry_rank_source_start','portfolio_cash_source_start',
        'base_entry_datetime','source_model_training_years','source_model_sha256','label_start','label_end','target_return',
        'execution_price','recovery_price','horizon_minutes')]
    expected_labels = training[trace].copy(); expected_labels['fitted'] = True
    exact(auditor.table(directory/'training_labels.csv'), expected_labels, '原始目标及全部来源trace')
    zeros = {'label_start': 0, 'label_end': 0}
    for (year, symbol), part in training.groupby([training.datetime.dt.year, 'symbol'], sort=False):
        raw = auditor.full_bars(year, symbol)
        for endpoint, price in (('label_start','execution_price'), ('label_end','recovery_price')):
            quotes = raw.reindex(pd.DatetimeIndex(part[endpoint]))
            require(np.isfinite(quotes.close).all() and np.array_equal(quotes.close.to_numpy(), part[price].to_numpy()), f'原始精确端点不同：{year}/{symbol}/{endpoint}')
            zeros[endpoint] += int(quotes.volume.eq(0).sum())
    hashes = {'full_training_rows_sha256': frame_digest(training[KEYS+['feature_source_start']]),
              'training_matrix_sha256': frame_digest(matrix), 'training_target_sha256': frame_digest(training[['target_return']])}
    for key, value in hashes.items(): require(meta[key] == value, f'实际训练内容SHA不同：{key}')
    for name, key in (('training_inputs.pkl','training_inputs_file_sha256'),('training_labels.csv','training_labels_file_sha256'),
                      ('return_model.txt','native_model_sha256'),('model_bundle.joblib','model_bundle_sha256')):
        require(digest(auditor.record(directory/name)) == meta[key], f'实际文件SHA不同：{name}')
    bundle = joblib.load(directory/'model_bundle.joblib')
    require(json.loads(json.dumps(bundle['metadata'], default=str)) == {k:v for k,v in meta.items() if k != 'model_bundle_sha256'}, 'bundle实际元数据不同')
    parameters = {'n_estimators':160,'learning_rate':.04,'max_depth':4,'num_leaves':15,'min_child_samples':50,
        'colsample_bytree':.85,'subsample':.85,'subsample_freq':1,'reg_lambda':8.,'reg_alpha':.1,
        'random_state':int(cfg.get('random_state',42)),'verbosity':-1,**cfg['t_model']}
    parameters.update(objective='huber', alpha=.01, max_depth=4 if profile == 'huber_depth4' else 6,
        num_leaves=15 if profile == 'huber_depth4' else 31, n_jobs=max(1,int(cfg['workers'])),
        device_type='cpu', deterministic=True, force_col_wise=True)
    require(parameters == meta['model_parameters'] and meta['lightgbm_version'] == lightgbm.__version__, '独立完整参数/库版本不同')
    native = lightgbm.Booster(model_file=str(directory/'return_model.txt'))
    require(bundle['features'] == features, 'bundle特征序不同')
    validate_native_parameters(native, bundle['model'], parameters, features)
    values = np.asarray(native.predict(matrix, num_threads=min(4, cfg['workers'])))
    require(values.shape == (len(matrix),) and np.isfinite(values).all(), '原生全拟合行预测非有限/shape错')
    idx = np.linspace(0, len(matrix)-1, min(len(matrix),1024), dtype=int)
    other = np.asarray(bundle['model'].predict(matrix.iloc[idx]))
    require(other.shape == (len(idx),) and np.isfinite(other).all(), 'joblib重载shape/有限性错')
    error = float(np.max(np.abs(values[idx]-other))); close(error,0.,'原生/joblib训练预测',1e-12)
    absolute = np.abs(training.target_return.to_numpy()-values); tail=absolute>.01
    ss, sa = float(np.square(absolute).sum()), float(absolute.sum())
    stats = meta['fit_residual_statistics']
    expected_stats = {'rows':len(absolute),'alpha':.01,'abs_residual_gt_alpha_rows':int(tail.sum()),
        'abs_residual_gt_alpha_fraction':float(tail.mean()),'sum_squared_residual':ss,'sum_absolute_residual':sa,
        'tail_squared_residual_share':float(np.square(absolute[tail]).sum()/ss) if ss else 0.,
        'tail_absolute_residual_share':float(absolute[tail].sum()/sa) if sa else 0.,
        'target_clipped':False,'tail_rows_removed':False,'scope':'in_sample_core_fit_description_only'}
    require(stats == expected_stats, '实际训练残差描述与原生全行复算不一致')
    reload = auditor.json(directory/'native_reload_audit.json')
    require(reload['passed'] is True and reload['max_error'] <= 1e-12 and reload['fitted_rows'] == len(training)
            and reload['training_inputs_sha256'] == meta['training_inputs_file_sha256'], '重载归档身份错')
    source = auditor.learning_source/f"year_{fold['test_year']}/models/h{horizon}_{PROFILES[profile]}"
    comparison = compare_source_training(auditor, directory, source, meta)
    recorded = auditor.json(directory.parent.parent/'training_identity'/f'h{horizon}_{profile}.json')
    require(recorded['passed'] is True and recorded['matrices_labels_features_and_split_exact'] == comparison['four_training_artifacts'], 'runner对照记录与独立实际文件复算不一致')
    return {'path':str(directory.relative_to(auditor.report)),'rows':len(training),'features':features,'passed':True,
        'horizon_minutes':horizon,'profile':profile,'training_split':split_meta,'source_overlap_counts':counts,
        'hashes':hashes,'all_native_trees_equal_joblib':True,'all_model_parameters_exact':True,
        'native_full_training_rows':len(values),'native_reload_sample_rows':len(idx),'native_reload_max_error':error,
        'native_huber_alpha':.01,'raw_zero_volume_reference_labels':zeros,'original_l2_comparison':comparison,
        'original_reference_target_unmodified':True,'independently_rebuilt_training_inputs':True}

