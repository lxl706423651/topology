"""v11 只读复用已审计的周 OOF 队列、完整行情和固定周期标签。

只冻结旧选股，不复用旧 T 目标/权重。每次从全截面当前特征重建核心队列并核对
原来源摘要；标签独立读取，不载入十三 GB 的旧恢复报价，也不改写原缓存。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_panel import write_json
from .weekly_t_cohort import CONTEXT_FEATURES, TRACE_COLUMNS, attach_weekly_context
from .weekly_t_quote_store import calculate_file_sha256


def frame_sha256(frame: pd.DataFrame) -> str:
    """按实际列序及行序记录身份，避免 CSV 格式差异被当成数值变化。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def validate_timing_sources(cfg: dict, root: Path) -> dict:
    """仅允许 T 学习目标/特征/选择协议变化；旧数据、周选股和执行经济合同逐项相同。"""
    source = Path(cfg['source_training_report_dir'])
    previous = json.loads((source/'resolved_config.json').read_text())
    allowed = {'research_version', 'candidate_activation_policy', 'source_training_report_dir',
               'dense_recovery_cache_paths', 'candidate_support_requirements', 'output_dir',
               'max_symbols', 'smoke_weeks', 'test_years', 't_target_mode', 't_model_profiles',
               't_market_prior', 't_residual_min_group_size'}
    changed = [key for key in set(cfg)|set(previous)
               if key not in allowed and not key.startswith('timing_') and cfg.get(key) != previous.get(key)]
    if changed:
        raise ValueError(f'旧行情、周选股、执行或拟合参数合同改变：{sorted(changed)}')
    if not set(cfg['test_years']).issubset(previous['test_years']):
        raise ValueError('外层年份不在已冻结周选择中')
    audit = json.loads((source/'independent_residual_audit.json').read_text())
    if not all(audit.get(key) is True for key in ('passed', 'complete', 'full_configured_scope_complete')):
        raise ValueError('旧 OOF 周模型及数据的独立审计不完整')
    hashes = json.loads((source/'source_hashes.json').read_text())
    for name, expected in hashes.items():
        if (calculate_file_sha256(root/name) != expected or
                calculate_file_sha256(source/'source_snapshot'/Path(name).name) != expected):
            raise ValueError(f'旧公共实现身份改变：{name}')
    return {'passed': True, 'unchanged_shared_sources': len(hashes),
            'economic_and_weekly_contract_unchanged': True, 'old_t_weights_reused': False,
            'source_audit_sha256': calculate_file_sha256(source/'independent_residual_audit.json')}


def restore_core_cohort(universe: pd.DataFrame, intraday: pd.DataFrame, fold: dict,
                        cfg: dict, out: Path, inputs: dict, record_input) -> pd.DataFrame:
    """按原审计周名单取当前行、重新传播上下文，与完整旧 lineage 逐值核对。"""
    source = Path(cfg['source_training_report_dir'])
    directory = source/f"year_{fold['test_year']}"/'oof_weekly_cohort'
    audit = json.loads((source/'independent_residual_audit.json').read_text())
    metadata_path = directory/'cohort_metadata.json'
    record_input(metadata_path, inputs)
    metadata = json.loads(metadata_path.read_text())
    core, excluded = sorted(fold['core_years']), sorted([fold['inner_year'], fold['test_year']])
    if metadata['core_years'] != core or metadata['excluded_years'] != excluded:
        raise ValueError('核心队列的年度合同改变')
    keys = ['trade_week', 'symbol']
    source_keys = keys+['feature_week', 'signal_datetime', 'feature_source_start', 'entry_date', 'exit_date']
    if frame_sha256(universe[source_keys]) != metadata['weekly_source_keys_sha256']:
        raise ValueError('完整周资格表来源与原 OOF 不一致')
    parts, proofs = [], []
    # 只索引一次完整分钟键；新增跨股票特征在调用之前已经全截面计算。
    minute_keys = intraday[keys+['datetime']].assign(_position=np.arange(len(intraday)))
    for target in core:
        selector = directory/f'selector_{target}'
        identity = audit['selectors'][str(selector.relative_to(source))]
        if identity['passed'] is not True:
            raise ValueError('原周 selector 未审计通过')
        selected_path = selector/'oof_weekly_selections.csv'
        for name, expected in [('model_bundle.joblib', identity['model_bundle_sha256']),
                               ('oof_weekly_selections.csv', identity['selected_csv_sha256'])]:
            record_input(selector/name, inputs)
            if inputs[str((selector/name).resolve())]['sha256'] != expected:
                raise ValueError(f'OOF 周 selector 身份改变：{name}')
        selected = pd.read_csv(selected_path, float_precision='round_trip')
        for column in ('trade_week', 'signal_datetime'):
            selected[column] = pd.to_datetime(selected[column])
        train = [year for year in core if year != target]
        if (not selected.source_model_training_years.eq(json.dumps(train, separators=(',', ':'))).all()
                or not selected.source_model_sha256.eq(identity['model_bundle_sha256']).all()
                or not selected.trade_week.dt.year.eq(target).all()):
            raise ValueError('周 OOF 模型不能拟合自身或内部/外层年')
        dates = selected[keys].merge(universe[keys+['entry_date', 'exit_date']], on=keys,
                                    how='left', validate='one_to_one')
        positions = minute_keys.merge(dates, on=keys, how='inner', validate='many_to_one')
        day = positions.datetime.dt.normalize()
        positions = positions.loc[day.gt(positions.entry_date)&day.lt(positions.exit_date)]
        current = intraday.iloc[positions._position.to_numpy()].copy()
        enriched = attach_weekly_context(current, selected, universe, cfg)
        enriched['cohort_target_year'] = target
        parts.append(enriched)
        proofs.append({'target_year': target, 'training_years': train, 'rows': len(enriched),
                       'selected_sha256': identity['selected_csv_sha256'],
                       'model_sha256': identity['model_bundle_sha256']})
    cohort = pd.concat(parts, ignore_index=True).sort_values(['datetime','symbol'], kind='stable').reset_index(drop=True)
    trace = ['datetime', 'symbol', 'feature_source_start', *CONTEXT_FEATURES.values(), *TRACE_COLUMNS]
    signature = frame_sha256(cohort[trace])
    expected = audit['cohorts'][f"year_{fold['test_year']}"]
    if (signature != metadata['cohort_context_sha256'] or signature != expected['cohort_context_sha256']
            or len(cohort) != expected['rows'] or expected['passed'] is not True):
        raise AssertionError('重建的核心周上下文/来源未精确复现')
    lineage_path = directory/'cohort_lineage.csv'
    record_input(lineage_path, inputs)
    original = pd.read_csv(lineage_path, float_precision='round_trip', parse_dates=['datetime','feature_source_start'])
    pd.testing.assert_frame_equal(cohort[trace+['cohort_target_year']], original,
                                  check_dtype=False, check_exact=True)
    dest = out/'oof_weekly_cohort'
    dest.mkdir()
    cohort[trace+['cohort_target_year']].to_csv(dest/'cohort_lineage.csv', index=False, encoding='utf-8-sig')
    write_json(dest/'cohort_reuse_audit.json', {'passed': True, 'core_years': core,
               'excluded_years': excluded, 'rows': len(cohort), 'cohort_context_sha256': signature,
               'selectors': proofs, 'full_market_features_before_selection': True,
               'old_t_models_or_targets_reused': False})
    return cohort


def load_labels_only(cfg: dict, out: Path, inputs: dict, record_input) -> pd.DataFrame:
    """读取已验证缓存全部标签；只用于核心拟合，不根据未来缺价筛选推断行。"""
    source = Path(cfg['source_training_report_dir'])
    manifest_path = source/'horizon_manifest.json'
    record_input(manifest_path, inputs)
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((source/'independent_residual_audit.json').read_text())['input_identity']['horizon']
    # 旧审计摘要对应缓存 metadata.json；报告副本的 cache_hit 字段允许不同。
    cache_metadata_path = Path(manifest['cache_path'])/'metadata.json'
    record_input(cache_metadata_path, inputs)
    cached = json.loads(cache_metadata_path.read_text())
    if calculate_file_sha256(cache_metadata_path) != audit['manifest_sha256'] or audit['passed'] is not True:
        raise ValueError('固定周期缓存 metadata 偏离原数据审计')
    keys = (set(manifest)|set(cached))-{'cache_hit'}
    if any(manifest.get(key) != cached.get(key) for key in keys):
        raise ValueError('标签报告 manifest 与已审计缓存 metadata 不一致')
    for key in ('signal_times', 'horizons', 'start_date', 'end_date', 'data_root'):
        if manifest[key] != cfg[key]:
            raise ValueError(f'标签缓存配置不一致：{key}')
    path = Path(manifest['labels_path'])
    record_input(path, inputs)
    labels = pd.read_pickle(path)
    if len(labels) != manifest['label_rows'] or list(labels.columns) != manifest['label_columns']:
        raise ValueError('标签表行列合同改变')
    write_json(out/'horizon_manifest.json', {**manifest, 'v11_reads_labels_only': True,
               'labels_file_sha256': inputs[str(path.resolve())]['sha256'],
               'future_targets_used_only_by_core_training': True})
    return labels
