"""只读独立审计 v11 核心原始收益/扣费分类、当前特征和模型；不读取外层账户。

用法：python scripts/audit_weekly_t_timing_models.py --report-dir reports/... [--smoke]。
仅写本报告的 independent_timing_model_audit.json/md；不拟合、不选参、不改缓存。
旧周来源/训练域使用此前独立审计的重建逻辑，新十一特征及双头标签另行重算。
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_weekly_t_short_horizon import (
    audit_support, file_digest, frame_digest, independent_training_rows,
    interval_overlap, read_json, verify_cache_inputs,
)
from scripts.audit_weekly_t_residual import (
    CONTEXT, TRACE, KEYS, independent_context, audit_training_domain,
)

TIMING_COLUMNS = [
    'f_timing_market_return_5m', 'f_timing_market_return_15m',
    'f_timing_market_breadth', 'f_timing_market_day_return_std',
    'f_timing_excess_return_5m', 'f_timing_excess_return_15m',
    'f_timing_volume_rate', 'f_timing_trend_efficiency', 'f_timing_sigma20',
    'f_timing_vwap_z', 'f_timing_reversal_confirmation',
]
TARGET_COLUMNS = ['datetime', 'symbol', 'feature_source_start', 'label_start', 'label_end',
                  'raw_target_return', 'buy_net_return', 'sell_net_return', 'profit_class', 'fitted']
PROBABILITIES = ['prob_neither', 'prob_buy_profit', 'prob_sell_profit']
CLASS_MAPPING = {'0': 'neither', '1': 'buy_net_positive', '2': 'sell_net_positive'}
COSTS = ['commission_rate', 'transfer_rate', 'slippage_rate',
         'stamp_rate_before_20230828', 'stamp_rate_from_20230828', 'min_commission']


def record(path: Path, result: dict) -> str:
    """记录实际读取证据的完整字节身份，避免将自报摘要当成独立核对。"""
    key = str(path.resolve())
    if key not in result['inputs']:
        result['inputs'][key] = {'sha256': file_digest(path), 'bytes': path.stat().st_size}
    return result['inputs'][key]['sha256']


def csv(path: Path, result: dict | None = None) -> pd.DataFrame:
    """以浮点往返模式读证据，并恢复明确命名的时间列。"""
    if result is not None:
        record(path, result)
    frame = pd.read_csv(path, float_precision='round_trip')
    for column in frame:
        if column in {'datetime', 'date', 'trade_week', 'feature_week', 'signal_datetime',
                      'entry_date', 'exit_date', 'feature_source_start', 'label_start', 'label_end',
                      'intraday_feature_source_start', 'weekly_context_feature_source_start',
                      'weekly_context_signal_datetime', 'planned_recovery_time'}:
            frame[column] = pd.to_datetime(frame[column], format='mixed')
    return frame


def exact(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """CSV 类型可改变，但全部值、键、列序及行序必须精确相同。"""
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=True)


def save(report: Path, result: dict) -> None:
    """只原子更新自有 JSON/Markdown；异常保留已完成范围。"""
    path = report/'independent_timing_model_audit.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)
    lines = ['# v11 双头择时模型独立审计', '',
             f"完成：{result.get('complete', False)}；通过：{result.get('passed', False)}。",
             '只核模型、输入与内部预测；未读取外层账户、收益或重新选参。', '',
             '|模型|核心年|训练行|排除区间重叠|特征/标签/来源 SHA|原生双头|',
             '|---|---|---:|---:|---|---|']
    for name, item in result.get('models', {}).items():
        lines.append(f"|{name}|{item['core_years']}|{item['training_rows']}|{item['overlap_rows']}|通过|通过|")
    lines += ['', '十一项择时特征由全市场当前截面独立重算，之后才取核心 OOF 队列；',
              '特征选择、训练标签、类别与支持域只来自严格来源隔离后的核心行。',
              'raw 标签是固定端点股票收益，未减同刻均值；类别按实际费率和日期税率重建。',
              '概率未经校准，标签未模拟容量/现金/恢复成功，不能解释为真实成交胜率。',
              '原始量价边界通过不等于全新测试；较早年份的留一年验证仍使用较晚年训练参数。',
              f"\n脚本 SHA：{result['audit_script_sha256']}；耗时：{result.get('elapsed_seconds', 0):.2f} 秒。"]
    if result.get('error'):
        lines += ['', '错误：'+result['error']]
    (report/'independent_timing_model_audit.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def sources(report: Path, smoke: bool, result: dict) -> dict:
    """核全部源快照；仅允许已知 smoke runner 后续新增归档列的身份差异。"""
    hashes = read_json(report/'source_hashes.json')
    record(report/'source_hashes.json', result)
    checked = {}
    for name, expected in hashes.items():
        snapshot = report/'source_snapshot'/Path(name).name
        actual = record(ROOT/name, result)
        assert record(snapshot, result) == expected, ('snapshot', name)
        exception = (smoke and name == 'scripts/run_weekly_t_timing.py'
                     and expected == '66957a3ed70c3d8a7b31bcf6f9546e7f4db0a079de5308775c06e25ebde022e3'
                     and actual == '08ffd705484ffe3fec8345f4d10758206c2993b2abb390a4a6eb69ab5a39c9fa')
        assert actual == expected or exception, ('current source', name)
        checked[name] = {'snapshot_sha256': expected, 'current_sha256': actual,
                         'smoke_archive_only_exception': exception}
    assert len(checked) == 32
    return {'passed': True, 'files': checked, 'current_all_match': all(
        row['snapshot_sha256'] == row['current_sha256'] for row in checked.values())}


def independent_timing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """独立重算当前市场横截面及固定数值稳定公式，不调用生产特征函数。"""
    names = ['f_return_5m', 'f_return_15m', 'f_open_return', 'f_day_return',
             'f_amount_vs_previous_day', 'f_session_fraction', 'f_known_amplitude',
             'f_realized_volatility', 'f_previous_day_amplitude', 'f_vwap_deviation']
    values = frame[names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    time_key = frame.datetime
    mean5 = values.f_return_5m.groupby(time_key, sort=False).transform('mean')
    mean15 = values.f_return_15m.groupby(time_key, sort=False).transform('mean')
    positive = values.f_open_return.gt(0).astype(float).where(values.f_open_return.notna())
    breadth = positive.groupby(time_key, sort=False).transform('mean')
    std = values.f_day_return.groupby(time_key, sort=False).transform('std', ddof=0)
    session = values.f_session_fraction.where(values.f_session_fraction.between(0, 1))
    amplitude = values.f_known_amplitude.where(values.f_known_amplitude.ge(0))
    previous = values.f_previous_day_amplitude.where(values.f_previous_day_amplitude.ge(0))
    volatility = values.f_realized_volatility.where(values.f_realized_volatility.ge(0))
    volume = values.f_amount_vs_previous_day.where(values.f_amount_vs_previous_day.ge(0))/session.clip(lower=.05)
    efficiency = (values.f_open_return.abs()/amplitude.clip(lower=1e-6)).clip(0, 1)
    sigma = pd.Series(np.maximum.reduce([
        (volatility*np.sqrt(20./(240.*session).clip(lower=5.))).to_numpy(),
        (previous*.25*np.sqrt(20./240.)).to_numpy(), np.full(len(frame), .0005),
    ]), index=frame.index)
    calculated = [mean5, mean15, breadth, std, values.f_return_5m-mean5, values.f_return_15m-mean15,
                  volume, efficiency, sigma, values.f_vwap_deviation/sigma,
                  -np.sign(values.f_vwap_deviation)*values.f_return_5m/sigma]
    for name, value in zip(TIMING_COLUMNS, calculated):
        assert name not in frame
        frame[name] = value.replace([np.inf, -np.inf], np.nan).astype('float32')
    return frame


def rebuild_cohort(universe: pd.DataFrame, minutes: pd.DataFrame, fold: dict,
                   cfg: dict, report: Path, result: dict) -> pd.DataFrame:
    """从已独立核验的另一个核心年 selector 名单重建全部队列，不读取目标年标签。"""
    source = Path(cfg['source_training_report_dir'])
    original_audit = read_json(source/'independent_residual_audit.json')
    base = source/f"year_{fold['test_year']}"/'oof_weekly_cohort'
    metadata = read_json(base/'cohort_metadata.json')
    record(base/'cohort_metadata.json', result)
    core, excluded = sorted(fold['core_years']), sorted([fold['inner_year'], fold['test_year']])
    assert metadata['core_years'] == core and metadata['excluded_years'] == excluded
    signature = frame_digest(universe[KEYS+['feature_week', 'signal_datetime', 'feature_source_start', 'entry_date', 'exit_date']])
    assert signature == metadata['weekly_source_keys_sha256']
    minute_keys = minutes[KEYS+['datetime']].assign(position=np.arange(len(minutes)))
    parts, selectors = [], []
    for target in core:
        path = base/f'selector_{target}'
        identity = original_audit['selectors'][str(path.relative_to(source))]
        model_sha = record(path/'model_bundle.joblib', result)
        assert identity['passed'] and model_sha == identity['model_bundle_sha256']
        assert record(path/'oof_weekly_selections.csv', result) == identity['selected_csv_sha256']
        selected = csv(path/'oof_weekly_selections.csv')
        train = [year for year in core if year != target]
        selector_metadata = read_json(path/'model_metadata.json')
        record(path/'model_metadata.json', result)
        assert selector_metadata['train_years'] == train
        assert selector_metadata['excluded_years'] == sorted(set(metadata['available_years'])-set(train))
        assert selected.trade_week.dt.year.eq(target).all() and selected.selected.eq(True).all()
        assert selected[TRACE[0]].eq(json.dumps(train, separators=(',', ':'))).all()
        assert selected[TRACE[1]].eq(model_sha).all()
        dates = selected[KEYS].merge(universe[KEYS+['entry_date', 'exit_date']], on=KEYS, validate='one_to_one')
        positions = minute_keys.merge(dates, on=KEYS, validate='many_to_one')
        day = positions.datetime.dt.normalize()
        positions = positions.loc[day.gt(positions.entry_date)&day.lt(positions.exit_date)]
        current = independent_context(minutes.iloc[positions.position.to_numpy()].copy(), selected, universe, cfg)
        current['cohort_target_year'] = target
        parts.append(current)
        selectors.append({'target_year': target, 'train_years': train, 'rows': len(current), 'model_sha256': model_sha})
    cohort = pd.concat(parts, ignore_index=True).sort_values(['datetime', 'symbol'], kind='stable').reset_index(drop=True)
    trace = ['datetime', 'symbol', 'feature_source_start', *CONTEXT.values(), *TRACE]
    original = original_audit['cohorts'][f"year_{fold['test_year']}"]
    signature = frame_digest(cohort[trace])
    assert signature == metadata['cohort_context_sha256'] == original['cohort_context_sha256']
    assert len(cohort) == original['rows'] and original['passed']
    for directory in [base, report/f"year_{fold['test_year']}"/'oof_weekly_cohort']:
        exact(cohort[trace+['cohort_target_year']], csv(directory/'cohort_lineage.csv', result))
    current = read_json(report/f"year_{fold['test_year']}"/'oof_weekly_cohort/cohort_reuse_audit.json')
    assert current['cohort_context_sha256'] == signature and current['rows'] == len(cohort)
    result['cohorts'][str(fold['test_year'])] = {'passed': True, 'rows': len(cohort),
        'core_years': core, 'excluded_years': excluded, 'selectors': selectors, 'context_sha256': signature,
        'full_core_preserved_in_smoke': True, 'weekly_and_minute_earliest_source_matches': True}
    return cohort


def attach_labels(frame: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """独立按当前唯一键连接固定端点；未来缺价行留待严格训练资格处理。"""
    keys = ['datetime', 'symbol']
    future = ['recovery_time', 'recovery_price', 'label_start', 'label_end', 'target_return']
    chosen = labels.loc[labels.horizon_minutes.eq(horizon)]
    assert not chosen.duplicated(keys).any()
    result = frame.drop(columns=[name for name in future+['label_available'] if name in frame]).merge(
        chosen[keys+future], on=keys, how='left', sort=False, validate='one_to_one')
    result['label_available'] = np.isfinite(result.target_return)
    result['horizon_minutes'] = horizon
    for name in ('recovery_open', 'recovery_high', 'recovery_low', 'recovery_volume', 'recovery_amount'):
        if name in result:
            result[name] = np.nan
    return result


def labels_from_prices(training: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """由真实端点直接重算 raw 及实际费率三分类，不调用模型/策略的标签函数。"""
    session = np.r_[np.arange(571, 691), np.arange(781, 901)]
    index = {int(clock): i for i, clock in enumerate(session)}
    execution = pd.to_datetime(training.execution_time)
    assert execution.eq(training.datetime+pd.Timedelta(minutes=1)).all()
    endpoint_positions = (execution.dt.hour*60+execution.dt.minute).map(index).to_numpy()+horizon
    assert np.all(endpoint_positions < len(session))
    end = execution.dt.normalize()+pd.to_timedelta(session[endpoint_positions], unit='m')
    assert training.label_start.eq(execution).all() and training.label_end.eq(end).all()
    assert training.recovery_time.eq(end).all()
    assert training.execution_price.gt(0).all() and training.recovery_price.gt(0).all()
    raw = training.recovery_price/training.execution_price-1
    np.testing.assert_allclose(raw, training.target_return, atol=1e-12, rtol=0)
    # 缓存原始浮点标签须逐位一致；不使用重新安排算术后的浮点数冒充原训练输入。
    assert np.array_equal(raw.to_numpy(), training.target_return.to_numpy())
    fee = float(cfg['commission_rate'])+float(cfg['transfer_rate'])
    slip = float(cfg['slippage_rate'])
    assert cfg['min_commission'] == 0
    stamp = pd.Series(np.where(training.datetime < pd.Timestamp('2023-08-28'),
                              cfg['stamp_rate_before_20230828'], cfg['stamp_rate_from_20230828']), index=training.index)
    ratio = 1.+raw
    buy = ratio*(1-slip)*(1-fee-stamp)-(1+slip)*(1+fee)
    sell = (1-slip)*(1-fee-stamp)-ratio*(1+slip)*(1+fee)
    assert np.isfinite(buy).all() and np.isfinite(sell).all() and not (buy.gt(0)&sell.gt(0)).any()
    labels = training[['datetime', 'symbol', 'feature_source_start', 'label_start', 'label_end']].copy()
    labels['raw_target_return'], labels['buy_net_return'], labels['sell_net_return'] = raw, buy, sell
    labels['profit_class'] = np.where(buy.gt(0), 1, np.where(sell.gt(0), 2, 0)).astype(np.int64)
    assert set(labels.profit_class.unique()) == {0, 1, 2}
    return labels


def feature_selection(training: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> tuple:
    """独立重建核心覆盖率、三个目标的秩相关及贪心去相关，不读取验证数据。"""
    names = sorted(name for name in training if name.startswith('f_') and name not in cfg['excluded_model_features'])
    values = training[names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    indices = training.sample(n=min(int(cfg['feature_selection_sample_rows']), len(training)), random_state=int(cfg['random_state'])).index
    sample = values.loc[indices]
    coverage, diversity = values.notna().mean(), sample.nunique(dropna=True)
    eligible = [name for name in names if coverage[name] >= cfg['feature_min_coverage'] and diversity[name] > 1]
    targets = pd.DataFrame({'return': labels.raw_target_return, 'buy_profit': labels.profit_class.eq(1).astype(int),
                            'sell_profit': labels.profit_class.eq(2).astype(int)})
    ranks, target_ranks = sample[eligible].rank(pct=True), targets.loc[indices].rank(pct=True)
    correlations = [ranks.corrwith(target_ranks[name]).abs() for name in target_ranks if target_ranks[name].nunique() > 1]
    relevance = pd.Series(0., index=names)
    if correlations and eligible:
        relevance.loc[eligible] = pd.concat(correlations, axis=1).max(axis=1).fillna(0)
    ordered = sorted(eligible, key=lambda name: (-float(relevance[name]), name))
    correlation = sample[ordered].corr(method='spearman')
    selected, rows = [], []
    for name in ordered:
        strongest = correlation.loc[name, selected].abs().dropna()
        maximum = float(strongest.max()) if len(strongest) else 0.
        reason = ('correlation_limit' if maximum > cfg['feature_max_correlation'] else
                  'feature_limit' if len(selected) >= cfg['t_max_features'] else 'selected')
        if reason == 'selected':
            selected.append(name)
        rows.append({'feature': name, 'coverage': coverage[name], 'target_rank_relevance': relevance[name],
                     'max_selected_correlation': maximum, 'selected': reason == 'selected', 'reason': reason})
    for name in names:
        if name not in eligible:
            rows.append({'feature': name, 'coverage': coverage[name], 'target_rank_relevance': relevance[name],
                         'max_selected_correlation': 0., 'selected': False,
                         'reason': 'coverage' if coverage[name] < cfg['feature_min_coverage'] else 'constant'})
    return selected, pd.DataFrame(rows)


def audit_model(path: Path, training: pd.DataFrame, labels: pd.DataFrame, selected: tuple,
                fold: dict, horizon: int, profile: str, cfg: dict, result: dict) -> dict:
    """核真实拟合行全部 SHA、双头参数/原生树及类别映射，另做未来字段清除测试。"""
    from quant_backtest.weekly_t_timing_models import predict_timing_model
    for file in path.iterdir():
        if file.is_file():
            record(file, result)
    metadata = read_json(path/'model_metadata.json')
    bundle = joblib.load(path/'model_bundle.joblib')
    model_cfg = {**cfg, 't_label_horizon_minutes': horizon, 't_model_profile': profile}
    assert metadata == bundle['metadata'] and metadata['resolved_config'] == model_cfg
    assert read_json(path/'resolved_config.json') == model_cfg
    core, excluded = sorted(fold['core_years']), sorted([fold['test_year'], fold['inner_year']])
    assert metadata['train_years'] == core and metadata['excluded_years'] == excluded
    assert metadata['strict_feature_provenance'] is True and metadata['training_rows'] == len(training)
    assert metadata['raw_target_unchanged'] is True and metadata['cohort_mean_subtracted'] is False
    assert not any(metadata[key] for key in ['class_reweighting', 'oversampling', 'probability_calibration'])
    assert bundle['target_mode'] == metadata['t_target_mode'] == 'raw_return_and_cost_profit_class'
    assert metadata['class_mapping'] == CLASS_MAPPING and metadata['horizon_minutes'] == horizon
    assert metadata['target_cost_config'] == {key: float(cfg[key]) for key in COSTS}
    features, audit = selected
    assert features == metadata['feature_names'] == bundle['features'] == (path/'feature_list.txt').read_text().splitlines()
    assert len(features) <= 60 and all(name.startswith('f_') for name in features) and 'f_history_days' not in features
    pd.testing.assert_frame_equal(audit, csv(path/'feature_selection_audit.csv'), check_dtype=False,
                                  check_exact=False, atol=1e-14, rtol=1e-14)
    cap = int(cfg['t_max_train_rows'])
    fitted = training.sample(n=cap, random_state=int(cfg['random_state'])).sort_index() if cap and len(training) > cap else training
    assert len(fitted) == metadata['fitted_rows'] and metadata['training_row_cap'] == cap
    labels = labels.copy()
    labels['fitted'] = labels.index.isin(fitted.index)
    labels = labels[TARGET_COLUMNS]
    exact(labels, csv(path/'training_targets.csv'))
    numeric = fitted[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    keys = fitted[['symbol', 'datetime']]
    fit_targets = labels.loc[fitted.index, ['raw_target_return', 'buy_net_return', 'sell_net_return', 'profit_class']]
    hashes = {
        'training_target_sha256': frame_digest(labels.drop(columns='fitted')),
        'fitted_target_sha256': frame_digest(pd.concat([keys, fit_targets], axis=1)),
        'training_input_sha256': frame_digest(pd.concat([keys, numeric], axis=1)),
        'training_sample_sha256': frame_digest(pd.concat([keys, numeric, fit_targets], axis=1)),
        'feature_source_sha256': frame_digest(fitted[['symbol', 'datetime', 'feature_source_start']]),
        'full_feature_source_sha256': frame_digest(training[['symbol', 'datetime', 'feature_source_start']]),
        'training_targets_file_sha256': record(path/'training_targets.csv', result),
        'support_file_sha256': record(path/'support.json', result),
    }
    for name, value in hashes.items():
        assert metadata[name] == value, (str(path), name, metadata[name], value)
    for name, values in [('training_class_counts', labels.profit_class), ('fitted_class_counts', fit_targets.profit_class)]:
        assert metadata[name] == {str(k): int(values.eq(k).sum()) for k in (0, 1, 2)}
    for name, count in {'training_clock_count': training.datetime.nunique(),
                        'training_day_count': training.datetime.dt.normalize().nunique(),
                        'training_week_count': training.datetime.dt.to_period('W').nunique(),
                        'training_symbol_count': training.symbol.nunique()}.items():
        assert metadata[name] == count
    assert metadata['source_sha256'] == {name: file_digest(ROOT/'quant_backtest'/name)
                                       for name in ['weekly_t_timing_models.py', 'weekly_t_models.py', 'weekly_t_short_horizon.py']}
    params = {'n_estimators': 160, 'learning_rate': .04, 'max_depth': 4, 'num_leaves': 15,
              'min_child_samples': 50, 'colsample_bytree': .85, 'subsample': .85, 'subsample_freq': 1,
              'reg_lambda': 8., 'reg_alpha': .1, 'random_state': int(cfg['random_state']), 'verbosity': -1}
    params.update(cfg['t_model'])
    params.update(objective='regression', max_depth=6 if profile == 'l2_depth6' else 4,
                  num_leaves=31 if profile == 'l2_depth6' else 15,
                  n_jobs=max(1, int(cfg['workers'])), device_type='cpu', deterministic=True, force_col_wise=True)
    classifier_params = {**params, 'objective': 'multiclass', 'num_class': 3}
    assert metadata['model_parameters'] == params and metadata['classifier_parameters'] == classifier_params
    assert set(bundle['models']) == {'return', 'profit_class'}
    np.testing.assert_array_equal(bundle['models']['profit_class'].classes_, [0, 1, 2])
    native_checks = {}
    for head, required in [('return', params), ('profit_class', classifier_params)]:
        model = bundle['models'][head]
        assert all(model.get_params()[key] == value for key, value in required.items())
        native = lightgbm.Booster(model_file=str(path/f'{head}_model.txt'))
        assert native.feature_name() == features
        assert native.model_to_string() == model.booster_.model_to_string()
        x = numeric.head(257)
        expected = model.predict(x, num_threads=1) if head == 'return' else model.predict_proba(x, num_threads=1)
        actual = native.predict(x, num_threads=1)
        np.testing.assert_array_equal(expected, actual)
        native_checks[head] = {'all_tree_weights_match': True, 'prediction_rows': len(x),
                               'max_abs_error': 0., 'native_sha256': record(path/f'{head}_model.txt', result)}
    source = training.head(257).copy()
    before_sha = frame_digest(source)
    first = predict_timing_model(bundle, source, model_cfg)
    cleared = source.copy()
    future = [name for name in cleared if name.startswith(('execution_', 'recovery_', 'target_', 'label_'))]
    cleared[future] = np.nan
    second = predict_timing_model(bundle, cleared, model_cfg)
    compare = ['datetime', 'symbol', *features, 'prediction', 'raw_prediction', *PROBABILITIES, 'horizon_minutes']
    exact(first[compare], second[compare])
    assert frame_digest(source) == before_sha
    np.testing.assert_allclose(first[PROBABILITIES].sum(axis=1), 1., atol=1e-12, rtol=0)
    exact(first[['prediction']], first[['raw_prediction']].rename(columns={'raw_prediction': 'prediction'}))
    support = audit_support(path/'support.json', training, cfg, core, excluded)
    assert support['passed'] and bundle['support'] == read_json(path/'support.json')
    assert metadata['support_training_rows'] == len(training)
    assert metadata['support_training_inputs_sha256'] == support['training_inputs_sha256']
    overlap, by_year = interval_overlap(training, excluded, int(cfg['embargo_days']))
    assert not overlap.any()
    return {'passed': True, 'core_years': core, 'excluded_years': excluded, 'training_rows': len(training),
            'fitted_rows': len(fitted), 'features': features, 'timing_features_selected': sorted(set(features)&set(TIMING_COLUMNS)),
            'overlap_rows': int(overlap.sum()), 'overlap_by_excluded_year': by_year,
            'hashes': hashes, 'raw_and_cost_labels_match': True, 'feature_selection_core_only_matches': True,
            'future_field_clear_predictions_unchanged': True, 'input_not_mutated': True,
            'native_heads': native_checks, 'support': support, 'training_class_counts': metadata['training_class_counts']}


def validation_context(report: Path, fold: dict, universe: pd.DataFrame, minutes: pd.DataFrame,
                       cfg: dict, result: dict) -> pd.DataFrame:
    """重建内部推断的冻结原始周名单/模型身份；不读取任何外层预测或收益。"""
    directory = report/f"year_{fold['test_year']}"
    original = Path(cfg['frozen_selection_report_dir'])/directory.name
    selected = csv(original/'validation_weekly_selections.csv', result)
    old_audit = read_json(Path(cfg['source_training_report_dir'])/'independent_residual_audit.json')
    expected = old_audit['frozen_contexts'][f'{directory.name}/validation']
    assert record(original/'validation_weekly_selections.csv', result) == expected['source_selection_sha256']
    if cfg['smoke_weeks']:
        selected = selected.loc[selected.trade_week.isin(sorted(selected.trade_week.unique())[:cfg['smoke_weeks']])].copy()
    if cfg['max_symbols']:
        selected = selected.loc[selected.symbol.isin(sorted(selected.symbol.unique())[:cfg['max_symbols']])].copy()
    exact(selected, csv(directory/'validation_weekly_selections.csv', result))
    path = original/'inner_weekly_model'
    model_sha = record(path/'model_bundle.joblib', result)
    meta_sha = record(path/'model_metadata.json', result)
    metadata = read_json(path/'model_metadata.json')
    assert model_sha == expected['context']['source_model_sha256']
    assert meta_sha == expected['context']['source_metadata_sha256']
    assert metadata['train_years'] == fold['core_years']
    assert metadata['excluded_years'] == sorted([fold['inner_year'], fold['test_year']])
    selected[TRACE[0]] = json.dumps(fold['core_years'], separators=(',', ':'))
    selected[TRACE[1]] = model_sha
    current = minutes.merge(selected[KEYS+['entry_date', 'exit_date']], on=KEYS, validate='many_to_one')
    current = current.loc[current.date.gt(current.entry_date)&current.date.lt(current.exit_date)].copy()
    current = independent_context(current, selected, universe, cfg)
    signature = frame_digest(current[['datetime', 'symbol', 'feature_source_start', *CONTEXT.values(), *TRACE]])
    manifest = read_json(directory/'validation_weekly_context_manifest.json')
    record(directory/'validation_weekly_context_manifest.json', result)
    assert manifest['source_model_sha256'] == model_sha and manifest['source_metadata_sha256'] == meta_sha
    assert manifest['context_inputs_sha256'] == signature and manifest['context_rows'] == len(current)
    result['validation_contexts'][str(fold['test_year'])] = {'passed': True, 'rows': len(current),
        'context_inputs_sha256': signature, 'weekly_model_sha256': model_sha, 'core_years': fold['core_years']}
    return current


def audit_predictions(path: Path, model_dir: Path, context: pd.DataFrame, cfg: dict, result: dict) -> dict:
    """逐行从独立完整截面重建模型输入，以原生两头核内部保存的数值和完整推断资格。"""
    saved = csv(path, result)
    metadata = read_json(model_dir/'model_metadata.json')
    features = metadata['feature_names']
    current = context.copy()
    if cfg.get('exclude_flat_signal_bars', True) and 'signal_bar_tradable' in current:
        current = current.loc[current.signal_bar_tradable.eq(True)].copy()
    current = current.sort_values(['datetime', 'symbol'], kind='stable').reset_index(drop=True)
    exact(current[['datetime', 'symbol', *features]], saved[['datetime', 'symbol', *features]])
    extra = [name for name in saved if name.startswith('f_') and name not in features]
    if extra:
        exact(current[extra], saved[extra])
    x = current[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    checks = {}
    for head, columns in [('return', ['prediction']), ('profit_class', PROBABILITIES)]:
        model = lightgbm.Booster(model_file=str(model_dir/f'{head}_model.txt'))
        actual = model.predict(x, num_threads=1) if len(x) else np.empty((0, len(columns)))
        expected = saved[columns].to_numpy()
        actual = np.asarray(actual).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        checks[head] = float(np.max(np.abs(actual-expected))) if len(actual) else 0.
    np.testing.assert_array_equal(saved.prediction, saved.raw_prediction)
    np.testing.assert_allclose(saved[PROBABILITIES].sum(axis=1), 1, rtol=0, atol=1e-12)
    assert saved.horizon_minutes.eq(metadata['horizon_minutes']).all()
    assert not any(name.startswith(('target_', 'label_', 'execution_', 'recovery_')) for name in saved)
    return {'passed': True, 'rows': len(current), 'independently_rebuilt_input_matches': True,
            'native_prediction_max_errors': checks, 'extra_causal_features_checked': extra,
            'prediction_csv_sha256': record(path, result)}


def execute(report: Path, smoke: bool, requested: list[int] | None, result: dict) -> None:
    """全缓存只读一次，逐折逐期限重建；禁止读取恢复报价或外层账户。"""
    from quant_backtest.weekly_t_features import add_weekly_execution_features, add_session_context_features
    from quant_backtest.weekly_t_provenance import add_feature_provenance
    cfg = read_json(report/'resolved_config.json')
    record(report/'resolved_config.json', result)
    assert smoke == bool(cfg['max_symbols'] or cfg['smoke_weeks'])
    assert cfg['research_version'] == 'weekly_t_v11_cost_aware_timing'
    assert cfg['strict_feature_provenance'] and not cfg['outer_t_refit']
    years = sorted(requested or cfg['test_years'])
    assert set(years).issubset(cfg['test_years'])
    result.update(requested_years=years, configured_years=cfg['test_years'], smoke=smoke)
    result['sources'] = sources(report, smoke, result)
    source = Path(cfg['source_training_report_dir'])
    old_cfg = read_json(source/'resolved_config.json')
    record(source/'resolved_config.json', result)
    old_audit = read_json(source/'independent_residual_audit.json')
    record(source/'independent_residual_audit.json', result)
    assert all(old_audit[name] for name in ['passed', 'complete', 'full_configured_scope_complete'])
    weekly_manifest, horizon_manifest, cache_identity = verify_cache_inputs(source, old_cfg)
    assert cache_identity['passed']
    result['cache_identity'] = cache_identity
    for name in ['data_manifest.json', 'horizon_manifest.json', 'feature_provenance_manifest.json']:
        record(report/name, result)
    horizon_report = read_json(report/'horizon_manifest.json')
    assert all(horizon_report.get(name) == value for name, value in horizon_manifest.items())
    assert horizon_report['future_targets_used_only_by_core_training'] is True
    protocol = read_json(report/'research_protocol.json')
    assert protocol['selection_uses_inner_only'] and protocol['all_rules_frozen_before_outer_test'] and not protocol['outer_model_refit']
    freeze = read_json(report/'all_rules_frozen_before_outer_test.json')
    marker = report/'all_rules_frozen_before_outer_test.json'
    record(marker, result)
    assert sorted(freeze['test_years']) == sorted(cfg['test_years']) and freeze['selection_uses_inner_only']
    folds = {}
    for year in cfg['test_years']:
        folder = report/f'year_{year}'
        fold = read_json(folder/'split.json')
        record(folder/'split.json', result)
        assert fold == read_json(source/f'year_{year}/split.json')
        assert sorted(fold['train_years']) == sorted(fold['core_years']+[fold['inner_year']])
        assert not set(fold['core_years']) & {year, fold['inner_year']}
        assert year not in fold['train_years'] and fold['outer_t_refit'] is False
        if year == 2026:
            assert fold['train_years'] == [2023, 2024, 2025] and fold['forward_only'] is True
        rule = read_json(folder/'selected_rule.json')
        assert record(folder/'selected_rule.json', result) == freeze['selected_rule_sha256'][str(year)]
        assert (folder/'selected_rule.json').stat().st_mtime_ns <= marker.stat().st_mtime_ns
        hashes = {f'h{h}_{p}': record(folder/f'inner_t_h{h}_{p}/model_bundle.joblib', result)
                  for h in cfg['horizons'] for p in cfg['t_model_profiles']}
        assert rule['model_bundle_sha256'] == hashes and rule['fold'] == fold
        assert rule['old_t_weights_reused'] is False and rule['frozen_before_any_outer_test'] is True
        for prediction in folder.glob('test_*_predictions.csv'):
            assert prediction.stat().st_mtime_ns >= marker.stat().st_mtime_ns
        folds[year] = fold
        result['frozen_models'][str(year)] = {'passed': True, 'model_sha256': hashes,
            'selected_rule_sha256': freeze['selected_rule_sha256'][str(year)], 'outer_refit': False}
    print('来源、缓存身份、全折冻结与全部模型 bundle 身份核对完成；读取三个旧行情表和短期标签。', flush=True)
    cache = Path(weekly_manifest['cache_path'])
    frames = []
    for name in ['weekly', 'intraday', 'daily']:
        path = cache/f'{name}.pkl'
        record(path, result)
        frame = pd.read_pickle(path)
        assert len(frame) == weekly_manifest[f'{name}_rows']
        frames.append(frame)
    weekly, minutes, daily = frames
    del frames
    weekly = add_weekly_execution_features(weekly, daily)
    minutes = add_session_context_features(minutes)
    weekly, minutes, provenance = add_feature_provenance(weekly, minutes, daily)
    assert provenance == read_json(report/'feature_provenance_manifest.json')
    del daily
    gc.collect()
    print('完整旧来源重建完成；独立计算全截面十一特征。', flush=True)
    minutes = independent_timing_features(minutes)
    amount = weekly.f_week_amount/weekly.f_week_days
    mask = (weekly.f_history_days.ge(cfg['min_history_days']) & weekly.f_last_close.ge(cfg['min_price'])
            & amount.ge(cfg['min_avg_daily_amount']) & weekly.feature_week_complete.eq(True))
    if cfg.get('min_prior_close_auction_amount', 0) > 0:
        mask &= weekly.f_prior_close_auction_amount_min.ge(cfg['min_prior_close_auction_amount'])
    universe = weekly.loc[mask].copy()
    universe['universe_previous_close'] = weekly.f_last_close.loc[mask]
    universe['universe_avg_daily_amount'] = amount.loc[mask]
    del weekly
    label_path = Path(horizon_manifest['labels_path'])
    assert record(label_path, result) == horizon_report['labels_file_sha256']
    labels = pd.read_pickle(label_path)
    assert len(labels) == horizon_manifest['label_rows'] and list(labels) == horizon_manifest['label_columns']
    result['full_cross_section'] = {'rows': len(minutes), 'symbols': minutes.symbol.nunique(),
        'clock_count': minutes.datetime.nunique(), 'independently_rebuilt_features': TIMING_COLUMNS,
        'provenance_matches': True, 'timing_features_before_selection': True, 'read_recovery_quotes': False}
    save(report, result)
    for year in years:
        fold = folds[year]
        directory = report/f'year_{year}'
        cohort = rebuild_cohort(universe, minutes, fold, cfg, report, result)
        core_labels = labels.merge(cohort[['datetime', 'symbol']], on=['datetime', 'symbol'], validate='many_to_one')
        for horizon in cfg['horizons']:
            kept, domain = audit_training_domain(cohort, universe, horizon, cfg, directory/f'training_domain_h{horizon}')
            record(directory/f'training_domain_h{horizon}.json', result)
            record(directory/f'training_domain_h{horizon}.csv', result)
            result['training_domains'][f'{year}/h{horizon}'] = domain
            joined = attach_labels(kept, core_labels, horizon)
            training = independent_training_rows(joined, fold['core_years'], [year, fold['inner_year']], cfg)
            targets = labels_from_prices(training, horizon, cfg)
            selected = feature_selection(training, targets, cfg)
            for profile in cfg['t_model_profiles']:
                name = f'inner_t_h{horizon}_{profile}'
                result['models'][f'{year}/{name}'] = audit_model(directory/name, training, targets, selected,
                                                                fold, horizon, profile, cfg, result)
                print(f'{year}/{name}：{len(training):,}核心行，独立 raw/类别/特征/来源 SHA 与双头全部通过。', flush=True)
                save(report, result)
            del joined, training, kept, targets, selected
        del cohort, core_labels
        gc.collect()
        context = validation_context(report, fold, universe, minutes, cfg, result)
        for horizon in cfg['horizons']:
            for profile in cfg['t_model_profiles']:
                name = f'h{horizon}_{profile}'
                result['validation_predictions'][f'{year}/{name}'] = audit_predictions(
                    directory/f'validation_{name}_predictions.csv', directory/f'inner_t_{name}', context, cfg, result)
        del context
        save(report, result)
        print(f'{year}：六模型内部预测与完整当前截面输入逐行一致。', flush=True)
    del minutes, universe, labels
    gc.collect()
    for path, identity in result['inputs'].items():
        assert file_digest(Path(path)) == identity['sha256'], ('审计期间来源改变', path)
    result['sources_end'] = sources(report, smoke, result)
    result['complete'] = result['passed'] = True
    result['full_configured_scope_complete'] = years == sorted(cfg['test_years'])
    result['strict_raw_price_feature_boundary_passed'] = True
    result['historical_retest_not_new_untouched_test'] = True


def main() -> None:
    """保留可复现脚本身份、异常栈和完整审计范围。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--test-years', nargs='+', type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    result = {'started_utc': datetime.now(timezone.utc).isoformat(), 'passed': False, 'complete': False,
              'audit_script_sha256': file_digest(Path(__file__)), 'inputs': {}, 'models': {}, 'cohorts': {},
              'training_domains': {}, 'validation_contexts': {}, 'validation_predictions': {}, 'frozen_models': {},
              'independent_helpers': {name: file_digest(ROOT/'scripts'/name) for name in
                  ['audit_weekly_t_short_horizon.py', 'audit_weekly_t_residual.py']},
              'scope': '独立重建核心训练、标签、十一新特征及内部预测；不读取外层账户/收益'}
    try:
        execute(args.report_dir.resolve(), args.smoke, args.test_years, result)
    except Exception as error:
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        save(args.report_dir.resolve(), result)
    print(f'全部通过：{len(result["models"])} 模型，{result["elapsed_seconds"]:.2f} 秒。', flush=True)


if __name__ == '__main__':
    main()
