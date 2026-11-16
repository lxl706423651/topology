"""新增因果择时特征及25个预声明候选，在信号当刻选择方向与20/40/60分钟期限。

特征须在完整同刻股票池计算后再切片；不读取未来标签、执行价或恢复结果。
决策适配原库存T引擎，保留后来请求；选参只消费内部真实账户净利及五项执行失败。
本模块不写文件、不修改旧缓存、行情、输入表或模型。
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from .weekly_t_activation import EXECUTION_FAILURE_COLUMNS, REQUIRED_SELECTION_COLUMNS, validate_selection_values
from .weekly_t_features import select_recoverable_signal_sides
from .weekly_t_short_horizon import feature_support_mask, planned_recovery_times, predicted_pair_edges


TIMING_PROFILES = ('l2_depth4', 'l2_depth6')
TIMING_HORIZONS = (20, 40, 60)
TIMING_REGIMES = ('learned', 'reversal', 'momentum')
TIMING_FEATURE_COLUMNS = [
    'f_timing_market_return_5m', 'f_timing_market_return_15m', 'f_timing_market_breadth',
    'f_timing_market_day_return_std', 'f_timing_excess_return_5m', 'f_timing_excess_return_15m',
    'f_timing_volume_rate', 'f_timing_trend_efficiency', 'f_timing_sigma20',
    'f_timing_vwap_z', 'f_timing_reversal_confirmation',
]
DECISION_COLUMNS = ['datetime', 'symbol', 'action', 'fraction', 'priority', 'planned_recovery_time',
                    'date', 'candidate_index', 'horizon_minutes', 'predicted_net_edge', 'prob_profit',
                    'prediction', 'raw_prediction']


def validate_timing_number(value, name: str, lower: float, upper: float) -> float:
    """验证固定规则数值，拒绝布尔、字符串、非有限值及越界值。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f'{name}必须为有限数值')
    if not lower <= value <= upper:
        raise ValueError(f'{name}不在允许范围')
    return float(value)


def validate_timing_keys(frame: pd.DataFrame) -> pd.Series:
    """检查当前datetime/symbol唯一；返回解析后的时刻，输入索引与列保持不变。"""
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError('择时输入必须是列名唯一的DataFrame')
    if not {'datetime', 'symbol'}.issubset(frame):
        raise ValueError('择时输入缺datetime/symbol')
    times = pd.to_datetime(frame.datetime, errors='raise')
    if times.isna().any() or times.dt.tz is not None or frame.symbol.isna().any():
        raise ValueError('择时键不能为空，datetime须为本地无时区时刻')
    if frame.symbol.astype(str).str.strip().eq('').any():
        raise ValueError('股票代码不能为空')
    keys = pd.DataFrame({'datetime':times.to_numpy(), 'symbol':frame.symbol.astype(str).to_numpy()})
    if keys.duplicated().any():
        raise ValueError('同一datetime/symbol只能存在一个当前观测')
    return times


def add_timing_features(full_intraday: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """复制完整股票池，按同刻有限观测增加11列；不扩大原feature_source_start。

市场std为完整当前截面的总体标准差(ddof=0)，breadth以有限开盘收益为分母。
当前5/15分钟收益、日内累计量价和已有前日振幅均在旧来源范围内；缺失值保留
NaN而不制造历史观测。调用方须在任何选股切片之前执行，不能传入未来标签列代替。
cfg保留为统一接口，本轮公式及数值稳定常量固定，不由外层结果重估。
"""
    times = validate_timing_keys(full_intraday)
    required = ['f_return_5m', 'f_return_15m', 'f_open_return', 'f_day_return',
                'f_amount_vs_previous_day', 'f_session_fraction', 'f_known_amplitude',
                'f_realized_volatility', 'f_previous_day_amplitude', 'f_vwap_deviation']
    missing = set(required).difference(full_intraday)
    if missing:
        raise ValueError(f'择时特征缺当前字段：{sorted(missing)}')
    result = full_intraday.copy(deep=True)
    values = {key:pd.to_numeric(result[key], errors='coerce').replace([np.inf, -np.inf], np.nan)
              for key in required}
    ret5, ret15 = values['f_return_5m'], values['f_return_15m']
    opened, day = values['f_open_return'], values['f_day_return']
    session = values['f_session_fraction'].where(values['f_session_fraction'].between(0., 1.))
    amplitude = values['f_known_amplitude'].where(values['f_known_amplitude'].ge(0.))
    previous = values['f_previous_day_amplitude'].where(values['f_previous_day_amplitude'].ge(0.))
    realized = values['f_realized_volatility'].where(values['f_realized_volatility'].ge(0.))
    market5 = ret5.groupby(times, sort=False).transform('mean')
    market15 = ret15.groupby(times, sort=False).transform('mean')
    breadth = opened.gt(0).astype(float).where(opened.notna()).groupby(times, sort=False).transform('mean')
    dispersion = day.groupby(times, sort=False).transform('std', ddof=0)
    volume = values['f_amount_vs_previous_day'].where(values['f_amount_vs_previous_day'].ge(0.)) / session.clip(lower=.05)
    efficiency = (opened.abs() / amplitude.clip(lower=1e-6)).clip(0., 1.)
    sigma = pd.Series(np.maximum.reduce([
        (realized * np.sqrt(20. / (240. * session).clip(lower=5.))).to_numpy(),
        (previous * .25 * np.sqrt(20. / 240.)).to_numpy(),
        np.full(len(result), .0005),
    ]), index=result.index)
    outputs = [market5, market15, breadth, dispersion, ret5-market5, ret15-market15,
               volume, efficiency, sigma, values['f_vwap_deviation']/sigma,
               -np.sign(values['f_vwap_deviation'])*ret5/sigma]
    for name, value in zip(TIMING_FEATURE_COLUMNS, outputs):
        result[name] = value.replace([np.inf, -np.inf], np.nan).astype('float32')
    return result


def timing_candidates(cfg: dict) -> list[dict]:
    """固定顺序生成显式hold及24个主动候选；不接受事后扩大参数空间。"""
    grids = [
        ('timing_profiles', tuple(cfg.get('timing_profiles', cfg.get('t_model_profiles', TIMING_PROFILES))), TIMING_PROFILES),
        ('timing_horizons', tuple(cfg.get('timing_horizons', TIMING_HORIZONS)), TIMING_HORIZONS),
        ('timing_regimes', tuple(cfg.get('timing_regimes', TIMING_REGIMES)), TIMING_REGIMES),
        ('timing_min_net_edges', tuple(cfg.get('timing_min_net_edges', (.0005, .001))), (.0005, .001)),
        ('timing_min_profit_probabilities', tuple(cfg.get('timing_min_profit_probabilities', (.4, .5))), (.4, .5)),
    ]
    for name, actual, expected in grids:
        if actual != expected or any(isinstance(value, (bool, np.bool_)) for value in actual):
            raise ValueError(f'{name}必须保持预声明候选：{expected}')
    fraction = validate_timing_number(cfg['short_t_fraction'], 'short_t_fraction', 0., 1.)
    if fraction != .25:
        raise ValueError('择时实验保留原25%库存比例')
    candidates = [{'candidate_index':0, 'enabled':False, 'profile':'hold', 'regime':'hold',
                   'min_net_edge':0., 'min_profit_probability':0., 'fraction':fraction, 'direction':'both'}]
    for profile in TIMING_PROFILES:
        for regime in TIMING_REGIMES:
            for margin in (.0005, .001):
                for probability in (.4, .5):
                    candidates.append({'candidate_index':len(candidates), 'enabled':True, 'profile':profile,
                                       'regime':regime, 'min_net_edge':margin, 'min_profit_probability':probability,
                                       'fraction':fraction, 'direction':'both'})
    return candidates


def make_empty_timing_decisions() -> pd.DataFrame:
    """返回可直接交给旧引擎的稳定空schema，显式hold不伪造请求。"""
    dates = {'datetime', 'date', 'planned_recovery_time'}
    integers = {'action', 'candidate_index', 'horizon_minutes'}
    return pd.DataFrame({name:pd.Series(dtype='datetime64[ns]' if name in dates else
                                      'int64' if name in integers else 'object' if name=='symbol' else 'float64')
                         for name in DECISION_COLUMNS})


def make_timing_decisions(predictions_by_horizon: dict, support_by_horizon: dict,
                          candidate: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """当前逐期限/方向门控后择最高正净边际；并列短期限、再买先，保留后续请求。

prediction仅保留为模型审计字段，成本换算使用raw_prediction。prob_buy_profit和
prob_sell_profit是对应方向的扣费盈利概率；没有未来价格、标签或可成交量资格筛选。
所有期限必须覆盖相同当前键；支持域分别使用各期限核心训练结果，不在推断期重拟合。
"""
    rules = timing_candidates(cfg)
    index = candidate.get('candidate_index')
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(rules):
        raise ValueError('candidate_index须对应预声明25个候选')
    fixed = rules[index]
    for key, value in fixed.items():
        if key in candidate and candidate[key] != value:
            raise ValueError(f'候选{index}的{key}偏离事前配置')
    audit = {'candidate_index':int(index), 'enabled':fixed['enabled'], 'horizon_side_funnel':[],
             'input_rows_by_horizon':{}, 'eligible_alternatives':0, 'selected_requests':0,
             'reason':'explicit_hold' if index==0 else 'no_current_opportunity',
             'scope':'当前信号资格及期限选择；不代表实际成交或未来可恢复'}
    if index == 0:
        return make_empty_timing_decisions(), audit
    if set(predictions_by_horizon) != set(TIMING_HORIZONS) or set(support_by_horizon) != set(TIMING_HORIZONS):
        raise ValueError('预测与支持域必须分别提供整数键20/40/60')
    day_limit = validate_timing_number(cfg.get('timing_max_abs_day_return', .03), 'timing_max_abs_day_return', 0., 1.)
    amplitude_limit = validate_timing_number(cfg.get('timing_max_known_amplitude', .06), 'timing_max_known_amplitude', 0., 1.)
    if day_limit != .03 or amplitude_limit != .06 or cfg['latest_planned_recovery_time'] != '14:30':
        raise ValueError('本轮保持3%日涨跌、6%振幅和14:30期限截止')
    alternatives, expected_keys = [], None
    for horizon in TIMING_HORIZONS:
        frame = predictions_by_horizon[horizon]
        times = validate_timing_keys(frame)
        if 'horizon_minutes' in frame and not frame.horizon_minutes.eq(horizon).all():
            raise ValueError('预测表期限身份与字典horizon键不一致')
        if 'model_profile' in frame and not frame.model_profile.eq(fixed['profile']).all():
            raise ValueError('预测表模型profile与当前候选不一致')
        required = {'prediction', 'raw_prediction', 'prob_buy_profit', 'prob_sell_profit',
                    'f_day_return', 'f_known_amplitude'}
        if fixed['regime'] != 'learned':
            required.update(['f_timing_trend_efficiency', 'f_vwap_deviation', 'f_return_5m'])
        if fixed['regime'] == 'reversal':
            required.add('f_timing_vwap_z')
        if fixed['regime'] == 'momentum':
            required.update(['f_return_15m', 'f_open_return'])
        missing = required.difference(frame)
        if missing:
            raise ValueError(f'择时预测缺当前字段：{sorted(missing)}')
        keys = pd.DataFrame({'datetime':times.to_numpy(), 'symbol':frame.symbol.astype(str).to_numpy()})
        keys = keys.sort_values(['datetime', 'symbol'], kind='stable').reset_index(drop=True)
        if expected_keys is not None and not keys.equals(expected_keys):
            raise ValueError('各期限当前预测键不一致，禁止通过未来标签缺失删候选')
        expected_keys = keys
        numeric = {key:pd.to_numeric(frame[key], errors='coerce') for key in required}
        raw, prediction = numeric['raw_prediction'], numeric['prediction']
        current = pd.DataFrame({'datetime':times, 'prediction':raw}, index=frame.index)
        buy, sell = predicted_pair_edges(current, cfg)
        planned = planned_recovery_times(times, horizon)
        common = [
            ('prediction', np.isfinite(raw) & raw.gt(-1.) & np.isfinite(prediction) & prediction.gt(-1.)),
            ('support', feature_support_mask(frame, support_by_horizon[horizon])),
            ('clock', planned.notna() & planned.dt.normalize().eq(times.dt.normalize())
             & planned.dt.strftime('%H:%M').le(cfg['latest_planned_recovery_time'])),
            ('day_risk', np.isfinite(numeric['f_day_return']) & numeric['f_day_return'].abs().le(day_limit)),
            ('amplitude_risk', np.isfinite(numeric['f_known_amplitude']) & numeric['f_known_amplitude'].between(0., amplitude_limit)),
        ]
        audit['input_rows_by_horizon'][str(horizon)] = len(frame)
        for action, edge, probability_name in [(1, buy, 'prob_buy_profit'), (-1, sell, 'prob_sell_profit')]:
            probability = numeric[probability_name]
            regime = pd.Series(True, index=frame.index)
            if fixed['regime'] != 'learned':
                efficiency, vwap, recent = (numeric[name] for name in ['f_timing_trend_efficiency', 'f_vwap_deviation', 'f_return_5m'])
                regime &= np.isfinite(efficiency) & efficiency.between(0., 1.)
                regime &= np.isfinite(vwap) & np.isfinite(recent) & (action*recent).gt(0.)
                if fixed['regime']=='reversal':
                    z = numeric['f_timing_vwap_z']
                    regime &= (action*vwap).lt(0.) & efficiency.le(.65) & np.isfinite(z) & z.abs().ge(.75)
                else:
                    ret15, opened = numeric['f_return_15m'], numeric['f_open_return']
                    regime &= (action*vwap).gt(0.) & efficiency.ge(.55)
                    regime &= np.isfinite(ret15) & np.isfinite(opened) & (action*ret15).gt(0.) & (action*opened).gt(0.)
            stages = [*common, ('net_edge', np.isfinite(edge) & edge.ge(fixed['min_net_edge'])),
                      ('profit_probability', np.isfinite(probability) & probability.between(0., 1.)
                       & probability.ge(fixed['min_profit_probability'])), ('regime', regime),
                      ('exit_guard', select_recoverable_signal_sides(frame, pd.Series(action, index=frame.index), cfg))]
            eligible = pd.Series(True, index=frame.index)
            counts = {'horizon_minutes':horizon, 'action':action, 'input_rows':len(frame)}
            for name, mask in stages:
                eligible &= mask
                counts['after_'+name] = int(eligible.sum())
            audit['horizon_side_funnel'].append(counts)
            part = pd.DataFrame({'datetime':times, 'symbol':frame.symbol.astype(str), 'action':action,
                                 'fraction':fixed['fraction'], 'priority':edge, 'planned_recovery_time':planned,
                                 'date':times.dt.normalize(), 'candidate_index':index, 'horizon_minutes':horizon,
                                 'predicted_net_edge':edge, 'prob_profit':probability,
                                 'prediction':prediction, 'raw_prediction':raw}, index=frame.index)
            alternatives.append(part.loc[eligible, DECISION_COLUMNS])
    eligible = pd.concat(alternatives, ignore_index=True)
    audit['eligible_alternatives'] = len(eligible)
    if eligible.empty:
        return make_empty_timing_decisions(), audit
    chosen = eligible.sort_values(['datetime', 'symbol', 'predicted_net_edge', 'horizon_minutes', 'action'],
                                  ascending=[True, True, False, True, False], kind='stable')
    chosen = chosen.drop_duplicates(['datetime', 'symbol'], keep='first').reset_index(drop=True)
    audit.update(selected_requests=len(chosen), selected_days=int(chosen.date.nunique()),
                 selected_stock_days=len(chosen[['date', 'symbol']].drop_duplicates()), reason='current_opportunities_selected',
                 horizon_request_counts={str(key):int(value) for key,value in chosen.horizon_minutes.value_counts().items()},
                 direction_request_counts={str(key):int(value) for key,value in chosen.action.value_counts().items()})
    return chosen[DECISION_COLUMNS], audit


def select_timing_candidate(table: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """仅内部净利为正、完整pair>0且五项执行失败为0者按净利选；否则显式hold。

返回选中行及完整注释表。不存在交易数下限、置信区间或频率优先门槛；index0
必须是真实零T基准，不能用关闭开关掩盖一个已经存在交易的候选。
"""
    if not isinstance(table, pd.DataFrame) or table.empty or table.columns.duplicated().any():
        raise ValueError('内部候选表须非空且列名唯一')
    missing = set(REQUIRED_SELECTION_COLUMNS).difference(table)
    if missing:
        raise ValueError(f'候选缺字段：{sorted(missing)}')
    numbers = {name:validate_selection_values(table, name, count=name!='incremental_net_pnl')
               for name in REQUIRED_SELECTION_COLUMNS}
    indices = numbers['candidate_index']
    if indices.duplicated().any() or not indices.eq(0).any():
        raise ValueError('候选编号须唯一并包含显式index0 hold')
    if indices.gt(24).any():
        raise ValueError('候选编号不能超出事前25候选')
    if 'completed_pair_count' in table:
        completed = validate_selection_values(table, 'completed_pair_count', count=True)
        if not np.array_equal(completed.to_numpy(), numbers['t_pair_count'].to_numpy()):
            raise ValueError('完整pair计数不能与恢复片段混用')
    hold_position = int(np.flatnonzero(indices.eq(0).to_numpy())[0])
    if any(numbers[name].iloc[hold_position] != 0 for name in REQUIRED_SELECTION_COLUMNS if name!='candidate_index'):
        raise ValueError('index0 hold必须为零T、零增量且无T执行失败的真实基准')
    reasons = []
    for position in range(len(table)):
        failed = [name for name in EXECUTION_FAILURE_COLUMNS if numbers[name].iloc[position] != 0]
        if indices.iloc[position]==0:
            failed.append('explicit_hold')
        else:
            if numbers['t_pair_count'].iloc[position] == 0:
                failed.append('no_completed_t_pairs')
            if numbers['incremental_net_pnl'].iloc[position] <= 0:
                failed.append('nonpositive_net_pnl')
            if 'enabled' in table and table.iloc[position]['enabled'] != True:
                failed.append('disabled_candidate')
        reasons.append('|'.join(failed))
    annotated = table.copy(deep=True)
    annotated['timing_eligible'] = np.array([not value for value in reasons], dtype=bool)
    annotated['timing_rejection_reasons'] = reasons
    eligible = annotated.loc[annotated.timing_eligible]
    selected_index = 0 if eligible.empty else int(eligible.sort_values(
        ['incremental_net_pnl', 'candidate_index'], ascending=[False, True], kind='stable').iloc[0].candidate_index)
    annotated['selected'] = indices.eq(selected_index).to_numpy()
    annotated['selection_reason'] = np.where(annotated.selected,
        'no_profitable_execution_valid_candidate' if selected_index==0 else 'maximum_internal_net_pnl', '')
    chosen = annotated.loc[annotated.selected].iloc[0].to_dict()
    chosen['enabled'] = selected_index != 0
    return chosen, annotated
