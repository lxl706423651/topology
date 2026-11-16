"""固定25候选的主动T频率规则；仅使用当前因果信号，真实费用与账户约束保持原引擎。

参考规则保留旧的每日请求去重；新绝对边际与VWAP排名规则保留后续合格时刻，
由原引擎限制每股每日最多一个成功首腿。排名不意味着预测已覆盖双腿成本。
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np
import pandas as pd

from .strategy import BaseStrategy
from .weekly_t_activation import select_active_candidate, validate_selection_values
from .weekly_t_features import select_recoverable_signal_sides
from .weekly_t_short_horizon import (ShortHorizonTStrategy, create_short_decisions,
    feature_support_mask, planned_recovery_times, predicted_pair_edges)

PROFILES = ('l2_depth4', 'l2_depth6', 'huber_depth4')
FUNNEL_STAGES = ('after_enabled', 'after_prediction', 'after_support', 'after_clock', 'after_direction',
                 'after_vwap_gap', 'after_reversal', 'after_day_risk', 'after_amplitude_risk',
                 'after_cost_margin', 'after_exit_guard', 'after_rank', 'request_eligible')
DECISION_COLUMNS = ['datetime', 'symbol', 'action', 'fraction', 'priority', 'planned_recovery_time', 'date']


def validate_finite_number(value, name: str, *, lower: float | None = None, upper: float | None = None) -> float:
    """验证规则数值而不接受布尔或字符串；返回有限浮点数。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f'{name}必须为有限数值')
    number = float(value)
    if (lower is not None and number < lower) or (upper is not None and number > upper):
        raise ValueError(f'{name}不在允许范围')
    return number


def create_frequency_rules(cfg: dict, reference_rule: dict) -> list[dict]:
    """按事前固定顺序生成1参考、6绝对门槛、18VWAP排名候选，不改变旧参考参数。"""
    profiles = tuple(cfg.get('t_model_profiles', PROFILES))
    margins = tuple(cfg.get('frequency_absolute_net_margins', [0., .0005]))
    quantiles = tuple(cfg.get('frequency_rank_quantiles', [.70, .85, .95]))
    gaps = tuple(cfg.get('frequency_min_vwap_gaps', [.0035, .005]))
    for name, values in (('frequency_absolute_net_margins', margins), ('frequency_rank_quantiles', quantiles), ('frequency_min_vwap_gaps', gaps)):
        for value in values:
            validate_finite_number(value, name)
    if profiles != PROFILES or margins != (0., .0005) or quantiles != (.70, .85, .95) or gaps != (.0035, .005):
        raise ValueError('本轮频率实验要求预声明的三个profile、两个绝对门槛、三个分位和两个VWAP偏差')
    fraction = validate_finite_number(cfg['short_t_fraction'], 'short_t_fraction', lower=0., upper=1.)
    if fraction != .25:
        raise ValueError('本轮保持原25% T比例')
    required = {'profile', 'horizon_minutes', 'net_margin', 'fraction', 'direction', 'enabled'}
    if not required.issubset(reference_rule):
        raise ValueError(f'原参考规则缺列：{sorted(required.difference(reference_rule))}')
    if reference_rule['enabled'] is not True or reference_rule['fraction'] != fraction or reference_rule['direction'] != 'both':
        raise ValueError('参考必须是原已开启的25%、双向T规则')
    reference = deepcopy(reference_rule)
    reference.update(candidate_index=0, signal_mode='reference', enabled=True)
    rules = [reference]
    for profile in profiles:
        for margin in margins:
            rules.append({'candidate_index':len(rules), 'signal_mode':'absolute', 'profile':profile,
                'horizon_minutes':20, 'net_margin':float(margin), 'fraction':fraction, 'direction':'both', 'enabled':True})
    for profile in profiles:
        for quantile in quantiles:
            for gap in gaps:
                rules.append({'candidate_index':len(rules), 'signal_mode':'vwap_rank', 'profile':profile,
                    'horizon_minutes':20, 'rank_quantile':float(quantile), 'min_vwap_gap':float(gap),
                    'fraction':fraction, 'direction':'both', 'enabled':True,
                    'prediction_cost_coverage_required':False,
                    'rank_selection':'同刻当前条件合格机会中abs(pred)降序、symbol升序，取前ceil((1-q)*N)，至少1个'})
    return rules


def build_frequency_signal_funnel(predictions: pd.DataFrame, rule: dict, cfg: dict, support: dict) -> pd.DataFrame:
    """返回保持输入索引的逐行门控、排序与请求漏斗，不读取未来价、标签或可成交结果。"""
    mode = rule.get('signal_mode')
    if mode not in ('reference', 'absolute', 'vwap_rank'):
        raise ValueError('signal_mode必须为reference、absolute或vwap_rank')
    required = {'datetime', 'symbol', 'prediction'}
    if mode == 'vwap_rank':
        required.update(['f_vwap_deviation', 'f_return_5m', 'f_day_return', 'f_known_amplitude'])
    if not required.issubset(predictions):
        raise ValueError(f'当前频率信号缺列：{sorted(required.difference(predictions))}')
    if predictions.duplicated(['datetime', 'symbol']).any():
        raise ValueError('同一datetime/symbol只能有一个当前预测')
    if mode != 'reference' and rule['horizon_minutes'] != 20:
        raise ValueError('新频率候选只允许20交易分钟')
    fraction = validate_finite_number(rule.get('fraction', cfg['short_t_fraction']), 'fraction', lower=0., upper=1.)
    if fraction != float(cfg['short_t_fraction']):
        raise ValueError('规则T比例必须与原账户配置一致')
    if rule.get('direction', 'both') != 'both':
        raise ValueError('本轮频率规则保持双向T')
    frame = predictions[['datetime', 'symbol']].copy()
    frame['datetime'] = pd.to_datetime(frame.datetime, errors='raise')
    if frame.datetime.isna().any() or frame.symbol.isna().any():
        raise ValueError('当前信号时间和股票代码不能缺失')
    values = pd.to_numeric(predictions.prediction, errors='coerce')
    prepared = predictions.copy(deep=False)
    prepared = prepared.assign(prediction=values, datetime=frame.datetime)
    buy, sell = predicted_pair_edges(prepared, cfg)
    frame['predicted_net_edge'] = np.maximum(buy, sell)
    frame['planned_recovery_time'] = planned_recovery_times(frame.datetime, int(rule['horizon_minutes']))
    frame['fraction'] = fraction
    frame['enabled_ok'] = bool(rule.get('enabled', True))
    frame['prediction_ok'] = np.isfinite(values) & values.gt(-1.)
    frame['support_ok'] = feature_support_mask(predictions, support)
    frame['clock_ok'] = (frame.planned_recovery_time.notna()
        & frame.planned_recovery_time.dt.normalize().eq(frame.datetime.dt.normalize())
        & frame.planned_recovery_time.dt.strftime('%H:%M').le(cfg['latest_planned_recovery_time']))
    for column in ('direction_ok', 'vwap_gap_ok', 'reversal_ok', 'day_risk_ok', 'amplitude_risk_ok', 'cost_margin_ok'):
        frame[column] = True
    if mode == 'vwap_rank':
        quantile = validate_finite_number(rule['rank_quantile'], 'rank_quantile', lower=0., upper=1.)
        if not 0 < quantile < 1:
            raise ValueError('rank_quantile必须严格位于0和1之间')
        gap = validate_finite_number(rule['min_vwap_gap'], 'min_vwap_gap', lower=0.)
        if gap == 0:
            raise ValueError('min_vwap_gap必须为正')
        day_limit = validate_finite_number(cfg.get('frequency_max_abs_day_return', .03), 'frequency_max_abs_day_return', lower=0.)
        amplitude_limit = validate_finite_number(cfg.get('frequency_max_known_amplitude', .06), 'frequency_max_known_amplitude', lower=0.)
        vwap = pd.to_numeric(predictions.f_vwap_deviation, errors='coerce')
        recent = pd.to_numeric(predictions.f_return_5m, errors='coerce')
        day = pd.to_numeric(predictions.f_day_return, errors='coerce')
        amplitude = pd.to_numeric(predictions.f_known_amplitude, errors='coerce')
        actions = pd.Series(np.sign(values.fillna(0)).astype(int), index=predictions.index)
        frame['direction_ok'] = values.ne(0) & np.isfinite(values)
        frame['vwap_gap_ok'] = np.isfinite(vwap) & ((values.gt(0) & vwap.le(-gap)) | (values.lt(0) & vwap.ge(gap)))
        frame['reversal_ok'] = np.isfinite(recent) & ((values.gt(0) & recent.ge(0)) | (values.lt(0) & recent.le(0)))
        frame['day_risk_ok'] = np.isfinite(day) & day.abs().le(day_limit)
        frame['amplitude_risk_ok'] = np.isfinite(amplitude) & amplitude.between(0., amplitude_limit)
        frame['priority'] = values.abs()
    else:
        margin = validate_finite_number(rule['net_margin'], 'net_margin', lower=0.)
        actions = pd.Series(np.where(buy.ge(sell), 1, -1), index=predictions.index)
        frame['cost_margin_ok'] = frame.predicted_net_edge.ge(margin)
        frame['priority'] = frame.predicted_net_edge
    frame['exit_guard_ok'] = select_recoverable_signal_sides(predictions, actions, cfg)
    cumulative = pd.Series(True, index=predictions.index)
    for gate in ('enabled', 'prediction', 'support', 'clock', 'direction', 'vwap_gap', 'reversal',
                 'day_risk', 'amplitude_risk', 'cost_margin', 'exit_guard'):
        cumulative = cumulative & frame[gate + '_ok']
        frame['after_' + gate] = cumulative
    frame['pre_rank_eligible'] = cumulative
    frame['rank_ok'] = True
    frame['rank_ordinal'] = np.nan
    frame['eligible_opportunity_count'] = 0
    if mode == 'vwap_rank':
        frame['rank_ok'] = False
        positions = np.flatnonzero(cumulative.to_numpy())
        ranked = pd.DataFrame({'position':positions, 'datetime':frame.datetime.to_numpy()[positions],
            'symbol':frame.symbol.astype(str).to_numpy()[positions], 'absolute_prediction':values.abs().to_numpy()[positions]})
        ranked = ranked.sort_values(['datetime', 'absolute_prediction', 'symbol'], ascending=[True, False, True], kind='stable')
        if len(ranked):
            groups = ranked.groupby('datetime', sort=False)
            ordinal = groups.cumcount() + 1
            size = groups.position.transform('size')
            # 固定q的二进制表示可能使理论整数略大，先取12位再ceil避免多选一股。
            keep_count = np.ceil(np.round((1 - quantile) * size, 12)).astype(int).clip(lower=1)
            selected_positions = ranked.loc[ordinal.le(keep_count), 'position'].to_numpy()
            frame.iloc[selected_positions, frame.columns.get_loc('rank_ok')] = True
            frame.iloc[ranked.position.to_numpy(), frame.columns.get_loc('rank_ordinal')] = ordinal.to_numpy()
            counts = groups.size()
            frame['eligible_opportunity_count'] = frame.datetime.map(counts).fillna(0).astype(int)
    frame['after_rank'] = frame.pre_rank_eligible & frame.rank_ok
    frame['signal_eligible'] = frame.after_rank
    frame['action'] = np.where(frame.signal_eligible, actions, 0).astype(int)
    frame['request_eligible'] = frame.signal_eligible
    if mode == 'reference' and len(frame):
        positions = np.flatnonzero(frame.signal_eligible.to_numpy())
        requests = frame.iloc[positions][['datetime', 'symbol']].copy()
        requests['position'] = positions
        requests['date'] = requests.datetime.dt.normalize()
        positions = requests.sort_values(['datetime', 'symbol'], kind='stable').drop_duplicates(['date', 'symbol']).position.to_numpy()
        frame['request_eligible'] = False
        frame.iloc[positions, frame.columns.get_loc('request_eligible')] = True
    frame.attrs['signal_mode'] = mode
    frame.attrs['prediction_cost_coverage_required'] = mode != 'vwap_rank'
    frame.attrs['retry_contract'] = '参考规则每日首次请求去重；新规则保留后续机会，原引擎每股每日仅一个成功首腿'
    return frame


def summarize_frequency_funnel(funnel: pd.DataFrame) -> dict:
    """统计每个累计门控后的行、股票日、日期和股票数，不把潜在请求当作成交。"""
    def count(part):
        dates = pd.to_datetime(part.datetime).dt.normalize()
        return {'rows':len(part), 'symbol_days':int(part.assign(date=dates)[['date', 'symbol']].drop_duplicates().shape[0]),
                'days':int(dates.nunique()), 'symbols':int(part.symbol.nunique())}
    return {'input':count(funnel), 'stages':{stage:count(funnel.loc[funnel[stage]]) for stage in FUNNEL_STAGES},
            'prediction_cost_coverage_required':funnel.attrs.get('prediction_cost_coverage_required'),
            'signal_mode':funnel.attrs.get('signal_mode'), 'scope':'当前信号/请求漏斗；不表示实际成交或未来可恢复'}


class FrequencyTStrategy(BaseStrategy):
    """保留周度底仓，依据当前绝对净边际或VWAP机会排名发出不做空的库存T请求。"""
    name = 'weekly_frequency_t'

    def __init__(self, rule: dict, cfg: dict, support: dict):
        """保存副本，仅使用当前特征和核心训练支持域。"""
        self.rule, self.cfg, self.support = deepcopy(rule), deepcopy(cfg), deepcopy(support)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """输出旧引擎信号格式及原索引；新模式保留后来满足条件的请求机会。"""
        if self.rule.get('signal_mode') == 'reference':
            return ShortHorizonTStrategy(self.rule, self.cfg, self.support).generate_signals(data)
        funnel = build_frequency_signal_funnel(data, self.rule, self.cfg, self.support)
        result = pd.DataFrame(index=data.index)
        result['target_position'] = 1.0
        result['t_action'] = funnel.action
        result['fraction'] = funnel.fraction
        result['planned_recovery_time'] = funnel.planned_recovery_time
        result['priority'] = funnel.priority
        return result


def create_frequency_decisions(predictions: pd.DataFrame, rule: dict, cfg: dict, support: dict) -> pd.DataFrame:
    """参考模式精确复用旧去重；新模式把全部当前合格时刻按旧字段格式传给原引擎。"""
    if rule.get('signal_mode') == 'reference':
        return create_short_decisions(predictions, rule, cfg, support)
    funnel = build_frequency_signal_funnel(predictions, rule, cfg, support)
    decisions = funnel.loc[funnel.request_eligible, DECISION_COLUMNS[:-1]].copy()
    decisions = decisions.sort_values(['datetime', 'symbol'], kind='stable')
    decisions['date'] = decisions.datetime.dt.normalize()
    return decisions.reset_index(drop=True)


def select_frequency_candidate(rows: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """先保留真实交易且执行合格者，优先接近每日3组目标，达标后按内层净利排序。

不要求净利润为正；统计/CI不参与开关。频率仅计完整pair/eligible_portfolio_days，
达到目标后封顶；并列再按candidate_index升序。没有合格候选显式报错。
"""
    if 'eligible_portfolio_days' not in rows:
        raise ValueError('频率选择缺少eligible_portfolio_days')
    days = validate_selection_values(rows, 'eligible_portfolio_days', count=True)
    if days.le(0).any():
        raise ValueError('eligible_portfolio_days必须为正整数，不能以零分母制造频率')
    target = validate_finite_number(cfg.get('frequency_target_pairs_per_day', 3.), 'frequency_target_pairs_per_day', lower=0.)
    if target == 0:
        raise ValueError('frequency_target_pairs_per_day必须为正')
    _, annotated = select_active_candidate(rows)
    annotated['pairs_per_eligible_portfolio_day'] = annotated.t_pair_count.to_numpy() / days.to_numpy()
    annotated['frequency_score'] = np.minimum(annotated.pairs_per_eligible_portfolio_day, target)
    annotated['frequency_target_pairs_per_day'] = target
    valid = annotated.loc[annotated.active_execution_eligible]
    chosen = valid.sort_values(['frequency_score', 'incremental_net_pnl', 'candidate_index'],
                              ascending=[False, False, True], kind='stable').iloc[0]
    annotated['selected'] = annotated.candidate_index.eq(chosen.candidate_index)
    return annotated.loc[annotated.selected].iloc[0].to_dict(), annotated
