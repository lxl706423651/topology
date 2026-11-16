"""只读 v14 已完成内部折的入场预测误差；不读取外层、不拟合或修改规则。

--stage 分析已落齐37内部指标及selected_rule的折；--final还要求完整主运行终态。
所有原预测行保留端点缺失计数，固定端点净价差只是标称反事实，不是可成交收益。
动态退出只统计原已选主动账户；hold不制造已选交易。每折派生表独立缓存。
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.audit_inventory_learning_t import decisions, recovery_times, rules, exact

SESSION = list(range(571, 691))+list(range(781, 901))
KEY = ['datetime', 'symbol']
FAIL = ['inventory_mismatch_days', 'recovery_failure_days', 'pending_pairs', 't1_violation_count', 'base_plan_mismatch_count']
TABLES = ['prediction_quality', 'argmax_premium', 'selected_pairs', 'selected_pair_quality', 'candidate_overview',
          'margin_candidate_summary']
BINS = [-np.inf, 0., 10., 20., 40., 60., 100., np.inf]
BIN_NAMES = ['negative', '0_10', '10_20', '20_40', '40_60', '60_100', '100_plus']


def sha(path):
    """记录实际源字节，不以文件名或stat代替内容身份。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    """JSON缺失为null；不把未知价差或不完整组补成零收益。"""
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    return value


def edges(returns, dates, cfg):
    """按每元首腿未滑点金额计算两方向名义净差，费用/印花税按实际日期。"""
    assert cfg['min_commission'] == 0
    fee = cfg['commission_rate']+cfg['transfer_rate']
    slip = cfg['slippage_rate']
    stamp = pd.Series(np.where(dates < pd.Timestamp('2023-08-28'), cfg['stamp_rate_before_20230828'],
                              cfg['stamp_rate_from_20230828']), index=returns.index)
    ratio = 1+returns
    return (ratio*(1-slip)*(1-fee-stamp)-(1+slip)*(1+fee),
            (1-slip)*(1-fee-stamp)-ratio*(1+slip)*(1+fee))


def summary(frame):
    """每个样本等权；预测均值另给共同端点子集，防止缺失造成分母错配。"""
    valid = frame.loc[frame.endpoint_available]
    values = {'rows': len(frame), 'independent_days': frame.datetime.dt.normalize().nunique(),
        'independent_weeks': frame.datetime.dt.to_period('W-SUN').nunique(), 'symbols': frame.symbol.nunique(),
        'known_old_inventory_rows': int(frame.opening_old_shares.gt(0).sum()),
        'invalid_clock_rows': int(frame.endpoint_clock_valid.eq(False).sum()),
        'missing_execution_rows': int(frame.missing_execution.sum()), 'missing_recovery_rows': int(frame.missing_recovery.sum()),
        'valid_endpoint_rows': len(valid), 'missing_endpoint_rows': len(frame)-len(valid),
        'mean_abs_raw_prediction_bp_all': frame.prediction.abs().mean()*1e4,
        'mean_predicted_gross_bp_all': frame.predicted_gross_bp.mean(),
        'mean_predicted_net_bp_all': frame.predicted_net_bp.mean(),
        'mean_predicted_net_bp_common': valid.predicted_net_bp.mean(),
        'mean_realized_reference_gross_bp': valid.fixed_gross_bp.mean(),
        'mean_fixed_nominal_net_bp': valid.fixed_net_bp.mean(),
        'mean_fixed_cost_bp': (valid.fixed_gross_bp-valid.fixed_net_bp).mean(),
        'mean_net_optimism_bp': (valid.predicted_net_bp-valid.fixed_net_bp).mean(),
        'median_fixed_net_bp': valid.fixed_net_bp.median(),
        'raw_direction_success_rate': valid.fixed_gross_bp.gt(0).mean() if len(valid) else np.nan,
        'fixed_cost_covered_rate': valid.fixed_net_bp.gt(0).mean() if len(valid) else np.nan,
        'positive_predicted_net_rows': int(frame.predicted_net_bp.gt(0).sum()),
        'actual_account_execution_not_implied': True}
    return clean(values)


class Diagnosis:
    """文件白名单只允许本折内部数据、全局配置及对应内部年原CSV。"""
    def __init__(self, report, year):
        self.report, self.year = report.resolve(), year
        self.folder = self.report/f'year_{year}'
        self.inputs, self.raw, self.tables = {}, {}, {k: [] for k in TABLES}
        self.cfg = self.json(self.report/'resolved_config.json')
        self.fold = self.json(self.folder/'split.json')
        self.inner = self.fold['inner_year']
        self.chosen = self.json(self.folder/'selected_rule.json')
        self.metrics = self.csv(self.folder/'validation_rule_metrics.csv')
        assert len(self.metrics) == 37 and set(self.metrics.candidate_index) == set(range(37))
        assert self.metrics.selected.sum() == 1
        assert int(self.metrics.loc[self.metrics.selected, 'candidate_index'].iloc[0]) == self.chosen['rule']['candidate_index']

    def record(self, path):
        path = Path(path).resolve()
        if path.is_relative_to(self.report):
            rel = path.relative_to(self.report)
            permitted = (len(rel.parts) == 1 and rel.name in ['resolved_config.json', 'research_protocol.json', 'source_hashes.json'])
            if rel.parts[0] == f'year_{self.year}':
                permitted |= (len(rel.parts) == 2 and rel.name in ['split.json', 'selected_rule.json', 'validation_rule_metrics.csv'])
                permitted |= len(rel.parts) > 2 and rel.parts[1] in ['validation_predictions', 'validation_candidates', 'validation_context', 'models']
            assert permitted, f'禁止读取外层或其它折文件:{rel}'
        else:
            assert hasattr(self, 'inner') and path.is_relative_to(Path(self.cfg['data_root'])/str(self.inner)), f'原CSV不属于内部年:{path}'
        self.inputs.setdefault(str(path), sha(path))
        return path

    def json(self, path): return json.loads(self.record(path).read_text())

    def csv(self, path):
        try: return pd.read_csv(self.record(path), float_precision='round_trip')
        except pd.errors.EmptyDataError: return pd.DataFrame()

    def closes(self, symbol):
        """按原CSV默认数值解析取精确会话close；不前填/后填，无量close保留但不声称可成交。"""
        if symbol not in self.raw:
            path = Path(self.cfg['data_root'])/str(self.inner)/f'{symbol}.csv'
            frame = pd.read_csv(self.record(path), usecols=['datetime', 'close'])
            frame['datetime'] = pd.to_datetime(frame.datetime, format='mixed')
            minute = frame.datetime.dt.hour*60+frame.datetime.dt.minute
            frame = frame.loc[frame.datetime.dt.year.eq(self.inner) & minute.isin(SESSION)].copy()
            assert not frame.datetime.duplicated().any()
            assert (frame.datetime.to_numpy().astype('datetime64[ns]').astype('int64') % 60_000_000_000 == 0).all()
            frame['close'] = pd.to_numeric(frame.close, errors='raise')
            self.raw[symbol] = frame.set_index('datetime').close.sort_index()
        return self.raw[symbol]

    def annotate(self, prediction, horizon):
        """所有当前预测左联固定端点；未来缺失只影响诊断统计可用性，不改变原机会。"""
        p = prediction.copy()
        p['datetime'] = pd.to_datetime(p.datetime, format='mixed')
        assert p.datetime.dt.year.eq(self.inner).all() and not p.duplicated(KEY).any()
        p['horizon_minutes'] = horizon
        p['fixed_start'] = p.datetime+pd.Timedelta(minutes=1)
        p['fixed_end'] = recovery_times(p.datetime, horizon)
        p['endpoint_clock_valid'] = p.fixed_end.notna()
        p['fixed_execution_price'], p['fixed_recovery_price'] = np.nan, np.nan
        for symbol, part in p.groupby('symbol', sort=False):
            close = self.closes(symbol)
            p.loc[part.index, 'fixed_execution_price'] = close.reindex(pd.DatetimeIndex(part.fixed_start)).to_numpy()
            p.loc[part.index, 'fixed_recovery_price'] = close.reindex(pd.DatetimeIndex(part.fixed_end)).to_numpy()
        p['missing_execution'] = ~(np.isfinite(p.fixed_execution_price) & p.fixed_execution_price.gt(0))
        p['missing_recovery'] = p.endpoint_clock_valid & ~(np.isfinite(p.fixed_recovery_price) & p.fixed_recovery_price.gt(0))
        p['endpoint_available'] = p.endpoint_clock_valid & ~p.missing_execution & ~p.missing_recovery
        p['fixed_raw_return'] = (p.fixed_recovery_price/p.fixed_execution_price-1).where(p.endpoint_available)
        buy, sell = edges(p.prediction, p.datetime, self.cfg)
        action = pd.Series(np.where(buy >= sell, 1, -1), index=p.index)
        realized_buy, realized_sell = edges(p.fixed_raw_return, p.datetime, self.cfg)
        p['action'], p['direction'] = action, np.where(action > 0, 'buy_first', 'sell_first')
        p['predicted_net_bp'] = np.maximum(buy, sell)*1e4
        p['predicted_gross_bp'] = action*p.prediction*1e4
        p['fixed_gross_bp'] = action*p.fixed_raw_return*1e4
        p['fixed_net_bp'] = np.where(action > 0, realized_buy, realized_sell)*1e4
        day_keys = pd.MultiIndex.from_arrays([p.datetime.dt.normalize(), p.symbol])
        p['opening_old_shares'] = self.inventory.reindex(day_keys).fillna(0).to_numpy()
        p['signal_clock'] = p.datetime.dt.strftime('%H:%M')
        p['predicted_net_bin'] = pd.cut(p.predicted_net_bp, BINS, labels=BIN_NAMES, right=False).astype(str)
        return p

    def record_quality(self, frame, profile, scope):
        """固定按期限/方向/信号时钟/事前数值箱报告，不搜索最佳阈值。"""
        for domain, subset in [('all_current_keys', frame), ('baseline_opening_old_inventory', frame.loc[frame.opening_old_shares.gt(0)])]:
            groupings = [('all', []), ('horizon_direction', ['horizon_minutes', 'direction']),
                         ('horizon_direction_clock', ['horizon_minutes', 'direction', 'signal_clock']),
                         ('predicted_net_bin', ['predicted_net_bin'])]
            for grouping, columns in groupings:
                groups = [((), subset)] if not columns else subset.groupby(columns, sort=True, dropna=False, observed=True)
                for key, part in groups:
                    key = key if isinstance(key, tuple) else (key,)
                    self.tables['prediction_quality'].append(dict(fold=self.year, inner_year=self.inner, profile=profile,
                        scope=scope, domain=domain, grouping=grouping, **dict(zip(columns, key)), **summary(part)))

    def actual_pairs(self, annotated):
        """把原选中真实首腿回连已知信号；动态实现与固定端点标称目标严格分列。"""
        rule = self.chosen['rule']
        if not rule['enabled']:
            return {'selected_hold': True, 'started_groups': 0, 'complete_groups': 0,
                    'reason': '原内部选择hold；不制造已选交易，全部active概览另列'}
        directory = self.folder/f"validation_candidates/candidate_{rule['candidate_index']:03d}"
        events, trips = self.csv(directory/'test_trade_events.csv'), self.csv(directory/'test_round_trips.csv')
        events['datetime'] = pd.to_datetime(events.datetime, format='mixed')
        events['signal_datetime'] = pd.to_datetime(events.signal_datetime, format='mixed')
        first = events.loc[events.event_type.eq('t') & events.fragment_id.eq(0)].copy()
        first = first.rename(columns={'datetime': 'first_datetime', 'signal_datetime': 'datetime'})
        requested = self.csv(directory/'t_decisions.csv')
        requested['datetime'] = pd.to_datetime(requested.datetime, format='mixed')
        keys = requested[KEY+['horizon_minutes']]
        p = first.merge(keys, on=KEY, how='left', validate='many_to_one')
        assert p.horizon_minutes.notna().all()
        source = pd.concat(list(annotated[rule['profile']].values()), ignore_index=True)
        fields = ['prediction', 'predicted_net_bp', 'predicted_gross_bp', 'fixed_net_bp', 'fixed_gross_bp',
                  'fixed_start', 'fixed_end', 'fixed_execution_price', 'endpoint_available', 'direction', 'signal_clock']
        p = p.merge(source[KEY+['horizon_minutes']+fields], on=KEY+['horizon_minutes'], how='left', validate='many_to_one')
        assert p.first_datetime.eq(p.fixed_start).all(), '首腿不是预测下一分钟，需另行解释而不能错配固定目标'
        assert np.allclose(p.reference_price, p.fixed_execution_price, rtol=0, atol=1e-10)
        assert p.side.eq(np.where(p.direction.eq('buy_first'), 'buy', 'sell')).all()
        if len(trips):
            trips['second_datetime'] = pd.to_datetime(trips.second_datetime, format='mixed')
            trips['allocated_fees'] = trips.buy_fee+trips.sell_fee
            grouped = trips.groupby('pair_id').agg(recovered_shares=('shares', 'sum'), fragments=('shares', 'size'),
                fragment_net=('net_pnl', 'sum'), slipped_gross=('gross_pnl', 'sum'), slippage=('slippage_cost', 'sum'),
                allocated_fees=('allocated_fees', 'sum'), final_recovery=('second_datetime', 'max'))
        else:
            grouped = pd.DataFrame(columns=['recovered_shares', 'fragments', 'fragment_net', 'slipped_gross', 'slippage', 'allocated_fees', 'final_recovery'])
        p = p.merge(grouped, left_on='pair_id', right_index=True, how='left', validate='one_to_one')
        for c in ['recovered_shares', 'fragments', 'fragment_net', 'slipped_gross', 'slippage', 'allocated_fees']:
            p[c] = p[c].fillna(0)
        p['complete'] = p.recovered_shares.eq(p.shares)
        p['first_reference_notional'] = p.shares*p.reference_price
        p['predicted_net_currency'] = p.predicted_net_bp/1e4*p.first_reference_notional
        p['fixed_nominal_net_currency'] = p.fixed_net_bp/1e4*p.first_reference_notional
        p['dynamic_complete_net_currency'] = p.fragment_net.where(p.complete)
        p['dynamic_complete_net_bp'] = p.dynamic_complete_net_currency/p.first_reference_notional*1e4
        p['dynamic_reference_gross_currency'] = (p.slipped_gross+p.slippage).where(p.complete)
        p['dynamic_reference_gross_bp'] = p.dynamic_reference_gross_currency/p.first_reference_notional*1e4
        p['all_actual_pair_fees'] = p.pair_id.map(events.loc[events.event_type.eq('t')].groupby('pair_id').fee.sum())
        p['fold'], p['inner_year'], p['selected_candidate_index'] = self.year, self.inner, rule['candidate_index']
        self.tables['selected_pairs'] = p.to_dict('records')
        metadata = self.json(directory/'metrics.json')
        assert int(p.complete.sum()) == metadata['t_pair_count']
        assert abs(p.loc[p.complete, 'fragment_net'].sum()-metadata['realized_complete_pair_pnl']) < 1e-5
        for grouping, columns in [('all', []), ('horizon_direction', ['horizon_minutes', 'direction']),
                                  ('horizon_direction_clock', ['horizon_minutes', 'direction', 'signal_clock'])]:
            for key, group in ([((), p)] if not columns else p.groupby(columns, dropna=False)):
                key = key if isinstance(key, tuple) else (key,)
                done = group.loc[group.complete]
                common = done.loc[done.endpoint_available]
                norm = done.first_reference_notional.sum()
                self.tables['selected_pair_quality'].append(dict(fold=self.year, inner_year=self.inner, grouping=grouping,
                    **dict(zip(columns, key)), started_groups=len(group), complete_groups=len(done),
                    incomplete_groups=len(group)-len(done), missing_fixed_endpoint_groups=int((~group.endpoint_available).sum()),
                    mean_predicted_net_bp_all=group.predicted_net_bp.mean(), mean_fixed_net_bp_available=group.fixed_net_bp.mean(),
                    same_complete_available_groups=len(common), common_predicted_net_bp=common.predicted_net_bp.mean(),
                    common_fixed_net_bp=common.fixed_net_bp.mean(), common_dynamic_net_bp=common.dynamic_complete_net_bp.mean(),
                    complete_reference_notional=norm, complete_dynamic_net_currency=done.fragment_net.sum(),
                    all_realized_fragment_net_currency=group.fragment_net.sum(), all_actual_fees=group.all_actual_pair_fees.sum(),
                    complete_reference_gross_bp_weighted=done.dynamic_reference_gross_currency.sum()/norm*1e4 if norm else np.nan,
                    complete_dynamic_net_bp_weighted=done.fragment_net.sum()/norm*1e4 if norm else np.nan,
                    common_fixed_nominal_currency=common.fixed_nominal_net_currency.sum(),
                    common_dynamic_currency=common.fragment_net.sum(),
                    dynamic_minus_fixed_common_currency=(common.fragment_net-common.fixed_nominal_net_currency).sum()))
        return dict(selected_hold=False, started_groups=len(p), complete_groups=int(p.complete.sum()),
            incomplete_groups=int((~p.complete).sum()), actual_marked_increment=metadata['incremental_net_pnl'],
            actual_complete_net=metadata['realized_complete_pair_pnl'], **{k: metadata[k] for k in FAIL})

    def run(self):
        """先全部模型行，再既定argmax及原已选实际组；不根据端点成绩改任何规则。"""
        inventory = self.csv(self.folder/'validation_context/baseline/daily_inventory_audit.csv')
        inventory['date'] = pd.to_datetime(inventory.date, format='mixed')
        assert not inventory.duplicated(['date', 'symbol']).any()
        self.inventory = inventory.set_index(['date', 'symbol']).baseline_opening_shares
        raw, annotated = {}, {}
        for profile in ['l2_depth4', 'l2_depth6']:
            raw[profile], annotated[profile] = {}, {}
            for horizon in [20, 40, 60]:
                directory = self.folder/f'validation_predictions/h{horizon}_{profile}'
                p = self.csv(directory/'current_predictions.csv')
                p['datetime'] = pd.to_datetime(p.datetime, format='mixed')
                proof = self.json(directory/'prediction_audit.json')
                model = self.json(self.folder/f'models/h{horizon}_{profile}/model_metadata.json')
                assert sha(directory/'current_predictions.csv') == proof['predictions_sha256']
                assert proof['model_bundle_sha256'] == model['model_bundle_sha256'] and proof['future_independent']
                assert model['train_years'] == self.fold['core_years'] and model['excluded_years'] == sorted([self.inner, self.year])
                raw[profile][horizon] = p
                enriched = self.annotate(p, horizon)
                annotated[profile][horizon] = enriched
            all_h = pd.concat(list(annotated[profile].values()), ignore_index=True)
            self.record_quality(all_h, profile, 'per_horizon_best_direction_all_predictions')
            base_rule = rules()[1 if profile == 'l2_depth4' else 19]
            argmax = decisions(raw[profile], base_rule, self.cfg)
            saved = self.csv(self.folder/f"validation_candidates/candidate_{base_rule['candidate_index']:03d}/t_decisions.csv")
            for c in ['datetime', 'planned_recovery_time']:
                saved[c] = pd.to_datetime(saved[c], format='mixed')
            if len(argmax): exact(saved[list(argmax)], argmax, '当前argmax复现原margin0候选')
            else: assert saved.empty
            joined = argmax[KEY+['horizon_minutes']].merge(all_h, on=KEY+['horizon_minutes'], how='left', validate='one_to_one')
            self.record_quality(joined, profile, 'argmax_current_margin0')
            for margin_bp, offset in [(10, 6), (30, 12)]:
                # 只描述原网格已有门槛；max(edge)>=margin与先过滤再argmax等价。
                margin_rule = rules()[base_rule['candidate_index']+offset]
                expected = decisions(raw[profile], margin_rule, self.cfg)
                source_decisions = self.csv(self.folder/f"validation_candidates/candidate_{margin_rule['candidate_index']:03d}/t_decisions.csv")
                for c in ['datetime', 'planned_recovery_time']:
                    source_decisions[c] = pd.to_datetime(source_decisions[c], format='mixed')
                if len(expected): exact(source_decisions[list(expected)], expected, '原网格门槛决策复现')
                else: assert source_decisions.empty
                subset = joined.loc[joined.predicted_net_bp.ge(margin_bp)].copy()
                exact(subset[KEY+['horizon_minutes']].sort_values(KEY), expected[KEY+['horizon_minutes']].sort_values(KEY), '门槛机会键')
                self.record_quality(subset, profile, f'argmax_current_margin{margin_bp}')
            # 单期限独立门控，其他期限置极低但有限预测并不能安全表示不参与；改为
            # 利用已知候选函数三份同期限时钟之外的纯当前逻辑，复算每期限资格。
            alternatives = []
            for h, part in annotated[profile].items():
                p = part.copy()
                day = p.datetime.dt.normalize()
                current = p.predicted_net_bp.gt(0) & p.fixed_end.notna()
                current &= p.fixed_end.le(day+pd.to_timedelta(self.cfg['fixed_recovery_time']+':00'))
                current &= p.datetime.le(day+pd.to_timedelta(self.cfg['last_entry_time']+':00'))
                current &= np.isfinite(p.signal_amount) & p.signal_amount.gt(0) & np.isfinite(p.signal_volume) & p.signal_volume.gt(0)
                price, prior = p.signal_price.astype(float), p.previous_close.astype(float)
                valid = np.isfinite(price) & price.gt(0) & np.isfinite(prior) & prior.gt(0)
                valid &= np.isfinite(p.signal_high) & np.isfinite(p.signal_low) & p.signal_high.ge(price) & p.signal_low.le(price)
                reverse = np.where(p.action > 0, -1., 1.)
                limit = np.floor(prior*(1+reverse*self.cfg['flat_st_limit_rate'])*100+.5+1e-9)/100
                locked = ((price/prior-1)*reverse >= self.cfg['nonflat_limit_proxy']) | (p.signal_high.eq(p.signal_low) & (price-limit).abs().le(1e-6))
                if self.cfg['t_avoid_locked_exit_side']: current &= valid & ~locked
                alternatives.append(p.loc[current])
            eligible = pd.concat(alternatives, ignore_index=True)
            self.record_quality(eligible, profile, 'legal_current_horizon_alternatives_margin0')
            group = eligible.groupby(KEY, sort=False).agg(alternatives=('horizon_minutes', 'size'),
                all_endpoints_available=('endpoint_available', 'all'),
                alternatives_mean_predicted_net_bp=('predicted_net_bp', 'mean'),
                alternatives_mean_fixed_net_bp=('fixed_net_bp', 'mean'))
            premium = joined.merge(group, on=KEY, how='left', validate='one_to_one')
            assert premium.alternatives.notna().all()
            assert len(group) == len(joined)
            for domain, part in [('all_current_keys', premium), ('baseline_opening_old_inventory', premium.loc[premium.opening_old_shares.gt(0)])]:
                comparable = part.loc[part.all_endpoints_available]
                self.tables['argmax_premium'].append(dict(fold=self.year, inner_year=self.inner, profile=profile, domain=domain,
                    argmax_rows=len(part), rows_with_missing_any_alternative_endpoint=int((~part.all_endpoints_available).sum()),
                    same_key_all_alternatives_available_rows=len(comparable),
                    mean_number_of_current_alternatives=comparable.alternatives.mean(),
                    mean_argmax_predicted_net_bp=comparable.predicted_net_bp.mean(),
                    mean_alternative_predicted_net_bp=comparable.alternatives_mean_predicted_net_bp.mean(),
                    mean_argmax_fixed_net_bp=comparable.fixed_net_bp.mean(),
                    mean_alternative_fixed_net_bp=comparable.alternatives_mean_fixed_net_bp.mean(),
                    forecast_selection_premium_bp=(comparable.predicted_net_bp-comparable.alternatives_mean_predicted_net_bp).mean(),
                    realized_selection_premium_bp=(comparable.fixed_net_bp-comparable.alternatives_mean_fixed_net_bp).mean(),
                    no_future_filter_in_actual_argmax=True))
        fields = ['candidate_index', 'enabled', 'profile', 'min_net_edge', 'fraction', 'exit_mode', 'request_count',
            'selected', 'execution_eligible', 't_pair_count', 'fragment_count', 'completed_t_pair_lots',
            'realized_fragment_reference_gross_pnl', 't_fee', 't_slippage_cost', 'incremental_net_pnl',
            'realized_complete_pair_pnl', 'all_reference_acquisition_cost', 'holding_cost_reduction_rate', *FAIL]
        self.tables['candidate_overview'] = self.metrics[fields].assign(fold=self.year, inner_year=self.inner).to_dict('records')
        for (profile, margin), part in self.metrics.loc[self.metrics.enabled].groupby(['profile', 'min_net_edge']):
            complete = part.loc[part.pending_pairs.eq(0)]
            drag = complete.t_fee+complete.t_slippage_cost
            comparable = drag.gt(0)
            coverage = complete.realized_fragment_reference_gross_pnl.loc[comparable]/drag.loc[comparable]
            self.tables['margin_candidate_summary'].append(dict(fold=self.year, inner_year=self.inner,
                profile=profile, margin_bp=margin*1e4, overlapping_accounts=len(part),
                execution_eligible_accounts=int(part.execution_eligible.sum()),
                positive_marked_net_accounts=int(part.incremental_net_pnl.gt(0).sum()),
                completed_groups_min=int(part.t_pair_count.min()), completed_groups_median=part.t_pair_count.median(),
                completed_groups_max=int(part.t_pair_count.max()), paired_lots_median=part.completed_t_pair_lots.median(),
                marked_increment_median=part.incremental_net_pnl.median(),
                no_pending_cost_comparable_accounts=int(comparable.sum()),
                reference_gross_to_cost_median_no_pending=coverage.median(),
                reference_gross_covers_cost_accounts_no_pending=int(coverage.gt(1).sum()),
                median_actual_fee=part.t_fee.median(), median_actual_slippage=part.t_slippage_cost.median(),
                all_amounts_are_distribution_statistics_not_summed_portfolio=True))
        actual = self.actual_pairs(annotated)
        assert all(sha(p) == h for p, h in self.inputs.items())
        return dict(fold=self.year, inner_year=self.inner, selected_rule=self.chosen['rule'],
            actual_selected=actual, inputs_sha256=self.inputs, no_outer_read=True, no_new_parameters=True,
            fixed_endpoint_is_nominal_not_account_execution=True, existing_v14_sources_unchanged=True)


def write_markdown(out, results):
    """解释总体误差与实际交易；不将固定端点当真实可成交利润。"""
    q = pd.read_csv(out/'prediction_quality.csv')
    chosen = q.loc[q.grouping.eq('all') & q.scope.eq('argmax_current_margin0') & q.domain.eq('baseline_opening_old_inventory')]
    lines = ['# 永久底仓模型内部入场预测诊断', '',
        '仅已完成内部验证折，不读取外层，不训练/校准，也不更改原选择。所有数值箱仅作固定描述。', '',
        '以下是原margin=0当前argmax机会，进一步标识无T基准当日开盘已有旧股；现金/容量/已耗旧股仍需实际执行器判断。未来缺失端点单列，不据此删除原决策。', '',
        '|留出折|内部年|模型|当前机会|有效端点|预测净差 bp（同有效行）|实际方向毛差 bp|标称固定端点净差 bp|模型乐观偏差 bp|成本覆盖比例|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in chosen.itertuples():
        lines.append(f'|{r.fold}|{r.inner_year}|{r.profile}|{r.rows}|{r.valid_endpoint_rows}|{r.mean_predicted_net_bp_common:.2f}|{r.mean_realized_reference_gross_bp:.2f}|{r.mean_fixed_nominal_net_bp:.2f}|{r.mean_net_optimism_bp:.2f}|{r.fixed_cost_covered_rate:.2%}|')
    lines += ['', '价差按每元首腿原始参考金额归一化，含原双腿佣金、过户费、当日印花税及滑点。平价来回基准成本在2023-08-28后约31.20 bp、此前约36.20 bp；实际净差公式随两端价格变化。缺价、无量、容量与真实回购准备金会进一步约束成交，标称目标不能等同实际收益。', '',
              '|留出折|原已选规则|实际完整/启动组|实际完整净额 元|账户含残仓净增量 元|库存差日/恢复失败日/期末待恢复|',
              '|---|---|---:|---:|---:|---|']
    for r in results:
        a = r['actual_selected']; rule = r['selected_rule']
        if a['selected_hold']:
            lines.append(f"|{r['fold']}|hold|0/0|—|—|原规则未启用；不制造已选交易|")
        else:
            lines.append(f"|{r['fold']}|{rule['candidate_index']} {rule['profile']}/{rule['exit_mode']}|{a['complete_groups']}/{a['started_groups']}|{a['actual_complete_net']:.2f}|{a['actual_marked_increment']:.2f}|{a['inventory_mismatch_days']}/{a['recovery_failure_days']}/{a['pending_pairs']}|")
    lines += ['', 'argmax_premium.csv 另按同一股票/信号时刻对比原合法期限平均值与最大预测选择，缺任一期限端点的时刻单列；这只是条件诊断，不反向选择期限。',
        'selected_pairs.csv / selected_pair_quality.csv 将原已选实际首腿回连固定标签，并把真实动态退出完成净额与固定端点标称反事实分开。全部原始首腿保留，未完整组的已实现片段另外计数。hold时仅提供36主动内部账户概览，不暗中选择另一个交易账户。',
        'per_horizon、方向、8个原信号时钟与预测净差箱详见prediction_quality.csv。不同期限/规则含重复机会，不能把各组金额相加当成一只组合。',
        '旧OOF选择器含历史跨年训练依赖，当前诊断属于重复历史研究；不能据此声称全新测试已证明盈利，也没有在两个交互OOF核心年上另做不独立的校准。']
    (out/'diagnosis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    """阶段缓存仅复用同脚本及同输入SHA；终版等待主运行成功并再次核全部身份。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--stage', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--years', type=int, nargs='+')
    args = parser.parse_args()
    assert args.stage != args.final
    report = Path(args.report_dir).resolve()
    cfg_path = report/'resolved_config.json'
    cfg = json.loads(cfg_path.read_text())
    assert cfg['research_version'] == 'inventory_learning_t_v14'
    years = args.years or cfg['test_years']
    assert set(years).issubset(cfg['test_years'])
    available = [y for y in years if all((report/f'year_{y}'/f).exists() for f in ['validation_rule_metrics.csv', 'selected_rule.json'])]
    if args.final:
        assert available == cfg['test_years'] and json.loads((report/'runtime_integrity_audit.json').read_text())['all_unchanged']
    script_sha = sha(__file__)
    dependency = ROOT/'scripts/audit_inventory_learning_t.py'
    dependency_sha = sha(dependency)
    stages = []
    for year in available:
        cached = sorted((OUT/f'fold_{year}').glob('run_*/analysis.json'), reverse=True)
        prior = None
        for p in cached:
            value = json.loads(p.read_text())
            if (value['source_report'] == str(report) and value['script_sha256'] == script_sha
                    and value['dependency_sha256'] == dependency_sha and all(sha(k) == v for k, v in value['inputs_sha256'].items())):
                assert all(sha(p.parent/k) == v for k, v in value['output_sha256'].items())
                prior = p.parent
                break
        if prior is not None:
            stages.append(prior)
            continue
        run = OUT/f'fold_{year}'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
        run.mkdir(parents=True, exist_ok=False)
        shutil.copy2(__file__, run/'diagnose.py')
        diagnostic = Diagnosis(report, year)
        try:
            result = diagnostic.run()
            for name, rows in diagnostic.tables.items():
                pd.DataFrame(rows).to_csv(run/f'{name}.csv', index=False, encoding='utf-8-sig')
            write_markdown(run, [result])
            result.update(passed=True, source_report=str(report), script_sha256=script_sha, dependency_sha256=dependency_sha,
                          output_sha256={p.name: sha(p) for p in run.iterdir() if p.is_file()})
            (run/'analysis.json').write_text(json.dumps(clean(result), ensure_ascii=False, indent=2))
        except Exception as error:
            (run/'failure.json').write_text(json.dumps(clean(dict(error=str(error), inputs_sha256=diagnostic.inputs)), ensure_ascii=False, indent=2))
            raise
        stages.append(run)
        print(f'fold={year} 内部预测/argmax/原选中首腿诊断完成：{run}', flush=True)
    if not args.final:
        print(json.dumps({'completed_folds': available, 'final': False, 'stage_directories': list(map(str, stages))}, ensure_ascii=False))
        return
    final = OUT/f'final_{datetime.now():%Y%m%d_%H%M%S_%f}'
    final.mkdir(exist_ok=False)
    results = [json.loads((p/'analysis.json').read_text()) for p in stages]
    for name in TABLES:
        frames = []
        for p in stages:
            try: frames.append(pd.read_csv(p/f'{name}.csv', float_precision='round_trip'))
            except pd.errors.EmptyDataError: pass
        (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(final/f'{name}.csv', index=False, encoding='utf-8-sig')
    write_markdown(final, results)
    inputs = {k: v for r in results for k, v in r['inputs_sha256'].items()}
    for name in ['resolved_config.json', 'source_hashes.json', 'runtime_integrity_audit.json']:
        inputs[str(report/name)] = sha(report/name)
    for name, expected in json.loads((report/'source_hashes.json').read_text()).items():
        path = report/'source_snapshot'/name
        assert sha(path) == expected
        inputs[str(path)] = expected
    assert all(sha(p) == v for p, v in inputs.items()) and sha(dependency) == dependency_sha
    shutil.copy2(__file__, final/'diagnose.py')
    shutil.copy2(dependency, final/'audit_inventory_learning_t.py')
    output = dict(passed=True, complete=True, configured_folds=cfg['test_years'], no_outer_read=True,
        source_report=str(report), script_sha256=script_sha, dependency_sha256=dependency_sha,
        no_refit_calibration_or_parameter_change=True, sources_verified=48, results=results, inputs_sha256=inputs,
        output_sha256={p.name: sha(p) for p in final.iterdir() if p.is_file()})
    (final/'verification.json').write_text(json.dumps(clean(output), ensure_ascii=False, indent=2))
    print(json.dumps({'passed': True, 'final_directory': str(final), 'inputs': len(inputs)}, ensure_ascii=False))


if __name__ == '__main__': main()
