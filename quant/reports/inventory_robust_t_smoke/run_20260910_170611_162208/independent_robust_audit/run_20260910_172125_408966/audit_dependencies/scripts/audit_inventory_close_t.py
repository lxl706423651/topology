"""独立核验v16固定14:55训练、预测、37规则及原guard真实账户。

仅在成功终态后新增 independent_close_audit/run_*；不改模型、行情或原报告，
不训练或重撮合账户。新目标/策略独立重算；旧来源和账簿使用已审计只读方法。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import joblib
import lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dynamic_inventory_t import (Auditor as AccountAuditor, FAILURE_KEYS,
    close, digest, next_minute, parse_time, require)
from scripts.audit_inventory_guarded_t import (GUARD_COLUMNS, own_fillable, verify_guard_row,
    self_tests as guard_self_tests)
from scripts.audit_inventory_learning_t import (LearningAuditor, DATE_COLUMNS, CONTEXT,
    rules, exact, frame_digest, feature_selection)
from scripts.audit_weekly_t_short_horizon import independent_training_rows, interval_overlap
from scripts.audit_weekly_t_timing_models import independent_timing_features, TIMING_COLUMNS
from scripts.inventory_close_audit_models import self_tests as model_self_tests

VERSION = 'inventory_close_t_v16'
KEYS = ['datetime', 'symbol']
IDENTITY = {'recovery_key': 'close', 'recovery_kind': 'fixed_clock', 'recovery_clock': '14:55'}
CLOCKS = ('09:35', '09:45', '10:00', '10:30', '11:00', '13:05', '13:30', '14:00')
LINEAGE = ['datetime', 'symbol', 'feature_source_start', 'intraday_feature_source_start',
    'entry_feature_source_start', 'entry_rank_source_start', 'portfolio_cash_source_start',
    'base_entry_datetime', 'baseline_opening_shares', 'frozen_base_shares',
    'source_model_training_years', 'source_model_sha256', 'cohort_target_year']


def current_clock(frame):
    """独立检查原8个信号与唯一键，固定终点不从未来label时间取得。"""
    require(not frame.columns.duplicated().any() and set(KEYS).issubset(frame), '当前列/键缺失')
    require(not frame[KEYS].isna().any().any() and not frame.duplicated(KEYS).any(), '当前键缺失/重复')
    times = parse_time(frame.datetime)
    require(times.dt.tz is None and times.dt.strftime('%H:%M').isin(CLOCKS).all()
            and times.dt.second.eq(0).all() and times.dt.microsecond.eq(0).all()
            and times.dt.nanosecond.eq(0).all(), '非原精确当前信号时钟')
    return times, times.dt.normalize()+pd.Timedelta(hours=14, minutes=55)


def attach_close_reference(cohort, minutes):
    """以全部core当前键取原两个参考端点，保留缺价行，不读取旧target遮罩。"""
    current_clock(cohort)
    require(not any(c.startswith(('execution_', 'recovery_', 'target_', 'label_'))
                    or c in ('horizon_minutes', 'trade_week_complete') for c in cohort), 'core提前带未来标签或旧周资格')
    source = minutes[KEYS+['execution_time', 'execution_price', 'recovery_time', 'recovery_price']]
    require(not source.duplicated(KEYS).any(), '原当前端点键重复')
    targets = cohort[KEYS].merge(source, on=KEYS, how='left', sort=False, validate='one_to_one')
    clock, end = current_clock(targets)
    first = clock.map(next_minute)
    require(parse_time(targets.execution_time).eq(first).all()
            and parse_time(targets.recovery_time).eq(end).all(), '原端点不是下一分钟/同日14:55')
    prices = targets[['execution_price', 'recovery_price']].apply(pd.to_numeric, errors='raise')
    require(not np.isinf(prices).any().any() and not (prices.notna() & prices.le(0)).any().any(), '非法参考价')
    result = cohort.copy()
    result['execution_time'], result['recovery_time'] = first.to_numpy(), end.to_numpy()
    result['execution_price'], result['recovery_price'] = prices.execution_price.to_numpy(), prices.recovery_price.to_numpy()
    result['label_start'], result['label_end'] = first.to_numpy(), end.to_numpy()
    result['target_return'] = result.recovery_price/result.execution_price-1
    result['label_available'] = prices.notna().all(axis=1).to_numpy()
    for name, value in IDENTITY.items(): result[name] = value
    require(len(result) == len(cohort), '连接标签删掉当前机会')
    return result


def close_decisions(predictions, rule, cfg):
    """独立双腿现金公式及当前风险门控；同一个close预测仅比较方向。"""
    if not rule['enabled']: return pd.DataFrame()
    p = predictions.copy()
    p['datetime'], deadline = current_clock(p)
    require('horizon_minutes' not in p and p.model_profile.eq(rule['profile']).all(), '预测身份错误')
    require(all(p[c].eq(v).all() for c, v in IDENTITY.items()), '不是固定14:55预测')
    raw = pd.to_numeric(p.prediction)
    require(np.isfinite(raw).all() and raw.gt(-1).all(), '非法参考价格比')
    transaction, slip = cfg['commission_rate']+cfg['transfer_rate'], cfg['slippage_rate']
    stamp = pd.Series(np.where(p.datetime < pd.Timestamp('2023-08-28'), cfg['stamp_rate_before_20230828'],
                               cfg['stamp_rate_from_20230828']), index=p.index)
    buy = (1+raw)*(1-slip)*(1-transaction-stamp)-(1+slip)*(1+transaction)
    sell = (1-slip)*(1-transaction-stamp)-(1+raw)*(1+slip)*(1+transaction)
    action = pd.Series(np.where(buy >= sell, 1, -1), index=p.index)
    edge = pd.Series(np.maximum(buy, sell), index=p.index)
    mask = edge.gt(0) & edge.ge(rule['min_net_edge'])
    if cfg['t_avoid_locked_exit_side']:
        price, previous = p.signal_price.astype(float), p.previous_close.astype(float)
        valid = (np.isfinite(price) & np.isfinite(previous) & price.gt(0) & previous.gt(0)
                 & np.isfinite(p.signal_high) & np.isfinite(p.signal_low)
                 & p.signal_high.ge(price) & p.signal_low.le(price))
        reverse = pd.Series(np.where(action > 0, -1., 1.), index=p.index)
        limit = np.floor(previous*(1+reverse*cfg['flat_st_limit_rate'])*100+.5+1e-9)/100
        mask &= valid & ~(((price/previous-1)*reverse).ge(cfg['nonflat_limit_proxy'])
            | (p.signal_high.eq(p.signal_low) & (price-limit).abs().le(1e-6)))
    mask &= np.isfinite(p.signal_amount) & p.signal_amount.gt(0) & np.isfinite(p.signal_volume) & p.signal_volume.gt(0)
    mask &= p.datetime.le(p.datetime.dt.normalize()+pd.to_timedelta(cfg['last_entry_time']+':00'))
    mask &= deadline.gt(p.datetime.map(next_minute))
    result = p.loc[mask, KEYS].copy()
    result['action'], result['priority'] = action.loc[mask], edge.loc[mask]
    result['planned_recovery_time'], result['prediction'] = deadline.loc[mask], raw.loc[mask]
    result['fraction'], result['candidate_index'] = rule['fraction'], rule['candidate_index']
    result['signal_strength'], result['predicted_net_edge'] = result.priority, result.priority
    result['regime'], result['model_profile'] = 'learned_permanent_fixed_close_inventory', rule['profile']
    for name, value in IDENTITY.items(): result[name] = value
    return result.sort_values(['datetime', 'priority', 'symbol'], ascending=[True, False, True], kind='stable').reset_index(drop=True)


def validate_goal_status(goal, cfg, outer_started):
    """未量化频率或非正式外层不得宣称整项目目标完成；不控制策略开关。"""
    quantified = any(cfg.get(name) is not None for name in ('minimum_completed_pair_lots', 'minimum_active_stock_days'))
    if not quantified:
        require(goal['goal_achieved'] is False, '未量化足够手数却宣称完整目标达到')
        if outer_started: require(goal['frequency_target_status'] == 'not_quantified', '未量化频率状态错')
    if cfg['report_scope'] == 'smoke' or not outer_started:
        require(goal['goal_achieved'] is False, '工程/内部范围不得达标')


class CloseAuditor(AccountAuditor):
    """原账户核是只读父类，初始化、版本、目标与模型审计属于独立v16。"""
    def __init__(self, report, out):
        self.report, self.out = report, out
        self.inputs, self.raw, self.full_raw, self.raw_files, self.results = {}, {}, {}, {}, []
        self.cfg, self.protocol = self.json(report/'resolved_config.json'), self.json(report/'research_protocol.json')
        require(self.cfg['research_version'] == VERSION and self.protocol['version'] == VERSION, '只支持v16身份')
        require(self.cfg['forecast_target'] == 'same_day_1455' and self.cfg['recovery_key'] == 'close'
                and self.cfg['t_recheck_exit_side_at_execution'] is True, '目标或guard合同错误')
        require('horizons' not in self.cfg and 't_label_horizon_minutes' not in self.cfg, '旧整数期限污染新目标')
        require(tuple(self.cfg['signal_times']) == CLOCKS and self.cfg['last_entry_time'] == '14:30'
                and self.cfg['fixed_recovery_time'] == '14:55', '当前信号或deadline改变')
        self.checked_raw_events = self.checked_exit_pairs = 0
        self.cache_audits, self.sparse_arrays, self.mark_cache, self.exit_cache, self.base_anchors = [], {}, {}, {}, {}
        self.valuation_mode = 'full_minute_reference_stream_then_daily_close'
        require(self.protocol['valuation_contract'] == self.valuation_mode, '估值合同错误')
        self.market_calendar = pd.DatetimeIndex([])
        self.execution_source = Path(self.cfg['source_execution_report_dir'])
        self.learning_source = Path(self.cfg['source_learning_report_dir'])
        self.guards, self.model_results, self.prediction_results, self.cohort_results, self.baselines_checked = [], [], [], [], []

    def table(self, path):
        """保存表按原round-trip与全部明确日期列恢复，标签端点也必须精确。"""
        frame = self.csv(path)
        for name in DATE_COLUMNS.intersection(frame): frame[name] = parse_time(frame[name])
        return frame

    def sources(self):
        """分别核v15执行对照与v14完整core证明，不能继承同预测的旧审计假设。"""
        terminal = self.json(self.report/'runtime_integrity_audit.json')
        require(terminal.get('all_unchanged') is True and (self.report/'final_report.md').exists(), '主运行未成功终态')
        self.record(self.report/'final_report.md')
        hashes = self.json(self.report/'source_hashes.json')
        require(len(hashes) == terminal['runtime_sources_verified'] == 57, '原51源+新6源不齐')
        for name, value in hashes.items():
            require(digest(self.record(ROOT/name)) == value and digest(self.record(self.report/'source_snapshot'/name)) == value, '运行源/快照变化')
        registered = self.json(self.report/'input_sources.json')
        for path, identity in registered.items(): require(digest(self.record(path)) == identity['sha256'], f'登记输入变化:{path}')
        self.source_audits = {}
        for source, kind, version in ((self.execution_source, 'guarded', 'inventory_guarded_t_v15'),
                                      (self.learning_source, 'learning', 'inventory_learning_t_v14')):
            passed = []
            for path in sorted(source.glob(f'independent_{kind}_audit/run_*/verification.json')):
                value = self.json(path)
                if value.get('research_version') == version and all(value.get(k) is True for k in ('passed', 'complete', 'all_read_inputs_unchanged')):
                    passed.append((path, value))
            require(passed, f'缺完整{version}来源独审')
            audit_path, audit = passed[-1]
            config = self.json(source/'resolved_config.json')
            require(config['research_version'] == version, '来源报告身份错')
            for name in ('resolved_config.json', 'runtime_integrity_audit.json', 'source_hashes.json',
                         'all_rules_frozen_before_outer_test.json'):
                path = (source/name).resolve()
                require(audit['inputs_sha256'].get(str(path)) == digest(self.record(path)), '来源关键文件偏离完整独审')
            self.source_audits[kind] = (audit_path, audit)
        original = self.json(self.execution_source/'resolved_config.json')
        ignored = {'research_version', 'configuration_source', 'source_configuration_source', 'source_report_scope',
            'output_dir', 'workers', 'candidate_workers', 'test_years', 'internal_only', 'horizons', 'source_learning_version',
            'scheduled_recovery_times', 'frozen_t_models_reused', 't_models_fitted_this_run'}
        for name, value in original.items():
            if name not in ignored: require(self.cfg.get(name) == value, f'非新预测目标的原合同改变:{name}')
        require(set(self.cfg['test_years']).issubset(original['test_years']), '扩张来源评价范围')
        require(Path(self.cfg['source_learning_audit']).resolve() == self.source_audits['learning'][0].resolve()
                and self.cfg['source_learning_audit_sha256'] == digest(self.source_audits['learning'][0]), 'core来源审计身份错')
        require(self.protocol['candidate_rules'] == rules() and self.protocol['models_per_fold'] == 2
                and self.protocol['models_fitted_this_run'] == 2*len(self.cfg['test_years'])
                and self.protocol['frozen_t_models_reused'] is False and self.protocol['outer_model_refit'] is False
                and self.protocol['cross_horizon_argmax'] is False, '新训练/目标/候选协议错误')
        require(all(self.protocol.get(k) == v for k, v in IDENTITY.items()), '协议混入旧期限身份')
        frozen = self.json(self.report/'all_rules_frozen_before_outer_test.json')
        require(frozen['test_years'] == self.cfg['test_years'] and len(frozen['model_artifacts_sha256']) == 16*len(self.cfg['test_years']), '新模型冻结不全')
        self.freeze, self.freeze_time = frozen, (self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns
        for name, expected in frozen['model_artifacts_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, '新模型应相对本报告冻结')
            path = self.report/name
            require(digest(self.record(path)) == expected and path.stat().st_mtime_ns <= self.freeze_time, '模型freeze后变化')
        self.source_summary = dict(runtime_sources=len(hashes), registered_inputs=len(registered),
            newly_fitted_models=2*len(self.cfg['test_years']), new_model_artifacts=len(frozen['model_artifacts_sha256']),
            source_execution_audit=str(self.source_audits['guarded'][0]), source_learning_audit=str(self.source_audits['learning'][0]),
            old_t_weights_used_for_new_prediction=False, source_prediction_equality_required=False)

    def market(self):
        """复用旧已审计量价来源链，十一当前量价特征独立复算，不加载horizon缓存。"""
        from quant_backtest.weekly_t_features import add_weekly_execution_features, add_session_context_features
        from quant_backtest.weekly_t_provenance import add_feature_provenance
        known = self.json(Path(self.cfg['source_audit_report_dir'])/'input_sources.json')
        frames = []
        for name in ('weekly', 'intraday', 'daily'):
            path = (Path(self.cfg['market_cache_dir'])/f'{name}.pkl').resolve()
            require(digest(self.record(path)) == known[str(path)]['sha256'], '原市场缓存SHA不符')
            frames.append(pd.read_pickle(path))
        weekly, minutes, daily = frames
        self.market_calendar = pd.DatetimeIndex(daily.date.unique()).normalize().sort_values()
        weekly = add_weekly_execution_features(weekly, daily)
        minutes = add_session_context_features(minutes)
        weekly, minutes, provenance = add_feature_provenance(weekly, minutes, daily)
        require(provenance == self.json(self.report/'feature_provenance_manifest.json'), '实际来源manifest不同')
        minutes = independent_timing_features(minutes)
        amount = weekly.f_avg_daily_amount if 'f_avg_daily_amount' in weekly else weekly.f_daily_amount_mean
        mask = (weekly.f_history_days.ge(self.cfg['min_history_days']) & weekly.f_last_close.ge(self.cfg['min_price'])
            & amount.ge(self.cfg['min_avg_daily_amount']) & weekly.feature_week_complete.eq(True)
            & weekly.f_prior_close_auction_amount_min.ge(self.cfg['min_prior_close_auction_amount']))
        universe = weekly.loc[mask].copy()
        universe['universe_previous_close'], universe['universe_avg_daily_amount'] = weekly.f_last_close.loc[mask], amount.loc[mask]
        self.market_summary = dict(full_cross_section_rows=len(minutes), symbols=int(minutes.symbol.nunique()),
            independently_rebuilt_timing_features=TIMING_COLUMNS, old_source_provenance_matches=True,
            old_integer_horizon_labels_loaded=False)
        return universe, minutes

    def cohort(self, fold, universe, minutes):
        """以真正v14来源调用旧独立lineage重建，再核新完整机会归档；不调生产planner。"""
        source = LearningAuditor(self.learning_source, self.out)
        source.market_calendar = self.market_calendar
        core = source.cohort(fold, universe, minutes)
        proof = self.source_audits['learning'][1]['inputs_sha256']
        for name, value in source.inputs.items():
            require(proof.get(name) == value, 'core来源读取未绑定原完整独审')
            self.inputs[name] = value
        folder = self.report/f"year_{fold['test_year']}/training_cohort/frozen_inventory_cohort"
        saved = self.table(folder/'cohort_lineage.csv')
        exact(saved, core[LINEAGE], 'v16全部未联标签机会lineage', dtype=True)
        meta = self.json(folder/'reuse_metadata.json')
        require(meta['training_opportunity_rows'] == len(core) and meta['columns'] == list(core)
            and meta['dtypes'] == core.dtypes.astype(str).to_dict()
            and meta['full_current_core_sha256'] == frame_digest(core)
            and meta['cohort_lineage_sha256'] == frame_digest(core[LINEAGE]), '完整core字段/机会/来源SHA错')
        require(meta['labels_attached'] is False and meta['old_training_matrix_used_as_opportunity_filter'] is False
                and meta['planner_or_model_run'] is False, '从旧有效标签反选训练机会')
        require(meta['core_years'] == fold['core_years'] and meta['excluded_years'] == sorted([fold['test_year'], fold['inner_year']]), 'core年份错')
        self.cohort_results.extend(source.cohort_results)
        self.checked_raw_events += source.checked_raw_events
        self.raw_files.update(source.raw_files)
        return core

    def source_baseline(self, year, scope):
        """实际来源有同scope时逐表精确比；无旧外层则独立基准核账并明确范围。"""
        relative = Path(f'year_{year}/{scope}_context')
        names = ['frozen_base_selections.csv', 'frozen_base_entries.csv', 'baseline/test_equity_curve.csv',
                 'baseline/baseline_trade_events.csv', 'planning_baseline/baseline_trade_events.csv']
        present = (self.execution_source/relative).exists()
        if present:
            for name in names: exact(self.csv(self.report/relative/name), self.csv(self.execution_source/relative/name), '同源底仓及每日基准')
        require(present or (scope == 'test' and self.cfg['report_scope'] != 'smoke'),
                '内部/工程外层须有已审计的同源基准可比')
        self.baselines_checked.append(dict(year=year, scope=scope, source_scope_exists=present,
            source_tables_exact=names if present else [], independent_account_replay_required=True))

    def guard_account(self, folder, predictions, rule):
        """独立按全部请求顺序重建guard前现金/旧股/used，取消不产生任何虚拟成交。"""
        audit = self.table(folder/'first_leg_exit_side_audit.csv')
        require(list(audit.columns) == GUARD_COLUMNS, 'guard表字段合同改变')
        require(not audit.duplicated(['datetime', 'symbol']).any(), 'guard请求重复')
        event = self.table(folder/'test_trade_events.csv')
        attempt = self.table(folder/'blocked_orders.csv')
        dec = self.table(folder/'t_decisions.csv')
        forecast = predictions.get(rule['profile'], pd.DataFrame())
        pmap = forecast.set_index(KEYS) if len(forecast) else pd.DataFrame()
        if len(dec):
            dec = dec.copy()
            dec['execution'] = dec.datetime.map(next_minute)
            dec = dec.sort_values(['execution', 'priority', 'symbol'], ascending=[True, False, True], kind='stable')
        days = parse_time(self.csv(folder/'test_equity_curve.csv').datetime).dt.normalize()
        event = event.reset_index(drop=True)
        cash, qty, old, active, cycles, used = float(self.cfg['initial_cash']), defaultdict(int), {}, {}, {}, defaultdict(float)
        observed_guard, observed_first, observed_cancel = set(), set(), set()
        records = []
        lot = self.cfg['lot_size']
        cutoff = pd.Timestamp('2000-01-01 '+self.cfg['last_entry_time']).time()
        def apply_event(row):
            nonlocal cash
            q = int(row.shares)
            if row.side == 'buy': cash -= row.notional+row.fee; qty[row.symbol] += q
            else: cash += row.notional-row.fee; qty[row.symbol] -= q; old[row.symbol] = old.get(row.symbol, 0)-q
            raw_time = row.datetime+pd.Timedelta(minutes=1) if row.datetime.time() == pd.Timestamp('09:30').time() else row.datetime
            used[raw_time, row.symbol] += q*row.reference_price
            if row.event_type == 't':
                if row.fragment_id == 0:
                    require(row.symbol not in active, '重叠首腿')
                    active[row.symbol] = [row.pair_id, q]
                    cycles[row.symbol] = cycles.get(row.symbol, 0)+1
                else:
                    require(row.symbol in active and active[row.symbol][0] == row.pair_id, '无首腿恢复')
                    active[row.symbol][1] -= q
                    if active[row.symbol][1] == 0: del active[row.symbol]
            close(cash, row.cash_after, 'guard独立事件现金')
        for day in days:
            opening, old, cycles = dict(qty), dict(qty), {}
            e = event.loc[event.datetime.dt.normalize().eq(day)]
            requests = dec.loc[dec.execution.dt.normalize().eq(day)] if len(dec) else pd.DataFrame()
            times = sorted(set(e.datetime).union(requests.execution if len(requests) else []))
            for stamp in times:
                at = e.loc[e.datetime.eq(stamp)]
                first = at.loc[at.event_type.eq('t') & at.fragment_id.eq(0)]
                # 真正的恢复和底仓先发生，再依原priority顺序处理首腿请求。
                for _, row in at.loc[~(at.event_type.eq('t') & at.fragment_id.eq(0))].iterrows(): apply_event(row)
                for _, request in (requests.loc[requests.execution.eq(stamp)].iterrows() if len(requests) else []):
                    symbol = request.symbol
                    side = 'buy' if request.action > 0 else 'sell'
                    key = (stamp, symbol)
                    rows = audit.loc[audit.datetime.eq(stamp) & audit.symbol.eq(symbol)]
                    fills = first.loc[first.symbol.eq(symbol)]
                    prior_old = old.get(symbol, 0)
                    requested = min(int(opening.get(symbol, 0)*request.fraction/lot)*lot, prior_old)
                    preblocked = symbol in active or (self.cfg['max_cycles_per_day'] and cycles.get(symbol, 0) >= self.cfg['max_cycles_per_day']) or stamp.time() > cutoff or prior_old < lot or opening.get(symbol, 0) < lot or requested < lot
                    raw = self.full_bars(stamp.year, symbol)
                    if stamp not in raw.index: quote = {k: np.nan for k in ('price','high','low','amount','volume','previous_close','limit_up','limit_down')}
                    else:
                        b = raw.loc[stamp]
                        previous = float(pmap.loc[(request.datetime, symbol), 'previous_close'])
                        if not np.isfinite(previous):
                            past = self.bars(stamp.year, symbol)
                            past = past.loc[(past.index < day) & (past.index.hour == 15) & (past.index.minute == 0)]
                            previous = float(past.close.iloc[-1]) if len(past) else np.nan
                        quote = {'price': float(b.close), **{k: float(b[k]) for k in ('high','low','amount','volume')},
                                 'previous_close': previous, 'limit_up': np.nan, 'limit_down': np.nan}
                    available = 0 if preblocked else own_fillable(quote, side, requested, used[key], self.cfg)
                    require(len(rows) == int(available > 0), '本方向可填请求与guard记录集合不一致')
                    cancelled = False
                    if available:
                        row = rows.iloc[0]
                        require(row.signal_datetime == request.datetime and row.side == side, 'guard与原决策身份不同')
                        cancelled = verify_guard_row(row, quote, cash, prior_old, used[key], requested, self.cfg)
                        observed_guard.add(rows.index[0])
                        if cancelled:
                            require(fills.empty, '已取消首腿仍成交')
                            blocks = attempt.loc[attempt.datetime.eq(stamp) & attempt.symbol.eq(symbol) & attempt.event_type.eq('t_entry') & attempt.reason.eq(row.reason)]
                            require(len(blocks) == 1 and blocks.iloc[0].filled_shares == 0 and blocks.iloc[0].requested_shares == requested, '取消缺独立0成交拒单')
                            observed_cancel.add(blocks.index[0])
                        records.append({'account': str(folder.relative_to(self.report)), 'datetime': stamp, 'symbol': symbol,
                            'side': side, 'cancelled': cancelled, 'reason': row.reason if pd.notna(row.reason) else '',
                            'cash_before': cash, 'old_before': prior_old, 'raw_amount_used_before': used[key],
                            'own_fillable_shares': available, 'same_current_raw_quote_verified': True,
                            'no_virtual_reverse_state_consumption': True})
                    require(len(fills) <= 1, '同请求多个首腿')
                    if len(fills):
                        require(available > 0 and not cancelled, '无通过guard的真实首腿')
                        row = fills.iloc[0]
                        require(row.signal_datetime == request.datetime and row.side == side, '真实首腿与请求不符')
                        observed_first.add(fills.index[0]); apply_event(row)
        require(observed_guard == set(audit.index), '存在未解释guard行')
        all_first = event.loc[event.event_type.eq('t') & event.fragment_id.eq(0)]
        require(observed_first == set(all_first.index), '存在无原请求首腿')
        all_cancel = attempt.loc[attempt.reason.fillna('').str.startswith('execution_exit_side_')]
        require(observed_cancel == set(all_cancel.index), '存在无guard的取消拒单')
        metrics = self.json(folder/'metrics.json')
        close(metrics['execution_exit_side_checks'], len(audit), '复查指标', 0)
        close(metrics['execution_exit_side_cancellations'], len(all_cancel), '取消指标', 0)
        detail = pd.DataFrame(records)
        name = str(folder.relative_to(self.report)).replace('/', '__')
        detail.to_csv(self.out/f'{name}__guard_checks.csv', index=False, encoding='utf-8-sig')
        summary = {'account': str(folder.relative_to(self.report)), 'requests': len(dec), 'guard_rows': len(audit),
            'cancelled': len(all_cancel), 'actual_first_legs': len(all_first), 'all_request_guard_membership_rebuilt': True,
            'cash_old_and_raw_capacity_before_every_guard_verified': True, 'all_cancellations_no_fill_no_virtual_consumption': True}
        self.guards.append(summary)
        return summary

    def predictions(self, fold, scope, minutes, profiles):
        """当前所有输入先与完整市场核对，再用本折新原生权重逐值复现，无未来资格。"""
        folder = self.report/f"year_{fold['test_year']}"
        context = folder/f'{scope}_context'
        identity = self.json(context/'full_minute_coverage_audit.json')
        year = fold['inner_year'] if scope == 'validation' else fold['test_year']
        days = self.market_calendar[self.market_calendar.year == year]
        if self.cfg['smoke_days']: days = days[:self.cfg['smoke_days']]
        current = minutes.loc[minutes.symbol.isin(identity['symbols']) & minutes.datetime.dt.normalize().isin(days)].copy()
        output = {}
        for profile in profiles:
            directory = folder/f'models/close_{profile}'
            meta = self.json(directory/'model_metadata.json')
            target = folder/f'{scope}_predictions/close_{profile}'
            saved = self.table(target/'current_predictions.csv')
            audit = self.json(target/'prediction_audit.json')
            require(digest(target/'current_predictions.csv') == audit['predictions_sha256']
                and audit['model_bundle_sha256'] == meta['model_bundle_sha256'], '当前预测与本折模型SHA不符')
            required = list(dict.fromkeys(KEYS+['signal_price', 'signal_high', 'signal_low', 'signal_amount',
                'signal_volume', 'previous_close']+meta['features']+['feature_source_start']))
            require(set(saved) == set(required+['prediction', 'raw_prediction', 'model_profile', *IDENTITY]), '当前预测schema改变')
            for name in required:
                if pd.api.types.is_numeric_dtype(current[name].dtype): saved[name] = saved[name].astype(current[name].dtype)
            exact(saved[required], current[required], '本折全部当前推断键/特征/来源', dtype=True)
            require(saved.model_profile.eq(profile).all() and all(saved[k].eq(v).all() for k, v in IDENTITY.items()), '预测close身份错')
            matrix = current[meta['features']].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
            native = lightgbm.Booster(model_file=str(self.record(directory/'return_model.txt')))
            values = native.predict(matrix, num_threads=1) if len(matrix) else np.array([])
            require(np.array_equal(values, saved.prediction) and np.array_equal(values, saved.raw_prediction), '新原生预测未精确复现')
            require(audit['future_independent'] is True and all(audit.get(k) == v for k, v in IDENTITY.items()), '原未来擦除/固定目标审计缺失')
            # 未来列既不属于矩阵也不用于当前行选择；归档已保存的每一行都必须存在。
            policy = current[KEYS+['signal_price', 'signal_high', 'signal_low', 'signal_amount', 'signal_volume', 'previous_close']].copy()
            policy['prediction'], policy['raw_prediction'], policy['model_profile'] = values, values, profile
            for key, value in IDENTITY.items(): policy[key] = value
            output[profile] = policy
            self.prediction_results.append(dict(path=str(target.relative_to(self.report)), current_rows=len(current),
                native_predictions_exact=True, all_current_inputs_exact=True, future_eligibility_filter_absent=True,
                same_core_model_before_and_after_inner_selection=True))
        return output

    def run(self):
        """先新目标训练重建，再内部/外层真实账户；不使用审计结果改变任何规则。"""
        from scripts.inventory_close_audit_models import audit_close_model
        self.sources()
        universe, minutes = self.market()
        print('57源与双来源身份通过，开始完整core/固定14:55训练独立重建。', flush=True)
        folds = []
        for year in self.cfg['test_years']:
            folder = self.report/f'year_{year}'
            fold = self.json(folder/'split.json')
            previous = self.json(self.execution_source/f'year_{year}/split.json')
            for key in ('test_year', 'inner_year', 'core_years', 'train_years'):
                require(fold[key] == previous[key], '年度隔离拆分偏离原执行来源')
            require(sorted(fold['core_years']+[fold['inner_year']]) == sorted(fold['train_years'])
                and year not in fold['train_years'] and fold['inner_year'] not in fold['core_years'], '核心/内部/外层年交集')
            core = self.cohort(fold, universe, minutes)
            full = attach_close_reference(core, minutes)
            label_audit = self.json(folder/'fixed_close_label_audit.json')
            require(label_audit['core_rows'] == len(core) and label_audit['label_available_rows'] == int(full.label_available.sum())
                and all(label_audit.get(k) == v for k, v in IDENTITY.items()), '核心端点标签审计行数/身份错')
            training = independent_training_rows(full, fold['core_years'], [year, fold['inner_year']], self.cfg)
            require(len(training) > 0 and training.cohort_target_year.eq(training.datetime.dt.year).all()
                and training.baseline_opening_shares.ge(self.cfg['lot_size']).all(), '实际训练不是核心旧股机会')
            features, feature_audit = feature_selection(training, self.cfg)
            for profile in ('l2_depth4', 'l2_depth6'):
                detail = audit_close_model(self, folder/f'models/close_{profile}', training, features,
                    feature_audit, fold, profile, full_frame=full)
                self.model_results.append(detail)
            print(f'fold={year} 两个新close模型/完整core/标签通过：{len(training):,}训练行。', flush=True)
            folds.append(fold)
            del core, full, training
            gc.collect()
        del universe
        gc.collect()
        evaluation = self.json(self.report/'outer_evaluation_decision.json')
        choices, count, outer, presented = [], 0, 0, {}
        for fold in folds:
            year, folder = fold['test_year'], self.report/f"year_{fold['test_year']}"
            selected = self.json(folder/'selected_rule.json')
            require(digest(folder/'selected_rule.json') == self.freeze['selected_rule_sha256'][str(year)]
                and (folder/'selected_rule.json').stat().st_mtime_ns <= self.freeze_time, '已选规则偏离统一冻结')
            self.audit_context(folder/'validation_context', fold['inner_year'])
            self.source_baseline(year, 'validation')
            forecasts = self.predictions(fold, 'validation', minutes, ['l2_depth4', 'l2_depth6'])
            table = self.table(folder/'validation_rule_metrics.csv')
            original = self.table(self.execution_source/f'year_{year}/validation_rule_metrics.csv')
            require(len(table) == 37 and set(table.candidate_index) == set(range(37)), '37内部候选不齐')
            actual = []
            for rule in rules():
                index = rule['candidate_index']
                target = folder/f'validation_candidates/candidate_{index:03d}'
                require(self.json(target/'rule.json') == rule, '事前规则改变')
                expected = close_decisions(forecasts.get(rule['profile'], pd.DataFrame()), rule, self.cfg)
                saved = self.table(target/'t_decisions.csv')
                if len(expected): exact(saved[list(expected)], expected, '独立单一close/成本/当前条件决策')
                else: require(saved.empty, '伪造空候选请求')
                self.guard_account(target, forecasts, rule)
                result = self.account(target)
                row, old = table.loc[table.candidate_index.eq(index)].iloc[0], original.loc[original.candidate_index.eq(index)].iloc[0]
                for name in ('incremental_net_pnl', 't_pair_count', 'all_reference_acquisition_cost', *FAILURE_KEYS):
                    close(row[name], result[name], f'内部实际账户指标:{name}')
                close(row.all_reference_acquisition_cost, old.all_reference_acquisition_cost, '原全部底仓成本分母', 1e-8)
                actual.append(dict(candidate_index=index, **result))
                count += 1
            eligible = [r for r in actual if r['candidate_index'] and r['t_pair_count'] > 0 and all(r[k] == 0 for k in FAILURE_KEYS)]
            winner = min(eligible, key=lambda r: (-r['incremental_net_pnl'], r['candidate_index'])) if eligible else actual[0]
            require(selected['rule'] == rules()[winner['candidate_index']], '内部选参不是执行合格后最大真实净額')
            expected_rate = winner['incremental_net_pnl']/winner['all_reference_acquisition_cost'] if winner['all_reference_acquisition_cost'] else 0.
            close(selected['inner_metrics']['holding_cost_reduction_rate'], expected_rate, '实际5%分母', 1e-12)
            choices.append(dict(test_year=year, candidate_index=winner['candidate_index'], inner_net=winner['incremental_net_pnl'], cost_rate=expected_rate))
            for name in ('planning_baseline', 'baseline'): self.account(folder/'validation_context'/name)
            if evaluation['outer_evaluation_started']:
                self.audit_context(folder/'test_context', year)
                self.source_baseline(year, 'test')
                rule = selected['rule']
                forecasts = self.predictions(fold, 'test', minutes, [rule['profile']]) if rule['enabled'] else {}
                expected = close_decisions(forecasts.get(rule['profile'], pd.DataFrame()), rule, self.cfg)
                saved = self.table(folder/'t_decisions.csv')
                if len(expected): exact(saved[list(expected)], expected, '外层本折原生close及已冻规则')
                else: require(saved.empty, 'hold外层产生请求')
                require((folder/'test_equity_curve.csv').stat().st_mtime_ns >= self.freeze_time, '外层在全部规则冻结前发生')
                self.guard_account(folder, forecasts, rule)
                presented[year] = self.account(folder)
                for name in ('planning_baseline', 'baseline'): self.account(folder/'test_context'/name)
                outer += 1
            print(f'fold={year} 37内部/已有外层、全部guard与原真实账簿通过。', flush=True)
        passed = all(item['candidate_index'] > 0 and item['cost_rate'] >= self.cfg['profit_target_rate'] for item in choices)
        require(evaluation['all_inner_cost_targets_met'] == passed, '实际内部5%判断错')
        require(evaluation['outer_evaluation_started'] == (not self.cfg['internal_only'] and (passed or self.cfg['report_scope'] == 'smoke')), '外评启停错误')
        goal = self.json(self.report/'goal_evaluation.json')
        validate_goal_status(goal, self.cfg, evaluation['outer_evaluation_started'])
        if presented:
            annual = self.csv(self.report/'annual_summary.csv')
            require(not annual.test_year.duplicated().any() and set(annual.test_year) == set(presented), '最终年度表范围错')
            sums = dict(sum_independent_account_t_increment=0., sum_completed_pair_lots=0.,
                sum_one_way_traded_lots=0., sum_complete_pairs=0, sum_recovery_fragments=0)
            for year, result in presented.items():
                row = annual.loc[annual.test_year.eq(year)].iloc[0]
                mapping = {'incremental_marked_pnl': 'incremental_net_pnl', 'completed_t_pair_lots': 'completed_t_pair_lots',
                    'total_t_traded_lots': 'total_t_traded_lots', 'complete_t_pairs': 't_pair_count',
                    'recovery_fragments': 'fragment_count', 't_fee': 't_fee', 'all_reference_acquisition_cost': 'all_reference_acquisition_cost',
                    **{k: k for k in FAILURE_KEYS}}
                for name, actual in mapping.items(): close(row[name], result[actual], f'最终年度实际账户:{year}/{name}')
                sums['sum_independent_account_t_increment'] += result['incremental_net_pnl']
                sums['sum_completed_pair_lots'] += result['completed_t_pair_lots']
                sums['sum_one_way_traded_lots'] += result['total_t_traded_lots']
                sums['sum_complete_pairs'] += result['t_pair_count']
                sums['sum_recovery_fragments'] += result['fragment_count']
            for name, actual in sums.items(): close(goal[name], actual, '根目标汇总:'+name)
        for path, value in self.inputs.items(): require(digest(path) == value, f'独审读取期间输入改变:{path}')
        return dict(passed=True, complete=True, research_version=VERSION, report_scope=self.cfg['report_scope'],
            source=self.source_summary, models=self.model_results, predictions=self.prediction_results,
            core_cohorts=self.cohort_results, candidate_accounts=count, outer_accounts=outer,
            checked_accounts=len(self.results), accounts=self.results, choices=choices, source_baselines=self.baselines_checked,
            guard_accounts=self.guards, guard_rows_verified=sum(x['guard_rows'] for x in self.guards),
            cancellations_verified=sum(x['cancelled'] for x in self.guards), full_minute_caches=self.cache_audits,
            raw_event_checks=self.checked_raw_events, raw_exit_intent_checks=self.checked_exit_pairs, raw_files=self.raw_files,
            market=self.market_summary, inputs_sha256=self.inputs, all_read_inputs_unchanged=True,
            no_production_close_model_label_policy_or_guard_engine_imported=True,
            no_model_fit_or_account_rerun=True, goal_achieved=goal['goal_achieved'],
            limitations=['旧全截面session/provenance使用已冻结已审计实现；新当前特征、固定端点、实际训练矩阵与策略另行复算。',
                '原lineage/排名与核心规划账来自完整v14证明，并用旧独立核心审计重新核对；不重撮合全部未成交基础请求。',
                '缺真实涨跌停字段仍使用原代理；并不补盘口排队或历史ST点时资料。',
                '账目及来源通过不代表所有候选每日恢复或5%收益达标；历史复测不是全新测试。'])


def self_tests():
    """固定端点、原始目标、完整机会、成本与未来擦除的独立反例。"""
    import unittest
    import tempfile
    class Tests(unittest.TestCase):
        def setUp(self):
            self.cfg = dict(commission_rate=.0008, transfer_rate=.00001, slippage_rate=.0005,
                stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005,
                last_entry_time='14:30', t_avoid_locked_exit_side=True,
                flat_st_limit_rate=.05, nonflat_limit_proxy=.098)
            self.current = pd.DataFrame({'datetime': pd.to_datetime(['2026-03-02 09:35', '2026-03-02 14:00']),
                'symbol': ['SH.600001', 'SH.600002'], 'prediction': [.01, -.01], 'model_profile': 'l2_depth4',
                'signal_price': 10., 'signal_high': 10.01, 'signal_low': 9.99, 'previous_close': 10.,
                'signal_amount': 1e6, 'signal_volume': 1e5, **IDENTITY})
            self.core = self.current[KEYS].copy()
            self.raw = self.core.copy()
            self.raw['execution_time'] = self.raw.datetime+pd.Timedelta(minutes=1)
            self.raw['recovery_time'] = self.raw.datetime.dt.normalize()+pd.Timedelta(hours=14, minutes=55)
            self.raw['execution_price'], self.raw['recovery_price'] = [10., 10.], [11., np.nan]

        def test_endpoint_raw_missing_labels_and_zero_volume(self):
            self.raw['target_return'], self.raw['trade_week_complete'], self.raw['execution_volume'] = np.nan, False, 0.
            result = attach_close_reference(self.core, self.raw)
            self.assertEqual(len(result), 2)
            self.assertAlmostEqual(result.target_return.iloc[0], .1)
            self.assertTrue(np.isnan(result.target_return.iloc[1]))
            self.assertEqual(result.label_available.tolist(), [True, False])
            self.assertTrue(result.label_end.dt.strftime('%H:%M').eq('14:55').all())

        def test_endpoint_wrong_clock_rejected(self):
            for name, value in [('recovery_time', pd.Timestamp('2026-03-02 15:00')),
                                ('execution_time', pd.Timestamp('2026-03-02 09:35'))]:
                raw = self.raw.copy(); raw.loc[0, name] = value
                with self.assertRaises(AssertionError): attach_close_reference(self.core, raw)

        def test_invalid_price_and_core_with_future_rejected(self):
            for value in (0., -1., np.inf):
                raw = self.raw.copy(); raw.loc[0, 'execution_price'] = value
                with self.assertRaises(AssertionError): attach_close_reference(self.core, raw)
            with self.assertRaises(AssertionError):
                attach_close_reference(self.core.assign(target_return=1.), self.raw)

        def test_single_target_direction_and_exact_cost_formula(self):
            result = close_decisions(self.current, rules()[1], self.cfg)
            self.assertEqual(result.action.tolist(), [1, -1])
            expected_buy = 10.1*100*.9995*(1-.00081-.0005)-10.*100*1.0005*1.00081
            self.assertAlmostEqual(result.priority.iloc[0], expected_buy/1000, places=14)
            self.assertTrue(result.planned_recovery_time.dt.strftime('%H:%M').eq('14:55').all())
            self.assertNotIn('horizon_minutes', result)

        def test_future_perturbation_no_prediction_row_selection(self):
            expected = close_decisions(self.current, rules()[1], self.cfg)
            changed = self.current.assign(execution_price=0., recovery_price=np.nan, label_available=False,
                target_return=999., planned_recovery_time=pd.Timestamp('2099-01-01'))
            exact(close_decisions(changed, rules()[1], self.cfg), expected, '未来扰动无效')

        def test_duplicate_profile_and_target_rejected(self):
            for key, value in [('model_profile', 'l2_depth6'), ('recovery_key', '20'),
                                ('recovery_kind', 'trading_minutes'), ('recovery_clock', '15:00'), ('horizon_minutes', 20)]:
                with self.assertRaises(AssertionError):
                    close_decisions(self.current.assign(**{key: value}), rules()[1], self.cfg)
            with self.assertRaises(AssertionError):
                close_decisions(pd.concat([self.current, self.current.iloc[:1]]), rules()[1], self.cfg)

        def test_exact_current_clock(self):
            for stamp in ('2026-03-02 13:00', '2026-03-02 09:35:01', '2026-03-02 14:55'):
                current = self.current.copy(); current.loc[0, 'datetime'] = pd.Timestamp(stamp)
                with self.assertRaises(AssertionError): close_decisions(current, rules()[1], self.cfg)

        def test_current_limit_volume_and_weak_forecast(self):
            self.assertTrue(close_decisions(self.current.assign(prediction=.001), rules()[1], self.cfg).empty)
            self.assertTrue(close_decisions(self.current.assign(signal_volume=0.), rules()[1], self.cfg).empty)
            current = self.current.iloc[:1].copy()
            current['signal_price'], current['signal_high'], current['signal_low'] = 9.5, 9.5, 9.5
            self.assertTrue(close_decisions(current, rules()[1], self.cfg).empty)

        def test_embargo_and_missing_labels_preserve_only_valid_training(self):
            frame = pd.DataFrame({'datetime': pd.to_datetime(['2024-01-08 09:35', '2024-03-04 09:35', '2024-03-05 09:35']),
                'symbol': 'SH.600001', 'feature_source_start': pd.to_datetime(['2023-12-11', '2024-02-05', '2024-02-05']),
                'target_return': [.01, .02, np.nan], 'label_available': [True, True, False]})
            frame['label_start'] = frame.datetime+pd.Timedelta(minutes=1)
            frame['label_end'] = frame.datetime.dt.normalize()+pd.Timedelta(hours=14, minutes=55)
            cfg = dict(embargo_days=7, strict_feature_provenance=True)
            actual = independent_training_rows(frame, [2024], [2023, 2025], cfg)
            self.assertEqual(actual.datetime.tolist(), [pd.Timestamp('2024-03-04 09:35')])

        def test_grid_hold_and_shared_cash_tie(self):
            self.assertEqual(len(rules()), 37)
            self.assertTrue(close_decisions(pd.DataFrame(), rules()[0], self.cfg).empty)
            frame = self.current.copy()
            frame['datetime'], frame['prediction'] = pd.Timestamp('2026-03-02 09:35'), .01
            frame['symbol'] = ['SH.600002', 'SH.600001']
            self.assertEqual(close_decisions(frame, rules()[1], self.cfg).symbol.tolist(), ['SH.600001', 'SH.600002'])

        def test_smoke_requires_existing_source_outer_context(self):
            with tempfile.TemporaryDirectory() as name:
                auditor = CloseAuditor.__new__(CloseAuditor)
                auditor.execution_source = Path(name)
                auditor.cfg = {'report_scope': 'smoke'}
                auditor.baselines_checked = []
                with self.assertRaises(AssertionError): auditor.source_baseline(2026, 'test')
                with self.assertRaises(AssertionError): auditor.source_baseline(2026, 'validation')
                auditor.cfg['report_scope'] = 'internal'
                auditor.source_baseline(2026, 'test')
                self.assertFalse(auditor.baselines_checked[0]['source_scope_exists'])

        def test_frequency_unknown_or_internal_never_claims_goal(self):
            cfg = {'report_scope': 'smoke', 'minimum_completed_pair_lots': None, 'minimum_active_stock_days': None}
            good = {'goal_achieved': False, 'frequency_target_status': 'not_quantified'}
            validate_goal_status(good, cfg, True)
            with self.assertRaises(AssertionError): validate_goal_status({**good, 'goal_achieved': True}, cfg, True)
            with self.assertRaises(AssertionError): validate_goal_status({**good, 'frequency_target_status': 'met'}, cfg, True)
            with self.assertRaises(AssertionError): validate_goal_status({'goal_achieved': True}, {'report_scope': 'internal'}, False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(result.wasSuccessful(), '固定尾盘独审反例失败')
    return {'passed': True, 'tests': result.testsRun}


def main():
    """成功或失败均新建独立审计身份；原报告、数据、模型与运行源只读。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({'close': self_tests(), 'guard': guard_self_tests(), 'models': model_self_tests()}, ensure_ascii=False)); return
    require(args.report_dir, '须指定成功终态--report-dir')
    report = (ROOT/args.report_dir).resolve()
    require((report/'runtime_integrity_audit.json').exists(), '主运行未成功终态，不开始整报告审计')
    out = report/'independent_close_audit'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/Path(__file__).name)
    dependency = out/'audit_dependencies'; dependency.mkdir()
    files = ['inventory_close_audit_models.py', 'audit_inventory_guarded_t.py', 'audit_dynamic_inventory_t.py',
        'audit_inventory_learning_t.py', 'audit_weekly_t_short_horizon.py', 'audit_weekly_t_timing_models.py', 'audit_weekly_t_residual.py']
    hashes = {}
    for name in files:
        path = ROOT/'scripts'/name
        hashes[str(path)] = digest(path)
        shutil.copy2(path, dependency/name)
    start, auditor = time.time(), None
    try:
        tests = {'close': self_tests(), 'guard': guard_self_tests(), 'models': model_self_tests()}
        auditor = CloseAuditor(report, out)
        result = auditor.run()
        require(all(digest(path) == expected for path, expected in hashes.items()), '独立helpers在审计期间改变')
        result.update(self_tests=tests, elapsed_seconds=time.time()-start, script_sha256=digest(__file__),
            audit_dependencies_sha256=hashes, audit_dependencies_unchanged=True)
    except Exception:
        result = dict(passed=False, complete=False, research_version=VERSION, error=traceback.format_exc(),
            elapsed_seconds=time.time()-start, script_sha256=digest(__file__), audit_dependencies_sha256=hashes,
            inputs_sha256=auditor.inputs if auditor else {})
    (out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n')
    rows = ['# v16固定尾盘预测与真实执行独立验收', '',
        f"结果：{'通过' if result['passed'] else '失败（原错误保留）'}。", '',
        '本次只读已保存原始结果，不训练、不重撮合账户、不修改原报告或候选。', '']
    if result['passed']:
        rows += [f"核对{len(result['models'])}个新close模型、{len(result['predictions'])}张当前预测表、"
            f"{result['candidate_accounts']}内部候选及{result['outer_accounts']}外层，含基准共{result['checked_accounts']}账户。",
            '', f"{result['guard_rows_verified']}行guard复查、{result['cancellations_verified']}次取消；"
            f"{result['raw_event_checks']}条成交和{result['raw_exit_intent_checks']}个退出意图逐行核算。跨候选取消合计有重复，不能当单一组合。",
            '', '实际训练从完整永久旧股core重建，原始下一分钟/14:55价格、严格来源隔离、特征筛选、矩阵SHA、原生权重及37规则独立核对。',
            '', *['- '+value for value in result['limitations']], '', '完整计数、源及输入SHA见[verification.json](verification.json)。']
    else: rows += ['```text', result['error'], '```']
    (out/'acceptance.md').write_text('\n'.join(rows)+'\n')
    print(out, flush=True)
    if not result['passed']: raise AssertionError(result['error'])


if __name__ == '__main__': main()
