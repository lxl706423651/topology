"""独立核验 v14 永久底仓学习：来源、拟合矩阵、原生预测、37规则和真实账簿。

必须等待主进程成功终态。只写新的 independent_learning_audit/run_* 目录；不训练、
不调用生产模型/策略/引擎，也不覆盖历史审计。旧量价上下文与来源算法复用已冻结
且此前已审计的版本；新增十一特征、核心队列、标签、选特征及策略另行重算。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import gc
import hashlib
from itertools import product
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
from scripts.audit_dynamic_inventory_t import (
    Auditor as AccountAuditor, FAILURE_KEYS, SESSION, close, digest, fee, next_minute, parse_time, require,
)
from scripts.audit_weekly_t_short_horizon import independent_training_rows, interval_overlap
from scripts.audit_weekly_t_timing_models import independent_timing_features, TIMING_COLUMNS

VERSION = 'inventory_learning_t_v14'
KEYS = ['datetime', 'symbol']
WEEK_KEYS = ['trade_week', 'symbol']
SOURCE_KEYS = WEEK_KEYS + ['feature_week', 'signal_datetime', 'feature_source_start', 'entry_date', 'exit_date']
CONTEXT = ['f_weekly_pred_return_rank', 'f_weekly_pred_amplitude_rank', 'f_weekly_score', 'f_weekly_selection_rank']
DATE_COLUMNS = {'datetime', 'date', 'trade_week', 'feature_week', 'signal_datetime', 'entry_date', 'exit_date',
                'feature_source_start', 'intraday_feature_source_start', 'entry_feature_source_start',
                'entry_rank_source_start', 'portfolio_cash_source_start', 'base_entry_datetime', '_entry_time',
                'label_start', 'label_end', 'execution_time', 'recovery_time', 'planned_recovery_time'}


def frame_digest(frame):
    """绑定真实行序、列序、dtype所产生的数值哈希。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def exact(left, right, label, *, dtype=False):
    """时间/输入数值严格比较；不以放宽容差处理训练输入身份。"""
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                      check_dtype=dtype, check_exact=True)
    except AssertionError as error:
        raise AssertionError(f'{label}: {error}') from error


def recovery_times(times, horizon):
    """独立按下一交易分钟再推进固定期限，跳午休、不跨日。"""
    output = {}
    for stamp in pd.DatetimeIndex(times.unique()):
        first = next_minute(stamp)
        minute = first.hour * 60 + first.minute
        index = SESSION.index(minute) + horizon
        output[stamp] = first.normalize() + pd.Timedelta(minutes=SESSION[index]) if index < len(SESSION) else pd.NaT
    return pd.to_datetime(times.map(output))


def rules():
    """独立枚举事前37候选；收益、次数不改变候选空间。"""
    output = [dict(candidate_index=0, enabled=False, profile='hold', min_net_edge=0., fraction=0., exit_mode='fixed')]
    for profile, margin, fraction, mode in product(['l2_depth4', 'l2_depth6'], [0., .001, .003], [.5, 1.],
                                                 ['fixed', 'profit_buffer', 'profit_buffer_stop']):
        output.append(dict(candidate_index=len(output), enabled=True, profile=profile,
                           min_net_edge=margin, fraction=fraction, exit_mode=mode))
    return output


def decisions(predictions, rule, cfg):
    """独立当前方向净额及期限竞争，无未来报价/标签资格门。"""
    if not rule['enabled']:
        return pd.DataFrame()
    pieces = []
    expected = None
    for horizon in [20, 40, 60]:
        p = predictions[horizon].copy()
        keys = p[KEYS].sort_values(KEYS).reset_index(drop=True)
        require(not keys.duplicated(KEYS).any(), '重复当前预测键')
        if expected is not None:
            exact(keys, expected, '期限当前键')
        expected = keys
        require(p.model_profile.eq(rule['profile']).all() and p.horizon_minutes.eq(horizon).all(), '预测来源模型错配')
        raw = pd.to_numeric(p.prediction)
        require(np.isfinite(raw).all() and raw.gt(-1).all(), '非法价格比预测')
        transaction = cfg['commission_rate'] + cfg['transfer_rate']
        slip = cfg['slippage_rate']
        stamp = pd.Series(np.where(p.datetime < pd.Timestamp('2023-08-28'), cfg['stamp_rate_before_20230828'],
                                   cfg['stamp_rate_from_20230828']), index=p.index)
        buy = (1+raw)*(1-slip)*(1-transaction-stamp)-(1+slip)*(1+transaction)
        sell = (1-slip)*(1-transaction-stamp)-(1+raw)*(1+slip)*(1+transaction)
        action = pd.Series(np.where(buy >= sell, 1, -1), index=p.index)
        edge = pd.Series(np.maximum(buy, sell), index=p.index)
        when = recovery_times(p.datetime, horizon)
        day = p.datetime.dt.normalize()
        mask = edge.gt(0) & edge.ge(rule['min_net_edge'])
        if cfg['t_avoid_locked_exit_side']:
            price, previous = p.signal_price.astype(float), p.previous_close.astype(float)
            valid = (np.isfinite(price) & np.isfinite(previous) & price.gt(0) & previous.gt(0)
                     & np.isfinite(p.signal_high) & np.isfinite(p.signal_low)
                     & p.signal_high.ge(price) & p.signal_low.le(price))
            reverse = pd.Series(np.where(action > 0, -1., 1.), index=p.index)
            move = (price / previous - 1) * reverse
            limit = np.floor(previous*(1+reverse*cfg['flat_st_limit_rate'])*100+.5+1e-9)/100
            mask &= valid & ~(move.ge(cfg['nonflat_limit_proxy'])
                              | (p.signal_high.eq(p.signal_low) & (price-limit).abs().le(1e-6)))
        mask &= np.isfinite(p.signal_amount) & p.signal_amount.gt(0) & np.isfinite(p.signal_volume) & p.signal_volume.gt(0)
        mask &= p.datetime.le(day+pd.to_timedelta(cfg['last_entry_time']+':00'))
        mask &= when.notna() & when.dt.normalize().eq(day)
        mask &= when.le(day+pd.to_timedelta(cfg['fixed_recovery_time']+':00'))
        mask &= when.gt(p.datetime+pd.Timedelta(minutes=1))
        part = p.loc[mask, KEYS].copy()
        part['action'], part['priority'] = action.loc[mask], edge.loc[mask]
        part['planned_recovery_time'], part['prediction'] = when.loc[mask], raw.loc[mask]
        part['horizon_minutes'] = horizon
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values(KEYS+['priority', 'horizon_minutes', 'action'],
        ascending=[True, True, False, True, False], kind='stable').drop_duplicates(KEYS)
    result['fraction'], result['candidate_index'] = rule['fraction'], rule['candidate_index']
    result['signal_strength'] = result.priority
    result['predicted_net_edge'] = result.priority
    result['regime'], result['model_profile'] = 'learned_permanent_inventory', rule['profile']
    return result.sort_values(['datetime', 'priority', 'symbol'], ascending=[True, False, True], kind='stable').reset_index(drop=True)


def feature_selection(training, cfg):
    """独立重算原始收益秩相关、完整训练覆盖率及贪心去相关。"""
    names = sorted(c for c in training if c.startswith('f_') and c not in cfg['excluded_model_features'])
    values = training[names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    indices = training.sample(n=min(cfg['feature_selection_sample_rows'], len(training)), random_state=cfg['random_state']).index
    sample = values.loc[indices]
    coverage, diversity = values.notna().mean(), sample.nunique(dropna=True)
    eligible = [c for c in names if coverage[c] >= cfg['feature_min_coverage'] and diversity[c] > 1]
    target = pd.to_numeric(training.target_return).loc[indices].rank(pct=True)
    relevance = pd.Series(0., index=names)
    if target.nunique() > 1 and eligible:
        relevance.loc[eligible] = sample[eligible].rank(pct=True).corrwith(target).abs().fillna(0)
    ordered = sorted(eligible, key=lambda c: (-float(relevance[c]), c))
    correlation = sample[ordered].corr(method='spearman')
    selected, rows = [], []
    for name in ordered:
        peers = correlation.loc[name, selected].abs().dropna()
        maximum = float(peers.max()) if len(peers) else 0.
        reason = ('correlation_limit' if maximum > cfg['feature_max_correlation'] else
                  'feature_limit' if len(selected) >= cfg['t_max_features'] else 'selected')
        if reason == 'selected':
            selected.append(name)
        rows.append(dict(feature=name, coverage=coverage[name], target_rank_relevance=relevance[name],
                         max_selected_correlation=maximum, selected=reason == 'selected', reason=reason))
    for name in names:
        if name not in eligible:
            rows.append(dict(feature=name, coverage=coverage[name], target_rank_relevance=relevance[name],
                max_selected_correlation=0., selected=False,
                reason='coverage' if coverage[name] < cfg['feature_min_coverage'] else 'constant'))
    return selected, pd.DataFrame(rows)


class LearningAuditor(AccountAuditor):
    """复用独立原分钟/账簿核算，不调用旧审计的v13版本/163规则入口。"""

    def __init__(self, report, out):
        self.report, self.out = report, out
        self.inputs, self.raw, self.full_raw, self.raw_files, self.results = {}, {}, {}, {}, []
        self.cfg = self.json(report/'resolved_config.json')
        self.protocol = self.json(report/'research_protocol.json')
        require(self.cfg['research_version'] == VERSION, '只支持v14')
        require(self.protocol['valuation_contract'] == 'full_minute_reference_stream_then_daily_close', '估值协议未知')
        self.market_calendar = pd.DatetimeIndex([])
        self.checked_raw_events = self.checked_exit_pairs = 0
        self.cache_audits, self.sparse_arrays, self.mark_cache, self.exit_cache, self.base_anchors = [], {}, {}, {}, {}
        self.valuation_mode = 'full_minute_reference_stream_then_daily_close'
        self.model_results, self.prediction_results, self.cohort_results = [], [], []

    def table(self, path, *, original=False):
        """明确区分原周名单默认float解析与新证据round-trip合同。"""
        if original:
            frame = pd.read_csv(self.record(path))
        else:
            frame = self.csv(path)
        for name in DATE_COLUMNS.intersection(frame):
            frame[name] = pd.to_datetime(frame[name], format='mixed', errors='raise')
        return frame

    def market(self):
        """复用旧已审计量价/来源，独立重算全截面十一新特征后才按账户切片。"""
        from quant_backtest.weekly_t_features import add_weekly_execution_features, add_session_context_features
        from quant_backtest.weekly_t_provenance import add_feature_provenance
        cfg = self.cfg
        cache = Path(cfg['market_cache_dir'])
        known = self.json(Path(cfg['source_audit_report_dir'])/'input_sources.json')
        frames = []
        for name in ['weekly', 'intraday', 'daily']:
            path = (cache/f'{name}.pkl').resolve()
            require(digest(self.record(path)) == known[str(path)]['sha256'], '行情PKL偏离原审计')
            frames.append(pd.read_pickle(path))
        weekly, minutes, daily = frames
        del frames
        self.market_calendar = pd.DatetimeIndex(daily.date.unique()).normalize().sort_values()
        weekly = add_weekly_execution_features(weekly, daily)
        minutes = add_session_context_features(minutes)
        weekly, minutes, provenance = add_feature_provenance(weekly, minutes, daily)
        require(provenance == self.json(self.report/'feature_provenance_manifest.json'), '实际来源manifest未复现')
        minutes = independent_timing_features(minutes)
        # 三列分别以float32保存，优先原均值字段才能严格保持原股票池边界。
        amount = (weekly.f_avg_daily_amount if 'f_avg_daily_amount' in weekly else
                  weekly.f_daily_amount_mean if 'f_daily_amount_mean' in weekly else weekly.f_week_amount/weekly.f_week_days)
        mask = (weekly.f_history_days.ge(cfg['min_history_days']) & weekly.f_last_close.ge(cfg['min_price'])
                & amount.ge(cfg['min_avg_daily_amount']) & weekly.feature_week_complete.eq(True))
        mask &= weekly.f_prior_close_auction_amount_min.ge(cfg['min_prior_close_auction_amount'])
        universe = weekly.loc[mask].copy()
        universe['universe_previous_close'], universe['universe_avg_daily_amount'] = weekly.f_last_close.loc[mask], amount.loc[mask]
        manifest = self.json(self.report/'horizon_manifest.json')
        label_path = Path(manifest['labels_path'])
        require(digest(self.record(label_path)) == manifest['labels_file_sha256'], '短周期标签PKL身份不符')
        labels = pd.read_pickle(label_path)
        require(len(labels) == manifest['label_rows'], '标签行数错误')
        self.market_summary = dict(full_cross_section_rows=len(minutes), symbols=int(minutes.symbol.nunique()),
            independent_current_features=TIMING_COLUMNS, source_provenance_matches=True)
        return universe, minutes, daily, labels

    def cohort(self, fold, universe, minutes):
        """由完整排名供体、原OOF名单、已保存实际基础成交独立恢复纯core机会。"""
        cfg = self.cfg
        parent = Path(cfg['source_training_report_dir'])
        old_audit = self.json(parent/'independent_residual_audit.json')
        require(all(old_audit[k] is True for k in ['passed', 'complete', 'full_configured_scope_complete']), '旧OOF审计不完整')
        original = self.json(parent/f"year_{fold['test_year']}/oof_weekly_cohort/cohort_metadata.json")
        require(original['weekly_source_keys_sha256'] == frame_digest(universe[SOURCE_KEYS]), '全universe来源键错误')
        require(original['core_years'] == fold['core_years'] and original['excluded_years'] == sorted([fold['test_year'], fold['inner_year']]), 'OOF折身份错')
        folder = self.report/f"year_{fold['test_year']}/training_cohort/inventory_learning_cohort"
        saved_meta = self.json(folder/'cohort_metadata.json')
        parts = []
        for year in fold['core_years']:
            out = folder/f'core_{year}'
            selector = parent/f"year_{fold['test_year']}/oof_weekly_cohort/selector_{year}"
            checked = old_audit['selectors'][str(selector.relative_to(parent))]
            model = self.json(selector/'model_metadata.json')
            train = sorted(set(fold['core_years'])-{year})
            excluded = sorted(set(original['available_years'])-set(train))
            require(checked['passed'] and model['train_years'] == train and model['excluded_years'] == excluded, 'selector训练年不符')
            require(model['training_sample_sha256'] == checked['sample_sha256']
                    and model['feature_source_sha256'] == checked['source_sha256'], 'selector实际训练/来源摘要偏离旧审计')
            for name, expected in [('model_bundle.joblib', checked['model_bundle_sha256']),
                                   ('oof_weekly_selections.csv', checked['selected_csv_sha256'])]:
                require(digest(self.record(selector/name)) == expected, 'OOF已审计权重/名单SHA不符')
            for head, detail in checked['native_checks'].items():
                require(digest(self.record(selector/f'{head}_model.txt')) == detail['native_sha256'], 'OOF原生权重SHA不符')
            targets = universe.loc[universe.entry_date.dt.year.eq(year) & universe.exit_date.dt.year.eq(year)
                & universe.entry_date.lt(universe.exit_date) & universe.trade_week.dt.year.eq(year)
                & universe.feature_week_complete.eq(True)].copy()
            original_predictions = self.table(selector/'oof_weekly_predictions.csv')
            # pandas usecols保留源CSV列序，而非调用方列表顺序；保存的donor表继承该顺序。
            donor_keys = original_predictions[[c for c in original_predictions if c in WEEK_KEYS]]
            require(not donor_keys.duplicated(WEEK_KEYS).any(), '供体键重复')
            exact(donor_keys[WEEK_KEYS].sort_values(WEEK_KEYS), targets[WEEK_KEYS].sort_values(WEEK_KEYS), '原完整排名供体')
            donors = donor_keys.merge(targets[SOURCE_KEYS], on=WEEK_KEYS, validate='one_to_one')
            exact(self.table(out/'ranking_donor_sources.csv'), donors, '供体实际来源')
            grouped = donors.groupby('trade_week', sort=False)
            starts = grouped.feature_source_start.min()
            starts.loc[grouped.feature_source_start.apply(lambda c: c.isna().any())] = pd.NaT
            selected = self.table(selector/'oof_weekly_selections.csv')
            requests = selected.merge(universe[SOURCE_KEYS+['f_last_close']], on=WEEK_KEYS,
                                      suffixes=('', '_original'), validate='one_to_one')
            requests['entry_rank_source_start'] = requests.trade_week.map(starts)
            requests['entry_feature_source_start'] = requests.feature_source_start
            start, signal = requests.entry_rank_source_start, requests.signal_datetime
            execution = requests.entry_date + pd.Timedelta(hours=9, minutes=30)
            invalid = start.isna() | signal.isna() | execution.isna() | start.gt(signal) | signal.ge(execution)
            wrong = ~start.dt.year.isin(fold['core_years']) | ~signal.dt.year.isin(fold['core_years']) | ~execution.dt.year.isin(fold['core_years'])
            crossed = pd.Series(False, index=requests.index)
            for banned in [fold['inner_year'], fold['test_year']]:
                crossed |= start.lt(pd.Timestamp(banned+1, 1, 1)+pd.Timedelta(days=7)) & execution.ge(pd.Timestamp(banned, 1, 1)-pd.Timedelta(days=7))
            keep = ~(invalid | wrong | crossed)
            entry_audit = self.table(out/'entry_source_audit.csv')
            exact(entry_audit[WEEK_KEYS+['entry_rank_source_start']], requests[WEEK_KEYS+['entry_rank_source_start']], '入场来源行未篡改')
            require(np.array_equal(entry_audit.entry_source_eligible, keep), '先purge后规划资格不符')
            requests = requests.loc[keep].copy()
            events = self.table(out/'base_entries.csv')
            frozen = self.table(out/'frozen_base_selections.csv')
            require(not events.symbol.duplicated().any() and not frozen.symbol.duplicated().any(), '自动底仓每股不止首次买入')
            inventory = self.table(out/'daily_inventory_audit.csv')
            equity = self.table(out/'planning_equity_curve.csv')
            year_days = self.market_calendar[self.market_calendar.year == year]
            require(pd.DatetimeIndex(equity.datetime).normalize().equals(year_days), 'smoke截断了核心全年日历')
            cash, quantities, used = float(cfg['initial_cash']), defaultdict(int), defaultdict(float)
            groups = {d: p for d, p in events.groupby(events.datetime.dt.normalize())}
            for day in year_days:
                daily_rows = inventory.loc[inventory.date.eq(day)]
                today = groups.get(day, pd.DataFrame(columns=events.columns))
                expected_symbols = {s for s, q in quantities.items() if q > 0} | set(today.symbol)
                require(not daily_rows.symbol.duplicated().any() and set(daily_rows.symbol) == expected_symbols,
                        '核心基准日库存漏行/提前出现持仓')
                for row in daily_rows.itertuples():
                    require(row.baseline_opening_shares == quantities[row.symbol], '核心基准旧股开盘数量与真实成交不符')
                for row in today.itertuples(index=False):
                    require(row.side == 'buy' and row.event_type == 'weekly' and row.base_order_type == 'automatic', '核心基准出现T/卖出')
                    require(row.datetime == day+pd.Timedelta(hours=9, minutes=30), '核心基础成交时钟不符')
                    matching = requests.loc[requests.symbol.eq(row.symbol) & requests.entry_date.eq(day)]
                    require(len(matching) == 1 and matching.signal_datetime.iloc[0] < row.datetime, '基础成交未对应干净已知请求')
                    self.verify_event(row, used)
                    cash -= row.notional+row.fee
                    quantities[row.symbol] += int(row.shares)
                    require(cash >= cfg['initial_cash']*(1-cfg['base_fraction'])-1e-6
                            and cfg['min_position_notional']-1e-6 <= row.notional <= cfg['max_position_notional']+1e-6, '核心基础金额/保留现金错误')
                for row in daily_rows.itertuples():
                    require(row.baseline_closing_shares == quantities[row.symbol], '核心基准收盘持股与全部买入累计不符')
                close(float(equity.loc[equity.datetime.dt.normalize().eq(day), 'baseline_cash'].iloc[0]), cash, '核心基准现金')
            require(set(events.symbol) == set(frozen.symbol), '核心冻结持股未对应真实买入')
            old = inventory.loc[inventory.baseline_opening_shares.gt(0), ['date', 'symbol', 'baseline_opening_shares']]
            lookup = pd.MultiIndex.from_frame(old[['date', 'symbol']])
            loc = lookup.get_indexer(pd.MultiIndex.from_arrays([minutes.datetime.dt.normalize(), minutes.symbol]))
            omitted = {c for c in minutes if c.startswith(('execution_', 'recovery_', 'target_', 'label_'))} | set(CONTEXT) | {'trade_week_complete'}
            current = minutes.loc[loc >= 0, [c for c in minutes if c not in omitted]].copy()
            current['date'] = current.datetime.dt.normalize()
            current['baseline_opening_shares'] = old.baseline_opening_shares.to_numpy()[loc[loc >= 0]]
            current['intraday_feature_source_start'] = current.feature_source_start
            first = frozen.sort_values('_entry_time', kind='stable').drop_duplicates('symbol').set_index('symbol')
            event_index = events.set_index('symbol')
            for symbol, row in first.iterrows():
                event = event_index.loc[symbol]
                require(row._entry_time == event.datetime and row.signal_datetime == event.signal_datetime,
                        '冻结底仓与实际成交时刻不符')
                for saved, actual in [('frozen_base_shares', 'shares'), ('frozen_base_reference_price', 'reference_price'),
                                      ('frozen_base_fee', 'fee'), ('frozen_base_notional', 'notional')]:
                    close(float(row[saved]), float(event[actual]), '冻结底仓'+saved, 0.)
            for name in ['entry_feature_source_start', 'entry_rank_source_start', 'source_model_training_years',
                         'source_model_sha256', '_entry_time', 'frozen_base_shares']:
                current['base_entry_datetime' if name == '_entry_time' else name] = current.symbol.map(first[name])
            request_index = requests.set_index(['entry_date', 'symbol'])
            for row in first.reset_index().itertuples(index=False):
                known = request_index.loc[(row.entry_date, row.symbol)]
                require(row.entry_rank_source_start == known.entry_rank_source_start
                        and row.entry_feature_source_start == known.entry_feature_source_start
                        and row.source_model_sha256 == checked['model_bundle_sha256']
                        and row.source_model_training_years == known.source_model_training_years
                        == json.dumps(train, separators=(',', ':')), '冻结底仓来源未追溯原请求')
            requests['_clock'] = requests.entry_date+pd.Timedelta(hours=9, minutes=30)
            cash_sources = requests.groupby('_clock').entry_rank_source_start.min().sort_index().cummin()
            index = np.searchsorted(cash_sources.index.to_numpy(), current.datetime.to_numpy(), side='right')-1
            require(not len(current) or (index >= 0).all(), '持仓机会早于基础请求')
            current['portfolio_cash_source_start'] = cash_sources.to_numpy()[index] if len(current) else pd.Series(dtype='datetime64[ns]')
            current['feature_source_start'] = np.minimum(np.minimum(current.intraday_feature_source_start.to_numpy(),
                current.entry_rank_source_start.to_numpy()), current.portfolio_cash_source_start.to_numpy())
            current['cohort_target_year'] = year
            current = current.sort_values(KEYS, kind='stable').reset_index(drop=True)
            lineage = self.table(out/'cohort_lineage.csv')
            exact(lineage, current[list(lineage)], '永久持股当前机会/现金/供体来源')
            summary = self.json(out/'metadata.json')
            require(frame_digest(current[list(lineage)]) == summary['cohort_lineage_sha256'], '核心lineageSHA不符')
            parts.append(current)
            self.cohort_results.append(dict(fold=fold['test_year'], core_year=year, rows=len(current),
                ranking_donors=len(donors), clean_requests=len(requests), actual_base_entries=len(events),
                all_ranking_and_cash_sources_propagated=True, calendar_days=len(year_days)))
        cohort = pd.concat(parts, ignore_index=True).sort_values(KEYS, kind='stable').reset_index(drop=True)
        require(len(cohort) == saved_meta['training_opportunity_rows'], '完整核心机会总数不符')
        return cohort

    def model(self, directory, training, features, feature_audit, fold, horizon, profile):
        """逐值恢复真正拟合矩阵/原始目标、严格排除区间及两份模型权重。"""
        cfg = self.cfg
        meta = self.json(directory/'model_metadata.json')
        require(meta['train_years'] == fold['core_years'] and meta['excluded_years'] == sorted([fold['inner_year'], fold['test_year']]), 'T模型年份泄漏')
        require(meta['features'] == features and all(c.startswith('f_') and c != 'f_history_days' for c in features), '特征集合非法')
        require(meta['research_version'] == VERSION and meta['profile'] == profile and meta['horizon_minutes'] == horizon, '模型身份错')
        exact(self.table(directory/'feature_manifest.csv'), feature_audit, '核心特征选择复算')
        require(meta['fitted_rows'] == meta['training_rows'] == len(training) and cfg['t_max_train_rows'] == 0, '实际拟合行数/采样错误')
        overlaps, counts = interval_overlap(training, meta['excluded_years'], 7)
        require(not overlaps.any(), '真实特征至标签区间穿越排除年')
        matrix = training[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        actual_inputs = pd.read_pickle(self.record(directory/'training_inputs.pkl'))
        expected = training[KEYS+['feature_source_start', 'cohort_target_year', 'baseline_opening_shares']].copy()
        for c in features:
            expected[c] = matrix[c].to_numpy()
        exact(actual_inputs, expected, '真实拟合input.pkl', dtype=True)
        saved_labels = self.table(directory/'training_labels.csv')
        expected_labels = training[[c for c in saved_labels if c != 'fitted']].copy()
        expected_labels['fitted'] = True
        exact(saved_labels, expected_labels, '保存原始收益标签/端点')
        require(training.label_start.eq(training.datetime+pd.Timedelta(minutes=1)).all()
                and training.label_end.eq(recovery_times(training.datetime, horizon)).all(), '标签期限错误')
        close(float(np.max(np.abs(training.recovery_price/training.execution_price-1-training.target_return))), 0., '原始标签非残差', 1e-12)
        for (year, symbol), part in training.groupby([training.datetime.dt.year, 'symbol'], sort=False):
            raw = self.full_bars(year, symbol)
            for endpoint, name in [('label_start', 'execution_price'), ('label_end', 'recovery_price')]:
                actual = raw.close.reindex(pd.DatetimeIndex(part[endpoint])).to_numpy()
                require(np.isfinite(actual).all() and np.allclose(actual, part[name], rtol=0, atol=1e-10),
                        f'训练标签{endpoint}未对应原始精确分钟:{year}/{symbol}')
        hashes = {'full_training_rows_sha256': frame_digest(training[KEYS+['feature_source_start']]),
                  'training_matrix_sha256': frame_digest(matrix),
                  'training_target_sha256': frame_digest(training[['target_return']])}
        for key, value in hashes.items():
            require(meta[key] == value, f'模型{key}不符')
        for file, field in [('training_inputs.pkl', 'training_inputs_file_sha256'), ('training_labels.csv', 'training_labels_file_sha256'),
                            ('return_model.txt', 'native_model_sha256'), ('model_bundle.joblib', 'model_bundle_sha256')]:
            require(digest(self.record(directory/file)) == meta[field], f'模型产物SHA不符:{file}')
        bundle = joblib.load(directory/'model_bundle.joblib')
        native = lightgbm.Booster(model_file=str(directory/'return_model.txt'))
        require(native.feature_name() == features == bundle['features'], '原生特征顺序不符')
        require(native.model_to_string() == bundle['model'].booster_.model_to_string(), 'joblib和原生树权重不同')
        parameters = meta['model_parameters']
        require(parameters['objective'] == 'regression' and parameters['max_depth'] == (4 if profile == 'l2_depth4' else 6)
                and parameters['num_leaves'] == (15 if profile == 'l2_depth4' else 31)
                and parameters['deterministic'] is True and parameters['n_jobs'] == cfg['workers'], 'L2模型参数错')
        for k, v in cfg['t_model'].items():
            require(parameters[k] == v, '实际模型超参偏离预声明')
        require(all(bundle['model'].get_params()[k] == v for k, v in parameters.items()), '实际拟合器与参数元数据不同')
        sample = matrix.iloc[np.linspace(0, len(matrix)-1, min(len(matrix), 1024), dtype=int)]
        close(float(np.max(np.abs(native.predict(sample)-bundle['model'].predict(sample)))), 0., '训练原生重载', 1e-12)
        self.model_results.append(dict(path=str(directory.relative_to(self.report)), rows=len(training), features=features,
            passed=True, independently_rebuilt_training_inputs=True, raw_reference_target=True, source_overlap_counts=counts, hashes=hashes))

    def predictions(self, fold, scope, minutes, model_keys):
        """每个推断行/特征逐值来自当前缓存；原生weights复算，不按未来标签删行。"""
        out = self.report/f"year_{fold['test_year']}"
        context = out/f'{scope}_context'
        identity = self.json(context/'full_minute_coverage_audit.json')
        names = identity['symbols']
        year = fold['inner_year'] if scope == 'validation' else fold['test_year']
        days = self.market_calendar[self.market_calendar.year == year]
        if self.cfg['smoke_days']:
            days = days[:self.cfg['smoke_days']]
        current = minutes.loc[minutes.symbol.isin(names) & minutes.datetime.dt.normalize().isin(days)].copy()
        forecasts = {}
        prefix = 'validation_predictions' if scope == 'validation' else 'test_predictions'
        for profile, horizon in model_keys:
            directory = out/f'models/h{horizon}_{profile}'
            meta = self.json(directory/'model_metadata.json')
            path = out/prefix/f'h{horizon}_{profile}'
            saved = self.table(path/'current_predictions.csv')
            own = self.json(path/'prediction_audit.json')
            require(digest(path/'current_predictions.csv') == own['predictions_sha256'], '预测CSV身份不符')
            require(own['model_bundle_sha256'] == meta['model_bundle_sha256'], '推断借用错误模型')
            fields = [c for c in saved if c in current and c not in ['prediction', 'raw_prediction', 'horizon_minutes', 'model_profile']]
            # pandas保存float32时采用短小十进制；恢复原缓存dtype，而非放宽gate或SHA容差。
            for name in fields:
                if pd.api.types.is_numeric_dtype(current[name].dtype):
                    saved[name] = saved[name].astype(current[name].dtype)
            exact(saved[fields], current[fields], '当前预测完整原始特征及资格输入')
            matrix = current[meta['features']].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
            native = lightgbm.Booster(model_file=str(directory/'return_model.txt'))
            values = native.predict(matrix) if len(matrix) else np.array([])
            require(np.array_equal(values, saved.prediction.to_numpy()) and np.array_equal(values, saved.raw_prediction.to_numpy()), '原生当前预测未逐值复现')
            require(saved.model_profile.eq(profile).all() and saved.horizon_minutes.eq(horizon).all(), '推断期限/档位不符')
            require(own['future_independent'] is True, '未保存未来擦除检查')
            policy = current[['datetime', 'symbol', 'signal_price', 'signal_high', 'signal_low', 'signal_amount', 'signal_volume', 'previous_close']].copy()
            policy['prediction'], policy['raw_prediction'] = values, values
            policy['horizon_minutes'], policy['model_profile'] = horizon, profile
            forecasts.setdefault(profile, {})[horizon] = policy
            self.prediction_results.append(dict(path=str(path.relative_to(self.report)), current_rows=len(current),
                native_predictions_exact=True, current_inputs_exact=True, future_eligibility_filter_absent=True))
        return forecasts

    def run(self):
        """先所有模型来源，后实际候选/账簿；外层只按原freeze检查，不选新规则。"""
        terminal = self.json(self.report/'runtime_integrity_audit.json')
        require(terminal.get('all_unchanged') is True and (self.report/'final_report.md').exists(), '主运行尚未成功终态')
        self.record(self.report/'final_report.md')
        sources = self.json(self.report/'source_hashes.json')
        require(len(sources) == terminal['runtime_sources_verified'] == 48, '实际48源码快照不完整')
        changes = {}
        for name, value in sources.items():
            require(digest(self.record(self.report/'source_snapshot'/name)) == value, '原运行source_snapshot改变')
            if digest(ROOT/name) != value:
                changes[name] = dict(frozen=value, current=digest(ROOT/name))
        original = self.json(self.report/'input_sources.json')
        for path, identity in original.items():
            require(digest(self.record(path)) == identity['sha256'], f'原输入已改变:{path}')
        freeze = self.json(self.report/'all_rules_frozen_before_outer_test.json')
        require(freeze['test_years'] == self.cfg['test_years'], '全折冻结不完整')
        freeze_time = (self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns
        for path, value in freeze['model_artifacts_sha256'].items():
            require(digest(self.record(self.report/path)) == value, f'拟合后模型产物改变:{path}')
            require((self.report/path).stat().st_mtime_ns <= freeze_time, '模型在全折freeze后生成/改写')
        require(self.protocol['candidate_rules'] == rules() and self.protocol['outer_model_refit'] is False, '事前规则/禁止外评refit不符')
        universe, minutes, daily, labels = self.market()
        del daily
        gc.collect()
        print('48源、原缓存和全截面已核验；开始独立恢复核心训练。', flush=True)
        folds = []
        for year in self.cfg['test_years']:
            fold = self.json(self.report/f'year_{year}/split.json')
            require(year == fold['test_year'] and year not in fold['train_years'] and fold['inner_year'] not in fold['core_years'], '年度交集错误')
            require(sorted(fold['core_years']+[fold['inner_year']]) == sorted(fold['train_years']), '核心/内部非原训练拆分')
            old = self.json(Path(self.cfg['source_training_report_dir'])/f'year_{year}/split.json')
            require(all(fold[k] == v for k, v in old.items()), '既有年度拆分改变')
            cohort = self.cohort(fold, universe, minutes)
            for horizon in [20, 40, 60]:
                fields = ['execution_time', 'execution_price', 'recovery_time', 'recovery_price', 'label_start', 'label_end', 'target_return']
                targets = labels.loc[labels.horizon_minutes.eq(horizon), KEYS+fields]
                joined = cohort.merge(targets, on=KEYS, how='left', validate='one_to_one', sort=False)
                joined['label_available'] = np.isfinite(joined.target_return)
                joined['horizon_minutes'] = horizon
                training = independent_training_rows(joined, fold['core_years'], [year, fold['inner_year']], self.cfg)
                selected, audit = feature_selection(training, self.cfg)
                for profile in ['l2_depth4', 'l2_depth6']:
                    self.model(self.report/f'year_{year}/models/h{horizon}_{profile}', training, selected, audit, fold, horizon, profile)
                print(f'fold={year} h{horizon} 两模型拟合输入/标签/原生权重核验完成：{len(training):,}行', flush=True)
                del training, joined
            del cohort
            folds.append(fold)
            gc.collect()
        del universe, labels
        gc.collect()
        evaluation = self.json(self.report/'outer_evaluation_decision.json')
        choices, candidates, outer = [], 0, 0
        marker_time = (self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns
        for fold in folds:
            year = fold['test_year']
            folder = self.report/f'year_{year}'
            selected = self.json(folder/'selected_rule.json')
            require(digest(folder/'selected_rule.json') == freeze['selected_rule_sha256'][str(year)], '选中规则偏离统一freeze')
            require((folder/'selected_rule.json').stat().st_mtime_ns <= marker_time, '规则全折freeze后重写')
            self.audit_context(folder/'validation_context', fold['inner_year'])
            forecasts = self.predictions(fold, 'validation', minutes, list(product(['l2_depth4', 'l2_depth6'], [20, 40, 60])))
            table = self.table(folder/'validation_rule_metrics.csv')
            require(len(table) == 37 and set(table.candidate_index) == set(range(37)), '37内部真实候选不齐')
            actual = []
            for rule in rules():
                index = rule['candidate_index']
                directory = folder/f'validation_candidates/candidate_{index:03d}'
                require(self.json(directory/'rule.json') == rule, '实际规则变更')
                expected = decisions(forecasts.get(rule['profile'], {}), rule, self.cfg)
                saved = self.table(directory/'t_decisions.csv')
                if len(expected):
                    exact(saved[list(expected)], expected, '独立当前净边际/期限决策')
                else:
                    require(saved.empty, '应为空的候选伪造信号')
                result = self.account(directory)
                row = table.loc[table.candidate_index.eq(index)].iloc[0]
                for key in ['t_pair_count', 'incremental_net_pnl', *FAILURE_KEYS]:
                    close(row[key], result[key], f'内部汇总{index}/{key}')
                actual.append(dict(candidate_index=index, **result))
                candidates += 1
            eligible = [r for r in actual if r['candidate_index'] > 0 and r['t_pair_count'] > 0 and all(r[k] == 0 for k in FAILURE_KEYS)]
            winner = sorted(eligible, key=lambda r: (-r['incremental_net_pnl'], r['candidate_index']))[0] if eligible else actual[0]
            require(winner['candidate_index'] == selected['rule']['candidate_index'] and selected['rule'] == rules()[winner['candidate_index']], '原内部选参未遵守真实净额/执行要求')
            choices.append(dict(test_year=year, candidate_index=winner['candidate_index'], inner_net_pnl=winner['incremental_net_pnl']))
            for name in ['planning_baseline', 'baseline']:
                self.account(folder/'validation_context'/name)
            if evaluation['outer_evaluation_started']:
                self.audit_context(folder/'test_context', year)
                chosen = selected['rule']
                keys = [(chosen['profile'], h) for h in [20, 40, 60]] if chosen['enabled'] else []
                predicted = self.predictions(fold, 'test', minutes, keys) if keys else {}
                expected = decisions(predicted.get(chosen['profile'], {}), chosen, self.cfg)
                saved = self.table(folder/'t_decisions.csv')
                if len(expected):
                    exact(saved[list(expected)], expected, '原生外层权重/当前规则')
                else:
                    require(saved.empty, '外层伪造请求')
                require((folder/'test_equity_curve.csv').stat().st_mtime_ns >= marker_time, '外层早于全折冻结')
                self.account(folder)
                for name in ['planning_baseline', 'baseline']:
                    self.account(folder/'test_context'/name)
                outer += 1
            print(f'fold={year} 37规则、原始行情/动态退出及两本账户全部核验完成', flush=True)
        passed = all(self.json(self.report/f"year_{r['test_year']}/selected_rule.json")['inner_metrics']['holding_cost_reduction_rate'] >= .05
                     and r['candidate_index'] > 0 for r in choices)
        require(evaluation['all_inner_cost_targets_met'] == passed, '5%内部晋级不符')
        expected_outer = not self.cfg['internal_only'] and (passed or self.cfg['report_scope'] == 'smoke')
        require(evaluation['outer_evaluation_started'] == expected_outer, '外评启停政策不符')
        goal = self.json(self.report/'goal_evaluation.json')
        if self.cfg['report_scope'] == 'smoke':
            require(not goal['goal_achieved'], '烟测不能宣称收益目标完成')
        for path, value in self.inputs.items():
            require(digest(path) == value, f'独立审计期间输入改变:{path}')
        return dict(passed=True, complete=True, research_version=VERSION, report_scope=self.cfg['report_scope'],
            goal_achieved=goal['goal_achieved'], candidate_accounts=candidates, outer_accounts=outer,
            checked_accounts=len(self.results), models=self.model_results, predictions=self.prediction_results,
            core_cohorts=self.cohort_results, frozen_source_count=48, original_input_count=len(original),
            all_read_inputs_unchanged=True, current_source_changes_after_run=changes, choices=choices,
            accounts=self.results, full_minute_caches=self.cache_audits, raw_event_checks=self.checked_raw_events,
            raw_exit_intent_checks=self.checked_exit_pairs, market=self.market_summary,
            no_model_training_or_parameter_selection_performed=True,
            historical_retest_not_untouched_test=True,
            scope_limitations=['旧基础入场成交独立核价/费/现金/来源；未重新撮合所有未成交基础请求。',
                '复用此前独立动态账簿审计；不重新计算每个未成交订单的具体拒绝理由。',
                '两core交互OOF为联合训练机会，不声称二者能再互为独立校准集。'])


def self_tests():
    """最小反例覆盖隔离边界、原始收益特征、方向/成本/期限和未来字段无效。"""
    import unittest
    class Tests(unittest.TestCase):
        def test_clock(self):
            values = pd.Series(pd.to_datetime(['2026-01-06 11:00', '2026-01-06 14:00']))
            actual = recovery_times(values, 60)
            self.assertEqual(actual.iloc[0], pd.Timestamp('2026-01-06 13:31'))
            self.assertTrue(pd.isna(actual.iloc[1]))
        def test_raw_and_source_boundary(self):
            p = pd.DataFrame({'datetime': pd.to_datetime(['2024-01-10 10:00', '2024-02-12 10:00']),
                'symbol': ['a', 'b'], 'feature_source_start': pd.to_datetime(['2023-12-01', '2024-02-01']),
                'label_start': pd.to_datetime(['2024-01-10 10:01', '2024-02-12 10:01']),
                'label_end': pd.to_datetime(['2024-01-10 10:21', '2024-02-12 10:21']), 'target_return': [.01, .02]})
            kept = independent_training_rows(p, [2024, 2025], [2023, 2026], {'embargo_days': 7, 'exclude_flat_signal_bars': False})
            self.assertEqual(list(kept.symbol), ['b'])
            self.assertEqual(kept.target_return.iloc[0], .02)
        def test_decisions_current_only(self):
            cfg = dict(commission_rate=.0008, transfer_rate=.00001, slippage_rate=.0005,
                stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005,
                t_avoid_locked_exit_side=True, flat_st_limit_rate=.05, nonflat_limit_proxy=.098,
                last_entry_time='14:30', fixed_recovery_time='14:55')
            p = pd.DataFrame({'datetime': pd.to_datetime(['2026-01-06 10:00']*2), 'symbol': ['a', 'b'],
                'prediction': [.01, .0001], 'model_profile': ['l2_depth4']*2, 'signal_price': [10., 10.],
                'previous_close': [10., 10.], 'signal_high': [10.1]*2, 'signal_low': [9.9]*2,
                'signal_amount': [1e6]*2, 'signal_volume': [10000.]*2})
            predictions = {h: p.assign(horizon_minutes=h) for h in [20, 40, 60]}
            original = {h: x.copy(deep=True) for h, x in predictions.items()}
            result = decisions(predictions, rules()[1], cfg)
            self.assertEqual(list(result.symbol), ['a'])
            self.assertEqual(int(result.horizon_minutes.iloc[0]), 20)
            dirty = {h: x.assign(target_return=999., execution_price=np.nan, recovery_price=-99.) for h, x in predictions.items()}
            exact(result, decisions(dirty, rules()[1], cfg), '未来字段不得影响决策')
            for h in predictions:
                exact(predictions[h], original[h], '输入副本')
        def test_grid(self):
            self.assertEqual(len(rules()), 37)
            self.assertEqual(len({json.dumps(r) for r in rules()}), 37)
        def test_donor_csv_column_order(self):
            from io import StringIO
            text = 'symbol,score,trade_week\na,0.7,2024-02-12\n'
            original = pd.read_csv(StringIO(text))
            restored = original[[c for c in original if c in WEEK_KEYS]]
            expected = pd.read_csv(StringIO(text), usecols=WEEK_KEYS)
            exact(restored, expected, 'usecols原文件列序')
        def test_stamp_change_uses_actual_date(self):
            cfg = dict(commission_rate=.0008, transfer_rate=.00001, slippage_rate=.0005,
                stamp_rate_before_20230828=.001, stamp_rate_from_20230828=.0005,
                t_avoid_locked_exit_side=False, last_entry_time='14:30', fixed_recovery_time='14:55')
            p = pd.DataFrame({'datetime': pd.to_datetime(['2023-08-25 10:00', '2023-08-28 10:00']),
                'symbol': ['a', 'a'], 'prediction': [.0034, .0034], 'model_profile': ['l2_depth4']*2,
                'signal_amount': [1e6]*2, 'signal_volume': [10000.]*2})
            result = decisions({h: p.assign(horizon_minutes=h) for h in [20, 40, 60]}, rules()[1], cfg)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.datetime.iloc[0], pd.Timestamp('2023-08-28 10:00'))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), '独立审计反例未通过')


def main():
    """成功/失败均归档自身源码和实际输入身份，默认只追加新审计目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_tests()
        return
    if not args.report_dir:
        parser.error('需要--report-dir')
    report = Path(args.report_dir).resolve()
    out = report/'independent_learning_audit'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/Path(__file__).name)
    dependencies = {}
    dependency_directory = out/'audit_dependencies'
    dependency_directory.mkdir()
    for name in ['audit_dynamic_inventory_t.py', 'audit_weekly_t_short_horizon.py',
                 'audit_weekly_t_timing_models.py', 'audit_weekly_t_residual.py']:
        path = ROOT/'scripts'/name
        dependencies[str(path)] = digest(path)
        shutil.copy2(path, dependency_directory/name)
    auditor, result, started = None, {}, time.monotonic()
    try:
        auditor = LearningAuditor(report, out)
        result = auditor.run()
    except Exception as error:
        result = dict(passed=False, complete=False, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(script_sha256=digest(__file__), elapsed_seconds=time.monotonic()-started,
                      inputs_sha256=auditor.inputs if auditor is not None else {}, archive=str(out),
                      independent_audit_dependencies_sha256=dependencies,
                      independent_audit_dependencies_unchanged=all(digest(p) == v for p, v in dependencies.items()))
        if not result['independent_audit_dependencies_unchanged']:
            result.update(passed=False, complete=False, error='审计自身依赖在运行期间变化')
        (out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        (out/'review.md').write_text('独立 v14 审计：'+('通过' if result.get('passed') else '失败，详见 verification.json')+'。\n\n'
            '核验纯核心实际来源/原始标签/拟合矩阵、原生权重和当前信号；账户复用独立原分钟/旧股/费用/动态恢复核算。'
            '未训练、未调参；历史复测不等于全新测试。\n', encoding='utf-8')
        print(json.dumps({k: result[k] for k in ['passed', 'complete', 'archive', 'elapsed_seconds', 'script_sha256']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
