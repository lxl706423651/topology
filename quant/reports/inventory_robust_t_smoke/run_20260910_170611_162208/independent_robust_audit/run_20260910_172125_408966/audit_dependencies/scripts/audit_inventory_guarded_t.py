"""独立审计v15：复用已审计模型/决策，复算同刻guard与真实账户。

主运行成功终态后使用--report-dir；仅新增independent_guarded_audit/run_*，
不改任何原产物，不导入guarded执行器/生产策略，不拟合或重跑策略账户。
--self-test执行合成反例。旧独立动态审计只复用原始行情及账簿重放方法。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dynamic_inventory_t import (Auditor as AccountAuditor, FAILURE_KEYS,
    close, digest, next_minute, parse_time, require, truth)
from scripts.audit_inventory_learning_t import rules, decisions, exact

VERSION = 'inventory_guarded_t_v15'
KEYS = ['datetime', 'symbol']
GUARD_COLUMNS = ['datetime', 'signal_datetime', 'symbol', 'side', 'opposite_side',
    'requested_shares', 'own_fillable_shares', 'guard_enabled', 'opposite_checked', 'cancelled',
    'reason', 'opposite_quote_reason', 'reference_price', 'high', 'low', 'amount', 'volume',
    'previous_close', 'limit_up', 'limit_down', 'minute_amount_used_before', 'cash_before', 'old_sellable_before']


def limit_reason(quote, side, cfg):
    """独立按当前quote判方向限价，真实限价优先；不询问容量或未来恢复。"""
    require(side in ('buy', 'sell'), '非法quote方向')
    price, high, low, amount = [float(quote[k]) for k in ['price', 'high', 'low', 'amount']]
    if not all(np.isfinite([price, high, low, amount])): return 'missing_quote'
    volume = float(quote.get('volume', np.nan))
    if price <= 0 or amount <= 0 or (np.isfinite(volume) and volume <= 0): return 'zero_price_or_volume'
    if high < low or price < low-1e-6 or price > high+1e-6: return 'invalid_ohlc'
    direction = 1 if side == 'buy' else -1
    bound = float(quote.get('limit_up' if side == 'buy' else 'limit_down', np.nan))
    if np.isfinite(bound) and bound > 0:
        return 'price_limit' if (price-bound)*direction >= -1e-8 else ''
    previous = float(quote.get('previous_close', np.nan))
    if np.isfinite(previous) and previous > 0:
        flat_bound = np.floor(previous*(1+direction*cfg['flat_st_limit_rate'])*100+.5+1e-8)/100
        if (price/previous-1)*direction >= cfg['nonflat_limit_proxy'] or (high == low and abs(price-flat_bound) <= 1e-6):
            return 'conservative_price_limit_proxy'
    return ''


def own_fillable(quote, side, requested, used, cfg):
    """仅计算本方向quote及真实剩余分钟容量，不扣虚拟反向量或现金。"""
    if limit_reason(quote, side, cfg): return 0
    lot = cfg['lot_size']
    capacity = int(max(quote['amount']*cfg['max_bar_amount_fraction']-used, 0.)/quote['price']/lot)*lot
    return min(int(requested), capacity)


def verify_guard_row(row, quote, cash, old, used, requested, cfg):
    """当前独立账簿快照与guard逐字段相等；返回是否取消。"""
    require(truth(row.guard_enabled) and truth(row.opposite_checked), 'v15存在未开启的复查行')
    require(row.datetime == next_minute(row.signal_datetime), '复查不是下一交易分钟')
    opposite = 'sell' if row.side == 'buy' else 'buy'
    require(row.opposite_side == opposite, '反向错误')
    close(float(row.cash_before), cash, 'guard前现金')
    close(float(row.old_sellable_before), old, 'guard前旧股', 0)
    close(float(row.minute_amount_used_before), used, 'guard前raw容量')
    close(float(row.requested_shares), requested, 'guard原请求数量', 0)
    quantity = own_fillable(quote, row.side, requested, used, cfg)
    require(quantity > 0, '本方向无可填股数却有guard')
    close(float(row.own_fillable_shares), quantity, 'guard本方向可填股数', 0)
    for field, key in [('reference_price', 'price'), *[(k, k) for k in
            ('high', 'low', 'amount', 'volume', 'previous_close', 'limit_up', 'limit_down')]]:
        expected, actual = quote[key], getattr(row, field)
        if pd.isna(expected): require(pd.isna(actual), f'guard伪造未知{field}')
        else: close(float(actual), float(expected), f'guard quote {field}', 1e-9)
    reason = limit_reason(quote, opposite, cfg)
    require(reason in ('', 'price_limit', 'conservative_price_limit_proxy'), '同quote反向非限价异常')
    cancelled = bool(reason)
    require(truth(row.cancelled) == cancelled, '取消标志与原quote独立复算不符')
    require(('' if pd.isna(row.opposite_quote_reason) else row.opposite_quote_reason) == reason, '反向quote原因错误')
    require(('' if pd.isna(row.reason) else row.reason) == ('execution_exit_side_'+reason if reason else ''), '取消原因错误')
    return cancelled


class GuardedAuditor(AccountAuditor):
    """独立v15身份，不调用旧v13入口或生产引擎；共享只读账簿方法。"""
    def __init__(self, report, out):
        self.report, self.out = report, out
        self.inputs, self.raw, self.full_raw, self.raw_files, self.results = {}, {}, {}, {}, []
        self.cfg = self.json(report/'resolved_config.json')
        self.protocol = self.json(report/'research_protocol.json')
        require(self.cfg['research_version'] == VERSION and self.cfg['t_recheck_exit_side_at_execution'] is True, '版本/复查合同错误')
        require(self.protocol['valuation_contract'] == 'full_minute_reference_stream_then_daily_close', '未知估值合同')
        self.checked_raw_events = self.checked_exit_pairs = 0
        self.cache_audits, self.sparse_arrays, self.mark_cache, self.exit_cache, self.base_anchors = [], {}, {}, {}, {}
        self.valuation_mode = 'full_minute_reference_stream_then_daily_close'
        market = pd.read_pickle(self.record(Path(self.cfg['market_cache_dir'])/'daily.pkl'))
        self.market_calendar = pd.DatetimeIndex(parse_time(market.date).unique()).normalize().sort_values()
        self.source = Path(self.cfg['source_learning_report_dir'])
        self.guards, self.predictions_checked, self.baselines_checked = [], [], []
        self.native, self.feature_dtypes = {}, {}

    def table(self, path):
        """仅对明确时间字段解析，保留其余浮点往返精度。"""
        frame = self.csv(path)
        for key in ('datetime', 'signal_datetime', 'planned_recovery_time', 'feature_source_start'):
            if key in frame: frame[key] = parse_time(frame[key])
        return frame

    def sources(self):
        """逐字节绑定51运行源、原输入与已通过审计的v14冻结模型，不重做训练。"""
        complete = self.json(self.report/'runtime_integrity_audit.json')
        require(complete.get('all_unchanged') is True and (self.report/'final_report.md').exists(), '主运行未成功结束')
        self.record(self.report/'final_report.md')
        hashes = self.json(self.report/'source_hashes.json')
        require(len(hashes) == 51, 'v15应冻结48旧源加3新源')
        for name, value in hashes.items():
            require(digest(self.record(ROOT/name)) == value, f'当前源变更:{name}')
            require(digest(self.record(self.report/'source_snapshot'/name)) == value, f'快照源错误:{name}')
        original_inputs = self.json(self.report/'input_sources.json')
        for name, identity in original_inputs.items():
            require(digest(self.record(name)) == identity['sha256'], f'登记输入已改变:{name}')
        original = self.json(self.source/'resolved_config.json')
        require(original['research_version'] == 'inventory_learning_t_v14', '来源非v14')
        require(self.json(self.source/'runtime_integrity_audit.json')['all_unchanged'] is True, '来源未完整结束')
        allowed = {'research_version', 'configuration_source', 'output_dir', 'workers', 'candidate_workers', 'test_years', 'internal_only'}
        require(all(self.cfg[k] == v for k, v in original.items() if k not in allowed), '原模型/金额/费用/范围合同改变')
        require(set(self.cfg['test_years']).issubset(original['test_years']), '扩张来源年份')
        audits = []
        for path in sorted(self.source.glob('independent_learning_audit/run_*/verification.json')):
            data = self.json(path)
            if data.get('passed') and data.get('complete') and data.get('all_read_inputs_unchanged'):
                audits.append((path, data))
        require(audits, '缺完整独立来源审计')
        audit_path, audit = audits[-1]
        require(len(self.json(self.source/'source_hashes.json')) == 48, '来源不是原48源')
        frozen = self.json(self.source/'all_rules_frozen_before_outer_test.json')
        reused = self.json(self.report/'reused_model_artifacts_sha256.json')
        expected = {str((self.source/k).resolve()): v for k, v in frozen['model_artifacts_sha256'].items()}
        require(reused == expected and len(expected) == 48*len(original['test_years']), '复用模型产物集合改变')
        for path, value in expected.items():
            require(digest(self.record(path)) == value, '复用模型字节改变')
            require(audit['inputs_sha256'].get(path) == value, '模型产物未绑定完整来源审计')
        for filename in ('resolved_config.json', 'source_hashes.json', 'all_rules_frozen_before_outer_test.json'):
            path = self.source/filename
            require(audit['inputs_sha256'].get(str(path.resolve())) == digest(path), '来源关键文件偏离独立审计')
        require(self.protocol['candidate_rules'] == rules() and self.protocol['models_fitted_this_run'] == 0, '候选或训练声明改变')
        self.source_summary = {'independent_source_audit': str(audit_path), 'model_artifacts_verified': len(expected),
            'models_reused': 6*len(self.cfg['test_years']), 'models_retrained': 0,
            'source_report_scope': original['report_scope'], 'current_runtime_sources': len(hashes),
            'registered_inputs_verified': len(original_inputs), 'model_training_audit_reused_not_repeated': True}

    def predictions(self, year, scope, profiles):
        """当前CSV精确比对来源；外层原生权重重算，输入用来源同期限表核查。"""
        result = {}
        for profile in profiles:
            for h in (20, 40, 60):
                relative = Path(f'year_{year}/{scope}_predictions/h{h}_{profile}/current_predictions.csv')
                saved = self.table(self.report/relative)
                own = self.json((self.report/relative).parent/'prediction_audit.json')
                model = self.source/f'year_{year}/models/h{h}_{profile}'
                meta = self.json(model/'model_metadata.json')
                require(own['predictions_sha256'] == digest(self.report/relative), '预测CSV摘要错误')
                require(own['model_bundle_sha256'] == meta['model_bundle_sha256'] and own['future_independent'] is True, '预测使用错误模型或无未来隔离记录')
                source_path = self.source/relative
                if scope == 'validation':
                    exact(saved, self.table(source_path), '内部当前预测与v14逐字段精确相等')
                else:
                    # 工程来源外层可能仅保存另一profile；同期限因果输入仍相同。
                    paths = sorted((self.source/f'year_{year}/test_predictions').glob(f'h{h}_*/current_predictions.csv'))
                    require(paths, '来源未有同期限外层输入；需先增加独立市场特征重建，不能省略输入证明')
                    original = self.table(paths[0])
                    fields = [c for c in saved if c not in ('prediction', 'raw_prediction', 'model_profile', 'horizon_minutes')]
                    exact(saved[fields], original[fields], '外层当前因果输入与已审计来源相等')
                    key = str(model)
                    if key not in self.native:
                        self.native[key] = lightgbm.Booster(model_file=str(self.record(model/'return_model.txt')))
                        train = pd.read_pickle(self.record(model/'training_inputs.pkl'))
                        self.feature_dtypes[key] = train[meta['features']].dtypes.to_dict()
                    matrix = saved[meta['features']].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
                    matrix = matrix.astype(self.feature_dtypes[key])
                    expected = self.native[key].predict(matrix, num_threads=1) if len(matrix) else np.array([])
                    require(np.array_equal(expected, saved.prediction) and np.array_equal(expected, saved.raw_prediction), '外层原生冻结权重未精确复现')
                self.predictions_checked.append({'year': year, 'scope': scope, 'profile': profile, 'horizon': h,
                    'rows': len(saved), 'source_inputs_exact': True, 'source_prediction_exact': scope == 'validation',
                    'native_predictions_exact': scope == 'test'})
                result.setdefault(profile, {})[h] = saved
        return result

    def source_baseline(self, year, scope):
        """同源底仓计划、实际基础成交和每日基准完整相等，不只核期末股数。"""
        relative = Path(f'year_{year}/{scope}_context')
        names = ['frozen_base_selections.csv', 'frozen_base_entries.csv', 'baseline/test_equity_curve.csv',
                 'baseline/baseline_trade_events.csv', 'planning_baseline/baseline_trade_events.csv']
        for filename in names:
            exact(self.csv(self.report/relative/filename), self.csv(self.source/relative/filename), '同源底仓与基准精确相同')
        self.baselines_checked.append({'year': year, 'scope': scope, 'tables_exact': names})

    def guard_account(self, folder, predictions, rule):
        """独立按全部请求顺序重建guard前现金/旧股/used，取消不产生任何虚拟成交。"""
        audit = self.table(folder/'first_leg_exit_side_audit.csv')
        require(list(audit.columns) == GUARD_COLUMNS, 'guard表字段合同改变')
        require(not audit.duplicated(['datetime', 'symbol']).any(), 'guard请求重复')
        event = self.table(folder/'test_trade_events.csv')
        attempt = self.table(folder/'blocked_orders.csv')
        dec = self.table(folder/'t_decisions.csv')
        forecast = predictions.get(rule['profile'], {}).get(20, pd.DataFrame())
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

    def run(self):
        """完成全部内部候选/存在的工程外层核账，并核选参、底仓与终态身份。"""
        self.sources()
        frozen = self.json(self.report/'all_rules_frozen_before_outer_test.json')
        evaluation = self.json(self.report/'outer_evaluation_decision.json')
        marker = (self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns
        choices, count, outer = [], 0, 0
        for year in self.cfg['test_years']:
            folder = self.report/f'year_{year}'
            fold = self.json(folder/'split.json')
            previous = self.json(self.source/f'year_{year}/split.json')
            for k in ('test_year', 'inner_year', 'core_years', 'train_years'): require(fold[k] == previous[k], '来源年度隔离改变')
            selected = self.json(folder/'selected_rule.json')
            require(digest(folder/'selected_rule.json') == frozen['selected_rule_sha256'][str(year)], '选择规则偏离全折冻结')
            require((folder/'selected_rule.json').stat().st_mtime_ns <= marker, '统一冻结后改规则')
            self.audit_context(folder/'validation_context', fold['inner_year'])
            self.source_baseline(year, 'validation')
            forecasts = self.predictions(year, 'validation', ['l2_depth4', 'l2_depth6'])
            table = self.csv(folder/'validation_rule_metrics.csv')
            original = self.csv(self.source/f'year_{year}/validation_rule_metrics.csv')
            require(len(table) == 37 and set(table.candidate_index) == set(range(37)), '37内部候选不齐')
            accounts = []
            for rule in rules():
                index = rule['candidate_index']
                target = folder/f'validation_candidates/candidate_{index:03d}'
                require(self.json(target/'rule.json') == rule, '候选规则改变')
                exact(self.table(target/'t_decisions.csv'), self.table(self.source/f'year_{year}/validation_candidates/candidate_{index:03d}/t_decisions.csv'), '37首腿决策逐字段同源')
                self.guard_account(target, forecasts, rule)
                r = self.account(target)
                row = table.loc[table.candidate_index.eq(index)].iloc[0]
                old = original.loc[original.candidate_index.eq(index)].iloc[0]
                for k in ('incremental_net_pnl', 't_pair_count', 'all_reference_acquisition_cost', *FAILURE_KEYS): close(row[k], r[k], f'内部指标{k}')
                close(row.all_reference_acquisition_cost, old.all_reference_acquisition_cost, '同源底仓原始成本', 1e-8)
                accounts.append({'candidate_index': index, **r}); count += 1
            eligible = [r for r in accounts if r['candidate_index'] and r['t_pair_count'] > 0 and all(r[k] == 0 for k in FAILURE_KEYS)]
            winner = min(eligible, key=lambda r: (-r['incremental_net_pnl'], r['candidate_index'])) if eligible else accounts[0]
            require(selected['rule'] == rules()[winner['candidate_index']], '内部选参未遵守执行合格后最大净额')
            choices.append({'test_year': year, 'candidate_index': winner['candidate_index'],
                            'inner_net': winner['incremental_net_pnl'], 'cost_rate': selected['inner_metrics']['holding_cost_reduction_rate']})
            for name in ('planning_baseline', 'baseline'): self.account(folder/'validation_context'/name)
            if evaluation['outer_evaluation_started']:
                self.audit_context(folder/'test_context', year)
                self.source_baseline(year, 'test')
                rule = selected['rule']
                forecast = self.predictions(year, 'test', [rule['profile']]) if rule['enabled'] else {}
                expected = decisions(forecast.get(rule['profile'], {}), rule, self.cfg)
                saved = self.table(folder/'t_decisions.csv')
                if len(expected): exact(saved[list(expected)], expected, '外层原冻结模型策略决策')
                else: require(saved.empty, 'hold外层有请求')
                require((folder/'test_equity_curve.csv').stat().st_mtime_ns >= marker, '外层早于统一冻结')
                self.guard_account(folder, forecast, rule)
                self.account(folder)
                for name in ('planning_baseline', 'baseline'): self.account(folder/'test_context'/name)
                outer += 1
            print(f'fold={year}：37内部、guard当前状态与所有真实账户核验通过', flush=True)
        passed = all(x['candidate_index'] > 0 and x['cost_rate'] >= self.cfg['profit_target_rate'] for x in choices)
        require(evaluation['all_inner_cost_targets_met'] == passed, '内部5%判断错误')
        require(evaluation['outer_evaluation_started'] == (not self.cfg['internal_only'] and (passed or self.cfg['report_scope'] == 'smoke')), '外层启停错误')
        goal = self.json(self.report/'goal_evaluation.json')
        if self.cfg['report_scope'] == 'smoke': require(not goal['goal_achieved'], '烟测不能达标')
        for path, value in self.inputs.items(): require(digest(path) == value, f'审计期间输入改变:{path}')
        return {'passed': True, 'complete': True, 'research_version': VERSION, 'report_scope': self.cfg['report_scope'],
            'source': self.source_summary, 'candidate_accounts': count, 'outer_accounts': outer,
            'checked_accounts': len(self.results), 'accounts': self.results, 'choices': choices,
            'guard_accounts': self.guards, 'guard_rows_verified': sum(x['guard_rows'] for x in self.guards),
            'cancellations_verified': sum(x['cancelled'] for x in self.guards),
            'predictions': self.predictions_checked, 'source_baselines': self.baselines_checked,
            'full_minute_caches': self.cache_audits, 'raw_event_checks': self.checked_raw_events,
            'raw_exit_intent_checks': self.checked_exit_pairs, 'raw_files': self.raw_files,
            'inputs_sha256': self.inputs, 'all_read_inputs_unchanged': True,
            'no_production_guarded_engine_or_policy_imported': True, 'no_model_fit_or_account_rerun': True,
            'goal_achieved': goal['goal_achieved'],
            'limitations': ['来源v14训练隔离复用其完整独立审计，本次逐字节核模型而不重新训练。',
                '原始CSV缺真实限价字段；本运行输入合同不携带limit列，实际复查核原代理；真实limit优先由合成反例验证。',
                '共享旧独立账户核价费/容量/T1/退出时刻；未重撮合所有原底仓未成交请求或复算每个非guard拒单原因。',
                '外层当前特征须有已审计来源同期限输入可比；来源未执行外层时不能省略特征来源证明。',
                '合法恢复失败如实保留，审计通过不代表每天恢复或5%盈利。历史复测不是全新验证。']}


def self_tests():
    """纯quote与快照反例；不接触行情缓存或运行账户。"""
    import unittest
    from types import SimpleNamespace
    cfg = {'flat_st_limit_rate': .05, 'nonflat_limit_proxy': .098, 'lot_size': 100, 'max_bar_amount_fraction': .05}
    base = {'price': 10., 'high': 10., 'low': 10., 'amount': 100000., 'volume': 10000.,
            'previous_close': 10., 'limit_up': np.nan, 'limit_down': np.nan}
    def row_for(q):
        return SimpleNamespace(datetime=pd.Timestamp('2026-01-05 10:01'), signal_datetime=pd.Timestamp('2026-01-05 10:00'),
            side='buy', opposite_side='sell', guard_enabled=True, opposite_checked=True, requested_shares=400,
            own_fillable_shares=400, cash_before=20000., old_sellable_before=800, minute_amount_used_before=0.,
            reference_price=q['price'], **{k:q[k] for k in ('high','low','amount','volume','previous_close','limit_up','limit_down')},
            cancelled=False, reason='', opposite_quote_reason='')
    class Tests(unittest.TestCase):
        def test_actual_limit_priority(self):
            self.assertEqual(limit_reason({**base, 'price': 11., 'high': 11., 'low': 11., 'limit_up': 12.}, 'buy', cfg), '')
            self.assertEqual(limit_reason({**base, 'limit_up': 10.}, 'buy', cfg), 'price_limit')
        def test_flat_five_and_normal_six(self):
            self.assertEqual(limit_reason({**base, 'price':10.5,'high':10.5,'low':10.5}, 'buy', cfg), 'conservative_price_limit_proxy')
            self.assertEqual(limit_reason({**base, 'price':10.6,'high':10.6,'low':10.6}, 'buy', cfg), '')
        def test_nonflat_limit_and_reverse(self):
            q={**base,'price':9.,'high':9.2,'low':9.}
            self.assertEqual(limit_reason(q,'sell',cfg), 'conservative_price_limit_proxy')
            self.assertEqual(limit_reason(q,'buy',cfg), '')
        def test_invalid_zero_and_capacity(self):
            self.assertEqual(limit_reason({**base,'volume':0.},'buy',cfg), 'zero_price_or_volume')
            self.assertEqual(limit_reason({**base,'low':11.},'buy',cfg), 'invalid_ohlc')
            self.assertEqual(own_fillable(base,'buy',800,2000.,cfg),300)
        def test_guard_no_virtual_reverse_capacity(self):
            row=row_for(base)
            self.assertFalse(verify_guard_row(row,base,20000.,800,0.,400,cfg))
        def test_state_tampering(self):
            for name in ['cash_before','old_sellable_before','minute_amount_used_before']:
                row=row_for(base);setattr(row,name,getattr(row,name)-1)
                with self.subTest(name=name),self.assertRaises(AssertionError): verify_guard_row(row,base,20000.,800,0.,400,cfg)
        def test_wrong_cancel_and_timing(self):
            row=row_for(base);row.cancelled=True
            with self.assertRaises(AssertionError):verify_guard_row(row,base,20000.,800,0.,400,cfg)
            row=row_for(base);row.datetime=row.signal_datetime
            with self.assertRaises(AssertionError):verify_guard_row(row,base,20000.,800,0.,400,cfg)
        def test_actual_locked_reverse_cancellation(self):
            q={**base,'price':9.,'high':9.,'low':9.}
            row=row_for(q);row.cancelled=True;row.opposite_quote_reason='conservative_price_limit_proxy'
            row.reason='execution_exit_side_conservative_price_limit_proxy'
            self.assertTrue(verify_guard_row(row,q,20000.,800,0.,400,cfg))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(result.wasSuccessful(), '独立审计自测失败')
    return {'passed':True,'tests':result.testsRun}


def main():
    """成功/失败都归档独立脚本、依赖和输入SHA；不覆盖任何原审计。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(),ensure_ascii=False));return
    require(args.report_dir, '须指定--report-dir')
    report=(ROOT/args.report_dir).resolve()
    require((report/'runtime_integrity_audit.json').exists(), '主运行尚未终态，不开始审计')
    out=report/'independent_guarded_audit'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/Path(__file__).name)
    dependencies={}
    (out/'audit_dependencies').mkdir()
    for name in ['audit_dynamic_inventory_t.py','audit_inventory_learning_t.py','audit_weekly_t_short_horizon.py',
                 'audit_weekly_t_timing_models.py','audit_weekly_t_residual.py']:
        path=ROOT/'scripts'/name;dependencies[str(path)]=digest(path)
        shutil.copy2(path,out/'audit_dependencies'/name)
    auditor=None;start=time.time()
    try:
        checks=self_tests()
        auditor=GuardedAuditor(report,out)
        result=auditor.run()
        result.update(elapsed_seconds=time.time()-start,self_tests=checks,script_sha256=digest(__file__),
                      audit_dependencies_sha256=dependencies,
                      audit_dependencies_unchanged=all(digest(p)==v for p,v in dependencies.items()))
        require(result['audit_dependencies_unchanged'],'独立helper运行期间改变')
    except Exception:
        result={'passed':False,'complete':False,'error':traceback.format_exc(),'elapsed_seconds':time.time()-start,
                'inputs_sha256':auditor.inputs if auditor else {},'script_sha256':digest(__file__),
                'audit_dependencies_sha256':dependencies}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
    rows=['# v15独立复查与账户验收', '', f"结果：{'通过' if result['passed'] else '失败（保留原始错误）'}。", '',
          '本审计不修改生产源、原报告、模型或候选。来源模型训练证明复用v14独立审计，实际guard和账户逐行另核。', '']
    if result['passed']:
        rows += [f"核对{result['candidate_accounts']}内部候选、{result['outer_accounts']}外层账户及基准，共{result['checked_accounts']}账户；"
                 f"{result['guard_rows_verified']}行复查、{result['cancellations_verified']}次取消，"
                 f"{result['raw_event_checks']}条真实成交、{result['raw_exit_intent_checks']}个首腿退出意图。",
                 '', '51运行源、全部登记输入、原6模型/折冻结产物及输入表身份通过；所有审计读取输入始末不变。',
                 '', '取消与实际首腿互斥；每个本方向可填请求的复查集合以及复查前现金、旧股和raw容量独立重建。'
                 '本方向可填不代表现金已批准，也不要求虚拟反向量能。', '',
                 '| 账户 | 完整组 | 配对手 | T净增量 | 股数差日 | 恢复失败日 | pending |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for a in result['accounts']:
            if '/validation_candidates/' in a['path'] or a['path'].endswith('baseline'):continue
            rows.append(f"| {a['path']} | {a['t_pair_count']} | {a['completed_t_pair_lots']:g} | {a['incremental_net_pnl']:.6f} | {a['inventory_mismatch_days']} | {a['recovery_failure_days']} | {a['pending_pairs']} |")
        rows += ['', *['- '+x for x in result['limitations']], '', '完整机器记录及输入SHA见[verification.json](verification.json)，逐请求guard复算另存本目录CSV。']
    else: rows += ['```text',result['error'],'```']
    (out/'acceptance.md').write_text('\n'.join(rows)+'\n')
    print(out,flush=True)
    if not result['passed']:raise AssertionError(result['error'])


if __name__=='__main__':main()
