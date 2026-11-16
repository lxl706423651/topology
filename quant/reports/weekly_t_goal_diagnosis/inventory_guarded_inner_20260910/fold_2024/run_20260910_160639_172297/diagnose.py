"""只读v15已完成内部折，同候选对照v14；不读外层、不重训或修改规则。

全部首腿按原数量匹配，单边及未完整组保留。取消记录仅表示到达报价反向受限，
本方向quote可填尚不代表现金已批准，更不能保证未来恢复。不同候选增量不可相加。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
KEY = ['symbol', 'first_datetime', 'side', 'shares']
FAIL = ['inventory_mismatch_days', 'recovery_failure_days', 'pending_pairs', 't1_violation_count', 'base_plan_mismatch_count']
VALUES = ['incremental_net_pnl', 'holding_cost_reduction_rate', 't_pair_count', 'completed_t_pair_lots',
          't_fee', 't_slippage_cost', 'inventory_mismatch_days', 'recovery_failure_days', 'pending_pairs',
          'all_reference_acquisition_cost']
TABLES = ['candidate_comparison', 'pair_comparison', 'cancelled_current_quotes', 'cancel_summary', 'cancel_board_summary']


def sha(path):
    """绑定实际读取字节，缓存不依赖文件名或stat。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    """缺失为null，不能把未完成组伪装成零收益。"""
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    return value


def limit_reason(row, side, cfg):
    """独立方向限价检查，仅消费到达quote；不模拟反向容量或成交。"""
    raw, prior = float(row['reference_price']), float(row['previous_close'])
    direction = 1 if side == 'buy' else -1
    limit = row['limit_up'] if side == 'buy' else row['limit_down']
    if np.isfinite(limit) and limit > 0:
        return 'price_limit' if (raw-limit)*direction >= -1e-8 else ''
    if np.isfinite(prior) and prior > 0:
        st = np.floor(prior*(1+direction*cfg['flat_st_limit_rate'])*100+.5+1e-8)/100
        exact_flat_st = row['high'] == row['low'] and abs(raw-st) <= 1e-6
        if (raw/prior-1)*direction >= cfg['nonflat_limit_proxy'] or exact_flat_st:
            return 'conservative_price_limit_proxy'
    return ''


def pairs(events, trips):
    """所有实际首腿保留，完整组与未完整已实现片段分列。"""
    first = events.loc[events.event_type.eq('t') & events.fragment_id.eq(0)].copy()
    first = first.rename(columns={'datetime': 'first_datetime'})
    fields = ['pair_id', *KEY, 'signal_datetime', 'reference_price', 'fee']
    first = first[fields]
    first['first_datetime'] = pd.to_datetime(first.first_datetime, format='mixed')
    first['signal_datetime'] = pd.to_datetime(first.signal_datetime, format='mixed')
    assert not first.duplicated(KEY).any() and not first.pair_id.duplicated().any()
    if len(trips):
        grouped = trips.groupby('pair_id').agg(recovered_shares=('shares', 'sum'), fragments=('shares', 'size'),
            fragment_net=('net_pnl', 'sum'), slipped_gross=('gross_pnl', 'sum'), slippage=('slippage_cost', 'sum'))
        grouped['allocated_fee'] = trips.assign(allocated_fee=trips.buy_fee+trips.sell_fee).groupby('pair_id').allocated_fee.sum()
    else:
        grouped = pd.DataFrame(columns=['recovered_shares', 'fragments', 'fragment_net', 'slipped_gross', 'slippage', 'allocated_fee'])
    first = first.merge(grouped, left_on='pair_id', right_index=True, how='left', validate='one_to_one')
    for name in grouped: first[name] = first[name].fillna(0)
    assert first.recovered_shares.le(first.shares).all()
    first['complete'] = first.recovered_shares.eq(first.shares)
    first['complete_net'] = first.fragment_net.where(first.complete)
    first['reference_gross'] = first.slipped_gross+first.slippage
    return first


def pair_contrast(old, new):
    """同股/同刻/同方向/同数量完全匹配，不只保留共同获利组。"""
    joined = old.merge(new, on=KEY, how='outer', suffixes=('_source', '_guarded'), indicator=True, validate='one_to_one')
    joined['matching'] = joined['_merge'].map({'both': 'common_first', 'left_only': 'source_only_first', 'right_only': 'guarded_only_first'}).astype(str)
    both = joined._merge.eq('both')
    assert np.allclose(joined.loc[both, 'reference_price_source'], joined.loc[both, 'reference_price_guarded'], rtol=0, atol=1e-10)
    joined['both_complete'] = both & joined.complete_source.eq(True) & joined.complete_guarded.eq(True)
    joined['both_complete_net_change'] = (joined.complete_net_guarded-joined.complete_net_source).where(joined.both_complete)
    return joined.drop(columns='_merge')


class Diagnosis:
    """只允许本折内部文件、已审计v14对应内部文件及内部年原CSV。"""
    def __init__(self, report, year):
        self.report, self.year = report.resolve(), year
        self.inputs, self.raw = {}, {}
        self.cfg = json.loads((self.report/'resolved_config.json').read_text())
        self.source = Path(self.cfg['source_learning_report_dir']).resolve()
        self.record(self.report/'resolved_config.json')
        self.fold = self.read_json(self.report/f'year_{year}/split.json')
        self.inner = self.fold['inner_year']
        self.results = {name: [] for name in TABLES}

    def record(self, path):
        path = Path(path).resolve()
        if path.is_relative_to(self.report) or path.is_relative_to(self.source):
            base = self.report if path.is_relative_to(self.report) else self.source
            rel = path.relative_to(base)
            allowed = len(rel.parts) == 1 and rel.name in ['resolved_config.json', 'source_hashes.json', 'research_protocol.json']
            if rel.parts[0] == f'year_{self.year}':
                allowed |= len(rel.parts) == 2 and rel.name in ['split.json', 'selected_rule.json', 'validation_rule_metrics.csv', 'guarded_vs_source_candidates.csv']
                allowed |= len(rel.parts) > 2 and rel.parts[1] in ['validation_candidates', 'validation_predictions']
            assert allowed, f'禁止读取非本折内部文件:{path}'
        else:
            assert path.is_relative_to(Path(self.cfg['data_root'])/str(self.inner)), f'原行情不是内部年:{path}'
        self.inputs.setdefault(str(path), sha(path))
        return path

    def read_json(self, path): return json.loads(self.record(path).read_text())

    def read_csv(self, path):
        try: return pd.read_csv(self.record(path), float_precision='round_trip')
        except pd.errors.EmptyDataError: return pd.DataFrame()

    def raw_quotes(self, symbol):
        """逐股读取准确原分钟OHLCAV，只缓存取消证据所需股票，不前后填价。"""
        if symbol not in self.raw:
            path = self.record(Path(self.cfg['data_root'])/str(self.inner)/f'{symbol}.csv')
            frame = pd.read_csv(path, usecols=['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'])
            frame['datetime'] = pd.to_datetime(frame.datetime, format='mixed')
            minute = frame.datetime.dt.hour*60+frame.datetime.dt.minute
            frame = frame.loc[minute.isin(list(range(571,691))+list(range(781,901))) & frame.datetime.dt.year.eq(self.inner)]
            assert not frame.datetime.duplicated().any()
            self.raw[symbol] = frame.set_index('datetime')
        return self.raw[symbol]

    def guards(self, audit, previous):
        """全部guard行独立复算纯限价；取消行再逐价核原CSV，未来恢复从不参与。"""
        if not len(audit): return audit.copy()
        out = audit.copy()
        for name in ['datetime', 'signal_datetime']: out[name] = pd.to_datetime(out[name], format='mixed')
        assert out.datetime.dt.year.eq(self.inner).all() and out.guard_enabled.eq(True).all() and out.opposite_checked.eq(True).all()
        assert out.datetime.eq(out.signal_datetime+pd.Timedelta(minutes=1)).all()
        assert not out.duplicated(['datetime','symbol','side']).any()
        assert out.own_fillable_shares.gt(0).all() and out.own_fillable_shares.le(out.requested_shares).all()
        assert out.opposite_side.eq(np.where(out.side.eq('buy'), 'sell', 'buy')).all()
        assert not out.apply(lambda r: bool(limit_reason(r, r.side, self.cfg)), axis=1).any()
        expected = out.apply(lambda r: limit_reason(r, r.opposite_side, self.cfg), axis=1)
        assert expected.eq(out.opposite_quote_reason.fillna('')).all()
        assert out.cancelled.eq(expected.ne('')).all()
        assert out.reason.fillna('').eq(np.where(out.cancelled, 'execution_exit_side_'+expected, '')).all()
        cancelled = out.loc[out.cancelled].copy()
        if not len(cancelled): return cancelled
        known = cancelled.merge(previous, on=['signal_datetime','symbol'], how='left', validate='many_to_one', indicator=True)
        assert known._merge.eq('both').all()
        usable = np.isfinite(known.original_previous_close)
        assert np.allclose(known.loc[usable,'previous_close'], known.loc[usable,'original_previous_close'], rtol=0, atol=1e-10)
        known['previous_close_matches_current_input'] = usable
        known['board_prefix'] = np.select([known.symbol.str.startswith(p) for p in ['SH.688','SZ.300','SZ.301']],
                                          ['SH.688','SZ.300','SZ.301'], default='other')
        direction = np.where(known.opposite_side.eq('buy'), 1., -1.)
        known['opposite_signed_price_change'] = (known.reference_price/known.previous_close-1)*direction
        known['growth_prefix_9_8_proxy_below_19_8'] = (known.board_prefix.ne('other')
            & known.opposite_quote_reason.eq('conservative_price_limit_proxy')
            & known.opposite_signed_price_change.ge(self.cfg['nonflat_limit_proxy'])
            & known.opposite_signed_price_change.lt(.198))
        known['raw_quote_verified'] = False
        for symbol, part in known.groupby('symbol', sort=False):
            raw = self.raw_quotes(symbol).reindex(pd.DatetimeIndex(part.datetime))
            for saved, original in [('reference_price','close'),('high','high'),('low','low'),('amount','amount'),('volume','volume')]:
                assert np.allclose(part[saved].to_numpy(), raw[original].to_numpy(), rtol=0, atol=1e-10, equal_nan=True), (symbol,saved)
            known.loc[part.index, 'raw_quote_verified'] = True
        return known.drop(columns='_merge')

    def run(self):
        """只读齐备的37候选；比较相同规则和全部首腿，原选择保持不动。"""
        folder, source = self.report/f'year_{self.year}', self.source/f'year_{self.year}'
        current, original = self.read_csv(folder/'validation_rule_metrics.csv'), self.read_csv(source/'validation_rule_metrics.csv')
        comparison = self.read_csv(folder/'guarded_vs_source_candidates.csv')
        selected = self.read_json(folder/'selected_rule.json')
        old_selected = self.read_json(source/'selected_rule.json')
        assert len(current)==len(original)==len(comparison)==37
        assert list(current.candidate_index)==list(original.candidate_index)==list(range(37))
        rule_fields=['candidate_index','enabled','profile','min_net_edge','fraction','exit_mode']
        pd.testing.assert_frame_equal(current[rule_fields],original[rule_fields],check_dtype=False,check_exact=True)
        for name in VALUES:
            assert np.allclose(comparison[name+'_source'],original[name],rtol=0,atol=1e-8)
            assert np.allclose(comparison[name+'_guarded'],current[name],rtol=0,atol=1e-8)
            assert np.allclose(comparison[name+'_change'],current[name]-original[name],rtol=0,atol=1e-8)
        assert comparison.all_reference_acquisition_cost_change.abs().lt(1e-8).all()
        previous=self.read_csv(folder/'validation_predictions/h20_l2_depth4/current_predictions.csv')[['datetime','symbol','previous_close']]
        previous=previous.rename(columns={'datetime':'signal_datetime','previous_close':'original_previous_close'})
        previous.signal_datetime=pd.to_datetime(previous.signal_datetime,format='mixed')
        for index in range(37):
            new_dir,old_dir=folder/f'validation_candidates/candidate_{index:03d}',source/f'validation_candidates/candidate_{index:03d}'
            identity=self.read_json(new_dir/'source_decisions_identity.json')
            new_decisions,old_decisions=self.read_csv(new_dir/'t_decisions.csv'),self.read_csv(old_dir/'t_decisions.csv')
            assert identity['same_as_source'] and Path(identity['path']).resolve()==old_dir/'t_decisions.csv'
            assert identity['sha256']==sha(old_dir/'t_decisions.csv')
            pd.testing.assert_frame_equal(new_decisions,old_decisions,check_dtype=False,check_exact=True)
            new_events,new_trips=self.read_csv(new_dir/'test_trade_events.csv'),self.read_csv(new_dir/'test_round_trips.csv')
            old_events,old_trips=self.read_csv(old_dir/'test_trade_events.csv'),self.read_csv(old_dir/'test_round_trips.csv')
            new,old=pairs(new_events,new_trips),pairs(old_events,old_trips)
            compare=pair_contrast(old,new)
            guards=self.read_csv(new_dir/'first_leg_exit_side_audit.csv')
            cancelled=self.guards(guards,previous)
            row=current.iloc[index];oldrow=original.iloc[index]
            assert len(guards)==row.execution_exit_side_checks and len(cancelled)==row.execution_exit_side_cancellations
            for group,metrics in [(new,row),(old,oldrow)]:
                assert group.complete.sum()==metrics.t_pair_count
                assert abs(group.loc[group.complete,'fragment_net'].sum()-metrics.realized_complete_pair_pnl)<1e-5
            current_keys=pd.MultiIndex.from_frame(new[['first_datetime','symbol','side']])
            if len(cancelled):
                blocked_keys=pd.MultiIndex.from_frame(cancelled[['datetime','symbol','side']])
                assert not current_keys.isin(blocked_keys).any(), '已取消首腿仍产生真实首腿'
                old_fields=['first_datetime','symbol','side','shares','complete','fragment_net','complete_net']
                cancelled=cancelled.merge(old[old_fields].rename(columns={'first_datetime':'datetime','shares':'source_first_shares',
                    'complete':'source_pair_complete','fragment_net':'source_pair_fragment_net','complete_net':'source_pair_complete_net'}),
                    on=['datetime','symbol','side'],how='left',validate='one_to_one')
                cancelled['source_had_actual_first']=cancelled.source_first_shares.notna()
                cancelled['fold'],cancelled['candidate_index']=self.year,index
                self.results['cancelled_current_quotes']+=cancelled.to_dict('records')
            compare['fold'],compare['candidate_index']=self.year,index
            self.results['pair_comparison']+=compare.to_dict('records')
            both=compare.matching.eq('common_first');both_complete=compare.both_complete
            metrics=dict(comparison.iloc[index]);metrics.update(dict(fold=self.year,inner_year=self.inner,
                **{name:row[name] for name in rule_fields if name!='candidate_index'},
                selected_source=bool(oldrow.selected),selected_guarded=bool(row.selected),
                source_started_groups=len(old),guarded_started_groups=len(new),common_first_groups=int(both.sum()),
                source_only_first_groups=int(compare.matching.eq('source_only_first').sum()),
                guarded_only_first_groups=int(compare.matching.eq('guarded_only_first').sum()),
                common_first_both_complete_groups=int(both_complete.sum()),
                common_complete_net_change=float(compare.both_complete_net_change.sum()),
                source_incomplete_groups=int((~old.complete).sum()),guarded_incomplete_groups=int((~new.complete).sum()),
                all_realized_fragment_net_change=float(new.fragment_net.sum()-old.fragment_net.sum()),
                marked_increment_minus_fragment_change=float(row.incremental_net_pnl-oldrow.incremental_net_pnl-new.fragment_net.sum()+old.fragment_net.sum()),
                cancelled_rows=len(cancelled),cancelled_rows_with_source_actual_first=int(cancelled.source_had_actual_first.sum()) if len(cancelled) else 0,
                source_actual_cancelled_complete_net=float(cancelled.source_pair_complete_net.sum()) if len(cancelled) else 0.))
            self.results['candidate_comparison'].append(clean(metrics))
            if len(cancelled):
                for (side,reason),group in cancelled.groupby(['side','opposite_quote_reason']):
                    self.results['cancel_summary'].append(dict(fold=self.year,candidate_index=index,side=side,reason=reason,
                        cancelled_rows=len(group),symbols=group.symbol.nunique(),distinct_days=group.datetime.dt.normalize().nunique(),
                        source_actual_first_rows=int(group.source_had_actual_first.sum()),
                        source_actual_complete_rows=int(group.source_pair_complete.eq(True).sum()),
                        source_actual_complete_net=float(group.source_pair_complete_net.sum())))
            for prefix in ['SH.688','SZ.300','SZ.301','other']:
                group=cancelled.loc[cancelled.board_prefix.eq(prefix)] if len(cancelled) else cancelled
                self.results['cancel_board_summary'].append(dict(fold=self.year,candidate_index=index,board_prefix=prefix,
                    cancelled_rows=len(group),symbols=group.symbol.nunique() if len(group) else 0,
                    proxy_9_8_below_19_8_rows=int(group.growth_prefix_9_8_proxy_below_19_8.sum()) if len(group) else 0,
                    not_claimed_as_actual_price_limit=True))
        assert all(sha(path)==s for path,s in self.inputs.items())
        return dict(passed=True,fold=self.year,inner_year=self.inner,source_report=str(self.source),
            original_selected=old_selected['rule'],guarded_selected=selected['rule'],inputs_sha256=self.inputs,
            no_outer_read=True,no_refit_or_policy_change=True,source_decisions_exact=True,
            all_cancelled_current_raw_quotes_verified=True,opposite_limits_independently_recomputed=True,
            limitations=['取消时本方向仅quote可填，尚未执行现金/准备金等后续条件。',
                '同键不同首腿数量列为单边组，不强配规模不同的收益。',
                '未重跑完整账户；本诊断复核保存片段/同候选差额及当前取消证据。',
                '不同候选含重复机会和独立账户，差额不能相加当组合。'])


def markdown(directory,results):
    """用相同候选的分布解释机制，不把不同候选delta合成收益。"""
    rows=pd.read_csv(directory/'candidate_comparison.csv')
    lines=['# 首腿同刻反向限价复核：内部同候选比较','','仅完成的内部折；不读外层、不重训、不改变原规则。下表统计36主动候选的配对变化，均值/中位数是实验分布，不能相加当组合。','',
        '|留出折|内部年|净额改善/变差/相同候选|净额变化中位数 元|完整组变化中位数|配对手变化中位数|费用变化中位数 元|库存差减少候选|恢复失败减少候选|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for result in results:
        g=rows.loc[rows.fold.eq(result['fold'])&rows.enabled]
        delta=g.incremental_net_pnl_change
        lines.append(f"|{result['fold']}|{result['inner_year']}|{int(delta.gt(1e-8).sum())}/{int(delta.lt(-1e-8).sum())}/{int(delta.abs().le(1e-8).sum())}|{delta.median():.2f}|{g.t_pair_count_change.median():g}|{g.completed_t_pair_lots_change.median():g}|{g.t_fee_change.median():.2f}|{int(g.inventory_mismatch_days_change.lt(0).sum())}|{int(g.recovery_failure_days_change.lt(0).sum())}|")
    lines+=['','candidate_comparison.csv 保留原、新每个候选完整指标及共同/单边首腿；pair_comparison.csv 全量保留双方未完整组，未按盈亏筛选。',
        'cancelled_current_quotes.csv 的取消证据逐个核原始到达分钟close/high/low/amount/volume，并核原信号previous_close。source_had_actual_first区分源账户真的成交与仅新路径中有检查，不把取消次数全当成避免的旧成交。',
        '共同首腿必须股票、首腿时刻、方向、数量全部相同。单边首腿可能来自直接取消，也可能来自现金/旧股/待恢复路径变化；不把全部净额差归因于直接删除亏损交易。',
        '标价增量与所有已实现恢复片段之差另外保留，涉及尚未完成首腿费用/残仓估值，不能仅比较共同完成组就声称策略总收益提高。',
        '同刻反向未受限仍不保证未来可恢复；库存失败、亏损与hold均按原结果保留。历史反复研究不等于全新外层验证。']
    lines+=['','板块前缀分布见cancel_board_summary.csv，包含SH.688/SZ.300/SZ.301的零计数。原9.8%是固定保守代理，不能称所有板块真实涨跌停。'
            'growth_prefix_9_8_proxy_below_19_8标记上述前缀、仅命中保守9.8%区间且尚低于19.8%的取消；它仅说明代理命中，'
            '不证明实际封板，也不反向宣称可成交或存在新择时利润。准确原价/前收/高低在取消明细逐行保存。']
    (directory/'diagnosis.md').write_text('\n'.join(lines)+'\n')


def main():
    """仅追加阶段目录；最终汇总须等待主运行终态并核全部原输入身份。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',required=True)
    parser.add_argument('--stage',action='store_true')
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args();assert args.stage!=args.final
    report=Path(args.report_dir).resolve();cfg=json.loads((report/'resolved_config.json').read_text())
    assert cfg['research_version']=='inventory_guarded_t_v15'
    available=[y for y in cfg['test_years'] if all((report/f'year_{y}'/name).exists() for name in ['validation_rule_metrics.csv','guarded_vs_source_candidates.csv','selected_rule.json'])]
    if args.final:
        assert available==cfg['test_years'] and json.loads((report/'runtime_integrity_audit.json').read_text())['all_unchanged'] and (report/'final_report.md').exists()
    script_sha=sha(__file__);stages=[]
    for year in available:
        prior=None
        for p in sorted((OUT/f'fold_{year}').glob('run_*/analysis.json'),reverse=True):
            value=json.loads(p.read_text())
            if value['source_current_report']==str(report) and value['script_sha256']==script_sha and all(sha(k)==s for k,s in value['inputs_sha256'].items()):
                assert all(sha(p.parent/name)==s for name,s in value['output_sha256'].items());prior=p.parent;break
        if prior is not None: stages.append(prior);continue
        out=OUT/f'fold_{year}'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}';out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(__file__,out/'diagnose.py');d=Diagnosis(report,year)
        try:
            result=d.run()
            for name,rows in d.results.items():pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
            markdown(out,[result])
            result.update(source_current_report=str(report),script_sha256=script_sha,output_sha256={p.name:sha(p) for p in out.iterdir()})
            (out/'analysis.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2))
        except Exception as error:
            (out/'failure.json').write_text(json.dumps(clean(dict(error=str(error),inputs_sha256=d.inputs)),ensure_ascii=False,indent=2));raise
        stages.append(out);print(f'fold={year} 同候选/全首腿/当前取消证据完成:{out}',flush=True)
    if not args.final:
        print(json.dumps(dict(complete=False,completed_folds=available,stages=list(map(str,stages))),ensure_ascii=False));return
    out=OUT/f'final_{datetime.now():%Y%m%d_%H%M%S_%f}';out.mkdir(exist_ok=False)
    results=[json.loads((p/'analysis.json').read_text()) for p in stages]
    for name in TABLES:
        frames=[]
        for p in stages:
            try:frames.append(pd.read_csv(p/f'{name}.csv',float_precision='round_trip'))
            except pd.errors.EmptyDataError:pass
        (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    markdown(out,results);shutil.copy2(__file__,out/'diagnose.py')
    inputs={k:s for r in results for k,s in r['inputs_sha256'].items()}
    for name in ['source_hashes.json','runtime_integrity_audit.json','reused_model_artifacts_sha256.json','learning_source_reuse_audit.json']:
        inputs[str(report/name)]=sha(report/name)
    sources=json.loads((report/'source_hashes.json').read_text());assert len(sources)==51
    for name,s in sources.items():
        assert sha(report/'source_snapshot'/name)==s and sha(ROOT/name)==s
        inputs[str(report/'source_snapshot'/name)]=s
    assert all(sha(k)==s for k,s in inputs.items())
    (out/'verification.json').write_text(json.dumps(clean(dict(passed=True,complete=True,no_outer_read=True,
        original_runtime_sources_unchanged=51,results=results,inputs_sha256=inputs,script_sha256=script_sha,
        output_sha256={p.name:sha(p) for p in out.iterdir()})),ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=True,final_directory=str(out),inputs=len(inputs)),ensure_ascii=False))


if __name__=='__main__':main()
