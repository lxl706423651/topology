"""验证v11核心OOF精确恢复、标签隔离、成本合同及全折冻结后多期限外评。

从quant环境执行。只创建临时小表和权重替身，真实运行来源恢复、严格purge与
择时决策函数；不训练额外大模型、不读取新外层结果或修改旧源和缓存。
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_weekly_t_timing as runner
from quant_backtest import weekly_t_timing_data as data_module
from quant_backtest.weekly_t_cohort import CONTEXT_FEATURES, TRACE_COLUMNS, attach_weekly_context
from quant_backtest.weekly_t_models import select_training_rows
from quant_backtest.weekly_t_short_horizon import planned_recovery_times
from quant_backtest.weekly_t_timing import timing_candidates
from scripts.test_weekly_t_cohort import cohort_config, cohort_frames, fake_weekly_predictions
from scripts.test_weekly_t_timing_models import timing_config


def sha(path):
    """独立计算临时文件身份。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    """仅写测试独占目录。"""
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def frame_sha(frame):
    """独立记录顺序敏感的表内容身份。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


class TimingWorkflowTests(unittest.TestCase):
    """来源和控制流反例：不以模拟账户的盈亏作为模型有效性证据。"""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='weekly_t_timing_workflow_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'v8'
        self.source.mkdir()
        self.cfg = {**timing_config(), **cohort_config(),
                    'initial_cash': 1000000., 'base_fraction': .95, 'lot_size': 100,
                    'short_t_fraction': .25, 'latest_planned_recovery_time': '14:30',
                    't_training_domain': 'known_slot_and_recovery_deadline',
                    'horizons': [20,40,60], 'timing_horizons': [20,40,60],
                    't_model_profiles': ['l2_depth4','l2_depth6'],
                    'timing_regimes': ['learned','reversal','momentum'],
                    'timing_min_net_edges': [.0005,.001], 'timing_min_profit_probabilities': [.4,.5],
                    'timing_max_abs_day_return': .03, 'timing_max_known_amplitude': .06,
                    't_avoid_locked_exit_side': False, 'max_symbols': 1, 'smoke_weeks': 1,
                    'research_version': runner.VERSION, 'candidate_activation_policy': 'positive_net_execution_valid',
                    'source_training_report_dir': str(self.source), 'test_years': [2023,2026],
                    'dense_recovery_cache_paths': {str(y):str(self.root/f'dense_{y}') for y in range(2023,2027)}}
        self.cfg.pop('t_label_horizon_minutes')
        self.weekly, original = cohort_frames()
        self.weekly['f_last_close'] = 10.
        pieces=[]
        for clock in ['09:35','13:30','14:00']:
            piece=original.copy()
            h,m=map(int,clock.split(':'))
            piece['datetime']=piece.date+pd.Timedelta(hours=h,minutes=m)
            piece['execution_time']=piece.datetime+pd.Timedelta(minutes=1)
            piece['f_day_return']=.01
            piece['f_known_amplitude']=.02
            piece['f_timing_trend_efficiency']=.6
            piece['f_timing_vwap_z']=-1.
            piece['f_vwap_deviation']=-.004
            piece['f_return_5m']=.001
            piece['f_return_15m']=.002
            piece['f_open_return']=.003
            pieces.append(piece)
        self.intraday=pd.concat(pieces,ignore_index=True).sort_values(['datetime','symbol']).reset_index(drop=True)
        self.daily=self.intraday[['date','symbol']].drop_duplicates()
        self.folds=[]
        self.expected={}
        self.audit={'passed':True,'complete':True,'full_configured_scope_complete':True,'selectors':{},'cohorts':{}}
        for year in self.cfg['test_years']:
            train=[y for y in range(2023,2027) if y!=year]
            inner=max(train)
            fold={'test_year':year,'train_years':train,'inner_year':inner,
                  'core_years':[y for y in train if y!=inner],'outer_t_refit':False}
            self.folds.append(fold)
            self.build_source_cohort(fold)
        dump(self.source/'independent_residual_audit.json',self.audit)
        labels=[]
        for horizon in self.cfg['horizons']:
            part=self.intraday[['datetime','symbol','target_return']].copy()
            part['horizon_minutes']=horizon
            part['label_start']=part.datetime+pd.Timedelta(minutes=1)
            part['label_end']=planned_recovery_times(part.datetime,horizon)
            part['recovery_time']=part.label_end
            part['recovery_price']=10.01
            # 即便所有留出年都有极端标签，也不能被传入本折fit。
            part.loc[part.datetime.dt.year.eq(2026),'target_return']=1000.
            labels.append(part)
        self.labels=pd.concat(labels,ignore_index=True)
        self.counter=0

    def build_source_cohort(self,fold):
        """建立有SHA审计的互为OOF名单及逐值lineage，不训练周模型。"""
        directory=self.source/f"year_{fold['test_year']}"/'oof_weekly_cohort'
        directory.mkdir(parents=True)
        parts=[]
        for target in fold['core_years']:
            selector=directory/f'selector_{target}'
            selector.mkdir()
            other=[y for y in fold['core_years'] if y!=target]
            (selector/'model_bundle.joblib').write_text(json.dumps({'train_years':other}))
            selected=fake_weekly_predictions(self.weekly[self.weekly.trade_week.dt.year.eq(target)],self.cfg)
            selected=selected[selected.selected].copy()
            selected['source_model_training_years']=json.dumps(other,separators=(',',':'))
            selected['source_model_sha256']=sha(selector/'model_bundle.joblib')
            selected.to_csv(selector/'oof_weekly_selections.csv',index=False)
            selected=pd.read_csv(selector/'oof_weekly_selections.csv',float_precision='round_trip',parse_dates=['trade_week','signal_datetime'])
            current=self.intraday[self.intraday.datetime.dt.year.eq(target)&self.intraday.datetime.dt.dayofweek.isin([1,2,3])]
            current=current.merge(selected[['trade_week','symbol']],on=['trade_week','symbol'],how='inner',validate='many_to_one')
            enriched=attach_weekly_context(current,selected,self.weekly,self.cfg)
            enriched['cohort_target_year']=target
            parts.append(enriched)
            self.audit['selectors'][str(selector.relative_to(self.source))]={'passed':True,
                'model_bundle_sha256':sha(selector/'model_bundle.joblib'),
                'selected_csv_sha256':sha(selector/'oof_weekly_selections.csv')}
        cohort=pd.concat(parts,ignore_index=True).sort_values(['datetime','symbol'],kind='stable').reset_index(drop=True)
        self.expected[fold['test_year']]=cohort
        trace=['datetime','symbol','feature_source_start',*CONTEXT_FEATURES.values(),*TRACE_COLUMNS]
        keys=['trade_week','symbol','feature_week','signal_datetime','feature_source_start','entry_date','exit_date']
        metadata={'core_years':fold['core_years'],'excluded_years':sorted([fold['test_year'],fold['inner_year']]),
                  'weekly_source_keys_sha256':frame_sha(self.weekly[keys]),'cohort_context_sha256':frame_sha(cohort[trace])}
        dump(directory/'cohort_metadata.json',metadata)
        cohort[trace+['cohort_target_year']].to_csv(directory/'cohort_lineage.csv',index=False)
        self.audit['cohorts'][f"year_{fold['test_year']}"]={'passed':True,'rows':len(cohort),
                                                           'cohort_context_sha256':metadata['cohort_context_sha256']}

    def out(self,fold_dirs=False):
        self.counter+=1
        directory=self.root/f'output_{self.counter}'
        directory.mkdir()
        if fold_dirs:
            for fold in self.folds:(directory/f"year_{fold['test_year']}").mkdir()
        return directory

    def test_restore_exact_oof_lineage_and_reject_wrong_source_or_model(self):
        """恢复完整核心队列不受smoke缩股影响；周参数年份/来源输入变化明确拒绝。"""
        fold=self.folds[0]
        original=self.intraday.copy(deep=True)
        actual=data_module.restore_core_cohort(self.weekly,self.intraday,fold,self.cfg,self.out(),{},runner.record_input)
        pd.testing.assert_frame_equal(actual,self.expected[fold['test_year']])
        self.assertEqual(set(actual.datetime.dt.year),set(fold['core_years']))
        self.assertTrue(set(CONTEXT_FEATURES.values()).issubset(actual))
        changed=self.weekly.copy()
        changed.loc[0,'feature_source_start']-=pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError,'完整周资格表来源'):
            data_module.restore_core_cohort(changed,self.intraday,fold,self.cfg,self.out(),{},runner.record_input)
        selector=self.source/'year_2023/oof_weekly_cohort/selector_2024/model_bundle.joblib'
        selector.write_bytes(selector.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'selector 身份改变'):
            data_module.restore_core_cohort(self.weekly,self.intraday,fold,self.cfg,self.out(),{},runner.record_input)
        pd.testing.assert_frame_equal(self.intraday,original)

    def test_label_cache_report_bytes_may_differ_only_by_cache_hit(self):
        """旧审计哈希属于缓存metadata；报告cache_hit不同不应错误拒绝同一数据。"""
        cache=self.root/'labels_cache';cache.mkdir()
        path=cache/'labels.pkl';self.labels.to_pickle(path)
        cfg={**self.cfg,'signal_times':['09:35'],'start_date':'2023-01-01','end_date':'2026-04-30','data_root':'synthetic'}
        manifest={key:cfg[key] for key in ['signal_times','horizons','start_date','end_date','data_root']}
        manifest.update(cache_path=str(cache),labels_path=str(path),label_rows=len(self.labels),
                        label_columns=list(self.labels),cache_hit=False)
        dump(cache/'metadata.json',manifest)
        dump(self.source/'horizon_manifest.json',{**manifest,'cache_hit':True})
        self.audit['input_identity']={'horizon':{'passed':True,'manifest_sha256':sha(cache/'metadata.json')}}
        dump(self.source/'independent_residual_audit.json',self.audit)
        actual=data_module.load_labels_only(cfg,self.out(),{},runner.record_input)
        pd.testing.assert_frame_equal(actual,self.labels)
        bad={**manifest,'cache_hit':True,'label_rows':len(self.labels)+1}
        dump(self.source/'horizon_manifest.json',bad)
        with self.assertRaises((ValueError,AssertionError)):
            data_module.load_labels_only(cfg,self.out(),{},runner.record_input)

    def test_reuse_rejects_cost_weekly_policy_or_legacy_source_mutation(self):
        """目标可以变化，费用、周70/30、持仓比例和旧公共源码不可悄悄变化。"""
        old={**self.cfg,'research_version':'weekly_t_v8_residual_zero_market','t_target_mode':'same_clock_cohort_residual_zero_market'}
        dump(self.source/'resolved_config.json',old)
        dependency=self.root/'legacy.py';dependency.write_text('legacy = True\n')
        snapshot=self.source/'source_snapshot';snapshot.mkdir()
        (snapshot/'legacy.py').write_bytes(dependency.read_bytes())
        dump(self.source/'source_hashes.json',{'legacy.py':sha(dependency)})
        self.assertTrue(data_module.validate_timing_sources(self.cfg,self.root)['passed'])
        for key,value in [('commission_rate',.0009),('weekly_return_weight',.8),('short_t_fraction',.5)]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'合同改变'):
                data_module.validate_timing_sources({**self.cfg,key:value},self.root)
        dependency.write_text('legacy = False\n')
        with self.assertRaisesRegex(ValueError,'公共实现身份改变'):
            data_module.validate_timing_sources(self.cfg,self.root)

    def patch_workflow(self,stack,out):
        """替代拟合和账户成本较高的I/O；真实恢复、标签连接和三期限决策照常执行。"""
        self.fits=[];self.predictions=[];self.executions=[]
        def fit(frame,years,cfg,directory,excluded_years=()):
            self.assertTrue(frame.datetime.dt.year.isin(years).all())
            self.assertFalse(set(years)&set(excluded_years))
            self.assertFalse(frame.target_return.eq(1000.).any())
            actual,meta,_=select_training_rows(frame,years,cfg,'intraday',excluded_years)
            self.assertTrue(all(name in actual for name in CONTEXT_FEATURES.values()))
            h=cfg['t_label_horizon_minutes'];profile=cfg['t_model_profile']
            self.fits.append((tuple(years),h,profile,len(actual)))
            directory.mkdir()
            joblib.dump({'metadata':meta,'marker':[h,profile]},directory/'model_bundle.joblib')
            return {}
        def context(fold,scope,universe,intraday,daily,cfg,directory,inputs):
            if scope=='test':
                self.assertTrue((out/'all_rules_frozen_before_outer_test.json').exists())
                self.assertEqual(len(self.fits),12)
            else:
                self.assertEqual(sum(c[0]==tuple(fold['core_years']) for c in self.fits),6)
            year=fold['inner_year'] if scope=='validation' else fold['test_year']
            selected=fake_weekly_predictions(universe[universe.trade_week.dt.year.eq(year)],cfg)
            rows=intraday[intraday.datetime.dt.year.eq(year)].copy()
            rows['target_return']=np.nan
            return selected[selected.selected],daily,rows
        def predict(frame,model,cfg,directory,prefix):
            self.assertTrue(frame.target_return.isna().all())
            self.predictions.append((prefix,cfg['t_label_horizon_minutes'],str(model)))
            value=frame.copy();h=cfg['t_label_horizon_minutes']
            value['raw_prediction']=value['prediction']={20:.006,40:.009,60:.008}[h]
            value['prob_neither']=.1;value['prob_buy_profit']=.8;value['prob_sell_profit']=.1
            value['horizon_minutes']=h
            return value,{'bounds':{'f_input':{'lower':-100.,'upper':100.}}}
        def engine(selected,daily,intraday,decisions,cfg,recovery_bars=None):
            if len(decisions):
                self.assertTrue(decisions.planned_recovery_time.notna().all())
                self.assertTrue(set(decisions.horizon_minutes).issubset({20,40,60}))
                self.assertEqual(set(decisions.horizon_minutes),{20,40})
                self.assertEqual(cfg['commission_rate'],self.cfg['commission_rate'])
            self.executions.append(len(decisions))
            curve=pd.DataFrame({'datetime':[intraday.datetime.min().normalize()+pd.Timedelta(hours=15)],
                                'equity':[1000000.],'baseline_equity':[1000000.]})
            return {'equity_curve':curve,'trade_events':pd.DataFrame(columns=['datetime','symbol','event_type','fragment_id']),
                    'metrics':{'t_pair_count':0,'t_added_equity':0.,'inventory_mismatch_days':0}}
        def evaluations(tasks,selected,daily,intraday,cfg,directory,recovery_store=None):
            self.assertEqual(len(tasks),25)
            self.assertFalse((out/'all_rules_frozen_before_outer_test.json').exists())
            self.assertEqual(len(json.loads((Path(directory).parent/'candidate_rules_frozen_before_inner_execution.json').read_text())),25)
            result={}
            for task in tasks:
                index=task['candidate_index']
                metrics={key:0 for key in ['inventory_mismatch_days','recovery_failure_days','pending_pairs','base_plan_mismatch_count','t1_violation_count']}
                metrics.update(t_pair_count=0 if index==0 else 1,completed_pair_count=0 if index==0 else 1)
                result[index]={'metrics':metrics,'incremental_net_pnl':0. if index==0 else 10. if index==1 else -1.,'active_days':int(index!=0),'active_stock_days':int(index!=0)}
            return result
        changes={'fit_timing_model':fit,'prepare_execution_context':context,'predict_and_audit':predict,
                 'run_weekly_portfolio':engine,'run_frequency_candidates':evaluations,
                 'get_store':lambda *args:types.SimpleNamespace(symbols=['SH.600000','SH.600004'],fingerprint='fake',metadata={'time_count':1}),
                 'calculate_frequency_opportunities':lambda *args:({'eligible_portfolio_days':2},pd.DataFrame()),
                 'save_result':lambda result,*args:result['metrics'], 'validate_candidate_result':lambda *args:{'passed':True}}
        for name,value in changes.items():stack.enter_context(patch.object(runner,name,side_effect=value))

    def test_core_only_training_all_horizons_freeze_before_outer_without_refit(self):
        """两折12模型只收core标签，全部冻结后外评沿用3周期权重，旧执行器接收原时钟。"""
        out=self.out(True)
        original=self.intraday.copy(deep=True)
        with ExitStack() as stack:
            self.patch_workflow(stack,out)
            table=runner.freeze_all_rules(self.folds,self.weekly,self.intraday,self.daily,self.labels,self.cfg,out,{}, {})
            self.assertEqual(len(table),50)
            self.assertEqual(table.loc[table.selected,'candidate_index'].tolist(),[1,1])
            self.assertEqual(len(self.fits),12)
            before={str(p):sha(p) for p in out.glob('year_*/inner_t_*/model_bundle.joblib')}
            annual=runner.evaluate_folds(self.folds,self.weekly,self.intraday,self.daily,self.cfg,out,{}, {})
            self.assertEqual(len(annual),2)
            self.assertEqual(len(self.fits),12)
            self.assertEqual(before,{str(p):sha(p) for p in out.glob('year_*/inner_t_*/model_bundle.joblib')})
            for fold in self.folds:
                rule=json.loads((out/f"year_{fold['test_year']}/selected_rule.json").read_text())
                self.assertEqual(len(rule['model_bundle_sha256']),6)
        pd.testing.assert_frame_equal(self.intraday,original)

    def test_later_fold_rule_or_any_model_change_blocks_first_outer_context(self):
        """后折篡改也必须在第一个外层预测前拒绝，不允许先计算前折外层。"""
        out=self.out(True)
        with ExitStack() as stack:
            self.patch_workflow(stack,out)
            runner.freeze_all_rules(self.folds,self.weekly,self.intraday,self.daily,self.labels,self.cfg,out,{}, {})
        rule=out/'year_2026/selected_rule.json'
        original=rule.read_bytes();rule.write_bytes(original+b' ')
        with patch.object(runner,'prepare_execution_context') as context,self.assertRaises(ValueError):
            runner.evaluate_folds(self.folds,self.weekly,self.intraday,self.daily,self.cfg,out,{}, {})
        context.assert_not_called()
        rule.write_bytes(original)
        # 连未选profile的已冻结模型都属于全折身份，不应遗漏。
        model=out/'year_2026/inner_t_h60_l2_depth6/model_bundle.joblib'
        model.write_bytes(model.read_bytes()+b'changed')
        with patch.object(runner,'prepare_execution_context') as context,self.assertRaises(ValueError):
            runner.evaluate_folds(self.folds,self.weekly,self.intraday,self.daily,self.cfg,out,{}, {})
        context.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
