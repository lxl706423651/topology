"""验证v10内层25候选、冻结顺序、来源身份与因果检查，不训练或运行真实账户。

在quant环境执行。复用临时v8/v9身份夹具，采用确定性的账户/模型替身；
正向路径仍使用真实频率规则、选择器及文件身份核对，不读取新外层业绩。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import json

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_weekly_t_active as active_runner
from scripts import run_weekly_t_frequency as runner
from scripts import test_weekly_t_active_workflow as active_fixture
from scripts.test_weekly_t_active_workflow import dump, sha
from scripts.test_weekly_t_frequency import opportunities
from quant_backtest.weekly_t_short_horizon import create_short_decisions


class TestFrequencyWorkflow(unittest.TestCase):
    """拒绝会改变训练/测试来源的反例，并核参考决策与内层频率选择。"""

    def setUp(self):
        """准备四折微型源和两个目标折，扩成三个h20模型并写原参考身份。"""
        self.fixture=active_fixture.TestActiveWorkflow(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root,self.source=self.fixture.root,self.fixture.source
        self.cfg=deepcopy(self.fixture.cfg)
        self.active=self.root/'active'
        self.active.mkdir()
        self.cfg.update(research_version=runner.VERSION,candidate_activation_policy='frequency_execution_valid',
            source_active_report_dir=str(self.active),frequency_horizon_minutes=20,
            frequency_absolute_net_margins=[0.,.0005],frequency_rank_quantiles=[.7,.85,.95],
            frequency_min_vwap_gaps=[.0035,.005],frequency_max_abs_day_return=.03,
            frequency_max_known_amplitude=.06,frequency_target_pairs_per_day=3.)
        self.audit=deepcopy(self.fixture.audit)
        self.audit['frozen_contexts']={}
        self.folds=[]
        self.reference={}
        self.support={'bounds':{'f_x':{'lower':-10.,'upper':10.}},'training_rows':20}
        for year in [2023,2024,2025,2026]:
            origin=self.source/f'year_{year}'
            fold=json.loads((origin/'split.json').read_text())
            if year in self.cfg['test_years']:
                self.folds.append(fold)
            first=origin/'inner_t_h20_l2_depth4'
            original=json.loads((first/'model_metadata.json').read_text())
            for profile in ['l2_depth6','huber_depth4']:
                model=origin/f'inner_t_h20_{profile}'
                model.mkdir()
                metadata=deepcopy(original)
                metadata['profile']=profile
                metadata['resolved_config']['t_model_profile']=profile
                dump(model/'model_metadata.json',metadata)
                joblib.dump({'metadata':metadata,'features':['f_x'],'models':{'return':'test'}},model/'model_bundle.joblib')
                (model/'return_model.txt').write_text(f'fake-native-{year}-{profile}')
                self.audit['models'][f'year_{year}/inner_t_h20_{profile}']={
                    'passed':True,'metadata_sha256':sha(model/'model_metadata.json'),
                    'model_bundle_sha256':sha(model/'model_bundle.joblib'),'native_model_sha256':sha(model/'return_model.txt')}
            prior={'candidate_index':1,'enabled':True,'horizon_minutes':20,'profile':'l2_depth4',
                   'net_margin':.002,'fraction':.25,'direction':'both'}
            self.reference[year]=prior
            folder=self.active/f'year_{year}'
            folder.mkdir()
            dump(folder/'selected_rule.json',prior)
            data=self.frame_for_fold(fold)
            decisions=create_short_decisions(data,prior,self.cfg,self.support)
            metrics=origin/'validation_rule_metrics.csv'
            rows=pd.read_csv(metrics)
            digest=__import__('hashlib').sha256(pd.util.hash_pandas_object(decisions,index=False).to_numpy().tobytes()).hexdigest()
            rows.loc[rows.candidate_index.eq(1),'decision_sha256']=digest
            rows.to_csv(metrics,index=False)
        dump(self.source/'independent_residual_audit.json',self.audit)
        self.counter=0

    def frame_for_fold(self,fold):
        """用对应内部年三个当前信号，不依赖任何真实市场收益。"""
        frame=opportunities(('09:35','09:45','10:00'),day=f"{fold['inner_year']}-04-15")
        frame['prediction']=.01
        frame['residual_prediction']=frame.prediction
        frame['market_prediction']=0.
        frame['f_x']=0.
        return frame

    def output_dir(self,with_folds=False):
        """每个检查独立输出，避免覆盖已经固定的测试文件。"""
        self.counter+=1
        out=self.root/f'frequency_out_{self.counter}'
        out.mkdir()
        if with_folds:
            for fold in self.folds:
                (out/f"year_{fold['test_year']}").mkdir()
        return out

    def test_config_reuse_rejects_changed_cost_or_training_and_smoke_only_scope(self):
        """新信号参数可声明，但费用/训练变化不能复用原内层结果。"""
        path=self.root/'frequency.yaml'
        path.write_text(yaml.safe_dump(self.cfg))
        with patch.object(active_runner,'PROJECT_ROOT',self.root):
            full=runner.load_configuration(['--config',str(path)])
            smoke=runner.load_configuration(['--config',str(path),'--max-symbols','8','--smoke-weeks','3','--test-years','2026'])
            changed={key for key in full if full[key]!=smoke[key]}
            self.assertEqual(changed,{'max_symbols','smoke_weeks','test_years'})
            for key,value in [('commission_rate',.001),('short_t_fraction',.5),('t_market_prior',.1)]:
                modified=deepcopy(self.cfg)
                modified[key]=value
                path.write_text(yaml.safe_dump(modified))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    runner.load_configuration(['--config',str(path)])

    def test_overlapping_fold_and_wrong_core_model_refused(self):
        """时间重叠或被替换的h20权重在账户执行前拒绝。"""
        split=self.source/'year_2023/split.json'
        bad=json.loads(split.read_text())
        bad['train_years']=[2023,2025,2026]
        dump(split,bad)
        with self.assertRaisesRegex(ValueError,'拆分合同'):
            runner.load_fold_definitions(self.cfg,self.output_dir(),{})
        fold=self.folds[0]
        model=self.source/'year_2023/inner_t_h20_l2_depth4/model_bundle.joblib'
        model.write_bytes(model.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'核心模型身份变化'):
            runner.copy_core_model(fold,'l2_depth4',self.cfg,self.output_dir(),{})

    def test_inner_and_outer_context_use_correct_model_and_year_identity(self):
        """核对v4原始名单而非再导出副本；inner/outer须使用相应训练域的模型。"""
        fold=self.folds[0]
        v4=self.root/'v4'
        cfg={**self.cfg,'frozen_selection_report_dir':str(v4)}
        selected=pd.DataFrame({'symbol':['TEST'],'trade_week':pd.to_datetime(['2026-04-13'])})
        frame=self.frame_for_fold(fold)
        out=self.output_dir()
        captured=[]
        for filename in ('weekly_predictions.csv','weekly_prediction_metrics.csv'):
            (self.source/'year_2023'/filename).write_text('symbol\nTEST\n')
        for scope,name,years in [('validation','inner_weekly_model',fold['core_years']),('test','weekly_model',fold['train_years'])]:
            filename='validation_weekly_selections.csv' if scope=='validation' else 'weekly_selections.csv'
            source=v4/'year_2023'/filename
            source.parent.mkdir(parents=True,exist_ok=True)
            selected.to_csv(source,index=False)
            derived=self.source/'year_2023'/filename
            derived.write_bytes(source.read_bytes().replace(b'\n',b'\r\n'))
            self.assertNotEqual(sha(source),sha(derived))
            model=v4/'year_2023'/name
            model.mkdir(parents=True,exist_ok=True)
            (model/'model_bundle.joblib').write_text(scope)
            dump(model/'model_metadata.json',{'train_years':years})
            context={'source_model_sha256':sha(model/'model_bundle.joblib'),'source_metadata_sha256':sha(model/'model_metadata.json'),
                     'context_inputs_sha256':scope+'-context'}
            self.audit['frozen_contexts'][f'year_2023/{scope}']={
                'passed':True,'source_selection_sha256':sha(source),'context':context}
            dump(self.source/'independent_residual_audit.json',self.audit)
            def attach(data,selection,universe,config,model_path,train_years):
                captured.append((Path(model_path).name,list(train_years)))
                result=data.copy()
                result.attrs['frozen_weekly_context_manifest']=context
                return result
            with patch.object(runner,'read_frozen_selections',return_value=selected) as read_selections, \
                 patch.object(runner,'select_execution_data',return_value=(pd.DataFrame(),frame)), \
                 patch.object(runner,'attach_frozen_context',side_effect=attach):
                runner.prepare_execution_context(fold,scope,pd.DataFrame(),frame,pd.DataFrame({'symbol':['TEST']}),cfg,out,{})
                self.assertEqual(read_selections.call_args.args[0],source)
            (model/'model_bundle.joblib').write_text('replaced')
            with self.assertRaisesRegex(ValueError,'周模型上下文身份'),patch.object(runner,'read_frozen_selections',return_value=selected), \
                 patch.object(runner,'select_execution_data',return_value=(pd.DataFrame(),frame)):
                runner.prepare_execution_context(fold,scope,pd.DataFrame(),frame,pd.DataFrame({'symbol':['TEST']}),cfg,out,{})
            source.write_bytes(source.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'冻结周股池偏离原审计身份'):
                runner.prepare_execution_context(fold,scope,pd.DataFrame(),frame,pd.DataFrame({'symbol':['TEST']}),cfg,out,{})
        self.assertEqual(captured,[('inner_weekly_model',fold['core_years']),('weekly_model',fold['train_years'])])

    def test_prediction_audit_catches_future_leak_and_native_nan(self):
        """未来依赖会在第二次预测变化；原生NaN不能靠NaN比较漏过。"""
        frame=self.frame_for_fold(self.folds[0])
        frame['f_x']=[10.,20.,30.]
        frame['execution_price']=frame.f_x
        bundle={'features':['f_x']}
        class Native:
            def feature_name(self): return ['f_x']
            def predict(self,matrix): return np.asarray(matrix)[:,0]
        def prediction(bundle,data,cfg):
            result=data.copy()
            result['prediction']=pd.to_numeric(data.f_x)
            result['residual_prediction']=result.prediction
            result['market_prediction']=0.
            return result
        with patch.object(runner.joblib,'load',return_value=bundle),patch.object(runner.lightgbm,'Booster',return_value=Native()), \
             patch.object(runner,'predict_residual_t',side_effect=prediction):
            actual=runner.predict_and_audit(frame,self.root,self.cfg,self.output_dir(),'valid')
            self.assertEqual(actual.prediction.tolist(),[10.,20.,30.])
        def leak(bundle,data,cfg):
            result=prediction(bundle,data,cfg)
            result['prediction']=data.execution_price.fillna(0.)
            return result
        with patch.object(runner.joblib,'load',return_value=bundle),patch.object(runner.lightgbm,'Booster',return_value=Native()), \
             patch.object(runner,'predict_residual_t',side_effect=leak),self.assertRaises(AssertionError):
            runner.predict_and_audit(frame,self.root,self.cfg,self.output_dir(),'leak')
        native=Native()
        native.predict=lambda matrix:np.full(len(matrix),np.nan)
        with patch.object(runner.joblib,'load',return_value=bundle),patch.object(runner.lightgbm,'Booster',return_value=native), \
             patch.object(runner,'predict_residual_t',side_effect=prediction),self.assertRaises(AssertionError):
            runner.predict_and_audit(frame,self.root,self.cfg,self.output_dir(),'invalid_native')

    def test_25_inner_rules_frozen_reference_exact_and_frequency_selection_before_outer(self):
        """候选文件先于内层账户；精确参考不重选，25候选完成后统一固定全部外层。"""
        out=self.output_dir(with_folds=True)
        seen=[]
        def context(fold,scope,universe,intraday,daily,cfg,directory,inputs):
            self.assertEqual(scope,'validation')
            return pd.DataFrame({'symbol':['TEST']}),pd.DataFrame(),self.frame_for_fold(fold)
        def evaluations(tasks,selected,daily,intraday,cfg,directory,recovery_store):
            year=int(Path(directory).parent.name.split('_')[1])
            rules=json.loads((out/f'year_{year}/candidate_rules_frozen_before_inner_execution.json').read_text())
            self.assertEqual(len(rules),25)
            self.assertEqual(len(tasks),25)
            self.assertFalse((out/'all_rules_frozen_before_outer_test.json').exists())
            expected=create_short_decisions(intraday,self.reference[year],cfg,self.support)
            actual=next(t['decisions'] for t in tasks if t['candidate_index']==0)
            pd.testing.assert_frame_equal(actual,expected)
            seen.append(year)
            results={}
            for task in tasks:
                index=task['candidate_index']
                pairs,net=(2,-3.) if index==0 else ((30,-10.) if index==1 else (1,-20.))
                metrics={'t_pair_count':pairs,'completed_pair_count':pairs,'inventory_mismatch_days':0,
                    'recovery_failure_days':0,'pending_pairs':0,'base_plan_mismatch_count':0,'t1_violation_count':0}
                results[index]={'metrics':metrics,'incremental_net_pnl':net,'active_days':2,'active_stock_days':2}
            return results
        with patch.object(runner,'prepare_execution_context',side_effect=context), \
             patch.object(runner,'get_store',return_value=types.SimpleNamespace(symbols=['TEST'])), \
             patch.object(runner,'run_weekly_portfolio',return_value={}), \
             patch.object(runner,'calculate_frequency_opportunities',return_value=({'eligible_portfolio_days':10},pd.DataFrame())), \
             patch.object(runner,'save_result',return_value={}), \
             patch.object(runner,'predict_and_audit',side_effect=lambda data,*args:data.copy()), \
             patch.object(runner,'run_frequency_candidates',side_effect=evaluations):
            table=runner.freeze_frequency_rules(self.folds,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),self.cfg,out,{}, {})
        self.assertEqual(seen,[2023,2026])
        self.assertEqual(len(table),50)
        self.assertEqual(table.loc[table.selected,'candidate_index'].tolist(),[1,1])
        marker=json.loads((out/'all_rules_frozen_before_outer_test.json').read_text())
        self.assertEqual(marker['test_years'],[2023,2026])
        for year in seen:
            rule=json.loads((out/f'year_{year}/selected_rule.json').read_text())
            self.assertEqual(rule['inner_complete_pairs'],30)
            self.assertEqual(rule['inner_incremental_net_pnl'],-10.)
            self.assertEqual(rule['signal_mode'],'absolute')
            self.assertTrue(rule['enabled'])
            self.assertEqual(sha(out/f'year_{year}/t_model/model_bundle.joblib'),
                             sha(self.source/f'year_{year}/inner_t_h20_l2_depth4/model_bundle.joblib'))

    def test_outer_guard_rejects_missing_incomplete_or_changed_freeze(self):
        """没有全部正确固定规则，连第一个外层context或账户都不能执行。"""
        out=self.output_dir(with_folds=True)
        with patch.object(runner,'prepare_execution_context') as context:
            with self.assertRaises((FileNotFoundError,ValueError)):
                runner.evaluate_frequency_folds(self.folds,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),self.cfg,out,{}, {})
            context.assert_not_called()
        for fold in self.folds:
            dump(out/f"year_{fold['test_year']}/selected_rule.json",{'enabled':True})
        marker={'test_years':[2023],'selected_rule_sha256':{str(f['test_year']):sha(out/f"year_{f['test_year']}/selected_rule.json") for f in self.folds}}
        dump(out/'all_rules_frozen_before_outer_test.json',marker)
        with patch.object(runner,'prepare_execution_context') as context,self.assertRaises(ValueError):
            runner.evaluate_frequency_folds(self.folds,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),self.cfg,out,{}, {})
        context.assert_not_called()
        marker['test_years']=[2023,2026]
        dump(out/'all_rules_frozen_before_outer_test.json',marker)
        rule=out/'year_2026/selected_rule.json'
        rule.write_text('{"enabled":false}')
        with patch.object(runner,'prepare_execution_context') as context,self.assertRaises(ValueError):
            runner.evaluate_frequency_folds(self.folds,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),self.cfg,out,{}, {})
        context.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
