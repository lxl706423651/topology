"""主动T审计新分解与冻结模型复制的合成反例；只写临时目录，不运行策略或修改原产物。"""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT=Path('/data/lxl/seed-emulator/topology/quant');sys.path.insert(0,str(ROOT))
from scripts.audit_weekly_t_active_execution import decompose_portfolio_increment,verify_active_training_lineage
SOURCE=ROOT/'reports/weekly_t_cross_validation/run_20260908_235818_404243'

def table(*records):
    return pd.DataFrame(records,columns=['event_type','side','notional','fee'])

class ActiveExecutionAuditTests(unittest.TestCase):
    def test_completed_pair_clean_plan(self):
        baseline=table(('weekly','buy',100.,0.))
        actual=table(('weekly','buy',100.,0.),('t','buy',30.,1.),('t','sell',36.,1.))
        result=decompose_portfolio_increment(actual,baseline,{'actual':{'qty':{'A':10}},'baseline':{'qty':{'A':10}}},{'A':12.},{})
        self.assertEqual(result['independent_total_increment_CNY'],4.)
        self.assertEqual(result['paired_T_net_pnl_CNY'],4.)
        self.assertEqual(result['weekly_plan_effect_CNY'],0.)

    def test_unrecovered_sell_cash_not_mistaken_for_profit(self):
        baseline=table(('weekly','buy',100.,0.))
        actual=table(('weekly','buy',100.,0.),('t','sell',30.,1.))
        first=SimpleNamespace(notional=30.,fee=1.,side='sell',shares=3,symbol='A')
        result=decompose_portfolio_increment(actual,baseline,{'actual':{'qty':{'A':7}},'baseline':{'qty':{'A':10}}},{'A':12.},{1:{'first':first,'remaining':3}})
        self.assertEqual(result['T_cashflow_CNY'],29.)
        self.assertEqual(result['paired_T_net_pnl_CNY'],0.)
        self.assertEqual(result['pending_T_inventory_market_value_CNY'],-36.)
        self.assertEqual(result['independent_total_increment_CNY'],-7.)

    def test_partial_fee_allocation_and_pending_mark(self):
        baseline=table(('weekly','buy',100.,0.))
        actual=table(('weekly','buy',100.,0.),('t','sell',40.,2.),('t','buy',18.,1.))
        first=SimpleNamespace(notional=40.,fee=2.,side='sell',shares=4,symbol='A')
        result=decompose_portfolio_increment(actual,baseline,{'actual':{'qty':{'A':8}},'baseline':{'qty':{'A':10}}},{'A':12.},{1:{'first':first,'remaining':2}})
        self.assertEqual(result['unmatched_T_first_leg_cashflow_CNY'],19.)
        self.assertEqual(result['paired_T_net_pnl_CNY'],0.)
        self.assertEqual(result['direct_T_marked_pnl_CNY'],-5.)

    def test_weekly_plan_difference_can_offset_nonzero_pair_pnl(self):
        baseline=table(('weekly','buy',100.,0.))
        actual=table(('weekly','buy',80.,0.),('t','buy',30.,1.),('t','sell',36.,1.))
        result=decompose_portfolio_increment(actual,baseline,{'actual':{'qty':{'A':8}},'baseline':{'qty':{'A':10}}},{'A':12.},{})
        self.assertEqual(result['paired_T_net_pnl_CNY'],4.)
        self.assertEqual(result['weekly_cashflow_difference_CNY'],20.)
        self.assertEqual(result['weekly_plan_effect_CNY'],-4.)
        self.assertEqual(result['independent_total_increment_CNY'],0.)

    def test_unrecovered_buy_keeps_negative_cash_and_extra_inventory(self):
        baseline=table(('weekly','buy',100.,0.))
        actual=table(('weekly','buy',100.,0.),('t','buy',50.,1.))
        first=SimpleNamespace(notional=50.,fee=1.,side='buy',shares=5,symbol='A')
        result=decompose_portfolio_increment(actual,baseline,{'actual':{'qty':{'A':15}},'baseline':{'qty':{'A':10}}},{'A':12.},{1:{'first':first,'remaining':5}})
        self.assertEqual(result['T_cashflow_CNY'],-51.)
        self.assertEqual(result['direct_T_marked_pnl_CNY'],9.)
        self.assertEqual(result['weekly_plan_effect_CNY'],0.)

    def create_copy(self,directory):
        cfg=json.loads((SOURCE/'resolved_config.json').read_text());cfg.update(test_years=[2026],source_training_report_dir=str(SOURCE),research_version='weekly_t_v9_active_execution_valid',candidate_activation_policy='active_execution_valid')
        folder=directory/'year_2026';folder.mkdir()
        original=SOURCE/'year_2026'
        rows=pd.read_csv(original/'validation_rule_metrics.csv',float_precision='round_trip')
        valid=rows.t_pair_count.gt(0)&rows[['inventory_mismatch_days','recovery_failure_days','pending_pairs','base_plan_mismatch_count','t1_violation_count']].eq(0).all(axis=1)
        chosen=rows.loc[valid].sort_values(['incremental_net_pnl','candidate_index'],ascending=[False,True]).iloc[0]
        name=f'inner_t_h{int(chosen.horizon_minutes)}_{chosen.profile}'
        model=original/name
        shutil.copytree(model,folder/'t_model')
        shutil.copyfile(original/'split.json',folder/'split.json')
        shutil.copyfile(original/f'support_h{int(chosen.horizon_minutes)}.json',folder/'selected_support.json')
        metadata=json.loads((SOURCE/'year_2026/selected_rule.json').read_text())
        from scripts.audit_weekly_t_active_execution import file_sha256
        metadata.update(enabled=True,horizon_minutes=int(chosen.horizon_minutes),profile=chosen.profile,source_model=name,
            model_bundle_sha256=file_sha256(model/'model_bundle.joblib'),candidate_index=int(chosen.candidate_index),
            fraction=float(chosen.fraction),direction=chosen.direction,net_margin=float(chosen.net_margin),
            inner_incremental_net_pnl=float(chosen.incremental_net_pnl),inner_complete_pairs=int(chosen.t_pair_count),
            activation_policy='active_execution_valid',statistical_gates_are_informational=True,frozen_before_any_outer_test=True)
        (folder/'selected_rule.json').write_text(json.dumps(metadata))
        return cfg,folder

    def test_actual_frozen_model_reference_and_exact_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);cfg,_=self.create_copy(directory)
            check=verify_active_training_lineage(directory,cfg)
            self.assertTrue(check['passed']);self.assertEqual(check['selected_models']['2026']['core_years'],[2023,2024])

    def test_changed_model_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);cfg,folder=self.create_copy(directory)
            path=folder/'t_model/return_model.txt';path.write_text(path.read_text()+'\n# corruption\n')
            with self.assertRaises(AssertionError):verify_active_training_lineage(directory,cfg)

    def test_cost_change_or_disabled_active_rule_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);cfg,folder=self.create_copy(directory)
            wrong=dict(cfg);wrong['commission_rate']=0.
            with self.assertRaises(AssertionError):verify_active_training_lineage(directory,wrong)
            path=folder/'selected_rule.json';rule=json.loads(path.read_text());rule['enabled']=False;path.write_text(json.dumps(rule))
            with self.assertRaises(AssertionError):verify_active_training_lineage(directory,cfg)


if __name__=='__main__':unittest.main(verbosity=2)
