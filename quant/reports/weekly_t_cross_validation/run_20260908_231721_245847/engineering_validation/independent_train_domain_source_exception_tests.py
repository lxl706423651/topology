"""验证v7仅允许特定并发SHA且费用/指标与调用隔离不变；不修改运行源。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path('/data/lxl/seed-emulator/topology/quant');sys.path.insert(0,str(ROOT))
from scripts.audit_weekly_t_train_domain import verify_sources,verify_annual_ast_isolation,ANNUAL_SNAPSHOT_SHA,ANNUAL_CONCURRENT_SHA

class SourceExceptionTests(unittest.TestCase):
    def setUp(self):
        self.report=ROOT/'reports/weekly_t_cross_validation_smoke/run_20260908_231228_518787'
        self.before=(self.report/'source_snapshot/annual_portfolio.py').read_text()
        self.after=(ROOT/'quant_backtest/annual_portfolio.py').read_text()

    def test_exact_known_version_preserves_explicit_mismatch(self):
        result=verify_sources(self.report)
        self.assertTrue(result['passed']);self.assertFalse(result['current_full_file_match'])
        self.assertFalse(result['files']['quant_backtest/annual_portfolio.py']['passed'])
        self.assertEqual(result['concurrent_source_exception']['proof']['run_lot_import_call_attribute_or_dynamic_name_references'],0)

    def test_other_module_ast_change_rejected(self):
        with self.assertRaises(AssertionError):verify_annual_ast_isolation(self.before,self.after+'\nNEW_GLOBAL = 1\n',{})

    def test_fee_text_change_even_without_ast_change_rejected(self):
        changed=self.after.replace('def calculate_fee(', 'def  calculate_fee(',1)
        self.assertNotEqual(changed,self.after)
        with self.assertRaises(AssertionError):verify_annual_ast_isolation(self.before,changed,{})

    def test_lot_import_rejected(self):
        with self.assertRaises(AssertionError):verify_annual_ast_isolation(self.before,self.after,{'bad.py':'from .annual_portfolio import run_lot_portfolio as delegated'})

    def test_lot_call_or_attribute_rejected(self):
        for text in ['run_lot_portfolio()','mod.run_lot_portfolio()','getattr(mod, "run_lot_portfolio")()']:
            with self.assertRaises(AssertionError):verify_annual_ast_isolation(self.before,self.after,{'bad.py':text})

    def test_unrestricted_annual_module_import_rejected(self):
        with self.assertRaises(AssertionError):verify_annual_ast_isolation(self.before,self.after,{'bad.py':'import quant_backtest.annual_portfolio as fees'})

    def test_unknown_annual_sha_rejected(self):
        checked={'passed':False,'files':{'quant_backtest/annual_portfolio.py':{'expected_sha256':ANNUAL_SNAPSHOT_SHA,
            'snapshot_sha256':ANNUAL_SNAPSHOT_SHA,'current_sha256':'0'*64,'passed':False}}}
        with patch('scripts.audit_weekly_t_train_domain.verify_frozen_sources',return_value=checked):
            with self.assertRaises(AssertionError):verify_sources(self.report)

    def test_any_other_file_change_rejected(self):
        checked={'passed':False,'files':{'quant_backtest/weekly_t_engine.py':{'expected_sha256':'a','snapshot_sha256':'a','current_sha256':'b','passed':False}}}
        with patch('scripts.audit_weekly_t_train_domain.verify_frozen_sources',return_value=checked):
            with self.assertRaises(AssertionError):verify_sources(self.report)

if __name__=='__main__':unittest.main(verbosity=2)
