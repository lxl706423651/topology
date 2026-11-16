"""v15报告反例：合成已有账目，验证原指标、取消请求及版本范围，不运行账户。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_backtest import guarded_t_reporting as reporting
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics
from test_inventory_t_learning_reporting import learning_config, learning_fixture


def guarded_config(**changes):
    """沿用原经济fixture，仅增版本、来源与复查开关。"""
    return learning_config(research_version=reporting.VERSION, t_recheck_exit_side_at_execution=True,
                           source_learning_report_dir="synthetic_v14_source", source_report_scope="smoke", **changes)


def guarded_fixture(**changes):
    """构造真实首腿通过、一个取消、一个通过后未成交，后两者不计实际组。"""
    result = learning_fixture(**changes)
    rows = []
    first = result['trade_events'].loc[lambda x: x.event_type.eq('t') & x.fragment_id.eq(0)]
    if len(first):
        f = first.iloc[0]
        base = {'datetime': f.datetime, 'signal_datetime': f.datetime-pd.Timedelta(minutes=1),
                'symbol': f.symbol, 'side': f.side, 'opposite_side': 'sell' if f.side == 'buy' else 'buy',
                'requested_shares': int(f.shares), 'own_fillable_shares': int(f.shares),
                'guard_enabled': True, 'opposite_checked': True, 'cancelled': False, 'reason': '',
                'opposite_quote_reason': '', 'reference_price': f.reference_price, 'high': f.reference_price,
                'low': f.reference_price, 'amount': 1000000., 'volume': 100000., 'previous_close': 10.,
                'limit_up': np.nan, 'limit_down': np.nan, 'minute_amount_used_before': 0.,
                'cash_before': 100000., 'old_sellable_before': 1000}
        rows.append(base)
        rows.append({**base, 'datetime': f.datetime+pd.Timedelta(minutes=1), 'cancelled': True,
                     'opposite_quote_reason': 'conservative_price_limit_proxy',
                     'reason': 'execution_exit_side_conservative_price_limit_proxy'})
        rows.append({**base, 'datetime': f.datetime+pd.Timedelta(minutes=2)})
    result['first_leg_exit_side_audit'] = pd.DataFrame(rows, columns=reporting.GUARD_COLUMNS)
    return result


class GuardedReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_sources = {name: (ROOT/name).read_bytes() for name in
            ('quant_backtest/balanced_t_reporting.py', 'quant_backtest/dynamic_t_reporting.py',
             'quant_backtest/inventory_t_learning_reporting.py')}

    @classmethod
    def tearDownClass(cls):
        for name, content in cls.old_sources.items():
            assert (ROOT/name).read_bytes() == content, name

    def write(self, result, cfg=None, directory=None):
        """所有产物在临时目录，不接触真实报告。"""
        if directory is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            directory = Path(temporary.name)
        status = reporting.write_guarded_inventory_report(directory, cfg or guarded_config(), [(2024, result)])
        return directory, status

    def test_original_cost_metrics_and_inputs_not_rewritten(self):
        result, cfg = guarded_fixture(net=2000.), guarded_config()
        original, original_cfg = deepcopy(result), deepcopy(cfg)
        expected, _ = calculate_balanced_metrics(result, cfg)
        writer, versions = reporting.write_balanced_report, []
        def observed(directory, actual_cfg, entries, validation_table):
            versions.append(actual_cfg['research_version'])
            return writer(directory, actual_cfg, entries, validation_table)
        with patch.object(reporting, 'write_balanced_report', observed):
            directory, status = self.write(result, cfg)
        self.assertEqual(versions, [reporting.VERSION])
        self.assertEqual(cfg, original_cfg)
        for key in ('all_reference_acquisition_cost', 'holding_cost_reduction_rate', 'incremental_marked_pnl',
                    't_fee', 't_slippage_cost', 'completed_t_pair_lots', 'total_t_traded_lots',
                    'goal_achieved', 'frequency_target_status'):
            self.assertEqual(status['annual_results'][0][key], expected[key])
        for key, value in original.items():
            if isinstance(value, pd.DataFrame): pd.testing.assert_frame_equal(result[key], value, check_exact=True)
            elif isinstance(value, pd.Index): pd.testing.assert_index_equal(result[key], value, exact=True)
            else: self.assertEqual(result[key], value)
        for name in ('annual_summary.csv', '年度.csv', 'test_trade_events.csv', 'test_round_trips.csv',
                     'test_equity_curve.csv', 'test_open_positions.csv', 'daily_inventory_audit.csv',
                     'baseline_trade_events.csv', 'per_stock_cost_reduction.csv', 't_pair_lot_summary.csv',
                     'daily_frequency_coverage.csv', 'goal_evaluation.json', 'exit_decisions.csv',
                     'dynamic_exit_summary.csv', 'first_leg_exit_side_audit.csv', 'guard_summary.csv',
                     'selected_guarded_rules.csv', 'inventory_guarded_summary.json'):
            self.assertTrue((directory/name).is_file(), name)

    def test_guard_requests_not_fills_or_fragment_groups(self):
        directory, status = self.write(guarded_fixture(fragments=(100, 500)))
        row = status['annual_results'][0]
        self.assertEqual((row['complete_t_pairs'], row['recovery_fragments'], row['completed_t_pair_lots']), (1, 2, 6.))
        g = pd.read_csv(directory/'guard_summary.csv').iloc[0]
        self.assertEqual((g.guard_audit_requests, g.guard_cancelled_requests, g.guard_passed_requests), (3, 1, 2))
        self.assertEqual((g.guard_actual_first_leg_groups, g.guard_passed_without_actual_first_leg, g.guard_cancelled_stock_days), (1, 1, 1))
        self.assertEqual(pd.read_csv(directory/'dynamic_exit_summary.csv').iloc[0].latched_exit_pairs, 1)
        meta = json.loads((directory/'inventory_guarded_summary.json').read_text())
        self.assertEqual((meta['research_version'], meta['model_learning_version'], meta['candidate_count']), (reporting.VERSION, 14, 37))
        self.assertFalse(meta['models_retrained'])
        self.assertFalse(meta['guard_pass_is_daily_recovery_guarantee'])

    def test_hold_smoke_scope_and_no_win_claim(self):
        directory, status = self.write(guarded_fixture(hold=True), guarded_config(report_scope='smoke'))
        self.assertFalse(status['goal_achieved'])
        self.assertEqual(status['inventory_guarded']['source_report_scope'], 'smoke')
        self.assertTrue(pd.read_csv(directory/'first_leg_exit_side_audit.csv').empty)
        text = (directory/'final_report.md').read_text()
        self.assertIn('v15', text.splitlines()[0])
        self.assertIn('工程烟测', text.splitlines()[0])
        self.assertIn('没有可评价的做T胜率', text)
        self.assertIn('无重新训练', text)
        self.assertNotIn('v13恢复协议', text)
        self.assertEqual((directory/'test_return_curve.png').read_bytes(), (directory/'测试集收益曲线.png').read_bytes())

    def test_guard_does_not_hide_actual_inventory_failure(self):
        directory, status = self.write(guarded_fixture(pending=True, net=-50., reason='stop_loss'))
        r = status['annual_results'][0]
        self.assertEqual((r['inventory_mismatch_days'], r['pending_pairs'], r['complete_t_pairs']), (1, 1, 0))
        self.assertEqual(r['incremental_marked_pnl'], -50.)
        self.assertFalse(status['goal_achieved'])
        self.assertEqual(pd.read_csv(directory/'guard_summary.csv').iloc[0].guard_actual_first_leg_groups, 1)

    def test_original_yearly_guard_exit_tables_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            folder = directory/'year_2024'
            folder.mkdir()
            result = guarded_fixture()
            before = {}
            for name, key in [('first_leg_exit_side_audit.csv', 'first_leg_exit_side_audit'), ('exit_decisions.csv', 'exit_decisions')]:
                result[key].to_csv(folder/name, index=False, encoding='utf-8')
                before[name] = (folder/name).read_bytes()
            self.write(result, directory=directory)
            for name, content in before.items(): self.assertEqual((folder/name).read_bytes(), content)

    def test_wrong_config_rule_or_missing_table_rejected(self):
        for failure in ('version', 'disabled', 'string_flag', 'profile', 'fraction', 'table', 'duplicate_year'):
            r, cfg = guarded_fixture(), guarded_config()
            if failure == 'version': cfg['research_version'] = 'inventory_learning_t_v14'
            elif failure == 'disabled': cfg['t_recheck_exit_side_at_execution'] = False
            elif failure == 'string_flag': cfg['t_recheck_exit_side_at_execution'] = 'true'
            elif failure == 'profile': r['selected_rule']['profile'] = 'huber'
            elif failure == 'fraction': r['selected_rule']['fraction'] = True
            elif failure == 'table': del r['first_leg_exit_side_audit']
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                if failure == 'duplicate_year':
                    reporting.write_guarded_inventory_report(Path('/unused'), cfg, [(2024, r), (2024, r)])
                else: self.write(r, cfg)

    def test_audit_timing_reason_duplicate_and_cancelled_fill_rejected(self):
        for failure in ('string_bool', 'time', 'opposite', 'negative_quantity', 'reason', 'duplicate', 'cancelled_fill'):
            r = guarded_fixture()
            a = r['first_leg_exit_side_audit']
            if failure == 'string_bool': a['cancelled'] = a.cancelled.astype(str)
            elif failure == 'time': a.loc[0, 'datetime'] = a.loc[0, 'signal_datetime']
            elif failure == 'opposite': a.loc[0, 'opposite_side'] = a.loc[0, 'side']
            elif failure == 'negative_quantity': a.loc[0, 'own_fillable_shares'] = -1
            elif failure == 'reason': a.loc[1, 'reason'] = 'capacity'
            elif failure == 'duplicate': r['first_leg_exit_side_audit'] = pd.concat([a, a.iloc[[0]]], ignore_index=True)
            else:
                a.loc[0, ['cancelled', 'reason', 'opposite_quote_reason']] = [True, 'execution_exit_side_price_limit', 'price_limit']
            with self.subTest(failure=failure), self.assertRaises(ValueError): self.write(r)


if __name__ == '__main__': unittest.main(verbosity=2)
