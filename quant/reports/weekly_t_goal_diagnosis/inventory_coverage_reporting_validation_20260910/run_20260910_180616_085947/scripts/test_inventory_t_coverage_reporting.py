"""v18独立报告反例；仅合成已有账表和临时PNG，不拟合、筛选或重跑真实账户。

验证原成本/手数/失败合同、版本与覆盖身份、缺失值及允许矩阵变化的准确说明。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from quant_backtest import inventory_t_coverage_reporting as reporting
from quant_backtest.balanced_t_reporting import calculate_balanced_metrics
from test_guarded_t_reporting import guarded_config, guarded_fixture


def coverage_config(**changes):
    """原经济/执行fixture仅声明新学习版本及固定覆盖门槛。"""
    cfg = guarded_config()
    cfg.update(research_version=reporting.VERSION, training_loss='regression_l2',
               feature_min_coverage=.70, source_feature_min_coverage=.80,
               horizons=[20, 40, 60], forecast_target='fixed_20_40_60_reference_return',
               frozen_t_models_reused=False, learning_model_profiles=['l2_depth4', 'l2_depth6'],
               source_execution_report_dir='synthetic_v15_execution',
               source_learning_report_dir='synthetic_v14_core')
    cfg.update(changes)
    return cfg


class CoverageReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = {name: (ROOT/name).read_bytes() for name in (
            'quant_backtest/balanced_t_reporting.py', 'quant_backtest/dynamic_t_reporting.py',
            'quant_backtest/guarded_t_reporting.py', 'quant_backtest/inventory_t_robust_reporting.py',
            'quant_backtest/guarded_t_engine.py', 'quant_backtest/weekly_t_models.py')}

    @classmethod
    def tearDownClass(cls):
        for name, content in cls.old.items():
            assert (ROOT/name).read_bytes() == content, name

    def write(self, result, cfg=None, directory=None, validation=None):
        """始终在临时目录写已有fixture；不调用任何模型、selector或执行器。"""
        if directory is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            directory = Path(temporary.name)
        return directory, reporting.write_coverage_inventory_report(
            directory, cfg or coverage_config(), [(2024, result)], validation)

    def test_original_cfg_identity_metrics_and_input_immutability(self):
        result, cfg = guarded_fixture(net=1600., fragments=(100, 500)), coverage_config()
        original, original_cfg = deepcopy(result), deepcopy(cfg)
        expected, _ = calculate_balanced_metrics(result, cfg)
        writer, seen, titles = reporting.write_balanced_report, [], []
        from matplotlib.figure import Figure
        suptitle = Figure.suptitle
        def observe_writer(directory, actual_cfg, entries, validation):
            self.assertIs(actual_cfg, cfg)
            seen.append(actual_cfg['research_version'])
            return writer(directory, actual_cfg, entries, validation)
        def observe_title(figure, title, *args, **kwargs):
            titles.append(title)
            return suptitle(figure, title, *args, **kwargs)
        with patch.object(reporting, 'write_balanced_report', observe_writer), patch.object(Figure, 'suptitle', observe_title):
            directory, status = self.write(result, cfg)
        self.assertEqual(seen, [reporting.VERSION])
        self.assertIn('v18', titles[-1])
        self.assertIn('0.80→0.70', titles[-1])
        self.assertNotIn('Huber', titles[-1])
        self.assertEqual(cfg, original_cfg)
        for key, value in original.items():
            if isinstance(value, pd.DataFrame): pd.testing.assert_frame_equal(result[key], value, check_exact=True)
            elif isinstance(value, pd.Index): pd.testing.assert_index_equal(result[key], value, exact=True)
            else: self.assertEqual(result[key], value)
        for name in ('all_reference_acquisition_cost', 'holding_cost_reduction_rate', 'incremental_marked_pnl',
                     't_fee', 't_slippage_cost', 'complete_t_pairs', 'recovery_fragments',
                     'completed_t_pair_lots', 'total_t_traded_lots'):
            self.assertEqual(status['annual_results'][0][name], expected[name])
        self.assertEqual(expected['all_reference_acquisition_cost'], 30002.)
        self.assertAlmostEqual(expected['holding_cost_reduction_rate'], 1600/30002.)
        self.assertFalse(status['goal_achieved'])
        self.assertEqual(status['frequency_target_status'], 'not_quantified')
        for name in ('final_report.md', '输出文件说明.md', 'annual_summary.csv', '年度.csv', 'test_trade_events.csv',
                     'test_round_trips.csv', 'test_equity_curve.csv', 'test_open_positions.csv', 'daily_inventory_audit.csv',
                     'baseline_trade_events.csv', 'per_stock_cost_reduction.csv', 'daily_frequency_coverage.csv',
                     'first_leg_exit_side_audit.csv', 'guard_summary.csv', 'dynamic_exit_summary.csv', 'exit_decisions.csv',
                     'selected_coverage_rules.csv', 'inventory_coverage_summary.json', 'planned_horizon_summary.csv',
                     'goal_evaluation.json'):
            self.assertTrue((directory/name).is_file(), name)
        self.assertEqual((directory/'test_return_curve.png').read_bytes(), (directory/'测试集收益曲线.png').read_bytes())
        self.assertFalse((directory/'inventory_robust_summary.json').exists())

    def test_feature_identity_contract_does_not_promise_three_added_columns(self):
        directory, status = self.write(guarded_fixture())
        meta = status['inventory_coverage']
        self.assertEqual((meta['model_learning_version'], meta['newly_fitted_models_per_fold'], meta['candidate_count']), (18, 6, 37))
        self.assertEqual((meta['training_loss'], meta['objective']), ('regression_l2', 'regression'))
        self.assertEqual((meta['source_feature_min_coverage'], meta['feature_min_coverage']), (.8, .7))
        self.assertEqual(meta['newly_fitted_model_profiles'], ['l2_depth4', 'l2_depth6'])
        self.assertTrue(meta['same_training_rows_labels_split_required'])
        self.assertTrue(meta['source_feature_screen_reproduction_required'])
        self.assertTrue(meta['common_feature_values_and_dtypes_equal_required'])
        self.assertFalse(meta['complete_training_matrix_identity_required'])
        self.assertTrue(meta['selected_features_and_column_order_may_change'])
        for name in ('additional_15m_features_guaranteed_selected', 'feature_gain_guaranteed',
                     'identity_verification_performed_by_reporter', 'missing_feature_rows_deleted',
                     'frozen_t_models_reused', 'target_is_portfolio_execution_net_pnl'):
            self.assertFalse(meta[name], name)
        self.assertTrue(meta['missing_feature_values_preserved_as_nan'])
        self.assertTrue(meta['cross_horizon_argmax'])
        self.assertFalse(meta['uniform_1455_forecast_target'])
        text = (directory/'final_report.md').read_text()
        self.assertIn('v18', text.splitlines()[0])
        self.assertIn('不保证三列实际入模或改善收益', text)
        self.assertIn('矩阵列序和权重允许改变', text)
        self.assertIn('报告器不读取模型', text)
        self.assertIn('不强填零、不删当前样本', text)
        self.assertNotIn('完整矩阵与原L2完全相同', text)
        self.assertNotIn('t_huber_alpha', meta)

    def test_partial_recovery_guard_requests_and_pairs_are_distinct(self):
        directory, status = self.write(guarded_fixture(fragments=(100, 500)))
        row = status['annual_results'][0]
        self.assertEqual((row['complete_t_pairs'], row['recovery_fragments'], row['completed_t_pair_lots']), (1, 2, 6.))
        guard = pd.read_csv(directory/'guard_summary.csv').iloc[0]
        self.assertEqual((guard.guard_audit_requests, guard.guard_cancelled_requests, guard.guard_passed_requests), (3, 1, 2))
        self.assertEqual((guard.guard_actual_first_leg_groups, guard.guard_passed_without_actual_first_leg), (1, 1))
        exit_row = status['inventory_coverage']['annual_exit_summaries'][0]
        self.assertEqual((exit_row['latched_exit_pairs'], exit_row['early_trigger_pairs']), (1, 1))

    def test_hold_smoke_not_goal_or_win_evidence(self):
        directory, status = self.write(guarded_fixture(hold=True), coverage_config(report_scope='smoke'))
        self.assertFalse(status['goal_achieved'])
        text = (directory/'final_report.md').read_text()
        self.assertIn('工程烟测', text.splitlines()[0])
        self.assertIn('没有可评价的做T胜率', text)
        self.assertTrue(pd.read_csv(directory/'exit_decisions.csv').empty)
        self.assertTrue(pd.read_csv(directory/'first_leg_exit_side_audit.csv').empty)

    def test_pending_inventory_loss_preserved(self):
        _, status = self.write(guarded_fixture(pending=True, net=-50., reason='stop_loss'))
        row = status['annual_results'][0]
        self.assertEqual((row['inventory_mismatch_days'], row['pending_pairs'], row['complete_t_pairs']), (1, 1, 0))
        self.assertEqual(row['incremental_marked_pnl'], -50.)
        self.assertFalse(row['profit_target_met'])
        self.assertFalse(status['goal_achieved'])

    def test_manual_cost_remains_in_whole_original_cost_denominator(self):
        result = guarded_fixture(net=2000.)
        manual = result['baseline_events'].iloc[0].copy()
        manual[['base_order_type', 'shares', 'notional', 'fee']] = ['manual', 500, 5000., 1.]
        result['baseline_events'] = pd.concat([result['baseline_events'], manual.to_frame().T], ignore_index=True)
        _, status = self.write(result)
        row = status['annual_results'][0]
        self.assertEqual(row['all_reference_acquisition_cost'], 35003.)
        self.assertAlmostEqual(row['holding_cost_reduction_rate'], 2000/35003.)

    def test_reject_wrong_version_loss_coverage_horizon_and_guard(self):
        changes = [{'research_version': 'inventory_robust_t_v17'}, {'training_loss': 'huber'},
                   {'feature_min_coverage': .75}, {'feature_min_coverage': True}, {'feature_min_coverage': float('nan')},
                   {'source_feature_min_coverage': .70}, {'source_feature_min_coverage': float('inf')},
                   {'horizons': [20]}, {'forecast_target': 'same_day_1455'}, {'recovery_key': 'close'},
                   {'frozen_t_models_reused': True}, {'learning_model_profiles': ['huber_depth4', 'huber_depth6']},
                   {'t_recheck_exit_side_at_execution': False}, {'source_learning_report_dir': ''}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.write(guarded_fixture(), coverage_config(**change))

    def test_reject_invalid_rule_or_missing_guard_or_cancelled_fill(self):
        for mode in ('huber', 'bad_fraction', 'hold_with_fill', 'missing_guard', 'cancelled_fill'):
            result = guarded_fixture()
            if mode == 'huber': result['selected_rule']['profile'] = 'huber_depth4'
            elif mode == 'bad_fraction': result['selected_rule']['fraction'] = True
            elif mode == 'hold_with_fill': result['selected_rule']['enabled'] = False
            elif mode == 'missing_guard': del result['first_leg_exit_side_audit']
            else:
                result['first_leg_exit_side_audit'].loc[0, ['cancelled', 'reason', 'opposite_quote_reason']] = [True, 'execution_exit_side_price_limit', 'price_limit']
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'v18|37候选'):
                self.write(result)

    def test_all_horizons_lunch_and_invalid_clock(self):
        starts = pd.to_datetime(['2024-01-03 10:01', '2024-01-03 11:01', '2024-01-03 13:06'])
        ends = pd.to_datetime(['2024-01-03 10:21', '2024-01-03 13:11', '2024-01-03 14:06'])
        row = reporting._horizon_summary(2024, pd.DataFrame({'first_execution_time': starts, 'original_deadline': ends}))
        self.assertEqual((row['planned_h20_pairs'], row['planned_h40_pairs'], row['planned_h60_pairs']), (1, 1, 1))
        for start, end in [('10:01', '10:31'), ('11:01', '11:41'), ('10:01', '14:55'), ('14:00', '15:00')]:
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, '20/40/60'):
                reporting._horizon_summary(2024, pd.DataFrame({'first_execution_time': [f'2024-01-03 {start}'],
                                                               'original_deadline': [f'2024-01-03 {end}']}))

    def test_existing_year_tables_and_validation_identity_preserved(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        folder = directory/'year_2024'
        folder.mkdir()
        result, original = guarded_fixture(reason='hard_deadline'), {}
        for name in ('exit_decisions', 'first_leg_exit_side_audit'):
            result[name].to_csv(folder/f'{name}.csv', index=False)
            original[name] = (folder/f'{name}.csv').read_bytes()
        validation = pd.DataFrame({'candidate_index': [0, 1], 'selected': [False, True], 'incremental_net_pnl': [0., -10.]})
        before = validation.copy(deep=True)
        _, status = self.write(result, directory=directory, validation=validation)
        for name, contents in original.items():
            self.assertEqual((folder/f'{name}.csv').read_bytes(), contents)
        pd.testing.assert_frame_equal(validation, before)
        pd.testing.assert_frame_equal(pd.read_csv(directory/'validation_rule_metrics.csv'), validation)
        self.assertEqual(status['inventory_coverage']['annual_exit_summaries'][0]['hard_deadline_trigger_pairs'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
