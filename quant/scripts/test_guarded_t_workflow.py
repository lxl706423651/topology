"""冻结模型复用入口的合同反例；不重复测试原训练器或组合执行器。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from scripts import run_inventory_guarded_t as runner


class GuardedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/"source"
        self.source.mkdir()
        self.original = {
            "research_version": "inventory_learning_t_v14", "configuration_source": "/frozen/v14.yaml",
            "test_years": [2026], "max_symbols": 12, "smoke_days": 15, "candidate_workers": 8,
            "workers": 12, "output_dir": "/frozen/reports", "internal_only": False,
            "commission_rate": .0008, "min_commission": 0., "slippage_rate": .0005,
            "base_fraction": .85, "profit_target_rate": .05, "report_scope": "smoke",
        }
        (self.source/"resolved_config.json").write_text(json.dumps(self.original))
        self.declared = {"research_version": runner.VERSION, "source_learning_report_dir": str(self.source),
            "output_dir": str(self.root/"output"), "t_recheck_exit_side_at_execution": True,
            "candidate_workers": 2, "workers": 4}
        self.config = self.root/"guard.yaml"

    def load(self, *extra):
        self.config.write_text(yaml.safe_dump(self.declared))
        with patch.object(runner, "load_learning_configuration", return_value=copy.deepcopy(self.original)):
            return runner.load_configuration(["--config", str(self.config), *extra])

    def test_source_scope_fees_and_weights_stay_frozen(self):
        cfg = self.load()
        for key in ("commission_rate", "min_commission", "slippage_rate", "base_fraction",
                    "profit_target_rate", "max_symbols", "smoke_days", "report_scope"):
            self.assertEqual(cfg[key], self.original[key])
        self.assertTrue(cfg["frozen_t_models_reused"])
        self.assertFalse(cfg["t_models_fitted_this_run"])
        self.assertEqual(cfg["source_learning_version"], "inventory_learning_t_v14")
        self.assertEqual(cfg["research_version"], "inventory_guarded_t_v15")

    def test_yaml_cannot_reduce_fees_or_turn_off_execution_condition(self):
        self.declared["commission_rate"] = 0.
        with self.assertRaises(ValueError):
            self.load()
        self.declared.pop("commission_rate")
        self.declared["t_recheck_exit_side_at_execution"] = False
        with self.assertRaises(ValueError):
            self.load()

    def test_cannot_claim_an_unfitted_year_or_duplicate_year(self):
        for years in (("2025",), ("2026", "2026")):
            with self.assertRaises(ValueError):
                self.load("--test-years", *years)

    def test_mutated_source_contract_is_rejected(self):
        self.config.write_text(yaml.safe_dump(self.declared))
        changed = {**self.original, "commission_rate": 0.}
        with patch.object(runner, "load_learning_configuration", return_value=changed):
            with self.assertRaises(ValueError):
                runner.load_configuration(["--config", str(self.config)])

    def test_exact_prediction_and_decision_comparison_detects_small_changes(self):
        original = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-05 09:45"]),
            "symbol": ["SH.600000"], "prediction": [.0123456789123456], "action": [1]})
        path = self.root/"predictions.csv"
        original.to_csv(path, index=False)
        inputs = {}
        runner.assert_source_table_equal(original, path, inputs)
        self.assertIn(str(path), inputs)
        changed = original.copy()
        changed.loc[0, "prediction"] += 1e-12
        with self.assertRaises(AssertionError):
            runner.assert_source_table_equal(changed, path, {})
        with self.assertRaises(AssertionError):
            runner.assert_source_table_equal(original.iloc[:0], path, {})

    def test_paired_comparison_preserves_cost_denominator_and_rule_identity(self):
        row = {"candidate_index": 1, "enabled": True, "profile": "l2_depth4", "min_net_edge": 0.,
            "fraction": .5, "exit_mode": "fixed", "incremental_net_pnl": 100.,
            "holding_cost_reduction_rate": .01, "t_pair_count": 10, "completed_t_pair_lots": 100.,
            "t_fee": 10., "t_slippage_cost": 5., "inventory_mismatch_days": 0,
            "recovery_failure_days": 0, "pending_pairs": 0, "all_reference_acquisition_cost": 10000.}
        original = pd.DataFrame([row])
        path = self.root/"source_metrics.csv"
        original.to_csv(path, index=False)
        current = original.assign(incremental_net_pnl=120., holding_cost_reduction_rate=.012,
                                  execution_exit_side_checks=12, execution_exit_side_cancellations=1)
        compared = runner.write_source_comparison(current, path, self.root, {})
        self.assertEqual(compared.incremental_net_pnl_change.iloc[0], 20.)
        changed = current.assign(all_reference_acquisition_cost=9999.)
        with self.assertRaises(AssertionError):
            runner.write_source_comparison(changed, path, self.root, {})
        changed = current.assign(min_net_edge=.001)
        with self.assertRaises(AssertionError):
            runner.write_source_comparison(changed, path, self.root, {})


if __name__ == "__main__":
    unittest.main()
