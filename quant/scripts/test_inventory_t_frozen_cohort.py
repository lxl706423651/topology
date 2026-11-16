"""冻结完整持仓队列复用的合成反例；只写临时来源/审计，不运行市场或账户。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.inventory_t_frozen_cohort import (
    BASE_KEYS, CONTEXT_CONFIG_KEYS, LINEAGE_COLUMNS, OLD_CONTEXT,
    calculate_frozen_file_sha256, calculate_frozen_frame_sha256,
    load_frozen_inventory_core_cohort,
)


def write_json(path, value):
    """写合成审计/模型身份；不构建任何模型。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def record_input(path, inputs):
    """正式runner两参数输入绑定合同。"""
    inputs[str(Path(path).resolve())] = {"sha256": calculate_frozen_file_sha256(path)}


class FrozenCohortTests(unittest.TestCase):
    """独立构造原完整机会和更小旧矩阵，检验后者不能删掉前者。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.folder = self.source / "year_2023/training_cohort/inventory_learning_cohort"
        self.fold = {"test_year": 2023, "inner_year": 2026, "core_years": [2024, 2025]}
        self.cfg = {name: 1 for name in CONTEXT_CONFIG_KEYS}
        self.cfg.update(initial_cash=1000000, base_fraction=.85, target_position_notional=15000,
                        min_position_notional=10000, max_position_notional=20000, lot_size=100,
                        source_training_report_dir="original_oof_source", embargo_days=7,
                        start_date="2023-01-01", end_date="2026-04-30", signal_times=["10:00", "14:00"],
                        horizons=[20], learning_model_profiles=["l2_depth4"], manual_buys_csv=None)
        import hashlib
        base_sha = hashlib.sha256(json.dumps({k: self.cfg[k] for k in BASE_KEYS}, sort_keys=True).encode()).hexdigest()
        write_json(self.source / "resolved_config.json", self.cfg)
        write_json(self.source / "runtime_integrity_audit.json", {"all_unchanged": True})
        snapshot = self.source / "source_snapshot/original.py"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text('"""冻结来源的合成接口。"""\n')
        write_json(self.source / "source_hashes.json", {"original.py": calculate_frozen_file_sha256(snapshot)})
        write_json(self.source / "year_2023/split.json", self.fold)
        parts, lines, summaries, checks = [], [], [], []
        for year in [2024, 2025]:
            times = pd.to_datetime([f"{year}-03-12 10:00", f"{year}-03-12 14:00", f"{year}-03-13 10:00"])
            current = pd.DataFrame({"datetime": times, "symbol": "SH.600000",
                                    "feature_source_start": [pd.Timestamp(year, 3, 11), pd.NaT, pd.Timestamp(year, 3, 12)],
                                    "date": times.normalize(), "f_price": np.array([.1, .2, .3], dtype="float32"),
                                    "f_history_days": np.array([20, 20, 21], dtype="float32"),
                                    "execution_price": [10., np.nan, 10.], "recovery_price": [np.nan, 11., 12.],
                                    "trade_week_complete": False})
            for name in OLD_CONTEXT:
                current[name] = np.float64(.5)
            parts.append(current)
            own = current.feature_source_start
            line = current[["datetime", "symbol"]].copy()
            for name in ["entry_feature_source_start", "entry_rank_source_start", "portfolio_cash_source_start"]:
                line[name] = pd.Timestamp(year, 2, 1)
            line["intraday_feature_source_start"] = own
            line["feature_source_start"] = np.minimum(own.to_numpy(), line.portfolio_cash_source_start.to_numpy())
            line["base_entry_datetime"] = pd.Timestamp(year, 3, 11, 9, 30)
            line["baseline_opening_shares"] = 1500
            line["frozen_base_shares"] = 1500
            line["source_model_training_years"] = json.dumps([2025 if year == 2024 else 2024])
            line["source_model_sha256"] = "a" * 64
            line["cohort_target_year"] = year
            line = line[LINEAGE_COLUMNS]
            for name in line:
                if pd.api.types.is_datetime64_any_dtype(line[name]):
                    line[name] = line[name].astype("datetime64[ns]")
            lines.append(line)
            directory = self.folder / f"core_{year}"
            directory.mkdir(parents=True)
            line.to_csv(directory / "cohort_lineage.csv", index=False)
            inventory = pd.DataFrame({"date": pd.to_datetime([f"{year}-03-11", f"{year}-03-12", f"{year}-03-13"]),
                                      "symbol": "SH.600000", "baseline_opening_shares": [0, 1500, 1500]})
            inventory.to_csv(directory / "daily_inventory_audit.csv", index=False)
            summary = {"target_year": year, "selector_training_years": [2025 if year == 2024 else 2024],
                       "base_config_sha256": base_sha, "training_opportunity_rows": len(line),
                       "cohort_lineage_sha256": calculate_frozen_frame_sha256(line)}
            write_json(directory / "metadata.json", summary)
            summaries.append(summary)
            checks.append({"fold": 2023, "core_year": year, "rows": 3, "all_ranking_and_cash_sources_propagated": True})
        self.intraday = pd.concat(parts, ignore_index=True)
        self.lineage = pd.concat(lines, ignore_index=True)
        self.meta = {"core_years": [2024, 2025], "excluded_years": [2023, 2026], "embargo_days": 7,
                     "selectors": summaries, "training_opportunity_rows": 6,
                     "labels_attached": False, "future_fill_or_return_filter": False, "core_sampling_performed": False,
                     "baseline_has_no_t_and_no_manual_buys": True, "full_market_features_before_selection": True,
                     "all_core_years_joint_fit_only_not_independent_calibration": True}
        write_json(self.folder / "cohort_metadata.json", self.meta)
        self.model = self.source / "year_2023/models/h20_l2_depth4"
        self.model.mkdir(parents=True)
        # 缺旧未来标签或来源的4行不在矩阵中，但在完整队列中必须保留。
        matrix = self.lineage.loc[[2, 5], ["datetime", "symbol", "feature_source_start", "cohort_target_year", "baseline_opening_shares"]].copy()
        matrix["f_price"] = self.intraday.loc[[2, 5], "f_price"].to_numpy()
        matrix.to_pickle(self.model / "training_inputs.pkl")
        write_json(self.model / "model_metadata.json", {"train_years": [2024, 2025], "excluded_years": [2023, 2026],
                   "features": ["f_price"], "training_inputs_file_sha256": calculate_frozen_file_sha256(self.model / "training_inputs.pkl")})
        self.audit = {"passed": True, "complete": True, "all_read_inputs_unchanged": True,
                      "research_version": "inventory_learning_t_v14", "core_cohorts": checks,
                      "models": [{"path": "year_2023/models/h20_l2_depth4", "passed": True,
                                  "independently_rebuilt_training_inputs": True}]}
        self.audit_path = self.source / "independent_learning_audit/run_1/verification.json"
        self.anchor()

    def anchor(self):
        """仅供恶意/边界夹具重新签一份合成审计，以测试更深层合同。"""
        self.audit["inputs_sha256"] = {str(p.resolve()): calculate_frozen_file_sha256(p)
                                        for p in self.source.rglob("*") if p.is_file() and p != self.audit_path}
        write_json(self.audit_path, self.audit)

    def load(self, current=None, cfg=None, suffix="out"):
        """真实调用待测入口，输出限临时目录。"""
        return load_frozen_inventory_core_cohort(self.intraday if current is None else current,
            self.fold, self.cfg if cfg is None else cfg, self.source, self.root / suffix, {}, record_input)

    def test_full_keys_preserve_missing_old_labels_provenance_dtype_and_cash_sources(self):
        before = self.intraday.copy(deep=True)
        result, meta = self.load()
        self.assertEqual(len(result), 6)
        self.assertEqual(len(meta["old_training_matrix_checks"]), 1)
        self.assertEqual(meta["old_training_matrix_checks"][0]["rows"], 2)
        pd.testing.assert_frame_equal(result[LINEAGE_COLUMNS], self.lineage, check_exact=True)
        self.assertEqual(result.f_price.dtype, np.dtype("float32"))
        self.assertEqual(result.feature_source_start.isna().sum(), 2)
        self.assertFalse(set(OLD_CONTEXT + ["execution_price", "recovery_price", "trade_week_complete"]) & set(result))
        pd.testing.assert_frame_equal(before, self.intraday, check_exact=True)

    def test_future_fields_changed_or_removed_do_not_change_core(self):
        first, _ = self.load()
        second = self.intraday.drop(columns=["execution_price", "recovery_price", "trade_week_complete"])
        second["target_return"] = np.inf
        actual, _ = self.load(second, suffix="other")
        pd.testing.assert_frame_equal(first, actual, check_exact=True)

    def test_missing_old_label_row_is_still_required_in_current_input(self):
        with self.assertRaisesRegex(ValueError, "缺少冻结机会键"):
            self.load(self.intraday.drop(index=0))

    def test_current_source_changed_even_outside_old_matrix_rejected(self):
        data = self.intraday.copy()
        data.loc[0, "feature_source_start"] = pd.Timestamp("2024-03-10")
        with self.assertRaisesRegex(ValueError, "当前原始来源"):
            self.load(data)

    def test_feature_value_or_dtype_change_rejected(self):
        for dtype in (False, True):
            with self.subTest(dtype=dtype):
                data = self.intraday.copy()
                if dtype:
                    data["f_price"] = data.f_price.astype("float64")
                else:
                    data.loc[2, "f_price"] = np.float32(.4)
                with self.assertRaisesRegex(ValueError, "旧矩阵当前值/dtype"):
                    self.load(data)

    def test_incomplete_or_changed_source_audit_rejected(self):
        self.audit["complete"] = False
        write_json(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, "完整独立审计"):
            self.load()

    def test_lineage_file_tampering_rejected_before_use(self):
        path = self.folder / "core_2024/cohort_lineage.csv"
        path.write_text(path.read_text().replace("1500", "1400"))
        with self.assertRaisesRegex(ValueError, "SHA 改变"):
            self.load()

    def test_reanchored_missing_lineage_row_cannot_hide_old_inventory_opportunity(self):
        path = self.folder / "core_2024/cohort_lineage.csv"
        line = self.lineage.iloc[:3].drop(index=0)
        line.to_csv(path, index=False)
        m = self.meta["selectors"][0]
        m.update(training_opportunity_rows=2, cohort_lineage_sha256=calculate_frozen_frame_sha256(line))
        self.meta["training_opportunity_rows"] = 5
        self.audit["core_cohorts"][0]["rows"] = 2
        write_json(self.folder / "core_2024/metadata.json", m)
        write_json(self.folder / "cohort_metadata.json", self.meta)
        self.anchor()
        with self.assertRaisesRegex(ValueError, "全部旧股机会全集"):
            self.load()

    def test_core_identity_and_economic_config_cannot_change(self):
        with self.assertRaisesRegex(ValueError, "配置改变"):
            self.load(cfg={**self.cfg, "base_fraction": .95})
        self.fold["inner_year"] = 2027
        with self.assertRaisesRegex(ValueError, "本折身份"):
            self.load()

    def test_duplicate_current_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "键缺失或重复"):
            self.load(pd.concat([self.intraday, self.intraday.iloc[:1]], ignore_index=True))

    def test_manual_configuration_does_not_rewrite_frozen_training_inventory(self):
        core, meta = self.load(cfg={**self.cfg, "manual_buys_csv": "never_read_this_external_file.csv"})
        pd.testing.assert_frame_equal(core[LINEAGE_COLUMNS], self.lineage, check_exact=True)
        self.assertTrue(meta["manual_buys_ignored_for_training"])

    def test_existing_output_not_overwritten_and_explicit_audit_sha_checked(self):
        self.load()
        with self.assertRaises(FileExistsError):
            self.load()
        with self.assertRaisesRegex(ValueError, "指定独审 SHA"):
            self.load(cfg={**self.cfg, "source_learning_audit_sha256": "0" * 64}, suffix="sha")


if __name__ == "__main__":
    unittest.main(verbosity=2)
