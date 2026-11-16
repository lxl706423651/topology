"""用微型临时来源链验证固定底仓研究的年度/周模型身份及因果入场。

仅创建本测试拥有的JSON、CSV和模型字节替身，不训练模型、不执行账户、不读取
真实大行情；从quant环境运行。拒绝案例同时覆盖内容身份及内容自洽但错年的情况。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.balanced_t_data import read_balanced_fold, read_balanced_selections


def write_json(path: Path, value) -> None:
    """只写当前测试的临时JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def file_sha(path: Path) -> str:
    """独立计算夹具字节身份，避免用被测函数产生预期值。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BalancedSourceContractsTests(unittest.TestCase):
    """测试v8拆分、v4原始周名单及v11审计输入链相互一致。"""

    def setUp(self):
        """构建一个2023外层/2026内层、2024/25核心的微型审计链。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.v8 = self.root / "v8"
        self.v4 = self.root / "v4"
        self.v11 = self.root / "v11"
        self.cfg = {
            "source_training_report_dir": str(self.v8),
            "frozen_selection_report_dir": str(self.v4),
            "source_audit_report_dir": str(self.v11),
            "test_years": [2023, 2024, 2025, 2026],
            "max_symbols": 0,
        }
        self.fold = {"test_year": 2023, "train_years": [2024, 2025, 2026],
                     "inner_year": 2026, "core_years": [2024, 2025],
                     "forward_only": False, "outer_t_refit": False}
        self.split_path = self.v8 / "year_2023/split.json"
        write_json(self.split_path, self.fold)
        self.audit_path = self.v8 / "independent_residual_audit.json"
        self.audit = {"passed": True, "complete": True, "frozen_contexts": {}}
        self.paths = {}
        for scope in ("validation", "test"):
            year = 2026 if scope == "validation" else 2023
            entry = pd.Timestamp(f"{year}-01-05")
            # 日期只模拟合法顺序；市场日资格由传入的微型日历明确给定。
            frame = pd.DataFrame({
                "symbol": ["SH.600000", "SZ.000001"], "rank": [2, 1],
                "trade_week": [entry, entry + pd.Timedelta(days=7)],
                "entry_date": [entry, entry + pd.Timedelta(days=7)],
                "exit_date": [entry + pd.Timedelta(days=4), entry + pd.Timedelta(days=11)],
                "signal_datetime": [entry - pd.Timedelta(days=3) + pd.Timedelta(hours=15),
                                    entry + pd.Timedelta(days=4, hours=15)],
                "score": [.8, .9],
            })
            filename = "validation_weekly_selections.csv" if scope == "validation" else "weekly_selections.csv"
            selection = self.v4 / "year_2023" / filename
            selection.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(selection, index=False)
            model_name = "inner_weekly_model" if scope == "validation" else "weekly_model"
            model = self.v4 / "year_2023" / model_name
            model.mkdir()
            bundle = model / "model_bundle.joblib"
            bundle.write_bytes(f"model bytes for {scope}".encode())
            training_years = self.fold["core_years"] if scope == "validation" else self.fold["train_years"]
            excluded_years = sorted(set(self.cfg["test_years"]) - set(training_years))
            metadata = model / "model_metadata.json"
            write_json(metadata, {"train_years": training_years, "excluded_years": excluded_years})
            self.paths[scope] = {"selection": selection, "bundle": bundle, "metadata": metadata,
                                 "frame": frame, "days": pd.DatetimeIndex(frame.entry_date)}
            self.audit["frozen_contexts"][f"year_2023/{scope}"] = {
                "passed": True, "source_selection_sha256": file_sha(selection),
                "context": {"source_model_path": str(bundle),
                            "source_model_sha256": file_sha(bundle),
                            "source_metadata_sha256": file_sha(metadata),
                            "source_model_training_years": training_years,
                            "excluded_years": excluded_years},
            }
        self.refresh_audited_identity()

    def refresh_audited_identity(self):
        """生成与当前夹具一致的上游字节链，用于隔离语义错误与普通字节错误。"""
        for scope, paths in self.paths.items():
            entry = self.audit["frozen_contexts"][f"year_2023/{scope}"]
            entry["source_selection_sha256"] = file_sha(paths["selection"])
            entry["context"]["source_model_sha256"] = file_sha(paths["bundle"])
            entry["context"]["source_metadata_sha256"] = file_sha(paths["metadata"])
        write_json(self.audit_path, self.audit)
        all_paths = [self.split_path, self.audit_path]
        for paths in self.paths.values():
            all_paths.extend(paths[k] for k in ("selection", "bundle", "metadata"))
        write_json(self.v11 / "input_sources.json", {
            str(path.resolve()): {"sha256": file_sha(path), "size_bytes": path.stat().st_size}
            for path in all_paths})

    def read_selected(self, scope="validation", fold=None, days=None):
        """直接调用被测来源读取，不mock任何身份或时间校验。"""
        return read_balanced_selections(self.cfg, self.fold if fold is None else fold, scope,
            self.paths[scope]["days"] if days is None else days, {})

    def test_correct_inner_outer_sources_and_inputs_recorded(self):
        inputs = {}
        actual = read_balanced_fold(self.cfg, 2023, inputs)
        self.assertEqual(actual["core_years"], [2024, 2025])
        self.assertFalse(actual["t_model_fit"])
        self.assertIn(str(self.split_path.resolve()), inputs)
        for scope in ("validation", "test"):
            with self.subTest(scope=scope):
                result = self.read_selected(scope, actual)
                self.assertEqual(result.symbol.tolist(), ["SH.600000", "SZ.000001"])
                self.assertEqual(result.entry_date.dt.year.unique().tolist(), [2026 if scope=="validation" else 2023])

    def test_self_consistent_changed_fold_rejected_by_frozen_sha(self):
        altered = {**self.fold, "inner_year": 2025, "core_years": [2024, 2026]}
        self.assertEqual(sorted(altered["core_years"]+[altered["inner_year"]]), altered["train_years"])
        write_json(self.split_path, altered)
        with self.assertRaises(ValueError):
            read_balanced_fold(self.cfg, 2023, {})

    def test_source_selection_mutation_rejected(self):
        paths = self.paths["validation"]
        frame = paths["frame"].copy()
        frame.loc[0, "symbol"] = "SH.600001"
        frame.to_csv(paths["selection"], index=False)
        with self.assertRaises(ValueError):
            self.read_selected()

    def test_model_bytes_and_metadata_bytes_must_match_audited_sha(self):
        for key in ("bundle", "metadata"):
            with self.subTest(key=key):
                path = self.paths["validation"][key]
                original = path.read_bytes()
                path.write_bytes(original+b" ")
                with self.assertRaises(ValueError):
                    self.read_selected()
                path.write_bytes(original)

    def test_self_consistent_wrong_model_training_years_rejected(self):
        paths = self.paths["validation"]
        wrong = {"train_years": [2023, 2025], "excluded_years": [2024, 2026]}
        write_json(paths["metadata"], wrong)
        context = self.audit["frozen_contexts"]["year_2023/validation"]["context"]
        context.update(source_model_training_years=wrong["train_years"], excluded_years=wrong["excluded_years"])
        self.refresh_audited_identity()
        with self.assertRaises(ValueError):
            self.read_selected()

    def test_wrong_excluded_years_rejected_even_when_bytes_match(self):
        paths = self.paths["validation"]
        write_json(paths["metadata"], {"train_years": [2024, 2025], "excluded_years": [2023]})
        self.audit["frozen_contexts"]["year_2023/validation"]["context"]["excluded_years"] = [2023]
        self.refresh_audited_identity()
        with self.assertRaises(ValueError):
            self.read_selected()

    def test_future_signal_cannot_enter_at_first_minute(self):
        frame = self.paths["validation"]["frame"].copy()
        frame.loc[0, "signal_datetime"] = frame.loc[0, "entry_date"] + pd.Timedelta(hours=9, minutes=31)
        frame.to_csv(self.paths["validation"]["selection"], index=False)
        self.refresh_audited_identity()
        with self.assertRaises(ValueError):
            self.read_selected()

    def test_later_week_not_imported_into_earlier_calendar(self):
        days = self.paths["validation"]["days"][:1]
        result = self.read_selected(days=days)
        self.assertEqual(result.symbol.tolist(), ["SH.600000"])
        self.assertEqual(result.entry_date.tolist(), list(days))

    def test_v8_derived_csv_cannot_replace_original_v4_selection(self):
        decoy = self.v8 / "year_2023/validation_weekly_selections.csv"
        decoy.write_text("symbol,rank\nSH.600999,1\n")
        result = self.read_selected()
        self.assertEqual(result.symbol.tolist(), ["SH.600000", "SZ.000001"])

    def test_current_fold_does_not_reuse_wrong_weekly_scope(self):
        # 字节来自正确v4内层模型，但调用方把另一个核心年集交给它，也须拒绝。
        altered = deepcopy(self.fold)
        altered.update(core_years=[2024, 2026], inner_year=2025)
        with self.assertRaises(ValueError):
            self.read_selected(fold=altered)


if __name__ == "__main__":
    unittest.main()
