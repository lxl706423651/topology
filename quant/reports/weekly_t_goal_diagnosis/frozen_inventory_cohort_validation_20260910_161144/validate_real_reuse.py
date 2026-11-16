"""一次真实首折冻结核心复用；完整同刻增强、不规划、不拟合、不读outer账户。"""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import pandas as pd
from quant_backtest.inventory_t_frozen_cohort import (
    calculate_frozen_file_sha256, load_frozen_inventory_core_cohort)
from quant_backtest.weekly_t_timing import add_timing_features
from run_weekly_t_active import load_frozen_market_data


def record_input(path, inputs):
    """与runner一致绑定当前读取文件，另比对独审缓存输入身份。"""
    path = Path(path).resolve()
    inputs[str(path)] = {"sha256": calculate_frozen_file_sha256(path)}


start = time.monotonic()
out = Path(__file__).resolve().parent
source = ROOT / "reports/inventory_learning_t/run_20260910_151232_841321"
cfg = json.loads((source / "resolved_config.json").read_text())
fold = json.loads((source / "year_2023/split.json").read_text())
audit_path = source / "independent_learning_audit/run_20260910_153539_071695/verification.json"
audit = json.loads(audit_path.read_text())
cfg["source_learning_audit"] = str(audit_path)
cfg["source_learning_audit_sha256"] = calculate_frozen_file_sha256(audit_path)
inputs = {}
market_output = out / "current_market_read"
market_output.mkdir()
print("loading original cache, whole 2024/2025 cross sections", flush=True)
universe, intraday, daily = load_frozen_market_data({**cfg, "test_years": [2024, 2025]}, market_output, inputs)
del universe, daily
intraday = add_timing_features(intraday, cfg)
for name, detail in inputs.items():
    assert detail["sha256"] == audit["inputs_sha256"][name], name
print("current rows", len(intraday), "symbols", intraday.symbol.nunique(), flush=True)
core, metadata = load_frozen_inventory_core_cohort(intraday, fold, cfg, source, out, inputs, record_input)
assert len(core) == 206488
assert sum(row["rows"] for row in metadata["core_years_detail"]) == len(core)
assert not any(name.startswith(("execution_", "recovery_", "label_", "target_")) for name in core)
path = out / "complete_current_core.pkl"
core.to_pickle(path)
summary = {"passed": True, "scope": "first_fold_complete_two_core_years_data_reuse_only",
           "core_years": fold["core_years"], "rows": len(core), "columns": len(core.columns),
           "missing_current_source_rows": int(core.feature_source_start.isna().sum()),
           "original_matrix_checks": metadata["old_training_matrix_checks"],
           "complete_current_core_sha256": metadata["full_current_core_sha256"],
           "complete_current_core_pickle_sha256": calculate_frozen_file_sha256(path),
           "cohort_lineage_sha256": metadata["cohort_lineage_sha256"],
           "data_planners_run": 0, "models_fit": 0, "outer_returns_read": False,
           "raw_csv_price_checks_repeated": False, "source_cache_bytes_equal_full_audit": True,
           "future_labels_used_to_select_opportunities": False,
           "all_read_inputs_sha256": {name: value["sha256"] for name, value in inputs.items()},
           "elapsed_seconds": time.monotonic()-start}
(out / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: summary[key] for key in ["passed", "rows", "columns", "data_planners_run", "models_fit", "elapsed_seconds"]}), flush=True)
