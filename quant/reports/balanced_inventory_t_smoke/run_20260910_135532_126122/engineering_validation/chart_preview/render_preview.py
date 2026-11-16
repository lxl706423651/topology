"""仅从已保存CSV绘制新预览；逐文件证明原烟测产物没有改变。"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
PROJECT = SOURCE.parents[2]
sys.path.insert(0, str(PROJECT))
from quant_backtest.balanced_t_reporting import _plot_balanced


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


protocol = json.loads((HERE / "protocol.json").read_text())
cfg = json.loads((SOURCE / "resolved_config.json").read_text())
annual = pd.read_csv(SOURCE / "annual_summary.csv", float_precision="round_trip")
saved = annual.copy(deep=True)
entries = [(int(year), {"equity_curve": pd.read_csv(
    SOURCE / f"year_{year}/test_equity_curve.csv", float_precision="round_trip")})
    for year in annual.test_year]
curve_before = [result["equity_curve"].copy(deep=True) for _, result in entries]
_plot_balanced(HERE, cfg, annual, entries)
pd.testing.assert_frame_equal(saved, annual, check_exact=True)
for before, (_, result) in zip(curve_before, entries):
    pd.testing.assert_frame_equal(before, result["equity_curve"], check_exact=True)
for relative, digest in protocol["original_files_sha256"].items():
    assert sha(SOURCE / relative) == digest, relative
reporter = PROJECT / "quant_backtest/balanced_t_reporting.py"
(HERE / "balanced_t_reporting_after.py").write_bytes(reporter.read_bytes())
result = {"passed": True, "mode": "render_saved_results_only", "account_rerun": False,
          "original_files_verified_unchanged": len(protocol["original_files_sha256"]),
          "annual_and_curve_frames_unchanged": True,
          "old_reporter_sha256": protocol["old_reporter_sha256"],
          "new_reporter_sha256": sha(reporter),
          "scope": cfg["report_scope"], "smoke_days": cfg["smoke_days"],
          "max_symbols": cfg["max_symbols"], "test_years": cfg["test_years"],
          "output_sha256": {name: sha(HERE / name) for name in
              ("test_return_curve.png", "测试集收益曲线.png", "render_preview.py")}}
(HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
