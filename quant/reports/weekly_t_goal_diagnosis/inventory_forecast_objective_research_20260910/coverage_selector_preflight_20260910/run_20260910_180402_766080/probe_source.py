"""仅重建已审计完整核心并比较一次0.80/0.70特征选择，不拟合或读取评价账户。"""
from __future__ import annotations

from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from quant_backtest.balanced_t_data import record_balanced_input
from quant_backtest.inventory_t_frozen_cohort import load_frozen_inventory_core_cohort
from quant_backtest.inventory_t_learning_models import attach_learning_labels, calculate_learning_frame_sha256
from quant_backtest.weekly_t_models import select_training_rows, select_training_features
from quant_backtest.weekly_t_timing import add_timing_features
from quant_backtest.weekly_t_timing_data import load_labels_only
from scripts.run_weekly_t_active import load_frozen_market_data


def exact(left, right, description):
    """精确核对真实值、列序和dtype，避免改变核心样本或只验证自报摘要。"""
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                 check_dtype=True, check_exact=True, obj=description)


def no_fit(*args, **kwargs):
    """进程内禁止误调用任何LightGBM拟合入口。"""
    raise AssertionError("本预检禁止拟合模型")


def main():
    """只对四折已冻结core进行0.70一次对照，保留原.80控制及全部来源。"""
    import lightgbm
    lightgbm.train = no_fit
    lightgbm.LGBMModel.fit = no_fit
    lightgbm.LGBMRegressor.fit = no_fit
    source = ROOT / "reports/inventory_learning_t/run_20260910_151232_841321"
    robust = ROOT / "reports/inventory_robust_t/run_20260910_172815_737873"
    out = Path(__file__).resolve().parent / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out / "probe.py")
    inputs, results = {}, []
    try:
        def js(path):
            record_balanced_input(path, inputs)
            return json.loads(Path(path).read_text())
        cfg = js(source / "resolved_config.json")
        robust_cfg = js(robust / "resolved_config.json")
        cfg.update(source_learning_audit=robust_cfg["source_learning_audit"],
                   source_learning_audit_sha256=robust_cfg["source_learning_audit_sha256"])
        assert cfg["feature_min_coverage"] == .8 and cfg["feature_max_correlation"] == .95
        assert cfg["feature_selection_sample_rows"] == 20000 and cfg["random_state"] == 42
        assert cfg["t_max_train_rows"] == 0 and cfg["horizons"] == [20, 40, 60]
        folds = [js(source / f"year_{year}" / "split.json") for year in [2023, 2024, 2025, 2026]]
        years = sorted({year for fold in folds for year in fold["core_years"]})
        assert years == [2023, 2024, 2025]
        frozen = js(robust / "source_hashes.json")
        for name, digest in frozen.items():
            record_balanced_input(ROOT / name, inputs, digest)
            record_balanced_input(robust / "source_snapshot" / name, inputs, digest)
        known = js(Path(cfg["source_audit_report_dir"]) / "input_sources.json")
        for name in ("weekly", "intraday", "daily"):
            path = (Path(cfg["market_cache_dir"]) / f"{name}.pkl").resolve()
            record_balanced_input(path, inputs, known[str(path)]["sha256"])
        print(f"输出目录：{out}；只重建核心年份{years}，禁止fit，coverage固定.70一次", flush=True)
        universe, intraday, daily = load_frozen_market_data({**cfg, "test_years": years}, out, inputs)
        assert set(intraday.datetime.dt.year.unique()) == set(years)
        intraday = add_timing_features(intraday, cfg)
        del universe, daily
        gc.collect()
        cores = {}
        for fold in folds:
            year = fold["test_year"]
            folder = out / f"year_{year}"
            folder.mkdir()
            cores[year], audit = load_frozen_inventory_core_cohort(intraday, fold, cfg, source,
                folder / "training_cohort", inputs, record_balanced_input)
            assert set(cores[year].datetime.dt.year.unique()) == set(fold["core_years"])
            print(f"[{year}] 完整未联标签core恢复{len(cores[year]):,}行，原矩阵/来源反证通过", flush=True)
        del intraday
        gc.collect()
        labels = load_labels_only(cfg, out, inputs, record_balanced_input)
        symbols = {symbol for core in cores.values() for symbol in core.symbol.unique()}
        labels = labels.loc[labels.datetime.dt.year.isin(years) & labels.symbol.isin(symbols)].copy()
        print(f"标签仅保留核心年份及{len(symbols)}只核心股票，共{len(labels):,}条；未读评价预测/账户", flush=True)
        for fold in folds:
            year = fold["test_year"]
            excluded = sorted([year, fold["inner_year"]])
            for horizon in [20, 40, 60]:
                folder = out / f"year_{year}" / f"h{horizon}"
                folder.mkdir()
                joined = attach_learning_labels(cores[year], labels, horizon)
                training, split, split_audit = select_training_rows(joined, fold["core_years"], cfg, "intraday", excluded)
                target = training[["target_return"]].rename(columns={"target_return": "return"})
                features80, audit80 = select_training_features(training, target, cfg, "intraday")
                features70, audit70 = select_training_features(training, target, {**cfg, "feature_min_coverage": .7}, "intraday")
                common = [c for c in features80 if c in features70]
                matrix80 = training[features80].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                matrix70 = training[features70].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                exact(matrix80[common], matrix70[common], "共同列原值")
                source_checks = []
                for profile in ["l2_depth4", "l2_depth6"]:
                    old = source / f"year_{year}" / "models" / f"h{horizon}_{profile}"
                    meta = js(old / "model_metadata.json")
                    assert meta["features"] == features80 and meta["training_rows"] == len(training)
                    assert meta["train_years"] == fold["core_years"] and meta["excluded_years"] == excluded
                    for key in split:
                        assert meta[key] == split[key], (year, horizon, key)
                    for name in ["feature_manifest.csv", "training_split_audit.csv", "training_inputs.pkl", "training_labels.csv"]:
                        record_balanced_input(old / name, inputs)
                    exact(audit80, pd.read_csv(old / "feature_manifest.csv", float_precision="round_trip"), ".80原manifest")
                    exact(split_audit, pd.read_csv(old / "training_split_audit.csv", float_precision="round_trip"), "原split审计")
                    matrix_saved = pd.read_pickle(old / "training_inputs.pkl")
                    rebuilt = training[["datetime", "symbol", "feature_source_start", "cohort_target_year", "baseline_opening_shares"]].copy()
                    for name in features80:
                        rebuilt[name] = matrix80[name].to_numpy()
                    exact(rebuilt, matrix_saved, "原实际拟合输入")
                    raw_labels = pd.read_csv(old / "training_labels.csv", float_precision="round_trip")
                    expected_labels = training[[name for name in raw_labels if name != "fitted"]].copy()
                    expected_labels["fitted"] = True
                    for name in raw_labels:
                        if pd.api.types.is_datetime64_any_dtype(expected_labels[name]):
                            raw_labels[name] = pd.to_datetime(raw_labels[name], format="mixed")
                    exact(expected_labels, raw_labels, "原完整标签与trace")
                    assert calculate_learning_frame_sha256(matrix80) == meta["training_matrix_sha256"]
                    assert calculate_learning_frame_sha256(training[["target_return"]]) == meta["training_target_sha256"]
                    assert calculate_learning_frame_sha256(training[["datetime", "symbol", "feature_source_start"]]) == meta["full_training_rows_sha256"]
                    source_checks.append(str(old))
                audit80.to_csv(folder / "feature_manifest_0p80.csv", index=False)
                audit70.to_csv(folder / "feature_manifest_0p70.csv", index=False)
                split_audit.to_csv(folder / "unchanged_training_split_audit.csv", index=False)
                coverage = training[[name for name in training if name.startswith("f_")]].notna()
                clock = training.datetime.dt.strftime("%H:%M")
                coverage.groupby(clock).agg(["count", "sum", "mean"]).to_csv(folder / "feature_availability_by_clock.csv")
                row = dict(fold=year, core_years=fold["core_years"], excluded_years=excluded, horizon_minutes=horizon,
                    training_rows=len(training), source_models=source_checks, all_source_artifacts_exact=True,
                    features80=features80, features70=features70, added=[c for c in features70 if c not in features80],
                    removed=[c for c in features80 if c not in features70], common_columns=common,
                    common_values_and_dtypes_identical=True, raw_target_and_split_unchanged=True,
                    training_rows_sha256=calculate_learning_frame_sha256(training[["datetime", "symbol", "feature_source_start"]]),
                    target_sha256=calculate_learning_frame_sha256(training[["target_return"]]),
                    matrix80_sha256=calculate_learning_frame_sha256(matrix80), matrix70_sha256=calculate_learning_frame_sha256(matrix70),
                    new_15m_records=audit70.loc[audit70.feature.isin(["f_return_15m", "f_timing_market_return_15m", "f_timing_excess_return_15m"])].to_dict("records"))
                (folder / "selection_comparison.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
                results.append(row)
                print(f"[{year}/h{horizon}] {len(features80)}→{len(features70)}列；新增{row['added']}，移除{row['removed']}；原行/标签/矩阵/隔离完全一致", flush=True)
            del cores[year]
            gc.collect()
        for path, identity in inputs.items():
            record_balanced_input(path, inputs, identity["sha256"])
        result = dict(passed=True, complete=True, coverage_control=.8, coverage_treatment=.7, selection_passes_per_fold_horizon=1,
            models_fitted=0, account_runs=0, validation_or_outer_artifacts_read=False,
            bulk_shared_cache_unpickled=True, feature_enhancement_years_only=years,
            no_core_row_target_change=True, no_production_changes=True, results=results, inputs_sha256=inputs,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"passed": True, "cases": len(results), "out": str(out), "models_fitted": 0}), flush=True)
    except Exception:
        (out / "failure.json").write_text(json.dumps(dict(traceback=traceback.format_exc(), inputs=inputs,
            completed_cases=results), ensure_ascii=False, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
