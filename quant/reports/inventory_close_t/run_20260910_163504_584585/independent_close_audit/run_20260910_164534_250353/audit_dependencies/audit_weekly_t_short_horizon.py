"""独立审计冻结v5短周期T的输入、实际训练行、核心支持域和模型保存一致性。

在quant目录使用quant环境Python运行：
  python scripts/audit_weekly_t_short_horizon.py --report-dir reports/.../run_...
小样本运行附加--smoke；该标记不改变冻结配置或抽样范围。不训练模型，不读取
外层收益，不重新选参；只写同报告下independent_short_horizon_audit.json/md。
先核对全部源代码与缓存身份，复用冻结的因果特征/来源/标签连接，训练资格、
区间相交、拟合抽样和支持域签名均独立重算；要求所审计折全部模型已落盘。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import joblib
import lightgbm
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def file_digest(path: Path) -> str:
    """分块计算文件SHA，避免将大权重或缓存文件整份复制到内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    """按模型的持久签名约定，对原列顺序和内容哈希，不加入索引。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    """只读UTF-8元数据，缺失或未写完时明确失败，不等待或修改输入。"""
    return json.loads(path.read_text(encoding="utf-8"))


def save_audit(report: Path, result: dict) -> None:
    """原子保存本脚本自有的JSON/中文MD，不改CSV、模型或原主报告。"""
    destination = report / "independent_short_horizon_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(destination)
    rows = ["# 短周期T独立模型与输入审计", "",
            f"审计完成：{result.get('complete', False)}；全部检查通过：{result.get('passed', False)}。",
            f"范围：{result.get('mode', '尚未读取配置')}；本次不读取外层收益、不训练或重新选参。", "",
            "|模型|训练行|拟合行|样本SHA|来源SHA|隔离重叠|原生预测一致|", "|---|---:|---:|---|---|---:|---|"]
    for name, audit in result.get("models", {}).items():
        rows.append(f"|{name}|{audit['training_rows']:,}|{audit['fitted_rows']:,}|"
                    f"{audit['training_sample_hash_matches']}|{audit['feature_source_hash_matches']}|"
                    f"{audit['training_overlap_rows']}|{audit['native_joblib_predictions_match']}|")
    rows += ["", f"已核对{len(result.get('models', {}))}个核心模型，"
             f"{len(result.get('supports', {}))}份核心训练支持域，{len(result.get('outer_model_copies', {}))}份外层模型复制。",
             f"折×周期范围{result.get('fold_horizon_scopes', 0)}，实际独特训练范围{result.get('unique_training_scopes', 0)}。",
             "", "strict raw price feature boundary passed仅表示实际量价来源至标签末端没有进入排除年份及其前后隔离区间。",
             "此前外层历史结果已被查看，本轮仍为历史复测，不能据此称为全新独立测试或保证未来收益。",
             "累计history_days仅可用于事前股票资格，本审计要求它未被模型选作特征。",
             "", f"审计脚本SHA256：{result.get('audit_script_sha256', '')}"]
    if result.get("error"):
        rows += ["", "审计中止原因：" + result["error"]]
    markdown = report / "independent_short_horizon_audit.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(markdown)


def verify_sources(report: Path) -> dict:
    """核对报告快照和当前全部运行依赖；版本不一致时禁止使用当前helper重算。"""
    expected = read_json(report / "source_hashes.json")
    if not expected:
        raise ValueError("冻结源码清单为空")
    entries = {}
    for name, signature in expected.items():
        relative = Path(name)
        current = relative if relative.is_absolute() else PROJECT_ROOT / relative
        snapshot = report / "source_snapshot" / relative.name
        entries[name] = {"expected_sha256": signature, "current_sha256": file_digest(current),
                         "snapshot_sha256": file_digest(snapshot)}
        entries[name]["passed"] = entries[name]["current_sha256"] == signature == entries[name]["snapshot_sha256"]
    return {"passed": all(item["passed"] for item in entries.values()), "files": entries}


def verify_cache_inputs(report: Path, cfg: dict) -> tuple[dict, dict, dict]:
    """只读验证源文件stat、主板并集、选股SHA及缓存有效配置，不调用会写缓存的加载器。"""
    from quant_backtest.weekly_t_data import MAINBOARD_PREFIXES

    weekly = read_json(report / "data_manifest.json")
    horizon = read_json(report / "horizon_manifest.json")
    root = Path(cfg["data_root"]).resolve()
    paths = sorted(path for path in root.glob("[0-9][0-9][0-9][0-9]/*.csv")
                   if path.stem.startswith(MAINBOARD_PREFIXES)
                   and pd.Timestamp(cfg["start_date"]).year <= int(path.parent.name) <= pd.Timestamp(cfg["end_date"]).year)
    symbols = sorted({path.stem for path in paths})
    if cfg.get("max_symbols", 0):
        symbols = symbols[:int(cfg["max_symbols"])]
    allowed = set(symbols)
    paths = [path for path in paths if path.stem in allowed]
    stats = [[str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]
    entries = {}
    fingerprint_keys = {
        "weekly": ["schema", "data_root", "files", "symbols", "start_date", "end_date", "signal_times",
                   "known_complete_end_date", "code_sha256", "universe_rule", "price_adjustment"],
        "horizon": ["schema", "data_root", "files", "symbols", "start_date", "end_date", "signal_times",
                    "horizons", "recovery_retry_steps", "incremental_quote_minutes", "frozen_selection_sources",
                    "quote_scope", "code_sha256", "reader_code_sha256"],
    }
    for name, manifest in (("weekly", weekly), ("horizon", horizon)):
        payload = {key: manifest[key] for key in fingerprint_keys[name]}
        reconstructed_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        disk = read_json(Path(manifest["cache_path"]) / "metadata.json")
        report_value = {key: value for key, value in manifest.items() if key != "cache_hit"}
        disk_value = {key: value for key, value in disk.items() if key != "cache_hit"}
        checks = {"source_stats_match": manifest["files"] == stats,
                  "source_symbols_match": manifest["symbols"] == symbols,
                  "source_root_matches": Path(manifest["data_root"]).resolve() == root,
                  "start_date_matches": manifest["start_date"] == str(cfg["start_date"]),
                  "end_date_matches": manifest["end_date"] == str(cfg["end_date"]),
                  "signal_times_match": sorted(manifest["signal_times"]) == sorted(cfg["signal_times"]),
                  "fingerprint_matches": manifest["fingerprint"] == reconstructed_key,
                  "report_matches_cache_metadata": report_value == disk_value}
        if name == "weekly":
            checks["calendar_completion_matches"] = manifest["known_complete_end_date"] == bool(cfg["known_complete_end_date"])
            checks["reader_code_matches"] = manifest["code_sha256"] == file_digest(PROJECT_ROOT / "quant_backtest/weekly_t_data.py")
        else:
            checks["horizons_match"] = manifest["horizons"] == sorted(cfg["horizons"])
            checks["reader_code_matches"] = manifest["reader_code_sha256"] == file_digest(PROJECT_ROOT / "quant_backtest/weekly_t_data.py")
            checks["horizon_code_matches"] = manifest["code_sha256"] == file_digest(PROJECT_ROOT / "quant_backtest/weekly_t_horizon_data.py")
        entries[name] = {"checks": checks, "passed": all(checks.values()),
                         "fingerprint": manifest["fingerprint"], "manifest_sha256": file_digest(Path(manifest["cache_path"]) / "metadata.json")}
    selections = []
    for item in horizon["frozen_selection_sources"]:
        path = Path(item["path"])
        stat = path.stat()
        signature = file_digest(path)
        selections.append({"path": str(path), "sha256": signature,
                           "passed": signature == item["sha256"] and stat.st_size == item["bytes"] and stat.st_mtime_ns == item["mtime_ns"]})
    entries["selections"] = {"files": selections,
                             "source_manifest_matches": read_json(report / "frozen_selection_sources.json") == horizon["frozen_selection_sources"],
                             "passed": bool(selections) and all(item["passed"] for item in selections)}
    entries["selections"]["passed"] &= entries["selections"]["source_manifest_matches"]
    entries["passed"] = all(entries[name]["passed"] for name in ("weekly", "horizon", "selections"))
    return weekly, horizon, entries


def interval_overlap(frame: pd.DataFrame, excluded_years: list[int], embargo_days: int) -> tuple[np.ndarray, dict]:
    """独立以纳秒半开区间交集判断闭合样本[source_start,label_end]是否碰到隔离年。"""
    starts = pd.to_datetime(frame.feature_source_start, errors="coerce")
    ends = pd.to_datetime(frame.label_end, errors="coerce")
    valid = starts.notna().to_numpy() & ends.notna().to_numpy()
    sample_left = starts.astype("int64").to_numpy()
    sample_right = ends.astype("int64").to_numpy() + 1
    overlap = np.zeros(len(frame), dtype=bool)
    per_year = {}
    for year in sorted(set(excluded_years)):
        left = (pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=embargo_days)).value
        right = (pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=embargo_days)).value
        intersect = valid & (np.minimum(sample_right, right) > np.maximum(sample_left, left))
        overlap |= intersect
        per_year[str(year)] = int(intersect.sum())
    return overlap, per_year


def independent_training_rows(frame: pd.DataFrame, years: list[int], excluded: list[int], cfg: dict) -> pd.DataFrame:
    """独立重建分钟训练资格和原排序索引，不调用select_training_rows或推断筛选器。"""
    current = np.ones(len(frame), dtype=bool)
    if cfg.get("exclude_flat_signal_bars", True) and "signal_bar_tradable" in frame:
        current &= frame.signal_bar_tradable.eq(True).to_numpy()
    source = frame.loc[current].reset_index(drop=True)
    times = {column: pd.to_datetime(source[column], errors="coerce")
             for column in ("datetime", "feature_source_start", "label_start", "label_end")}
    eligible = np.ones(len(source), dtype=bool)
    for values in times.values():
        eligible &= values.dt.year.isin(years).to_numpy()
    eligible &= ((times["feature_source_start"] <= times["datetime"])
                 & (times["datetime"] < times["label_start"])
                 & (times["label_start"] <= times["label_end"])).to_numpy()
    eligible &= (times["datetime"].dt.normalize().eq(times["label_start"].dt.normalize())
                 & times["datetime"].dt.normalize().eq(times["label_end"].dt.normalize())).to_numpy()
    eligible &= np.isfinite(pd.to_numeric(source.target_return, errors="coerce").to_numpy())
    for column in ("trade_week_complete", "label_available"):
        if column in source:
            eligible &= source[column].eq(True).to_numpy()
    overlaps, _ = interval_overlap(source, excluded, int(cfg["embargo_days"]))
    eligible &= ~overlaps
    return source.loc[eligible].sort_values(["datetime", "symbol"], kind="stable").copy()


def audit_support(path: Path, training: pd.DataFrame, cfg: dict, years: list[int], excluded: list[int]) -> dict:
    """从完整核心训练行独立重算分位边界及签名，内部年和外层年均不得出现。"""
    stored = read_json(path)
    columns = list(cfg["t_support_features"])
    lower, upper = map(float, cfg["t_support_quantiles"])
    bounds = {}
    for column in columns:
        numeric = pd.to_numeric(training[column], errors="coerce")
        finite = numeric.loc[np.isfinite(numeric)]
        bounds[column] = {"lower": float(finite.quantile(lower)), "upper": float(finite.quantile(upper)), "finite_rows": len(finite)}
    signature = frame_digest(training[["datetime", "symbol", *columns]])
    actual_years = sorted(int(year) for year in training.datetime.dt.year.unique())
    overlaps, counts = interval_overlap(training, excluded, int(cfg["embargo_days"]))
    checks = {"training_inputs_sha256_matches": signature == stored["training_inputs_sha256"],
              "bounds_match": bounds == stored["bounds"], "quantiles_match": [lower, upper] == stored["quantiles"],
              "row_count_matches": len(training) == stored["training_rows"],
              "only_core_years": set(actual_years).issubset(years) and not set(actual_years).intersection(excluded),
              "no_excluded_interval_overlap": not overlaps.any(),
              "only_causal_support_features": all(name.startswith("f_") and name != "f_history_days" for name in columns)}
    return {"passed": all(checks.values()), "checks": checks, "training_rows": len(training),
            "actual_signal_years": actual_years, "expected_core_years": years, "excluded_years": excluded,
            "training_inputs_sha256": signature, "overlap_by_excluded_year": counts,
            "support_file_sha256": file_digest(path)}


def audit_model(path: Path, training: pd.DataFrame, cfg: dict, years: list[int],
                excluded: list[int], horizon: int, profile: str) -> dict:
    """重算拟合样本/来源SHA，再用固定小样本比较joblib与原生Booster的实际输出。"""
    metadata = read_json(path / "model_metadata.json")
    features = list(metadata["feature_names"])
    cap = int(metadata["training_row_cap"])
    fitted = (training.sample(n=cap, random_state=int(metadata["model_parameters"]["random_state"])).sort_index()
              if cap and len(training) > cap else training)
    numeric = fitted[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    target = pd.DataFrame({"return": pd.to_numeric(fitted.target_return, errors="raise")})
    sample_sha = frame_digest(pd.concat([fitted[["symbol", "datetime"]], numeric, target], axis=1))
    source_sha = frame_digest(fitted[["symbol", "datetime", "feature_source_start"]])
    overlaps, counts = interval_overlap(training, excluded, int(cfg["embargo_days"]))
    fitted_overlap, _ = interval_overlap(fitted, excluded, int(cfg["embargo_days"]))
    sample = numeric.sample(n=min(128, len(numeric)), random_state=20260908)
    bundle = joblib.load(path / "model_bundle.joblib")
    native = lightgbm.Booster(model_file=str(path / "return_model.txt"))
    restored = np.asarray(bundle["models"]["return"].predict(sample, num_threads=1), dtype=float)
    native_values = np.asarray(native.predict(sample, num_threads=1), dtype=float)
    native_match = bool(np.isfinite(restored).all() and np.isfinite(native_values).all()
                        and np.allclose(restored, native_values, rtol=1e-12, atol=1e-12))
    checks = {"training_sample_hash_matches": sample_sha == metadata["training_sample_sha256"],
              "feature_source_hash_matches": source_sha == metadata["feature_source_sha256"],
              "training_rows_match": len(training) == metadata["training_rows"],
              "fitted_rows_match": len(fitted) == metadata["fitted_rows"],
              "native_joblib_predictions_match": native_match,
              "no_excluded_interval_overlap": not overlaps.any() and not fitted_overlap.any(),
              "correct_year_sets": sorted(metadata["train_years"]) == sorted(years) and sorted(metadata["excluded_years"]) == sorted(excluded),
              "strict_provenance_enabled": metadata.get("strict_feature_provenance") is True,
              "only_f_features": all(name.startswith("f_") for name in features),
              "history_count_absent": "f_history_days" not in features,
              "configured_exclusions_absent": not set(features).intersection(cfg["excluded_model_features"]),
              "feature_list_matches": (path / "feature_list.txt").read_text().splitlines() == features,
              "bundle_features_match": bundle["features"] == features == native.feature_name(),
              "bundle_metadata_matches": bundle["metadata"] == metadata,
              "profile_matches": metadata["profile"] == profile,
              "horizon_matches": metadata["resolved_config"]["t_label_horizon_minutes"] == horizon,
              "task_matches": metadata["task"] == bundle["task"] == "intraday",
              "row_cap_matches": cap == int(cfg["t_max_train_rows"]),
              "lightgbm_version_matches": metadata["lightgbm_version"] == lightgbm.__version__}
    result = {"passed": all(checks.values()), "checks": checks, "training_rows": len(training),
              "fitted_rows": len(fitted), "training_sample_sha256": sample_sha, "feature_source_sha256": source_sha,
              "training_sample_hash_matches": checks["training_sample_hash_matches"],
              "feature_source_hash_matches": checks["feature_source_hash_matches"],
              "training_overlap_rows": int(overlaps.sum()), "fitted_overlap_rows": int(fitted_overlap.sum()),
              "training_overlap_by_excluded_year": counts, "native_joblib_predictions_match": native_match,
              "native_prediction_rows": len(sample), "native_prediction_max_abs_error": float(np.max(np.abs(restored - native_values))),
              "train_years": metadata["train_years"], "excluded_years": metadata["excluded_years"],
              "feature_source_min": str(training.feature_source_start.min()), "signal_min": str(training.datetime.min()),
              "label_end_max": str(training.label_end.max()), "feature_names": features,
              "metadata_sha256": file_digest(path / "model_metadata.json"),
              "model_bundle_sha256": file_digest(path / "model_bundle.joblib"),
              "native_model_sha256": file_digest(path / "return_model.txt")}
    del numeric, bundle, native, fitted
    return result


def audit_outer_copy(directory: Path, cfg: dict) -> dict:
    """核对外层保留选中核心权重及支持域，禁止重拟合或把内部年并入权重。"""
    rule = read_json(directory / "selected_rule.json")
    source = directory / rule["source_model"]
    destination = directory / "t_model"
    source_files = {str(path.relative_to(source)): file_digest(path) for path in source.rglob("*") if path.is_file()}
    destination_files = {str(path.relative_to(destination)): file_digest(path) for path in destination.rglob("*") if path.is_file()}
    support_source = directory / f"support_h{rule['horizon_minutes']}.json"
    support_selected = directory / "selected_support.json"
    checks = {"all_model_files_identical": bool(source_files) and source_files == destination_files,
              "bundle_matches_selected_sha": destination_files.get("model_bundle.joblib") == rule["model_bundle_sha256"],
              "support_exact_copy": file_digest(support_source) == file_digest(support_selected),
              "source_model_matches_rule": rule["source_model"] == f"inner_t_h{rule['horizon_minutes']}_{rule['profile']}",
              "outer_refit_disabled": cfg.get("outer_t_refit") is False and rule["fold"]["outer_t_refit"] is False,
              "rule_declared_frozen_before_outer": rule.get("frozen_before_any_outer_test") is True}
    return {"passed": all(checks.values()), "checks": checks, "source_model": rule["source_model"],
            "source_file_sha256": source_files, "outer_file_sha256": destination_files,
            "selected_rule_sha256": file_digest(directory / "selected_rule.json"),
            "selected_support_sha256": file_digest(support_selected)}


def execute_audit(report: Path, smoke: bool, result: dict) -> None:
    """核对冻结身份后一次读取全市场数据，按周期重建并复用独特核心训练行。"""
    cfg = read_json(report / "resolved_config.json")
    small_run = bool(cfg.get("max_symbols", 0) or cfg.get("smoke_weeks", 0))
    if smoke != small_run:
        raise ValueError("--smoke须与冻结配置的小样本状态一致；审计不会修改max_symbols或smoke_weeks")
    result.update({"mode": "小样本接线验证" if smoke else "完整冻结年度范围", "requested_years": cfg["test_years"],
                   "configured_max_symbols": cfg.get("max_symbols", 0), "configured_smoke_weeks": cfg.get("smoke_weeks", 0)})
    if cfg.get("outer_t_refit") is not False or cfg.get("strict_feature_provenance") is not True or "f_history_days" not in cfg["excluded_model_features"]:
        raise ValueError("v5必须保持核心权重、真实来源隔离并排除f_history_days")
    result["sources"] = verify_sources(report)
    if not result["sources"]["passed"]:
        raise ValueError("冻结源码快照或当前依赖SHA不匹配，拒绝跨版本重算")
    weekly_manifest, horizon_manifest, result["input_identity"] = verify_cache_inputs(report, cfg)
    if not result["input_identity"]["passed"]:
        raise ValueError("原始数据stat、缓存配置指纹或冻结选股SHA不匹配")
    folds = {int(year): read_json(report / f"year_{year}/split.json") for year in cfg["test_years"]}
    for year in folds:
        for horizon in cfg["horizons"]:
            for profile in cfg["t_model_profiles"]:
                path = report / f"year_{year}/inner_t_h{horizon}_{profile}/model_metadata.json"
                if not path.exists():
                    raise FileNotFoundError(f"需要所有请求模型完成后再审计：{path}")
            if not (report / f"year_{year}/support_h{horizon}.json").exists():
                raise FileNotFoundError(f"缺少核心支持域：year_{year}/support_h{horizon}.json")
        result["outer_model_copies"][str(year)] = audit_outer_copy(report / f"year_{year}", cfg)
    freeze = read_json(report / "all_rules_frozen_before_outer_test.json")
    result["rule_freeze_hashes_match"] = set(map(int, freeze["test_years"])) == set(folds) and all(
        freeze["selected_rule_sha256"][str(year)] == result["outer_model_copies"][str(year)]["selected_rule_sha256"] for year in folds)
    save_audit(report, result)

    # 当前源码已逐文件核对，可复用冻结计算；不导入runner的训练/回测入口。
    from quant_backtest.weekly_t_features import add_session_context_features, add_weekly_execution_features
    from quant_backtest.weekly_t_provenance import add_feature_provenance
    from quant_backtest.weekly_t_short_horizon import attach_horizon_labels

    print("独立审计读取原周/日内/日表及短周期标签；不加载2亿行恢复报价", flush=True)
    cache = Path(weekly_manifest["cache_path"])
    weekly, intraday, daily = [pd.read_pickle(cache / f"{name}.pkl") for name in ("weekly", "intraday", "daily")]
    labels = pd.read_pickle(Path(horizon_manifest["labels_path"]))
    row_counts = {"weekly": len(weekly), "intraday": len(intraday), "daily": len(daily), "labels": len(labels)}
    if any(row_counts[name] != weekly_manifest[name + "_rows"] for name in ("weekly", "intraday", "daily")) or len(labels) != horizon_manifest["label_rows"]:
        raise ValueError("冻结缓存实际行数与manifest不符")
    all_years = sorted(int(year) for year in daily.date.dt.year.unique())
    result["fold_protocol"] = {}
    for year, fold in folds.items():
        train_years = [value for value in all_years if value != year]
        inner = max(train_years)
        core = [value for value in train_years if value != inner]
        checks = {"train_years_match": fold["train_years"] == train_years, "inner_year_matches": fold["inner_year"] == inner,
                  "core_years_match": fold["core_years"] == core, "outer_refit_disabled": fold["outer_t_refit"] is False,
                  "forward_only_matches": fold["forward_only"] == (max(train_years) < year)}
        result["fold_protocol"][str(year)] = {"passed": all(checks.values()), "checks": checks, "split": fold}
    print("重建完整截面会话特征及真实最早来源，再按过去周资格形成T训练母表", flush=True)
    weekly = add_weekly_execution_features(weekly, daily)
    intraday = add_session_context_features(intraday)
    weekly, intraday, provenance = add_feature_provenance(weekly, intraday, daily)
    result["provenance"] = {"recomputed": provenance, "reported": read_json(report / "feature_provenance_manifest.json")}
    result["provenance"]["passed"] = result["provenance"]["recomputed"] == result["provenance"]["reported"] and provenance["market_previous_close_dependency_enabled"]
    mask = ((weekly.f_history_days >= cfg["min_history_days"]) & (weekly.f_last_close >= cfg["min_price"])
            & (weekly.f_avg_daily_amount >= cfg["min_avg_daily_amount"]) & weekly.feature_week_complete.fillna(False))
    if cfg.get("min_prior_close_auction_amount", 0) > 0:
        mask &= weekly.f_prior_close_auction_amount_min >= cfg["min_prior_close_auction_amount"]
    universe = weekly.loc[mask]
    training_source = intraday.merge(universe[["trade_week", "symbol"]], on=["trade_week", "symbol"], how="inner", validate="many_to_one")
    result["reconstructed_rows"] = {**row_counts, "causal_weekly_universe": len(universe), "t_training_source": len(training_source)}
    del weekly, intraday, daily, universe
    gc.collect()
    unique_scopes = set()
    for horizon in cfg["horizons"]:
        print(f"重建h{horizon}标签与核心范围", flush=True)
        frame = attach_horizon_labels(training_source, labels, horizon)
        scope_tables = {}
        for year, fold in folds.items():
            years = sorted(fold["core_years"])
            excluded = sorted([year, fold["inner_year"]])
            scope = (int(horizon), tuple(years), tuple(excluded))
            if scope not in scope_tables:
                scope_tables[scope] = independent_training_rows(frame, years, excluded, cfg)
                unique_scopes.add(scope)
            training = scope_tables[scope]
            if len(training) < int(cfg["min_training_rows"]):
                raise ValueError(f"独立重建核心训练样本不足：{scope}")
            name = f"year_{year}/support_h{horizon}.json"
            result["supports"][name] = audit_support(report / name, training, cfg, years, excluded)
            for profile in cfg["t_model_profiles"]:
                name = f"year_{year}/inner_t_h{horizon}_{profile}"
                audit = audit_model(report / name, training, cfg, years, excluded, horizon, profile)
                result["models"][name] = audit
                print(f"{name}：{audit['training_rows']:,}训练行；样本SHA={audit['training_sample_hash_matches']}，"
                      f"来源SHA={audit['feature_source_hash_matches']}，区间重叠={audit['training_overlap_rows']}，"
                      f"原生权重一致={audit['native_joblib_predictions_match']}", flush=True)
            result["fold_horizon_scopes"] = len(result["supports"])
            result["unique_training_scopes"] = len(unique_scopes)
            save_audit(report, result)
        del frame, training, scope_tables
        gc.collect()
    del training_source, labels
    gc.collect()
    final_sources = verify_sources(report)
    result["source_files_unchanged_during_audit"] = final_sources == result["sources"]
    result["audit_script_unchanged"] = file_digest(Path(__file__)) == result["audit_script_sha256"]
    expected_models = len(folds) * len(cfg["horizons"]) * len(cfg["t_model_profiles"])
    result["complete"] = len(result["models"]) == expected_models and len(result["supports"]) == len(folds) * len(cfg["horizons"])
    result["passed"] = bool(result["complete"] and result["rule_freeze_hashes_match"] and result["provenance"]["passed"]
        and result["source_files_unchanged_during_audit"] and result["audit_script_unchanged"]
        and all(value["passed"] for value in result["fold_protocol"].values())
        and all(value["passed"] for value in result["models"].values())
        and all(value["passed"] for value in result["supports"].values())
        and all(value["passed"] for value in result["outer_model_copies"].values()))
    result["strict_raw_price_feature_boundary_passed"] = bool(result["complete"] and all(
        audit["training_overlap_rows"] == 0 and audit["fitted_overlap_rows"] == 0
        and audit["training_sample_hash_matches"] and audit["feature_source_hash_matches"]
        and audit["checks"]["history_count_absent"] for audit in result["models"].values()))


def main() -> None:
    """执行显式报告路径审计；异常也保存已完成证据，并以非零状态退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="确认审计的是冻结配置中的小样本，不改变配置")
    args = parser.parse_args()
    report = args.report_dir.resolve()
    if not report.is_dir():
        parser.error("report-dir必须是已存在的报告目录")
    started = time.monotonic()
    result = {"audit_script_sha256": file_digest(Path(__file__)), "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "report_dir": str(report), "complete": False, "passed": False, "models": {}, "supports": {},
              "outer_model_copies": {}, "historical_retest_not_fresh_holdout": True,
              "outer_performance_read": False, "model_training_performed": False,
              "method": "冻结helper只重建因果特征/来源/标签；训练资格、时间交集、拟合抽样、两类签名、支持域及权重复制独立核对"}
    try:
        execute_audit(report, args.smoke, result)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["error_traceback"] = traceback.format_exc()
        print(result["error_traceback"], file=sys.stderr, flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    save_audit(report, result)
    print(f"审计完成={result['complete']}，全部通过={result['passed']}；{report / 'independent_short_horizon_audit.json'}", flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
