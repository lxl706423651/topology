"""独立重算v4实际训练行和签名，不调用训练筛选函数、不重训、不读取外层收益。

从项目根目录使用quant环境Python执行本文件；只新增/更新同报告的独立审计JSON。
来源helper和会话增强可复用，区间相交、训练资格、抽样和签名由本文件独立核对。
默认首折；--all-folds一次读取大表覆盖全部24模型并另存full审计，不覆盖首折证据。
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from quant_backtest.weekly_t_features import add_session_context_features, add_weekly_execution_features
from quant_backtest.weekly_t_provenance import add_feature_provenance


REPORT = Path(__file__).resolve().parent
OUTPUT = REPORT / "independent_strict_feature_audit.json"


def digest_frame(frame):
    """按持久元数据约定，对表内容及列顺序计算签名，不加入索引。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def interval_overlap(frame, excluded, embargo):
    """独立采用闭合样本区间与半开隔离区间交集长度判断，不调用模型掩码。"""
    left = pd.to_datetime(frame.feature_source_start).astype("int64").to_numpy()
    right = pd.to_datetime(frame.label_end).astype("int64").to_numpy() + 1
    overlap = np.zeros(len(frame), dtype=bool)
    per_year = {}
    for year in excluded:
        excluded_start = (pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=embargo)).value
        excluded_stop = (pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=embargo)).value
        intersection_start = np.maximum(left, excluded_start)
        intersection_stop = np.minimum(right, excluded_stop)
        overlaps_this_year = intersection_stop > intersection_start
        overlap |= overlaps_this_year
        per_year[str(year)] = int(overlaps_this_year.sum())
    return overlap, per_year


def independent_training_rows(frame, years, excluded, task, cfg):
    """从完整增强面板独立重建资格、端点顺序和隔离，再按真实时钟及代码排序。"""
    causal = np.ones(len(frame), dtype=bool)
    if task == "weekly":
        causal &= frame.feature_week_complete.eq(True).to_numpy()
    elif cfg.get("exclude_flat_signal_bars", False) and "signal_bar_tradable" in frame:
        causal &= frame.signal_bar_tradable.eq(True).to_numpy()
    source = frame.loc[causal].reset_index(drop=True)
    key = "signal_datetime" if task == "weekly" else "datetime"
    eligible = np.ones(len(source), dtype=bool)
    dates = {column: pd.to_datetime(source[column]) for column in
             ["feature_source_start", key, "label_start", "label_end"]}
    for values in dates.values():
        eligible &= values.dt.year.isin(years).to_numpy()
    eligible &= ((dates["feature_source_start"] <= dates[key]) & (dates[key] < dates["label_start"])
                 & (dates["label_start"] <= dates["label_end"])).to_numpy()
    if task == "weekly":
        eligible &= source.feature_week.dt.year.isin(years).to_numpy()
        eligible &= np.isfinite(source.target_amplitude.to_numpy()) & (source.target_amplitude.to_numpy() >= 0)
    else:
        eligible &= (dates[key].dt.normalize().eq(dates["label_start"].dt.normalize())
                     & dates[key].dt.normalize().eq(dates["label_end"].dt.normalize())).to_numpy()
    eligible &= np.isfinite(pd.to_numeric(source.target_return).to_numpy())
    for column in ["trade_week_complete", "label_available"]:
        if column in source:
            eligible &= source[column].eq(True).to_numpy()
    overlaps, _ = interval_overlap(source, excluded, int(cfg["embargo_days"]))
    eligible &= ~overlaps
    training = source.loc[eligible].sort_values([key, "symbol"], kind="stable").copy()
    return training


def audit_model(metadata, training, excluded, cfg):
    """用独立行集、冻结特征名和原随机种子重算拟合签名，逐行检查隔离区间。"""
    task = metadata["task"]
    key = "signal_datetime" if task == "weekly" else "datetime"
    cap = int(metadata["training_row_cap"])
    fitted = training.sample(n=cap, random_state=int(metadata["model_parameters"]["random_state"])).sort_index() if cap and len(training) > cap else training
    targets = (pd.DataFrame({"return": training.groupby("trade_week").target_return.rank(pct=True),
                             "amplitude": training.groupby("trade_week").target_amplitude.rank(pct=True)})
               if task == "weekly" else pd.DataFrame({"return": pd.to_numeric(training.target_return)}))
    values = fitted[metadata["feature_names"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    signature = digest_frame(pd.concat([fitted[["symbol", key]], values, targets.loc[fitted.index]], axis=1))
    source_signature = digest_frame(fitted[["symbol", key, "feature_source_start"]])
    all_overlap, by_year = interval_overlap(training, excluded, int(cfg["embargo_days"]))
    fit_overlap, fitted_by_year = interval_overlap(fitted, excluded, int(cfg["embargo_days"]))
    return {"training_rows": len(training), "metadata_training_rows": metadata["training_rows"],
            "fitted_rows": len(fitted), "metadata_fitted_rows": metadata["fitted_rows"],
            "training_sample_sha256": signature, "metadata_training_sample_sha256": metadata["training_sample_sha256"],
            "feature_source_sha256": source_signature, "metadata_feature_source_sha256": metadata["feature_source_sha256"],
            "training_sample_hash_matches": signature == metadata["training_sample_sha256"],
            "feature_source_hash_matches": source_signature == metadata["feature_source_sha256"],
            "all_training_overlap_rows": int(all_overlap.sum()), "fitted_overlap_rows": int(fit_overlap.sum()),
            "training_overlap_by_excluded_year": by_year, "fitted_overlap_by_excluded_year": fitted_by_year,
            "feature_source_min": str(training.feature_source_start.min()),
            "signal_min": str(training[key].min()), "label_end_max": str(training.label_end.max()),
            "only_f_features": all(name.startswith("f_") for name in metadata["feature_names"]),
            "history_count_absent": "f_history_days" not in metadata["feature_names"],
            "train_years": metadata["train_years"], "excluded_years": metadata["excluded_years"]}


def collect_metadata_audit():
    """轻量统计当前已产生模型的冻结协议、特征清单和来源签名，不读取收益。"""
    entries = []
    for path in sorted(REPORT.glob("year_*/*/model_metadata.json")):
        try:
            metadata = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        split = json.loads((path.parent.parent / "split.json").read_text())
        inner = path.parent.name.startswith("inner_")
        train = split["core_years"] if inner else split["train_years"]
        excluded = [split["test_year"], split["inner_year"]] if inner else [split["test_year"]]
        entries.append({"model": str(path.relative_to(REPORT)), "train_years": metadata["train_years"],
                        "excluded_years": metadata["excluded_years"], "training_rows": metadata["training_rows"],
                        "fitted_rows": metadata["fitted_rows"], "feature_source_start_min": metadata["feature_source_start_min"],
                        "label_end_max": metadata["label_end_max"],
                        "correct_year_sets": set(train) == set(metadata["train_years"]) and set(excluded) == set(metadata["excluded_years"]),
                        "strict_provenance_enabled": bool(metadata.get("strict_feature_provenance")),
                        "source_hash_present": len(metadata.get("feature_source_sha256", "")) == 64,
                        "history_count_absent": "f_history_days" not in metadata["feature_names"],
                        "only_f_features": all(name.startswith("f_") for name in metadata["feature_names"]),
                        "feature_list_matches": (path.parent / "feature_list.txt").read_text().splitlines() == metadata["feature_names"],
                        "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return entries


def main():
    """重建指定折的核心/外层训练签名并持久化审计，不运行模型fit或候选回测。"""
    cfg = json.loads((REPORT / "resolved_config.json").read_text())
    all_folds = "--all-folds" in sys.argv
    output = REPORT / "independent_strict_feature_audit_full.json" if all_folds else OUTPUT
    manifest = json.loads((REPORT / "data_manifest.json").read_text())
    source_hashes = json.loads((REPORT / "source_hashes.json").read_text())
    snapshot_matches = {name: hashlib.sha256((REPORT / "source_snapshot" / Path(name).name).read_bytes()).hexdigest() == signature
                        for name, signature in source_hashes.items()}
    current_matches = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == signature
                       for name, signature in source_hashes.items()}
    if not all(snapshot_matches.values()) or not all(current_matches.values()):
        raise RuntimeError("冻结源文件哈希不匹配，拒绝用其他版本重算")
    print("独立审计读取冻结缓存，随后重建完整截面来源；不训练模型", flush=True)
    cache = Path(manifest["cache_path"])
    weekly = pd.read_pickle(cache / "weekly.pkl")
    daily = pd.read_pickle(cache / "daily.pkl")
    intraday = pd.read_pickle(cache / "intraday.pkl")
    print("缓存读取完成；增强当前特征与真实来源", flush=True)
    weekly = add_weekly_execution_features(weekly, daily)
    intraday = add_session_context_features(intraday)
    weekly, intraday, provenance = add_feature_provenance(weekly, intraday, daily)
    del daily
    gc.collect()
    mask = ((weekly.f_history_days >= cfg["min_history_days"]) & (weekly.f_last_close >= cfg["min_price"])
            & (weekly.f_avg_daily_amount >= cfg["min_avg_daily_amount"]) & weekly.feature_week_complete.eq(True)
            & (weekly.f_prior_close_auction_amount_min >= cfg["min_prior_close_auction_amount"]))
    universe = weekly.loc[mask].copy()
    t_training = intraday.merge(universe[["trade_week", "symbol"]], on=["trade_week", "symbol"],
                                how="inner", validate="many_to_one")
    del weekly, intraday
    gc.collect()
    print(f"完整截面来源完成，周候选{len(universe):,}行，盘中训练母表{len(t_training):,}行", flush=True)
    years_to_audit = cfg["test_years"] if all_folds else [2023]
    outputs = {}
    training_tables = {}
    start = time.monotonic()
    model_names = {"inner_weekly_model", "inner_t_l2_depth4", "inner_t_l2_depth6", "inner_t_huber_depth4", "weekly_model", "t_model"}
    expected_names = {f"year_{year}/{name}" for year in years_to_audit for name in model_names}
    while set(outputs) != expected_names:
        metadata_paths = [path for year in years_to_audit for path in (REPORT / f"year_{year}").glob("*/model_metadata.json")]
        for path in sorted(metadata_paths):
            name = str(path.parent.relative_to(REPORT))
            if name in outputs:
                continue
            try:
                metadata = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            task = metadata["task"]
            key = (task, tuple(sorted(metadata["train_years"])), tuple(sorted(metadata["excluded_years"])))
            if key not in training_tables:
                frame = universe if task == "weekly" else t_training
                training_tables[key] = independent_training_rows(frame, metadata["train_years"],
                                                                 metadata["excluded_years"], task, cfg)
            outputs[name] = audit_model(metadata, training_tables[key], metadata["excluded_years"], cfg)
            print(f"{name}：训练行{outputs[name]['training_rows']:,}，签名匹配{outputs[name]['training_sample_hash_matches']}，来源签名{outputs[name]['feature_source_hash_matches']}，隔离重叠{outputs[name]['all_training_overlap_rows']}", flush=True)
        metadata_audits = collect_metadata_audit()
        passed = bool(outputs) and all(v["training_sample_hash_matches"] and v["feature_source_hash_matches"]
            and v["all_training_overlap_rows"] == 0 and v["fitted_overlap_rows"] == 0 and v["history_count_absent"]
            and v["training_rows"] == v["metadata_training_rows"] and v["fitted_rows"] == v["metadata_fitted_rows"] for v in outputs.values())
        result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "scope": ("四折全部模型逐行重建训练与来源签名" if all_folds else "首折逐行重建训练与来源签名，其余折核对已生成元数据") + "；不读取外层收益，不重训，不调整参数。",
                  "strict_raw_price_feature_boundary_passed_for_audited_rows": passed,
                  "requested_folds_row_audit_complete": set(outputs) == expected_names,
                  "all_folds_row_audit_complete": all_folds and set(outputs) == expected_names,
                  "requested_years": years_to_audit, "unique_training_scopes": len(training_tables),
                  "snapshot_hash_matches": snapshot_matches, "current_source_hash_matches": current_matches,
                  "recomputed_provenance_manifest": provenance,
                  "reported_provenance_manifest": json.loads((REPORT / "feature_provenance_manifest.json").read_text()),
                  "row_audits": outputs, "metadata_audits": metadata_audits,
                  "all_folds_metadata_complete": len(metadata_audits) == 24,
                  "historical_retest_not_fresh_holdout": True,
                  "interpretation": "strict raw price feature boundary passed仅表示核对的原始量价输入区间与标签不跨排除年及±7日；历次外层历史结果已参与研究，此次仍是历史复测，不能据隔离修复声称全新测试或保证增益。",
                  "method": "复用冻结因果增强/来源helper；训练资格与区间交集采用独立实现，再重算拟合抽样、完整训练数据签名和来源签名。累计history_days仅作事前股票资格，不入模型。"}
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        if set(outputs) == expected_names:
            break
        if time.monotonic() - start > 900:
            print("请求的模型尚未全部落盘，已保存明确标记为部分完成的独立审计", flush=True)
            break
        print("等待其余冻结模型元数据，已完成结果保留于独立审计JSON", flush=True)
        time.sleep(10)
    print(f"独立训练来源审计已写入 {output}", flush=True)


if __name__ == "__main__":
    main()
