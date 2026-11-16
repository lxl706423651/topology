"""v16独立模型审计辅助：核实际矩阵、固定尾盘原始标签、purge和原生权重。

由 audit_inventory_close_t 调用；不导入生产close模型/标签/队列模块，不拟合，
不运行账户。当前core和独立选特征结果由主审计提供；raw通过其只读full_bars读取。
直接运行 --self-test 只验证纯时钟/隔离反例，不读取市场或训练模型。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dynamic_inventory_t import close, digest, next_minute, require
from scripts.audit_inventory_learning_t import exact, frame_digest
from scripts.audit_weekly_t_short_horizon import interval_overlap

VERSION = "inventory_close_t_v16"
KEYS = ["datetime", "symbol"]
IDENTITY = {"recovery_key": "close", "recovery_kind": "fixed_clock", "recovery_clock": "14:55"}
CLOCKS = ("09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00")


def independent_close_training_split(frame: pd.DataFrame, fold: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """独立生成完整机会的逐原因隔离统计；不调用生产select_training_rows。

本轮当前资格保留全部8信号，包括平价；只在这里以来源/未来标签做训练purge。
原因计数允许重叠，与保留行数分别报告，不把缺标签行当不存在的持仓机会。
    """
    require(cfg.get("exclude_flat_signal_bars") is False and cfg.get("strict_feature_provenance") is True,
            "本轮当前资格/严格来源配置错误")
    require(cfg.get("embargo_days") == 7, "本轮隔离须7日")
    require(not any(c in frame for c in ("trade_week_complete", "horizon_minutes")), "旧周或整数期限污染训练机会")
    require(not frame[KEYS].isna().any().any() and not frame.duplicated(KEYS).any(), "原完整机会键缺失/重复")
    source = frame.reset_index(drop=True)
    values = {name: pd.to_datetime(source[name], errors="coerce")
              for name in ("datetime", "feature_source_start", "label_start", "label_end")}
    start, signal, first, end = [values[name] for name in ("feature_source_start", "datetime", "label_start", "label_end")]
    years = sorted(fold["core_years"])
    excluded = sorted([fold["inner_year"], fold["test_year"]])
    require(len(set(years)) == 2 and len(set(excluded)) == 2 and not set(years) & set(excluded), "核心/排除年交叉")
    bad = {
        "feature_year_not_training": ~start.dt.year.isin(years),
        "signal_year_not_training": ~signal.dt.year.isin(years),
        "label_start_year_not_training": ~first.dt.year.isin(years),
        "label_end_year_not_training": ~end.dt.year.isin(years),
        "noncausal_time_order": ~(start.le(signal) & signal.lt(first) & first.le(end)),
        "intraday_label_crosses_day": ~(signal.dt.normalize().eq(first.dt.normalize()) & signal.dt.normalize().eq(end.dt.normalize())),
        "invalid_target_return": ~np.isfinite(pd.to_numeric(source.target_return, errors="coerce")),
        "label_unavailable": ~source.label_available.eq(True),
    }
    # 与独立ns区间相交算法并列校验，NaT在前面时序检查拒绝。
    boundary = pd.Series(False, index=source.index)
    for year in excluded:
        left = pd.Timestamp(year, 1, 1)-pd.Timedelta(days=7)
        right = pd.Timestamp(year+1, 1, 1)+pd.Timedelta(days=7)
        boundary |= start.lt(right) & end.ge(left)
    independent_overlap, _ = interval_overlap(source, excluded, 7)
    require(np.array_equal(boundary.to_numpy(), independent_overlap), "隔离区间独立算法不一致")
    bad["excluded_year_or_boundary_embargo"] = boundary
    kept = source.loc[~pd.DataFrame(bad).any(axis=1)].sort_values(KEYS, kind="stable").copy()
    rows = [{"reason": "causal_inference_filter", "excluded_rows": 0}]
    rows += [{"reason": name, "excluded_rows": int(mask.sum())} for name, mask in bad.items()]
    rows.append({"reason": "retained_training_rows", "excluded_rows": len(kept)})
    summary = {"input_rows": len(frame), "causally_eligible_rows": len(frame), "training_rows": len(kept),
               "training_start": str(kept.datetime.min()), "training_end": str(kept.datetime.max()),
               "label_start_min": str(kept.label_start.min()), "label_end_max": str(kept.label_end.max()),
               "feature_source_start_min": str(kept.feature_source_start.min())}
    return kept, pd.DataFrame(rows), summary


def validate_close_training_identity(training: pd.DataFrame, fold: dict, cfg: dict) -> dict:
    """核已选训练行的固定时钟、价格比、核心旧股与完整历史来源，不读成交收益。"""
    require(len(training) > 0 and not training.duplicated(KEYS).any(), "训练为空或重复键")
    require("horizon_minutes" not in training and "trade_week_complete" not in training, "旧期限/周完整资格进入训练")
    require(all(name in training and training[name].eq(value).all() for name, value in IDENTITY.items()), "训练目标身份不是固定14:55")
    signal = pd.to_datetime(training.datetime)
    require(signal.dt.tz is None and signal.dt.strftime("%H:%M").isin(CLOCKS).all()
            and signal.dt.second.eq(0).all() and signal.dt.microsecond.eq(0).all()
            and signal.dt.nanosecond.eq(0).all(), "训练信号不在原8个精确时钟")
    first = signal.map(next_minute)
    end = signal.dt.normalize()+pd.Timedelta(hours=14, minutes=55)
    for name, expected in (("execution_time", first), ("label_start", first), ("recovery_time", end), ("label_end", end)):
        require(name in training and pd.to_datetime(training[name]).eq(expected).all(), f"{name}不是独立固定尾盘端点")
    require(training.cohort_target_year.eq(signal.dt.year).all()
            and training.cohort_target_year.isin(fold["core_years"]).all(), "实际目标年非core")
    require(training.baseline_opening_shares.ge(cfg["lot_size"]).all(), "实际训练无足够日初旧股")
    require(training.label_available.eq(True).all(), "实际拟合包含缺标签行")
    prices = training[["execution_price", "recovery_price"]].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(training.target_return, errors="raise")
    require(np.isfinite(prices).all().all() and prices.gt(0).all().all()
            and np.isfinite(target).all() and target.gt(-1).all(), "训练价格/原收益非有限正价格比")
    # 原cache端点经过同一种double除法，应精确重现，不能用旧target或净额标签替代。
    require(np.array_equal((prices.recovery_price/prices.execution_price-1).to_numpy(), target.to_numpy()), "标签不是原始精确参考价收益")
    overlap, counts = interval_overlap(training, [fold["inner_year"], fold["test_year"]], 7)
    require(not overlap.any(), "来源至14:55跨留出年/隔离边界")
    starts = pd.to_datetime(training.feature_source_start)
    require(starts.notna().all() and starts.le(signal).all() and starts.dt.year.isin(fold["core_years"]).all(), "实际训练来源无效")
    return counts


def audit_close_model(auditor, directory: Path, training: pd.DataFrame, features: list[str],
                      feature_audit: pd.DataFrame, fold: dict, profile: str,
                      *, full_frame: pd.DataFrame | None = None) -> dict:
    """核一个新close模型并返回可归档dict；auditor提供json/table/record/full_bars。

training/features来自主审计独立构造。full_frame建议必传，以核完整机会→训练的
逐原因统计和原索引；无此参数时清楚记录未重复核split，不冒称完整split通过。
    """
    directory = Path(directory)
    cfg = auditor.cfg
    meta = auditor.json(directory/"model_metadata.json")
    require(cfg["research_version"] == VERSION and meta["research_version"] == VERSION, "v16模型版本错误")
    require(profile in ("l2_depth4", "l2_depth6") and meta["profile"] == profile, "模型profile错误")
    require(meta["train_years"] == sorted(fold["core_years"])
            and meta["excluded_years"] == sorted([fold["inner_year"], fold["test_year"]]), "模型训练/排除年身份错误")
    require(meta["features"] == features and all(name.startswith("f_") and name != "f_history_days" for name in features), "实际模型特征身份不符")
    require(all(meta.get(name) == value for name, value in IDENTITY.items()) and "horizon_minutes" not in meta, "模型期限身份不是close")
    require(meta.get("fixed_integer_horizon_used") is False and meta.get("target_is_portfolio_execution_net_pnl") is False
            and meta.get("probability_gate_used") is False, "模型目标/门控元数据错误")
    require(meta["target_definition"] == "same_day_14_55_reference_price/next_trading_minute_reference_price-1", "参考标签定义错误")
    require(meta["training_cohort"] == "permanent_inventory_core_oof_opening_old_shares", "训练场景非永久底仓旧股")
    require(meta["resolved_config"] == {**cfg, "t_model_profile": profile}, "模型解析配置偏离本轮预声明")
    require(meta["fitted_rows"] == meta["training_rows"] == len(training) and cfg["t_max_train_rows"] == 0, "拟合样本/禁额外抽样合同错误")
    require(meta["strict_feature_provenance"] is True and meta["embargo_days"] == 7
            and meta["purge_reason_counts_overlap"] is True, "训练隔离metadata错误")
    counts = validate_close_training_identity(training, fold, cfg)
    split_summary = None
    if full_frame is not None:
        actual_training, split, split_summary = independent_close_training_split(full_frame, fold, cfg)
        exact(actual_training, training, "独立完整机会→训练purge", dtype=True)
        require(actual_training.index.equals(training.index), "选特征抽样用训练索引已变化")
        exact(auditor.table(directory/"training_split_audit.csv"), split, "逐原因训练隔离审计", dtype=True)
        for name, value in split_summary.items():
            require(meta[name] == value, f"训练隔离metadata {name}不符")
    exact(auditor.table(directory/"feature_manifest.csv"), feature_audit, "独立close特征选择")
    matrix = training[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    actual_inputs = pd.read_pickle(auditor.record(directory/"training_inputs.pkl"))
    expected = training[KEYS+["feature_source_start", "cohort_target_year", "baseline_opening_shares"]].copy()
    for name in features:
        expected[name] = matrix[name].to_numpy()
    exact(actual_inputs, expected, "实际close拟合input.pkl全部current值/dtype", dtype=True)
    actual_labels = auditor.table(directory/"training_labels.csv")
    trace = [name for name in training if name in (
        "datetime", "symbol", "feature_source_start", "cohort_target_year", "baseline_opening_shares",
        "intraday_feature_source_start", "entry_feature_source_start", "entry_rank_source_start",
        "portfolio_cash_source_start", "base_entry_datetime", "source_model_training_years", "source_model_sha256",
        "label_start", "label_end", "target_return", "execution_price", "recovery_price", *IDENTITY, "label_available")]
    expected_labels = training[trace].copy()
    expected_labels["fitted"] = True
    exact(actual_labels, expected_labels, "完整训练原始标签及全部来源列")
    raw_zero_volume = {"label_start": 0, "label_end": 0}
    for (year, symbol), part in training.groupby([training.datetime.dt.year, "symbol"], sort=False):
        raw = auditor.full_bars(year, symbol)
        for endpoint, name in (("label_start", "execution_price"), ("label_end", "recovery_price")):
            quotes = raw.reindex(pd.DatetimeIndex(part[endpoint]))
            actual = quotes.close.to_numpy()
            require(np.isfinite(actual).all() and np.array_equal(actual, part[name].to_numpy()),
                    f"训练{endpoint}价格偏离原始精确分钟：{year}/{symbol}")
            raw_zero_volume[endpoint] += int(quotes.volume.eq(0).sum())
    hashes = {"full_training_rows_sha256": frame_digest(training[KEYS+["feature_source_start"]]),
              "training_matrix_sha256": frame_digest(matrix),
              "training_target_sha256": frame_digest(training[["target_return"]])}
    for name, value in hashes.items():
        require(meta[name] == value, f"模型{name}不符")
    for filename, key in (("training_inputs.pkl", "training_inputs_file_sha256"), ("training_labels.csv", "training_labels_file_sha256"),
                          ("return_model.txt", "native_model_sha256"), ("model_bundle.joblib", "model_bundle_sha256")):
        require(digest(auditor.record(directory/filename)) == meta[key], f"模型实际文件SHA不符：{filename}")
    bundle = joblib.load(directory/"model_bundle.joblib")
    normalized = json.loads(json.dumps(bundle["metadata"], default=str))
    require(normalized == {key: value for key, value in meta.items() if key != "model_bundle_sha256"}, "bundle与外部metadata身份不同")
    native = lightgbm.Booster(model_file=str(directory/"return_model.txt"))
    require(native.feature_name() == features == bundle["features"], "原生模型特征顺序不符")
    require(native.model_to_string() == bundle["model"].booster_.model_to_string(), "原生全部树与joblib权重不同")
    parameters = {"n_estimators": 160, "learning_rate": .04, "max_depth": 4, "num_leaves": 15,
                  "min_child_samples": 50, "colsample_bytree": .85, "subsample": .85, "subsample_freq": 1,
                  "reg_lambda": 8., "reg_alpha": .1, "random_state": int(cfg.get("random_state", 42)), "verbosity": -1}
    parameters.update(cfg["t_model"])
    parameters.update({"objective": "regression", "max_depth": 4 if profile == "l2_depth4" else 6,
                       "num_leaves": 15 if profile == "l2_depth4" else 31,
                       "n_jobs": max(1, int(cfg["workers"])), "device_type": "cpu",
                       "deterministic": True, "force_col_wise": True})
    require(meta["model_parameters"] == parameters, "完整实际模型参数偏离独立预声明重建")
    require(all(bundle["model"].get_params()[key] == value for key, value in parameters.items()), "拟合器实际参数与metadata不同")
    require(meta["lightgbm_version"] == lightgbm.__version__, "重载LightGBM版本与训练不同")
    sample = matrix.iloc[np.linspace(0, len(matrix)-1, min(len(matrix), 1024), dtype=int)]
    first, second = native.predict(sample), bundle["model"].predict(sample)
    require(np.isfinite(first).all() and np.isfinite(second).all(), "原生重载非有限预测")
    error = float(np.max(np.abs(first-second)))
    close(error, 0., "训练原生重载", 1e-12)
    reload_audit = auditor.json(directory/"native_reload_audit.json")
    require(reload_audit["passed"] is True and reload_audit["max_error"] <= 1e-12
            and reload_audit["fitted_rows"] == len(training)
            and reload_audit["training_inputs_sha256"] == meta["training_inputs_file_sha256"], "原生重载生产证明身份不符")
    return {"path": str(directory.relative_to(auditor.report)), "rows": len(training), "features": features,
            "passed": True, "independently_rebuilt_training_inputs": True, "raw_reference_target": True,
            "recovery_identity": IDENTITY, "source_overlap_counts": counts, "hashes": hashes,
            "training_split_independently_checked": full_frame is not None, "training_split": split_summary,
            "raw_zero_volume_reference_labels": raw_zero_volume,
            "all_native_trees_equal_joblib": True, "native_reload_sample_rows": len(sample),
            "native_reload_max_error": error, "all_model_parameters_exact": True}


def self_tests() -> dict:
    """六种独立时钟/收益/隔离反例；不构建LightGBM、市场或账户。"""
    fold = {"core_years": [2024, 2025], "inner_year": 2026, "test_year": 2023}
    cfg = {"exclude_flat_signal_bars": False, "strict_feature_provenance": True, "embargo_days": 7, "lot_size": 100}
    clocks = pd.to_datetime(["2024-03-12 10:00", "2025-03-12 14:00"])
    source = pd.to_datetime(["2024-02-01", "2025-02-01"])
    frame = pd.DataFrame({"datetime": clocks, "symbol": ["A", "B"], "feature_source_start": source,
                          "execution_time": clocks+pd.Timedelta(minutes=1), "label_start": clocks+pd.Timedelta(minutes=1),
                          "recovery_time": clocks.normalize()+pd.Timedelta(hours=14, minutes=55),
                          "label_end": clocks.normalize()+pd.Timedelta(hours=14, minutes=55),
                          "execution_price": [10., 20.], "recovery_price": [10.1, 19.9], "label_available": True,
                          "baseline_opening_shares": [1500, 700], "cohort_target_year": clocks.year})
    frame["target_return"] = frame.recovery_price/frame.execution_price-1
    for name, value in IDENTITY.items(): frame[name] = value
    validate_close_training_identity(frame, fold, cfg)
    kept, _, _ = independent_close_training_split(frame, fold, cfg)
    exact(kept, frame, "有效两core样本", dtype=True)
    for failure in ("integer_horizon", "wrong_clock", "net_target", "new_inventory"):
        bad = frame.copy()
        if failure == "integer_horizon": bad["horizon_minutes"] = 20
        elif failure == "wrong_clock": bad["label_end"] -= pd.Timedelta(minutes=1)
        elif failure == "net_target": bad["target_return"] -= .001
        else: bad["baseline_opening_shares"] = 0
        try:
            validate_close_training_identity(bad, fold, cfg)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"未拒绝反例：{failure}")
    bad = frame.copy()
    bad.loc[0, "feature_source_start"] = pd.Timestamp("2023-12-31")
    bad.loc[1, ["recovery_price", "target_return"]] = np.nan
    bad.loc[1, "label_available"] = False
    kept, split, _ = independent_close_training_split(bad, fold, cfg)
    require(kept.empty and len(bad) == 2, "隔离和缺标签必须只作用训练不消灭原机会")
    counts = split.set_index("reason").excluded_rows
    require(counts["excluded_year_or_boundary_embargo"] == 1 and counts["label_unavailable"] == 1,
            "来源/缺标签原因遗漏")
    return {"passed": True, "cases": 6, "models_fit": 0, "raw_market_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(), ensure_ascii=False, indent=2))
    else:
        parser.error("本文件由主审计器导入；独立运行仅支持--self-test")
