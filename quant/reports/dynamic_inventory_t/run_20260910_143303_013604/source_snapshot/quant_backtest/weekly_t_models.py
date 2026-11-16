"""周度选股及库存做 T 模型：输入因果 f_* 面板，训练年份内拟合并保存模型审计。

从 quant 仓库调用。周模型分别预测下一周收益/日均振幅的截面排名，再按 70/30
组合排序；T 模型预测下一分钟成交至同日恢复时刻的原始收益。只有 f_* 是输入，
未来价格、标签及成交资格不作为特征。输出目录写入模型与配置，不改行情或缓存。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import lightgbm
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


T_MODEL_PROFILES = {
    "l2_depth4": {"objective": "regression", "max_depth": 4, "num_leaves": 15},
    "l2_depth6": {"objective": "regression", "max_depth": 6, "num_leaves": 31},
    "huber_depth4": {"objective": "huber", "alpha": 0.01, "max_depth": 4, "num_leaves": 15},
}


def write_model_json(path: Path, value: object) -> None:
    """将模型元数据以 UTF-8 JSON 写入调用方输出目录，不修改其他产物。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_t_model_profiles() -> list[dict]:
    """返回预声明的三个 T 回归配置；调用方只能用训练内部结果选择。"""
    return [{"name": name, **parameters} for name, parameters in T_MODEL_PROFILES.items()]


def select_causal_inference_rows(frame: pd.DataFrame, cfg: dict, task: str) -> pd.DataFrame:
    """只凭当前已知状态选择推断行，不读取标签、未来周完整性或实际恢复价格。"""
    key = "signal_datetime" if task == "weekly" else "datetime"
    required = {key, "symbol"} | ({"trade_week"} if task == "weekly" else set())
    if not required.issubset(frame):
        raise ValueError(f"{task} 推断缺少键列：{sorted(required.difference(frame.columns))}")
    if frame[list(required)].isna().any().any() or frame.duplicated([key, "symbol"]).any():
        raise ValueError(f"{task} 面板存在缺失或重复时间/股票键")
    if task == "weekly" and frame.duplicated(["trade_week", "symbol"]).any():
        raise ValueError("同一 trade_week/symbol 不能提供多条周预测样本")
    mask = pd.Series(True, index=frame.index)
    if task == "weekly" and "feature_week_complete" in frame:
        mask &= frame.feature_week_complete.eq(True)
    if task == "intraday" and cfg.get("exclude_flat_signal_bars", True) and "signal_bar_tradable" in frame:
        mask &= frame.signal_bar_tradable.eq(True)
    return frame.loc[mask].copy().reset_index(drop=True)


def select_training_rows(frame: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                         task: str, excluded_years: Iterable[int] = ()) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """校验特征/标签端点年份并清除留出年边界前后 embargo_days 天，返回训练及审计。

    周样本检查 feature_week、signal_datetime、label_start、label_end；分钟样本检查
    datetime 和 label_start/end（缺列时分别用 execution_time/recovery_time）。标签
    区间不得穿越排除年份。训练可检查未来完整标签，推断不会采用这些检查。
    """
    if task not in {"weekly", "intraday"}:
        raise ValueError("task 必须为 weekly 或 intraday")
    years, excluded = sorted(set(map(int, train_years))), sorted(set(map(int, excluded_years)))
    if not years or set(years).intersection(excluded):
        raise ValueError("训练年份须非空且与排除年份互斥")
    embargo = int(cfg.get("embargo_days", 7))
    if embargo < 0:
        raise ValueError("embargo_days 不能为负数")
    source = select_causal_inference_rows(frame, cfg, task)
    signal_column = "signal_datetime" if task == "weekly" else "datetime"
    start_column = "label_start" if "label_start" in source else "execution_time"
    end_column = "label_end" if "label_end" in source else "recovery_time"
    required = {signal_column, start_column, end_column, "target_return"}
    if task == "weekly":
        required |= {"feature_week", "trade_week", "target_amplitude"}
    if not required.issubset(source):
        raise ValueError(f"{task} 训练缺列：{sorted(required.difference(source.columns))}")
    signal = pd.to_datetime(source[signal_column], errors="coerce")
    label_start = pd.to_datetime(source[start_column], errors="coerce")
    label_end = pd.to_datetime(source[end_column], errors="coerce")
    feature_start = pd.to_datetime(source.feature_week, errors="coerce") if task == "weekly" else signal
    strict_provenance = bool(cfg.get("strict_feature_provenance", False))
    if strict_provenance:
        if "feature_source_start" not in source:
            raise ValueError("严格年度隔离需要feature_source_start真实历史来源列")
        feature_start = pd.to_datetime(source.feature_source_start, errors="coerce")
    checks = {
        "feature_year_not_training": ~feature_start.dt.year.isin(years),
        "signal_year_not_training": ~signal.dt.year.isin(years),
        "label_start_year_not_training": ~label_start.dt.year.isin(years),
        "label_end_year_not_training": ~label_end.dt.year.isin(years),
        "noncausal_time_order": ~((feature_start <= signal) & (signal < label_start) & (label_start <= label_end)),
    }
    if task == "weekly":
        checks["feature_week_year_not_training"] = ~pd.to_datetime(source.feature_week).dt.year.isin(years)
    if task == "intraday":
        checks["intraday_label_crosses_day"] = ~((signal.dt.normalize() == label_start.dt.normalize()) &
                                                  (signal.dt.normalize() == label_end.dt.normalize()))
    target_columns = ["target_return", "target_amplitude"] if task == "weekly" else ["target_return"]
    for column in target_columns:
        values = pd.to_numeric(source[column], errors="coerce")
        checks[f"invalid_{column}"] = ~np.isfinite(values)
        if column == "target_amplitude":
            checks["negative_target_amplitude"] = values < 0
    if "trade_week_complete" in source:
        checks["incomplete_label_week"] = ~source.trade_week_complete.eq(True)
    if "label_available" in source:
        checks["label_unavailable"] = ~source.label_available.eq(True)
    boundary_mask = pd.Series(False, index=source.index)
    for year in excluded:
        left = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=embargo)
        right = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=embargo)
        boundary_mask |= (feature_start < right) & (label_end >= left)
    checks["excluded_year_or_boundary_embargo"] = boundary_mask
    valid = ~pd.DataFrame(checks, index=source.index).any(axis=1)
    selected = source.loc[valid].sort_values([signal_column, "symbol"], kind="stable").copy()
    min_rows = int(cfg.get("min_training_rows", 100))
    if min_rows < 2 or len(selected) < min_rows:
        raise ValueError(f"{task} 有效训练样本 {len(selected)} 少于 min_training_rows={min_rows}")
    rows = [{"reason": "causal_inference_filter", "excluded_rows": len(frame) - len(source)}]
    rows.extend({"reason": reason, "excluded_rows": int(mask.sum())} for reason, mask in checks.items())
    rows.append({"reason": "retained_training_rows", "excluded_rows": len(selected)})
    metadata = {"task": task, "train_years": years, "excluded_years": excluded,
                "input_rows": len(frame), "causally_eligible_rows": len(source), "training_rows": len(selected),
                "training_start": str(pd.to_datetime(selected[signal_column]).min()),
                "training_end": str(pd.to_datetime(selected[signal_column]).max()),
                "label_start_min": str(pd.to_datetime(selected[start_column]).min()),
                "label_end_max": str(pd.to_datetime(selected[end_column]).max()), "embargo_days": embargo,
                "embargo_definition": "训练样本feature_start至label_end不得与排除年份及其前后日历隔离天数相交",
                "strict_feature_provenance": strict_provenance,
                "feature_source_start_min": str(feature_start.loc[selected.index].min()),
                "embargo_limitation": ("按真实最早量价来源隔离；包含未来训练年的LOYO仍为回顾性研究"
                    if strict_provenance else "隔离窗口不重置更长历史特征；包含未来训练年的LOYO仅为回顾性研究"),
                "purge_reason_counts_overlap": True}
    return selected, metadata, pd.DataFrame(rows)


def select_training_features(training: pd.DataFrame, targets: pd.DataFrame, cfg: dict,
                             task: str) -> tuple[list[str], pd.DataFrame]:
    """只用训练行 f_* 覆盖率、标签秩相关和特征相关性选特征，缺失留给 LightGBM。"""
    excluded_features = set(cfg.get("excluded_model_features", []))
    columns = sorted(column for column in training if column.startswith("f_") and column not in excluded_features)
    if not columns:
        raise ValueError("面板没有因果 f_* 输入特征")
    maximum = int(cfg.get("weekly_max_features" if task == "weekly" else "t_max_features", 48))
    sample_size = int(cfg.get("feature_selection_sample_rows", 20000))
    coverage_floor = float(cfg.get("feature_min_coverage", .8))
    correlation_limit = float(cfg.get("feature_max_correlation", .95))
    if maximum < 1 or sample_size < 2 or not 0 <= coverage_floor <= 1 or not 0 < correlation_limit <= 1:
        raise ValueError("非法特征选择配置")
    seed = int(cfg.get("random_state", 42))
    indices = training.sample(n=min(sample_size, len(training)), random_state=seed).index
    values = training[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    sample = values.loc[indices]
    coverage = values.notna().mean()
    diversity = sample.nunique(dropna=True)
    eligible = [column for column in columns if coverage[column] >= coverage_floor and diversity[column] > 1]
    target_rank = targets.loc[indices].rank(pct=True)
    ranks = sample[eligible].rank(pct=True)
    relevance = pd.Series(0.0, index=columns)
    # 常数/全缺失列已排除，不让无意义相关计算污染日志或触发除零警告。
    correlations_to_target = [ranks.corrwith(target_rank[column]).abs() for column in target_rank
                              if target_rank[column].nunique() > 1]
    if correlations_to_target and eligible:
        relevance.loc[eligible] = pd.concat(correlations_to_target, axis=1).max(axis=1).fillna(0)
    ordered = sorted(eligible, key=lambda column: (-float(relevance[column]), column))
    correlations = sample[ordered].corr(method="spearman")
    selected, audit = [], []
    for column in ordered:
        peers = correlations.loc[column, selected].abs().dropna() if selected else pd.Series(dtype=float)
        strongest = float(peers.max()) if len(peers) else 0.0
        reason = "correlation_limit" if strongest > correlation_limit else "feature_limit" if len(selected) >= maximum else "selected"
        if reason == "selected":
            selected.append(column)
        audit.append({"feature": column, "coverage": coverage[column], "target_rank_relevance": relevance[column],
                      "max_selected_correlation": strongest, "selected": reason == "selected", "reason": reason})
    for column in columns:
        if column not in eligible:
            audit.append({"feature": column, "coverage": coverage[column], "target_rank_relevance": relevance[column],
                          "max_selected_correlation": 0.0, "selected": False,
                          "reason": "coverage" if coverage[column] < coverage_floor else "constant"})
    if not selected:
        raise ValueError("训练覆盖率及相关性筛选后没有可用特征")
    return selected, pd.DataFrame(audit)


def create_model_parameters(cfg: dict, task: str) -> tuple[dict, str]:
    """合并固定 CPU LightGBM 配置；T profile 仅在预声明三个候选中选择。"""
    profile = str(cfg.get("t_model_profile", "l2_depth4")) if task == "intraday" else "weekly_dual_rank"
    if task == "intraday" and profile not in T_MODEL_PROFILES:
        raise ValueError(f"未知 t_model_profile：{profile}")
    parameters = {"n_estimators": 160, "learning_rate": .04, "max_depth": 4, "num_leaves": 15,
                  "min_child_samples": 50, "colsample_bytree": .85, "subsample": .85,
                  "subsample_freq": 1, "reg_lambda": 8.0, "reg_alpha": .1,
                  "random_state": int(cfg.get("random_state", 42)), "verbosity": -1}
    parameters.update(cfg.get("weekly_model" if task == "weekly" else "t_model", {}))
    parameters.update(T_MODEL_PROFILES[profile] if task == "intraday" else {"objective": "regression"})
    if profile == "huber_depth4":
        # LightGBM 的 Huber alpha 是原始残差截断值；默认0.9对分钟收益几乎等于L2。
        parameters["alpha"] = float(cfg.get("t_huber_alpha", .01))
        if not np.isfinite(parameters["alpha"]) or parameters["alpha"] <= 0:
            raise ValueError("t_huber_alpha 须为有限正数")
    parameters.update({"n_jobs": max(1, int(cfg.get("workers", 4))), "device_type": "cpu",
                       "deterministic": True, "force_col_wise": True})
    return parameters, profile


def fit_models(frame: pd.DataFrame, train_years: Iterable[int], cfg: dict,
               output_dir: Path | str, task: str, excluded_years: Iterable[int] = ()) -> dict:
    """拟合周双模型或分钟单模型并保存原生权重、bundle、特征和年份审计。"""
    training, metadata, split_audit = select_training_rows(frame, train_years, cfg, task, excluded_years)
    training["target_return"] = pd.to_numeric(training.target_return, errors="raise")
    if task == "weekly":
        training["target_amplitude"] = pd.to_numeric(training.target_amplitude, errors="raise")
        targets = pd.DataFrame({"return": training.groupby("trade_week").target_return.rank(pct=True),
                                "amplitude": training.groupby("trade_week").target_amplitude.rank(pct=True)})
    else:
        targets = pd.DataFrame({"return": pd.to_numeric(training.target_return, errors="raise")})
    features, feature_audit = select_training_features(training, targets, cfg, task)
    parameters, profile = create_model_parameters(cfg, task)
    cap = int(cfg.get("weekly_max_train_rows" if task == "weekly" else "t_max_train_rows", cfg.get("max_train_rows", 0)))
    if cap < 0:
        raise ValueError("max_train_rows 不能为负数")
    fitted = training.sample(n=cap, random_state=int(parameters["random_state"])).sort_index() if cap and len(training) > cap else training
    x = fitted[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    models = {}
    print(f"{task}/{profile}：训练年{metadata['train_years']}，{len(fitted):,}行，{len(features)}特征", flush=True)
    for name in targets:
        model = LGBMRegressor(**parameters)
        model.fit(x, targets.loc[fitted.index, name])
        models[name] = model
    signature_frame = pd.concat([fitted[["symbol", "signal_datetime" if task == "weekly" else "datetime"]],
                                 x, targets.loc[fitted.index]], axis=1)
    signature = hashlib.sha256(pd.util.hash_pandas_object(signature_frame, index=False).to_numpy().tobytes()).hexdigest()
    if cfg.get("strict_feature_provenance"):
        source_keys = fitted[["symbol", "signal_datetime" if task == "weekly" else "datetime", "feature_source_start"]]
        metadata["feature_source_sha256"] = hashlib.sha256(
            pd.util.hash_pandas_object(source_keys, index=False).to_numpy().tobytes()).hexdigest()
    metadata.update({"fitted_rows": len(fitted), "feature_names": features, "profile": profile,
                     "model_parameters": parameters, "lightgbm_version": lightgbm.__version__,
                     "training_sample_sha256": signature, "input_contract": "仅训练选择的f_*；未来价格及标签不是模型输入",
                     "label_definition": "下一周收益/日均振幅分别按trade_week做截面百分位排名" if task == "weekly" else
                     "recovery_price/execution_price-1原始未来收益；费用门槛与仓位规则由训练内部验证选择",
                     "inference_contract": "不按未来标签、未来周是否完整或恢复行情是否存在删除推断行",
                     "profile_selection_scope": "调用者只能使用外层训练内部验证结果选择；本函数不读取外层评价指标",
                     "resolved_config": cfg, "training_row_cap": cap})
    bundle = {"task": task, "models": models, "features": features, "metadata": metadata}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.booster_.save_model(str(directory / f"{name}_model.txt"))
    joblib.dump(bundle, directory / "model_bundle.joblib")
    write_model_json(directory / "model_metadata.json", metadata)
    write_model_json(directory / "resolved_config.json", cfg)
    split_audit.to_csv(directory / "training_split_audit.csv", index=False)
    feature_audit.to_csv(directory / "feature_selection_audit.csv", index=False)
    manifest = pd.DataFrame({"feature": features, "selection_order": range(1, len(features) + 1)})
    manifest.merge(feature_audit, on="feature", validate="one_to_one").to_csv(directory / "used_feature_manifest.csv", index=False)
    (directory / "feature_list.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    return bundle


def fit_weekly_models(frame: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                      output_dir: Path | str, excluded_years: Iterable[int] = ()) -> dict:
    """使用训练年份完整周标签拟合周收益及振幅两个截面秩模型，返回持久化 bundle。"""
    return fit_models(frame, train_years, cfg, output_dir, "weekly", excluded_years)


def fit_t_model(frame: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                output_dir: Path | str, excluded_years: Iterable[int] = ()) -> dict:
    """使用训练年份的完整同日未来收益拟合指定 T profile，不在函数内选择交易门槛。"""
    return fit_models(frame, train_years, cfg, output_dir, "intraday", excluded_years)


def predict_model_values(bundle: dict, frame: pd.DataFrame, cfg: dict, task: str) -> tuple[pd.DataFrame, dict]:
    """严格按训练特征顺序推断，通过原生缺失分支处理 NaN，不查看任何未来标签。"""
    if bundle.get("task") != task:
        raise ValueError("模型 bundle 的 task 与推断 API 不一致")
    missing = set(bundle["features"]).difference(frame.columns)
    if missing:
        raise ValueError(f"推断缺少训练特征：{sorted(missing)}")
    source = select_causal_inference_rows(frame, cfg, task)
    x = source[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    predicted = {name: model.predict(x) if len(x) else np.array([], dtype=float) for name, model in bundle["models"].items()}
    return source, predicted


def predict_weekly(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """返回每周全部当前合格股票的双预测及 70/30 排名；同分股票按 symbol 稳定排序。

    pred_return/pred_amplitude 是两个模型预测的截面排名尺度，不是百分比价格回报。
    默认 selected 为每个 trade_week 前 100 只，score 只使用本周股票预测排名。
    """
    source, predicted = predict_model_values(bundle, frame, cfg, "weekly")
    weights = (float(cfg.get("weekly_return_weight", .7)), float(cfg.get("weekly_amplitude_weight", .3)))
    if min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("周收益/振幅排名权重须非负且和为 1")
    top_n = int(cfg.get("weekly_top_n", 100))
    if top_n < 1:
        raise ValueError("weekly_top_n 须为正数")
    keys = [column for column in ("symbol", "feature_week", "signal_datetime", "trade_week") if column in source]
    result = source[keys].copy()
    result["pred_return"], result["pred_amplitude"] = predicted["return"], predicted["amplitude"]
    result["pred_return_rank"] = result.groupby("trade_week").pred_return.rank(pct=True)
    result["pred_amplitude_rank"] = result.groupby("trade_week").pred_amplitude.rank(pct=True)
    result["score"] = weights[0] * result.pred_return_rank + weights[1] * result.pred_amplitude_rank
    result = result.sort_values(["trade_week", "score", "symbol"], ascending=[True, False, True], kind="stable")
    result["rank"] = result.groupby("trade_week").cumcount() + 1
    result["selected"] = result["rank"] <= top_n
    result.attrs["metadata"] = {"weights": weights, "top_n": top_n,
                                 "prediction_scale": "两模型的训练目标为周截面排名；pred_return不是原始预期收益率",
                                 "feature_names": bundle["features"], "train_years": bundle["metadata"]["train_years"]}
    return result.reset_index(drop=True)


def predict_t(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """返回原分钟字段及原始预期收益 prediction；保留未来执行字段但绝不输入模型。

    prediction 为 recovery/execution-1，上涨为正；调用方按符号选择先买/先卖，
    费用门槛只用训练内部回测选择。保留字段供执行引擎使用，不属于模型特征。
    """
    source, predicted = predict_model_values(bundle, frame, cfg, "intraday")
    result = source.copy()
    result["prediction"] = predicted["return"]
    result["model_profile"] = bundle["metadata"]["profile"]
    result.attrs["metadata"] = {"prediction_scale": "recovery_price/execution_price-1",
                                 "profile": bundle["metadata"]["profile"], "costs_in_prediction": False}
    return result.sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
