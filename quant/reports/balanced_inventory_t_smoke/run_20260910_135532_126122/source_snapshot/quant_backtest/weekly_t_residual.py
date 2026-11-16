"""严格核心年度内按同刻队列居中训练T残差，推断市场收益使用固定零先验。

只消费调用方的本折OOF训练域；未来raw收益仅作标签，不能进入推断特征。
准备函数无写入副作用；拟合函数在新目录保存完整供体、目标/来源审计和模型。
零市场先验是预声明假设，最终盈亏仍须按原股票、T+1、费用和真实成交判断。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Real
from pathlib import Path
from typing import Iterable

import joblib
import lightgbm
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from .weekly_t_models import (
    create_model_parameters, predict_model_values, select_training_features,
    select_training_rows, write_model_json,
)


TARGET_MODE = "same_clock_cohort_residual_zero_market"
DECOMPOSITION_COLUMNS = ["datetime", "symbol", "raw_target_return", "cohort_mean_return",
                         "residual_target_return", "cohort_size", "feature_source_start",
                         "label_start", "label_end", "kept_for_training"]


def validate_residual_config(cfg: dict) -> dict:
    """要求事前固定残差模式、零市场先验、至少两股以及严格量价来源隔离配置。"""
    if not isinstance(cfg, dict) or cfg.get("t_target_mode") != TARGET_MODE:
        raise ValueError(f"t_target_mode 必须为 {TARGET_MODE}")
    prior = cfg.get("t_market_prior")
    if isinstance(prior, (bool, np.bool_)) or not isinstance(prior, Real) or not np.isfinite(prior) or prior != 0:
        raise ValueError("t_market_prior 必须为有限数值0，不能拟合或传入实际未来市场收益")
    size = cfg.get("t_residual_min_group_size")
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size != 2:
        raise ValueError("t_residual_min_group_size 必须为预声明整数2")
    if cfg.get("strict_feature_provenance") is not True:
        raise ValueError("残差供体必须启用 strict_feature_provenance=True")
    excluded = cfg.get("excluded_model_features", [])
    if not isinstance(excluded, (list, tuple, set)) or "f_history_days" not in excluded:
        raise ValueError("严格隔离配置必须显式从模型排除 f_history_days")
    return {"t_target_mode": TARGET_MODE, "t_market_prior": 0.0, "t_residual_min_group_size": 2}


def calculate_residual_frame_sha256(frame: pd.DataFrame) -> str:
    """对既定列顺序、行顺序及类型摘要，不包含DataFrame索引。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def prepare_residual_training(frame: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                              excluded_years: Iterable[int] = ()) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """先严格筛raw标签，再用完整同刻合法供体居中，返回训练/元数据/排除审计/供体表。

training保留原raw target_return，另加非f_列target_residual_return。供体包括全部
原严格筛选后的行，singleton也留在decomposition供审计，但不进入训练或支持域。
同刻标签起止时刻必须相同，不能混合不同horizon。均值计算早于特征选择和行采样，
不会读取调用表外的股票、内部验证年或外层标签；所有输入及其原标签保持不变。
"""
    resolved = validate_residual_config(cfg)
    raw, metadata, split_audit = select_training_rows(frame, train_years, cfg, "intraday", excluded_years)
    raw = raw.copy(deep=True)
    raw["target_return"] = pd.to_numeric(raw.target_return, errors="raise")
    start = "label_start" if "label_start" in raw else "execution_time"
    end = "label_end" if "label_end" in raw else "recovery_time"
    decomposition = raw[["datetime", "symbol", "feature_source_start"]].copy()
    decomposition["raw_target_return"] = raw.target_return.astype(float)
    decomposition["label_start"] = pd.to_datetime(raw[start])
    decomposition["label_end"] = pd.to_datetime(raw[end])
    endpoint_counts = decomposition.groupby("datetime")[["label_start", "label_end"]].nunique()
    if endpoint_counts.gt(1).any().any():
        raise ValueError("同datetime残差供体的label_start/label_end必须一致，不能混合周期")
    groups = decomposition.groupby("datetime", sort=False).raw_target_return
    decomposition["cohort_mean_return"] = groups.transform("mean")
    decomposition["cohort_size"] = groups.transform("size").astype("int64")
    decomposition["residual_target_return"] = decomposition.raw_target_return - decomposition.cohort_mean_return
    if not np.isfinite(decomposition[["cohort_mean_return", "residual_target_return"]]).all().all():
        raise ValueError("供体均值或残差不是有限数值")
    decomposition["kept_for_training"] = decomposition.cohort_size.ge(resolved["t_residual_min_group_size"])
    decomposition = decomposition[DECOMPOSITION_COLUMNS].copy()
    keep = decomposition.kept_for_training
    training = raw.loc[keep].copy(deep=True)
    training["target_residual_return"] = decomposition.loc[keep, "residual_target_return"]
    minimum = int(cfg.get("min_training_rows", 100))
    if len(training) < minimum:
        raise ValueError(f"去除不足2股同刻组后，有效残差训练样本{len(training)}少于min_training_rows={minimum}")
    kept_targets = decomposition.loc[keep, ["symbol", "datetime", "residual_target_return"]]
    source_columns = ["symbol", "datetime", "feature_source_start"]
    group_sizes = decomposition.groupby("datetime", sort=False).cohort_size.first()
    group_residual_means = decomposition.groupby("datetime", sort=False).residual_target_return.mean()
    metadata = {**metadata, **resolved,
        "raw_training_rows": len(raw), "training_rows": len(training),
        "donor_rows": len(decomposition), "donor_clock_count": len(group_sizes),
        "training_clock_count": int(training.datetime.nunique()),
        "singleton_rows_excluded": int((~keep).sum()), "singleton_clock_count": int(group_sizes.lt(2).sum()),
        "donor_group_size_min": int(group_sizes.min()), "donor_group_size_max": int(group_sizes.max()),
        "donor_group_size_median_over_clocks": float(group_sizes.median()),
        "max_absolute_donor_clock_mean_residual": float(group_residual_means.abs().max()),
        "training_start": str(training.datetime.min()), "training_end": str(training.datetime.max()),
        "label_start_min": str(pd.to_datetime(training[start]).min()),
        "label_end_max": str(pd.to_datetime(training[end]).max()),
        "feature_source_start_min": str(pd.to_datetime(training.feature_source_start).min()),
        "raw_training_target_sha256": calculate_residual_frame_sha256(
            decomposition[["symbol", "datetime", "raw_target_return"]]),
        "residual_training_target_sha256": calculate_residual_frame_sha256(kept_targets),
        "target_decomposition_sha256": calculate_residual_frame_sha256(decomposition),
        "donor_source_sha256": calculate_residual_frame_sha256(
            decomposition[source_columns + ["label_start", "label_end"]]),
        "feature_source_full_training_sha256": calculate_residual_frame_sha256(training[source_columns]),
        "target_decomposition_columns": list(DECOMPOSITION_COLUMNS),
        "donor_scope": "仅调用方OOF训练域经过原raw标签/真实来源严格年度purge后的完整同刻队列；不读表外市场或留出年",
        "residual_formula": "target_residual_return=raw_target_return-mean(raw_target_return|datetime,strict_core_donors)",
        "residual_centering_stage": "先raw年度/来源/标签purge，再全供体均值，再去singleton，再特征选择与模型行采样",
        "sample_cap_contract": "行采样不重估同刻均值；采样后单组残差均值不要求仍为0",
        "raw_label_unchanged": True,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("weekly_t_residual.py", "weekly_t_models.py")}}
    audit = pd.concat([split_audit, pd.DataFrame([
        {"reason": "residual_singleton_rows_excluded", "excluded_rows": int((~keep).sum())},
        {"reason": "retained_residual_training_rows", "excluded_rows": len(training)}])], ignore_index=True)
    return training, metadata, audit, decomposition


def fit_residual_t_model(frame: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                         output_dir: Path | str, excluded_years: Iterable[int] = ()) -> dict:
    """独立拟合真实残差标签，保留v7的profile/特征选择/采样规则并写可核验新产物。

不调用旧fit_t_model，也不将raw标签模型改名。完整供体表和居中前后标签SHA在采样
之前确定，训练样本SHA则按实际所选特征、实际采样行和残差标签计算。输出目录
必须为空，避免覆盖已冻结模型。全部输入市场f_*仍按原训练特征选择规则处理。
"""
    validate_residual_config(cfg)
    directory = Path(output_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("残差模型输出目录必须不存在或为空，不能覆盖冻结产物")
    training, metadata, split_audit, decomposition = prepare_residual_training(frame, train_years, cfg, excluded_years)
    targets = pd.DataFrame({"return": training.target_residual_return})
    features, feature_audit = select_training_features(training, targets, cfg, "intraday")
    parameters, profile = create_model_parameters(cfg, "intraday")
    cap = int(cfg.get("t_max_train_rows", cfg.get("max_train_rows", 0)))
    if cap < 0:
        raise ValueError("max_train_rows不能为负数")
    fitted = training.sample(n=cap, random_state=int(parameters["random_state"])).sort_index() if cap and len(training) > cap else training
    x = fitted[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    residual_targets = targets.loc[fitted.index, "return"]
    print(f"intraday_residual/{profile}：训练年{metadata['train_years']}，完整供体{len(decomposition):,}行，"
          f"可拟合{len(training):,}行/{metadata['training_clock_count']:,}时刻，采样{len(fitted):,}行，{len(features)}特征", flush=True)
    model = LGBMRegressor(**parameters)
    model.fit(x, residual_targets)
    keys = fitted[["symbol", "datetime"]]
    source_keys = fitted[["symbol", "datetime", "feature_source_start"]]
    residual_signature = pd.concat([keys, x, residual_targets], axis=1)
    raw_signature = pd.concat([keys, x, fitted.target_return.rename("return")], axis=1)
    metadata.update({"fitted_rows": len(fitted), "feature_names": features, "profile": profile,
        "model_parameters": parameters, "lightgbm_version": lightgbm.__version__,
        "training_sample_sha256": calculate_residual_frame_sha256(residual_signature),
        "raw_training_sample_sha256": calculate_residual_frame_sha256(raw_signature),
        "feature_source_sha256": calculate_residual_frame_sha256(source_keys),
        "training_row_cap": cap, "resolved_config": deepcopy(cfg),
        "input_contract": "仅训练选择的因果f_*；同刻未来均值、供体数量及所有未来价格/标签不是推断输入",
        "label_definition": "严格core同刻OOF队列raw horizon收益减完整供体均值；以无量纲原始收益为单位的残差",
        "prediction_scale": "raw股票收益的建模近似=residual_prediction+固定market_prediction(0)",
        "inference_contract": "只按已有当前因果资格推断，不按未来标签、未来均值、未来供体数量或恢复价删行；不加回真实未来共同项",
        "profile_selection_scope": "仅外层训练内部验证选择；外层仍使用同一核心模型权重",
        "economic_limitation": "市场零先验不代表未来市场为0；最终必须核原股票实际净损益，残差PnL不能代替账户PnL"})
    directory.mkdir(parents=True, exist_ok=True)
    decomposition_path = directory / "target_decomposition.csv"
    decomposition.to_csv(decomposition_path, index=False, encoding="utf-8-sig")
    metadata["target_decomposition_file_sha256"] = hashlib.sha256(decomposition_path.read_bytes()).hexdigest()
    model.booster_.save_model(str(directory / "return_model.txt"))
    bundle = {"task": "intraday", "target_mode": TARGET_MODE, "models": {"return": model},
              "features": features, "metadata": metadata}
    joblib.dump(bundle, directory / "model_bundle.joblib")
    write_model_json(directory / "model_metadata.json", metadata)
    write_model_json(directory / "resolved_config.json", cfg)
    split_audit.to_csv(directory / "training_split_audit.csv", index=False)
    feature_audit.to_csv(directory / "feature_selection_audit.csv", index=False)
    manifest = pd.DataFrame({"feature": features, "selection_order": range(1, len(features) + 1)})
    manifest.merge(feature_audit, on="feature", validate="one_to_one").to_csv(directory / "used_feature_manifest.csv", index=False)
    (directory / "feature_list.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    return bundle


def predict_residual_t(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """保留已有因果推断行，给出残差预测加固定0；不在当前推断表上重新居中。

单股推断合法，未来标签及同刻未来供体数量完全不消费。与旧predict_t一样按
datetime/symbol稳定排序；原字段保留，新增prediction/residual_prediction/
market_prediction和model_profile。禁止误用旧raw目标bundle或非零市场先验。
"""
    validate_residual_config(cfg)
    metadata = bundle.get("metadata", {})
    if (bundle.get("target_mode") != TARGET_MODE or metadata.get("t_target_mode") != TARGET_MODE
            or metadata.get("t_market_prior") != 0.0 or set(bundle.get("models", {})) != {"return"}):
        raise ValueError("推断必须使用已声明残差目标和固定零市场先验的单模型bundle")
    source, predicted = predict_model_values(bundle, frame, cfg, "intraday")
    values = np.asarray(predicted["return"], dtype=float)
    if values.shape != (len(source),) or not np.isfinite(values).all():
        raise ValueError("残差模型预测形状或有限性非法")
    result = source.copy(deep=True)
    result["residual_prediction"] = values
    result["market_prediction"] = 0.0
    result["prediction"] = result.residual_prediction + result.market_prediction
    result["model_profile"] = metadata["profile"]
    result.attrs["metadata"] = {"target_mode": TARGET_MODE, "market_prior": 0.0,
                                 "prediction_scale": metadata["prediction_scale"],
                                 "future_market_mean_consumed": False, "minimum_future_group_size_required": False}
    return result.sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
