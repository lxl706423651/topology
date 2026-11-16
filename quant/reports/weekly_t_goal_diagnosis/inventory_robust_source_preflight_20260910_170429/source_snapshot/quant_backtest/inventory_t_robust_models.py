"""永久底仓原始短周期标签的Huber回归；保持纯核心样本，独立保存真实拟合输入。

标签是下一分钟到固定期限的原始参考价格收益，不是共享现金账户的实际T利润。
Huber输出为收益位置分数，不保证是校准条件均值或共享账户净额。alpha=0.01
只改变原始残差的训练损失，不截断标签或删去尾部样本；内部/外层只推断。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import lightgbm
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from .weekly_t_models import (create_model_parameters, select_training_features,
    select_training_rows, write_model_json)
from .weekly_t_quote_store import calculate_file_sha256
from .weekly_t_short_horizon import planned_recovery_times
from .inventory_t_learning_models import attach_learning_labels


VERSION = "inventory_robust_t_v17"
PROFILES = ("huber_depth4", "huber_depth6")
SOURCE_PROFILE_MAP = {"huber_depth4": "l2_depth4", "huber_depth6": "l2_depth6"}
HORIZONS = (20, 40, 60)


def calculate_learning_frame_sha256(frame: pd.DataFrame) -> str:
    """固定实际列序和行序的内容摘要，供训练输入与原生重载审计。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def validate_robust_model_config(cfg: dict) -> tuple[str, int]:
    """固定两种Huber深度、原三期限及七日来源隔离，禁止额外抽样或改写目标。"""
    profile, horizon = cfg.get("t_model_profile"), cfg.get("t_label_horizon_minutes")
    if profile not in PROFILES or isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError("训练必须指定预声明的回归器和20/40/60分钟期限")
    if (cfg.get("training_loss") != "huber" or isinstance(cfg.get("t_huber_alpha"), bool)
            or cfg.get("t_huber_alpha") != .01
            or cfg.get("forecast_target") != "fixed_20_40_60_reference_return"):
        raise ValueError("本轮只检验原固定期限原始收益的Huber损失alpha=0.01")
    if tuple(cfg.get("horizons", ())) != HORIZONS or any(isinstance(value, bool) for value in cfg["horizons"]):
        raise ValueError("必须保留原20/40/60预测期限")
    if "objective" in cfg.get("t_model", {}) or "alpha" in cfg.get("t_model", {}):
        raise ValueError("objective和alpha由本轮固定损失配置唯一确定")
    if cfg.get("strict_feature_provenance") is not True or "f_history_days" not in cfg.get("excluded_model_features", []):
        raise ValueError("必须严格隔离实际来源并排除累计历史天数特征")
    if cfg.get("exclude_flat_signal_bars") is not False:
        raise ValueError("本轮保持所有当前信号；可成交性由实际执行器判断")
    if cfg.get("embargo_days") != 7:
        raise ValueError("训练保持预声明7个日历日边界隔离")
    if type(cfg.get("t_max_train_rows")) is not int or cfg["t_max_train_rows"] != 0:
        raise ValueError("本轮保留全部有效核心行，t_max_train_rows必须为0")
    return str(profile), int(horizon)


def create_robust_model_parameters(cfg: dict) -> dict:
    """从显式映射的原L2参数工厂继承全部参数，仅替换objective并添加固定alpha。"""
    profile, _ = validate_robust_model_config(cfg)
    parameters, _ = create_model_parameters({**cfg, "t_model_profile": SOURCE_PROFILE_MAP[profile]}, "intraday")
    return {**parameters, "objective": "huber", "alpha": .01}


def calculate_fit_residual_statistics(target: np.ndarray, prediction: np.ndarray) -> dict:
    """描述实际未裁剪训练残差；行分母与损失分母分别保存，不据此再选择参数。"""
    residual = np.asarray(target, dtype=float)-np.asarray(prediction, dtype=float)
    absolute = np.abs(residual)
    tail = absolute > .01
    squared_sum, absolute_sum = float(np.square(residual).sum()), float(absolute.sum())
    return {"rows": len(residual), "alpha": .01, "abs_residual_gt_alpha_rows": int(tail.sum()),
        "abs_residual_gt_alpha_fraction": float(tail.mean()),
        "sum_squared_residual": squared_sum, "sum_absolute_residual": absolute_sum,
        "tail_squared_residual_share": float(np.square(residual[tail]).sum()/squared_sum) if squared_sum else 0.,
        "tail_absolute_residual_share": float(absolute[tail].sum()/absolute_sum) if absolute_sum else 0.,
        "target_clipped": False, "tail_rows_removed": False, "scope": "in_sample_core_fit_description_only"}


def fit_inventory_robust_model(frame: pd.DataFrame, train_years: list[int], cfg: dict,
                               output_dir: Path | str, excluded_years: list[int]) -> dict:
    """在入场前已隔离的永久底仓机会中再次按特征/标签边界purge后拟合。

选择特征和全部模型参数均只消费核心行，保留全部有效行。未来缺失标签只影响
训练有效性；不会影响内部/外层预测行。写新目录，拒绝覆盖已有模型。
    """
    profile, horizon = validate_robust_model_config(cfg)
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("模型目录必须为空，不能覆盖历史模型")
    if not {"cohort_target_year", "baseline_opening_shares"}.issubset(frame):
        raise ValueError("训练必须来自真实无T永久底仓日初旧股队列")
    if (not pd.to_numeric(frame.baseline_opening_shares).ge(cfg["lot_size"]).all()
            or not frame.cohort_target_year.isin(train_years).all()):
        raise ValueError("训练机会不属于核心旧股持有日")
    training, split, split_audit = select_training_rows(frame, train_years, cfg, "intraday", excluded_years)
    if not training.cohort_target_year.eq(pd.to_datetime(training.datetime).dt.year).all():
        raise ValueError("训练队列年份与当前时刻不一致")
    expected = planned_recovery_times(pd.to_datetime(training.datetime), horizon)
    if (not pd.to_datetime(training.label_start).eq(pd.to_datetime(training.datetime)+pd.Timedelta(minutes=1)).all()
            or not pd.to_datetime(training.label_end).eq(expected).all()
            or not training.horizon_minutes.eq(horizon).all()):
        raise ValueError("标签端点与当前时钟声明的固定期限不一致")
    prices = training[["execution_price", "recovery_price"]].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(training.target_return, errors="raise")
    if (not np.isfinite(prices).all().all() or not prices.gt(0).all().all()
            or not target.gt(-1).all()
            or not np.allclose(prices.recovery_price/prices.execution_price-1, target, atol=1e-12, rtol=0)):
        raise ValueError("固定端点参考价格和收益标签不一致")
    features, feature_audit = select_training_features(training, pd.DataFrame({"return": target}), cfg, "intraday")
    parameters = create_robust_model_parameters(cfg)
    fitted = training
    matrix = fitted[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    model = LGBMRegressor(**parameters)
    model.fit(matrix, pd.to_numeric(fitted.target_return))
    predicted = np.asarray(model.predict(matrix))
    if predicted.shape != (len(matrix),) or not np.isfinite(predicted).all():
        raise AssertionError("训练分数必须逐行且全为有限数")
    directory.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(directory/"return_model.txt"))
    inputs = fitted[["datetime", "symbol", "feature_source_start", "cohort_target_year",
                     "baseline_opening_shares"]].copy()
    for name in features:
        inputs[name] = matrix[name].to_numpy()
    inputs.to_pickle(directory/"training_inputs.pkl")
    trace = [name for name in training if name in ("datetime", "symbol", "feature_source_start", "cohort_target_year",
        "baseline_opening_shares", "intraday_feature_source_start", "entry_feature_source_start",
        "entry_rank_source_start", "portfolio_cash_source_start", "base_entry_datetime", "source_model_training_years",
        "source_model_sha256", "label_start", "label_end", "target_return", "execution_price", "recovery_price", "horizon_minutes")]
    labels = training[trace].copy()
    labels["fitted"] = labels.index.isin(fitted.index)
    labels.to_csv(directory/"training_labels.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(directory/"training_split_audit.csv", index=False, encoding="utf-8-sig")
    feature_audit.to_csv(directory/"feature_manifest.csv", index=False, encoding="utf-8-sig")
    metadata = {**split, "research_version": VERSION, "profile": profile, "horizon_minutes": horizon,
        "features": features, "fitted_rows": len(fitted), "model_parameters": parameters,
        "source_l2_profile": SOURCE_PROFILE_MAP[profile], "training_loss": "huber", "huber_alpha": .01,
        "only_parameter_changes_from_source_l2": {"objective": "huber", "alpha": .01},
        "score_definition": "Huber_location_score_of_original_raw_reference_return",
        "score_is_calibrated_conditional_mean": False, "score_is_actual_execution_pnl": False,
        "training_targets_clipped": False, "tail_training_rows_removed": False,
        "fit_residual_statistics": calculate_fit_residual_statistics(fitted.target_return.to_numpy(), predicted),
        "training_cohort": "permanent_inventory_core_oof_opening_old_shares",
        "target_definition": "reference_recovery_price/reference_next_minute_price-1",
        "target_is_portfolio_execution_net_pnl": False, "probability_gate_used": False,
        "full_training_rows_sha256": calculate_learning_frame_sha256(training[["datetime", "symbol", "feature_source_start"]]),
        "training_matrix_sha256": calculate_learning_frame_sha256(matrix),
        "training_target_sha256": calculate_learning_frame_sha256(fitted[["target_return"]]),
        "training_inputs_file_sha256": calculate_file_sha256(directory/"training_inputs.pkl"),
        "training_labels_file_sha256": calculate_file_sha256(directory/"training_labels.csv"),
        "native_model_sha256": calculate_file_sha256(directory/"return_model.txt"),
        "lightgbm_version": lightgbm.__version__, "resolved_config": cfg}
    bundle = {"model": model, "features": features, "metadata": metadata}
    joblib.dump(bundle, directory/"model_bundle.joblib")
    metadata = {**metadata, "model_bundle_sha256": calculate_file_sha256(directory/"model_bundle.joblib")}
    write_model_json(directory/"model_metadata.json", metadata)
    native = lightgbm.Booster(model_file=str(directory/"return_model.txt"))
    reloaded = np.asarray(native.predict(matrix))
    if reloaded.shape != (len(matrix),) or not np.isfinite(reloaded).all():
        raise AssertionError("原生Huber重载必须逐行且全为有限数")
    if native.params.get("objective") != "huber" or float(native.params.get("alpha", np.nan)) != .01:
        raise AssertionError("原生模型实际损失不是固定Huber alpha=0.01")
    error = float(np.max(np.abs(predicted-reloaded)))
    if error > 1e-12 or native.feature_name() != features:
        raise AssertionError("原生模型重载未精确复现训练预测")
    write_model_json(directory/"native_reload_audit.json", {"passed": True, "max_error": error,
        "training_inputs_sha256": metadata["training_inputs_file_sha256"], "fitted_rows": len(fitted)})
    return metadata


def predict_inventory_robust_model(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """所有当前唯一键返回Huber分数；校验真实损失身份，未来缺价不删推断行。"""
    profile, horizon = validate_robust_model_config(cfg)
    metadata, features = bundle["metadata"], bundle["features"]
    if (metadata["research_version"] != VERSION or metadata["profile"] != profile
            or metadata["horizon_minutes"] != horizon or metadata["features"] != features
            or metadata.get("training_loss") != "huber" or metadata.get("huber_alpha") != .01
            or metadata.get("source_l2_profile") != SOURCE_PROFILE_MAP[profile]):
        raise ValueError("推断模型身份与本轮配置不一致")
    actual = bundle["model"].get_params()
    declared = metadata["model_parameters"]
    booster = bundle["model"].booster_
    if (declared.get("objective") != "huber" or declared.get("alpha") != .01
            or any(actual.get(name) != value for name, value in declared.items())
            or booster.params.get("objective") != "huber" or float(booster.params.get("alpha", np.nan)) != .01
            or booster.feature_name() != features):
        raise ValueError("模型实际参数/原生损失与Huber元数据不一致")
    if (frame.columns.duplicated().any() or not {"datetime", "symbol", *features}.issubset(frame)
            or frame[["datetime", "symbol"]].isna().any().any()
            or frame.duplicated(["datetime", "symbol"]).any()):
        raise ValueError("推断须有唯一当前键及原顺序特征列")
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = frame.copy()
    values = np.asarray(bundle["model"].predict(matrix)) if len(matrix) else np.array([], dtype=float)
    if values.shape != (len(matrix),) or not np.isfinite(values).all():
        raise ValueError("模型输出非有限收益")
    result["prediction"] = values
    result["raw_prediction"] = result.prediction
    result["return_score"] = result.prediction
    result["horizon_minutes"], result["model_profile"] = horizon, profile
    result["model_version"], result["forecast_loss"], result["forecast_alpha"] = VERSION, "huber", .01
    return result
