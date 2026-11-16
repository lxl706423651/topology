"""永久底仓L2模型的单一覆盖率消融；保留原标签、NaN及纯核心来源隔离。

仅将训练特征最低覆盖率由0.80降至0.70，同时保存同批训练行的0.80控制输入。
不修改原模型、原缓存或原目标，不把回归输出视为校准净利润或可成交保证。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from .inventory_t_learning_models import calculate_learning_frame_sha256, attach_learning_labels
from .weekly_t_models import create_model_parameters, select_training_features, select_training_rows, write_model_json
from .weekly_t_quote_store import calculate_file_sha256
from .weekly_t_short_horizon import planned_recovery_times


VERSION = "inventory_coverage_t_v18"
PROFILES = ("l2_depth4", "l2_depth6")
HORIZONS = (20, 40, 60)
SOURCE_PROFILE_MAP = {name: name for name in PROFILES}
INPUT_KEYS = ["datetime", "symbol", "feature_source_start", "cohort_target_year", "baseline_opening_shares"]


def validate_coverage_model_config(cfg: dict) -> tuple[str, int]:
    """只接受原L2/三期限、0.70对0.80及原七日来源隔离，禁止抽样和手工改损失。"""
    profile, horizon = cfg.get("t_model_profile"), cfg.get("t_label_horizon_minutes")
    if profile not in PROFILES or isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError("须指定原L2模型及20/40/60分钟期限")
    if (cfg.get("training_loss") != "regression_l2"
            or cfg.get("forecast_target") != "fixed_20_40_60_reference_return"
            or cfg.get("feature_min_coverage") != .70 or cfg.get("source_feature_min_coverage") != .80):
        raise ValueError("本轮只改变特征覆盖率0.80→0.70，保持原L2和原参考目标")
    if tuple(cfg.get("horizons", ())) != HORIZONS or any(isinstance(v, bool) for v in cfg["horizons"]):
        raise ValueError("须保留原20/40/60期限")
    if "objective" in cfg.get("t_model", {}) or "alpha" in cfg.get("t_model", {}):
        raise ValueError("原L2模型参数不得额外声明objective或alpha")
    if cfg.get("strict_feature_provenance") is not True or "f_history_days" not in cfg.get("excluded_model_features", []):
        raise ValueError("必须隔离真实量价来源并排除跨年累计历史天数")
    if cfg.get("exclude_flat_signal_bars") is not False or cfg.get("embargo_days") != 7:
        raise ValueError("保持原全部当前机会与7个日历日来源隔离")
    if type(cfg.get("t_max_train_rows")) is not int or cfg["t_max_train_rows"] != 0:
        raise ValueError("全部有效核心行须保留，不允许训练抽样")
    return str(profile), int(horizon)


def create_coverage_model_parameters(cfg: dict) -> dict:
    """直接使用原L2参数工厂；覆盖率仅作用于特征选择，不改变回归器参数。"""
    validate_coverage_model_config(cfg)
    parameters, _ = create_model_parameters(cfg, "intraday")
    if parameters.get("objective") != "regression" or "alpha" in parameters:
        raise ValueError("实际参数必须为原L2，不能含Huber阈值")
    return parameters


def make_training_inputs(training: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """保持原五个非特征键及实际选择顺序，NaN和float32原样保存。"""
    result = training[INPUT_KEYS].copy()
    for name in matrix:
        result[name] = matrix[name].to_numpy()
    return result


def fit_inventory_coverage_model(frame: pd.DataFrame, train_years: list[int], cfg: dict,
                                 output_dir: Path | str, excluded_years: list[int]) -> dict:
    """仅在purge后的原核心行拟合L2，同时持久化0.80控制与0.70真实特征输入。"""
    profile, horizon = validate_coverage_model_config(cfg)
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("模型目录须为空，不能覆盖历史模型")
    if not {"cohort_target_year", "baseline_opening_shares"}.issubset(frame):
        raise ValueError("核心机会须包含原年份及昨日可交易底仓")
    if (not pd.to_numeric(frame.baseline_opening_shares).ge(cfg["lot_size"]).all()
            or not frame.cohort_target_year.isin(train_years).all()):
        raise ValueError("训练机会不属于核心旧股持有日")
    training, split, split_audit = select_training_rows(frame, train_years, cfg, "intraday", excluded_years)
    if not training.cohort_target_year.eq(pd.to_datetime(training.datetime).dt.year).all():
        raise ValueError("核心年份与信号时刻不一致")
    expected = planned_recovery_times(pd.to_datetime(training.datetime), horizon)
    if (not pd.to_datetime(training.label_start).eq(pd.to_datetime(training.datetime) + pd.Timedelta(minutes=1)).all()
            or not pd.to_datetime(training.label_end).eq(expected).all()
            or not training.horizon_minutes.eq(horizon).all()):
        raise ValueError("原固定期限标签端点不一致")
    prices = training[["execution_price", "recovery_price"]].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(training.target_return, errors="raise")
    if (not np.isfinite(prices).all().all() or not prices.gt(0).all().all() or not target.gt(-1).all()
            or not np.allclose(prices.recovery_price / prices.execution_price - 1, target, rtol=0, atol=1e-12)):
        raise ValueError("固定端点参考价格与原始收益标签不一致")
    targets = pd.DataFrame({"return": target})
    source_features, source_audit = select_training_features(training, targets,
        {**cfg, "feature_min_coverage": cfg["source_feature_min_coverage"]}, "intraday")
    features, feature_audit = select_training_features(training, targets, cfg, "intraday")
    matrix = training[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    source_matrix = training[source_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    parameters = create_coverage_model_parameters(cfg)
    model = LGBMRegressor(**parameters)
    model.fit(matrix, target)
    predicted = np.asarray(model.predict(matrix))
    if predicted.shape != (len(matrix),) or not np.isfinite(predicted).all():
        raise AssertionError("训练预测须逐行且全为有限数")
    directory.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(directory / "return_model.txt"))
    make_training_inputs(training, matrix).to_pickle(directory / "training_inputs.pkl")
    make_training_inputs(training, source_matrix).to_pickle(directory / "source_training_inputs.pkl")
    source_audit.to_csv(directory / "source_feature_manifest.csv", index=False, encoding="utf-8-sig")
    feature_audit.to_csv(directory / "feature_manifest.csv", index=False, encoding="utf-8-sig")
    trace = [name for name in training if name in ("datetime", "symbol", "feature_source_start", "cohort_target_year",
        "baseline_opening_shares", "intraday_feature_source_start", "entry_feature_source_start",
        "entry_rank_source_start", "portfolio_cash_source_start", "base_entry_datetime", "source_model_training_years",
        "source_model_sha256", "label_start", "label_end", "target_return", "execution_price", "recovery_price", "horizon_minutes")]
    labels = training[trace].copy()
    labels["fitted"] = True
    labels.to_csv(directory / "training_labels.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(directory / "training_split_audit.csv", index=False, encoding="utf-8-sig")
    metadata = {**split, "research_version": VERSION, "profile": profile, "horizon_minutes": horizon,
        "features": features, "source_features": source_features, "fitted_rows": len(training),
        "model_parameters": parameters, "source_l2_profile": SOURCE_PROFILE_MAP[profile],
        "training_loss": "regression_l2", "feature_min_coverage": .70, "source_feature_min_coverage": .80,
        "added_features": [name for name in features if name not in source_features],
        "removed_features": [name for name in source_features if name not in features],
        "only_selector_configuration_change": {"feature_min_coverage": {"source": .80, "current": .70}},
        "score_definition": "raw_reference_return_L2_regression_prediction",
        "score_is_calibrated_conditional_mean": False, "score_is_actual_execution_pnl": False,
        "nan_values_filled": False, "training_targets_clipped": False, "tail_training_rows_removed": False,
        "training_cohort": "permanent_inventory_core_oof_opening_old_shares",
        "target_definition": "reference_recovery_price/reference_next_minute_price-1",
        "target_is_portfolio_execution_net_pnl": False, "probability_gate_used": False,
        "full_training_rows_sha256": calculate_learning_frame_sha256(training[["datetime", "symbol", "feature_source_start"]]),
        "training_matrix_sha256": calculate_learning_frame_sha256(matrix),
        "source_training_matrix_sha256": calculate_learning_frame_sha256(source_matrix),
        "training_target_sha256": calculate_learning_frame_sha256(training[["target_return"]]),
        "training_inputs_file_sha256": calculate_file_sha256(directory / "training_inputs.pkl"),
        "source_training_inputs_file_sha256": calculate_file_sha256(directory / "source_training_inputs.pkl"),
        "source_feature_manifest_file_sha256": calculate_file_sha256(directory / "source_feature_manifest.csv"),
        "training_labels_file_sha256": calculate_file_sha256(directory / "training_labels.csv"),
        "native_model_sha256": calculate_file_sha256(directory / "return_model.txt"),
        "lightgbm_version": lightgbm.__version__, "resolved_config": cfg}
    joblib.dump({"model": model, "features": features, "metadata": metadata}, directory / "model_bundle.joblib")
    metadata = {**metadata, "model_bundle_sha256": calculate_file_sha256(directory / "model_bundle.joblib")}
    write_model_json(directory / "model_metadata.json", metadata)
    native = lightgbm.Booster(model_file=str(directory / "return_model.txt"))
    restored = np.asarray(native.predict(matrix))
    if restored.shape != (len(matrix),) or not np.isfinite(restored).all():
        raise AssertionError("原生L2重载须逐行且全为有限数")
    if native.params.get("objective") != "regression" or native.feature_name() != features:
        raise AssertionError("原生模型实际L2损失或特征身份不一致")
    error = float(np.max(np.abs(predicted - restored)))
    if error > 1e-12:
        raise AssertionError("原生模型未精确复现实际预测")
    write_model_json(directory / "native_reload_audit.json", {"passed": True, "max_error": error,
        "training_inputs_sha256": metadata["training_inputs_file_sha256"], "fitted_rows": len(training)})
    return metadata


def predict_inventory_coverage_model(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按实际v18/L2/0.70身份返回全部当前唯一键；未来缺价不删行，保留NaN特征。"""
    profile, horizon = validate_coverage_model_config(cfg)
    metadata, features, model = bundle["metadata"], bundle["features"], bundle["model"]
    if (metadata.get("research_version") != VERSION or metadata.get("profile") != profile
            or metadata.get("horizon_minutes") != horizon or metadata.get("features") != features
            or metadata.get("training_loss") != "regression_l2" or metadata.get("feature_min_coverage") != .70
            or metadata.get("source_feature_min_coverage") != .80 or metadata.get("source_l2_profile") != profile):
        raise ValueError("模型身份须与v18/L2/0.70本轮配置一致")
    actual, declared, booster = model.get_params(), metadata["model_parameters"], model.booster_
    if (declared != create_coverage_model_parameters(cfg) or declared.get("objective") != "regression"
            or any(actual.get(name) != value for name, value in declared.items())
            or booster.params.get("objective") != "regression" or booster.feature_name() != features):
        raise ValueError("模型实际参数或原生特征与声明不一致")
    if (frame.columns.duplicated().any() or not {"datetime", "symbol", *features}.issubset(frame)
            or frame[["datetime", "symbol"]].isna().any().any()
            or frame.duplicated(["datetime", "symbol"]).any()):
        raise ValueError("推断须有唯一当前键和全部模型特征")
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = np.asarray(model.predict(matrix)) if len(matrix) else np.array([], dtype=float)
    if values.shape != (len(matrix),) or not np.isfinite(values).all():
        raise ValueError("模型输出须逐行且为有限收益")
    result = frame.drop(columns=["forecast_alpha"], errors="ignore").copy()
    result["prediction"], result["raw_prediction"], result["return_score"] = values, values, values
    result["horizon_minutes"], result["model_profile"] = horizon, profile
    result["model_version"], result["forecast_loss"], result["feature_min_coverage"] = VERSION, "regression_l2", .70
    return result
