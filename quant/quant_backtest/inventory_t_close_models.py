"""永久底仓队列的同日14:55参考收益回归；只在纯核心年份拟合并保存可重算输入。

标签是下一分钟到固定期限的原始参考价格收益，不是共享现金账户的实际T利润。
当前特征、标签及来源由调用方分别提供；内部/外层仅使用保存的原生模型推断。
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
from .inventory_t_close_labels import (IDENTITY, validate_fixed_close_config,
    calculate_fixed_close_times, validate_reference_times)


VERSION = "inventory_close_t_v16"
PROFILES = ("l2_depth4", "l2_depth6")
def calculate_learning_frame_sha256(frame: pd.DataFrame) -> str:
    """固定实际列序和行序的内容摘要，供训练输入与原生重载审计。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def validate_close_model_config(cfg: dict) -> str:
    """固定两种原回归器，只改变参考目标，不混入固定长度期限或旧概率门控。"""
    validate_fixed_close_config(cfg)
    profile = cfg.get("t_model_profile")
    if (profile not in PROFILES or cfg.get("forecast_target") != "same_day_1455"
            or cfg.get("recovery_key") != "close" or "t_label_horizon_minutes" in cfg
            or "horizons" in cfg):
        raise ValueError("本轮只支持独立同日14:55目标，不使用整数期限")
    if cfg.get("strict_feature_provenance") is not True or "f_history_days" not in cfg.get("excluded_model_features", []):
        raise ValueError("必须严格隔离来源并排除累计历史天数特征")
    if cfg.get("exclude_flat_signal_bars") is not False or cfg.get("embargo_days") != 7:
        raise ValueError("保留原全部当前信号及七日来源隔离")
    if type(cfg.get("t_max_train_rows")) is not int or cfg["t_max_train_rows"] != 0:
        raise ValueError("本轮使用全部有效核心行，不追加抽样")
    return str(profile)


def fit_inventory_close_model(frame: pd.DataFrame, train_years: list[int], cfg: dict,
                               output_dir: Path | str, excluded_years: list[int]) -> dict:
    """在入场前已隔离的永久底仓机会中再次按特征/标签边界purge后拟合。

选择特征、可选确定性抽样、全部模型参数均只消费核心行。未来缺失标签只影响
训练有效性；不会影响内部/外层预测行。写新目录，拒绝覆盖已有模型。
    """
    profile = validate_close_model_config(cfg)
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
    first, end = calculate_fixed_close_times(pd.to_datetime(training.datetime))
    validate_reference_times(training, first, end, require_all=True)
    if not all(name in training and training[name].eq(value).all() for name, value in IDENTITY.items()):
        raise ValueError("核心标签必须具有独立固定尾盘身份")
    if not training.label_available.eq(True).all():
        raise ValueError("实际拟合行必须有两个有效参考端点")
    prices = training[["execution_price", "recovery_price"]].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(training.target_return, errors="raise")
    if (not np.isfinite(prices).all().all() or not prices.gt(0).all().all()
            or not target.gt(-1).all()
            or not np.allclose(prices.recovery_price/prices.execution_price-1, target, atol=1e-12, rtol=0)):
        raise ValueError("固定端点参考价格和收益标签不一致")
    features, feature_audit = select_training_features(training, pd.DataFrame({"return": target}), cfg, "intraday")
    parameters, _ = create_model_parameters(cfg, "intraday")
    cap = cfg.get("t_max_train_rows", 0)
    fitted = training.sample(n=cap, random_state=cfg["random_state"]).sort_index() if cap and len(training)>cap else training
    matrix = fitted[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    model = LGBMRegressor(**parameters)
    model.fit(matrix, pd.to_numeric(fitted.target_return))
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
        "source_model_sha256", "label_start", "label_end", "target_return", "execution_price", "recovery_price", "recovery_key", "recovery_kind", "recovery_clock", "label_available")]
    labels = training[trace].copy()
    labels["fitted"] = labels.index.isin(fitted.index)
    labels.to_csv(directory/"training_labels.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(directory/"training_split_audit.csv", index=False, encoding="utf-8-sig")
    feature_audit.to_csv(directory/"feature_manifest.csv", index=False, encoding="utf-8-sig")
    metadata = {**split, "research_version": VERSION, "profile": profile, **IDENTITY,
        "features": features, "fitted_rows": len(fitted), "model_parameters": parameters,
        "training_cohort": "permanent_inventory_core_oof_opening_old_shares",
        "target_definition": "same_day_14_55_reference_price/next_trading_minute_reference_price-1",
        "fixed_integer_horizon_used": False,
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
    predicted, reloaded = np.asarray(model.predict(matrix)), np.asarray(native.predict(matrix))
    if (predicted.shape != (len(matrix),) or reloaded.shape != (len(matrix),)
            or not np.isfinite(predicted).all() or not np.isfinite(reloaded).all()):
        raise AssertionError("训练预测和原生重载输出必须逐行且全为有限数")
    error = float(np.max(np.abs(predicted-reloaded)))
    if error > 1e-12 or native.feature_name() != features:
        raise AssertionError("原生模型重载未精确复现训练预测")
    write_model_json(directory/"native_reload_audit.json", {"passed": True, "max_error": error,
        "training_inputs_sha256": metadata["training_inputs_file_sha256"], "fitted_rows": len(fitted)})
    return metadata


def predict_inventory_close_model(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """对每个当前唯一键推断，保留未来缺报价的行，不在推断期间重拟合或裁掉支持域。"""
    profile = validate_close_model_config(cfg)
    metadata, features = bundle["metadata"], bundle["features"]
    if (metadata["research_version"] != VERSION or metadata["profile"] != profile
            or any(metadata.get(name) != value for name, value in IDENTITY.items()) or metadata["features"] != features):
        raise ValueError("推断模型身份与本轮配置不一致")
    if (frame.columns.duplicated().any() or not {"datetime", "symbol", *features}.issubset(frame)
            or frame[["datetime", "symbol"]].isna().any().any()
            or frame.duplicated(["datetime", "symbol"]).any()):
        raise ValueError("推断须有唯一当前键及原顺序特征列")
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = frame.copy()
    result["prediction"] = bundle["model"].predict(matrix) if len(matrix) else pd.Series(dtype=float)
    if not np.isfinite(result.prediction).all():
        raise ValueError("模型输出非有限收益")
    result["raw_prediction"] = result.prediction
    result["model_profile"] = profile
    for name, value in IDENTITY.items():
        result[name] = value
    return result
