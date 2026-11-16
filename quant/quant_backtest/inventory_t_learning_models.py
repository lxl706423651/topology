"""永久底仓队列的固定端点收益回归；只在纯核心年份拟合并保存可重算输入。

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
from .weekly_t_short_horizon import planned_recovery_times, attach_horizon_labels


VERSION = "inventory_learning_t_v14"
PROFILES = ("l2_depth4", "l2_depth6")
HORIZONS = (20, 40, 60)


def attach_learning_labels(cohort: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """核心机会冻结后才左联完整固定端点；缺未来价保留行供训练purge审计。

cohort构造阶段已删所有execution字段；这里显式取已审计标签的下一分钟参考价，
避免把当前信号价或原14:55恢复价误当训练端点。此函数不能用于推断资格。
    """
    if isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError("仅支持预声明20/40/60分钟训练标签")
    if any(name.startswith(("execution_", "recovery_", "target_", "label_")) for name in cohort):
        raise ValueError("连接标签之前的核心机会须已去除全部未来字段")
    result = attach_horizon_labels(cohort, labels, horizon)
    selected = labels.loc[labels.horizon_minutes.eq(horizon), ["datetime", "symbol", "execution_time", "execution_price"]]
    result = result.merge(selected, on=["datetime", "symbol"], how="left", validate="one_to_one", sort=False)
    if len(result) != len(cohort):
        raise AssertionError("标签左联改变了核心机会行数")
    return result


def calculate_learning_frame_sha256(frame: pd.DataFrame) -> str:
    """固定实际列序和行序的内容摘要，供训练输入与原生重载审计。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def validate_learning_model_config(cfg: dict) -> tuple[str, int]:
    """约束本轮固定两种回归器/三期限及完整来源隔离，不继承旧概率门控。"""
    profile, horizon = cfg.get("t_model_profile"), cfg.get("t_label_horizon_minutes")
    if profile not in PROFILES or isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError("训练必须指定预声明的回归器和20/40/60分钟期限")
    if cfg.get("strict_feature_provenance") is not True or "f_history_days" not in cfg.get("excluded_model_features", []):
        raise ValueError("必须严格隔离实际来源并排除累计历史天数特征")
    if cfg.get("exclude_flat_signal_bars") is not False:
        raise ValueError("本轮保持所有当前信号；可成交性由实际执行器判断")
    if cfg.get("embargo_days") != 7:
        raise ValueError("训练保持预声明7个日历日边界隔离")
    cap = cfg.get("t_max_train_rows", 0)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("t_max_train_rows须为非负整数，0代表不抽样")
    return str(profile), int(horizon)


def fit_inventory_return_model(frame: pd.DataFrame, train_years: list[int], cfg: dict,
                               output_dir: Path | str, excluded_years: list[int]) -> dict:
    """在入场前已隔离的永久底仓机会中再次按特征/标签边界purge后拟合。

选择特征、可选确定性抽样、全部模型参数均只消费核心行。未来缺失标签只影响
训练有效性；不会影响内部/外层预测行。写新目录，拒绝覆盖已有模型。
    """
    profile, horizon = validate_learning_model_config(cfg)
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
        "source_model_sha256", "label_start", "label_end", "target_return", "execution_price", "recovery_price", "horizon_minutes")]
    labels = training[trace].copy()
    labels["fitted"] = labels.index.isin(fitted.index)
    labels.to_csv(directory/"training_labels.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(directory/"training_split_audit.csv", index=False, encoding="utf-8-sig")
    feature_audit.to_csv(directory/"feature_manifest.csv", index=False, encoding="utf-8-sig")
    metadata = {**split, "research_version": VERSION, "profile": profile, "horizon_minutes": horizon,
        "features": features, "fitted_rows": len(fitted), "model_parameters": parameters,
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
    error = float(np.max(np.abs(model.predict(matrix)-native.predict(matrix))))
    if error > 1e-12 or native.feature_name() != features:
        raise AssertionError("原生模型重载未精确复现训练预测")
    write_model_json(directory/"native_reload_audit.json", {"passed": True, "max_error": error,
        "training_inputs_sha256": metadata["training_inputs_file_sha256"], "fitted_rows": len(fitted)})
    return metadata


def predict_inventory_return_model(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """对每个当前唯一键推断，保留未来缺报价的行，不在推断期间重拟合或裁掉支持域。"""
    profile, horizon = validate_learning_model_config(cfg)
    metadata, features = bundle["metadata"], bundle["features"]
    if (metadata["research_version"] != VERSION or metadata["profile"] != profile
            or metadata["horizon_minutes"] != horizon or metadata["features"] != features):
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
    result["horizon_minutes"], result["model_profile"] = horizon, profile
    return result
