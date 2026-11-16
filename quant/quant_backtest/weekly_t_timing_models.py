"""v11 库存做 T 的原始收益回归与扣费盈利三分类模型。

调用方提供因果 f_*、原始固定周期标签和真实特征来源；仅核心训练行经过严格
年度隔离后参与目标生成、特征选择及支持域拟合。写独立模型目录，不改旧缓存。
三类概率描述固定端点标称净收益事件，不代表真实成交成功率或校准后的盈利概率。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable

import joblib
import lightgbm
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd

from .weekly_t_models import (
    create_model_parameters, select_causal_inference_rows, select_training_features,
    select_training_rows, write_model_json,
)
from .weekly_t_short_horizon import fit_feature_support, planned_recovery_times, predicted_pair_edges


VERSION = "weekly_t_v11_cost_aware_timing"
TARGET_MODE = "raw_return_and_cost_profit_class"
CLASS_MAPPING = {"0": "neither", "1": "buy_net_positive", "2": "sell_net_positive"}
COST_CONFIG_KEYS = ("commission_rate", "transfer_rate", "slippage_rate", "stamp_rate_before_20230828",
                    "stamp_rate_from_20230828", "min_commission")
TARGET_COLUMNS = ["datetime", "symbol", "feature_source_start", "label_start", "label_end",
                  "raw_target_return", "buy_net_return", "sell_net_return", "profit_class", "fitted"]


def calculate_timing_frame_sha256(frame: pd.DataFrame) -> str:
    """固定列序和行序的内容哈希；不将临时 pandas 索引纳入身份。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def validate_timing_config(cfg: dict) -> dict:
    """验证原始收益双头协议、零最低佣金及不重加权的三分类配置。"""
    if cfg.get("strict_feature_provenance") is not True or "f_history_days" not in cfg.get("excluded_model_features", []):
        raise ValueError("择时模型必须严格来源隔离并排除 f_history_days")
    if cfg.get("t_target_mode", TARGET_MODE) != TARGET_MODE:
        raise ValueError(f"t_target_mode 必须为 {TARGET_MODE}")
    profile = cfg.get("t_model_profile", "l2_depth4")
    if profile not in ("l2_depth4", "l2_depth6"):
        raise ValueError("择时模型只支持 l2_depth4/l2_depth6 原始收益回归")
    maximum = cfg.get("t_max_features", 60)
    cap = cfg.get("t_max_train_rows", cfg.get("max_train_rows", 0))
    for name, value, lower, upper in (("t_max_features", maximum, 1, 60), ("t_max_train_rows", cap, 0, None)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < lower or (upper is not None and value > upper):
            raise ValueError(f"{name} 超出支持的整数范围")
    for name in COST_CONFIG_KEYS:
        value = cfg.get(name, 0. if name == "min_commission" else None)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} 必须是有限非负费率")
    if cfg.get("min_commission", 0.) != 0:
        raise ValueError("择时目标的每元名义额公式要求 min_commission=0")
    if cfg["commission_rate"] + cfg["transfer_rate"] + max(cfg["stamp_rate_before_20230828"], cfg["stamp_rate_from_20230828"]) >= 1:
        raise ValueError("累计费率必须小于1")
    parameters = cfg.get("t_model", {})
    if (parameters.get("class_weight") is not None or parameters.get("is_unbalance", False)
            or parameters.get("unbalance", False) or parameters.get("is_unbalanced", False)
            or parameters.get("scale_pos_weight", 1.) != 1.
            or parameters.get("pos_bagging_fraction", 1.) != 1.
            or parameters.get("neg_bagging_fraction", 1.) != 1.):
        raise ValueError("三分类禁止类重加权或正负样本重采样")
    horizons = [cfg[key] for key in ("t_label_horizon_minutes", "horizon_minutes", "horizon") if key in cfg]
    for value in horizons:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in (20, 40, 60):
            raise ValueError("固定标签周期只能为20/40/60交易分钟")
    if len(set(horizons)) > 1:
        raise ValueError("配置中的固定周期相互冲突")
    return {"profile": profile, "max_features": int(maximum), "training_row_cap": int(cap),
            "horizon_minutes": int(horizons[0]) if horizons else None}


def _resolve_horizon(cfg: dict, metadata: dict | None = None) -> int:
    """只从配置或已训练模型元数据确定周期；推断不读取未来标签端点。"""
    requested = validate_timing_config(cfg)["horizon_minutes"]
    stored = None if metadata is None else metadata.get("horizon_minutes")
    if stored is not None and (isinstance(stored, bool) or stored not in (20, 40, 60)):
        raise ValueError("模型周期元数据非法")
    if requested is not None and stored is not None and requested != stored:
        raise ValueError("推断配置与模型标签周期不一致")
    horizon = requested if requested is not None else stored
    if horizon is None:
        raise ValueError("训练配置必须指定 t_label_horizon_minutes 或 horizon_minutes/horizon")
    return int(horizon)


def _class_counts(values: pd.Series) -> dict:
    """按固定0/1/2顺序保留各类数量，包括缺失类别的零计数。"""
    return {str(key): int(values.eq(key).sum()) for key in (0, 1, 2)}


def fit_timing_model(frame: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                     output_dir: Path | str, excluded_years: Iterable[int] = ()) -> dict:
    """仅严格核心训练行拟合原始收益与扣费事件两个头，并保存完整标签和输入审计。

完整purged行决定特征、类别与支持域；两模型使用同一份确定性cap采样行。无类
平衡、过采样或概率校准，不减未来同刻均值。固定周期标签只供训练使用。
"""
    policy = validate_timing_config(cfg)
    horizon = _resolve_horizon(cfg)
    directory = Path(output_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("择时输出目录必须为空，不能覆盖已有模型")
    training, metadata, split_audit = select_training_rows(frame, train_years, cfg, "intraday", excluded_years)
    training["target_return"] = pd.to_numeric(training.target_return, errors="raise")
    start = "label_start" if "label_start" in training else "execution_time"
    end = "label_end" if "label_end" in training else "recovery_time"
    expected_end = planned_recovery_times(training.datetime, horizon)
    if (not pd.to_datetime(training[start]).eq(pd.to_datetime(training.datetime)+pd.Timedelta(minutes=1)).all()
            or not pd.to_datetime(training[end]).eq(expected_end).all()
            or ("horizon_minutes" in training and not training.horizon_minutes.eq(horizon).all())):
        raise ValueError("严格训练行的标签端点与固定周期不一致")
    if not training.target_return.gt(-1).all():
        raise ValueError("原始股票收益必须大于-1")
    if {"execution_price", "recovery_price"}.issubset(training):
        prices = training[["execution_price", "recovery_price"]].apply(pd.to_numeric, errors="coerce")
        if (not np.isfinite(prices).all().all() or not prices.gt(0).all().all()
                or not np.allclose(prices.recovery_price/prices.execution_price-1,
                                   training.target_return, atol=1e-12, rtol=0)):
            raise ValueError("原始收益标签与正数端点价格不一致")
    buy, sell = predicted_pair_edges(training.assign(prediction=training.target_return), cfg)
    if not np.isfinite(buy).all() or not np.isfinite(sell).all() or (buy.gt(0) & sell.gt(0)).any():
        raise ValueError("扣费双向标签非法或同时盈利")
    labels = training[["datetime", "symbol", "feature_source_start"]].copy()
    labels["label_start"], labels["label_end"] = pd.to_datetime(training[start]), pd.to_datetime(training[end])
    labels["raw_target_return"], labels["buy_net_return"], labels["sell_net_return"] = training.target_return, buy, sell
    labels["profit_class"] = np.where(buy.gt(0), 1, np.where(sell.gt(0), 2, 0)).astype(np.int64)
    if set(labels.profit_class.unique()) != {0, 1, 2}:
        raise ValueError("严格核心训练行必须包含全部三类0/1/2")
    targets = pd.DataFrame({"return": training.target_return,
                            "buy_profit": labels.profit_class.eq(1).astype(int),
                            "sell_profit": labels.profit_class.eq(2).astype(int)})
    features, feature_audit = select_training_features(training, targets, cfg, "intraday")
    parameters, profile = create_model_parameters(cfg, "intraday")
    cap = policy["training_row_cap"]
    fitted = training.sample(n=cap, random_state=int(parameters["random_state"])).sort_index() if cap and len(training) > cap else training
    fitted_labels = labels.loc[fitted.index]
    if set(fitted_labels.profit_class.unique()) != {0, 1, 2}:
        raise ValueError("实际cap采样训练行缺少三类之一；不通过过采样补类")
    labels["fitted"] = labels.index.isin(fitted.index)
    labels = labels[TARGET_COLUMNS]
    x = fitted[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    classifier_parameters = {**parameters, "objective": "multiclass", "num_class": 3}
    print(f"intraday_timing/{profile}/h{horizon}：核心{metadata['train_years']}，"
          f"{len(training):,}行/{training.datetime.nunique():,}时刻，拟合{len(fitted):,}行，{len(features)}特征", flush=True)
    regressor = LGBMRegressor(**parameters).fit(x, fitted.target_return)
    classifier = LGBMClassifier(**classifier_parameters).fit(x, fitted_labels.profit_class)
    if not np.array_equal(classifier.classes_, np.array([0, 1, 2])):
        raise ValueError("分类器的类别顺序不是固定0/1/2")
    support = fit_feature_support(training, cfg)
    fitted_targets = labels.loc[fitted.index, ["raw_target_return", "buy_net_return", "sell_net_return", "profit_class"]]
    keys = fitted[["symbol", "datetime"]]
    metadata.update({
        "research_version": VERSION, "t_target_mode": TARGET_MODE, "horizon_minutes": horizon,
        "fitted_rows": len(fitted), "training_clock_count": int(training.datetime.nunique()),
        "training_day_count": int(pd.to_datetime(training.datetime).dt.normalize().nunique()),
        "training_week_count": int(pd.to_datetime(training.datetime).dt.to_period("W").nunique()),
        "training_symbol_count": int(training.symbol.nunique()), "class_mapping": CLASS_MAPPING.copy(),
        "training_class_counts": _class_counts(labels.profit_class),
        "fitted_class_counts": _class_counts(fitted_labels.profit_class),
        "training_target_columns": TARGET_COLUMNS.copy(), "feature_names": features, "profile": profile,
        "model_parameters": parameters, "classifier_parameters": classifier_parameters,
        "lightgbm_version": lightgbm.__version__, "training_row_cap": cap,
        "training_target_sha256": calculate_timing_frame_sha256(labels.drop(columns="fitted")),
        "fitted_target_sha256": calculate_timing_frame_sha256(pd.concat([keys, fitted_targets], axis=1)),
        "training_input_sha256": calculate_timing_frame_sha256(pd.concat([keys, x], axis=1)),
        "training_sample_sha256": calculate_timing_frame_sha256(pd.concat([keys, x, fitted_targets], axis=1)),
        "feature_source_sha256": calculate_timing_frame_sha256(fitted[["symbol", "datetime", "feature_source_start"]]),
        "full_feature_source_sha256": calculate_timing_frame_sha256(training[["symbol", "datetime", "feature_source_start"]]),
        "support_training_rows": len(training), "support_training_inputs_sha256": support["training_inputs_sha256"],
        "raw_target_unchanged": True, "cohort_mean_subtracted": False,
        "target_cost_config": {key: float(cfg.get(key, 0.)) for key in COST_CONFIG_KEYS},
        "class_reweighting": False, "oversampling": False, "probability_calibration": False,
        "sample_weight_rule": "每行等权；同一时钟股票并非独立统计样本",
        "label_definition": "raw=recovery_price/execution_price-1；0两方向扣费非正，1买先净额>0，2卖先净额>0；无量纲名义额比率",
        "input_contract": "只用严格核心选择的当前因果f_*；标签、未来成交/恢复价格及类别不是模型输入",
        "prediction_scale": "原始股票收益，不去同刻均值；三类输出是未经校准的固定端点标称扣费事件概率",
        "inference_contract": "只用当前因果推断资格，不读取或过滤未来标签/价格；保持原predict_t的稳定时刻/股票排序",
        "economic_limitation": "三类不代表真实成交成功率；容量、现金、整手和恢复失败由真实到账分钟账户验证",
        "resolved_config": deepcopy(cfg),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("weekly_t_timing_models.py", "weekly_t_models.py", "weekly_t_short_horizon.py")},
    })
    directory.mkdir(parents=True, exist_ok=True)
    labels.to_csv(directory/"training_targets.csv", index=False, encoding="utf-8-sig")
    write_model_json(directory/"support.json", support)
    metadata["training_targets_file_sha256"] = hashlib.sha256((directory/"training_targets.csv").read_bytes()).hexdigest()
    metadata["support_file_sha256"] = hashlib.sha256((directory/"support.json").read_bytes()).hexdigest()
    native_audit = {"sample_rows": min(128, len(x)), "class_order": [0, 1, 2]}
    for name, model in (("return", regressor), ("profit_class", classifier)):
        path = directory/f"{name}_model.txt"
        model.booster_.save_model(str(path))
        native = lightgbm.Booster(model_file=str(path))
        sample = x.iloc[:128]
        expected = model.predict(sample) if name == "return" else model.predict_proba(sample)
        actual = native.predict(sample)
        passed = native.feature_name() == features and np.array_equal(actual, expected)
        native_audit[name] = {"passed": bool(passed), "max_abs_difference": float(np.max(abs(actual-expected))),
                              "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if not passed:
            raise AssertionError(f"{name} 原生重载与训练对象预测不一致")
    metadata["native_reload_audit"] = native_audit
    bundle = {"task": "intraday", "target_mode": TARGET_MODE,
              "models": {"return": regressor, "profit_class": classifier},
              "features": features, "metadata": metadata, "support": support}
    joblib.dump(bundle, directory/"model_bundle.joblib")
    write_model_json(directory/"model_metadata.json", metadata)
    write_model_json(directory/"resolved_config.json", cfg)
    write_model_json(directory/"native_reload_audit.json", native_audit)
    split_audit.to_csv(directory/"training_split_audit.csv", index=False)
    feature_audit.to_csv(directory/"feature_selection_audit.csv", index=False)
    manifest = pd.DataFrame({"feature": features, "selection_order": range(1, len(features)+1)})
    manifest.merge(feature_audit, on="feature", validate="one_to_one").to_csv(directory/"used_feature_manifest.csv", index=False)
    (directory/"feature_list.txt").write_text("\n".join(features)+"\n")
    return bundle


def predict_timing_model(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """返回当前合格行、原始收益与固定0/1/2顺序概率；未来列缺失不改变输出选择。

复用原模型的当前资格/缺失值/稳定排序合同，不调用分类器predict把类别误作概率。
保留原字段供独立执行器使用，执行价、恢复价、标签和未来分组信息绝不进入x。
"""
    metadata = bundle.get("metadata", {})
    if (bundle.get("task") != "intraday" or bundle.get("target_mode") != TARGET_MODE
            or metadata.get("t_target_mode") != TARGET_MODE
            or set(bundle.get("models", {})) != {"return", "profit_class"}
            or metadata.get("class_mapping") != CLASS_MAPPING):
        raise ValueError("必须使用原始收益与固定三分类的择时bundle")
    horizon = _resolve_horizon(cfg, metadata)
    costs = {key: float(cfg.get(key, 0.)) for key in COST_CONFIG_KEYS}
    if metadata.get("target_cost_config") != costs:
        raise ValueError("推断成本配置与分类目标的训练成本定义不一致")
    if "t_model_profile" in cfg and cfg["t_model_profile"] != metadata.get("profile"):
        raise ValueError("推断profile与模型身份不一致")
    features = bundle["features"]
    if (features != metadata.get("feature_names") or any(not name.startswith("f_") or name == "f_history_days" for name in features)
            or len(features) != len(set(features)) or not features):
        raise ValueError("择时模型因果特征列表非法")
    missing = set(features).difference(frame)
    if missing:
        raise ValueError(f"推断缺少训练特征：{sorted(missing)}")
    classifier = bundle["models"]["profit_class"]
    if not np.array_equal(classifier.classes_, np.array([0, 1, 2])):
        raise ValueError("分类器类别顺序必须为0/1/2")
    source = select_causal_inference_rows(frame, cfg, "intraday")
    x = source[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw = np.asarray(bundle["models"]["return"].predict(x), dtype=float) if len(x) else np.empty(0)
    probabilities = np.asarray(classifier.predict_proba(x), dtype=float) if len(x) else np.empty((0, 3))
    if (raw.shape != (len(source),) or probabilities.shape != (len(source), 3)
            or not np.isfinite(raw).all() or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError("择时原始收益/三类概率的形状、有限性或概率和非法")
    result = source.copy()
    result["prediction"], result["raw_prediction"] = raw, raw
    result["prob_neither"], result["prob_buy_profit"], result["prob_sell_profit"] = probabilities.T
    result["horizon_minutes"], result["model_profile"] = horizon, metadata["profile"]
    result.attrs["metadata"] = {"target_mode": TARGET_MODE, "horizon_minutes": horizon,
                               "prediction_scale": metadata["prediction_scale"], "probability_calibration": False,
                               "class_mapping": CLASS_MAPPING.copy(), "future_fields_consumed": False}
    return result.sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
