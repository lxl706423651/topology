"""交易路径净收益的LightGBM回归/盈利概率模型，仅消费固定29个当时可见输入。

标签由独立批次执行器构造；账户净收益不是单股裸收益。概率模型用核心训练的
正/非正收益均值转为期望账户收益，未成交意图包含在非正类，不能把概率当成交胜率。
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier

from quant_backtest.pathwise_labels import PATH_FEATURES
from quant_backtest.learned_exit_models import select_window_rows
from quant_backtest.simple_signals import calculate_day_weights, select_spaced_training

METHOD_NAMES = {"method1": "方法一新研究：执行路径净收益回归",
                "method2": "方法二新研究：盈利概率与收益幅度模型"}


def create_path_profiles(cfg: dict) -> list[dict]:
    """按cfg.path_horizons/path_model_profiles生成各期限、复杂度和两种监督目标的固定网格。"""
    return [{"name": f"path_h{h}_{size['name']}_{algorithm}", "method": method,
             "algorithm": algorithm, "horizon": h, "parameters": {k: v for k, v in size.items() if k != "name"}}
            for h in cfg["path_horizons"] for size in cfg["path_model_profiles"]
            for method, algorithm in [("method1", "regression"), ("method2", "classification")]]


def select_path_rows(frame: pd.DataFrame, windows: list[dict], horizon: int, cfg: dict,
                     times=None, spaced: bool = False) -> pd.DataFrame:
    """从frame选horizon路径标签完整落在windows的行；cfg给宽限期，spaced时用times去掉同股潜在重叠窗口。"""
    prefix = f"path_h{horizon}_"
    out = frame.assign(label_end=frame[prefix + "window_end"], path_target=frame[prefix + "net_return"])
    out = select_window_rows(out, windows)
    out = out.loc[np.isfinite(out.path_target)].copy()
    if spaced:
        if times is None:
            raise ValueError("训练抽样缺少实际交易日历")
        out = select_spaced_training(out, times, horizon + cfg["path_exit_grace_days"])
    return out


def fit_path_model(frame: pd.DataFrame, profile: dict, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """只用frame核心样本及profile超参拟合；cfg给种子/线程/截断，输出冻结模型及特征重要性中文表。"""
    if len(frame) < cfg["simple_min_training_rows"]:
        raise ValueError("路径模型核心样本不足")
    features = list(PATH_FEATURES)
    cap = cfg["label_clip"] * cfg["entry_equity_fraction"]
    target = np.clip(frame.path_target.to_numpy(), -cap, cap)
    weights = calculate_day_weights(frame)
    positive = target > 0
    parameters = {"learning_rate": .03, "reg_alpha": 1., "subsample": 1., "colsample_bytree": 1.,
        "n_jobs": cfg["workers"], "random_state": cfg["model"]["random_state"],
        "deterministic": True, "force_col_wise": True, "verbosity": -1, **profile["parameters"]}
    x = frame[features].replace([np.inf, -np.inf], np.nan)
    win_mean = float(np.average(target[positive], weights=weights[positive])) if positive.any() else 0.
    loss_mean = float(np.average(target[~positive], weights=weights[~positive])) if (~positive).any() else 0.
    if profile["algorithm"] == "regression":
        model = LGBMRegressor(objective="regression", **parameters)
        model.fit(x, target * 10000, sample_weight=weights)
    else:
        if len(np.unique(positive)) < 2:
            raise ValueError("盈利概率模型需要核心正/非正两类样本")
        model = LGBMClassifier(objective="binary", **parameters)
        model.fit(x, positive.astype(int), sample_weight=weights)
    bundle = {"model": model, "profile": profile, "features": features, "training_rows": len(frame),
              "positive_mean": win_mean, "nonpositive_mean": loss_mean,
              "target_unit": "相对独立初始账户的净权益收益率，回归训练时乘10000",
              "label_clip_account_return": cap}
    details = pd.DataFrame({"特征代码": features, "中文含义": [PATH_FEATURES[c] for c in features],
        "训练非缺失比例": x.notna().mean().to_numpy(), "分裂增益": model.booster_.feature_importance("gain"),
        "分裂次数": model.booster_.feature_importance("split")})
    return bundle, details


def generate_path_scores(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """只用bundle冻结模型及frame29特征预测；cfg保留统一接口，预测资格绝不受标签/未来是否成交影响。"""
    if not len(frame):
        return pd.DataFrame(columns=["datetime", "symbol", "prediction", "eligible"])
    x = frame[bundle["features"]].replace([np.inf, -np.inf], np.nan)
    if bundle["profile"]["algorithm"] == "regression":
        scores = bundle["model"].predict(x) / 10000
    else:
        probability = bundle["model"].predict_proba(x)[:, 1]
        scores = probability * bundle["positive_mean"] + (1 - probability) * bundle["nonpositive_mean"]
    out = frame[["datetime", "symbol"]].copy()
    out["prediction"] = np.where(np.isfinite(scores), scores, -1e9)
    out["eligible"] = np.isfinite(scores)
    return out


def create_path_exits(profile: dict, cfg: dict) -> list[dict]:
    """按profile.horizon返回唯一与训练路径一致的ATR/到期退出；cfg的宽限窗口不是真实强制卖出日期。"""
    return [{"name": f"path_h{profile['horizon']}", "exit_mode": "matched_horizon",
             "initial": cfg["path_initial_atr"], "trailing": cfg["path_trailing_atr"], "max_days": 0}]


def calculate_path_diagnostics(bundle: dict, frame: pd.DataFrame, windows: list, cfg: dict) -> list[dict]:
    """对frame内部windows完整路径标签分箱比较预测和实际账户收益；不查看外层或据此改规则。"""
    selected = select_path_rows(frame, windows, bundle["profile"]["horizon"], cfg)
    if not len(selected):
        return []
    scores = generate_path_scores(bundle, selected, cfg)
    bins = [-np.inf, 0, .0002, .0005, .001, .002, np.inf]
    grouped = pd.cut(scores.prediction.to_numpy(), bins)
    values = pd.DataFrame({"group": grouped, "predicted": scores.prediction.to_numpy(),
                           "realized_path": selected.path_target.to_numpy()})
    out = values.groupby("group", observed=True).agg(rows=("predicted", "size"),
        predicted=("predicted", "mean"), realized_account_label=("realized_path", "mean")).reset_index()
    out["group"] = out["group"].astype(str)
    return out.to_dict("records")

