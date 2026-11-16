"""完整核心路径样本的加权净收益模型，只使用已有29个当时特征与训练标签。

核心内部允许同股潜在窗口重叠，以平均唯一性降权并令每日总权重相同；
跨拟合、校准、验证与留出年的边界仍完整隔离。分解模型把同刻共同净收益
与个股相对收益分开拟合，推断相加后仍是账户净收益，不是多空收益。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from quant_backtest.multiscale_training import MACRO_COLUMNS
from quant_backtest.pathwise_labels import PATH_FEATURES
from quant_backtest.pathwise_models import select_path_rows

METHOD_NAMES = {"method1": "方法一新研究：完整核心加权净收益",
                "method2": "方法二新研究：环境与个股分解净收益"}
STOCK_FEATURES = [c for c in PATH_FEATURES if c not in MACRO_COLUMNS]


def calculate_frame_hash(frame: pd.DataFrame) -> str:
    """对有序frame的列名、类型与值生成指纹，用于真实核心输入、标签和权重审计，不写入原数据。"""
    digest = hashlib.sha256()
    digest.update(str([(c, str(frame[c].dtype)) for c in frame]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def calculate_interval_uniqueness(starts: np.ndarray, ends: np.ndarray, size: int) -> np.ndarray:
    """对单股starts/ends闭区间计算平均1/并发数；size为共同钟点数，不读取收益或实际提前退出。"""
    starts, ends = np.asarray(starts), np.asarray(ends)
    if (len(starts) != len(ends) or size < 1 or not np.issubdtype(starts.dtype, np.integer)
            or not np.issubdtype(ends.dtype, np.integer) or (starts < 0).any()
            or (ends >= size).any() or (ends < starts).any()):
        raise ValueError("潜在标签区间无效")
    delta = np.zeros(size + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends + 1, -1)
    concurrency = np.cumsum(delta[:-1])
    inverse = np.divide(1., concurrency, out=np.zeros(size), where=concurrency > 0)
    integral = np.r_[0., np.cumsum(inverse)]
    return (integral[ends + 1] - integral[starts]) / (ends - starts + 1)


def select_pooled_training(frame: pd.DataFrame, windows: list, horizon: int,
                           cfg: dict, times: pd.DatetimeIndex) -> pd.DataFrame:
    """从frame选择完整处于windows的horizon路径；cfg给同刻最低标签数，times给固定潜在窗口权重日历。

    核心内部不再删掉重叠入口，但跨角色边界不放宽；同股每天仍只有缓存预定的一个钟。
    唯一性仅按同股同年核心潜在区间计算，不因早盈/早亏改变观察长度。
    """
    out = select_path_rows(frame, windows, horizon, cfg).sort_values(["datetime", "symbol"]).copy()
    counts = out.groupby("datetime").symbol.transform("size")
    out = out.loc[counts >= cfg["pooled_min_label_group"]].reset_index(drop=True)
    if out.duplicated(["datetime", "symbol"]).any():
        raise ValueError("完整核心入口键重复")
    if out.assign(day=out.datetime.dt.normalize()).duplicated(["day", "symbol"]).any():
        raise ValueError("路径缓存不满足一股一天预定一个入口")
    if out.empty:
        return out.assign(sample_uniqueness=pd.Series(dtype=float), sample_weight=pd.Series(dtype=float))
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("共同钟必须唯一有序")
    starts = times.get_indexer(out.datetime)
    ends = times.get_indexer(out.label_end - pd.Timedelta(minutes=1))
    if (starts < 0).any() or (ends < 0).any():
        raise ValueError("完整核心信号或潜在终点不在共同钟内")
    uniqueness = np.empty(len(out))
    for positions in out.groupby([out.symbol, out.datetime.dt.year], sort=False).indices.values():
        uniqueness[positions] = calculate_interval_uniqueness(starts[positions], ends[positions], len(times))
    out["sample_uniqueness"] = uniqueness
    totals = out.groupby(out.datetime.dt.normalize()).sample_uniqueness.transform("sum")
    weights = uniqueness / totals
    out["sample_weight"] = weights / weights.mean()
    if not (np.isfinite(out.sample_weight) & (out.sample_weight > 0)).all():
        raise ValueError("训练样本权重非正或非有限")
    return out


def create_pooled_profiles(cfg: dict) -> list[dict]:
    """按cfg.path_horizons/pooled_model_profiles生成固定直接与分解模型网格，保持同期限退出。"""
    return [{"name": f"pooled_h{h}_{size['name']}_{kind}", "method": method, "algorithm": kind,
             "horizon": h, "parameters": {k: v for k, v in size.items() if k != "name"}}
            for h in cfg["path_horizons"] for size in cfg["pooled_model_profiles"]
            for method, kind in [("method1", "direct"), ("method2", "decomposed")]]


def prepare_pooled_targets(frame: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """由frame核心标签/权重及cfg截断生成账户目标、同刻环境目标和个股残差，环境量必须同刻一致。"""
    if frame.empty or not np.isfinite(frame.path_target).all():
        raise ValueError("核心目标为空或非有限")
    macro = frame[MACRO_COLUMNS].replace([np.inf, -np.inf], np.nan)
    # 将NaN也计为独立值，拒绝同刻一部分股票缺环境、一部分有值的不一致输入。
    if (macro.groupby(frame.datetime).nunique(dropna=False) > 1).any().any():
        raise ValueError("同一时点环境特征不一致")
    cap = cfg["label_clip"] * cfg["entry_equity_fraction"]
    target = frame.path_target.clip(-cap, cap)
    totals = frame.groupby("datetime").sample_weight.sum()
    common = (target * frame.sample_weight).groupby(frame.datetime).sum() / totals
    baseline = macro.assign(datetime=frame.datetime).drop_duplicates("datetime").set_index("datetime").sort_index()
    baseline["target"] = common
    baseline["weight"] = totals
    residual = target - frame.datetime.map(common)
    return target, baseline, residual


def fit_booster(frame: pd.DataFrame, target, weights, parameters: dict, cfg: dict):
    """只对frame核心特征、target账户收益和weights拟合；parameters给树超参，cfg给种子和线程。"""
    params = {"objective": "regression", "learning_rate": .03, "reg_alpha": 1.,
              "subsample": 1., "colsample_bytree": 1., "deterministic": True,
              "force_col_wise": True, "verbosity": -1, "random_state": cfg["model"]["random_state"],
              "n_jobs": cfg["workers"], **parameters}
    model = LGBMRegressor(**params)
    model.fit(frame.replace([np.inf, -np.inf], np.nan), np.asarray(target) * 10000,
              sample_weight=np.asarray(weights))
    return model


def summarize_pooled_training(frame: pd.DataFrame, cfg: dict) -> dict:
    """对frame实际核心输入、权重、潜在标签与分解目标记录摘要；cfg给截断，摘要可重建核验。"""
    target, baseline, residual = prepare_pooled_targets(frame, cfg)
    return {"training_rows": len(frame), "training_days": int(frame.datetime.dt.normalize().nunique()),
            "training_clocks": len(baseline), "mean_uniqueness": float(frame.sample_uniqueness.mean()),
            "minimum_uniqueness": float(frame.sample_uniqueness.min()),
            "core_keys_sha256": calculate_frame_hash(frame[["datetime", "symbol", "feature_source_start", "label_end"]]),
            "core_features_sha256": calculate_frame_hash(frame[list(PATH_FEATURES)]),
            "core_supervision_sha256": calculate_frame_hash(pd.DataFrame({
                "target": target, "residual": residual, "weight": frame.sample_weight,
                "uniqueness": frame.sample_uniqueness})),
            "core_environment_sha256": calculate_frame_hash(baseline.reset_index()),
            "target_unit": "相对独立初始账户净收益；各回归器训练乘10000、推断除10000",
            "label_clip_account_return": cfg["label_clip"] * cfg["entry_equity_fraction"]}


def fit_pooled_model(frame: pd.DataFrame, profile: dict, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """只用frame完整核心训练profile直接/分解模型；cfg给环境树配置，返回可重载模型包和中文分量重要性。"""
    if len(frame) < cfg["simple_min_training_rows"]:
        raise ValueError("完整核心模型样本不足")
    target, baseline, residual = prepare_pooled_targets(frame, cfg)
    components = {}
    if profile["algorithm"] == "direct":
        components["direct"] = fit_booster(frame[list(PATH_FEATURES)], target, frame.sample_weight, profile["parameters"], cfg)
    elif profile["algorithm"] == "decomposed":
        components["stock"] = fit_booster(frame[STOCK_FEATURES], residual, frame.sample_weight, profile["parameters"], cfg)
        environment_weight = baseline.weight / baseline.weight.mean()
        components["environment"] = fit_booster(baseline[MACRO_COLUMNS], baseline.target,
            environment_weight, cfg["pooled_environment_model"], cfg)
    else:
        raise ValueError("未知完整核心算法")
    details = []
    for name, model in components.items():
        names = model.booster_.feature_name()
        details.extend({"分量代码": name, "分量含义": {"direct": "总净收益", "stock": "个股相对净收益", "environment": "同刻共同净收益"}[name],
            "特征代码": c, "中文含义": PATH_FEATURES[c], "分裂增益": float(gain), "分裂次数": int(count)}
            for c, gain, count in zip(names, model.booster_.feature_importance("gain"), model.booster_.feature_importance("split")))
    return {"components": components, "profile": profile, "features": list(PATH_FEATURES),
            **summarize_pooled_training(frame, cfg)}, pd.DataFrame(details)


def generate_pooled_scores(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """只对frame当时特征使用bundle各冻结分量相加；cfg为统一接口，不读取标签或其他未来样本。"""
    out = frame[["datetime", "symbol"]].copy()
    score = np.zeros(len(out))
    for model in bundle["components"].values():
        if len(frame):
            features = model.booster_.feature_name()
            score += model.predict(frame[features].replace([np.inf, -np.inf], np.nan)) / 10000
    out["prediction"] = np.where(np.isfinite(score), score, -1e9)
    out["eligible"] = np.isfinite(score)
    return out


def calculate_pooled_diagnostics(bundle: dict, frame: pd.DataFrame, windows: list, cfg: dict) -> list[dict]:
    """仅在frame内部windows完整标签上比较冻结bundle的预测与实际账户标签；cfg给潜在期限，不用于外层选参。"""
    part = select_path_rows(frame, windows, bundle["profile"]["horizon"], cfg)
    if part.empty:
        return []
    scored = generate_pooled_scores(bundle, part, cfg)
    groups = pd.cut(scored.prediction.to_numpy(), [-np.inf, 0, .0002, .0005, .001, .002, np.inf])
    grouped = pd.DataFrame({"group": groups, "prediction": scored.prediction.to_numpy(),
                           "target": part.path_target.to_numpy()})
    table = grouped.groupby("group", observed=True).agg(rows=("target", "size"),
        predicted=("prediction", "mean"), realized_account_label=("target", "mean")).reset_index()
    table["group"] = table.group.astype(str)
    return table.to_dict("records")
