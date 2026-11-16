"""可解释的有限窗口收益模型：输入有界特征面板，仅在核心训练片段拟合。

岭回归和分箱保序回归共享净收益标签；推断不读取未来收益、成交量或标签。
技术信号只决定入场资格，独立批次的T+1、ATR与到期退出由既有引擎负责。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from quant_backtest.multiscale_training import CONTEXT_NAMES
from quant_backtest.learned_exit_models import calculate_entry_target, select_window_rows
from quant_backtest.exit_alignment import HorizonExitPolicy

FEATURE_NAMES = {
    **{k: v for k, v in CONTEXT_NAMES.items() if "60d" not in k},
    **{f"cs_rank_reversal_{n}": f"过去{n}原始分钟反转值的截面百分位" for n in [20, 60, 120]},
    **{f"cs_rank_boll_pctb_{n}": f"{n}原始分钟布林带位置的截面百分位" for n in [20, 60]},
    **{f"cs_rank_money_flow_imbalance_{n}": f"{n}原始分钟资金流不平衡的截面百分位" for n in [20, 60]},
    "cs_rank_breakout_pos_60": "过去60原始分钟区间位置的截面百分位",
    "signal_daily_return": "本股当前已完成分钟收盘相对已知前收盘的收益",
}
FAMILY_NAMES = {
    "oversold_reversal": "超跌反转",
    "pullback_trend": "趋势中的回调",
    "trend_breakout": "趋势突破",
}
METHOD_NAMES = {"method1": "方法一新对照：低维岭回归净收益",
                "method2": "方法二新对照：技术信号保序收益校准"}


def build_simple_panel(panel: pd.DataFrame, market, matrices: dict) -> pd.DataFrame:
    """从panel挑固定特征及标签/来源列，market提供键，matrices提供当前daily_return；不重建因子。"""
    metadata = ["datetime", "symbol", "forward_return", "label_end",
                "feature_source_start", "cross_section_count"]
    labels = [c for c in panel if c.startswith(("future_return_", "future_end_"))]
    out = panel[list(dict.fromkeys(metadata + labels + [c for c in FEATURE_NAMES if c != "signal_daily_return"]))].copy()
    rows = market.times.get_indexer(out.datetime)
    columns = np.array([market.asset_index[s] for s in out.symbol])
    if (rows < 0).any():
        raise ValueError("特征时刻不在共同行情时钟")
    out["signal_daily_return"] = matrices["daily_return"][rows, columns]
    return out


def select_horizon_rows(frame: pd.DataFrame, windows: list[dict], horizon: int) -> pd.DataFrame:
    """把frame的horizon日标签映射到统一列，只保留完整落在windows内的有限标签。"""
    if horizon < 1 or int(horizon) != horizon:
        raise ValueError("持有期限必须为正整数交易日")
    out = frame if horizon == 1 else frame.assign(
        forward_return=frame[f"future_return_{horizon}d"], label_end=frame[f"future_end_{horizon}d"])
    out = select_window_rows(out, windows)
    return out.loc[np.isfinite(out.forward_return)].copy()


def select_spaced_training(frame: pd.DataFrame, times: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """对frame按times当年交易日序号每horizon+1日抽样，每股每天预定一个信号钟，不依标签/行数选择。"""
    if horizon < 1 or int(horizon) != horizon:
        raise ValueError("抽样期限必须为正整数")
    if frame.duplicated(["datetime", "symbol"]).any():
        raise ValueError("训练入口键重复")
    days = pd.DatetimeIndex(sorted(times.normalize().unique()))
    ordinal = pd.Series(np.arange(len(days)), index=days).groupby(days.year).cumcount()
    eligible_day = frame.datetime.dt.normalize().map(ordinal).mod(horizon + 1).eq(0)
    offsets = sorted(set(times[times.hour != 15] - times[times.hour != 15].normalize()))
    # 股票和自然日决定钟点；该钟缺行情时不补选当天其他钟，避免依赖后续可用性。
    keys = pd.DataFrame({"day": frame.datetime.dt.normalize(), "symbol": frame.symbol.astype(str)})
    slot = pd.util.hash_pandas_object(keys, index=False).to_numpy() % len(offsets)
    selected_offset = pd.to_timedelta(np.array(offsets, dtype="timedelta64[ns]")[slot])
    mask = eligible_day & ((frame.datetime - frame.datetime.dt.normalize()).to_numpy() == selected_offset)
    out = frame.loc[mask].sort_values(["symbol", "datetime"]).copy()
    previous_end = out.groupby("symbol", observed=True).label_end.shift()
    if (previous_end >= out.datetime).any():
        raise ValueError("抽样后仍存在同股重叠标签")
    return out


def calculate_day_weights(frame: pd.DataFrame) -> np.ndarray:
    """按frame自然交易日内样本数倒数赋权并归一至均值1；跨股票关联未因此消失。"""
    weights = 1 / frame.groupby(frame.datetime.dt.normalize()).datetime.transform("size").to_numpy()
    return weights / weights.mean()


def calculate_family_signal(frame: pd.DataFrame, family: str) -> tuple[np.ndarray, np.ndarray]:
    """从frame固定当时特征构造family的0至1强度及入场资格；缺资格输入失败关闭，不查看标签。"""
    daily = frame.signal_daily_return
    daily_rank = frame.cs_rank_context_daily_return
    momentum = frame.cs_rank_context_momentum_5d
    flow = frame.cs_rank_money_flow_imbalance_20
    if family == "oversold_reversal":
        score = (.5 * (1 - daily_rank) + .25 * (1 - frame.cs_rank_context_momentum_1d)
                 + .25 * (1 - frame.cs_rank_boll_pctb_60))
        eligible = daily < 0
    elif family == "pullback_trend":
        score = .5 * (1 - daily_rank) + .25 * (1 - frame.cs_rank_boll_pctb_20) + .25 * flow
        eligible = (daily < 0) & (momentum >= .6) & (flow >= .5)
    elif family == "trend_breakout":
        score = .5 * momentum + .25 * frame.cs_rank_breakout_pos_60 + .25 * flow
        eligible = (daily > 0) & (momentum > .5)
    else:
        raise ValueError(f"未知技术信号：{family}")
    valid = eligible & np.isfinite(daily) & np.isfinite(score)
    return score.fillna(.5).clip(0, 1).to_numpy(), valid.to_numpy()


def create_profiles(cfg: dict) -> list[dict]:
    """从cfg.simple_horizons/ridge_penalties/isotonic_families生成事前候选，不按外层成绩扩网格。"""
    profiles = []
    for h in cfg["simple_horizons"]:
        for penalty in cfg["ridge_penalties"]:
            profiles.append({"name": f"ridge_h{h}_p{penalty:g}", "method": "method1", "algorithm": "ridge",
                             "horizon": h, "penalty": penalty})
        for family in cfg["isotonic_families"]:
            profiles.append({"name": f"{family}_h{h}", "method": "method2", "algorithm": "isotonic",
                             "horizon": h, "family": family})
    return profiles


def select_profile_training(frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """根据profile技术族过滤frame核心样本；岭回归保留全部，保序模型仅拟合其事前资格区域。"""
    if profile["algorithm"] == "ridge":
        return frame.copy()
    _, eligible = calculate_family_signal(frame, profile["family"])
    return frame.loc[eligible].copy()


def fit_simple_model(frame: pd.DataFrame, profile: dict, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """只在frame上拟合profile模型及预处理；cfg给费用、最少样本、分箱及线程数，返回模型包和中文解释表。"""
    if len(frame) < cfg["simple_min_training_rows"]:
        raise ValueError("核心训练样本不足")
    features = list(FEATURE_NAMES)
    target = calculate_entry_target(frame, cfg)
    weights = calculate_day_weights(frame)
    if not np.isfinite(target).all():
        raise ValueError("核心标签含非有限值")
    if profile["algorithm"] == "ridge":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=profile["penalty"] * len(frame), solver="svd")),
        ])
        with threadpool_limits(limits=cfg["workers"]):
            model.fit(frame[features].replace([np.inf, -np.inf], np.nan), target,
                      scaler__sample_weight=weights, ridge__sample_weight=weights)
        details = pd.DataFrame({"特征代码": features, "中文含义": [FEATURE_NAMES[c] for c in features],
            "训练中位数": model["imputer"].statistics_, "训练均值": model["scaler"].mean_,
            "训练标准差": model["scaler"].scale_, "标准化系数_净收益基点": model["ridge"].coef_})
    else:
        strength, eligible = calculate_family_signal(frame, profile["family"])
        if not eligible.all():
            raise ValueError("保序训练包含不属于该技术族的样本")
        bins = cfg["isotonic_bins"]
        bucket = np.minimum((strength * bins).astype(int), bins - 1)
        detail = pd.DataFrame({"bin": bucket, "weight": weights, "weighted_target": target * weights})
        grouped = detail.groupby("bin").agg(rows=("bin", "size"), weight=("weight", "sum"), value=("weighted_target", "sum"))
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit((grouped.index.to_numpy() + .5) / bins, grouped.value / grouped.weight, sample_weight=grouped.weight)
        details = pd.DataFrame({"信号强度分箱": grouped.index, "核心样本数": grouped.rows,
            "日期权重和": grouped.weight, "实际净收益基点": grouped.value / grouped.weight,
            "单调校准净收益基点": model.predict((grouped.index.to_numpy() + .5) / bins)}).reset_index(drop=True)
    return {"model": model, "profile": profile, "features": features, "target_unit": "净收益基点",
            "training_rows": len(frame)}, details


def generate_simple_scores(bundle: dict, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """只用bundle冻结模型和frame固定特征预测；cfg给线程数，输出含资格掩码但绝不含标签。"""
    profile, model = bundle["profile"], bundle["model"]
    if not len(frame):
        return pd.DataFrame(columns=["datetime", "symbol", "prediction", "eligible"])
    if profile["algorithm"] == "ridge":
        with threadpool_limits(limits=cfg["workers"]):
            prediction = model.predict(frame[bundle["features"]].replace([np.inf, -np.inf], np.nan)) / 10000
        eligible = np.ones(len(frame), dtype=bool)
    else:
        strength, eligible = calculate_family_signal(frame, profile["family"])
        prediction = model.predict(strength) / 10000
    out = frame[["datetime", "symbol"]].copy()
    out["prediction"] = np.where(eligible & np.isfinite(prediction), prediction, -1e9)
    out["eligible"] = eligible & np.isfinite(prediction)
    return out


def calculate_gate(scores: pd.DataFrame, quantile: float, floor: float) -> float:
    """仅从scores校准时钟的合资格非负预测最大值取quantile分位，floor为最低净收益门槛。"""
    valid = scores.loc[scores.eligible & (scores.prediction >= floor)]
    top = valid.groupby("datetime").prediction.max()
    return max(floor, float(top.quantile(quantile))) if len(top) else 1e9


def create_exit_policy(market, rule: dict):
    """依rule.exit_mode/horizon创建到期退出器；ATR吊顶由market对应原批次引擎处理。"""
    if rule["exit_mode"] == "matched_horizon":
        return HorizonExitPolicy(market, rule["horizon"], "matched_horizon")
    if rule["exit_mode"] != "atr_trailing":
        raise ValueError("未知退出类型")
    return None

