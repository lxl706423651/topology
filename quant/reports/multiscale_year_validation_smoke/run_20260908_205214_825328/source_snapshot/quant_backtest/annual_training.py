"""年度向前验证的训练与防泄漏边界；不读取测试收益，也不负责下单。

模型沿用现有XGBoost截面排序思路，标签改为下一交易日同一分钟的延迟收益。
特征缺失率、相关性筛选和重要性全部在传入的训练片段上计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def describe_feature(name: str) -> str:
    """把name因子代码转为中文族名/窗口说明；仅解释命名，不改变特征计算。"""
    mapping = {"oc_ret": "开收盘收益", "hl_range": "高低价振幅", "upper_shadow": "上影线比例", "lower_shadow": "下影线比例",
               "close_pos_in_bar": "收盘在分钟振幅内的位置", "vwap_gap": "相对分钟成交均价偏离", "momentum": "动量收益",
               "reversal": "反转收益", "ret_mean": "平均分钟收益", "ret_std": "分钟收益波动率", "ret_skew": "分钟收益偏度",
               "weighted_momentum": "线性加权动量", "downside_vol": "下行波动率", "parkinson_vol": "高低价波动率估计",
               "garman_klass_vol": "开高低收波动率估计", "rogers_satchell_vol": "趋势修正振幅波动率估计",
               "drawdown_from_high": "相对窗口最高价回撤", "rolling_sharpe": "滚动收益与波动比", "amihud_illiq": "价格变动与成交额比",
               "sma_gap": "简单均线偏离", "ema_gap": "指数均线偏离", "price_rank": "历史窗口内价格百分位",
               "dist_high": "距窗口最高价幅度", "dist_low": "距窗口最低价幅度", "breakout_pos": "区间突破位置",
               "volume_ratio": "成交量与均量比", "amount_ratio": "成交额与均额比", "volume_change": "成交量变化率",
               "amount_change": "成交额变化率", "volume_z": "成交量标准分", "amount_z": "成交额标准分",
               "money_flow_imbalance": "涨跌方向成交额不平衡", "ret_volume_corr": "收益与成交量变化相关性",
               "ret_amount_corr": "收益与成交额变化相关性", "range_mean": "平均振幅", "range_std": "振幅波动",
               "vwap_gap_mean": "平均成交均价偏离", "vol_change": "波动率变化", "trend_slope": "对数价格趋势斜率",
               "trend_r2": "趋势拟合优度", "ma_cross": "短长均线相对差", "boll_pctb": "布林带内价格位置",
               "boll_width": "布林带宽", "boll_upper_gap": "相对布林上轨偏离", "boll_lower_gap": "相对布林下轨偏离",
               "rsi": "相对强弱指标", "roc": "价格变动率", "atr_ratio": "真实波幅与价格比", "cci": "顺势指标",
               "willr": "威廉超买超卖指标", "mfi": "资金流量指标", "adx": "平均趋向强度", "macd_dif": "快慢指数均线差",
               "macd_dea": "均线差信号线", "macd_hist": "均线差柱状动量", "stoch_k": "随机指标快线",
               "stoch_d": "随机指标慢线", "obv_z": "能量潮标准分", "minute_sin": "日内时间正弦编码",
               "minute_cos": "日内时间余弦编码"}
    raw = name.removeprefix("cs_rank_")
    for prefix in sorted(mapping, key=len, reverse=True):
        if raw == prefix or raw.startswith(prefix + "_"):
            parameters = raw[len(prefix):].strip("_")
            suffix = f"，分钟窗口或参数{parameters}" if parameters else ""
            return "当时股票池内百分位秩：" + mapping[prefix] + suffix
    return "当时股票池内百分位秩：" + raw


def create_year_folds(cfg: dict) -> list[dict]:
    """由 cfg 的测试年/数据终点生成核心训练、训练内校准、验证和测试的半开区间。"""
    folds = []
    for year in cfg["test_years"]:
        test_start = pd.Timestamp(f"{year}-01-01")
        end = min(pd.Timestamp(f"{year + 1}-01-01"), pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1))
        core_end = pd.Timestamp(f"{year - 1}-07-01")
        validation_start = pd.Timestamp(f"{year - 1}-10-01")
        if core_end <= pd.Timestamp(cfg["start_date"]) or end <= test_start:
            raise ValueError(f"{year}没有足够的过去训练数据或测试行情")
        folds.append({"year": year, "train_start": pd.Timestamp(cfg["start_date"]), "train_end": core_end,
                      "calibration_start": core_end, "calibration_end": validation_start,
                      "validation_start": validation_start, "validation_end": test_start,
                      "test_start": test_start, "test_end": end})
    return folds


def select_signal_tradable_panel(panel: pd.DataFrame, market: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按信号时已知market行情筛选panel中的可买候选；cfg提供容量/整手约束。

    只读取当前close/volume/amount和历史previous_close，绝不读取execution_*。
    筛后重新计算标签截面秩和截面数量；缓存原始因子不改动。旧协议缺少开关时不筛。
    """
    if not cfg.get("filter_signal_untradable", False):
        return panel
    cols = ["datetime", "symbol", "close", "previous_close", "volume", "amount"]
    quotes = market[cols].set_index(["datetime", "symbol"])
    aligned = panel[["datetime", "symbol"]].join(quotes, on=["datetime", "symbol"], validate="one_to_one")
    valid = aligned.close.notna() & (aligned.previous_close > 0) & (aligned.volume > 0)
    valid &= aligned.close / aligned.previous_close < 1.095
    valid &= aligned.amount * cfg["max_bar_amount_fraction"] >= aligned.close * cfg["lot_size"]
    out = panel.loc[valid].copy()
    out["target_rank"] = out.groupby("datetime").forward_return.rank(pct=True).astype("float32")
    out["cross_section_count"] = out.groupby("datetime").symbol.transform("size")
    return out


def select_training_rows(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, minimum: int) -> pd.DataFrame:
    """从 panel 取[start,end)训练行；标签结束也须早于end，minimum限制当时截面规模。"""
    mask = (panel.datetime >= start) & (panel.datetime < end) & (panel.label_end < end)
    mask &= panel.target_rank.notna() & (panel.cross_section_count >= minimum)
    return panel.loc[mask].copy()


def select_training_features(train: pd.DataFrame, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    """在 train 内按覆盖率、秩相关及冗余阈值筛选最多cfg.max_features列，返回清单及审计。"""
    cols = [c for c in train if c.startswith("cs_rank_")]
    if len(train) < 100:
        raise ValueError("有效训练行不足100，拒绝生成误导模型")
    # 固定等间隔抽样限制相关矩阵计算开销；抽样仅来自该轮训练期。
    stride = max(1, len(train) // 50000)
    sample = train.iloc[::stride]
    coverage = train[cols].notna().mean()
    usable = [c for c in cols if coverage[c] >= cfg["feature_min_coverage"] and sample[c].std() > 1e-6]
    standardized = sample[usable].fillna(0.5)
    strength = standardized.corrwith(sample.target_rank).fillna(0.0)
    correlation = standardized.corr().abs()
    selected, rows = [], []
    ordered = sorted(usable, key=lambda c: (-abs(strength[c]), c))
    for col in ordered:
        maximum = float(correlation.loc[col, selected].max()) if selected else 0.0
        keep = len(selected) < cfg["max_features"] and maximum < cfg["feature_max_correlation"]
        rows.append({"feature": col, "说明": describe_feature(col), "training_rank_correlation": strength[col],
                     "coverage": coverage[col], "max_correlation_to_selected": maximum, "selected": keep})
        if keep:
            selected.append(col)
    for col in sorted(set(cols).difference(usable)):
        rows.append({"feature": col, "说明": describe_feature(col) + "；覆盖率不足或近似常量，排除", "coverage": coverage[col], "selected": False})
    if not selected:
        raise ValueError("训练期没有可用特征")
    return selected, pd.DataFrame(rows)


def fit_rank_model(train: pd.DataFrame, features: list[str], cfg: dict) -> XGBRegressor:
    """仅对train的features拟合截面排序回归；cfg提供固定超参数、线程数，不接收测试eval_set。"""
    model = XGBRegressor(**cfg["model"], objective="reg:squarederror", tree_method="hist",
                         n_jobs=cfg["workers"], eval_metric="rmse")
    weights = 1 / train.groupby("datetime").symbol.transform("size").to_numpy()
    weights /= weights.mean()
    model.fit(train[features], train.target_rank, sample_weight=weights)
    return model
