"""双XGBoost开仓/继续持仓模型与四折回顾切分，无券商副作用。

复用年度面板和共同时钟行情；新增价格状态矩阵与虚拟持仓样本pkl。
行情特征只用当前/过去收盘，未来收益仅是训练标签，不能传入决策回调。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from quant_backtest.annual_portfolio import MarketStore


BASE_FEATURES = {
    "momentum_1": "前1个采样点收益", "momentum_5": "前5个采样点收益",
    "momentum_11": "前1交易日同刻收益", "momentum_33": "前3交易日同刻收益",
    "momentum_55": "前5交易日同刻收益", "volatility_11": "最近11点收益波动率",
    "volatility_55": "最近55点收益波动率", "atr_ratio": "240分钟ATR除以当前价",
    "amount_log": "当前分钟成交额对数", "volume_ratio": "当前量相对11点均量",
    "daily_return": "相对前收盘收益", "session_fraction": "当前交易时段位置",
}
STATE_FEATURES = {"lot_return": "本批次相对含滑点买价浮动收益",
                  "lot_age": "本批次持有交易日数", "lot_return_atr": "本批次浮盈除以当前ATR"}
EXIT_FEATURES = list(BASE_FEATURES) + list(STATE_FEATURES)


def build_research_folds(times: pd.DatetimeIndex, cfg: dict) -> list[dict]:
    """按times实际交易日及cfg比例构建四折；每个非测试年独立切拟合/校准/验证并隔离7自然日。"""
    days = pd.DatetimeIndex(sorted(times.normalize().unique()))
    years = sorted(days.year.unique())
    folds = []
    for test_year in cfg["test_years"]:
        if test_year not in years:
            raise ValueError(f"缺少测试年{test_year}，不能静默删除该折")
        windows = {"fit": [], "calibration": [], "validation": []}
        for year in years:
            if year == test_year:
                continue
            yd = days[days.year == year]
            first = int(len(yd) * cfg["fit_fraction"])
            second = int(len(yd) * (cfg["fit_fraction"] + cfg["calibration_fraction"]))
            boundaries = [yd[0], yd[first], yd[second], yd[-1] + pd.Timedelta(days=1)]
            for j, role in enumerate(windows):
                start = boundaries[j] + pd.Timedelta(days=cfg["embargo_days"])
                end = boundaries[j + 1]
                if start >= end:
                    raise ValueError("切片太短，隔离后没有数据")
                windows[role].append({"year": int(year), "start": start, "end": end})
        td = days[days.year == test_year]
        folds.append({"year": int(test_year), "training_years": [int(y) for y in years if y != test_year],
                      "future_trained": bool(max(y for y in years if y != test_year) > test_year),
                      "test_start": td[0], "test_end": td[-1] + pd.Timedelta(days=1), **windows})
    return folds


def select_window_rows(frame: pd.DataFrame, windows: list[dict], require_label: bool = True) -> pd.DataFrame:
    """按windows选frame；训练要求标签终点和虚拟买入时间同属单个窗口，避免跨边界收益泄漏。"""
    mask = np.zeros(len(frame), dtype=bool)
    for window in windows:
        part = (frame.datetime >= window["start"]) & (frame.datetime < window["end"])
        if require_label:
            part &= (frame.label_end > frame.datetime) & (frame.label_end < window["end"])
            if "entry_datetime" in frame:
                part &= frame.entry_datetime >= window["start"]
        mask |= part.to_numpy()
    return frame.loc[mask].copy()


def calculate_entry_target(frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    """用frame.forward_return/label_end算下一交易日退出净收益基点，cfg指定税费/滑点；买费不是沉没成本。"""
    stamp = np.where(frame.label_end >= pd.Timestamp("2023-08-28"), cfg["stamp_rate_from_20230828"],
                     cfg["stamp_rate_before_20230828"])
    fee = cfg["commission_rate"] + cfg["transfer_rate"]
    gross = 1 + frame.forward_return.to_numpy(dtype=float)
    net = gross * (1 - cfg["slippage_rate"]) * (1 - fee - stamp) / ((1 + cfg["slippage_rate"]) * (1 + fee)) - 1
    return np.clip(net, -cfg["label_clip"], cfg["label_clip"]) * 10000


def fit_return_model(frame: pd.DataFrame, features: list[str], target: np.ndarray, cfg: dict, profile: dict) -> XGBRegressor:
    """仅以frame[features]/target拟合回归；cfg/profile给超参。按时间等权，无测试eval_set或提前停止。"""
    params = {**cfg["model"], **{k: v for k, v in profile.items() if k != "name"}}
    model = XGBRegressor(**params, objective="reg:squarederror", tree_method="hist", n_jobs=cfg["workers"])
    weights = 1 / frame.groupby("datetime").datetime.transform("size").to_numpy()
    weights /= weights.mean()
    model.fit(frame[features], target, sample_weight=weights)
    return model


def calculate_base_matrices(market: MarketStore) -> dict[str, np.ndarray]:
    """从market仅当前close/volume/amount/atr/previous_close构造过去窗口矩阵；11点含日终，不填未来缺口。"""
    close = pd.DataFrame(market.values["close"])
    changes = close.pct_change(fill_method=None)
    output = {f"momentum_{n}": close.pct_change(n, fill_method=None) for n in [1, 5, 11, 33, 55]}
    output.update({f"volatility_{n}": changes.rolling(n, min_periods=max(3, n // 2)).std() for n in [11, 55]})
    volume = pd.DataFrame(market.values["volume"])
    output.update(atr_ratio=market.values["atr"] / close, amount_log=np.log1p(market.values["amount"]),
                  volume_ratio=volume / volume.rolling(11, min_periods=3).mean(),
                  daily_return=close / market.values["previous_close"] - 1)
    minutes = market.times.hour * 60 + market.times.minute - 571
    minutes = np.asarray(minutes - np.where(market.times.hour >= 13, 90, 0), dtype=float) / 240
    output["session_fraction"] = np.broadcast_to(minutes[:, None], close.shape)
    return {k: np.where(np.isfinite(output[k]), output[k], np.nan).astype("float32") for k in BASE_FEATURES}


def build_lot_features(matrices: dict, market: MarketStore, indices: np.ndarray, assets: np.ndarray,
                       buy_prices: np.ndarray, entry_indices: np.ndarray) -> np.ndarray:
    """使用matrices当前特征及每批indices/assets/buy_prices/entry_indices生成固定列序持仓状态；不访问执行价。"""
    current = market.values["close"][indices, assets]
    atr = market.values["atr"][indices, assets]
    age = market.day_index[indices] - market.day_index[entry_indices]
    values = [matrices[key][indices, assets] for key in BASE_FEATURES]
    values.extend([current / buy_prices - 1, age, (current - buy_prices) / np.where(atr > 0, atr, np.nan)])
    out = np.column_stack(values).astype("float32")
    out[~np.isfinite(out)] = np.nan
    return out


def build_exit_samples(panel: pd.DataFrame, market: MarketStore, matrices: dict, cfg: dict) -> pd.DataFrame:
    """从panel等间隔抽虚拟买点，再于cfg.exit_ages各交易日同刻产生持仓状态，标签仅供训练。

    虚拟批次不依赖任何模型/测试策略结果，覆盖离池后的持仓；忽略实际资金/止损存活筛选，
    属监督学习近似，不能解释为最优停止策略。继续一天与立即卖出比较，旧买费已沉没不重复扣。
    """
    sampled = panel.sort_values(["datetime", "symbol"]).iloc[::cfg["exit_entry_stride"]]
    entries = market.times.get_indexer(sampled.datetime)
    assets = np.array([market.asset_index[s] for s in sampled.symbol])
    days = pd.DatetimeIndex(sorted(market.days.unique()))
    offsets = market.times - market.days
    output = []
    for age in cfg["exit_ages"]:
        target_day = market.day_index[entries] + age
        valid = target_day + 1 < len(days)
        entry, asset, day = entries[valid], assets[valid], target_day[valid]
        state = market.times.get_indexer(days[day] + offsets[entry])
        future = market.times.get_indexer(days[day + 1] + offsets[entry])
        if (state < 0).any() or (future < 0).any():
            raise ValueError("共同时钟缺少同刻映射")
        buy = market.values["execution_close"][entry, asset] * (1 + cfg["slippage_rate"])
        now = market.values["execution_close"][state, asset]
        later = market.values["execution_close"][future, asset]
        stamp_now = np.where(market.times[state] >= pd.Timestamp("2023-08-28"), cfg["stamp_rate_from_20230828"], cfg["stamp_rate_before_20230828"])
        stamp_later = np.where(market.times[future] >= pd.Timestamp("2023-08-28"), cfg["stamp_rate_from_20230828"], cfg["stamp_rate_before_20230828"])
        fee = cfg["commission_rate"] + cfg["transfer_rate"]
        target = later * (1 - fee - stamp_later) / (now * (1 - fee - stamp_now)) - 1
        frame = pd.DataFrame(build_lot_features(matrices, market, state, asset, buy, entry), columns=EXIT_FEATURES)
        frame["datetime"] = market.times[state]
        frame["entry_datetime"] = market.times[entry] + pd.Timedelta(minutes=1)
        frame["label_end"] = market.times[future] + pd.Timedelta(minutes=1)
        frame["symbol"] = np.asarray(market.symbols)[asset]
        frame["continuation_target"] = np.clip(target, -cfg["label_clip"], cfg["label_clip"]) * 10000
        good = np.isfinite(buy) & (buy > 0) & np.isfinite(target) & (now > 0) & (later > 0)
        good &= market.values["execution_volume"][entry, asset] > 0
        good &= (market.values["execution_volume"][state, asset] > 0) & (market.values["execution_volume"][future, asset] > 0)
        output.append(frame.loc[good])
    return pd.concat(output, ignore_index=True)


def load_exit_cache(panel: pd.DataFrame, market: MarketStore, manifest: dict, cfg: dict) -> tuple[dict, pd.DataFrame, dict]:
    """复用/构建cfg.aux_cache_dir状态矩阵及监督样本；key包含源缓存、代码、费用和抽样，pkl仅限本地可信来源。"""
    source = Path(manifest["cache_path"])
    specs = {"source": str(source), "source_metadata": hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest(),
             "source_files": [(n, (source / n).stat().st_size, (source / n).stat().st_mtime_ns) for n in ["panel.pkl", "market.pkl"]],
             "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "panel_rows": len(panel),
             "eligibility_sha256": hashlib.sha256(pd.util.hash_pandas_object(panel[["datetime", "symbol"]], index=False).to_numpy().tobytes()).hexdigest(),
             "cfg": {k: cfg[k] for k in ["exit_ages", "exit_entry_stride", "label_clip", "slippage_rate", "commission_rate", "transfer_rate", "stamp_rate_from_20230828", "stamp_rate_before_20230828", "max_bar_amount_fraction", "lot_size", "filter_signal_untradable"]}}
    key = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cfg["aux_cache_dir"]) / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "exit_cache.pkl"
    if path.exists():
        print(f"复用持仓特征缓存：{path}", flush=True)
        obj = pd.read_pickle(path)
    else:
        print(f"构建持仓特征缓存（不重建原因子）：{path}", flush=True)
        matrices = calculate_base_matrices(market)
        obj = {"matrices": matrices, "samples": build_exit_samples(panel, market, matrices, cfg)}
        temp = path.with_suffix(".tmp.pkl")
        pd.to_pickle(obj, temp)
        temp.replace(path)
        (directory / "metadata.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj["matrices"], obj["samples"], {"cache_path": str(path), "samples": len(obj["samples"]), "key": key}


class LearnedExitPolicy:
    """独立批次继续持仓预测器；只对已满足T+1的批次发退出意图，不要求仍在选股池。"""

    def __init__(self, model: XGBRegressor, matrices: dict, market: MarketStore, threshold: float):
        """model为冻结持仓回归器；matrices为因果行情特征；market给当前状态；threshold为收益单位退出阈值。"""
        self.model, self.matrices, self.market, self.threshold = model, matrices, market, threshold

    def __call__(self, index: int, positions: list) -> dict[int, str]:
        """对index时刻positions逐批预测；当预测次日卖出相对立即卖出增益<=threshold时退出，缺报价不预测。"""
        market = self.market
        if market.times[index].hour == 15:
            return {}
        lots = [p for p in positions if market.days[index] > p.buy_time.normalize()
                and np.isfinite(market.values["close"][index, p.asset]) and not p.pending_reason]
        if not lots:
            return {}
        features = build_lot_features(self.matrices, market, np.full(len(lots), index),
                                     np.array([p.asset for p in lots]), np.array([p.buy_price for p in lots]),
                                     np.array([p.entry_index for p in lots]))
        scores = self.model.get_booster().inplace_predict(features, validate_features=False) / 10000
        return {p.lot_id: f"持仓模型预测继续一天收益{score:.4%}不高于阈值{self.threshold:.4%}"
                for p, score in zip(lots, scores) if np.isfinite(score) and score <= self.threshold}
