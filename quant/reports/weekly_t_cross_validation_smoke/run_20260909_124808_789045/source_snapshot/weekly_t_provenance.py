"""追踪周度/盘中特征的历史量价来源，不改特征值、不读取未来标签或执行行情。

前四周与前一周均指逐股实际有观测的周，停牌空周不占滚动窗口。周起点使用
对应自然周周一零时，保守覆盖周内全部量价；盘中还包括实际前一观测日，以及
同刻市场日收益的前收来源。累计history_days另列来源，调用方须从模型中删除
f_history_days后，才能把feature_source_start解释为全部模型量价输入的边界。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_PREVIOUS_CLOSE_FEATURES = ("f_market_day_return", "f_relative_day_return")
HISTORY_METADATA_FEATURES = ("f_history_days",)


def prepare_provenance_calendar(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从完整逐股观测日历建立前日、前周、前四观测周和累计历史起点，不读价格。

返回按symbol/date及symbol/feature_week索引的日历表。不能先截掉留出年或做
股票池筛选；缺少的历史行无法从滚动特征数值中恢复其真实来源。
    """
    if not {"date", "symbol"}.issubset(daily):
        raise ValueError("daily缺少date/symbol，无法重建量价依赖日历")
    if daily[["date", "symbol"]].isna().any().any():
        raise ValueError("daily日期或股票键缺失")
    days = daily[["symbol", "date"]].copy()
    days["symbol"] = days.symbol.astype(str)
    days["date"] = pd.to_datetime(days.date, errors="raise").dt.normalize()
    if days.duplicated(["symbol", "date"]).any():
        raise ValueError("daily的symbol/date必须唯一")
    days = days.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    grouped = days.groupby("symbol", sort=False)
    days["previous_date"] = grouped.date.shift(1)
    days["history_metadata_source_start"] = grouped.date.transform("first")
    days["feature_week"] = days.date - pd.to_timedelta(days.date.dt.dayofweek, unit="D")
    weeks = days[["symbol", "feature_week"]].drop_duplicates().reset_index(drop=True)
    grouped_weeks = weeks.groupby("symbol", sort=False)
    weeks["previous_week"] = grouped_weeks.feature_week.shift(1)
    fourth = grouped_weeks.feature_week.shift(4)
    weeks["feature_source_start"] = fourth.fillna(grouped_weeks.feature_week.transform("first"))
    weeks["history_metadata_source_start"] = weeks.symbol.map(
        days.groupby("symbol", sort=False).date.first())
    weeks = weeks.set_index(["symbol", "feature_week"])
    week_keys = pd.MultiIndex.from_frame(days[["symbol", "feature_week"]])
    days["previous_week"] = weeks.previous_week.reindex(week_keys).to_numpy()
    return days.set_index(["symbol", "date"]), weeks


def add_feature_provenance(weekly: pd.DataFrame, intraday: pd.DataFrame,
                           daily: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """返回新增来源列的周表、盘中表和审计metadata，原表和全部f_*数值保持不变。

weekly新增feature_source_start及history_metadata_source_start；intraday还新增
market_previous_close_source_start。存在市场日收益相关特征或cfg明确启用
add_session_context时，传播同刻有限日收益贡献者的前收来源。特征还未生成时
按当前正数signal_price/previous_close计算贡献资格，不需等待新增f_*。
仅有旧f_market_return/cs_return_rank且未配置新增市场特征时，
市场项使用当前开盘以来收益，不传播其他股票的历史来源。

调用方须提供未按年份或股票池裁剪的完整daily，以及构建市场特征时的完整
intraday截面，再进行训练/测试拆分。有限市场日收益若找不到其前日来源，
受影响的同刻feature_source_start设NaT以使严格训练筛选失败关闭。
    """
    if not {"symbol", "feature_week"}.issubset(weekly):
        raise ValueError("weekly缺少symbol/feature_week")
    if not {"symbol", "datetime"}.issubset(intraday):
        raise ValueError("intraday缺少symbol/datetime")
    for name, table, keys in (("weekly", weekly, ["symbol", "feature_week"]),
                               ("intraday", intraday, ["symbol", "datetime"])):
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError(f"{name}时间/股票键缺失或重复")
    days, weeks = prepare_provenance_calendar(daily)
    w, i = weekly.copy(deep=True), intraday.copy(deep=True)
    feature_week = pd.to_datetime(weekly.feature_week, errors="raise")
    if not ((feature_week == feature_week.dt.normalize()) & feature_week.dt.dayofweek.eq(0)).all():
        raise ValueError("feature_week必须为自然周周一零时")
    weekly_keys = pd.MultiIndex.from_arrays([weekly.symbol.astype(str), feature_week],
                                           names=["symbol", "feature_week"])
    if (weeks.index.get_indexer(weekly_keys) < 0).any():
        raise ValueError("weekly存在daily未覆盖的观测周，不能猜测历史依赖")
    aligned_weeks = weeks.reindex(weekly_keys)
    w["feature_source_start"] = aligned_weeks.feature_source_start.to_numpy()
    w["history_metadata_source_start"] = aligned_weeks.history_metadata_source_start.to_numpy()

    signal = pd.to_datetime(intraday.datetime, errors="raise")
    signal_date = signal.dt.normalize()
    if "date" in intraday and not pd.to_datetime(intraday.date).dt.normalize().eq(signal_date).all():
        raise ValueError("intraday.date必须与datetime属于同一日")
    daily_keys = pd.MultiIndex.from_arrays([intraday.symbol.astype(str), signal_date],
                                          names=["symbol", "date"])
    locations = days.index.get_indexer(daily_keys)
    if (locations < 0).any():
        raise ValueError("intraday存在daily未覆盖的交易日，不能猜测前日来源")
    previous_day = days.previous_date.to_numpy()[locations]
    previous_week = days.previous_week.to_numpy()[locations]
    current_day = signal_date.to_numpy()
    own_start = np.minimum(np.where(pd.isna(previous_day), current_day, previous_day),
                           np.where(pd.isna(previous_week), current_day, previous_week))
    i["history_metadata_source_start"] = days.history_metadata_source_start.to_numpy()[locations]
    cfg = {} if cfg is None else cfg
    market_enabled = bool(cfg.get("add_session_context", False)) or any(
        column in intraday for column in MARKET_PREVIOUS_CLOSE_FEATURES)
    unknown_market_rows = 0
    if market_enabled:
        if "f_day_return" in intraday:
            contribution = np.isfinite(pd.to_numeric(intraday.f_day_return, errors="raise").to_numpy())
        else:
            if not {"signal_price", "previous_close"}.issubset(intraday):
                raise ValueError("市场日收益来源审计需要f_day_return或当前signal_price/previous_close")
            price = pd.to_numeric(intraday.signal_price, errors="coerce").to_numpy()
            previous = pd.to_numeric(intraday.previous_close, errors="coerce").to_numpy()
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                day_return = price / previous - 1
            contribution = ((price > 0) & (previous > 0) & np.isfinite(price)
                            & np.isfinite(previous) & np.isfinite(day_return))
        source = pd.Series(previous_day).where(contribution)
        clock = signal.to_numpy()
        market_start = source.groupby(clock, sort=False).transform("min").to_numpy()
        unknown = pd.Series(contribution & pd.isna(previous_day)).groupby(clock, sort=False).transform("any").to_numpy()
        start = np.minimum(own_start, np.where(pd.isna(market_start), own_start, market_start))
        start[unknown] = np.datetime64("NaT")
        market_start[unknown] = np.datetime64("NaT")
        unknown_market_rows = int(unknown.sum())
    else:
        start = own_start
        market_start = np.full(len(intraday), np.datetime64("NaT"), dtype="datetime64[ns]")
    i["feature_source_start"] = start
    i["market_previous_close_source_start"] = market_start
    if (w.feature_source_start.to_numpy() > feature_week.to_numpy()).any():
        raise AssertionError("周特征来源晚于特征周，时间映射错误")
    if (i.feature_source_start.to_numpy() > signal.to_numpy()).any():
        raise AssertionError("盘中特征来源晚于当前信号，时间映射错误")
    metadata = {"weekly_rows": len(w), "intraday_rows": len(i), "daily_rows": len(days),
                "observed_stock_weeks": len(weeks), "market_previous_close_dependency_enabled": market_enabled,
                "unknown_market_source_rows": unknown_market_rows,
                "weekly_missing_source_rows": int(w.feature_source_start.isna().sum()),
                "intraday_missing_source_rows": int(i.feature_source_start.isna().sum()),
                "weekly_rule": "本股前4个实际有观测周的最早周一起点；不足4个使用最早已观测周",
                "intraday_rule": "本股前一个实际观测周起点与前一个实际观测日之较早者；无历史时为当前日",
                "market_rule": "仅市场日收益特征传播同datetime有限f_day_return贡献者的前收来源日",
                "required_model_feature_exclusions": list(HISTORY_METADATA_FEATURES),
                "history_metadata_rule": "累计history_days源自本股首个已观测日，单独标记；feature_source_start不包含该累计计数",
                "input_scope_requirement": "完整daily历史和构建市场特征时的完整intraday截面；先加来源列再做年度/股票池筛选",
                "no_feature_values_modified": True,
                "future_columns_read": []}
    return w, i, metadata
