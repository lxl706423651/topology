"""在同一外层折内生成OOF周top100分钟训练队列，并传播周预测的真实历史来源。

读取已具备因果资格和feature_source_start的周表、分钟表与配置；周模型仅用
另一个核心年份拟合。输出独立模型、预测、选择及元数据，不改输入行情或v5源码。
周排名是当周已知预测，未来周标签和未来恢复是否成功不参与队列/上下文选择。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .weekly_t_models import fit_weekly_models, predict_weekly, write_model_json


CONTEXT_FEATURES = {
    "pred_return_rank": "f_weekly_pred_return_rank",
    "pred_amplitude_rank": "f_weekly_pred_amplitude_rank",
    "score": "f_weekly_score",
    "rank": "f_weekly_selection_rank",
}
TRACE_COLUMNS = ("source_model_training_years", "source_model_sha256")
WEEK_KEYS = ["trade_week", "symbol"]


def calculate_cohort_file_sha256(path: Path) -> str:
    """流式记录实际读取/保存文件的SHA256，不载入大型模型文件的第二份副本。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cohort_years(values, name: str) -> list[int]:
    """规范年份集合；拒绝字符串、布尔和小数，避免悄悄改变隔离范围。"""
    if isinstance(values, str):
        raise ValueError(f"{name}须为整数年份列表")
    try:
        years = list(values)
    except TypeError as exc:
        raise ValueError(f"{name}须为整数年份列表") from exc
    if any(isinstance(year, (bool, np.bool_)) or not isinstance(year, (int, np.integer))
           or year < 1900 or year > 2200 for year in years):
        raise ValueError(f"{name}包含非法年份")
    return sorted(set(map(int, years)))


def validate_cohort_config(cfg: dict) -> int:
    """要求真实来源隔离和固定周排名配置；返回选股数量供rank归一化。"""
    if cfg.get("strict_feature_provenance") is not True or "f_history_days" not in cfg.get("excluded_model_features", []):
        raise ValueError("OOF队列必须启用strict_feature_provenance并排除模型f_history_days")
    raw_top = cfg.get("top_n", cfg.get("weekly_top_n", 100))
    if isinstance(raw_top, (bool, np.bool_)) or not isinstance(raw_top, (int, np.integer)) or raw_top < 1:
        raise ValueError("top_n须为正整数")
    if cfg.get("weekly_top_n", raw_top) != raw_top:
        raise ValueError("top_n与weekly_top_n必须一致")
    weights = [cfg.get("weekly_return_weight", .7), cfg.get("weekly_amplitude_weight", .3)]
    if not np.isfinite(weights).all() or min(weights) < 0 or not np.isclose(sum(weights), 1.):
        raise ValueError("周预测权重必须有限非负且合计1")
    return int(raw_top)


def prepare_cohort_keys(frame: pd.DataFrame, name: str, required: set[str], keys: list[str]) -> pd.DataFrame:
    """复制必需键、规范本地时间并拒绝重复，不按标签是否存在删除当前行。"""
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame):
        raise ValueError(f"{name}缺少必需列：{sorted(required)}")
    result = frame.copy(deep=True)
    if result.symbol.isna().any() or result.symbol.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}股票代码缺失")
    result["symbol"] = result.symbol.astype(str)
    for column in set(keys + ["trade_week"]) - {"symbol"}:
        result[column] = pd.to_datetime(result[column], errors="raise")
        values = pd.DatetimeIndex(result[column])
        if values.tz is not None or values.hasnans:
            raise ValueError(f"{name}.{column}须为非缺失无时区时间")
    if not (result.trade_week.dt.normalize().eq(result.trade_week) & result.trade_week.dt.dayofweek.eq(0)).all():
        raise ValueError(f"{name}.trade_week须为周一零点")
    if result.duplicated(keys).any():
        raise ValueError(f"{name}键{keys}重复")
    return result


def normalize_source_year_trace(value) -> str:
    """将模型训练年记录为稳定JSON字符串；这个溯源字段不会进入f_*模型输入。"""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("source_model_training_years须为JSON整数列表") from exc
    years = validate_cohort_years(value, "source_model_training_years")
    if not years:
        raise ValueError("source_model_training_years不能为空")
    return json.dumps(years, separators=(",", ":"))


def attach_weekly_context(intraday: pd.DataFrame, weekly_predictions: pd.DataFrame,
                          weekly_universe: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """给所传分钟行附唯一已知周预测，保留原顺序/索引/未来标签并扩展最早量价来源。

调用者先选定股周；本函数不偷偷丢失缺预测行，也不重新排序或选股。输入须有
datetime/trade_week/symbol/feature_source_start；预测须有周信号时间、四个排名列。
模型训练年及bundle SHA由预测非f列给出，或由cfg的weekly_context_model_*填缺列。
周来源来自同股周universe，不信任预测表携带的来源。未知/未来来源失败关闭。
    """
    top_n = validate_cohort_config(cfg)
    reserved = {*CONTEXT_FEATURES.values(), "intraday_feature_source_start", "weekly_context_feature_source_start",
                "weekly_context_signal_datetime", *TRACE_COLUMNS, "_weekly_context_order"}
    if reserved.intersection(intraday.columns):
        raise ValueError("intraday已含周上下文列，不能静默覆盖已有来源或预测")
    minutes = prepare_cohort_keys(intraday, "intraday", {"datetime", "trade_week", "symbol", "feature_source_start"},
                                  ["datetime", "symbol"])
    predictions = prepare_cohort_keys(weekly_predictions, "weekly_predictions",
                                      {*WEEK_KEYS, "signal_datetime", *CONTEXT_FEATURES}, WEEK_KEYS)
    universe = prepare_cohort_keys(weekly_universe, "weekly_universe",
                                   {*WEEK_KEYS, "signal_datetime", "feature_week", "feature_source_start"}, WEEK_KEYS)
    for column in CONTEXT_FEATURES:
        values = pd.to_numeric(predictions[column], errors="raise")
        valid = np.isfinite(values) & (values.gt(0) if column == "rank" else values.between(0, 1))
        if not valid.all():
            raise ValueError(f"weekly_predictions.{column}数值非法")
        if column == "rank" and not values.eq(np.floor(values)).all():
            raise ValueError("weekly_predictions.rank须为正整数")
        predictions[column] = values
    expected_score = (predictions.pred_return_rank * cfg.get("weekly_return_weight", .7)
                      + predictions.pred_amplitude_rank * cfg.get("weekly_amplitude_weight", .3))
    if not np.allclose(predictions.score, expected_score, atol=1e-12, rtol=0):
        raise ValueError("周score与配置的预测排名权重不一致")
    for column, key in zip(TRACE_COLUMNS, ["weekly_context_model_training_years", "weekly_context_model_sha256"]):
        if column not in predictions:
            if key not in cfg:
                raise ValueError(f"缺少模型溯源列{column}或配置{key}")
            value = normalize_source_year_trace(cfg[key]) if column == TRACE_COLUMNS[0] else cfg[key]
            predictions[column] = value
    predictions[TRACE_COLUMNS[0]] = predictions[TRACE_COLUMNS[0]].map(normalize_source_year_trace)
    if not predictions[TRACE_COLUMNS[1]].map(lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None).all():
        raise ValueError("source_model_sha256须为64位小写十六进制摘要")
    if any(int(year) in json.loads(trace) for year, trace in zip(predictions.trade_week.dt.year, predictions[TRACE_COLUMNS[0]])):
        raise ValueError("周预测模型不能拟合自己的目标年份")

    context = predictions[[*WEEK_KEYS, "signal_datetime", *CONTEXT_FEATURES, *TRACE_COLUMNS]].rename(
        columns={"signal_datetime": "weekly_context_signal_datetime"})
    source = universe[[*WEEK_KEYS, "signal_datetime", "feature_week", "feature_source_start"]].rename(
        columns={"signal_datetime": "_universe_signal_datetime", "feature_week": "_universe_feature_week",
                 "feature_source_start": "weekly_context_feature_source_start"})
    context = context.merge(source, on=WEEK_KEYS, how="left", validate="one_to_one", indicator="_source_merge")
    if not context._source_merge.eq("both").all():
        raise ValueError("周预测缺少对应weekly_universe来源")
    for column in ["weekly_context_signal_datetime", "_universe_signal_datetime", "_universe_feature_week", "weekly_context_feature_source_start"]:
        context[column] = pd.to_datetime(context[column], errors="raise")
        if pd.DatetimeIndex(context[column]).tz is not None or context[column].isna().any():
            raise ValueError(f"周上下文{column}缺失或带时区")
    if not (context.weekly_context_signal_datetime.eq(context._universe_signal_datetime)
            & context.weekly_context_feature_source_start.le(context._universe_feature_week)
            & context._universe_feature_week.le(context.weekly_context_signal_datetime)
            & context.weekly_context_signal_datetime.lt(context.trade_week)).all():
        raise ValueError("周来源/信号时间不一致或晚于当周可知时点")
    context = context.drop(columns=["_source_merge", "_universe_signal_datetime", "_universe_feature_week"])
    context = context.rename(columns=CONTEXT_FEATURES)
    context[CONTEXT_FEATURES["rank"]] = context[CONTEXT_FEATURES["rank"]] / top_n
    minutes["_weekly_context_order"] = np.arange(len(minutes))
    merged = minutes.merge(context, on=WEEK_KEYS, how="left", validate="many_to_one", indicator="_prediction_merge")
    if not merged._prediction_merge.eq("both").all():
        raise ValueError("分钟样本缺少唯一对应周预测，不能静默删行")
    own_source = pd.to_datetime(merged.feature_source_start, errors="raise")
    if pd.DatetimeIndex(own_source).tz is not None or own_source.isna().any():
        raise ValueError("分钟feature_source_start缺失或带时区")
    if not (own_source.le(merged.datetime) & merged.weekly_context_signal_datetime.lt(merged.datetime)
            & merged.datetime.ge(merged.trade_week) & merged.datetime.lt(merged.trade_week + pd.Timedelta(days=7))).all():
        raise ValueError("分钟或周预测来源晚于当前信号，或分钟不属于该股周")
    merged["intraday_feature_source_start"] = own_source
    merged["feature_source_start"] = np.minimum(own_source.to_numpy(), merged.weekly_context_feature_source_start.to_numpy())
    result = merged.sort_values("_weekly_context_order", kind="stable").drop(columns=["_weekly_context_order", "_prediction_merge"])
    result.index = intraday.index.copy()
    result.attrs = dict(intraday.attrs)
    result.attrs["weekly_context_contract"] = "仅周预测排名/score；最早来源=min(分钟原来源,周特征真实来源)"
    return result


def select_cohort_target_weeks(weekly_universe: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """只按已知周信号完整性及同年首末交易日选目标周，不读股票未来标签完整性。

不按smoke_weeks裁剪核心队列，避免前三周恰被来源隔离清空；小样本资源限制由
调用方股票数/模型参数控制。末尾全局不完整周只保留已给出的合法周中信号。
    """
    required = {*WEEK_KEYS, "entry_date", "exit_date", "feature_week_complete"}
    if not required.issubset(weekly_universe):
        raise ValueError(f"OOF周目标缺少日历/完整性列{required}")
    start = pd.to_datetime(weekly_universe.entry_date, errors="raise")
    end = pd.to_datetime(weekly_universe.exit_date, errors="raise")
    mask = (start.dt.year.eq(target_year) & end.dt.year.eq(target_year) & start.lt(end)
            & weekly_universe.trade_week.dt.year.eq(target_year) & weekly_universe.feature_week_complete.eq(True))
    return weekly_universe.loc[mask].copy()


def build_core_t_cohort(weekly_universe: pd.DataFrame, intraday: pd.DataFrame,
                        core_years, excluded_years, cfg: dict, output_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """用同折两个核心年的互为OOF周选择生成分钟T队列，返回未按T标签删行的训练表。

每个selector只拟合另一个核心年；排除自身目标年、内部/外层和全部其余可用年。
只保存当前输出目录下新产物，不复用其他折模型。返回行包含所选股周中间交易日、
四个新f_*及模型溯源；调用方接短周期标签后仍须按扩展来源执行T训练严格purge。
    """
    top_n = validate_cohort_config(cfg)
    core = validate_cohort_years(core_years, "core_years")
    excluded = validate_cohort_years(excluded_years, "excluded_years")
    if len(core) != 2 or set(core).intersection(excluded):
        raise ValueError("OOF队列要求两个核心年份，且不能与排除年份重叠")
    weekly = prepare_cohort_keys(weekly_universe, "weekly_universe",
                                 {*WEEK_KEYS, "feature_week", "signal_datetime", "feature_source_start", "entry_date", "exit_date", "feature_week_complete"}, WEEK_KEYS)
    minutes = prepare_cohort_keys(intraday, "intraday", {"datetime", "trade_week", "symbol", "feature_source_start"}, ["datetime", "symbol"])
    available = sorted(set(map(int, weekly.trade_week.dt.year)) | set(map(int, minutes.datetime.dt.year)) | set(core) | set(excluded))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "cohort_metadata.json").exists() or any((directory / f"selector_{year}").exists() for year in core):
        raise ValueError("OOF输出已有模型或元数据；请使用新的目录，不能覆盖冻结实验")
    parts, selectors = [], []
    effective_cfg = {**cfg, "top_n": top_n, "weekly_top_n": top_n}
    for target_year in core:
        train = [year for year in core if year != target_year]
        selector_excluded = [year for year in available if year not in train]
        model_dir = directory / f"selector_{target_year}"
        bundle = fit_weekly_models(weekly, train, effective_cfg, model_dir, excluded_years=selector_excluded)
        model_meta = bundle["metadata"]
        if model_meta.get("train_years") != train or sorted(model_meta.get("excluded_years", [])) != selector_excluded:
            raise ValueError("周selector模型元数据的训练/排除年与本折OOF协议不一致")
        model_path = model_dir / "model_bundle.joblib"
        model_sha = calculate_cohort_file_sha256(model_path)
        targets = select_cohort_target_weeks(weekly, target_year)
        predictions = predict_weekly(bundle, targets, effective_cfg)
        if "selected" not in predictions or not predictions.selected.isin([True, False]).all():
            raise ValueError("周模型预测须给出合法selected标记")
        selected = predictions.loc[predictions.selected.eq(True)].copy()
        if selected.duplicated(WEEK_KEYS).any() or (selected.groupby("trade_week").size() > top_n).any():
            raise ValueError("OOF周选择重复或超过top_n")
        selected[TRACE_COLUMNS[0]] = json.dumps(train, separators=(",", ":"))
        selected[TRACE_COLUMNS[1]] = model_sha
        dates = targets[[*WEEK_KEYS, "entry_date", "exit_date"]].copy()
        dates["entry_date"] = pd.to_datetime(dates.entry_date)
        dates["exit_date"] = pd.to_datetime(dates.exit_date)
        allowed = selected[WEEK_KEYS].merge(dates, on=WEEK_KEYS, how="left", validate="one_to_one")
        if allowed[["entry_date", "exit_date"]].isna().any().any():
            raise ValueError("OOF预测包含目标周外的股票或缺失日历")
        indices = minutes[[*WEEK_KEYS, "datetime"]].assign(_cohort_position=np.arange(len(minutes))).merge(
            allowed, on=WEEK_KEYS, how="inner", validate="many_to_one")
        day = indices.datetime.dt.normalize()
        indices = indices.loc[day.gt(indices.entry_date) & day.lt(indices.exit_date)]
        target_minutes = minutes.iloc[indices._cohort_position.to_numpy()].copy()
        enriched = attach_weekly_context(target_minutes, selected, weekly, effective_cfg)
        enriched["cohort_target_year"] = target_year
        parts.append(enriched)
        predictions.to_csv(model_dir / "oof_weekly_predictions.csv", index=False, encoding="utf-8-sig")
        selected.to_csv(model_dir / "oof_weekly_selections.csv", index=False, encoding="utf-8-sig")
        selector = {"target_year": target_year, "source_model_training_years": train,
                    "excluded_years": selector_excluded, "source_model_sha256": model_sha,
                    "source_model_path": str(model_path.resolve()), "predicted_rows": len(predictions),
                    "selected_stockweeks": len(selected), "selected_weeks": int(selected.trade_week.nunique()),
                    "cohort_rows_before_t_label_purge": len(enriched),
                    "selected_source_sha256": calculate_cohort_file_sha256(model_dir / "oof_weekly_selections.csv")}
        selectors.append(selector)
    training = pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
    if training.duplicated(["datetime", "symbol"]).any():
        raise AssertionError("OOF核心队列重复分钟行")
    input_columns = [*WEEK_KEYS, "feature_week", "signal_datetime", "feature_source_start", "entry_date", "exit_date"]
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(weekly[input_columns], index=False).to_numpy().tobytes()).hexdigest()
    trace_keys = ["datetime", "symbol", "feature_source_start", *CONTEXT_FEATURES.values(), *TRACE_COLUMNS]
    metadata = {"core_years": core, "excluded_years": excluded, "available_years": available,
                "selector_count": len(selectors), "selectors": selectors,
                "training_rows_before_t_label_purge": len(training), "weekly_context_features": list(CONTEXT_FEATURES.values()),
                "weekly_source_keys_sha256": input_hash,
                "cohort_context_sha256": hashlib.sha256(pd.util.hash_pandas_object(training[trace_keys], index=False).to_numpy().tobytes()).hexdigest(),
                "source_sha256": {Path(__file__).name: calculate_cohort_file_sha256(Path(__file__))},
                "resolved_config": effective_cfg,
                "source_rule": "min(旧分钟feature_source_start,同股周真实feature_source_start)，新增周输入覆盖前4实际观测周",
                "label_policy": "未来股票标签/恢复缺失不删队列；调用方接各周期标签后按扩展来源严格T训练purge",
                "selection_policy": "仅同折核心年互为OOF；当前特征周完整、entry/exit同目标年且entry<exit、selected及周中交易日",
                "limitation": "每个selector仅用另一个核心年；较早核心年由未来核心年预测时仍属于回顾性OOF，不是向前验证"}
    training[trace_keys + ["cohort_target_year"]].to_csv(directory / "cohort_lineage.csv", index=False, encoding="utf-8-sig")
    write_model_json(directory / "cohort_metadata.json", metadata)
    return training, metadata
