"""从已审计周 OOF 名单建立永久底仓的纯训练机会，不训练模型或连接未来标签。

调用方提供完整已传播 provenance 的 universe/daily/当前分钟截面及 v8 来源目录。
每个 core 年从干净现金规划无 T 底仓；先隔离入场来源，再取全部开盘已有旧股的
信号。只向新的输出子目录写计划和溯源，不修改原模型、行情、缓存或执行引擎。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_panel import write_json
from .dynamic_t_engine import plan_dynamic_entries
from .weekly_t_cohort import CONTEXT_FEATURES, select_cohort_target_weeks
from .weekly_t_timing_data import frame_sha256

WEEK_KEYS = ["trade_week", "symbol"]
SOURCE_KEYS = WEEK_KEYS + ["feature_week", "signal_datetime", "feature_source_start", "entry_date", "exit_date"]
FUTURE_PREFIXES = ("execution_", "recovery_", "target_", "label_")


def calculate_learning_file_sha256(path: Path) -> str:
    """以流式 SHA 绑定只读输入，避免一次载入模型或大 CSV 字节。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_clean_entry_requests(requests: pd.DataFrame, core_years: list[int],
                                excluded_years: list[int], embargo_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅按已知来源区间筛入场；缺来源拒绝，失败请求保留原因，不读未来行情。"""
    audit = requests.copy()
    start = pd.to_datetime(audit.entry_rank_source_start, errors="coerce")
    signal = pd.to_datetime(audit.signal_datetime, errors="coerce")
    execution = pd.to_datetime(audit.entry_date, errors="coerce") + pd.Timedelta(hours=9, minutes=30)
    invalid = start.isna() | signal.isna() | execution.isna() | ~start.le(signal) | ~signal.lt(execution)
    wrong_year = ~start.dt.year.isin(core_years) | ~signal.dt.year.isin(core_years) | ~execution.dt.year.isin(core_years)
    overlap = pd.Series(False, index=audit.index)
    for year in excluded_years:
        left = pd.Timestamp(year, 1, 1) - pd.Timedelta(days=embargo_days)
        right = pd.Timestamp(year + 1, 1, 1) + pd.Timedelta(days=embargo_days)
        overlap |= start.lt(right) & execution.ge(left)
    audit["entry_source_invalid"] = invalid
    audit["entry_source_outside_core"] = wrong_year
    audit["entry_source_excluded_or_embargo"] = overlap
    audit["entry_source_eligible"] = ~(invalid | wrong_year | overlap)
    return audit.loc[audit.entry_source_eligible].copy(), audit


def load_learning_selector_requests(universe: pd.DataFrame, fold: dict, target_year: int,
                                   cfg: dict, inputs: dict, record_input, source_audit: dict,
                                   cohort_metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """核对本折已审计 selector、全部排名供体键和来源，再补原始已知入场字段。

预测 CSV 只读取键；它的完整目标键必须精确等于原 universe 的目标年集合。
selected CSV/bundle/native 另以旧独立审计摘要核对，训练/排除年不得借用其它折。
    """
    root = Path(cfg["source_training_report_dir"])
    directory = root / f"year_{fold['test_year']}" / "oof_weekly_cohort" / f"selector_{target_year}"
    key = str(directory.relative_to(root))
    checked = source_audit["selectors"][key]
    core = sorted(map(int, fold["core_years"]))
    train = [year for year in core if year != target_year]
    excluded = sorted(set(cohort_metadata["available_years"]) - set(train))
    metadata_path = directory / "model_metadata.json"
    record_input(metadata_path, inputs)
    model = json.loads(metadata_path.read_text())
    if (checked.get("passed") is not True or model["train_years"] != train
            or sorted(model["excluded_years"]) != excluded or checked["train_years"] != train
            or sorted(checked["excluded_years"]) != excluded
            or model["training_sample_sha256"] != checked["sample_sha256"]
            or model["feature_source_sha256"] != checked["source_sha256"]):
        raise ValueError("周 OOF 模型身份或训练/排除年份与已审计本折不一致")
    expected = {"model_bundle.joblib": checked["model_bundle_sha256"],
                "oof_weekly_selections.csv": checked["selected_csv_sha256"]}
    expected.update({f"{head}_model.txt": item["native_sha256"]
                     for head, item in checked["native_checks"].items()})
    for name, digest in expected.items():
        path = directory / name
        record_input(path, inputs)
        if calculate_learning_file_sha256(path) != digest:
            raise ValueError(f"已审计 OOF 文件被改变：{path}")
    selected = pd.read_csv(directory / "oof_weekly_selections.csv", float_precision="round_trip")
    for name in ("trade_week", "feature_week", "signal_datetime"):
        selected[name] = pd.to_datetime(selected[name], format="mixed")
    if (selected.duplicated(WEEK_KEYS).any() or selected[WEEK_KEYS].isna().any().any()
            or not selected.source_model_sha256.eq(expected["model_bundle.joblib"]).all()
            or not selected.source_model_training_years.eq(json.dumps(train, separators=(",", ":"))).all()
            or not selected.trade_week.dt.year.eq(target_year).all()
            or not (selected["rank"].gt(0) & selected["rank"].mod(1).eq(0)).all()):
        raise ValueError("周名单键、排名或模型 lineage 非法")
    prediction_path = directory / "oof_weekly_predictions.csv"
    record_input(prediction_path, inputs)
    donor_keys = pd.read_csv(prediction_path, usecols=WEEK_KEYS)
    donor_keys.trade_week = pd.to_datetime(donor_keys.trade_week, format="mixed")
    targets = select_cohort_target_weeks(universe, target_year)
    order = lambda frame: frame[WEEK_KEYS].sort_values(WEEK_KEYS, kind="stable").reset_index(drop=True)
    if donor_keys.duplicated(WEEK_KEYS).any():
        raise ValueError("完整排名供体键重复")
    pd.testing.assert_frame_equal(order(donor_keys), order(targets), check_dtype=False, check_exact=True)
    donors = donor_keys.merge(targets[SOURCE_KEYS], on=WEEK_KEYS, validate="one_to_one")
    if len(donors) != checked["target_rows"]:
        raise ValueError("完整排名供体数与原独立审计不一致")
    # 同周被排名但未入选的股票也会影响 rank/score，不能只传播成功持股来源。
    grouped = donors.groupby("trade_week", sort=False)
    starts = grouped.feature_source_start.min()
    missing = grouped.feature_source_start.apply(lambda column: column.isna().any())
    starts.loc[missing] = pd.NaT
    extra = ["f_last_close"]
    if "observed_last_datetime" in universe:
        extra.append("observed_last_datetime")
    joined = selected.merge(universe[SOURCE_KEYS + extra], on=WEEK_KEYS, how="left",
                            suffixes=("", "_original"), validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("OOF 入选对象缺少原 universe 行")
    for name in ("feature_week", "signal_datetime"):
        if not joined[name].eq(joined[f"{name}_original"]).all():
            raise ValueError("OOF 信号时间或观测周与原 universe 不一致")
    price = joined.f_last_close
    if not (np.isfinite(price) & price.gt(0)).all():
        raise ValueError("入场计划的已知前收必须有限且为正")
    if "observed_last_datetime" in joined and not joined.observed_last_datetime.le(joined.signal_datetime).all():
        raise ValueError("入场前收来源晚于原周信号")
    joined["universe_previous_close"] = price
    joined["entry_feature_source_start"] = joined.feature_source_start
    joined["entry_rank_source_start"] = joined.trade_week.map(starts)
    joined["entry_rank_donor_count"] = joined.trade_week.map(grouped.size()).astype(int)
    joined = joined.drop(columns=["_merge", "feature_week_original", "signal_datetime_original"])
    proof = {"target_year": target_year, "selector_training_years": train, "selector_excluded_years": excluded,
             "selector_path": str(directory.resolve()), "model_bundle_sha256": expected["model_bundle.joblib"],
             "selected_csv_sha256": expected["oof_weekly_selections.csv"],
             "training_sample_sha256": model["training_sample_sha256"],
             "feature_source_sha256": model["feature_source_sha256"], "donor_rows": len(donors),
             "donor_keys_and_sources_sha256": frame_sha256(donors[SOURCE_KEYS]),
             "ranking_source_rule": "min(feature_source_start of all original same-week ranking donors)",
             "all_original_donors_included": True, "new_model_training_performed": False}
    return joined, donors, proof


def build_inventory_core_cohort(universe: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame,
                               fold: dict, cfg: dict, out: Path, inputs: dict, record_input,
                               recovery_store=None) -> tuple[pd.DataFrame, dict]:
    """建立两个 core 年永久底仓当前队列及 metadata；不按未来标签/恢复情况筛行。

universe/daily/intraday 必须在调用前完成全截面特征及 provenance；record_input
接受(path, inputs)。训练禁用人工加仓，不继承评价账户现金。输出在 out 下新增
inventory_learning_cohort/；不存在新取样/拟合参数，不使用 smoke_days 截训练期。
    """
    core = sorted(map(int, fold["core_years"]))
    excluded = sorted([int(fold["inner_year"]), int(fold["test_year"])])
    embargo = cfg.get("embargo_days", 7)
    if (len(core) != 2 or len(set(core)) != 2 or set(core) & set(excluded)
            or isinstance(embargo, bool) or not isinstance(embargo, (int, np.integer)) or embargo < 0):
        raise ValueError("要求两个互异 core 年、独立 inner/outer 及非负整数 embargo_days")
    if not set(SOURCE_KEYS + ["f_last_close", "feature_week_complete"]).issubset(universe):
        raise ValueError("完整 universe 缺少真实来源、日期或已知前收")
    if universe.duplicated(WEEK_KEYS).any() or universe[WEEK_KEYS].isna().any().any():
        raise ValueError("完整 universe 股周键缺失或重复")
    if not {"datetime", "symbol", "feature_source_start"}.issubset(intraday):
        raise ValueError("当前分钟缺少时间/股票/真实特征来源")
    if intraday.duplicated(["datetime", "symbol"]).any() or intraday[["datetime", "symbol"]].isna().any().any():
        raise ValueError("当前分钟键缺失或重复")
    root = Path(cfg["source_training_report_dir"])
    audit_path = root / "independent_residual_audit.json"
    metadata_path = root / f"year_{fold['test_year']}" / "oof_weekly_cohort" / "cohort_metadata.json"
    for path in (audit_path, metadata_path):
        record_input(path, inputs)
    source_audit = json.loads(audit_path.read_text())
    original = json.loads(metadata_path.read_text())
    if not all(source_audit.get(name) is True for name in ("passed", "complete", "full_configured_scope_complete")):
        raise ValueError("原 OOF 独立模型审计未完整通过")
    if original["core_years"] != core or original["excluded_years"] != excluded:
        raise ValueError("当前 fold 与原 OOF 年份合同不一致")
    if frame_sha256(universe[SOURCE_KEYS]) != original["weekly_source_keys_sha256"]:
        raise ValueError("完整原 universe 来源键 SHA 改变；不能先裁剪股票/年份")
    directory = Path(out) / "inventory_learning_cohort"
    directory.mkdir(parents=True, exist_ok=False)
    parts, summaries = [], []
    future_columns = [name for name in intraday if name.startswith(FUTURE_PREFIXES)]
    # 旧表的trade_week_complete仅用于旧周退出标签；永久持仓不应因此丢失末周。
    omitted = set(future_columns) | set(CONTEXT_FEATURES.values()) | {"trade_week_complete"}
    # 仅复制实际持仓分钟；完整大截面不在每个 fold 再复制一遍。
    minute_dates = pd.to_datetime(intraday.datetime).dt.normalize()
    for target in core:
        destination = directory / f"core_{target}"
        destination.mkdir()
        requests, donors, proof = load_learning_selector_requests(
            universe, fold, target, cfg, inputs, record_input, source_audit, original)
        requests, entry_audit = select_clean_entry_requests(requests, core, excluded, int(embargo))
        entry_audit.to_csv(destination / "entry_source_audit.csv", index=False, encoding="utf-8-sig")
        donors.to_csv(destination / "ranking_donor_sources.csv", index=False, encoding="utf-8-sig")
        year_daily = daily.loc[pd.to_datetime(daily.date).dt.year.eq(target)].copy()
        if year_daily.empty:
            raise ValueError(f"core {target} 缺少全市场真实日历")
        local_cfg = {**cfg, "manual_buys_csv": None, "dynamic_exit_mode": "fixed", "report_scope": "core_training"}
        planned = plan_dynamic_entries(requests, year_daily, local_cfg, recovery_bars=recovery_store)
        frozen, baseline = planned["frozen_selections"], planned["baseline_result"]
        events, audit = baseline["baseline_events"], baseline["daily_inventory_audit"]
        for name, frame in {"frozen_base_selections": frozen, "base_entries": events,
                            "base_planning_attempts": planned["planning_attempts"],
                            "daily_inventory_audit": audit, "planning_equity_curve": baseline["equity_curve"],
                            "position_amount_audit": baseline["position_amount_audit"]}.items():
            frame.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig")
        old = audit.loc[audit.baseline_opening_shares.gt(0), ["date", "symbol", "baseline_opening_shares"]].copy()
        index = pd.MultiIndex.from_frame(old[["date", "symbol"]])
        locations = index.get_indexer(pd.MultiIndex.from_arrays([minute_dates, intraday.symbol]))
        current = intraday.loc[locations >= 0, [name for name in intraday if name not in omitted]].copy()
        current["date"] = pd.to_datetime(current.datetime).dt.normalize()
        current["baseline_opening_shares"] = old.baseline_opening_shares.to_numpy()[locations[locations >= 0]]
        own = pd.to_datetime(current.feature_source_start, errors="coerce")
        current["intraday_feature_source_start"] = own
        first = frozen.sort_values("_entry_time", kind="stable").drop_duplicates("symbol")
        for name in ["entry_feature_source_start", "entry_rank_source_start", "source_model_training_years",
                     "source_model_sha256", "_entry_time", "frozen_base_shares"]:
            target_name = "base_entry_datetime" if name == "_entry_time" else name
            current[target_name] = current.symbol.map(first.set_index("symbol")[name])
        # 所有已处理请求都可能改变现金竞争，即使其实际成交失败；不能只取本股。
        clock_sources = requests.assign(_clock=requests.entry_date + pd.Timedelta(hours=9, minutes=30))
        clock_sources = clock_sources.groupby("_clock").entry_rank_source_start.min().sort_index().cummin()
        clock_index = np.searchsorted(clock_sources.index.to_numpy(), current.datetime.to_numpy(), side="right") - 1
        if len(current) and (clock_index < 0).any():
            raise AssertionError("持有期信号早于任何已知基础决策")
        current["portfolio_cash_source_start"] = clock_sources.to_numpy()[clock_index] if len(current) else pd.Series(dtype="datetime64[ns]")
        current["feature_source_start"] = np.minimum(np.minimum(own.to_numpy(), current.entry_rank_source_start.to_numpy()),
                                                     current.portfolio_cash_source_start.to_numpy())
        current["cohort_target_year"] = target
        current = current.sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
        trace = ["datetime", "symbol", "feature_source_start", "intraday_feature_source_start",
                 "entry_feature_source_start", "entry_rank_source_start", "portfolio_cash_source_start",
                 "base_entry_datetime", "baseline_opening_shares", "frozen_base_shares",
                 "source_model_training_years", "source_model_sha256", "cohort_target_year"]
        current[trace].to_csv(destination / "cohort_lineage.csv", index=False, encoding="utf-8-sig")
        summary = {**proof, "original_selected_stockweeks": len(entry_audit), "source_eligible_requests": len(requests),
                   "source_rejected_requests": int((~entry_audit.entry_source_eligible).sum()),
                   "actual_base_entry_count": len(events), "held_symbols": len(first), "market_days": len(planned["market_dates"]),
                   "eligible_inventory_stock_days": len(old), "training_opportunity_rows": len(current),
                   "missing_current_provenance_rows": int(own.isna().sum()),
                   "cohort_lineage_sha256": frame_sha256(current[trace]),
                   "base_config_sha256": planned["base_config_sha256"], "initial_cash": float(cfg["initial_cash"]),
                   "manual_input_configured_but_disabled_for_training": bool(cfg.get("manual_buys_csv")),
                   "pending_pairs": int(baseline["metrics"]["pending_pairs"]),
                   "no_terminal_liquidation": bool(baseline["metrics"]["no_terminal_liquidation"])}
        write_json(destination / "metadata.json", summary)
        summaries.append(summary)
        parts.append(current)
    cohort = pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
    if cohort.duplicated(["datetime", "symbol"]).any():
        raise AssertionError("两个 core 年产生重复信号键")
    metadata = {"core_years": core, "excluded_years": excluded, "embargo_days": int(embargo),
                "source_audit_sha256": calculate_learning_file_sha256(audit_path), "selectors": summaries,
                "training_opportunity_rows": len(cohort), "future_columns_removed": future_columns,
                "old_weekly_label_eligibility_omitted": ["trade_week_complete"] if "trade_week_complete" in intraday else [],
                "old_weekly_context_features_omitted": list(CONTEXT_FEATURES.values()),
                "labels_attached": False, "future_fill_or_return_filter": False, "core_sampling_performed": False,
                "all_core_years_joint_fit_only_not_independent_calibration": True,
                "daily_old_inventory_gate": "baseline_opening_shares > 0; all original signal clocks retained",
                "source_rule": "min(current intraday, entry full ranking donors, all prior portfolio entry requests)",
                "baseline_has_no_t_and_no_manual_buys": True, "full_market_features_before_selection": True,
                "output_directory": str(directory.resolve())}
    write_json(directory / "cohort_metadata.json", metadata)
    return cohort, metadata
