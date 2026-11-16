"""复用完整独审绑定的永久底仓核心机会；不规划账户、不拟合、不连接未来标签。

调用方须先在完整市场截面完成原 session/provenance/timing 增强。本模块读取
v14 完整 lineage、基准日库存及旧训练矩阵，只向新的输出目录写复用证据。
旧矩阵仅作当前字段反证核对，绝不成为机会筛选条件；两 core 年仅可联合拟合。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["datetime", "symbol"]
LINEAGE_COLUMNS = [
    *KEYS, "feature_source_start", "intraday_feature_source_start",
    "entry_feature_source_start", "entry_rank_source_start", "portfolio_cash_source_start",
    "base_entry_datetime", "baseline_opening_shares", "frozen_base_shares",
    "source_model_training_years", "source_model_sha256", "cohort_target_year",
]
TIME_COLUMNS = ["datetime", "feature_source_start", "intraday_feature_source_start",
                "entry_feature_source_start", "entry_rank_source_start",
                "portfolio_cash_source_start", "base_entry_datetime"]
FUTURE_PREFIXES = ("execution_", "recovery_", "target_", "label_")
OLD_CONTEXT = ["f_weekly_pred_return_rank", "f_weekly_pred_amplitude_rank",
               "f_weekly_score", "f_weekly_selection_rank"]
# 这些已解析配置决定原自动底仓或当前特征；新标签/模型/退出参数不在本合同内。
BASE_KEYS = ("initial_cash", "base_fraction", "target_position_notional", "min_position_notional",
             "max_position_notional", "lot_size", "commission_rate", "min_commission", "transfer_rate",
             "stamp_rate_before_20230828", "stamp_rate_from_20230828", "slippage_rate",
             "max_bar_amount_fraction", "flat_st_limit_rate", "nonflat_limit_proxy")
CONTEXT_CONFIG_KEYS = (*BASE_KEYS, "source_training_report_dir", "start_date", "end_date", "embargo_days",
                       "signal_times", "min_history_days", "min_avg_daily_amount", "min_price",
                       "top_n", "weekly_top_n", "weekly_return_weight", "weekly_amplitude_weight",
                       "add_session_context", "use_weekly_execution_features",
                       "min_prior_close_auction_amount", "strict_feature_provenance")


def calculate_frozen_file_sha256(path: Path) -> str:
    """流式绑定已审计的只读文件，不把大模型矩阵字节一次放入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_frozen_frame_sha256(frame: pd.DataFrame) -> str:
    """与原 lineage metadata 相同的真实列序/dtype/行序摘要。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def require_frozen_equal(actual: pd.DataFrame, expected: pd.DataFrame, description: str) -> None:
    """无容差核对 dtype 和值；索引已按原键对齐，保留 NaT/NaN 相等语义。"""
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_exact=True, check_dtype=True)
    except AssertionError as error:
        raise ValueError(f"冻结核心 {description} 不一致：{error}") from error


def load_frozen_inventory_core_cohort(intraday: pd.DataFrame, fold: dict, cfg: dict,
                                     source_report: Path, out: Path, inputs: dict,
                                     record_input) -> tuple[pd.DataFrame, dict]:
    """恢复本折全部未联标签 core 机会；record_input 接受(path, inputs)。

source_report 为 v14 完整报告。可选 cfg.source_learning_audit 指定该报告内完整
verification.json；否则使用最新完整通过记录。若给 source_learning_audit_sha256，
额外严格核其身份。cfg 必须继承原解析的底仓/当前特征参数；人工指令不参与训练。
out/frozen_inventory_cohort 必须不存在。输入必须完整增强、键唯一、来源为 datetime64；
丢原键、来源改写、旧矩阵当前值/dtype 改写或不完整独审都会拒绝。
    """
    source = Path(source_report).resolve()
    destination = Path(out) / "frozen_inventory_cohort"
    if destination.exists():
        raise FileExistsError(f"不覆盖已有复用证据：{destination}")
    core_years = sorted(map(int, fold["core_years"]))
    excluded = sorted([int(fold["test_year"]), int(fold["inner_year"])])
    if len(set(core_years)) != 2 or len(set(excluded)) != 2 or set(core_years) & set(excluded):
        raise ValueError("要求两个互异 core 年及互异 inner/outer")
    if not {*KEYS, "feature_source_start"}.issubset(intraday):
        raise ValueError("完整当前表缺键或真实来源")
    if intraday[KEYS].isna().any().any() or intraday.duplicated(KEYS).any():
        raise ValueError("完整当前表键缺失或重复")
    for name in ("datetime", "feature_source_start"):
        if intraday[name].dtype != np.dtype("datetime64[ns]"):
            raise ValueError(f"当前 {name} 必须保持原 datetime64[ns] dtype")
    if not intraday.symbol.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("当前 symbol 必须为非空原字符串")

    explicit = cfg.get("source_learning_audit")
    candidates = [Path(explicit).resolve()] if explicit else sorted(source.glob("independent_learning_audit/run_*/verification.json"))
    successful = []
    for path in candidates:
        if not path.is_relative_to(source):
            raise ValueError("来源独审必须位于 source_report 内")
        value = json.loads(path.read_text())
        if all(value.get(key) is True for key in ("passed", "complete", "all_read_inputs_unchanged")):
            successful.append((path, value))
    if not successful:
        raise ValueError("来源完整独立审计未通过或输入已改变")
    audit_path, audit = successful[-1]
    if audit.get("research_version") != "inventory_learning_t_v14":
        raise ValueError("要求原永久底仓 v14 完整独审")
    verified = {str(audit_path): calculate_frozen_file_sha256(audit_path)}
    if cfg.get("source_learning_audit_sha256", verified[str(audit_path)]) != verified[str(audit_path)]:
        raise ValueError("指定独审 SHA 不符")
    record_input(audit_path, inputs)

    def read_path(path: Path) -> Path:
        """任何使用的来源文件先与完整独审字节 SHA 比较，再记录到新输入清单。"""
        path = Path(path).resolve()
        expected = audit["inputs_sha256"].get(str(path))
        if expected is None:
            raise ValueError(f"完整独审未绑定所需文件：{path}")
        actual = calculate_frozen_file_sha256(path)
        if actual != expected:
            raise ValueError(f"已审计来源文件 SHA 改变：{path}")
        verified[str(path)] = actual
        record_input(path, inputs)
        return path

    def read_json(path: Path) -> dict:
        """读取 SHA 已核的结构化来源。"""
        return json.loads(read_path(path).read_text())

    original_cfg = read_json(source / "resolved_config.json")
    for name in CONTEXT_CONFIG_KEYS:
        if name not in original_cfg or name not in cfg or cfg[name] != original_cfg[name]:
            raise ValueError(f"原底仓/当前特征配置改变或缺失：{name}")
    integrity = read_json(source / "runtime_integrity_audit.json")
    if integrity.get("all_unchanged") is not True:
        raise ValueError("来源运行完整性核对未完成")
    source_hashes = read_json(source / "source_hashes.json")
    for name, expected in source_hashes.items():
        snapshot = read_path(source / "source_snapshot" / name)
        if verified[str(snapshot)] != expected:
            raise ValueError("来源冻结快照与自身 manifest 不符")
    year_directory = source / f"year_{fold['test_year']}"
    split = read_json(year_directory / "split.json")
    for name in ("test_year", "inner_year", "core_years"):
        if split[name] != fold[name]:
            raise ValueError(f"本折身份不符：{name}")
    folder = year_directory / "training_cohort" / "inventory_learning_cohort"
    original_meta = read_json(folder / "cohort_metadata.json")
    if (original_meta["core_years"] != core_years or original_meta["excluded_years"] != excluded
            or original_meta["embargo_days"] != cfg["embargo_days"]):
        raise ValueError("原未联标签队列年份/隔离合同不符")
    if (any(original_meta.get(name) is not False for name in
            ("labels_attached", "future_fill_or_return_filter", "core_sampling_performed"))
            or any(original_meta.get(name) is not True for name in
                   ("baseline_has_no_t_and_no_manual_buys", "full_market_features_before_selection",
                    "all_core_years_joint_fit_only_not_independent_calibration"))):
        raise ValueError("来源不是完整无标签/无未来筛选/无人工核心队列")
    base_sha = hashlib.sha256(json.dumps({key: cfg[key] for key in BASE_KEYS}, sort_keys=True).encode()).hexdigest()
    omitted = {name for name in intraday if name.startswith(FUTURE_PREFIXES)} | set(OLD_CONTEXT) | {"trade_week_complete", "horizon_minutes"}
    current_columns = [name for name in intraday if name not in omitted]
    market_index = pd.MultiIndex.from_frame(intraday[KEYS])
    market_days = pd.MultiIndex.from_arrays([intraday.datetime.dt.normalize(), intraday.symbol])
    parts, summaries = [], []
    for year in core_years:
        directory = folder / f"core_{year}"
        meta = read_json(directory / "metadata.json")
        checked = [item for item in audit["core_cohorts"]
                   if item["fold"] == fold["test_year"] and item["core_year"] == year]
        if (len(checked) != 1 or checked[0].get("all_ranking_and_cash_sources_propagated") is not True
                or checked[0]["rows"] != meta["training_opportunity_rows"] or meta["base_config_sha256"] != base_sha
                or meta["selector_training_years"] != [value for value in core_years if value != year]
                or meta != next(item for item in original_meta["selectors"] if item["target_year"] == year)):
            raise ValueError("本 core 年的来源摘要/底仓/selector 与完整独审不符")
        lineage = pd.read_csv(read_path(directory / "cohort_lineage.csv"), float_precision="round_trip")
        if list(lineage) != LINEAGE_COLUMNS:
            raise ValueError("完整 lineage 字段或列顺序改变")
        for name in TIME_COLUMNS:
            lineage[name] = pd.to_datetime(lineage[name], format="mixed")
        if (lineage.duplicated(KEYS).any() or lineage[KEYS].isna().any().any()
                or not lineage.cohort_target_year.eq(year).all() or not lineage.datetime.dt.year.eq(year).all()
                or not lineage.baseline_opening_shares.gt(0).all()
                or len(lineage) != meta["training_opportunity_rows"]
                or calculate_frozen_frame_sha256(lineage) != meta["cohort_lineage_sha256"]):
            raise ValueError("完整 lineage 键/旧股/dtype/原摘要不符")
        positions = market_index.get_indexer(pd.MultiIndex.from_frame(lineage[KEYS]))
        if (positions < 0).any():
            raise ValueError("完整当前行情缺少冻结机会键；不能按旧标签取交集")
        current = intraday.iloc[positions][current_columns].copy().reset_index(drop=True)
        require_frozen_equal(current[["feature_source_start"]].rename(columns={"feature_source_start": "intraday_feature_source_start"}),
                             lineage[["intraday_feature_source_start"]], "当前原始来源")
        # 原无 T 基准旧股股日 × 原当前表必须恰等于全部 lineage，额外/缺少机会也拒绝。
        inventory = pd.read_csv(read_path(directory / "daily_inventory_audit.csv"), float_precision="round_trip")
        inventory["date"] = pd.to_datetime(inventory.date, format="mixed")
        old = inventory.loc[inventory.baseline_opening_shares.gt(0), ["date", "symbol", "baseline_opening_shares"]]
        if old.duplicated(["date", "symbol"]).any():
            raise ValueError("原基准旧股股日重复")
        old_index = pd.MultiIndex.from_frame(old[["date", "symbol"]])
        old_locations = old_index.get_indexer(market_days)
        full_keys = intraday.loc[old_locations >= 0, KEYS].sort_values(KEYS, kind="stable")
        require_frozen_equal(full_keys, lineage[KEYS], "全部旧股机会全集")
        located = old_index.get_indexer(pd.MultiIndex.from_arrays([lineage.datetime.dt.normalize(), lineage.symbol]))
        if (located < 0).any() or not np.array_equal(old.baseline_opening_shares.to_numpy()[located], lineage.baseline_opening_shares):
            raise ValueError("冻结旧股股数与基准日初不符")
        current["date"] = current.datetime.dt.normalize()
        # 保持旧 build 函数追加字段的精确顺序，而不以13列CSV顺序重排原frame。
        append_order = ["baseline_opening_shares", "intraday_feature_source_start", "entry_feature_source_start",
                        "entry_rank_source_start", "source_model_training_years", "source_model_sha256",
                        "base_entry_datetime", "frozen_base_shares", "portfolio_cash_source_start",
                        "feature_source_start", "cohort_target_year"]
        for name in append_order:
            current[name] = lineage[name].to_numpy()
        require_frozen_equal(current[LINEAGE_COLUMNS], lineage, "全部来源及账本")
        parts.append(current)
        summaries.append({"core_year": year, "rows": len(current), "cohort_lineage_sha256": meta["cohort_lineage_sha256"],
                          "eligible_inventory_stock_days": len(old), "missing_current_provenance_rows": int(current.intraday_feature_source_start.isna().sum())})
    core = pd.concat(parts, ignore_index=True).sort_values(KEYS, kind="stable").reset_index(drop=True)
    if len(core) != original_meta["training_opportunity_rows"] or core.duplicated(KEYS).any():
        raise ValueError("两年完整机会数或键不符")
    # 仅用实际拟合矩阵反证当前features/来源；缺原label的行始终保留在core。
    index = pd.MultiIndex.from_frame(core[KEYS])
    matrix_checks, matrix_hashes_checked = [], set()
    for horizon in original_cfg["horizons"]:
        for profile in original_cfg["learning_model_profiles"]:
            directory = year_directory / "models" / f"h{horizon}_{profile}"
            meta = read_json(directory / "model_metadata.json")
            reference = str(directory.relative_to(source))
            checked = [item for item in audit["models"] if item["path"] == reference]
            if (len(checked) != 1 or checked[0].get("passed") is not True
                    or checked[0].get("independently_rebuilt_training_inputs") is not True
                    or meta["train_years"] != core_years or meta["excluded_years"] != excluded):
                raise ValueError("旧矩阵必须有完整独立模型审计和同一core身份")
            path = read_path(directory / "training_inputs.pkl")
            matrix_sha = verified[str(path)]
            if matrix_sha != meta["training_inputs_file_sha256"]:
                raise ValueError("旧矩阵文件与模型metadata不符")
            if matrix_sha not in matrix_hashes_checked:
                matrix = pd.read_pickle(path)
                allowed = {*KEYS, "feature_source_start", "cohort_target_year", "baseline_opening_shares", *meta["features"]}
                if set(matrix) != allowed or matrix.duplicated(KEYS).any() or not all(name.startswith("f_") for name in meta["features"]):
                    raise ValueError("旧矩阵当前字段schema不符")
                locations = index.get_indexer(pd.MultiIndex.from_frame(matrix[KEYS]))
                if (locations < 0).any():
                    raise ValueError("重建完整机会漏掉原拟合键")
                actual = core.iloc[locations][list(matrix)].copy()
                actual[meta["features"]] = actual[meta["features"]].replace([np.inf, -np.inf], np.nan)
                require_frozen_equal(actual, matrix, "旧矩阵当前值/dtype反证")
                matrix_hashes_checked.add(matrix_sha)
                matrix_checks.append({"model": reference, "rows": len(matrix), "features": meta["features"],
                                      "matrix_sha256": matrix_sha, "exact_current_values_and_dtypes": True})
    for name, expected in verified.items():
        if calculate_frozen_file_sha256(Path(name)) != expected:
            raise ValueError(f"复用读取期间来源改变：{name}")
    metadata = {"core_years": core_years, "excluded_years": excluded, "source_report": str(source),
                "source_learning_audit": str(audit_path), "source_learning_audit_sha256": verified[str(audit_path)],
                "training_opportunity_rows": len(core), "core_years_detail": summaries,
                "columns": list(core), "dtypes": core.dtypes.astype(str).to_dict(),
                "full_current_core_sha256": calculate_frozen_frame_sha256(core),
                "cohort_lineage_sha256": calculate_frozen_frame_sha256(core[LINEAGE_COLUMNS]),
                "old_training_matrix_checks": matrix_checks, "labels_attached": False,
                "old_training_matrix_used_as_opportunity_filter": False, "planner_or_model_run": False,
                "all_core_years_joint_fit_only_not_independent_calibration": True,
                "manual_buys_ignored_for_training": bool(cfg.get("manual_buys_csv")),
                "full_market_enhancement_required_before_call": True,
                "value_identity_scope": "原完整lineage及现有实际矩阵精确核对；新全current摘要另存，不虚称原lineage保存全部因子数值",
                "omitted_columns": [name for name in intraday if name in omitted],
                "all_read_inputs_unchanged": True, "inputs_sha256": verified}
    destination.mkdir(parents=True, exist_ok=False)
    core[LINEAGE_COLUMNS].to_csv(destination / "cohort_lineage.csv", index=False, encoding="utf-8-sig")
    (destination / "reuse_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return core, metadata
