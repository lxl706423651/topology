"""等金额固定底仓研究的数据复用层；只读已审计缓存、周模型及年度拆分。

不重建全市场因子、不训练旧T模型，也不按未来收益或退市筛选股票。完整市场
日历用于现金期及持有期估值；固定底仓一旦成功买入，就保留其之后全部信号日，
不再只取入选周中间日。输出身份清单由调用方写入新的研究报告目录。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .weekly_t_quote_store import calculate_file_sha256


INTRADAY_COLUMNS = [
    "datetime", "symbol", "date", "trade_week", "previous_close", "signal_price",
    "signal_high", "signal_low", "signal_amount", "signal_volume", "f_vwap_deviation",
    "f_return_5m", "f_return_15m", "f_open_return", "execution_time", "execution_price",
    "execution_high", "execution_low", "execution_amount", "execution_volume",
]


def record_balanced_input(path: Path | str, inputs: dict, expected: str | None = None) -> dict:
    """记录只读输入内容SHA；已有记录再次调用时验证当前字节没有变化。"""
    path = Path(path).resolve()
    digest = calculate_file_sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"输入偏离已审计来源：{path}")
    identity = {"sha256": digest, "size_bytes": path.stat().st_size}
    if str(path) in inputs and inputs[str(path)] != identity:
        raise ValueError(f"研究运行期间输入发生变化：{path}")
    inputs[str(path)] = identity
    return identity


def validate_balanced_sources(cfg: dict, inputs: dict, project_root: Path) -> dict:
    """核原模型/执行审计、缓存身份、原源码及原始CSV状态；不借旧收益选择新规则。"""
    origin = Path(cfg["source_audit_report_dir"])
    model_path = origin / "independent_timing_model_audit.json"
    execution_path = origin / "independent_timing_execution_audit_archive/verification.json"
    for path in (model_path, execution_path, origin / "input_sources.json", origin / "source_hashes.json"):
        record_balanced_input(path, inputs)
    model = json.loads(model_path.read_text())
    execution = json.loads(execution_path.read_text())
    known = json.loads((origin / "input_sources.json").read_text())
    source_cfg_path = (Path(cfg["source_training_report_dir"]) / "resolved_config.json").resolve()
    if str(source_cfg_path) not in known:
        raise ValueError("基础研究配置不在原已审计输入中")
    record_balanced_input(source_cfg_path, inputs, known[str(source_cfg_path)]["sha256"])
    if not model.get("passed") or not model.get("complete") or model.get("smoke"):
        raise ValueError("复用数据及周模型的独立审计必须完整通过")
    if not execution.get("passed") or not execution.get("full_default_scope"):
        raise ValueError("原始行情执行证据不完整")
    if calculate_file_sha256(model_path) != execution["outputs_sha256"]["independent_timing_model_audit.json"]:
        raise ValueError("模型审计身份与完整执行审计不一致")
    hashes = json.loads((origin / "source_hashes.json").read_text())
    for name, digest in hashes.items():
        record_balanced_input(project_root / name, inputs, digest)
        record_balanced_input(origin / "source_snapshot" / Path(name).name, inputs, digest)
    metadata_path = Path(cfg["market_cache_dir"]) / "metadata.json"
    record_balanced_input(metadata_path, inputs, model["cache_identity"]["weekly"]["manifest_sha256"])
    metadata = json.loads(metadata_path.read_text())
    for name in ("start_date", "end_date", "signal_times"):
        if metadata[name] != cfg[name]:
            raise ValueError(f"缓存与当前研究的{name}不同")
    raw_root = Path(cfg["data_root"]).resolve()
    if Path(metadata["data_root"]).resolve() != raw_root:
        raise ValueError("原始行情目录与已审计缓存不一致")
    for name, size, mtime in metadata["files"]:
        stat = (raw_root / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError(f"原始行情已变更，不能复用旧缓存：{name}")
    return {"passed": True, "old_runtime_sources_verified": len(hashes),
            "raw_file_stats_verified": len(metadata["files"]),
            "raw_stat_check_is_not_full_raw_sha_scan": True,
            "cache_fingerprint": metadata["fingerprint"], "old_t_models_reused": False,
            "weekly_ranking_reused": True, "position_and_holding_contract_changed_explicitly": True}


def load_balanced_market(cfg: dict, inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """读取原全市场daily/intraday PKL并逐字节核SHA；丢弃所有T标签及恢复端点列。

这里只复用个股当日量价特征，不拟合预处理、不重算截面排名。所有日期先保留；
具体年份/持股筛选由调用方在冻结周名单和无T底仓计划产生后完成。
    """
    origin = Path(cfg["source_audit_report_dir"])
    known = json.loads((origin / "input_sources.json").read_text())
    cache = Path(cfg["market_cache_dir"])
    tables = []
    for name in ("daily", "intraday"):
        path = (cache / f"{name}.pkl").resolve()
        if str(path) not in known:
            raise ValueError("市场PKL不在原运行已核验输入中")
        record_balanced_input(path, inputs, known[str(path)]["sha256"])
        frame = pd.read_pickle(path)
        if name == "intraday":
            missing = set(INTRADAY_COLUMNS).difference(frame)
            if missing:
                raise ValueError(f"原分钟缓存缺字段{sorted(missing)}")
            frame = frame[INTRADAY_COLUMNS].copy()
        tables.append(frame)
    daily, intraday = tables
    daily["date"] = pd.to_datetime(daily.date)
    intraday["datetime"] = pd.to_datetime(intraday.datetime)
    days = pd.DatetimeIndex(daily.date.unique()).sort_values()
    if daily.duplicated(["date", "symbol"]).any() or intraday.duplicated(["datetime", "symbol"]).any():
        raise ValueError("已审计市场键出现重复")
    metadata = json.loads((cache / "metadata.json").read_text())
    if len(days) != metadata["calendar"]["trading_days"]:
        raise ValueError("完整市场交易日日历与原清单不一致")
    return daily, intraday, days


def read_balanced_fold(cfg: dict, year: int, inputs: dict) -> dict:
    """保留既有年度core/inner/outer划分，不因任何新成绩更换留出年。"""
    path = Path(cfg["source_training_report_dir"]) / f"year_{year}/split.json"
    known = json.loads((Path(cfg["source_audit_report_dir"]) / "input_sources.json").read_text())
    if str(path.resolve()) not in known:
        raise ValueError("年度拆分不在原已审计输入身份中")
    record_balanced_input(path, inputs, known[str(path.resolve())]["sha256"])
    fold = json.loads(path.read_text())
    if (fold["test_year"] != year or year in fold["train_years"] or
            fold["inner_year"] in fold["core_years"] or
            sorted(fold["train_years"]) != sorted(fold["core_years"] + [fold["inner_year"]])):
        raise ValueError("年度隔离不完整")
    return {**fold, "t_model_fit": False, "t_rule_selection_year": fold["inner_year"],
            "base_model_protocol_unchanged": True}


def read_balanced_selections(cfg: dict, fold: dict, scope: str, days: pd.DatetimeIndex,
                             inputs: dict) -> pd.DataFrame:
    """读取原已审计周排名作为逐周新增底仓候选；不能将全年并集提前到年初买入。"""
    if scope not in {"validation", "test"}:
        raise ValueError("scope须为validation/test")
    year = fold["test_year"]
    filename = "validation_weekly_selections.csv" if scope == "validation" else "weekly_selections.csv"
    path = Path(cfg["frozen_selection_report_dir"]) / f"year_{year}" / filename
    audit_path = Path(cfg["source_training_report_dir"]) / "independent_residual_audit.json"
    record_balanced_input(audit_path, inputs)
    audit = json.loads(audit_path.read_text())["frozen_contexts"][f"year_{year}/{scope}"]
    if not audit["passed"]:
        raise ValueError("对应周模型/名单未通过原审计")
    record_balanced_input(path, inputs, audit["source_selection_sha256"])
    # 原冻结周选择器使用pandas默认CSV解析；保留该数值合同，不加入新的重排序。
    selected = pd.read_csv(path, dtype={"symbol": str})
    for column in ("trade_week", "entry_date", "exit_date", "signal_datetime"):
        selected[column] = pd.to_datetime(selected[column], errors="raise")
    if not selected.signal_datetime.lt(selected.entry_date + pd.Timedelta(hours=9, minutes=31)).all():
        raise ValueError("基础周名单不能在其信号之前开仓")
    model_name = "inner_weekly_model" if scope == "validation" else "weekly_model"
    model = Path(cfg["frozen_selection_report_dir"]) / f"year_{year}" / model_name
    for filename, key in (("model_bundle.joblib", "source_model_sha256"),
                          ("model_metadata.json", "source_metadata_sha256")):
        record_balanced_input(model / filename, inputs, audit["context"][key])
    metadata = json.loads((model / "model_metadata.json").read_text())
    expected_train = fold["core_years"] if scope == "validation" else fold["train_years"]
    expected_excluded = [fold["inner_year"], year] if scope == "validation" else [year]
    if (sorted(metadata["train_years"]) != sorted(expected_train) or
            sorted(metadata["excluded_years"]) != sorted(expected_excluded)):
        raise ValueError("周模型实际训练/排除年份与拆分不一致")
    selected = selected.loc[selected.entry_date.isin(days)].copy()
    maximum = int(cfg.get("max_symbols", 0))
    if maximum:
        order = selected.sort_values(["entry_date", "rank", "symbol"], kind="stable")
        symbols = order.symbol.drop_duplicates().head(maximum)
        selected = selected.loc[selected.symbol.isin(symbols)].copy()
    return selected.sort_values(["entry_date", "rank", "symbol"], kind="stable").reset_index(drop=True)
