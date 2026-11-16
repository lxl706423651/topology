"""评价冻结的高频率T候选并保存完整账户，分母只取事前可交易旧股和合法时钟。

从quant项目调用；公共行情由spawn worker初始化一次，候选不改变执行配置、不
重算bootstrap，也不选择规则。原行情和模型只读，输出写入独立candidate_XXX目录。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
from numbers import Real
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from .weekly_t_engine import run_weekly_portfolio
from .weekly_t_short_horizon import planned_recovery_times
from .weekly_t_validation import validate_candidate_result


RESULT_FILES = {
    "equity_curve": "test_equity_curve.csv", "trade_events": "test_trade_events.csv",
    "round_trips": "test_round_trips.csv", "open_positions": "test_open_positions.csv",
    "daily_inventory_audit": "daily_inventory_audit.csv", "order_attempts": "blocked_orders.csv",
    "baseline_events": "baseline_trade_events.csv", "weekly_entry_plans": "weekly_entry_plans.csv",
    "base_plan_mismatches": "base_plan_mismatches.csv",
}
OPPORTUNITY_COLUMNS = ["date", "symbol", "opening_sellable", "eligible_t_shares",
                       "eligible_signal_count", "first_eligible_signal", "last_eligible_signal"]
_FREQUENCY_CONTEXT: dict = {}


def calculate_frequency_opportunities(baseline_result: dict, intraday: pd.DataFrame,
                                     cfg: dict, horizon_minutes: int = 20) -> tuple[dict, pd.DataFrame]:
    """以no-T账户日初旧股和当前信号时钟计算分母，返回摘要和唯一日期/股票表。

intraday已由调用方限制为选股周的中间交易日。本函数仅消费datetime/symbol，
不按预测、未来恢复价量、成交结果或当日现金删分母；当日新买不增加旧股数。
    """
    lot, fraction = cfg.get("lot_size", 100), cfg.get("short_t_fraction", 0.25)
    if isinstance(lot, (bool, np.bool_)) or not isinstance(lot, (int, np.integer)) or lot < 100 or lot % 100:
        raise ValueError("lot_size必须为100的正整数倍")
    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real) or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("short_t_fraction必须处于(0,1]")
    if isinstance(horizon_minutes, (bool, np.bool_)) or not isinstance(horizon_minutes, (int, np.integer)) or horizon_minutes < 1:
        raise ValueError("horizon_minutes必须为正整数交易分钟")
    if "trade_events" in baseline_result and baseline_result["trade_events"].event_type.eq("t").any():
        raise ValueError("机会分母必须来自未做T的基准账户")
    inventory = baseline_result["daily_inventory_audit"][["date", "symbol", "opening_sellable"]].copy()
    inventory["date"] = pd.to_datetime(inventory.date, errors="raise")
    if inventory.date.dt.tz is not None or inventory.date.isna().any() or not inventory.date.eq(inventory.date.dt.normalize()).all():
        raise ValueError("基准库存日期必须为非缺失交易日")
    if not inventory.symbol.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("基准库存股票代码必须为非空字符串")
    if inventory.duplicated(["date", "symbol"]).any():
        raise ValueError("基准库存date/symbol必须唯一")
    old = pd.to_numeric(inventory.opening_sellable, errors="raise")
    if not (np.isfinite(old) & old.ge(0) & old.mod(1).eq(0)).all():
        raise ValueError("开盘可卖旧股必须为有限非负整数")
    inventory["opening_sellable"] = old.astype("int64")
    inventory["eligible_t_shares"] = (np.floor(old * float(fraction) / lot) * lot).astype("int64")
    clocks = intraday[["datetime", "symbol"]].copy()
    clocks["datetime"] = pd.to_datetime(clocks.datetime, errors="raise")
    if clocks.datetime.dt.tz is not None or clocks.datetime.isna().any() or clocks.duplicated(["datetime", "symbol"]).any():
        raise ValueError("信号datetime/symbol必须非缺失且唯一")
    if not clocks.symbol.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("信号股票代码必须为非空字符串")
    clocks["date"] = clocks.datetime.dt.normalize()
    recovery = planned_recovery_times(clocks.datetime, int(horizon_minutes))
    deadline = cfg.get("latest_planned_recovery_time", "14:30")
    if not isinstance(deadline, str) or re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", deadline) is None:
        raise ValueError("latest_planned_recovery_time必须为HH:MM当日时钟")
    deadline_offset = pd.to_timedelta(f"{deadline}:00")
    valid = recovery.notna() & recovery.dt.normalize().eq(clocks.date)
    valid &= recovery.le(clocks.date + deadline_offset)
    legal = clocks.loc[valid].groupby(["date", "symbol"], sort=True).agg(
        eligible_signal_count=("datetime", "size"), first_eligible_signal=("datetime", "min"),
        last_eligible_signal=("datetime", "max")).reset_index()
    eligible = inventory.loc[inventory.eligible_t_shares.ge(lot)].merge(
        legal, on=["date", "symbol"], how="inner", validate="one_to_one")
    eligible = eligible[OPPORTUNITY_COLUMNS].sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    summary = {"eligible_portfolio_days": int(eligible.date.nunique()),
               "eligible_inventory_stock_days": len(eligible),
               "eligible_signal_rows": int(eligible.eligible_signal_count.sum()),
               "horizon_minutes": int(horizon_minutes), "short_t_fraction": float(fraction),
               "lot_size": int(lot), "latest_planned_recovery_time": deadline,
               "denominator_scope": "no-T基准开盘旧股整手及中间交易日当前信号时钟；不按未来成交、恢复或现金筛选"}
    return summary, eligible


def write_frequency_frame(frame: pd.DataFrame, path: Path) -> None:
    """原子写入UTF-8 BOM候选CSV，失败时不留下貌似完整的截断文件。"""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_frequency_json(value: dict, path: Path) -> None:
    """原子保存配置、指标和失败证据，转换numpy标量及时间但不改原对象。"""
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        raise TypeError(f"不能序列化{type(item).__name__}")
    text = json.dumps(value, ensure_ascii=False, indent=2, default=convert)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_frequency_worker(selections: pd.DataFrame, daily: pd.DataFrame,
                                intraday: pd.DataFrame, cfg: dict, audit_dir: str,
                                recovery_store: object | None) -> None:
    """spawn已隔离公共frames；每worker只初始化一次并复用只读恢复索引。"""
    global _FREQUENCY_CONTEXT
    _FREQUENCY_CONTEXT = {"selections": selections, "daily": daily, "intraday": intraday,
                          "cfg": deepcopy(cfg), "audit_dir": Path(audit_dir), "recovery_store": recovery_store}


def evaluate_frequency_task(task: dict) -> tuple[int, dict]:
    """真实执行一个候选，先保全完整账本再验证；异常留证并原样传播。"""
    context = _FREQUENCY_CONTEXT
    index = int(task["candidate_index"])
    directory = context["audit_dir"] / f"candidate_{index:03d}"
    directory.mkdir(exist_ok=False)
    write_frequency_frame(task["decisions"], directory / "decisions.csv")
    identity = {"candidate_index": index, "decision_sha256": task["decision_sha256"],
                "saved_decisions_sha256": hashlib.sha256((directory / "decisions.csv").read_bytes()).hexdigest()}
    write_frequency_json(identity, directory / "candidate.json")
    write_frequency_json(context["cfg"], directory / "resolved_config.json")
    try:
        kwargs = {} if context["recovery_store"] is None else {"recovery_bars": context["recovery_store"]}
        result = run_weekly_portfolio(context["selections"], context["daily"], context["intraday"],
                                      task["decisions"], context["cfg"], **kwargs)
        for key, frame in result.items():
            if isinstance(frame, pd.DataFrame):
                write_frequency_frame(frame, directory / RESULT_FILES.get(key, f"{key}.csv"))
        write_frequency_json(result["metrics"], directory / "metrics.json")
        write_frequency_frame(pd.DataFrame([result["metrics"]]), directory / "test_metrics.csv")
        execution_audit = validate_candidate_result(result, context["cfg"])
        events = result["trade_events"]
        first = events.loc[events.event_type.eq("t") & events.fragment_id.eq(0)].copy()
        if len(first) != first.pair_id.nunique() or len(first) != result["metrics"]["started_pair_count"]:
            raise AssertionError("T首腿fragment_id与实际组合数量不一致")
        first["date"] = pd.to_datetime(first.datetime).dt.normalize()
        curve = result["equity_curve"]
        summary = {**identity, "original_index": index,
                   "incremental_net_pnl": float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1]),
                   "metrics": result["metrics"], "execution_audit": execution_audit,
                   "active_days": int(first.date.nunique()),
                   "active_stock_days": len(first[["date", "symbol"]].drop_duplicates()),
                   "t_first_leg_count": len(first), "request_count": len(task["decisions"]),
                   "candidate_directory": str(directory.resolve())}
        write_frequency_json(execution_audit, directory / "execution_audit.json")
        write_frequency_json(summary, directory / "summary.json")
        return index, summary
    except Exception as exc:
        write_frequency_json({**identity, "passed": False, "error_type": type(exc).__name__,
                              "error": str(exc)}, directory / "failure.json")
        raise


def run_frequency_candidates(tasks: list[dict], selections: pd.DataFrame, daily: pd.DataFrame,
                             intraday: pd.DataFrame, cfg: dict, audit_dir: str | Path,
                             recovery_store: object | None = None) -> dict[int, dict]:
    """按candidate_index返回真实独立账户；重复决策哈希允许，索引和目录不得重复。

candidate_workers默认6，使用spawn并一次性初始化公共上下文。每个完成候选打印
进度；任务异常取消未开始任务并传播，恢复/周退出等合法失败仍保留完整账户。
    """
    workers = cfg.get("candidate_workers", 6)
    if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("candidate_workers必须为正整数")
    indices = []
    for task in tasks:
        if not {"candidate_index", "decision_sha256", "decisions"}.issubset(task):
            raise ValueError("候选缺少candidate_index/decision_sha256/decisions")
        index = task["candidate_index"]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError("candidate_index必须为非负整数")
        if not isinstance(task["decision_sha256"], str) or not task["decision_sha256"]:
            raise ValueError("decision_sha256必须为非空字符串")
        if not isinstance(task["decisions"], pd.DataFrame):
            raise TypeError("decisions必须是DataFrame")
        indices.append(int(index))
    if len(set(indices)) != len(indices):
        raise ValueError("candidate_index重复会覆盖候选完整账本")
    if not tasks:
        return {}
    directory = Path(audit_dir)
    if any((directory / f"candidate_{index:03d}").exists() for index in indices):
        raise FileExistsError("候选账户目录已存在；请使用新的独立audit_dir")
    directory.mkdir(parents=True, exist_ok=True)
    arguments = (selections, daily, intraday, cfg, str(directory), recovery_store)
    outputs = {}
    if workers == 1:
        initialize_frequency_worker(*arguments)
        try:
            for completed, task in enumerate(tasks, 1):
                index, summary = evaluate_frequency_task(task)
                outputs[index] = summary
                print(f"候选账户完成 {completed}/{len(tasks)}：candidate_{index:03d}", flush=True)
        finally:
            _FREQUENCY_CONTEXT.clear()
    else:
        print(f"内部候选完整账户：{len(tasks)}个候选，{workers}个spawn进程", flush=True)
        pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initialize_frequency_worker, initargs=arguments)
        futures = []
        try:
            futures = [pool.submit(evaluate_frequency_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                index, summary = future.result()
                outputs[index] = summary
                print(f"候选账户完成 {completed}/{len(tasks)}：candidate_{index:03d}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return {index: outputs[index] for index in indices}
