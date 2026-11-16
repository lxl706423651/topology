"""在独立账户中并行评价已经冻结的内部做T候选，不读取外层数据或选择参数。

多进程显式采用spawn，避免复制已训练LightGBM的线程状态；公共行情在每个
worker初始化一次。任务只携带决策表，返回轻量指标，逐候选审计CSV独立落盘。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .inventory_t import calculate_next_trading_minute
from .weekly_t_engine import run_weekly_portfolio


_WORKER_CONTEXT: dict = {}


def validate_candidate_result(result: dict, cfg: dict) -> dict:
    """核对已成交现金、整手、昨日卖额和信号时序；合法但未恢复的候选照常返回。

共同底仓计划一致且无待恢复组合时，额外权益须等于全部恢复片段净利润。
恢复失败、库存目标未达成和基准计划偏离属于选参指标，不因此隐藏候选结果；
实际负现金、T+1违规、穿越时序或错误记账则立即抛出AssertionError。
    """
    configured_lot = cfg.get("lot_size", 100)
    lot = int(configured_lot)
    if lot != configured_lot or lot < 100 or lot % 100:
        raise ValueError("lot_size必须为100的正整数倍")
    metrics = result["metrics"]
    curve, inventory = result["equity_curve"], result["daily_inventory_audit"]
    if curve.empty:
        raise AssertionError("候选没有权益记录，不能评价增量收益")
    checks = {"integer_lots": True, "nonnegative_cash": True, "t1_legal": True,
              "execution_after_signal": True, "t_first_leg_next_minute": True,
              "paired_time_order": True, "paired_pnl_identity": True}
    for key in ("trade_events", "baseline_events"):
        events = result[key]
        if events.empty:
            continue
        checks["integer_lots"] &= bool((np.isfinite(events.shares) & (events.shares > 0)
                                         & (events.shares % lot == 0)).all())
        checks["nonnegative_cash"] &= bool((np.isfinite(events.cash_after)
                                            & (events.cash_after >= -1e-6)).all())
        sales = events.loc[events.side == "sell"]
        checks["t1_legal"] &= bool((sales.shares <= sales.sellable_before).all()
                                    and (events.sellable_before >= 0).all()
                                    and (events.sellable_after >= 0).all())
        buys = events.loc[events.side == "buy"]
        checks["t1_legal"] &= bool(buys.sellable_after.eq(buys.sellable_before).all()
                                    and sales.sellable_after.eq(sales.sellable_before - sales.shares).all())
        checks["execution_after_signal"] &= bool((pd.to_datetime(events.datetime)
                                                    > pd.to_datetime(events.signal_datetime)).all())
        t_events = events.loc[events.event_type == "t"].sort_values("datetime", kind="stable")
        first_legs = t_events.drop_duplicates("pair_id", keep="first")
        checks["t_first_leg_next_minute"] &= all(
            pd.Timestamp(row.datetime) == calculate_next_trading_minute(pd.Timestamp(row.signal_datetime))
            for row in first_legs.itertuples(index=False))
    for key in ("cash", "baseline_cash"):
        checks["nonnegative_cash"] &= bool((np.isfinite(curve[key]) & (curve[key] >= -1e-6)).all())
    checks["nonnegative_cash"] &= bool(np.isfinite(curve[["equity", "baseline_equity"]]).all().all())
    if len(inventory):
        checks["t1_legal"] &= bool((inventory.day_sold >= 0).all()
                                    and (inventory.day_sold <= inventory.opening_sellable).all()
                                    and (inventory.closing_sellable >= 0).all())
    checks["t1_legal"] &= metrics["t1_violation_count"] == 0
    trips = result["round_trips"]
    if len(trips):
        checks["paired_time_order"] = bool((pd.to_datetime(trips.second_datetime)
                                               > pd.to_datetime(trips.first_datetime)).all())
    identity_checked = bool(metrics["same_base_plan"] and metrics["pending_pairs"] == 0)
    if identity_checked:
        increment = float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1])
        checks["paired_pnl_identity"] = bool(np.isclose(increment, trips.net_pnl.sum(), atol=1e-5, rtol=1e-8))
    if not all(checks.values()):
        raise AssertionError(f"候选成交不变量失败: {checks}")
    return {**checks, "paired_pnl_identity_checked": identity_checked}


def initialize_candidate_worker(selections: pd.DataFrame, daily: pd.DataFrame,
                                intraday: pd.DataFrame, cfg: dict, audit_dir: str,
                                recovery_store: object | None) -> None:
    """每个worker只复制一次公共只读输入；恢复索引交由执行器按到达时钟读取。"""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {"selections": selections.copy(deep=True), "daily": daily.copy(deep=True),
                       "intraday": intraday.copy(deep=True), "cfg": deepcopy(cfg),
                       "audit_dir": Path(audit_dir), "recovery_store": recovery_store}


def write_candidate_audit(frame: pd.DataFrame, path: Path) -> None:
    """原子导出单候选UTF-8 BOM审计CSV，不覆盖其他候选的文件。"""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def evaluate_candidate_task(task: dict) -> tuple[str, dict]:
    """以独立账户执行一份冻结decisions，保存失败细节并仅返回轻量指标。"""
    context = _WORKER_CONTEXT
    kwargs = {} if context["recovery_store"] is None else {"recovery_bars": context["recovery_store"]}
    result = run_weekly_portfolio(context["selections"], context["daily"], context["intraday"],
                                  task["decisions"], context["cfg"], **kwargs)
    execution_audit = validate_candidate_result(result, context["cfg"])
    index = int(task["candidate_index"])
    attempts = result["order_attempts"]
    t_attempts = attempts.loc[attempts.event_type.astype(str).str.startswith("t")]
    inventory = result["daily_inventory_audit"]
    failures = inventory.loc[(~inventory.inventory_preserved) | inventory.recovery_failure]
    write_candidate_audit(t_attempts, context["audit_dir"] / f"candidate_{index:03d}_t_attempts.csv")
    write_candidate_audit(failures, context["audit_dir"] / f"candidate_{index:03d}_t_inventory_failures.csv")
    curve = result["equity_curve"]
    summary = {"incremental_net_pnl": float(curve.equity.iloc[-1] - curve.baseline_equity.iloc[-1]),
               "metrics": result["metrics"], "original_index": index, "execution_audit": execution_audit}
    if context["cfg"].get("candidate_support_requirements"):
        from .weekly_t_short_horizon import candidate_support_statistics
        summary["support"] = candidate_support_statistics(result, context["cfg"])
    return str(task["decision_sha256"]), summary


def run_candidate_evaluations(tasks: list[dict], selections: pd.DataFrame, daily: pd.DataFrame,
                              intraday: pd.DataFrame, cfg: dict, audit_dir: str | Path,
                              recovery_store: object | None = None) -> dict:
    """运行事前去重的内部候选并按原tasks顺序返回哈希到指标的映射，不作参数选择。

tasks每项含candidate_index、decision_sha256及decisions；索引和哈希须唯一。
candidate_workers默认1；大于1时使用spawn和一次性初始化，不从训练进程fork。
worker异常原样传回调用者并取消尚未执行的任务，不能把异常当作零收益候选。
    """
    workers = cfg.get("candidate_workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("candidate_workers必须为正整数")
    indices, hashes = [], []
    for task in tasks:
        if not {"candidate_index", "decision_sha256", "decisions"}.issubset(task):
            raise ValueError("候选任务缺少candidate_index/decision_sha256/decisions")
        index = task["candidate_index"]
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError("candidate_index必须为非负整数")
        if not isinstance(task["decision_sha256"], str) or not task["decision_sha256"]:
            raise ValueError("decision_sha256必须为非空字符串")
        if not isinstance(task["decisions"], pd.DataFrame):
            raise TypeError("decisions必须是DataFrame")
        indices.append(int(index))
        hashes.append(task["decision_sha256"])
    if len(set(indices)) != len(indices) or len(set(hashes)) != len(hashes):
        raise ValueError("候选索引和决策哈希必须事前去重，避免审计覆盖或重复账户")
    if not tasks:
        return {}
    directory = Path(audit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    arguments = (selections, daily, intraday, cfg, str(directory), recovery_store)
    outputs = {}
    if workers == 1:
        initialize_candidate_worker(*arguments)
        try:
            for task in tasks:
                key, value = evaluate_candidate_task(task)
                outputs[key] = value
        finally:
            _WORKER_CONTEXT.clear()
    else:
        print(f"内部候选并行评价：{len(tasks)}个独立账户，{workers}个spawn进程", flush=True)
        pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initialize_candidate_worker, initargs=arguments)
        futures = []
        try:
            futures = [pool.submit(evaluate_candidate_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                key, value = future.result()
                outputs[key] = value
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"内部候选执行完成 {completed}/{len(tasks)}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return {key: outputs[key] for key in hashes}
