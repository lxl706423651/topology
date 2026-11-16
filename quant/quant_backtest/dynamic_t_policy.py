"""v13预声明入场×动态恢复候选；仅扩展退出，不用未来价格产生首腿信号。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .balanced_t_policy import build_balanced_candidates, generate_balanced_decisions


VERSION = "dynamic_inventory_t_v13"


def build_dynamic_candidates(cfg: dict) -> list[dict]:
    """按原入场候选稳定顺序交叉三个退出结构，保留一个真实hold对照。"""
    modes = cfg.get("dynamic_exit_modes", ["fixed", "profit_buffer", "profit_buffer_stop"])
    if modes != ["fixed", "profit_buffer", "profit_buffer_stop"]:
        raise ValueError("v13要求三个事前退出结构及固定顺序")
    take = cfg.get("dynamic_take_net_return", .003)
    stop = cfg.get("dynamic_stop_net_return", .01)
    if isinstance(take, bool) or isinstance(stop, bool) or not np.isfinite([take, stop]).all() or not 0 < take < stop < 1:
        raise ValueError("动态净额阈值须为有效正比率且止损幅度大于止盈缓冲")
    entries = build_balanced_candidates(cfg)
    candidates = [{**entries[0], "entry_candidate_index": 0, "exit_mode": "fixed",
                   "take_net_return": float(take), "stop_net_return": float(stop)}]
    for entry in entries[1:]:
        for mode in modes:
            candidates.append({**entry, "candidate_index": len(candidates),
                "entry_candidate_index": int(entry["candidate_index"]), "exit_mode": mode,
                "take_net_return": float(take), "stop_net_return": float(stop)})
    return candidates


def generate_dynamic_decisions(intraday: pd.DataFrame, rule: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """首腿完全沿用当前入场信号；恢复由引擎后续真实当前bar独立触发。"""
    candidates = build_dynamic_candidates(cfg)
    index = rule.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(candidates):
        raise ValueError("动态候选编号非法")
    if rule != candidates[int(index)]:
        raise ValueError("动态候选与事前配置不一致")
    entry = build_balanced_candidates(cfg)[rule["entry_candidate_index"]]
    decisions, audit = generate_balanced_decisions(intraday, entry, cfg)
    decisions["candidate_index"] = int(index)
    return decisions, {**audit, "entry_candidate_index": rule["entry_candidate_index"],
                       "candidate_index": int(index), "exit_mode": rule["exit_mode"],
                       "exit_decisions_not_precomputed_from_future_path": True}


def create_dynamic_rule_config(cfg: dict, rule: dict) -> dict:
    """给单账户传递已冻结退出模式，不改变共同费用、底仓或行情配置。"""
    return {**cfg, "dynamic_exit_mode": rule["exit_mode"],
            "dynamic_take_net_return": rule["take_net_return"],
            "dynamic_stop_net_return": rule["stop_net_return"]}
