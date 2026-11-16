"""从已有内部回测选择实际做过T且满足执行要求的候选，不以统计门槛关闭T。

本模块只处理已完成的内部候选指标，不读行情、外层结果或模型，也不修改
输入表。净增量只用于排序；置信区间、样本数量和集中度保留为报告参考。
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


EXECUTION_FAILURE_COLUMNS = (
    "inventory_mismatch_days",
    "recovery_failure_days",
    "pending_pairs",
    "base_plan_mismatch_count",
    "t1_violation_count",
)
REQUIRED_SELECTION_COLUMNS = (
    "candidate_index",
    "incremental_net_pnl",
    "t_pair_count",
    *EXECUTION_FAILURE_COLUMNS,
)


def validate_selection_values(rows: pd.DataFrame, column: str, *, count: bool) -> pd.Series:
    """检查选择所需数值；拒绝缺失、非有限、布尔以及非法计数，返回数值副本。"""
    values = rows[column]
    valid_type = values.map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_)))
    if not valid_type.all():
        raise ValueError(f"候选字段{column}必须为数值，不能缺失、字符串或布尔")
    numeric = values.astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"候选字段{column}存在非有限数值")
    if count and ((numeric < 0) | (numeric != np.floor(numeric))).any():
        raise ValueError(f"候选字段{column}必须为非负整数计数")
    return numeric


def select_active_candidate(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """按内部收益选真实有T且执行合格的候选，返回选中行及保留原顺序的注释表。

    要求完整T配对数大于零，五项执行失败计数均为零；可选候选即使亏损也
    参与排序，并列时取较小candidate_index。若存在completed_pair_count，
    则必须与t_pair_count逐行一致。没有合格候选直接报错，不选择零交易行。
    原strict_eligible/rejection_reasons不作为门槛也不改写；enabled由调用方
    明确构造为True。本函数不保证未来每天必有交易或盈利。
    """
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("候选必须为DataFrame")
    if rows.empty:
        raise ValueError("没有内部候选，无法选择主动T规则")
    if rows.columns.duplicated().any():
        raise ValueError("候选表存在重复列名")
    missing = [column for column in REQUIRED_SELECTION_COLUMNS if column not in rows]
    if missing:
        raise ValueError("候选缺少必要字段：" + ", ".join(missing))

    numeric = {}
    for column in REQUIRED_SELECTION_COLUMNS:
        numeric[column] = validate_selection_values(rows, column, count=column != "incremental_net_pnl")
    if numeric["candidate_index"].duplicated().any():
        raise ValueError("candidate_index必须唯一，避免并列选择存在歧义")
    if "completed_pair_count" in rows:
        completed = validate_selection_values(rows, "completed_pair_count", count=True)
        if not np.array_equal(completed.to_numpy(), numeric["t_pair_count"].to_numpy()):
            raise ValueError("t_pair_count与completed_pair_count不一致")

    reasons = []
    for position in range(len(rows)):
        failures = [column for column in EXECUTION_FAILURE_COLUMNS if numeric[column].iloc[position] != 0]
        if numeric["t_pair_count"].iloc[position] == 0:
            failures.append("no_completed_t_pairs")
        reasons.append("|".join(failures))
    eligible = np.array([not value for value in reasons], dtype=bool)
    if not eligible.any():
        raise ValueError("没有完成真实T且满足库存恢复、基准计划和T+1要求的候选；不能用零交易替代")

    # 不依赖输入行序、索引或旧selected状态；相同收益时严格按候选编号确定。
    ordering = pd.DataFrame({"position": np.flatnonzero(eligible),
                            "net": numeric["incremental_net_pnl"].to_numpy()[eligible],
                            "candidate_index": numeric["candidate_index"].to_numpy()[eligible]})
    chosen_position = int(ordering.sort_values(["net", "candidate_index"], ascending=[False, True],
                                              kind="stable").iloc[0]["position"])
    annotated = rows.copy(deep=True)
    annotated["active_execution_eligible"] = eligible
    annotated["execution_rejection_reasons"] = reasons
    annotated["selected"] = False
    annotated.iloc[chosen_position, annotated.columns.get_loc("selected")] = True
    return annotated.iloc[chosen_position].to_dict(), annotated
