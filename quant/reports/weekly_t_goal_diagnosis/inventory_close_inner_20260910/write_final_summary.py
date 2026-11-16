"""只汇总已通过的v16内部诊断CSV；不读账户/原分钟、不择参，新增宏观误差与失败说明。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def sha(path):
    """小型已生成诊断产物按完整字节绑定。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(final):
    """每折统计36个相互重叠的主动账户分布，绝不合计成可投资组合。"""
    proof_path = final/"verification.json"
    proof = json.loads(proof_path.read_text())
    assert proof["passed"] is True and proof["complete"] is True and proof["no_outer_read"] is True
    assert all(result["selected_rule"]["candidate_index"] == 0 for result in proof["results"])
    names = ["candidate_overview.csv", "candidate_source_comparison.csv", "prediction_quality.csv"]
    for name in names:
        assert sha(final/name) == proof["output_sha256"][name]
    metrics, comparison, quality = [pd.read_csv(final/name, float_precision="round_trip") for name in names]
    active = metrics.loc[metrics.enabled]
    rows = []
    for (fold, inner), frame in active.groupby(["fold", "inner_year"]):
        assert len(frame) == 36
        compared = comparison.loc[comparison.fold.eq(fold) & comparison.candidate_index.gt(0)]
        rate = frame.realized_complete_pair_pnl/frame.all_reference_acquisition_cost
        assert np.allclose(rate, frame.holding_cost_reduction_rate, atol=1e-12, rtol=0)
        row = dict(fold=fold, inner_year=inner, active_candidates=len(frame),
            marked_positive=int(frame.incremental_net_pnl.gt(0).sum()),
            marked_negative=int(frame.incremental_net_pnl.lt(0).sum()),
            marked_zero=int(frame.incremental_net_pnl.eq(0).sum()),
            complete_net_positive=int(frame.realized_complete_pair_pnl.gt(0).sum()),
            execution_eligible=int(frame.execution_eligible.sum()),
            marked_net_min=frame.incremental_net_pnl.min(), marked_net_max=frame.incremental_net_pnl.max(),
            complete_net_min=frame.realized_complete_pair_pnl.min(), complete_net_max=frame.realized_complete_pair_pnl.max(),
            complete_groups_min=int(frame.t_pair_count.min()), complete_groups_max=int(frame.t_pair_count.max()),
            paired_lots_min=frame.completed_t_pair_lots.min(), paired_lots_max=frame.completed_t_pair_lots.max(),
            actual_fee_min=frame.t_fee.min(), actual_fee_max=frame.t_fee.max(),
            actual_slippage_min=frame.t_slippage_cost.min(), actual_slippage_max=frame.t_slippage_cost.max(),
            cost_denominator=float(frame.all_reference_acquisition_cost.iloc[0]),
            complete_cost_rate_min=rate.min(), complete_cost_rate_max=rate.max(),
            inventory_failed_candidates=int(frame.inventory_mismatch_days.gt(0).sum()),
            recovery_failed_candidates=int(frame.recovery_failure_days.gt(0).sum()),
            pending_candidates=int(frame.pending_pairs.gt(0).sum()),
            pending_pairs_max=int(frame.pending_pairs.max()),
            t1_failed_candidates=int(frame.t1_violation_count.gt(0).sum()),
            base_failed_candidates=int(frame.base_plan_mismatch_count.gt(0).sum()),
            same_numeric_net_better=int(compared.incremental_net_pnl_change.gt(0).sum()),
            same_numeric_net_worse=int(compared.incremental_net_pnl_change.lt(0).sum()),
            same_numeric_net_equal=int(compared.incremental_net_pnl_change.eq(0).sum()),
            same_numeric_net_difference_median=compared.incremental_net_pnl_change.median())
        rows.append(row)
    overview = pd.DataFrame(rows)
    out = final/"mechanism_explanation"
    out.mkdir(exist_ok=False)
    overview.to_csv(out/"annual_active_risk_overview.csv", index=False, encoding="utf-8-sig")
    q = quality.loc[quality.grouping.eq("all") & quality.scope.eq("original_current_margin0bp")
                    & quality.domain.eq("baseline_opening_old_inventory")]
    q.to_csv(out/"known_inventory_prediction_error.csv", index=False, encoding="utf-8-sig")
    lines = ["# 固定尾盘目标：内部误差和恢复约束结论", "",
        "本说明只汇总已完成内部数据。四折原选择均为hold，正式外层没有执行；零T增量不是盈利改善。", "",
        "|留出折/内部年|主动正/负/零净额账户|执行合格|完整组范围|配对手范围|含残仓净增量范围 元|库存失败/恢复失败/pending账户|",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in overview.itertuples():
        lines.append(f"|{row.fold}/{row.inner_year}|{row.marked_positive}/{row.marked_negative}/{row.marked_zero}|{row.execution_eligible}|{row.complete_groups_min}–{row.complete_groups_max}|{row.paired_lots_min:.0f}–{row.paired_lots_max:.0f}|{row.marked_net_min:.2f}～{row.marked_net_max:.2f}|{row.inventory_failed_candidates}/{row.recovery_failed_candidates}/{row.pending_candidates}|")
    lines += ["", "表中每一列是36个相互重叠候选的范围或计数，不可把不同候选利润、费用或手数相加当作一只组合。配对手指一个完整T两腿匹配的100股单位，恢复片段另计。", "",
        "|留出折|模型|旧股当前机会|预测净差 bp|实现方向毛差 bp|固定14:55标称净差 bp|预测乐观偏差 bp|标称覆盖成本比例|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in q.itertuples():
        lines.append(f"|{row.fold}|{row.profile}|{row.rows}|{row.mean_predicted_net_bp_common:.2f}|{row.mean_fixed_gross_bp:.2f}|{row.mean_fixed_nominal_net_bp:.2f}|{row.mean_net_optimism_bp:.2f}|{row.nominal_cost_covered_rate:.2%}|")
    lines += ["", "以上固定端点是在原0边际当前条件和日初旧股机会中的诊断，不等于实际成交。预测方向先由当前两腿扣费预估选定，随后用内部年真实精确端点评估；未来缺报价被计数，没有据此删除原决策。当前量价合格、预测净差正或名义方向胜率高，都不证明容量、现金、可卖旧股及当日恢复可行。", "",
        "|留出折|同数字候选净改善/恶化/相等|差额中位 元|实际含费底仓分母 元|完整组成本降低率范围|",
        "|---|---:|---:|---:|---:|"]
    for row in overview.itertuples():
        lines.append(f"|{row.fold}|{row.same_numeric_net_better}/{row.same_numeric_net_worse}/{row.same_numeric_net_equal}|{row.same_numeric_net_difference_median:.2f}|{row.cost_denominator:.2f}|{row.complete_cost_rate_min:.4%}～{row.complete_cost_rate_max:.4%}|")
    lines += ["", "v15和v16保持数字参数及底仓，但预测目标、实际首腿集合和恢复期限不同。以上差额是两套完整策略的内部结果比较，不能声称仅改变退出时点的因果效果。总体没有给出固定14:55模型稳定改善盈利或每日恢复的证据。", "",
        "成本降低率按原报告合同使用“实际完整组净额÷原始含费底仓取得成本”；含残仓账户增量还包含未完成首腿费用和期末库存差额估值，因此两者不能混用。所有恢复片段、未恢复股数与实际完整组在candidate_pairs.csv中分列。即使期末pending=0，也可能经历跨日恢复而违反每日股数要求。", "",
        "时段效应等研究只能支持检验持有期限的假设，不能为固定14:55提供收益保证。这里仍是重复历史研究，年度样本和OOF来源的统计依赖未因换目标消失；没有据已见内部诊断重拟合、调阈值、反向信号或删去失败股票。"]
    (out/"macro_error_and_risk.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    shutil.copy2(__file__, out/"write_final_summary.py")
    metadata = dict(passed=True, derived_only_from_complete_internal_diagnosis=True, source_report=proof["source_report"],
        no_new_raw_or_account_read=True, no_outer_read=True, no_rule_or_model_changed=True,
        cost_rate_uses_complete_pair_net_not_marked_increment=True,
        input_sha256={str(path): sha(path) for path in [proof_path, *[final/name for name in names]]},
        output_sha256={path.name: sha(path) for path in out.iterdir() if path.is_file()})
    (out/"verification.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    print(overview.to_json(orient="records"))
    print(json.dumps(dict(output=str(out), sha256=sha(out/"verification.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_directory", type=Path)
    summarize(parser.parse_args().final_directory)
