"""仅从已通过的内部诊断派生表写烟测机制摘要，不读取原市场、模型或外层。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def sha(path):
    """绑定本次实际读取的派生输入。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """仅一个已终核的烟测内部折，不用候选对照修改选择。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", required=True)
    args = parser.parse_args()
    folder = Path(args.diagnosis_dir).resolve()
    verification = json.loads((folder/"verification.json").read_text())
    assert verification["passed"] and verification["complete"] and verification["no_outer_read"]
    assert "_smoke/" in verification["source_report"] and len(verification["results"]) == 1
    names = ["verification.json", "candidate_overview.csv", "candidate_source_comparison.csv", "prediction_quality.csv"]
    inputs = {str(folder/name):sha(folder/name) for name in names}
    for name in names[1:]: assert sha(folder/name) == verification["output_sha256"][name]
    metrics = pd.read_csv(folder/"candidate_overview.csv", float_precision="round_trip")
    comparison = pd.read_csv(folder/"candidate_source_comparison.csv", float_precision="round_trip")
    quality = pd.read_csv(folder/"prediction_quality.csv", float_precision="round_trip")
    active = metrics.loc[metrics.enabled]
    source = comparison.loc[comparison.candidate_index.gt(0)]
    chosen = metrics.loc[metrics.selected].iloc[0]
    pair = comparison.loc[comparison.candidate_index.eq(chosen.candidate_index)].iloc[0]
    fee_saving = -pair.t_fee_change-pair.t_slippage_cost_change
    change = pair.incremental_net_pnl_change
    population = quality.loc[quality.grouping.eq("all") & quality.scope.eq("original_current_margin0bp")
        & quality.domain.eq("baseline_opening_old_inventory")]
    result = dict(passed=True, scope="engineering_smoke_internal_only", source_report=verification["source_report"],
        internal_year=int(chosen.inner_year), fold=int(chosen.fold), active_accounts=len(active),
        active_negative_accounts=int(active.incremental_net_pnl.lt(0).sum()),
        execution_eligible_accounts=int(active.execution_eligible.sum()),
        complete_group_range=[int(active.t_pair_count.min()),int(active.t_pair_count.max())],
        paired_lot_range=[float(active.completed_t_pair_lots.min()),float(active.completed_t_pair_lots.max())],
        active_marked_net_range=[float(active.incremental_net_pnl.min()),float(active.incremental_net_pnl.max())],
        same_numeric_better_accounts=int(source.incremental_net_pnl_change.gt(0).sum()),
        same_numeric_worse_accounts=int(source.incremental_net_pnl_change.lt(0).sum()),
        same_numeric_median_change=float(source.incremental_net_pnl_change.median()),
        selected_candidate=int(chosen.candidate_index), selected_marked_net=float(chosen.incremental_net_pnl),
        selected_complete_groups=int(chosen.t_pair_count), selected_paired_lots=float(chosen.completed_t_pair_lots),
        source_same_numeric_complete_groups=int(pair.t_pair_count_source),
        source_same_numeric_paired_lots=float(pair.completed_t_pair_lots_source),
        source_same_numeric_marked_net=float(pair.incremental_net_pnl_source),
        selected_same_numeric_net_change=float(change), selected_same_numeric_fee_slippage_saving=float(fee_saving),
        selected_same_numeric_reference_gross_change=float(pair.realized_fragment_reference_gross_pnl_change),
        fee_saving_share_of_change=float(fee_saving/change) if change else None,
        score_quality=population.to_dict("records"), inputs_sha256=inputs,
        no_raw_model_or_outer_read=True, no_refit_no_parameter_change=True,
        score_is_not_calibrated_conditional_mean=True,
        reduced_loss_from_less_trading_is_not_proven_timing_improvement=True)
    # 两个账户都无未完成仓位时，此分解精确成立；否则必须保留marked残仓差异。
    assert pair.pending_pairs_robust == pair.pending_pairs_source == 0
    assert abs(change-fee_saving-pair.realized_fragment_reference_gross_pnl_change) < 1e-7
    out = folder/('interpretation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    shutil.copy2(__file__,out/'summarize_smoke.py')
    lines = ["# v17 工程烟测：仅内部2025机制摘要", "",
        "完整core训练保持不变；本摘要的账户评价仅12股、15个交易日。不是全年策略结论，也没有读工程外层。",
        f"36个主动候选全部亏损，其中{result['execution_eligible_accounts']}个执行合格。完整组范围{result['complete_group_range']}，配对手范围{result['paired_lot_range']}。",
        f"与v15同数字候选逐账户比较：{result['same_numeric_better_accounts']}个净額改善、{result['same_numeric_worse_accounts']}个恶化，中位变化{result['same_numeric_median_change']:.2f}元；不是可相加组合。", "",
        f"原选择{int(chosen.candidate_index)}：Huber {int(chosen.t_pair_count)}组/{chosen.completed_t_pair_lots:.0f}手，净{chosen.incremental_net_pnl:.2f}元；对应L2 {int(pair.t_pair_count_source)}组/{pair.completed_t_pair_lots_source:.0f}手，净{pair.incremental_net_pnl_source:.2f}元。",
        f"减亏{change:.2f}元中，费用与滑点减少{fee_saving:.2f}元（{fee_saving/change:.2%}），参考毛差改善{pair.realized_fragment_reference_gross_pnl_change:.2f}元。成交大幅缩减，且新候选仍亏，不能据此宣称择时盈利能力已改善。", "",
        "|模型|原0边际及旧股机会|评分换算净边际bp|实际方向毛差bp|标称固定端点净差bp|",
        "|---|---:|---:|---:|---:|"]
    for row in population.itertuples():
        lines.append(f"|{row.profile}|{row.rows}|{row.mean_predicted_net_bp_common:.2f}|{row.mean_fixed_gross_bp:.2f}|{row.mean_fixed_nominal_net_bp:.2f}|")
    lines += ["", "这些机会全部有精确固定端点；门控没有使用未来价格。评分换算的净边际仍未被同方向价差兑现；固定端点净差仅为含原成本的标称反事实，不能等同真实账户收益。",
        "Huber是稳健位置评分，不是已经校准的条件均值。本摘要不反转方向、不改变alpha、候选、门槛或退出方式；完整检验应遵循原冻结协议。"]
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(passed=True,script_sha256=sha(__file__),inputs_sha256=inputs,
        outputs_sha256={path.name:sha(path) for path in out.iterdir() if path.is_file()})
    (out/'verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(out)


if __name__ == '__main__': main()
