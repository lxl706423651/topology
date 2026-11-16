"""仅从已终核四折内部派生表归档机制摘要，不读取市场、模型或外层。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def sha(path):
    """绑定实际读取的派生文件字节。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """保持既有选择，分别说明候选分布、总额分解与单位质量边界。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", required=True)
    folder = Path(parser.parse_args().diagnosis_dir).resolve()
    verification = json.loads((folder / "verification.json").read_text())
    assert verification["passed"] and verification["complete"] and verification["no_outer_read"]
    assert "_smoke/" not in verification["source_report"]
    assert verification["configured_folds"] == [2023, 2024, 2025, 2026]
    names = ["verification.json", "candidate_overview.csv", "candidate_source_comparison.csv", "prediction_quality.csv"]
    inputs = {str(folder / name): sha(folder / name) for name in names}
    for name in names[1:]:
        assert sha(folder / name) == verification["output_sha256"][name]
    metrics, comparison, quality = [pd.read_csv(folder / name, float_precision="round_trip") for name in names[1:]]
    failures = ["inventory_mismatch_days", "recovery_failure_days", "pending_pairs", "t1_violation_count", "base_plan_mismatch_count"]
    population = quality.loc[quality.grouping.eq("all") & quality.scope.eq("original_current_margin0bp")
        & quality.domain.eq("baseline_opening_old_inventory")]
    results = []
    for year in verification["configured_folds"]:
        m = metrics.loc[metrics.fold.eq(year)]
        c = comparison.loc[comparison.fold.eq(year)]
        assert len(m) == len(c) == 37 and m.candidate_index.nunique() == 37
        active = m.loc[m.enabled]
        same = c.loc[c.candidate_index.gt(0)]
        assert len(active) == len(same) == 36 and m.selected.sum() == 1
        chosen = m.loc[m.selected].iloc[0]
        pair = c.loc[c.candidate_index.eq(chosen.candidate_index)].iloc[0]
        saving = -pair.t_fee_change - pair.t_slippage_cost_change
        gross_change = pair.realized_fragment_reference_gross_pnl_change
        # 两个同参数账户的期末配对均完成时，总额现金流分解应精确相等。
        assert pair.pending_pairs_robust == pair.pending_pairs_source == 0
        assert abs(pair.incremental_net_pnl_change - saving - gross_change) < 1e-7
        results.append(dict(fold=year, inner_year=int(chosen.inner_year),
            positive=int(active.incremental_net_pnl.gt(0).sum()), negative=int(active.incremental_net_pnl.lt(0).sum()),
            zero=int(active.incremental_net_pnl.eq(0).sum()), execution_eligible=int(active.execution_eligible.sum()),
            groups_range=[int(active.t_pair_count.min()), int(active.t_pair_count.max())],
            paired_lots_range=[float(active.completed_t_pair_lots.min()), float(active.completed_t_pair_lots.max())],
            net_range=[float(active.incremental_net_pnl.min()), float(active.incremental_net_pnl.max())],
            failing_candidate_counts={key: int(active[key].gt(0).sum()) for key in failures},
            positive_but_execution_ineligible=active.loc[active.incremental_net_pnl.gt(0) & ~active.execution_eligible,
                ["candidate_index", "incremental_net_pnl"] + failures].to_dict("records"),
            selected=chosen[["candidate_index", "profile", "min_net_edge", "fraction", "exit_mode", "t_pair_count",
                "fragment_count", "completed_t_pair_lots", "incremental_net_pnl", "realized_complete_pair_pnl",
                "all_reference_acquisition_cost", "holding_cost_reduction_rate", "t_fee", "t_slippage_cost",
                "realized_fragment_reference_gross_pnl", "active_days"] + failures].to_dict(),
            source_same_numeric=pair.to_dict(), same_numeric_better=int(same.incremental_net_pnl_change.gt(0).sum()),
            same_numeric_worse=int(same.incremental_net_pnl_change.lt(0).sum()),
            same_numeric_median_net_change=float(same.incremental_net_pnl_change.median()),
            source_execution_eligible=int(same.execution_eligible_source.sum()),
            selected_fee_slippage_saving=float(saving), selected_reference_gross_change=float(gross_change)))
    lines = ["# v17 四折内部机制摘要", "",
        "仅复用已完成内部诊断。前三折内部评价年均为2026，第四折内部年为2025；不是四个年度外层收益，也不能将各候选或各折净额相加当作组合表现。没有读取外层、拟合或改变参数。", "",
        "|留出折/内部年|所选|完整组/片段|配对手|实际净额 元|含费底仓成本 元|成本降低率|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        s = r["selected"]
        lines.append(f"|{r['fold']}/{r['inner_year']}|{int(s['candidate_index'])}|{int(s['t_pair_count'])}/{int(s['fragment_count'])}|{s['completed_t_pair_lots']:.0f}|{s['incremental_net_pnl']:.2f}|{s['all_reference_acquisition_cost']:.2f}|{s['holding_cost_reduction_rate']:.4%}|")
    lines += ["", "所选四个账户的库存差、恢复失败、期末待恢复、T+1及底仓计划不一致均为零。仍有两折亏损，两折小幅盈利；没有一折达到5%成本降低目标。", "",
        "|留出折|36主动正/负/零|执行合格|完整组范围|配对手范围|净额范围 元|库存/恢复/期末待恢复失败候选数|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        f = r["failing_candidate_counts"]
        lines.append(f"|{r['fold']}|{r['positive']}/{r['negative']}/{r['zero']}|{r['execution_eligible']}|{r['groups_range'][0]}–{r['groups_range'][1]}|{r['paired_lots_range'][0]:.0f}–{r['paired_lots_range'][1]:.0f}|{r['net_range'][0]:.2f}–{r['net_range'][1]:.2f}|{f[failures[0]]}/{f[failures[1]]}/{f[failures[2]]}|")
    lines += ["", "以上失败计数是分别独立的候选账户数量，不是可相加的组合失败天数。正收益但执行不合格的候选仍完整保留于JSON。", "",
        "|留出折|同数字所选 L2→Huber 组数/手数|净额变化 元|总费用及滑点减少 元|总参考毛差变化 元|36候选净改善/变差|",
        "|---|---|---:|---:|---:|---:|"]
    for r in results:
        s, p = r["selected"], r["source_same_numeric"]
        lines.append(f"|{r['fold']}|{int(p['t_pair_count_source'])}/{p['completed_t_pair_lots_source']:.0f}→{int(s['t_pair_count'])}/{s['completed_t_pair_lots']:.0f}|{p['incremental_net_pnl_change']:.2f}|{r['selected_fee_slippage_saving']:.2f}|{r['selected_reference_gross_change']:.2f}|{r['same_numeric_better']}/{r['same_numeric_worse']}|")
    lines += ["", "同数字规则保持原门槛、份额、退出和费用，但模型评分、真实信号及成交规模已改变。总额分解能说明减费对净额的贡献；不能由总毛差下降推断单笔择时变差。源L2和Huber单位名义交易质量尚未在本摘要以共同分母核验，不作方向性结论。", "",
        "|留出折/模型|原0净边际当前旧股机会/有效端点|评分换算净差 bp|方向毛差 bp|标称固定净差 bp|评分乐观偏差 bp|",
        "|---|---:|---:|---:|---:|---:|"]
    for r in population.itertuples():
        lines.append(f"|{r.fold}/{r.profile}|{r.rows}/{r.valid_endpoint_rows}|{r.mean_predicted_net_bp_common:.2f}|{r.mean_fixed_gross_bp:.2f}|{r.mean_fixed_nominal_net_bp:.2f}|{r.mean_net_optimism_bp:.2f}|")
    lines += ["", "机会由当前条件决定，未来端点仅用于误差描述，缺价计数保留于原派生表。固定20/40/60分钟价差使用原精确费税和滑点，属于标称反事实，不保证现金、旧股额度、容量或实际回购；真实动态退出账户净額另列。Huber是稳健位置评分，并非校准条件均值。", "不同日、周只表示不同日期数量；重复使用历史数据且前三折共用内部2026，不构成四份独立盈利证据。此摘要不产生新选择。"]
    out = folder / ("interpretation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = dict(passed=True, scope="four_fold_internal_only", source_report=verification["source_report"],
        results=results, score_quality=population.to_dict("records"), inputs_sha256=inputs,
        no_raw_model_or_outer_read=True, no_refit_or_rule_change=True, per_notional_source_quality_not_verified=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    assert all(sha(path) == digest for path, digest in inputs.items())
    manifest = dict(passed=True, script_sha256=sha(__file__), inputs_sha256=inputs,
        outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out / "verification.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
