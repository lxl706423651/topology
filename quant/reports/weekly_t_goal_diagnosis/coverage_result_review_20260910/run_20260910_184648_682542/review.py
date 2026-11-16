"""只读v18完整内部候选及既存诊断，审核结果表述；不读取外层、不拟合或重放。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

FAIL = ["inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
        "t1_violation_count", "base_plan_mismatch_count"]
RULE = ["candidate_index", "enabled", "profile", "min_net_edge", "fraction", "exit_mode"]
METRICS = ["incremental_net_pnl", "holding_cost_reduction_rate", "t_pair_count", "completed_t_pair_lots",
           "t_fee", "t_slippage_cost", "inventory_mismatch_days", "recovery_failure_days",
           "pending_pairs", "all_reference_acquisition_cost"]


def sha(path):
    """真实字节摘要；不使用stat代替内容身份。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """仅消费本报告已结束内部文件、v15同规则指标和完整只读诊断。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--diagnosis-dir", required=True)
    args = parser.parse_args()
    report, diagnosis = Path(args.report_dir).resolve(), Path(args.diagnosis_dir).resolve()
    inputs = {}
    def record(path):
        path = Path(path).resolve()
        if path.is_relative_to(report):
            r = path.relative_to(report)
            allowed = r.as_posix() in ("resolved_config.json", "runtime_integrity_audit.json", "source_hashes.json", "outer_evaluation_decision.json")
            allowed |= len(r.parts) == 2 and r.parts[0] in [f"year_{y}" for y in (2023, 2024, 2025, 2026)] and r.name in (
                "split.json", "selected_rule.json", "validation_rule_metrics.csv", "coverage_vs_source_candidates.csv")
            assert allowed, "禁止读取本报告外层或其它文件"
        elif path.is_relative_to(diagnosis):
            assert path.name in ("verification.json", "diagnosis.md", "prediction_quality.csv", "selected_complete_notional_comparison.csv")
        elif path != Path(__file__).resolve():
            assert path.is_relative_to(source) and path.name in ("selected_rule.json", "validation_rule_metrics.csv"), "禁止读取源外层账户"
        value = sha(path)
        assert str(path) not in inputs or inputs[str(path)] == value
        inputs[str(path)] = value
        return path
    def read_json(path):
        return json.loads(record(path).read_text())
    def read_csv(path):
        return pd.read_csv(record(path), float_precision="round_trip")
    cfg = read_json(report / "resolved_config.json")
    source = Path(cfg["source_execution_report_dir"]).resolve()
    terminal = read_json(report / "runtime_integrity_audit.json")
    source_hashes = read_json(report / "source_hashes.json")
    evaluation = read_json(report / "outer_evaluation_decision.json")
    assert cfg["research_version"] == "inventory_coverage_t_v18" and cfg["feature_min_coverage"] == .7
    assert cfg["source_feature_min_coverage"] == .8 and cfg["test_years"] == [2023, 2024, 2025, 2026]
    assert terminal["all_unchanged"] and terminal["runtime_sources_verified"] == len(source_hashes) == 56
    assert evaluation["outer_evaluation_started"] is False and evaluation["all_inner_cost_targets_met"] is False
    for year in cfg["test_years"]:
        assert not (report / f"year_{year}/test_equity_curve.csv").exists()
        assert not (report / f"year_{year}/test_predictions").exists()
    diagnostic = read_json(diagnosis / "verification.json")
    assert diagnostic["passed"] and diagnostic["complete"] and diagnostic["source_report"] == str(report)
    assert diagnostic["no_outer_read"] and diagnostic["no_refit_calibration_or_parameter_change"]
    for filename in ("diagnosis.md", "prediction_quality.csv", "selected_complete_notional_comparison.csv"):
        assert sha(record(diagnosis / filename)) == diagnostic["output_sha256"][filename]
    summaries, paired, active_rows, positive_rows = [], [], [], []
    for year in cfg["test_years"]:
        directory = report / f"year_{year}"
        split = read_json(directory / "split.json")
        selected = read_json(directory / "selected_rule.json")
        current = read_csv(directory / "validation_rule_metrics.csv")
        previous = read_csv(source / f"year_{year}/validation_rule_metrics.csv")
        source_choice = read_json(source / f"year_{year}/selected_rule.json")["rule"]
        comparison = read_csv(directory / "coverage_vs_source_candidates.csv")
        assert current.candidate_index.tolist() == previous.candidate_index.tolist() == list(range(37))
        pd.testing.assert_frame_equal(current[RULE], previous[RULE], check_exact=True)
        for key in METRICS:
            for suffix, values in (("coverage", current[key]), ("source", previous[key]), ("change", current[key]-previous[key])):
                np.testing.assert_array_equal(comparison[key+"_"+suffix].to_numpy(), values.to_numpy())
        active = current.loc[current.enabled].copy()
        eligible = active.t_pair_count.gt(0) & active[FAIL].eq(0).all(axis=1)
        np.testing.assert_array_equal(active.execution_eligible.to_numpy(), eligible.to_numpy())
        expected = active.loc[eligible].sort_values(["incremental_net_pnl", "candidate_index"], ascending=[False, True])
        expected_index = int(expected.candidate_index.iloc[0]) if len(expected) else 0
        assert selected["rule"]["candidate_index"] == expected_index
        row = current.loc[current.candidate_index.eq(expected_index)].iloc[0]
        assert all(selected["rule"][key] == row[key] for key in RULE)
        assert np.isclose(row.incremental_net_pnl, selected["inner_metrics"]["incremental_net_pnl"], rtol=0, atol=1e-9)
        summary = dict(fold=year, inner_year=split["inner_year"], selected_index=expected_index,
            source_selected_index=source_choice["candidate_index"], selected_net=float(row.incremental_net_pnl),
            selected_cost_rate=float(row.holding_cost_reduction_rate), selected_groups=int(row.t_pair_count),
            selected_pair_lots=float(row.completed_t_pair_lots), selection_reason=selected["inner_metrics"]["selection_reason"],
            active_candidates=len(active), actually_traded_candidates=int(active.t_pair_count.gt(0).sum()),
            execution_eligible_candidates=int(eligible.sum()), positive_marked_net_candidates=int(active.incremental_net_pnl.gt(0).sum()),
            **{key+"_failing_candidates": int(active[key].gt(0).sum()) for key in FAIL})
        for key in ("t_pair_count", "completed_t_pair_lots", "incremental_net_pnl", "holding_cost_reduction_rate", "t_fee", "t_slippage_cost"):
            summary[key+"_min"], summary[key+"_max"] = float(active[key].min()), float(active[key].max())
        summaries.append(summary)
        for key in ("incremental_net_pnl", "completed_t_pair_lots", "t_pair_count", "t_fee", "t_slippage_cost"):
            delta = comparison.loc[comparison.candidate_index.gt(0), key+"_change"]
            paired.append(dict(fold=year, metric=key, higher=int(delta.gt(0).sum()), lower=int(delta.lt(0).sum()),
                               equal=int(delta.eq(0).sum()), median_change=float(delta.median()), non_additive_description=True))
        active_rows.append(active[[*RULE, *METRICS, "realized_complete_pair_pnl", *[k for k in FAIL if k not in METRICS]]].assign(fold=year))
        positive_rows.append(active.loc[active.incremental_net_pnl.gt(0), ["candidate_index", "incremental_net_pnl", *FAIL]].assign(fold=year))
    quality = read_csv(diagnosis / "prediction_quality.csv")
    selected_quality = quality.loc[quality.grouping.eq("all") & quality.scope.eq("original_current_margin0bp")
                                  & quality.domain.eq("baseline_opening_old_inventory")]
    assert len(selected_quality) == 8 and selected_quality.mean_fixed_nominal_net_bp.lt(0).all()
    units = read_csv(diagnosis / "selected_complete_notional_comparison.csv")
    assert units.candidate_index.tolist() == [0, 15, 0, 0]
    conclusions = [
        "最终内部选择是0/15/0/0：2023、2025、2026留出折的36个主动候选全部实际成交，却全部存在库存差和恢复失败，故没有执行合格候选；hold的0不是主动策略收益改善。2024折仅4个候选执行合格且均亏，选择15：378完整组、3597配对手、净−3887.12元，原始底仓成本降低率−0.4617%。",
        "频率变化并不一致：四折同数字规则的手数增加候选数为24/14/21/31（各36个），净额改善数仅17/32/1/2；对应净额变化中位数−375.45/+1996.29/−3478.22/−8779.62元。新增交易没有形成跨折稳定净增益，候选分布不能相加成一个组合。第一折4个正净额候选也均违反恢复要求，最高完整组成本降低率仅约0.5196%，不能当作达标。",
        "既存内部误差诊断仍显示成本覆盖不足：原0净边际、基准旧股机会下，8个模型/内部折的预测净差均值约+13.39至+19.48bp，而同方向固定端点标称净差均值全部为负（−42.03至−14.35bp）；标称成本覆盖率约35.55%至40.16%。这属于机会级参考端点诊断，不是实际账户成交收益，但不支持已经解决预测乐观问题。",
        "存在局部改善，应如实保留：2024折两轮均选择数字规则15，新/旧完整组378/381，单位首腿参考名义金额的毛差16.31/9.06bp，费税滑点约31.20/31.20bp，净差−14.89/−22.14bp；总净額从−5900.97改善至−3887.12元。因此不能把该处改善全部归为少交易省费，也不能由总毛额变化否定单位质量。但它仍亏损，且只覆盖这一内部样本。",
        "同数字规则对照控制了原37规则参数、费用与底仓，但新旧预测及实际成交集合不同；只强调当前内部选中的规则会保留选择乐观偏差。全37组对照可呈现异质性，仍不能替代独立样本。前三留出折内部年同为2026，最后为2025；本轮未执行正式外层，不能称2023—2026四年度验证结果，更不能把多个候选或内部折净额相加。"]
    output = Path(__file__).resolve().parent / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(exist_ok=False)
    pd.DataFrame(summaries).to_csv(output / "fold_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired).to_csv(output / "same_rule_change_distribution.csv", index=False, encoding="utf-8-sig")
    pd.concat(active_rows, ignore_index=True).to_csv(output / "all_144_active_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(positive_rows, ignore_index=True).to_csv(output / "positive_candidates_execution_failures.csv", index=False, encoding="utf-8-sig")
    selected_quality.to_csv(output / "fixed_endpoint_quality_evidence.csv", index=False, encoding="utf-8-sig")
    units.to_csv(output / "selected_rule_unit_notional_evidence.csv", index=False, encoding="utf-8-sig")
    links = [str(report / f"year_{year}/selected_rule.json") for year in cfg["test_years"]]
    lines = ["# v18完整内部结果的独立表述复核", "", "本复核仅重算已保存的小表与既存内部诊断；完整原行情、模型和账户独审由另一审计流程负责。", ""]
    lines.extend(f"{i}. {item}\n" for i, item in enumerate(conclusions, 1))
    lines += ["证据：", "", f"- [正式内部结果目录]({report})", f"- [完整内部诊断]({diagnosis / 'diagnosis.md'})",
              "- 当前目录 fold_summary.csv、same_rule_change_distribution.csv、positive_candidates_execution_failures.csv 保留上述计数与区间。",
              "- fixed_endpoint_quality_evidence.csv 与 selected_rule_unit_notional_evidence.csv 为已核SHA的既存诊断摘录，未重新读取原始分钟。"]
    (output / "review.md").write_text("\n".join(lines)+"\n")
    record(__file__)
    shutil.copy2(__file__, output / "review.py")
    assert all(sha(path) == value for path, value in inputs.items()), "复核期间输入变化"
    result = dict(passed=True, complete=True, report=str(report), source_diagnosis=str(diagnosis),
        conclusions=conclusions, folds=summaries, paired_distributions=paired,
        no_outer_read=True, no_model_fit=True, no_account_run=True, no_parameter_change=True,
        complete_raw_model_account_audit_not_repeated=True, reported_runtime_integrity=terminal,
        input_sha256=inputs, all_read_inputs_unchanged=True, script_sha256=sha(__file__),
        outputs_sha256={path.name: sha(path) for path in output.iterdir() if path.is_file()})
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True, output=str(output), input_count=len(inputs),
        verification_sha256=sha(output / "verification.json")), ensure_ascii=False))


if __name__ == "__main__":
    main()
