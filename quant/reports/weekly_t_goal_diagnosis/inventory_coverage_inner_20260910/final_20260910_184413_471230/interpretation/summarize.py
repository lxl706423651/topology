"""只汇总完整内部诊断的冻结派生表，记录精确分母与阅读SHA；不读外层或执行策略。"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent
inputs = {}


def sha(path):
    """实际内容SHA用于绑定已完成诊断结果。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name, csv=True):
    """只读本次完整内部诊断根下指定表，不追索生产账户或市场。"""
    path = SOURCE/name
    inputs[str(path)] = sha(path)
    return pd.read_csv(path, float_precision="round_trip") if csv else json.loads(path.read_text())


verification = read("verification.json", False)
overview = read("candidate_overview.csv")
comparison = read("candidate_source_comparison.csv")
quality = read("prediction_quality.csv")
unit = read("selected_complete_notional_comparison.csv")
models = read("model_training_summary.csv")
active = overview.loc[overview.enabled]
annual = []
for fold, a in active.groupby("fold"):
    b = comparison.loc[comparison.fold.eq(fold) & comparison.candidate_index.gt(0)]
    selected = next(r["selected_account"] for r in verification["results"] if r["fold"] == fold)
    annual.append(dict(fold=int(fold), inner_year=int(a.inner_year.iloc[0]), active_accounts=len(a),
        positive_marked_increment_accounts=int(a.incremental_net_pnl.gt(0).sum()),
        negative_marked_increment_accounts=int(a.incremental_net_pnl.lt(0).sum()),
        zero_marked_increment_accounts=int(a.incremental_net_pnl.eq(0).sum()),
        execution_eligible_accounts=int(a.execution_eligible.sum()),
        complete_group_min=int(a.t_pair_count.min()), complete_group_max=int(a.t_pair_count.max()),
        paired_lot_min=float(a.completed_t_pair_lots.min()), paired_lot_max=float(a.completed_t_pair_lots.max()),
        marked_increment_min=float(a.incremental_net_pnl.min()), marked_increment_max=float(a.incremental_net_pnl.max()),
        inventory_failure_accounts=int(a.inventory_mismatch_days.gt(0).sum()),
        recovery_failure_accounts=int(a.recovery_failure_days.gt(0).sum()), pending_accounts=int(a.pending_pairs.gt(0).sum()),
        t1_failure_accounts=int(a.t1_violation_count.gt(0).sum()), base_plan_failure_accounts=int(a.base_plan_mismatch_count.gt(0).sum()),
        same_numeric_improved=int(b.incremental_net_pnl_change.gt(0).sum()),
        same_numeric_worse=int(b.incremental_net_pnl_change.lt(0).sum()),
        same_numeric_equal=int(b.incremental_net_pnl_change.eq(0).sum()),
        same_numeric_change_median=float(b.incremental_net_pnl_change.median()),
        selected_candidate_index=selected["candidate_index"], selected_complete_net=selected["complete_net"],
        selected_marked_increment=selected["marked_increment"],
        all_original_fee_inclusive_base_cost=selected["cost_denominator"],
        selected_cost_reduction_rate=selected["holding_cost_reduction_rate"],
        selected_complete_groups=selected["complete_groups"], selected_paired_lots=selected["paired_lots"]))
q = quality.loc[quality.grouping.eq("all") & quality.scope.eq("original_current_margin0bp")
    & quality.domain.eq("baseline_opening_old_inventory")]
fields = ["fold", "inner_year", "profile", "rows", "valid_endpoint_rows", "mean_predicted_net_bp_common",
          "mean_fixed_gross_bp", "mean_fixed_nominal_net_bp", "mean_net_optimism_bp", "nominal_cost_covered_rate"]
summary = dict(scope="all_four_completed_internal_folds_not_outer", years_are_fold_labels_not_independent_inner_years=True,
    annual_candidate_summary=annual, total_overlapping_active_accounts=len(active),
    total_positive_marked_increment_accounts=int(active.incremental_net_pnl.gt(0).sum()),
    total_negative_marked_increment_accounts=int(active.incremental_net_pnl.lt(0).sum()),
    positive_and_execution_eligible_accounts=int((active.incremental_net_pnl.gt(0) & active.execution_eligible).sum()),
    current_old_inventory_zero_margin_score_quality=q[fields].to_dict("records"),
    selected_complete_notional_comparison=json.loads(unit.to_json(orient="records", double_precision=15)),
    model_features_described=models[["fold", "profile", "horizon_minutes", "features", "source_features",
        "added_features", "removed_features"]].to_dict("records"),
    complete_diagnostic_input_count=len(verification["inputs_sha256"]), production_sources_verified=verification["sources_verified"],
    no_outer_read=True, no_new_fit_or_execution=True, no_parameter_change=True,
    original_base_cost_denominator_preserved=True, same_numeric_unit_comparison_uses_own_complete_pairs=True,
    inputs_sha256=inputs, diagnostic_verification_sha256=sha(SOURCE/"verification.json"))
pd.DataFrame(annual).to_csv(OUT/"candidate_summary.csv", index=False, encoding="utf-8-sig")
q[fields].to_csv(OUT/"score_quality_summary.csv", index=False, encoding="utf-8-sig")
(OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
lines = ["# v18覆盖门槛研究：完整内部诊断", "",
    "本轮内部证据不支持稳定获得5%额外利润。四折最终选择hold / 15 / hold / hold；正式外层未执行。144个主动候选中4个账面含残仓增量为正、140个为负，正数候选均有执行失败。候选数只是重叠账户的描述计数，不能将其收益相加。", "",
    "前三个留出标签2023/2024/2025均在内部2026年评价，最后2026留出标签的内部年是2025。2026行情只到4月30日；以下不是四个独立完整年度验证收益，也不作年化5%过线。", "",
    "|留出折/内部年|主动正/负|执行合格|完整组范围|配对手范围|库存失败/恢复失败账户|期末pending账户|已选规则|已选完整净额 元|全部原始含费底仓成本 元|成本降低率|",
    "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
for r in annual:
    lines.append(f"|{r['fold']}/{r['inner_year']}|{r['positive_marked_increment_accounts']}/{r['negative_marked_increment_accounts']}|"
        f"{r['execution_eligible_accounts']}|{r['complete_group_min']}–{r['complete_group_max']}|"
        f"{r['paired_lot_min']:.0f}–{r['paired_lot_max']:.0f}|{r['inventory_failure_accounts']}/{r['recovery_failure_accounts']}|"
        f"{r['pending_accounts']}|{r['selected_candidate_index']}|{r['selected_complete_net']:.2f}|"
        f"{r['all_original_fee_inclusive_base_cost']:.2f}|{r['selected_cost_reduction_rate']:.4%}|")
lines += ["", "全部候选T1与底仓计划差异计数为0；日内恢复/库存失败和期末pending另列，不能因T1通过或hold没有成交而写成全部执行目标已满足。每手100股，配对手按完整首腿仅计一次；它不等于双腿成交手数或恢复片段数。", "",
    "唯一主动选中为2024留出折的15号规则：378完整组、443恢复片段、3597配对手，参考毛差4256.00−滑点2609.843−T费用5533.27644357＝净−3887.11944357元。其五项执行失败为0，成本降低率−0.46167844%；总净值收益及持仓涨跌不混入T降低成本分子。其余三折hold没有T成交，单位交易质量为空，不是100%胜率。", "",
    "相同数字参数与v15的比较：", "",
    "|留出折|净额改善/恶化候选|净额变化中位数 元|", "|---|---:|---:|"]
for r in annual:
    lines.append(f"|{r['fold']}|{r['same_numeric_improved']}/{r['same_numeric_worse']}|{r['same_numeric_change_median']:.2f}|")
lines += ["", "两轮原参考标签、L2损失、费用、37规则与首腿复查合同保持，覆盖筛选由0.80改为0.70。保存的24模型清单均由30列到33列，仅增加三个15分钟个股/市场/超额收益列且无移除。清单事实和局部改善不证明新增列稳定有效。此工具读取训练身份证明，没有重建完整训练矩阵或重训；正式模型审计另行负责。", "",
    "本轮已选15与v15同数字15使用各自完整组原首腿参考金额归一化：本轮2610181元、v15为2665515元。对应参考毛差16.3054/9.0602bp，实际费滑31.1975/31.1983bp，完整净差−14.8921/−22.1382bp。本轮单位净差高7.2460bp、总完整净额高2013.85元，但仍亏损。两边成交集合不同，且首腿参考金额不是全部底仓成本，不得拿该较小/循环成交分母替代5%目标，也不能外推另外三折。", "",
    "原0净边际、当前价量与时钟门控后，基准日初有旧股的机会（不是实际可成交保证）：", "",
    "|留出折|模型|机会数/有效固定端点|预测净差 bp|固定端点方向毛差 bp|标称扣费净差 bp|乐观差 bp|标称成本覆盖率|",
    "|---|---|---:|---:|---:|---:|---:|---:|"]
for r in q.itertuples():
    lines.append(f"|{r.fold}|{r.profile}|{r.rows}/{r.valid_endpoint_rows}|{r.mean_predicted_net_bp_common:.2f}|"
        f"{r.mean_fixed_gross_bp:.2f}|{r.mean_fixed_nominal_net_bp:.2f}|{r.mean_net_optimism_bp:.2f}|{r.nominal_cost_covered_rate:.2%}|")
lines += ["", "全部八个模型/内部域平均标称扣费净差为负。前三折平均方向毛差虽为正，仍小于原约31bp来回费滑；最后内部2025年平均方向毛差本身也为负。预测净差普遍乐观27.75–61.31bp。这些固定端点结果没有容量、现金与真实分片恢复保证，不是账户净额或真实成交胜率；不根据这里的未来诊断结果改信号、阈值或剔除股票。", "",
    "首折全部主动失败有具体执行路径：[三代表事件说明](../../fold_2023/run_20260910_182937_945796/recovery_attribution/interpretation.md)。36候选共187个跨日/未完成首腿，来自21个股票×首腿日；三个事件并集覆盖全部36。两例恢复方向达到9.8%代理阈值，一例为精确5%flat代理且有大量零量。5%代理不能当作真实ST认定；首腿当刻反向检查通过不能保证之后恢复。全部事件和其余未逐一复核的失败均保留，不设事后黑名单。", "",
    "完整阶段各执行一次，最终只合并已有阶段并复核脚本、输入和输出SHA。1332实际输入与56生产源码始末一致，无外层读取、模型拟合、账户重放或参数修改。此次是历史重复研究；该诊断通过只表明读取边界/算术/保存身份一致，不是盈利目标验收。", "",
    "证据：[原完整诊断](../diagnosis.md)、[原verification](../verification.json)、[同数字比较](../candidate_source_comparison.csv)、[完整首腿金额比较](../selected_complete_notional_comparison.csv)、[全部原失败行](../candidate_failures.csv)。"]
(OUT/"summary.md").write_text("\n".join(lines)+"\n")
shutil.copy2("/tmp/inventory_coverage_inner_full_final_v18.log", OUT/"final_command.log")
assert all(sha(path) == h for path, h in inputs.items())
(OUT/"verification.json").write_text(json.dumps(dict(passed=True, inputs_unchanged=True, inputs_sha256=inputs,
    final_command_returncode=0, final_terminal_session=13408, stages_not_rerun=True,
    outputs_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}), ensure_ascii=False, indent=2)+"\n")
print(json.dumps(dict(directory=str(OUT), verification_sha256=sha(OUT/"verification.json")), ensure_ascii=False))
