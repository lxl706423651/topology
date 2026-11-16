"""只汇总冻结的内部诊断表，归档终端日志；不读取外层、原模型或原行情。"""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent


def sha(path):
    """记录本次实际读取派生表的内容身份。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


names = ["verification.json", "candidate_overview.csv", "candidate_source_comparison.csv",
         "prediction_quality.csv", "selected_complete_notional_comparison.csv", "model_training_summary.csv"]
inputs = {str(SOURCE/name): sha(SOURCE/name) for name in names}
verification = json.loads((SOURCE/"verification.json").read_text())
active = pd.read_csv(SOURCE/"candidate_overview.csv", float_precision="round_trip")
active = active.loc[active.enabled]
comparison = pd.read_csv(SOURCE/"candidate_source_comparison.csv", float_precision="round_trip")
comparison = comparison.loc[comparison.candidate_index.gt(0)]
quality = pd.read_csv(SOURCE/"prediction_quality.csv", float_precision="round_trip")
quality = quality.loc[quality.grouping.eq("all") & quality.scope.eq("original_current_margin0bp")
                      & quality.domain.eq("baseline_opening_old_inventory")]
unit = pd.read_csv(SOURCE/"selected_complete_notional_comparison.csv", float_precision="round_trip").iloc[0]
error = active.realized_fragment_reference_gross_pnl-active.t_slippage_cost-active.t_fee-active.incremental_net_pnl
summary = dict(scope="completed_smoke_inner_only", fold=2026, inner_year=2025, outer_read=False,
    active_accounts=len(active), positive=int(active.incremental_net_pnl.gt(0).sum()),
    negative=int(active.incremental_net_pnl.lt(0).sum()), zero=int(active.incremental_net_pnl.eq(0).sum()),
    execution_eligible=int(active.execution_eligible.sum()),
    complete_group_range=[int(active.t_pair_count.min()), int(active.t_pair_count.max())],
    paired_lot_range=[float(active.completed_t_pair_lots.min()), float(active.completed_t_pair_lots.max())],
    inventory_failure_accounts=int(active.inventory_mismatch_days.gt(0).sum()),
    recovery_failure_accounts=int(active.recovery_failure_days.gt(0).sum()), pending_accounts=int(active.pending_pairs.gt(0).sum()),
    same_numeric_net_improved=int(comparison.incremental_net_pnl_change.gt(0).sum()),
    same_numeric_net_worse=int(comparison.incremental_net_pnl_change.lt(0).sum()),
    same_numeric_net_equal=int(comparison.incremental_net_pnl_change.eq(0).sum()),
    same_numeric_net_change_median=float(comparison.incremental_net_pnl_change.median()),
    all_saved_account_decomposition_max_error=float(error.abs().max()),
    all_saved_decompositions_match=bool(error.abs().le(1e-6).all()),
    selected_account=verification["results"][0]["selected_account"], selected_complete_notional_comparison=unit.to_dict(),
    current_old_inventory_score_quality=quality[["profile", "rows", "valid_endpoint_rows", "mean_predicted_net_bp_common",
        "mean_fixed_gross_bp", "mean_fixed_nominal_net_bp", "mean_net_optimism_bp", "nominal_cost_covered_rate"]].to_dict("records"),
    training_manifest_description=pd.read_csv(SOURCE/"model_training_summary.csv")[["profile", "horizon_minutes", "features",
        "source_features", "added_features", "removed_features"]].to_dict("records"),
    inputs_sha256=inputs, inputs_unchanged=all(sha(p) == h for p, h in inputs.items()),
    diagnostic_verification_sha256=sha(SOURCE/"verification.json"))
assert summary["inputs_unchanged"] and summary["all_saved_decompositions_match"]
(OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
text = f"""# v18 烟测内部结果解释

这里只读2026留出折的内部2025年及v15同折同数字控制，未读工程外层，未重训或重跑账户。诊断一次成功，37候选、6个模型的保存预测/元数据、384实际输入及56源开结身份通过。完整训练矩阵未重建，本工具不替代模型与执行独立审计。

36主动账户全部亏损，24个执行合格；完整组24—212、配对手280—2179。12个账户有库存差异和恢复失败，期末pending均为0。同数字规则相对v15净额改善12个、恶化24个、中位变化−663.55元。这些候选共用行情和底仓，不能相加为组合业绩。

已选15的35完整组/368配对手：原始参考毛差84.00元−实际滑点230.691元−T手续费489.02820273元＝净−635.71920273元。执行价gross_pnl已含滑点，先加回得到参考毛差，再扣一次滑点及费用。原始含费底仓成本172266.443386605元未变，成本降低率−0.3690325%，不是总净值收益率。此选中账户五项执行失败为0，不能代表其他候选无失败。

同数字v15候选15为31完整组、净−1194.81470528元。本轮/原控制的完整pair原首腿参考金额分别230617/198656元；每组仅计原首腿一次，分片不重复计。

|口径|v18|v15同数字规则|
|---|---:|---:|
|参考毛差/上述金额 bp|3.64|−28.99|
|实际费滑/上述金额 bp|31.21|31.15|
|完整组净额/上述金额 bp|−27.57|−60.14|

该规则的单位金额毛差、净差相对控制更好，但仍不足覆盖成本；这是各自真实成交集合的描述，未使用共同首腿交集。它不是5%目标分母，也不能由一个候选和小范围工程烟测推断稳定择时改善。

原0边际当前门控与日初旧股机会中，depth4/depth6分别272/324行。平均预测净差+16.51/+21.41bp，对应固定20/40/60分钟端点实际方向毛差−19.70/−19.77bp，标称扣费净差−50.93/−51.00bp，乐观偏差67.44/72.41bp；标称覆盖成本比例35.29%/36.73%。这是当前可知门控后全部旧股预测机会，未假定现金、实际旧股余额或容量可满足，不能替代实际组合净额，也不等于真实执行胜率。

保存的6模型清单均由30增到33列，新增三个15分钟个股/市场/超额收益列且无移除；这只是清单事实，不构成新增特征有效性的证据。当前结果未通过盈利目标；完整年度研究尚不由本内部烟测诊断代表。所有分箱/比较均为事后描述，不用于本轮规则修改。

证据：[原诊断](../diagnosis.md)、[原验收](../verification.json)、[单位金额明细](../selected_complete_notional_comparison.csv)。当前派生表净额分解最大误差{summary['all_saved_account_decomposition_max_error']:.3g}元；这是保存账表核对，未重放实际账户。
"""
(OUT/"summary.md").write_text(text)
shutil.copy2("/tmp/inventory_coverage_inner_smoke_v18.log", OUT/"diagnostic_command.log")
proof = dict(passed=True, command="python -u reports/weekly_t_goal_diagnosis/inventory_coverage_inner_20260910/diagnose.py --report-dir reports/inventory_coverage_t_smoke/run_20260910_181434_954042 --final",
    returncode=0, terminal_session=20580, inputs_sha256=inputs,
    outputs_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
(OUT/"verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2)+"\n")
print(OUT)
