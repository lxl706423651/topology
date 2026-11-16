"""仅核已完成v18烟测六个核心模型的原生树使用和NaN路径，不读取评价预测或账户。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
FEATURES = ("f_return_15m", "f_timing_market_return_15m", "f_timing_excess_return_15m")
INPUT_KEYS = ["datetime", "symbol", "feature_source_start", "cohort_target_year", "baseline_opening_shares"]
VERSION = "inventory_coverage_t_v18"


def require(value, message):
    """身份或分裂路径不符立即停止，不输出成功结论。"""
    if not value:
        raise AssertionError(message)


def sha(path):
    """流式计算真实文件字节摘要。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def traverse_tree(tree, matrix, sample_ids, feature_names, tree_index):
    """从已保存树逐行走核心输入，统计实际到达新增字段NaN分支的次数。

仅支持本轮数值分裂。None缺失类型按原生预测约定将NaN视为0；NaN缺失类型
走default_left。随后以原生pred_leaf的核心样本逐树核路径，不计算预测收益。
    """
    leaves = np.full(len(sample_ids), -1, dtype=np.int32)
    sample_positions = np.full(len(matrix), -1, dtype=np.int32)
    sample_positions[sample_ids] = np.arange(len(sample_ids))
    seen_missing = {f: np.zeros(len(matrix), dtype=bool) for f in FEATURES}
    records = []
    stack = [(tree, np.arange(len(matrix), dtype=np.int64))]
    while stack:
        node, rows = stack.pop()
        if "split_index" not in node:
            positions = sample_positions[rows]
            keep = positions >= 0
            leaves[positions[keep]] = node["leaf_index"]
            continue
        feature = feature_names[int(node["split_feature"])]
        require(node["decision_type"] == "<=", "本轮不应出现类别分裂")
        missing_type = node["missing_type"]
        require(missing_type in ("None", "NaN"), "出现本轮未声明的零值缺失类型")
        values = matrix[rows, int(node["split_feature"])].astype(np.float64, copy=True)
        missing = np.isnan(values)
        if missing_type == "None":
            values[missing] = 0.
        left = values <= float(node["threshold"])
        if missing_type == "NaN":
            left[missing] = bool(node["default_left"])
        if feature in FEATURES:
            missing_rows = rows[missing]
            seen_missing[feature][missing_rows] = True
            records.append(dict(tree_index=tree_index, split_index=node["split_index"], feature=feature,
                split_gain=float(node["split_gain"]), threshold=float(node["threshold"]),
                missing_type=missing_type, default_left=bool(node["default_left"]),
                all_core_rows_reaching_node=len(rows), nan_core_rows_reaching_node=len(missing_rows),
                nan_rows_routed_left=int((missing & left).sum()), nan_rows_routed_right=int((missing & ~left).sum()),
                training_bagged_internal_count=node["internal_count"]))
        stack.append((node["right_child"], rows[~left]))
        stack.append((node["left_child"], rows[left]))
    require((leaves >= 0).all(), "核心样本有未到达叶子的行")
    return leaves, records, seen_missing


def main():
    """只读六个模型目录并新增证据，全部输入SHA在结束时重新核对。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    start = time.time()
    report = Path(args.report_dir).resolve()
    output = Path(__file__).resolve().parent / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(exist_ok=False)
    inputs = {}
    def record(path):
        path = Path(path).resolve()
        if path.is_relative_to(report):
            relative = path.relative_to(report)
            allowed = relative.as_posix() in ("resolved_config.json", "runtime_integrity_audit.json", "source_hashes.json")
            allowed |= relative.parts[0] == "source_snapshot"
            allowed |= len(relative.parts) == 4 and relative.parts[1] == "models" and relative.name in (
                "model_metadata.json", "return_model.txt", "training_inputs.pkl")
            require(allowed, "禁止打开非核心模型或评价产物：" + str(path))
        else:
            require(path == Path(__file__).resolve() or path.is_relative_to(ROOT / "quant_backtest")
                or path.is_relative_to(ROOT / "scripts") or path.is_relative_to(ROOT / "config"), "非授权输入范围")
        value = sha(path)
        require(str(path) not in inputs or inputs[str(path)] == value, "已读取文件发生变化")
        inputs[str(path)] = value
        return path
    def read_json(path):
        return json.loads(record(path).read_text())
    cfg = read_json(report / "resolved_config.json")
    terminal = read_json(report / "runtime_integrity_audit.json")
    require(cfg["research_version"] == VERSION and cfg["report_scope"] == "smoke", "只读取本轮已完成工程烟测")
    require(cfg["test_years"] == [2026] and cfg["feature_min_coverage"] == .7, "本轮模型范围错误")
    require(terminal["all_unchanged"] is True and terminal["model_artifacts_verified"] == 60, "模型尚未全部终态")
    source_hashes = read_json(report / "source_hashes.json")
    require(len(source_hashes) == terminal["runtime_sources_verified"] == 56, "冻结源码计数不符")
    for relative, expected in source_hashes.items():
        for path in (ROOT / relative, report / "source_snapshot" / relative):
            require(sha(record(path)) == expected, "冻结源码或快照已变：" + relative)
    record(__file__)
    model_results, importance_rows, node_rows, clock_rows = [], [], [], []
    for horizon in (20, 40, 60):
        for profile in ("l2_depth4", "l2_depth6"):
            key = f"h{horizon}_{profile}"
            directory = report / "year_2026/models" / key
            meta = read_json(directory / "model_metadata.json")
            require(meta["research_version"] == VERSION and meta["profile"] == profile and meta["horizon_minutes"] == horizon,
                "真实模型身份错误")
            require(meta["train_years"] == [2023, 2024] and meta["excluded_years"] == [2025, 2026], "非本折纯核心模型")
            require(meta["model_parameters"]["objective"] == "regression" and meta["feature_min_coverage"] == .7,
                "并非本轮原L2覆盖实验")
            raw_inputs = pd.read_pickle(record(directory / "training_inputs.pkl"))
            require(sha(directory / "training_inputs.pkl") == meta["training_inputs_file_sha256"], "核心矩阵文件SHA不符")
            features = meta["features"]
            require(list(raw_inputs) == INPUT_KEYS + features and len(raw_inputs) == meta["fitted_rows"], "实际核心矩阵字段或行数不符")
            require(raw_inputs.datetime.dt.year.isin([2023, 2024]).all(), "矩阵含非核心年")
            frame = raw_inputs[features]
            digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()
            require(digest == meta["training_matrix_sha256"], "实际核心矩阵内容SHA不符")
            native = lightgbm.Booster(model_file=str(record(directory / "return_model.txt")))
            require(sha(directory / "return_model.txt") == meta["native_model_sha256"], "真实原生模型SHA不符")
            require(native.params["objective"] == "regression" and native.feature_name() == features, "原生损失或特征身份错误")
            dump = native.dump_model()
            gains = native.feature_importance(importance_type="gain")
            splits = native.feature_importance(importance_type="split")
            gain_total = float(gains.sum())
            matrix = frame.to_numpy()
            sample_ids = np.linspace(0, len(matrix)-1, min(1024, len(matrix)), dtype=int)
            expected_leaves = np.asarray(native.predict(frame.iloc[sample_ids], pred_leaf=True, num_threads=1))
            actual_leaves = np.empty_like(expected_leaves)
            current_nodes = []
            any_missing = {f: np.zeros(len(matrix), dtype=bool) for f in FEATURES}
            for tree in dump["tree_info"]:
                index = int(tree["tree_index"])
                leaves, nodes, reached = traverse_tree(tree["tree_structure"], matrix, sample_ids, features, index)
                actual_leaves[:, index] = leaves
                current_nodes.extend(nodes)
                for feature in FEATURES:
                    any_missing[feature] |= reached[feature]
            require(np.array_equal(actual_leaves, expected_leaves), "独立NaN路径与原生核心pred_leaf不一致")
            for i, feature in enumerate(features):
                values = frame[feature]
                nodes = [n for n in current_nodes if n["feature"] == feature]
                if feature in FEATURES:
                    require(len(nodes) == int(splits[i]), "原生节点数与importance不一致")
                    require(np.isclose(sum(n["split_gain"] for n in nodes), gains[i], rtol=1e-12, atol=1e-12), "原生节点gain合计不符")
                importance_rows.append(dict(model=key, horizon_minutes=horizon, profile=profile, feature=feature,
                    added_15m=feature in FEATURES, training_rows=len(frame), nonmissing_rows=int(values.notna().sum()),
                    nan_rows=int(values.isna().sum()), coverage=float(values.notna().mean()), dtype=str(values.dtype),
                    split_count=int(splits[i]), gain=float(gains[i]), all_feature_gain=gain_total,
                    gain_share=float(gains[i]/gain_total) if gain_total else None,
                    nan_type_split_count=sum(n["missing_type"] == "NaN" for n in nodes) if feature in FEATURES else None,
                    default_left_split_count=sum(n["default_left"] for n in nodes) if feature in FEATURES else None,
                    default_right_split_count=sum(not n["default_left"] for n in nodes) if feature in FEATURES else None,
                    splits_reached_by_core_nan=sum(n["nan_core_rows_reaching_node"] > 0 for n in nodes) if feature in FEATURES else None,
                    nan_node_visits=sum(n["nan_core_rows_reaching_node"] for n in nodes) if feature in FEATURES else None,
                    distinct_core_rows_reaching_nan_split=int(any_missing[feature].sum()) if feature in FEATURES else None,
                    nan_visits_routed_left=sum(n["nan_rows_routed_left"] for n in nodes) if feature in FEATURES else None,
                    nan_visits_routed_right=sum(n["nan_rows_routed_right"] for n in nodes) if feature in FEATURES else None))
            for feature in FEATURES:
                for clock, rows in raw_inputs.groupby(raw_inputs.datetime.dt.strftime("%H:%M")):
                    clock_rows.append(dict(model=key, horizon_minutes=horizon, profile=profile, feature=feature,
                        clock=clock, rows=len(rows), valid=int(rows[feature].notna().sum()),
                        nan=int(rows[feature].isna().sum()), coverage=float(rows[feature].notna().mean())))
            node_rows.extend(dict(model=key, **n) for n in current_nodes)
            selected = [r for r in importance_rows if r["model"] == key and r["added_15m"]]
            model_results.append(dict(model=key, core_years=meta["train_years"], excluded_years=meta["excluded_years"],
                training_rows=len(frame), trees=len(dump["tree_info"]), features=len(features),
                added_fields_all_have_actual_splits=all(r["split_count"] > 0 for r in selected),
                added_fields_split_count=sum(r["split_count"] for r in selected),
                added_fields_gain_share=sum(r["gain_share"] for r in selected),
                native_core_leaf_path_rows=len(sample_ids), native_core_leaf_paths_exact=True,
                actual_full_core_nan_routes_counted=True, field_usage=selected))
            print(key, [(r["feature"], r["split_count"], r["gain_share"]) for r in selected], flush=True)
            del raw_inputs, frame, matrix, native
    pd.DataFrame(importance_rows).to_csv(output / "all_feature_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([r for r in importance_rows if r["added_15m"]]).to_csv(output / "added_15m_usage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(node_rows).to_csv(output / "added_15m_native_nodes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(clock_rows).to_csv(output / "added_15m_coverage_by_clock.csv", index=False, encoding="utf-8-sig")
    lines = ["# v18 核心模型实际使用新增15分钟字段的证据", "",
        "只读取已完成烟测的六份核心训练模型；核心年为2023/2024，排除2025/2026。没有读取内部或外层评价预测、标签或账户，没有重新拟合、改变阈值或撮合。", "",
        "| 模型 | 核心行数 | 个股15m分裂/gain占比 | 市场15m分裂/gain占比 | 相对市场15m分裂/gain占比 | 三列合计gain占比 |",
        "|---|---:|---:|---:|---:|---:|"]
    for result in model_results:
        fields = {r["feature"]: r for r in result["field_usage"]}
        columns = [f"{fields[f]['split_count']} / {fields[f]['gain_share']:.2%}" for f in FEATURES]
        lines.append(f"|{result['model']}|{result['training_rows']}|" + "|".join(columns) + f"|{result['added_fields_gain_share']:.2%}|")
    lines += ["", "gain share 的分母是该模型全部特征的原生 split_gain 合计；split count 是原生树实际分裂节点数。它们不是样本外贡献或因果重要性，也不能证明盈利。", "",
        "NaN 核验同时读取保存矩阵和原生节点：记录 missing_type、default_left/default_right，并将全部核心行沿已保存树遍历，统计真实到达这些 NaN 分裂的行/访问次数。每模型另取固定1024个核心行逐树核原生 pred_leaf 完全一致。访问次数会跨树重复，不能当独立样本；这只是既存模型的路径核查，没有计算评价收益。", "",
        "字段覆盖率与逐时钟情况见 added_15m_usage.csv / added_15m_coverage_by_clock.csv；原生节点证据见 added_15m_native_nodes.csv。早盘09:35/09:45不足15个历史分钟，缺失保留；h60的14:00信号无同日端点而不进入拟合，使原6/8覆盖变为5/7。", "",
        "树实际使用新增信息与研究收益是否改善是两个问题，本证据只回答前者。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(__file__, output / "probe.py")
    (output / "command.txt").write_text("python " + str(Path(__file__).resolve()) + " --report-dir " + str(report) + "\n")
    require(all(sha(path) == value for path, value in inputs.items()), "核查期间已读取输入改变")
    result = dict(passed=True, complete=True, source_report=str(report), models=model_results,
        sources_verified=56, inputs_sha256=inputs, all_read_inputs_unchanged=True,
        no_inner_or_outer_predictions_labels_or_accounts_read=True, models_fitted=0, accounts_run=0,
        importance_is_in_sample_tree_usage_not_out_of_sample_alpha=True,
        elapsed_seconds=time.time()-start, script_sha256=sha(__file__),
        outputs_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=True, output=str(output), verification_sha256=sha(output / "verification.json"),
        elapsed_seconds=result["elapsed_seconds"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
