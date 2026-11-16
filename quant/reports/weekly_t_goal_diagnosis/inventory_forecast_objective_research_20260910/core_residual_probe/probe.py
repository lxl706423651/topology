"""只读已审计 v14 核心拟合文件，诊断固定 1% 残差截断的实际覆盖。

不拟合、不改变标签、不读取内部/外层预测或账户。仅将统计与输入始末 SHA 写到
自身新诊断目录；相同实际输入、目标和原生权重身份允许复用预测计算。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import lightgbm
import numpy as np
import pandas as pd

ALPHA = 0.01
VERSION = "inventory_learning_t_v14"
KEYS = ["datetime", "symbol"]
PREFIX = [*KEYS, "feature_source_start", "cohort_target_year", "baseline_opening_shares"]
FILES = ("model_metadata.json", "training_inputs.pkl", "training_labels.csv", "return_model.txt")
FOLDS = {2023: ([2024, 2025], [2023, 2026]), 2024: ([2023, 2025], [2024, 2026]),
         2025: ([2023, 2024], [2025, 2026]), 2026: ([2023, 2024], [2025, 2026])}


def require(value, message):
    """任何身份、行对应或有限性错误立即终止，禁止隐式丢掉异常行。"""
    if not value:
        raise AssertionError(message)


def file_sha(path):
    """流式核对真实文件字节，不能用路径/stat 代替内容身份。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha(frame):
    """按原列序和原 dtype 重建已冻结的 pandas 内容 SHA。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def residual_statistics(prediction, target):
    """原标签不变，以 prediction-target 计算唯一预声明 alpha 的贡献比例。"""
    prediction, target = np.asarray(prediction), np.asarray(target)
    require(prediction.shape == target.shape and target.ndim == 1 and len(target), "预测/标签形状不一致或为空")
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), "不允许丢弃非有限预测/标签")
    residual = prediction-target
    absolute = np.abs(residual)
    squared = residual**2
    above = absolute > ALPHA
    absolute_sum, squared_sum = float(absolute.sum()), float(squared.sum())
    quantiles = np.quantile(absolute, [.5, .9, .95, .99], method="linear")
    result = dict(rows=len(target), alpha=ALPHA, residual_definition="native_prediction-original_fitted_target",
                  above_alpha_rows=int(above.sum()), above_alpha_fraction=float(above.mean()),
                  equal_alpha_rows=int((absolute == ALPHA).sum()),
                  absolute_gradient_total=absolute_sum, above_alpha_absolute_gradient=float(absolute[above].sum()),
                  squared_error_total=squared_sum, above_alpha_squared_error=float(squared[above].sum()),
                  absolute_gradient_share=float(absolute[above].sum()/absolute_sum) if absolute_sum else 0.,
                  squared_error_share=float(squared[above].sum()/squared_sum) if squared_sum else 0.,
                  gradient_reduction_if_clipped=float((absolute-np.minimum(absolute, ALPHA)).sum()/absolute_sum) if absolute_sum else 0.,
                  mean_residual=float(residual.mean()), rmse=float(np.sqrt(squared.mean())),
                  abs_max=float(absolute.max()),
                  native_predictions_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                  fitted_residuals_sha256=hashlib.sha256(residual.tobytes()).hexdigest())
    result.update({f"abs_p{q}": float(v) for q, v in zip((50, 90, 95, 99), quantiles)})
    return result


def self_test():
    """验证严格大于边界、两种贡献分母及 NaN 拒绝，均不调用模型拟合。"""
    prediction = np.array([.001, .005, .01, .02, -.03])
    original = prediction.copy()
    result = residual_statistics(prediction, np.zeros(5))
    require(result["above_alpha_rows"] == 2 and result["equal_alpha_rows"] == 1, "固定阈值边界错误")
    require(np.isclose(result["absolute_gradient_share"], .05/.066), "绝对梯度分母错误")
    require(np.isclose(result["squared_error_share"], .0013/.001426), "L2 平方误差分母错误")
    require(np.isclose(result["gradient_reduction_if_clipped"], .03/.066), "截断后总梯度变化错误")
    require(np.array_equal(prediction, original), "输入被修改")
    zero = residual_statistics(np.zeros(2), np.zeros(2))
    require(zero["squared_error_share"] == zero["absolute_gradient_share"] == 0, "零分母处理错误")
    try:
        residual_statistics(np.array([np.nan]), np.array([0.]))
    except AssertionError:
        pass
    else:
        raise AssertionError("非有限值未拒绝")
    print("PASS: fixed-alpha boundary, squared/absolute contributions, zero denominator, nonfinite rejection, immutable inputs", flush=True)


def run(args):
    """核对每折六个核心模型，只对唯一三重身份执行原生前向推断。"""
    started = time.monotonic()
    report, audit_file, out = args.report_dir.resolve(), args.audit_file.resolve(), args.output_dir.resolve()
    require(out != report and not out.is_relative_to(report), "不得把诊断写入原报告")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("verification.json", "model_residual_statistics.csv", "input_sha256.json", "README.md"):
        require(not (out/name).exists(), f"拒绝覆盖已有诊断：{name}")
    require(file_sha(audit_file) == args.audit_sha256, "原完整独审身份不一致")
    audit = json.loads(audit_file.read_text())
    require(audit["passed"] is True and audit["complete"] is True and audit["all_read_inputs_unchanged"] is True
            and audit["research_version"] == VERSION, "仅允许已完整通过的原审计")
    # 只取已审计文件身份，不消费此 JSON 中的模型评分、内部/外层选择或账户结果。
    anchor = audit["inputs_sha256"]
    del audit
    before = {str(audit_file): args.audit_sha256}
    rows, cached = [], {}
    for fold_year, (core_years, excluded_years) in FOLDS.items():
        for horizon in (20, 40, 60):
            for profile in ("l2_depth4", "l2_depth6"):
                identity = f"year_{fold_year}/models/h{horizon}_{profile}"
                directory = report/identity
                hashes = {}
                for name in FILES:
                    path = directory/name
                    actual_sha = file_sha(path)
                    require(str(path) in anchor and anchor[str(path)] == actual_sha, f"原独审未锚定或文件改变：{path}")
                    before[str(path)] = hashes[name] = actual_sha
                meta = json.loads((directory/"model_metadata.json").read_text())
                require(meta["research_version"] == VERSION and meta["profile"] == profile
                        and meta["horizon_minutes"] == horizon, f"模型版本/期限错误：{identity}")
                require(sorted(meta["train_years"]) == core_years and sorted(meta["excluded_years"]) == excluded_years,
                        f"训练/排除年份错误：{identity}")
                require(meta["model_parameters"]["objective"] == "regression" and meta["lightgbm_version"] == lightgbm.__version__,
                        f"并非原 L2 或运行库版本改变：{identity}")
                for name, key in (("training_inputs.pkl", "training_inputs_file_sha256"),
                                  ("training_labels.csv", "training_labels_file_sha256"),
                                  ("return_model.txt", "native_model_sha256")):
                    require(hashes[name] == meta[key], f"元数据与实际文件 SHA 不同：{identity}/{name}")
                inputs = pd.read_pickle(directory/"training_inputs.pkl")
                labels = pd.read_csv(directory/"training_labels.csv", float_precision="round_trip")
                require(labels.fitted.dtype == bool and not labels.duplicated(KEYS).any()
                        and not inputs.duplicated(KEYS).any(), f"拟合标记/当前键不合法：{identity}")
                fitted = labels.loc[labels.fitted].copy().reset_index(drop=True)
                inputs = inputs.reset_index(drop=True)
                require(len(inputs) == len(fitted) == meta["fitted_rows"], f"实际拟合行数不同：{identity}")
                for name in ("datetime", "feature_source_start", "label_start", "label_end"):
                    fitted[name] = pd.to_datetime(fitted[name], format="mixed").astype("datetime64[ns]")
                pd.testing.assert_frame_equal(inputs[PREFIX], fitted[PREFIX], check_exact=True, check_dtype=True)
                require(fitted.datetime.dt.year.isin(core_years).all() and fitted.cohort_target_year.eq(fitted.datetime.dt.year).all()
                        and fitted.horizon_minutes.eq(horizon).all(), f"拟合标签非本核心年份/期限：{identity}")
                features = meta["features"]
                require(list(inputs.columns) == [*PREFIX, *features], f"实际列序与原模型不同：{identity}")
                matrix = inputs[features]
                target_frame = fitted[["target_return"]]
                require(frame_sha(matrix) == meta["training_matrix_sha256"]
                        and frame_sha(target_frame) == meta["training_target_sha256"], f"实际矩阵/目标内容 SHA 不同：{identity}")
                require(np.isfinite(target_frame.to_numpy()).all(), f"实际拟合目标含无效数：{identity}")
                key = (hashes["training_inputs.pkl"], meta["training_matrix_sha256"], meta["training_target_sha256"], hashes["return_model.txt"])
                cache_hit = key in cached
                if not cache_hit:
                    native = lightgbm.Booster(model_file=str(directory/"return_model.txt"))
                    require(native.feature_name() == features and native.num_trees() == meta["model_parameters"]["n_estimators"],
                            f"原生实际特征/树数不同：{identity}")
                    prediction = np.asarray(native.predict(matrix, num_threads=args.prediction_threads))
                    require(prediction.dtype == np.float64, "原生输出 dtype 改变")
                    cached[key] = (identity, residual_statistics(prediction, target_frame.target_return.to_numpy(copy=True)))
                origin, statistics = cached[key]
                rows.append({"fold_test_year": fold_year, "core_years": ",".join(map(str, core_years)),
                             "excluded_years": ",".join(map(str, excluded_years)), "horizon_minutes": horizon,
                             "profile": profile, "model_directory": identity, "prediction_cache_hit": cache_hit,
                             "prediction_source_model": origin, **statistics, "target_dtype": str(target_frame.target_return.dtype),
                             "training_inputs_file_sha256": hashes["training_inputs.pkl"],
                             "training_labels_file_sha256": hashes["training_labels.csv"],
                             "training_matrix_sha256": meta["training_matrix_sha256"],
                             "training_target_sha256": meta["training_target_sha256"], "native_model_sha256": hashes["return_model.txt"]})
                print(f"{identity}: n={statistics['rows']}, >1%={statistics['above_alpha_fraction']:.6%}, "
                      f"L2 share={statistics['squared_error_share']:.6%}, abs-gradient share={statistics['absolute_gradient_share']:.6%}, "
                      f"cache={cache_hit}", flush=True)
    after = {path: file_sha(Path(path)) for path in before}
    require(before == after, "诊断期间输入字节改变")
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"model_residual_statistics.csv", index=False, encoding="utf-8-sig")
    (out/"input_sha256.json").write_text(json.dumps({"before": before, "after": after, "all_unchanged": True}, indent=2)+"\n")
    verification = {"passed": True, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_report": str(report), "source_audit": str(audit_file), "source_audit_sha256": args.audit_sha256,
                    "fixed_alpha": ALPHA, "models_checked": len(rows), "unique_prediction_identities": len(cached),
                    "prediction_cache_hits": len(rows)-len(cached), "core_model_files_checked": len(before)-1,
                    "all_input_sha256_unchanged": before == after, "lightgbm_version": lightgbm.__version__,
                    "no_training_or_model_selection": True, "no_inner_or_outer_predictions_or_accounts_read": True,
                    "source_audit_fields_consumed": ["passed", "complete", "all_read_inputs_unchanged", "research_version", "inputs_sha256"],
                    "no_label_modification": True, "residual_target": "unchanged original fitted target_return from exact CSV parser",
                    "threshold_search_performed": False, "script_sha256": file_sha(Path(__file__).resolve()),
                    "command": [sys.executable, *sys.argv], "elapsed_seconds": time.monotonic()-started,
                    "statistics": rows}
    (out/"verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    lines = ["# 固定 1% 截断的纯核心残差预检", "", "只使用原 v14 已审计实际拟合输入、标签、元数据与原生权重；无训练、无标签修改、无参数扫描，无内部/外层预测或账户读取。",
             "", f"24 个模型身份全部核对；{len(cached)} 个唯一预测计算，{len(rows)-len(cached)} 次严格身份相同的复用。96 份模型文件及完整审计身份均始末 SHA 一致。",
             "", "残差=原生预测−原精确拟合标签。表中分位数及 max 均为原始收益率单位，乘 10,000 为 bp；超阈值严格为 abs(residual)>0.01。",
             "L2 贡献分母为全部残差平方和，绝对梯度贡献分母为全部绝对残差和。不是仅在超阈值行内计算比例；相同折模型不作为独立样本合并。", "",
             "| 折标识 | 核心年 | 期限 | 深度 | 拟合行 | 超 1% 行比例 | L2 平方误差份额 | 绝对梯度份额 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['fold_test_year']} | {row['core_years']} | {row['horizon_minutes']} | {row['profile']} | {row['rows']} | {row['above_alpha_fraction']:.4%} | {row['squared_error_share']:.4%} | {row['absolute_gradient_share']:.4%} |")
    lines += ["", "解释边界：这只回答已拟合 L2 的哪些残差会被 1% 梯度截断；并非 Huber 重训后的残差、内部校准或择时收益证据。Huber 拟合路径改变后残差集合也会改变。真实大涨跌可能是真实信息，截断不能视为纠错。",
              "", "没有按核心残差更改 alpha，也没有用内部/外层选择周期或深度。是否足以覆盖交易成本，仍须在事先声明的独立训练与真实执行协议中验证。",
              "", "执行命令：", "", "```bash", " ".join(verification["command"]), "```", "", "本目录 probe.py 为独立诊断；model_residual_statistics.csv 为完整分位数/贡献统计，verification.json 为身份与结果，input_sha256.json 为输入始末 SHA。"]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({key: verification[key] for key in ("passed", "models_checked", "unique_prediction_identities", "prediction_cache_hits", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--audit-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        require(all((args.report_dir, args.audit_file, args.audit_sha256, args.output_dir)) and args.prediction_threads > 0,
                "必须显式提供原报告/完整审计身份/新输出目录与正预测线程数")
        run(args)
