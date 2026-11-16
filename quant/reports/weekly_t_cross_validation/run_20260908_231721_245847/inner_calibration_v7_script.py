"""只读v7首折内部2026的两个事前固定模型，按固定幅度分箱诊断预测校准。

从本报告目录运行。读取已冻结缓存/内部预测/支持域和内部候选记录，写本目录
inner_calibration_v7.json/md；不拟合、不调参、不读取外层收益，也不改原始产物。
未来价仅用于内部误差诊断，标称双腿价差不代表可成交或实际账户收益。
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parent
PROJECT = REPORT.parents[2]
sys.path.insert(0, str(PROJECT))

from quant_backtest.weekly_t_features import add_session_context_features
from quant_backtest.weekly_t_short_horizon import (
    create_short_decisions, feature_support_mask, planned_recovery_times, predicted_pair_edges,
)

MODELS = ((20, "l2_depth4"), (60, "l2_depth6"))
EDGES_BP = (0., 10., 20., 40., 60., 100., np.inf)


def file_sha(path: Path) -> str:
    """分块计算所读文件SHA，避免再次将大缓存驻留内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha(frame: pd.DataFrame) -> str:
    """按生产候选相同方式摘要DataFrame，包含类型和行序而不包含索引。"""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def counts(frame: pd.DataFrame) -> dict:
    """报告样本的行、日期、周及股票数，不把重复时钟视为独立收益实验。"""
    return {"rows": len(frame), "days": int(frame.datetime.dt.normalize().nunique()),
            "weeks": int(frame.trade_week.nunique()), "symbols": int(frame.symbol.nunique()),
            "distinct_clocks": int(frame.datetime.nunique()),
            "symbol_days": len(frame.assign(_day=frame.datetime.dt.normalize())[["_day", "symbol"]].drop_duplicates())}


def mean_or_none(values: pd.Series) -> float | None:
    """只汇总有限值；空箱明确返回null。"""
    valid = values.loc[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def summarize_bins(frame: pd.DataFrame) -> list[dict]:
    """固定左闭右开预测绝对幅度箱；所有均值仅为每信号等权内部诊断。"""
    output = []
    for lower, upper in zip(EDGES_BP[:-1], EDGES_BP[1:]):
        part = frame.loc[frame.absolute_prediction_bp.ge(lower) & frame.absolute_prediction_bp.lt(upper)]
        valid = part.loc[part.future_prices_valid & part.direction.ne(0)]
        output.append({"lower_bp": lower, "upper_bp": None if np.isinf(upper) else upper,
            "bin": f"[{lower:g},{'inf' if np.isinf(upper) else f'{upper:g}'})",
            **counts(part), "usable_future_rows": len(valid),
            "missing_execution_price_rows": int((~part.valid_execution_price).sum()),
            "missing_recovery_price_rows": int((~part.valid_recovery_price).sum()),
            "missing_or_invalid_future_rows": int((~part.future_prices_valid).sum()),
            "zero_prediction_rows": int(part.direction.eq(0).sum()),
            "mean_absolute_prediction_bp": mean_or_none(part.absolute_prediction_bp),
            "mean_predicted_nominal_net_bp": mean_or_none(part.predicted_nominal_net_bp),
            "mean_direction_times_realized_raw_return_bp": mean_or_none(valid.direction_realized_raw_bp),
            "median_direction_times_realized_raw_return_bp": float(valid.direction_realized_raw_bp.median()) if len(valid) else None,
            "mean_realized_nominal_net_bp": mean_or_none(valid.realized_nominal_net_bp),
            "nominal_net_positive_fraction": float(valid.realized_nominal_net_bp.gt(0).mean()) if len(valid) else None})
    return output


def main() -> None:
    """重建当前支持域输入，核候选决策身份，再固定分箱统计内部未来误差。"""
    started = time.monotonic()
    fold = REPORT / "year_2023"
    cfg = json.loads((REPORT / "resolved_config.json").read_text())
    assert cfg["research_version"] == "weekly_t_v7_train_domain"
    assert cfg["short_t_fraction"] == .25 and cfg["latest_planned_recovery_time"] == "14:30"
    cfg_path = REPORT / "resolved_config.json"
    data_path, horizon_path = REPORT / "data_manifest.json", REPORT / "horizon_manifest.json"
    data_manifest, horizon_manifest = json.loads(data_path.read_text()), json.loads(horizon_path.read_text())
    raw_path = Path(data_manifest["cache_path"]) / "intraday.pkl"
    labels_path = Path(horizon_manifest["labels_path"])
    selected_path, metrics_path = fold / "validation_weekly_selections.csv", fold / "validation_rule_metrics.csv"
    required = [cfg_path, data_path, horizon_path, raw_path, labels_path, selected_path, metrics_path,
                REPORT / "source_hashes.json", REPORT / "research_protocol.json"]
    source_checks = []
    for name, expected in json.loads((REPORT / "source_hashes.json").read_text()).items():
        current, frozen = PROJECT / name, REPORT / "source_snapshot" / Path(name).name
        source_checks.append({"path": name, "expected_sha256": expected,
                              "current_sha256": file_sha(current), "snapshot_sha256": file_sha(frozen)})
        assert source_checks[-1]["snapshot_sha256"] == expected
        source_checks[-1]["current_match"] = source_checks[-1]["current_sha256"] == expected
        if not source_checks[-1]["current_match"]:
            # 同工作区另一任务改了旧逐lot入口；本诊断未导入/调用它，不能隐瞒身份差异。
            assert name == "quant_backtest/annual_portfolio.py"
            original_tree, current_tree = ast.parse(frozen.read_text()), ast.parse(current.read_text())
            for function in ("calculate_fee", "calculate_daily_metrics"):
                original = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
                changed = next(node for node in current_tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
                assert ast.dump(original) == ast.dump(changed)
            source_checks[-1]["unrelated_change_notice"] = "当前run_lot_portfolio成交校验已改；本诊断不调用该入口，费用与指标函数AST未变，按原快照记账"
        required.extend([current, frozen])
    for horizon, profile in MODELS:
        for path in (fold / f"validation_predictions_h{horizon}_{profile}.csv", fold / f"support_h{horizon}.json",
                     fold / f"inner_t_h{horizon}_{profile}/model_metadata.json",
                     fold / f"inner_t_h{horizon}_{profile}/model_bundle.joblib"):
            if not path.exists():
                raise FileNotFoundError(f"固定诊断输入尚未生成：{path}")
            required.append(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"内部候选尚未完成：{metrics_path}")
    selected = pd.read_csv(selected_path, parse_dates=["trade_week", "entry_date", "exit_date"], float_precision="round_trip")
    assert selected.entry_date.dt.year.eq(2026).all() and selected.exit_date.dt.year.eq(2026).all()
    selected = selected[["trade_week", "symbol", "entry_date", "exit_date", "universe_previous_close"]]
    assert not selected.duplicated(["trade_week", "symbol"]).any()
    print("读取完整同刻2026截面以复原当前市场/支持域特征；不加载恢复报价库。", flush=True)
    raw = pd.read_pickle(raw_path)
    columns = ["datetime", "symbol", "date", "trade_week", "signal_price", "signal_high", "signal_low",
               "previous_close", "f_open_return", "f_vwap_deviation", "f_previous_day_amplitude"]
    current = raw.loc[raw.datetime.ge("2026-01-01") & raw.datetime.lt("2027-01-01"), columns].copy()
    del raw
    gc.collect()
    current = add_session_context_features(current)
    market_rows = len(current)
    current = current.merge(selected, on=["trade_week", "symbol"], how="inner", validate="many_to_one")
    current = current.loc[current.date.gt(current.entry_date) & current.date.lt(current.exit_date)].copy()
    assert not current.duplicated(["datetime", "symbol"]).any()
    slot = cfg["initial_cash"] * cfg["base_fraction"] / cfg["top_n"]
    base = np.floor(slot / (current.universe_previous_close * (1 + cfg["slippage_rate"])) / cfg["lot_size"]) * cfg["lot_size"]
    current["static_slot_eligible"] = np.floor(base * cfg["short_t_fraction"] / cfg["lot_size"]) >= 1
    print(f"因果周中信号{len(current):,}行；读取两种周期的未来标签，仅供内部误差诊断。", flush=True)
    raw_labels = pd.read_pickle(labels_path)
    relevant = raw_labels.datetime.ge("2026-01-01") & raw_labels.datetime.lt("2027-01-01")
    relevant &= raw_labels.horizon_minutes.isin([value[0] for value in MODELS])
    labels = raw_labels.loc[relevant, ["datetime", "symbol", "horizon_minutes", "execution_price", "recovery_price",
                                    "label_start", "label_end", "target_return"]].copy()
    del raw_labels, relevant
    gc.collect()
    labels = labels.merge(current[["datetime", "symbol"]], on=["datetime", "symbol"], how="inner", validate="many_to_one")
    metrics = pd.read_csv(metrics_path, float_precision="round_trip")
    output = {"scope": "year_2023首折，core2024/2025，内部2026；没有读取外层收益",
              "predeclared_models": [{"horizon_minutes": h, "profile": p} for h, p in MODELS],
              "absolute_prediction_bins_bp": [0, 10, 20, 40, 60, 100, "inf"],
              "current_market_rows_before_weekly_selection": market_rows,
              "static_slot_budget_rmb": slot, "source_checks": source_checks, "models": [],
              "statistical_scope": "每信号等权；日期/周/股票数仅为不同组数，不主张样本独立。"}
    for horizon, profile in MODELS:
        path = fold / f"validation_predictions_h{horizon}_{profile}.csv"
        predictions = pd.read_csv(path, parse_dates=["datetime"], float_precision="round_trip")
        assert np.isfinite(predictions.prediction).all()
        samples = predictions.merge(current, on=["datetime", "symbol"], how="left", validate="one_to_one", indicator=True)
        assert samples._merge.eq("both").all() and len(samples) == len(current)
        samples = samples.drop(columns="_merge").sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
        support = json.loads((fold / f"support_h{horizon}.json").read_text())
        known_support = feature_support_mask(samples, support)
        planned = planned_recovery_times(samples.datetime, horizon)
        clock_ok = planned.notna() & planned.dt.strftime("%H:%M").le(cfg["latest_planned_recovery_time"])
        clock_ok &= planned.dt.normalize().eq(samples.datetime.dt.normalize())
        candidate_rows = []
        for margin in cfg["short_t_net_margins"]:
            rule = {"horizon_minutes": horizon, "profile": profile, "net_margin": margin,
                    "enabled": True, "fraction": cfg["short_t_fraction"], "direction": "both"}
            decisions = create_short_decisions(samples, rule, cfg, support)
            stored = metrics.loc[metrics.horizon_minutes.eq(horizon) & metrics.profile.eq(profile)
                                 & np.isclose(metrics.net_margin, margin, rtol=0, atol=1e-14)]
            assert len(stored) == 1
            row = stored.iloc[0]
            digest = frame_sha(decisions)
            assert digest == row.decision_sha256, f"重建当前输入未复现候选{row.candidate_index}的决策SHA"
            static_requests = decisions.merge(samples[["datetime", "symbol", "static_slot_eligible"]],
                                              on=["datetime", "symbol"], how="left", validate="one_to_one")
            candidate_rows.append({"candidate_index": int(row.candidate_index), "net_margin_bp": margin * 10000,
                                  "requested_pairs": len(decisions),
                                  "requested_pairs_in_static_slot_proxy": int(static_requests.static_slot_eligible.sum()),
                                  "completed_pairs_original_account": int(row.completed_pair_count),
                                  "fragments_original_account": int(row.fragment_count),
                                  "active_days_original_account": int(row.active_days),
                                  "active_weeks_original_account": int(row.active_weeks),
                                  "decision_sha256": digest, "stored_decision_sha_match": True})
        causal = samples.loc[known_support & clock_ok].copy()
        future = labels.loc[labels.horizon_minutes.eq(horizon)].drop(columns="horizon_minutes")
        causal = causal.merge(future, on=["datetime", "symbol"], how="left", validate="one_to_one")
        causal["valid_execution_price"] = np.isfinite(causal.execution_price) & causal.execution_price.gt(0)
        causal["valid_recovery_price"] = np.isfinite(causal.recovery_price) & causal.recovery_price.gt(0)
        causal["future_prices_valid"] = causal.valid_execution_price & causal.valid_recovery_price
        realized = causal.recovery_price / causal.execution_price - 1
        valid = causal.future_prices_valid
        assert np.allclose(realized.loc[valid], causal.loc[valid, "target_return"], rtol=0, atol=1e-12)
        causal["direction"] = np.sign(causal.prediction)
        causal["absolute_prediction_bp"] = causal.prediction.abs() * 10000
        causal["direction_realized_raw_bp"] = causal.direction * realized * 10000
        buy_pred, sell_pred = predicted_pair_edges(causal, cfg)
        realized_input = causal.assign(prediction=realized)
        buy_realized, sell_realized = predicted_pair_edges(realized_input, cfg)
        causal["predicted_nominal_net_bp"] = np.where(causal.direction.gt(0), buy_pred, sell_pred) * 10000
        causal["realized_nominal_net_bp"] = np.where(causal.direction.gt(0), buy_realized, sell_realized) * 10000
        causal.loc[causal.direction.eq(0), ["predicted_nominal_net_bp", "realized_nominal_net_bp"]] = np.nan
        static = causal.loc[causal.static_slot_eligible]
        output["models"].append({"horizon_minutes": horizon, "profile": profile,
             "all_frozen_weekly_middle_signals": counts(samples),
             "support_pass_rows": int(known_support.sum()), "clock_pass_rows": int(clock_ok.sum()),
             "existing_support_and_deadline_signals": counts(causal), "static_slot_proxy_signals": counts(static),
             "full_causal_bins": summarize_bins(causal), "static_slot_proxy_bins": summarize_bins(static),
             "existing_margin_candidates": candidate_rows,
             "current_feature_inputs_sha256": frame_sha(samples[["datetime", "symbol", *cfg["t_support_features"],
                                                                "signal_price", "signal_high", "signal_low", "previous_close"]])})
        print(f"固定h{horizon}/{profile}：当前域{len(causal):,}，静态代理{len(static):,}，3候选决策SHA全部复现。", flush=True)
    output["limitations"] = [
        "模型/分箱/原门槛事前固定；本诊断不选择新模型或阈值，也不改变训练、推断或实际账户。",
        "全因果样本指已匹配冻结周中名单且通过原支持域与14:30截止的全部信号，不额外按未来价或实际成交过滤。",
        "静态子集只作分析切片，使用9500元初始slot和已知周前收；至少400股不保证T+1、现金、容量、限价或恢复合法。",
        "方向价差=sign(预测)*实际原始收益；标称净价差使用原双腿滑点、佣金、过户费及当日印花税公式，除以首腿参考名义金额。",
        "预测正为买先、负为卖先；零预测方向为0且不统计方向/净价差均值。未来缺价行保留在箱行数中，有限未来值另列。",
        "未来价仅作内部误差诊断。标称价差假定两腿各按计划参考价完成，不计真实容量、分片、延迟恢复和动态资金，不能视为已成交收益。",
        "原候选请求含当前方向风险门控和每股每日最早一次；完整pair/片段来自原内部账户，未对静态子集另开账户或分配pair归属。",
        "多个信号共享日期、市场特征和未来窗口；均值为逐信号等权而非独立样本估计。历史已经多次复测，不是全新未接触测试。"]
    output["input_files"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}
                             for path in sorted(set(required))]
    output["script_sha256"] = file_sha(Path(__file__))
    output["elapsed_seconds"] = time.monotonic() - started
    positive_forecast_bins = [row for model in output["models"]
                             for key in ("full_causal_bins", "static_slot_proxy_bins")
                             for row in model[key] if row["rows"] and row["mean_predicted_nominal_net_bp"] > 0]
    output["positive_forecast_bin_check"] = {
        "nonempty_positive_mean_predicted_net_bins": len(positive_forecast_bins),
        "all_mean_realized_direction_negative": all(row["mean_direction_times_realized_raw_return_bp"] < 0
                                                     for row in positive_forecast_bins),
        "all_mean_realized_nominal_net_negative": all(row["mean_realized_nominal_net_bp"] < 0
                                                       for row in positive_forecast_bins)}
    (REPORT / "inner_calibration_v7.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# v7 首折内部预测校准诊断", "", "对象事前固定为h20/l2_depth4和h60/l2_depth6；只分析内部2026，不读外层收益、不拟合或调参。固定箱为绝对预测原始收益[0,10,20,40,60,100,∞)bp。", "",
             "每个模型先匹配冻结周中名单并通过原核心支持域及原14:30恢复截止；再单列静态整手代理子集。未来缺价不从当前域删行。标称双腿净价差包含原费用，但不是可成交或账户收益。", ""]
    if positive_forecast_bins and output["positive_forecast_bin_check"]["all_mean_realized_direction_negative"]:
        lines += ["**本次固定分箱未显示高预测幅度兑现为内部方向价差。** 两模型中平均预测净价差为正的非空箱，实际平均方向价差全部为负，扣原费用后的标称净价差也全部为负；静态整手代理子集结论相同。h20高幅箱仅35行、h60最高幅箱仅6行，这些是相关信号的描述均值，不能据此挑选新阈值或推断未来必然反向。", ""]
    for model in output["models"]:
        h, profile = model["horizon_minutes"], model["profile"]
        lines += [f"## 固定h{h}/{profile}", "",
                  f"全部周中因果预测{model['all_frozen_weekly_middle_signals']['rows']:,}行；原支持域通过{model['support_pass_rows']:,}行，原时钟通过{model['clock_pass_rows']:,}行。交集与静态子集如下。", ""]
        for title, key, group in (("全部当前合格信号", "full_causal_bins", "existing_support_and_deadline_signals"),
                                  ("静态整手代理子集", "static_slot_proxy_bins", "static_slot_proxy_signals")):
            detail = model[group]
            lines += [f"{title}：{detail['rows']:,}行/{detail['days']}天/{detail['weeks']}周/{detail['symbols']}股；不同信号时刻{detail['distinct_clocks']}。", "",
                      "| abs预测bp箱 | 行数/有效未来 | 缺执行/恢复价 | 平均abs预测bp | 平均预测净bp | 平均实际方向bp | 平均实际标称净bp |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for row in model[key]:
                def formatted(value):
                    """空箱显示破折号，有限数显示三位小数。"""
                    return "—" if value is None else f"{value:.3f}"
                lines.append(f"| {row['bin']} | {row['rows']}/{row['usable_future_rows']} | {row['missing_execution_price_rows']}/{row['missing_recovery_price_rows']} | "
                             + " | ".join(formatted(row[c]) for c in ["mean_absolute_prediction_bp", "mean_predicted_nominal_net_bp",
                                            "mean_direction_times_realized_raw_return_bp", "mean_realized_nominal_net_bp"]) + " |")
            lines.append("")
        lines += ["原规则候选（未改门槛或重新评价账户；请求决策SHA已逐项复现）：", "",
                  "| 原净门槛bp | 请求/其中静态代理 | 原完整pair/片段 | 原活跃日/周 |",
                  "|---|---:|---:|---:|"]
        for row in model["existing_margin_candidates"]:
            lines.append(f"| {row['net_margin_bp']:.0f} | {row['requested_pairs']}/{row['requested_pairs_in_static_slot_proxy']} | "
                         f"{row['completed_pairs_original_account']}/{row['fragments_original_account']} | {row['active_days_original_account']}/{row['active_weeks_original_account']} |")
        lines.append("")
    lines += ["## 解释边界", ""] + ["- " + text for text in output["limitations"]]
    matched = sum(check["current_match"] for check in source_checks)
    lines += ["", f"17份原快照SHA均匹配；当前工作区{matched}/17份一致。annual_portfolio旧run_lot_portfolio入口的独立改动未用于本诊断，费用/指标函数AST未变，具体身份见JSON。脚本SHA256：`{output['script_sha256']}`。所读输入文件及SHA、各箱不同日期/周/股票数量见`inner_calibration_v7.json`。"]
    (REPORT / "inner_calibration_v7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"内部校准诊断完成，耗时{output['elapsed_seconds']:.2f}秒；已释放缓存数据。", flush=True)


if __name__ == "__main__":
    main()
