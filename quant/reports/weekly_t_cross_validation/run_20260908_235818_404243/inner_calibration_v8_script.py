"""只读v8/v7首折内部2026固定模型，比较残差训练后的预测校准。

从本报告目录运行。读取已冻结缓存/内部预测/支持域和内部候选记录，写本目录
inner_calibration_v8.json/md；不拟合、不调参、不读取外层收益，也不改原始产物。
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


def read_native_importance(report: Path, required: list) -> list[dict]:
    """读取事前固定全部9个核心模型；只记录原生gain/split，不选择模型。"""
    import lightgbm as lgb
    rows = []
    for horizon in (20, 40, 60):
        for profile in ("l2_depth4", "l2_depth6", "huber_depth4"):
            folder = report / "year_2023" / f"inner_t_h{horizon}_{profile}"
            mp, native = folder / "model_metadata.json", folder / "return_model.txt"
            required.extend([mp, native])
            meta = json.loads(mp.read_text())
            assert meta["train_years"] == [2024, 2025] and meta["excluded_years"] == [2023, 2026]
            model = lgb.Booster(model_file=str(native))
            assert model.feature_name() == meta["feature_names"]
            gains, splits = model.feature_importance("gain"), model.feature_importance("split")
            total = float(gains.sum())
            features = [{"feature": name, "gain": float(gain), "split": int(split),
                         "normalized_gain": float(gain / total) if total else 0.}
                        for name, gain, split in zip(model.feature_name(), gains, splits)]
            rows.append({"horizon_minutes": horizon, "profile": profile,
                "feature_names_match": True, "training_rows": meta["training_rows"],
                "fitted_rows": meta["fitted_rows"], "training_clock_count": meta.get("training_clock_count"),
                "singleton_rows_excluded": meta.get("singleton_rows_excluded"),
                "market_two_feature_gain_share": sum(r["normalized_gain"] for r in features
                    if r["feature"] in ("f_market_day_return", "f_market_return")),
                "session_gain_share": sum(r["normalized_gain"] for r in features if r["feature"] == "f_session_fraction"),
                "features": sorted(features, key=lambda row: -row["gain"])})
    return rows


def prediction_decomposition(frame: pd.DataFrame) -> dict:
    """同一已知信号集合内，度量预测的同刻共同分量；未来均值只供相关诊断。"""
    values = frame.prediction.astype(float)
    common = frame.groupby("datetime", sort=False).prediction.transform("mean")
    centered = values - values.mean()
    common_centered = common - values.mean()
    residual = values - common
    denominator = float(np.square(centered).sum())
    energy = float(np.square(values).sum())
    finite = frame.future_prices_valid
    actual = frame.loc[finite, "target_return"]
    actual_common = frame.loc[finite].groupby("datetime", sort=False).target_return.transform("mean")
    actual_residual = actual - actual_common
    def corr(left, right):
        """零方差或不足两行不产生虚构相关系数。"""
        return float(left.corr(right)) if len(left) > 1 and left.std() > 0 and right.std() > 0 else None
    return {**counts(frame), "mean_prediction_bp": float(values.mean() * 10000),
        "prediction_rms_bp": float(np.sqrt(np.square(values).mean()) * 10000),
        "same_clock_prediction_rms_bp": float(np.sqrt(np.square(common).mean()) * 10000),
        "within_clock_prediction_rms_bp": float(np.sqrt(np.square(residual).mean()) * 10000),
        "same_clock_centered_variance_share": float(np.square(common_centered).sum() / denominator) if denominator else None,
        "same_clock_uncentered_energy_share": float(np.square(common).sum() / energy) if energy else None,
        "prediction_vs_raw_target_correlation": corr(values.loc[finite], actual),
        "prediction_vs_current_sample_clock_mean_future_correlation": corr(values.loc[finite], actual_common),
        "prediction_vs_within_clock_future_residual_correlation": corr(values.loc[finite], actual_residual),
        "decomposition_scope": "本段完全相同因果信号集合，按datetime分组、逐行等权；未来共同收益不是全市场均值且从不作为输入。"}


def add_future_diagnostics(samples: pd.DataFrame, future: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """保留全部当前样本；标签只计算误差与原费用公式下的标称两腿价差。"""
    causal = samples.merge(future, on=["datetime", "symbol"], how="left", validate="one_to_one")
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
    buy_actual, sell_actual = predicted_pair_edges(causal.assign(prediction=realized), cfg)
    causal["predicted_nominal_net_bp"] = np.where(causal.direction.gt(0), buy_pred, sell_pred) * 10000
    causal["realized_nominal_net_bp"] = np.where(causal.direction.gt(0), buy_actual, sell_actual) * 10000
    causal.loc[causal.direction.eq(0), ["predicted_nominal_net_bp", "realized_nominal_net_bp"]] = np.nan
    return causal


def describe_candidates(metrics: pd.DataFrame) -> dict:
    """只记录本折全部27个内部候选原结果，不能用本诊断重选规则。"""
    assert len(metrics) == 27 and metrics.test_year.eq(2023).all() and metrics.inner_year.eq(2026).all()
    fields = ["candidate_index", "horizon_minutes", "profile", "net_margin", "incremental_net_pnl",
        "completed_pair_count", "fragment_count", "active_days", "active_weeks", "active_symbols",
        "strict_eligible", "rejection_reasons", "pending_pairs", "inventory_mismatch_days",
        "recovery_failure_days", "base_plan_mismatch_count", "weekly_increment_ci_low_rmb", "t_fee"]
    net = metrics.incremental_net_pnl
    return {"rows": len(metrics), "positive": int(net.gt(0).sum()), "zero": int(net.eq(0).sum()),
        "negative": int(net.lt(0).sum()), "strict_eligible": int(metrics.strict_eligible.eq(True).sum()),
        "candidate_details": json.loads(metrics[fields].to_json(orient="records")),
        "scope": "仅year_2023首折inner2026独立账户；未读取任何外层记录。"}


def main() -> None:
    """比较固定首折、模型、样本交集与预测幅度箱；不拟合、不重跑账户。"""
    started = time.monotonic()
    previous = REPORT.parent / "run_20260908_231721_245847"
    reports = {"v7": previous, "v8": REPORT}
    configs = {name: json.loads((path / "resolved_config.json").read_text()) for name, path in reports.items()}
    assert configs["v8"]["research_version"] == "weekly_t_v8_residual_zero_market"
    cfg = configs["v8"]
    required = []
    native = {}
    selections = {}
    metrics = {}
    source_checks = []
    for version, report in reports.items():
        for name, expected in json.loads((report / "source_hashes.json").read_text()).items():
            frozen, current = report / "source_snapshot" / Path(name).name, PROJECT / name
            snapshot_sha, current_sha = file_sha(frozen), file_sha(current)
            assert snapshot_sha == expected
            if version == "v8":
                assert current_sha == expected, f"v8冻结来源变更：{name}"
            source_checks.append({"version": version, "path": name, "snapshot_sha256": snapshot_sha,
                "current_sha256": current_sha, "current_match": current_sha == expected})
            required.extend([frozen, current])
        for file in ("source_hashes.json", "resolved_config.json", "research_protocol.json", "data_manifest.json", "horizon_manifest.json"):
            required.append(report / file)
        fold = report / "year_2023"
        mp = fold / "validation_rule_metrics.csv"
        if not mp.exists():
            raise FileNotFoundError(f"固定首折内部27候选尚未齐全：{mp}")
        metrics[version] = pd.read_csv(mp, float_precision="round_trip")
        required.append(mp)
        selected_path = fold / "validation_weekly_selections.csv"
        required.append(selected_path)
        selections[version] = pd.read_csv(selected_path, parse_dates=["trade_week", "entry_date", "exit_date"], float_precision="round_trip")
        native[version] = read_native_importance(report, required)
    fields = ["trade_week", "symbol", "entry_date", "exit_date", "universe_previous_close"]
    selected = selections["v8"][fields].sort_values(["trade_week", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(selected, selections["v7"][fields].sort_values(["trade_week", "symbol"]).reset_index(drop=True))
    assert selected.entry_date.dt.year.eq(2026).all() and selected.exit_date.dt.year.eq(2026).all()
    assert not selected.duplicated(["trade_week", "symbol"]).any()
    economic_keys = ["initial_cash", "base_fraction", "top_n", "lot_size", "short_t_fraction", "commission_rate", "min_commission",
        "transfer_rate", "stamp_rate_before_20230828", "stamp_rate_from_20230828", "slippage_rate",
        "short_t_net_margins", "latest_planned_recovery_time", "t_support_features"]
    assert all(configs["v7"][key] == cfg[key] for key in economic_keys)
    data = json.loads((REPORT / "data_manifest.json").read_text())
    horizon_data = json.loads((REPORT / "horizon_manifest.json").read_text())
    raw_path, labels_path = Path(data["cache_path"]) / "intraday.pkl", Path(horizon_data["labels_path"])
    old_data = json.loads((previous / "data_manifest.json").read_text())
    old_labels = json.loads((previous / "horizon_manifest.json").read_text())
    assert old_data["cache_path"] == data["cache_path"] and old_labels["labels_path"] == horizon_data["labels_path"]
    required.extend([raw_path, labels_path])
    print("固定只读取2026内部年份信号；从全市场当前截面重建因果支持域输入。", flush=True)
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
    slot = cfg["initial_cash"] * cfg["base_fraction"] / cfg["top_n"]
    base = np.floor(slot / (current.universe_previous_close * (1 + cfg["slippage_rate"])) / cfg["lot_size"]) * cfg["lot_size"]
    current["static_slot_eligible"] = np.floor(base * cfg["short_t_fraction"] / cfg["lot_size"]) >= 1
    raw_labels = pd.read_pickle(labels_path)
    relevant = raw_labels.datetime.ge("2026-01-01") & raw_labels.datetime.lt("2027-01-01")
    relevant &= raw_labels.horizon_minutes.isin([row[0] for row in MODELS])
    labels = raw_labels.loc[relevant, ["datetime", "symbol", "horizon_minutes", "execution_price", "recovery_price", "target_return"]].copy()
    del raw_labels, relevant
    gc.collect()
    labels = labels.merge(current[["datetime", "symbol"]], on=["datetime", "symbol"], how="inner", validate="many_to_one")
    output = {"scope": "仅首折core2024/25模型与inner2026，绝不读取新旧outer账户或预测。",
        "predeclared_models": [{"horizon_minutes": h, "profile": p} for h, p in MODELS],
        "absolute_prediction_bins_bp": [0, 10, 20, 40, 60, 100, "inf"],
        "source_checks": source_checks, "native_importance": native,
        "identical_weekly_selections": True, "identical_economic_config": True,
        "current_market_rows": market_rows, "static_slot_budget_rmb": slot,
        "inner_candidates": {version: describe_candidates(frame) for version, frame in metrics.items()}, "models": []}
    for horizon, profile in MODELS:
        model_rows = {}
        masks = {}
        future = labels.loc[labels.horizon_minutes.eq(horizon)].drop(columns="horizon_minutes")
        for version, report in reports.items():
            fold = report / "year_2023"
            pred_path = fold / f"validation_predictions_h{horizon}_{profile}.csv"
            support_path = fold / f"support_h{horizon}.json"
            required.extend([pred_path, support_path])
            pred = pd.read_csv(pred_path, parse_dates=["datetime"], float_precision="round_trip")
            assert np.isfinite(pred.prediction).all()
            if version == "v8":
                assert pred.market_prediction.eq(0).all() and pred.prediction.equals(pred.residual_prediction)
            samples = pred.merge(current, on=["datetime", "symbol"], how="left", validate="one_to_one", indicator=True)
            assert samples._merge.eq("both").all() and len(samples) == len(current)
            samples = samples.drop(columns="_merge").sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
            support = json.loads(support_path.read_text())
            support_ok = feature_support_mask(samples, support)
            planned = planned_recovery_times(samples.datetime, horizon)
            clock_ok = planned.notna() & planned.dt.strftime("%H:%M").le(cfg["latest_planned_recovery_time"])
            clock_ok &= planned.dt.normalize().eq(samples.datetime.dt.normalize())
            candidates = []
            for margin in cfg["short_t_net_margins"]:
                rule = {"horizon_minutes": horizon, "profile": profile, "net_margin": margin,
                    "enabled": True, "fraction": cfg["short_t_fraction"], "direction": "both"}
                decisions = create_short_decisions(samples, rule, cfg, support)
                stored = metrics[version].loc[metrics[version].horizon_minutes.eq(horizon) & metrics[version].profile.eq(profile)
                    & np.isclose(metrics[version].net_margin, margin, rtol=0, atol=1e-14)]
                assert len(stored) == 1
                row = stored.iloc[0]
                digest = frame_sha(decisions)
                assert digest == row.decision_sha256, (version, horizon, profile, margin, "候选SHA未复现")
                joined = decisions.merge(samples[["datetime", "symbol", "static_slot_eligible"]], on=["datetime", "symbol"], validate="one_to_one")
                candidates.append({"candidate_index": int(row.candidate_index), "net_margin_bp": margin * 10000,
                    "requested_pairs": len(decisions), "requested_pairs_static_proxy": int(joined.static_slot_eligible.sum()),
                    "completed_pairs": int(row.completed_pair_count), "fragments": int(row.fragment_count),
                    "incremental_net_pnl_rmb": float(row.incremental_net_pnl), "t_fee_rmb": float(row.t_fee),
                    "active_days": int(row.active_days), "active_weeks": int(row.active_weeks),
                    "strict_eligible": bool(row.strict_eligible), "rejection_reasons": str(row.rejection_reasons),
                    "decision_sha256": digest, "stored_decisions_exactly_reproduced": True})
            masks[version] = (support_ok & clock_ok).to_numpy()
            model_rows[version] = add_future_diagnostics(samples, future, cfg)
            own = model_rows[version].loc[masks[version]]
            output_row = {"version": version, "all_current_signals": counts(samples), "support_pass_rows": int(support_ok.sum()),
                "clock_pass_rows": int(clock_ok.sum()), "own_domain": counts(own), "own_domain_bins": summarize_bins(own),
                "own_domain_static_bins": summarize_bins(own.loc[own.static_slot_eligible]),
                "existing_margin_candidates": candidates,
                "current_inputs_sha256": frame_sha(samples[["datetime", "symbol", *cfg["t_support_features"],
                    "signal_price", "signal_high", "signal_low", "previous_close"]])}
            model_rows[version].attrs["output"] = output_row
        pd.testing.assert_frame_equal(model_rows["v7"][["datetime", "symbol"]], model_rows["v8"][["datetime", "symbol"]])
        intersection = masks["v7"] & masks["v8"]
        record = {"horizon_minutes": horizon, "profile": profile, "versions": {}}
        for version in reports:
            paired = model_rows[version].loc[intersection]
            row = model_rows[version].attrs["output"]
            row.update({"paired_identical_domain": counts(paired), "paired_domain_keys_sha256": frame_sha(paired[["datetime", "symbol"]]),
                "paired_domain_bins": summarize_bins(paired), "paired_domain_static_bins": summarize_bins(paired.loc[paired.static_slot_eligible]),
                "paired_prediction_decomposition": prediction_decomposition(paired),
                "paired_static_prediction_decomposition": prediction_decomposition(paired.loc[paired.static_slot_eligible])})
            record["versions"][version] = row
        assert record["versions"]["v7"]["paired_domain_keys_sha256"] == record["versions"]["v8"]["paired_domain_keys_sha256"]
        output["models"].append(record)
        print(f"固定h{horizon}/{profile}相同信号交集{int(intersection.sum()):,}，v7/v8共6候选SHA逐项精确复现。", flush=True)
    output["limitations"] = [
        "固定模型和分箱、原门槛不变，不据该诊断重选本轮规则。仅内部结果，历史已多次复测，不能称全新测试。",
        "v7/v8预测各自重新分箱，箱内不是相同信号成员；即使整体集合完全对齐，也不能将同名箱视为配对改善。",
        "固定零市场先验只令显式market_prediction=0；残差树OOS的同刻截面均值仍可能非零，故独立量化其预测方差占比。",
        "共同项方差占比=逐行重复的同刻预测均值的中心化平方和/全部预测中心化平方和；不是因果归因或收益解释度。",
        "市场与session gain仅是训练树分裂损失改进份额；标签改为残差后总损失尺度变化，gain占比不代表未来收益贡献。",
        "未来股票原始价格仅用于内部误差诊断；方向价差为sign(prediction)*raw_return。标称净价差按原双腿费率滑点及当日印花税精算，假定按计划价格完成，不能称实际成交PnL。",
        "静态9500元slot与至少400股只是已知信息切片，不保证现金、T1、容量、限价或恢复可执行。",
        "全部当前信号先匹配冻结周中名单和既有支持域/14:30截止，未来缺价不删除当前行；每信号相关性强，日周股计数不等于独立样本。",
        "只有原内部候选表中的incremental_net_pnl属于真实回测账户量，未对分箱另开账户或重新分配已成交pair。"]
    output["input_files"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)} for path in sorted(set(required))]
    output["script_sha256"] = file_sha(Path(__file__))
    output["elapsed_seconds"] = time.monotonic() - started
    (REPORT / "inner_calibration_v8.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    render_markdown(output)
    print(f"v8固定内部诊断完成，耗时{output['elapsed_seconds']:.2f}秒；无模型/账户重跑。", flush=True)


def render_markdown(output: dict) -> None:
    """把固定比较结果写成短中文报告；完整各箱独立日周股数量留JSON。"""
    def fmt(value):
        """空值不伪装零。"""
        return "—" if value is None else f"{value:.3f}"
    lines = ["# v8 首折内部预测校准诊断", "", "只看核心2024/25模型和内部2026；固定h20/l2_depth4、h60/l2_depth6与原幅度箱，没有读取外层，也没有重训、修改门槛或重跑账户。", "",
        "先检查原周中名单及各自既有支持域，再在两个版本支持域交集的完全相同信号上比较。下表采用该交集；原域和静态整手子集详见JSON。不同版本按自身预测重新分箱，箱内成员不同，不能直接称同组改善。", "",
        "## 训练特征分裂份额", "", "| 周期/模型 | 市场两项gain v7→v8 | session gain v7→v8 |", "|---|---:|---:|"]
    for old, new in zip(output["native_importance"]["v7"], output["native_importance"]["v8"]):
        lines.append(f"| h{old['horizon_minutes']}/{old['profile']} | {old['market_two_feature_gain_share']:.2%} → {new['market_two_feature_gain_share']:.2%} | {old['session_gain_share']:.2%} → {new['session_gain_share']:.2%} |")
    for model in output["models"]:
        h, profile = model["horizon_minutes"], model["profile"]
        old, new = model["versions"]["v7"], model["versions"]["v8"]
        d = new["paired_identical_domain"]
        lines += ["", f"## 固定h{h}/{profile}", "", f"相同因果信号{d['rows']:,}行，{d['days']}天/{d['weeks']}周/{d['symbols']}股/{d['distinct_clocks']}个时刻；候选决策SHA已逐项复现。", "",
            "| 指标 | v7 | v8 |", "|---|---:|---:|"]
        for title, key in (("同刻共同预测方差占比", "same_clock_centered_variance_share"), ("同刻共同预测RMS bp", "same_clock_prediction_rms_bp"), ("股间残差预测RMS bp", "within_clock_prediction_rms_bp")):
            a, b = old["paired_prediction_decomposition"][key], new["paired_prediction_decomposition"][key]
            if "share" in key:
                lines.append(f"| {title} | {a:.2%} | {b:.2%} |")
            else:
                lines.append(f"| {title} | {fmt(a)} | {fmt(b)} |")
        lines += ["", "| 版本/abs预测bp | 行数/未来有效 | 平均abs预测bp | 预测标称净bp | 实际方向bp | 实际标称净bp |", "|---|---:|---:|---:|---:|---:|"]
        for version, row in model["versions"].items():
            for group in row["paired_domain_bins"]:
                lines.append(f"| {version} {group['bin']} | {group['rows']}/{group['usable_future_rows']} | " + " | ".join(fmt(group[key]) for key in
                    ("mean_absolute_prediction_bp", "mean_predicted_nominal_net_bp", "mean_direction_times_realized_raw_return_bp", "mean_realized_nominal_net_bp")) + " |")
        lines += ["", "原内部候选账户（与上述标称误差统计不同）：", "", "| 版本/门槛bp | 请求/静态代理 | 完整pair/片段 | 原净增量元 |", "|---|---:|---:|---:|"]
        for version, row in model["versions"].items():
            for candidate in row["existing_margin_candidates"]:
                lines.append(f"| {version}/{candidate['net_margin_bp']:.0f} | {candidate['requested_pairs']}/{candidate['requested_pairs_static_proxy']} | {candidate['completed_pairs']}/{candidate['fragments']} | {candidate['incremental_net_pnl_rmb']:.3f} |")
    lines += ["", "## 本折全部内部候选", ""]
    for version, summary in output["inner_candidates"].items():
        lines.append(f"{version}：27候选中{summary['positive']}正/{summary['zero']}零/{summary['negative']}负，原严格合格{summary['strict_eligible']}；逐项原因和完整配对支持见JSON。")
    lines += ["", "## 解释边界", ""] + ["- " + row for row in output["limitations"]]
    lines += ["", f"脚本SHA：`{output['script_sha256']}`。全部输入SHA、18个原生模型逐列gain/split、原域与交集分箱、静态子集计数及内部候选细节见同名JSON。"]
    (REPORT / "inner_calibration_v8.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
