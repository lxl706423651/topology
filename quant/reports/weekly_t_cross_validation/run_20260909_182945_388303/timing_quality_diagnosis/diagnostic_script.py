"""只读比较已完成 v11 与 v10 的实际完整 T 配对，不拟合、选参或重跑账户。

从仓库根目录运行 --report-dir 新报告目录；默认参考已冻结 v10。只写新报告下
timing_quality_diagnosis/，保留读取输入及输出 SHA。胜率按完整 pair_id，不按恢复片段。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reports/weekly_t_cross_validation/run_20260909_135906_004933"
EDGE_BINS = [-np.inf, 0, 5, 10, 20, 40, 80, 160, np.inf]
PROBABILITY_BINS = [0, .4, .5, .6, .7, .8, .9, 1.0000001]
FAILURE_KEYS = ["inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
                "base_plan_mismatch_count", "t1_violation_count", "weekly_liquidation_failure_days"]


def sha256(path: Path) -> str:
    """流式记录读取文件字节身份。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path, inputs: dict, columns: list | None = None) -> pd.DataFrame:
    """不改变 CSV，保留浮点往返精度；空文件表示空记录而非缺文件。"""
    inputs[str(path.resolve())] = sha256(path)
    try:
        return pd.read_csv(path, float_precision="round_trip", usecols=(lambda c: c in columns) if columns else None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path, inputs: dict) -> dict:
    """读取并冻结 JSON 来源。"""
    inputs[str(path.resolve())] = sha256(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, values: dict) -> None:
    """严格 JSON 编码；无配对时的胜率和均值用 null。"""
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (Path, pd.Timestamp)):
            return str(value)
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def boolean(value) -> bool:
    """不会把 CSV 字符串 False 当成真。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"非法布尔值 {value!r}")


def numeric_sum(frame: pd.DataFrame, column: str) -> float:
    """缺失非空账目拒绝静默当零。"""
    if frame.empty:
        return 0.
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"非法账目列 {column}")
    return float(values.sum())


def trading_horizon(first: pd.Timestamp, planned: pd.Timestamp) -> int:
    """独立按真实会话时钟计算决策恢复期限，跨午休而不跨日；不读取规则单h。"""
    if first.normalize() != planned.normalize():
        raise ValueError("计划恢复与首腿必须同日")
    clocks = list(range(9 * 60 + 31, 11 * 60 + 31)) + list(range(13 * 60 + 1, 15 * 60 + 1))
    begin, end = first.hour * 60 + first.minute, planned.hour * 60 + planned.minute
    if first.second or planned.second or begin not in clocks or end not in clocks:
        raise ValueError("计划或首腿不在支持的完整交易分钟")
    horizon = clocks.index(end) - clocks.index(begin)
    if horizon not in (20, 40, 60):
        raise ValueError(f"计划恢复期限非法 {horizon}")
    return horizon


def net_edges(prediction: np.ndarray, times: pd.Series, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """按首腿参考名义额计算预测净边际；费用与滑点各只扣一次。"""
    if float(cfg["min_commission"]) != 0:
        raise ValueError("诊断预测净边际公式只支持原最低佣金0配置")
    fee = cfg["commission_rate"] + cfg["transfer_rate"]
    slip = cfg["slippage_rate"]
    stamp = np.where(times < pd.Timestamp("2023-08-28"), cfg["stamp_rate_before_20230828"], cfg["stamp_rate_from_20230828"])
    ratio = 1 + prediction
    return (ratio * (1-slip) * (1-fee-stamp) - (1+slip) * (1+fee),
            (1-slip) * (1-fee-stamp) - ratio * (1+slip) * (1+fee))


def read_predictions(folder: Path, inputs: dict) -> pd.DataFrame:
    """读取保存的外层预测，不加载模型；新周期分表和旧单表明确区分。"""
    paths = sorted(folder.glob("t_h*_predictions.csv"))
    if not paths and (folder / "t_predictions.csv").exists():
        paths = [folder / "t_predictions.csv"]
    columns = ["datetime", "symbol", "prediction", "raw_prediction", "prob_buy_profit",
               "prob_sell_profit", "horizon_minutes"]
    pieces = []
    for path in paths:
        frame = read_csv(path, inputs, columns)
        if frame.empty:
            continue
        frame["datetime"] = pd.to_datetime(frame.datetime, format="mixed")
        if "horizon_minutes" not in frame:
            frame["horizon_minutes"] = -1  # 旧v10单模型表，实际周期仍由决策恢复时钟决定。
        if frame[["datetime", "symbol", "horizon_minutes"]].duplicated().any():
            raise ValueError(f"重复预测键 {path}")
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    if result[["datetime", "symbol", "horizon_minutes"]].duplicated().any():
        raise ValueError("多份预测表包含相同周期键，不能任选其一")
    return result


def describe_pairs(pairs: pd.DataFrame) -> dict:
    """完整组定义为已恢复全部股数；部分恢复仅单独披露，不进入净胜率分母。"""
    complete = pairs.loc[pairs.complete] if len(pairs) else pairs
    count = len(complete)
    gross = numeric_sum(complete, "reference_gross_pnl")
    net = numeric_sum(complete, "net_pnl")
    return {"started_pairs": len(pairs), "complete_pairs": count, "pending_pairs": len(pairs)-count,
            "recovery_fragments": int(pairs.fragment_count.sum()) if len(pairs) else 0,
            "net_wins": int(complete.net_pnl.gt(0).sum()) if count else 0,
            "net_losses": int(complete.net_pnl.lt(0).sum()) if count else 0,
            "net_zeros": int(complete.net_pnl.eq(0).sum()) if count else 0,
            "net_win_rate": float(complete.net_pnl.gt(0).mean()) if count else None,
            "gross_win_rate": float(complete.reference_gross_pnl.gt(0).mean()) if count else None,
            "reference_gross_pnl": gross, "slippage_cost": numeric_sum(complete, "slippage_cost"),
            "fees": numeric_sum(complete, "fees"), "net_pnl": net,
            "mean_reference_gross_per_pair": gross/count if count else None,
            "mean_net_per_pair": net/count if count else None,
            "mean_realized_net_bps": float(complete.realized_net_bps.mean()) if count else None,
            "cross_day_pairs": int(complete.cross_day.sum()) if count else 0,
            "active_days": int(pairs.signal_datetime.dt.normalize().nunique()) if len(pairs) else 0}


def diagnose_year(report: Path, year: int, dataset: str, cfg: dict, inputs: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """首腿(signal_datetime,symbol)精确连接决策及对应周期预测，并重建完整pair。"""
    folder = report / f"year_{year}"
    rule = read_json(folder / "selected_rule.json", inputs)
    events = read_csv(folder / "test_trade_events.csv", inputs)
    trips = read_csv(folder / "test_round_trips.csv", inputs)
    decisions = read_csv(folder / "t_decisions.csv", inputs)
    curve = read_csv(folder / "test_equity_curve.csv", inputs)
    metrics = read_csv(folder / "test_metrics.csv", inputs)
    if len(metrics) != 1:
        raise ValueError("年度指标必须一行")
    metrics = metrics.iloc[0].to_dict()
    predictions = read_predictions(folder, inputs)
    enabled = boolean(rule["enabled"])
    first = events.loc[events.event_type.eq("t") & events.fragment_id.eq(0)].copy()
    if len(first) and first.pair_id.duplicated().any():
        raise ValueError("pair_id首腿不唯一")
    if not enabled and (len(first) or len(trips) or len(decisions)):
        raise ValueError("hold不能隐藏请求或真实T成交")
    if len(decisions):
        decisions["datetime"] = pd.to_datetime(decisions.datetime, format="mixed")
        if decisions[["datetime", "symbol"]].duplicated().any():
            raise ValueError("实际首腿对应的同刻股票决策不唯一")
    rows = []
    if len(first):
        if decisions.empty or predictions.empty:
            raise ValueError("存在实际首腿却没有保存决策或预测")
        first["signal_datetime"] = pd.to_datetime(first.signal_datetime, format="mixed")
        first["datetime"] = pd.to_datetime(first.datetime, format="mixed")
        decisions = decisions.rename(columns={"datetime": "decision_datetime"})
        merged = first.merge(decisions, left_on=["signal_datetime", "symbol"], right_on=["decision_datetime", "symbol"],
                             how="left", validate="one_to_one", suffixes=("", "_decision"), indicator=True)
        if not merged._merge.eq("both").all():
            raise ValueError("实际首腿未匹配到唯一原始决策")
        prediction_index = predictions.set_index(["datetime", "symbol", "horizon_minutes"], verify_integrity=True)
        if len(trips) and not set(trips.pair_id).issubset(set(first.pair_id)):
            raise ValueError("存在没有首腿的恢复片段")
        for event in merged.to_dict("records"):
            fragments = trips.loc[trips.pair_id.eq(event["pair_id"])] if len(trips) else trips
            planned = pd.Timestamp(event["planned_recovery_time_decision"])
            if planned != pd.Timestamp(event["planned_recovery_time"]):
                raise ValueError("首腿实际恢复计划偏离原始决策")
            horizon = trading_horizon(event["datetime"], planned)
            if "horizon_minutes" in event and int(event["horizon_minutes"]) != horizon:
                raise ValueError("决策horizon与实际首腿计划时钟不符")
            key = (event["signal_datetime"], event["symbol"], horizon)
            if key not in prediction_index.index:
                key = (event["signal_datetime"], event["symbol"], -1)
            if key not in prediction_index.index:
                raise ValueError("实际首腿缺所选周期的保存预测")
            prediction = prediction_index.loc[key]
            raw = float(prediction.get("raw_prediction", prediction["prediction"]))
            action = int(event["action"])
            if action not in (-1, 1) or (event["side"] == "buy") != (action == 1):
                raise ValueError("实际首腿方向与决策相反")
            buy, sell = net_edges(np.array([raw]), pd.Series([event["signal_datetime"]]), cfg)
            edge = float((buy if action == 1 else sell)[0])
            prob = prediction.get("prob_buy_profit" if action == 1 else "prob_sell_profit", np.nan)
            for column, expected in (("predicted_net_edge", edge), ("raw_prediction", raw), ("prob_profit", prob)):
                if column in event and not np.isclose(float(event[column]), expected, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"决策字段 {column} 与保存模型预测不符")
            if len(fragments):
                if not fragments.symbol.eq(event["symbol"]).all():
                    raise ValueError("同pair混入股票")
                complete_flags = fragments.pair_fully_recovered.map(boolean)
                if complete_flags.nunique() != 1:
                    raise ValueError("同pair完成状态冲突")
                complete = bool(complete_flags.iloc[-1])
                recovered = numeric_sum(fragments, "shares")
                if (recovered > event["shares"] or (complete and recovered != event["shares"])):
                    raise ValueError("恢复股数与首腿不符")
                end = pd.to_datetime(fragments.second_datetime, format="mixed").max()
            else:
                complete, recovered, end = False, 0, pd.NaT
            executed_gross = numeric_sum(fragments, "gross_pnl")
            slip = numeric_sum(fragments, "slippage_cost")
            fees = numeric_sum(fragments, "buy_fee") + numeric_sum(fragments, "sell_fee")
            net = numeric_sum(fragments, "net_pnl")
            if not np.isclose(executed_gross-fees, net, rtol=1e-10, atol=1e-7):
                raise ValueError("配对毛差减费用不等于净收益")
            pair_events = events.loc[events.event_type.eq("t") & events.pair_id.eq(event["pair_id"])]
            if complete and (not np.isclose(numeric_sum(pair_events, "fee"), fees, atol=1e-7)
                             or not np.isclose(numeric_sum(pair_events, "slippage_cost"), slip, atol=1e-7)):
                raise ValueError("完整pair片段的费用/滑点与事件不符")
            notional = float(event["shares"] * event["reference_price"])
            if not np.isfinite(notional) or notional <= 0:
                raise ValueError("非法首腿参考名义额")
            signal = event["signal_datetime"]
            rows.append({"dataset":dataset, "test_year":year, "pair_id":event["pair_id"], "symbol":event["symbol"],
                         "signal_datetime":signal, "first_datetime":event["datetime"], "last_recovery_datetime":end,
                         "planned_recovery_time":planned, "horizon_minutes":horizon,
                         "horizon_source":"saved_decision_and_execution_clock", "signal_clock":signal.strftime("%H:%M"),
                         "session":"morning" if signal.hour < 12 else "afternoon",
                         "direction":"buy_first" if action==1 else "sell_first", "shares":int(event["shares"]),
                         "recovered_shares":int(recovered), "complete":complete, "fragment_count":len(fragments),
                         "cross_day":bool(pd.notna(end) and end.normalize()!=event["datetime"].normalize()),
                         "first_reference_notional":notional, "reference_gross_pnl":executed_gross+slip,
                         "executed_gross_pnl":executed_gross, "slippage_cost":slip, "fees":fees, "net_pnl":net,
                         "realized_net_bps":net/notional*10000 if complete else np.nan,
                         "raw_prediction":raw, "predicted_net_edge_bps":edge*10000,
                         "predicted_net_pnl":edge*notional, "profit_event_probability":prob,
                         "net_prediction_error_bps":net/notional*10000-edge*10000 if complete else np.nan})
    columns = ["dataset","test_year","pair_id","complete","fragment_count","net_pnl","reference_gross_pnl",
               "slippage_cost","fees","cross_day","signal_datetime","realized_net_bps","profit_event_probability",
               "predicted_net_edge_bps","direction","horizon_minutes","signal_clock","session"]
    pairs = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    stats = {"dataset":dataset,"test_year":year,"enabled":enabled,"state":"hold" if not enabled else "active_traded" if len(first) else "active_no_fills",
             "decision_rows":len(decisions),"saved_prediction_rows":len(predictions),**describe_pairs(pairs),
             "all_t_event_fees":numeric_sum(events.loc[events.event_type.eq("t")],"fee"),
             "all_t_event_slippage":numeric_sum(events.loc[events.event_type.eq("t")],"slippage_cost"),
             "account_t_increment":float(curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1]),
             **{key:int(metrics[key]) for key in FAILURE_KEYS}}
    if (stats["complete_pairs"] != int(metrics["t_pair_count"])
            or len(pairs)-stats["complete_pairs"] != int(metrics["pending_pairs"])):
        raise ValueError("重建组数与保存年度指标不符")
    return pairs, stats, curve


def group_diagnostics(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """固定方向/期限/时刻分组及预测分箱，只作事后归因，不能用来改阈值。"""
    records, calibration = [], []
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    for column in ("direction", "horizon_minutes", "signal_clock", "session"):
        for key, part in pairs.groupby(["dataset","test_year",column], dropna=False, sort=True):
            records.append({"dataset":key[0],"test_year":int(key[1]),"dimension":column,"group":str(key[2]),**describe_pairs(part)})
    complete = pairs.loc[pairs.complete].copy()
    for kind, column, bins in (("predicted_net_edge_bps","predicted_net_edge_bps",EDGE_BINS),
                               ("profit_event_probability","profit_event_probability",PROBABILITY_BINS)):
        observed = complete.loc[pd.to_numeric(complete[column], errors="coerce").notna()].copy()
        observed["bin"] = pd.cut(observed[column].astype(float), bins, right=False)
        for key, part in observed.groupby(["dataset","test_year","bin"], observed=True, sort=True):
            count = len(part)
            probability = pd.to_numeric(part.profit_event_probability, errors="coerce")
            prob_count = int(probability.notna().sum())
            calibration.append({"dataset":key[0],"test_year":int(key[1]),"dimension":kind,"bin":str(key[2]),
                                "complete_pairs":count,"net_win_rate":float(part.net_pnl.gt(0).mean()),
                                "mean_predicted_net_edge_bps":float(part.predicted_net_edge_bps.mean()),
                                "mean_realized_net_bps":float(part.realized_net_bps.mean()),
                                "mean_net_prediction_error_bps":float(part.net_prediction_error_bps.mean()),
                                "net_pnl":float(part.net_pnl.sum()), "probability_sample_count":prob_count,
                                "mean_profit_event_probability":float(probability.mean()) if prob_count else np.nan,
                                "realized_win_minus_probability":float((part.loc[probability.notna(),"net_pnl"].gt(0).astype(float)-probability.dropna()).mean()) if prob_count else np.nan})
    return pd.DataFrame(records), pd.DataFrame(calibration)


def run_diagnosis(report: Path, reference: Path) -> Path:
    """完成报告后才读取外层结果，输出独立子目录并核对所有输入未变。"""
    report, reference = report.resolve(), reference.resolve()
    if report == reference:
        raise ValueError("新报告与参考目录不能相同")
    for path in (report, reference):
        if not (path/"final_report.md").is_file():
            raise ValueError(f"只诊断已完成报告：{path}")
    output = report / "timing_quality_diagnosis"
    output.mkdir(exist_ok=False)
    (output/"diagnostic_script.py").write_bytes(Path(__file__).read_bytes())
    inputs = {}
    protocol = {"purpose":"posthoc_fixed_rule_timing_quality_diagnosis", "report_dir":str(report),
                "reference_frequency_dir":str(reference), "retrain":False,"rerun_accounts":False,
                "changes_strategy_selection":False,"win_rate_unit":"fully_recovered_pair_id_within_year",
                "incomplete_pairs_excluded_from_win_rate":True,"hold_win_rate":None,
                "net_edge_bins_bps":["-inf",0,5,10,20,40,80,160,"inf"],
                "probability_bins":PROBABILITY_BINS,"script_sha256":sha256(Path(__file__)),
                "probability_note":"uncalibrated fixed-endpoint fee-profit event; not actual fill probability"}
    write_json(output/"protocol.json",protocol)
    pieces, summaries, comparisons = [], [], []
    current_cfg = read_json(report/"resolved_config.json",inputs)
    reference_cfg = read_json(reference/"resolved_config.json",inputs)
    years = sorted(int(year) for year in current_cfg["test_years"])
    for year in years:
        current, current_stats, current_curve = diagnose_year(report,year,"v11",current_cfg,inputs)
        previous, previous_stats, previous_curve = diagnose_year(reference,year,"v10",reference_cfg,inputs)
        pieces += [current,previous]; summaries += [current_stats,previous_stats]
        same_dates = pd.to_datetime(current_curve.datetime,format="mixed").equals(pd.to_datetime(previous_curve.datetime,format="mixed"))
        error = float(np.max(np.abs(current_curve.baseline_equity.to_numpy()-previous_curve.baseline_equity.to_numpy()))) if same_dates else None
        comparable = same_dates and error == 0
        comparisons.append({"test_year":year,"same_dates":same_dates,"baseline_max_difference":error,
                            "same_baseline_comparable":comparable,
                            "current_complete_pairs":current_stats["complete_pairs"],"reference_complete_pairs":previous_stats["complete_pairs"],
                            "current_net_win_rate":current_stats["net_win_rate"],"reference_net_win_rate":previous_stats["net_win_rate"],
                            "net_win_rate_change":current_stats["net_win_rate"]-previous_stats["net_win_rate"] if current_stats["complete_pairs"] and previous_stats["complete_pairs"] else None,
                            "current_account_t_increment":current_stats["account_t_increment"],"reference_account_t_increment":previous_stats["account_t_increment"],
                            "account_increment_change":current_stats["account_t_increment"]-previous_stats["account_t_increment"] if comparable else None})
    all_pairs = pd.concat(pieces,ignore_index=True)
    groups, calibration = group_diagnostics(all_pairs)
    annual, comparison = pd.DataFrame(summaries), pd.DataFrame(comparisons)
    for name, frame in (("actual_pairs.csv",all_pairs),("annual_pair_quality.csv",annual),
                        ("grouped_pair_quality.csv",groups),("prediction_realization.csv",calibration),
                        ("comparison_v10.csv",comparison)):
        frame.to_csv(output/name,index=False,encoding="utf-8-sig")
    totals = {name:describe_pairs(all_pairs.loc[all_pairs.dataset.eq(name)]) for name in ("v11","v10")}
    lines = ["# v11择时能力：固定外层结果事后诊断", "",
             "此处只比较保存的原规则实际成交，不训练、不重跑账户、不选择新阈值。"
             "净胜率以完整pair为单位，部分恢复片段不重复算交易；hold没有胜率，不能写成100%。", "",
             "| 年份 | v11/v10完整组 | v11/v10净胜率 | v11/v10平均净额（元/组） | v11/v10账户T增量（元） |",
             "| --- | ---: | ---: | ---: | ---: |"]
    percent = lambda value:"无成交" if value is None or pd.isna(value) else f"{value:.2%}"
    number = lambda value:"无成交" if value is None or pd.isna(value) else f"{value:.2f}"
    for year in years:
        new = next(row for row in summaries if row["dataset"]=="v11" and row["test_year"]==year)
        old = next(row for row in summaries if row["dataset"]=="v10" and row["test_year"]==year)
        lines.append(f"| {year} | {new['complete_pairs']}/{old['complete_pairs']} | {percent(new['net_win_rate'])}/{percent(old['net_win_rate'])} | "
                     f"{number(new['mean_net_per_pair'])}/{number(old['mean_net_per_pair'])} | {new['account_t_increment']:.2f}/{old['account_t_increment']:.2f} |")
    lines += ["", *[f"{name}完整组：参考毛差 {data['reference_gross_pnl']:.2f} − 滑点 {data['slippage_cost']:.2f} − "
                       f"费用 {data['fees']:.2f} = 净额 {data['net_pnl']:.2f} 元；完整组 {data['complete_pairs']}、"
                       f"恢复片段 {data['recovery_fragments']}、待恢复 {data['pending_pairs']}。" for name,data in totals.items()], "",
              "参考毛差=gross_pnl+slippage_cost；gross_pnl已使用含滑点成交价，net_pnl已扣费用，不能再次扣滑点。"
              "完整组净额与账户增量可能因未恢复库存估值不同，原年度库存、恢复、T+1、底仓差异与周退出失败保存在annual_pair_quality.csv。", "",
              "actual_pairs.csv将实际首腿以signal_datetime和symbol精确连接原决策；实际选择的20/40/60分钟来自决策和执行时钟，"
              "不从规则假设固定单周期。grouped_pair_quality.csv按方向、周期、信号时刻和上午/下午报告净额、费用、组数与胜率。", "",
              "prediction_realization.csv报告固定预测净边际分箱的真实净额bp、误差及样本数。盈利概率分箱仅包含保存该概率的完整成交组，"
              "并给出概率均值、真实胜率及差值；它不是重新概率校准。模型目标是假设固定端点成交的扣费事件，"
              "实际成交受到选择、费用、分批和恢复失败影响，两个概率概念不能直接等同；小组样本不能证明改进。", "",
              "same_baseline_comparable=false时不计算可比账户增量变化。四年是独立账户，合计不代表复利；2026仅到4月30日。"
              "本轮在反复研究的历史外层上作事后归因，不能根据这些分组重新选择方向、时段或门槛，也不能由它宣称未来盈利。", ""]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    changed = [path for path,signature in inputs.items() if sha256(Path(path)) != signature]
    if changed:
        raise AssertionError(f"诊断期间原输入改变：{changed}")
    output_hashes = {path.name:sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output/"diagnosis.json",{"protocol":protocol,"annual":summaries,"totals":totals,
               "comparisons":comparisons,"inputs_sha256":inputs,"outputs_sha256":output_hashes,
               "all_inputs_unchanged":True,"posthoc_only":True})
    return output


def main(argv: list[str] | None = None) -> None:
    """新目录只执行一次；已有诊断不覆盖，避免改写已归档证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--reference-frequency-dir",type=Path,default=REFERENCE)
    args = parser.parse_args(argv)
    print(run_diagnosis(args.report_dir,args.reference_frequency_dir),flush=True)


if __name__ == "__main__":
    main()
