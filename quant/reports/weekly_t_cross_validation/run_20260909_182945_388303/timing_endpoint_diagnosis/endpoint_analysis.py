"""一次性只读重建 v11 已成交86组的原固定端点扣费事件，不调参或重跑账户。

脚本位于正式报告的独立诊断子目录，只读取对应分股缓存labels；不读取/重建
61m完整标签或217m报价。缺失标签保留为缺失，不从比较分母静默删除。
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
REPORT = OUT.parent


def sha(path):
    """流式记录实际读取文件的字节身份。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, data):
    """严格JSON，无端点信息用null，不伪造零标签。"""
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def net_edges(raw, timestamp, cfg):
    """独立按首腿未滑点参考名义额重建两方向固定端点净额。"""
    commission = float(cfg["commission_rate"]) + float(cfg["transfer_rate"])
    slip = float(cfg["slippage_rate"])
    tax = float(cfg["stamp_rate_before_20230828"] if timestamp < pd.Timestamp("2023-08-28")
                else cfg["stamp_rate_from_20230828"])
    buy = (1+raw)*(1-slip)*(1-commission-tax) - (1+slip)*(1+commission)
    sell = (1-slip)*(1-commission-tax) - (1+raw)*(1+slip)*(1+commission)
    return buy, sell


def summarize(frame):
    """全部实际组均在分母审计中；事件频率仅对明确列出的已知端点报告。"""
    known = frame.loc[frame.endpoint_available]
    if known.empty:
        absent = {key:None for key in ("mean_predicted_net_bps","mean_endpoint_net_bps",
                  "mean_actual_execution_net_bps","mean_endpoint_minus_prediction_bps",
                  "mean_actual_minus_endpoint_bps","mean_predicted_profit_event_probability",
                  "endpoint_net_wins","endpoint_net_win_rate","actual_execution_net_wins",
                  "actual_execution_net_win_rate","endpoint_net_pnl","actual_execution_net_pnl",
                  "actual_minus_endpoint_pnl","different_execution_outcome_pairs",
                  "nontrivial_actual_endpoint_difference_pairs")}
        return {"actual_pairs":len(frame), "endpoint_available_pairs":0, "endpoint_missing_pairs":len(frame),**absent}
    return {"actual_pairs":len(frame),"endpoint_available_pairs":len(known),"endpoint_missing_pairs":len(frame)-len(known),
            "mean_predicted_net_bps":float(known.predicted_net_edge_bps.mean()),
            "mean_endpoint_net_bps":float(known.endpoint_net_bps.mean()),
            "mean_actual_execution_net_bps":float(known.realized_net_bps.mean()),
            "mean_endpoint_minus_prediction_bps":float((known.endpoint_net_bps-known.predicted_net_edge_bps).mean()),
            "mean_actual_minus_endpoint_bps":float((known.realized_net_bps-known.endpoint_net_bps).mean()),
            "mean_predicted_profit_event_probability":float(known.profit_event_probability.mean()),
            "endpoint_net_wins":int(known.endpoint_net_pnl.gt(0).sum()),
            "endpoint_net_win_rate":float(known.endpoint_net_pnl.gt(0).mean()),
            "actual_execution_net_wins":int(known.net_pnl.gt(0).sum()),
            "actual_execution_net_win_rate":float(known.net_pnl.gt(0).mean()),
            "endpoint_net_pnl":float(known.endpoint_net_pnl.sum()),
            "actual_execution_net_pnl":float(known.net_pnl.sum()),
            "actual_minus_endpoint_pnl":float((known.net_pnl-known.endpoint_net_pnl).sum()),
            "different_execution_outcome_pairs":int((known.net_pnl.gt(0)!=known.endpoint_net_pnl.gt(0)).sum()),
            "nontrivial_actual_endpoint_difference_pairs":int((known.net_pnl-known.endpoint_net_pnl).abs().gt(1e-7).sum())}


def main():
    """先固化协议，再一次性按86个已知信号/股票/周期键读取原分股标签。"""
    if (OUT/"summary.json").exists():
        raise FileExistsError("已归档的端点诊断不得覆盖")
    inputs = {}
    def record(path):
        path = Path(path)
        inputs[str(path.resolve())] = sha(path)
        return path
    def read_json(path):
        return json.loads(record(path).read_text(encoding="utf-8"))
    protocol = {"purpose":"fixed_endpoint_event_check_for_86_executed_v11_pairs",
                "posthoc_only":True,"changes_parameters_or_models":False,"reruns_accounts":False,
                "expected_pair_count":86,"label_source":"original horizon_cache per-symbol labels",
                "probability_bins":[0,.4,.5,.6,.7,.8,.9,1.0000001],
                "confidence_intervals":"none; no IID extrapolation",
                "missing_endpoint_policy":"retain all groups with explicit missing reason",
                "script_sha256":sha(__file__)}
    write_json(OUT/"protocol.json",protocol)
    cfg = read_json(REPORT/"resolved_config.json")
    if cfg["min_commission"] != 0:
        raise ValueError("原固定端点公式必须为最低佣金0")
    manifest = read_json(REPORT/"horizon_manifest.json")
    cache = Path(manifest["cache_path"])
    cached_metadata = read_json(cache/"metadata.json")
    if cached_metadata["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("原标签缓存身份不一致")
    prior = read_json(REPORT/"timing_quality_diagnosis/diagnosis.json")
    pairs_path = record(REPORT/"timing_quality_diagnosis/actual_pairs.csv")
    if sha(pairs_path) != prior["outputs_sha256"]["actual_pairs.csv"]:
        raise ValueError("已成交配对诊断已改变")
    pairs = pd.read_csv(pairs_path, float_precision="round_trip")
    pairs = pairs.loc[pairs.dataset.eq("v11")].copy()
    if len(pairs)!=86 or pairs[["test_year","pair_id"]].duplicated().any():
        raise ValueError("不是预声明的86个唯一实际配对")
    for column in ("signal_datetime","first_datetime","last_recovery_datetime","planned_recovery_time"):
        pairs[column] = pd.to_datetime(pairs[column],format="mixed")
    rows=[]
    for symbol, selected in pairs.groupby("symbol",sort=True):
        path = cache/f"{symbol}.pkl"
        if path.exists():
            payload = pd.read_pickle(record(path))
            labels = payload["labels"]
            if labels[["datetime","symbol","horizon_minutes"]].duplicated().any():
                raise ValueError(f"原分股标签重复：{symbol}")
            indexed = labels.set_index(["datetime","symbol","horizon_minutes"],verify_integrity=True)
        else:
            indexed = None
        for original in selected.to_dict("records"):
            item={**original,"endpoint_available":False,"endpoint_missing_reason":"",
                  "endpoint_execution_price":np.nan,"endpoint_recovery_price":np.nan,
                  "endpoint_raw_return":np.nan,"endpoint_buy_net_return":np.nan,"endpoint_sell_net_return":np.nan,
                  "endpoint_profit_class":np.nan,"endpoint_net_bps":np.nan,"endpoint_net_pnl":np.nan,
                  "actual_minus_endpoint_pnl":np.nan,"label_execution_time":pd.NaT,"label_recovery_time":pd.NaT}
            key=(original["signal_datetime"],symbol,int(original["horizon_minutes"]))
            if indexed is None or key not in indexed.index:
                item["endpoint_missing_reason"]="missing_symbol_cache" if indexed is None else "missing_label_key"
                rows.append(item);continue
            label=indexed.loc[key]
            item["label_execution_time"],item["label_recovery_time"]=label.execution_time,label.recovery_time
            prices=np.asarray([label.execution_price,label.recovery_price,label.target_return],dtype=float)
            if not np.isfinite(prices).all() or prices[0]<=0 or prices[1]<=0:
                item["endpoint_missing_reason"]="missing_or_invalid_endpoint_price_or_target"
                rows.append(item);continue
            if (pd.Timestamp(label.execution_time)!=original["first_datetime"]
                    or pd.Timestamp(label.recovery_time)!=original["planned_recovery_time"]):
                raise ValueError("实际组与原标签的执行/计划恢复时钟不符")
            raw=float(label.recovery_price/label.execution_price-1)
            if not np.isclose(raw,label.target_return,rtol=1e-12,atol=1e-14):
                raise ValueError("原标签目标与原价格比不符")
            reference=original["shares"]*float(label.execution_price)
            if not np.isclose(reference,original["first_reference_notional"],rtol=1e-12,atol=1e-9):
                raise ValueError("原首腿参考价与标签执行参考价不符")
            buy,sell=net_edges(raw,original["signal_datetime"],cfg)
            target=buy if original["direction"]=="buy_first" else sell
            predicted_buy,predicted_sell=net_edges(original["raw_prediction"],original["signal_datetime"],cfg)
            predicted=predicted_buy if original["direction"]=="buy_first" else predicted_sell
            if not np.isclose(predicted*10000,original["predicted_net_edge_bps"],rtol=1e-12,atol=1e-9):
                raise ValueError("保存预测边际与独立费率换算不符")
            item.update(endpoint_available=True,endpoint_execution_price=float(label.execution_price),
                        endpoint_recovery_price=float(label.recovery_price),endpoint_raw_return=raw,
                        endpoint_buy_net_return=buy,endpoint_sell_net_return=sell,
                        endpoint_profit_class=1 if buy>0 else 2 if sell>0 else 0,
                        endpoint_net_bps=target*10000,endpoint_net_pnl=target*reference,
                        actual_minus_endpoint_pnl=original["net_pnl"]-target*reference)
            rows.append(item)
    result=pd.DataFrame(rows).sort_values(["test_year","pair_id"],kind="stable").reset_index(drop=True)
    result.to_csv(OUT/"endpoint_pairs.csv",index=False,encoding="utf-8-sig")
    aggregate=summarize(result)
    groups=[]
    for dimension in ("test_year","horizon_minutes","direction"):
        for key,part in result.groupby(dimension,sort=True):
            groups.append({"dimension":dimension,"group":str(key),**summarize(part)})
    pd.DataFrame(groups).to_csv(OUT/"endpoint_group_summary.csv",index=False,encoding="utf-8-sig")
    result["probability_bin"]=pd.cut(result.profit_event_probability,protocol["probability_bins"],right=False)
    bins=[{"probability_bin":str(key),**summarize(part)} for key,part in result.groupby("probability_bin",observed=True,sort=True)]
    pd.DataFrame(bins).to_csv(OUT/"endpoint_probability_bins.csv",index=False,encoding="utf-8-sig")
    fmt=lambda value,kind='.4f':"未知" if value is None else format(value,kind)
    lines=["# 原固定端点的预测兑现诊断", "",
           "本诊断只追踪已经成交的86组，不改变信号、训练或交易规则。使用原分股标签的执行参考价和计划恢复参考价，"
           "按原佣金、滑点、过户费及当日卖出印花税独立计算同一方向净额；没有把实际延迟/分批恢复终点充当原标签。", "",
           f"实际组数 {aggregate['actual_pairs']}，原固定端点可用 {aggregate['endpoint_available_pairs']}，"
           f"缺失 {aggregate['endpoint_missing_pairs']}。缺失行仍完整保留在endpoint_pairs.csv，未默认为亏损、胜利或零收益。", "",
           f"预测净边际均值 **{fmt(aggregate['mean_predicted_net_bps'])}bp**，原固定端点实际标称净额均值 "
           f"**{fmt(aggregate['mean_endpoint_net_bps'])}bp**，真实执行组均值 {fmt(aggregate['mean_actual_execution_net_bps'])}bp；"
           "均以每组首腿参考名义额归一后等权平均。", "",
           f"原固定端点净胜率 {fmt(aggregate['endpoint_net_win_rate'],'.2%')}（{aggregate['endpoint_net_wins']}组）；"
           f"真实执行组净胜率 {fmt(aggregate['actual_execution_net_win_rate'],'.2%')}。"
           f"固定端点标称净额合计 {fmt(aggregate['endpoint_net_pnl'],'.6f')}元，实际组净额合计 {fmt(aggregate['actual_execution_net_pnl'],'.6f')}元，"
           f"实际减固定端点为 {fmt(aggregate['actual_minus_endpoint_pnl'],'.6f')}元；"
           f"超过1e-7元差异的组 {aggregate['nontrivial_actual_endpoint_difference_pairs']}，盈利符号改变 {aggregate['different_execution_outcome_pairs']}组。", "",
           "| 预测事件概率区间 | 全部组/有效端点 | 预测概率均值 | 原同一端点事件频率 | 平均预测净bp | 平均端点净bp |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in bins:
        lines.append(f"| {item['probability_bin']} | {item['actual_pairs']}/{item['endpoint_available_pairs']} | "
                     f"{fmt(item['mean_predicted_profit_event_probability'],'.2%')} | {fmt(item['endpoint_net_win_rate'],'.2%')} | "
                     f"{fmt(item['mean_predicted_net_bps'])} | {fmt(item['mean_endpoint_net_bps'])} |")
    lines += ["", "此处概率比较已经使用原同一固定端点的扣费事件，但仅覆盖被策略选择且实际成交的样本；"
              "不代表全体预测分布、全市场概率校准或未来胜率。不给出IID置信区间，不用这些历史分箱重新选阈值。"
              "原模型概率未经校准，概率标签也不包含真实成交可行性。", "",
              "所有净额均已包含一次滑点与费用；没有在实际net_pnl上再次扣滑点。"
              "分组表和逐组端点保留原86组及缺失原因，输入文件和脚本SHA见summary.json。", ""]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    if any(sha(Path(path))!=expected for path,expected in inputs.items()):
        raise AssertionError("只读诊断期间输入改变")
    outputs={path.name:sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name!="summary.json"}
    write_json(OUT/"summary.json",{"protocol":protocol,"aggregate":aggregate,"groups":groups,"probability_bins":bins,
               "input_sha256":inputs,"output_sha256":outputs,"all_inputs_unchanged":True,
               "symbol_cache_files_read":int(result.symbol.nunique()),"full_label_cache_loaded":False,"quotes_rebuilt":False})
    print(json.dumps(aggregate,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
