"""补充区分9.8%与精确5%flat代理，核09:30恢复使用原09:31报价；不改首次归因。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p/"quant_backtest").is_dir())
REPORT = ROOT/"reports/inventory_coverage_t/run_20260910_182452_383731"
inputs = {}


def read(path, csv=False, **kw):
    """仅已归档三事件/配置/对应内部年股票原行情与一个当前h40表。"""
    path = Path(path).resolve()
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, **kw) if csv else json.loads(path.read_text())


summary = read(OUT/"summary.json")
cfg = read(REPORT/"resolved_config.json")
results = []
for e in summary["episodes"]:
    index = e["candidate_index"]
    raw = read(OUT/f"candidate_{index:03d}_episode_raw.csv", True, float_precision="round_trip")
    stamps = pd.to_datetime(raw.datetime, format="mixed")
    current = raw.loc[stamps.ge(pd.Timestamp(e["exit_earliest_time"]))
        & stamps.dt.normalize().eq(pd.Timestamp(e["first_datetime"]).normalize())]
    sign = 1 if e["recovery_side"] == "buy" else -1
    rate = current.close/e["previous_close"]-1
    flat_price = round(e["previous_close"]*(1+sign*.05), 2)
    high_low_flat = (current.high-current.low).abs().le(1e-6)
    normal_proxy = rate.mul(sign).ge(.098)
    flat_proxy = high_low_flat & current.close.sub(flat_price).abs().le(1e-6)
    full = read(ROOT/cfg["data_root"]/"2026"/f"{e['symbol']}.csv", True)
    full["datetime"] = pd.to_datetime(full.datetime, format="mixed")
    recovery = pd.Timestamp(e["final_recovery"])
    assert recovery.strftime("%H:%M") == "09:30"
    quote_clock = recovery+pd.Timedelta(minutes=1)
    quote = full.loc[full.datetime.eq(quote_clock)]
    assert len(quote) == 1
    trip = read(OUT/f"candidate_{index:03d}_episode_trips.csv", True, float_precision="round_trip")
    expected = float(quote.close.iloc[0])*(1+cfg["slippage_rate"] if sign > 0 else 1-cfg["slippage_rate"])
    actual = trip.buy_price if sign > 0 else trip.sell_price
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)
    if e["horizon_minutes"] == 40:
        # 最初归因只取h60同当前键的previous_close；实际h40相同当前字段在此独立核对。
        forecast = read(REPORT/"year_2023/validation_predictions/h40_l2_depth6/current_predictions.csv", True,
            float_precision="round_trip", usecols=["datetime", "symbol", "previous_close"])
        row = forecast.loc[forecast.symbol.eq(e["symbol"]) & pd.to_datetime(forecast.datetime, format="mixed").eq(
            pd.Timestamp(e["first_datetime"])-pd.Timedelta(minutes=1))]
        assert len(row) == 1 and float(row.previous_close.iloc[0]) == e["previous_close"]
    results.append(dict(candidate_index=index, symbol=e["symbol"], actual_horizon_minutes=e["horizon_minutes"],
        observed_minute_rows=len(current), nonflat_9_8_proxy_rows=int(normal_proxy.sum()),
        exact_5_flat_proxy_rows=int(flat_proxy.sum()), exact_5_candidate_price=flat_price,
        zero_volume_rows=int(current.volume.le(0).sum()), true_st_status_inferred=False,
        next_day_recovery_clock=str(recovery), actual_raw_quote_clock=str(quote_clock),
        raw_recovery_price=float(quote.close.iloc[0]), actual_slipped_recovery_price=expected,
        saved_recovery_price_matches_raw=True))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in inputs.items())
result = dict(passed=True, rows=results, inputs_sha256=inputs, no_outer_read=True,
    no_blacklist_or_rule_changes=True, original_diagnosis_or_runtime_modified=False)
(OUT/"interpretation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
text = """# 三个内部恢复事件的机制说明

全部36主动候选都发生日内恢复失败。跨候选187个延迟/未完成首腿来自21个股票×首腿日；没有单一事件覆盖全部候选。SH600266在1月7日、SH600228在3月23日、SH600893在2月13日三事件的候选并集覆盖全部36，属于事后说明，不能据此删股或使用未来恢复结果过滤入场。

- 规则16活动最少（119完整组），唯一跨日组：SH600266于1月7日09:36卖出，参考6.51、前收6.14，原h60期限10:36。从期限到15:00的175个观测分钟均6.75（约+9.93%），174次9.8%代理拒单和1次零量，1月8日09:30才真实买回。
- 规则18活动次少（128完整组），唯一跨日组：SH600228于3月23日10:01买入，参考5.19、前收5.43，原h60期限11:01。之后150分钟均5.16，等于round(5.43×0.95,2)的精确5%候选下限；其中76次代理拒单、74次零量。这是5%flat保守代理，不能据此声称当时真实ST；3月24日09:30才卖回。
- 规则34（174完整组）的唯一跨日组：SH600893于2月13日13:31卖出，参考56.38、前收51.97。实际选择h40，期限14:11，之后50分钟均57.17（约+10.00%），49次9.8%代理拒单、1次零量；下一观测交易日2月24日09:30才买回。

三组最终都恢复，所以期末pending为0，但不能抹去首次未恢复日及次日库存差异。首腿当刻反向方向检查通过，只说明那一刻不被该价格规则拦截，不保证后来价格/量能保持可恢复。三组意图均实际锁存为hard_deadline，规则18虽允许止损，记录不支持假称它已经提前止损却未执行。

三笔09:30真实恢复按引擎合同使用该日09:31原bar，原价与保存成交价（含原滑点）逐值核对通过。这是已有记录的只读核对，没有补成交、重放账户或改动模型。其他18股票×首腿日原因未逐一重查，不能由三例推断所有失败同因，也不替代完整独立审计。

证据：[全部延迟或未完成首腿](all_delayed_or_pending_pairs.csv)、[原归因及读取SHA](summary.json)、[代理及恢复报价核对](interpretation.json)。原阶段工具与56生产源保持冻结。
"""
(OUT/"interpretation.md").write_text(text)
(OUT/"interpretation_verification.json").write_text(json.dumps(dict(passed=True, inputs_sha256=inputs,
    outputs_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
        for name in ("explain.py", "interpretation.md", "interpretation.json")}), ensure_ascii=False, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
