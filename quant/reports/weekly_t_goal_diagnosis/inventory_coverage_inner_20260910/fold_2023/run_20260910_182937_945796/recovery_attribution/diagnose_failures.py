"""首折三个事后代表恢复事件的只读归因，不排除股票、不修改门槛或重放账户。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
STAGE = OUT.parent
ROOT = next(parent for parent in OUT.parents if (parent/"quant_backtest").is_dir())
REPORT = ROOT/"reports/inventory_coverage_t/run_20260910_182452_383731"
inputs = {}


def read(path, csv=False, **kwargs):
    """读前SHA、写输出前重核；仅已冻结首折内部及2026三只原CSV。"""
    path = Path(path).resolve()
    assert (path.is_relative_to(STAGE) or path == REPORT/"resolved_config.json"
        or path.is_relative_to(REPORT/"year_2023/validation_candidates")
        or path.is_relative_to(REPORT/"year_2023/validation_predictions")
        or (path.parent.name == "2026" and path.stem in {"SH.600266", "SH.600228", "SH.600893"}))
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    assert str(path) not in inputs or inputs[str(path)] == value
    inputs[str(path)] = value
    return pd.read_csv(path, **kwargs) if csv else json.loads(path.read_text())


cfg = read(REPORT/"resolved_config.json")
failures = read(STAGE/"candidate_failures.csv", True, float_precision="round_trip")
pairs = read(STAGE/"candidate_pairs.csv", True, float_precision="round_trip")
overview = read(STAGE/"candidate_overview.csv", True, float_precision="round_trip")
pairs["first_date"] = pd.to_datetime(pairs.first_datetime, format="mixed").dt.strftime("%Y-%m-%d")
last_date = pd.to_datetime(pairs.final_recovery, format="mixed").dt.strftime("%Y-%m-%d")
bad = pairs.loc[~pairs.complete | last_date.ne(pairs.first_date)].copy()
assert set(bad.candidate_index) == set(range(1, 37))
bad.to_csv(OUT/"all_delayed_or_pending_pairs.csv", index=False, encoding="utf-8-sig")
failures.groupby(["symbol", "date"]).agg(candidate_count=("candidate_index", "nunique"),
    candidate_indices=("candidate_index", lambda values: json.dumps(sorted(set(values))))).reset_index().to_csv(
    OUT/"failure_stock_days.csv", index=False, encoding="utf-8-sig")
groups = {key: set(g.candidate_index) for key, g in bad.groupby(["symbol", "first_date"])}
remaining, cover = set(range(1, 37)), []
while remaining:
    key = max(groups, key=lambda k: (len(groups[k] & remaining), k))
    new = groups[key] & remaining
    assert new
    cover.append(dict(symbol=key[0], first_date=key[1], candidates=sorted(groups[key]), added_candidates=sorted(new)))
    remaining -= new

episodes = []
for candidate, symbol, first_date in ((16, "SH.600266", "2026-01-07"),
                                     (18, "SH.600228", "2026-03-23"), (34, "SH.600893", "2026-02-13")):
    selected = bad.loc[bad.candidate_index.eq(candidate) & bad.symbol.eq(symbol) & bad.first_date.eq(first_date)]
    assert len(selected) == 1
    pair = selected.iloc[0]
    folder = REPORT/f"year_2023/validation_candidates/candidate_{candidate:03d}"
    block = read(folder/"blocked_orders.csv", True, float_precision="round_trip")
    intent = read(folder/"exit_decisions.csv", True, float_precision="round_trip")
    trip = read(folder/"test_round_trips.csv", True, float_precision="round_trip")
    intent = intent.loc[intent.pair_id.eq(pair.pair_id)]
    trip = trip.loc[trip.pair_id.eq(pair.pair_id)]
    assert len(intent) == 1
    intent = intent.iloc[0]
    first, end = pd.Timestamp(pair.first_datetime), pd.Timestamp(pair.final_recovery)
    stamps = pd.to_datetime(block.datetime, format="mixed")
    block = block.loc[block.symbol.eq(symbol) & block.event_type.eq("t_recovery") & stamps.ge(first) & stamps.le(end)].copy()
    block.to_csv(OUT/f"candidate_{candidate:03d}_episode_blocked.csv", index=False, encoding="utf-8-sig")
    trip.to_csv(OUT/f"candidate_{candidate:03d}_episode_trips.csv", index=False, encoding="utf-8-sig")
    profile = overview.loc[overview.candidate_index.eq(candidate), "profile"].iloc[0]
    pred = read(REPORT/f"year_2023/validation_predictions/h60_{profile}/current_predictions.csv", True,
        float_precision="round_trip", usecols=["datetime", "symbol", "previous_close"])
    pred = pred.loc[pred.symbol.eq(symbol) & pd.to_datetime(pred.datetime, format="mixed").eq(pd.Timestamp(pair.datetime))]
    assert len(pred) == 1
    previous_close = float(pred.previous_close.iloc[0])
    raw = read(ROOT/cfg["data_root"]/"2026"/f"{symbol}.csv", True)
    raw["datetime"] = pd.to_datetime(raw.datetime, format="mixed")
    raw = raw.loc[raw.datetime.ge(first) & raw.datetime.le(end)].copy()
    raw.to_csv(OUT/f"candidate_{candidate:03d}_episode_raw.csv", index=False, encoding="utf-8-sig")
    first_day = raw.loc[raw.datetime.dt.normalize().eq(first.normalize())]
    after_intent = first_day.loc[first_day.datetime.ge(pd.Timestamp(intent.exit_earliest_time))]
    return_from_previous_close = after_intent.close/previous_close-1
    opposite_side = "buy" if pair.direction == "sell_first" else "sell"
    proxy = return_from_previous_close.ge(.098) if opposite_side == "buy" else return_from_previous_close.le(-.098)
    episode = dict(candidate_index=candidate, pair_id=int(pair.pair_id), symbol=symbol,
        first_datetime=str(first), original_deadline=str(pair.fixed_end), final_recovery=str(end),
        direction=pair.direction, recovery_side=opposite_side, horizon_minutes=int(pair.horizon_minutes),
        original_shares=int(pair.shares), complete=bool(pair.complete), fragment_count=int(pair.fragments),
        exit_mode=pair.exit_mode, exit_signal_time=intent.exit_signal_time,
        exit_earliest_time=intent.exit_earliest_time, exit_reason=intent.exit_reason,
        previous_close=previous_close, first_reference_price=float(pair.reference_price),
        blocked_reason_counts=block.reason.value_counts().to_dict(),
        earliest_block=str(pd.to_datetime(block.datetime, format="mixed").min()),
        latest_block=str(pd.to_datetime(block.datetime, format="mixed").max()),
        first_day_observed_rows_after_intent=len(after_intent),
        first_day_price_proxy_rows_after_intent=int(proxy.sum()),
        first_day_zero_volume_rows_after_intent=int(after_intent.volume.le(0).sum()),
        first_day_min_reference_price=float(after_intent.close.min()),
        first_day_max_reference_price=float(after_intent.close.max()),
        actual_trip_net=float(trip.net_pnl.sum()),
        actual_trip_reference_gross=float((trip.gross_pnl+trip.slippage_cost).sum()),
        actual_trip_slippage=float(trip.slippage_cost.sum()),
        actual_trip_fees=float((trip.buy_fee+trip.sell_fee).sum()),
        saved_exit_intent=intent.to_dict())
    assert np.isclose(episode["actual_trip_reference_gross"]-episode["actual_trip_slippage"]-episode["actual_trip_fees"], episode["actual_trip_net"], atol=1e-7)
    episodes.append(episode)

assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == h for path, h in inputs.items())
summary = dict(passed=True, scope="first_fold_inner_2026_only", candidates=36, all_candidates_have_delayed_or_pending_pairs=True,
    candidate_overlapping_delayed_or_pending_pairs=len(bad), direction_counts=bad.direction.value_counts().to_dict(),
    distinct_stock_first_dates=len(groups), unique_failure_stock_days=len(failures.groupby(["symbol", "date"])),
    greedy_descriptive_cover=cover, cover_is_not_blacklist_or_rule_change=True,
    representative_choice="fewest complete groups among failures represented by three descriptive cover events; no PnL ranking",
    episodes=episodes, no_outer_read=True, no_fit_account_replay_or_rule_change=True, inputs_sha256=inputs)
(OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
lines = ["# v18首折恢复失败的有界事后归因", "", "仅2023留出折的内部2026年，不读取外层或修改信号。跨候选记录不能当作一只组合。", "",
    f"36个主动候选共{len(bad)}个跨日或期末未完成首腿，分布在{len(groups)}个股票×首腿日；卖先{int(bad.direction.eq('sell_first').sum())}、买先{int(bad.direction.eq('buy_first').sum())}。没有一个事件覆盖全部候选。三个事件的并集已覆盖36个候选，此集合仅用来解释失败，不是剔除清单。", "",
    "|代表规则|首腿|计划期限/恢复意图|真实恢复|恢复拒单原因|", "|---|---|---|---|---|"]
for r in episodes:
    lines.append(f"|{r['candidate_index']} {r['symbol']}|{r['first_datetime']} {r['direction']} {r['original_shares']}股|{r['original_deadline']} / {r['exit_reason']} / {r['exit_earliest_time']}|{r['final_recovery']}|{r['blocked_reason_counts']}|")
lines += ["", "quote原价与拒单解释保存在逐事件CSV/JSON。9.8%分支和其他原始价量事实只用于解释已发生执行，不能事后把未来可恢复性加到入场筛选中。首腿当刻反向复查通过，不保证随后期限恢复方向仍可成交。", "",
    "next-day恢复保留真实T+1与日初旧股预算；真实跨日即违反每日股数恢复要求，不能用最终pending为零抹去。首腿、预定期限、锁存意图、拒单及分片全部来自原记录，没有补成交或重放账户。该有界描述不能替代完整源码/会计审计，也不能由三个代表事件推断其余事件原因完全相同。"]
(OUT/"summary.md").write_text("\n".join(lines)+"\n")
proof = dict(passed=True, inputs_sha256=inputs, output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
(OUT/"verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
