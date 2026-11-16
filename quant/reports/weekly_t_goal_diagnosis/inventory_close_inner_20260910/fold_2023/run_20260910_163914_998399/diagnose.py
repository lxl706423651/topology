"""只读v16已完成内部折，区分14:55参考目标、实际配对与含残仓净额。

--stage只读已落齐37候选和selected_rule的内部折；--final还须主运行成功终态。
不读取外层、不拟合/校准/重跑账户或改规则。原预测键及失败未完成组全部保留，
未来精确报价仅用于事后内部误差诊断，不能将标称固定端点收益当作可成交利润。
每次新增独立输出目录，阶段复用需脚本、实际输入和派生输出SHA全部一致。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
VERSION = "inventory_close_t_v16"
KEYS = ["datetime", "symbol"]
FAIL = ["inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
        "t1_violation_count", "base_plan_mismatch_count"]
RULE_FIELDS = ["candidate_index", "enabled", "profile", "min_net_edge", "fraction", "exit_mode"]
METRIC_FIELDS = ["incremental_net_pnl", "holding_cost_reduction_rate", "t_pair_count", "fragment_count",
    "completed_t_pair_lots", "realized_fragment_reference_gross_pnl", "t_fee", "t_slippage_cost",
    "realized_complete_pair_pnl", "all_reference_acquisition_cost", *FAIL]
IDENTITY = {"recovery_key": "close", "recovery_kind": "fixed_clock", "recovery_clock": "14:55"}
CLOCKS = ("09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00")
SESSION = set(range(571, 691)) | set(range(781, 901))
BINS = [-np.inf, 0., 10., 20., 40., 60., 100., np.inf]
BIN_LABELS = ["negative", "0_10", "10_20", "20_40", "40_60", "60_100", "100_plus"]
TABLES = ["prediction_quality", "candidate_overview", "candidate_source_comparison", "candidate_pairs",
          "candidate_pair_quality", "candidate_failures", "candidate_blocked_reasons", "margin_candidate_summary"]


def require(condition, message):
    """诊断遇到身份或算术不一致必须失败，不把缺失/失败当收益零。"""
    if not condition:
        raise AssertionError(message)


def sha(path):
    """流式记录真实输入字节；原文件stat不作为内容身份。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    """缺失值写null，未完成/未知端点的收益不补零。"""
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    return value


def edges(returns, dates, cfg):
    """每元首腿未滑点金额的双向标称净差，保留原佣金/过户费/滑点和日期印花税。"""
    require(cfg["min_commission"] == 0, "本公式不能忽略非零最低佣金")
    fee = cfg["commission_rate"] + cfg["transfer_rate"]
    slip = cfg["slippage_rate"]
    stamp = pd.Series(np.where(dates < pd.Timestamp("2023-08-28"), cfg["stamp_rate_before_20230828"],
                              cfg["stamp_rate_from_20230828"]), index=returns.index)
    return ((1+returns)*(1-slip)*(1-fee-stamp)-(1+slip)*(1+fee),
            (1-slip)*(1-fee-stamp)-(1+returns)*(1+slip)*(1+fee))


def annotate_values(predictions, execution, recovery, inventory, cfg):
    """当前键原样保留；传入端点只用于标称目标，缺价不参与原方向或机会选择。"""
    p = predictions.copy()
    p["datetime"] = pd.to_datetime(p.datetime, format="mixed")
    require(not p.duplicated(KEYS).any() and p.datetime.dt.strftime("%H:%M").isin(CLOCKS).all(), "预测当前键或时钟错误")
    require(np.isfinite(p.prediction).all() and p.prediction.gt(-1).all(), "预测非有效价格比")
    p["fixed_start"] = p.datetime + pd.Timedelta(minutes=1)
    p["fixed_end"] = p.datetime.dt.normalize()+pd.Timedelta(hours=14, minutes=55)
    p["fixed_execution_price"], p["fixed_recovery_price"] = np.asarray(execution), np.asarray(recovery)
    p["missing_execution"] = ~(np.isfinite(p.fixed_execution_price) & p.fixed_execution_price.gt(0))
    p["missing_recovery"] = ~(np.isfinite(p.fixed_recovery_price) & p.fixed_recovery_price.gt(0))
    p["endpoint_available"] = ~(p.missing_execution | p.missing_recovery)
    p["fixed_raw_return"] = (p.fixed_recovery_price/p.fixed_execution_price-1).where(p.endpoint_available)
    buy, sell = edges(p.prediction, p.datetime, cfg)
    action = pd.Series(np.where(buy >= sell, 1, -1), index=p.index)
    actual_buy, actual_sell = edges(p.fixed_raw_return, p.datetime, cfg)
    p["action"], p["direction"] = action, np.where(action > 0, "buy_first", "sell_first")
    p["predicted_net_return"] = np.maximum(buy, sell)
    p["predicted_gross_bp"], p["predicted_net_bp"] = action*p.prediction*1e4, p.predicted_net_return*1e4
    p["fixed_gross_bp"], p["fixed_net_bp"] = action*p.fixed_raw_return*1e4, np.where(action > 0, actual_buy, actual_sell)*1e4
    day_keys = pd.MultiIndex.from_arrays([p.datetime.dt.normalize(), p.symbol])
    p["opening_old_shares"] = inventory.reindex(day_keys).fillna(0).to_numpy()
    p["signal_clock"] = p.datetime.dt.strftime("%H:%M")
    p["predicted_net_bin"] = pd.cut(p.predicted_net_bp, BINS, labels=BIN_LABELS, right=False).astype(str)
    return p


def current_eligibility(p, margin, cfg):
    """仅当前预测、价量和合法时钟复算原条件；不触未来价或endpoint_available。"""
    mask = p.predicted_net_return.gt(0) & p.predicted_net_return.ge(margin)
    if cfg["t_avoid_locked_exit_side"]:
        price, previous = p.signal_price.astype(float), p.previous_close.astype(float)
        reverse = pd.Series(np.where(p.action > 0, -1., 1.), index=p.index)
        valid = (np.isfinite(price) & price.gt(0) & np.isfinite(previous) & previous.gt(0)
            & np.isfinite(p.signal_high) & np.isfinite(p.signal_low) & p.signal_high.ge(price) & p.signal_low.le(price))
        limit = np.floor(previous*(1+reverse*cfg["flat_st_limit_rate"])*100+.5+1e-9)/100
        locked = ((price/previous-1)*reverse).ge(cfg["nonflat_limit_proxy"]) | (
            p.signal_high.eq(p.signal_low) & (price-limit).abs().le(1e-6))
        mask &= valid & ~locked
    mask &= np.isfinite(p.signal_amount) & p.signal_amount.gt(0) & np.isfinite(p.signal_volume) & p.signal_volume.gt(0)
    mask &= p.datetime.le(p.datetime.dt.normalize()+pd.to_timedelta(cfg["last_entry_time"]+":00"))
    return mask


def summarize_predictions(frame):
    """等权描述机会，未知端点分开；不同日/周计数不意味着统计独立。"""
    valid = frame.loc[frame.endpoint_available]
    return clean(dict(rows=len(frame), distinct_days=frame.datetime.dt.normalize().nunique(),
        distinct_weeks=frame.datetime.dt.to_period("W-SUN").nunique(), symbols=frame.symbol.nunique(),
        valid_endpoint_rows=len(valid), missing_execution_rows=int(frame.missing_execution.sum()),
        missing_recovery_rows=int(frame.missing_recovery.sum()), missing_endpoint_rows=len(frame)-len(valid),
        positive_predicted_net_rows=int(frame.predicted_net_bp.gt(0).sum()),
        mean_predicted_net_bp_all=frame.predicted_net_bp.mean(), mean_predicted_net_bp_common=valid.predicted_net_bp.mean(),
        mean_fixed_gross_bp=valid.fixed_gross_bp.mean(), mean_fixed_nominal_net_bp=valid.fixed_net_bp.mean(),
        mean_fixed_cost_bp=(valid.fixed_gross_bp-valid.fixed_net_bp).mean(),
        mean_net_optimism_bp=(valid.predicted_net_bp-valid.fixed_net_bp).mean(),
        direction_success_rate=valid.fixed_gross_bp.gt(0).mean() if len(valid) else np.nan,
        nominal_cost_covered_rate=valid.fixed_net_bp.gt(0).mean() if len(valid) else np.nan,
        actual_execution_not_implied=True))


def pair_details(events, trips, predictions):
    """保留每条真实首腿及未完成组；参考目标和账户已实现片段/完整净额分列。"""
    event = events.copy()
    if event.empty: return pd.DataFrame()
    event["datetime"] = pd.to_datetime(event.datetime, format="mixed")
    event["signal_datetime"] = pd.to_datetime(event.signal_datetime, format="mixed")
    first = event.loc[event.event_type.eq("t") & event.fragment_id.eq(0)].copy()
    if first.empty: return first
    require(not first.pair_id.duplicated().any(), "同账户首腿pair_id重复")
    first = first.rename(columns={"datetime": "first_datetime", "signal_datetime": "datetime"})
    fields = ["prediction", "predicted_net_bp", "predicted_gross_bp", "fixed_net_bp", "fixed_gross_bp",
              "fixed_start", "fixed_end", "fixed_execution_price", "endpoint_available", "direction", "signal_clock"]
    p = first.merge(predictions[KEYS+fields], on=KEYS, how="left", validate="many_to_one")
    require(p.prediction.notna().all() and p.first_datetime.eq(p.fixed_start).all(), "实际首腿未匹配原下一分钟预测")
    require(np.allclose(p.reference_price, p.fixed_execution_price, rtol=0, atol=1e-10), "实际首腿参考价与原分钟不一致")
    require(p.side.eq(np.where(p.direction.eq("buy_first"), "buy", "sell")).all(), "真实方向与当前预测不一致")
    columns = ["recovered_shares", "fragments", "fragment_net", "slipped_gross", "slippage", "allocated_fees", "final_recovery"]
    if len(trips):
        trip = trips.copy()
        require(trip.pair_id.isin(p.pair_id).all(), "恢复片段没有对应首腿")
        trip["second_datetime"] = pd.to_datetime(trip.second_datetime, format="mixed")
        trip["allocated_fees"] = trip.buy_fee+trip.sell_fee
        grouped = trip.groupby("pair_id").agg(recovered_shares=("shares", "sum"), fragments=("shares", "size"),
            fragment_net=("net_pnl", "sum"), slipped_gross=("gross_pnl", "sum"), slippage=("slippage_cost", "sum"),
            allocated_fees=("allocated_fees", "sum"), final_recovery=("second_datetime", "max"))
    else:
        grouped = pd.DataFrame(columns=columns)
    p = p.merge(grouped, left_on="pair_id", right_index=True, how="left", validate="one_to_one")
    for name in columns[:-1]: p[name] = p[name].fillna(0)
    require(p.recovered_shares.le(p.shares).all(), "恢复超过原始股数")
    p["complete"] = p.recovered_shares.eq(p.shares)
    p["unrecovered_shares"] = p.shares-p.recovered_shares
    p["first_reference_notional"] = p.shares*p.reference_price
    p["predicted_net_currency"] = p.predicted_net_bp/1e4*p.first_reference_notional
    p["fixed_nominal_net_currency"] = p.fixed_net_bp/1e4*p.first_reference_notional
    p["actual_complete_net_currency"] = p.fragment_net.where(p.complete)
    p["actual_complete_net_bp"] = p.actual_complete_net_currency/p.first_reference_notional*1e4
    p["actual_complete_reference_gross_currency"] = (p.slipped_gross+p.slippage).where(p.complete)
    p["all_actual_pair_fees"] = p.pair_id.map(event.loc[event.event_type.eq("t")].groupby("pair_id").fee.sum())
    return p


def compare_candidates(current, source):
    """同数字参数但不同预测的逐账户对照，不将候选差额相加成组合收益。"""
    require(len(current) == len(source) == 37, "内部候选必须齐37个")
    a, b = current.sort_values("candidate_index"), source.sort_values("candidate_index")
    pd.testing.assert_frame_equal(a[RULE_FIELDS].reset_index(drop=True), b[RULE_FIELDS].reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    require(np.allclose(a.all_reference_acquisition_cost, b.all_reference_acquisition_cost, rtol=0, atol=1e-8), "底仓原始含费成本分母改变")
    merged = a[["candidate_index", *METRIC_FIELDS]].merge(b[["candidate_index", *METRIC_FIELDS]],
        on="candidate_index", validate="one_to_one", suffixes=("_close", "_source"))
    for name in METRIC_FIELDS: merged[name+"_change"] = merged[name+"_close"]-merged[name+"_source"]
    merged["same_forecasts"], merged["same_execution_guard"] = False, True
    return merged


def inventory_failures(daily):
    """从原每日库存表提取全部恢复/股数/旧股预算失败，不假设存在独立失败CSV。"""
    if daily.empty: return daily.copy()
    bad = (~daily.inventory_preserved.eq(True) | daily.recovery_failure.eq(True)
        | daily.baseline_inventory_difference.ne(0) | daily.t1_violation_count.ne(0)
        | daily.day_sold.gt(daily.opening_sellable))
    return daily.loc[bad].copy()


class Diagnosis:
    """显式文件白名单：本折内部、固定配置和同内部年raw，不允许读取任一外层账户。"""
    def __init__(self, report, year):
        self.report, self.year = report.resolve(), int(year)
        self.folder = self.report/f"year_{year}"
        self.inputs, self.raw, self.tables = {}, {}, {key: [] for key in TABLES}
        self.cfg = self.json(self.report/"resolved_config.json")
        require(self.cfg["research_version"] == VERSION, "非v16报告")
        self.source = (ROOT/self.cfg["source_execution_report_dir"]).resolve()
        self.fold = self.json(self.folder/"split.json")
        self.inner = self.fold["inner_year"]
        self.chosen = self.json(self.folder/"selected_rule.json")
        self.metrics = self.csv(self.folder/"validation_rule_metrics.csv")
        require(len(self.metrics) == 37 and set(self.metrics.candidate_index) == set(range(37)), "内部37候选未齐")
        self.metrics = self.metrics.sort_values("candidate_index").reset_index(drop=True)
        require(self.metrics.selected.sum() == 1 and int(self.metrics.loc[self.metrics.selected, "candidate_index"].iloc[0])
                == self.chosen["rule"]["candidate_index"], "内部选择归档不一致")

    def record(self, path):
        """先检查允许读取范围再取SHA，不能为记录身份先打开外层文件。"""
        path = Path(path).resolve()
        permitted = False
        if path.is_relative_to(self.report):
            parts = path.relative_to(self.report).parts
            permitted = len(parts) == 1 and parts[0] in ("resolved_config.json", "research_protocol.json", "source_hashes.json")
            if parts[0] == f"year_{self.year}":
                permitted |= len(parts) == 2 and parts[1] in ("split.json", "selected_rule.json", "validation_rule_metrics.csv", "close_vs_source_candidates.csv")
                permitted |= len(parts) > 2 and parts[1] in ("validation_predictions", "validation_candidates", "validation_context", "models")
        elif hasattr(self, "source") and path.is_relative_to(self.source):
            permitted = path.relative_to(self.source).as_posix() in ("resolved_config.json", f"year_{self.year}/validation_rule_metrics.csv", f"year_{self.year}/split.json")
        elif hasattr(self, "inner"):
            raw_root = (ROOT/self.cfg["data_root"]/str(self.inner)).resolve()
            permitted = path.parent == raw_root and path.suffix == ".csv"
        require(permitted, f"禁止读取外层/其他折或非内部年原CSV：{path}")
        identity = sha(path)
        require(str(path) not in self.inputs or self.inputs[str(path)] == identity, "诊断期间已读输入改变")
        self.inputs[str(path)] = identity
        return path

    def json(self, path): return json.loads(self.record(path).read_text())

    def csv(self, path):
        try: return pd.read_csv(self.record(path), float_precision="round_trip")
        except pd.errors.EmptyDataError: return pd.DataFrame()

    def closes(self, symbol):
        """原CSV按生产数值解析合同读取，会话外记录忽略；精确close缺失不填补。"""
        if symbol not in self.raw:
            path = (ROOT/self.cfg["data_root"]/str(self.inner)/f"{symbol}.csv").resolve()
            frame = pd.read_csv(self.record(path), usecols=["datetime", "close"])
            frame["datetime"] = pd.to_datetime(frame.datetime, format="mixed")
            minute = frame.datetime.dt.hour*60+frame.datetime.dt.minute
            frame = frame.loc[frame.datetime.dt.year.eq(self.inner) & minute.isin(SESSION)].copy()
            require(not frame.datetime.duplicated().any(), "原会话分钟重复")
            require((frame.datetime.astype("int64") % 60_000_000_000 == 0).all(), "会话内非整分钟")
            self.raw[symbol] = frame.set_index("datetime").close.sort_index()
        return self.raw[symbol]

    def annotate(self, frame):
        """只查当前内部年精确next-minute和14:55，缺端点全部保留。"""
        frame = frame.copy()
        frame["datetime"] = pd.to_datetime(frame.datetime, format="mixed")
        require(frame.datetime.dt.year.eq(self.inner).all(), "预测不是本折内部年")
        require(all(frame[name].eq(value).all() for name, value in IDENTITY.items()), "预测固定尾盘身份错")
        first = frame.datetime+pd.Timedelta(minutes=1)
        last = frame.datetime.dt.normalize()+pd.Timedelta(hours=14, minutes=55)
        a, b = pd.Series(np.nan, index=frame.index), pd.Series(np.nan, index=frame.index)
        for symbol, group in frame.groupby("symbol", sort=False):
            closes = self.closes(symbol)
            a.loc[group.index] = closes.reindex(pd.DatetimeIndex(first.loc[group.index])).to_numpy()
            b.loc[group.index] = closes.reindex(pd.DatetimeIndex(last.loc[group.index])).to_numpy()
        return annotate_values(frame, a, b, self.inventory, self.cfg)

    def quality(self, frame, profile, scope):
        """固定全量、旧股、方向、8时钟与原净差箱，不搜索阈值。"""
        for domain, part in (("all_current_keys", frame), ("baseline_opening_old_inventory", frame.loc[frame.opening_old_shares.gt(0)])):
            for grouping, fields in (("all", []), ("direction", ["direction"]),
                    ("direction_clock", ["direction", "signal_clock"]), ("predicted_net_bin", ["predicted_net_bin"])):
                groups = [((), part)] if not fields else part.groupby(fields, observed=True, dropna=False)
                for keys, values in groups:
                    keys = keys if isinstance(keys, tuple) else (keys,)
                    self.tables["prediction_quality"].append(dict(fold=self.year, inner_year=self.inner, profile=profile,
                        scope=scope, domain=domain, grouping=grouping, **dict(zip(fields, keys)), **summarize_predictions(values)))

    def account(self, row, predictions):
        """读取既存账户，不重放执行；核统计到全部首腿/片段并列全部失败记录。"""
        index = int(row.candidate_index)
        directory = self.folder/f"validation_candidates/candidate_{index:03d}"
        metrics = self.json(directory/"metrics.json")
        for name in METRIC_FIELDS:
            require(np.isclose(row[name], metrics[name], rtol=0, atol=1e-7), f"候选表与实际metrics不同：{index}/{name}")
        event, trip = self.csv(directory/"test_trade_events.csv"), self.csv(directory/"test_round_trips.csv")
        p = pair_details(event, trip, predictions.get(row.profile, pd.DataFrame()))
        actual_t = event.loc[event.event_type.eq("t")] if len(event) else event
        require(np.isclose(actual_t.fee.sum() if len(actual_t) else 0., row.t_fee, rtol=0, atol=1e-7), "实际全部T事件费用不符")
        complete = p.loc[p.complete] if len(p) else p
        require(len(complete) == row.t_pair_count and len(trip) == row.fragment_count, "完整组/恢复片段计数不同")
        lots = complete.shares.sum()/self.cfg["lot_size"] if len(complete) else 0.
        require(lots == row.completed_t_pair_lots, "原始完整组配对手数错误")
        done_net = complete.fragment_net.sum() if len(complete) else 0.
        require(np.isclose(done_net, row.realized_complete_pair_pnl, atol=1e-7, rtol=0), "完整组净额不同")
        if len(p):
            p["fold"], p["inner_year"], p["candidate_index"] = self.year, self.inner, index
            p["profile"], p["fraction"], p["exit_mode"], p["selected"] = row.profile, row.fraction, row.exit_mode, row.selected
            self.tables["candidate_pairs"].extend(p.to_dict("records"))
            for grouping, fields in (("all", []), ("direction", ["direction"]), ("direction_clock", ["direction", "signal_clock"])):
                groups = [((), p)] if not fields else p.groupby(fields, observed=True, dropna=False)
                for keys, group in groups:
                    keys = keys if isinstance(keys, tuple) else (keys,)
                    done = group.loc[group.complete]
                    common = done.loc[done.endpoint_available]
                    self.tables["candidate_pair_quality"].append(dict(fold=self.year, inner_year=self.inner,
                        candidate_index=index, selected=row.selected, profile=row.profile, exit_mode=row.exit_mode,
                        grouping=grouping, **dict(zip(fields, keys)), started_groups=len(group), complete_groups=len(done),
                        incomplete_groups=len(group)-len(done), missing_fixed_endpoint_groups=int((~group.endpoint_available).sum()),
                        realized_all_fragment_net=group.fragment_net.sum(), complete_actual_net=done.fragment_net.sum(),
                        same_complete_endpoint_groups=len(common), common_predicted_net_bp=common.predicted_net_bp.mean(),
                        common_fixed_nominal_net_bp=common.fixed_net_bp.mean(), common_actual_complete_net_bp=common.actual_complete_net_bp.mean(),
                        common_fixed_nominal_currency=common.fixed_nominal_net_currency.sum(),
                        common_actual_complete_currency=common.fragment_net.sum(),
                        actual_minus_fixed_common_currency=(common.fragment_net-common.fixed_nominal_net_currency).sum(),
                        all_actual_pair_fees=group.all_actual_pair_fees.sum()))
        inventory = self.csv(directory/"daily_inventory_audit.csv")
        if len(inventory):
            days = pd.to_datetime(inventory.date, format="mixed").dt.normalize()
            require(days.loc[~inventory.inventory_preserved.eq(True)].nunique() == row.inventory_mismatch_days,
                    "每日库存不守恒计数与指标不同")
            require(days.loc[inventory.recovery_failure.eq(True)].nunique() == row.recovery_failure_days,
                    "每日恢复失败计数与指标不同")
        require(len(p)-len(complete) == row.pending_pairs, "未完成真实组与期末pending数量不同")
        failure = inventory_failures(inventory)
        if len(failure): self.tables["candidate_failures"].extend(failure.assign(fold=self.year, inner_year=self.inner, candidate_index=index).to_dict("records"))
        blocked = self.csv(directory/"blocked_orders.csv")
        if len(blocked):
            blocked["clock"] = pd.to_datetime(blocked.datetime, format="mixed").dt.strftime("%H:%M")
            for (kind, reason, clock), group in blocked.groupby(["event_type", "reason", "clock"], dropna=False):
                self.tables["candidate_blocked_reasons"].append(dict(fold=self.year, inner_year=self.inner, candidate_index=index,
                    event_type=kind, reason=reason, clock=clock, rows=len(group), symbols=group.symbol.nunique()))
        return dict(candidate_index=index, started_groups=len(p), complete_groups=len(complete), incomplete_groups=len(p)-len(complete),
            complete_net=done_net, marked_increment=row.incremental_net_pnl, cost_denominator=row.all_reference_acquisition_cost,
            holding_cost_reduction_rate=row.holding_cost_reduction_rate, paired_lots=lots, **{key: row[key] for key in FAIL})

    def run(self):
        """先已保存两close预测，再原37账户；全部比较只描述既有内部结果。"""
        inv = self.csv(self.folder/"validation_context/baseline/daily_inventory_audit.csv")
        inv["date"] = pd.to_datetime(inv.date, format="mixed")
        require(not inv.duplicated(["date", "symbol"]).any(), "基准旧股股日重复")
        self.inventory = inv.set_index(["date", "symbol"]).baseline_opening_shares
        predictions = {}
        for profile in ("l2_depth4", "l2_depth6"):
            directory = self.folder/f"validation_predictions/close_{profile}"
            raw = self.csv(directory/"current_predictions.csv")
            audit = self.json(directory/"prediction_audit.json")
            meta = self.json(self.folder/f"models/close_{profile}/model_metadata.json")
            require(sha(directory/"current_predictions.csv") == audit["predictions_sha256"], "预测CSV身份错")
            require(audit["model_bundle_sha256"] == meta["model_bundle_sha256"] and audit["future_independent"] is True, "模型与当前推断证明错")
            require(meta["train_years"] == sorted(self.fold["core_years"]) and meta["excluded_years"] == sorted([self.inner, self.year]), "模型训练/排除年错")
            require(raw.model_profile.eq(profile).all(), "预测profile不一致")
            predictions[profile] = self.annotate(raw)
            self.quality(predictions[profile], profile, "all_close_predictions")
            for margin in (0., .001, .003):
                mask = current_eligibility(predictions[profile], margin, self.cfg)
                eligible = predictions[profile].loc[mask]
                rows = self.metrics.loc[self.metrics.profile.eq(profile) & self.metrics.min_net_edge.eq(margin)]
                for row in rows.itertuples():
                    saved = self.csv(self.folder/f"validation_candidates/candidate_{row.candidate_index:03d}/t_decisions.csv")
                    if len(saved): saved["datetime"] = pd.to_datetime(saved.datetime, format="mixed")
                    actual_keys = saved[KEYS] if len(saved) else eligible.iloc[:0][KEYS]
                    pd.testing.assert_frame_equal(actual_keys.sort_values(KEYS).reset_index(drop=True),
                        eligible[KEYS].sort_values(KEYS).reset_index(drop=True), check_exact=True)
                self.quality(eligible, profile, f"original_current_margin{margin*1e4:g}bp")
        original = self.csv(self.source/f"year_{self.year}/validation_rule_metrics.csv")
        source_split = self.json(self.source/f"year_{self.year}/split.json")
        require(all(source_split[key] == self.fold[key] for key in ("test_year", "inner_year", "core_years", "train_years")), "原v15拆分与新折不一致")
        comparisons = compare_candidates(self.metrics, original)
        saved_comparison = self.csv(self.folder/"close_vs_source_candidates.csv")
        shared = list(saved_comparison.columns)
        pd.testing.assert_frame_equal(comparisons[shared], saved_comparison, check_dtype=False, check_exact=False, rtol=0, atol=1e-7)
        self.tables["candidate_source_comparison"] = comparisons.assign(fold=self.year, inner_year=self.inner).to_dict("records")
        self.tables["candidate_overview"] = self.metrics.assign(fold=self.year, inner_year=self.inner).to_dict("records")
        accounts = [self.account(row, predictions) for row in self.metrics.itertuples()]
        for (profile, margin), group in self.metrics.loc[self.metrics.enabled].groupby(["profile", "min_net_edge"]):
            self.tables["margin_candidate_summary"].append(dict(fold=self.year, inner_year=self.inner, profile=profile,
                margin_bp=margin*1e4, overlapping_accounts=len(group), positive_net_accounts=int(group.incremental_net_pnl.gt(0).sum()),
                execution_eligible_accounts=int(group.execution_eligible.sum()), complete_groups_min=group.t_pair_count.min(),
                complete_groups_median=group.t_pair_count.median(), complete_groups_max=group.t_pair_count.max(),
                paired_lots_median=group.completed_t_pair_lots.median(), marked_increment_median=group.incremental_net_pnl.median(),
                actual_fee_median=group.t_fee.median(), actual_slippage_median=group.t_slippage_cost.median(),
                distribution_not_additive_portfolio=True))
        require(all(sha(path) == value for path, value in self.inputs.items()), "诊断期间输入改变")
        self.raw.clear()
        return dict(fold=self.year, inner_year=self.inner, selected_rule=self.chosen["rule"],
            selected_account=accounts[int(self.chosen["rule"]["candidate_index"])], accounts=accounts,
            inputs_sha256=self.inputs, no_outer_read=True, no_new_parameters=True,
            reference_endpoint_is_nominal_not_executable=True, same_numeric_rules_not_same_forecasts=True)


def write_markdown(out, results):
    """突出预测误差与账户实际差异，hold/亏损均如实保留。"""
    q = pd.read_csv(out/"prediction_quality.csv")
    selected = q.loc[q.grouping.eq("all") & q.scope.eq("original_current_margin0bp") & q.domain.eq("baseline_opening_old_inventory")]
    lines = ["# 固定14:55预测的内部诊断", "", "仅已完成内部评价折。没有读取外层、重新拟合或依据诊断修改本轮规则。", "",
        "下表使用原0边际当前门控及基准日初旧股机会。未来端点缺失单列，现金、已耗旧股与真实容量未被假定满足。", "",
        "|留出折|内部年|模型|原机会/有效端点|预测净差 bp|实际方向毛差 bp|标称固定净差 bp|乐观偏差 bp|成本覆盖率|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected.itertuples():
        lines.append(f"|{r.fold}|{r.inner_year}|{r.profile}|{r.rows}/{r.valid_endpoint_rows}|{r.mean_predicted_net_bp_common:.2f}|{r.mean_fixed_gross_bp:.2f}|{r.mean_fixed_nominal_net_bp:.2f}|{r.mean_net_optimism_bp:.2f}|{r.nominal_cost_covered_rate:.2%}|")
    lines += ["", "参考价差以首腿原始参考金额归一化，按当日原费税及双腿滑点扣减。平价来回成本在2023-08-28后约31.20 bp，此前约36.20 bp；它不是盘口或真实回购保证。", "",
        "|留出折|原已选规则|启动/完整组|配对手|完整组净额 元|账户含残仓净额 元|原始含费底仓成本 元|成本降低率|库存差/恢复失败/期末pending|",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for result in results:
        a, rule = result["selected_account"], result["selected_rule"]
        lines.append(f"|{result['fold']}|{rule['candidate_index']} {rule['profile']}/{rule['exit_mode']}|{a['started_groups']}/{a['complete_groups']}|{a['paired_lots']:.0f}|{a['complete_net']:.2f}|{a['marked_increment']:.2f}|{a['cost_denominator']:.2f}|{a['holding_cost_reduction_rate']:.4%}|{a['inventory_mismatch_days']}/{a['recovery_failure_days']}/{a['pending_pairs']}|")
    lines += ["", "candidate_overview.csv列全部37个已保存账户；candidate_source_comparison.csv按相同数字参数对照v15。两轮预测目标和实际信号不同，不能把它写成相同信号的单项退出改进，也不能将候选差额相加当作组合收益。",
        "candidate_pairs.csv保留全部候选每个首腿，包括无恢复、部分恢复及亏损。完整组净额、已实现片段净额、固定14:55标称目标和含残仓账户增量分列；candidate_pair_quality.csv按方向/信号时段比较同一完整且有端点子集。",
        "candidate_failures.csv保留所有已存库存失败行；candidate_blocked_reasons.csv按原拒单原因和时钟计数。每股不同候选含重复机会，属于分布描述。",
        "prediction_quality.csv包含全部当前预测、旧股机会、方向、8时钟及固定净差箱。不同日/周数量不证明统计独立；本轮是历史重复研究，不能称全新样本盈利证明。hold也不代表收益目标成功。"]
    (out/"diagnosis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    """等待明确已完成折；阶段缓存与终版均新增目录，并绑定脚本/输入/输出SHA。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--years", type=int, nargs="+")
    args = parser.parse_args(argv)
    report = Path(args.report_dir).resolve()
    cfg = json.loads((report/"resolved_config.json").read_text())
    require(cfg["research_version"] == VERSION, "仅接受v16")
    years = args.years or cfg["test_years"]
    require(set(years).issubset(cfg["test_years"]) and len(years) == len(set(years)), "诊断年份无效")
    available = [y for y in years if all((report/f"year_{y}"/name).exists() for name in (
        "validation_rule_metrics.csv", "selected_rule.json", "close_vs_source_candidates.csv"))]
    if args.final:
        require(available == cfg["test_years"] and (report/"final_report.md").exists()
                and json.loads((report/"runtime_integrity_audit.json").read_text())["all_unchanged"] is True, "主运行未成功终态或折未齐")
    script_sha = sha(__file__)
    stages = []
    for year in available:
        prior = None
        for path in sorted((OUT/f"fold_{year}").glob("run_*/analysis.json"), reverse=True):
            value = json.loads(path.read_text())
            if value["source_report"] == str(report) and value["script_sha256"] == script_sha \
                    and all(sha(name) == digest for name, digest in value["inputs_sha256"].items()) \
                    and all(sha(path.parent/name) == digest for name, digest in value["output_sha256"].items()):
                prior = path.parent
                break
        if prior is not None:
            stages.append(prior)
            continue
        out = OUT/f"fold_{year}"/f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
        out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(__file__, out/"diagnose.py")
        diagnostic = None
        try:
            diagnostic = Diagnosis(report, year)
            result = diagnostic.run()
            for name, rows in diagnostic.tables.items():
                pd.DataFrame(rows).to_csv(out/f"{name}.csv", index=False, encoding="utf-8-sig")
            write_markdown(out, [result])
            result.update(passed=True, source_report=str(report), script_sha256=script_sha,
                output_sha256={path.name: sha(path) for path in out.iterdir() if path.is_file()})
            (out/"analysis.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2)+"\n")
        except Exception:
            (out/"failure.json").write_text(json.dumps(dict(traceback=traceback.format_exc(),
                inputs_sha256=diagnostic.inputs if diagnostic else {}), ensure_ascii=False, indent=2)+"\n")
            raise
        stages.append(out)
        print(f"fold={year} 已完成内部固定尾盘/37账户诊断：{out}", flush=True)
    if not args.final:
        print(json.dumps(dict(final=False, completed_folds=available, stage_directories=list(map(str, stages))), ensure_ascii=False))
        return
    final = OUT/f"final_{datetime.now():%Y%m%d_%H%M%S_%f}"
    final.mkdir(exist_ok=False)
    results = [json.loads((path/"analysis.json").read_text()) for path in stages]
    for name in TABLES:
        frames = []
        for stage in stages:
            try: frames.append(pd.read_csv(stage/f"{name}.csv", float_precision="round_trip"))
            except pd.errors.EmptyDataError: pass
        (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(final/f"{name}.csv", index=False, encoding="utf-8-sig")
    write_markdown(final, results)
    inputs = {name: digest for result in results for name, digest in result["inputs_sha256"].items()}
    for name in ("resolved_config.json", "source_hashes.json", "runtime_integrity_audit.json", "research_protocol.json"):
        inputs[str(report/name)] = sha(report/name)
    source_hashes = json.loads((report/"source_hashes.json").read_text())
    require(len(source_hashes) == 57, "v16冻结运行源数量不符")
    for name, digest in source_hashes.items():
        for path in (report/"source_snapshot"/name, ROOT/name):
            require(sha(path) == digest, "运行源码或原快照改变")
            inputs[str(path)] = digest
    require(all(sha(path) == digest for path, digest in inputs.items()), "终版输入身份变化")
    shutil.copy2(__file__, final/"diagnose.py")
    result = dict(passed=True, complete=True, source_report=str(report), configured_folds=cfg["test_years"],
        script_sha256=script_sha, no_outer_read=True, no_refit_calibration_or_parameter_change=True,
        sources_verified=57, inputs_sha256=inputs, results=results,
        output_sha256={path.name: sha(path) for path in final.iterdir() if path.is_file()})
    (final/"verification.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(passed=True, final_directory=str(final), inputs=len(inputs)), ensure_ascii=False))


if __name__ == "__main__": main()
