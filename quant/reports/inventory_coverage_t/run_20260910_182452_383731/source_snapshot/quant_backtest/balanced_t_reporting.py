"""固定金额、永久底仓做 T 的收益与标准手报告；不选参、不改变交易开关。

从 balanced_t_engine 的实际/无T基准事件、配对和每日库存重建全部含费底仓成本。
5%默认相对全部原始参考成本，账户本金比率并列；未量化手数不冒称目标完成。
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


STANDARD_FILES = {"equity_curve":"test_equity_curve.csv", "trade_events":"test_trade_events.csv",
                  "round_trips":"test_round_trips.csv", "open_positions":"test_open_positions.csv",
                  "daily_inventory_audit":"daily_inventory_audit.csv", "baseline_events":"baseline_trade_events.csv",
                  "order_attempts":"blocked_orders.csv", "position_amount_audit":"position_amount_audit.csv",
                  "base_plan_mismatches":"base_plan_mismatches.csv"}
FAILURE_KEYS = ("inventory_mismatch_days", "recovery_failure_days", "pending_pairs",
                "t1_violation_count", "base_plan_mismatch_count")
FREQUENCY_KEYS = ("minimum_completed_pair_lots", "minimum_active_stock_days")


def _number_sum(frame: pd.DataFrame, column: str) -> float:
    """空记录为零；非空记录缺字段/非有限金额不能静默忽略。"""
    if frame.empty:
        return 0.
    values = pd.to_numeric(frame[column],errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"非有限账目：{column}")
    return float(values.sum())


def _boolean(value) -> bool:
    """显式布尔转换，避免字符串False变成True。"""
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,str) and value.lower() in ("true","false"):
        return value.lower()=="true"
    raise ValueError(f"非法布尔值：{value!r}")


def _ratio(numerator: float, denominator: float):
    """零成本/零机会用未知，不伪造零收益率或完整覆盖。"""
    return float(numerator/denominator) if denominator>0 else None


def _json_ready(value):
    """递归清理输出标量，严格JSON用null表示缺失统计。"""
    if isinstance(value,dict):
        return {str(k):_json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value,np.generic):
        return _json_ready(value.item())
    if isinstance(value,(pd.Timestamp,Path)):
        return str(value)
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict) -> None:
    """写派生报告JSON，不改变配置或账户结果。"""
    path.write_text(json.dumps(_json_ready(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _validate_cfg(cfg: dict) -> tuple[float,int,float,dict]:
    """目标由显式配置冻结，频率无默认门槛且没有审批开关。"""
    initial,lot,target=float(cfg["initial_cash"]),cfg.get("lot_size",100),float(cfg.get("profit_target_rate",.05))
    if not np.isfinite(initial) or initial<=0 or isinstance(lot,bool) or lot!=100 or not np.isfinite(target) or target<=0:
        raise ValueError("初始本金须正、主板标准手100股、收益目标须正")
    low=float(cfg.get("min_position_notional",10000));high=float(cfg.get("max_position_notional",20000))
    nominal=float(cfg.get("target_position_notional",15000))
    if not np.isfinite([low,nominal,high]).all() or not 0<low<=nominal<=high:
        raise ValueError("单股金额区间及目标非法")
    frequency={}
    for name in FREQUENCY_KEYS:
        if cfg.get(name) is not None:
            value=cfg[name]
            if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value<=0:
                raise ValueError(f"{name}必须是明确的正数，缺省为未量化")
            frequency[name]=float(value)
    return initial,int(lot),target,frequency


def _pair_table(result: dict, lot: int) -> pd.DataFrame:
    """按首腿唯一pair重建股数和净额；600股分片只计1组6配对手。"""
    events,trips=result["trade_events"],result["round_trips"]
    first=events.loc[events.event_type.eq("t") & events.fragment_id.eq(0)] if len(events) else events
    if len(first) and first.pair_id.duplicated().any():
        raise ValueError("T首腿pair_id必须唯一")
    if len(trips) and not set(trips.pair_id).issubset(set(first.pair_id)):
        raise ValueError("恢复片段缺原首腿")
    rows=[]
    for row in first.itertuples(index=False):
        parts=trips.loc[trips.pair_id.eq(row.pair_id)] if len(trips) else trips
        amount=float(row.shares)
        if not np.isfinite(amount) or amount<=0 or amount%lot:
            raise ValueError("首腿必须为100股整手")
        complete=False
        if len(parts):
            if not parts.symbol.eq(row.symbol).all() or not parts.shares.gt(0).all() or not parts.shares.mod(lot).eq(0).all():
                raise ValueError("恢复片段股票或整手非法")
            flags=parts.pair_fully_recovered.map(_boolean)
            if flags.nunique()!=1:
                raise ValueError("同pair完成状态冲突")
            complete=bool(flags.iloc[0])
        recovered=_number_sum(parts,"shares")
        if recovered>amount or (complete and recovered!=amount):
            raise ValueError("配对恢复股数不符")
        gross=_number_sum(parts,"gross_pnl");fees=_number_sum(parts,"buy_fee")+_number_sum(parts,"sell_fee")
        net=_number_sum(parts,"net_pnl");slip=_number_sum(parts,"slippage_cost")
        if not np.isclose(gross-fees,net,rtol=1e-10,atol=1e-7):
            raise ValueError("配对净额不等于成交价毛差减费用")
        rows.append({"pair_id":row.pair_id,"symbol":row.symbol,"first_datetime":pd.Timestamp(row.datetime),
                     "direction":"buy_first" if row.side=="buy" else "sell_first", "original_shares":int(amount),
                     "recovered_shares":int(recovered),"pending_shares":int(amount-recovered),"complete":complete,
                     "completed_pair_lots":amount/lot if complete else 0.,"recovery_fragments":len(parts),
                     "reference_gross_pnl":gross+slip,"slippage_cost":slip,"fees":fees,"realized_fragment_net_pnl":net,
                     "complete_pair_net_pnl":net if complete else 0.})
    return pd.DataFrame(rows,columns=["pair_id","symbol","first_datetime","direction","original_shares",
                        "recovered_shares","pending_shares","complete","completed_pair_lots","recovery_fragments",
                        "reference_gross_pnl","slippage_cost","fees","realized_fragment_net_pnl","complete_pair_net_pnl"])


def _frequency(result: dict, pairs: pd.DataFrame, cfg: dict, lot: int) -> tuple[dict,pd.DataFrame]:
    """用全市场日期和无T基准开盘昨日股数统计频率；不以成交成功筛分母。"""
    curve=result["equity_curve"]
    dates=pd.DatetimeIndex(pd.to_datetime(curve.datetime,format="mixed").dt.normalize())
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("净值必须每天一行并升序")
    supplied=result.get("market_dates")
    market=pd.DatetimeIndex(pd.to_datetime(supplied)).normalize().unique().sort_values() if supplied is not None else dates
    if not dates.equals(market):
        raise ValueError("净值没有覆盖本次全部声明市场交易日")
    audit=result["daily_inventory_audit"].copy()
    if len(audit):
        if "baseline_opening_shares" not in audit:
            raise ValueError("固定底仓频率需baseline_opening_shares，不能用收盘股数倒推")
        audit["date"]=pd.to_datetime(audit.date,format="mixed").dt.normalize()
        if audit[["date","symbol"]].duplicated().any() or not audit.date.isin(market).all():
            raise ValueError("库存股票日重复或不在评价日历")
        held=audit.loc[audit.baseline_opening_shares.gt(0),["date","symbol"]]
        eligible=audit.loc[audit.baseline_opening_shares.ge(lot),["date","symbol"]]
    else:
        held=eligible=pd.DataFrame(columns=["date","symbol"])
    active=pairs[["first_datetime","symbol"]].copy()
    active["date"]=pd.to_datetime(active.first_datetime).dt.normalize()
    active=active[["date","symbol"]].drop_duplicates()
    matched=active.merge(eligible,on=["date","symbol"],how="inner",validate="one_to_one")
    daily=pd.DataFrame({"date":market})
    first_by_day=pairs.assign(date=pd.to_datetime(pairs.first_datetime).dt.normalize())
    for name,table,column in (("baseline_holding_stock_days",held,"symbol"),("baseline_old_stock_opportunities",eligible,"symbol"),
                              ("active_stock_days",active,"symbol"),("active_stock_days_in_opportunities",matched,"symbol"),
                              ("started_t_pairs",first_by_day,"symbol")):
        values=table.groupby("date")[column].size() if len(table) else pd.Series(dtype=float)
        daily[name]=daily.date.map(values).fillna(0).astype(int)
    lots=first_by_day.groupby("date").completed_pair_lots.sum() if len(first_by_day) else pd.Series(dtype=float)
    completed=first_by_day.loc[first_by_day.complete].groupby("date").size() if len(first_by_day) else pd.Series(dtype=float)
    daily["completed_pair_lots"]=daily.date.map(lots).fillna(0.)
    daily["complete_t_pairs"]=daily.date.map(completed).fillna(0).astype(int)
    holding_days=int(daily.baseline_holding_stock_days.gt(0).sum());old_days=int(daily.baseline_old_stock_opportunities.gt(0).sum())
    active_days=int(daily.active_stock_days.gt(0).sum());total_lots=float(daily.completed_pair_lots.sum())
    return {"market_evaluation_days":len(market),"market_calendar_source":"explicit_market_dates" if supplied is not None else "saved_equity_curve",
            "market_calendar_independently_supplied":supplied is not None,"baseline_holding_days":holding_days,
            "baseline_old_stock_days":old_days,"baseline_holding_stock_days":len(held),"baseline_old_stock_opportunity_stock_days":len(eligible),
            "active_days":active_days,"active_stock_days":len(active),"active_stock_days_in_opportunities":len(matched),
            "active_stock_days_outside_opportunities":len(active)-len(matched),"active_stock_day_coverage":_ratio(len(matched),len(eligible)),
            "completed_pair_lots_per_market_day":_ratio(total_lots,len(market)),
            "completed_pair_lots_per_holding_day":_ratio(total_lots,holding_days),
            "completed_pair_lots_per_old_stock_day":_ratio(total_lots,old_days),
            "completed_pair_lots_per_active_day":_ratio(total_lots,active_days),
            "median_completed_pair_lots_per_market_day":float(daily.completed_pair_lots.median()),
            "zero_activity_day_fraction":_ratio(len(market)-active_days,len(market)),
            "opportunity_definition":"无T基准开盘昨日股数>=100的股票日；未用预测、份额、未来成交或恢复成功过滤"},daily


def calculate_balanced_metrics(result: dict, cfg: dict) -> tuple[dict,pd.DataFrame]:
    """重建年度固定底仓成本/标准手与验收，返回摘要及含零T股票的每股表。

要求旧weekly结果表和新增position_amount_audit；base_order_type标识automatic/manual，
baseline_opening_shares用于频率。自动退出、每日库存偏差与未量化手数不得冒称目标通过。
"""
    initial,lot,target,thresholds=_validate_cfg(cfg)
    required={"equity_curve","trade_events","round_trips","open_positions","daily_inventory_audit",
              "baseline_events","position_amount_audit","metrics"}
    if required.difference(result):
        raise ValueError(f"缺结果表：{sorted(required.difference(result))}")
    metrics=result["metrics"]
    if set(FAILURE_KEYS).difference(metrics):
        raise ValueError("必须显式提供五项T执行失败计数")
    curve=result["equity_curve"]
    if curve.empty or not np.isfinite(curve[["equity","baseline_equity","cash","holdings_value"]].to_numpy(dtype=float)).all():
        raise ValueError("净值/现金/持仓估值必须非空有限")
    for key in FAILURE_KEYS:
        if not np.isfinite(float(metrics[key])) or float(metrics[key])<0 or int(metrics[key])!=metrics[key]:
            raise ValueError(f"非法执行失败计数 {key}")
    audit=result["daily_inventory_audit"]
    stock_day_failures={}
    for column,key in (("inventory_difference","inventory_mismatch_days"),("recovery_failure","recovery_failure_days")):
        bad=audit.loc[audit[column].ne(0) if column=="inventory_difference" else audit[column].map(_boolean)] if len(audit) else audit
        count=int(pd.to_datetime(bad.date,format="mixed").dt.normalize().nunique()) if len(bad) else 0
        if count!=int(metrics[key]):
            raise ValueError(f"{key}与真实逐日库存不符")
        stock_day_failures[key.replace('_days','_stock_days')]=len(bad)
    pairs=_pair_table(result,lot)
    frequency,daily=_frequency(result,pairs,cfg,lot)
    events,baseline=result["trade_events"],result["baseline_events"]
    buys=baseline.loc[baseline.side.eq("buy") & baseline.event_type.ne("t")].copy() if len(baseline) else baseline.copy()
    sells=baseline.loc[baseline.side.eq("sell") & baseline.event_type.ne("t")] if len(baseline) else baseline
    if len(buys) and ("base_order_type" not in buys or not buys.base_order_type.isin(["automatic","manual"]).all()):
        raise ValueError("底仓事件需要automatic/manual明确分类")
    if len(buys):
        buys["acquisition_cost"]=pd.to_numeric(buys.notional)+pd.to_numeric(buys.fee)
        if not np.isfinite(buys.acquisition_cost).all() or not buys.acquisition_cost.gt(0).all():
            raise ValueError("底仓购入成本必须有限正数")
    positions=result["open_positions"]
    symbols=sorted(set(buys.symbol if len(buys) else [])|set(pairs.symbol)|set(positions.symbol if len(positions) else []))
    rows=[]
    for symbol in symbols:
        bought=buys.loc[buys.symbol.eq(symbol)] if len(buys) else buys
        stock_pairs=pairs.loc[pairs.symbol.eq(symbol)]
        opened=positions.loc[positions.symbol.eq(symbol)] if len(positions) else positions
        cost=_number_sum(bought,"acquisition_cost");net=_number_sum(stock_pairs,"complete_pair_net_pnl")
        auto=bought.loc[bought.base_order_type.eq("automatic")] if len(bought) else bought
        manual=bought.loc[bought.base_order_type.eq("manual")] if len(bought) else bought
        shares=_number_sum(opened,"shares");reference_shares=_number_sum(bought,"shares")
        rows.append({"symbol":symbol,"reference_acquisition_cost":cost,
                     "automatic_entry_cost":_number_sum(auto,"acquisition_cost"),"manual_addition_cost":_number_sum(manual,"acquisition_cost"),
                     "reference_acquired_shares":reference_shares,"final_actual_shares":shares,
                     "final_market_value":_number_sum(opened,"market_value"),"complete_t_pairs":int(stock_pairs.complete.sum()),
                     "completed_t_pair_lots":_number_sum(stock_pairs,"completed_pair_lots"),"complete_t_net_pnl":net,
                     "realized_fragment_net_pnl":_number_sum(stock_pairs,"realized_fragment_net_pnl"),
                     "pending_t_shares":_number_sum(stock_pairs,"pending_shares"),"cost_reduction_rate":_ratio(net,cost),
                     "cost_reduction_per_reference_share":_ratio(net,reference_shares),
                     "economic_cost_per_reference_share":_ratio(cost-net,reference_shares)})
    stock=pd.DataFrame(rows,columns=["symbol","reference_acquisition_cost","automatic_entry_cost","manual_addition_cost",
              "reference_acquired_shares","final_actual_shares","final_market_value","complete_t_pairs","completed_t_pair_lots",
              "complete_t_net_pnl","realized_fragment_net_pnl","pending_t_shares","cost_reduction_rate",
              "cost_reduction_per_reference_share","economic_cost_per_reference_share"])
    cost=_number_sum(stock,"reference_acquisition_cost");net=_number_sum(stock,"complete_t_net_pnl")
    marked=float(curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1])
    stock["reference_cost_weight"]=stock.reference_acquisition_cost/cost if cost else np.nan
    positive=float(stock.complete_t_net_pnl.clip(lower=0).sum());negative=float((-stock.complete_t_net_pnl.clip(upper=0)).sum())
    stock["positive_profit_contribution"]=stock.complete_t_net_pnl.clip(lower=0)/positive if positive else np.nan
    stock["loss_contribution"]=(-stock.complete_t_net_pnl.clip(upper=0))/negative if negative else np.nan
    total_events=events.loc[events.event_type.eq("t")] if len(events) else events
    first=total_events.loc[total_events.fragment_id.eq(0)] if len(total_events) else total_events
    recovery=total_events.loc[total_events.fragment_id.gt(0)] if len(total_events) else total_events
    total_lots=_number_sum(total_events,"shares")/lot;paired_lots=_number_sum(pairs,"completed_pair_lots")
    for key,value in (("total_t_traded_lots",total_lots),("completed_t_pair_lots",paired_lots)):
        for source in (result,metrics):
            if key in source and not np.isclose(float(source[key]),value,rtol=0,atol=1e-9):
                raise ValueError(f"引擎 {key} 与实际配对/事件不一致")
    pending=int((~pairs.complete).sum()) if len(pairs) else 0
    if pending!=int(metrics["pending_pairs"]) or ("t_pair_count" in metrics and int(pairs.complete.sum())!=int(metrics["t_pair_count"])):
        raise ValueError("配对/挂起计数与引擎不符")
    equity_error=float((curve.equity-curve.cash-curve.holdings_value).abs().max())
    negative_cash_days=int(curve.cash.lt(-1e-8).sum())
    reconciliation=marked-net
    ledger_valid=bool(abs(reconciliation)<=1e-6 and equity_error<=1e-6)
    inventory_ok=bool(not any(int(metrics[k]) for k in FAILURE_KEYS) and negative_cash_days==0 and len(sells)==0 and ledger_valid)
    amounts=result["position_amount_audit"]
    automatic=amounts.loc[amounts.order_type.eq("automatic")] if len(amounts) else amounts
    filled=automatic.loc[automatic.actual_shares.gt(0)] if len(automatic) else automatic
    low=float(cfg.get("min_position_notional",10000));high=float(cfg.get("max_position_notional",20000))
    violations=int((~filled.actual_notional.between(low,high)).sum()) if len(filled) else 0
    allocation_met=bool(len(filled)>0 and violations==0)
    contributions=0.;withdrawals=0.
    if "external_cash_flows" in result and len(result["external_cash_flows"]):
        flow=pd.to_numeric(result["external_cash_flows"].amount,errors="raise")
        if not np.isfinite(flow).all():raise ValueError("外部现金流非法")
        contributions=float(flow.clip(lower=0).sum());withdrawals=float(-flow.clip(upper=0).sum())
    capital=initial+contributions
    unknown_cost_stocks=int((stock.final_actual_shares.gt(0)&stock.reference_acquisition_cost.le(0)).sum())
    reduction=_ratio(net,cost) if unknown_cost_stocks==0 else None;account_return=marked/capital
    profit_hit=bool(reduction is not None and reduction>=target)
    profit_met=profit_hit and inventory_ok
    count_map={"minimum_completed_pair_lots":paired_lots,"minimum_active_stock_days":frequency["active_stock_days"]}
    enough=bool(thresholds and all(count_map[k]>=value for k,value in thresholds.items()))
    last=pd.Timestamp(curve.datetime.iloc[-1]);stale=positions.loc[pd.to_datetime(positions.mark_datetime,format="mixed").ne(last)&positions.shares.gt(0)] if len(positions) else positions
    hhi=float(stock.reference_cost_weight.pow(2).sum()) if cost else None
    enabled=result.get('selected_rule',{}).get('enabled',metrics.get('t_enabled'))
    if enabled is not None:enabled=_boolean(enabled)
    if enabled is False and len(first):raise ValueError('关闭T规则存在真实T首腿')
    activity='hold' if enabled is False else 'active_traded' if len(first) else 'active_no_fills' if enabled is True else 'state_not_provided'
    value={**metrics,**frequency,**stock_day_failures,"profit_target_rate":target,"profit_target_metric":"complete_t_net_pnl/all_reference_acquisition_cost",
           "t_enabled":enabled,"t_activity_status":activity,
           "initial_cash":initial,"positive_external_contributions":contributions,"external_withdrawals":withdrawals,
           "account_capital_denominator":capital,"all_reference_acquisition_cost":cost,
           "automatic_entry_cost":_number_sum(stock,"automatic_entry_cost"),"manual_addition_cost":_number_sum(stock,"manual_addition_cost"),
           "realized_complete_pair_pnl":net,"realized_all_fragment_pnl":_number_sum(pairs,"realized_fragment_net_pnl"),
           "incremental_marked_pnl":marked,"unpaired_or_baseline_difference_pnl":reconciliation,
           "holding_cost_reduction_rate":reduction,"account_increment_return":account_return,
           "holding_cost_5pct_hit":profit_hit,"account_5pct_hit":bool(account_return>=target),
           "profit_target_status":"not_evaluable" if reduction is None else "met" if profit_met else "not_met",
           "profit_target_met":profit_met,"frequency_target_status":"not_quantified" if not thresholds else "met" if enough else "not_met",
           "frequency_thresholds":thresholds,"frequency_target_met":enough,
           "inventory_contract_status":"met" if inventory_ok else "not_met","inventory_contract_met":inventory_ok,
           "allocation_goal_status":"met" if allocation_met else "not_evaluable" if filled.empty else "not_met",
           "allocation_goal_met":allocation_met,"automatic_entry_attempts":len(automatic),"filled_automatic_entries":len(filled),
           "unfilled_automatic_entry_attempts":len(automatic)-len(filled),"automatic_entry_amount_violations":violations,
           "manual_entry_attempts":int(amounts.order_type.eq("manual").sum()) if len(amounts) else 0,
           "goal_achieved":bool(profit_met and enough and allocation_met),"report_controls_activation":False,
           "complete_t_pairs":int(pairs.complete.sum()),"recovery_fragments":len(result["round_trips"]),
           "completed_t_pair_lots":paired_lots,"total_t_traded_lots":total_lots,
           "first_leg_lots":_number_sum(first,"shares")/lot,"recovery_leg_lots":_number_sum(recovery,"shares")/lot,
           "pending_t_lots":_number_sum(pairs,"pending_shares")/lot,"lot_size":lot,
           "t_fee":_number_sum(total_events,"fee"),"t_slippage_cost":_number_sum(total_events,"slippage_cost"),
           "realized_fragment_reference_gross_pnl":_number_sum(pairs,"reference_gross_pnl"),
           "reference_stock_count":int(stock.reference_acquisition_cost.gt(0).sum()),"zero_t_reference_stock_count":int((stock.reference_acquisition_cost.gt(0)&stock.complete_t_pairs.eq(0)).sum()),
           "unknown_reference_cost_stock_count":unknown_cost_stocks,
           "max_reference_cost_weight":float(stock.reference_cost_weight.max()) if cost else None,
           "top5_reference_cost_weight":float(stock.reference_cost_weight.nlargest(5).sum()) if cost else None,
           "reference_cost_hhi":hhi,"effective_reference_stock_count":1/hhi if hhi else None,
           "max_positive_profit_contribution":float(stock.positive_profit_contribution.max()) if positive else None,
           "max_loss_contribution":float(stock.loss_contribution.max()) if negative else None,
           "mean_holdings_market_value":float(curve.holdings_value.mean()),"peak_holdings_market_value":float(curve.holdings_value.max()),
           "final_unused_cash":float(curve.cash.iloc[-1]),"automatic_base_sell_count":len(sells),
           "negative_cash_days":negative_cash_days,"equity_identity_error":equity_error,
           "complete_pnl_reconciles_with_increment":ledger_valid,"final_stale_position_count":len(stale),
           "final_stale_market_value":_number_sum(stale,"market_value"),"final_open_position_rows":int(positions.shares.gt(0).sum()) if len(positions) else 0,
           "total_return":float((curve.equity.iloc[-1]-initial-contributions+withdrawals)/capital),
           "baseline_total_return":float((curve.baseline_equity.iloc[-1]-initial-contributions+withdrawals)/capital),
           "evaluation_start":str(curve.datetime.iloc[0]),"evaluation_end":str(curve.datetime.iloc[-1])}
    return value,stock


def write_balanced_report(directory: Path, cfg: dict, annual_results, validation_table=None) -> dict:
    """写每折独立账户标准报告；未量化频率只报告不关闭T，期末正常底仓不算失败。"""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    entries=sorted(annual_results.items() if isinstance(annual_results,dict) else annual_results,key=lambda x:int(x[0]))
    if not entries or len(set(int(y) for y,_ in entries))!=len(entries):
        raise ValueError("年度结果必须非空且年份唯一")
    scope=cfg.get("report_scope","smoke" if cfg.get("max_symbols",0) or cfg.get("smoke_weeks",0) else "formal")
    if scope not in ("formal","smoke","internal"):
        raise ValueError("report_scope只能formal/smoke/internal")
    records=[];stock_frames=[];pair_frames=[];daily_frames=[]
    for year,result in entries:
        year=int(year);folder=directory/f"year_{year}";folder.mkdir(exist_ok=True)
        summary,stocks=calculate_balanced_metrics(result,cfg)
        records.append({"test_year":year,"report_scope":scope,**summary})
        stock_frames.append(stocks.assign(test_year=year))
        pairs=_pair_table(result,100);pair_frames.append(pairs.assign(test_year=year))
        _,daily=_frequency(result,pairs,cfg,100);daily_frames.append(daily.assign(test_year=year))
        stocks.to_csv(folder/"per_stock_cost_reduction.csv",index=False,encoding="utf-8-sig")
        pairs.to_csv(folder/"t_pair_lot_summary.csv",index=False,encoding="utf-8-sig")
        daily.to_csv(folder/"daily_frequency_coverage.csv",index=False,encoding="utf-8-sig")
        _write_json(folder/"balanced_goal_evaluation.json",summary)
    annual=pd.DataFrame(records)
    for name in ("annual_summary.csv","年度.csv","test_metrics.csv"):
        annual.to_csv(directory/name,index=False,encoding="utf-8-sig")
    for name,frames in (("per_stock_cost_reduction.csv",stock_frames),("t_pair_lot_summary.csv",pair_frames),("daily_frequency_coverage.csv",daily_frames)):
        pd.concat(frames,ignore_index=True).to_csv(directory/name,index=False,encoding="utf-8-sig")
    for key,name in STANDARD_FILES.items():
        frames=[result[key].assign(test_year=int(year)) for year,result in entries if key in result]
        if frames:pd.concat(frames,ignore_index=True).to_csv(directory/name,index=False,encoding="utf-8-sig")
    if validation_table is not None:
        validation_table.to_csv(directory/"validation_rule_metrics.csv",index=False,encoding="utf-8-sig")
    all_profit=bool(annual.profit_target_met.all());all_inventory=bool(annual.inventory_contract_met.all())
    frequency_quantified=bool(cfg.get("minimum_completed_pair_lots") is not None or cfg.get("minimum_active_stock_days") is not None)
    complete_years=set(map(int,annual.test_year))==set(map(int,cfg.get("test_years",annual.test_year.tolist())))
    status={"report_scope":scope,"report_only":True,"controls_activation":False,
            "profit_denominator":"all_reference_acquisition_cost","profit_target_rate":float(cfg.get("profit_target_rate",.05)),
            "each_validation_year_profit_target_met":all_profit,"each_validation_year_inventory_contract_met":all_inventory,
            "all_configured_years_present":complete_years,"frequency_target_status":"not_quantified" if not frequency_quantified else "met" if bool(annual.frequency_target_met.all()) else "not_met",
            "each_year_allocation_goal_met":bool(annual.allocation_goal_met.all()),
            "goal_achieved":bool(scope=="formal" and complete_years and annual.goal_achieved.all()),
            "sum_independent_account_t_increment":float(annual.incremental_marked_pnl.sum()),
            "sum_completed_pair_lots":float(annual.completed_t_pair_lots.sum()),
            "sum_one_way_traded_lots":float(annual.total_t_traded_lots.sum()),
            "sum_complete_pairs":int(annual.complete_t_pairs.sum()),"sum_recovery_fragments":int(annual.recovery_fragments.sum()),
            "annual_results":records,"not_compounded":True,"partial_2026_not_annualized_for_target":True}
    _write_json(directory/"goal_evaluation.json",status)
    _plot_balanced(directory,cfg,annual,entries)
    _write_balanced_text(directory,cfg,annual,status)
    return status


def _plot_balanced(directory: Path, cfg: dict, annual: pd.DataFrame, entries: list) -> None:
    """仅展示已保存指标；各年独立作图，明确烟测范围及短区间日期。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    fonts=sorted(Path("/usr/share/fonts/opentype/noto").glob("*CJK*"))
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]));plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(fonts[0])).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    scope=cfg.get("report_scope","formal")
    scope_text={"formal":"正式年度验证","smoke":"工程烟测，不得作为收益证据",
                "internal":"训练内部评价，不是外层验证"}.get(scope,str(scope))
    fig,axes=plt.subplots(len(entries),2,figsize=(15,5.4*len(entries)),squeeze=False)
    fig.suptitle(f"均衡固定底仓做T（v12）｜{scope_text}",fontsize=15,y=.995)
    fmt=lambda value,spec: "不可评价" if value is None or pd.isna(value) else format(value,spec)
    for index,(year,result) in enumerate(entries):
        row=annual.loc[annual.test_year.eq(int(year))].iloc[0];curve=result["equity_curve"]
        metric=lambda name,spec='.2f':fmt(row.get(name),spec)
        time=pd.to_datetime(curve.datetime,format="mixed");left,right=axes[index]
        left.plot(time,(curve.baseline_equity/cfg["initial_cash"]-1)*100,label="相同固定底仓")
        left.plot(time,(curve.equity/cfg["initial_cash"]-1)*100,'--',label="固定底仓＋做T")
        left.set_ylabel("原始账户本金收益（%）");left.legend()
        left.set_title(f"{int(year)}｜{time.iloc[0]:%Y-%m-%d}—{time.iloc[-1]:%Y-%m-%d}")
        right.plot(time,curve.equity-curve.baseline_equity,color="#177f75")
        right.axhline(0,color='grey',lw=.7);right.set_ylabel("真实净增量（元）")
        right.set_title(f"净增量 {row.incremental_marked_pnl:,.2f} | 股数偏差 {row.inventory_mismatch_days}日")
        left_text=(f"总收益 {metric('total_return','.2%')}｜基准 {metric('baseline_total_return','.2%')}\n"
                   f"年化估计 {metric('annual_return','.2%')}｜年化波动 {metric('annual_volatility','.2%')}\n"
                   f"Sharpe {metric('sharpe')}｜最大回撤 {metric('max_drawdown','.2%')}")
        wins=metric('win_rate','.2%') if row.complete_t_pairs else "不可评价（无完整T组）"
        right_text=(f"完整T组 {int(row.complete_t_pairs)}｜配对手 {row.completed_t_pair_lots:g}｜净胜率 {wins}\n"
                    f"成本降低率 {metric('holding_cost_reduction_rate','.2%')}｜期末持仓 {metric('final_open_position_rows','.0f')}笔\n"
                    f"底仓股票 {metric('reference_stock_count','.0f')}只｜平均持仓市值 {metric('mean_holdings_market_value',',.0f')}元")
        for ax,content in ((left,left_text),(right,right_text)):
            ax.text(0,-.22,content,transform=ax.transAxes,fontsize=9,va='top',ha='left',
                    bbox={"boxstyle":"round,pad=.4","facecolor":"#f2f5f7","edgecolor":"#d7dfe5"})
        short_span=(time.iloc[-1]-time.iloc[0]).days<=90 and time.iloc[0].year==time.iloc[-1].year
        for ax in (left,right):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d' if short_span else '%Y-%m'));ax.grid(alpha=.25)
    fig.text(.5,.012,"各年独立账户；年化仅为区间换算，不用于5%目标验收。收益与风险指标按原账户口径，胜率按完整T组。",
             ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.045,1,.95),h_pad=3.0);fig.savefig(directory/"test_return_curve.png",dpi=150);plt.close(fig)
    shutil.copyfile(directory/"test_return_curve.png",directory/"测试集收益曲线.png")


def _write_balanced_text(directory: Path, cfg: dict, annual: pd.DataFrame, status: dict) -> None:
    """收益、手数、持仓、均衡与失败逐项披露，不将少量交易或未清底仓误判。"""
    scope={"formal":"正式年度验证","smoke":"工程烟测，不证明收益","internal":"训练内部评价"}[status["report_scope"]]
    fmt=lambda x,spec='.2f':"不可评价" if x is None or pd.isna(x) else format(x,spec)
    lines=[f"# 固定金额底仓做T：{scope}","",
           f"整体目标达成：**{status['goal_achieved']}**。各验证年5%成本降低目标：{status['each_validation_year_profit_target_met']}；"
           f"手数目标：{status['frequency_target_status']}。未指定数值手数要求时只列实际活动，不将一组或少量交易称为足够多；这不影响训练或开启。", "",
           f"每股自动建仓目标名义额 {cfg.get('target_position_notional',15000):,.0f}元，范围 "
           f"{cfg.get('min_position_notional',10000):,.0f}—{cfg.get('max_position_notional',20000):,.0f}元（含滑点成交价×股数，不含买入费用）。"
           "固定底仓不自动卖出；人工加仓为明确金额区间例外，其全部含费成本仍加入分母。", "",
           "| 年度 | T状态 | 原始本金 | 全部底仓含费成本 | T净增量 | 成本降低率 | 账户额外收益 | 底仓/加入T总收益 |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in annual.itertuples(index=False):
        lines.append(f"| {r.test_year} | {r.t_activity_status} | {r.initial_cash:,.0f} | {r.all_reference_acquisition_cost:,.2f} | {r.incremental_marked_pnl:+,.2f} | "
                     f"{fmt(r.holding_cost_reduction_rate,'.4%')} | {r.account_increment_return:.4%} | {r.baseline_total_return:.4%}/{r.total_return:.4%} |")
    lines += ["", "5%默认分母是全部无T基准的原始实际买入金额＋买入费用，包含零T股票及人工加仓；T买回、交易额、平均占用、期末市值不进入这个分母。"
              "T盈利不先减成本再算比例。账户初始本金＋正外部入金比率并列，不能事后挑较小分母。"
              "分母为0时不可评价。净价差已扣费/含滑点，不能重复扣滑点。", "",
              "| 年度 | 完整组/恢复片段 | 配对标准手/双腿成交手 | 每市场日/旧股日配对手 | 活跃日/市场日 | 活跃股票日/旧股股票日 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in annual.itertuples(index=False):
        lines.append(f"| {r.test_year} | {r.complete_t_pairs}/{r.recovery_fragments} | {r.completed_t_pair_lots:g}/{r.total_t_traded_lots:g} | "
                     f"{fmt(r.completed_pair_lots_per_market_day)}/{fmt(r.completed_pair_lots_per_old_stock_day)} | {r.active_days}/{r.market_evaluation_days} | "
                     f"{r.active_stock_days}/{r.baseline_old_stock_opportunity_stock_days} |")
    lines += ["", "100股=1标准手；600股分两次恢复仍是1组6配对手，双腿共12成交手。未完成组不计完整配对手，实际首腿/恢复/待恢复手数均另列。"
              "旧股日来自无T基准日初至少100股可卖库存，不用模型门槛或未来成交成功筛分母；全市场评价日包括0成交日。"
              "覆盖分子为实际活动与基准股票日交集，基准之外的实际活动单列。", "",
              "| 年度 | 基准股票/零T股票 | 金额违规/未成交尝试 | 股数偏差/恢复失败日 | 待恢复/T1/底仓差 | T费用/滑点 | 期末底仓/陈旧市值 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in annual.itertuples(index=False):
        lines.append(f"| {r.test_year} | {r.reference_stock_count}/{r.zero_t_reference_stock_count} | "
                     f"{r.automatic_entry_amount_violations}/{r.unfilled_automatic_entry_attempts} | {r.inventory_mismatch_days}/{r.recovery_failure_days} | "
                     f"{r.pending_pairs}/{r.t1_violation_count}/{r.base_plan_mismatch_count} | {r.t_fee:.2f}/{r.t_slippage_cost:.2f} | "
                     f"{r.final_open_position_rows}/{r.final_stale_market_value:.2f} |")
    lines += ["", "期末正常底仓是本策略应保留的股份，不按未清仓判失败；未恢复T债务、每日股数差及自动底仓卖出则必须单列。"
              "有库存或共同底仓偏差时，账户净值差不能全称为降成本；完整配对净额和未配对/估值差分别保存。"
              "每股成本降低率按原始成本加权汇总，零T与亏损股票不删除；原成本权重HHI、前五大和正/负收益贡献在CSV中。", "",
              "各年度独立账户，要求每个配置验证年达到目标，不用平均收益或跨年复利替代。2026仅至4月30日，直接评价该区间，不年化后过5%。"
              "2023—2025留一年训练可含未来年份，是回顾性检验；2026虽前推，既有历史已反复研究，不能称全新测试。", "",
              "本轮按训练内部实际净收益比较执行合格主动规则，负收益也如实保留；只有没有执行合格主动候选时才持有。"
              "本报告不增设收益、样本或概率停用门槛，不依据外层结果换规则。", "",
              "实际区间："+"；".join(f"{r.test_year} {r.evaluation_start}至{r.evaluation_end}" for r in annual.itertuples())+"。", "",
              "仅OHLCV与近似限价不能保证未来成交，缺报价和陈旧估值须结合原始审计查看。"
              "工程检查通过与5%/频率目标通过是不同结论。", ""]
    (directory/"final_report.md").write_text("\n".join(lines),encoding="utf-8")
    (directory/"输出文件说明.md").write_text("# 固定底仓报告文件\n\n"
        "- annual_summary.csv、年度.csv、test_metrics.csv：原始成本/账户双口径、手数、覆盖与失败。\n"
        "- per_stock_cost_reduction.csv：全部底仓股票，包括零T；原始成本、人工加仓、净价差与贡献集中度。\n"
        "- t_pair_lot_summary.csv：每个首腿pair的原股数、完整状态、配对手、恢复片段及费用净額。\n"
        "- daily_frequency_coverage.csv：全市场日历中包括零成交日的配对手数及旧股/活动股票日。\n"
        "- position_amount_audit.csv：实际自动建仓金额区间与人工加仓例外、失败尝试。\n"
        "- test_equity_curve.csv、test_trade_events.csv、test_round_trips.csv、test_open_positions.csv、daily_inventory_audit.csv、baseline_trade_events.csv：原账户记录按年合并。\n"
        "- goal_evaluation.json：各年5%/金额/库存分别验收，未量化手数不判整体达成，也不控制开启。\n"
        "- test_return_curve.png、测试集收益曲线.png：各年独立账户及T人民币净增量。\n",encoding="utf-8")
