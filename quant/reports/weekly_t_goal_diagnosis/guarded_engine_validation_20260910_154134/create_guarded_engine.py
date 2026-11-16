"""仅生成新的执行条件引擎副本，不修改冻结原引擎。"""
from pathlib import Path
source=Path('quant_backtest/dynamic_t_engine.py').read_text()
text=source[:source.index('\ndef plan_dynamic_entries(')]
old='''"""固定金额底仓的因果动态恢复实验；从冻结v12复制账户核，旧源/缓存不改。

固定、净利缓冲、净利缓冲加止损三模式共享完整会话时钟及逐分钟恢复。当前
完成分钟只锁存退出意图，最早下一交易分钟成交；硬期限事前已知，前一交易
分钟锁存。真实分片、费用、现金、旧股预算及未恢复债务全部保留。
"""'''
new='''"""同刻首腿反方向限价复核；复制冻结动态账户核，旧引擎与缓存保持原样。

事前开启t_recheck_exit_side_at_execution后，仅在首腿本方向已有可填数量时，
用同一到达分钟quote检查反方向限价，命中则整笔取消。该条件属于到达报价的
成交限制，不是上一信号分钟已知的新因子；不能回看后续恢复价量或成功状态。
复核不模拟反向成交、不消耗容量/现金/旧股。默认关闭时原全部账户字段精确
复现，另增first_leg_exit_side_audit。底仓仍由原plan_dynamic_entries冻结。
"""'''
assert old in text;text=text.replace(old,new,1)
text=text.replace('def run_dynamic_portfolio(', 'def run_guarded_portfolio(',1)
text=text.replace('''    balanced_t_engine.plan_balanced_entries生成后重放；不重新规划或扩大基础股数。''','''    dynamic_t_engine.plan_dynamic_entries生成后重放；不重新规划或扩大基础股数。''',1)
text=text.replace('''    cfg = {**BALANCED_DEFAULTS, **DYNAMIC_DEFAULTS, **cfg}
    validate_balanced_config(cfg)''','''    cfg = {**BALANCED_DEFAULTS, **DYNAMIC_DEFAULTS, "t_recheck_exit_side_at_execution": False, **cfg}
    if not isinstance(cfg["t_recheck_exit_side_at_execution"], (bool, np.bool_)):
        raise ValueError("t_recheck_exit_side_at_execution须为显式布尔值")
    validate_balanced_config(cfg)''',1)
text=text.replace('''    exit_decisions = []
    base_opened''','''    exit_decisions = []
    first_leg_exit_side_audits = []
    base_opened''',1)
marker='''DYNAMIC_DEFAULTS = {"dynamic_exit_mode": "fixed", "dynamic_take_net_return": .003,
                    "dynamic_stop_net_return": .01}'''
newmarker=marker+'''
FIRST_LEG_EXIT_SIDE_AUDIT_COLUMNS = [
    "datetime", "signal_datetime", "symbol", "side", "opposite_side", "requested_shares",
    "own_fillable_shares", "guard_enabled", "opposite_checked", "cancelled", "reason",
    "opposite_quote_reason", "reference_price", "high", "low", "amount", "volume",
    "previous_close", "limit_up", "limit_down", "minute_amount_used_before", "cash_before",
    "old_sellable_before"]'''
assert marker in text;text=text.replace(marker,newmarker,1)
oldblock='''                quantity, price, reason = calculate_quote_fill(quote, side, quantity, actual["used"].get(symbol, 0.0))
                notional_clipped = False'''
newblock='''                quantity, price, reason = calculate_quote_fill(quote, side, quantity, actual["used"].get(symbol, 0.0))
                if quantity:
                    opposite_side = "sell" if side == "buy" else "buy"
                    enabled = bool(cfg["t_recheck_exit_side_at_execution"])
                    opposite_reason = "not_checked"
                    if enabled:
                        # 原纯函数先检查quote及方向限价。零数量仅询问限价，不能
                        # 以虚拟反向整笔容量筛首腿，更不能调用fill_book/recover_pair。
                        _, _, opposite_reason = calculate_quote_fill(quote, opposite_side, 0, 0.0)
                        if opposite_reason not in ("", "price_limit", "conservative_price_limit_proxy"):
                            raise AssertionError("本方向已可填的同一quote出现非限价反向拒因")
                    cancelled = enabled and opposite_reason in ("price_limit", "conservative_price_limit_proxy")
                    cancel_reason = "execution_exit_side_" + opposite_reason if cancelled else ""
                    first_leg_exit_side_audits.append({
                        "datetime": timestamp, "signal_datetime": row["datetime"], "symbol": symbol,
                        "side": side, "opposite_side": opposite_side, "requested_shares": requested_quantity,
                        "own_fillable_shares": quantity, "guard_enabled": enabled,
                        "opposite_checked": enabled, "cancelled": cancelled, "reason": cancel_reason,
                        "opposite_quote_reason": opposite_reason, "reference_price": quote["price"],
                        **{key: quote[key] for key in ("high", "low", "amount", "volume", "previous_close", "limit_up", "limit_down")},
                        "minute_amount_used_before": actual["used"].get(symbol, 0.0),
                        "cash_before": actual["cash"], "old_sellable_before": old})
                    if cancelled:
                        record_attempt(actual, symbol, "", side, requested_quantity, 0, timestamp,
                                       row["datetime"], "t_entry", cancel_reason)
                        continue
                notional_clipped = False'''
assert text.count(oldblock)==1;text=text.replace(oldblock,newblock,1)
oldreturn='''            "exit_decisions": pd.DataFrame(exit_decisions, columns=EXIT_DECISION_COLUMNS)}'''
newreturn='''            "exit_decisions": pd.DataFrame(exit_decisions, columns=EXIT_DECISION_COLUMNS),
            "first_leg_exit_side_audit": pd.DataFrame(first_leg_exit_side_audits,
                                                      columns=FIRST_LEG_EXIT_SIDE_AUDIT_COLUMNS)}'''
assert text.count(oldreturn)==1;text=text.replace(oldreturn,newreturn,1)
path=Path('quant_backtest/guarded_t_engine.py');assert not path.exists();path.write_text(text)
print(path)
