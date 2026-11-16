"""预测期限与退出时刻的配对研究：在旧独立批次引擎上增加到期与到期后模型退出。

仅消费当时信号、实际交易日历和冻结持仓模型；不访问未来收益或未来成交量。
ATR保护及T+1/整手/容量/待平仓处理仍由annual_portfolio负责，不改变旧引擎。
"""

from __future__ import annotations

from quant_backtest.learned_exit_models import LearnedExitPolicy


class HorizonExitPolicy:
    """仅做多批次的期限退出器；离池不退出，同票不同买入信号分别计算到期时刻。"""

    def __init__(self, market, horizon: int, mode: str, learned_policy=None):
        """market给共同时钟/交易日序号；horizon为正交易日数；mode选到期退出或到期后交给learned_policy。"""
        if horizon < 1 or int(horizon) != horizon:
            raise ValueError("预测期限必须为正整数交易日")
        if mode not in {"matched_horizon", "minimum_horizon_model"}:
            raise ValueError("未知的期限退出模式")
        if mode == "minimum_horizon_model" and learned_policy is None:
            raise ValueError("到期后模型模式缺少持仓预测器")
        self.market, self.horizon, self.mode, self.learned = market, int(horizon), mode, learned_policy

    def __call__(self, index: int, positions: list) -> dict[int, str]:
        """对index时刻positions检查到期；到期是入场信号后horizon交易日同刻，不要求仍有选股分数。"""
        market = self.market
        now = market.times[index]
        if now.hour == 15:
            return {}
        due = []
        for lot in positions:
            if now.normalize() <= lot.buy_time.normalize() or lot.pending_reason:
                continue
            age = market.day_index[index] - market.day_index[lot.entry_index]
            signal_offset = lot.buy_signal - lot.buy_signal.normalize()
            if age > self.horizon or (age == self.horizon and now - now.normalize() >= signal_offset):
                due.append(lot)
        if not due:
            return {}
        if self.mode == "minimum_horizon_model":
            return self.learned(index, due)
        return {lot.lot_id: f"达到开仓预测期限{self.horizon}个交易日同刻，下一分钟尝试退出" for lot in due}


def create_alignment_policy(model, matrices: dict, market, rule: dict, cfg: dict):
    """按rule.alignment_mode与cfg.model_profiles对应期限建退出器；省略模型退出的对照返回None或到期器。"""
    mode = rule["alignment_mode"]
    if mode == "next_day_control":
        return None
    learned = LearnedExitPolicy(model, matrices, market, rule["exit_threshold"])
    if mode == "model_control":
        return learned
    horizons = {p["name"]: p["horizon"] for p in cfg["model_profiles"]}
    return HorizonExitPolicy(market, horizons[rule["profile"]], mode, learned)
