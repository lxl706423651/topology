"""逐验证集搜索的口径、时间和撮合一致性测试，不读取2026历史收益。"""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from test_cta_research import create_market
from quant_backtest.cta_validation_search import run_fast_metrics, build_signals, rank_common_rules, ValidationCtaStrategy
from quant_backtest.cta_research_engine import run_portfolio
import quant_backtest.cta_research_engine as execution_engine


@pytest.mark.parametrize('gate', [0.5, 1.0])
def test_fast_metrics_match_full_engine_with_fills_or_no_entries(gate):
    """gate分别生成真实成交或全现金，快速通道必须和完整通道的全部公共指标一致。"""
    market = create_market(days=6)
    market.quotes[480:, :, :3] = [94.0, 94.5, 93.5]
    frame = pd.DataFrame([{'datetime': market.times[t], 'symbol': symbol, 'score_exit0_regularized_probability': 0.7}
        for t in range(0, len(market.times), 120) for symbol in market.symbols])
    rule = dict(model_name='exit0_regularized_probability', exit_id=0, style='model_only', frequency=120, context='none', gate=gate)
    fold = dict(validation_start=market.times[0].normalize(), validation_end=market.times[-1].normalize()+pd.Timedelta(days=1))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/xgb_cta_expanded.yaml').read_text())
    source = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/xgb_cta_local_history.yaml').read_text())
    cfg = {**source, **cfg}
    fast = run_fast_metrics(frame, rule, market, fold, cfg)
    full = run_portfolio(build_signals(frame, rule), market, fold['validation_start'], fold['validation_end'], cfg)
    for key, value in fast.items():
        assert np.isclose(value, full['metrics'][key], rtol=1e-10, atol=1e-10), key
    if gate < 1:
        assert fast['buy_count']>0 and fast['sell_count']>0


def test_all_folds_pass_rejects_high_mean_or_exact_target():
    """平均值很高、某折等于2或者现金夏普占位，都不能满足每折严格>2。"""
    rows = []
    for name, scores in [('uneven', [4.0, 4.0, 1.9]), ('edge', [2.0, 2.1, 2.2]), ('pass', [2.01, 2.1, 2.2])]:
        for year, score in zip([2023, 2024, 2025], scores):
            rows.append(dict(rule_id=name, year=year, sharpe=score, total_return=0.1, annual_volatility=0.1, closed_lot_count=20))
    cfg = dict(minimum_closed_lots_per_fold=20, sharpe_target=2.0)
    ranked = rank_common_rules(pd.DataFrame(rows), [{'year': y} for y in [2023, 2024, 2025]], cfg)
    assert ranked.iloc[0].rule_id=='pass' and ranked.all_folds_pass.sum()==1


def test_missing_validation_fold_cannot_be_omitted():
    """缺少某个验证年时必须拒绝评价，不允许悄悄缩小目标。"""
    rows = pd.DataFrame([dict(rule_id='partial', year=y, sharpe=4, total_return=0.1, annual_volatility=0.1, closed_lot_count=100)
                         for y in [2023, 2024]])
    with pytest.raises(ValueError, match='全部验证'):
        rank_common_rules(rows, [{'year': y} for y in [2023, 2024, 2025]], dict(minimum_closed_lots_per_fold=20, sharpe_target=2))


def test_strategy_preserves_rows_and_ignores_future_fields():
    """未声明的未来收益列不影响信号；函数不得修改原数据或用零开仓信号指示退出。"""
    data = pd.DataFrame(dict(datetime=pd.to_datetime(['2024-02-01 09:31']), symbol=['SH.600000'],
                             score_exit0_regularized_probability=[0.6], future_return=[100]))
    rule = dict(model_name='exit0_regularized_probability', exit_id=0, style='model_only', frequency=120, context='none', gate=0.5)
    before = data.copy(deep=True)
    strategy = ValidationCtaStrategy(rule)
    signals = strategy.generate_signals(data)
    data.future_return = -100
    pd.testing.assert_frame_equal(signals, strategy.generate_signals(data))
    pd.testing.assert_frame_equal(before.drop(columns='future_return'), data.drop(columns='future_return'))
    assert signals.open_signal.iloc[0]==1 and signals.close_signal.iloc[0]==0


def test_unavailable_pending_exit_does_not_rescan_past_prices(monkeypatch):
    """monkeypatch禁止旧价扫描；已知不可卖的待退出批次应直接拒绝，仍保留T+1优先及原因1。"""
    market = create_market(days=3)
    market.quotes[120:, :, :3] = [94.0, 94.5, 93.5]
    market.flags[240:, :, 1] = 0
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/xgb_cta_local_history.yaml').read_text())
    shape = (len(market.times), 5)
    candidates, predictions, profiles = np.full(shape, -1, dtype=np.int64), np.ones(shape), np.zeros(shape, dtype=np.int64)
    candidates[0, 0] = 0
    parameters, exits, stamps = execution_engine.create_parameters(cfg, market.times)
    def reject_scan(*args):
        """不可卖时调用args指向的历史扫描意味着优化契约未成立。"""
        raise AssertionError('不可卖时不应回溯旧报价')
    monkeypatch.setattr(execution_engine, 'find_past_price', reject_scan)
    curve, events, trips, blocks, lots, marks = execution_engine.simulate_account.py_func(
        market.quotes, market.flags, market.boards, candidates, predictions, profiles, exits, parameters, stamps, 0, len(market.times))
    assert len(events)==1 and len(trips)==0 and lots[0, 1]>0
    assert (blocks[:, 4]==4).any() and (blocks[:, 4]==1).any()
