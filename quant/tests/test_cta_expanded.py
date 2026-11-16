"""扩展CTA研究的因果性和训练边界测试；合成数据不访问网络或测试收益。"""

import numpy as np
import pandas as pd
import json
import pytest

from quant_backtest.cta_expanded_features import build_daily_features, align_daily_features
from quant_backtest.cta_expanded_training import create_internal_folds, ExpandedCtaStrategy, select_robust_rule, fit_candidate


def create_history() -> pd.DataFrame:
    """生成单股票确定性日线，供未来扰动反例使用。"""
    c = np.arange(100, dtype=float) + 100
    return pd.DataFrame(dict(date=pd.date_range('2023-01-01', periods=100), open=c, high=c+2,
                             low=c-2, close=c+1, volume=c*100, amount=(c+1)*c*100))


def test_daily_features_ignore_current_and_future_days():
    """改变决策日及未来全部价格，已可用的历史特征不能变化。"""
    history = create_history()
    signals = pd.DataFrame({'datetime': pd.to_datetime(['2023-03-01 09:31', '2023-03-01 14:19'])})
    before = align_daily_features(signals, build_daily_features(history, 20, [5, 20]))
    history.loc[history.date >= '2023-03-01', ['open', 'high', 'low', 'close', 'volume', 'amount']] *= 7
    after = align_daily_features(signals, build_daily_features(history, 20, [5, 20]))
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_series_equal(before.iloc[0], before.iloc[1], check_names=False)


def test_daily_alignment_keeps_input_order_and_no_backfill():
    """打乱信号顺序后仍对应原行，没有历史的日期必须缺失。"""
    daily = build_daily_features(create_history(), 2, [5])
    signals = pd.DataFrame({'datetime': pd.to_datetime(['2023-03-01', '2022-01-01', '2023-02-01'])})
    aligned = align_daily_features(signals, daily)
    assert aligned.iloc[1].isna().all()
    assert aligned.loc[0, 'ctx_daily_close_lag1'] != aligned.loc[2, 'ctx_daily_close_lag1']


def test_folds_exclude_2026():
    """所有内部选择期间严格早于2026，训练与验证不交叠。"""
    for fold in create_internal_folds():
        assert fold['train_start'] < fold['train_end'] <= fold['validation_start'] < fold['validation_end']
        assert fold['validation_end'] <= pd.Timestamp('2026-01-01')


def test_strategy_only_reads_fixed_features_and_preserves_input():
    """模型只能收到固定列；额外未来标签不影响信号，且不修改调用方数据。"""
    class FixedModel:
        def predict(self, frame):
            assert list(frame) == ['ctx_feature']
            return np.full(len(frame), 1.0)
    data = pd.DataFrame(dict(datetime=pd.to_datetime(['2024-01-02 09:31']), symbol=['SH.600000'],
                             ctx_feature=[2.0], ctx_momentum20=[0.1], ctx_ema20_gap=[0.1],
                             ctx_momentum5=[0.1], ctx_ema5_gap=[0.1], future_return=[100]))
    before = data.copy(deep=True)
    rule = dict(objective='return', style='trend', entry_minutes=24, gate=0, exit_profile=0)
    result = ExpandedCtaStrategy(FixedModel(), ['ctx_feature'], rule).generate_signals(data)
    assert result.open_signal.iloc[0] == 1 and result.close_signal.iloc[0] == 0
    pd.testing.assert_frame_equal(before, data)


def test_selection_retains_failed_candidates():
    """跨折波动惩罚能避免只选单折最高值，失败候选仍保留。"""
    rows = []
    for name, scores in [('stable', [1.5, 1.5]), ('unstable', [4.0, -1.0])]:
        for score in scores:
            rows.append(dict(rule_id=name, objective='return', exit_profile=0, profile_name='test', style='trend',
                             entry_minutes=24, gate=0, sharpe=score, closed_lot_count=20, total_return=0.1))
    selected, table = select_robust_rule(pd.DataFrame(rows), {'validation_min_closed_lots': 20})
    assert selected['rule_id'] == 'stable' and len(table) == 2


@pytest.mark.parametrize('objective', ['return', 'probability', 'downside'])
def test_training_weights_unchanged_when_2026_is_perturbed(tmp_path, objective):
    """对三类objective将2026特征/目标任意改动，过去拟合权重必须逐字节不变；tmp_path保存临时模型。"""
    dates = pd.date_range('2025-01-01', periods=400)
    index = np.arange(len(dates))
    panel = pd.DataFrame(dict(datetime=dates, symbol='SH.600000', ctx_x=np.sin(index/7), ctx_y=np.cos(index/11)))
    labels = pd.DataFrame(dict(net_return_0=np.sin(index/13)*0.02, label_end_0=dates+pd.Timedelta(days=30)))
    fold = dict(year='test', train_start=pd.Timestamp('2025-01-01'), train_end=pd.Timestamp('2026-01-01'))
    cfg = dict(feature_min_coverage=0.8, feature_max_correlation=0.95, max_features=2, random_seed=42, workers=1)
    profile = dict(name='unit', n_estimators=3, max_depth=2, min_child_weight=1, reg_lambda=1)
    fit_candidate(panel, labels, fold, 0, profile, objective, cfg, tmp_path/'before')
    panel.loc[panel.datetime >= fold['train_end'], ['ctx_x', 'ctx_y']] = 1e20
    labels.loc[panel.datetime >= fold['train_end'], 'net_return_0'] = -100
    fit_candidate(panel, labels, fold, 0, profile, objective, cfg, tmp_path/'after')
    before = json.loads((tmp_path/'before/model_metadata.json').read_text())
    after = json.loads((tmp_path/'after/model_metadata.json').read_text())
    assert before['model_sha256'] == after['model_sha256']
    assert before['feature_values_sha256'] == after['feature_values_sha256']
    rows = pd.read_csv(tmp_path/'after/training_rows.csv', parse_dates=['datetime', 'label_end'])
    assert (rows.label_end < fold['train_end']).all()
    assert len(rows) < int((panel.datetime < fold['train_end']).sum())
