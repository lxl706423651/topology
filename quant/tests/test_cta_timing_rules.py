"""开仓时钟与信号对齐测试，不访问历史收益或回填未来数据。"""

import pandas as pd
import pytest

from quant_backtest.cta_timing_rules import TimingValidationStrategy, calculate_entry_clock


def test_clock_handles_lunch_and_keeps_only_declared_offset():
    """上午下午均按交易分钟编号，午休不算可开仓分钟。"""
    dates = pd.Series(pd.to_datetime(['2024-01-02 09:31', '2024-01-02 09:55', '2024-01-02 13:01', '2024-01-02 13:25']))
    assert calculate_entry_clock(dates, 240, 24).tolist()==[False, True, False, False]
    assert calculate_entry_clock(dates, 240, 120).tolist()==[False, False, True, False]


@pytest.mark.parametrize('frequency,offset', [(60,0), (240,1), (240,240), (0,0)])
def test_clock_rejects_unavailable_sampling_slots(frequency, offset):
    """frequency/offset不在24分钟缓存时点上必须报错，不能把更稀疏信号冒充该频率。"""
    with pytest.raises(ValueError):
        calculate_entry_clock(pd.Series(pd.to_datetime(['2024-01-02 09:31'])), frequency, offset)


def test_timing_strategy_never_relabels_feature_times():
    """选择09:55时点后成交依据仍为09:55，不能把当时特征伪装成09:31已知。"""
    data = pd.DataFrame(dict(datetime=pd.to_datetime(['2024-01-02 09:31','2024-01-02 09:55']),
                             symbol='SH.600000', score_exit0_shallow_probability=0.6))
    rule = dict(model_name='exit0_shallow_probability', exit_id=0, style='model_only', context='none', frequency=240, offset=24, gate=0.4)
    output = TimingValidationStrategy(rule).generate_signals(data)
    pd.testing.assert_series_equal(output.datetime, data.datetime)
    assert output.open_signal.tolist()==[0,1] and output.close_signal.eq(0).all()
