"""独立审计器的小反例检查；不拟合模型或读取外层账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_weekly_t_timing_models import independent_timing_features, labels_from_prices

cfg = json.loads(Path('reports/weekly_t_cross_validation_smoke/run_20260909_182533_118901/resolved_config.json').read_text())
times = pd.to_datetime(['2023-08-25 09:35', '2023-08-28 09:35']).repeat(3)
frame = pd.DataFrame({'datetime': times, 'symbol': ['A', 'B', 'C']*2,
    'feature_source_start': times.normalize(), 'execution_price': [100.]*6,
    'recovery_price': [100., 101., 99.]*2})
frame['execution_time'] = frame.datetime+pd.Timedelta(minutes=1)
frame['label_start'] = frame.execution_time
frame['label_end'] = frame.execution_time+pd.Timedelta(minutes=20)
frame['recovery_time'] = frame.label_end
frame['target_return'] = frame.recovery_price/frame.execution_price-1
labels = labels_from_prices(frame, 20, cfg)
assert labels.profit_class.tolist() == [0, 1, 2, 0, 1, 2]
assert labels.buy_net_return.iloc[3] > labels.buy_net_return.iloc[0]
print('PASS exact two-date stamp contract and three classes')
for column, value in [('target_return', .8), ('label_end', pd.Timestamp('2023-08-25 10:30'))]:
    bad = frame.copy()
    bad.loc[0, column] = value
    try:
        labels_from_prices(bad, 20, cfg)
    except AssertionError:
        print('PASS reject corrupt', column)
    else:
        raise AssertionError('missed corrupt '+column)
features = pd.DataFrame({'datetime': pd.to_datetime(['2024-06-03 09:35']*3), 'symbol': ['A', 'B', 'C']})
for column, values in {
    'f_return_5m': [.01, -.01, .02], 'f_return_15m': [.02, -.01, .01],
    'f_open_return': [.03, 0., -.01], 'f_day_return': [.02, .04, -.03],
    'f_amount_vs_previous_day': [.4, .2, .5], 'f_session_fraction': [.1, .1, .1],
    'f_known_amplitude': [.03, .04, .05], 'f_realized_volatility': [.02, .01, .03],
    'f_previous_day_amplitude': [.03, .01, .02], 'f_vwap_deviation': [.001, -.002, .003],
}.items():
    features[column] = pd.Series(values, dtype='float32')
base = independent_timing_features(features.copy())
future = features.copy()
future.datetime += pd.Timedelta(days=1)
# 保持原缓存的 float32 合同；追加行不能先把整个历史列提升成 float64。
future['f_return_5m'] = np.float32(100.)
combined = independent_timing_features(pd.concat([features, future], ignore_index=True)).iloc[:3]
pd.testing.assert_frame_equal(base, combined, check_exact=True)
print('PASS future append does not change full current-clock features')
print('4 independent auditor checks passed')
