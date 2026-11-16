# 日频小市值特征缓存

- 成功股票数：5 / 5
- 行数：4020
- 日期范围：2023-01-03 至 2026-04-30
- pkl：`daily_size_features.pkl`
- csv：`daily_size_features.csv`
- 计算方法：`derived_float_market_cap = close * volume_hands * 100 / (turnover_rate / 100)`

注意：这是从换手率反推的流通市值，不是交易所原始披露字段。若换手率为 0 或缺失，该日市值为 NaN。
