# 分钟数据特征工程说明

生成时间：2026-05-17 12:55:33

## 输出位置

- 特征 pkl：`E:\quant\一分钟沪深股票，2023-2026.4\processed\features_pkl`
- 特征清单：`E:\quant\一分钟沪深股票，2023-2026.4\analysis\feature_columns.txt`
- 特征 manifest：`E:\quant\一分钟沪深股票，2023-2026.4\analysis\feature_manifest.csv`
- 特征 slice：`E:\quant\一分钟沪深股票，2023-2026.4\slices\feature_slice_2000.pkl`

## 当前结果

- 特征列数量：197
- 已生成行数：23,131,444
- 单个切片最大股票数：8

## 特征类别

1. 收益与动量：`ret_*`, `logret_*`, `roc_*`
2. 波动率与极值：`ret_std_*`, `atr_ratio_*`, `range_*`, `dist_high_*`, `dist_low_*`
3. 均线与趋势：`sma_gap_*`, `ema_gap_*`, `ma_cross_*`
4. 价量关系：`volume_ratio_*`, `amount_ratio_*`, `vwap_gap_*`, `obv_z_*`
5. 技术指标：`rsi_*`, `cci_*`, `willr_*`, `mfi_*`, `adx_*`, `macd_*`, `boll_*`, `stoch_*`
6. 日内时间特征：`minute_sin`, `minute_cos`, `weekday_sin`, `weekday_cos`, `is_morning`

## 标签

- `future_return_1`：未来 1 分钟收益
- `future_return_5`：未来 5 分钟收益
- `future_return_24` / `Return_24`：未来 24 分钟收益，兼容 `邦盛期间量化策略` 里的旧脚本命名

特征只使用当前和历史 K 线，未来收益只作为训练标签，回测时不能作为模型输入。
