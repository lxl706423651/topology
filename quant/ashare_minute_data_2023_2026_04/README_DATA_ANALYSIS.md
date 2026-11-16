# 一分钟沪深股票数据分析说明

生成时间：2026-05-17 12:16:16

## 数据概览

- 原始目录：`E:\quant\一分钟沪深股票，2023-2026.4`
- 解压目录：`E:\quant\一分钟沪深股票，2023-2026.4\extracted`
- 清洗后 pkl：`E:\quant\一分钟沪深股票，2023-2026.4\processed\pkl`
- slice 示例：`E:\quant\一分钟沪深股票，2023-2026.4\slices`
- 覆盖年份：2023, 2024, 2025, 2026
- 股票文件数/代码数：5646
- 原始行数：1,029,640,606
- 清洗后行数：1,029,640,606
- 行数保留率：100.00%
- 时间范围：2023-01-03 09:31:00 到 2026-04-30 15:00:00
- 当前 pkl 总大小：32.61 GB

## 清洗规则

1. 每个 ZIP 按年份解压到 `extracted/<year>/`，避免不同年份同名股票 CSV 互相覆盖。
2. 标准字段统一为 `datetime, symbol, open, high, low, close, volume, amount`。
3. `datetime` 转成 pandas 时间类型；价格、成交量、成交额转成数值。
4. 过滤非 A 股常规分钟时段，只保留 09:30-11:30 与 13:00-15:00。
5. 删除缺失 OHLC、非正价格、`high < low`、`high` 小于开/收、`low` 大于开/收的异常行。
6. 同一股票同一分钟重复时保留最后一条。
7. 为了照顾 32GB 内存，pkl 按 `year=<年>/half=H1|H2/part_xxxx.pkl` 切片保存。

## 质量统计文件

- `analysis/data_profile_by_symbol.csv`：每个股票 CSV 的行数、缺失、重复、异常、时间范围。
- `analysis/pkl_manifest.csv`：每个 pkl 切片的行数、股票数、起止时间。

## 如何读取一个切片

```python
import pandas as pd

df = pd.read_pickle(r"E:\quant\一分钟沪深股票，2023-2026.4\processed\pkl\year=2023\half=H1\part_0000.pkl")
print(df.head())
print(df.dtypes)
```

## slice 示例用途

- `slice_head_1000.pkl`：快速看字段和类型。
- `slice_one_symbol_*.pkl`：看单只股票的分钟序列。
- `slice_5_symbols_5000_rows.pkl`：看多股票横截面与时间排序。

后续特征工程会基于这些清洗后的 pkl 继续生成 100-200 个技术和统计特征。
