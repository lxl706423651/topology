# 量化数据处理、回测框架与模型实验说明

生成时间：2026-05-17

这份文档给刚入行的量化开发者看，重点不是堆概念，而是解释本项目现在有什么、我改了什么、数据在哪里、怎么继续做实验。

## 1. 本次完成结果

### 数据层

- 原始数据目录：`E:\quant\ashare_minute_data_2023_2026_04`
- 原始 ZIP：4 个，合计约 11.84 GB
- 解压后 CSV：21,828 个，位置：`E:\quant\ashare_minute_data_2023_2026_04\extracted`
- 清洗后分钟数据：1,029,640,606 行
- 清洗后 pkl：478 个切片，合计约 35.01 GB
- pkl 位置：`E:\quant\ashare_minute_data_2023_2026_04\processed\pkl`
- slice 示例：`E:\quant\ashare_minute_data_2023_2026_04\slices`
- 数据画像：`E:\quant\ashare_minute_data_2023_2026_04\analysis\data_profile_by_symbol.csv`
- pkl manifest：`E:\quant\ashare_minute_data_2023_2026_04\analysis\pkl_manifest.csv`

### 特征层

- 选股池：按 2026 年成交额选前 120 只高流动性股票
- 股票池文件：`E:\quant\ashare_minute_data_2023_2026_04\analysis\training_universe_top120_2026.csv`
- 特征 pkl：344 个切片，23,131,444 行，约 19.30 GB
- 特征数量：197 个模型特征
- 特征位置：`E:\quant\ashare_minute_data_2023_2026_04\processed\features_pkl`
- 特征清单：`E:\quant\ashare_minute_data_2023_2026_04\analysis\feature_columns.txt`
- 特征说明：`E:\quant\ashare_minute_data_2023_2026_04\README_FEATURES.md`

### 回测与模型

- Demo 回测报告：`E:\quant\reports\demo_after_improvements`
- 真实分钟数据冒烟回测：`E:\quant\reports\real_data_smoke_bollinger`
- 模型实验结果：`E:\quant\reports\model_experiments\model_metrics.csv`
- 模型产物：`linear_svm.joblib`, `lightgbm.joblib`, `lstm_model.pt`

## 2. 环境说明

已新建 conda 环境：

```powershell
conda activate quant
```

本次已创建 `quant` 环境并保存复现文件：`environment_quant.yml`。

关键版本：

| 组件 | 版本 |
|---|---:|
| Python | 3.10.20 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| LightGBM | 4.6.0 |
| XGBoost | 3.2.0 |
| CatBoost | 1.2.10 |
| TA-Lib | 0.6.4 |
| pyarrow | 24.0.0 |
| akshare | 1.18.62 |

环境创建命令：

```bash
conda env create -f environment_quant.yml
conda activate quant
```

## 3. 数据目录结构

```text
E:\quant\ashare_minute_data_2023_2026_04
├─ 2023.zip / 2024.zip / 2025.zip / 2026.zip
├─ extracted\
│  ├─ 2023\
│  ├─ 2024\
│  ├─ 2025\
│  └─ 2026\
├─ processed\
│  ├─ pkl\
│  │  ├─ year=2023\half=H1\part_0000.pkl
│  │  └─ ...
│  └─ features_pkl\
│     ├─ year=2023\half=H1\part_0003.pkl
│     └─ ...
├─ analysis\
│  ├─ data_profile_by_symbol.csv
│  ├─ pkl_manifest.csv
│  ├─ feature_manifest.csv
│  ├─ feature_columns.txt
│  └─ training_universe_top120_2026.csv
├─ slices\
│  ├─ slice_head_1000.pkl
│  ├─ slice_one_symbol_BJ.920000.pkl
│  ├─ slice_5_symbols_5000_rows.pkl
│  └─ feature_slice_2000.pkl
├─ README_DATA_ANALYSIS.md
└─ README_FEATURES.md
```

为什么按 `year/half/part` 切片：全量分钟数据超过 10 亿行，一次性读入会压垮内存。现在可以按年份、半年、part 小块读取。

## 4. 如何读取数据

读取清洗后的 OHLCV：

```python
import pandas as pd

path = r"E:\quant\ashare_minute_data_2023_2026_04\processed\pkl\year=2026\half=H1\part_0208.pkl"
df = pd.read_pickle(path)
print(df.head())
print(df.dtypes)
```

读取特征：

```python
import pandas as pd

path = r"E:\quant\ashare_minute_data_2023_2026_04\processed\features_pkl\year=2026\half=H1\part_0208.pkl"
fea = pd.read_pickle(path)
print(fea.shape)
```

注意：原始 `volume` 单位是“手”，`amount` 是成交额。估算分钟 VWAP 时应使用：

```python
vwap = amount / (volume * 100)
```

这个逻辑已经写入 `quant_backtest/features.py`。

## 5. 本次新增脚本

### 解压

```powershell
conda run -n quant python scripts\extract_stock_minute_data.py
```

作用：把每个 ZIP 解压到 `extracted/<year>`，避免不同年份同名 CSV 覆盖。

### 清洗并切片

```powershell
conda run -n quant python scripts\preprocess_stock_minute_data.py --batch-size 80
```

作用：标准化字段、检查异常、输出 pkl、生成数据画像和 slice。

### 选择训练股票池

```powershell
conda run -n quant python scripts\select_training_universe.py --year 2026 --top-n 120
```

作用：按成交额选高流动性股票，避免一开始用全市场训练导致内存和训练时间失控。

### 生成特征

```powershell
conda run -n quant python scripts\build_stock_minute_features.py --universe-file "E:\quant\ashare_minute_data_2023_2026_04\analysis\training_universe_top120_2026.csv"
```

作用：生成 197 个代表性特征和 `Return_24` 标签。

### 真实数据回测冒烟测试

```powershell
conda run -n quant python scripts\run_framework_smoke_tests.py --max-symbols 20 --max-rows 200000
```

作用：确认框架能跑真实 pkl 数据。

### 模型实验

```powershell
conda run -n quant python scripts\run_model_experiments.py --max-files 40 --max-rows 300000 --lstm-epochs 2 --lstm-max-sequences 60000 --batch-size 512
```

作用：跑 Linear SVM、LightGBM、LSTM，输出 IC 和多空收益评估。

## 6. 回测框架结构

核心目录：`E:\quant\quant_backtest`

| 文件 | 作用 |
|---|---|
| `data.py` | 读取和标准化 OHLCV 数据 |
| `engine.py` | 回测引擎，负责信号、持仓、收益、成本、组合权益 |
| `metrics.py` | 绩效指标，如收益、波动、Sharpe、回撤、交易胜率 |
| `indicators.py` | 技术指标，TA-Lib 可用时优先用 TA-Lib |
| `features.py` | 本次新增，生成分钟级技术/统计特征 |
| `strategy.py` | 策略接口，支持普通策略、函数策略、模型策略 |
| `modeling.py` | 加载 joblib/pickle/torch/keras 模型 |
| `reporting.py` | 保存 metrics、equity、trades、图表 |

运行入口：

```powershell
conda run -n quant python run_strategy.py
conda run -n quant python run_strategy.py --data path\to\data.pkl --output reports\my_test
```

## 7. 本次对框架的改进

1. `data.py` 新增 `load_ohlcv()`，现在支持 CSV、PKL、Parquet。
2. `data.py` 保留并标准化 `amount` 字段。
3. `run_strategy.py` 支持直接传入 pkl，不再只支持 CSV。
4. `engine.py` 的组合等权逻辑改为按真实持仓数量计 active count，而不是按收益是否为 0 判断。
5. `engine.py` 增加持仓和换手诊断指标：
   - `avg_abs_position`
   - `long_bar_rate`
   - `short_bar_rate`
   - `flat_bar_rate`
   - `avg_turnover`
   - `total_turnover`
6. 新增真实数据冒烟测试脚本。
7. 新增分钟特征工程模块和模型实验脚本。

## 8. 量化回测需要特别注意什么

本项目已经开始处理其中一部分，但还没有全部覆盖：

1. 避免未来函数：框架默认 `signal_lag=1`，信号下一根 K 线才生效。
2. 手续费和滑点：已支持 `commission_rate` 和 `slippage_bps`，但还不是完整撮合。
3. 时间切分：模型训练使用按时间切分，不随机打乱未来数据。
4. 生存者偏差：目前没有单独处理退市、停牌、指数成分历史变化。
5. 复权和公司行为：当前数据是否前复权/后复权需要再确认。
6. A 股交易制度：当前框架还没有严格实现 T+1、涨跌停不可成交、最小买卖单位 100 股。
7. 组合约束：还没有行业暴露、单票权重上限、现金占用、杠杆、融资融券等约束。
8. 过拟合控制：分钟数据噪音很大，后续需要 walk-forward、purged split、样本外年份验证。

参考资料：

- PyTorch 官方安装：https://pytorch.org/get-started/locally/
- scikit-learn 时间序列切分：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Backtrader 滑点说明：https://www.backtrader.com/docu/slippage/slippage/
- Backtrader 手续费说明：https://www.backtrader.com/docu/commission-schemes/commission-schemes/

## 9. 特征工程说明

当前 197 个特征覆盖：

1. 收益与动量：`ret_*`, `logret_*`, `roc_*`
2. 波动率与极值：`ret_std_*`, `atr_ratio_*`, `range_*`, `dist_high_*`, `dist_low_*`
3. 均线与趋势：`sma_gap_*`, `ema_gap_*`, `ma_cross_*`
4. 价量关系：`volume_ratio_*`, `amount_ratio_*`, `vwap_gap_*`, `obv_z_*`
5. 技术指标：`rsi_*`, `cci_*`, `willr_*`, `mfi_*`, `adx_*`, `macd_*`, `boll_*`, `stoch_*`
6. 日内时间特征：`minute_sin`, `minute_cos`, `weekday_sin`, `weekday_cos`, `is_morning`

标签：

- `future_return_1`
- `future_return_5`
- `future_return_24`
- `Return_24`

`Return_24` 是为了兼容 `legacy_bangsheng_quant_strategies` 里的旧训练脚本命名。

## 10. 邦盛策略代码思路

`E:\quant\legacy_bangsheng_quant_strategies` 里的脚本主线是：

1. 读入 `fea_df2.pkl`。
2. 使用 `Return_24` 作为未来收益标签。
3. 按 `candle_begin_time` 做横截面排序，把每个时刻的收益转成 0-1 排名。
4. 部分脚本也把特征按横截面排序或分箱。
5. 使用 LightGBM、MLP、LSTM、pairwise loss 等模型学习横截面相对强弱。
6. 评估指标主要是每个时间截面的 Spearman 相关，即预测排序和未来收益排序的一致性。

我本次没有直接运行旧脚本，因为它们依赖旧文件名 `fea_df2.pkl` 和旧字段；我把它们的思路迁移到新脚本 `scripts\run_model_experiments.py`：

- 标签：`Return_24`
- 目标：横截面 rank 后的 `target_rank`
- 评估：Spearman IC、rank RMSE/MAE、多空 top/bottom 分组收益
- 模型：Linear SVM、LightGBM、LSTM

## 11. 模型实验结果

实验设置：

- 数据：特征 pkl 前 40 个文件
- 样本：300,000 行
- 训练/验证/测试：按时间 60% / 20% / 20%
- 特征：197 个
- LSTM：window=24，epochs=2，最多 60,000 条序列，GPU 训练

结果：

| 模型 | RMSE(rank) | MAE(rank) | Mean Spearman IC | 多空均值 | 多空波动 | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| Linear SVM | 0.288860 | 0.243727 | 0.142603 | 0.001196 | 0.008030 | 36.637136 |
| LightGBM | 0.282162 | 0.243449 | 0.147011 | 0.001633 | 0.007243 | 55.444541 |
| LSTM | 0.292619 | 0.246510 | 0.121508 | 0.001309 | 0.007528 | 42.752875 |

解释：

- LightGBM 目前最好，符合分钟横截面表格特征的常见经验。
- Linear SVM 能跑通，但线性表达能力有限。
- LSTM 已确认能用 GPU 训练，但这里只跑了小样本和 2 个 epoch，结果不能代表最终深度学习效果。
- `Sharpe-like` 基于重叠的 24 分钟未来收益，不能直接当成实盘年化 Sharpe，只适合横向比较模型。

## 12. 下一步建议

短期最有价值的改进：

1. 加入 A 股真实交易规则：T+1、涨跌停、100 股整数手、停牌过滤。
2. 明确复权方式：确认原始价格是否前复权，避免长期训练标签扭曲。
3. 加 benchmark：沪深 300、中证 500、中证 1000 或全市场等权。
4. 模型验证改成 walk-forward：例如 2023 训练、2024 验证、2025 测试、2026 样本外。
5. 对 `Return_24` 做 purging，避免重叠标签造成验证偏乐观。
6. 把 LightGBM 预测信号接入 `ModelSignalStrategy`，做真实成本下的模型回测。
7. 使用 Parquet 作为可选格式，pkl 快但不适合跨语言和列裁剪。
8. 做特征重要性、相关性过滤、缺失率过滤，减少冗余特征。
9. 对分钟数据增加市场状态特征：指数收益、行业收益、全市场成交额、涨跌家数。
10. 对 LSTM 增加更稳的训练流程：分股票构造序列、AMP 混合精度、early stopping、按年份样本外验证。

## 13. 推荐开发路径

如果你下一步想稳稳推进，可以按这个顺序：

1. 先读 `slices\feature_slice_2000.pkl`，理解字段。
2. 用 `model_metrics.csv` 里的 LightGBM 结果作为 baseline。
3. 把 LightGBM 模型接入回测，先做 top/bottom 组合。
4. 增加真实 A 股交易约束。
5. 再扩大训练数据，不要一开始就全市场全年份深度学习。
