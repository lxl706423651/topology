# Quant Backtest Framework

这是从当前 notebook 研究代码中抽出的轻量回测框架。核心目标是：你以后主要改
`run_strategy.py` 里的 `build_strategy()` 或策略类，就能正常跑回测并输出关键指标。

## 快速运行

```powershell
python run_strategy.py
python run_strategy.py --data path\to\ohlcv.csv --output reports\my_strategy
```

输出文件：

- `metrics.csv`: 夏普、年化收益、年化波动率、最大回撤、Calmar、胜率、Profit Factor 等
- `equity_curve.csv`: 权益曲线
- `bars_with_signals.csv`: 每根 K 线的信号、持仓、收益
- `trades.csv`: 逐笔交易
- `equity_curve.png`: 权益图

## 数据格式

CSV 至少需要时间和 OHLC 四列。框架会自动兼容：

- 时间列：`datetime/date/time/trade_time/timestamp`
- 品种列：`symbol/underlying_symbol/code/ticker/instrument`
- 价格列：`open/high/low/close`
- 成交量列：`volume/vol`，没有也可以，会填 0

## 写普通策略

推荐继承 `BaseStrategy` 并返回 `target_position`，值为 `-1/0/1`：

```python
class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def generate_signals(self, data):
        position = ...
        return pd.DataFrame({"target_position": position}, index=data.index)
```

也兼容旧 notebook 的风格，返回 `open_signal` 和 `close_signal`：

```python
def cal_feature(data, paras):
    open_signal = ...
    close_signal = ...
    return open_signal, close_signal

strategy = FunctionSignalStrategy(cal_feature, paras={"n": 20})
```

## 接入训练好的模型

`ModelSignalStrategy` 支持 `.joblib/.pkl/.pickle`，如果本机安装了对应库，也支持
`.pt/.pth` 和 `.keras/.h5`。模型需要提供 `predict()` 或 `predict_proba()`。

```python
from quant_backtest.strategy import ModelSignalStrategy

def feature_fn(data):
    return pd.DataFrame({
        "ret_1": data["close"].pct_change(),
        "ma_gap": data["close"] / data["close"].rolling(20).mean() - 1,
    })

strategy = ModelSignalStrategy(
    model_path="models/my_model.joblib",
    feature_fn=feature_fn,
)
```

## TA-Lib

当前环境未检测到 `talib`。框架内置了 pandas/numpy 版本的 Bollinger、RSI、MACD、ATR，
所以不安装 TA-Lib 也能跑。若你要在隔离环境安装最新版，建议：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install TA-Lib
```

这样不会改动当前 Anaconda 环境。

## 已补强的不足

- 旧代码依赖缺失的 `api.*` 模块，新框架已移除这些隐式依赖。
- 旧代码主要散落在 notebook，新框架改为可复用 `.py` 模块。
- TA-Lib 变成可选依赖，不再阻塞运行。
- 默认加入 `signal_lag=1`，避免用本根 K 线信号直接吃本根收益的前视偏差。
- 加入手续费、滑点、逐笔交易、权益曲线、指标输出和多品种基础支持。
- 加入模型策略适配层，方便接入 sklearn/joblib/pickle/torch/keras 打包模型。
