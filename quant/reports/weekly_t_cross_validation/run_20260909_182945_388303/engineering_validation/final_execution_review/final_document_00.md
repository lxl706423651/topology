# Quant Backtest Developer Guide

## AI/新人先读入口

本仓库包含两条线：一个轻量通用回测框架，以及当前主线 A 股 XGBoost long-only 分钟线研究。新接手的人或 AI 不要只读本文件就开始改代码，建议先按下面顺序读取：

1. `AI_QUICKSTART.md`：项目背景、当前代表性结果、复现命令、AI 接手规则和注意事项。
2. `1method/README.md`：当前 XGBoost long-only 策略方法、交易规则、文件用途和完整流程。
3. `reports/README.md`：哪些报告已经进 Git，哪些大数据/大产物被排除。
4. `reports/xgboost_long_only_backtest/run_20260526_202355/final_report.md` 和 `model_metadata.json`：当前代表性严格 T+1 回测结果。
5. `codex_worklog.md`：历史实验和验证记录。
6. [三板块XGBoost CTA新研究](6method_xgb_cta/README.md)：755只本地子集完成严格向前重训及状态修复后的独立审计，连续净收益-2.59%、夏普-0.126，未达标。创业板/科创板各仅3只，完整市场覆盖仍不足，不能直接实盘。

当前核心实验是 A 股 long-only：257 只稳定沪深 300 股票、156 个训练期筛选特征、XGBoost 模型、验证集选规则、测试期从 `2025-05-01` 开始，约束包括 signal_lag=1、T+1、100 股整数手、不做空、不加杠杆、初始资金 100 万和万1交易成本。完整上下文见 `AI_QUICKSTART.md`。

本项目已经把原来散落在 notebook 里的回测逻辑整理成一套 Python 回测框架。开发者后续主要做三件事：

1. 准备行情数据 CSV。
2. 在 `strategies/` 里写策略，或在 `run_strategy.py` 里替换策略入口。
3. 运行 `python run_strategy.py --data 数据路径 --output 输出路径` 查看指标和报告。

## 当前代码框架

```text
E:\quant
├─ run_strategy.py                  # 主运行入口，开发者最常改这里
├─ README.md                        # 当前开发者说明
├─ README_backtest.md               # 回测框架快速使用说明
├─ requirements.txt                 # 必要依赖
├─ requirements-optional.txt        # 可选依赖，如 TA-Lib、torch、tensorflow
├─ quant_backtest\                  # 回测框架核心包
│  ├─ __init__.py                   # 对外导出核心类
│  ├─ data.py                       # 数据读取、字段标准化、K线重采样
│  ├─ engine.py                     # 回测引擎，信号转持仓、收益、交易、权益曲线
│  ├─ indicators.py                 # 技术指标，BOLL/RSI/MACD/ATR 等
│  ├─ metrics.py                    # 绩效指标，夏普、最大回撤、Calmar 等
│  ├─ modeling.py                   # 模型加载，兼容 joblib/pickle/torch/keras
│  ├─ reporting.py                  # 报告保存、指标打印、权益图输出
│  └─ strategy.py                   # 策略接口，普通策略/旧函数策略/模型策略
├─ strategies\
│  ├─ __init__.py                   # 策略包标识
│  └─ example_bollinger_strategy.py # 示例 BOLL 突破策略
└─ reports\                         # 回测输出目录
```

## 每个 Python 文件是干嘛的

### `run_strategy.py`

回测主入口。你平时最常改这个文件。

里面有三个关键部分：

- `build_strategy()`：返回要运行的策略。你要换策略，优先改这里。
- `demo_data()`：没有传入外部数据时生成一份示例数据，用于自检框架。
- `main()`：读取参数、加载数据、创建回测配置、运行引擎、保存报告。

常用运行方式：

```powershell
python run_strategy.py
python run_strategy.py --data data\IF_15m.csv --output reports\if_boll
```

### `quant_backtest/data.py`

负责数据输入和格式统一。

核心函数：

- `load_csv(path)`：读取 CSV 并标准化字段。
- `standardize_ohlcv(data)`：把不同来源的数据统一成框架需要的列。
- `resample_ohlcv(data, rule="15min")`：按品种重采样 K 线。

框架最终需要的数据列是：

```text
datetime, symbol, open, high, low, close, volume
```

其中 `volume` 可以没有，框架会自动填 0。

### `quant_backtest/engine.py`

回测核心引擎。

核心类：

- `BacktestConfig`：回测参数，包括初始资金、手续费、滑点、合约乘数、收益模式等。
- `BacktestEngine`：执行回测。
- `BacktestResult`：保存回测结果，包括权益曲线、逐 K 线结果、逐笔交易、绩效指标。

它主要做这些事：

- 调用策略生成信号。
- 把 `open_signal/close_signal` 或 `target_position` 转成持仓。
- 默认 `signal_lag=1`，避免用当前 K 线信号吃当前 K 线收益。
- 计算手续费、滑点、收益、权益曲线。
- 提取逐笔交易。
- 汇总多品种组合收益。

### `quant_backtest/strategy.py`

策略接口和适配层。

核心类：

- `BaseStrategy`：所有新策略建议继承它，实现 `generate_signals(data)`。
- `FunctionSignalStrategy`：兼容旧 notebook 里的 `cal_feature(data, paras)`。
- `ModelSignalStrategy`：兼容训练好的模型，支持 `predict()` 或 `predict_proba()`。

新策略推荐返回：

```python
return pd.DataFrame({"target_position": position}, index=data.index)
```

其中 `position` 通常是：

```text
1   做多
0   空仓
-1  做空
```

旧 notebook 策略也可以继续返回：

```python
return open_signal, close_signal
```

### `quant_backtest/indicators.py`

技术指标模块。

当前提供：

- `bollinger_bands()`：布林带
- `rsi()`：RSI
- `macd()`：MACD
- `atr()`：ATR
- `sma()`：简单均线
- `ema()`：指数均线
- `zscore()`：滚动标准分

TA-Lib 是可选依赖。有 `talib` 时部分指标会优先用 TA-Lib；没有时自动走 pandas/numpy 实现，所以当前环境不装 TA-Lib 也能跑。

### `quant_backtest/metrics.py`

绩效指标模块。

当前输出指标包括但不限于：

- 总收益 `total_return`
- 年化收益 `annual_return`
- 年化波动率 `annual_volatility`
- 夏普 `sharpe`
- Sortino `sortino`
- 最大回撤 `max_drawdown`
- 最大回撤区间 `max_drawdown_start/end`
- Calmar `calmar`
- 单根 K 线胜率 `win_bar_rate`
- VaR/CVaR
- 偏度、峰度
- 交易次数、交易胜率、平均单笔收益、最大盈利/亏损、Profit Factor、平均持仓周期

### `quant_backtest/modeling.py`

模型加载模块，用于把训练好的模型接入回测。

支持：

- `.joblib`
- `.pkl`
- `.pickle`
- `.pt/.pth`，需要本机安装 `torch`
- `.keras/.h5`，需要本机安装 `tensorflow`

模型对象需要提供 `predict()` 或 `predict_proba()`。

### `quant_backtest/reporting.py`

报告输出模块。

核心函数：

- `print_metrics(metrics)`：在终端打印核心指标。
- `save_report(result, output_dir)`：保存全部回测结果。
- `plot_equity(equity, path)`：保存权益曲线图。

输出文件一般包括：

```text
metrics.csv
equity_curve.csv
bars_with_signals.csv
trades.csv
equity_curve.png
```

### `strategies/example_bollinger_strategy.py`

示例策略文件。当前是一个 BOLL 突破策略，用来演示新策略应该怎么写。

它继承 `BaseStrategy`，实现：

```python
generate_signals(self, data)
```

返回：

```text
open_signal
close_signal
boll_upper
boll_middle
boll_lower
```

你可以复制这个文件，改成自己的策略文件。

## 数据放在哪里

建议在项目根目录新建一个 `data/` 文件夹，例如：

```text
E:\quant\data\
├─ IF_15m.csv
├─ rb_15m.csv
└─ stock_daily.csv
```

当前我没有把 `data/` 强制写死在代码里。运行时通过 `--data` 指定即可：

```powershell
python run_strategy.py --data data\IF_15m.csv --output reports\IF_15m_boll
```

如果不传 `--data`，框架会使用 `run_strategy.py` 里的 `demo_data()` 生成模拟数据，只用于测试框架是否能跑通，不代表真实策略表现。

## CSV 数据格式要求

最推荐格式：

```csv
datetime,symbol,open,high,low,close,volume
2024-01-02 09:30:00,IF,3500,3510,3498,3506,1000
2024-01-02 09:45:00,IF,3506,3518,3502,3512,1200
```

也兼容旧 notebook 常见列名：

- `underlying_symbol` 会被识别为 `symbol`
- `date/time/trade_time/timestamp` 会被识别为 `datetime`
- `vol` 会被识别为 `volume`

最低要求是必须有：

```text
datetime, open, high, low, close
```

如果没有 `symbol`，框架会自动设为 `SINGLE`。

## 如何接入自己的策略

推荐新建一个文件，例如：

```text
strategies\my_strategy.py
```

示例：

```python
import pandas as pd
from quant_backtest.strategy import BaseStrategy
from quant_backtest.indicators import sma


class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ma20 = sma(data["close"], 20)
        position = (data["close"] > ma20).astype(float)
        return pd.DataFrame({"target_position": position}, index=data.index)
```

然后在 `run_strategy.py` 里改：

```python
from strategies.my_strategy import MyStrategy


def build_strategy():
    return MyStrategy()
```

## 如何接入旧 notebook 的 `cal_feature`

如果你有旧代码：

```python
def cal_feature(data, paras):
    open_signal = ...
    close_signal = ...
    return open_signal, close_signal
```

可以这样接入：

```python
from quant_backtest.strategy import FunctionSignalStrategy


def build_strategy():
    return FunctionSignalStrategy(cal_feature, paras={"n": 20})
```

## 如何接入训练好的模型

示例：

```python
import pandas as pd
from quant_backtest.strategy import ModelSignalStrategy


def feature_fn(data):
    return pd.DataFrame({
        "ret_1": data["close"].pct_change(),
        "ma_gap": data["close"] / data["close"].rolling(20).mean() - 1,
    }, index=data.index)


def build_strategy():
    return ModelSignalStrategy(
        model_path=r"models\my_model.joblib",
        feature_fn=feature_fn,
    )
```

模型输出建议是 `-1/0/1`，分别表示做空、空仓、做多。

## TA-Lib 说明

当前框架不强依赖 TA-Lib。为了不破坏现有 Python 环境，建议不要直接在当前 Anaconda 环境里强装。

如果确实要装，建议新建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install TA-Lib
```

如果 TA-Lib 安装失败，框架仍然可以用内置指标实现继续运行。

## 回测输出在哪里

默认输出到：

```text
reports\latest\
```

可以通过 `--output` 修改：

```powershell
python run_strategy.py --data data\IF_15m.csv --output reports\IF_15m_my_strategy
```

每次输出包含：

- `metrics.csv`：绩效指标
- `equity_curve.csv`：权益曲线
- `bars_with_signals.csv`：K 线、信号、持仓、收益明细
- `trades.csv`：逐笔交易
- `equity_curve.png`：权益曲线图片

## 固定底仓做 T 与年度交叉验证

[4method_inventory_t/README.md](4method_inventory_t/README.md) 说明新增的固定底仓日内做 T 策略：沿用 3method 入场思路，保留底仓，按每日昨日可卖股份完成买卖配对，并与相同底仓买入持有比较净增量收益。入口为 `scripts/run_inventory_t_cross_validation.py`，配置为 `config/inventory_t_cross_validation.yaml`。年度留出覆盖 2023—2026；较早年度采用含未来年份训练的回顾式 LOYO，2026 数据仅截至 04-30。

## 周度前100与学习型做 T

当前主动回测入口为 `scripts/run_weekly_t_active.py`，默认配置 `config/weekly_t_cross_validation_v9.yaml`。按用户要求，样本数、置信区间、集中度和内部收益必须为正不再控制是否启用；仅用内部实际有T且执行合格的候选选择原核心权重。四折均开启并实际成交，共34组、扣费净增量合计276.31元；2025仍有4天股数偏差，不能称为每天都恢复成功。见[主动做T验证说明](5method_weekly_t/主动做T验证说明.md)及[本轮报告](reports/weekly_t_cross_validation/run_20260909_124918_833635/final_report.md)。原费用、T+1及实际成交约束保留，未用外层收益重新选参。

此前残差目标实验见[市场残差做T验证说明](5method_weekly_t/市场残差做T验证说明.md)和[v8报告](reports/weekly_t_cross_validation/run_20260908_235818_404243/final_report.md)。v8 保持周选股及费用规则，仅改变 T 训练目标；四折内部均无合格候选，全部关闭 T，额外净收益为 0，未达盈利目标。完整独立模型与执行审计通过，验证的是实现与账目；历史复测和周底仓退出风险仍需保留。

已知交易条件匹配实验见[执行条件匹配做T验证说明](5method_weekly_t/执行条件匹配做T验证说明.md)，入口为 `scripts/run_weekly_t_train_domain.py`，配置为 `config/weekly_t_cross_validation_v7.yaml`。v7按已知静态股数和原恢复截止时间筛选T训练样本，四折正式验证均关闭T、额外净收益为0；实际账户仍按原成交与库存规则执行。

选股队列实验见[选股队列做T验证说明](5method_weekly_t/选股队列做T验证说明.md)，入口为 `scripts/run_weekly_t_cohort.py`，配置为 `config/weekly_t_cross_validation_v6.yaml`。v6在核心年度交叉拟合产生的周前100队列上训练做T，并接入已知周预测；沿用v5费用和候选门槛，四折正式验证均关闭T、额外净收益为0。

后续短周期做T实验的固定协议、配置和运行命令见[短周期做T验证说明](5method_weekly_t/短周期做T验证说明.md)，入口为 `scripts/run_weekly_t_short_horizon.py`，配置为 `config/weekly_t_cross_validation_v5.yaml`。该实验冻结v4周度选股，再比较提前恢复与持有底仓的净差额。

[5method_weekly_t/README.md](5method_weekly_t/README.md) 说明周度双模型选股：上一周全部分钟聚合特征预测下一周收益和日均振幅，预测排名按70%/30%组合取前100，周初开盘买入、周末收盘退出。周中另训练做T回报模型，与相同股数的周度持有基准比较扣费净增量。入口为 `scripts/run_weekly_t_cross_validation.py`，当前配置为 `config/weekly_t_cross_validation_v4.yaml`，旧版配置与报告保留；选股、做T模型及门槛均在各年度留出折中独立训练。当前实现包含真实尾盘分钟分批恢复和过去收盘流动性门槛；v4进一步按实际滚动历史来源隔离年度训练样本，v2/v3旧协议存在滚动历史跨留出年问题。各版历史已被查看，修正版仍是研究复测，不能称为全新未见测试。原固定底仓实验保留。


## 做 T 频率改进（v10）

新增频率研究入口 `scripts/run_weekly_t_frequency.py`，配置 `config/weekly_t_cross_validation_v10.yaml`，见[做T频率改进说明](5method_weekly_t/做T频率改进说明.md)。[正式四折](reports/weekly_t_cross_validation/run_20260909_135906_004933/final_report.md)完成1760组，是v9的34组的51.76倍；平均3.911组/可计划日，或2.262组/全部估值日。频率提高的同时，毛价差352元不足以覆盖滑点与费用10643.38元，净增量 **−10291.38元**，未实现盈利。

v10的T库存/恢复/pending/T+1/底仓计划五项失败均为0，原共同基准逐日相同；周退出仍有171个失败年度日和期末遗留持仓。旧v9结果与代码保留；完整方法、覆盖分母及真实费用见频率说明；[v10独立执行审计](reports/weekly_t_cross_validation/run_20260909_135906_004933/independent_execution_audit.md)已完成通过，不能把频率目标达成当作收益目标达成。

## 做 T 择时能力研究（v11）

新增[做T择时能力研究（v11）](5method_weekly_t/做T择时能力改进说明.md)，入口 `scripts/run_weekly_t_timing.py`、配置 `config/weekly_t_cross_validation_v11.yaml`。结合一手研究，新训练原始价差与扣费盈利概率模型，并按当前状态选择反转/趋势规则及20/40/60分钟恢复周期；四折内部先冻结，再统一外评。[完整回测](reports/weekly_t_cross_validation/run_20260909_182945_388303/final_report.md)为86组、净增量−573.72元；每组平均亏6.67元，比v10的5.85元更差，**择时改善未通过验证**。[质量诊断](reports/weekly_t_cross_validation/run_20260909_182945_388303/timing_quality_diagnosis/report.md)显示扣费前毛差也为−131元，总亏损减少主要来自交易数减少。四年均开启且有成交，T库存恢复与T+1均无失败，原底仓及费用合同保留；2026仅至4月30日，所有结果仍是已被多次研究的历史复测。

v11的[独立模型审计](reports/weekly_t_cross_validation/run_20260909_182945_388303/independent_timing_model_audit.md)与[独立执行审计](reports/weekly_t_cross_validation/run_20260909_182945_388303/independent_execution_audit.md)均已完成通过，覆盖24组模型、100个内部候选及四年原始行情核账。审计确认实现与账目，未验证盈利能力改善。
