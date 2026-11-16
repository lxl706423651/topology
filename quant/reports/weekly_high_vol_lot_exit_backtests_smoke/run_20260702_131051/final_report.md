# 周度高波动 top5 独立 lot 平仓回测报告

## 实验边界

- 股票池固定为每周用上一周高波动 top300。
- 每个采样点最多买入 top5，且每只股票分数必须超过 method 固定入场门控阈值。
- 每次开仓按当前总资产的 10% 计算目标买入金额；现金不足时按现金约束向下缩减。
- 同一股票后续交易日仍在 top5 且有现金时可以继续加仓；每次加仓形成独立 lot。
- 每个 lot 按自己的买入价、入场 ATR、持仓后最高价和技术指标独立平仓。
- 候选退出规则先在策略训练集筛选，再在验证集确认，测试集只做最终验收。
- 策略训练集：`2024-11-01 <= datetime < 2025-03-01`。
- 策略验证集：`2025-03-01 <= datetime < 2025-05-01`。
- 测试集：`datetime >= 2025-05-01`。
- 周度高波动特征缓存：`/data/lxl/seed-emulator/topology/quant/reports/dynamic_high_vol_method_backtests/panel_cache/dynamic_high_vol_panel_2e134805e8fb3860.pkl`。

## 候选退出规则

- Chandelier/ATR 吊灯止损：最高价减 ATR 倍数。
- 入场 ATR 初始止损 + ATR 移动止损。
- EMA 跌破退出 + Chandelier 保护。
- MACD 空头信号退出 + Chandelier 保护。
- Parabolic SAR 退出 + Chandelier 保护。
- RSI 过热后回落退出 + Chandelier 保护。
- 模型分数衰减退出 + Chandelier 保护。
- Chandelier 保护 + 最大持仓周期约束。

## 训练集候选前十

| 方法 | 退出规则 | 训练 Sharpe | 训练最大回撤 | 训练买入次数 |
|---|---|---:|---:|---:|
| `method1_xgboost_lot_exit` | `chandelier_atr120_3p0` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `chandelier_atr240_3p0` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `atr_initial2_trail3_atr120` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `atr_initial2_trail4_atr240` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `ema96_break_chandelier3` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `macd_bearish_chandelier3` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `psar_chandelier3` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `rsi120_reversal_chandelier3` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `score_decay_chandelier3` | -67.110965 | -0.000242 | 5 |
| `method1_xgboost_lot_exit` | `chandelier3_max3d` | -67.110965 | -0.000242 | 5 |

## 验证集 shortlist 结果

| 方法 | 退出规则 | 验证 Sharpe | 验证最大回撤 | 验证买入次数 |
|---|---|---:|---:|---:|
| `method1_xgboost_lot_exit` | `chandelier_atr120_3p0` | -35.496479 | -0.000081 | 1 |
| `method1_xgboost_lot_exit` | `chandelier_atr240_3p0` | -35.496479 | -0.000081 | 1 |
| `method2_cta_lot_exit` | `chandelier_atr120_3p0` | -35.496479 | -0.000081 | 1 |
| `method2_cta_lot_exit` | `chandelier_atr240_3p0` | -35.496479 | -0.000081 | 1 |

## 选中规则与测试结果

| 方法 | 退出规则 | 训练 Sharpe | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 胜率 | 平均持仓天数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `method1_xgboost_lot_exit` | `chandelier_atr120_3p0` | -67.110965 | -35.496479 | 30.610811 | 3.222543 | 0.000000 | 2 | 0.000000 | 0.000 |
| `method2_cta_lot_exit` | `chandelier_atr120_3p0` | -67.110965 | -35.496479 | 30.610811 | 3.222543 | 0.000000 | 2 | 0.000000 | 0.000 |

## 重要限制

- 当前仍未完整模拟涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。
- 买入金额按总资产 10% 计算，但现金不足时不能加杠杆，只能减少或跳过买入。
- 本项目历史测试集已经被查看过，新的高 Sharpe 仍需要未来样本或 walk-forward 验证。
