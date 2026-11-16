# 周度高波动 top300 持仓级平仓策略报告

## 实验边界

- 股票池固定为 `weekly_previous_week_high_vol_top300`：本周使用上一交易周高波动 top300。
- 不重新训练 method1 XGBoost 模型，不重新选择模型特征。
- 不再因为持仓股票掉出当前 topN 目标组合就卖出；买入后只由持仓级退出规则平仓。
- topN、退出规则只按验证集选择，测试集只用于最终验收。
- 单票买入目标仓位上限：`10.00%`。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`。
- 测试集：`datetime >= 2025-05-01`。
- 特征缓存：`/data/lxl/seed-emulator/topology/quant/reports/dynamic_high_vol_method_backtests/panel_cache/dynamic_high_vol_panel_2e134805e8fb3860.pkl`。

## 候选平仓规则

- 固定止损/止盈：收益达到预设亏损或盈利比例时退出。
- 移动止损：价格从持仓后最高价回撤到阈值时退出。
- ATR/Chandelier Exit：用最高价减去若干倍 ATR 作为多头移动止损。
- 最大持仓时间：持仓超过固定采样周期后退出。
- 模型分数衰减：当前持仓自身分数低于入场阈值减去缓冲时退出。

## 验证集候选结果前十

| 方法 | topN | 平仓规则 | 验证 Sharpe | 验证最大回撤 | 验证买入次数 |
|---|---:|---|---:|---:|---:|
| `method1_xgboost_exit_managed` | 3 | `time_2d_stop_6_take_12` | -7.976648 | -0.012613 | 5 |
| `method1_xgboost_exit_managed` | 3 | `trail_5_stop_5_take_12_2d` | -7.976648 | -0.012613 | 5 |
| `method2_cta_exit_managed` | 3 | `time_2d_stop_6_take_12` | -7.981241 | -0.012611 | 5 |
| `method2_cta_exit_managed` | 3 | `trail_5_stop_5_take_12_2d` | -7.981241 | -0.012611 | 5 |
| `method1_xgboost_exit_managed` | 3 | `trail_8_stop_6_take_15_3d` | -14.277942 | -0.020639 | 5 |
| `method2_cta_exit_managed` | 3 | `trail_8_stop_6_take_15_3d` | -14.284308 | -0.020639 | 5 |
| `method1_xgboost_exit_managed` | 2 | `score_decay_stop_6_take_12_2d` | -14.575500 | -0.019112 | 4 |
| `method1_xgboost_exit_managed` | 2 | `time_2d_stop_6_take_12` | -14.575500 | -0.019112 | 4 |
| `method1_xgboost_exit_managed` | 2 | `trail_5_stop_5_take_12_2d` | -14.575500 | -0.019112 | 4 |
| `method2_cta_exit_managed` | 2 | `score_decay_stop_6_take_12_2d` | -14.601263 | -0.019112 | 4 |

## 选中规则与测试结果

| 方法 | topN | 平仓规则 | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 已平仓胜率 | 平均持仓天数 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `method1_xgboost_exit_managed` | 3 | `time_2d_stop_6_take_12` | 12.496880 | 9.509291 | -0.000081 | 3 | 0.000000 | 0.000 |
| `method2_cta_exit_managed` | 3 | `time_2d_stop_6_take_12` | 12.496880 | 9.509291 | -0.000081 | 3 | 0.000000 | 0.000 |

## 重要限制

- 当前仍未完整模拟涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。
- 单票 10% 是买入时的目标仓位上限；价格上涨后可能短暂超过 10%，本脚本不做强制市值再平衡。
- 这些结果不能和旧 top1 满仓策略直接比较收益率；新策略按用户要求降低了单票集中度和总仓位。
- 测试集历史已经在本项目中被查看过，后续实盘前仍应使用未来新增数据或 walk-forward 验证。
