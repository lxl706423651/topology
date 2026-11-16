# 动态高波动股票池 method1/method2 回测报告

## 实验边界

- 本报告不重新训练 XGBoost 模型，不重新选择 method1/method2 门控阈值。
- 股票池每天或每周动态变化，且每个交易日的股票池只使用该日之前的历史波动率数据确定。
- 每个动态股票池既作为可交易范围，也作为当日截面 rank 特征的计算截面。
- top-N：每次选取 `300` 只高波动股票。
- 最小历史窗口交易日数：`3`。
- 日度滚动窗口：过去 `7` 个自然日，不含当日。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`，只做外推诊断。
- 测试集：`datetime >= 2025-05-01`，用于最终比较。
- ST 剔除状态：`loaded_latest_cache:/data/lxl/seed-emulator/topology/quant/config/stock_name_cache/a_stock_names_20260614.csv`。

## 两种动态股票池

- `weekly_previous_week_high_vol_top300`：每个交易周使用上一周周一到周五内日内分钟收益波动率均值最高的 top300 股票，并在本周所有交易日使用同一股票池。
- `daily_rolling_7d_high_vol_top300`：每个交易日使用过去 7 个自然日内日内分钟收益波动率均值最高的 top300 股票，每天更新股票池。

## 重要说明

- 这两个股票池都是动态事前规则；与此前 `recent_week_high_vol_top300` 静态池不同，不使用测试末尾的一周固定覆盖整个测试集。
- 当前回测仍未完整建模涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。
- method2 CTA 仍是 method1 预测分的小权重趋势/突破包装，不应简单视为完全独立 alpha。

## 策略结果

| 股票池 | 方法 | 并集股票数 | 日均股票数 | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 信号期占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `daily_rolling_7d_high_vol_top300` | `method1_xgboost_long_only` | 2439 | 300.00 | 2.215921 | 3.689603 | 3.179945 | -0.307858 | 141 | 0.134392 |
| `daily_rolling_7d_high_vol_top300` | `method2_cta_trend_breakout` | 2439 | 300.00 | 2.215919 | 3.689603 | 3.179945 | -0.307858 | 141 | 0.135036 |
| `weekly_previous_week_high_vol_top300` | `method1_xgboost_long_only` | 2315 | 300.00 | 1.440584 | 3.648573 | 2.842638 | -0.255614 | 147 | 0.123863 |
| `weekly_previous_week_high_vol_top300` | `method2_cta_trend_breakout` | 2315 | 300.00 | 1.440584 | 3.654623 | 2.846792 | -0.255614 | 147 | 0.124080 |

## 特征缓存

- `weekly_previous_week_high_vol_top300`：rows=`1045431`，cache=`/data/lxl/seed-emulator/topology/quant/reports/dynamic_high_vol_method_backtests/panel_cache/dynamic_high_vol_panel_2e134805e8fb3860.pkl`，loaded_from_cache=`False`。
- `daily_rolling_7d_high_vol_top300`：rows=`1051468`，cache=`/data/lxl/seed-emulator/topology/quant/reports/dynamic_high_vol_method_backtests/panel_cache/dynamic_high_vol_panel_d3e4bebb43133a7f.pkl`，loaded_from_cache=`False`。

## 产物说明

- `strategy_summary.csv`：四组回测核心指标。
- 每个动态股票池目录下保存每日成员、每日摘要、收益曲线、交易事件、平仓复盘、持仓和指标。
