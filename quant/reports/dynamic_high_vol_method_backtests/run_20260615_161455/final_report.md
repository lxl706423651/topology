# 动态高波动股票池 method1/method2 回测报告

## 实验边界

- 本报告不重新训练 XGBoost 模型，不重新选择 method1/method2 门控阈值。
- 股票池每天或每周动态变化，且每个交易日的股票池只使用该日之前的历史波动率数据确定。
- 每个动态股票池既作为可交易范围，也作为当日截面 rank 特征的计算截面。
- top-N：每次选取 `20` 只高波动股票。
- 最小历史窗口交易日数：`3`。
- 日度滚动窗口：过去 `7` 个自然日，不含当日。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`，只做外推诊断。
- 测试集：`datetime >= 2025-05-01`，用于最终比较。
- ST 剔除状态：`loaded_latest_cache:/data/lxl/seed-emulator/topology/quant/config/stock_name_cache/a_stock_names_20260614.csv`。

## 两种动态股票池

- `weekly_previous_week_high_vol_top20`：每个交易周使用上一周周一到周五内日内分钟收益波动率均值最高的 top20 股票，并在本周所有交易日使用同一股票池。
- `daily_rolling_7d_high_vol_top20`：每个交易日使用过去 7 个自然日内日内分钟收益波动率均值最高的 top20 股票，每天更新股票池。

## 重要说明

- 这两个股票池都是动态事前规则；与此前 `recent_week_high_vol_top300` 静态池不同，不使用测试末尾的一周固定覆盖整个测试集。
- 当前回测仍未完整建模涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。
- method2 CTA 仍是 method1 预测分的小权重趋势/突破包装，不应简单视为完全独立 alpha。

## 策略结果

| 股票池 | 方法 | 并集股票数 | 日均股票数 | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 信号期占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `daily_rolling_7d_high_vol_top20` | `method1_xgboost_long_only` | 52 | 20.00 | 2.190929 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 |
| `daily_rolling_7d_high_vol_top20` | `method2_cta_trend_breakout` | 52 | 20.00 | 1.809225 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 |
| `weekly_previous_week_high_vol_top20` | `method1_xgboost_long_only` | 52 | 20.00 | 0.721938 | 3.059216 | 0.006385 | -0.003700 | 2 | 0.018182 |
| `weekly_previous_week_high_vol_top20` | `method2_cta_trend_breakout` | 52 | 20.00 | 0.721938 | 3.059216 | 0.006385 | -0.003700 | 2 | 0.018182 |

## 特征缓存

- `weekly_previous_week_high_vol_top20`：rows=`25770`，cache=``，loaded_from_cache=`False`。
- `daily_rolling_7d_high_vol_top20`：rows=`25170`，cache=``，loaded_from_cache=`False`。

## 产物说明

- `strategy_summary.csv`：四组回测核心指标。
- 每个动态股票池目录下保存每日成员、每日摘要、收益曲线、交易事件、平仓复盘、持仓和指标。
