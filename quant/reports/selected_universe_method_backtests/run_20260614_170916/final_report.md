# 三类精选股票池 method1/method2 对比报告

## 实验边界

- 本报告不重新训练 XGBoost 模型，不重新选择 method1/method2 门控阈值。
- 每个股票池既作为可交易股票范围，也作为本次截面 rank 特征的计算截面；method1/method2 的 top1、分数阈值、分差阈值保持原样。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`，只做外推诊断。
- 测试集：`datetime >= 2025-05-01`，用于比较不同股票池在同一固定规则下的表现。
- ST 剔除状态：`loaded_cache:/data/lxl/seed-emulator/topology/quant/config/stock_name_cache/a_stock_names_20260614.csv`。
- 所有股票池都限定在本地沪深主板非 ST 股票中，剔除北交所、科创板和创业板。

## 三种股票池

- `tech_theme_direct`：联网确认科技主线后，直接指定半导体、存储、AI、AI应用、电池、商业航天、机器人等主题股票；入选 `162` 只。
- `recent_week_high_vol_top300`：本地数据截至 2026-04-30 的最近 5 个交易日分钟收益波动率最高 top300；入选 `300` 只。
- `pretest_liquidity_momentum_top300`：测试集开始前 40 个交易日，按流动性、趋势动量和低波动综合得分 top300；入选 `300` 只。

## 科技主题资料依据

- 21世纪经济报道：2026-05-27 报道 AI 基建带动存储芯片需求，存储芯片概念股受到关注。 资料：<https://www.21jingji.com/article/20260527/herald/0f0697308627701c632d54307bba5599.html>
- 华尔街见闻：2026-06-04 报道 A 股半导体产业链、存储器、半导体材料、半导体设备等方向活跃。 资料：<https://wallstreetcn.com/articles/3773833>
- 证券时报：2026-01-06 报道商业航天、锂电池、AI应用是市场热门方向。 资料：<https://www.stcn.com/article/detail/3573069.html>

## 重要解释限制

- 科技主题池是当前资料和人工指定股票池，存在主题主观性和当前成分幸存者偏差。
- 最近一周高波动 top300 使用本地数据截至测试末尾附近的最近可用一周，并要求股票行情覆盖本地数据末日；该选择仍属于后验诊断，不能当作严格独立测试结论。
- 自选流动性趋势质量 top300 只使用测试集开始前数据选股，更接近可复现的事前股票池规则。
- 当前回测仍未完整建模涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。

## 策略结果

| 股票池 | 方法 | 股票数 | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 测试信号期占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pretest_liquidity_momentum_top300` | `method1_xgboost_long_only` | 300 | 5.342636 | 3.596191 | 3.450267 | -0.147131 | 144 | 0.058909 |
| `pretest_liquidity_momentum_top300` | `method2_cta_trend_breakout` | 300 | 5.342636 | 3.597386 | 3.453041 | -0.147131 | 145 | 0.059322 |
| `recent_week_high_vol_top300` | `method1_xgboost_long_only` | 300 | 3.155202 | 4.139625 | 11.150365 | -0.269970 | 239 | 0.105209 |
| `recent_week_high_vol_top300` | `method2_cta_trend_breakout` | 300 | 3.239634 | 4.232361 | 12.050728 | -0.269970 | 233 | 0.107069 |
| `tech_theme_direct` | `method1_xgboost_long_only` | 162 | 3.757855 | 3.431970 | 4.278671 | -0.199829 | 174 | 0.078338 |
| `tech_theme_direct` | `method2_cta_trend_breakout` | 162 | 3.757855 | 3.388650 | 4.170798 | -0.199829 | 175 | 0.078958 |

## 特征缓存

- `tech_theme_direct`：rows=`586785`，cache=`/data/lxl/seed-emulator/topology/quant/reports/selected_universe_method_backtests/panel_cache/all_a_method_panel_e9f7658603768c26.pkl`，loaded_from_cache=`True`。
- `recent_week_high_vol_top300`：rows=`1083003`，cache=`/data/lxl/seed-emulator/topology/quant/reports/selected_universe_method_backtests/panel_cache/all_a_method_panel_d06f8c5a54ed7110.pkl`，loaded_from_cache=`False`。
- `pretest_liquidity_momentum_top300`：rows=`1086535`，cache=`/data/lxl/seed-emulator/topology/quant/reports/selected_universe_method_backtests/panel_cache/all_a_method_panel_7d70dc480d638a70.pkl`，loaded_from_cache=`False`。

## 产物说明

- `universe_selection_audit.csv`：所有候选股票和选择指标。
- `universe_summary.csv`：三类股票池选择口径。
- `strategy_summary.csv`：六组回测核心指标。
- 每个股票池目录下保存 method1/method2 的收益曲线、交易事件、平仓复盘、持仓和指标。
