# 多因子 CTA 风格策略回测报告

## 研究边界

- 本报告包含两个策略：`cta_trend_breakout` 和 `cta_volatility_scaled`。
- 当前市场和数据口径是 A 股股票分钟数据，所以这是 CTA 风格 long-only 股票择时策略，不是期货多空 CTA。
- 两个策略都不做空、不加杠杆，`short_exposure=0`，并沿用 signal_lag=1、T+1、100 股整数手、万1成本。
- CTA 因子叠加强度：`0.0005`；主信号仍为已训练好的 156 因子 XGBoost alpha 预测。
- 验证集：`2024-11-01 <= datetime < 2025-05-01`；测试集：`datetime >= 2025-05-01`。

## 因子公式

### cta_trend_breakout

- 策略名：`cta_trend_breakout_alpha`
- 打分列：`cta_trend_breakout_score`
- 构造方法：XGBoost多因子alpha预测 + 小权重趋势斜率/均线差/通道突破/动量/ADX趋势质量/量价资金流叠加项
- 交易思路：借鉴时间序列动量和通道突破思想，只在模型alpha很强且个股趋势、突破、量价确认较好的截面中买入；A股现货限制下只做多，不做空。
- 验证集选中规则：`top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`
- 验证 Sharpe：2.898413

### cta_volatility_scaled

- 策略名：`cta_volatility_scaled_alpha`
- 打分列：`cta_volatility_scaled_score`
- 构造方法：XGBoost多因子alpha预测 + 小权重风险调整动量/低波动/通道突破/量价资金流叠加项
- 交易思路：借鉴 CTA 波动控制和风险平价思想，在模型alpha基础上偏向低下行波动、趋势质量更好且量价确认的股票，降低追涨噪声。
- 验证集选中规则：`top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`
- 验证 Sharpe：2.898413

## 最终测试集验收

### cta_trend_breakout

- 策略：`cta_trend_breakout_alpha`
- 规则：`top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`
- Sharpe：3.909777
- 年化收益：3.148437
- 年化波动率：0.382021
- 最大回撤：-0.127325
- 成本后总收益：2.918502
- 平均多头暴露：0.108982
- short_exposure：0.000000
- 已平仓胜率：0.654321
- 总交易手数：269666
- Sharpe 目标 `3` 是否达成：`True`

### cta_volatility_scaled

- 策略：`cta_volatility_scaled_alpha`
- 规则：`top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`
- Sharpe：3.909777
- 年化收益：3.148437
- 年化波动率：0.382021
- 最大回撤：-0.127325
- 成本后总收益：2.918502
- 平均多头暴露：0.108982
- short_exposure：0.000000
- 已平仓胜率：0.654321
- 总交易手数：269666
- Sharpe 目标 `3` 是否达成：`True`

## 结论

- 两个命名策略是否全部达到测试集 Sharpe > 3：`True`。
- 两个策略验证集是否选中同一交易规则：`True`。
- 去重后的可执行交易路径数量：`1`。
- 是否可视为两个独立策略同时成功：`False`。两条命名策略的 `test_equity_curve.csv` 和 `test_trade_events.csv` 完全一致，`test_holdings.csv`、`test_round_trips.csv` 只在预测分解释字段上不同。
- 本目录是 2026-05-27 20:46 对 `run_20260527_192923` 的同命令重跑复现；`test_strategy_metrics.csv` 与原目录 SHA256 完全一致。
- 由于本项目历史测试集已经被多次查看，本结果仍应作为当前数据口径下的研究回测；后续生产判断需要新增未来数据或预先声明的 walk-forward。

## 产物说明

- `资料依据.md`：CTA/趋势跟踪资料和本地因子映射。
- `数据切分审计.md`、`split_audit.csv`：验证集和测试集边界。
- `strategy_validation_summary.csv`：两个 CTA 策略在验证集上选中的规则和指标。
- `test_strategy_metrics.csv`：两个 CTA 策略的最终测试指标。
- `交易路径一致性审计.md`、`trade_path_audit.csv`：多个命名策略是否共享同一条可执行交易路径。
- `cta_trend_breakout/`、`cta_volatility_scaled/`：每个策略的权益曲线、收益图、持仓、逐笔交易、平仓复盘、期末持仓和元数据。
