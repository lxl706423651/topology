# Reports Tracking Policy

本目录只把能帮助理解、审计和复现实验的轻量报告文件纳入 Git，不提交原始数据、特征缓存、预测明细和二进制模型缓存。

## 已纳入的关键记录

- `xgboost_long_only_backtest/run_*/final_report.md`：每次 XGBoost long-only 回测的中文总结报告。
- `xgboost_long_only_backtest/run_*/model_metadata.json`：训练窗口、验证窗口、测试窗口、参数、样本规模、成本假设和所选交易规则。
- `xgboost_long_only_backtest/run_*/used_feature_manifest.csv`、`feature_list.txt`、`feature_importance.csv`：模型使用的特征清单和重要性。
- `xgboost_long_only_backtest/run_*/test_metrics.csv`、`validation_rule_metrics.csv`、`test_yearly_returns.csv`、`test_monthly_returns.csv`：验证选规则和测试期核心指标。
- `xgboost_long_only_backtest/run_20260526_202355/`：当前正式 T+1 代表性结果，额外保留收益曲线、持仓、交易事件、闭环交易、未平仓和 `xgboost_model.json`。
- `factor_feature_audit/run_20260525_161905/`：当前模型特征审计结果、因子审计明细和无未来函数检查。
- `stable_hs300_factor_correlation/run_20260518_223759/`：稳定沪深300股票池、样本稳定性和因子相关性摘要。

## 仍然排除的内容

- `ashare_minute_data_2023_2026_04/` 下的原始行情、解压 CSV、处理后 pkl 和切片数据。
- `reports/feature_panel_cache/*.pkl` 等特征面板缓存。
- `test_predictions.csv`、`validation_predictions.csv`、`train_predictions_sample.csv` 等大体积预测明细。
- `xgboost_model.joblib`、`*.joblib`、`*.pt` 等二进制模型文件。

复现实验时，优先从最新正式报告 `reports/xgboost_long_only_backtest/run_20260526_202355/model_metadata.json` 和 `final_report.md` 读取配置，再使用 `scripts/run_xgboost_long_only_backtest.py` 按 README 中的命令重跑。
