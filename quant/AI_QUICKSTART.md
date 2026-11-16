# AI and Human Quickstart

本文档把本项目的 Codex/quant skill 经验、项目背景、当前关键实验和继续工作的注意事项固化到仓库里。目标是让人或 AI 读取仓库后，先看这一页就能知道项目在做什么、当前可靠结论是什么、下一步应该怎么做，以及哪些坑不能踩。

## 先读顺序

1. `AI_QUICKSTART.md`：项目总览、当前状态、操作边界和复现命令。
2. `1method/README.md`：当前 XGBoost long-only 策略方法、交易规则、文件用途和完整流程。
3. `reports/README.md`：Git 中保留哪些轻量报告、哪些大文件被排除。
4. `reports/xgboost_long_only_backtest/run_20260526_202355/final_report.md`：当前代表性严格 T+1 回测报告。
5. `reports/xgboost_long_only_backtest/run_20260526_202355/model_metadata.json`：机器可读配置、模型参数、规则阈值和关键指标。
6. `codex_worklog.md`：历史任务、实验、验证命令和注意事项。

## 项目一句话

这是一个 A 股分钟线量化研究项目：从本地沪深 300 分钟行情中筛稳定股票池，构造无未来函数因子，做训练期因子审计，用 XGBoost 预测同截面未来收益 rank，再用验证集选择 long-only 组合规则，最后在 2025-05-01 之后做测试集回测。

当前主线不是通用 demo 回测，而是：

- 稳定沪深 300 股票池。
- 严格训练期因子审计。
- XGBoost long-only 模型。
- A 股约束：不做空、不加杠杆、T+1、100 股整数手、100 万初始资金、万1交易成本、signal_lag=1。

## 当前代表性结果

当前最重要的正式结果目录：

`reports/xgboost_long_only_backtest/run_20260526_202355/`

核心结论：

- 规则：`top_n_1p0_hold_1_exposure_1p0_confq_0p9_spreadq_0p7`
- 特征数：156 个 `cs_rank_*` 特征
- 股票池：257 只稳定沪深 300 股票
- 训练/测试切分：测试集从 `2025-05-01` 开始
- 内部验证集：从 `2024-11-01` 开始，仍在训练期内
- 测试 Sharpe：`3.706244`
- 年化收益：`2.874256`
- 年化波动：`0.384842`
- 最大回撤：`-0.127287`
- 成本后总收益：`2.669561`
- short exposure：`0`
- lot size：`100`

必须同时记住：项目历史上曾围绕测试集目标做过探索，因此当前测试集不能被表述成完全未触碰的 pristine holdout。后续更严谨的结论应保留新增未来数据，或做 walk-forward。

## 关键目录和文件

- `quant_backtest/factor_library.py`：核心因子库、未来收益标签、未来函数检查辅助逻辑。
- `scripts/analyze_stable_hs300_factor_correlation.py`：稳定股票池和因子相关性分析。
- `scripts/audit_factor_feature_library.py`：严格训练期因子审计和特征筛选。
- `scripts/run_xgboost_long_only_backtest.py`：主模型训练、验证选规则、T+1/整数手回测和报告输出。
- `reports/stable_hs300_factor_correlation/run_20260518_223759/`：当前稳定股票池和相关性摘要。
- `reports/factor_feature_audit/run_20260525_161905/`：当前模型特征审计结果。
- `reports/xgboost_long_only_backtest/run_20260526_202355/`：当前代表性 XGBoost long-only 回测。
- `reports/feature_panel_cache/`：本地大体积 pkl 特征面板缓存，被 Git 忽略。
- `ashare_minute_data_2023_2026_04/`：本地原始和处理后行情数据，被 Git 忽略；仓库克隆后不会自带这些数据。

## 标准复现流程

优先使用 conda 环境：

```bash
/home/lxl/anaconda3/envs/quant/bin/python
```

1. 稳定股票池和因子相关性：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/analyze_stable_hs300_factor_correlation.py --data-root ashare_minute_data_2023_2026_04/extracted --universe-file config/hs300_current_constituents.csv --horizon 24 --eval-step 24 --high-corr 0.80 --workers 4
```

2. 严格训练期因子审计：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/audit_factor_feature_library.py --workers 4 --train-start 2023-01-03 --train-end 2025-04-30 --min-assets 20 --min-rank-ic-periods 120
```

3. 构建或刷新特征面板缓存：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgboost_long_only_backtest.py --feature-manifest reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv --workers 4 --build-feature-panel-only --rebuild-feature-panel
```

4. 运行当前代表性 XGBoost long-only 回测：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_xgboost_long_only_backtest.py --feature-manifest reports/factor_feature_audit/run_20260525_161905/model_feature_manifest.csv --workers 4 --split-date 2025-05-01 --validation-start 2024-11-01 --commission-rate 0.0001 --initial-cash 1000000 --top-n-candidates 1,2,3 --top-quantile-candidates 0.02 --hold-steps-candidates 1 --target-exposure-candidates 1.0 --confidence-quantile-candidates none,0.9 --spread-quantile-candidates none,0.7 --min-assets 30 --min-holding-count 1 --lot-size 100 --sharpe-target 2
```

## AI 接手时的操作规则

- 先读 `AI_QUICKSTART.md`、`1method/README.md`、`reports/README.md`、最新 `final_report.md` 和 `model_metadata.json`。
- 修改策略或模型前，先看 `codex_worklog.md`，避免重复踩历史问题。
- 不要默认重建完整特征面板；完整缓存约 1.4G，先确认是否能复用 `reports/feature_panel_cache/feature_panel_7226a2c8f3dd389a.pkl`。
- 不要用测试集 Sharpe 反向调特征、模型参数或交易规则。规则选择必须来自训练期内部验证。
- 不要把 `run_20260525_190654` 当作严格结果；它历史 Sharpe 高，但 omitted T+1 且带测试目标探索色彩。
- 报告里必须明确未建模约束：涨跌停无法成交、ST/退市、最低佣金、冲击成本、点时历史指数成分。
- A 股股票策略默认 long-only，不做空，不加杠杆，目标仓位必须在 `(0, 1]`。
- 保持 `signal_lag=1` 或等价执行延迟；如果改成同 bar 执行，必须在报告中显式标注为不现实假设。
- 新增因子必须只使用当前或历史已完成 bar；rolling、expanding、lag、rank、resample 不能读未来行。
- 任何 scaler、imputer、feature selector、model preprocessing 只能在训练集 fit，再用于验证和测试。
- 截面特征必须按 `datetime` 分组；单股票 rolling 必须按 `symbol` 分组。

## 验证清单

Python 修改后至少运行：

```bash
python3 -m py_compile <changed_python_files>
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile <changed_python_files>
```

回测或策略修改后优先跑小样本 smoke test，再跑全量。涉及 T+1 或整数手时检查：

- `test_round_trips.csv` 中卖出日期不能早于或等于买入日期。
- `test_trade_events.csv` 中股数必须是 100 的整数倍。
- `test_metrics.csv` 中 `short_exposure` 必须为 0。
- `model_metadata.json` 必须记录数据切分、成本、lot size、所选规则、缓存使用情况和随机种子。

## Git 和数据策略

Git 中保留：

- 源码、配置、README、方法文档。
- 轻量报告摘要、关键指标、特征清单、模型 metadata、当前代表性图和必要交易摘要。

Git 中排除：

- 原始行情、解压 CSV、处理后 pkl、切片数据。
- `reports/feature_panel_cache/*.pkl`。
- 大体积预测明细，例如 `test_predictions.csv`、`validation_predictions.csv`、`train_predictions_sample.csv`。
- `*.joblib`、`*.pt` 等二进制模型缓存。

如果未来需要让别人完整复现，必须单独提供行情数据或说明数据获取方式；当前 Git 仓库本身不包含 116G 级别的本地行情数据。

## 后续更严格研究方向

- 保留 2026-05 之后新增行情作为完全未触碰 forward test。
- 做 walk-forward，每个窗口只用过去训练和验证，再评估下一个未来窗口。
- 加入涨跌停、停牌、ST/退市、最低佣金、冲击成本和成交容量约束。
- 降低单票集中风险，例如设置最低持仓数或行业/市值分散约束。
- 如果继续调参，必须把调参依据限制在训练期内部验证或新建 walk-forward 验证框架内。
