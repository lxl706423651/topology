# Method2 CTA 独立 lot 动态平仓策略回测结果

正式报告目录：

`reports/weekly_high_vol_lot_exit_backtests/run_20260702_123216/`

对应脚本：

`scripts/run_weekly_high_vol_lot_exit_backtests.py`

## 这次改了什么

- 股票池：每周使用上一周高波动 top300 股票。
- 入场：按 method2 的 `cta_trend_breakout_score` 排序，每个采样点最多选 top5，且必须超过 method2 入场门控阈值 `0.5529991269111634`。
- 仓位：每次开仓按当前总资产 `10%` 计算目标买入金额；现金不足时向下缩减，不能加杠杆。
- 加仓：同一股票如果后续交易日仍在 top5，且有现金，可以再次开仓。
- lot：每次开仓形成独立 lot，同一股票不同买入批次独立计算平仓条件。
- 平仓：不再因为股票掉出当前目标 top5 就卖出；只按动态 ATR 和指标规则卖出。
- 交易约束：A 股 long-only、不能做空、T+1、100 股整数手、佣金万八、卖出印花税 `0.0005`、过户费 `0.00001`、免 5。

## 训练、验证、测试边界

- 策略训练集：`2024-11-01 <= datetime < 2025-03-01`。
- 策略验证集：`2025-03-01 <= datetime < 2025-05-01`。
- 测试集：`datetime >= 2025-05-01`。
- 训练集先评估全部退出候选；每个 method 只把训练集表现靠前的 shortlist 放到验证集；验证集选中最终规则后，测试集只跑这一个规则。
- 本次没有根据测试集 Sharpe 回头改规则。

## 选中规则

验证集选中：

`atr_initial2_trail4_atr240`

中文含义：

- 入场 ATR 初始止损：买入价减 `2 * 入场时 ATR240`。
- ATR 移动止损：持仓后最高价减 `4 * 当前 ATR240`。
- 两个止损价同时存在时，使用更严格的那个。
- 每个 lot 单独计算自己的买入价、入场 ATR 和最高价。

## 测试集结果

- Sharpe：`5.720239`
- 年化收益：`5.111388`
- 年化波动：`0.325594`
- 最大回撤：`-0.080580`
- 总收益：`26.582429`
- 买入次数：`1449`
- 卖出次数：`1440`
- 已平仓 lot 数：`1440`
- 已平仓胜率：`0.596528`
- 平均持仓天数：`1.775694`
- 平均持仓股票数：`3.368774`
- 平均 open lot 数：`4.221524`
- 平均多头暴露：`0.418376`
- 最大多头暴露：`1.000000`
- short exposure：`0.0`

## 约束审计

- 负暴露行数：`0`
- 非 100 股整数手交易数：`0`
- 卖出日早于或等于买入日的 T+1 违规数：`0`
- 因“未进入当前目标组合”平仓次数：`0`
- 同一股票多 lot 已出现，例如 `SZ.000638` 有 `22` 个独立 lot。

## 输出文件

- `final_report.md`：本次总报告。
- `selected_strategy_summary.csv`：method1/method2 选中规则和测试指标。
- `train_candidate_metrics.csv`：训练集全部候选退出规则结果。
- `validation_candidate_metrics.csv`：训练集 shortlist 在验证集上的结果。
- `method2_cta_lot_exit/test_equity_curve.csv`：测试集权益曲线。
- `method2_cta_lot_exit/test_return_curve.png`：测试集收益曲线图，图中含 Sharpe、收益率、年化、波动、最大回撤、交易手数、胜率等指标。
- `method2_cta_lot_exit/test_trade_events.csv`：逐笔买卖记录。
- `method2_cta_lot_exit/test_round_trips.csv`：每个已平仓 lot 的买入、卖出、收益和触发原因。
- `method2_cta_lot_exit/test_open_positions.csv`：测试期末未平仓 lot。

## 与 method1 的关系

method2 仍是 method1 XGBoost alpha 的 CTA 包装版本。虽然本次 method2 测试 Sharpe 略高，但交易路径和 method1 很接近，不应直接把 method1 与 method2 当作完全独立策略加倍配仓。

## 调用方式

正式回测命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_high_vol_lot_exit_backtests.py
```

快速烟测命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python -u scripts/run_weekly_high_vol_lot_exit_backtests.py \
  --max-train-periods 30 \
  --max-validation-periods 20 \
  --max-test-periods 20 \
  --output-dir reports/weekly_high_vol_lot_exit_backtests_smoke \
  --train-shortlist-size 3
```

## 实盘前仍要补的内容

- 涨跌停无法成交、停牌、撤单、部分成交和成交额容量还没有完整模拟。
- 当前 ST 过滤不是严格历史点时 ST。
- 总收益和 Sharpe 很高，应先做未来样本 forward test 或 walk-forward，再决定实盘规模。
- 每次开仓是总资产 10%，但权益增长后绝对成交额会快速变大，实盘必须加单票成交额占比和最大委托金额限制。
