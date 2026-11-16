# 研究来源与改编（2026-09-10）

## 特征

Microsoft Qlib官方实现：`https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py`。
参考Alpha360的历史OHLC/VWAP归一化价量序列及Alpha158多滚动窗口的组织方式。
本地不是完整复刻：采用20期日线序列及5/10/20/30/60窗口；全部只连接上一已完成日，
同时复用274项分钟原因子和已有池内截面排名。日线volume单位为股，VWAP=amount/volume。
不照搬Qlib标签或交易引擎，不把其基准业绩等同于本项目收益。

## 预测目标

XGBoost官方参数：`https://xgboost.readthedocs.io/en/stable/parameter.html`。
比较reg:squarederror（净收益回归）、binary:logistic（净盈利概率）和
reg:quantileerror（25%下行分位数）。实际运行版本、全部参数随每个模型保存。
分类概率未经校准，不直接称为可信的真实胜率；不同目标不能混用门槛单位。

## 退出

StockCharts原始说明：`https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit`。
吊顶退出使用高点减ATR倍数，常见说明为22期最高点减3倍ATR22。
本地改编为每批持仓高水位、上一完成日ATR14及只上移保护线，不是原公式完全复刻。
叠加初始ATR止损、可选EMA20趋势失效与10/20交易日折算分钟期限，保留T+1及不可成交等待。
这里只比较这两套已实现退出，不宣称RSI/MACD/学习退出模型已测试。

## 选择偏差

Bailey与López de Prado的Deflated Sharpe Ratio论文：
`https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf`。
多次试验后选最大夏普会带来选择偏差，因此保存全部内部候选、按时间验证、模型先冻结再测试。
本轮没有计算论文中的DSR统计量，不能把简单年际稳定性惩罚称为DSR。
2026历史已曝光，严格不参与本轮拟合也不能恢复成从未见过的新测试集。
