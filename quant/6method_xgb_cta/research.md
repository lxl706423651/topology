# XGBoost与A股趋势择时研究依据

检索日期：2026-09-09。下述资料支持研究假设和实现规则，不承诺A股夏普超过3。

## 趋势择时与模型

- [AQR：A Century of Evidence on Trend-Following Investing（2017）](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing)：
  在跨资产历史中研究时间序列动量。本轮只借鉴多周期趋势，不能照搬其期货多空收益。
- [Levy和Lopes：Trend-Following Strategies via Dynamic Momentum Learning（2021）](https://arxiv.org/abs/2106.08420)：
  研究动态学习不同动量周期的重要性。用XGBoost预测候选交易净收益是本项目的研究
  改编，并非该论文的原始模型，也不是已经证明有效的A股策略。
- [XGBoost官方参数](https://xgboost.readthedocs.io/en/stable/parameter.html)：采用直方图树、
  浅层/正则化候选、固定种子与线程数。训练目标从旧截面排名改为可执行路径净收益。

## 退出

- [StockCharts：Chandelier Exit](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit)：
  高点减ATR倍数形成波动自适应保护线。原始指标可以下降，本项目额外约束保护线只收紧。
- [Fidelity：ATR](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr)：
  ATR度量波动而非方向，涵盖跳空。本轮使用日尺度ATR，避免将很小的单分钟ATR
  当作跨日股票风险尺度；止损触发后仍可能受T+1及涨跌停阻挡。

## 板块规则与费用

- [上交所科创板交易问答](https://edu.sse.com.cn/tib/qa/)：20%涨跌幅、上市前五日
  无涨跌幅限制；最低200股，超过200股可逐股递增，不足200股余额须一次卖出。
- [深交所创业板交易特别规定解读](https://investor.szse.cn/institute/rules/t20200807_580310.html)：
  20%涨跌幅，上市前五日无涨跌幅限制，100股及整数倍申报。
- [创业板风险警示股票规则](https://www.szse.cn/disclosure/notice/general/t20200710_579459.html)：
  创业板ST不应误套主板5%制度；本策略不新买ST，但历史持仓退出仍须正确识别板块。
- [上交所2026规则生效公告](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)：
  2026-07-06生效，不把尚未生效的修订倒用于截至2026-04的本地历史。
- [税务总局2023年第39号公告](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)：
  2023-08-28起证券交易印花税减半。本轮按日期计费；万八免5是用户指定假设。
- 历史日线、ST和证券上市日期使用BaoStock公开接口；它是第三方数据，必须与
  本地分钟收盘对照，不能因字段齐全就视为交易所逐笔成交、完整退市和公司行动数据。

## 验证与多重尝试

- [Bailey和Lopez de Prado：Deflated Sharpe Ratio（2014）](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)：
  多重试验、选择性报告和非正态收益会夸大回测表现。本轮预先限定候选，完整留痕，
  报告日收益块自助法区间而不把点估计当成保证；新历史回测不会消除过去研究的暴露。
- 牛熊震荡按基准上一已完成日60日收益是否超过正负5%分组，仅用于预先固定的
  分层诊断，不根据策略收益事后给年度贴标签，也不据分组结果再次调参。
