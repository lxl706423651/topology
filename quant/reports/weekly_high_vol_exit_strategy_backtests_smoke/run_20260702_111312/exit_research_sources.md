# 平仓策略资料来源

- [Kaminski and Lo stop-loss framework](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338)：止损规则在动量环境中可能改善风险收益，但随机游走环境下可能降低期望收益。
- [TA-Lib ATR documentation](https://ta-lib.github.io/ta-lib-python/func_groups/volatility_indicators.html)：ATR 是波动率指标，可用 high、low、close 计算，用于波动率自适应止损。
- [Chandelier Exit description](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit)：Chandelier Exit 使用最高价减去若干倍 ATR 作为多头移动止损。
- [Fidelity exit strategies](https://www.fidelity.com/learning-center/trading-investing/trading/exit-strategies)：时间退出用于定义最大持仓暴露时间，并提示到期后至少重新评估仓位。
