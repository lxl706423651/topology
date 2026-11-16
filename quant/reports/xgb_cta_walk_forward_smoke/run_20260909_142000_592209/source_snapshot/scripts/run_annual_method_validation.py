"""在quant项目根目录执行年度重训、中文报告和含2026数据的部署候选训练。

输入为config/annual_method_validation.yaml和只读分钟行情；生成可复用pkl、
年度模型、训练/验证审计、测试收益图和逐笔交易。首次构建缓存及重训较耗时，
不覆盖旧报告、不连接券商。测试年绝不参与该轮训练；最终全量模型没有测试成绩。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import load_annual_panel, write_json
from quant_backtest.annual_portfolio import MarketStore, run_lot_portfolio, calculate_daily_metrics
from quant_backtest.annual_training import create_year_folds, select_training_rows, select_training_features, fit_rank_model, select_signal_tradable_panel
from scripts.run_multi_factor_cta_backtests import add_cta_factor_scores

METHODS = {"method1": ("方法一：XGBoost排序与独立ATR退出", "prediction"),
           "method2": ("方法二：XGBoost加CTA趋势叠加与独立ATR退出", "cta_trend_breakout_score")}

LIMITATIONS = """- 历史数据已被此前研究查看；本轮是预先固定流程的历史稳健性检查，不是全新未见样本。
- 原始行情无完整历史ST状态、退市总表、复权因子和公司行动。使用2026年名称快照排除ST/退，仍有前视筛选及幸存者偏差；原始价格的分红拆股也未完整记账。
- 交易只在信号之后的下一分钟收盘模拟成交。使用双边万八佣金、免5、日期对应印花税、双边十万分之一过户费和双边万五滑点。
- 每次成交额最多占下一分钟成交额5%，零量/缺失不成交；主板前收盘±9.5%是保守阻成交代理，不等于完整涨跌停、集合竞价及排队模型。停牌持仓用最后可见价格估值，无法反映退市损失。
- 止损在24分钟采样点和日终检查，不是逐分钟监控；T+1或无法卖出时保存待平仓状态，不保证以止损价成交。入场后的首个监控区间保守只用当前收盘更新最高价，避免引入买入前的分钟内高点。
- 各年从相同现金独立启动，年末持仓按市值保留，不虚构强制成交。跨年拼接只是独立年度收益的统计连接，不是连续持仓实盘仿真。
- 方法二沿用方法一的XGBoost核心，仅增加CTA叠加，不能假定两者为独立收益来源。
- 最终全量模型包含2026训练数据，必须用2026年4月之后未参与研究的新数据或模拟盘验收；不能引用年度模型成绩充当它的独立测试成绩。
"""


def load_config(args: argparse.Namespace) -> dict:
    """读取args.config的YAML，并应用明确的烟测覆盖；所有相对路径相对项目根目录。"""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ["data_root", "output_dir", "cache_dir", "name_snapshot"]:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if args.max_symbols is not None:
        cfg["max_symbols"] = args.max_symbols
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.smoke:
        cfg.update(max_symbols=8, universe_size=8, min_cross_section=4, min_closed_lots=1,
                   max_features=12, gate_quantiles=[0.5], train_shortlist=1)
        cfg["model"]["n_estimators"] = 8
        cfg["exit_candidates"] = cfg["exit_candidates"][:1]
        cfg["output_dir"] += "_smoke"
    if not 0 < cfg["entry_equity_fraction"] <= 0.1 or cfg["entry_top_n"] > 5:
        raise ValueError("本实验要求每笔不超过总资产10%、最多前5候选")
    cfg["is_smoke"] = args.smoke
    return cfg


def generate_scores(model: object, frame: pd.DataFrame, features: list[str], cfg: dict) -> pd.DataFrame:
    """仅用frame的已完成特征和model生成双方法分数；返回结果不包含标签或未来收益。"""
    out = frame[["datetime", "symbol"] + [c for c in frame if c.startswith("cs_rank_")]].copy()
    out["prediction"] = model.predict(out[features])
    out = add_cta_factor_scores(out, cfg["cta_blend"])
    return out[["datetime", "symbol", "prediction", "cta_trend_breakout_score"]]


def select_method_rule(calibration: pd.DataFrame, validation: pd.DataFrame, market: MarketStore,
                       fold: dict, method: str, cfg: dict, directory: Path) -> dict:
    """在fold训练内calibration筛选退出规则，再用validation确认；不接收测试分数或指标。"""
    score_col = METHODS[method][1]
    top_scores = calibration.groupby("datetime")[score_col].max()
    candidates, train_rows = {}, []
    for q in cfg["gate_quantiles"]:
        threshold = float(top_scores.quantile(q))
        for exit_rule in cfg["exit_candidates"]:
            rule = {**exit_rule, "quantile": q, "threshold": threshold}
            key = f"{exit_rule['name']}_q{q:g}"
            candidates[key] = rule
            result = run_lot_portfolio(calibration, market, fold["calibration_start"], fold["calibration_end"], score_col, rule, cfg)
            train_rows.append({"rule_id": key, **result["metrics"], **rule})
    train_metrics = pd.DataFrame(train_rows).sort_values(["sharpe", "max_drawdown", "rule_id"], ascending=[False, False, True])
    train_metrics.to_csv(directory / f"{method}_training_rule_metrics.csv", index=False, encoding="utf-8-sig")
    eligible = train_metrics.loc[train_metrics.closed_lot_count >= cfg["min_closed_lots"]].head(cfg["train_shortlist"])
    validation_rows = []
    for key in eligible.rule_id:
        result = run_lot_portfolio(validation, market, fold["validation_start"], fold["validation_end"], score_col, candidates[key], cfg)
        validation_rows.append({"rule_id": key, **result["metrics"], **candidates[key]})
    table = pd.DataFrame(validation_rows, columns=train_metrics.columns)
    table.to_csv(directory / f"{method}_validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    # 事前固定的现金基准：验证期没有正夏普且足够交易的规则时，不强行入市。
    if len(table):
        eligible_val = table.loc[(table.closed_lot_count >= cfg["min_closed_lots"]) & (table.sharpe > 0)]
    else:
        eligible_val = table
    if len(eligible_val):
        winner = eligible_val.sort_values(["sharpe", "max_drawdown", "rule_id"], ascending=[False, False, True]).iloc[0]
        selected = {**candidates[winner.rule_id], "rule_id": winner.rule_id, "selection": "验证集正夏普且交易数量达标"}
    else:
        selected = {**cfg["exit_candidates"][0], "threshold": 1e9, "quantile": None,
                    "rule_id": "cash_validation_failed", "selection": "验证未达标，按事前规则保持现金"}
    return selected


def plot_result(directory: Path, result: dict, title: str) -> None:
    """把result的日终收益、回撤及交易指标绘成中文图，保存到directory，不展示训练收益。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font).exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    curve, metrics = result["curve"], result["metrics"]
    daily = curve.groupby(curve.datetime.dt.normalize()).last()
    initial = result["initial_cash"]
    equity = pd.concat([pd.Series([initial], index=[daily.index[0] - pd.Timedelta(days=1)]), daily.equity])
    fig, (ax, dd) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=15, y=0.97)
    sharpe_text = f"{metrics['sharpe']:.2f}" if metrics['annual_volatility'] > 0 else "未定义（收益无波动）"
    subtitle = (f"夏普 {sharpe_text} | 最大回撤 {metrics['max_drawdown']:.2%} | 收益 {metrics['total_return']:.2%}\n"
                f"年化 {metrics['annual_return']:.2%} | 年化波动 {metrics['annual_volatility']:.2%} | 已平仓胜率 {metrics['win_rate']:.2%}\n"
                f"买入 {metrics['buy_count']} 次 | 卖出 {metrics['sell_count']} 次 | 成交 {metrics['trade_lots']:.0f} 手 | 换手 {metrics['turnover']:.2f} 倍")
    fig.text(0.5, 0.88, subtitle, ha="center", va="center", fontsize=10)
    ax.plot(equity.index, (equity / initial - 1) * 100, color="#147d64", label="策略累计收益（含成本）")
    ax.set_ylabel("累计收益（%）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    drawdown = equity / equity.cummax() - 1
    dd.fill_between(equity.index, drawdown.to_numpy() * 100, 0, color="#bf5960", alpha=0.5)
    dd.set_ylabel("回撤（%）")
    dd.grid(alpha=0.2)
    fig.text(0.5, 0.015, result.get("report_note", "年度模型样本外研究；非实盘承诺。2026仅覆盖已有月份，期末持仓按市值计价。"), ha="center", fontsize=9)
    fig.subplots_adjust(top=0.79, bottom=0.09, hspace=0.1)
    fig.savefig(directory / "测试集收益曲线.png", dpi=140)
    shutil.copyfile(directory / "测试集收益曲线.png", directory / "test_return_curve.png")
    plt.close(fig)


def write_strategy_summary(directory: Path, result: dict, title: str, rule: dict) -> None:
    """把result指标写成易读中文纵表；directory/title/rule指定位置、策略名和冻结规则。"""
    labels = {"total_return": "总收益率", "annual_return": "年化收益率", "annual_volatility": "年化波动率",
              "sharpe": "夏普比率", "sortino": "下行风险调整收益比", "max_drawdown": "最大回撤",
              "calmar": "年化收益与最大回撤比", "buy_count": "买入成交次数", "sell_count": "卖出成交次数（含部分卖出）",
              "closed_lot_count": "完全平仓批次数", "trade_lots": "双边成交手数", "win_rate": "完全平仓批次胜率",
              "mean_holding_days": "平均持仓自然日", "turnover": "双边换手倍数", "fees": "手续费合计（元，不含价格内滑点）",
              "mean_exposure": "平均多头仓位比例", "trading_days": "交易日数", "open_lots": "期末未平仓批次"}
    percentages = {"total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate", "mean_exposure"}
    metrics = result["metrics"]
    rows = []
    for key, label in labels.items():
        value = metrics[key]
        display = f"{value:.2%}" if key in percentages else f"{value:.4f}"
        if key == "sharpe" and metrics["annual_volatility"] == 0:
            display = "未定义（收益无波动）"
        if key == "win_rate" and metrics["closed_lot_count"] == 0:
            display = "未定义（没有完全平仓批次）"
        rows.append({"指标": label, "结果": display})
    (directory / "final_report.md").write_text(f"# {title}\n\n规则：`{rule['rule_id']}`（{rule['selection']}）。\n\n"
        + ("全程现金、没有交易，夏普在统计上未定义；CSV中的0仅是无波动占位值，不是有效策略表现。\n\n" if metrics['annual_volatility'] == 0 else "")
        + pd.DataFrame(rows).to_markdown(index=False) + "\n\n![测试集收益与回撤](测试集收益曲线.png)\n\n## 限制\n\n" + result.get("limitations", LIMITATIONS), encoding="utf-8")


def calculate_cross_year_summary(table: pd.DataFrame) -> pd.DataFrame:
    """汇总table的年度指标；空仓无波动年的夏普未定义，不把CSV占位0作为有效夏普求均值。"""
    active = table.assign(sharpe=table.sharpe.where(table.annual_volatility > 0))
    return active.groupby("method").agg(mean_sharpe=("sharpe", "mean"), worst_sharpe=("sharpe", "min"),
        active_years=("sharpe", "count"), positive_years=("total_return", lambda x: int((x > 0).sum())),
        all_years_sharpe_above3=("sharpe", lambda x: bool((x > 3).all())))


def save_result(directory: Path, result: dict, title: str, rule: dict, cfg: dict) -> None:
    """保存result的指标/逐笔记录/图像/中文说明及cfg资金和rule规则，不覆盖已有运行。"""
    directory.mkdir(parents=True, exist_ok=True)
    names = {"curve": "test_equity_curve.csv", "events": "test_trade_events.csv", "trips": "test_round_trips.csv",
             "open": "test_open_positions.csv", "holdings": "test_holdings.csv"}
    for key, filename in names.items():
        result[key].to_csv(directory / filename, index=False, encoding="utf-8-sig")
    pd.DataFrame([result["metrics"]]).to_csv(directory / "test_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "strategy_metadata.json", {"strategy": title, "rule": rule, "blocked": result["blocked"]})
    result["initial_cash"] = cfg["initial_cash"]
    plot_result(directory, result, title)
    write_strategy_summary(directory, result, title, rule)
    (directory / "输出文件说明.md").write_text("""# 输出文件说明

| 文件 | 作用 |
|---|---|
| final_report.md | 本策略测试摘要与已知限制 |
| test_equity_curve.csv | 各估值点权益equity、现金cash、多头暴露exposure；不是价格收益简单相加 |
| 测试集收益曲线.png / test_return_curve.png | 同一张中文收益/回撤图的双文件名 |
| test_trade_events.csv | 实际模拟成交事件：datetime成交时刻、signal_datetime信号时刻、side方向、shares股数、price价格、fee费用、reason触发原因 |
| test_round_trips.csv | 买卖匹配：lot_id独立批次、买卖时间和价格、net_pnl扣费盈亏、holding_days自然日持仓时长；部分卖出会有多行 |
| test_open_positions.csv | 期末未平仓股数、买入价、估值价、止损价及待执行原因 |
| test_holdings.csv | 每日收盘各批次持仓与止损线 |
| test_metrics.csv | 收益total_return、年化annual_return、波动annual_volatility、夏普sharpe、回撤max_drawdown、换手turnover、胜率win_rate、交易数等 |
| strategy_metadata.json | 训练/验证选出的冻结规则、成交受限计数 |

夏普使用含空仓日的每日净收益、252交易日、无风险收益0；年化收益按实际自然日复合。
换手为双边成交额/平均日终权益，不是单边换手；胜率按完全平仓的批次聚合，部分未平仓批次不计入胜率。
涨跌停阻挡是代理模型，行情复权和历史ST状态尚不完整。详细限制见final_report.md。
""", encoding="utf-8")


def validate_result(result: dict, cfg: dict) -> dict:
    """审计result的成交滞后、T+1、整手、10%预算、现金和卖出原因；cfg提供约束。"""
    events, trips, curve = result["events"], result["trips"], result["curve"]
    checks = {"signals_strictly_before_fills": bool((events.signal_datetime < events.datetime).all()),
              "t1_sell": bool((trips.sell_datetime.dt.normalize() > trips.buy_datetime.dt.normalize()).all()) if len(trips) else True,
              "board_lots": bool((events.shares % cfg["lot_size"] == 0).all()),
              "cash_nonnegative": bool((curve.cash >= -1e-6).all()), "long_only": bool(curve.exposure.between(0, 1 + 1e-8).all()),
              "common_clock_unique": bool(curve.datetime.is_monotonic_increasing and not curve.datetime.duplicated().any())}
    buys = events.loc[events.side == "buy"]
    checks["entry_budget_10pct"] = bool((buys.notional <= buys.signal_equity * cfg["entry_equity_fraction"] + 1e-5).all())
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def run_annual_validation(cfg: dict, build_only: bool = False) -> Path:
    """按cfg先冻结所有年度模型/规则及最终候选，最后统一评估年度测试；build_only只建缓存。"""
    panel, raw_market, membership, manifest = load_annual_panel(cfg)
    if build_only:
        return Path(manifest["cache_path"])
    original_rows = len(panel)
    panel = select_signal_tradable_panel(panel, raw_market, cfg)
    print(f"信号时可买候选过滤：{original_rows} -> {len(panel)}行，只读取当时及过去行情", flush=True)
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(directory / "data_manifest.json", manifest)
    membership.to_csv(directory / "dynamic_universe_membership.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "protocol.json", {"freeze_before_test": True, "year_folds": create_year_folds(cfg),
               "label": "信号t，t+1分钟收盘买入，下一交易日同一分钟收盘的未来收益截面秩",
               "training": "上年7月前核心拟合；7-9月训练内样本外校准；10-12月验证；测试全年",
               "signal_tradability": "训练期门控审计发现高分偏向涨停代理股票，因此在信号时已知可买候选内拟合/校准/排序；不使用execution字段过滤",
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "code_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__), ROOT / "quant_backtest/annual_panel.py", ROOT / "quant_backtest/annual_training.py", ROOT / "quant_backtest/annual_portfolio.py"]}})
    market = MarketStore(raw_market)
    del raw_market
    fold_models = []
    split_audit = []
    for fold in create_year_folds(cfg):
        year = fold["year"]
        fold_dir = directory / f"year_{year}"
        fold_dir.mkdir()
        print(f"{year}轮：仅过去数据拟合与规则选择", flush=True)
        train = select_training_rows(panel, fold["train_start"], fold["train_end"], cfg["min_cross_section"])
        features, feature_audit = select_training_features(train, cfg)
        feature_audit.to_csv(fold_dir / "feature_selection.csv", index=False, encoding="utf-8-sig")
        print(f"训练行数 {len(train)}，训练期入选特征 {len(features)}", flush=True)
        model = fit_rank_model(train, features, cfg)
        model.save_model(fold_dir / "xgboost_model.json")
        assert train.label_end.max() < fold["train_end"] <= fold["test_start"]
        audit = {"year": year, "train_rows": len(train), "train_first": train.datetime.min(), "train_last": train.datetime.max(),
                 "last_training_label": train.label_end.max(), "features": len(features), **fold}
        split_audit.append(audit)
        write_json(fold_dir / "model_metadata.json", {**audit, "feature_columns": features, "model_parameters": model.get_params(),
                                                     "label_horizon": "下一交易日同一分钟", "model_shared_by_methods": ["method1", "method2"]})
        calibration_frame = panel.loc[panel.datetime.between(fold["calibration_start"], fold["calibration_end"], inclusive="left")]
        validation_frame = panel.loc[panel.datetime.between(fold["validation_start"], fold["validation_end"], inclusive="left")]
        calibration = generate_scores(model, calibration_frame, features, cfg)
        validation = generate_scores(model, validation_frame, features, cfg)
        selected = {method: select_method_rule(calibration, validation, market, fold, method, cfg, fold_dir) for method in METHODS}
        write_json(fold_dir / "selected_rules.json", selected)
        print(f"{year}轮冻结规则：{selected}", flush=True)
        fold_models.append((fold, model, features, selected))
        del train, calibration_frame, validation_frame, calibration, validation
    pd.DataFrame(split_audit).to_csv(directory / "split_audit.csv", index=False, encoding="utf-8-sig")
    if cfg["fit_deployment_model"]:
        print("训练含2026数据的最终候选，不计算该模型的历史测试成绩", flush=True)
        deployment = directory / "deployment_candidate"
        deployment.mkdir()
        end = pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1)
        full_train = select_training_rows(panel, pd.Timestamp(cfg["start_date"]), end, cfg["min_cross_section"])
        features, selection = select_training_features(full_train, cfg)
        model = fit_rank_model(full_train, features, cfg)
        model.save_model(deployment / "xgboost_model.json")
        selection.to_csv(deployment / "feature_selection.csv", index=False, encoding="utf-8-sig")
        write_json(deployment / "model_metadata.json", {"feature_columns": features, "model_parameters": model.get_params(),
                   "train_first": full_train.datetime.min(), "train_last": full_train.datetime.max(), "last_training_label": full_train.label_end.max(),
                   "train_rows": len(full_train), "train_rows_2026": int((full_train.datetime.dt.year == 2026).sum()),
                   "test_metrics": None, "status": "含2026训练数据，尚无独立未来测试成绩",
                   "rule_source": "最后年度轮的测试前年份验证；不按年度测试表现选规则",
                   "model_sha256": hashlib.sha256((deployment / "xgboost_model.json").read_bytes()).hexdigest()})
        write_json(deployment / "selected_rules.json", fold_models[-1][3])
        (deployment / "README.md").write_text("# 含2026数据的最终候选\n\n模型在2023至2026年4月可用标签数据上重新拟合。"
            "年度测试模型与本模型是不同模型，不能共享测试成绩。模型特征顺序见model_metadata.json，"
            "构造方法见quant_backtest/annual_panel.py；需用完整当时股票池计算截面秩。"
            "selected_rules.json仅继承2026年度轮的测试前验证规则，不按测试收益调参。"
            "模型重拟合可能改变分数分布，门控阈值迁移必须先在未来模拟盘验证。"
            "尚未接入实盘，不应直接自动委托。\n", encoding="utf-8")
        del full_train, model
    write_json(directory / "frozen_before_test.json", {"timestamp": datetime.now().isoformat(), "all_folds_frozen": True,
                                                      "deployment_is_not_tested": True})
    metrics_rows, audit_rows, daily_results, overlap_rows = [], [], {}, []
    for fold, model, features, selected in fold_models:
        year = fold["year"]
        test_frame = panel.loc[panel.datetime.between(fold["test_start"], fold["test_end"], inclusive="left")]
        test_scores = generate_scores(model, test_frame, features, cfg)
        test_scores.to_pickle(directory / f"year_{year}" / "test_predictions.pkl")
        print(f"{year}轮：执行冻结规则的年度测试，不再改参数", flush=True)
        results = {}
        for method, (name, score_col) in METHODS.items():
            result = run_lot_portfolio(test_scores, market, fold["test_start"], fold["test_end"], score_col, selected[method], cfg)
            checks = validate_result(result, cfg)
            destination = directory / f"year_{year}" / method
            save_result(destination, result, f"{year}年 {name}", selected[method], cfg)
            metrics_rows.append({"year": year, "method": method, "策略名称": name, "rule_id": selected[method]["rule_id"], **result["metrics"]})
            audit_rows.append({"year": year, "method": method, **checks})
            daily_results[(year, method)] = result["curve"].groupby(result["curve"].datetime.dt.normalize()).last().equity
            results[method] = result
        a = results["method1"]["events"].query("side == 'buy'")
        b = results["method2"]["events"].query("side == 'buy'")
        keys_a, keys_b = set(zip(a.datetime, a.symbol)), set(zip(b.datetime, b.symbol))
        da = daily_results[(year, "method1")].pct_change()
        db = daily_results[(year, "method2")].pct_change()
        overlap_rows.append({"year": year, "buy_overlap_method1": len(keys_a & keys_b) / len(keys_a) if keys_a else None,
                             "buy_overlap_method2": len(keys_a & keys_b) / len(keys_b) if keys_b else None,
                             "daily_return_correlation": da.corr(db) if da.std() > 0 and db.std() > 0 else None})
    table = pd.DataFrame(metrics_rows)
    table.to_csv(directory / "annual_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(directory / "execution_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlap_rows).to_csv(directory / "method_overlap.csv", index=False, encoding="utf-8-sig")
    overview = table[["year", "method", "total_return", "annual_return", "sharpe", "max_drawdown", "buy_count", "win_rate"]].copy()
    overview.columns = ["测试年份", "方法", "总收益率", "年化收益率", "夏普", "最大回撤", "买入次数", "已平仓胜率"]
    overview.loc[table.annual_volatility == 0, "夏普"] = np.nan
    summary = calculate_cross_year_summary(table)
    summary.to_csv(directory / "cross_year_summary.csv", encoding="utf-8-sig")
    (directory / "final_report.md").write_text("# 双方法年度滚动验证与含2026数据重训\n\n"
        "## 结果\n\n" + overview.to_markdown(index=False, floatfmt=".4f") + "\n\n收益字段为小数，0.10代表10%。2026年仅截至4月，不是完整年度。\n\n"
        + summary.to_markdown(floatfmt=".4f") + "\n\n无波动年度夏普未定义（nan），CSV原始0只是占位；平均和最差夏普仅统计有波动年度，不能当作拼接组合夏普。"
        "mean_sharpe为有效年度平均夏普，worst_sharpe为最差年度夏普，active_years为有效年度数，positive_years为盈利年度数。"
        "所有年度均超过3才视为跨年份夏普目标达标，禁止用最好一年替代整体结论。\n\n"
        "## 流程\n\n每轮只用测试前年份：7月前核心训练；7至9月训练内样本外校准门控/筛退出候选；10至12月验证选择。"
        "训练标签结束时间必须早于分割点。先冻结全部年度模型和规则，再统一读取年度测试分数、执行回测。"
        "原先在2025年筛选的156特征不直接搬到2024；使用固定274因子库，每轮只在核心训练期筛选。\n\n"
        "股票池每周使用上一周平均分钟波动率前300；统一24分钟时钟；先按当前已知行情筛可买候选，再选前5且过门控，每笔不超过信号时总资产10%。"
        "隔交易日可加独立批次，退出使用只能收紧的ATR240止损或训练验证选中的持仓时限，掉出目标不卖出。"
        "验证没有正夏普且足够交易的规则时，下一年保持现金，这是事前规则而非看到测试后的选择。\n\n"
        "## 2026训练说明\n\n2026年度测试模型不接触2026训练数据；另存deployment_candidate，使用包括2026在内的全部已完成标签重新训练。"
        "最后一交易日的下一交易日标签尚不可见，不能用于拟合。最终模型不报告样本外夏普。\n\n"
        "## 与旧回测的差别\n\n取消2025/2026文件交集作为存续要求；不沿用后期筛选的特征；"
        "缺失分钟线不再改变股票采样相位；退出信号和执行行情分离；日终夏普统一包含空仓日和初始资金。"
        "另增加滑点/成交容量/保守涨跌停代理，因此本轮与旧的高收益数字不是单独改变年份的完全同口径比较。"
        "第一版训练期审计发现门控受已知不可买股票主导，本版在信号时可买候选内训练和校准；这一修正未使用年度测试预测挑参数。\n\n"
        "## 限制\n\n" + LIMITATIONS + "\n## 方法参考\n\n"
        "- [scikit-learn时间序列切分](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)：按时间向前，避免未来训练过去。\n"
        "- [税务部门2023年印花税公告](https://shanghai.chinatax.gov.cn/zcfw/zcfgk/yhs/202308/t468451.html)：2023-08-28起减半。\n", encoding="utf-8")
    (directory / "输出文件说明.md").write_text("""# 输出文件说明

| 文件/目录 | 作用 |
|---|---|
| final_report.md | 年度结果、策略原理、2026训练与限制的总入口 |
| resolved_config.yaml | 本轮实际使用的全部配置 |
| protocol.json / frozen_before_test.json | 时间切分、标签定义、代码指纹及测试前冻结时间 |
| data_manifest.json | 原始文件大小/修改时间指纹、名称快照、缓存路径及数据限制 |
| dynamic_universe_membership.csv | 每日当时使用的上一周高波动股票池和选股窗口 |
| split_audit.csv | 每轮训练/验证/测试边界以及训练标签最后时刻 |
| annual_test_metrics.csv | 两方法逐年测试指标，英文列解释见各策略目录说明 |
| cross_year_summary.csv | 各方法平均/最差年度夏普、盈利年度数、是否每年都超过3 |
| execution_audit.csv | 成交延迟、T+1、整手、现金、预算和时钟检查 |
| method_overlap.csv | 按时间与股票的买入重合率、每日收益相关性 |
| year_YYYY/xgboost_model.json | 该年测试前拟合的模型，两方法共用基础预测 |
| year_YYYY/model_metadata.json | 训练窗口、超参数、严格有序特征列、标签定义 |
| year_YYYY/feature_selection.csv | 仅训练期计算的特征相关性、覆盖率和入选理由 |
| year_YYYY/selected_rules.json | 验证期选出的门控和退出规则 |
| year_YYYY/*_training_rule_metrics.csv | 训练内部校准期的全部候选成绩 |
| year_YYYY/*_validation_rule_metrics.csv | 训练短名单在验证期的成绩 |
| year_YYYY/test_predictions.pkl | 冻结模型的测试分数，仅本地留存，禁止据此调参 |
| year_YYYY/method1/ 与 method2/ | 各自报告、收益图、净值、交易、未平仓及输出说明 |
| deployment_candidate/ | 包括2026的全量重训候选、特征与规则；没有独立测试成绩 |

大型pkl缓存位于data_manifest.json指定目录，首次构建后复用；不上传Git。
""", encoding="utf-8")
    print(f"年度研究完成：{directory}", flush=True)
    return directory


def main() -> None:
    """解析YAML入口、缓存模式和烟测参数并执行；输出目录不会复用或覆盖旧实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/annual_method_validation.yaml")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    print(run_annual_validation(load_config(args), args.build_only))


if __name__ == "__main__":
    main()
