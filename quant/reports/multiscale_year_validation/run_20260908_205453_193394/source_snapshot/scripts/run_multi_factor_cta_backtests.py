"""运行两个多因子 CTA 风格 A 股 long-only 策略回测。

输入为已有 XGBoost 预测文件和可复用特征面板 pkl 缓存。本脚本不重训模型、
不重建大特征面板；它用当前及历史已完成 bar 形成趋势、突破、波动、量价
和模型 alpha 的 CTA 风格打分，只在训练期内部验证集上选择交易规则，随后
对测试集执行一次最终验收。输出包括资料依据、数据切分审计、验证规则表、
测试收益曲线、逐笔交易、平仓复盘和中文报告；不会修改原始行情数据。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_xgboost_long_only_backtest as xb
from scripts.tune_xgboost_long_only_parameters import create_selected_rule


PREDICTION_RUN_DIR = PROJECT_ROOT / "reports" / "xgboost_long_only_backtest" / "run_20260526_202355"
FEATURE_PANEL_CACHE = PROJECT_ROOT / "reports" / "feature_panel_cache" / "feature_panel_7226a2c8f3dd389a.pkl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "multi_factor_cta_backtests"
SCORE_BLEND_STRENGTH = 0.0005


@dataclass(frozen=True)
class CtaStrategy:
    """单个 CTA 风格策略定义。

    name: 稳定英文标识，用于目录和文件名。
    family: 策略家族；本脚本固定输出两个家族，各自独立验证和测试。
    score_col: 已构造好的策略打分列，回测时映射到 prediction 字段。
    formula: 中文因子公式说明。
    rationale: 中文交易逻辑说明。
    """

    name: str
    family: str
    score_col: str
    formula: str
    rationale: str


def parse_args() -> argparse.Namespace:
    """解析策略路径、交易约束和验证规则候选参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-run-dir", type=Path, default=PREDICTION_RUN_DIR)
    parser.add_argument("--feature-panel-cache", type=Path, default=FEATURE_PANEL_CACHE)
    parser.add_argument("--stable-symbols-file", type=Path, default=xb.STABLE_SYMBOLS_FILE)
    parser.add_argument("--feature-manifest", type=Path, default=xb.MODEL_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--commission-rate", type=float, default=xb.COMMISSION_RATE)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--top-n-candidates", default="1,2,3")
    parser.add_argument("--top-quantile-candidates", default="0.02")
    parser.add_argument("--hold-steps-candidates", default="1")
    parser.add_argument("--target-exposure-candidates", default="1.0")
    parser.add_argument("--confidence-quantile-candidates", default="none,0.9")
    parser.add_argument("--spread-quantile-candidates", default="none,0.7")
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--min-holding-count", type=int, default=1)
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--score-blend-strength", type=float, default=SCORE_BLEND_STRENGTH)
    parser.add_argument("--sharpe-target", type=float, default=3.0)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--workers", type=int, default=xb.WORKERS)
    parser.add_argument("--seed", type=int, default=xb.RANDOM_SEED)
    return parser.parse_args()


def create_backtest_config(args: argparse.Namespace) -> xb.ExperimentConfig:
    """生成与主 XGBoost 回测脚本兼容的回测配置对象。"""

    namespace = argparse.Namespace(
        data_root=xb.DATA_ROOT,
        stable_symbols_file=args.stable_symbols_file,
        feature_manifest=args.feature_manifest,
        output_dir=args.output_dir,
        feature_panel_cache_dir=xb.FEATURE_PANEL_CACHE_DIR,
        split_date=args.split_date,
        validation_start=args.validation_start,
        horizon=args.horizon,
        eval_step=args.eval_step,
        commission_rate=args.commission_rate,
        initial_cash=args.initial_cash,
        top_n_candidates=args.top_n_candidates,
        top_quantile_candidates=args.top_quantile_candidates,
        min_assets=args.min_assets,
        min_holding_count=args.min_holding_count,
        hold_steps_candidates=args.hold_steps_candidates,
        target_exposure_candidates=args.target_exposure_candidates,
        confidence_quantile_candidates=args.confidence_quantile_candidates,
        spread_quantile_candidates=args.spread_quantile_candidates,
        lot_size=args.lot_size,
        sharpe_target=args.sharpe_target,
        max_symbols=args.max_symbols,
        workers=args.workers,
        seed=args.seed,
        rebuild_feature_panel=False,
        no_feature_panel_cache=False,
        build_feature_panel_only=False,
    )
    return xb.create_config(namespace)


def read_prediction_frame(path: Path) -> pd.DataFrame:
    """读取既有预测文件，并标准化回测需要的时间列。"""

    frame = pd.read_csv(path)
    for col in ["datetime", "entry_datetime", "label_exit_datetime"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def read_feature_panel(path: Path, symbols: Sequence[str]) -> pd.DataFrame:
    """读取 pkl 特征面板并只保留 CTA 打分需要的列。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到特征面板缓存：{path}")
    panel = pd.read_pickle(path)
    allowed = set(symbols)
    if allowed:
        panel = panel.loc[panel["symbol"].astype(str).isin(allowed)].copy()
    needed = [
        "datetime",
        "symbol",
        "cs_rank_trend_slope_20",
        "cs_rank_trend_slope_60",
        "cs_rank_ema_gap_30",
        "cs_rank_ma_cross_5_20",
        "cs_rank_ma_cross_10_60",
        "cs_rank_breakout_pos_20",
        "cs_rank_breakout_pos_60",
        "cs_rank_breakout_pos_120",
        "cs_rank_price_rank_20",
        "cs_rank_price_rank_60",
        "cs_rank_roc_24",
        "cs_rank_roc_48",
        "cs_rank_weighted_momentum_20",
        "cs_rank_macd_dif_12_26_9",
        "cs_rank_macd_hist_12_26_9",
        "cs_rank_adx_14",
        "cs_rank_adx_24",
        "cs_rank_trend_r2_20",
        "cs_rank_trend_r2_60",
        "cs_rank_ret_std_20",
        "cs_rank_ret_std_60",
        "cs_rank_downside_vol_60",
        "cs_rank_boll_width_60",
        "cs_rank_volume_z_20",
        "cs_rank_volume_z_60",
        "cs_rank_volume_change_20",
        "cs_rank_money_flow_imbalance_20",
        "cs_rank_obv_z_60",
        "cs_rank_mfi_14",
        "cs_rank_vwap_gap_mean_20",
        "cs_rank_dist_high_20",
    ]
    available = [col for col in needed if col in panel.columns]
    return panel.loc[:, available].copy()


def calculate_mean_factor(frame: pd.DataFrame, cols: Sequence[str], invert: bool = False) -> pd.Series:
    """计算一组 0-1 截面 rank 因子的均值，缺失因子按 0.5 中性值处理。

    cols: 因子列名列表，因子应只使用当前及历史完成 bar。
    invert: 为 True 时使用 1-rank，适合低波动、低回撤这类防御因子。
    """

    available = [col for col in cols if col in frame.columns]
    if not available:
        return pd.Series(np.float32(0.5), index=frame.index)
    work = frame.loc[:, available].astype("float32").replace([np.inf, -np.inf], np.nan)
    if invert:
        work = 1.0 - work
    return work.mean(axis=1).fillna(0.5).astype("float32")


def add_cta_factor_scores(frame: pd.DataFrame, blend_strength: float) -> pd.DataFrame:
    """构造两个 CTA 策略使用的多因子打分列。

    打分由已有 156 因子 XGBoost alpha 预测作为主信号，再叠加很小的 CTA
    风格因子项。这样不会在测试集重训模型，也不会改变 signal_lag=1 的
    时间对齐；所有叠加因子均来自当前及历史已完成 bar 的截面 rank。
    """

    out = frame.copy()
    out["trend_factor"] = calculate_mean_factor(
        out,
        [
            "cs_rank_trend_slope_20",
            "cs_rank_trend_slope_60",
            "cs_rank_ema_gap_30",
            "cs_rank_ma_cross_5_20",
            "cs_rank_ma_cross_10_60",
        ],
    )
    out["breakout_factor"] = calculate_mean_factor(
        out,
        [
            "cs_rank_breakout_pos_20",
            "cs_rank_breakout_pos_60",
            "cs_rank_breakout_pos_120",
            "cs_rank_price_rank_20",
            "cs_rank_price_rank_60",
        ],
    )
    out["momentum_factor"] = calculate_mean_factor(
        out,
        [
            "cs_rank_roc_24",
            "cs_rank_roc_48",
            "cs_rank_weighted_momentum_20",
            "cs_rank_macd_dif_12_26_9",
            "cs_rank_macd_hist_12_26_9",
        ],
    )
    out["trend_quality_factor"] = calculate_mean_factor(
        out,
        ["cs_rank_adx_14", "cs_rank_adx_24", "cs_rank_trend_r2_20", "cs_rank_trend_r2_60"],
    )
    out["low_volatility_factor"] = calculate_mean_factor(
        out,
        ["cs_rank_ret_std_20", "cs_rank_ret_std_60", "cs_rank_downside_vol_60", "cs_rank_boll_width_60"],
        invert=True,
    )
    out["volume_flow_factor"] = calculate_mean_factor(
        out,
        [
            "cs_rank_volume_z_20",
            "cs_rank_volume_z_60",
            "cs_rank_volume_change_20",
            "cs_rank_money_flow_imbalance_20",
            "cs_rank_obv_z_60",
            "cs_rank_mfi_14",
        ],
    )
    out["risk_adjusted_momentum_factor"] = (
        0.45 * out["momentum_factor"]
        + 0.25 * out["trend_quality_factor"]
        + 0.30 * out["low_volatility_factor"]
    ).astype("float32")
    out["pullback_breakout_factor"] = (
        0.45 * out["breakout_factor"]
        + 0.25 * (1.0 - out.get("cs_rank_dist_high_20", pd.Series(0.5, index=out.index)).fillna(0.5))
        + 0.30 * out["volume_flow_factor"]
    ).astype("float32")

    blend = float(blend_strength)
    trend_overlay = (
        0.25 * (out["trend_factor"] - 0.5)
        + 0.25 * (out["breakout_factor"] - 0.5)
        + 0.20 * (out["momentum_factor"] - 0.5)
        + 0.15 * (out["trend_quality_factor"] - 0.5)
        + 0.15 * (out["volume_flow_factor"] - 0.5)
    )
    risk_overlay = (
        0.30 * (out["risk_adjusted_momentum_factor"] - 0.5)
        + 0.30 * (out["low_volatility_factor"] - 0.5)
        + 0.20 * (out["breakout_factor"] - 0.5)
        + 0.20 * (out["volume_flow_factor"] - 0.5)
    )
    out["cta_trend_breakout_score"] = (out["prediction"].astype("float32") + blend * trend_overlay).astype("float32")
    out["cta_volatility_scaled_score"] = (out["prediction"].astype("float32") + blend * risk_overlay).astype("float32")
    return out


def merge_predictions_with_cta_features(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    symbols: Sequence[str],
    blend_strength: float,
) -> pd.DataFrame:
    """把既有预测和 CTA 特征按 datetime/symbol 对齐并生成策略打分。"""

    allowed = set(symbols)
    work = predictions.loc[predictions["symbol"].astype(str).isin(allowed)].copy()
    merged = work.merge(features, on=["datetime", "symbol"], how="left", validate="many_to_one")
    return add_cta_factor_scores(merged, blend_strength)


def create_cta_strategies() -> list[CtaStrategy]:
    """返回两个预先定义的多因子 CTA 风格策略。"""

    return [
        CtaStrategy(
            name="cta_trend_breakout_alpha",
            family="cta_trend_breakout",
            score_col="cta_trend_breakout_score",
            formula=(
                "XGBoost多因子alpha预测 + 小权重趋势斜率/均线差/通道突破/动量/"
                "ADX趋势质量/量价资金流叠加项"
            ),
            rationale=(
                "借鉴时间序列动量和通道突破思想，只在模型alpha很强且个股趋势、突破、"
                "量价确认较好的截面中买入；A股现货限制下只做多，不做空。"
            ),
        ),
        CtaStrategy(
            name="cta_volatility_scaled_alpha",
            family="cta_volatility_scaled",
            score_col="cta_volatility_scaled_score",
            formula=(
                "XGBoost多因子alpha预测 + 小权重风险调整动量/低波动/通道突破/"
                "量价资金流叠加项"
            ),
            rationale=(
                "借鉴 CTA 波动控制和风险平价思想，在模型alpha基础上偏向低下行波动、"
                "趋势质量更好且量价确认的股票，降低追涨噪声。"
            ),
        ),
    ]


def evaluate_cta_strategy(
    strategy: CtaStrategy,
    validation_frame: pd.DataFrame,
    config: xb.ExperimentConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """只用验证集评估单个 CTA 策略及其交易规则候选。"""

    validation_eval = validation_frame.copy()
    validation_eval["prediction"] = validation_eval[strategy.score_col].astype("float32")
    table = xb.evaluate_rules(validation_eval, config)
    table.to_csv(run_dir / f"validation_rules_{strategy.name}.csv", index=False, encoding="utf-8-sig")
    best = table.iloc[0]
    row: dict[str, Any] = {
        "strategy": strategy.name,
        "family": strategy.family,
        "score_col": strategy.score_col,
        "formula": strategy.formula,
        "rationale": strategy.rationale,
        "best_rule": best["rule"],
    }
    for key in [
        "sharpe",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "total_return",
        "avg_turnover",
        "avg_gross_exposure",
        "round_trip_win_rate",
        "trade_count",
        "confidence_quantile",
        "confidence_threshold",
        "spread_quantile",
        "spread_threshold",
    ]:
        row[f"validation_{key}"] = best.get(key)
    return row


def write_source_references(run_dir: Path) -> None:
    """写出本脚本参考资料及其到本地因子的映射。"""

    text = """# 多因子 CTA 策略资料依据

## 参考资料

- Moskowitz、Ooi、Pedersen《Time Series Momentum》：时间序列动量强调资产自身过去收益对未来收益的预测能力，并在股指、债券、外汇、商品期货中检验趋势效应。本地映射为 `roc`、`weighted_momentum`、`macd`、`trend_slope` 等历史动量和趋势因子。
- Hurst、Ooi、Pedersen《A Century of Evidence on Trend-Following Investing》：长期趋势跟踪研究显示，时间序列动量在多个年代和宏观环境中都存在，且和传统资产相关性较低。本地映射为趋势斜率、均线差、突破位置和趋势质量组合。
- Marshall、Cahan、Cahan《Trend-following trading strategies in commodity futures: A re-examination》：研究双均线交叉和通道策略。本地映射为 `ma_cross`、`breakout_pos`、`price_rank`。
- Kim、Tse、Wald《Time series momentum and volatility scaling》：指出波动缩放/风险平价会显著影响趋势策略表现。本地映射为低波动、低下行波动、布林带宽度和风险调整动量。

## 本地实现限制

- 本仓库当前数据是 A 股分钟股票数据，不是期货连续合约数据；因此脚本实现的是“CTA 风格择时/趋势跟踪 long-only 股票策略”，不是可多空的传统期货 CTA。
- 策略不做空、不加杠杆，`short_exposure=0`；所有交易仍使用现有现金账户、signal_lag=1、T+1、100 股整数手和万1成本假设。
- 策略候选和规则阈值只由内部验证集选择，测试集只做最终验收。

参考链接：

- https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
- https://research.cbs.dk/en/publications/a-century-of-evidence-on-trend-following-investing/
- https://www.sciencedirect.com/science/article/abs/pii/S037842660900199X
- https://www.sciencedirect.com/science/article/abs/pii/S1386418116301379
"""
    (run_dir / "资料依据.md").write_text(text, encoding="utf-8")


def write_split_audit(
    run_dir: Path,
    config: xb.ExperimentConfig,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """写出验证集和测试集边界，明确测试集没有参与规则选择。"""

    rows = [
        {
            "数据段(split)": "内部验证集",
            "用途(use)": "选择每个 CTA 策略的 top-N、置信度门控和分差门控规则",
            "行数(rows)": int(len(validation)),
            "股票数(symbol_count)": int(validation["symbol"].nunique()),
            "开始时间(start_datetime)": str(validation["datetime"].min()),
            "结束时间(end_datetime)": str(validation["datetime"].max()),
        },
        {
            "数据段(split)": "最终测试集",
            "用途(use)": "策略和规则选定后只做一次最终回测",
            "行数(rows)": int(len(test)),
            "股票数(symbol_count)": int(test["symbol"].nunique()),
            "开始时间(start_datetime)": str(test["datetime"].min()),
            "结束时间(end_datetime)": str(test["datetime"].max()),
        },
    ]
    pd.DataFrame(rows).to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    text = f"""# 数据切分审计

- 内部验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`，只用于选择策略规则。
- 最终测试集：`datetime >= {config.split_date.date()}`，在策略公式和验证规则确定后一次性验收。
- 本脚本没有在测试集上重新估计分位数阈值；置信度和分差门控阈值来自验证集。
- 本脚本没有用测试集 Sharpe 反向选择两个 CTA 策略的因子公式。
"""
    (run_dir / "数据切分审计.md").write_text(text, encoding="utf-8")


def calculate_frame_signature(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    include_index: bool = False,
) -> str:
    """计算 DataFrame 内容哈希，用于判断两个策略是否产生同一条交易路径。

    frame: 需要审计的表，例如权益曲线、交易事件、持仓或平仓复盘。
    columns: 可选列清单；用于排除预测分这类不影响真实成交路径的解释字段。
    include_index: 是否把索引写入哈希；权益曲线重置索引后通常不需要。
    """

    if columns is None:
        work = frame.copy()
    else:
        available = [col for col in columns if col in frame.columns]
        work = frame.loc[:, available].copy()
    csv_text = work.to_csv(index=include_index, lineterminator="\n")
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def save_strategy_outputs(
    run_dir: Path,
    strategy: CtaStrategy,
    validation_table: pd.DataFrame,
    selected_rule: xb.PortfolioRule,
    test_frame: pd.DataFrame,
    config: xb.ExperimentConfig,
) -> dict[str, Any]:
    """保存单个 CTA 策略的最终测试集产物。"""

    strategy_dir = run_dir / strategy.family
    strategy_dir.mkdir(parents=True, exist_ok=True)
    test_eval = test_frame.copy()
    test_eval["prediction"] = test_eval[strategy.score_col].astype("float32")
    curve, holdings, trades, round_trips, open_positions, metrics = xb.run_long_only_backtest(
        test_eval,
        selected_rule,
        config,
    )
    metrics.update(
        {
            "strategy_family": strategy.family,
            "strategy_name": strategy.name,
            "strategy_formula": strategy.formula,
            "strategy_rationale": strategy.rationale,
            "test_start": str(test_eval["datetime"].min()),
            "test_end": str(test_eval["datetime"].max()),
            "test_rows": int(len(test_eval)),
            "test_symbol_count": int(test_eval["symbol"].nunique()),
        }
    )
    holding_execution_cols = [col for col in holdings.columns if col != "prediction"]
    round_trip_execution_cols = [col for col in round_trips.columns if col != "买入预测分(prediction)"]
    equity_curve_signature = calculate_frame_signature(curve.reset_index())
    trade_events_signature = calculate_frame_signature(trades)
    holdings_execution_signature = calculate_frame_signature(holdings, holding_execution_cols)
    round_trip_execution_signature = calculate_frame_signature(round_trips, round_trip_execution_cols)
    metrics.update(
        {
            "equity_curve_signature": equity_curve_signature,
            "trade_events_signature": trade_events_signature,
            "holdings_execution_signature": holdings_execution_signature,
            "round_trip_execution_signature": round_trip_execution_signature,
            "executable_path_signature": hashlib.sha256(
                "\n".join(
                    [
                        equity_curve_signature,
                        trade_events_signature,
                        holdings_execution_signature,
                        round_trip_execution_signature,
                    ]
                ).encode("utf-8")
            ).hexdigest(),
        }
    )
    validation_table.to_csv(strategy_dir / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(strategy_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    holdings.to_csv(strategy_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(strategy_dir / "test_trade_events.csv", index=False, encoding="utf-8-sig")
    round_trips.to_csv(strategy_dir / "test_round_trips.csv", index=False, encoding="utf-8-sig")
    open_positions.to_csv(strategy_dir / "test_open_positions.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(strategy_dir / "test_metrics.csv", header=["value"])
    xb.plot_test_return_curve(strategy_dir, curve, metrics)
    xb.write_output_guides(strategy_dir)
    metadata = {
        "strategy": asdict(strategy),
        "selected_rule": asdict(selected_rule),
        "config": {
            key: str(value) if isinstance(value, Path | pd.Timestamp) else value
            for key, value in asdict(config).items()
        },
        "test_metrics": metrics,
        "no_test_tuning_statement": "策略公式和规则只按内部验证集确定；测试集只用于最终验收。",
    }
    (strategy_dir / "strategy_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def write_report(
    run_dir: Path,
    config: xb.ExperimentConfig,
    strategy_rows: pd.DataFrame,
    test_rows: list[dict[str, Any]],
    blend_strength: float,
) -> None:
    """写出中文多因子 CTA 策略报告。"""

    all_reached = all(float(row.get("sharpe", 0.0)) >= config.sharpe_target for row in test_rows)
    selected_same_rule = bool(strategy_rows["best_rule"].nunique() == 1) if not strategy_rows.empty else False
    path_signatures = {
        str(row.get("executable_path_signature"))
        for row in test_rows
        if row.get("executable_path_signature")
    }
    unique_path_count = len(path_signatures) if path_signatures else len(test_rows)
    duplicated_paths = unique_path_count < len(test_rows)
    lines = [
        "# 多因子 CTA 风格策略回测报告",
        "",
        "## 研究边界",
        "",
        "- 本报告包含两个策略：`cta_trend_breakout` 和 `cta_volatility_scaled`。",
        "- 当前市场和数据口径是 A 股股票分钟数据，所以这是 CTA 风格 long-only 股票择时策略，不是期货多空 CTA。",
        "- 两个策略都不做空、不加杠杆，`short_exposure=0`，并沿用 signal_lag=1、T+1、100 股整数手、万1成本。",
        f"- CTA 因子叠加强度：`{blend_strength}`；主信号仍为已训练好的 156 因子 XGBoost alpha 预测。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`；测试集：`datetime >= {config.split_date.date()}`。",
        "",
        "## 因子公式",
        "",
    ]
    for _, row in strategy_rows.iterrows():
        lines.extend(
            [
                f"### {row['family']}",
                "",
                f"- 策略名：`{row['strategy']}`",
                f"- 打分列：`{row['score_col']}`",
                f"- 构造方法：{row['formula']}",
                f"- 交易思路：{row['rationale']}",
                f"- 验证集选中规则：`{row['best_rule']}`",
                f"- 验证 Sharpe：{float(row['validation_sharpe']):.6f}",
                "",
            ]
        )
    lines.extend(["## 最终测试集验收", ""])
    for metrics in test_rows:
        reached = float(metrics.get("sharpe", 0.0)) >= config.sharpe_target
        lines.extend(
            [
                f"### {metrics['strategy_family']}",
                "",
                f"- 策略：`{metrics['strategy_name']}`",
                f"- 规则：`{metrics.get('rule')}`",
                f"- Sharpe：{float(metrics.get('sharpe', 0.0)):.6f}",
                f"- 年化收益：{float(metrics.get('annual_return', 0.0)):.6f}",
                f"- 年化波动率：{float(metrics.get('annual_volatility', 0.0)):.6f}",
                f"- 最大回撤：{float(metrics.get('max_drawdown', 0.0)):.6f}",
                f"- 成本后总收益：{float(metrics.get('total_return', 0.0)):.6f}",
                f"- 平均多头暴露：{float(metrics.get('avg_gross_exposure', 0.0)):.6f}",
                f"- short_exposure：{float(metrics.get('short_exposure', 0.0)):.6f}",
                f"- 已平仓胜率：{float(metrics.get('round_trip_win_rate', 0.0)):.6f}",
                f"- 总交易手数：{float(metrics.get('total_trade_lots', 0.0)):.0f}",
                f"- Sharpe 目标 `{config.sharpe_target:g}` 是否达成：`{reached}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 结论",
            "",
            f"- 两个命名策略是否全部达到测试集 Sharpe > {config.sharpe_target:g}：`{all_reached}`。",
            f"- 两个策略验证集是否选中同一交易规则：`{selected_same_rule}`。",
            f"- 去重后的可执行交易路径数量：`{unique_path_count}`。",
            f"- 是否存在重复交易路径：`{duplicated_paths}`。如果为 True，说明不同打分公式没有改变最终成交序列，不能把它们视为两个可叠加的独立 alpha。",
            "- 由于本项目历史测试集已经被多次查看，本结果仍应作为当前数据口径下的研究回测；后续生产判断需要新增未来数据或预先声明的 walk-forward。",
            "",
            "## 产物说明",
            "",
            "- `资料依据.md`：CTA/趋势跟踪资料和本地因子映射。",
            "- `数据切分审计.md`、`split_audit.csv`：验证集和测试集边界。",
            "- `strategy_validation_summary.csv`：两个 CTA 策略在验证集上选中的规则和指标。",
            "- `test_strategy_metrics.csv`：两个 CTA 策略的最终测试指标。",
            "- `交易路径一致性审计.md`、`trade_path_audit.csv`：多个命名策略是否共享同一条可执行交易路径。",
            "- `cta_trend_breakout/`、`cta_volatility_scaled/`：每个策略的权益曲线、收益图、持仓、逐笔交易、平仓复盘、期末持仓和元数据。",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    (run_dir / "multi_factor_cta_report.md").write_text(report_text, encoding="utf-8")
    (run_dir / "final_report.md").write_text(report_text, encoding="utf-8")


def write_trade_path_audit(run_dir: Path, test_rows: list[dict[str, Any]]) -> None:
    """写出交易路径去重审计，避免把同一路径误解为多个独立策略。

    run_dir: 当前实验输出目录。
    test_rows: 每个策略测试后返回的指标字典，必须包含交易路径签名字段。
    """

    rows = [
        {
            "strategy_family": row.get("strategy_family"),
            "strategy_name": row.get("strategy_name"),
            "rule": row.get("rule"),
            "sharpe": row.get("sharpe"),
            "equity_curve_signature": row.get("equity_curve_signature"),
            "trade_events_signature": row.get("trade_events_signature"),
            "holdings_execution_signature": row.get("holdings_execution_signature"),
            "round_trip_execution_signature": row.get("round_trip_execution_signature"),
            "executable_path_signature": row.get("executable_path_signature"),
        }
        for row in test_rows
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(run_dir / "trade_path_audit.csv", index=False, encoding="utf-8-sig")
    unique_path_count = int(audit["executable_path_signature"].nunique(dropna=True)) if not audit.empty else 0
    duplicated_paths = bool(unique_path_count < len(audit)) if not audit.empty else False
    text = f"""# 交易路径一致性审计

- 审计目的：判断不同命名策略是否真正产生不同成交序列，避免把同一条路径误当成多个独立 alpha。
- 策略数量：`{len(audit)}`。
- 去重后的可执行交易路径数量：`{unique_path_count}`。
- 是否存在重复交易路径：`{duplicated_paths}`。
- 审计口径：权益曲线、逐笔交易事件、持仓执行字段和平仓复盘执行字段的 SHA256 签名；持仓和复盘中的预测分只作为解释字段，不参与执行路径去重。
- 结论：如果重复交易路径为 True，这些策略不能按多个独立策略叠加配仓，实盘只能作为同一个 XGBoost alpha 核心的不同解释或确认方式。
"""
    (run_dir / "交易路径一致性审计.md").write_text(text, encoding="utf-8")


def write_root_output_guide(run_dir: Path) -> None:
    """写出根目录输出文件说明，保持和主 XGBoost 报告目录一致。"""

    text = """# 输出文件说明

- `final_report.md`：标准入口报告，汇总策略定义、验证规则、测试指标、限制和结论。
- `multi_factor_cta_report.md`：多因子 CTA 策略报告副本，内容与 `final_report.md` 一致，保留策略专属命名便于识别。
- `资料依据.md`：CTA/趋势跟踪资料和本地因子映射说明。
- `数据切分审计.md`：训练期内部验证集和最终测试集的边界说明。
- `split_audit.csv`：机器可读的数据切分审计表。
- `resolved_config.json`：本次运行解析后的配置、输入路径、交易成本、股票池和随机种子。
- `strategy_validation_summary.csv`：两个 CTA 策略在验证集上选出的规则和指标。
- `test_strategy_metrics.csv`：两个 CTA 策略在测试集上的最终指标汇总。
- `交易路径一致性审计.md`、`trade_path_audit.csv`：判断多个命名策略是否产生同一条可执行交易路径。
- `validation_rules_cta_trend_breakout_alpha.csv`：趋势突破型 CTA 策略的验证规则明细。
- `validation_rules_cta_volatility_scaled_alpha.csv`：波动控制型 CTA 策略的验证规则明细。
- `cta_trend_breakout/`：趋势突破型 CTA 策略的测试集权益曲线、收益图、持仓、交易记录、平仓复盘和指标。
- `cta_volatility_scaled/`：波动控制型 CTA 策略的测试集权益曲线、收益图、持仓、交易记录、平仓复盘和指标。
"""
    (run_dir / "输出文件说明.md").write_text(text, encoding="utf-8")


def run_multi_factor_cta_backtests(args: argparse.Namespace) -> Path:
    """执行两个多因子 CTA 策略的验证选择和测试集验收。"""

    config = create_backtest_config(args)
    symbols = xb.read_stable_symbols(config.stable_symbols_file, config.max_symbols)
    features = read_feature_panel(args.feature_panel_cache, symbols)
    validation_predictions = read_prediction_frame(args.prediction_run_dir / "validation_predictions.csv")
    test_predictions = read_prediction_frame(args.prediction_run_dir / "test_predictions.csv")
    validation = merge_predictions_with_cta_features(
        validation_predictions,
        features,
        symbols,
        args.score_blend_strength,
    )
    test = merge_predictions_with_cta_features(
        test_predictions,
        features,
        symbols,
        args.score_blend_strength,
    )

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_source_references(run_dir)
    write_split_audit(run_dir, config, validation, test)
    resolved_config = {
        key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items()
    }
    resolved_config["prediction_run_dir"] = str(args.prediction_run_dir)
    resolved_config["feature_panel_cache"] = str(args.feature_panel_cache)
    resolved_config["score_blend_strength"] = float(args.score_blend_strength)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    strategies = create_cta_strategies()
    validation_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        print(f"正在验证 CTA 策略：{strategy.family}/{strategy.name}", flush=True)
        validation_rows.append(evaluate_cta_strategy(strategy, validation, config, run_dir))
    validation_summary = pd.DataFrame(validation_rows)
    validation_summary.to_csv(run_dir / "strategy_validation_summary.csv", index=False, encoding="utf-8-sig")

    strategy_map = {strategy.name: strategy for strategy in strategies}
    test_rows: list[dict[str, Any]] = []
    for _, row in validation_summary.iterrows():
        strategy = strategy_map[str(row["strategy"])]
        validation_table = pd.read_csv(run_dir / f"validation_rules_{strategy.name}.csv")
        selected_rule = create_selected_rule(validation_table.iloc[0])
        print(f"正在最终测试 CTA 策略：{strategy.family}/{strategy.name}", flush=True)
        metrics = save_strategy_outputs(run_dir, strategy, validation_table, selected_rule, test, config)
        test_rows.append(metrics)
    pd.DataFrame(test_rows).to_csv(run_dir / "test_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    write_trade_path_audit(run_dir, test_rows)
    write_report(run_dir, config, validation_summary, test_rows, float(args.score_blend_strength))
    write_root_output_guide(run_dir)
    print(f"多因子 CTA 策略结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """命令行入口。"""

    run_multi_factor_cta_backtests(parse_args())


if __name__ == "__main__":
    main()
