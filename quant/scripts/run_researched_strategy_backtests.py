"""运行资料驱动的风格中性和小市值代理 A 股策略回测。

输入为既有 XGBoost 预测结果、完整特征面板 pkl 缓存和稳定沪深 300 股票池。
脚本不重训模型、不重建因子面板；它只在训练期内部验证集上选择策略候选
和交易规则，随后对每个策略家族在测试集执行一次最终回测。输出包括资料
来源说明、候选验证表、测试收益曲线、逐笔交易记录和中文报告。当前实现
遵守 A 股现货 long-only、signal_lag=1、T+1、100 股整数手和交易成本约束；
“中性策略”指去除主要风格暴露后的 long-only 选股，不是股票多空市场中性。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
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
OUTPUT_DIR = PROJECT_ROOT / "reports" / "researched_strategy_backtests"
FULL_FEATURE_PANEL_CACHE = PROJECT_ROOT / "reports" / "feature_panel_cache" / "feature_panel_7226a2c8f3dd389a.pkl"
DAILY_SIZE_FEATURE_CACHE = PROJECT_ROOT / "reports" / "daily_size_features" / "latest" / "daily_size_features.pkl"


@dataclass(frozen=True)
class StrategyCandidate:
    """单个研究策略候选。

    name: 稳定英文标识，用于文件名。
    family: 策略家族，当前支持 neutral 和 small_cap。
    score_col: 已构造好的打分列，后续映射为 backtest 的 prediction。
    rationale: 中文策略逻辑说明。
    """

    name: str
    family: str
    score_col: str
    rationale: str


def parse_args() -> argparse.Namespace:
    """解析策略研究脚本参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-run-dir", type=Path, default=PREDICTION_RUN_DIR)
    parser.add_argument("--feature-panel-cache", type=Path, default=FULL_FEATURE_PANEL_CACHE)
    parser.add_argument(
        "--daily-size-feature-cache",
        type=Path,
        default=DAILY_SIZE_FEATURE_CACHE,
        help="可选日频流通市值特征 pkl；存在时小市值策略优先使用真实市值派生得分。",
    )
    parser.add_argument("--stable-symbols-file", type=Path, default=xb.STABLE_SYMBOLS_FILE)
    parser.add_argument("--feature-manifest", type=Path, default=xb.MODEL_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--commission-rate", type=float, default=xb.COMMISSION_RATE)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--top-n-candidates", default="1,2,3,5")
    parser.add_argument("--top-quantile-candidates", default="0.02,0.05")
    parser.add_argument("--hold-steps-candidates", default="1")
    parser.add_argument("--target-exposure-candidates", default="1.0")
    parser.add_argument("--confidence-quantile-candidates", default="none,0.85,0.9,0.92")
    parser.add_argument("--spread-quantile-candidates", default="none,0.6,0.7,0.8")
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--min-holding-count", type=int, default=1)
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--sharpe-target", type=float, default=3.0)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--workers", type=int, default=xb.WORKERS)
    parser.add_argument("--seed", type=int, default=xb.RANDOM_SEED)
    return parser.parse_args()


def create_backtest_config(args: argparse.Namespace) -> xb.ExperimentConfig:
    """生成与主 XGBoost 回测兼容的配置对象。"""

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
    """读取既有预测文件，并标准化时间列。"""

    frame = pd.read_csv(path)
    for col in ["datetime", "entry_datetime", "label_exit_datetime"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def calculate_percent_rank(frame: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    """按 datetime 计算截面百分位排名。"""

    rank = frame.groupby("datetime")[col].rank(method="average", pct=True, ascending=ascending)
    return rank.astype("float32")


def read_daily_size_features(path: Path) -> pd.DataFrame | None:
    """读取日频市值特征缓存；不存在时返回 None 并继续使用代理特征。"""

    if not path.exists():
        return None
    frame = pd.read_pickle(path)
    frame = frame.loc[:, ["symbol", "date", "log_float_market_cap", "derived_float_market_cap"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def add_research_features(panel: pd.DataFrame, daily_size: pd.DataFrame | None = None) -> pd.DataFrame:
    """在特征面板上构造只依赖当前和历史数据的研究辅助特征。"""

    needed = [
        "datetime",
        "symbol",
        "close",
        "amount",
        "cs_rank_amihud_illiq_20",
        "cs_rank_ret_std_60",
        "cs_rank_ret_mean_20",
        "cs_rank_volume_z_60",
        "cs_rank_trend_slope_20",
        "cs_rank_downside_vol_60",
    ]
    available = [col for col in needed if col in panel.columns]
    out = panel.loc[:, available].copy().sort_values(["symbol", "datetime"])
    out["amount_ma_20"] = (
        out.groupby("symbol", sort=False)["amount"]
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    out = out.sort_values(["datetime", "symbol"])
    out["low_price_score"] = 1.0 - calculate_percent_rank(out, "close", ascending=True)
    out["low_amount_score"] = 1.0 - calculate_percent_rank(out, "amount_ma_20", ascending=True)
    if "cs_rank_amihud_illiq_20" in out.columns:
        out["illiquidity_score"] = out["cs_rank_amihud_illiq_20"].astype("float32")
    else:
        out["illiquidity_score"] = np.nan
    out["size_proxy_score"] = (
        0.35 * out["low_price_score"].fillna(0.5)
        + 0.35 * out["low_amount_score"].fillna(0.5)
        + 0.30 * out["illiquidity_score"].fillna(0.5)
    ).astype("float32")
    if "cs_rank_downside_vol_60" in out.columns:
        out["defensive_score"] = (1.0 - out["cs_rank_downside_vol_60"].fillna(0.5)).astype("float32")
    else:
        out["defensive_score"] = np.float32(0.5)
    if daily_size is not None and not daily_size.empty:
        out["date"] = pd.to_datetime(out["datetime"]).dt.date
        out = out.merge(daily_size, on=["symbol", "date"], how="left", validate="many_to_one")
        out["real_small_cap_score"] = 1.0 - calculate_percent_rank(out, "log_float_market_cap", ascending=True)
        out["size_proxy_score"] = (
            0.70 * out["real_small_cap_score"].fillna(out["size_proxy_score"])
            + 0.20 * out["low_amount_score"].fillna(0.5)
            + 0.10 * out["illiquidity_score"].fillna(0.5)
        ).astype("float32")
    else:
        out["real_small_cap_score"] = np.nan
    return out


def merge_predictions_with_features(predictions: pd.DataFrame, features: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """把预测结果与研究辅助特征按 datetime/symbol 对齐。"""

    allowed = set(symbols)
    work = predictions.loc[predictions["symbol"].astype(str).isin(allowed)].copy()
    merged = work.merge(features, on=["datetime", "symbol"], how="left", validate="many_to_one")
    merged["prediction_rank"] = calculate_percent_rank(merged, "prediction", ascending=True)
    return merged


def residualize_by_datetime(frame: pd.DataFrame, target_col: str, exposure_cols: Sequence[str]) -> pd.Series:
    """按截面回归剔除风格暴露，并返回残差百分位得分。"""

    residual = pd.Series(np.nan, index=frame.index, dtype="float32")
    for _, group in frame.groupby("datetime", sort=False):
        valid_cols = [col for col in exposure_cols if col in group.columns]
        if not valid_cols:
            residual.loc[group.index] = group[target_col].astype("float32")
            continue
        work = group[[target_col, *valid_cols]].replace([np.inf, -np.inf], np.nan).copy()
        work[valid_cols] = work[valid_cols].fillna(work[valid_cols].median()).fillna(0.5)
        y = work[target_col].fillna(work[target_col].median()).fillna(0.5).to_numpy(dtype="float64")
        x = work[valid_cols].to_numpy(dtype="float64")
        x = np.column_stack([np.ones(len(x)), x])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            values = y - x @ beta
        except np.linalg.LinAlgError:
            values = y - np.nanmean(y)
        residual.loc[group.index] = values.astype("float32")
    scored = frame.assign(_residual=residual)
    return calculate_percent_rank(scored, "_residual", ascending=True)


def add_strategy_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """为中性策略和小市值代理策略生成多个候选打分。"""

    out = frame.copy()
    base_exposures = ["cs_rank_ret_std_60", "cs_rank_ret_mean_20", "cs_rank_volume_z_60"]
    style_exposures = [
        "size_proxy_score",
        "cs_rank_amihud_illiq_20",
        "cs_rank_ret_std_60",
        "cs_rank_ret_mean_20",
        "cs_rank_trend_slope_20",
    ]
    defensive_exposures = ["cs_rank_downside_vol_60", "cs_rank_volume_z_60", "size_proxy_score"]
    out["neutral_residual_basic_score"] = residualize_by_datetime(out, "prediction_rank", base_exposures)
    out["neutral_residual_style_score"] = residualize_by_datetime(out, "prediction_rank", style_exposures)
    out["neutral_residual_defensive_score"] = residualize_by_datetime(out, "prediction_rank", defensive_exposures)
    out["small_cap_proxy_score"] = out["size_proxy_score"].astype("float32")
    out["small_cap_alpha_30_score"] = (
        0.70 * out["prediction_rank"].fillna(0.5)
        + 0.30 * out["size_proxy_score"].fillna(0.5)
    ).astype("float32")
    out["small_cap_alpha_50_score"] = (
        0.50 * out["prediction_rank"].fillna(0.5)
        + 0.50 * out["size_proxy_score"].fillna(0.5)
    ).astype("float32")
    out["small_cap_quality_score"] = (
        0.55 * out["prediction_rank"].fillna(0.5)
        + 0.30 * out["size_proxy_score"].fillna(0.5)
        + 0.15 * out["defensive_score"].fillna(0.5)
    ).astype("float32")
    if "real_small_cap_score" in out.columns and out["real_small_cap_score"].notna().any():
        real_size = out["real_small_cap_score"].fillna(out["size_proxy_score"]).fillna(0.5)
        out["small_cap_real_size_score"] = real_size.astype("float32")
        out["small_cap_real_alpha_30_score"] = (
            0.70 * out["prediction_rank"].fillna(0.5)
            + 0.30 * real_size
        ).astype("float32")
        out["small_cap_real_alpha_50_score"] = (
            0.50 * out["prediction_rank"].fillna(0.5)
            + 0.50 * real_size
        ).astype("float32")
    return out


def create_strategy_candidates(has_real_size: bool = False) -> list[StrategyCandidate]:
    """返回预先定义、由验证集选择的两类研究策略候选。"""

    candidates = [
        StrategyCandidate(
            "neutral_residual_basic",
            "neutral",
            "neutral_residual_basic_score",
            "将 XGBoost 预测 rank 对波动、动量、成交活跃度做截面中性化，买入残差 alpha 最高的股票。",
        ),
        StrategyCandidate(
            "neutral_residual_style",
            "neutral",
            "neutral_residual_style_score",
            "在基础中性化上进一步剔除小市值代理、非流动性、波动、动量和趋势暴露。",
        ),
        StrategyCandidate(
            "neutral_residual_defensive",
            "neutral",
            "neutral_residual_defensive_score",
            "剔除下行波动、成交活跃度和小市值代理暴露，保留更接近个股残差信号的 alpha。",
        ),
        StrategyCandidate(
            "small_cap_proxy",
            "small_cap",
            "small_cap_proxy_score",
            "只使用低价格、低历史成交额和高非流动性构造小市值代理得分，不使用预测 alpha。",
        ),
        StrategyCandidate(
            "small_cap_alpha_30",
            "small_cap",
            "small_cap_alpha_30_score",
            "以 XGBoost 预测 rank 为主，加入 30% 小市值代理暴露，避免纯小市值噪声。",
        ),
        StrategyCandidate(
            "small_cap_alpha_50",
            "small_cap",
            "small_cap_alpha_50_score",
            "预测 alpha 和小市值代理各占一半，检验更强小市值倾斜是否有效。",
        ),
        StrategyCandidate(
            "small_cap_quality",
            "small_cap",
            "small_cap_quality_score",
            "在小市值代理和预测 alpha 上叠加低下行波动，避免买入高噪声小票代理。",
        ),
    ]
    if has_real_size:
        candidates.extend(
            [
                StrategyCandidate(
                    "small_cap_real_size",
                    "small_cap",
                    "small_cap_real_size_score",
                    "使用日频换手率反推流通市值，买入真实流通市值较小的股票。",
                ),
                StrategyCandidate(
                    "small_cap_real_alpha_30",
                    "small_cap",
                    "small_cap_real_alpha_30_score",
                    "XGBoost 预测 rank 占 70%，真实流通小市值得分占 30%。",
                ),
                StrategyCandidate(
                    "small_cap_real_alpha_50",
                    "small_cap",
                    "small_cap_real_alpha_50_score",
                    "XGBoost 预测 rank 和真实流通小市值得分各占 50%。",
                ),
            ]
        )
    return candidates


def evaluate_strategy_candidate(
    candidate: StrategyCandidate,
    validation_frame: pd.DataFrame,
    config: xb.ExperimentConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """只用验证集评估一个策略候选及其全部交易规则。"""

    validation_eval = validation_frame.copy()
    validation_eval["prediction"] = validation_eval[candidate.score_col].astype("float32")
    table = xb.evaluate_rules(validation_eval, config)
    table.to_csv(run_dir / f"validation_rules_{candidate.name}.csv", index=False, encoding="utf-8-sig")
    best = table.iloc[0]
    row: dict[str, Any] = {
        "candidate": candidate.name,
        "family": candidate.family,
        "score_col": candidate.score_col,
        "rationale": candidate.rationale,
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
    """写出本脚本参考资料和如何映射到当前可用数据。"""

    text = """# 资料依据和本地实现映射

- Fama-French 三因子框架把市值 ME 定义为股价乘以股数，并使用 size/SMB 作为股票共同风险因子。本地分钟数据没有历史股本和市值字段；若提供日频市值缓存，脚本会用换手率反推流通市值，否则退化为低价格、低历史成交额和高非流动性代理。
- 中国 A 股异常研究显示，A 股存在价值、风险、交易类异常，size/quality/past-return 类证据相对弱，但残差动量和反转更强。因此中性策略先剔除波动、动量、流动性和小市值代理等风格暴露，再交易残差 alpha。
- 市场中性组合通常需要多空或衍生品对冲。当前 A 股现货框架禁止做空且没有股指期货/融券数据，所以本脚本的“中性策略”是风格中性 long-only，不是严格市场中性。
- AKShare 东方财富日行情接口提供日收盘价、成交量和换手率；`scripts/build_daily_size_features.py` 用这些字段反推流通股本和流通市值，再把分钟信号按同日对齐。

参考链接：

- Fama 和 French, Common Risk Factors in the Returns on Stocks and Bonds: https://docslib.org/doc/4169268/common-risk-factors-in-the-returns-on-stocks-and-bonds
- Jansen、Swinkels、Zhou, Anomalies in the China A-share Market: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3810114
- Li、Li、Men, Size effect in China: https://ideas.repec.org/a/taf/applec/v53y2021i37p4358-4370.html
- Market-neutral portfolio selection: https://www.sciencedirect.com/science/article/abs/pii/S0378426698001149
- AKShare A 股历史行情接口说明: https://akshare.akfamily.xyz/data/stock/stock.html
"""
    (run_dir / "资料依据.md").write_text(text, encoding="utf-8")


def write_split_audit(run_dir: Path, config: xb.ExperimentConfig, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """写出本次策略验证和测试边界。"""

    rows = [
        {
            "数据段(split)": "内部验证集",
            "用途(use)": "选择策略候选和交易规则",
            "行数(rows)": int(len(validation)),
            "股票数(symbol_count)": int(validation["symbol"].nunique()),
            "开始时间(start_datetime)": str(validation["datetime"].min()),
            "结束时间(end_datetime)": str(validation["datetime"].max()),
        },
        {
            "数据段(split)": "最终测试集",
            "用途(use)": "选定后只做一次最终回测",
            "行数(rows)": int(len(test)),
            "股票数(symbol_count)": int(test["symbol"].nunique()),
            "开始时间(start_datetime)": str(test["datetime"].min()),
            "结束时间(end_datetime)": str(test["datetime"].max()),
        },
    ]
    pd.DataFrame(rows).to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    text = f"""# 数据切分审计

- 内部验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`，只用于选择策略候选、top-N、置信度门控和分差门控。
- 最终测试集：`datetime >= {config.split_date.date()}`，每个策略家族只在验证选定后执行一次回测。
- 本脚本没有用测试集 Sharpe 反向选择策略候选或交易规则。
"""
    (run_dir / "数据切分审计.md").write_text(text, encoding="utf-8")


def save_family_outputs(
    run_dir: Path,
    family: str,
    candidate: StrategyCandidate,
    validation_table: pd.DataFrame,
    selected_rule: xb.PortfolioRule,
    test_frame: pd.DataFrame,
    config: xb.ExperimentConfig,
) -> dict[str, Any]:
    """保存单个策略家族的最终测试回测产物。"""

    family_dir = run_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    test_eval = test_frame.copy()
    test_eval["prediction"] = test_eval[candidate.score_col].astype("float32")
    curve, holdings, trades, round_trips, open_positions, metrics = xb.run_long_only_backtest(
        test_eval,
        selected_rule,
        config,
    )
    metrics["strategy_family"] = family
    metrics["strategy_candidate"] = candidate.name
    metrics["strategy_rationale"] = candidate.rationale
    metrics["test_start"] = str(test_eval["datetime"].min())
    metrics["test_end"] = str(test_eval["datetime"].max())
    metrics["test_rows"] = int(len(test_eval))
    metrics["test_symbol_count"] = int(test_eval["symbol"].nunique())
    validation_table.to_csv(family_dir / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(family_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    holdings.to_csv(family_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(family_dir / "test_trade_events.csv", index=False, encoding="utf-8-sig")
    round_trips.to_csv(family_dir / "test_round_trips.csv", index=False, encoding="utf-8-sig")
    open_positions.to_csv(family_dir / "test_open_positions.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(family_dir / "test_metrics.csv", header=["value"])
    xb.plot_test_return_curve(family_dir, curve, metrics)
    xb.write_output_guides(family_dir)
    metadata = {
        "family": family,
        "candidate": asdict(candidate),
        "selected_rule": asdict(selected_rule),
        "config": {key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items()},
        "test_metrics": metrics,
        "no_test_tuning_statement": "策略候选和交易规则只按内部验证集选择；测试集只用于最终回测。",
    }
    (family_dir / "strategy_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_report(
    run_dir: Path,
    config: xb.ExperimentConfig,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    test_rows: list[dict[str, Any]],
) -> None:
    """写出中文策略研究报告。"""

    lines = [
        "# 资料驱动策略回测报告",
        "",
        "## 研究边界",
        "",
        "- 当前实现包含两个策略家族：风格中性 long-only 和小市值代理 long-only。",
        "- 风格中性不是股票多空市场中性；当前框架没有融券、股指期货或 ETF 对冲腿。",
        "- 小市值策略优先使用日频换手率反推的历史流通市值；未提供日频市值缓存时退化为低价格、低历史成交额和高非流动性代理。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`。",
        f"- 测试集：`datetime >= {config.split_date.date()}`，只用于最终验收。",
        "",
        "## 验证集候选排名",
        "",
        "| 策略家族 | 候选 | 验证 Sharpe | 年化收益 | 最大回撤 | 最优规则 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in summary.sort_values(["family", "validation_sharpe"], ascending=[True, False]).iterrows():
        lines.append(
            f"| `{row['family']}` | `{row['candidate']}` | {float(row['validation_sharpe']):.6f} | "
            f"{float(row['validation_annual_return']):.6f} | {float(row['validation_max_drawdown']):.6f} | "
            f"`{row['best_rule']}` |"
        )
    lines.extend(["", "## 每个策略家族的选中结果", ""])
    for _, row in selected.iterrows():
        lines.append(
            f"- `{row['family']}` 选中 `{row['candidate']}`，验证 Sharpe "
            f"{float(row['validation_sharpe']):.6f}，规则 `{row['best_rule']}`。"
        )
    lines.extend(["", "## 最终测试集验收", ""])
    for metrics in test_rows:
        reached = float(metrics.get("sharpe", 0.0)) >= config.sharpe_target
        lines.extend(
            [
                f"### {metrics['strategy_family']}",
                "",
                f"- 候选：`{metrics['strategy_candidate']}`",
                f"- Sharpe：{float(metrics.get('sharpe', 0.0)):.6f}",
                f"- 年化收益：{float(metrics.get('annual_return', 0.0)):.6f}",
                f"- 年化波动率：{float(metrics.get('annual_volatility', 0.0)):.6f}",
                f"- 最大回撤：{float(metrics.get('max_drawdown', 0.0)):.6f}",
                f"- 成本后总收益：{float(metrics.get('total_return', 0.0)):.6f}",
                f"- short_exposure：{float(metrics.get('short_exposure', 0.0)):.6f}",
                f"- Sharpe 目标 `{config.sharpe_target:g}` 是否达成：`{reached}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 产物说明",
            "",
            "- `strategy_candidate_summary.csv`：所有候选在验证集上的最优规则和指标。",
            "- `strategy_selection_summary.csv`：每个策略家族验证集选中的候选。",
            "- `neutral/`、`small_cap/`：各策略家族最终测试回测产物，含收益曲线和逐笔交易。",
            "- `资料依据.md`：论文和数据接口资料，以及本地实现的限制说明。",
        ]
    )
    (run_dir / "research_strategy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_researched_strategies(args: argparse.Namespace) -> Path:
    """执行策略候选验证选择和最终测试回测。"""

    config = create_backtest_config(args)
    symbols = xb.read_stable_symbols(config.stable_symbols_file, config.max_symbols)
    if not args.feature_panel_cache.exists():
        raise FileNotFoundError(f"找不到特征面板缓存：{args.feature_panel_cache}")
    panel = pd.read_pickle(args.feature_panel_cache)
    if config.max_symbols:
        panel = panel.loc[panel["symbol"].astype(str).isin(set(symbols))].copy()
    daily_size = read_daily_size_features(args.daily_size_feature_cache)
    features = add_research_features(panel, daily_size)
    validation_predictions = read_prediction_frame(args.prediction_run_dir / "validation_predictions.csv")
    test_predictions = read_prediction_frame(args.prediction_run_dir / "test_predictions.csv")
    validation = merge_predictions_with_features(validation_predictions, features, symbols)
    test = merge_predictions_with_features(test_predictions, features, symbols)
    validation = add_strategy_scores(validation)
    test = add_strategy_scores(test)

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_source_references(run_dir)
    write_split_audit(run_dir, config, validation, test)
    (run_dir / "resolved_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    has_real_size = "real_small_cap_score" in validation.columns and validation["real_small_cap_score"].notna().any()
    candidates = create_strategy_candidates(has_real_size)
    rows = []
    for candidate in candidates:
        print(f"正在验证策略候选：{candidate.family}/{candidate.name}", flush=True)
        rows.append(evaluate_strategy_candidate(candidate, validation, config, run_dir))
    summary = pd.DataFrame(rows).sort_values(["family", "validation_sharpe"], ascending=[True, False])
    summary.to_csv(run_dir / "strategy_candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected = (
        summary.sort_values(["family", "validation_sharpe", "validation_annual_return"], ascending=[True, False, False])
        .groupby("family", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected.to_csv(run_dir / "strategy_selection_summary.csv", index=False, encoding="utf-8-sig")

    candidate_map = {candidate.name: candidate for candidate in candidates}
    test_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        candidate = candidate_map[str(row["candidate"])]
        validation_table = pd.read_csv(run_dir / f"validation_rules_{candidate.name}.csv")
        selected_rule = create_selected_rule(validation_table.iloc[0])
        print(f"正在最终测试策略：{candidate.family}/{candidate.name}", flush=True)
        metrics = save_family_outputs(run_dir, candidate.family, candidate, validation_table, selected_rule, test, config)
        test_rows.append(metrics)
    pd.DataFrame(test_rows).to_csv(run_dir / "test_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    write_report(run_dir, config, summary, selected, test_rows)
    print(f"资料驱动策略结果已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """运行资料驱动策略回测流程。"""

    run_researched_strategies(parse_args())


if __name__ == "__main__":
    main()
