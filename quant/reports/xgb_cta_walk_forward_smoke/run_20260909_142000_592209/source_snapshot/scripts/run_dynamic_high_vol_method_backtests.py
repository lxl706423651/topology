"""回测动态高波动股票池下的 method1/method2 固定规则。

输入为本地 A 股分钟 OHLCV 数据、method1 已训练模型、严格训练期特征清单、
method2 CTA 元数据和当前或最近一次 A 股名称缓存。脚本构造两种动态股票池：
每周用上一交易周高波动 top-N 股票，或每天用过去 7 个自然日高波动 top-N
股票。每个交易日的股票池只由该日之前的数据决定；脚本不重新训练模型、不
重新选择门控阈值。输出包括动态股票池成员、验证/测试指标、交易记录、收益
曲线和中文报告。原始行情数据不会被修改。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_all_a_method_backtests as all_a
import scripts.run_multi_factor_cta_backtests as cta
import scripts.run_xgboost_long_only_backtest as xb


OUTPUT_DIR = PROJECT_ROOT / "reports" / "dynamic_high_vol_method_backtests"
PANEL_CACHE_DIR = OUTPUT_DIR / "panel_cache"
CACHE_VERSION = "dynamic_high_vol_panel_v1"


def parse_args() -> argparse.Namespace:
    """解析动态高波动股票池回测参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=xb.DATA_ROOT)
    parser.add_argument("--feature-manifest", type=Path, default=xb.MODEL_FEATURE_MANIFEST)
    parser.add_argument("--method1-run-dir", type=Path, default=all_a.METHOD1_RUN_DIR)
    parser.add_argument("--method2-run-dir", type=Path, default=all_a.METHOD2_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--panel-cache-dir", type=Path, default=PANEL_CACHE_DIR)
    parser.add_argument("--validation-start", default=xb.VALIDATION_START)
    parser.add_argument("--split-date", default=xb.SPLIT_DATE)
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--horizon", type=int, default=xb.FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=xb.EVALUATION_STEP_BARS)
    parser.add_argument("--initial-cash", type=float, default=xb.INITIAL_CASH)
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--lot-size", type=int, default=xb.LOT_SIZE)
    parser.add_argument("--workers", type=int, default=xb.WORKERS)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--daily-lookback-days", type=int, default=7)
    parser.add_argument("--min-window-days", type=int, default=3)
    parser.add_argument("--max-market-symbols", type=int, default=0, help="烟测用，0 表示使用全部沪深主板非 ST 股票。")
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--no-panel-cache", action="store_true")
    parser.add_argument("--no-akshare-st-filter", action="store_true")
    parser.add_argument("--broker-commission-rate", type=float, default=0.0008)
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--min-commission", type=float, default=0.0)
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> all_a.AllAMethodConfig:
    """把命令行参数转换为复用全 A 回测函数所需配置。"""

    return all_a.AllAMethodConfig(
        data_root=args.data_root,
        feature_manifest=args.feature_manifest,
        method1_run_dir=args.method1_run_dir,
        method2_run_dir=args.method2_run_dir,
        output_dir=args.output_dir,
        panel_cache_dir=args.panel_cache_dir,
        validation_start=pd.Timestamp(args.validation_start),
        split_date=pd.Timestamp(args.split_date),
        end_date=pd.Timestamp(args.end_date),
        forward_horizon_bars=int(args.horizon),
        evaluation_step_bars=int(args.eval_step),
        initial_cash=float(args.initial_cash),
        min_assets_per_rebalance=int(args.min_assets),
        lot_size=int(args.lot_size),
        max_symbols=0,
        workers=max(int(args.workers), 1),
        rebuild_panel=bool(args.rebuild_panel),
        no_panel_cache=bool(args.no_panel_cache),
        use_akshare_st_filter=not bool(args.no_akshare_st_filter),
        fee=all_a.FeeConfig(
            broker_commission_rate=float(args.broker_commission_rate),
            stamp_duty_rate=float(args.stamp_duty_rate),
            transfer_fee_rate=float(args.transfer_fee_rate),
            min_commission=float(args.min_commission),
            commission_includes_exchange_fees=True,
        ),
    )


def load_latest_name_map(use_akshare: bool) -> tuple[dict[str, str], str]:
    """优先读取当天或最近一次 A 股名称缓存，必要时再尝试 AKShare。"""

    if not use_akshare:
        return {}, "disabled"
    today_cache = all_a.STOCK_NAME_CACHE_DIR / f"a_stock_names_{datetime.now():%Y%m%d}.csv"
    if today_cache.exists():
        frame = pd.read_csv(today_cache, dtype={"code": str})
        return dict(zip(frame["code"].astype(str).str.zfill(6), frame["name"].astype(str), strict=False)), f"loaded_cache:{today_cache}"
    cached = sorted(all_a.STOCK_NAME_CACHE_DIR.glob("a_stock_names_*.csv"))
    if cached:
        latest = cached[-1]
        frame = pd.read_csv(latest, dtype={"code": str})
        return dict(zip(frame["code"].astype(str).str.zfill(6), frame["name"].astype(str), strict=False)), f"loaded_latest_cache:{latest}"
    return all_a.load_current_name_map(use_akshare)


def discover_mainboard_symbols(config: all_a.AllAMethodConfig, max_market_symbols: int) -> tuple[list[str], pd.DataFrame, str]:
    """发现本地沪深主板非 ST 股票池，支持最近名称缓存兜底。"""

    year_dirs = [config.data_root / "2025", config.data_root / "2026"]
    year_sets = []
    for year_dir in year_dirs:
        if not year_dir.exists():
            raise FileNotFoundError(f"找不到年份目录：{year_dir}")
        year_sets.append({path.stem for path in year_dir.glob("*.csv")})
    candidates = sorted(set.intersection(*year_sets))
    name_map, st_status = load_latest_name_map(config.use_akshare_st_filter)
    rows: list[dict[str, object]] = []
    selected: list[str] = []
    for symbol in candidates:
        code = all_a.normalize_symbol_code(symbol)
        name = name_map.get(code, "")
        reason = ""
        if not all_a.is_mainboard_symbol(symbol):
            reason = "非沪深主板或属于北交所/科创板/创业板"
        elif name_map and ("ST" in name.upper() or "退" in name):
            reason = "当前名称含 ST 或退"
        if reason:
            rows.append({"symbol": symbol, "code": code, "name": name, "included": False, "exclude_reason": reason})
            continue
        rows.append({"symbol": symbol, "code": code, "name": name, "included": True, "exclude_reason": ""})
        selected.append(symbol)
    if max_market_symbols:
        selected = selected[:max_market_symbols]
        selected_set = set(selected)
        for row in rows:
            if row["included"] and row["symbol"] not in selected_set:
                row["included"] = False
                row["exclude_reason"] = "max_market_symbols 烟测截断"
    if not selected:
        raise RuntimeError("动态高波动基础股票池为空。")
    return selected, pd.DataFrame(rows), st_status


def read_minute_data(symbol: str, data_root: Path, years: Sequence[int]) -> pd.DataFrame:
    """读取单只股票多个年份的分钟数据。"""

    parts: list[pd.DataFrame] = []
    usecols = {"datetime", "close", "amount"}
    for year in years:
        path = data_root / str(year) / f"{symbol}.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path, usecols=lambda col: col in usecols)
        part["symbol"] = symbol
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "amount" in frame.columns:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    else:
        frame["amount"] = np.nan
    return frame.sort_values("datetime").drop_duplicates(["symbol", "datetime"], keep="last")


def calculate_symbol_daily_volatility(task: tuple[str, Path, pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """计算单只股票每日分钟收益波动率。

    symbol: 股票代码。
    data_root: 本地分钟数据根目录。
    start_date: 为了给首个回测日提供历史窗口而提前读取的起始日期。
    end_date: 回测结束日期。
    """

    symbol, data_root, start_date, end_date = task
    years = sorted(range(start_date.year, end_date.year + 1))
    frame = read_minute_data(symbol, data_root, years)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.loc[(frame["datetime"] >= start_date) & (frame["datetime"] <= end_date + pd.Timedelta(days=1))].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["trade_date"] = frame["datetime"].dt.date
    frame["minute_return"] = frame.groupby("trade_date", sort=False)["close"].pct_change()
    daily = (
        frame.groupby("trade_date", sort=True)
        .agg(
            symbol=("symbol", "first"),
            minute_volatility=("minute_return", "std"),
            amount=("amount", "sum"),
            bar_count=("close", "count"),
        )
        .reset_index()
    )
    daily["minute_volatility"] = daily["minute_volatility"].replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(subset=["minute_volatility"])
    daily = daily.loc[daily["bar_count"] >= 120]
    return daily


def build_daily_volatility_panel(config: all_a.AllAMethodConfig, symbols: Sequence[str]) -> pd.DataFrame:
    """并行计算所有候选股票的日内分钟波动率面板。"""

    start_date = config.validation_start - pd.Timedelta(days=21)
    tasks = [(symbol, config.data_root, start_date, config.end_date) for symbol in symbols]
    frames: list[pd.DataFrame] = []
    print(f"正在计算动态高波动选股指标：symbols={len(symbols)}, workers={config.workers}", flush=True)
    if config.workers == 1:
        for idx, task in enumerate(tasks, start=1):
            frame = calculate_symbol_daily_volatility(task)
            if not frame.empty:
                frames.append(frame)
            if idx % 500 == 0 or idx == len(tasks):
                print(f"已计算日波动 {idx}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(calculate_symbol_daily_volatility, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                if idx % 500 == 0 or idx == len(futures):
                    print(f"已计算日波动 {idx}/{len(futures)}", flush=True)
    if not frames:
        raise RuntimeError("没有构建出任何日波动指标。")
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    return out


def select_window_top_symbols(
    daily_volatility: pd.DataFrame,
    window_start: date,
    window_end: date,
    top_n: int,
    min_window_days: int,
) -> pd.DataFrame:
    """在给定历史窗口内选择平均分钟波动率最高的股票。"""

    window = daily_volatility.loc[
        (daily_volatility["trade_date"] >= window_start) & (daily_volatility["trade_date"] <= window_end)
    ]
    if window.empty:
        return pd.DataFrame()
    ranked = (
        window.groupby("symbol", sort=False)
        .agg(
            selection_metric=("minute_volatility", "mean"),
            active_days=("trade_date", "nunique"),
            avg_amount=("amount", "mean"),
            avg_bar_count=("bar_count", "mean"),
        )
        .reset_index()
    )
    ranked = ranked.loc[ranked["active_days"] >= min_window_days].copy()
    ranked = ranked.sort_values(["selection_metric", "avg_amount"], ascending=[False, False]).head(top_n)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    ranked["window_start"] = str(window_start)
    ranked["window_end"] = str(window_end)
    return ranked


def get_evaluation_dates(daily_volatility: pd.DataFrame, config: all_a.AllAMethodConfig) -> list[date]:
    """取得验证和测试期间的本地交易日期。"""

    start = config.validation_start.date()
    end = config.end_date.date()
    dates = sorted(date_value for date_value in daily_volatility["trade_date"].dropna().unique() if start <= date_value <= end)
    if not dates:
        raise RuntimeError("没有找到验证/测试期间交易日。")
    return dates


def build_weekly_membership(
    daily_volatility: pd.DataFrame,
    evaluation_dates: Sequence[date],
    top_n: int,
    min_window_days: int,
) -> pd.DataFrame:
    """为每个交易日生成“上一交易周高波动 top-N”股票池。"""

    rows: list[pd.DataFrame] = []
    by_week: dict[date, pd.DataFrame] = {}
    for trade_date in evaluation_dates:
        week_start = trade_date - timedelta(days=trade_date.weekday())
        if week_start not in by_week:
            prev_start = week_start - timedelta(days=7)
            prev_end = week_start - timedelta(days=1)
            selected = select_window_top_symbols(daily_volatility, prev_start, prev_end, top_n, min_window_days)
            selected["selection_window_type"] = "previous_workweek"
            by_week[week_start] = selected
        selected = by_week[week_start].copy()
        if selected.empty:
            continue
        selected["trade_date"] = trade_date
        rows.append(selected)
    if not rows:
        raise RuntimeError("周度动态高波动股票池为空。")
    out = pd.concat(rows, ignore_index=True)
    out["universe_id"] = f"weekly_previous_week_high_vol_top{top_n}"
    return out


def build_daily_membership(
    daily_volatility: pd.DataFrame,
    evaluation_dates: Sequence[date],
    top_n: int,
    lookback_days: int,
    min_window_days: int,
) -> pd.DataFrame:
    """为每个交易日生成“过去7个自然日高波动 top-N”股票池。"""

    rows: list[pd.DataFrame] = []
    for trade_date in evaluation_dates:
        window_start = trade_date - timedelta(days=lookback_days)
        window_end = trade_date - timedelta(days=1)
        selected = select_window_top_symbols(daily_volatility, window_start, window_end, top_n, min_window_days)
        if selected.empty:
            continue
        selected["trade_date"] = trade_date
        selected["selection_window_type"] = f"rolling_{lookback_days}_calendar_days"
        rows.append(selected)
    if not rows:
        raise RuntimeError("日度滚动高波动股票池为空。")
    out = pd.concat(rows, ignore_index=True)
    out["universe_id"] = f"daily_rolling_{lookback_days}d_high_vol_top{top_n}"
    return out


def calculate_dynamic_cache_key(
    config: all_a.AllAMethodConfig,
    universe_id: str,
    membership: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    """生成动态股票池特征面板缓存键。"""

    member_hash = hashlib.sha256(
        membership.loc[:, ["trade_date", "symbol", "selection_rank", "window_start", "window_end"]]
        .sort_values(["trade_date", "selection_rank", "symbol"])
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()
    manifest_hash = hashlib.sha256(
        manifest.loc[:, ["model_feature", "factor_base"]].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    payload = {
        "cache_version": CACHE_VERSION,
        "universe_id": universe_id,
        "membership_hash": member_hash,
        "manifest_hash": manifest_hash,
        "validation_start": str(config.validation_start),
        "split_date": str(config.split_date),
        "end_date": str(config.end_date),
        "horizon": config.forward_horizon_bars,
        "eval_step": config.evaluation_step_bars,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_or_load_dynamic_rank_panel(
    config: all_a.AllAMethodConfig,
    universe_id: str,
    membership: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, Path | None, bool]:
    """按每日动态成员构建或读取截面 rank 特征面板。"""

    cache_key = calculate_dynamic_cache_key(config, universe_id, membership, manifest)
    config.panel_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.panel_cache_dir / f"dynamic_high_vol_panel_{cache_key[:16]}.pkl"
    metadata_path = config.panel_cache_dir / f"dynamic_high_vol_panel_{cache_key[:16]}.metadata.json"
    if cache_path.exists() and metadata_path.exists() and not config.rebuild_panel and not config.no_panel_cache:
        print(f"正在读取动态高波动特征缓存：{cache_path}", flush=True)
        return pd.read_pickle(cache_path), cache_path, True

    base_factors = manifest["factor_base"].astype(str).drop_duplicates().tolist()
    symbols = sorted(membership["symbol"].astype(str).unique())
    year_dirs = all_a.discover_year_dirs(config.data_root)
    tasks = [(symbol, year_dirs, base_factors, config) for symbol in symbols]
    frames: list[pd.DataFrame] = []
    print(f"正在构建动态股票池基础因子：{universe_id}, symbols={len(symbols)}, workers={config.workers}", flush=True)
    if config.workers == 1:
        for idx, task in enumerate(tasks, start=1):
            frame = all_a.build_symbol_base_panel(task)
            if not frame.empty:
                frames.append(frame)
            if idx % 100 == 0 or idx == len(tasks):
                print(f"已构建基础因子 {idx}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(all_a.build_symbol_base_panel, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                if idx % 100 == 0 or idx == len(futures):
                    print(f"已构建基础因子 {idx}/{len(futures)}", flush=True)
    if not frames:
        raise RuntimeError(f"{universe_id} 没有构建出任何基础因子样本。")
    base_panel = pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    base_panel["trade_date"] = pd.to_datetime(base_panel["datetime"]).dt.date
    member_keys = membership.loc[:, ["trade_date", "symbol"]].drop_duplicates().copy()
    member_keys["trade_date"] = pd.to_datetime(member_keys["trade_date"]).dt.date
    panel_base = base_panel.merge(member_keys, on=["trade_date", "symbol"], how="inner")
    if panel_base.empty:
        raise RuntimeError(f"{universe_id} 动态成员过滤后样本为空。")
    print(f"正在计算动态截面 rank：{universe_id}, rows={len(panel_base)}, factors={len(base_factors)}", flush=True)
    ranked = panel_base.groupby("datetime", sort=False)[base_factors].rank(method="average", pct=True).astype("float32")
    ranked.columns = [f"cs_rank_{col}" for col in ranked.columns]
    keep_cols = [
        "datetime",
        "symbol",
        "close",
        "volume",
        "amount",
        "forward_return",
        "entry_datetime",
        "entry_price",
        "exit_price",
        "label_exit_datetime",
    ]
    panel = pd.concat([panel_base[keep_cols].reset_index(drop=True), ranked.reset_index(drop=True)], axis=1)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    if not config.no_panel_cache:
        panel.to_pickle(cache_path)
        metadata = {
            "cache_version": CACHE_VERSION,
            "universe_id": universe_id,
            "rows": int(len(panel)),
            "symbol_count": int(panel["symbol"].nunique()),
            "start": str(panel["datetime"].min()),
            "end": str(panel["datetime"].max()),
            "feature_count": int(len(ranked.columns)),
            "membership_days": int(membership["trade_date"].nunique()),
            "cache_path": str(cache_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return panel, cache_path, False
    return panel, None, False


def create_strategy_specs(config: all_a.AllAMethodConfig) -> list[all_a.StrategySpec]:
    """读取 method1/method2 固定规则。"""

    return [
        all_a.StrategySpec(
            "method1_xgboost_long_only",
            "method1 XGBoost 156因子 long-only top1门控",
            "prediction",
            all_a.load_method_rule(config.method1_run_dir),
            "复用 run_20260526_202355 的 XGBoost 模型和固定 top1 分数/分差门控。",
        ),
        all_a.StrategySpec(
            "method2_cta_trend_breakout",
            "method2 CTA trend breakout",
            "cta_trend_breakout_score",
            all_a.load_cta_rule(config.method2_run_dir, "cta_trend_breakout"),
            "复用 method2 CTA trend 包装：XGBoost 分数 + 小权重趋势/突破/动量/量价确认。",
        ),
    ]


def score_panel(
    panel: pd.DataFrame,
    model: Any,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """为动态股票池特征面板生成 method1/method2 分数。"""

    missing = [col for col in feature_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"动态特征面板缺少模型特征：{missing[:10]}")
    out = panel.copy()
    out["prediction"] = model.predict(out.loc[:, feature_cols]).astype("float32")
    out = cta.add_cta_factor_scores(out, cta.SCORE_BLEND_STRENGTH)
    return out


def save_universe_outputs(
    run_dir: Path,
    universe_id: str,
    membership: pd.DataFrame,
    panel: pd.DataFrame,
    config: all_a.AllAMethodConfig,
    xb_config: xb.ExperimentConfig,
    strategies: Sequence[all_a.StrategySpec],
) -> list[dict[str, Any]]:
    """保存单个动态股票池的成员、切分审计和两种方法回测产物。"""

    universe_dir = run_dir / universe_id
    universe_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(universe_dir / "dynamic_universe_membership.csv", index=False, encoding="utf-8-sig")
    daily_summary = (
        membership.groupby("trade_date", sort=True)
        .agg(symbol_count=("symbol", "nunique"), avg_selection_metric=("selection_metric", "mean"))
        .reset_index()
    )
    daily_summary.to_csv(universe_dir / "dynamic_universe_daily_summary.csv", index=False, encoding="utf-8-sig")
    validation = panel.loc[(panel["datetime"] >= config.validation_start) & (panel["datetime"] < config.split_date)].copy()
    test = panel.loc[panel["datetime"] >= config.split_date].copy()
    pd.DataFrame(
        [
            {
                "split": "validation",
                "rows": len(validation),
                "symbols": validation["symbol"].nunique(),
                "start": validation["datetime"].min(),
                "end": validation["datetime"].max(),
            },
            {
                "split": "test",
                "rows": len(test),
                "symbols": test["symbol"].nunique(),
                "start": test["datetime"].min(),
                "end": test["datetime"].max(),
            },
        ]
    ).to_csv(universe_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    rows = []
    for strategy in strategies:
        print(f"正在回测 {universe_id} / {strategy.method_id}", flush=True)
        row = all_a.save_strategy_outputs(universe_dir, strategy, validation, test, xb_config, config)
        row["universe_id"] = universe_id
        row["avg_daily_symbol_count"] = float(daily_summary["symbol_count"].mean())
        row["min_daily_symbol_count"] = int(daily_summary["symbol_count"].min())
        row["max_daily_symbol_count"] = int(daily_summary["symbol_count"].max())
        row["union_symbol_count"] = int(membership["symbol"].nunique())
        rows.append(row)
    return rows


def write_output_guide(run_dir: Path) -> None:
    """写出根目录输出文件说明。"""

    text = """# 输出文件说明

- `final_report.md`：动态高波动股票池 method1/method2 对比主报告。
- `resolved_config.json`：解析后的路径、日期、费用、top-N、窗口参数和固定规则。
- `strategy_summary.csv`：两种动态股票池乘以两种方法的验证/测试指标汇总。
- `<universe_id>/dynamic_universe_membership.csv`：每日动态股票池成员及选股窗口指标。
- `<universe_id>/dynamic_universe_daily_summary.csv`：每日股票池数量和平均波动指标。
- `<universe_id>/split_audit.csv`：验证集和测试集样本边界。
- `<universe_id>/method1_xgboost_long_only/`：method1 回测产物。
- `<universe_id>/method2_cta_trend_breakout/`：method2 CTA 回测产物。
"""
    (run_dir / "输出文件说明.md").write_text(text, encoding="utf-8")


def write_final_report(
    run_dir: Path,
    config: all_a.AllAMethodConfig,
    summary: pd.DataFrame,
    cache_rows: list[dict[str, Any]],
    st_status: str,
    top_n: int,
    daily_lookback_days: int,
    min_window_days: int,
) -> None:
    """写出动态高波动股票池中文报告。"""

    weekly_universe_id = f"weekly_previous_week_high_vol_top{top_n}"
    daily_universe_id = f"daily_rolling_{daily_lookback_days}d_high_vol_top{top_n}"
    lines = [
        "# 动态高波动股票池 method1/method2 回测报告",
        "",
        "## 实验边界",
        "",
        "- 本报告不重新训练 XGBoost 模型，不重新选择 method1/method2 门控阈值。",
        "- 股票池每天或每周动态变化，且每个交易日的股票池只使用该日之前的历史波动率数据确定。",
        "- 每个动态股票池既作为可交易范围，也作为当日截面 rank 特征的计算截面。",
        f"- top-N：每次选取 `{top_n}` 只高波动股票。",
        f"- 最小历史窗口交易日数：`{min_window_days}`。",
        f"- 日度滚动窗口：过去 `{daily_lookback_days}` 个自然日，不含当日。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`，只做外推诊断。",
        f"- 测试集：`datetime >= {config.split_date.date()}`，用于最终比较。",
        f"- ST 剔除状态：`{st_status}`。",
        "",
        "## 两种动态股票池",
        "",
        f"- `{weekly_universe_id}`：每个交易周使用上一周周一到周五内日内分钟收益波动率均值最高的 top{top_n} 股票，并在本周所有交易日使用同一股票池。",
        f"- `{daily_universe_id}`：每个交易日使用过去 {daily_lookback_days} 个自然日内日内分钟收益波动率均值最高的 top{top_n} 股票，每天更新股票池。",
        "",
        "## 重要说明",
        "",
        "- 这两个股票池都是动态事前规则；与此前 `recent_week_high_vol_top300` 静态池不同，不使用测试末尾的一周固定覆盖整个测试集。",
        "- 当前回测仍未完整建模涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。",
        "- method2 CTA 仍是 method1 预测分的小权重趋势/突破包装，不应简单视为完全独立 alpha。",
        "",
        "## 策略结果",
        "",
        "| 股票池 | 方法 | 并集股票数 | 日均股票数 | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 信号期占比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["universe_id", "method_id"]).itertuples(index=False):
        lines.append(
            f"| `{row.universe_id}` | `{row.method_id}` | {int(row.union_symbol_count)} | "
            f"{float(row.avg_daily_symbol_count):.2f} | {float(row.validation_sharpe):.6f} | "
            f"{float(row.test_sharpe):.6f} | {float(row.test_annual_return):.6f} | "
            f"{float(row.test_max_drawdown):.6f} | {int(row.test_buy_trade_count)} | "
            f"{float(row.test_signal_period_ratio):.6f} |"
        )
    lines.extend(["", "## 特征缓存", ""])
    for row in cache_rows:
        lines.append(
            f"- `{row['universe_id']}`：rows=`{row['rows']}`，cache=`{row['cache_path']}`，"
            f"loaded_from_cache=`{row['loaded_from_cache']}`。"
        )
    lines.extend(
        [
            "",
            "## 产物说明",
            "",
            "- `strategy_summary.csv`：四组回测核心指标。",
            "- 每个动态股票池目录下保存每日成员、每日摘要、收益曲线、交易事件、平仓复盘、持仓和指标。",
        ]
    )
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dynamic_high_vol_backtests(args: argparse.Namespace) -> Path:
    """执行动态高波动股票池 method1/method2 回测。"""

    config = create_config(args)
    symbols, universe_audit, st_status = discover_mainboard_symbols(config, args.max_market_symbols)
    daily_volatility = build_daily_volatility_panel(config, symbols)
    evaluation_dates = get_evaluation_dates(daily_volatility, config)
    weekly_membership = build_weekly_membership(daily_volatility, evaluation_dates, args.top_n, args.min_window_days)
    daily_membership = build_daily_membership(
        daily_volatility,
        evaluation_dates,
        args.top_n,
        args.daily_lookback_days,
        args.min_window_days,
    )

    manifest = xb.read_feature_manifest(config.feature_manifest)
    model = joblib.load(config.method1_run_dir / "xgboost_model.joblib")
    feature_cols = (config.method1_run_dir / "feature_list.txt").read_text(encoding="utf-8").splitlines()
    xb_config = all_a.create_xb_config(config)
    strategies = create_strategy_specs(config)

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    universe_audit.to_csv(run_dir / "base_universe_audit.csv", index=False, encoding="utf-8-sig")
    daily_volatility.to_csv(run_dir / "daily_volatility_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    for universe_id, membership in [
        (f"weekly_previous_week_high_vol_top{args.top_n}", weekly_membership),
        (f"daily_rolling_{args.daily_lookback_days}d_high_vol_top{args.top_n}", daily_membership),
    ]:
        panel, cache_path, loaded_from_cache = build_or_load_dynamic_rank_panel(config, universe_id, membership, manifest)
        scored_panel = score_panel(panel, model, feature_cols)
        cache_rows.append(
            {
                "universe_id": universe_id,
                "rows": int(len(scored_panel)),
                "symbols": int(scored_panel["symbol"].nunique()),
                "cache_path": str(cache_path) if cache_path else "",
                "loaded_from_cache": bool(loaded_from_cache),
            }
        )
        summary_rows.extend(save_universe_outputs(run_dir, universe_id, membership, scored_panel, config, xb_config, strategies))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **{key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items() if key != "fee"},
                "fee": asdict(config.fee),
                "akshare_st_filter_status": st_status,
                "top_n": int(args.top_n),
                "daily_lookback_days": int(args.daily_lookback_days),
                "min_window_days": int(args.min_window_days),
                "max_market_symbols": int(args.max_market_symbols),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_final_report(run_dir, config, summary, cache_rows, st_status, args.top_n, args.daily_lookback_days, args.min_window_days)
    write_output_guide(run_dir)
    print(f"动态高波动 method 回测已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """命令行入口。"""

    run_dynamic_high_vol_backtests(parse_args())


if __name__ == "__main__":
    main()
