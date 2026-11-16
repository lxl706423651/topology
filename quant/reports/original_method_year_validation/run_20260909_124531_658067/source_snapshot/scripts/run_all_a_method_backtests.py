"""把既有 method1/method2 外推到全 A 主板股票池回测。

输入为 method1 已训练好的 XGBoost 模型、严格训练期特征清单、本地 A 股
分钟 OHLCV 数据，以及 method2 CTA 的固定规则元数据。本脚本不重新训练模型、
不在全 A 测试集上调参；它按本地全市场数据构造与 method1 一致的 156 个
截面 rank 特征，剔除北交所、科创板、创业板，并尽量通过 AKShare 当前名称
剔除 ST/退市风险股票。输出包括全 A 股票池审计、验证/测试指标、交易记录、
收益曲线和中文报告。原始行情数据不会被修改。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
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

from quant_backtest.factor_library import build_factor_library, calculate_forward_return
from quant_backtest.metrics import performance_metrics
import scripts.run_multi_factor_cta_backtests as cta
import scripts.run_xgboost_long_only_backtest as xb


OUTPUT_DIR = PROJECT_ROOT / "reports" / "all_a_method_backtests"
PANEL_CACHE_DIR = OUTPUT_DIR / "panel_cache"
METHOD1_RUN_DIR = PROJECT_ROOT / "reports" / "xgboost_long_only_backtest" / "run_20260526_202355"
METHOD2_RUN_DIR = PROJECT_ROOT / "reports" / "multi_factor_cta_backtests" / "run_20260527_192923"
CACHE_VERSION = "all_a_method_panel_v1_mainboard_non_st"
STOCK_NAME_CACHE_DIR = PROJECT_ROOT / "config" / "stock_name_cache"


@dataclass(frozen=True)
class FeeConfig:
    """A 股现货费用口径。

    broker_commission_rate: 券商佣金，按成交金额双边收取；本项目按用户指定万八。
    stamp_duty_rate: 股票卖出印花税，按卖出成交金额单边收取。
    transfer_fee_rate: A 股过户费，按成交金额双边收取。
    min_commission: 最低佣金；用户指定免 5，因此为 0。
    commission_includes_exchange_fees: 佣金是否已含经手费/证管费，避免重复计费。
    """

    broker_commission_rate: float
    stamp_duty_rate: float
    transfer_fee_rate: float
    min_commission: float
    commission_includes_exchange_fees: bool


@dataclass(frozen=True)
class AllAMethodConfig:
    """全 A method 外推回测配置。"""

    data_root: Path
    feature_manifest: Path
    method1_run_dir: Path
    method2_run_dir: Path
    output_dir: Path
    panel_cache_dir: Path
    validation_start: pd.Timestamp
    split_date: pd.Timestamp
    end_date: pd.Timestamp
    forward_horizon_bars: int
    evaluation_step_bars: int
    initial_cash: float
    min_assets_per_rebalance: int
    lot_size: int
    max_symbols: int
    workers: int
    rebuild_panel: bool
    no_panel_cache: bool
    use_akshare_st_filter: bool
    fee: FeeConfig


@dataclass(frozen=True)
class StrategySpec:
    """固定规则策略定义。"""

    method_id: str
    strategy_name: str
    score_col: str
    rule: xb.PortfolioRule
    description: str


def parse_args() -> argparse.Namespace:
    """解析全 A 外推回测参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=xb.DATA_ROOT)
    parser.add_argument("--feature-manifest", type=Path, default=xb.MODEL_FEATURE_MANIFEST)
    parser.add_argument("--method1-run-dir", type=Path, default=METHOD1_RUN_DIR)
    parser.add_argument("--method2-run-dir", type=Path, default=METHOD2_RUN_DIR)
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
    parser.add_argument("--max-symbols", type=int, default=0, help="烟测用股票数上限，0 表示使用所有主板非 ST 股票。")
    parser.add_argument("--workers", type=int, default=xb.WORKERS)
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--no-panel-cache", action="store_true")
    parser.add_argument("--no-akshare-st-filter", action="store_true", help="不联网读取当前股票名称剔除 ST。")
    parser.add_argument("--broker-commission-rate", type=float, default=0.0008, help="券商佣金，万八为 0.0008。")
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005, help="卖出印花税，0.5‰ 为 0.0005。")
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001, help="过户费，0.01‰ 为 0.00001。")
    parser.add_argument("--min-commission", type=float, default=0.0, help="最低佣金；免 5 时为 0。")
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> AllAMethodConfig:
    """校验参数并生成配置对象。"""

    if args.horizon < 1 or args.eval_step < 1:
        raise ValueError("--horizon 和 --eval-step 必须为正数。")
    if args.min_assets < 1:
        raise ValueError("--min-assets 必须为正数。")
    fee = FeeConfig(
        broker_commission_rate=float(args.broker_commission_rate),
        stamp_duty_rate=float(args.stamp_duty_rate),
        transfer_fee_rate=float(args.transfer_fee_rate),
        min_commission=float(args.min_commission),
        commission_includes_exchange_fees=True,
    )
    return AllAMethodConfig(
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
        max_symbols=max(int(args.max_symbols), 0),
        workers=max(int(args.workers), 1),
        rebuild_panel=bool(args.rebuild_panel),
        no_panel_cache=bool(args.no_panel_cache),
        use_akshare_st_filter=not bool(args.no_akshare_st_filter),
        fee=fee,
    )


def is_mainboard_symbol(symbol: str) -> bool:
    """判断是否为沪深主板 A 股代码，剔除北交所、科创板和创业板。"""

    return symbol.startswith(("SH.600", "SH.601", "SH.603", "SH.605", "SZ.000", "SZ.001", "SZ.002", "SZ.003"))


def normalize_symbol_code(symbol: str) -> str:
    """把 `SH.600000`/`SZ.000001` 转成 AKShare 使用的 6 位代码。"""

    return symbol.split(".", maxsplit=1)[-1]


def load_current_name_map(use_akshare: bool) -> tuple[dict[str, str], str]:
    """用 AKShare 读取当前 A 股代码名称，用于剔除名称含 ST 或退的股票。"""

    if not use_akshare:
        return {}, "disabled"
    cache_path = STOCK_NAME_CACHE_DIR / f"a_stock_names_{datetime.now():%Y%m%d}.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path, dtype={"code": str})
        return dict(zip(frame["code"].astype(str).str.zfill(6), frame["name"].astype(str), strict=False)), f"loaded_cache:{cache_path}"
    try:
        import akshare as ak  # type: ignore

        errors: list[str] = []
        for source_name, loader in [
            ("stock_zh_a_spot", ak.stock_zh_a_spot),
            ("stock_info_a_code_name", ak.stock_info_a_code_name),
        ]:
            try:
                frame = loader()
                if {"代码", "名称"}.issubset(frame.columns):
                    out = pd.DataFrame(
                        {
                            "code": frame["代码"].astype(str).str.extract(r"(\d{6})$", expand=False).str.zfill(6),
                            "name": frame["名称"].astype(str),
                        }
                    )
                else:
                    out = pd.DataFrame(
                        {
                            "code": frame["code"].astype(str).str.extract(r"(\d{6})$", expand=False).str.zfill(6),
                            "name": frame["name"].astype(str),
                        }
                    )
                out = out.dropna(subset=["code"]).drop_duplicates("code")
                STOCK_NAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                out.to_csv(cache_path, index=False, encoding="utf-8-sig")
                name_map = dict(zip(out["code"].astype(str).str.zfill(6), out["name"].astype(str), strict=False))
                return name_map, f"loaded:{source_name};cached:{cache_path}"
            except Exception as exc:  # pragma: no cover - 网络接口不稳定，只在报告中记录
                errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
        return {}, "failed: " + " | ".join(errors)
    except Exception as exc:  # pragma: no cover
        return {}, f"failed: import akshare: {type(exc).__name__}: {exc}"


def discover_all_a_mainboard_symbols(config: AllAMethodConfig) -> tuple[list[str], pd.DataFrame, str]:
    """发现本地全 A 主板股票池，并记录剔除原因。"""

    year_dirs = [config.data_root / "2025", config.data_root / "2026"]
    year_sets = []
    for year_dir in year_dirs:
        if not year_dir.exists():
            raise FileNotFoundError(f"找不到年份目录：{year_dir}")
        year_sets.append({path.stem for path in year_dir.glob("*.csv")})
    candidates = sorted(set.intersection(*year_sets))
    name_map, st_status = load_current_name_map(config.use_akshare_st_filter)
    rows: list[dict[str, object]] = []
    selected: list[str] = []
    for symbol in candidates:
        code = normalize_symbol_code(symbol)
        name = name_map.get(code, "")
        reason = ""
        if not is_mainboard_symbol(symbol):
            reason = "非沪深主板或属于北交所/科创板/创业板"
        elif name_map and ("ST" in name.upper() or "退" in name):
            reason = "当前名称含 ST 或退"
        if reason:
            rows.append({"symbol": symbol, "code": code, "name": name, "included": False, "exclude_reason": reason})
            continue
        rows.append({"symbol": symbol, "code": code, "name": name, "included": True, "exclude_reason": ""})
        selected.append(symbol)
    if config.max_symbols:
        selected_set = set(selected[: config.max_symbols])
        for row in rows:
            if row["included"] and row["symbol"] not in selected_set:
                row["included"] = False
                row["exclude_reason"] = "max_symbols 烟测截断"
        selected = selected[: config.max_symbols]
    if not selected:
        raise RuntimeError("全 A 主板股票池为空。")
    return selected, pd.DataFrame(rows), st_status


def create_xb_config(config: AllAMethodConfig) -> xb.ExperimentConfig:
    """生成复用主回测选择函数所需的配置对象。"""

    return xb.ExperimentConfig(
        data_root=config.data_root,
        stable_symbols_file=Path(""),
        feature_manifest=config.feature_manifest,
        output_dir=config.output_dir,
        feature_panel_cache_dir=config.panel_cache_dir,
        split_date=config.split_date,
        validation_start=config.validation_start,
        forward_horizon_bars=config.forward_horizon_bars,
        evaluation_step_bars=config.evaluation_step_bars,
        commission_rate=0.0,
        initial_cash=config.initial_cash,
        top_n_candidates=(1,),
        top_quantile_candidates=(0.02,),
        min_assets_per_rebalance=config.min_assets_per_rebalance,
        min_holding_count=1,
        hold_steps_candidates=(1,),
        target_exposure_candidates=(1.0,),
        confidence_quantile_candidates=(None,),
        spread_quantile_candidates=(None,),
        lot_size=config.lot_size,
        sharpe_target=3.0,
        max_symbols=config.max_symbols,
        workers=config.workers,
        random_seed=xb.RANDOM_SEED,
        rebuild_feature_panel=False,
        no_feature_panel_cache=False,
        build_feature_panel_only=False,
    )


def load_method_rule(path: Path) -> xb.PortfolioRule:
    """从 method1 `model_metadata.json` 读取固定组合规则。"""

    metadata = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
    rule = metadata["selected_rule"]
    return xb.PortfolioRule(
        rule_type=str(rule["rule_type"]),
        value=float(rule["value"]),
        hold_steps=int(rule["hold_steps"]),
        target_exposure=float(rule["target_exposure"]),
        confidence_quantile=rule.get("confidence_quantile"),
        confidence_threshold=rule.get("confidence_threshold"),
        spread_quantile=rule.get("spread_quantile"),
        spread_threshold=rule.get("spread_threshold"),
    )


def load_cta_rule(path: Path, family: str = "cta_trend_breakout") -> xb.PortfolioRule:
    """从 method2 CTA `strategy_metadata.json` 读取固定组合规则。"""

    metadata_path = path / family / "strategy_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rule = metadata["selected_rule"]
    return xb.PortfolioRule(
        rule_type=str(rule["rule_type"]),
        value=float(rule["value"]),
        hold_steps=int(rule["hold_steps"]),
        target_exposure=float(rule["target_exposure"]),
        confidence_quantile=rule.get("confidence_quantile"),
        confidence_threshold=rule.get("confidence_threshold"),
        spread_quantile=rule.get("spread_quantile"),
        spread_threshold=rule.get("spread_threshold"),
    )


def calculate_panel_cache_key(config: AllAMethodConfig, symbols: Sequence[str], manifest: pd.DataFrame) -> str:
    """生成全 A method 特征面板缓存键。"""

    payload = {
        "cache_version": CACHE_VERSION,
        "symbols": list(symbols),
        "feature_manifest": str(config.feature_manifest.resolve()),
        "manifest_hash": hashlib.sha256(
            manifest.loc[:, ["model_feature", "factor_base"]].to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "validation_start": str(config.validation_start),
        "split_date": str(config.split_date),
        "end_date": str(config.end_date),
        "horizon": config.forward_horizon_bars,
        "eval_step": config.evaluation_step_bars,
        "max_symbols": config.max_symbols,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def discover_year_dirs(data_root: Path) -> list[Path]:
    """返回排序后的年份数据目录。"""

    return sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit())


def build_symbol_base_panel(args: tuple[str, list[Path], list[str], AllAMethodConfig]) -> pd.DataFrame:
    """构造单只股票的基础因子采样面板。

    symbol: 股票代码。
    year_dirs: 本地年份目录，按全历史读取以保持采样相位。
    base_factors: 需要进入截面 rank 的基础因子列。
    config: 全 A method 配置，决定标签 horizon、采样步长和日期范围。
    """

    symbol, year_dirs, base_factors, config = args
    parts: list[pd.DataFrame] = []
    for year_dir in year_dirs:
        path = year_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path)
        part["symbol"] = symbol
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    data = pd.concat(parts, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["datetime"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values("datetime").drop_duplicates(["symbol", "datetime"], keep="last").reset_index(drop=True)
    factors = build_factor_library(data).reindex(columns=base_factors)
    future_return = calculate_forward_return(data, config.forward_horizon_bars)
    entry_datetime = data["datetime"].shift(-1)
    entry_price = data["close"].shift(-1)
    exit_price = data["close"].shift(-(config.forward_horizon_bars + 1))
    exit_datetime = data["datetime"].shift(-(config.forward_horizon_bars + 1))
    sampled = np.arange(len(data)) % config.evaluation_step_bars == 0
    date_mask = (data["datetime"] >= config.validation_start) & (data["datetime"] <= config.end_date + pd.Timedelta(days=1))
    keep = (
        sampled
        & date_mask
        & future_return.notna()
        & entry_datetime.notna()
        & entry_price.notna()
        & exit_price.notna()
        & exit_datetime.notna()
        & (entry_price > 0)
        & (exit_price > 0)
    )
    if not bool(keep.any()):
        return pd.DataFrame()
    base = data.loc[keep, ["datetime", "symbol", "close", "volume", "amount"]].copy()
    base["forward_return"] = future_return.loc[keep].to_numpy(dtype="float32")
    base["entry_datetime"] = pd.to_datetime(entry_datetime.loc[keep]).to_numpy()
    base["entry_price"] = entry_price.loc[keep].to_numpy(dtype="float32")
    base["exit_price"] = exit_price.loc[keep].to_numpy(dtype="float32")
    base["label_exit_datetime"] = pd.to_datetime(exit_datetime.loc[keep]).to_numpy()
    feature_part = factors.loc[keep].astype("float32")
    return pd.concat([base.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1)


def build_or_load_rank_panel(
    config: AllAMethodConfig,
    symbols: Sequence[str],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, Path | None, bool]:
    """构建或读取全 A 主板截面 rank 特征面板缓存。"""

    base_factors = manifest["factor_base"].astype(str).drop_duplicates().tolist()
    cache_key = calculate_panel_cache_key(config, symbols, manifest)
    config.panel_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.panel_cache_dir / f"all_a_method_panel_{cache_key[:16]}.pkl"
    metadata_path = config.panel_cache_dir / f"all_a_method_panel_{cache_key[:16]}.metadata.json"
    if cache_path.exists() and metadata_path.exists() and not config.rebuild_panel and not config.no_panel_cache:
        print(f"正在读取全 A method 特征缓存：{cache_path}", flush=True)
        return pd.read_pickle(cache_path), cache_path, True

    year_dirs = discover_year_dirs(config.data_root)
    frames: list[pd.DataFrame] = []
    print(f"正在构建全 A 主板基础因子：symbols={len(symbols)}, workers={config.workers}", flush=True)
    if config.workers == 1:
        for idx, symbol in enumerate(symbols, start=1):
            frame = build_symbol_base_panel((symbol, year_dirs, base_factors, config))
            if not frame.empty:
                frames.append(frame)
            if idx % 100 == 0 or idx == len(symbols):
                print(f"已构建基础因子 {idx}/{len(symbols)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(build_symbol_base_panel, (symbol, year_dirs, base_factors, config)): symbol
                for symbol in symbols
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                if idx % 100 == 0 or idx == len(symbols):
                    print(f"已构建基础因子 {idx}/{len(symbols)}", flush=True)
    if not frames:
        raise RuntimeError("没有构建出任何全 A method 样本。")
    base_panel = pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    print(f"正在计算截面 rank 特征：rows={len(base_panel)}, factors={len(base_factors)}", flush=True)
    ranked = base_panel.groupby("datetime", sort=False)[base_factors].rank(method="average", pct=True).astype("float32")
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
    panel = pd.concat([base_panel[keep_cols], ranked], axis=1).replace([np.inf, -np.inf], np.nan)
    if not config.no_panel_cache:
        panel.to_pickle(cache_path)
        metadata = {
            "cache_version": CACHE_VERSION,
            "rows": int(len(panel)),
            "symbol_count": int(panel["symbol"].nunique()),
            "start": str(panel["datetime"].min()),
            "end": str(panel["datetime"].max()),
            "feature_count": int(len(ranked.columns)),
            "cache_path": str(cache_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return panel, cache_path, False
    return panel, None, False


def calculate_trade_cost(side: str, traded_value: float, fee: FeeConfig) -> float:
    """按买卖方向计算交易成本。"""

    commission = traded_value * fee.broker_commission_rate
    if fee.min_commission > 0 and traded_value > 0:
        commission = max(commission, fee.min_commission)
    transfer = traded_value * fee.transfer_fee_rate
    stamp = traded_value * fee.stamp_duty_rate if side == "卖出" else 0.0
    return float(commission + transfer + stamp)


def reduce_buy_plan_to_cash(
    buy_plan: dict[str, int],
    entry_prices: dict[str, float],
    cash: float,
    config: AllAMethodConfig,
) -> dict[str, int]:
    """在现金和精确买入费用约束下缩减买入计划。"""

    plan = {symbol: shares for symbol, shares in buy_plan.items() if shares > 0}
    while plan:
        required = sum(
            plan[symbol] * entry_prices[symbol] + calculate_trade_cost("买入", plan[symbol] * entry_prices[symbol], config.fee)
            for symbol in plan
        )
        if required <= cash + 1e-6:
            return plan
        largest = max(plan, key=lambda symbol: plan[symbol] * entry_prices[symbol])
        reduced = plan[largest] - config.lot_size
        if reduced > 0:
            plan[largest] = reduced
        else:
            del plan[largest]
    return {}


def sell_position_lots(
    position_lots: dict[str, list[xb.PositionLot]],
    symbol: str,
    shares_to_sell: int,
    sell_datetime: pd.Timestamp,
    sell_price: float,
    reason: str,
    config: AllAMethodConfig,
) -> tuple[int, float, float, list[dict[str, object]], list[dict[str, object]]]:
    """按 FIFO 卖出满足 T+1 的持仓批次，并按新费用口径生成复盘记录。"""

    lots = position_lots.get(symbol, [])
    if not lots or shares_to_sell <= 0:
        return 0, 0.0, 0.0, [], []
    execution_date = pd.Timestamp(sell_datetime).date()
    remaining = int(shares_to_sell)
    sold_shares = 0
    sell_value = 0.0
    sell_cost = 0.0
    events: list[dict[str, object]] = []
    round_trips: list[dict[str, object]] = []
    for lot in list(lots):
        if remaining <= 0:
            break
        if lot.buy_date >= execution_date or lot.shares <= 0:
            continue
        quantity = min(lot.shares, remaining)
        value = quantity * sell_price
        proportional_buy_cost = lot.buy_cost * quantity / lot.shares
        cost = calculate_trade_cost("卖出", value, config.fee)
        gross_pnl = (sell_price - lot.buy_price) * quantity
        net_pnl = gross_pnl - proportional_buy_cost - cost
        denominator = lot.buy_price * quantity + proportional_buy_cost
        round_trips.append(
            {
                "股票代码(symbol)": symbol,
                "买入时间(buy_datetime)": lot.buy_datetime,
                "卖出时间(sell_datetime)": sell_datetime,
                "持有天数(holding_days)": (pd.Timestamp(sell_datetime).date() - lot.buy_date).days,
                "买入价格(buy_price)": lot.buy_price,
                "卖出价格(sell_price)": sell_price,
                "股数(shares)": quantity,
                "手数(lots)": quantity / config.lot_size,
                "买入原因(buy_reason)": lot.reason,
                "卖出原因(sell_reason)": reason,
                "买入信号时间(signal_datetime)": lot.signal_datetime,
                "买入预测分(prediction)": lot.prediction,
                "毛收益(gross_pnl)": gross_pnl,
                "交易成本(cost)": proportional_buy_cost + cost,
                "净收益(net_pnl)": net_pnl,
                "单笔收益率(return)": net_pnl / denominator if denominator else 0.0,
            }
        )
        lot.shares -= quantity
        lot.buy_cost -= proportional_buy_cost
        remaining -= quantity
        sold_shares += quantity
        sell_value += value
        sell_cost += cost
    position_lots[symbol] = [lot for lot in lots if lot.shares > 0]
    if sold_shares > 0:
        events.append(
            {
                "交易时间(execution_datetime)": sell_datetime,
                "股票代码(symbol)": symbol,
                "方向(side)": "卖出",
                "股数(shares)": sold_shares,
                "手数(lots)": sold_shares / config.lot_size,
                "价格(price)": sell_price,
                "成交金额(traded_value)": sell_value,
                "交易成本(cost)": sell_cost,
                "触发原因(trigger_reason)": reason,
            }
        )
    return sold_shares, sell_value, sell_cost, events, round_trips


def run_long_only_backtest_with_fees(
    predictions: pd.DataFrame,
    rule: xb.PortfolioRule,
    xb_config: xb.ExperimentConfig,
    config: AllAMethodConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按 signal_lag=1、T+1、100 股整数手和实盘费用口径回测。"""

    rows: list[dict[str, object]] = []
    holdings: list[pd.DataFrame] = []
    trade_events: list[dict[str, object]] = []
    round_trips: list[dict[str, object]] = []
    position_lots: dict[str, list[xb.PositionLot]] = {}
    last_prices: dict[str, float] = {}
    cash = float(config.initial_cash)
    rebalance_count = 0
    grouped = list(predictions.groupby("datetime", sort=True))
    for period_index, (dt, group) in enumerate(grouped):
        tradable = group.dropna(subset=["entry_datetime", "entry_price", "exit_price"]).copy()
        tradable["entry_datetime"] = pd.to_datetime(tradable["entry_datetime"])
        entry_prices = dict(zip(tradable["symbol"].astype(str), tradable["entry_price"].astype(float), strict=False))
        exit_prices = dict(zip(tradable["symbol"].astype(str), tradable["exit_price"].astype(float), strict=False))
        valuation_entry_prices = {**last_prices, **entry_prices}
        valuation_exit_prices = {**last_prices, **exit_prices}
        entry_times = dict(zip(tradable["symbol"].astype(str), tradable["entry_datetime"], strict=False))
        predictions_by_symbol = dict(zip(tradable["symbol"].astype(str), tradable["prediction"].astype(float), strict=False))
        default_execution_time = pd.Timestamp(tradable["entry_datetime"].min()) if not tradable.empty else pd.Timestamp(dt)
        current_shares = xb.summarize_position_lots(position_lots)
        pre_trade_equity = cash + xb.value_position_lots(position_lots, valuation_entry_prices)
        should_rebalance = (period_index % rule.hold_steps == 0) or not current_shares

        buy_value = sell_value = turnover_value = cost_value = 0.0
        if should_rebalance:
            selected = xb.select_symbols_for_rule(tradable, rule, xb_config)
            desired_shares = xb.create_desired_shares(selected, pre_trade_equity, rule, xb_config)
            current_shares = xb.summarize_position_lots(position_lots)
            selected_symbols = set(desired_shares)
            for symbol, current_quantity in current_shares.items():
                desired_quantity = desired_shares.get(symbol, 0)
                quantity_to_sell = max(0, current_quantity - desired_quantity)
                if quantity_to_sell <= 0 or symbol not in entry_prices:
                    continue
                sell_time = pd.Timestamp(entry_times.get(symbol, default_execution_time))
                reason = (
                    "调仓减仓：目标股数下降，且持仓已满足 T+1"
                    if symbol in selected_symbols
                    else "调仓平仓：股票未进入当前目标组合，且持仓已满足 T+1"
                )
                sold, value, cost, events, trips = sell_position_lots(
                    position_lots,
                    symbol,
                    quantity_to_sell,
                    sell_time,
                    float(entry_prices[symbol]),
                    reason,
                    config,
                )
                if sold > 0:
                    sell_value += value
                    cost_value += cost
                    cash += value - cost
                    trade_events.extend(events)
                    round_trips.extend(trips)

            current_after_sell = xb.summarize_position_lots(position_lots)
            raw_buy_plan = {
                symbol: desired_quantity - current_after_sell.get(symbol, 0)
                for symbol, desired_quantity in desired_shares.items()
                if desired_quantity > current_after_sell.get(symbol, 0) and symbol in entry_prices
            }
            buy_plan = reduce_buy_plan_to_cash(raw_buy_plan, entry_prices, cash, config)
            selected_predictions = dict(zip(selected["symbol"].astype(str), selected["prediction"].astype(float), strict=False))
            for symbol, quantity in buy_plan.items():
                price = float(entry_prices[symbol])
                buy_time = pd.Timestamp(entry_times.get(symbol, default_execution_time))
                value = quantity * price
                cost = calculate_trade_cost("买入", value, config.fee)
                if value + cost > cash + 1e-6:
                    continue
                reason = f"买入：预测分进入当前规则 {rule.name} 的目标组合"
                lot = xb.PositionLot(
                    symbol=symbol,
                    shares=int(quantity),
                    buy_datetime=buy_time,
                    buy_date=buy_time.date(),
                    buy_price=price,
                    buy_cost=cost,
                    signal_datetime=pd.Timestamp(dt),
                    prediction=float(selected_predictions.get(symbol, predictions_by_symbol.get(symbol, np.nan))),
                    reason=reason,
                )
                position_lots.setdefault(symbol, []).append(lot)
                cash -= value + cost
                buy_value += value
                cost_value += cost
                trade_events.append(
                    {
                        "交易时间(execution_datetime)": buy_time,
                        "股票代码(symbol)": symbol,
                        "方向(side)": "买入",
                        "股数(shares)": int(quantity),
                        "手数(lots)": quantity / config.lot_size,
                        "价格(price)": price,
                        "成交金额(traded_value)": value,
                        "交易成本(cost)": cost,
                        "触发原因(trigger_reason)": reason,
                    }
                )
            turnover_value = buy_value + sell_value
            rebalance_count += int(turnover_value > 0)

        current_shares = xb.summarize_position_lots(position_lots)
        entry_position_value = xb.value_position_lots(position_lots, valuation_entry_prices)
        exit_position_value = xb.value_position_lots(position_lots, valuation_exit_prices)
        gross_end_equity = pre_trade_equity - entry_position_value + exit_position_value
        end_equity = cash + exit_position_value
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "gross_return": gross_end_equity / pre_trade_equity - 1.0 if pre_trade_equity > 0 else 0.0,
                "cost_return": cost_value / pre_trade_equity if pre_trade_equity > 0 else 0.0,
                "return": end_equity / pre_trade_equity - 1.0 if pre_trade_equity > 0 else 0.0,
                "turnover": turnover_value / pre_trade_equity if pre_trade_equity > 0 else 0.0,
                "buy_turnover": buy_value / pre_trade_equity if pre_trade_equity > 0 else 0.0,
                "sell_turnover": sell_value / pre_trade_equity if pre_trade_equity > 0 else 0.0,
                "holding_count": len(current_shares),
                "gross_exposure": entry_position_value / pre_trade_equity if pre_trade_equity > 0 else 0.0,
                "cash_weight": cash / pre_trade_equity if pre_trade_equity > 0 else 0.0,
                "short_exposure": 0.0,
                "equity": end_equity,
            }
        )

        if current_shares:
            held = tradable.loc[tradable["symbol"].astype(str).isin(current_shares)].copy()
            if not held.empty:
                held["shares"] = held["symbol"].astype(str).map(current_shares).astype(int)
                held["entry_value"] = held["shares"] * held["entry_price"].astype(float)
                held["exit_value"] = held["shares"] * held["exit_price"].astype(float)
                held["entry_weight"] = held["entry_value"] / pre_trade_equity if pre_trade_equity > 0 else 0.0
                held["rule"] = rule.name
                holdings.append(
                    held[
                        [
                            "datetime",
                            "symbol",
                            "prediction",
                            "forward_return",
                            "entry_price",
                            "exit_price",
                            "shares",
                            "entry_value",
                            "exit_value",
                            "entry_weight",
                            "rule",
                        ]
                    ].copy()
                )
        for symbol, price in exit_prices.items():
            if np.isfinite(price) and price > 0:
                last_prices[symbol] = float(price)

    curve = pd.DataFrame(rows).sort_values("datetime")
    if curve.empty:
        return curve, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    curve["cumulative_return"] = curve["equity"] / config.initial_cash - 1.0
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    curve = curve.set_index("datetime")
    annualization = 252 * 240 / config.evaluation_step_bars
    metrics: dict[str, Any] = performance_metrics(curve["equity"], curve["return"], annualization=annualization)
    metrics.update(
        {
            "rule": rule.name,
            "gross_total_return": float((1 + curve["gross_return"].fillna(0)).prod() - 1),
            "total_cost_return": float(curve["cost_return"].sum()),
            "avg_turnover": float(curve["turnover"].mean()),
            "total_turnover": float(curve["turnover"].sum()),
            "avg_buy_turnover": float(curve["buy_turnover"].mean()),
            "avg_sell_turnover": float(curve["sell_turnover"].mean()),
            "avg_holding_count": float(curve["holding_count"].mean()),
            "avg_gross_exposure": float(curve["gross_exposure"].mean()),
            "avg_cash_weight": float(curve["cash_weight"].mean()),
            "short_exposure": 0.0,
            "broker_commission_rate": float(config.fee.broker_commission_rate),
            "stamp_duty_rate": float(config.fee.stamp_duty_rate),
            "transfer_fee_rate": float(config.fee.transfer_fee_rate),
            "min_commission": float(config.fee.min_commission),
            "signal_lag_bars": 1,
            "rebalance_horizon_bars": int(config.forward_horizon_bars),
            "hold_steps": int(rule.hold_steps),
            "target_exposure": float(rule.target_exposure),
            "confidence_quantile": rule.confidence_quantile,
            "confidence_threshold": rule.confidence_threshold,
            "spread_quantile": rule.spread_quantile,
            "spread_threshold": rule.spread_threshold,
            "lot_size": int(config.lot_size),
            "trade_count": int(len([event for event in trade_events if event.get("方向(side)") in {"买入", "卖出"}])),
            "buy_trade_count": int(len([event for event in trade_events if event.get("方向(side)") == "买入"])),
            "sell_trade_count": int(len([event for event in trade_events if event.get("方向(side)") == "卖出"])),
            "rebalance_count": int(rebalance_count),
            "t_plus_1_sell_rule": "enabled",
        }
    )
    holdings_frame = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    trade_events_frame = pd.DataFrame(trade_events)
    round_trips_frame = pd.DataFrame(round_trips)
    open_rows: list[dict[str, object]] = []
    final_prices = grouped[-1][1].set_index("symbol")["exit_price"].astype(float).to_dict() if grouped else {}
    for symbol, lots in position_lots.items():
        for lot in lots:
            if lot.shares <= 0:
                continue
            price = float(final_prices.get(symbol, lot.buy_price))
            open_rows.append(
                {
                    "股票代码(symbol)": symbol,
                    "买入时间(buy_datetime)": lot.buy_datetime,
                    "股数(shares)": lot.shares,
                    "手数(lots)": lot.shares / config.lot_size,
                    "买入价格(buy_price)": lot.buy_price,
                    "期末估值价格(final_price)": price,
                    "期末市值(final_value)": lot.shares * price,
                    "买入原因(buy_reason)": lot.reason,
                }
            )
    open_positions_frame = pd.DataFrame(open_rows)
    if not round_trips_frame.empty:
        metrics["round_trip_count"] = int(len(round_trips_frame))
        metrics["round_trip_win_rate"] = float((round_trips_frame["净收益(net_pnl)"].astype(float) > 0).mean())
        metrics["avg_round_trip_return"] = float(round_trips_frame["单笔收益率(return)"].astype(float).mean())
    else:
        metrics["round_trip_count"] = 0
        metrics["round_trip_win_rate"] = 0.0
        metrics["avg_round_trip_return"] = 0.0
    metrics["total_trade_lots"] = (
        float(trade_events_frame["手数(lots)"].astype(float).sum()) if not trade_events_frame.empty else 0.0
    )
    return curve, holdings_frame, trade_events_frame, round_trips_frame, open_positions_frame, metrics


def calculate_signal_frequency(frame: pd.DataFrame, rule: xb.PortfolioRule, xb_config: xb.ExperimentConfig) -> dict[str, Any]:
    """统计固定规则在样本中的触发频率。"""

    signal_rows = []
    for dt, group in frame.groupby("datetime", sort=True):
        selected = xb.select_symbols_for_rule(group, rule, xb_config)
        signal_rows.append({"datetime": pd.Timestamp(dt), "selected_count": int(len(selected))})
    signals = pd.DataFrame(signal_rows)
    if signals.empty:
        return {"period_count": 0, "signal_period_count": 0, "signal_period_ratio": 0.0, "max_no_signal_periods": 0}
    active = signals["selected_count"] > 0
    max_no_signal = 0
    current = 0
    for value in active:
        if value:
            max_no_signal = max(max_no_signal, current)
            current = 0
        else:
            current += 1
    max_no_signal = max(max_no_signal, current)
    return {
        "period_count": int(len(signals)),
        "signal_period_count": int(active.sum()),
        "signal_period_ratio": float(active.mean()),
        "max_no_signal_periods": int(max_no_signal),
    }


def save_strategy_outputs(
    run_dir: Path,
    strategy: StrategySpec,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    xb_config: xb.ExperimentConfig,
    config: AllAMethodConfig,
) -> dict[str, Any]:
    """保存单个 method 的验证诊断和测试回测产物。"""

    strategy_dir = run_dir / strategy.method_id
    strategy_dir.mkdir(parents=True, exist_ok=True)
    validation_eval = validation.copy()
    test_eval = test.copy()
    validation_eval["prediction"] = validation_eval[strategy.score_col].astype("float32")
    test_eval["prediction"] = test_eval[strategy.score_col].astype("float32")
    validation_curve, _, _, _, _, validation_metrics = run_long_only_backtest_with_fees(
        validation_eval,
        strategy.rule,
        xb_config,
        config,
    )
    curve, holdings, trades, round_trips, open_positions, metrics = run_long_only_backtest_with_fees(
        test_eval,
        strategy.rule,
        xb_config,
        config,
    )
    validation_metrics.update(calculate_signal_frequency(validation_eval, strategy.rule, xb_config))
    metrics.update(calculate_signal_frequency(test_eval, strategy.rule, xb_config))
    metrics.update(
        {
            "method_id": strategy.method_id,
            "strategy_name": strategy.strategy_name,
            "strategy_description": strategy.description,
            "score_col": strategy.score_col,
            "test_start": str(test_eval["datetime"].min()),
            "test_end": str(test_eval["datetime"].max()),
            "test_rows": int(len(test_eval)),
            "test_symbol_count": int(test_eval["symbol"].nunique()),
        }
    )
    validation_metrics.update({"method_id": strategy.method_id, "strategy_name": strategy.strategy_name})
    validation_curve.to_csv(strategy_dir / "validation_equity_curve.csv", encoding="utf-8-sig")
    pd.Series(validation_metrics).to_csv(strategy_dir / "validation_metrics.csv", header=["value"])
    curve.to_csv(strategy_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    holdings.to_csv(strategy_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(strategy_dir / "test_trade_events.csv", index=False, encoding="utf-8-sig")
    round_trips.to_csv(strategy_dir / "test_round_trips.csv", index=False, encoding="utf-8-sig")
    open_positions.to_csv(strategy_dir / "test_open_positions.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(strategy_dir / "test_metrics.csv", header=["value"])
    xb.plot_test_return_curve(strategy_dir, curve, metrics)
    xb.write_output_guides(strategy_dir)
    return {
        "method_id": strategy.method_id,
        "strategy_name": strategy.strategy_name,
        "validation_sharpe": validation_metrics.get("sharpe"),
        "validation_total_return": validation_metrics.get("total_return"),
        "validation_max_drawdown": validation_metrics.get("max_drawdown"),
        "validation_trade_count": validation_metrics.get("trade_count"),
        "validation_signal_period_ratio": validation_metrics.get("signal_period_ratio"),
        "test_sharpe": metrics.get("sharpe"),
        "test_annual_return": metrics.get("annual_return"),
        "test_annual_volatility": metrics.get("annual_volatility"),
        "test_total_return": metrics.get("total_return"),
        "test_max_drawdown": metrics.get("max_drawdown"),
        "test_trade_count": metrics.get("trade_count"),
        "test_buy_trade_count": metrics.get("buy_trade_count"),
        "test_round_trip_win_rate": metrics.get("round_trip_win_rate"),
        "test_total_trade_lots": metrics.get("total_trade_lots"),
        "test_signal_period_ratio": metrics.get("signal_period_ratio"),
        "test_max_no_signal_periods": metrics.get("max_no_signal_periods"),
        "rule": strategy.rule.name,
        "confidence_threshold": strategy.rule.confidence_threshold,
        "spread_threshold": strategy.rule.spread_threshold,
    }


def write_report(
    run_dir: Path,
    config: AllAMethodConfig,
    universe_audit: pd.DataFrame,
    st_status: str,
    cache_path: Path | None,
    loaded_from_cache: bool,
    summary: pd.DataFrame,
) -> None:
    """写出全 A method 外推报告。"""

    included = universe_audit.loc[universe_audit["included"] == True]  # noqa: E712
    lines = [
        "# Method1/Method2 全 A 主板外推回测报告",
        "",
        "## 实验边界",
        "",
        "- 本报告不重新训练模型、不重新选择门控阈值，只把既有 method1/method2 固定规则迁移到全 A 主板股票池。",
        "- 股票池：本地 2025/2026 都存在的上交所和深交所主板 A 股；剔除北交所、科创板 688、创业板 300/301。",
        "- 注意：股票池不是严格点时成分，ST 剔除使用当前名称缓存，可能低估历史 ST、退市和幸存者偏差风险。",
        f"- ST 剔除状态：`{st_status}`。若为 failed/disabled，则本次报告只能保证代码前缀剔除，不能保证完全剔除 ST。",
        f"- 入选股票数：`{len(included)}`。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`，只做外推诊断。",
        f"- 测试集：`datetime >= {config.split_date.date()}`，用于最终外推验收。",
        f"- 特征缓存：`{cache_path}`，loaded_from_cache=`{loaded_from_cache}`。",
        "",
        "## 费用口径",
        "",
        f"- 券商佣金：`{config.fee.broker_commission_rate}`，即万八，双边收取。",
        f"- 最低佣金：`{config.fee.min_commission}`，用户指定免 5。",
        f"- 卖出印花税：`{config.fee.stamp_duty_rate}`。",
        f"- 过户费：`{config.fee.transfer_fee_rate}`，双边收取。",
        "- 本报告默认券商佣金已包含经手费/证管费，因此不再单独叠加规费，避免重复计费。",
        "",
        "## 策略结果",
        "",
        "| 方法 | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试交易事件 | 测试信号期占比 | 最长无信号采样期 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.method_id}` | {float(row.validation_sharpe):.6f} | {float(row.test_sharpe):.6f} | "
            f"{float(row.test_annual_return):.6f} | {float(row.test_max_drawdown):.6f} | "
            f"{int(row.test_trade_count)} | {float(row.test_signal_period_ratio):.6f} | {int(row.test_max_no_signal_periods)} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- 本报告是旧模型和旧阈值的全 A 外推，不是全 A 重新训练，也没有根据全 A 测试集反向调参。",
            "- 如果全 A 外推显著强于沪深 300，优先检查幸存者偏差、当前 ST 过滤、停牌/涨跌停、复权口径和成交容量，而不是直接放大实盘仓位。",
            "- 当前回测未完整建模涨跌停不能成交、停牌、冲击成本和成交额容量；top1 满仓路径对单票可成交规模非常敏感。",
            "- 如果交易次数很少或信号期占比很低，通常是因为既有 top1 分数阈值和分差阈值来自沪深 300 验证集，迁移到全 A 后触发条件更苛刻或分布发生偏移。",
            "- 后续如要做全 A 实盘，应在新的训练/验证协议下重新训练全 A 模型或做 walk-forward，不能根据本次测试表现反向改阈值。",
            "",
            "## 产物说明",
            "",
            "- `resolved_config.json`：本次配置和费用口径。",
            "- `universe_audit.csv`：股票池入选/剔除原因。",
            "- `split_audit.csv`：验证集和测试集边界。",
            "- `strategy_summary.csv`：method1/method2 验证和测试核心指标。",
            "- `method1_xgboost_long_only/`、`method2_cta_trend_breakout/`：各方法收益曲线、交易事件、平仓复盘、持仓和指标。",
        ]
    )
    text = "\n".join(lines) + "\n"
    (run_dir / "final_report.md").write_text(text, encoding="utf-8")
    (run_dir / "all_a_method_report.md").write_text(text, encoding="utf-8")


def write_output_guide(run_dir: Path) -> None:
    """写出根目录输出文件说明。"""

    text = """# 输出文件说明

- `final_report.md`：标准入口报告，汇总全 A 主板外推边界、费用、指标和结论。
- `all_a_method_report.md`：同内容报告副本，保留实验命名。
- `resolved_config.json`：解析后的路径、日期、费用、缓存和股票池配置。
- `universe_audit.csv`：每只本地股票是否纳入、股票名称和剔除原因。
- `split_audit.csv`：验证集和测试集样本边界。
- `strategy_summary.csv`：method1/method2 的验证和测试指标汇总。
- `method1_xgboost_long_only/`：method1 固定 XGBoost top1 门控策略的外推回测产物。
- `method2_cta_trend_breakout/`：method2 CTA trend 包装策略的外推回测产物。
"""
    (run_dir / "输出文件说明.md").write_text(text, encoding="utf-8")


def run_all_a_method_backtests(args: argparse.Namespace) -> Path:
    """执行全 A 主板 method1/method2 外推回测。"""

    config = create_config(args)
    manifest = xb.read_feature_manifest(config.feature_manifest)
    symbols, universe_audit, st_status = discover_all_a_mainboard_symbols(config)
    panel, cache_path, loaded_from_cache = build_or_load_rank_panel(config, symbols, manifest)
    feature_cols = (config.method1_run_dir / "feature_list.txt").read_text(encoding="utf-8").splitlines()
    missing = [col for col in feature_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"全 A 特征面板缺少模型特征：{missing[:10]}")
    model = joblib.load(config.method1_run_dir / "xgboost_model.joblib")
    print(f"正在预测全 A method1 分数：rows={len(panel)}, features={len(feature_cols)}", flush=True)
    panel["prediction"] = model.predict(panel.loc[:, feature_cols]).astype("float32")
    panel = cta.add_cta_factor_scores(panel, cta.SCORE_BLEND_STRENGTH)

    validation = panel.loc[(panel["datetime"] >= config.validation_start) & (panel["datetime"] < config.split_date)].copy()
    test = panel.loc[panel["datetime"] >= config.split_date].copy()
    if validation.empty or test.empty:
        raise ValueError(f"切分后样本为空：validation={len(validation)} test={len(test)}")

    method1_rule = load_method_rule(config.method1_run_dir)
    method2_rule = load_cta_rule(config.method2_run_dir, "cta_trend_breakout")
    xb_config = create_xb_config(config)
    strategies = [
        StrategySpec(
            "method1_xgboost_long_only",
            "method1 XGBoost 156因子 long-only top1门控",
            "prediction",
            method1_rule,
            "复用 run_20260526_202355 训练好的 XGBoost 模型和沪深300验证集选出的 top1 分数/分差门控。",
        ),
        StrategySpec(
            "method2_cta_trend_breakout",
            "method2 CTA trend breakout",
            "cta_trend_breakout_score",
            method2_rule,
            "复用 method2 CTA trend 包装：XGBoost 分数 + 小权重趋势/突破/动量/量价确认叠加项。",
        ),
    ]

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **{key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items() if key != "fee"},
                "fee": asdict(config.fee),
                "cache_path": str(cache_path) if cache_path else None,
                "loaded_from_cache": loaded_from_cache,
                "akshare_st_filter_status": st_status,
                "method1_rule": asdict(method1_rule),
                "method2_rule": asdict(method2_rule),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    universe_audit.to_csv(run_dir / "universe_audit.csv", index=False, encoding="utf-8-sig")
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
    ).to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for strategy in strategies:
        print(f"正在回测全 A 外推策略：{strategy.method_id}", flush=True)
        summary_rows.append(save_strategy_outputs(run_dir, strategy, validation, test, xb_config, config))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    write_report(run_dir, config, universe_audit, st_status, cache_path, loaded_from_cache, summary)
    write_output_guide(run_dir)
    print(f"全 A method 外推回测已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """命令行入口。"""

    run_all_a_method_backtests(parse_args())


if __name__ == "__main__":
    main()
