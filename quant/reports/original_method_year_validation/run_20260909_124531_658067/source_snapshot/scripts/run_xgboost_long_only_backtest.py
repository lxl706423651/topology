"""训练 XGBoost alpha 模型并执行 A 股 long-only 样本外回测。

必需输入包括已解压的 A 股分钟 OHLCV 文件、稳定股票池报告，以及
``audit_factor_feature_library`` 生成的最终特征清单。本脚本使用
``quant_backtest.factor_library`` 构造因子，只在 ``datetime < 2025-05-01``
的数据上训练，在训练期内部验证集上选择 long-only top-N/top-quantile
组合规则，再用完整训练期重训模型，最后在 ``datetime >= 2025-05-01``
上评估。脚本会将可复用的特征面板缓存为 pkl，并将模型、预测、
持仓、权益曲线和报告写入 ``reports/xgboost_long_only_backtest``，
不会修改原始行情数据。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

try:  # pragma: no cover - 绘图只在报告生成阶段使用
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except Exception:  # pragma: no cover
    plt = None
    font_manager = None

try:  # pragma: no cover - 预期在 quant conda 环境中可用
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.factor_library import build_factor_library, calculate_forward_return, validate_no_future_leakage
from quant_backtest.metrics import performance_metrics


DATA_ROOT = PROJECT_ROOT / "ashare_minute_data_2023_2026_04" / "extracted"
STABLE_SYMBOLS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "stable_hs300_factor_correlation"
    / "run_20260518_223759"
    / "stable_hs300_symbols.csv"
)
MODEL_FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "reports"
    / "factor_feature_audit"
    / "run_20260525_161905"
    / "model_feature_manifest.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "xgboost_long_only_backtest"
FEATURE_PANEL_CACHE_DIR = PROJECT_ROOT / "reports" / "feature_panel_cache"
FEATURE_PANEL_CACHE_VERSION = "feature_panel_v1_t1_entry_rank"

SPLIT_DATE = "2025-05-01"
VALIDATION_START = "2024-11-01"
FORWARD_HORIZON_BARS = 24
EVALUATION_STEP_BARS = 24
COMMISSION_RATE = 0.0001
INITIAL_CASH = 1_000_000.0
TOP_N_CANDIDATES = (10, 20, 30, 50)
TOP_QUANTILE_CANDIDATES = (0.05, 0.10, 0.20)
MIN_ASSETS_PER_REBALANCE = 30
MIN_HOLDING_COUNT = 2
HOLD_STEPS_CANDIDATES = (1, 2, 3, 5, 10, 20)
TARGET_EXPOSURE_CANDIDATES = (1.0,)
CONFIDENCE_QUANTILE_CANDIDATES: tuple[float | None, ...] = (None,)
SPREAD_QUANTILE_CANDIDATES: tuple[float | None, ...] = (None,)
LOT_SIZE = 100
SHARPE_TARGET = 2.0
RANDOM_SEED = 42
WORKERS = min(4, os.cpu_count() or 1)


@dataclass(frozen=True)
class ExperimentConfig:
    """XGBoost 训练和 long-only 回测的解析后配置。"""

    data_root: Path
    stable_symbols_file: Path
    feature_manifest: Path
    output_dir: Path
    feature_panel_cache_dir: Path
    split_date: pd.Timestamp
    validation_start: pd.Timestamp
    forward_horizon_bars: int
    evaluation_step_bars: int
    commission_rate: float
    initial_cash: float
    top_n_candidates: tuple[int, ...]
    top_quantile_candidates: tuple[float, ...]
    min_assets_per_rebalance: int
    min_holding_count: int
    hold_steps_candidates: tuple[int, ...]
    target_exposure_candidates: tuple[float, ...]
    confidence_quantile_candidates: tuple[float | None, ...]
    spread_quantile_candidates: tuple[float | None, ...]
    lot_size: int
    sharpe_target: float
    max_symbols: int
    workers: int
    random_seed: int
    rebuild_feature_panel: bool
    no_feature_panel_cache: bool
    build_feature_panel_only: bool


def parse_args() -> argparse.Namespace:
    """解析实验参数和路径覆盖项。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--stable-symbols-file", type=Path, default=STABLE_SYMBOLS_FILE)
    parser.add_argument("--feature-manifest", type=Path, default=MODEL_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--feature-panel-cache-dir", type=Path, default=FEATURE_PANEL_CACHE_DIR)
    parser.add_argument("--split-date", default=SPLIT_DATE)
    parser.add_argument("--validation-start", default=VALIDATION_START)
    parser.add_argument("--horizon", type=int, default=FORWARD_HORIZON_BARS)
    parser.add_argument("--eval-step", type=int, default=EVALUATION_STEP_BARS)
    parser.add_argument("--commission-rate", type=float, default=COMMISSION_RATE)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--top-n-candidates", default=",".join(map(str, TOP_N_CANDIDATES)))
    parser.add_argument("--top-quantile-candidates", default=",".join(map(str, TOP_QUANTILE_CANDIDATES)))
    parser.add_argument("--min-assets", type=int, default=MIN_ASSETS_PER_REBALANCE)
    parser.add_argument("--min-holding-count", type=int, default=MIN_HOLDING_COUNT)
    parser.add_argument("--hold-steps-candidates", default=",".join(map(str, HOLD_STEPS_CANDIDATES)))
    parser.add_argument("--target-exposure-candidates", default=",".join(map(str, TARGET_EXPOSURE_CANDIDATES)))
    parser.add_argument(
        "--confidence-quantile-candidates",
        default="none",
        help="验证集 top1 预测分的门控分位数候选，例如 none,0.9；测试集使用验证集学到的绝对阈值。",
    )
    parser.add_argument(
        "--spread-quantile-candidates",
        default="none",
        help="验证集 top1-top2 预测分差的门控分位数候选，例如 none,0.7；测试集使用验证集学到的绝对阈值。",
    )
    parser.add_argument("--lot-size", type=int, default=LOT_SIZE)
    parser.add_argument("--sharpe-target", type=float, default=SHARPE_TARGET)
    parser.add_argument("--max-symbols", type=int, default=0, help="调试用股票数上限，0 表示使用全部稳定股票。")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rebuild-feature-panel", action="store_true", help="忽略已有 pkl 缓存并重建特征面板。")
    parser.add_argument("--no-feature-panel-cache", action="store_true", help="不读取也不写入特征面板 pkl 缓存。")
    parser.add_argument("--build-feature-panel-only", action="store_true", help="只构建或刷新特征面板缓存，不训练模型。")
    return parser.parse_args()


def parse_int_tuple(text: str) -> tuple[int, ...]:
    """解析逗号分隔的正整数。"""

    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("候选值必须是正整数。")
    return values


def parse_float_tuple(text: str) -> tuple[float, ...]:
    """解析逗号分隔且位于 (0, 1] 的浮点数。"""

    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 or value > 1 for value in values):
        raise ValueError("候选分位数或仓位必须位于 (0, 1]。")
    return values


def parse_optional_float_tuple(text: str) -> tuple[float | None, ...]:
    """解析可包含 none 的逗号分隔分位数候选。"""

    values: list[float | None] = []
    for raw_item in text.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in {"none", "null", "off", "false"}:
            values.append(None)
            continue
        value = float(item)
        if value <= 0 or value >= 1:
            raise ValueError("门控分位数候选必须位于 (0, 1)，或使用 none。")
        values.append(value)
    if not values:
        return (None,)
    deduped: list[float | None] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def create_config(args: argparse.Namespace) -> ExperimentConfig:
    """校验命令行参数并生成不可变实验配置。"""

    if XGBRegressor is None:
        raise RuntimeError("当前 Python 环境未安装 xgboost。")
    if args.horizon < 1 or args.eval_step < 1:
        raise ValueError("--horizon 和 --eval-step 必须为正数。")
    if args.commission_rate < 0:
        raise ValueError("--commission-rate 不能为负数。")
    if args.min_holding_count < 1:
        raise ValueError("--min-holding-count 必须为正数。")
    if args.lot_size < 1:
        raise ValueError("--lot-size 必须为正数。")
    if args.build_feature_panel_only and args.no_feature_panel_cache:
        raise ValueError("--build-feature-panel-only 需要启用特征面板缓存，不能同时使用 --no-feature-panel-cache。")
    exposure_candidates = parse_float_tuple(args.target_exposure_candidates)
    if any(value > 1 for value in exposure_candidates):
        raise ValueError("无杠杆 long-only 股票交易要求 --target-exposure-candidates 位于 (0, 1]。")
    return ExperimentConfig(
        data_root=args.data_root,
        stable_symbols_file=args.stable_symbols_file,
        feature_manifest=args.feature_manifest,
        output_dir=args.output_dir,
        feature_panel_cache_dir=args.feature_panel_cache_dir,
        split_date=pd.Timestamp(args.split_date),
        validation_start=pd.Timestamp(args.validation_start),
        forward_horizon_bars=args.horizon,
        evaluation_step_bars=args.eval_step,
        commission_rate=args.commission_rate,
        initial_cash=args.initial_cash,
        top_n_candidates=parse_int_tuple(args.top_n_candidates),
        top_quantile_candidates=parse_float_tuple(args.top_quantile_candidates),
        min_assets_per_rebalance=args.min_assets,
        min_holding_count=int(args.min_holding_count),
        hold_steps_candidates=parse_int_tuple(args.hold_steps_candidates),
        target_exposure_candidates=exposure_candidates,
        confidence_quantile_candidates=parse_optional_float_tuple(args.confidence_quantile_candidates),
        spread_quantile_candidates=parse_optional_float_tuple(args.spread_quantile_candidates),
        lot_size=int(args.lot_size),
        sharpe_target=float(args.sharpe_target),
        max_symbols=max(int(args.max_symbols), 0),
        workers=max(int(args.workers), 1),
        random_seed=int(args.seed),
        rebuild_feature_panel=bool(args.rebuild_feature_panel),
        no_feature_panel_cache=bool(args.no_feature_panel_cache),
        build_feature_panel_only=bool(args.build_feature_panel_only),
    )


def discover_year_dirs(data_root: Path) -> list[Path]:
    """返回按年份名称排序的已解压行情目录。"""

    dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not dirs:
        raise FileNotFoundError(f"{data_root} 下未找到已解压年份目录。")
    return dirs


def read_stable_symbols(path: Path, max_symbols: int = 0) -> list[str]:
    """从稳定股票池报告读取股票代码。"""

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "symbol" not in frame.columns:
        raise ValueError(f"稳定股票池文件必须包含 symbol 列：{path}")
    symbols = frame["symbol"].astype(str).drop_duplicates().tolist()
    return symbols[:max_symbols] if max_symbols else symbols


def read_feature_manifest(path: Path) -> pd.DataFrame:
    """读取最终模型特征清单，并要求特征为截面 rank 形式。"""

    manifest = pd.read_csv(path, encoding="utf-8-sig")
    required = {"model_feature", "factor_base"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"特征清单缺少列 {sorted(missing)}：{path}")
    if not manifest["model_feature"].astype(str).str.startswith("cs_rank_").all():
        raise ValueError("本实验要求模型特征名以 cs_rank_* 开头。")
    return manifest.drop_duplicates("model_feature").reset_index(drop=True)


def read_symbol_data(symbol: str, year_dirs: Iterable[Path]) -> pd.DataFrame:
    """读取单只股票所有可用的原始分钟数据。"""

    parts: list[pd.DataFrame] = []
    for year_dir in year_dirs:
        path = year_dir / f"{symbol}.csv"
        if path.exists():
            part = pd.read_csv(path)
            part["symbol"] = symbol
            parts.append(part)
    if not parts:
        raise FileNotFoundError(f"未找到股票 {symbol} 的原始数据文件。")
    frame = pd.concat(parts, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("datetime").drop_duplicates(["symbol", "datetime"], keep="last").reset_index(drop=True)


def build_symbol_panel(args: tuple[str, list[Path], list[str], ExperimentConfig]) -> pd.DataFrame:
    """为单只股票构建采样后的基础因子和标签。

    所有因子只使用当前及历史已完成 bar。``future_return`` 带交易延迟：
    bar t 的信号在 t+1 close 入场，并在配置的 horizon 后退出。
    该字段仅作为训练和评估标签，不能进入特征。
    """

    symbol, year_dirs, base_factors, config = args
    data = read_symbol_data(symbol, year_dirs)
    factors = build_factor_library(data).reindex(columns=base_factors)
    future_return = calculate_forward_return(data, config.forward_horizon_bars)
    entry_datetime = data["datetime"].shift(-1)
    entry_price = data["close"].shift(-1)
    exit_price = data["close"].shift(-(config.forward_horizon_bars + 1))
    exit_datetime = data["datetime"].shift(-(config.forward_horizon_bars + 1))
    sampled = np.arange(len(data)) % config.evaluation_step_bars == 0
    keep = (
        sampled
        & future_return.notna()
        & entry_datetime.notna()
        & entry_price.notna()
        & exit_price.notna()
        & exit_datetime.notna()
    )
    base = data.loc[keep, ["datetime", "symbol", "close", "volume", "amount"]].copy()
    base["forward_return"] = future_return.loc[keep].to_numpy(dtype="float32")
    base["entry_datetime"] = pd.to_datetime(entry_datetime.loc[keep]).to_numpy()
    base["entry_price"] = entry_price.loc[keep].to_numpy(dtype="float32")
    base["exit_price"] = exit_price.loc[keep].to_numpy(dtype="float32")
    base["label_exit_datetime"] = pd.to_datetime(exit_datetime.loc[keep]).to_numpy()
    feature_part = factors.loc[keep].astype("float32")
    return pd.concat([base.reset_index(drop=True), feature_part.reset_index(drop=True)], axis=1)


def build_feature_panel(config: ExperimentConfig, symbols: list[str], manifest: pd.DataFrame) -> pd.DataFrame:
    """构建训练和测试共用的截面 rank 特征面板。"""

    year_dirs = discover_year_dirs(config.data_root)
    base_factors = manifest["factor_base"].astype(str).tolist()
    parts: list[pd.DataFrame] = []
    print(f"正在构建 {len(symbols)} 只股票的基础因子面板，workers={config.workers}", flush=True)
    if config.workers == 1:
        for idx, symbol in enumerate(symbols, start=1):
            parts.append(build_symbol_panel((symbol, year_dirs, base_factors, config)))
            if idx % 10 == 0 or idx == len(symbols):
                print(f"已构建股票面板 {idx}/{len(symbols)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(build_symbol_panel, (symbol, year_dirs, base_factors, config)): symbol
                for symbol in symbols
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                parts.append(future.result())
                if idx % 10 == 0 or idx == len(symbols):
                    print(f"已构建股票面板 {idx}/{len(symbols)}", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    ranked = panel.groupby("datetime", sort=False)[base_factors].rank(method="average", pct=True).astype("float32")
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
    out = pd.concat([panel[keep_cols], ranked], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def calculate_data_file_fingerprint(data_root: Path, symbols: list[str]) -> str:
    """根据入选股票原始 CSV 的路径、大小和修改时间生成数据指纹。"""

    records: list[str] = []
    for year_dir in discover_year_dirs(data_root):
        for symbol in symbols:
            path = year_dir / f"{symbol}.csv"
            if path.exists():
                stat = path.stat()
                records.append(f"{year_dir.name}/{symbol}.csv:{stat.st_size}:{stat.st_mtime_ns}")
            else:
                records.append(f"{year_dir.name}/{symbol}.csv:missing")
    payload = "\n".join(records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_feature_panel_cache_metadata(
    config: ExperimentConfig,
    symbols: list[str],
    manifest: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    """生成特征面板 pkl 缓存键和元数据，避免误用不同配置下的旧缓存。"""

    manifest_view = manifest.loc[:, ["model_feature", "factor_base"]].astype(str)
    manifest_fingerprint = hashlib.sha256(
        manifest_view.to_csv(index=False).encode("utf-8"),
    ).hexdigest()
    metadata: dict[str, object] = {
        "cache_version": FEATURE_PANEL_CACHE_VERSION,
        "data_root": str(config.data_root.resolve()),
        "stable_symbols_file": str(config.stable_symbols_file.resolve()),
        "feature_manifest": str(config.feature_manifest.resolve()),
        "manifest_fingerprint": manifest_fingerprint,
        "data_file_fingerprint": calculate_data_file_fingerprint(config.data_root, symbols),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "base_factors": manifest_view["factor_base"].tolist(),
        "model_features": manifest_view["model_feature"].tolist(),
        "forward_horizon_bars": int(config.forward_horizon_bars),
        "evaluation_step_bars": int(config.evaluation_step_bars),
        "max_symbols": int(config.max_symbols),
    }
    cache_key = hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_key, metadata


def get_feature_panel_cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    """返回特征面板 pkl 和对应元数据 JSON 的路径。"""

    stem = f"feature_panel_{cache_key[:16]}"
    return cache_dir / f"{stem}.pkl", cache_dir / f"{stem}.metadata.json"


def validate_cached_feature_panel(panel: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """读取缓存后检查必要列，防止损坏或旧格式 pkl 进入训练。"""

    required = {
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
    }
    required.update(manifest["model_feature"].astype(str).tolist())
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"特征面板缓存缺少必要列：{sorted(missing)[:20]}")
    if panel.empty:
        raise ValueError("特征面板缓存为空。")


def load_or_build_feature_panel(
    config: ExperimentConfig,
    symbols: list[str],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, Path | None, bool]:
    """优先读取匹配配置的 pkl 特征面板缓存；不存在时构建并写入缓存。"""

    if config.no_feature_panel_cache:
        panel = build_feature_panel(config, symbols, manifest)
        validate_cached_feature_panel(panel, manifest)
        return panel, None, False

    cache_key, metadata = create_feature_panel_cache_metadata(config, symbols, manifest)
    cache_path, metadata_path = get_feature_panel_cache_paths(config.feature_panel_cache_dir, cache_key)
    if cache_path.exists() and metadata_path.exists() and not config.rebuild_feature_panel:
        print(f"正在读取特征面板缓存：{cache_path}", flush=True)
        panel = pd.read_pickle(cache_path)
        validate_cached_feature_panel(panel, manifest)
        return panel, cache_path, True

    panel = build_feature_panel(config, symbols, manifest)
    validate_cached_feature_panel(panel, manifest)
    config.feature_panel_cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(cache_path)
    metadata.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "row_count": int(len(panel)),
            "column_count": int(len(panel.columns)),
            "datetime_start": str(pd.to_datetime(panel["datetime"]).min()),
            "datetime_end": str(pd.to_datetime(panel["datetime"]).max()),
            "pkl_file": str(cache_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入特征面板缓存：{cache_path}", flush=True)
    return panel, cache_path, False


def add_target_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """根据标签收益添加同一时间截面的训练目标 rank。"""

    out = frame.copy()
    counts = out.groupby("datetime")["symbol"].transform("count")
    rank = out.groupby("datetime")["forward_return"].rank(method="average", pct=True)
    out["target_rank"] = np.where(counts > 1, rank, np.nan).astype("float32")
    return out


def split_panel(frame: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """切分内部训练集、验证集和测试集，并防止标签窗口跨越边界。"""

    label_exit = pd.to_datetime(frame["label_exit_datetime"])
    core_train = frame.loc[
        (frame["datetime"] < config.validation_start) & (label_exit < config.validation_start)
    ].copy()
    validation = frame.loc[
        (frame["datetime"] >= config.validation_start)
        & (frame["datetime"] < config.split_date)
        & (label_exit < config.split_date)
    ].copy()
    test = frame.loc[frame["datetime"] >= config.split_date].copy()
    return core_train, validation, test


def build_training_matrix(frame: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    """返回目标有限样本的 X/y；XGBoost 可处理 NaN 特征。"""

    work = frame.dropna(subset=["target_rank"]).copy()
    return work.loc[:, list(feature_cols)], work["target_rank"].astype("float32")


def create_xgboost_model(config: ExperimentConfig) -> XGBRegressor:
    """创建用于 alpha 打分的固定参数 XGBoost 模型。"""

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=220,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        reg_lambda=5.0,
        reg_alpha=0.1,
        tree_method="hist",
        n_jobs=max(config.workers, 1),
        random_state=config.random_seed,
        eval_metric="rmse",
    )


def mean_spearman_ic(frame: pd.DataFrame, pred_col: str = "prediction") -> float:
    """按时间计算截面 Spearman IC，并返回均值。"""

    values: list[float] = []
    for _, group in frame.groupby("datetime", sort=False):
        if group[pred_col].nunique() < 2 or group["forward_return"].nunique() < 2:
            continue
        corr = group[[pred_col, "forward_return"]].corr(method="spearman").iloc[0, 1]
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


@dataclass(frozen=True)
class PortfolioRule:
    """在验证集上选择的 long-only 组合规则。"""

    rule_type: str
    value: float
    hold_steps: int
    target_exposure: float
    confidence_quantile: float | None = None
    confidence_threshold: float | None = None
    spread_quantile: float | None = None
    spread_threshold: float | None = None

    @property
    def name(self) -> str:
        suffix = str(self.value).replace(".", "p")
        exposure = str(self.target_exposure).replace(".", "p")
        parts = [f"{self.rule_type}_{suffix}", f"hold_{self.hold_steps}", f"exposure_{exposure}"]
        if self.confidence_quantile is not None:
            parts.append(f"confq_{str(self.confidence_quantile).replace('.', 'p')}")
        if self.spread_quantile is not None:
            parts.append(f"spreadq_{str(self.spread_quantile).replace('.', 'p')}")
        return "_".join(parts)


@dataclass
class PositionLot:
    """单笔买入形成的持仓批次，用于执行 A 股 T+1 卖出约束和交易复盘。"""

    symbol: str
    shares: int
    buy_datetime: pd.Timestamp
    buy_date: object
    buy_price: float
    buy_cost: float
    signal_datetime: pd.Timestamp
    prediction: float
    reason: str


def calculate_gate_thresholds(
    predictions: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[dict[float | None, float | None], dict[float | None, float | None]]:
    """只用验证集预测分布计算持仓门控阈值。

    top1 预测分和 top1-top2 分差都只依赖当前时间截面的模型输出。
    返回的绝对阈值会写入选中规则，并原样用于测试集，避免在测试集上重估分位数。
    """

    confidence_thresholds: dict[float | None, float | None] = {None: None}
    spread_thresholds: dict[float | None, float | None] = {None: None}
    needed_confidence = [value for value in config.confidence_quantile_candidates if value is not None]
    needed_spread = [value for value in config.spread_quantile_candidates if value is not None]
    if not needed_confidence and not needed_spread:
        return confidence_thresholds, spread_thresholds

    top_scores: list[float] = []
    spreads: list[float] = []
    for _, group in predictions.groupby("datetime", sort=False):
        tradable = group.dropna(subset=["prediction", "forward_return", "entry_price", "exit_price"]).sort_values(
            "prediction",
            ascending=False,
        )
        if len(tradable) < max(config.min_assets_per_rebalance, 2):
            continue
        top_scores.append(float(tradable.iloc[0]["prediction"]))
        spreads.append(float(tradable.iloc[0]["prediction"] - tradable.iloc[1]["prediction"]))

    top_score_series = pd.Series(top_scores, dtype="float64")
    spread_series = pd.Series(spreads, dtype="float64")
    for quantile in needed_confidence:
        confidence_thresholds[quantile] = float(top_score_series.quantile(quantile)) if not top_score_series.empty else None
    for quantile in needed_spread:
        spread_thresholds[quantile] = float(spread_series.quantile(quantile)) if not spread_series.empty else None
    return confidence_thresholds, spread_thresholds


def create_candidate_rules(config: ExperimentConfig, calibration_predictions: pd.DataFrame | None = None) -> list[PortfolioRule]:
    """生成 top-N/top-quantile、持有周期、目标仓位和验证集门控候选规则。"""

    selectors: list[tuple[str, float]] = [
        ("top_n", float(n)) for n in config.top_n_candidates if n >= config.min_holding_count
    ]
    selectors.extend(("top_quantile", float(q)) for q in config.top_quantile_candidates)
    if calibration_predictions is not None:
        confidence_thresholds, spread_thresholds = calculate_gate_thresholds(calibration_predictions, config)
    else:
        confidence_thresholds, spread_thresholds = ({None: None}, {None: None})
    rules = [
        PortfolioRule(
            rule_type,
            value,
            hold_steps,
            exposure,
            confidence_quantile,
            confidence_thresholds.get(confidence_quantile),
            spread_quantile,
            spread_thresholds.get(spread_quantile),
        )
        for rule_type, value in selectors
        for hold_steps in config.hold_steps_candidates
        for exposure in config.target_exposure_candidates
        for confidence_quantile in config.confidence_quantile_candidates
        for spread_quantile in config.spread_quantile_candidates
    ]
    return rules


def select_symbols_for_rule(group: pd.DataFrame, rule: PortfolioRule, config: ExperimentConfig) -> pd.DataFrame:
    """按预测分数在单个时间点选择 long-only 持仓股票。"""

    tradable = group.dropna(subset=["prediction", "forward_return", "entry_price", "exit_price"]).sort_values(
        "prediction",
        ascending=False,
    )
    if len(tradable) < config.min_assets_per_rebalance:
        return tradable.iloc[0:0].copy()
    if rule.confidence_threshold is not None and float(tradable.iloc[0]["prediction"]) < rule.confidence_threshold:
        return tradable.iloc[0:0].copy()
    if rule.spread_threshold is not None:
        if len(tradable) < 2:
            return tradable.iloc[0:0].copy()
        spread = float(tradable.iloc[0]["prediction"] - tradable.iloc[1]["prediction"])
        if spread < rule.spread_threshold:
            return tradable.iloc[0:0].copy()
    if rule.rule_type == "top_n":
        n = min(int(rule.value), len(tradable))
    else:
        n = max(config.min_holding_count, int(math.ceil(len(tradable) * float(rule.value))))
    if n < config.min_holding_count:
        return tradable.iloc[0:0].copy()
    selected = tradable.head(n).copy()
    return selected


def summarize_position_lots(position_lots: dict[str, list[PositionLot]]) -> dict[str, int]:
    """汇总每只股票当前持有的总股数。"""

    return {
        symbol: int(sum(lot.shares for lot in lots if lot.shares > 0))
        for symbol, lots in position_lots.items()
        if sum(lot.shares for lot in lots if lot.shares > 0) > 0
    }


def value_position_lots(position_lots: dict[str, list[PositionLot]], prices: dict[str, float]) -> float:
    """用给定价格映射计算当前所有持仓批次市值。

    prices: 股票代码到当前可用估值价格的映射。全 A 股票池可能有个别持仓在
    某个采样时点没有行情；调用方应传入最近可得价格，避免把缺失行情误当成
    持仓市值归零。
    """

    total = 0.0
    for symbol, lots in position_lots.items():
        price = prices.get(symbol)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        total += sum(lot.shares for lot in lots if lot.shares > 0) * float(price)
    return float(total)


def create_desired_shares(selected: pd.DataFrame, pre_trade_equity: float, rule: PortfolioRule, config: ExperimentConfig) -> dict[str, int]:
    """根据目标仓位和 100 股整数手生成目标股数。"""

    if selected.empty or pre_trade_equity <= 0:
        return {}
    desired: dict[str, int] = {}
    allocation = pre_trade_equity * rule.target_exposure / len(selected)
    for row in selected.itertuples(index=False):
        symbol = str(row.symbol)
        price = float(row.entry_price)
        if not np.isfinite(price) or price <= 0:
            continue
        shares = int(math.floor(allocation / (price * config.lot_size)) * config.lot_size)
        if shares > 0:
            desired[symbol] = shares
    return desired


def reduce_buy_plan_to_cash(
    buy_plan: dict[str, int],
    entry_prices: dict[str, float],
    cash: float,
    config: ExperimentConfig,
) -> dict[str, int]:
    """在现金约束下缩减买入计划，确保买入金额和税费不超过现金。"""

    plan = {symbol: shares for symbol, shares in buy_plan.items() if shares > 0}
    while plan:
        required = sum(plan[symbol] * entry_prices[symbol] * (1 + config.commission_rate) for symbol in plan)
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
    position_lots: dict[str, list[PositionLot]],
    symbol: str,
    shares_to_sell: int,
    sell_datetime: pd.Timestamp,
    sell_price: float,
    reason: str,
    config: ExperimentConfig,
) -> tuple[int, float, float, list[dict[str, object]], list[dict[str, object]]]:
    """按 FIFO 卖出已满足 T+1 的持仓批次，并生成交易事件和已平仓复盘记录。"""

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
        cost = value * config.commission_rate
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


def run_long_only_backtest(
    predictions: pd.DataFrame,
    rule: PortfolioRule,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | str | None]]:
    """按 signal_lag=1、现金账户、T+1 和 100 股整数手回测 long-only 预测排序策略。"""

    rows: list[dict[str, object]] = []
    holdings: list[pd.DataFrame] = []
    trade_events: list[dict[str, object]] = []
    round_trips: list[dict[str, object]] = []
    position_lots: dict[str, list[PositionLot]] = {}
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
        current_shares = summarize_position_lots(position_lots)
        pre_trade_equity = cash + value_position_lots(position_lots, valuation_entry_prices)
        should_rebalance = (period_index % rule.hold_steps == 0) or not current_shares

        buy_value = sell_value = turnover_value = cost_value = 0.0
        if should_rebalance:
            selected = select_symbols_for_rule(tradable, rule, config)
            desired_shares = create_desired_shares(selected, pre_trade_equity, rule, config)
            current_shares = summarize_position_lots(position_lots)
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

            current_after_sell = summarize_position_lots(position_lots)
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
                cost = value * config.commission_rate
                if value + cost > cash + 1e-6:
                    continue
                reason = f"买入：预测分进入当前规则 {rule.name} 的目标组合"
                lot = PositionLot(
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

        current_shares = summarize_position_lots(position_lots)
        entry_position_value = value_position_lots(position_lots, valuation_entry_prices)
        exit_position_value = value_position_lots(position_lots, valuation_exit_prices)
        gross_end_equity = pre_trade_equity - entry_position_value + exit_position_value
        end_equity = cash + exit_position_value
        gross_return = gross_end_equity / pre_trade_equity - 1.0 if pre_trade_equity > 0 else 0.0
        net_return = end_equity / pre_trade_equity - 1.0 if pre_trade_equity > 0 else 0.0
        turnover = turnover_value / pre_trade_equity if pre_trade_equity > 0 else 0.0
        buy_turnover = buy_value / pre_trade_equity if pre_trade_equity > 0 else 0.0
        sell_turnover = sell_value / pre_trade_equity if pre_trade_equity > 0 else 0.0
        cost_return = cost_value / pre_trade_equity if pre_trade_equity > 0 else 0.0
        gross_exposure = entry_position_value / pre_trade_equity if pre_trade_equity > 0 else 0.0
        cash_weight = cash / pre_trade_equity if pre_trade_equity > 0 else 0.0
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "gross_return": gross_return,
                "cost_return": cost_return,
                "return": net_return,
                "turnover": turnover,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "holding_count": len(current_shares),
                "gross_exposure": gross_exposure,
                "cash_weight": cash_weight,
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
    metrics = performance_metrics(curve["equity"], curve["return"], annualization=annualization)
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
            "commission_rate": float(config.commission_rate),
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
    if not trade_events_frame.empty:
        metrics["total_trade_lots"] = float(trade_events_frame["手数(lots)"].astype(float).sum())
    else:
        metrics["total_trade_lots"] = 0.0
    return curve, holdings_frame, trade_events_frame, round_trips_frame, open_positions_frame, metrics


def evaluate_rules(validation_predictions: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """只使用验证集 Sharpe 和诊断指标选择组合规则。"""

    rows: list[dict[str, object]] = []
    for rule in create_candidate_rules(config, validation_predictions):
        curve, _, _, _, _, metrics = run_long_only_backtest(validation_predictions, rule, config)
        row = {
            "rule": rule.name,
            "rule_type": rule.rule_type,
            "rule_value": rule.value,
            "hold_steps": rule.hold_steps,
            "target_exposure": rule.target_exposure,
            "confidence_quantile": rule.confidence_quantile,
            "confidence_threshold": rule.confidence_threshold,
            "spread_quantile": rule.spread_quantile,
            "spread_threshold": rule.spread_threshold,
        }
        row.update(metrics)
        row["periods"] = int(len(curve))
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)


def calculate_periodic_returns(curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按自然月和自然年聚合净收益。"""

    returns = curve["return"].copy()
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    yearly = (1 + returns).groupby(returns.index.to_period("Y")).prod() - 1
    monthly_frame = monthly.rename("return").reset_index()
    yearly_frame = yearly.rename("return").reset_index()
    monthly_frame["datetime"] = monthly_frame["datetime"].astype(str)
    yearly_frame["datetime"] = yearly_frame["datetime"].astype(str)
    return monthly_frame, yearly_frame


def configure_chinese_font() -> bool:
    """尽量为 Matplotlib 配置可显示中文的字体，返回是否成功。"""

    if plt is None or font_manager is None:
        return False
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


def plot_test_return_curve(run_dir: Path, curve: pd.DataFrame, metrics: dict[str, float | int | str | None]) -> None:
    """绘制测试集收益折线图，并在图中附上关键绩效指标。"""

    if plt is None or curve.empty:
        return
    can_show_chinese = configure_chinese_font()
    plot_frame = curve.copy()
    if "datetime" in plot_frame.columns:
        plot_frame = plot_frame.set_index("datetime")
    title = str(metrics.get("rule", "策略"))
    if can_show_chinese:
        summary = (
            f"策略：{title}\n"
            f"Sharpe：{float(metrics.get('sharpe', 0.0)):.3f}\n"
            f"收益率：{float(metrics.get('total_return', 0.0)):.2%}\n"
            f"年化：{float(metrics.get('annual_return', 0.0)):.2%}\n"
            f"年化波动：{float(metrics.get('annual_volatility', 0.0)):.2%}\n"
            f"最大回撤：{float(metrics.get('max_drawdown', 0.0)):.2%}\n"
            f"交易手数：{float(metrics.get('total_trade_lots', 0.0)):.0f}\n"
            f"平仓胜率：{float(metrics.get('round_trip_win_rate', 0.0)):.2%}\n"
            f"平均持仓：{float(metrics.get('avg_holding_count', 0.0)):.2f} 只"
        )
        main_label = "累计收益"
        title_text = f"测试集收益曲线 - {title}"
        y_return = "累计收益"
        drawdown_label = "回撤"
        x_label = "时间"
    else:
        summary = (
            f"Strategy: {title}\n"
            f"Sharpe: {float(metrics.get('sharpe', 0.0)):.3f}\n"
            f"Return: {float(metrics.get('total_return', 0.0)):.2%}\n"
            f"Annual return: {float(metrics.get('annual_return', 0.0)):.2%}\n"
            f"Annual vol: {float(metrics.get('annual_volatility', 0.0)):.2%}\n"
            f"Max drawdown: {float(metrics.get('max_drawdown', 0.0)):.2%}\n"
            f"Trade lots: {float(metrics.get('total_trade_lots', 0.0)):.0f}\n"
            f"Closed-trade win rate: {float(metrics.get('round_trip_win_rate', 0.0)):.2%}\n"
            f"Avg holdings: {float(metrics.get('avg_holding_count', 0.0)):.2f}"
        )
        main_label = "Cumulative return"
        title_text = f"Test equity curve - {title}"
        y_return = "Cumulative return"
        drawdown_label = "Drawdown"
        x_label = "Time"
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(plot_frame.index, plot_frame["cumulative_return"], color="#1f77b4", linewidth=1.5, label=main_label)
    axes[0].set_title(title_text, fontsize=14)
    axes[0].set_ylabel(y_return)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[0].text(
        0.012,
        0.98,
        summary,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.86, "edgecolor": "#999999"},
    )
    axes[1].fill_between(plot_frame.index, plot_frame["drawdown"], 0, color="#d62728", alpha=0.35)
    axes[1].set_ylabel(drawdown_label)
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel(x_label)
    fig.tight_layout()
    fig.savefig(run_dir / "测试集收益曲线.png", dpi=160)
    fig.savefig(run_dir / "test_return_curve.png", dpi=160)
    plt.close(fig)


def write_output_guides(run_dir: Path) -> None:
    """写出中文文件说明，帮助人工阅读报告目录。"""

    text = """# 输出文件说明

- `final_report.md`：中文最终报告。
- `model_metadata.json`：模型配置、参数、特征数量、股票池和绩效指标，字段名保留英文以便程序读取。
- `xgboost_model.json`：XGBoost 原生模型。
- `xgboost_model.joblib`：Python joblib 格式模型。
- `used_feature_manifest.csv`：本次使用的特征清单。
- `feature_list.txt`：模型特征名列表。
- `feature_importance.csv`：模型特征重要性。
- `validation_rule_metrics.csv`：训练期内部验证集规则表现。
- `train_predictions_sample.csv`：训练集预测样本。
- `validation_predictions.csv`：验证集预测。
- `test_predictions.csv`：测试集预测。
- `test_holdings.csv`：测试集逐期持仓。
- `test_trade_events.csv`：逐笔买入和卖出事件，含触发原因。
- `test_round_trips.csv`：已平仓交易复盘，按买入批次和卖出批次匹配。
- `test_open_positions.csv`：测试期末仍未平仓的持仓。
- `test_equity_curve.csv`：权益、累计收益、回撤、换手和暴露。
- `测试集收益曲线.png`：带关键指标的测试集收益曲线。
- `test_return_curve.png`：同一张收益曲线的英文文件名副本，便于跨系统打开。
- `test_monthly_returns.csv`：月度收益。
- `test_yearly_returns.csv`：年度收益。
- `test_metrics.csv`：测试集指标。
"""
    (run_dir / "输出文件说明.md").write_text(text, encoding="utf-8")


def write_report(
    run_dir: Path,
    config: ExperimentConfig,
    selected_rule: PortfolioRule,
    validation_table: pd.DataFrame,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float | int | str | None],
    feature_count: int,
    symbol_count: int,
    used_feature_source_count: int,
) -> None:
    """写出包含假设、指标和目标是否达成的 Markdown 报告。"""

    reached = bool(float(test_metrics.get("sharpe", 0.0)) >= config.sharpe_target)
    text = f"""# XGBoost A 股 Long-Only 回测报告

## 数据与切分

- 训练集：`datetime < {config.split_date.date()}`。内部验证集从 `{config.validation_start.date()}` 开始，仍然完全位于训练期内。
- 测试集：`datetime >= {config.split_date.date()}`。
- 股票池：{symbol_count} 只稳定沪深 300 股票，来源为 `{config.stable_symbols_file}`。
- 使用原始字段：`datetime`、`symbol`、`open`、`high`、`low`、`close`、`volume`、`amount`。
- 当前严格训练期特征清单包含 {feature_count} 个模型特征，来自 {used_feature_source_count} 个基础因子。

## 防未来函数控制

- 因子生成使用 `quant_backtest.factor_library`；单股票 rolling、lag、pct_change、趋势、波动率、流动性等特征只使用当前及历史已完成 bar。
- 截面 rank 只在同一个 `datetime` 内按当时可交易股票横截面计算。
- 标签是带交易延迟的未来收益：bar t 生成特征和信号，t+1 close 入场，持有 {config.forward_horizon_bars} 根 bar 后退出。
- 内部训练集会排除标签退出时间跨过验证集起点的样本；最终训练集会排除标签退出时间跨过切分日的样本。
- XGBoost 拟合、组合规则选择、特征筛选都只使用训练期或训练期内部验证集。
- 测试集用于最终预测和评估。

## 模型与交易策略

- 模型：`XGBRegressor`，固定参数保存在 `model_metadata.json`。
- 训练目标：同一时间截面的带交易延迟未来收益 rank。
- 策略：按预测分数排序，执行 long-only 整数手规则 `{selected_rule.name}`。
- 做空：禁止。杠杆：禁止。目标总多头仓位：`{selected_rule.target_exposure}`。
- 持仓门控：top1 预测分验证集分位数 `{selected_rule.confidence_quantile}`、阈值 `{selected_rule.confidence_threshold}`；top1-top2 分差验证集分位数 `{selected_rule.spread_quantile}`、阈值 `{selected_rule.spread_threshold}`。
- 调仓：每 `{selected_rule.hold_steps}` 个采样周期调仓一次；每个采样周期为 `{config.evaluation_step_bars}` 根 bar。
- 整数手：买入向下取整到 `{config.lot_size}` 股整数手，初始资金 `{config.initial_cash}`。
- 交易成本：所有税费合计按成交金额 `{config.commission_rate}` 计算；滑点设为 0。
- T+1：当天买入的股票当天不能卖出，下一交易日才允许平仓或减仓。
- 尚未完整建模：涨跌停无法成交、ST/退市、最低佣金、冲击成本和历史点时成分股。

## 验证集规则选择

组合规则只在训练期内部验证集上选择。

完整验证规则表保存在 `validation_rule_metrics.csv`。

## 测试集指标

- Sharpe: {float(test_metrics.get("sharpe", 0.0)):.6f}
- 年化收益: {float(test_metrics.get("annual_return", 0.0)):.6f}
- 年化波动率: {float(test_metrics.get("annual_volatility", 0.0)):.6f}
- 成本前总收益: {float(test_metrics.get("gross_total_return", 0.0)):.6f}
- 成本后总收益: {float(test_metrics.get("total_return", 0.0)):.6f}
- 总成本收益扣减: {float(test_metrics.get("total_cost_return", 0.0)):.6f}
- 最大回撤: {float(test_metrics.get("max_drawdown", 0.0)):.6f}
- Calmar: {float(test_metrics.get("calmar", 0.0)):.6f}
- 单期胜率: {float(test_metrics.get("win_bar_rate", 0.0)):.6f}
- 平均换手率: {float(test_metrics.get("avg_turnover", 0.0)):.6f}
- 总换手率: {float(test_metrics.get("total_turnover", 0.0)):.6f}
- 平均总多头暴露: {float(test_metrics.get("avg_gross_exposure", 0.0)):.6f}
- 平均现金权重: {float(test_metrics.get("avg_cash_weight", 0.0)):.6f}
- 交易事件数: {int(test_metrics.get("trade_count", 0))}
- 调仓次数: {int(test_metrics.get("rebalance_count", 0))}
- 已平仓交易数: {int(test_metrics.get("round_trip_count", 0))}
- 已平仓交易胜率: {float(test_metrics.get("round_trip_win_rate", 0.0)):.6f}
- 总交易手数: {float(test_metrics.get("total_trade_lots", 0.0)):.2f}
- 平均持仓数量: {float(test_metrics.get("avg_holding_count", 0.0)):.2f}

## Sharpe 目标

测试集 Sharpe >= {config.sharpe_target:g}: `{reached}`。

## 验证诊断

- 训练集平均 Spearman IC: {train_metrics.get("mean_spearman_ic", 0.0):.6f}
- 验证集平均 Spearman IC: {validation_metrics.get("mean_spearman_ic", 0.0):.6f}
- 选中的验证规则: `{selected_rule.name}`。
- 最优验证 Sharpe: {float(validation_table.iloc[0].get("sharpe", 0.0)) if not validation_table.empty else 0.0:.6f}
"""
    (run_dir / "final_report.md").write_text(text, encoding="utf-8")


def save_outputs(
    config: ExperimentConfig,
    run_dir: Path,
    model: XGBRegressor,
    feature_cols: Sequence[str],
    feature_manifest: pd.DataFrame,
    validation_table: pd.DataFrame,
    selected_rule: PortfolioRule,
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_curve: pd.DataFrame,
    test_holdings: pd.DataFrame,
    test_trade_events: pd.DataFrame,
    test_round_trips: pd.DataFrame,
    test_open_positions: pd.DataFrame,
    test_metrics: dict[str, float | int | str | None],
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    symbols: list[str],
    feature_panel_cache_path: Path | None,
    feature_panel_cache_loaded: bool,
) -> None:
    """持久化模型、预测、回测结果和报告产物。"""

    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(run_dir / "xgboost_model.json")
    joblib.dump(model, run_dir / "xgboost_model.joblib")
    feature_manifest.to_csv(run_dir / "used_feature_manifest.csv", index=False, encoding="utf-8-sig")
    (run_dir / "feature_list.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    validation_table.to_csv(run_dir / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    train_predictions.to_csv(run_dir / "train_predictions_sample.csv", index=False, encoding="utf-8-sig")
    validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_curve.to_csv(run_dir / "test_equity_curve.csv", encoding="utf-8-sig")
    if not test_holdings.empty:
        test_holdings.to_csv(run_dir / "test_holdings.csv", index=False, encoding="utf-8-sig")
    if not test_trade_events.empty:
        test_trade_events.to_csv(run_dir / "test_trade_events.csv", index=False, encoding="utf-8-sig")
    if not test_round_trips.empty:
        test_round_trips.to_csv(run_dir / "test_round_trips.csv", index=False, encoding="utf-8-sig")
    if not test_open_positions.empty:
        test_open_positions.to_csv(run_dir / "test_open_positions.csv", index=False, encoding="utf-8-sig")
    monthly, yearly = calculate_periodic_returns(test_curve)
    monthly.to_csv(run_dir / "test_monthly_returns.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(run_dir / "test_yearly_returns.csv", index=False, encoding="utf-8-sig")
    pd.Series(test_metrics).to_csv(run_dir / "test_metrics.csv", header=["value"])
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_test_return_curve(run_dir, test_curve, test_metrics)
    write_output_guides(run_dir)
    metadata = {
        "config": {key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items()},
        "model_params": model.get_params(),
        "selected_rule": asdict(selected_rule),
        "feature_count": len(feature_cols),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_panel_cache": {
            "enabled": not config.no_feature_panel_cache,
            "loaded_from_cache": feature_panel_cache_loaded,
            "path": str(feature_panel_cache_path) if feature_panel_cache_path is not None else None,
        },
        "no_test_tuning_statement": (
            "测试集仅用于最终预测和评估。模型拟合、组合规则选择和特征筛选均来自训练期或训练期内部验证产物。"
        ),
    }
    (run_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        run_dir,
        config,
        selected_rule,
        validation_table,
        train_metrics,
        validation_metrics,
        test_metrics,
        len(feature_cols),
        len(symbols),
        int(feature_manifest["factor_base"].nunique()),
    )


def run_experiment(config: ExperimentConfig) -> Path:
    """执行完整训练、验证规则选择和测试集回测流程。"""

    feature_manifest = read_feature_manifest(config.feature_manifest)
    feature_cols = feature_manifest["model_feature"].astype(str).tolist()
    symbols = read_stable_symbols(config.stable_symbols_file, config.max_symbols)
    panel_base, feature_panel_cache_path, feature_panel_cache_loaded = load_or_build_feature_panel(
        config,
        symbols,
        feature_manifest,
    )
    if config.build_feature_panel_only:
        if feature_panel_cache_path is None:
            raise RuntimeError("当前配置未启用特征面板缓存，无法只构建缓存后退出。")
        print(f"特征面板缓存已就绪：{feature_panel_cache_path}", flush=True)
        return feature_panel_cache_path

    panel = add_target_rank(panel_base)
    core_train, validation, test = split_panel(panel, config)
    if core_train.empty or validation.empty or test.empty:
        raise ValueError(
            f"切分后样本量无效：core_train={len(core_train)} validation={len(validation)} test={len(test)}"
        )

    leak_checks = []
    year_dirs = discover_year_dirs(config.data_root)
    for symbol in symbols[:3]:
        leak_checks.append(validate_no_future_leakage(read_symbol_data(symbol, year_dirs).head(1_000)))
    if not all(check["passed"] for check in leak_checks):
        raise ValueError(f"未来扰动防泄露检查失败：{leak_checks}")

    x_core, y_core = build_training_matrix(core_train, feature_cols)
    x_validation, _ = build_training_matrix(validation, feature_cols)
    validation_eval = validation.dropna(subset=["target_rank"]).copy()

    validation_model = create_xgboost_model(config)
    print(f"正在拟合验证模型：rows={len(x_core)} features={len(feature_cols)}", flush=True)
    validation_model.fit(x_core, y_core)
    validation_eval["prediction"] = validation_model.predict(x_validation).astype("float32")
    validation_table = evaluate_rules(validation_eval, config)
    best = validation_table.iloc[0]
    selected_rule = PortfolioRule(
        str(best["rule_type"]),
        float(best["rule_value"]),
        int(best["hold_steps"]),
        float(best["target_exposure"]),
        None if pd.isna(best.get("confidence_quantile")) else float(best.get("confidence_quantile")),
        None if pd.isna(best.get("confidence_threshold")) else float(best.get("confidence_threshold")),
        None if pd.isna(best.get("spread_quantile")) else float(best.get("spread_quantile")),
        None if pd.isna(best.get("spread_threshold")) else float(best.get("spread_threshold")),
    )

    train_eval = core_train.dropna(subset=["target_rank"]).copy()
    train_eval["prediction"] = validation_model.predict(x_core).astype("float32")
    train_metrics = {"mean_spearman_ic": mean_spearman_ic(train_eval)}
    validation_metrics = {"mean_spearman_ic": mean_spearman_ic(validation_eval)}

    train_full = panel.loc[
        (panel["datetime"] < config.split_date)
        & (pd.to_datetime(panel["label_exit_datetime"]) < config.split_date)
    ].copy()
    x_train_full, y_train_full = build_training_matrix(train_full, feature_cols)
    final_model = create_xgboost_model(config)
    print(f"正在拟合最终模型：rows={len(x_train_full)} features={len(feature_cols)}", flush=True)
    final_model.fit(x_train_full, y_train_full)

    test_eval = test.dropna(subset=["forward_return"]).copy()
    test_eval["prediction"] = final_model.predict(test_eval[feature_cols]).astype("float32")
    test_curve, test_holdings, test_trade_events, test_round_trips, test_open_positions, test_metrics = run_long_only_backtest(
        test_eval,
        selected_rule,
        config,
    )
    test_metrics["mean_spearman_ic"] = mean_spearman_ic(test_eval)
    test_metrics["test_start"] = str(test_eval["datetime"].min())
    test_metrics["test_end"] = str(test_eval["datetime"].max())
    test_metrics["test_rows"] = int(len(test_eval))
    test_metrics["test_symbol_count"] = int(test_eval["symbol"].nunique())

    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    save_outputs(
        config,
        run_dir,
        final_model,
        feature_cols,
        feature_manifest,
        validation_table,
        selected_rule,
        train_eval[
            [
                "datetime",
                "symbol",
                "close",
                "forward_return",
                "entry_datetime",
                "entry_price",
                "exit_price",
                "target_rank",
                "prediction",
            ]
        ].head(200_000),
        validation_eval[
            [
                "datetime",
                "symbol",
                "close",
                "forward_return",
                "entry_datetime",
                "entry_price",
                "exit_price",
                "target_rank",
                "prediction",
            ]
        ],
        test_eval[
            [
                "datetime",
                "symbol",
                "close",
                "forward_return",
                "entry_datetime",
                "entry_price",
                "exit_price",
                "target_rank",
                "prediction",
            ]
        ],
        test_curve,
        test_holdings,
        test_trade_events,
        test_round_trips,
        test_open_positions,
        test_metrics,
        train_metrics,
        validation_metrics,
        symbols,
        feature_panel_cache_path,
        feature_panel_cache_loaded,
    )
    return run_dir


def main() -> None:
    """运行 XGBoost long-only 样本外实验。"""

    config = create_config(parse_args())
    run_dir = run_experiment(config)
    if config.build_feature_panel_only:
        print(f"已完成特征面板缓存构建：{run_dir}")
        return
    metrics = pd.read_csv(run_dir / "test_metrics.csv", index_col=0)["value"].to_dict()
    print(f"已保存 XGBoost long-only 回测结果到 {run_dir}")
    print(
        "测试集指标："
        f"sharpe={metrics.get('sharpe')} annual_return={metrics.get('annual_return')} "
        f"annual_volatility={metrics.get('annual_volatility')} max_drawdown={metrics.get('max_drawdown')} "
        f"avg_turnover={metrics.get('avg_turnover')}"
    )


if __name__ == "__main__":
    main()
