"""原XGBoost/CTA逐年隔离数据：复用有限原因子，新增按年初始化递推列与24分钟标签pkl。

只读年度缓存及原分钟CSV，不读取旧模型预测或测试收益。股票池沿用上一自然周高波动，
因子定义不换成低维净收益；年界初始化、公共时钟和去除未来缺失筛选是显式隔离调整。
"""

from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import talib

from quant_backtest.annual_panel import load_symbol_bars, write_json
from quant_backtest.bounded_training import create_finite_factor_names
from quant_backtest.factor_library import calculate_atr, calculate_rsi, calculate_adx, rolling_trend_stats, rolling_zscore
from scripts.run_multi_factor_cta_backtests import add_cta_factor_scores


def calculate_file_hash(path: Path) -> str:
    """分块计算path指纹，避免大pkl整体复制到内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_original_features(directory: Path) -> tuple[list[str], list[str]]:
    """从directory旧模型清单读取固定156列，并以AST取得原CTA函数依赖的因子字面量。"""
    features = (directory / "feature_list.txt").read_text().splitlines()
    if len(features) != 156 or len(set(features)) != 156 or not all(c.startswith("cs_rank_") for c in features):
        raise ValueError("原始模型必须是明确的156列，不能偷偷换成29列或154列新模型")
    tree = ast.parse(inspect.getsource(add_cta_factor_scores))
    overlay = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
               and isinstance(node.value, str) and node.value.startswith("cs_rank_")}
    return features, sorted(set(features) | overlay)


def calculate_recursive_columns(raw: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """按单股单年raw和names计算原库相同的递推/趋势公式；参数只含该年已完成分钟。

    返回与raw同索引列，不把递推指标替换为简单均值；未识别列失败，不静默补0。
    """
    close, high, low = raw.close.astype(float), raw.high.astype(float), raw.low.astype(float)
    out = {}
    macds, trends = {}, {}
    obv = None
    for name in names:
        if name.startswith("ema_gap_"):
            n = int(name.split("_")[-1])
            out[name] = close / close.ewm(span=n, adjust=False, min_periods=n).mean() - 1
        elif name.startswith("rsi_"):
            out[name] = calculate_rsi(close, int(name.split("_")[-1]))
        elif name.startswith("atr_ratio_"):
            out[name] = calculate_atr(high, low, close, int(name.split("_")[-1])) / close
        elif name.startswith("adx_"):
            out[name] = calculate_adx(high, low, close, int(name.split("_")[-1]))
        elif name.startswith("macd_"):
            _, kind, fast, slow, signal = name.split("_")
            key = (int(fast), int(slow), int(signal))
            if key not in macds:
                macds[key] = talib.MACD(close.to_numpy(), fastperiod=key[0], slowperiod=key[1], signalperiod=key[2])
            out[name] = pd.Series(macds[key][["dif", "dea", "hist"].index(kind)], index=raw.index) / close
        elif name.startswith("trend_"):
            n = int(name.split("_")[-1])
            if n not in trends:
                trends[n] = rolling_trend_stats(np.log(close), n)
            out[name] = trends[n][0 if name.startswith("trend_slope_") else 1]
        elif name.startswith("obv_z_"):
            if obv is None:
                obv = pd.Series(talib.OBV(close.to_numpy(), raw.volume.to_numpy(dtype=float)), index=raw.index)
            out[name] = rolling_zscore(obv, int(name.split("_")[-1]))
        else:
            raise ValueError(f"未识别的原始递推列：{name}")
    return pd.DataFrame(out, index=raw.index).replace([np.inf, -np.inf], np.nan)


def build_original_symbol(raw: pd.DataFrame, cached: dict, columns: list[str]) -> dict:
    """复用cached单股有限因子，以raw分年补columns非有限列和标签；返回panel/market及来源。

    有限列仅在该年已观察241条后使用，避免跨年窗口；截面排名稍后在全池统一计算。
    周池来源跨年的首周不进训练/校准/验证。未来缺价只令标签缺失，不删推断股票。
    """
    bases = [c.removeprefix("cs_rank_") for c in columns]
    finite = create_finite_factor_names()
    recursive = [c for c in bases if c not in finite]
    source_panel = cached["panel"].set_index("datetime")
    source_market = cached["market"].set_index("datetime")
    frames, markets = [], []
    for year, segment in raw.groupby(raw.datetime.dt.year, sort=True):
        segment = segment.reset_index(drop=True)
        bars = segment.set_index("datetime")
        clock = source_market.index[source_market.index.year == year]
        market = source_market.reindex(clock).copy()
        updated = calculate_recursive_columns(segment, recursive)
        updated.index = bars.index
        atr = calculate_atr(bars.high, bars.low, bars.close, 240)
        market["atr"] = atr.reindex(clock).to_numpy()
        market["high"] = bars.high.reindex(clock).to_numpy()
        market["execution_atr"] = atr.reindex(clock + pd.Timedelta(minutes=1)).to_numpy()
        days = pd.DatetimeIndex(sorted(clock.normalize().unique()))
        previous = bars.close.groupby(bars.index.normalize()).last().reindex(days).ffill().shift(1)
        market["previous_close"] = previous.reindex(clock.normalize()).to_numpy()
        market["symbol"] = cached["audit"]["symbol"]
        market.index.name = "datetime"
        markets.append(market.reset_index())
        times = source_panel.index[source_panel.index.year == year]
        locations = bars.index.get_indexer(times)
        # 仅按已知历史决定暖机资格，不使用未来标签或未来能否成交筛股票。
        times = times[locations >= 240]
        panel = source_panel.reindex(times)[bases].copy()
        panel.loc[:, recursive] = updated.reindex(times)[recursive].to_numpy(dtype="float32")
        next_time = pd.Series(bars.index, index=bars.index).shift(-25)
        panel["label_end"] = next_time.reindex(times).to_numpy()
        panel["forward_return"] = (bars.close.shift(-25) / bars.close.shift(-1) - 1).reindex(times).to_numpy(dtype="float32")
        panel["source_start"] = bars.index[0]
        panel["pool_source_start"] = times.normalize() - pd.to_timedelta(times.dayofweek + 7, unit="D")
        panel["symbol"] = cached["audit"]["symbol"]
        panel.index.name = "datetime"
        frames.append(panel.reset_index())
    return {"panel": pd.concat(frames, ignore_index=True), "market": pd.concat(markets, ignore_index=True)}


def build_original_symbol_file(task: tuple) -> str:
    """task给单股、cfg、列和缓存目录；只新增pkl，已有结果在同一指纹下直接复用。"""
    symbol, cfg, columns, directory = task
    path = Path(directory) / f"{symbol}.pkl"
    if path.exists():
        return str(path)
    cached = pd.read_pickle(Path(cfg["annual_cache"]) / f"{symbol}.pkl")
    raw, audit = load_symbol_bars(Path(cfg["data_root"]), symbol, cfg["start_date"], cfg["end_date"])
    output = build_original_symbol(raw, cached, columns)
    output["quality"] = audit
    temp = path.with_suffix(".tmp.pkl")
    pd.to_pickle(output, temp)
    temp.replace(path)
    return str(path)


def load_original_panel(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """由cfg限定数据/年度原缓存/旧156列模型目录；重用新pkl或只补递推与24分钟标签。

    cache_dir是用户YAML的持久路径；原缓存每股文件、原行情清单和定义进入身份指纹。
    不调用load_annual_panel覆盖旧元数据，不删除原156列的历史选择偏差。
    """
    features, columns = read_original_features(Path(cfg["original_model_run"]))
    source = Path(cfg["annual_cache"])
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["factor_code_sha256"] != calculate_file_hash(Path(inspect.getfile(calculate_atr))):
        raise ValueError("旧有限因子缓存的原库源码已经改变")
    symbols = sorted(p.stem for p in source.glob("*.pkl") if p.name.startswith(("SH.", "SZ.")))
    if cfg["max_symbols"]:
        symbols = symbols[:cfg["max_symbols"]]
    raw_files = [(name, size, mtime) for name, size, mtime in metadata["files"] if Path(name).stem in set(symbols)]
    for name, size, mtime in raw_files:
        stat = (Path(cfg["data_root"]) / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError(f"原行情已改变：{name}")
    specs = {"schema": 1, "symbols": symbols, "features": features, "columns": columns,
             "source": str(source), "date_range": [cfg["start_date"], cfg["end_date"]], "raw_files": raw_files,
             "source_metadata_sha256": calculate_file_hash(source / "metadata.json"),
             "files": [(s, (source / f"{s}.pkl").stat().st_size, (source / f"{s}.pkl").stat().st_mtime_ns) for s in symbols],
             "code": {str(p): calculate_file_hash(p) for p in [Path(__file__), Path(inspect.getfile(calculate_atr)),
                       Path(inspect.getfile(create_finite_factor_names)), Path(inspect.getfile(add_cta_factor_scores))]}}
    key = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cfg["cache_dir"]) / key
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "panel.pkl").exists() or not (directory / "market.pkl").exists():
        pieces = directory / "symbols"
        pieces.mkdir(exist_ok=True)
        tasks = [(s, cfg, columns, str(pieces)) for s in symbols]
        files = []
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as executor:
            for index, file in enumerate(executor.map(build_original_symbol_file, tasks), 1):
                files.append(file)
                if index % 100 == 0 or index == len(tasks):
                    print(f"复用有限因子/补原递推与标签：{index}/{len(tasks)}", flush=True)
        panels, markets = [], []
        for file in files:
            part = pd.read_pickle(file)
            panels.append(part["panel"])
            markets.append(part["market"])
        panel = pd.concat(panels, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
        bases = [c.removeprefix("cs_rank_") for c in columns]
        ranked = panel.groupby("datetime", sort=False)[bases].rank(pct=True).astype("float32")
        ranked.columns = columns
        panel = pd.concat([panel.drop(columns=bases), ranked], axis=1)
        panel["target_rank"] = panel.groupby("datetime").forward_return.rank(pct=True).astype("float32")
        panel["cross_section_count"] = panel.groupby("datetime").symbol.transform("size")
        # 全截面来源：任一贡献者来自外年时不能仅删自己而继续用受污染的截面秩。
        panel["source_start"] = panel.groupby("datetime").source_start.transform("min")
        market = pd.concat(markets, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
        for name, value in [("panel", panel), ("market", market)]:
            temp = directory / f"{name}.tmp.pkl"
            value.to_pickle(temp)
            temp.replace(directory / f"{name}.pkl")
        write_json(directory / "metadata.json", specs)
    else:
        print(f"复用原模型年度隔离pkl：{directory}", flush=True)
        panel, market = pd.read_pickle(directory / "panel.pkl"), pd.read_pickle(directory / "market.pkl")
    manifest = {"key": key, "path": str(directory), "rows": len(panel), "symbols": len(symbols),
                "features": features, "columns": columns,
                "panel_sha256": calculate_file_hash(directory / "panel.pkl"),
                "market_sha256": calculate_file_hash(directory / "market.pkl")}
    return panel, market, manifest
