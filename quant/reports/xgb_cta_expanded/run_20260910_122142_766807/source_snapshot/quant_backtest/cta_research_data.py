"""A股CTA研究的数据层：只读原始分钟数据，生成可复用状态、因子PKL和报价缓存。

输入来自YAML路径和BaoStock公开接口，输出按来源指纹隔离；不读取研究结果选股，
不覆盖旧缓存。历史ST缺失禁止新入场，公司行动现金/税务仍须在报告披露。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import signal
import time

import numpy as np
import pandas as pd

from .annual_panel import load_symbol_bars, write_json
from .factor_library import build_factor_library
from scripts.run_dynamic_high_vol_method_backtests import build_weekly_membership

SCHEMA_VERSION = 1
QUOTE_FIELDS = ["close", "high", "low", "amount", "atr", "ema20", "limit_up", "limit_down"]
CONTEXT_NAMES = {
    "ctx_momentum5": "当前价相对五个交易日前收盘的收益",
    "ctx_momentum20": "当前价相对二十个交易日前收盘的收益",
    "ctx_ema20_gap": "当前价相对上一已完成日EMA20的偏离",
    "ctx_ema5_gap": "当前价相对上一已完成日EMA5的偏离",
    "ctx_breakout10": "当前价相对过去十个已完成日最高价的偏离",
    "ctx_atr_ratio": "上一已完成日ATR14与当前价之比",
    "ctx_daily_return": "当前价相对历史参考前收盘的收益",
    "ctx_pool_return": "当时合资格周池平均日内收益",
    "ctx_pool_breadth": "当时合资格周池上涨比例",
    "ctx_pool_momentum20": "当时合资格周池平均二十日动量",
    "ctx_board": "板块编码：主板0、创业板1、科创板2",
}


def calculate_sha256(path: Path) -> str:
    """流式计算path文件指纹，避免把大报价文件整体复制进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_identity(value: object) -> str:
    """将value的稳定JSON表示映射为缓存目录键。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def save_pickle(path: Path, value: object) -> None:
    """把value原子保存到生成物path，不改原始输入文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pkl")
    pd.to_pickle(value, temporary)
    temporary.replace(path)


def read_history(directory: Path, symbol: str) -> pd.DataFrame:
    """从directory读取symbol.pkl或symbol.csv的授权历史日线，验证结构但不擅自补未知状态。"""
    path = directory / f"{symbol}.pkl"
    if path.exists():
        frame = pd.read_pickle(path).copy()
    else:
        frame = pd.read_csv(directory / f"{symbol}.csv")
    required = ["date", "open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"]
    if not set(required).issubset(frame):
        raise ValueError(f"{symbol}历史数据缺少必需列：{set(required) - set(frame)}")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError(f"{symbol}历史数据日期缺失或重复")
    for key in required[1:]:
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def find_history_path(directory: Path, symbol: str) -> Path:
    """在directory优先定位symbol的PKL，否则定位CSV，不将不存在文件当成正常状态。"""
    path = directory / f"{symbol}.pkl"
    return path if path.exists() else directory / f"{symbol}.csv"


def classify_board(symbol: str) -> int:
    """按symbol识别普通A股板块；主板0、创业板1、科创板2，不支持品种返回-1。"""
    if symbol.startswith("SH.688"):
        return 2
    if symbol.startswith(("SZ.300", "SZ.301")):
        return 1
    if symbol.startswith(("SH.600", "SH.601", "SH.603", "SH.605", "SZ.000", "SZ.001", "SZ.002", "SZ.003")):
        return 0
    return -1


def create_minute_clock(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """为dates创建沪深连续交易分钟结束时钟，跳过午休，不生成竞价前或盘后分钟。"""
    offsets = np.r_[np.arange(571, 691), np.arange(781, 901)]
    return pd.DatetimeIndex((dates.normalize().to_numpy()[:, None] + offsets[None, :] * np.timedelta64(1, "m")).ravel())


def create_signal_clock(dates: pd.DatetimeIndex, step: int) -> pd.DatetimeIndex:
    """从每个交易日首个完成分钟起按step采样，不使用股票各自缺失行号作时钟。"""
    if step < 1 or 240 % step:
        raise ValueError("信号间隔必须整除240分钟")
    return create_minute_clock(dates)[::step]


def initialize_provider(timeout: int) -> None:
    """每个下载进程用timeout秒连接公开匿名日线服务，不在多线程间共享socket。"""
    import baostock as bs
    import baostock.common.context as context
    previous = getattr(context, "default_socket", None)
    if previous is not None:
        previous.close()
    socket.setdefaulttimeout(timeout)
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock登录失败：{result.error_msg}")


def raise_download_timeout(signum, frame) -> None:
    """给下载进程的单次请求设置硬超时；signum/frame为系统回调参数，不读行情。"""
    raise TimeoutError("历史日线请求超过限定时间")


def download_history(task: tuple) -> dict:
    """按task=(symbol,配置,日线缓存目录)下载或复用日状态，不用当前名称替代isST。"""
    import baostock as bs
    symbol, cfg, directory = task
    path = Path(directory) / f"{symbol}.pkl"
    if path.exists():
        frame = pd.read_pickle(path)
        return {"symbol": symbol, "rows": len(frame), "cache_hit": True, "error": ""}
    denied = Path(directory) / "provider_access_denied.json"
    if denied.exists():
        return {"symbol": symbol, "rows": 0, "cache_hit": False, "error": "数据服务明确拒绝访问，未重试"}
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,tradestatus,pctChg,isST"
    signal.signal(signal.SIGALRM, raise_download_timeout)
    for attempt in range(3):
        try:
            signal.alarm(cfg["network_timeout"])
            result = bs.query_history_k_data_plus(symbol.lower(), fields, start_date="2022-09-01",
                                                end_date=cfg["end_date"], frequency="d", adjustflag="3")
            if result.error_code != "0":
                if "黑名单" in result.error_msg or "权限" in result.error_msg:
                    write_json(denied, {"message": result.error_msg, "action": "停止请求，等待用户提供授权数据源"})
                    return {"symbol": symbol, "rows": 0, "cache_hit": False, "error": result.error_msg}
                raise RuntimeError(result.error_msg)
            frame = result.get_data()
            if result.error_code != "0":
                raise RuntimeError(result.error_msg)
            if frame.empty:
                raise ValueError("历史日线为空")
            frame["date"] = pd.to_datetime(frame.date)
            if frame.date.duplicated().any():
                raise ValueError("历史日线日期重复")
            for col in frame.columns.difference(["date", "code"]):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.sort_values("date").reset_index(drop=True)
            save_pickle(path, frame)
            return {"symbol": symbol, "rows": len(frame), "cache_hit": False, "error": ""}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(1)
                initialize_provider(cfg["network_timeout"])
        finally:
            signal.alarm(0)
    return {"symbol": symbol, "rows": 0, "cache_hit": False, "error": error}


def build_daily_statistics(task: tuple) -> str:
    """按task=(股票,配置,目录)保存分钟日波动与质量统计，复用匹配的旧日波动值。"""
    symbol, cfg, directory = task
    path = Path(directory) / f"{symbol}.pkl"
    if path.exists():
        return str(path)
    raw, audit = load_symbol_bars(Path(cfg["data_root"]), symbol, cfg["start_date"], cfg["end_date"])
    day = raw.datetime.dt.normalize()
    minute_return = raw.groupby(day).close.pct_change(fill_method=None)
    daily = raw.assign(minute_return=minute_return, trade_date=day).groupby("trade_date").agg(
        minute_volatility=("minute_return", "std"), amount=("amount", "sum"), bar_count=("close", "count"),
        close=("close", "last"), high=("high", "max"), low=("low", "min"))
    daily["symbol"] = symbol
    save_pickle(path, {"daily": daily.reset_index(), "audit": audit})
    return str(path)


def calculate_daily_context(history: pd.DataFrame, cfg: dict, board: int) -> pd.DataFrame:
    """仅用history已完成日构造次日ATR/均线/通道和当日公开状态；cfg指定窗口，board指定涨跌幅。"""
    daily = history.set_index("date").sort_index().copy()
    true_range = pd.concat([daily.high - daily.low, (daily.high - daily.preclose).abs(),
                            (daily.low - daily.preclose).abs()], axis=1).max(axis=1)
    daily["atr"] = true_range.rolling(cfg["atr_days"], min_periods=cfg["atr_days"]).mean().shift(1)
    for window in [5, 20]:
        daily[f"ema{window}"] = daily.close.ewm(span=window, adjust=False, min_periods=window).mean().shift(1)
        daily[f"past_close{window}"] = daily.close.shift(window)
    daily["past_high10"] = daily.high.rolling(10, min_periods=10).max().shift(1)
    # 过去交易日数量只会增加；不按数据末尾是否存续过滤股票。
    daily["observed_days"] = (daily.tradestatus.eq(1)).cumsum().shift(1).fillna(0)
    limit_rate = pd.Series(0.20 if board else 0.10, index=daily.index)
    if board == 0:
        limit_rate = limit_rate.where(~daily.isST.eq(1), 0.05)
    for name, sign in [("limit_up", 1), ("limit_down", -1)]:
        daily[name] = np.floor(daily.preclose * (1 + sign * limit_rate) * 100 + 0.5 + 1e-8) / 100
    # 当前日收盘/日成交额等未来字段不返回给特征或撮合器。
    columns = ["atr", "ema5", "ema20", "past_close5", "past_close20", "past_high10", "observed_days",
               "preclose", "limit_up", "limit_down", "isST", "tradestatus"]
    return daily[columns]


def build_symbol_features(task: tuple) -> str:
    """按task=(股票,配置,缓存,交易日,入池日)复用分钟因子，补历史日尺度特征和逐分钟报价。

    原274因子完全匹配时间键时直接使用；缺行仅重算该股票。推断行不依赖未来标签。
    """
    symbol, cfg, directory, dates, member_dates = task
    directory = Path(directory)
    output = directory / "symbols" / f"{symbol}.pkl"
    if output.exists():
        return str(output)
    raw, audit = load_symbol_bars(Path(cfg["data_root"]), symbol, cfg["start_date"], cfg["end_date"])
    raw = raw.set_index("datetime")
    history = read_history(Path(cfg["history_path"]), symbol)
    board = classify_board(symbol)
    daily = calculate_daily_context(history, cfg, board)
    context = daily.reindex(raw.index.normalize()).set_axis(raw.index)
    raw = raw.join(context)
    times = create_signal_clock(pd.DatetimeIndex(dates), cfg["base_signal_minutes"])
    times = times[times.normalize().isin(pd.to_datetime(member_dates))]
    at_signal = raw.reindex(times)
    valid = at_signal.isST.eq(0) & at_signal.tradestatus.eq(1) & (at_signal.observed_days >= cfg["minimum_history_days"])
    valid &= (at_signal.close > at_signal.limit_down) & (at_signal.close < at_signal.limit_up)
    valid &= (at_signal.amount >= cfg["minimum_signal_amount"]) & (at_signal.volume > 0) & (at_signal.atr > 0)
    times = times[valid]
    reused = False
    old = Path(cfg["reuse_factor_cache"]) / f"{symbol}.pkl"
    if cfg.get("reuse_layout") == "cta":
        old = old.parent / "symbols" / old.name
    if cfg["reuse_validated"] and old.exists():
        key = "features" if cfg.get("reuse_layout") == "cta" else "panel"
        previous = pd.read_pickle(old)[key].set_index("datetime")
        if times.isin(previous.index).all():
            excluded = {"symbol", "label_end", "forward_return"}
            # CTA分股票文件保留的是原因子；不复用旧日状态上下文或旧池截面秩。
            excluded.update(c for c in previous if c.startswith(("ctx_", "cs_rank_")))
            factors = previous.loc[times, previous.columns.difference(excluded)].copy()
            reused = True
    if not reused:
        factors = build_factor_library(raw.reset_index()[["datetime", "open", "high", "low", "close", "volume", "amount"]])
        factors.index = raw.index
        factors = factors.reindex(times)
    factors = factors.replace([np.inf, -np.inf], np.nan).astype("float32")
    current = raw.reindex(times)
    for window in [5, 20]:
        factors[f"ctx_momentum{window}"] = (current.close / current[f"past_close{window}"] - 1).astype("float32")
        factors[f"ctx_ema{window}_gap"] = (current.close / current[f"ema{window}"] - 1).astype("float32")
    factors["ctx_breakout10"] = (current.close / current.past_high10 - 1).astype("float32")
    factors["ctx_atr_ratio"] = (current.atr / current.close).astype("float32")
    factors["ctx_daily_return"] = (current.close / current.preclose - 1).astype("float32")
    factors["ctx_board"] = np.float32(board)
    factors["symbol"] = symbol
    factors.index.name = "datetime"
    quotes = raw[QUOTE_FIELDS].copy()
    quotes["allow_buy"] = (raw.isST.eq(0) & raw.tradestatus.eq(1) & (raw.observed_days >= cfg["minimum_history_days"])).astype("uint8")
    # 日状态与分钟成交相矛盾时保留原始报价用于审计，但两侧均不撮合。
    quotes["status_known"] = (raw.isST.isin([0, 1]) & raw.tradestatus.eq(1)).astype("uint8")
    quotes["tradestatus"] = raw.tradestatus
    quotes["is_st"] = raw.isST.eq(1).astype("uint8")
    quotes["volume"] = raw.volume
    audit.update(symbol=symbol, board=board, factors_reused=reused, feature_rows=len(factors))
    local_close = raw.close.groupby(raw.index.normalize()).last()
    reference = history.set_index("date").close.reindex(local_close.index)
    difference = (local_close - reference).abs()
    audit["daily_close_mismatch_count"] = int((difference > 0.021).sum())
    audit["daily_close_compared"] = int(reference.notna().sum())
    audit["status_missing_minutes"] = int(raw.isST.isna().sum())
    audit["non_normal_active_minutes"] = int((~raw.tradestatus.eq(1) & raw.volume.gt(0) & raw.amount.gt(0)).sum())
    audit["reused_factor_source"] = str(old) if reused else "当前只读分钟行情重新计算"
    save_pickle(output, {"features": factors.reset_index(), "quotes": quotes, "audit": audit})
    return str(output)


def select_history_universe(symbols: list[str], history_dir: Path, cfg: dict) -> tuple[list[str], dict]:
    """按cfg显式覆盖政策筛选symbols；history_dir只查本地文件，绝不以未来收益或全期无缺失选股。"""
    policy = cfg.get("history_coverage_policy", "require_all")
    if policy not in {"require_all", "available_local"}:
        raise ValueError("history_coverage_policy仅支持require_all或available_local")
    if policy == "available_local" and not cfg.get("history_data_dir"):
        raise ValueError("本地子集模式必须显式指定history_data_dir，禁止网络补齐")
    available = {s for s in symbols if find_history_path(history_dir, s).exists()}
    selected = [s for s in symbols if policy == "require_all" or s in available]
    if not selected:
        raise ValueError("本地没有可用股票历史状态，不能进行有效训练")
    selected_set = set(selected)
    names = ["主板", "创业板", "科创板"]
    scope = {"policy": policy, "eligible_raw_count": len(symbols), "selected_count": len(selected),
        "selected_symbols": selected, "excluded_symbols": [s for s in symbols if s not in selected_set],
        "boards": [{"板块": name, "原始合格股票": sum(classify_board(s) == i for s in symbols),
                    "本轮股票": sum(classify_board(s) == i for s in selected)} for i, name in enumerate(names)],
        "limitation": ("仅为已缓存历史状态的本地覆盖子集，下载顺序偏向沪市主板；不是全市场或均衡三板块样本。"
                       "周度Top300仅在该子集中计算，未要求股票在整个未来期间持续存在或无缺失。"
                       if policy == "available_local" else "要求所选原始股票全部具备历史状态文件；逐日未知状态仍禁止新买。")}
    return selected, scope


def prepare_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按cfg下载历史状态、构造三板块周池及扩展PKL；返回特征、周池和来源清单。

    不加载任何旧测试预测；下载失败股票留审计并禁止入场，不悄悄用当前ST快照补齐。
    """
    root, cache_root = Path(cfg["data_root"]), Path(cfg["cache_dir"])
    paths = sorted(root.glob("[0-9][0-9][0-9][0-9]/*.csv"))
    symbols = sorted({p.stem for p in paths if classify_board(p.stem) >= 0})
    if cfg["max_symbols"]:
        # 小样本均衡覆盖三个板块，不能只取排序前几只主板。
        symbols = [s for board in range(3)
                   for s in [x for x in symbols if classify_board(x) == board][:cfg["max_symbols"]]]
    history_dir = Path(cfg["history_data_dir"]) if cfg.get("history_data_dir") else cache_root / f"daily_history_{cfg['end_date']}"
    symbols, scope = select_history_universe(symbols, history_dir, cfg)
    print(f"历史覆盖政策{scope['policy']}：使用{len(symbols)}/{scope['eligible_raw_count']}只；{scope['limitation']}", flush=True)
    chosen = set(symbols)
    files = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.stem in chosen]
    identity = {"schema": SCHEMA_VERSION, "files": files, "symbols": symbols, "start": cfg["start_date"],
                "end": cfg["end_date"], "settings": {k: cfg[k] for k in ["universe_size", "min_week_days", "base_signal_minutes",
                "minimum_history_days", "atr_days", "minimum_signal_amount"]},
                "factor_code": calculate_sha256(Path(__file__).with_name("factor_library.py")),
                "data_code": calculate_sha256(Path(__file__))}
    identity["universe_scope"] = scope
    reuse_root = Path(cfg["reuse_factor_cache"])
    reuse_meta = reuse_root / "complete.json"
    if reuse_meta.exists():
        identity["factor_reuse_source"] = {"path": str(reuse_root), "manifest_sha256": calculate_sha256(reuse_meta),
            "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted((reuse_root / "symbols").glob("*.pkl"))]}
    task_symbols = sorted(set(symbols + [cfg["benchmark_symbol"].upper()]))
    identity["history_source"] = str(history_dir)
    identity["history_files"] = {s: calculate_sha256(find_history_path(history_dir, s))
                                 for s in task_symbols if find_history_path(history_dir, s).exists()}
    directory = cache_root / calculate_identity(identity)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "universe_scope.json", scope)
    if (directory / "complete.json").exists():
        manifest = json.loads((directory / "complete.json").read_text())
        print(f"复用三板块扩展特征PKL：{directory}", flush=True)
        return pd.read_pickle(directory / "panel.pkl"), pd.read_pickle(directory / "membership.pkl"), manifest
    if not cfg.get("history_data_dir"):
        history_dir.mkdir(parents=True, exist_ok=True)
    cached = [s for s in task_symbols if find_history_path(history_dir, s).exists()]
    missing = [s for s in task_symbols if s not in set(cached)]
    download_audit = [{"symbol": s, "rows": len(read_history(history_dir, s)), "cache_hit": True, "error": ""} for s in cached]
    if missing and (cfg.get("history_data_dir") or (history_dir / "provider_access_denied.json").exists()):
        write_json(directory / "data_blocker.json", {"required": len(task_symbols), "available": len(cached),
            "missing_symbols": missing, "reason": "数据服务已明确拒绝请求，不能继续全市场回测或用当前名称补历史ST"})
        raise PermissionError(f"历史数据尚缺{len(missing)}只；服务明确拒绝访问，需授权数据源。已有缓存保留。")
    if missing:
        with ProcessPoolExecutor(max_workers=cfg["download_workers"], initializer=initialize_provider,
                                 initargs=(cfg["network_timeout"],)) as pool:
            tasks = [(s, cfg, str(history_dir)) for s in missing]
            for i, result in enumerate(pool.map(download_history, tasks), 1):
                download_audit.append(result)
                if i % 100 == 0 or i == len(tasks):
                    print(f"历史日状态 {i + len(cached)}/{len(task_symbols)}，失败{sum(bool(x['error']) for x in download_audit)}", flush=True)
    else:
        print(f"离线复用{len(cached)}份已下载历史日状态，不连接外部服务", flush=True)
    pd.DataFrame(download_audit).to_csv(directory / "history_download_audit.csv", index=False, encoding="utf-8-sig")
    success = {x["symbol"] for x in download_audit if not x["error"]}
    unresolved = sorted(set(task_symbols) - success)
    if unresolved:
        write_json(directory / "data_blocker.json", {"required": len(task_symbols), "available": len(success),
            "missing_symbols": unresolved, "reason": "历史数据覆盖不完整，未启动完整股票池训练"})
        raise ValueError(f"{len(unresolved)}只历史数据未能获得，不能静默缩小全市场股票池")
    symbols = [s for s in symbols if s in success]
    if not symbols or cfg["benchmark_symbol"].upper() not in success:
        raise ValueError("没有有效历史状态或基准数据，不启动误导性训练")
    identity["history_files"] = {s: calculate_sha256(find_history_path(history_dir, s)) for s in task_symbols if s in success}
    directory = cache_root / calculate_identity(identity)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(download_audit).to_csv(directory / "history_download_audit.csv", index=False, encoding="utf-8-sig")
    old_meta = Path(cfg["reuse_factor_cache"]) / "metadata.json"
    cfg = {**cfg, "history_path": str(history_dir), "reuse_validated": False}
    if reuse_meta.exists():
        previous = json.loads(reuse_meta.read_text())["identity"]
        old_files = {row[0]: tuple(row[1:]) for row in previous["files"]}
        cfg["reuse_validated"] = (previous["factor_code"] == identity["factor_code"]
            and all(old_files.get(name) == (size, stamp) for name, size, stamp in files))
        cfg["reuse_layout"] = "cta"
    elif old_meta.exists():
        previous = json.loads(old_meta.read_text())
        old_files = {row[0]: tuple(row[1:]) for row in previous["files"]}
        cfg["reuse_validated"] = (previous["factor_code_sha256"] == identity["factor_code"]
            and all(old_files.get(name, (size, stamp)) == (size, stamp) for name, size, stamp in files))
    daily_dir = cache_root / ("minute_daily_" + calculate_identity({"files": files, "range": [cfg["start_date"], cfg["end_date"]]}))
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_frames = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        tasks = [(s, cfg, str(daily_dir)) for s in symbols]
        for i, path in enumerate(pool.map(build_daily_statistics, tasks), 1):
            daily_frames.append(pd.read_pickle(path)["daily"])
            if i % 250 == 0 or i == len(tasks):
                print(f"三板块历史分钟日统计 {i}/{len(tasks)}", flush=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    dates = pd.DatetimeIndex(sorted(daily.trade_date.unique()))
    # 历史日内数据只能用于下一周选股；日ST为该日生效状态，不读取日终价格过滤本日信号。
    eligible_daily = []
    for symbol, frame in daily.groupby("symbol", sort=False):
        history = read_history(history_dir, symbol)
        status = calculate_daily_context(history, cfg, classify_board(symbol))
        frame = frame.join(status[["isST", "tradestatus", "observed_days"]], on="trade_date")
        mask = frame.isST.eq(0) & frame.tradestatus.eq(1) & (frame.observed_days >= cfg["minimum_history_days"]) & (frame.bar_count >= 120)
        eligible_daily.append(frame.loc[mask])
    daily = pd.concat(eligible_daily, ignore_index=True)
    daily["trade_date"] = daily.trade_date.dt.date
    membership = build_weekly_membership(daily, dates.date, cfg["universe_size"], cfg["min_week_days"])
    save_pickle(directory / "membership.pkl", membership)
    member_dates = membership.groupby("symbol").trade_date.unique()
    frames, audits = [], []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        tasks = [(s, cfg, str(directory), dates, d) for s, d in member_dates.items()]
        for i, path in enumerate(pool.map(build_symbol_features, tasks), 1):
            value = pd.read_pickle(path)
            frames.append(value["features"])
            audits.append(value["audit"])
            if i % 100 == 0 or i == len(tasks):
                print(f"三板块特征 {i}/{len(tasks)}；已复用{sum(x['factors_reused'] for x in audits)}只原因子", flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    base = [c for c in panel if c not in {"datetime", "symbol"} and not c.startswith("ctx_")]
    # 在新增板块后的真实当前周池重算截面秩，旧股票池的秩不能直接搬用。
    ranks = panel.groupby("datetime", sort=False)[base].rank(pct=True).astype("float32")
    ranks.columns = [f"cs_rank_{c}" for c in base]
    panel = pd.concat([panel.drop(columns=base), ranks], axis=1)
    panel["ctx_pool_return"] = panel.groupby("datetime").ctx_daily_return.transform("mean").astype("float32")
    panel["ctx_pool_breadth"] = panel.ctx_daily_return.gt(0).groupby(panel.datetime).transform("mean").astype("float32")
    panel["ctx_pool_momentum20"] = panel.groupby("datetime").ctx_momentum20.transform("mean").astype("float32")
    save_pickle(directory / "panel.pkl", panel)
    pd.DataFrame(audits).to_csv(directory / "data_quality.csv", index=False, encoding="utf-8-sig")
    manifest = {"identity": identity, "cache_path": str(directory), "history_path": str(history_dir), "universe_scope": scope,
                "rows": len(panel), "feature_count": len(panel.columns) - 2, "symbols": sorted(panel.symbol.unique()),
                "dates": [str(x.date()) for x in dates], "raw_symbol_count": len(chosen), "history_symbol_count": len(success) - 1,
                "reused_factor_symbols": sum(x["factors_reused"] for x in audits),
                "daily_close_mismatch_count": sum(x["daily_close_mismatch_count"] for x in audits),
                "non_normal_active_minutes": sum(x["non_normal_active_minutes"] for x in audits),
                "panel_sha256": calculate_sha256(directory / "panel.pkl"),
                "status_policy": "历史isST=0且历史足够才允许新买；tradestatus必须为1才允许撮合买卖，未知或非正常一律拒绝；未用当前名称快照"}
    write_json(directory / "complete.json", manifest)
    return panel, membership, manifest


class MinuteMarket:
    """只读分钟矩阵，供路径标签及共享账户重复使用；缺失成交绝不前向填充。"""

    def __init__(self, directory: Path):
        """directory须为build_minute_market完成的独立缓存。"""
        self.directory = directory
        self.metadata = json.loads((directory / "complete.json").read_text())
        self.symbols = self.metadata["symbols"]
        self.symbol_index = {s: i for i, s in enumerate(self.symbols)}
        self.times = pd.DatetimeIndex(np.load(directory / "times.npy"))
        self.quotes = np.load(directory / "quotes.npy", mmap_mode="r")
        self.flags = np.load(directory / "flags.npy", mmap_mode="r")
        self.boards = np.array([classify_board(s) for s in self.symbols], dtype=np.int64)


def build_minute_market(manifest: dict) -> MinuteMarket:
    """由manifest的单股票报价PKL构建可复用分钟mmap；只生成执行缓存，不重建特征。"""
    root = Path(manifest["cache_path"])
    directory = root / "minute_market"
    if (directory / "complete.json").exists():
        return MinuteMarket(directory)
    directory.mkdir(parents=True, exist_ok=True)
    symbols = manifest["symbols"]
    times = create_minute_clock(pd.DatetimeIndex(manifest["dates"]))
    np.save(directory / "times.npy", times.to_numpy())
    quotes = np.lib.format.open_memmap(directory / "quotes.npy", mode="w+", dtype="float64", shape=(len(times), len(symbols), len(QUOTE_FIELDS)))
    flags = np.lib.format.open_memmap(directory / "flags.npy", mode="w+", dtype="uint8", shape=(len(times), len(symbols), 3))
    quotes[:] = np.nan
    flags[:] = 0
    for i, symbol in enumerate(symbols):
        frame = pd.read_pickle(root / "symbols" / f"{symbol}.pkl")["quotes"]
        position = times.get_indexer(frame.index)
        valid = position >= 0
        quotes[position[valid], i, :] = frame.loc[valid, QUOTE_FIELDS].to_numpy()
        executable = frame.volume.gt(0) & frame.amount.gt(0) & frame.status_known.eq(1)
        flags[position[valid], i, 0] = (executable & frame.allow_buy.eq(1)).to_numpy(dtype="uint8")[valid]
        flags[position[valid], i, 1] = executable.to_numpy(dtype="uint8")[valid]
        flags[position[valid], i, 2] = frame.is_st.to_numpy(dtype="uint8")[valid]
        if (i + 1) % 200 == 0 or i + 1 == len(symbols):
            print(f"分钟执行矩阵 {i + 1}/{len(symbols)}", flush=True)
    quotes.flush()
    flags.flush()
    write_json(directory / "complete.json", {"symbols": symbols, "fields": QUOTE_FIELDS,
        "time_count": len(times), "quote_shape": quotes.shape, "parent_panel_sha256": manifest["panel_sha256"]})
    return MinuteMarket(directory)
