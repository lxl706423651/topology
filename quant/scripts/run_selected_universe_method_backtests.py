"""对三类精选股票池分别运行 method1/method2 固定规则回测。

输入为本地 A 股分钟数据、method1 已训练 XGBoost 模型、严格训练期特征清单、
method2 CTA 元数据，以及当前 A 股名称缓存。脚本会构造三类股票池：联网主题
资料驱动的科技主题池、最近可用一周高波动 top300、测试集之前流动性和趋势
质量 top300。输出为每个股票池在 method1/method2 下的验证/测试指标、交易
记录、收益曲线、股票池明细和中文报告。脚本不重新训练模型、不根据测试表现
调参，原始行情数据不会被修改。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
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


OUTPUT_DIR = PROJECT_ROOT / "reports" / "selected_universe_method_backtests"
PANEL_CACHE_DIR = OUTPUT_DIR / "panel_cache"

TECH_THEME_REFERENCES = [
    {
        "source": "21世纪经济报道",
        "url": "https://www.21jingji.com/article/20260527/herald/0f0697308627701c632d54307bba5599.html",
        "note": "2026-05-27 报道 AI 基建带动存储芯片需求，存储芯片概念股受到关注。",
    },
    {
        "source": "华尔街见闻",
        "url": "https://wallstreetcn.com/articles/3773833",
        "note": "2026-06-04 报道 A 股半导体产业链、存储器、半导体材料、半导体设备等方向活跃。",
    },
    {
        "source": "证券时报",
        "url": "https://www.stcn.com/article/detail/3573069.html",
        "note": "2026-01-06 报道商业航天、锂电池、AI应用是市场热门方向。",
    },
]

TECH_THEME_CANDIDATES: list[tuple[str, str, str]] = [
    ("SH.600171", "半导体/存储", "上海贝岭"),
    ("SH.600198", "半导体/通信芯片", "大唐电信"),
    ("SH.600206", "半导体材料", "有研新材"),
    ("SH.600360", "功率半导体", "华微电子"),
    ("SH.600460", "半导体/IDM", "士兰微"),
    ("SH.600584", "半导体封测", "长电科技"),
    ("SH.600667", "半导体工程", "太极实业"),
    ("SH.600703", "半导体/LED", "三安光电"),
    ("SH.600745", "半导体/消费电子", "闻泰科技"),
    ("SH.603005", "半导体封测", "晶方科技"),
    ("SH.603160", "芯片设计", "汇顶科技"),
    ("SH.603290", "功率半导体", "斯达半导"),
    ("SH.603501", "图像传感器/芯片", "韦尔股份"),
    ("SH.603986", "存储芯片", "兆易创新"),
    ("SH.605111", "功率半导体", "新洁能"),
    ("SZ.000021", "存储/电子制造", "深科技"),
    ("SZ.000050", "显示面板", "深天马A"),
    ("SZ.000063", "通信/算力", "中兴通讯"),
    ("SZ.000670", "存储/芯片", "盈方微"),
    ("SZ.000733", "军工电子", "振华科技"),
    ("SZ.000938", "云计算/AI基础设施", "紫光股份"),
    ("SZ.000988", "激光/光通信", "华工科技"),
    ("SZ.001309", "存储模组", "德明利"),
    ("SZ.002049", "芯片设计", "紫光国微"),
    ("SZ.002079", "半导体器件", "苏州固锝"),
    ("SZ.002119", "半导体材料", "康强电子"),
    ("SZ.002156", "半导体封测", "通富微电"),
    ("SZ.002185", "半导体封测", "华天科技"),
    ("SZ.002222", "光学/激光", "福晶科技"),
    ("SZ.002236", "AI视觉/安防", "大华股份"),
    ("SZ.002241", "消费电子/AI硬件", "歌尔股份"),
    ("SZ.002371", "半导体设备", "北方华创"),
    ("SZ.002384", "PCB/电子制造", "东山精密"),
    ("SZ.002409", "半导体材料", "雅克科技"),
    ("SZ.002415", "AI视觉/安防", "海康威视"),
    ("SZ.002436", "PCB/半导体载板", "兴森科技"),
    ("SZ.002463", "AI服务器PCB", "沪电股份"),
    ("SZ.002484", "电子元件", "江海股份"),
    ("SZ.002579", "PCB", "中京电子"),
    ("SZ.002600", "消费电子/AI硬件", "领益智造"),
    ("SZ.002859", "电子材料", "洁美科技"),
    ("SZ.002916", "AI服务器PCB", "深南电路"),
    ("SZ.003031", "军工电子", "中瓷电子"),
    ("SH.600271", "软件/数字政务", "航天信息"),
    ("SH.600536", "基础软件", "中国软件"),
    ("SH.600570", "金融科技", "恒生电子"),
    ("SH.600588", "企业软件", "用友网络"),
    ("SH.600633", "AI应用/传媒", "浙数文化"),
    ("SH.600718", "软件/AI应用", "东软集团"),
    ("SH.600728", "智能城市/AI", "佳都科技"),
    ("SH.600756", "软件", "浪潮软件"),
    ("SH.600797", "云计算", "浙大网新"),
    ("SH.600845", "工业软件", "宝信软件"),
    ("SH.600850", "数字化/信创", "电科数字"),
    ("SH.600986", "AI营销", "浙文互联"),
    ("SH.603000", "AI传媒", "人民网"),
    ("SH.603019", "算力服务器", "中科曙光"),
    ("SH.603189", "AI应用/视频", "网达软件"),
    ("SH.603228", "PCB", "景旺电子"),
    ("SH.603533", "AI内容", "掌阅科技"),
    ("SZ.000034", "云计算/IT分销", "神州数码"),
    ("SZ.000066", "信创/计算机", "中国长城"),
    ("SZ.000158", "华为/AI算力", "常山北明"),
    ("SZ.000555", "金融科技", "神州信息"),
    ("SZ.000681", "AI内容/版权", "视觉中国"),
    ("SZ.000815", "数据中心", "美利云"),
    ("SZ.000977", "AI服务器", "浪潮信息"),
    ("SZ.002065", "软件", "东华软件"),
    ("SZ.002095", "互联网应用", "生意宝"),
    ("SZ.002123", "AI通信应用", "梦网科技"),
    ("SZ.002131", "AI营销", "利欧股份"),
    ("SZ.002153", "行业软件", "石基信息"),
    ("SZ.002195", "AI应用", "岩山科技"),
    ("SZ.002229", "算力/AI应用", "鸿博股份"),
    ("SZ.002230", "AI语音", "科大讯飞"),
    ("SZ.002261", "AI教育/华为", "拓维信息"),
    ("SZ.002268", "网络安全", "电科网安"),
    ("SZ.002291", "AI营销", "遥望科技"),
    ("SZ.002315", "跨境电商/AI应用", "焦点科技"),
    ("SZ.002368", "信创/软件", "太极股份"),
    ("SZ.002400", "AI营销", "省广集团"),
    ("SZ.002405", "智能驾驶/地图", "四维图新"),
    ("SZ.002410", "建筑软件/AI应用", "广联达"),
    ("SZ.002439", "网络安全", "启明星辰"),
    ("SZ.002555", "游戏/AI应用", "三七互娱"),
    ("SZ.002558", "游戏/AI应用", "巨人网络"),
    ("SZ.002602", "游戏/AI应用", "世纪华通"),
    ("SZ.002605", "游戏/AI应用", "姚记科技"),
    ("SZ.002624", "游戏/AI应用", "完美世界"),
    ("SZ.002642", "云计算", "荣联科技"),
    ("SZ.002649", "软件外包/AI应用", "博彦科技"),
    ("SZ.002657", "金融科技", "中科金财"),
    ("SZ.002912", "网络安全", "中新赛克"),
    ("SH.600110", "锂电铜箔", "诺德股份"),
    ("SH.600152", "锂电池", "维科技术"),
    ("SH.600482", "动力电池/军工", "中国动力"),
    ("SH.600549", "电池材料", "厦门钨业"),
    ("SH.600884", "锂电材料", "杉杉股份"),
    ("SH.600989", "储能材料", "宝丰能源"),
    ("SH.601012", "光伏/新能源", "隆基绿能"),
    ("SH.603659", "锂电材料", "璞泰来"),
    ("SH.603799", "锂电材料", "华友钴业"),
    ("SH.603806", "光伏材料", "福斯特"),
    ("SH.603876", "电池铝箔", "鼎胜新材"),
    ("SH.605117", "储能逆变器", "德业股份"),
    ("SZ.000009", "锂电材料", "中国宝安"),
    ("SZ.000049", "消费电池", "德赛电池"),
    ("SZ.000155", "锂电/新能源", "川能动力"),
    ("SZ.000559", "智能汽车", "万向钱潮"),
    ("SZ.000625", "智能电动车", "长安汽车"),
    ("SZ.000792", "盐湖提锂", "盐湖股份"),
    ("SZ.002056", "磁材/新能源", "横店东磁"),
    ("SZ.002074", "动力电池", "国轩高科"),
    ("SZ.002080", "风电/电池材料", "中材科技"),
    ("SZ.002129", "光伏硅片", "TCL中环"),
    ("SZ.002176", "锂矿", "江特电机"),
    ("SZ.002240", "锂矿", "盛新锂能"),
    ("SZ.002340", "电池回收", "格林美"),
    ("SZ.002407", "电解液/新材料", "多氟多"),
    ("SZ.002460", "锂矿", "赣锋锂业"),
    ("SZ.002466", "锂矿", "天齐锂业"),
    ("SZ.002594", "动力电池/整车", "比亚迪"),
    ("SZ.002709", "电解液", "天赐材料"),
    ("SZ.002738", "锂矿", "中矿资源"),
    ("SZ.002756", "锂电材料", "永兴材料"),
    ("SZ.002812", "隔膜", "恩捷股份"),
    ("SZ.002850", "电池结构件", "科达利"),
    ("SH.600038", "航空装备", "中直股份"),
    ("SH.600118", "卫星应用", "中国卫星"),
    ("SH.600151", "商业航天", "航天机电"),
    ("SH.600316", "航空装备", "洪都航空"),
    ("SH.600372", "航空电子", "中航机载"),
    ("SH.600391", "航发产业链", "航发科技"),
    ("SH.600435", "卫星导航", "北方导航"),
    ("SH.600562", "雷达/军工电子", "国睿科技"),
    ("SH.600760", "航空装备", "中航沈飞"),
    ("SH.600765", "航空装备", "中航重机"),
    ("SH.600879", "航天电子", "航天电子"),
    ("SH.600893", "航空发动机", "航发动力"),
    ("SH.601698", "卫星通信", "中国卫通"),
    ("SZ.000768", "航空装备", "中航西飞"),
    ("SZ.002025", "航天连接器", "航天电器"),
    ("SZ.002111", "航空地面设备", "威海广泰"),
    ("SZ.002151", "卫星导航", "北斗星通"),
    ("SZ.002179", "航空连接器", "中航光电"),
    ("SZ.002214", "红外/军工", "大立科技"),
    ("SZ.002297", "航空材料", "博云新材"),
    ("SZ.002383", "卫星导航", "合众思壮"),
    ("SZ.002465", "卫星通信", "海格通信"),
    ("SZ.002544", "军工通信", "普天科技"),
    ("SZ.002829", "卫星导航/无人系统", "星网宇达"),
    ("SH.600580", "机器人/电驱", "卧龙电驱"),
    ("SH.600699", "智能汽车", "均胜电子"),
    ("SH.601689", "智能汽车", "拓普集团"),
    ("SH.603486", "服务机器人", "科沃斯"),
    ("SH.603583", "智能装备", "捷昌驱动"),
    ("SH.603728", "机器人电机", "鸣志电器"),
    ("SZ.000821", "智能装备", "京山轻机"),
    ("SZ.002008", "激光装备", "大族激光"),
    ("SZ.002031", "机器人", "巨轮智能"),
    ("SZ.002050", "机器人/热管理", "三花智控"),
    ("SZ.002139", "智能控制器", "拓邦股份"),
    ("SZ.002472", "机器人减速器", "双环传动"),
    ("SZ.002527", "工业机器人", "新时达"),
    ("SZ.002747", "工业机器人", "埃斯顿"),
    ("SZ.002920", "智能驾驶", "德赛西威"),
]


def parse_args() -> argparse.Namespace:
    """解析精选股票池对比回测参数。"""

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
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--no-panel-cache", action="store_true")
    parser.add_argument("--no-akshare-st-filter", action="store_true")
    parser.add_argument("--broker-commission-rate", type=float, default=0.0008)
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--min-commission", type=float, default=0.0)
    parser.add_argument("--volatility-top-n", type=int, default=300)
    parser.add_argument("--self-top-n", type=int, default=300)
    parser.add_argument("--recent-week-days", type=int, default=5, help="高波动池使用最近多少个交易日。")
    parser.add_argument("--self-lookback-days", type=int, default=40, help="自选池使用测试前多少个交易日。")
    parser.add_argument("--max-symbols-per-universe", type=int, default=0, help="烟测用，每个股票池最多保留多少只。")
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> all_a.AllAMethodConfig:
    """把本脚本参数转换为全 A 回测配置对象。"""

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


def read_symbol_year_data(symbol: str, data_root: Path, years: Sequence[int]) -> pd.DataFrame:
    """读取单只股票指定年份分钟数据，返回按时间排序的 DataFrame。"""

    parts: list[pd.DataFrame] = []
    for year in years:
        path = data_root / str(year) / f"{symbol}.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path, usecols=lambda col: col in {"datetime", "open", "high", "low", "close", "volume", "amount"})
        part["symbol"] = symbol
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("datetime").drop_duplicates(["symbol", "datetime"], keep="last").reset_index(drop=True)


def summarize_recent_volatility(task: tuple[str, Path, pd.Timestamp, int]) -> dict[str, Any]:
    """计算单只股票最近可用若干交易日的分钟收益波动率。"""

    symbol, data_root, end_date, recent_week_days = task
    years = sorted({end_date.year, max(end_date.year - 1, 2000)})
    frame = read_symbol_year_data(symbol, data_root, years)
    if frame.empty:
        return {"symbol": symbol, "selection_metric": np.nan, "selection_reason": "无行情"}
    frame = frame.loc[frame["datetime"] <= end_date + pd.Timedelta(days=1)].copy()
    if frame.empty:
        return {"symbol": symbol, "selection_metric": np.nan, "selection_reason": "截至选择日无行情"}
    frame["trade_date"] = frame["datetime"].dt.date
    dates = sorted(frame["trade_date"].dropna().unique())
    if not dates or max(dates) < end_date.date():
        return {
            "symbol": symbol,
            "selection_metric": np.nan,
            "selection_reason": "最近行情未覆盖选择日，排除出过去一周高波动池",
            "selection_end": str(max(dates)) if dates else "",
        }
    dates = dates[-recent_week_days:]
    window = frame.loc[frame["trade_date"].isin(dates)].copy()
    returns = window["close"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan)
    volatility = float(returns.std(skipna=True))
    amount = float(window.get("amount", pd.Series(dtype=float)).sum(skipna=True))
    return {
        "symbol": symbol,
        "selection_metric": volatility,
        "selection_start": str(min(dates)) if dates else "",
        "selection_end": str(max(dates)) if dates else "",
        "selection_amount": amount,
        "selection_reason": f"最近{recent_week_days}个交易日分钟收益波动率",
    }


def summarize_pretest_quality(task: tuple[str, Path, pd.Timestamp, int]) -> dict[str, Any]:
    """计算测试集开始前的流动性、趋势和低波动综合选股指标。"""

    symbol, data_root, split_date, lookback_days = task
    years = sorted({split_date.year, max(split_date.year - 1, 2000)})
    frame = read_symbol_year_data(symbol, data_root, years)
    if frame.empty:
        return {"symbol": symbol, "selection_metric": np.nan, "selection_reason": "无行情"}
    frame = frame.loc[frame["datetime"] < split_date].copy()
    if frame.empty:
        return {"symbol": symbol, "selection_metric": np.nan, "selection_reason": "测试前无行情"}
    frame["trade_date"] = frame["datetime"].dt.date
    dates = sorted(frame["trade_date"].dropna().unique())[-lookback_days:]
    window = frame.loc[frame["trade_date"].isin(dates)].copy()
    daily = (
        window.groupby("trade_date", sort=True)
        .agg(close=("close", "last"), amount=("amount", "sum"), bars=("close", "count"))
        .reset_index()
    )
    daily["return"] = daily["close"].astype(float).pct_change()
    if len(daily) < max(10, lookback_days // 3):
        return {"symbol": symbol, "selection_metric": np.nan, "selection_reason": "测试前有效交易日不足"}
    if pd.Timestamp(daily["trade_date"].iloc[-1]) < split_date - pd.Timedelta(days=10):
        return {"symbol": symbol, "selection_metric": np.nan, "selection_reason": "测试前最近行情过旧"}
    first_close = float(daily["close"].iloc[0])
    last_close = float(daily["close"].iloc[-1])
    momentum = last_close / first_close - 1.0 if first_close > 0 else np.nan
    volatility = float(daily["return"].std(skipna=True))
    avg_amount = float(daily["amount"].mean(skipna=True))
    return {
        "symbol": symbol,
        "selection_metric": np.nan,
        "selection_start": str(daily["trade_date"].iloc[0]),
        "selection_end": str(daily["trade_date"].iloc[-1]),
        "avg_amount": avg_amount,
        "momentum": momentum,
        "volatility": volatility,
        "active_days": int(len(daily)),
        "selection_reason": "测试前流动性+趋势质量综合分",
    }


def run_parallel_selection_metrics(
    symbols: Sequence[str],
    workers: int,
    task_factory: Any,
    metric_function: Any,
) -> pd.DataFrame:
    """并行计算股票池选择指标。"""

    rows: list[dict[str, Any]] = []
    tasks = [task_factory(symbol) for symbol in symbols]
    if workers == 1:
        for task in tasks:
            rows.append(metric_function(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(metric_function, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if idx % 500 == 0 or idx == len(futures):
                    print(f"已计算选股指标 {idx}/{len(futures)}", flush=True)
    return pd.DataFrame(rows)


def select_tech_theme_universe(
    available_symbols: set[str],
    universe_audit: pd.DataFrame,
    max_symbols: int,
) -> tuple[list[str], pd.DataFrame]:
    """按联网资料确定的科技主题直接指定股票池，并过滤到本地主板非 ST 股票。"""

    rows = []
    selected: list[str] = []
    name_by_symbol = dict(zip(universe_audit["symbol"].astype(str), universe_audit["name"].fillna("").astype(str), strict=False))
    for symbol, theme, fallback_name in TECH_THEME_CANDIDATES:
        included = symbol in available_symbols
        rows.append(
            {
                "universe_id": "tech_theme_direct",
                "symbol": symbol,
                "name": name_by_symbol.get(symbol, fallback_name),
                "theme": theme,
                "selection_metric": np.nan,
                "selection_reason": "联网资料确认科技主线后人工指定",
                "included": included,
            }
        )
        if included:
            selected.append(symbol)
    selected = sorted(dict.fromkeys(selected))
    if max_symbols:
        selected = selected[:max_symbols]
        for row in rows:
            row["included"] = bool(row["symbol"] in selected)
            if row["symbol"] not in selected and row["symbol"] in available_symbols:
                row["selection_reason"] = "max_symbols_per_universe 烟测截断"
    return selected, pd.DataFrame(rows)


def select_high_volatility_universe(
    all_symbols: Sequence[str],
    universe_audit: pd.DataFrame,
    config: all_a.AllAMethodConfig,
    top_n: int,
    recent_week_days: int,
    max_symbols: int,
) -> tuple[list[str], pd.DataFrame]:
    """选择最近可用一周分钟收益波动率最高的股票。"""

    print("正在计算最近一周高波动股票池", flush=True)
    metrics = run_parallel_selection_metrics(
        all_symbols,
        config.workers,
        lambda symbol: (symbol, config.data_root, config.end_date, recent_week_days),
        summarize_recent_volatility,
    )
    metrics = metrics.dropna(subset=["selection_metric"]).sort_values("selection_metric", ascending=False)
    selected = metrics["symbol"].head(top_n).astype(str).tolist()
    if max_symbols:
        selected = selected[:max_symbols]
    names = universe_audit.loc[:, ["symbol", "name"]].copy()
    out = metrics.merge(names, on="symbol", how="left")
    out["universe_id"] = "recent_week_high_vol_top300"
    out["included"] = out["symbol"].isin(selected)
    return selected, out


def select_pretest_quality_universe(
    all_symbols: Sequence[str],
    universe_audit: pd.DataFrame,
    config: all_a.AllAMethodConfig,
    top_n: int,
    lookback_days: int,
    max_symbols: int,
) -> tuple[list[str], pd.DataFrame]:
    """选择测试集之前流动性、趋势和低波动综合得分最高的股票。"""

    print("正在计算测试前流动性趋势质量股票池", flush=True)
    metrics = run_parallel_selection_metrics(
        all_symbols,
        config.workers,
        lambda symbol: (symbol, config.data_root, config.split_date, lookback_days),
        summarize_pretest_quality,
    )
    metrics = metrics.dropna(subset=["avg_amount", "momentum", "volatility"]).copy()
    metrics["liquidity_rank"] = metrics["avg_amount"].rank(pct=True)
    metrics["momentum_rank"] = metrics["momentum"].rank(pct=True)
    metrics["low_vol_rank"] = (-metrics["volatility"]).rank(pct=True)
    metrics["selection_metric"] = (
        0.45 * metrics["liquidity_rank"] + 0.35 * metrics["momentum_rank"] + 0.20 * metrics["low_vol_rank"]
    )
    metrics = metrics.sort_values("selection_metric", ascending=False)
    selected = metrics["symbol"].head(top_n).astype(str).tolist()
    if max_symbols:
        selected = selected[:max_symbols]
    names = universe_audit.loc[:, ["symbol", "name"]].copy()
    out = metrics.merge(names, on="symbol", how="left")
    out["universe_id"] = "pretest_liquidity_momentum_top300"
    out["included"] = out["symbol"].isin(selected)
    return selected, out


def create_universes(
    config: all_a.AllAMethodConfig,
    args: argparse.Namespace,
    all_symbols: Sequence[str],
    universe_audit: pd.DataFrame,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """生成三类股票池及其选择明细。"""

    available = set(all_symbols)
    tech_symbols, tech_audit = select_tech_theme_universe(available, universe_audit, args.max_symbols_per_universe)
    high_vol_symbols, high_vol_audit = select_high_volatility_universe(
        all_symbols,
        universe_audit,
        config,
        args.volatility_top_n,
        args.recent_week_days,
        args.max_symbols_per_universe,
    )
    quality_symbols, quality_audit = select_pretest_quality_universe(
        all_symbols,
        universe_audit,
        config,
        args.self_top_n,
        args.self_lookback_days,
        args.max_symbols_per_universe,
    )
    universes = {
        "tech_theme_direct": tech_symbols,
        "recent_week_high_vol_top300": high_vol_symbols,
        "pretest_liquidity_momentum_top300": quality_symbols,
    }
    audit = pd.concat([tech_audit, high_vol_audit, quality_audit], ignore_index=True, sort=False)
    return universes, audit


def create_strategy_specs(config: all_a.AllAMethodConfig) -> list[all_a.StrategySpec]:
    """读取 method1/method2 固定规则，生成待评估策略列表。"""

    method1_rule = all_a.load_method_rule(config.method1_run_dir)
    method2_rule = all_a.load_cta_rule(config.method2_run_dir, "cta_trend_breakout")
    return [
        all_a.StrategySpec(
            "method1_xgboost_long_only",
            "method1 XGBoost 156因子 long-only top1门控",
            "prediction",
            method1_rule,
            "复用 run_20260526_202355 的 XGBoost 模型和固定 top1 分数/分差门控。",
        ),
        all_a.StrategySpec(
            "method2_cta_trend_breakout",
            "method2 CTA trend breakout",
            "cta_trend_breakout_score",
            method2_rule,
            "复用 method2 CTA trend 包装：XGBoost 分数 + 小权重趋势/突破/动量/量价确认。",
        ),
    ]


def build_scored_panel(
    config: all_a.AllAMethodConfig,
    symbols: Sequence[str],
    manifest: pd.DataFrame,
    model: Any,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, Path | None, bool]:
    """构建或读取股票池 rank 特征面板，并生成 method1/method2 打分。"""

    panel, cache_path, loaded_from_cache = all_a.build_or_load_rank_panel(config, symbols, manifest)
    missing = [col for col in feature_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"特征面板缺少模型特征：{missing[:10]}")
    print(f"正在预测精选股票池分数：symbols={len(symbols)}, rows={len(panel)}", flush=True)
    panel["prediction"] = model.predict(panel.loc[:, feature_cols]).astype("float32")
    panel = cta.add_cta_factor_scores(panel, cta.SCORE_BLEND_STRENGTH)
    return panel, cache_path, loaded_from_cache


def run_universe_backtests(
    run_dir: Path,
    config: all_a.AllAMethodConfig,
    universes: dict[str, list[str]],
    universe_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """对每个股票池分别运行 method1/method2 固定规则回测。"""

    manifest = xb.read_feature_manifest(config.feature_manifest)
    feature_cols = (config.method1_run_dir / "feature_list.txt").read_text(encoding="utf-8").splitlines()
    model = joblib.load(config.method1_run_dir / "xgboost_model.joblib")
    xb_config = all_a.create_xb_config(config)
    strategies = create_strategy_specs(config)
    summary_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    for universe_id, symbols in universes.items():
        if len(symbols) < config.min_assets_per_rebalance:
            raise ValueError(f"{universe_id} 股票数 {len(symbols)} 小于 min_assets={config.min_assets_per_rebalance}")
        universe_dir = run_dir / universe_id
        universe_dir.mkdir(parents=True, exist_ok=True)
        (universe_dir / "selected_symbols.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")
        panel, cache_path, loaded_from_cache = build_scored_panel(config, symbols, manifest, model, feature_cols)
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
        members = universe_audit.loc[universe_audit["universe_id"].eq(universe_id) & universe_audit["included"].eq(True)].copy()
        members.to_csv(universe_dir / "universe_members.csv", index=False, encoding="utf-8-sig")
        cache_rows.append(
            {
                "universe_id": universe_id,
                "symbol_count": len(symbols),
                "panel_cache_path": str(cache_path) if cache_path else "",
                "loaded_from_cache": loaded_from_cache,
                "rows": len(panel),
                "validation_rows": len(validation),
                "test_rows": len(test),
            }
        )
        for strategy in strategies:
            print(f"正在回测 {universe_id} / {strategy.method_id}", flush=True)
            row = all_a.save_strategy_outputs(universe_dir, strategy, validation, test, xb_config, config)
            row["universe_id"] = universe_id
            row["universe_symbol_count"] = len(symbols)
            summary_rows.append(row)
    return pd.DataFrame(summary_rows), cache_rows


def write_root_output_guide(run_dir: Path) -> None:
    """写出根目录文件说明。"""

    text = """# 输出文件说明

- `final_report.md`：本次三类精选股票池 method1/method2 对比的主报告。
- `resolved_config.json`：解析后的路径、日期、费用、股票池参数和固定规则。
- `universe_selection_audit.csv`：所有候选股票的选择指标、主题、是否入选和选择原因。
- `universe_summary.csv`：每个股票池的股票数量和选择口径。
- `strategy_summary.csv`：三个股票池乘以两个方法的验证/测试核心指标汇总。
- `<universe_id>/selected_symbols.txt`：该股票池最终入选股票代码。
- `<universe_id>/universe_members.csv`：该股票池最终入选股票明细。
- `<universe_id>/split_audit.csv`：该股票池验证集和测试集样本边界。
- `<universe_id>/method1_xgboost_long_only/`：method1 在该股票池上的回测产物。
- `<universe_id>/method2_cta_trend_breakout/`：method2 在该股票池上的回测产物。
"""
    (run_dir / "输出文件说明.md").write_text(text, encoding="utf-8")


def write_final_report(
    run_dir: Path,
    config: all_a.AllAMethodConfig,
    summary: pd.DataFrame,
    universe_summary: pd.DataFrame,
    cache_rows: list[dict[str, Any]],
    st_status: str,
) -> None:
    """写出三类股票池对比中文报告。"""

    cache_frame = pd.DataFrame(cache_rows)
    lines = [
        "# 三类精选股票池 method1/method2 对比报告",
        "",
        "## 实验边界",
        "",
        "- 本报告不重新训练 XGBoost 模型，不重新选择 method1/method2 门控阈值。",
        "- 每个股票池既作为可交易股票范围，也作为本次截面 rank 特征的计算截面；method1/method2 的 top1、分数阈值、分差阈值保持原样。",
        f"- 验证集：`{config.validation_start.date()} <= datetime < {config.split_date.date()}`，只做外推诊断。",
        f"- 测试集：`datetime >= {config.split_date.date()}`，用于比较不同股票池在同一固定规则下的表现。",
        f"- ST 剔除状态：`{st_status}`。",
        "- 所有股票池都限定在本地沪深主板非 ST 股票中，剔除北交所、科创板和创业板。",
        "",
        "## 三种股票池",
        "",
    ]
    for row in universe_summary.itertuples(index=False):
        lines.append(f"- `{row.universe_id}`：{row.description}；入选 `{int(row.symbol_count)}` 只。")
    lines.extend(
        [
            "",
            "## 科技主题资料依据",
            "",
        ]
    )
    for ref in TECH_THEME_REFERENCES:
        lines.append(f"- {ref['source']}：{ref['note']} 资料：<{ref['url']}>")
    lines.extend(
        [
            "",
            "## 重要解释限制",
            "",
            "- 科技主题池是当前资料和人工指定股票池，存在主题主观性和当前成分幸存者偏差。",
            "- 最近一周高波动 top300 使用本地数据截至测试末尾附近的最近可用一周，并要求股票行情覆盖本地数据末日；该选择仍属于后验诊断，不能当作严格独立测试结论。",
            "- 自选流动性趋势质量 top300 只使用测试集开始前数据选股，更接近可复现的事前股票池规则。",
            "- 当前回测仍未完整建模涨跌停不能成交、停牌、冲击成本、成交额容量和历史点时 ST 状态。",
            "",
            "## 策略结果",
            "",
            "| 股票池 | 方法 | 股票数 | 验证 Sharpe | 测试 Sharpe | 测试年化收益 | 测试最大回撤 | 测试买入次数 | 测试信号期占比 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered = summary.sort_values(["universe_id", "method_id"])
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| `{row.universe_id}` | `{row.method_id}` | {int(row.universe_symbol_count)} | "
            f"{float(row.validation_sharpe):.6f} | {float(row.test_sharpe):.6f} | "
            f"{float(row.test_annual_return):.6f} | {float(row.test_max_drawdown):.6f} | "
            f"{int(row.test_buy_trade_count)} | {float(row.test_signal_period_ratio):.6f} |"
        )
    lines.extend(["", "## 特征缓存", ""])
    for row in cache_frame.itertuples(index=False):
        lines.append(
            f"- `{row.universe_id}`：rows=`{int(row.rows)}`，cache=`{row.panel_cache_path}`，"
            f"loaded_from_cache=`{row.loaded_from_cache}`。"
        )
    lines.extend(
        [
            "",
            "## 产物说明",
            "",
            "- `universe_selection_audit.csv`：所有候选股票和选择指标。",
            "- `universe_summary.csv`：三类股票池选择口径。",
            "- `strategy_summary.csv`：六组回测核心指标。",
            "- 每个股票池目录下保存 method1/method2 的收益曲线、交易事件、平仓复盘、持仓和指标。",
        ]
    )
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selected_universe_method_backtests(args: argparse.Namespace) -> Path:
    """执行精选股票池 method1/method2 对比回测。"""

    config = create_config(args)
    all_symbols, base_universe_audit, st_status = all_a.discover_all_a_mainboard_symbols(config)
    universes, selection_audit = create_universes(config, args, all_symbols, base_universe_audit)
    run_dir = config.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_audit.to_csv(run_dir / "universe_selection_audit.csv", index=False, encoding="utf-8-sig")
    universe_summary = pd.DataFrame(
        [
            {
                "universe_id": "tech_theme_direct",
                "symbol_count": len(universes["tech_theme_direct"]),
                "description": "联网确认科技主线后，直接指定半导体、存储、AI、AI应用、电池、商业航天、机器人等主题股票",
                "selection_bias_note": "当前主题人工池，存在当前成分和主观选择偏差",
            },
            {
                "universe_id": "recent_week_high_vol_top300",
                "symbol_count": len(universes["recent_week_high_vol_top300"]),
                "description": f"本地数据截至 {config.end_date.date()} 的最近 {args.recent_week_days} 个交易日分钟收益波动率最高 top{args.volatility_top_n}",
                "selection_bias_note": "使用测试末尾附近最近一周，是后验诊断，不是严格测试独立股票池",
            },
            {
                "universe_id": "pretest_liquidity_momentum_top300",
                "symbol_count": len(universes["pretest_liquidity_momentum_top300"]),
                "description": f"测试集开始前 {args.self_lookback_days} 个交易日，按流动性、趋势动量和低波动综合得分 top{args.self_top_n}",
                "selection_bias_note": "只使用测试前数据，三组中最接近事前可复现规则",
            },
        ]
    )
    universe_summary.to_csv(run_dir / "universe_summary.csv", index=False, encoding="utf-8-sig")
    summary, cache_rows = run_universe_backtests(run_dir, config, universes, selection_audit)
    summary.to_csv(run_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **{key: str(value) if isinstance(value, Path | pd.Timestamp) else value for key, value in asdict(config).items() if key != "fee"},
                "fee": asdict(config.fee),
                "akshare_st_filter_status": st_status,
                "volatility_top_n": args.volatility_top_n,
                "self_top_n": args.self_top_n,
                "recent_week_days": args.recent_week_days,
                "self_lookback_days": args.self_lookback_days,
                "max_symbols_per_universe": args.max_symbols_per_universe,
                "tech_theme_references": TECH_THEME_REFERENCES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_final_report(run_dir, config, summary, universe_summary, cache_rows, st_status)
    write_root_output_guide(run_dir)
    print(f"精选股票池 method 对比回测已保存到 {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """命令行入口。"""

    run_selected_universe_method_backtests(parse_args())


if __name__ == "__main__":
    main()
