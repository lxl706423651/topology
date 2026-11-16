"""在quant根目录离线核查三板块历史状态覆盖，生成数据就绪报告，不连接数据商或券商。

输入为YAML、本地行情文件名及已授权历史日线，输出覆盖CSV/JSON和中文说明；
不训练模型、不读取旧测试预测，不把部分下载股票冒充完整研究股票池。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.cta_research_data import classify_board, find_history_path, read_history
from quant_backtest.annual_panel import write_json


def audit_data() -> None:
    """读取CLI配置并逐文件统计覆盖；新建唯一输出目录，原始行情及已下载缓存均只读。"""
    parser = argparse.ArgumentParser(description="离线检查CTA历史ST及日线覆盖")
    parser.add_argument("--config", type=Path, default=ROOT / "config/xgb_cta_walk_forward.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = ROOT / cfg["data_root"]
    history = ROOT / cfg["history_data_dir"] if cfg.get("history_data_dir") else ROOT / cfg["cache_dir"] / f"daily_history_{cfg['end_date']}"
    symbols = sorted({p.stem for p in root.glob("[0-9][0-9][0-9][0-9]/*.csv") if classify_board(p.stem) >= 0})
    rows = []
    for symbol in symbols:
        path = find_history_path(history, symbol)
        row = {"symbol": symbol, "板块": ["主板", "创业板", "科创板"][classify_board(symbol)], "history_available": path.exists()}
        if path.exists():
            frame = read_history(history, symbol)
            row.update(rows=len(frame), first_date=str(frame.date.min().date()), last_date=str(frame.date.max().date()),
                       unknown_st_rows=int((~frame.isST.isin([0, 1])).sum()), source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        rows.append(row)
    table = pd.DataFrame(rows)
    counts = table.groupby("板块").agg(本地股票数=("symbol", "size"), 已缓存历史状态=("history_available", "sum"))
    counts["仍缺历史状态"] = counts["本地股票数"] - counts["已缓存历史状态"]
    output = args.output_dir or ROOT / cfg["output_dir"] / datetime.now().strftime("data_readiness_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "history_coverage.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(output / "board_coverage.csv", encoding="utf-8-sig")
    complete = bool(table.history_available.all())
    write_json(output / "data_readiness.json", {"complete": complete, "stock_count": len(table),
        "available_stock_count": int(table.history_available.sum()), "missing_stock_count": int((~table.history_available).sum()),
        "provider_denied": (history / "provider_access_denied.json").exists(), "network_requests": 0,
        "formal_model_training_completed": False, "sharpe_target_achieved": False})
    write_json(output / "environment.json", {name: version(name) for name in ["pandas", "numpy", "xgboost", "numba", "baostock", "pytest"]})
    text = "# 三板块CTA数据就绪状态\n\n**这是数据准备报告，不是策略收益报告。**\n\n" + counts.to_markdown()
    text += "\n\nBaoStock此前返回‘黑名单用户，请与管理员联系’，已停止请求并保留拒访标记。"
    text += "本次命令完全离线，没有重试或绕过服务限制。\n\n"
    text += "完整股票池仍缺历史ST及日参考价，正式训练/年度策略验收尚未完成，不能声称夏普超过3。"
    text += "9只股票流程烟测只验证训练、撮合、报告和账务接口，不代表全市场收益。\n\n"
    text += "## 后续输入\n\n提供合法授权的历史日线目录，在config/xgb_cta_walk_forward.yaml的history_data_dir中指定。"
    text += "每股一个SH.688001.pkl或CSV，字段与转换说明见6method_xgb_cta/README.md。"
    text += "[Tushare历史ST接口](https://tushare.pro/document/2?doc_id=397)可按日期查询，但有授权积分要求；未擅自购买或使用他人凭证。\n\n"
    text += "## 文件说明\n\n- history_coverage.csv：逐股票历史状态缓存是否存在及起止日期。\n"
    text += "- board_coverage.csv：三板块覆盖数量，不包含基准指数。\n- data_readiness.json：机器可读阻塞状态。\n"
    text += "- environment.json：本轮环境版本，新增baostock及pytest已记录。\n- final_report.md：本说明。\n"
    (output / "final_report.md").write_text(text, encoding="utf-8")
    print(counts.to_string())
    print(f"离线状态报告：{output}")


if __name__ == "__main__":
    audit_data()
