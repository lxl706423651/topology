"""在quant根目录重训有限窗口开仓/持仓模型，输入YAML和原年度pkl，不读取旧测试收益。

build-only创建来源/简单ATR/有限因子支持缓存并审计，不训练；train内部选参后冻结；
evaluate仅消费同一有界数据契约。写新报告，不改原行情、旧缓存或券商，默认不评价外层。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.bounded_training import load_bounded_data, select_bounded_rows, create_finite_factor_names
from quant_backtest.learned_exit_models import build_research_folds
from scripts.run_multiscale_year_validation import load_config as load_base_config, run_training
from scripts.run_learned_exit_year_validation import run_evaluation, calculate_sha256


def load_config(args: argparse.Namespace) -> dict:
    """合并args配置，沿用费用/仓位/模型网格；额外声明有限ATR、白名单及来源日期，不放宽训练隔离。"""
    cfg = load_base_config(args)
    cfg["bounded_cache_dir"] = str((ROOT / cfg["bounded_cache_dir"]).resolve()) + ("_smoke" if args.smoke else "")
    cfg["data_contract"] = "bounded_observation_v1"
    cfg["candidate_basis"] = "修正已确认的递推量价来源、截面贡献传播、跨年内部拟合来源及全局步长抽样依赖；不从外层成绩改模型/门控/退出网格"
    cfg["atr_definition"] = "最近240个有效分钟真实波幅的简单均值（另需一条前收，非Wilder递推ATR）"
    cfg["report_note"] = "有限窗口与实际来源隔离重训；四折留一年回顾，前三折包含未来年份信息，非实盘证明。"
    cfg["entry_strategy_description"] = "仅使用原库有限窗口白名单及有界多尺度环境，逐折内部选最多154列并拟合3/5日净收益；前5门控及每批总资产10%不变。"
    cfg["report_limitations"] = "\n".join(
        line for line in cfg["report_limitations"].splitlines()
        if "旧分钟因子含EMA/RSI" not in line)
    cfg["report_limitations"] += "\n- 本轮删除EMA/RSI/ADX/MACD/累计OBV及全局序号趋势列；方法二沿用原叠加公式，只用剩余可用列，缺失整组为中性0.5，不能冒称与旧CTA数值相同。\n"
    cfg["report_limitations"] += "- ATR改为240个原始有效分钟真实波幅简单均值。旧模型不复用，开仓、持仓及回测共同消费新缓存；公式已变，不是旧Wilder ATR模型的复现。\n"
    cfg["report_limitations"] += "- 新增环境/截面特征最长20交易日，删除原60日动量/回撤两列；外层仍额外隔离60交易日。实际来源包络传播完整原截面、长期前收、20日窗口与上一自然周，来源触及留出年的内部行拒绝使用。\n"
    cfg["report_limitations"] += "- 核心拟合及虚拟入场额外要求来源不早于当年1月1日，不能借用上年验证行情；校准/验证可读取其正常历史。2026是否有核心拟合以实际split_audit.csv为准。虚拟持仓按固定键抽样，其他年份增删入口行不平移本年抽样。\n"
    cfg["report_limitations"] += "- 来源审计保守覆盖量价依赖，不保证非历史ST、观测日历、公司行动和撮合代理无偏；有限窗口浮点累计误差的合成检查不是全量逐值重算。\n"
    return cfg


def write_data_audit(cfg: dict, data: tuple) -> Path:
    """对data已构建有限面板逐行核对四折来源包络，按cfg新建中文审计目录，不训练或生成外层预测。"""
    panel, market, _, samples, manifest, cache = data
    out = Path(cfg["output_dir"]) / datetime.now().strftime("data_audit_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for fold in build_research_folds(market.times, cfg):
        selected = select_bounded_rows(panel, market.times, fold["year"], cfg["feature_embargo_trading_days"])
        for window in fold["fit"] + fold["calibration"] + fold["validation"]:
            part = selected.loc[(selected.datetime >= window["start"]) & (selected.datetime < window["end"])]
            a, b = pd.Timestamp(f"{fold['year']}-01-01"), pd.Timestamp(f"{fold['year'] + 1}-01-01")
            if part.feature_source_start.isna().any() or ((part.feature_source_start < b) & (part.datetime >= a)).any():
                raise ValueError("内部行仍有未知或留出年来源")
        rows.append({"留出年": fold["year"], "内部合格入口行": len(selected),
                     "仍有留出年量价来源行": 0, "来源缺失剔除行": int(panel.feature_source_start.isna().sum())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "四折来源隔离.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"因子代码": sorted(create_finite_factor_names())}).to_csv(out / "有限因子白名单.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"剔除因子代码": cache["excluded_base_factors"], "原因": "递推、累计或全局序号数值依赖，保守不复用"}).to_csv(out / "剔除因子清单.csv", index=False, encoding="utf-8-sig")
    write_json(out / "data_metadata.json", {"cache": cache, "annual_cache": manifest["cache_path"],
        "rows": len(panel), "exit_sample_rows": len(samples), "code_sha256": calculate_sha256(Path(__file__)),
        "test_predictions_generated": False, "model_fitting_performed": False})
    text = ["# 有限窗口数据审计", "", f"复用原库中的{cache['kept_base_factors']}个有限因子，剔除{len(cache['excluded_base_factors'])}个递推/累计或全局序号列。",
            f"面板{len(panel):,}行，虚拟持仓样本{len(samples):,}行；没有重建原274因子，没有训练或预测外层。",
            "", table.to_markdown(index=False), "", "来源缺失数是整个输入面板统计，各折重复展示，不能相加当独立样本。",
            "ATR采用240分钟真实波幅简单均值，不是Wilder递推；新增特征最长20交易日，外层60日隔离仍保留，另逐行排除实际包络触及留出年。",
            "白名单是候选输入，不是154个最终入模特征；最终筛选只在各折拟合片段完成。",
            "", "## 限制", "", cfg["report_limitations"]]
    (out / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "输出文件说明.md").write_text("# 文件角色\n\n- final_report.md：只审计数据，不能当收益报告。\n"
        "- 四折来源隔离.csv：来源筛选后的内部入口数及留出年交叉检查。\n"
        "- 有限因子白名单.csv、剔除因子清单.csv：允许候选和排除原因，不是按收益择优名单。\n"
        "- data_metadata.json：缓存路径/数量与审计代码指纹；大型pkl仍只保存在cache目录。\n", encoding="utf-8")
    print(f"有界数据审计完成：{out}", flush=True)
    return out


def run_cli() -> None:
    """解析config、烟测、build-only及阶段；默认训练内部研究，外层评价必须指定本轮冻结目录。"""
    parser = argparse.ArgumentParser(description="有限来源隔离与开仓/持仓模型重训")
    parser.add_argument("--config", type=Path, default=ROOT / "config/bounded_year_validation.yaml")
    parser.add_argument("--phase", choices=["train", "evaluate"], default="train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate":
        if args.run_dir is None or args.build_only:
            parser.error("evaluate必须指定--run-dir且不能带--build-only")
        run_evaluation(args.run_dir.resolve(), load_bounded_data)
        return
    cfg = load_config(args)
    if args.build_only:
        write_data_audit(cfg, load_bounded_data(cfg))
        return
    sources = ("quant_backtest/bounded_training.py", "scripts/run_bounded_year_validation.py")
    directory = run_training(cfg, load_bounded_data, select_bounded_rows, sources)
    (directory / "training_report.md").write_bytes((directory / "final_report.md").read_bytes())
    with (directory / "输出文件说明.md").open("a", encoding="utf-8") as handle:
        handle.write("\n- training_report.md：原内部报告归档，外层报告不能覆盖。\n"
                     "- 年份/feature_source_audit.csv：各内部年份实际最早量价来源及信号范围，随其他表一并冻结。\n")
    print(f"本轮只完成内部重训；须先检查完整四折内部启用，不能自动评价：{directory}", flush=True)


if __name__ == "__main__":
    run_cli()
