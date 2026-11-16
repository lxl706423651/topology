"""在quant根目录训练三板块XGBoost CTA并逐年向前测试，不连接券商。

输入为YAML及只读行情，输出新增PKL/报价缓存、原生模型、中文图表和交易账本。
--prepare-only只准备缓存，--smoke仅做小样本流程验证；正式研究可能耗时较长。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.cta_research_data import prepare_data, build_minute_market


def load_config(args: argparse.Namespace) -> dict:
    """读取args指定YAML和烟测选项，所有相对路径以项目ROOT为准。"""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["config_source"] = str(args.config.resolve())
    for name in ["data_root", "output_dir", "cache_dir", "reuse_factor_cache"]:
        cfg[name] = str((ROOT / cfg[name]).resolve())
    if cfg.get("history_data_dir"):
        cfg["history_data_dir"] = str((ROOT / cfg["history_data_dir"]).resolve())
    if args.workers:
        cfg["workers"] = args.workers
    cfg["is_smoke"] = args.smoke
    if args.smoke:
        cfg.update(max_symbols=3, universe_size=9, max_features=16,
                   entry_minutes_candidates=[240], entry_styles=["trend"], net_return_gates=[0.0],
                   validation_min_closed_lots=0)
        cfg["exit_profiles"] = cfg["exit_profiles"][:1]
        cfg["model_profiles"] = [{**cfg["model_profiles"][0], "n_estimators": 8}]
        cfg["output_dir"] += "_smoke"
    if (not 0 < cfg["entry_equity_fraction"] <= 0.10 or not 1 <= cfg["entry_top_n"] <= 5
            or not cfg["entry_equity_fraction"] <= cfg["max_symbol_exposure"] <= cfg["max_total_exposure"] <= 1):
        raise ValueError("仓位必须满足每笔最多总资产10%、前5、不融资，且单股/总仓上限递增")
    step = cfg["base_signal_minutes"]
    if not isinstance(step, int) or step <= 0 or 240 % step:
        raise ValueError("特征采样步长必须是整除240的正整数")
    if any(not isinstance(x, int) or x <= 0 or 240 % x or x % step for x in cfg["entry_minutes_candidates"]):
        raise ValueError("开仓频率必须整除240且是特征采样步长的整数倍")
    return cfg


def run_research() -> None:
    """解析CLI后先构建/复用缓存，再交由冻结协议的训练入口执行，不覆写旧轮次。"""
    parser = argparse.ArgumentParser(description="三板块XGBoost CTA严格向前年度验证")
    parser.add_argument("--config", type=Path, default=ROOT / "config/xgb_cta_walk_forward.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    cfg = load_config(args)
    panel, membership, manifest = prepare_data(cfg)
    market = build_minute_market(manifest)
    print(f"数据准备完成：{len(panel):,}行，{len(market.symbols)}只股票，缓存{manifest['cache_path']}", flush=True)
    if not args.prepare_only:
        from quant_backtest.cta_research_training import run_walk_forward
        output = run_walk_forward(panel, membership, market, manifest, cfg)
        print(f"本轮研究完成：{output}", flush=True)


if __name__ == "__main__":
    run_research()
