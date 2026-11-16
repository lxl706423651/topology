"""复核年度模型为何少交易，只读取训练内部校准和验证，不读取测试预测或收益。

在quant根目录传入--run-dir年度报告目录；复用既有pkl和模型，生成中文门控审计。
不会重训、改规则、连接实盘或覆盖原始行情。审计后的修改必须另立研究协议。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_annual_method_validation import generate_scores, METHODS
from quant_backtest.annual_training import select_signal_tradable_panel


def audit_training_gates(directory: Path) -> pd.DataFrame:
    """读取directory里的冻结模型和缓存，仅在各轮校准/验证期核对门控及客观可成交性。"""
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    manifest = json.loads((directory / "data_manifest.json").read_text())
    cache = Path(manifest["cache_path"])
    panel = pd.read_pickle(cache / "panel.pkl")
    market = pd.read_pickle(cache / "market.pkl")
    panel = select_signal_tradable_panel(panel, market, cfg)
    rows = []
    for year in cfg["test_years"]:
        fold_dir = directory / f"year_{year}"
        metadata = json.loads((fold_dir / "model_metadata.json").read_text())
        model = XGBRegressor()
        model.load_model(fold_dir / "xgboost_model.json")
        for stage, prefix in [("训练内部校准", "calibration"), ("验证", "validation")]:
            start, end = pd.Timestamp(metadata[f"{prefix}_start"]), pd.Timestamp(metadata[f"{prefix}_end"])
            assert end <= pd.Timestamp(metadata["test_start"])
            frame = panel.loc[panel.datetime.between(start, end, inclusive="left")]
            scores = generate_scores(model, frame, metadata["feature_columns"], cfg)
            for method, (_, column) in METHODS.items():
                training = pd.read_csv(fold_dir / f"{method}_training_rule_metrics.csv")
                threshold = float(training.loc[training["quantile"] == min(cfg["gate_quantiles"]), "threshold"].iloc[0])
                top = scores.sort_values(["datetime", column, "symbol"], ascending=[True, False, True]).groupby("datetime", sort=False).head(cfg["entry_top_n"])
                candidates = top.loc[top[column] >= threshold]
                matched = candidates.merge(market.loc[market.datetime.between(start, end, inclusive="left")], on=["datetime", "symbol"], how="left", validate="one_to_one")
                missing = matched.execution_close.isna() | (matched.execution_volume.fillna(0) <= 0)
                limit = matched.execution_close / matched.previous_close >= 1.095
                capacity = matched.execution_amount.fillna(0) * cfg["max_bar_amount_fraction"] / matched.execution_close
                small = capacity < cfg["lot_size"]
                rows.append({"测试年份": year, "方法": method, "审计阶段": stage,
                             "候选信号数": len(top), "最低候选门控": threshold, "过门控候选数": len(matched),
                             "下一分钟缺失或零量": int(missing.sum()), "涨停代理阻挡": int(limit.sum()),
                             "容量不足一手": int(small.sum()), "通过上述条件": int((~missing & ~limit & ~small).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(directory / "training_gate_audit.csv", index=False, encoding="utf-8-sig")
    (directory / "训练验证门控审计.md").write_text("# 训练与验证门控审计\n\n"
        "本审计只使用测试年之前的训练内部校准/验证数据，不读取年度测试预测或收益，不改变冻结规则。"
        "按最宽松的候选门控核对前5股票；指标可重叠，不能简单相加。通过上述条件仍不保证符合持仓/现金/ATR等全部下单条件。\n\n"
        + result.to_markdown(index=False) + "\n\n这些统计用于解释被淘汰的原因，不能直接通过放宽涨跌停或容量限制来恢复旧高收益。\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(audit_training_gates(parser.parse_args().run_dir).to_string(index=False))
