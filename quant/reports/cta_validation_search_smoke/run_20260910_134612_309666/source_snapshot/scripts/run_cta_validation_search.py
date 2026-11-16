"""在quant根目录迭代固定三个内部验证集，不重新评估2026、不覆盖旧模型。

输入--config YAML及过去模型/特征；输出新验证预测PKL、候选表、中文曲线和逐笔账本。
--prepare-only构建/复用预测，--smoke为少量股票/规则流程验证，不算达标研究。
"""

import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.cta_validation_search import load_validation_predictions, run_search


def run_research() -> None:
    """从命令行读取YAML；烟测仅减少待交易股票和候选，不改历史模型或撮合合同。"""
    parser = argparse.ArgumentParser(description='目标：所有固定验证集夏普>2，禁止读取2026测试结果')
    parser.add_argument('--config', type=Path, default=ROOT/'config/cta_validation_search.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg['config_source'], cfg['is_smoke'] = str(args.config.resolve()), args.smoke
    for key in ['source_report', 'prediction_cache', 'output_dir']:
        cfg[key] = str((ROOT/cfg[key]).resolve())
    data, cache, folds = load_validation_predictions(cfg)
    if args.prepare_only:
        return
    if args.smoke:
        cfg['output_dir'] += '_smoke'
        cfg.update(objectives=['probability'], profiles=['regularized'], exit_ids=[0], styles=['trend', 'recovery'],
                   frequencies=[120], contexts=['none', 'breadth'], probability_gates=[0.5], save_every=1)
        for year, frame in data.items():
            symbols = sorted(frame.symbol.unique())[:12]
            data[year] = frame.loc[frame.symbol.isin(symbols)].copy()
    directory = run_search(data, cache, folds, cfg)
    print(f'阶段产物：{directory}', flush=True)


if __name__ == '__main__':
    run_research()
