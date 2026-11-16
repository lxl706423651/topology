"""在quant根目录执行扩展CTA研究，不联网获取行情、不连接券商、不覆盖旧轮次。

输入为--config YAML及已审计缓存；输出新增PKL/模型/中文收益图与逐笔交易。
--prepare-only仅准备复用缓存；--smoke使用少量股票/树，不能用作策略成绩。
"""

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.cta_expanded_features import load_expanded_panel
from quant_backtest.cta_expanded_training import run_expanded_research
from quant_backtest.cta_research_data import MinuteMarket


def load_config(path: Path, smoke: bool) -> dict:
    """合并path研究YAML与显式基础合同；smoke仅缩减计算量，不更改费用或执行约束。"""
    overrides = yaml.safe_load(path.read_text())
    base = (ROOT / overrides['base_config']).resolve()
    cfg = {**yaml.safe_load(base.read_text()), **overrides, 'config_source': str(path.resolve()), 'is_smoke': smoke}
    for key in ['base_config', 'source_cache', 'expanded_cache', 'output_dir', 'history_data_dir', 'data_root', 'cache_dir']:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if smoke:
        cfg['output_dir'] += '_smoke'
        cfg['max_features'] = 24
        cfg['model_profiles'] = [{**cfg['model_profiles'][0], 'n_estimators': 8}]
        cfg['entry_minutes_candidates'], cfg['entry_styles'] = [120], ['trend']
        cfg['net_return_gates'], cfg['probability_gates'] = [0.0], [0.5]
        cfg['validation_min_closed_lots'] = 0
    if not (0 < cfg['entry_equity_fraction'] <= 0.1 and 1 <= cfg['entry_top_n'] <= 5 and
            cfg['entry_equity_fraction'] <= cfg['max_symbol_exposure'] <= cfg['max_total_exposure'] <= 1):
        raise ValueError('仓位配置违反只做多合同')
    if cfg['end_date'] != '2026-04-30':
        raise ValueError('本轮预注册测试截止固定为2026-04-30；新时间段须建立新协议')
    return cfg


def run_research() -> None:
    """读取命令行，构建/复用特征后执行冻结协议；所有输出均写入新目录。"""
    parser = argparse.ArgumentParser(description='2023—2025内部研究，冻结后2026单次测试')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/xgb_cta_expanded.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)
    panel, manifest = load_expanded_panel(cfg, max_symbols=12 if args.smoke else 0)
    if not args.prepare_only:
        market = MinuteMarket(Path(cfg['source_cache']) / 'minute_market')
        directory = run_expanded_research(panel, market, manifest, cfg)
        print(f'研究完成：{directory}', flush=True)


if __name__ == '__main__':
    run_research()
