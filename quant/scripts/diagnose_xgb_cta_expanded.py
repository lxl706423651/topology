"""对冻结后的2026结果做预定诊断：双倍滑点和20日块抽样，不选择新模型。

输入为--report-dir；只复用已保存测试信号和报价，输出新诊断子目录及JSON。
不重算特征、不拟合、不改变原始报告/规则；费用压力测试增加计算但不连接实盘。
"""

import argparse
from pathlib import Path
import json
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.cta_research_data import MinuteMarket, calculate_sha256
from quant_backtest.cta_research_engine import run_portfolio
from quant_backtest.cta_research_reporting import audit_result, save_result, calculate_bootstrap


def run_diagnostics() -> None:
    """固定诊断唯一report-dir的信号，重复调用已存在目录时拒绝覆盖历史产物。"""
    parser = argparse.ArgumentParser(description='冻结2026策略的成本和夏普不确定性诊断')
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.report_dir
    output = directory / 'fixed_diagnostics'
    output.mkdir(exist_ok=False)
    cfg = yaml.safe_load((directory / 'resolved_config.yaml').read_text())
    rule = json.loads((directory / 'selected_rule.json').read_text())
    paths = [directory / 'frozen_test_signals.pkl', directory / 'selected_rule.json',
             directory / 'final_model/xgboost_model.json', directory / 'test_equity_curve.csv']
    identity = {str(p): calculate_sha256(p) for p in paths}
    signals = pd.read_pickle(directory / 'frozen_test_signals.pkl')
    market = MinuteMarket(Path(cfg['source_cache']) / 'minute_market')
    stress_cfg = {**cfg, 'slippage_rate': cfg['slippage_rate'] * 2}
    result = run_portfolio(signals, market, pd.Timestamp('2026-01-01'),
                           pd.Timestamp(cfg['end_date']) + pd.Timedelta(days=1), stress_cfg)
    result['audit'] = audit_result(result, market, stress_cfg)
    save_result(output / 'double_slippage', result, rule, stress_cfg, '扩展XGB CTA：2026固定信号双倍滑点压力测试')
    daily = pd.read_csv(directory / 'test_equity_curve.csv', float_precision='round_trip')
    bootstrap = calculate_bootstrap(daily.daily_return.to_numpy(), cfg['random_seed'])
    write_json(output / 'sharpe_uncertainty.json', bootstrap)
    assert all(calculate_sha256(Path(p)) == digest for p, digest in identity.items())
    write_json(output / 'diagnostic_manifest.json', {'inputs': identity, 'source': calculate_sha256(Path(__file__)),
        'selection_performed': False, 'stress_metrics': result['metrics'], 'bootstrap': bootstrap})
    text = '# 冻结策略的预定诊断\n\n只将滑点从万五增至千一，固定原模型及开仓信号，不据此选择新门槛。\n\n'
    text += f"双倍滑点净收益{result['metrics']['total_return']:.2%}，日夏普{result['metrics']['sharpe']:.4f}。\n\n"
    text += f"20日连续块自助抽样的夏普95%区间：{bootstrap['sharpe_95pct_interval']}。\n\n"
    text += '区间仅刻画已有短样本收益的不确定性，不能消除多次选参、历史曝光、股票覆盖或公司行动偏差。\n'
    (output / 'README.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    run_diagnostics()
