"""从已冻结年度权益CSV重绘日期刻度，仅改报告图片，不重跑策略或更改数据。"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager
import pandas as pd

directory = Path(__file__).resolve().parent
cfg = json.loads((directory / 'resolved_config.json').read_text())
initial = float(cfg['initial_cash'])
years = pd.read_csv(directory / 'annual_summary.csv').test_year.tolist()
fonts = list(Path('/usr/share/fonts/opentype/noto').glob('*CJK*'))
if fonts:
    font_manager.fontManager.addfont(str(fonts[0]))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
plt.rcParams['axes.unicode_minus'] = False
output = directory / 'test_return_curve.png'
original = directory / 'test_return_curve_original.png'
if not original.exists():
    shutil.copyfile(output, original)
fig, axes = plt.subplots(len(years), 2, figsize=(14, 3.2 * len(years)), squeeze=False)
hashes = {}
for row, year in enumerate(years):
    path = directory / f'year_{year}' / 'test_equity_curve.csv'
    hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    curve = pd.read_csv(path, parse_dates=['datetime'])
    axes[row, 0].plot(curve.datetime, (curve.baseline_equity / initial - 1) * 100,
                      color='#d88729', label='周度持有')
    axes[row, 0].plot(curve.datetime, (curve.equity / initial - 1) * 100,
                      color='#2868a6', label='周度持有＋做T')
    axes[row, 0].set_title(f'{year} 留出年收益（%）')
    axes[row, 0].legend()
    axes[row, 1].plot(curve.datetime, (curve.equity - curve.baseline_equity) / initial * 100,
                      color='#1e746c')
    axes[row, 1].axhline(0, color='gray', linewidth=.7)
    axes[row, 1].set_title(f'{year} 做T净增量（百分点）')
    for ax in axes[row]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(output, dpi=160)
plt.close(fig)
shutil.copyfile(output, directory / '测试集收益曲线.png')
assert all(hashlib.sha256((directory / path).read_bytes()).hexdigest() == sha for path, sha in hashes.items())
(directory / 'plot_revision.json').write_text(json.dumps({
    'reason': '减少横轴日期刻度，解决原2023面板文字重叠；数据和策略不变',
    'original_image_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
    'revised_image_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'input_equity_csv_sha256': hashes, 'source': Path(__file__).name,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'strategy_models_or_trade_results_changed': False,
}, ensure_ascii=False, indent=2), encoding='utf-8')
