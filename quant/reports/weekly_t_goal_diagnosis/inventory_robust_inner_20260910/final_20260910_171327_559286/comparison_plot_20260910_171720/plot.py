"""仅展示已冻结烟测内部同规则比较，不改变参数、不读取工程外层。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

root = Path('/data/lxl/seed-emulator/topology/quant')
parent = root/'reports/weekly_t_goal_diagnosis/inventory_robust_inner_20260910/final_20260910_171327_559286'
source = parent/'candidate_source_comparison.csv'
sha = hashlib.sha256(source.read_bytes()).hexdigest()
out = parent/f'comparison_plot_{datetime.now():%Y%m%d_%H%M%S}'
out.mkdir()
shutil.copy2(__file__, out/'plot.py')
f = pd.read_csv(source, float_precision='round_trip')
f = f.loc[f.candidate_index.ne(0)]
assert len(f) == 36 and f.incremental_net_pnl_robust.lt(0).all()
fonts = sorted(Path('/usr/share/fonts/opentype/noto').glob('*CJK*'))
if fonts:
    font_manager.fontManager.addfont(str(fonts[0]))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4), gridspec_kw={'width_ratios': [1.45, 1]})
for tag, color, marker, label in [('source','#346994','o','原L2：36个独立候选'),
                                   ('robust','#d87928','^','Huber：36个独立候选')]:
    ax[0].scatter(f['completed_t_pair_lots_'+tag], f['incremental_net_pnl_'+tag],
        color=color, marker=marker, alpha=.8, label=label, s=40)
ax[0].axhline(0,color='#666666',lw=1)
ax[0].set(xlabel='完成配对手数（100股为1手）',ylabel='实际账户T净增量（元）',title='交易规模与扣费净额')
ax[0].grid(alpha=.2)
ax[0].legend(loc='lower left',fontsize=9)
r = f.loc[f.candidate_index.eq(33)].iloc[0]
ax[0].annotate('所选规则33：1组 / 4手 / -51.93元',
    xy=(r.completed_t_pair_lots_robust,r.incremental_net_pnl_robust), xytext=(450, 350),
    fontsize=9, arrowprops={'arrowstyle':'->','color':'#555555'})
gross = r.realized_fragment_reference_gross_pnl_change
cost = -r.t_fee_change-r.t_slippage_cost_change
net = r.incremental_net_pnl_change
assert abs(gross+cost-net) < 1e-7
values = [gross,cost,net]
bars=ax[1].bar(['参考毛差改善','费用和滑点减少','净额改善'], values, color=['#779f9c','#e0b56b','#667a96'],width=.58)
for bar,v in zip(bars,values):
    ax[1].text(bar.get_x()+bar.get_width()/2, v+25, f'{v:,.2f}', ha='center',fontsize=10)
ax[1].set(title='相同规则33：减亏来源',ylabel='改善金额（元）',ylim=(0,1600))
ax[1].grid(axis='y',alpha=.2)
fig.suptitle('Huber尚未证明择时改善｜内部2025，12股、15个交易日',fontsize=15)
fig.text(.5,.03,'36个候选分别运行，不能相加为一个组合；同33由53组586手降至1组4手，费用减少解释约82.77%的减亏。\n两个同33账户仍均亏损；本图仅读内部，不能当正式年度验证收益。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.13,1,.93])
fig.savefig(out/'same_rule_comparison.png',dpi=170)
fig.savefig(out/'same_rule_comparison.svg')
plt.close(fig)
assert hashlib.sha256(source.read_bytes()).hexdigest() == sha
(out/'verification.json').write_text(json.dumps({'passed':True,'source':str(source),'source_sha256':sha,
    'candidate_count':36,'only_internal_validation_read':True,'no_refit_or_account_rerun':True,
    'same_rule_33_gross_change':gross,'same_rule_33_cost_reduction':cost,'same_rule_33_net_change':net,
    'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}},ensure_ascii=False,indent=2))
print(out)
