"""绘制已冻结v18内部候选；只读诊断CSV，不训练或执行账户。"""
from pathlib import Path
from datetime import datetime
import hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
root=Path('/data/lxl/seed-emulator/topology/quant')
source=root/'reports/weekly_t_goal_diagnosis/inventory_coverage_inner_20260910/final_20260910_184413_471230'
out=source/'figures'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
out.mkdir(parents=True,exist_ok=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=['candidate_overview.csv','selected_complete_notional_comparison.csv','verification.json']
identities={name:sha(source/name) for name in files}
df=pd.read_csv(source/files[0],float_precision='round_trip')
comparison=pd.read_csv(source/files[1],float_precision='round_trip')
assert len(df)==148 and df.groupby('fold').size().eq(37).all()
active=df.loc[df.candidate_index.ne(0)].copy()
active['cost_rate_pct']=100*active.realized_complete_pair_pnl/active.all_reference_acquisition_cost
np.testing.assert_allclose(active.cost_rate_pct,100*active.holding_cost_reduction_rate,rtol=1e-10,atol=1e-10)
assert active.groupby('fold').execution_eligible.sum().tolist()==[0,4,0,0]
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11})
fig,axes=plt.subplots(2,2,figsize=(13,9.7),sharey=True)
for ax,(fold,g) in zip(axes.flat,active.groupby('fold')):
 for ok,color,marker,label in [(False,'#cf623a','x','执行不合格'),(True,'#197e86','o','执行合格')]:
  s=g.loc[g.execution_eligible.eq(ok)]
  ax.scatter(s.completed_t_pair_lots,s.cost_rate_pct,c=color,marker=marker,s=42,alpha=.85,label=label)
 ax.axhline(0,color='#8a939b',linewidth=1)
 ax.axhline(5,color='#347442',linestyle='--',linewidth=1)
 ax.grid(alpha=.16)
 inner=int(g.inner_year.iloc[0]); title=f'留出年标签 {fold} · 内部评价 {inner}'
 ax.set_title(title,loc='left',pad=12,fontweight='bold')
 ax.set_xlabel('完成配对手数（每手100股）')
 ax.text(.02,.93,f'36个主动候选；执行合格 {int(g.execution_eligible.sum())} 个',transform=ax.transAxes,fontsize=10,va='top')
 selected=g.loc[g.selected]
 if len(selected):
  s=selected.iloc[0]
  ax.scatter([s.completed_t_pair_lots],[s.cost_rate_pct],s=140,facecolors='none',edgecolors='#172e46',linewidths=1.5)
  ax.annotate('所选15：3597手，−0.4617%',(s.completed_t_pair_lots,s.cost_rate_pct),xytext=(.08,.35),textcoords='axes fraction',fontsize=10,color='#172e46',arrowprops={'arrowstyle':'-','color':'#172e46'})
 else:
  ax.text(.02,.05,'最终hold：无执行合格主动候选',transform=ax.transAxes,fontsize=10,color='#773d2b')
 ax.set_ylim(min(-6,float(active.cost_rate_pct.min())-1),6.5)
for ax in axes[:,0]: ax.set_ylabel('完整配对净额 / 全部原底仓成本（%）')
handles,labels=axes[0,1].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.97,.963),ncol=2,frameon=False)
fig.suptitle('十五分钟特征覆盖研究：增加信息后，仍未同时满足收益与每日恢复要求',x=.07,y=.99,ha='left',fontsize=17,fontweight='bold')
fig.text(.07,.938,'v18内部结果；虚线为原底仓成本5%目标。每个点是一个候选方案，不能相加为组合收益。',fontsize=10,color='#52606d')
fig.text(.07,.025,'前三折内部数据为2026年1—4月，末折为2025全年；正式外层未执行。\n不合格候选含未恢复风险，图中完整配对利润不代表全部残仓已消除。',fontsize=10,color='#52606d')
fig.subplots_adjust(top=.87,bottom=.13,left=.08,right=.97,hspace=.35,wspace=.16)
fig.savefig(out/'内部手数与成本收益.png',dpi=160,facecolor='white')
fig.savefig(out/'内部手数与成本收益.svg',facecolor='white')
plt.close(fig)
r=comparison.loc[comparison.fold.eq(2024)].iloc[0]
for prefix in ['source','coverage']:
 np.testing.assert_allclose(r[f'{prefix}_complete_gross_bp']-r[f'{prefix}_complete_cost_bp'],r[f'{prefix}_complete_net_bp'],atol=1e-10)
fig,ax=plt.subplots(figsize=(10,5.8))
x=np.arange(3); old=[r.source_complete_gross_bp,-r.source_complete_cost_bp,r.source_complete_net_bp]; new=[r.coverage_complete_gross_bp,-r.coverage_complete_cost_bp,r.coverage_complete_net_bp]
for dx,values,color,label in [(-.18,old,'#899aa9','v15：覆盖0.80'),(.18,new,'#237f88','v18：覆盖0.70')]:
 bars=ax.bar(x+dx,values,width=.34,color=color,label=label)
 for bar,v in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,v+(1 if v>=0 else -1),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=12)
ax.axhline(0,color='#88929b',lw=1);ax.set_xticks(x,['参考毛价差','费用及滑点','实际净差']);ax.set_ylabel('每万元完整首腿参考金额的盈亏（元）')
ax.set_ylim(-38,24);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper right')
ax.set_title('同规则15有所改善，但价差仍不足以覆盖交易成本',loc='left',fontsize=16,pad=20,fontweight='bold')
fig.text(.11,.11,'留出年标签2024，内部评价2026年1—4月；v15为381组，v18为378组。',fontsize=10,color='#52606d')
fig.text(.11,.06,'各自按实际完整组首腿参考金额归一化，成交集合不同；此分母不是5%目标的原底仓成本。',fontsize=10,color='#52606d')
fig.subplots_adjust(top=.84,bottom=.22,left=.11,right=.97)
fig.savefig(out/'同规则单位名义收益.png',dpi=160,facecolor='white')
fig.savefig(out/'同规则单位名义收益.svg',facecolor='white')
plt.close(fig)
shutil.copy2(__file__,out/'plot.py')
assert identities=={name:sha(source/name) for name in identities}
active[['fold','inner_year','candidate_index','execution_eligible','selected','completed_t_pair_lots','cost_rate_pct']].to_csv(out/'plot_data.csv',index=False,encoding='utf-8-sig')
proof={'passed':True,'read_inputs_unchanged':True,'input_sha256':identities,'candidate_count':144,'formal_outer_read':False,'fit_or_account_executed':False,'source_dir':str(source),'visual_review_status':'pending','artifacts_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
(out/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(out)
print('verification_sha256',sha(out/'verification.json'))
