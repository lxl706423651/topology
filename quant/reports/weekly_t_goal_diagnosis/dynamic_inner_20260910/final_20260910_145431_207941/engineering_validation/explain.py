"""从已完成只读诊断小表生成中文解释；不再读取账户、模型或任何外层。"""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('final_directory',type=Path)
    args=parser.parse_args();root=args.final_directory.resolve()
    files=['candidate_summary.csv','paired_exit_contrasts.csv','pair_match_categories.csv','unmatched_entry_causes.csv','trigger_realization.csv','diagnosis.json']
    h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inputs={str(root/f):h(root/f) for f in files};d=json.loads((root/'diagnosis.json').read_text())
    assert d['passed'] and d['no_outer_account_read']
    tables={f:pd.read_csv(root/f,float_precision='round_trip') for f in files if f.endswith('.csv')}
    c=tables['candidate_summary.csv'];p=tables['paired_exit_contrasts.csv'];t=tables['trigger_realization.csv'];u=tables['unmatched_entry_causes.csv']
    lines=['# 完整内部成对诊断解释','',
        '这是同一入场规则、同一完整分钟报价覆盖下的退出对照。不同候选账户相互重叠，以下跨候选统计不是一只组合的累计收益；没有读取外层，也没有重新选择规则。','',
        '| 留出折 | 动态退出 | 54个整账户改善/恶化 | 整账户净额差中位数（元） | 共同且双方完成首腿差中位数（元） | 其他组贡献差中位数（元） | 执行合格 fixed/动态 |',
        '|---|---|---|---:|---:|---:|---|']
    for (year,mode),part in p.groupby(['fold','dynamic_mode']):
        own=c.loc[c.fold.eq(year)];fix=own.loc[own.exit_mode.eq('fixed')&own.candidate_index.gt(0)]
        alt=own.loc[own.exit_mode.eq(mode)];delta=part.incremental_marked_net_delta
        lines.append(f'| {year} | {mode} | {int(delta.gt(1e-8).sum())}/{int(delta.lt(-1e-8).sum())} | {delta.median():.2f} | {part.common_complete_net_delta.median():.2f} | {part.remaining_complete_net_delta.median():.2f} | {int(fix.execution_eligible.sum())}/{int(alt.execution_eligible.sum())} |')
    lines+=['',
        '“其他组贡献”包含仅一侧出现、同信号成交股数不同、或只有一侧完整的组。每个对照的完整组净差可以准确分解，但各分量的中位数不能相加去还原整体中位数；未完整残仓的估值差另外保留在账户净差中。共同且双方完成的组是条件子集，不是对所有首腿的无偏效果估计。','',
        '| 动态退出 | 54对照×4折中共同完整组改善/恶化 | 整账户改善/恶化 | 动态侧独有首腿中位数 | fixed侧独有首腿中位数 |',
        '|---|---|---|---:|---:|']
    for mode,part in p.groupby('dynamic_mode'):
        a=part.common_complete_net_delta;b=part.incremental_marked_net_delta
        lines.append(f'| {mode} | {int(a.gt(1e-8).sum())}/{int(a.lt(-1e-8).sum())} | {int(b.gt(1e-8).sum())}/{int(b.lt(-1e-8).sum())} | {part.only_dynamic_first_legs.median():.1f} | {part.only_fixed_first_legs.median():.1f} |')
    lines+=['','改变退出时刻会改变后续现金、回购准备金、待恢复额度和旧股消耗，因而首腿集合也会改变。下表是另一真实账户在同一时刻的已保存记录；数量变化单列，未用事后盈利反推理由。计数跨候选重复，仅用于定位机制。','',
        '| 出现首腿的一侧 | 另一侧同刻记录 | 首腿数（重叠实验） |',
        '|---|---|---:|']
    for (side,cause),part in u.groupby(['present_mode','other_account_observed_reason']):
        lines.append(f'| {side} | {cause} | {int(part.first_legs.sum())} |')
    lines+=['','| 模式/退出原因 | 可估值且完整组（重叠实验） | 同组预计净额均值（元） | 同组实际净额均值（元） | 正预计但实际亏损比例 |',
        '|---|---:|---:|---:|---:|']
    for (mode,trigger),part in t.groupby(['exit_mode','trigger']):
        n=part.complete_with_finite_estimate.sum()
        if not n:continue
        est=part.estimated_net_sum_same_complete_groups.sum()/n;actual=part.actual_net_sum_same_complete_groups.sum()/n
        positive_losses=part.realized_loss_despite_positive_estimate.sum()
        # 止盈触发组的预计值全为正；其他触发分母没有所有正预计组计数，故只报绝对数量。
        loss_label=f'{positive_losses/n:.2%}' if trigger=='profit_buffer' else f'{int(positive_losses)}组（不估比例）'
        lines.append(f'| {mode}/{trigger} | {int(n)} | {est:.2f} | {actual:.2f} | {loss_label} |')
    active=c.loc[c.candidate_index.gt(0)]
    lines+=['',f'全部{len(active)}个主动内部账户中，净额正/负/零为{int(active.incremental_marked_net.gt(0).sum())}/{int(active.incremental_marked_net.lt(0).sum())}/{int(active.incremental_marked_net.eq(0).sum())}。hold的0仅是未交易对照，不能视为实现5%成本降低目标。',
        '',
        '止盈信号使用当刻收盘价估算净额，实际成交最早在下一交易分钟，并可能分片或受阻。预计与实际差额不是被忽略的手续费：预计已包含既有两腿费用，实际使用各片真实费和真实成交价。未完整组、缺当前估值组不被补成成功交易。','',
        'v13 fixed也使用新增完整分钟恢复覆盖。本诊断不把相对v12的恢复覆盖差额归为动态择时收益，不使用已知外层表现来改变方向、阈值或规则。2026内部数据只到2026-04-30；跨年留一训练的旧周度选股属于历史复测协议。']
    target=root/'interpretation.md';assert not target.exists();target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert all(h(p)==v for p,v in inputs.items())
    (root/'interpretation_sources.json').write_text(json.dumps({'passed':True,'inputs_sha256':inputs,
        'script_sha256':h(__file__),'interpretation_sha256':h(target),'no_additional_account_reads':True},ensure_ascii=False,indent=2))
    print(target)

if __name__=='__main__':main()
