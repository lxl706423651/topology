"""仅整理已经通过身份核对的内部诊断，不重算策略、不读取外层或重新选参。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def sha(path):
    """绑定当前读入文件的完整字节。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    """只生成新解释文件，已有正式诊断及其校验清单均不覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir', required=True)
    args = parser.parse_args()
    folder = Path(args.diagnostic_dir).resolve()
    inputs = {}

    def read_json(path):
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())

    def read_csv(name):
        path = folder / name
        inputs[str(path)] = sha(path)
        return pd.read_csv(path, float_precision='round_trip')

    verification = read_json(folder / 'verification.json')
    assert verification['passed'] and verification['complete'] and verification['no_outer_read']
    assert verification['no_refit_calibration_or_parameter_change']
    assert all(sha(folder / name) == value for name, value in verification['output_sha256'].items())
    quality = read_csv('prediction_quality.csv')
    margin = read_csv('margin_candidate_summary.csv')
    premium = read_csv('argmax_premium.csv')
    candidates = read_csv('candidate_overview.csv')
    common = quality.loc[quality.grouping.eq('all') & quality.scope.str.startswith('argmax_current')
                         & quality.domain.eq('baseline_opening_old_inventory')].copy()
    common['margin_bp'] = common.scope.str.replace('argmax_current_margin', '', regex=False).astype(int)
    common = common.sort_values(['fold', 'profile', 'margin_bp'])
    summary = []
    lines = ['# 内部预测与成本覆盖机制复核', '',
        '本文件仅解释原有四折内部结果。未读取外层账户或预测，未重训、拟合校准、反转方向或改变候选。', '',
        '前三个留出折的内部评价年均为 2026，最后一个内部评价年为 2025；不同模型和股池的这些结果不能相加视作四个独立获利年度。', '',
        '当前机会按原信号及原始最大预测净边际选择，并要求无 T 基准当天开盘已经持有旧股。该标识不保证实际现金、旧股剩余额度、整手、涨跌停或分钟容量均允许成交。', '',
        '|留出折|内部年|模型|原门槛 bp|当前机会|预测净差 bp|真实方向毛差 bp|固定端点标称净差 bp|固定端点覆盖成本比例|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in common.itertuples():
        lines.append(f'|{r.fold}|{r.inner_year}|{r.profile}|{r.margin_bp}|{r.rows}|{r.mean_predicted_net_bp_common:.2f}|'
                     f'{r.mean_realized_reference_gross_bp:.2f}|{r.mean_fixed_nominal_net_bp:.2f}|{r.fixed_cost_covered_rate:.2%}|')
    lines += ['', '以上机会每行等权。预测均值与实际固定端点均值使用同一端点可用子集，原始缺失行仍计数保存。'
              '标称净差使用原始准确分钟收盘价及当日双腿税费、滑点公式；约 31.20 bp 是 2023-08-28 后平价来回成本，'
              '此前约 36.20 bp。它不是实际可成交保证，更不是账户盈利。', '',
              '|留出折|模型|门槛 bp|6 个账户完整组中位数|净增量中位数 元|实际费用中位数 元|滑点中位数 元|执行合格/正净额账户数|',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for r in margin.sort_values(['fold', 'profile', 'margin_bp']).itertuples():
        assert r.overlapping_accounts == 6
        lines.append(f'|{r.fold}|{r.profile}|{r.margin_bp:g}|{r.completed_groups_median:g}|{r.marked_increment_median:.2f}|'
                     f'{r.median_actual_fee:.2f}|{r.median_actual_slippage:.2f}|{r.execution_eligible_accounts}/{r.positive_marked_net_accounts}|')
    lines += ['', '每档六个账户是两种旧股比例乘三种退出模式。上表是六个相互重叠实验的分布中位数，'
              '不是将它们合成一个组合；提高门槛造成少交易、少支出，与改善方向判断是不同事实。'
              '净额为原账户含残仓估值的增量，负值原样保留。', '',
              '|留出折|内部年|原选中规则|主动账户正/负/零|执行合格主动账户|原选中净额 元|',
              '|---|---|---|---|---:|---:|']
    for result in verification['results']:
        year = result['fold']
        part = candidates.loc[candidates.fold.eq(year)]
        active = part.loc[part.enabled]
        assert len(part) == 37 and len(active) == 36 and part.selected.sum() == 1
        selected = part.loc[part.selected].iloc[0]
        rule = result['selected_rule']
        item = dict(fold=year, inner_year=result['inner_year'],
                    selected_candidate_index=int(selected.candidate_index), selected_enabled=bool(selected.enabled),
                    selected_incremental_net_pnl=float(selected.incremental_net_pnl),
                    positive_active=int(active.incremental_net_pnl.gt(0).sum()),
                    negative_active=int(active.incremental_net_pnl.lt(0).sum()),
                    zero_active=int(active.incremental_net_pnl.eq(0).sum()),
                    execution_eligible_active=int(active.execution_eligible.sum()))
        summary.append(item)
        label = (f"{rule['candidate_index']} {rule['profile']}/{rule['exit_mode']}" if rule['enabled'] else 'hold（无已选 T）')
        lines.append(f"|{year}|{result['inner_year']}|{label}|{item['positive_active']}/{item['negative_active']}/{item['zero_active']}|"
                     f"{item['execution_eligible_active']}|{selected.incremental_net_pnl:.2f}|")
    lines += ['', 'hold 的零收益表示原协议没有启用主动账户，不代表做 T 成功。存在正净额但发生库存或恢复失败的候选时，'
              '也不能忽略失败将其视作满足每日库存要求。原被选账户的完整组、所有首腿、未完成组、固定端点标称反事实及'
              '真实退出净额分别见 selected_pairs.csv / selected_pair_quality.csv。', '',
              '|留出折|模型|同键完整备选时刻数|选择最大预测带来的预测溢价 bp|同键实际固定端点溢价 bp|',
              '|---|---|---:|---:|---:|']
    for r in premium.loc[premium.domain.eq('baseline_opening_old_inventory')].itertuples():
        lines.append(f'|{r.fold}|{r.profile}|{r.same_key_all_alternatives_available_rows}|'
                     f'{r.forecast_selection_premium_bp:.2f}|{r.realized_selection_premium_bp:.2f}|')
    lines += ['', '该溢价比较同一股票、同一信号时刻的当前合法期限均值与原最大预测选择；只为量化择大后的乐观程度，'
              '不改变实际机会或挑选新期限。至少一个备选端点缺失的时刻单列，不拿可用子集冒充全部成交。', '',
              'CSV 的 independent_days / independent_weeks 字段仅计不同日期与周数，不证明统计独立。'
              '多个股票的同刻收益有共同项，多个期限也重叠；未据这些行数计算新的显著性、校准模型或可信区间。', '',
              '来源隔离和模型原生权重的完整验收由另行独立审计负责；本诊断核对保存元数据、预测身份和原决策，'
              '仅在内部数据上解释误差。既有 OOF 选择器的历史跨年依赖仍存在，不能将反复研究的数据声称为全新测试。']
    assert all(sha(path) == digest for path, digest in inputs.items())
    output = folder / 'mechanism_explanation'
    output.mkdir(exist_ok=False)
    (output / 'mechanism_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copy2(__file__, output / 'explain.py')
    (output / 'verification.json').write_text(json.dumps(dict(passed=True, no_outer_read=True,
        no_refit_or_rule_change=True, script_sha256=sha(__file__), input_sha256=inputs,
        fold_summary=summary, output_sha256={p.name: sha(p) for p in output.iterdir()}), ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
