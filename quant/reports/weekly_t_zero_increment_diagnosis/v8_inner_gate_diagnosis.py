"""只读核查v8四折内部候选停用原因、预测与费率单位；不读外层或修改策略。

在quant项目环境运行，输出同目录JSON/Markdown及所读输入SHA。仅复核已有
内部指标和冻结规则，调用纯门控/费用函数，不重新拟合、不重新运行账户。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
sys.path.insert(0, str(PROJECT))
from quant_backtest.weekly_t_short_horizon import candidate_passes_support, predicted_pair_edges

REPORT = PROJECT / 'reports/weekly_t_cross_validation/run_20260908_235818_404243'
EXECUTION = ('inventory_mismatch_days', 'recovery_failure_days', 'pending_pairs', 'base_plan_mismatch_count', 't1_violation_count')
SUPPORT = ('active_days', 'active_weeks', 'active_symbols', 'initiated_pairs', 'max_day_pair_fraction',
           'max_week_pair_fraction', 'max_day_abs_pnl_fraction', 'max_week_abs_pnl_fraction',
           'weekly_increment_ci_low_rmb', 'weekly_increment_ci_high_rmb')
CONCENTRATION = ('max_day_pair_fraction', 'max_week_pair_fraction', 'max_day_abs_pnl_fraction', 'max_week_abs_pnl_fraction')
TRANSLATE = {'inventory_mismatch_days':'每日库存不一致', 'recovery_failure_days':'日内恢复失败',
 'pending_pairs':'期末未恢复T', 'base_plan_mismatch_count':'底仓计划不一致', 't1_violation_count':'T+1违规',
 'nonpositive_net_pnl':'净增量不为正', 'nonpositive_weekly_bootstrap_lower':'周bootstrap下界不为正',
 'insufficient_complete_pairs':'完整T配对不足100', 'insufficient_active_days':'活跃日不足20',
 'insufficient_active_weeks':'活跃周不足8', 'insufficient_active_symbols':'活跃股票不足20',
 'max_day_pair_fraction':'单日配对过度集中', 'max_week_pair_fraction':'单周配对过度集中',
 'max_day_abs_pnl_fraction':'单日绝对盈亏过度集中', 'max_week_abs_pnl_fraction':'单周绝对盈亏过度集中'}


def sha(path: Path) -> str:
    """记录所读文件身份，不触及原文件内容。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reasons_from_row(row: dict, cfg: dict) -> list[str]:
    """独立按数值比较重建门槛原因，不调用生产门控判断。"""
    requirement = cfg['candidate_support_requirements']
    failures = [key for key in EXECUTION if float(row[key]) != 0]
    values = {key: row[key] for key in ('incremental_net_pnl', 't_pair_count', *SUPPORT)}
    failures.extend('nonfinite_' + key for key, value in values.items() if not np.isfinite(value))
    if float(row['incremental_net_pnl']) <= 0:
        failures.append('nonpositive_net_pnl')
    if float(row['weekly_increment_ci_low_rmb']) <= 0:
        failures.append('nonpositive_weekly_bootstrap_lower')
    if int(row['t_pair_count']) < requirement['min_complete_pairs']:
        failures.append('insufficient_complete_pairs')
    for key in ('active_days','active_weeks','active_symbols'):
        if int(row[key]) < requirement['min_' + key]:
            failures.append('insufficient_' + key)
    for key in CONCENTRATION:
        if float(row[key]) < 0 or float(row[key]) > requirement[key]:
            failures.append(key)
    return failures


def summarize(frame: pd.DataFrame) -> dict:
    """汇总候选配置数量；不同门槛可能有相同决策，绝不累加其收益为组合收益。"""
    positive = frame.loc[frame.incremental_net_pnl.gt(0)]
    def reason_counts(part):
        return dict(Counter(reason for value in part.rejection_reasons for reason in str(value).split('|') if reason))
    def groups(part):
        counts = Counter()
        for value in part.rejection_reasons:
            reasons = value.split('|')
            labels = set()
            for reason in reasons:
                if reason in EXECUTION:
                    labels.add('交易合法性与库存')
                elif reason.startswith('insufficient_'):
                    labels.add('最少样本门槛')
                elif reason in CONCENTRATION:
                    labels.add('集中度门槛')
                elif reason == 'nonpositive_weekly_bootstrap_lower':
                    labels.add('周CI门槛')
                elif reason == 'nonpositive_net_pnl':
                    labels.add('盈利门槛')
                else:
                    labels.add(reason)
            counts[' + '.join(sorted(labels))] += 1
        return dict(counts)
    return {'candidates':len(frame), 'positive':len(positive), 'zero':int(frame.incremental_net_pnl.eq(0).sum()),
        'negative':int(frame.incremental_net_pnl.lt(0).sum()), 'strict_eligible':int(frame.strict_eligible.sum()),
        'unique_decisions':int(frame[['test_year','decision_sha256']].drop_duplicates().shape[0]),
        'nonzero_completed_pair_candidates':int(frame.t_pair_count.gt(0).sum()),
        'complete_pair_range':[int(frame.t_pair_count.min()),int(frame.t_pair_count.max())],
        'fragment_range':[int(frame.fragment_count.min()),int(frame.fragment_count.max())],
        'incremental_net_pnl_range_rmb':[float(frame.incremental_net_pnl.min()),float(frame.incremental_net_pnl.max())],
        'positive_complete_pair_range':[int(positive.t_pair_count.min()),int(positive.t_pair_count.max())] if len(positive) else None,
        'all_rejection_reason_counts':reason_counts(frame), 'positive_rejection_reason_counts':reason_counts(positive),
        'all_rejection_group_combinations':groups(frame), 'positive_rejection_group_combinations':groups(positive),
        'positive_without_execution_failure':int((positive[list(EXECUTION)].eq(0).all(axis=1)).sum()),
        'positive_exact_reason_combinations':dict(Counter(positive.rejection_reasons)),
        'all_exact_reason_combinations':dict(Counter(frame.rejection_reasons))}


def main() -> None:
    """读取108内部候选及36预测，独立复核并生成中文解释。"""
    inputs = [REPORT/'resolved_config.json', REPORT/'source_hashes.json']
    cfg = json.loads(inputs[0].read_text())
    sources = json.loads(inputs[1].read_text())
    for name, expected in sources.items():
        current, frozen = PROJECT/name, REPORT/'source_snapshot'/Path(name).name
        assert sha(current) == expected and sha(frozen) == expected
        inputs.extend([current,frozen])
    frames, forecasts, yearly = [], [], {}
    checked = []
    for year in (2023,2024,2025,2026):
        fold = REPORT/f'year_{year}'
        mp, rule_path = fold/'validation_rule_metrics.csv', fold/'selected_rule.json'
        inputs.extend([mp,rule_path])
        frame = pd.read_csv(mp,float_precision='round_trip')
        assert len(frame) == 27 and frame.test_year.eq(year).all()
        assert frame.enabled.eq(True).all()
        assert frame.t_pair_count.equals(frame.completed_pair_count)
        assert (frame.t_pair_count <= frame.started_pair_count).all()
        assert (frame.fragment_count >= frame.t_pair_count).all()
        rule = json.loads(rule_path.read_text())
        assert type(rule['enabled']) is bool and rule['enabled'] is False
        for row in frame.to_dict('records'):
            stored = row['rejection_reasons'].split('|')
            independent = reasons_from_row(row,cfg)
            assert independent == stored
            evaluation = {'incremental_net_pnl':row['incremental_net_pnl'], 'metrics':row,
                          'support':{key:row[key] for key in SUPPORT}}
            passed, production = candidate_passes_support(evaluation,cfg)
            assert type(passed) is bool and production == independent
            assert passed == row['strict_eligible'] == (len(independent)==0)
            checked.append({'test_year':year,'candidate_index':row['candidate_index'],
                            'independent_reasons_match':True,'production_bool_match':True})
        frames.append(frame)
        yearly[str(year)] = {'inner_year':int(frame.inner_year.iloc[0]),**summarize(frame),'selected_enabled':rule['enabled']}
        paths = sorted(fold.glob('validation_predictions_h*.csv'))
        assert len(paths) == 9
        for path in paths:
            inputs.append(path)
            pred = pd.read_csv(path,usecols=['datetime','prediction','residual_prediction','market_prediction'],float_precision='round_trip')
            assert len(pred) and np.isfinite(pred.prediction).all()
            assert pred.prediction.equals(pred.residual_prediction) and pred.market_prediction.eq(0).all()
            assert pred.prediction.ne(0).any()
            forecasts.append({'test_year':year,'path':str(path),'rows':len(pred),
                'prediction_nonzero_rows':int(pred.prediction.ne(0).sum()),
                'prediction_min_bp':float(pred.prediction.min()*10000),
                'prediction_max_bp':float(pred.prediction.max()*10000),
                'prediction_abs_p99_bp':float(pred.prediction.abs().quantile(.99)*10000),
                'market_prediction_zero_is_intentional':True})
    all_rows = pd.concat(frames,ignore_index=True)
    fields = ['test_year','inner_year','candidate_index','horizon_minutes','profile','net_margin','incremental_net_pnl',
        't_pair_count','fragment_count','active_days','active_weeks','active_symbols','weekly_increment_ci_low_rmb',
        *EXECUTION,'rejection_reasons','decision_sha256','same_decisions_as_candidate']
    # 一个满足全部条件的纯字典应返回True，排除函数硬编码/列表真假反转；不运行账户。
    synthetic = {'incremental_net_pnl':1., 'metrics':{'t_pair_count':100,**{key:0 for key in EXECUTION}},
        'support':{'active_days':20,'active_weeks':8,'active_symbols':20,'initiated_pairs':100,
          'max_day_pair_fraction':.1,'max_week_pair_fraction':.2,'max_day_abs_pnl_fraction':.1,
          'max_week_abs_pnl_fraction':.2,'weekly_increment_ci_low_rmb':.01,'weekly_increment_ci_high_rmb':2.}}
    assert candidate_passes_support(synthetic,cfg) == (True,[])
    fees = []
    for date,stamp in [('2023-08-25',cfg['stamp_rate_before_20230828']),('2023-08-28',cfg['stamp_rate_from_20230828']),('2026-01-05',cfg['stamp_rate_from_20230828'])]:
        slip = cfg['slippage_rate']; fee = cfg['commission_rate']+cfg['transfer_rate']
        a=(1-slip)*(1-fee-stamp); b=(1+slip)*(1+fee)
        zero=pd.DataFrame({'datetime':pd.to_datetime([date]),'prediction':[0.]})
        buy,sell=predicted_pair_edges(zero,cfg)
        assert np.isclose(buy.iloc[0],a-b,atol=1e-15) and np.isclose(sell.iloc[0],a-b,atol=1e-15)
        margins=[]
        for margin in cfg['short_t_net_margins']:
            up=(b+margin)/a-1; down=(b-a+margin)/b
            samples=pd.DataFrame({'datetime':pd.to_datetime([date,date]),'prediction':[up,-down]})
            actual_buy,actual_sell=predicted_pair_edges(samples,cfg)
            assert np.isclose(actual_buy.iloc[0],margin,rtol=0,atol=1e-15)
            assert np.isclose(actual_sell.iloc[1],margin,rtol=0,atol=1e-15)
            margins.append({'net_margin_decimal':margin,'net_margin_bp':margin*10000,
                'buy_first_required_positive_raw_prediction_bp':up*10000,
                'sell_first_required_negative_raw_prediction_abs_bp':down*10000})
        fees.append({'date':date,'zero_move_nominal_net_bp':float((a-b)*10000),'thresholds':margins})
    output={'scope':'v8四折已有内部候选、核心预测和源码；无外层账户或预测读取、无训练/参数变更、无账户重跑。',
      'report_dir':str(REPORT),'yearly':yearly,'overall':summarize(all_rows),
      'positive_candidates':json.loads(all_rows.loc[all_rows.incremental_net_pnl.gt(0),fields].to_json(orient='records')),
      'all_candidates':json.loads(all_rows[fields].to_json(orient='records')),
      'gate_rows_checked':len(checked),'gate_checks':checked,'gate_bool_synthetic_all_pass':True,
      'pair_counts_match_completed_counts':True,'forecasts':forecasts,'fee_unit_checks':fees,
      'candidate_support_requirements':cfg['candidate_support_requirements'],
      'reason_chinese':TRANSLATE,
      'inspection_findings':[
       '108内部候选有真实T配对和非零净增量，收益不是统一被计算成零；36个模型预测均存在非零值。',
       '4个selected_rule均是JSON布尔false；runner无合格候选时主动关闭T，策略enabled=false会把所有action保持0。',
       '108行存储拒绝原因、生产函数返回布尔与独立数值判断逐项相符；全部合格的纯字典返回True，不存在恒false/布尔反转。',
       '0.001表示0.1%=10bp净边际；当时费用滑点约31.2bp，需约41.2bp原始方向预测才触发10bp门槛，单位和代数公式一致。',
       '100完整pair、20日、8周、20股、周CI下界大于0及集中度约束是研究者额外设定的启用标准，不是T+1制度要求；最终流程是在测试是否值得启用T。',
       'T+1、每日库存恢复、底仓计划一致是经济可行性条件，不能与研究者设定的最少样本/CI阈值混为一谈。',
       'active_days/weeks/symbols按已发起pair统计，完整配对单独计数；片段没有当作多个完整pair。',
       '周CI使用每周累计权益差的变化重采样求总增量分位，不把收益率误当元；本文只检查公式与已存支持值，未重跑内部账户以独立重建全部CI。',
       '部分单pair候选CI下界约-7e-10元属于浮点残差；即使视为0，严格要求大于0及样本门槛仍会拒绝，不能解释四折总停用。',
       '重复门槛可能生成同一决策，108配置和27正配置不是108/27个独立实验；不得把候选收益相加。'],
      'code_references':{'cost_formula':'quant_backtest/weekly_t_short_horizon.py:78','signal_enable_switch':'quant_backtest/weekly_t_short_horizon.py:106',
        'decision_filter':'quant_backtest/weekly_t_short_horizon.py:127','support_statistics':'quant_backtest/weekly_t_short_horizon.py:139',
        'candidate_gate':'quant_backtest/weekly_t_short_horizon.py:169','off_fallback':'scripts/run_weekly_t_residual.py:217',
        'account_increment':'quant_backtest/weekly_t_validation.py:122','complete_pair_metric':'quant_backtest/weekly_t_engine.py:768',
        'support_config':'config/weekly_t_cross_validation_v8.yaml:114','margin_config':'config/weekly_t_cross_validation_v8.yaml:97'},
      'input_files':[{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(set(inputs))],
      'script_sha256':sha(Path(__file__))}
    (OUT/'v8_inner_gate_diagnosis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    render(output)
    print(json.dumps({'overall':output['overall'],'models':len(forecasts),'prediction_rows':sum(x['rows'] for x in forecasts),
        'all_nonzero_rows':sum(x['prediction_nonzero_rows'] for x in forecasts),'inputs':len(output['input_files'])},ensure_ascii=False,indent=2))


def render(output: dict) -> None:
    """中文说明主动停用的流程差异，包含全部27正候选和拒绝理由。"""
    total=output['overall']
    lines=['# v8 做T增量全零的内部候选排查','','**四折最终为零的直接原因是规则主动关闭T。** 内部108个候选已有27正、17零、64负，36个模型预测均非全零。没有发现费用单位、支持门控布尔或把所有内部收益写成0的逻辑错误。','',
      '**研究流程加了额外的“是否值得启用”筛选。** 100完整pair、20活跃日、8周、20只股票、周bootstrap下界必须大于0及集中度约束，不是T+1交易制度要求。因此最终结果不能解释成“主动做T四年的收益恰好都是0”；它表示四折均没有候选通过所设全部门槛。','','| 留出折 | 内部验证年 | 候选正/零/负 | 完整配对范围 | 原严格合格 |','|---|---:|---:|---:|---:|']
    for year,row in output['yearly'].items():
        lines.append(f"| {year} | {row['inner_year']} | {row['positive']}/{row['zero']}/{row['negative']} | {row['complete_pair_range'][0]}–{row['complete_pair_range'][1]} | {row['strict_eligible']} |")
    lines += ['',f"108配置中有{total['unique_decisions']}套不同决策；相同交易可能对应多个门槛，不能把候选收益相加。",'',
      '## 27个正候选为什么仍被拒','','| 拒绝条件 | 全108候选 | 其中27正候选 |','|---|---:|---:|']
    for reason,count in total['all_rejection_reason_counts'].items():
        lines.append(f"| {TRANSLATE.get(reason,reason)} | {count} | {total['positive_rejection_reason_counts'].get(reason,0)} |")
    lines += ['', '拒绝原因可同时发生。27正候选中12个没有库存/恢复/基准计划/T+1错误，但仍被统计或样本门槛拒绝。', '',
      '明确例子：2025留出折候选4，内部净增量 **+220.21元**、周CI下界 **+34.37元**，没有库存/恢复/底仓计划错误；却因只有15完整pair、12活跃日、活跃股票不足20而不合格。这是额外研究标准生效，不是交易制度或收益计算错误。','',
      '反例：2026留出折有候选完成295组并净增量+824.61元，仍有日内恢复/库存与基准计划问题，周CI下界为负。不能把所有拒绝都归因于样本门槛。','',
      '## 费用与开关','','0.001 是 0.1%=10bp的**扣费后**预期净边际，不是0.001bp。2023-08-28后，原股票价格不变时，按两腿滑点、佣金、过户费和印花税计算约亏31.1975bp。触发门槛需要的原始方向预测为：','','| 净边际门槛 | 买先需上涨预测 | 卖先需下跌预测绝对值 |','|---|---:|---:|']
    for row in output['fee_unit_checks'][-1]['thresholds']:
        lines.append(f"| {row['net_margin_bp']:.0f}bp | {row['buy_first_required_positive_raw_prediction_bp']:.4f}bp | {row['sell_first_required_negative_raw_prediction_abs_bp']:.4f}bp |")
    lines += ['', '所有108行拒绝原因与独立比较、生产返回的bool均完全一致；一个满足全部条件的纯测试字典返回True。原函数没有恒false或反转。此处只调用纯函数，没有运行新账户或调整条件。','','源码位置：','',
      '- [候选硬门槛](../../quant_backtest/weekly_t_short_horizon.py#L169)：盈利、CI、样本和库存等全部通过才为True。',
      '- [无合格候选时关闭T](../../scripts/run_weekly_t_residual.py#L217)：写入 `enabled: false`。',
      '- [信号开关](../../quant_backtest/weekly_t_short_horizon.py#L106)：先把所有 `t_action` 置0，只有enabled才计算非零动作。',
      '- [净增量计算](../../quant_backtest/weekly_t_validation.py#L122)：最终实际账户权益减同基准权益。',
      '- [配置门槛](../../config/weekly_t_cross_validation_v8.yaml#L114)；[费用公式](../../quant_backtest/weekly_t_short_horizon.py#L78)。','',
      '## 解释和核查边界','','- 本文未读取任何外层账户或预测，也未重新训练、运行账户、改变阈值或修改生产源。',
      '- 每日股数恢复和T+1属于用户策略的执行要求，应与人为设定的CI/样本门槛分开解释。',
      '- CI公式按每周权益差变化重采样，单位为元；本次核对公式与已存数值，没有从完整账户逐周曲线重建全部CI。',
      '- 活跃日周股票由发起pair统计；完整pair单独计数，恢复片段没有冒充完整pair。',
      '- 少量单pair CI约−7×10⁻¹⁰元是浮点残差，视为0仍不满足“必须大于0”，且其他样本条件未过。',
      '- 全部27个正候选、各拒绝组合、36份预测统计、来源SHA及108行独立门控结果见JSON。历史多次复测不等同全新样本。']
    (OUT/'v8_inner_gate_diagnosis.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
