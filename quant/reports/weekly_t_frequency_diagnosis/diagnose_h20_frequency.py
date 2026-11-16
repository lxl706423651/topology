"""只读v8四折内部h20预测，统计成本门槛和相对分位对潜在交易频率的影响。

从quant环境运行，输出本目录小型CSV/JSON/Markdown。不读取外层预测/收益，
不调用模型训练或账户引擎，不改变任何生产规则；相对分位仅是待验证的新规则。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
sys.path.insert(0,str(PROJECT))
from quant_backtest.weekly_t_short_horizon import predicted_pair_edges

SOURCE = PROJECT/'reports/weekly_t_cross_validation/run_20260908_235818_404243'
PROFILES = ('l2_depth4','l2_depth6','huber_depth4')
MARGINS_BP = (0,5,10,20)
QUANTILES = (.70,.85,.95)


def file_sha(path: Path) -> str:
    """分块摘要全部实际读取文件，保留诊断可重现的来源身份。"""
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def count_signal_units(frame: pd.DataFrame) -> dict:
    """区分信号行与股票日；多个信号、模型及候选不能视为独立交易。"""
    return {'rows':len(frame),'symbol_days':int(frame[['symbol','date']].drop_duplicates().shape[0]),
        'days':int(frame.date.nunique()),'weeks':int(frame.week.nunique()),'symbols':int(frame.symbol.nunique()),
        'distinct_timestamps':int(frame.datetime.nunique())}


def finite_mean(values: pd.Series):
    """空集合返回null，不把无信号伪装成零值均值。"""
    return float(values.mean()) if len(values) else None


def main() -> None:
    """只消费内部预测和指标，固定比较四种绝对门槛及三种相对分位。"""
    cfg_path=SOURCE/'resolved_config.json'
    cfg=json.loads(cfg_path.read_text())
    source_hashes=SOURCE/'source_hashes.json'
    fee_source=PROJECT/'quant_backtest/weekly_t_short_horizon.py'
    assert file_sha(fee_source)==json.loads(source_hashes.read_text())['quant_backtest/weekly_t_short_horizon.py']
    inputs=[cfg_path,source_hashes,fee_source]
    profiles,absolute,relative,original=[],[],[],[]
    for year in (2023,2024,2025,2026):
        fold=SOURCE/f'year_{year}'
        split_path,metrics_path=fold/'split.json',fold/'validation_rule_metrics.csv'
        inputs.extend([split_path,metrics_path])
        split=json.loads(split_path.read_text())
        metrics=pd.read_csv(metrics_path,float_precision='round_trip')
        inner=int(split['inner_year'])
        assert metrics.test_year.eq(year).all() and metrics.inner_year.eq(inner).all()
        selected=metrics.loc[metrics.horizon_minutes.eq(20)]
        for row in selected.to_dict('records'):
            original.append({key:row[key] for key in ('test_year','inner_year','candidate_index','profile','net_margin',
                't_pair_count','started_pair_count','fragment_count','active_days','active_weeks','active_symbols',
                'inventory_mismatch_days','recovery_failure_days','base_plan_mismatch_count')})
        for profile in PROFILES:
            model=fold/f'inner_t_h20_{profile}'
            metadata_path=model/'model_metadata.json'
            prediction_path=fold/f'validation_predictions_h20_{profile}.csv'
            inputs.extend([metadata_path,prediction_path])
            metadata=json.loads(metadata_path.read_text())
            assert sorted(metadata['train_years'])==sorted(split['core_years'])
            assert sorted(metadata['excluded_years'])==sorted([year,inner])
            assert metadata['profile']==profile and metadata['resolved_config']['t_label_horizon_minutes']==20
            frame=pd.read_csv(prediction_path,parse_dates=['datetime'],float_precision='round_trip')
            assert frame.datetime.dt.year.eq(inner).all() and not frame.duplicated(['datetime','symbol']).any()
            assert np.isfinite(frame.prediction).all() and frame.prediction.gt(-1).all()
            assert frame.prediction.equals(frame.residual_prediction) and frame.market_prediction.eq(0).all()
            frame['date']=frame.datetime.dt.normalize()
            frame['week']=frame.datetime.dt.to_period('W-SUN').dt.start_time
            buy,sell=predicted_pair_edges(frame,cfg)
            frame['priority']=np.maximum(buy,sell)
            frame['direction']=np.where(buy.ge(sell),'buy_first','sell_first')
            magnitude=frame.prediction.abs()*10000
            units=count_signal_units(frame)
            base={'test_year':year,'inner_year':inner,'horizon_minutes':20,'profile':profile}
            profiles.append({**base,**units,'core_years':split['core_years'],'training_rows':metadata['training_rows'],
                'training_clocks':metadata['training_clock_count'],'features':len(metadata['feature_names']),
                'mean_absolute_prediction_bp':float(magnitude.mean()),'median_absolute_prediction_bp':float(magnitude.median()),
                'p70_absolute_prediction_bp':float(magnitude.quantile(.70)),
                'p85_absolute_prediction_bp':float(magnitude.quantile(.85)),
                'p95_absolute_prediction_bp':float(magnitude.quantile(.95)),
                'p99_absolute_prediction_bp':float(magnitude.quantile(.99)),
                'max_absolute_prediction_bp':float(magnitude.max()),'positive_prediction_fraction':float(frame.prediction.gt(0).mean()),
                'mean_best_predicted_net_bp':float(frame.priority.mean()*10000)})
            for margin_bp in MARGINS_BP:
                passing=frame.loc[frame.priority.ge(margin_bp/10000)]
                passed=count_signal_units(passing)
                absolute.append({**base,'net_margin_bp':margin_bp,**passed,
                    'total_rows':units['rows'],'total_symbol_days':units['symbol_days'],
                    'row_fraction':passed['rows']/units['rows'],'symbol_day_fraction':passed['symbol_days']/units['symbol_days'],
                    'mean_predicted_net_bp':finite_mean(passing.priority*10000),
                    'buy_first_rows':int(passing.direction.eq('buy_first').sum()),
                    'sell_first_rows':int(passing.direction.eq('sell_first').sum())})
            # 相对阈值只用当前同刻已有预测；不是用完整未来验证年分布拟合固定界限。
            for q in QUANTILES:
                threshold=frame.groupby('datetime',sort=False).priority.transform(lambda values:values.quantile(q))
                passing=frame.loc[frame.priority.ge(threshold)]
                passed=count_signal_units(passing)
                relative.append({**base,'same_timestamp_quantile':q,**passed,'total_rows':units['rows'],
                    'total_symbol_days':units['symbol_days'],'row_fraction':passed['rows']/units['rows'],
                    'symbol_day_fraction':passed['symbol_days']/units['symbol_days'],
                    'mean_predicted_net_bp':finite_mean(passing.priority*10000),
                    'median_predicted_net_bp':float(passing.priority.median()*10000),
                    'predicted_net_positive_rows':int(passing.priority.gt(0).sum()),
                    'predicted_net_nonpositive_rows':int(passing.priority.le(0).sum()),
                    'predicted_net_nonpositive_fraction':float(passing.priority.le(0).mean()),
                    'mean_current_quantile_threshold_bp':float(threshold.mean()*10000)})
    artifacts={'profile_amplitudes.csv':profiles,'absolute_net_margin_counts.csv':absolute,
               'relative_quantile_counts.csv':relative,'original_inner_h20_pair_counts.csv':original}
    for name,rows in artifacts.items():
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    zero=pd.DataFrame({'datetime':pd.to_datetime(['2026-01-05']),'prediction':[0.]})
    buy,sell=predicted_pair_edges(zero,cfg)
    cost=float(-buy.iloc[0]*10000)
    a=(1-cfg['slippage_rate'])*(1-cfg['commission_rate']-cfg['transfer_rate']-cfg['stamp_rate_from_20230828'])
    b=(1+cfg['slippage_rate'])*(1+cfg['commission_rate']+cfg['transfer_rate'])
    thresholds=[{'net_margin_bp':margin,'buy_first_raw_prediction_bp':((b+margin/10000)/a-1)*10000,
        'sell_first_raw_prediction_abs_bp':(b-a+margin/10000)/b*10000} for margin in MARGINS_BP]
    output={'source_report':str(SOURCE),'scope':'仅四折核心模型元数据和原内部预测/配对指标；没有读取任何新外层业绩或标签。',
        'profiles':profiles,'absolute_net_margin_counts':absolute,'relative_quantile_counts':relative,
        'original_inner_pair_counts':original,'zero_move_two_leg_cost_bp':cost,'absolute_raw_thresholds':thresholds,
        'relative_rule_definition':'每模型每datetime，对现有已选周中股票的max(买先净边际,卖先净边际)求q分位，当前预测>=阈值入选；不使用后续时刻或全验证年分布。',
        'limited_proposal':{'horizons':[20],'profiles':list(PROFILES),'absolute_net_margins_bp':list(MARGINS_BP),
            'relative_current_clock_quantiles':list(QUANTILES),'total_candidate_configurations':21,
            'unchanged':'实际费率/滑点/T+1/库存恢复/基准计划/资金/容量/整手、T比例0.25和每股每日最多首个有效请求保持原值；不放大预测或删除实际费用。',
            'relative_rule_requires_explicit_change':'相对排名候选可在模型预测净边际为负时请求T，属于放弃原预测覆盖费用的假设，绝不能称已经覆盖成本。',
            'selection_scope':'全部21种仅在各折原内部年份运行真实账户后比较；先满足原执行要求，不使用新外层结果选择。',
            'frequency_objective_notice':'沿用净PnL最大排序可能继续选稀疏规则；若目标明确提高频率，需事前明确频率约束/排序政策，不能在看到外层数量后反复改变。',
            'no_account_evaluation_performed':True},
        'limitations':['表内行数/股票日仅费用或排名门槛的潜在上界，未加入support、方向限价、持仓/可卖库存、现金和容量；不等于订单或完整T配对。',
            '股票日假设每股每日取首个有效请求，仍未证明当时实际有可T整手持仓；同刻分位全名单与实际持仓子集的排序会不同。',
            '分位统计在每个当前datetime计算，不把全年预测分布作为当时已知的门槛；跨模型、门槛和不同信号不能相加当独立样本。',
            '20分钟残差条件均值预测的幅度小于逐笔真实噪声是可能现象；小幅预测不能通过缩放就变成真实优势。',
            '不以预测之外的未来标签提高信号频率；不能因希望更频繁而掩盖净亏损或恢复失败。'],
        'input_files':[{'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path)} for path in sorted(set(inputs))],
        'script_sha256':file_sha(Path(__file__))}
    (OUT/'h20_frequency_diagnosis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    render(output)
    print(json.dumps({'models':len(profiles),'margin_rows':len(absolute),'quantile_rows':len(relative),
        'p99_abs_prediction_bp_range':[min(x['p99_absolute_prediction_bp'] for x in profiles),max(x['p99_absolute_prediction_bp'] for x in profiles)],
        'q_nonpositive_predicted_net_fraction_range':[min(x['predicted_net_nonpositive_fraction'] for x in relative),max(x['predicted_net_nonpositive_fraction'] for x in relative)],
        'input_files':len(output['input_files'])},ensure_ascii=False,indent=2))


def render(output: dict) -> None:
    """用固定口径比较相同内部信号池，避免把潜在行数称为成交收益。"""
    lines=['# h20 内部交易频率诊断','','只看v8核心训练元数据和内部预测/已有配对数；没有读取v9或任何新外层收益、没有运行新账户。', '',
      f"原双腿费用及滑点合计约 **{output['zero_move_two_leg_cost_bp']:.2f}bp**；20bp扣费后边际还要求约51.2bp的原始方向预测。把净门槛降至0也仍须覆盖约31.2bp成本。",'',
      '## 幅度与费用门槛','','以下均为潜在信号上界，尚未通过support/现有库存/现金/容量/整手/方向限价等完整条件。股票日是同一股票同一天去重，不是已成交T配对。','','| 留出折/内部年/模型 | abs预测P95/P99 bp | 全股票日 | 净0bp 行/股票日 | 净5bp 行/股票日 | 净10bp 行/股票日 | 净20bp 行/股票日 |','|---|---:|---:|---:|---:|---:|---:|']
    for profile in output['profiles']:
        subsets=[x for x in output['absolute_net_margin_counts'] if x['test_year']==profile['test_year'] and x['profile']==profile['profile']]
        lines.append(f"| {profile['test_year']}/{profile['inner_year']}/{profile['profile']} | {profile['p95_absolute_prediction_bp']:.2f}/{profile['p99_absolute_prediction_bp']:.2f} | {profile['symbol_days']} | "+' | '.join(f"{x['rows']}/{x['symbol_days']}" for x in subsets)+' |')
    lines += ['', '## 相对分位的含义','','相对q70/q85/q95在每个当前时刻对预测净边际排名，约取前30%/15%/5%。它能产生更多请求，但排名不能抵消成本；多数新增信号的模型预测仍不足以支付费用。它是明确的新策略假设，不能宣传为“预测净利已覆盖成本”。', '',
      '| 留出折/模型 | q70股票日/预测净≤0占比 | q85股票日/预测净≤0占比 | q95股票日/预测净≤0占比 |','|---|---:|---:|---:|']
    for profile in output['profiles']:
        subsets=[x for x in output['relative_quantile_counts'] if x['test_year']==profile['test_year'] and x['profile']==profile['profile']]
        lines.append(f"| {profile['test_year']}/{profile['profile']} | "+' | '.join(f"{x['symbol_days']}/{x['predicted_net_nonpositive_fraction']:.1%}" for x in subsets)+' |')
    lines += ['', '## 有限改进方案','','仅固定20分钟、三个现有profile，内层比较12种绝对净门槛候选（0/5/10/20bp）和9种当前时刻相对分位候选（q70/q85/q95），共21种，不增加模型或方向反转。', '',
      '相对分位候选必须显式标记“允许模型预测未覆盖费用”，实际账户仍完整扣原费用，保留T+1、库存恢复、现金、整手、容量、限价、25% T比例和每股每日最多首个有效请求。先在内层真实回测，真实失败和亏损原样报告；不使用新外层选择门槛。', '',
      '**只扩大候选而仍按内层净利润最大排序，不能保证交易次数提高。** 若本轮目标是更高频率，需事前明确频率的选参地位；不能事后按外层数量反复修改。当前诊断只提交频率上界和有限候选方案，没有决定频率目标或修改生产政策。', '',
      '## 边界','']+['- '+x for x in output['limitations']]
    lines += ['', '每模型完整幅度分布、原内部已成交配对数、绝对门槛/相对分位统计及全部输入SHA见同目录CSV与JSON。']
    (OUT/'h20_frequency_diagnosis.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
