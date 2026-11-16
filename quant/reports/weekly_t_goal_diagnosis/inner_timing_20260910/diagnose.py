"""只读分析 v11 核心训练及内部验证门控与固定端点，不读取外层账户或拟合模型。

在 quant 环境从仓库根目录运行；输出仅写本脚本目录。标签缓存先按内部预测键
取交集，未来收益仅用于内部误差诊断，不用于生成信号、改参数或重新选参。
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from quant_backtest.weekly_t_short_horizon import feature_support_mask, planned_recovery_times
from quant_backtest.weekly_t_features import select_recoverable_signal_sides

OUT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/weekly_t_cross_validation/run_20260909_182945_388303'
INPUTS = {}
TABLES = {}


def sha(path):
    """计算实际读取文件的完整内容身份。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def record(path):
    """登记输入身份，防止把诊断变成未记录的数据改写。"""
    path = Path(path)
    if str(path) not in INPUTS:
        INPUTS[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read_json(path):
    """读取只读 JSON 并登记身份。"""
    return json.loads(record(path).read_text())


def read_csv(path, **kwargs):
    """按已归档预测的 round-trip 数值合同读取 CSV。"""
    return pd.read_csv(record(path), float_precision='round_trip', **kwargs)


def row(table, **values):
    """收集小规模聚合结果，不导出全部内部预测行。"""
    TABLES.setdefault(table, []).append(values)


def edges(times, returns, cfg):
    """独立按旧现金公式换算双腿名义净价差；不含成交失败或资金容量模拟。"""
    ratio = 1 + np.asarray(returns, dtype=float)
    slip = cfg['slippage_rate']
    fee = cfg['commission_rate'] + cfg['transfer_rate']
    stamp = np.where(times < pd.Timestamp('2023-08-28'),
                     cfg['stamp_rate_before_20230828'], cfg['stamp_rate_from_20230828'])
    buy = ratio*(1-slip)*(1-fee-stamp) - (1+slip)*(1+fee)
    sell = (1-slip)*(1-fee-stamp) - ratio*(1+slip)*(1+fee)
    return buy, sell


def dimensions(frame):
    """同时报告日、周、股票和同刻数量，避免把关联行当成独立样本。"""
    dt = frame.datetime
    return dict(rows=len(frame), days=int(dt.dt.normalize().nunique()),
                weeks=int(dt.dt.to_period('W-SUN').nunique()),
                symbols=int(frame.symbol.nunique()), clocks=int(dt.nunique()))


def interval_rows(frame, bins, variable, table, extra):
    """固定分箱诊断预测与真实端点；缺标签行保留计数，绝不改变信号资格。"""
    for lower, upper in zip(bins[:-1], bins[1:]):
        part = frame.loc[frame[variable].ge(lower) & frame[variable].lt(upper)]
        finite = part.loc[np.isfinite(part.actual_net_bp)]
        row(table, **extra, lower=lower, upper=upper, **dimensions(part),
            finite_labels=len(finite), missing_labels=len(part)-len(finite),
            mean_pred_raw_abs_bp=float(part.pred_abs_bp.mean()),
            mean_pred_net_bp=float(part.pred_net_bp.mean()),
            mean_actual_directional_raw_bp=float(finite.actual_directional_raw_bp.mean()),
            mean_actual_net_bp=float(finite.actual_net_bp.mean()),
            mean_reported_probability=float(part.probability.mean()),
            endpoint_positive_fraction=float(finite.actual_net_bp.gt(0).mean()) if len(finite) else np.nan)


def main():
    """对固定四折全部内部模型作聚合，不触碰外层预测或重新执行账户。"""
    start = time.monotonic()
    cfg = read_json(REPORT/'resolved_config.json')
    protocol = {
        'scope': 'v11四折core训练与inner验证；不读取外层预测/账户；不拟合、不回测、不选新规则',
        'abs_prediction_bins_bp': [0,10,20,40,60,100,'inf'],
        'probability_bins': [0,.4,.5,.6,.7,.8,.9,1.0000001],
        'endpoint_direction': '每个原模型在当前信号按预测双腿净额较高方向，未来端点只作诊断',
        'mask_scope': '所有当前行，以及原support/合法时钟/日风险/振幅后的当前行',
        'signal_funnel': '原选中候选按原门控重算；同刻不同方向/期限先计机会再去重，不能当独立交易',
        'new_strategy_selection': 'none',
    }
    (OUT/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    frames, rules = {}, {}
    for year in range(2023,2027):
        folder = REPORT/f'year_{year}'
        split = read_json(folder/'split.json')
        rules[year] = read_json(folder/'selected_rule.json')
        metrics = read_csv(folder/'validation_rule_metrics.csv')
        for _, entry in metrics.iterrows():
            keep = ['candidate_index','profile','regime','min_net_edge','min_profit_probability',
                    't_pair_count','incremental_net_pnl','submitted_t_requests','active_stock_days',
                    'eligible_inventory_stock_days','eligible_signal_rows','timing_eligible','selected']
            row('inner_candidates', fold=year, inner_year=split['inner_year'], **{k:entry[k] for k in keep})
        funnels = read_json(folder/'validation_signal_funnels.json')
        chosen = next(f for f in funnels if f['candidate_index']==rules[year]['candidate_index'])
        for item in chosen['horizon_side_funnel']:
            row('selected_funnels', fold=year, candidate_index=chosen['candidate_index'], **item)
        row('selected_summary', fold=year, inner_year=split['inner_year'],
            candidate_index=chosen['candidate_index'], profile=rules[year]['profile'],
            probability_opportunities=sum(f['after_profit_probability'] for f in chosen['horizon_side_funnel']),
            regime_opportunities=sum(f['after_regime'] for f in chosen['horizon_side_funnel']),
            requests=chosen['selected_requests'], complete_pairs=rules[year]['inner_complete_pairs'],
            inner_net_pnl=rules[year]['inner_incremental_net_pnl'])
        for horizon in (20,40,60):
            domain = read_json(folder/f'training_domain_h{horizon}.json')
            model = folder/f'inner_t_h{horizon}_l2_depth4'
            meta = read_json(model/'model_metadata.json')
            target = read_csv(model/'training_targets.csv', parse_dates=['datetime'])
            assert set(target.datetime.dt.year) <= set(split['core_years'])
            counts = target.groupby('symbol').size()
            means = target.groupby('datetime').raw_target_return.transform('mean')
            raw = target.raw_target_return
            common_share = float(((means-raw.mean())**2).sum()/((raw-raw.mean())**2).sum())
            row('core_summary', fold=year, horizon=horizon, core_years=split['core_years'],
                input_domain_rows=domain['input_rows'], domain_kept_rows=domain['kept_rows'],
                **dimensions(target), fit_rows=int(target.fitted.sum()),
                median_rows_per_symbol=float(counts.median()), min_rows_per_symbol=int(counts.min()),
                common_clock_variance_fraction=common_share,
                raw_mean_bp=float(raw.mean()*1e4), raw_std_bp=float(raw.std()*1e4),
                classes=meta['training_class_counts'])
            for half, part in target.groupby(target.datetime.dt.year.astype(str)+'H'+((target.datetime.dt.month-1)//6+1).astype(str)):
                row('core_time_blocks', fold=year, horizon=horizon, half=half, **dimensions(part))
            for clock, part in target.groupby(target.datetime.dt.strftime('%H:%M')):
                row('core_clock_targets', fold=year, horizon=horizon, clock=clock,
                    **dimensions(part), mean_raw_bp=float(part.raw_target_return.mean()*1e4),
                    buy_positive_fraction=float(part.buy_net_return.gt(0).mean()),
                    sell_positive_fraction=float(part.sell_net_return.gt(0).mean()))
            for profile in ('l2_depth4','l2_depth6'):
                model = folder/f'inner_t_h{horizon}_{profile}'
                read_json(model/'model_metadata.json')
                frame = read_csv(folder/f'validation_h{horizon}_{profile}_predictions.csv', parse_dates=['datetime'])
                assert frame.datetime.dt.year.eq(split['inner_year']).all()
                weekly = {'f_weekly_pred_return_rank','f_weekly_pred_amplitude_rank','f_weekly_score','f_weekly_selection_rank'}
                for col in frame:
                    if col.startswith('f_') and col not in weekly:
                        frame[col] = frame[col].astype('float32')
                support = read_json(model/'support.json')
                planned = planned_recovery_times(frame.datetime, horizon)
                mask = feature_support_mask(frame,support)
                clock_mask = planned.notna() & planned.dt.strftime('%H:%M').le('14:30')
                mask &= clock_mask & frame.f_day_return.abs().le(.03) & frame.f_known_amplitude.between(0,.06)
                frame['current_eligible'] = mask
                frames[year,horizon,profile] = frame
        print(f'核心和内部当前输入已读：fold={year}', flush=True)
    keys = pd.concat([f[['datetime','symbol']] for f in frames.values()]).drop_duplicates()
    manifest = read_json(REPORT/'horizon_manifest.json')
    labels_path = record(Path(manifest['labels_path']))
    labels = pd.read_pickle(labels_path)
    # 先限制内部精确时刻，再按内部股票键取交集；不分析其余缓存标签。
    labels = labels.loc[labels.datetime.isin(keys.datetime.unique())].merge(keys, on=['datetime','symbol'], how='inner', validate='many_to_one')
    assert not labels.duplicated(['datetime','symbol','horizon_minutes']).any()
    print(f'仅内部预测键匹配标签：{len(labels)}行', flush=True)
    row('label_scope', **dimensions(labels), horizons=sorted(labels.horizon_minutes.unique().tolist()),
        finite_labels=int(np.isfinite(labels.target_return).sum()))
    for (year,horizon,profile), frame in frames.items():
        frame = frame.merge(labels.loc[labels.horizon_minutes.eq(horizon), ['datetime','symbol','target_return']],
                            on=['datetime','symbol'], how='left', validate='one_to_one')
        buy,sell = edges(frame.datetime, frame.raw_prediction, cfg)
        abuy,asell = edges(frame.datetime, frame.target_return, cfg)
        side = np.where(buy>=sell,1,-1)
        frame['pred_abs_bp'] = frame.raw_prediction.abs()*1e4
        frame['pred_net_bp'] = np.maximum(buy,sell)*1e4
        frame['actual_net_bp'] = np.where(side==1,abuy,asell)*1e4
        frame['actual_directional_raw_bp'] = side*frame.target_return*1e4
        frame['probability'] = np.where(side==1,frame.prob_buy_profit,frame.prob_sell_profit)
        extra = dict(fold=year,horizon=horizon,profile=profile)
        for scope, part in [('all_current',frame),('current_risk_clock_support',frame.loc[frame.current_eligible])]:
            finite = part.loc[np.isfinite(part.target_return)]
            row('prediction_summary', **extra, scope=scope, **dimensions(part),
                finite_labels=len(finite), missing_labels=len(part)-len(finite),
                raw_q05_bp=float(part.raw_prediction.quantile(.05)*1e4),
                raw_q50_bp=float(part.raw_prediction.quantile(.5)*1e4),
                raw_q95_bp=float(part.raw_prediction.quantile(.95)*1e4),
                mean_pred_abs_bp=float(part.pred_abs_bp.mean()),
                mean_pred_net_bp=float(part.pred_net_bp.mean()),
                mean_actual_directional_raw_bp=float(finite.actual_directional_raw_bp.mean()),
                mean_actual_net_bp=float(finite.actual_net_bp.mean()))
            interval_rows(part,[0,10,20,40,60,100,np.inf],'pred_abs_bp','prediction_bins',{**extra,'scope':scope})
            interval_rows(part,[0,.4,.5,.6,.7,.8,.9,1.0000001],'probability','probability_bins',{**extra,'scope':scope})
            for margin in (0,.0005,.001):
                selected = part.loc[part.pred_net_bp.ge(margin*1e4)]
                finite_sel = selected.loc[np.isfinite(selected.target_return)]
                row('positive_predicted_edge', **extra,scope=scope,margin_bp=margin*1e4,
                    **dimensions(selected),finite_labels=len(finite_sel),
                    mean_predicted_net_bp=float(selected.pred_net_bp.mean()),
                    mean_actual_net_bp=float(finite_sel.actual_net_bp.mean()),
                    endpoint_positive_fraction=float(finite_sel.actual_net_bp.gt(0).mean()) if len(finite_sel) else np.nan)
        rule = rules[year]
        if profile == rule['profile']:
            for action,edge,prob in [(1,buy,frame.prob_buy_profit),(-1,sell,frame.prob_sell_profit)]:
                eligible = frame.current_eligible & np.isfinite(edge) & (edge>=rule['min_net_edge'])
                eligible &= prob.between(rule['min_profit_probability'],1.)
                # 以下固定順序只是分解原AND条件，不选择新的门槛。
                stages = [
                    ('before_shape',pd.Series(True,index=frame.index)),
                    ('finite_efficiency',np.isfinite(frame.f_timing_trend_efficiency)&frame.f_timing_trend_efficiency.between(0,1)),
                    ('recent_direction',np.isfinite(frame.f_return_5m)&(action*frame.f_return_5m>0)),
                    ('vwap_opposite',np.isfinite(frame.f_vwap_deviation)&(action*frame.f_vwap_deviation<0)),
                    ('efficiency_le_065',frame.f_timing_trend_efficiency.le(.65)),
                    ('vwap_z_ge_075',np.isfinite(frame.f_timing_vwap_z)&frame.f_timing_vwap_z.abs().ge(.75)),
                    ('exit_guard',select_recoverable_signal_sides(frame,pd.Series(action,index=frame.index),cfg))]
                for stage, condition in stages:
                    eligible &= condition
                    row('selected_shape_components',**extra, action=action,stage=stage, **dimensions(frame.loc[eligible]))
    for table, records in TABLES.items():
        pd.DataFrame(records).to_csv(OUT/f'{table}.csv',index=False)
    sources = {path: {'sha256':sha(ROOT/path)} for path in [
        'quant_backtest/weekly_t_timing.py','quant_backtest/weekly_t_timing_models.py',
        'quant_backtest/weekly_t_cohort.py','quant_backtest/weekly_t_short_horizon.py',
        'quant_backtest/weekly_t_features.py','quant_backtest/weekly_t_train_domain.py']}
    result = dict(protocol=protocol, inputs=INPUTS, production_sources=sources,
                  script_sha256=sha(__file__), elapsed_seconds=time.monotonic()-start,
                  tables={k:len(v) for k,v in TABLES.items()},
                  source_report=str(REPORT), selected_summary=TABLES['selected_summary'])
    (OUT/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'tables':result['tables'],
                      'selected_summary':result['selected_summary']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
