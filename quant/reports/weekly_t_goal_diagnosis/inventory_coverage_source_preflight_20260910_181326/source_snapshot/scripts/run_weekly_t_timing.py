"""从 quant 根目录运行成本感知的做 T 择时研究：核心拟合、内部选参、统一年度外评。

只新建 v11 输出，不覆盖历史实验。周选股、资金、T+1、整手、费用及恢复执行器
保持原合同；新模型学习原始收益与扣费盈利事件，在当前时点选择20/40/60分钟。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.weekly_t_engine import run_weekly_portfolio
from quant_backtest.weekly_t_frequency_validation import calculate_frequency_opportunities, run_frequency_candidates
from quant_backtest.weekly_t_horizon_data import validate_horizon_config
from quant_backtest.weekly_t_quote_store import calculate_file_sha256
from quant_backtest.weekly_t_short_horizon import attach_horizon_labels
from quant_backtest.weekly_t_timing import add_timing_features, timing_candidates, make_timing_decisions, select_timing_candidate
from quant_backtest.weekly_t_timing_data import frame_sha256, load_labels_only, restore_core_cohort, validate_timing_sources
from quant_backtest.weekly_t_timing_models import VERSION, fit_timing_model, predict_timing_model, validate_timing_config
from quant_backtest.weekly_t_timing_reporting import write_timing_report
from quant_backtest.weekly_t_train_domain import filter_core_t_training_domain
from quant_backtest.weekly_t_validation import validate_candidate_result
from scripts.run_weekly_t_active import PATH_FIELDS, load_frozen_market_data, record_input
from scripts.run_weekly_t_cross_validation import save_result, write_frame
from scripts.run_weekly_t_frequency import DECISION_COLUMNS, get_store, load_fold_definitions, prepare_execution_context, snapshot_frequency_sources


def load_configuration(argv: list[str] | None = None) -> dict:
    """读取预声明的小型候选协议；smoke仅缩小执行范围，核心拟合保持一致。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'config/weekly_t_cross_validation_v11.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-symbols', type=int)
    parser.add_argument('--smoke-weeks', type=int)
    parser.add_argument('--test-years')
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    for key in ('output_dir', 'max_symbols', 'smoke_weeks'):
        if getattr(args, key) is not None:
            value = getattr(args, key)
            cfg[key] = str(value) if isinstance(value, Path) else value
    if args.test_years:
        cfg['test_years'] = [int(y) for y in args.test_years.split(',')]
    for key in PATH_FIELDS:
        cfg[key] = str((PROJECT_ROOT/cfg[key]).resolve())
    cfg['dense_recovery_cache_paths'] = {str(y): str((PROJECT_ROOT/p).resolve())
                                         for y, p in cfg['dense_recovery_cache_paths'].items()}
    if cfg.get('research_version') != VERSION or cfg.get('candidate_activation_policy') != 'positive_net_execution_valid':
        raise ValueError('本入口要求 v11 原始收益+扣费盈利概率、内部净收益优先协议')
    if cfg['max_symbols'] < 0 or cfg['smoke_weeks'] < 0:
        raise ValueError('smoke范围不能为负')
    if not cfg['test_years'] or len(set(cfg['test_years'])) != len(cfg['test_years']):
        raise ValueError('外层年份必须非空且不重复')
    if cfg['t_model_profiles'] != ['l2_depth4', 'l2_depth6'] or cfg['horizons'] != [20, 40, 60]:
        raise ValueError('本次预声明模型与周期不允许静默扩展')
    _, _, minutes = validate_horizon_config(cfg)
    cfg['scheduled_recovery_times'] = [f'{v//60:02d}:{v%60:02d}' for v in minutes]
    validate_timing_config(cfg)
    if len(timing_candidates(cfg)) != 25:
        raise ValueError('要求24个active候选及一个显式等待基准')
    validate_timing_sources(cfg, PROJECT_ROOT)
    return cfg


def snapshot_sources(directory: Path) -> None:
    """冻结全部实际旧依赖和本次新增源码，后续逐文件检测运行期间的修改。"""
    snapshot_frequency_sources(directory)
    hashes = json.loads((directory/'source_hashes.json').read_text())
    for name in ('scripts/run_weekly_t_timing.py', 'quant_backtest/weekly_t_timing.py',
                 'quant_backtest/weekly_t_timing_models.py', 'quant_backtest/weekly_t_timing_data.py',
                 'quant_backtest/weekly_t_timing_reporting.py'):
        path = PROJECT_ROOT/name
        shutil.copyfile(path, directory/'source_snapshot'/path.name)
        hashes[name] = calculate_file_sha256(path)
    write_json(directory/'source_hashes.json', hashes)


def predict_and_audit(frame: pd.DataFrame, model_dir: Path, cfg: dict, out: Path, prefix: str) -> tuple:
    """核对两个原生模型预测及未来字段消除不变性，保存实际输入以便独立复核。"""
    bundle = joblib.load(model_dir/'model_bundle.joblib')
    predictions = predict_timing_model(bundle, frame, cfg)
    features = bundle['features']
    matrix = predictions[features].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    native_errors = {}
    for head, columns in (('return', ['prediction']),
                           ('profit_class', ['prob_neither', 'prob_buy_profit', 'prob_sell_profit'])):
        native = lightgbm.Booster(model_file=str(model_dir/f'{head}_model.txt'))
        if native.feature_name() != features:
            raise AssertionError('原生特征顺序改变')
        expected = predictions[columns].to_numpy()
        actual = np.asarray(native.predict(matrix)).reshape(expected.shape) if len(matrix) else expected.copy()
        error = float(np.max(np.abs(actual-expected))) if len(actual) else 0.
        if not np.isfinite(actual).all() or error > 1e-12:
            raise AssertionError('原生权重预测不一致')
        native_errors[head] = error
    future = [c for c in frame if c.startswith(('execution_', 'recovery_', 'target_', 'label_'))]
    scrubbed = frame.copy()
    scrubbed[future] = np.nan
    second = predict_timing_model(bundle, scrubbed, cfg)
    columns = ['datetime','symbol',*features,'prediction','raw_prediction','prob_neither',
               'prob_buy_profit','prob_sell_profit','horizon_minutes']
    pd.testing.assert_frame_equal(predictions[columns], second[columns], check_exact=True)
    # 归档模型未选中的当前门控字段，供独立复算支持域、反转/趋势和退出方向锁板。
    # 保存不会使字段进入模型；未来标签与实际执行/恢复价仍不写入此预测证据表。
    gates = [c for c in predictions if c.startswith('f_')]
    gates += [c for c in ('signal_price','signal_high','signal_low','previous_close',
                          'feature_source_start','intraday_feature_source_start',
                          'weekly_context_feature_source_start','source_model_training_years',
                          'source_model_sha256','model_profile') if c in predictions]
    archived_columns = list(dict.fromkeys([*columns,*gates]))
    prediction_path=out/f'{prefix}_predictions.csv'
    write_frame(predictions[archived_columns], prediction_path)
    write_json(out/f'{prefix}_causality_audit.json', {'passed': True, 'prediction_rows': len(predictions),
               'native_max_errors': native_errors, 'future_fields_removed': future,
               'current_inputs_and_predictions_unchanged': True,
               'archived_current_columns': archived_columns,
               'prediction_csv_sha256': calculate_file_sha256(prediction_path),
               'model_bundle_sha256': calculate_file_sha256(model_dir/'model_bundle.joblib')})
    return predictions, bundle['support']


def audited_decisions(predictions: dict, support: dict, rule: dict, cfg: dict) -> tuple:
    """对真实规则复算删除未来字段的请求，禁止用未来恢复成功或事后最优周期择时。"""
    decisions, audit = make_timing_decisions(predictions, support, rule, cfg)
    clean = {}
    for horizon, frame in predictions.items():
        clean[horizon] = frame.copy()
        fields = [c for c in frame if c.startswith(('execution_', 'recovery_', 'target_', 'label_'))]
        clean[horizon][fields] = np.nan
    repeated, _ = make_timing_decisions(clean, support, rule, cfg)
    pd.testing.assert_frame_equal(decisions, repeated, check_exact=True)
    audit['future_field_independence_passed'] = True
    return decisions, audit


def freeze_all_rules(folds: list, universe: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                     labels: pd.DataFrame, cfg: dict, directory: Path, inputs: dict, stores: dict) -> pd.DataFrame:
    """每折仅 core 拟合，内部完整逐候选真实执行；全部年度选定后才准许外层预测。"""
    tables = []
    for fold in folds:
        year = fold['test_year']
        out = directory/f'year_{year}'
        print(f'[{year}] 重建核心 {fold["core_years"]} OOF 训练队列', flush=True)
        cohort = restore_core_cohort(universe, intraday, fold, cfg, out, inputs, record_input)
        # 标签仅对本折核心当前键做join；其他年的收益从未传入 fit。
        core_keys = cohort[['datetime','symbol']]
        core_labels = labels.merge(core_keys, on=['datetime','symbol'], how='inner', validate='many_to_one')
        for horizon in cfg['horizons']:
            current, audit, metadata = filter_core_t_training_domain(cohort, universe, horizon, cfg)
            write_frame(audit, out/f'training_domain_h{horizon}.csv')
            write_json(out/f'training_domain_h{horizon}.json', metadata)
            frame = attach_horizon_labels(current, core_labels, horizon)
            for profile in cfg['t_model_profiles']:
                model_dir = out/f'inner_t_h{horizon}_{profile}'
                model_cfg = {**cfg, 't_label_horizon_minutes': horizon, 't_model_profile': profile}
                fit_timing_model(frame, fold['core_years'], model_cfg, model_dir,
                                 excluded_years=[year,fold['inner_year']])
            del current, audit, frame
        del cohort, core_labels
        gc.collect()
        selected, inner_daily, inner_intraday = prepare_execution_context(
            fold, 'validation', universe, intraday, daily, cfg, out, inputs)
        store = get_store(fold['inner_year'], cfg, inputs, stores)
        baseline = run_weekly_portfolio(selected, inner_daily, inner_intraday,
                                       pd.DataFrame(columns=DECISION_COLUMNS), cfg, recovery_bars=store)
        baseline_dir = out/'validation_baseline'
        baseline_dir.mkdir()
        save_result(baseline, baseline_dir, cfg)
        opportunities, eligible = calculate_frequency_opportunities(baseline, inner_intraday, cfg, 20)
        write_frame(eligible, out/'validation_eligible_inventory_stock_days.csv')
        write_json(out/'validation_frequency_opportunities.json', opportunities)
        rules = timing_candidates(cfg)
        write_json(out/'candidate_rules_frozen_before_inner_execution.json', rules)
        hold = rules[0]
        decisions, funnel = audited_decisions({}, {}, hold, cfg)
        tasks = [{'candidate_index': hold['candidate_index'], 'decisions': decisions,
                  'decision_sha256': frame_sha256(decisions)}]
        funnels = [{'candidate_index': hold['candidate_index'], **funnel}]
        for profile in cfg['t_model_profiles']:
            predictions, supports = {}, {}
            for horizon in cfg['horizons']:
                model_cfg = {**cfg,'t_label_horizon_minutes':horizon,'t_model_profile':profile}
                predictions[horizon], supports[horizon] = predict_and_audit(
                    inner_intraday, out/f'inner_t_h{horizon}_{profile}', model_cfg, out,
                    f'validation_h{horizon}_{profile}')
            for rule in (r for r in rules if r.get('profile') == profile):
                decisions, funnel = audited_decisions(predictions, supports, rule, cfg)
                tasks.append({'candidate_index':rule['candidate_index'], 'decisions':decisions,
                              'decision_sha256':frame_sha256(decisions)})
                funnels.append({'candidate_index':rule['candidate_index'], **funnel})
            del predictions, supports
        write_json(out/'validation_signal_funnels.json', funnels)
        print(f'[{year}] 内部 {fold["inner_year"]} 执行 {len(tasks)} 个候选账户', flush=True)
        evaluated = run_frequency_candidates(tasks, selected, inner_daily, inner_intraday, cfg,
                                             out/'validation_candidate_audits', recovery_store=store)
        task_map = {t['candidate_index']:t for t in tasks}
        rows=[]
        for rule in rules:
            item, task = evaluated[rule['candidate_index']], task_map[rule['candidate_index']]
            rows.append({'test_year':year,'inner_year':fold['inner_year'],**rule,**item['metrics'],**opportunities,
                         'incremental_net_pnl':item['incremental_net_pnl'], 'active_days':item['active_days'],
                         'active_stock_days':item['active_stock_days'], 'submitted_t_requests':len(task['decisions']),
                         'decision_sha256':task['decision_sha256']})
        write_frame(pd.DataFrame(rows), out/'validation_rule_metrics_unselected.csv')
        chosen, annotated = select_timing_candidate(pd.DataFrame(rows))
        rule = next(r for r in rules if r['candidate_index']==int(chosen['candidate_index']))
        model_hashes = {f'h{h}_{p}': calculate_file_sha256(out/f'inner_t_h{h}_{p}/model_bundle.joblib')
                        for h in cfg['horizons'] for p in cfg['t_model_profiles']}
        frozen = {**rule,'fold':fold,'inner_incremental_net_pnl':float(chosen['incremental_net_pnl']),
                  'inner_complete_pairs':int(chosen['t_pair_count']), 'model_bundle_sha256':model_hashes,
                  'selection_reason':chosen.get('selection_reason', 'positive_net_execution_valid' if rule.get('enabled') else 'no_profitable_execution_valid_inner_candidate'),
                  'activation_policy':cfg['candidate_activation_policy'], 'frozen_before_any_outer_test':True,
                  'probabilities_calibrated':False, 'old_t_weights_reused':False}
        write_json(out/'selected_rule.json', frozen)
        write_frame(annotated, out/'validation_rule_metrics.csv')
        tables.append(annotated)
        print(f'[{year}] 冻结候选 {rule["candidate_index"]}，内部 {chosen["t_pair_count"]} 对 / '
              f'净 T {chosen["incremental_net_pnl"]:.2f}元', flush=True)
        del tasks, task_map, inner_intraday, baseline
        gc.collect()
    combined = pd.concat(tables, ignore_index=True)
    write_frame(combined, directory/'validation_rule_metrics.csv')
    write_json(directory/'all_rules_frozen_before_outer_test.json', {
        'created_at':datetime.now().isoformat(),'test_years':cfg['test_years'],'selection_uses_inner_only':True,
        'selected_rule_sha256':{str(f['test_year']):calculate_file_sha256(directory/f'year_{f["test_year"]}/selected_rule.json') for f in folds}})
    return combined


def evaluate_folds(folds: list, universe: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame,
                   cfg: dict, directory: Path, inputs: dict, stores: dict) -> pd.DataFrame:
    """只评价已经冻结的规则；等待规则也保留完整无 T 底仓账户，不伪装为择时成功。"""
    marker = json.loads((directory/'all_rules_frozen_before_outer_test.json').read_text())
    if marker['test_years'] != [f['test_year'] for f in folds]:
        raise ValueError('必须等待全部年度规则冻结')
    # 在任何外层数据切片、预测或账户之前，先核所有折，不能部分先评后报错。
    for fold in folds:
        out=directory/f'year_{fold["test_year"]}'
        if calculate_file_sha256(out/'selected_rule.json') != marker['selected_rule_sha256'][str(fold['test_year'])]:
            raise ValueError('冻结后规则被修改')
        frozen=json.loads((out/'selected_rule.json').read_text())
        for key, expected in frozen['model_bundle_sha256'].items():
            if calculate_file_sha256(out/f'inner_t_{key}/model_bundle.joblib') != expected:
                raise ValueError('核心模型在全折冻结后改变')
    rows=[]
    for fold in folds:
        year=fold['test_year']
        out=directory/f'year_{year}'
        if calculate_file_sha256(out/'selected_rule.json') != marker['selected_rule_sha256'][str(year)]:
            raise ValueError('冻结后规则被修改')
        rule=json.loads((out/'selected_rule.json').read_text())
        selected, actual_daily, actual_intraday=prepare_execution_context(
            fold,'test',universe,intraday,daily,cfg,out,inputs)
        predictions,supports={},{}
        if rule.get('enabled', False):
            for horizon in cfg['horizons']:
                profile=rule['profile']
                model=out/f'inner_t_h{horizon}_{profile}'
                if calculate_file_sha256(model/'model_bundle.joblib') != rule['model_bundle_sha256'][f'h{horizon}_{profile}']:
                    raise ValueError('核心模型在冻结后改变')
                predictions[horizon],supports[horizon]=predict_and_audit(
                    actual_intraday,model,{**cfg,'t_label_horizon_minutes':horizon,'t_model_profile':profile},
                    out,f't_h{horizon}_{profile}')
        decisions,funnel=audited_decisions(predictions,supports,rule,cfg)
        write_frame(decisions,out/'t_decisions.csv')
        write_json(out/'test_signal_funnel_summary.json',funnel)
        store=get_store(year,cfg,inputs,stores)
        if not set(selected.symbol).issubset(store.symbols):
            raise ValueError('恢复行情未覆盖外层股票')
        write_json(out/'dense_recovery_reference.json',{'path':cfg['dense_recovery_cache_paths'][str(year)],
                   'fingerprint':store.fingerprint,'arrays_sha256_verified':True,'time_count':store.metadata['time_count']})
        baseline=run_weekly_portfolio(selected,actual_daily,actual_intraday,pd.DataFrame(columns=DECISION_COLUMNS),cfg,recovery_bars=store)
        opportunities,eligible=calculate_frequency_opportunities(baseline,actual_intraday,cfg,20)
        write_frame(eligible,out/'eligible_inventory_stock_days.csv')
        write_json(out/'frequency_opportunities.json',opportunities)
        result=run_weekly_portfolio(selected,actual_daily,actual_intraday,decisions,cfg,recovery_bars=store)
        write_json(out/'candidate_execution_audit.json',validate_candidate_result(result,cfg))
        metrics=save_result(result,out,cfg)
        curve=result['equity_curve']
        pd.testing.assert_series_equal(curve.baseline_equity,baseline['equity_curve'].equity,check_exact=True,check_names=False)
        if not cfg['max_symbols'] and not cfg['smoke_weeks']:
            origin=Path(cfg['source_training_report_dir'])/f'year_{year}/test_equity_curve.csv'
            record_input(origin,inputs)
            previous=pd.read_csv(origin,parse_dates=['datetime'],float_precision='round_trip')
            if not curve.datetime.reset_index(drop=True).equals(previous.datetime.reset_index(drop=True)):
                raise AssertionError('原底仓日期改变')
            error=float(np.max(abs(curve.baseline_equity.to_numpy()-previous.baseline_equity.to_numpy())))
            if error>1e-8:raise AssertionError('原底仓权益改变')
            write_json(out/'frozen_baseline_audit.json',{'same_dates':True,'max_equity_error_rmb':error})
        first=result['trade_events'].loc[lambda x:x.event_type.eq('t')&x.fragment_id.eq(0)]
        active=first.assign(date=pd.to_datetime(first.datetime).dt.normalize())
        rows.append({'test_year':year,'inner_year':fold['inner_year'],**metrics,**opportunities,
                     't_enabled':bool(rule.get('enabled')),'enabled':bool(rule.get('enabled')),
                     'submitted_t_requests':len(decisions),'request_count':len(decisions),
                     'prediction_rows':sum(len(p) for p in predictions.values()),
                     'active_days':int(active.date.nunique()),'active_stock_days':len(active.drop_duplicates(['date','symbol'])),
                     'selected_candidate_index':rule['candidate_index'],'signal_mode':rule.get('regime','hold')})
        print(f'[{year}] 外层 {len(decisions)} 请求 / {metrics["t_pair_count"]} 对 / '
              f'净 T {metrics["t_added_equity"]:.2f}元，股数未恢复 {metrics["inventory_mismatch_days"]} 日',flush=True)
        del predictions,actual_intraday,baseline,result
        gc.collect()
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """保存协议后才拟合；全折内部冻结后才外评；检查源身份并输出完整研究结果。"""
    cfg=load_configuration(argv)
    directory=Path(cfg['output_dir'])/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    directory.mkdir(parents=True)
    snapshot_sources(directory)
    write_json(directory/'resolved_config.json',cfg)
    write_json(directory/'research_protocol.json',{
        'research_version':VERSION,'previous_outer_results_seen':True,'selection_uses_inner_only':True,
        'all_rules_frozen_before_outer_test':True,'outer_model_refit':False,
        'core_models':'2 profiles x 3 horizons x raw return regression + unweighted 3-class cost-profit classification',
        'candidate_rules':timing_candidates(cfg),'selection':'positive inner net profit and >0 pairs, five execution failures zero; profit descending then index',
        'no_qualifying_candidate':'explicit hold, no evidence that timing is profitable; no forced loss-making trades',
        'horizon_choice':'highest currently predicted absolute net edge across legal h20/40/60 and directions; no future optimal endpoint',
        'probability_limitation':'uncalibrated fixed-endpoint cost-profit class probability, not fill/recovery probability',
        'execution_engine_and_costs_unchanged':True,'is_smoke':bool(cfg['max_symbols'] or cfg['smoke_weeks']),
        'smoke_scope':'same full core training, only smaller execution stock/week selection',
        'frequency_denominator':'no-T old shares and legal h20 clock; h20 contains the union of valid h20/h40/h60 opportunities'})
    print(f'输出目录：{directory}',flush=True)
    inputs,stores={},{}
    for name in ('resolved_config.json','source_hashes.json','independent_residual_audit.json'):
        record_input(Path(cfg['source_training_report_dir'])/name,inputs)
    write_json(directory/'source_reuse_audit.json',validate_timing_sources(cfg,PROJECT_ROOT))
    folds=load_fold_definitions(cfg,directory,inputs)
    years=sorted({y for f in folds for y in [f['test_year'],f['inner_year'],*f['core_years']]})
    universe,intraday,daily=load_frozen_market_data({**cfg,'test_years':years},directory,inputs)
    print('完整市场截面与真实来源准备完成，生成当前择时特征',flush=True)
    intraday=add_timing_features(intraday,cfg)
    labels=load_labels_only(cfg,directory,inputs,record_input)
    freeze_all_rules(folds,universe,intraday,daily,labels,cfg,directory,inputs,stores)
    del labels
    gc.collect()
    annual=evaluate_folds(folds,universe,intraday,daily,cfg,directory,inputs,stores)
    write_frame(annual,directory/'cross_validation_results.csv')
    write_frame(annual,directory/'annual_summary.csv')
    write_json(directory/'input_sources.json',inputs)
    for path,identity in inputs.items():
        if calculate_file_sha256(Path(path))!=identity['sha256']:
            raise AssertionError(f'运行期间只读输入改变：{path}')
    hashes=json.loads((directory/'source_hashes.json').read_text())
    for name,expected in hashes.items():
        if calculate_file_sha256(PROJECT_ROOT/name)!=expected:
            raise AssertionError(f'运行期间源码改变：{name}')
    write_json(directory/'runtime_integrity_audit.json',{'all_unchanged':True,
               'input_files_verified':len(inputs),'runtime_sources_verified':len(hashes)})
    write_timing_report(directory,cfg)
    print(f'报告完成：{directory/"final_report.md"}',flush=True)


if __name__=='__main__':
    main()
