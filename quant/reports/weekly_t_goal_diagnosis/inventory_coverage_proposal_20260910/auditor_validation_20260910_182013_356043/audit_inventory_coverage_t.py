"""独立核验v18原始短期限L2覆盖率训练、37规则及guard真实账户。

仅在成功终态后新增 independent_coverage_audit/run_*；不改模型、行情或原报告，
不训练或重撮合账户。0.70与原0.80选择独立重算；来源和账簿复用已审计只读方法。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import joblib
import lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dynamic_inventory_t import (Auditor as AccountAuditor, FAILURE_KEYS,
    close, digest, next_minute, parse_time, require)
from scripts.audit_inventory_guarded_t import (GUARD_COLUMNS, own_fillable, verify_guard_row,
    self_tests as guard_self_tests)
from scripts.audit_inventory_learning_t import (LearningAuditor, DATE_COLUMNS, CONTEXT,
    rules as original_rules, exact, frame_digest, feature_selection)
from scripts.audit_weekly_t_short_horizon import independent_training_rows, interval_overlap
from scripts.audit_weekly_t_timing_models import independent_timing_features, TIMING_COLUMNS
from scripts.audit_inventory_close_t import CloseAuditor, validate_goal_status
from scripts.audit_inventory_guarded_t import GuardedAuditor
from scripts.inventory_coverage_audit_models import (audit_coverage_model, compare_source_training, validate_native_parameters, coverage_feature_selection)
from scripts.audit_inventory_learning_t import decisions as short_decisions, recovery_times

VERSION = 'inventory_coverage_t_v18'
KEYS = ['datetime', 'symbol']
PROFILES = ('l2_depth4', 'l2_depth6')
PROFILE_MAP = {'l2_depth4': 'l2_depth4', 'l2_depth6': 'l2_depth6'}
CLOCKS = ('09:35', '09:45', '10:00', '10:30', '11:00', '13:05', '13:30', '14:00')
NEW_SOURCES = {'scripts/run_inventory_coverage_t.py', 'quant_backtest/inventory_t_frozen_cohort.py',
               'quant_backtest/inventory_t_coverage_models.py', 'quant_backtest/inventory_t_coverage_policy.py',
               'quant_backtest/inventory_t_coverage_reporting.py'}
COMPARISON_FIELDS = ('incremental_net_pnl', 'holding_cost_reduction_rate', 't_pair_count',
    'completed_t_pair_lots', 't_fee', 't_slippage_cost', 'inventory_mismatch_days',
    'recovery_failure_days', 'pending_pairs', 'all_reference_acquisition_cost')


def verify_comparison(current, original, saved):
    """独立重算所有同数字规则映射/分母/变化；不把不同预测误称相同信号。"""
    keys = list(coverage_rules()[0])
    exact(current[keys], pd.DataFrame(coverage_rules()), '候选表全部规则字段')
    exact(original[keys], pd.DataFrame(original_rules()), '来源候选表原规则字段')
    require(len(saved) == 37 and saved.candidate_index.tolist() == list(range(37)), '比较表候选缺失/乱序')
    for field in COMPARISON_FIELDS:
        for suffix, expected in (('coverage', current[field]), ('source', original[field]),
                                 ('change', current[field]-original[field])):
            require(np.array_equal(saved[field+'_'+suffix].to_numpy(), expected.to_numpy()), '对照列实际算术不同:'+field+'_'+suffix)
    require(np.allclose(current.all_reference_acquisition_cost, original.all_reference_acquisition_cost, rtol=0, atol=1e-8), '共同成本分母改变')
    require(saved.current_profile.equals(current.profile) and saved.source_profile.equals(original.profile), 'L2覆盖率/L2对照映射错误')
    require(saved.same_forecast_target.eq(True).all() and saved.same_forecasts.eq(False).all()
            and saved.same_execution_guard.eq(True).all(), '比较语义错误')
    require(saved.same_learning_objective.eq(True).all() and saved.source_feature_min_coverage.eq(.8).all()
            and saved.current_feature_min_coverage.eq(.7).all(),'唯一覆盖率对照语义错误')


def score_clock_funnel(predictions, rule, cfg):
    """描述分数门槛与事前时钟的分流，不以未来行情解释无请求。"""
    if not rule['enabled']: return {'enabled': False, 'horizons': []}
    rows=[]
    for horizon,p in predictions.items():
        fee=cfg['commission_rate']+cfg['transfer_rate']; slip=cfg['slippage_rate']
        stamp=np.where(p.datetime.lt(pd.Timestamp('2023-08-28')),cfg['stamp_rate_before_20230828'],cfg['stamp_rate_from_20230828'])
        buy=(1+p.prediction)*(1-slip)*(1-fee-stamp)-(1+slip)*(1+fee)
        sell=(1-slip)*(1-fee-stamp)-(1+p.prediction)*(1+slip)*(1+fee)
        net=np.maximum(buy,sell); score=(net>0)&(net>=rule['min_net_edge'])
        when=recovery_times(p.datetime,horizon)
        clock=when.notna()&when.le(p.datetime.dt.normalize()+pd.to_timedelta(cfg['fixed_recovery_time']+':00'))
        rejected=p.loc[score&~clock,KEYS+['prediction']].copy()
        rejected['predicted_net_edge']=net.loc[rejected.index]
        rows.append({'horizon_minutes':horizon,'current_rows':len(p),'score_margin_pass_rows':int(score.sum()),
            'score_pass_and_clock_legal_rows':int((score&clock).sum()),'score_pass_but_clock_invalid_rows':int((score&~clock).sum()),
            'first_clock_rejected_examples':rejected.head(5).to_dict('records')})
    return {'enabled':True,'funnel_definition':'score_and_predetermined_clock_only_other_current_gates_checked_by_full_decision_rebuild','horizons':rows}


def complete_pair_cost_rate(result):
    """目标分子仅完整组利润，不能用pending残仓的marked账户净额替代。"""
    cost=result['all_reference_acquisition_cost']
    return result['realized_complete_pair_pnl']/cost if cost else None


def report_selection(year, rule):
    """本轮新旧均为原L2档位，报告不虚构Huber映射列。"""
    return dict(test_year=year,**{k:rule[k] for k in ('enabled','candidate_index','profile','min_net_edge','fraction','exit_mode')})


def assert_no_outer_products(report, years):
    """关闭外评时实际产物也必须不存在，不能只相信布尔标记；仅检查路径不读取收益。"""
    names=('test_context','test_predictions','test_equity_curve.csv','test_trade_events.csv',
        'test_round_trips.csv','test_open_positions.csv','t_decisions.csv','metrics.json',
        'signal_causality_audit.json','first_leg_exit_side_audit.csv','exit_decisions.csv')
    unexpected=[str(Path(report)/f'year_{year}'/name) for year in years for name in names
                if (Path(report)/f'year_{year}'/name).exists()]
    require(not unexpected,'外评关闭但实际存在外评产物:'+str(unexpected))


def coverage_rules():
    """保留原独立37数值规则/L2档位，不导入生产候选器。"""
    return original_rules()


def attach_short_reference(core, labels, horizon):
    """完整core冻结之后左联原标签，列序复现原合同，缺价机会不被筛掉。"""
    require(horizon in (20,40,60) and not any(c.startswith(('execution_','recovery_','target_','label_'))
        or c in ('horizon_minutes','trade_week_complete') for c in core), 'core提前带未来/旧周资格')
    fields = ['recovery_time','recovery_price','label_start','label_end','target_return']
    source = labels.loc[labels.horizon_minutes.eq(horizon)]
    require(not core.duplicated(KEYS).any() and not source.duplicated(KEYS).any(), '重复core/标签身份')
    output = core.merge(source[KEYS+fields], on=KEYS, how='left', validate='one_to_one', sort=False)
    output['label_available'] = np.isfinite(output.target_return)
    output['horizon_minutes'] = horizon
    output['planned_recovery_time'] = recovery_times(output.datetime, horizon)
    output = output.merge(source[KEYS+['execution_time','execution_price']], on=KEYS, how='left', validate='one_to_one', sort=False)
    exact(output[list(core)], core, '联标签不得删掉或改变完整当前机会', dtype=True)
    return output


def coverage_decisions(predictions, rule, cfg):
    """独立核L2覆盖率身份/原分数，再用旧独立短期公式/时钟重建全部请求。"""
    if not rule['enabled']: return pd.DataFrame()
    require(set(predictions) == {20,40,60}, '三期限预测缺失')
    require(cfg['min_commission'] == 0, '不能忽略非零最低佣金')
    for horizon, frame in predictions.items():
        require(not frame.columns.duplicated().any() and not frame[KEYS].isna().any().any(), '当前字段或身份缺失')
        time = parse_time(frame.datetime)
        require(time.dt.tz is None and time.dt.strftime('%H:%M').isin(CLOCKS).all()
                and time.eq(time.dt.floor('min')).all(), '不合法原精确信号时钟')
        require('forecast_alpha' not in frame and frame.model_version.eq(VERSION).all()
                and frame.forecast_loss.eq('regression_l2').all() and frame.feature_min_coverage.eq(.7).all(), '预测实际版本/L2/覆盖率错')
        require(np.array_equal(frame.prediction, frame.raw_prediction) and np.array_equal(frame.prediction, frame.return_score), '预测分数与原L2覆盖率位置评分不同')
    output = short_decisions(predictions, rule, cfg)
    output['model_version'], output['forecast_loss'], output['feature_min_coverage'] = VERSION, 'regression_l2', .7
    return output


class CoverageAuditor(CloseAuditor):
    """复用独立lineage/估值/账户父核，新版本身份/模型/预测/规则均在此核验。"""
    def __init__(self, report, out):
        self.report, self.out = report, out
        self.inputs, self.raw, self.full_raw, self.raw_files, self.results = {}, {}, {}, {}, []
        self.cfg, self.protocol = self.json(report/'resolved_config.json'), self.json(report/'research_protocol.json')
        require(self.cfg['research_version'] == VERSION and self.protocol['version'] == VERSION, '只支持v18身份')
        require(self.cfg['training_loss'] == 'regression_l2' and self.cfg['feature_min_coverage'] == .7
                and self.cfg['source_feature_min_coverage'] == .8 and 't_huber_alpha' not in self.cfg
                and self.cfg['forecast_target'] == 'fixed_20_40_60_reference_return'
                and self.cfg['horizons'] == [20,40,60] and self.cfg['learning_model_profiles'] == list(PROFILES)
                and self.cfg['t_recheck_exit_side_at_execution'] is True, '新loss/原期限/guard合同错误')
        require(tuple(self.cfg['signal_times']) == CLOCKS and self.cfg['last_entry_time'] == '14:30'
                and self.cfg['fixed_recovery_time'] == '14:55', '当前信号/deadline改变')
        self.checked_raw_events = self.checked_exit_pairs = 0
        self.cache_audits, self.sparse_arrays, self.mark_cache, self.exit_cache, self.base_anchors = [], {}, {}, {}, {}
        self.valuation_mode = 'full_minute_reference_stream_then_daily_close'
        require(self.protocol['valuation_contract'] == self.valuation_mode, '估值合同错误')
        self.market_calendar = pd.DatetimeIndex([])
        self.execution_source = Path(self.cfg['source_execution_report_dir'])
        self.learning_source = Path(self.cfg['source_learning_report_dir'])
        self.guards, self.model_results, self.prediction_results, self.cohort_results, self.baselines_checked = [], [], [], [], []
        self.decision_funnels = []

    # 真正继承的是v15独立三期限guard重放方法；不传伪造cfg/版本。
    guard_account = GuardedAuditor.guard_account

    def account(self, folder):
        """旧逐笔核已验证每腿滑点；新返回与报告补独立T腿滑点合计。"""
        result=AccountAuditor.account(self,folder)
        events=self.table(folder/'test_trade_events.csv')
        t=events.loc[events.event_type.eq('t')]
        result['t_slippage_cost']=float((t.shares*(t.price-t.reference_price).abs()).sum())
        close(float(t.slippage_cost.sum()),result['t_slippage_cost'],'全部T腿参考价/成交价独立滑点')
        close(self.json(folder/'metrics.json')['t_slippage_cost'],result['t_slippage_cost'],'逐T腿滑点合计')
        return result

    def sources(self):
        """分别核v15执行对照与v14完整core证明，不能继承同预测的旧审计假设。"""
        terminal = self.json(self.report/'runtime_integrity_audit.json')
        require(terminal.get('all_unchanged') is True and (self.report/'final_report.md').exists(), '主运行未成功终态')
        self.record(self.report/'final_report.md')
        hashes = self.json(self.report/'source_hashes.json')
        require(len(hashes) == terminal['runtime_sources_verified'] == 56, '原51源+新5源不齐')
        for name, value in hashes.items():
            require(digest(self.record(ROOT/name)) == value and digest(self.record(self.report/'source_snapshot'/name)) == value, '运行源/快照变化')
        registered = self.json(self.report/'input_sources.json')
        for path, identity in registered.items(): require(digest(self.record(path)) == identity['sha256'], f'登记输入变化:{path}')
        self.source_audits = {}
        for source, kind, version in ((self.execution_source, 'guarded', 'inventory_guarded_t_v15'),
                                      (self.learning_source, 'learning', 'inventory_learning_t_v14')):
            passed = []
            for path in sorted(source.glob(f'independent_{kind}_audit/run_*/verification.json')):
                value = self.json(path)
                if value.get('research_version') == version and all(value.get(k) is True for k in ('passed', 'complete', 'all_read_inputs_unchanged')):
                    passed.append((path, value))
            require(passed, f'缺完整{version}来源独审')
            audit_path, audit = (next((p,v) for p,v in passed if p.resolve() == Path(self.cfg['source_learning_audit']).resolve()) if kind == 'learning' else passed[-1])
            config = self.json(source/'resolved_config.json')
            require(config['research_version'] == version, '来源报告身份错')
            for name in ('resolved_config.json', 'runtime_integrity_audit.json', 'source_hashes.json',
                         'all_rules_frozen_before_outer_test.json'):
                path = (source/name).resolve()
                require(audit['inputs_sha256'].get(str(path)) == digest(self.record(path)), '来源关键文件偏离完整独审')
            self.source_audits[kind] = (audit_path, audit)
        original = self.json(self.execution_source/'resolved_config.json')
        ignored = {'research_version', 'configuration_source', 'source_configuration_source', 'source_report_scope',
            'output_dir', 'workers', 'candidate_workers', 'test_years', 'internal_only', 'learning_model_profiles', 'source_learning_version',
            'frozen_t_models_reused', 't_models_fitted_this_run', 'feature_min_coverage'}
        for name, value in original.items():
            if name not in ignored: require(self.cfg.get(name) == value, f'非loss的原合同改变:{name}')
        require(set(self.cfg['test_years']).issubset(original['test_years']), '扩张来源评价范围')
        require(Path(self.cfg['source_learning_audit']).resolve() == self.source_audits['learning'][0].resolve()
                and self.cfg['source_learning_audit_sha256'] == digest(self.source_audits['learning'][0]), 'core来源审计身份错')
        require(self.protocol['candidate_rules'] == coverage_rules() and self.protocol['models_per_fold'] == 6
                and self.protocol['models_fitted_this_run'] == 6*len(self.cfg['test_years'])
                and self.protocol['frozen_t_models_reused'] is False and self.protocol['outer_model_refit'] is False
                and self.protocol['cross_horizon_argmax'] is True, '新训练/目标/候选协议错误')
        require(original['feature_min_coverage']==.8 and self.protocol['training_loss'] == 'regression_l2'
            and self.protocol['source_loss']=='regression_l2' and self.protocol['feature_min_coverage']==.7
            and self.protocol['source_feature_min_coverage']==.8 and self.protocol['only_feature_coverage_threshold_changed'] is True
            and self.protocol['same_selected_features_required'] is False and self.protocol['same_training_rows_and_labels_required'] is True
            and self.protocol['reconstructed_source_feature_selection_required'] is True
            and self.protocol['native_missing_values_preserved'] is True and self.protocol['coverage_threshold_selected_using_validation_returns'] is False
            and self.protocol['labels_winsorized_or_deleted'] is False
            and self.protocol['score_is_calibrated_conditional_mean'] is False, '原始标签/L2覆盖率协议错误')
        old_sources = self.json(self.execution_source/'source_hashes.json')
        require(len(old_sources) == 51 and set(hashes) == set(old_sources)|NEW_SOURCES, '实际runtime路径closure不符')
        require(all(hashes[k] == v for k,v in old_sources.items()), '原51运行源改变')
        require(self.cfg['source_learning_model_profiles'] == original['learning_model_profiles'], '旧L2 profile映射身份错')
        frozen = self.json(self.report/'all_rules_frozen_before_outer_test.json')
        require(frozen['test_years'] == self.cfg['test_years'] and len(frozen['model_artifacts_sha256']) == 60*len(self.cfg['test_years']), '每折6模型/每模型10产物冻结不全')
        self.freeze, self.freeze_time = frozen, (self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns
        for name, expected in frozen['model_artifacts_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, '新模型应相对本报告冻结')
            path = self.report/name
            require(digest(self.record(path)) == expected and path.stat().st_mtime_ns <= self.freeze_time, '模型freeze后变化')
        self.source_summary = dict(runtime_sources=len(hashes), registered_inputs=len(registered),
            newly_fitted_models=6*len(self.cfg['test_years']), new_model_artifacts=len(frozen['model_artifacts_sha256']),
            source_execution_audit=str(self.source_audits['guarded'][0]), source_learning_audit=str(self.source_audits['learning'][0]),
            old_t_weights_used_for_new_prediction=False, source_prediction_equality_required=False)

    def market(self):
        """复用旧独立全市场增强/标签读取，原标签文件必须再被完整v14证明锚定。"""
        universe, minutes, daily, labels = LearningAuditor.market(self)
        del daily
        manifest = self.json(self.report/'horizon_manifest.json')
        path = Path(manifest['labels_path']).resolve()
        require(self.source_audits['learning'][1]['inputs_sha256'].get(str(path)) == digest(path), '完整原标签cache未被v14完整独审锚定')
        return universe, minutes, labels


    def predictions(self, fold, scope, minutes, profiles):
        """完整当前键/字段复算并用本折L2覆盖率原生预测逐值核对，无未来标签资格。"""
        folder = self.report/f"year_{fold['test_year']}"
        context = folder/f'{scope}_context'
        identity = self.json(context/'full_minute_coverage_audit.json')
        year = fold['inner_year'] if scope == 'validation' else fold['test_year']
        days = self.market_calendar[self.market_calendar.year == year]
        if self.cfg['smoke_days']: days = days[:self.cfg['smoke_days']]
        current = minutes.loc[minutes.symbol.isin(identity['symbols']) & minutes.datetime.dt.normalize().isin(days)].copy()
        output = {}
        for profile in profiles:
            output[profile] = {}
            for horizon in (20,40,60):
                key = f'h{horizon}_{profile}'
                directory = folder/'models'/key
                meta = self.json(directory/'model_metadata.json')
                target = folder/f'{scope}_predictions'/key
                saved, audit = self.table(target/'current_predictions.csv'), self.json(target/'prediction_audit.json')
                require(digest(target/'current_predictions.csv') == audit['predictions_sha256']
                        and audit['model_bundle_sha256'] == meta['model_bundle_sha256'], '当前预测与本折模型SHA不符')
                current_names = list(dict.fromkeys(KEYS+['signal_price','signal_high','signal_low','signal_amount',
                    'signal_volume','previous_close']+meta['features']+['feature_source_start']))
                predicted_names = ['prediction','raw_prediction','model_profile','horizon_minutes','return_score',
                                   'model_version','forecast_loss','feature_min_coverage']
                require(set(saved) == set(current_names+predicted_names), '归档当前预测字段合同改变')
                for name in current_names:
                    if pd.api.types.is_numeric_dtype(current[name].dtype): saved[name] = saved[name].astype(current[name].dtype)
                exact(saved[current_names], current[current_names], '全部当前推断键/特征/来源', dtype=True)
                require(saved.model_profile.eq(profile).all() and saved.horizon_minutes.eq(horizon).all()
                        and saved.model_version.eq(VERSION).all() and saved.forecast_loss.eq('regression_l2').all()
                        and saved.feature_min_coverage.eq(.7).all(), '预测版本/期限/L2/覆盖率身份错')
                matrix = current[meta['features']].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
                native = lightgbm.Booster(model_file=str(self.record(directory/'return_model.txt')))
                require(native.params['objective'] == 'regression', '预测原生损失错')
                values = np.asarray(native.predict(matrix, num_threads=1)) if len(matrix) else np.array([])
                require(values.shape == (len(current),) and np.isfinite(values).all(), '预测shape/有限性错')
                require(all(np.array_equal(values,saved[name]) for name in ('prediction','raw_prediction','return_score')), '原生全部当前预测不一致')
                require(audit['future_independent'] is True and audit['horizon_minutes'] == horizon
                        and audit['forecast_loss'] == 'regression_l2' and audit['feature_min_coverage'] == .7
                        and audit['current_rows'] == len(current), '未来擦除/预测归档身份错')
                policy = current[KEYS+['signal_price','signal_high','signal_low','signal_amount','signal_volume','previous_close']].copy()
                for name in predicted_names:
                    policy[name] = saved[name].to_numpy()
                output[profile][horizon] = policy
                self.prediction_results.append(dict(path=str(target.relative_to(self.report)), current_rows=len(current),
                    profile=profile, horizon_minutes=horizon, native_predictions_exact=True, all_current_inputs_exact=True,
                    future_eligibility_filter_absent=True, same_core_model_before_and_after_inner_selection=True))
        return output
    def run(self):
        """先新目标训练重建，再内部/外层真实账户；不使用审计结果改变任何规则。"""
        # 新模型核显式调用独立helper，不导入生产模型。
        self.sources()
        universe, minutes, labels = self.market()
        print('56源与双来源身份通过，开始完整core/三期限L2覆盖率训练独立重建。', flush=True)
        folds = []
        for year in self.cfg['test_years']:
            folder = self.report/f'year_{year}'
            fold = self.json(folder/'split.json')
            previous = self.json(self.execution_source/f'year_{year}/split.json')
            for key in ('test_year', 'inner_year', 'core_years', 'train_years'):
                require(fold[key] == previous[key], '年度隔离拆分偏离原执行来源')
            require(sorted(fold['core_years']+[fold['inner_year']]) == sorted(fold['train_years'])
                and year not in fold['train_years'] and fold['inner_year'] not in fold['core_years'], '核心/内部/外层年交集')
            core = self.cohort(fold, universe, minutes)
            for horizon in (20,40,60):
                full = attach_short_reference(core, labels, horizon)
                training = independent_training_rows(full, fold['core_years'], [year, fold['inner_year']], self.cfg)
                require(len(training)>0, '实际核心训练为空')
                features, feature_audit = coverage_feature_selection(training, self.cfg)
                for profile in PROFILES:
                    detail = audit_coverage_model(self, folder/f'models/h{horizon}_{profile}', training, features,
                        feature_audit, fold, horizon, profile, full_frame=full)
                    self.model_results.append(detail)
                print(f'fold={year} h{horizon}两个L2覆盖率模型/原L2四产物/完整core及raw标签通过：{len(training):,}行。', flush=True)
            folds.append(fold)
            del core, full, training
            gc.collect()
        del universe, labels
        gc.collect()
        evaluation = self.json(self.report/'outer_evaluation_decision.json')
        if not evaluation['outer_evaluation_started']:assert_no_outer_products(self.report,self.cfg['test_years'])
        choices, count, outer, presented, comparisons, final_rules = [], 0, 0, {}, [], {}
        for fold in folds:
            year, folder = fold['test_year'], self.report/f"year_{fold['test_year']}"
            selected = self.json(folder/'selected_rule.json')
            require(digest(folder/'selected_rule.json') == self.freeze['selected_rule_sha256'][str(year)]
                and (folder/'selected_rule.json').stat().st_mtime_ns <= self.freeze_time, '已选规则偏离统一冻结')
            self.audit_context(folder/'validation_context', fold['inner_year'])
            self.source_baseline(year, 'validation')
            forecasts = self.predictions(fold, 'validation', minutes, PROFILES)
            table = self.table(folder/'validation_rule_metrics.csv')
            original = self.table(self.execution_source/f'year_{year}/validation_rule_metrics.csv')
            require(len(table) == 37 and set(table.candidate_index) == set(range(37)), '37内部候选不齐')
            comparison = self.csv(folder/'coverage_vs_source_candidates.csv')
            verify_comparison(table, original, comparison)
            comparisons.append(comparison.assign(test_year=year, inner_year=fold['inner_year']))
            actual = []
            for rule in coverage_rules():
                index = rule['candidate_index']
                target = folder/f'validation_candidates/candidate_{index:03d}'
                require(self.json(target/'rule.json') == rule, '事前规则改变')
                expected = coverage_decisions(forecasts.get(rule['profile'], {}), rule, self.cfg)
                self.decision_funnels.append(dict(test_year=year, scope='validation', candidate_index=index,
                    request_count=len(expected), **score_clock_funnel(forecasts.get(rule['profile'], {}),rule,self.cfg)))
                saved = self.table(target/'t_decisions.csv')
                if len(expected): exact(saved[list(expected)], expected, '独立L2覆盖率三期限/成本/当前条件决策')
                else: require(saved.empty, '伪造空候选请求')
                self.guard_account(target, forecasts, rule)
                result = self.account(target)
                row, old = table.loc[table.candidate_index.eq(index)].iloc[0], original.loc[original.candidate_index.eq(index)].iloc[0]
                for name in ('incremental_net_pnl', 't_pair_count', 'completed_t_pair_lots', 't_fee', 't_slippage_cost', 'all_reference_acquisition_cost', *FAILURE_KEYS):
                    close(row[name], result[name], f'内部实际账户指标:{name}')
                if result['all_reference_acquisition_cost']:
                    close(row.holding_cost_reduction_rate, complete_pair_cost_rate(result), '内部完整组真实成本比例', 1e-12)
                else: require(pd.isna(row.holding_cost_reduction_rate), '零底仓成本比例须缺失')
                require(row.request_count == len(expected), '候选请求计数不同')
                close(row.all_reference_acquisition_cost, old.all_reference_acquisition_cost, '原全部底仓成本分母', 1e-8)
                actual.append(dict(candidate_index=index, **result))
                count += 1
            eligible = [r for r in actual if r['candidate_index'] and r['t_pair_count'] > 0 and all(r[k] == 0 for k in FAILURE_KEYS)]
            winner = min(eligible, key=lambda r: (-r['incremental_net_pnl'], r['candidate_index'])) if eligible else actual[0]
            require(selected['rule'] == coverage_rules()[winner['candidate_index']], '内部选参不是执行合格后最大真实净額')
            expected_rate = winner['incremental_net_pnl']/winner['all_reference_acquisition_cost'] if winner['all_reference_acquisition_cost'] else 0.
            close(selected['inner_metrics']['holding_cost_reduction_rate'], expected_rate, '实际5%分母', 1e-12)
            choices.append(dict(test_year=year, candidate_index=winner['candidate_index'], inner_net=winner['incremental_net_pnl'], cost_rate=expected_rate))
            for name in ('planning_baseline', 'baseline'): self.account(folder/'validation_context'/name)
            if evaluation['outer_evaluation_started']:
                self.audit_context(folder/'test_context', year)
                self.source_baseline(year, 'test')
                rule = selected['rule']
                forecasts = self.predictions(fold, 'test', minutes, [rule['profile']]) if rule['enabled'] else {}
                expected = coverage_decisions(forecasts.get(rule['profile'], {}), rule, self.cfg)
                self.decision_funnels.append(dict(test_year=year,scope='test',candidate_index=rule['candidate_index'],
                    request_count=len(expected),**score_clock_funnel(forecasts.get(rule['profile'],{}),rule,self.cfg)))
                saved = self.table(folder/'t_decisions.csv')
                if len(expected): exact(saved[list(expected)], expected, '外层本折原生L2覆盖率及已冻规则')
                else: require(saved.empty, 'hold外层产生请求')
                require((folder/'test_equity_curve.csv').stat().st_mtime_ns >= self.freeze_time, '外层在全部规则冻结前发生')
                self.guard_account(folder, forecasts, rule)
                presented[year] = self.account(folder)
                final_rules[year] = rule
                for name in ('planning_baseline', 'baseline'): self.account(folder/'test_context'/name)
                outer += 1
            print(f'fold={year} 37内部/已有外层、全部guard与原真实账簿通过。', flush=True)
        exact(self.csv(self.report/'coverage_vs_source_candidates.csv'),pd.concat(comparisons,ignore_index=True),'根所有候选对照完整',dtype=True)
        passed = all(item['candidate_index'] > 0 and item['cost_rate'] >= self.cfg['profit_target_rate'] for item in choices)
        require(evaluation['all_inner_cost_targets_met'] == passed, '实际内部5%判断错')
        require(evaluation['outer_evaluation_started'] == (not self.cfg['internal_only'] and (passed or self.cfg['report_scope'] == 'smoke')), '外评启停错误')
        goal = self.json(self.report/'goal_evaluation.json')
        validate_goal_status(goal, self.cfg, evaluation['outer_evaluation_started'])
        if presented:
            annual = self.csv(self.report/'annual_summary.csv')
            require(not annual.test_year.duplicated().any() and set(annual.test_year) == set(presented), '最终年度表范围错')
            sums = dict(sum_independent_account_t_increment=0., sum_completed_pair_lots=0.,
                sum_one_way_traded_lots=0., sum_complete_pairs=0, sum_recovery_fragments=0)
            for year, result in presented.items():
                row = annual.loc[annual.test_year.eq(year)].iloc[0]
                mapping = {'incremental_marked_pnl': 'incremental_net_pnl', 'completed_t_pair_lots': 'completed_t_pair_lots',
                    'total_t_traded_lots': 'total_t_traded_lots', 'complete_t_pairs': 't_pair_count',
                    'recovery_fragments': 'fragment_count', 't_fee': 't_fee', 't_slippage_cost': 't_slippage_cost', 'all_reference_acquisition_cost': 'all_reference_acquisition_cost',
                    **{k: k for k in FAILURE_KEYS}}
                for name, actual in mapping.items(): close(row[name], result[actual], f'最终年度实际账户:{year}/{name}')
                sums['sum_independent_account_t_increment'] += result['incremental_net_pnl']
                sums['sum_completed_pair_lots'] += result['completed_t_pair_lots']
                sums['sum_one_way_traded_lots'] += result['total_t_traded_lots']
                sums['sum_complete_pairs'] += result['t_pair_count']
                sums['sum_recovery_fragments'] += result['fragment_count']
            for name, actual in sums.items(): close(goal[name], actual, '根目标汇总:'+name)
            self.verify_presented_summary(final_rules)
        for path, value in self.inputs.items(): require(digest(path) == value, f'独审读取期间输入改变:{path}')
        return dict(passed=True, complete=True, research_version=VERSION, report_scope=self.cfg['report_scope'],
            source=self.source_summary, models=self.model_results, predictions=self.prediction_results,
            core_cohorts=self.cohort_results, candidate_accounts=count, outer_accounts=outer,
            checked_accounts=len(self.results), accounts=self.results, choices=choices, source_baselines=self.baselines_checked,
            no_outer_products_when_disabled_verified=not evaluation['outer_evaluation_started'],
            current_decision_funnels=self.decision_funnels, all_37_numeric_rule_fields_and_source_comparison_changes_checked=True,
            guard_accounts=self.guards, guard_rows_verified=sum(x['guard_rows'] for x in self.guards),
            cancellations_verified=sum(x['cancelled'] for x in self.guards), full_minute_caches=self.cache_audits,
            raw_event_checks=self.checked_raw_events, raw_exit_intent_checks=self.checked_exit_pairs, raw_files=self.raw_files,
            market=self.market_summary, inputs_sha256=self.inputs, all_read_inputs_unchanged=True,
            no_production_coverage_model_policy_frozen_cohort_or_guard_engine_imported=True,
            no_model_fit_or_account_rerun=True, goal_achieved=goal['goal_achieved'],
            limitations=['旧全截面session/provenance使用已冻结已审计实现；当前特征、短期原始端点和策略另行复算；从完整core独立重建0.70/0.80选择，原行/标签/隔离和共同列保持，新完整矩阵允许改变。',
                '原lineage/排名与核心规划账来自完整v14证明，并用旧独立核心审计重新核对；不重撮合全部未成交基础请求。',
                '缺真实涨跌停字段仍使用原代理；并不补盘口排队或历史ST点时资料。',
                '账目及来源通过不代表所有候选每日恢复或5%收益达标；历史复测不是全新测试。'])

    def verify_presented_summary(self, final_rules):
        """从已经逐笔核过的年度表再核报告统计，不调用生产报告器自证。"""
        meta=self.json(self.report/'inventory_coverage_summary.json')
        identities={'research_version':VERSION,'frozen_t_models_reused':False,'models_retrained':True,
            'newly_fitted_models_per_fold':6,'newly_fitted_model_profiles':list(PROFILES),
            'prediction_horizons_minutes':[20,40,60],'training_loss':'regression_l2','objective':'regression','feature_min_coverage':.7,'source_feature_min_coverage':.8,
            'candidate_count':37,'uniform_1455_forecast_target':False,'cross_horizon_argmax':True,
            'complete_training_matrix_identity_required':False,'selected_features_and_column_order_may_change':True,
            'source_feature_screen_reproduction_required':True,'common_feature_values_and_dtypes_equal_required':True,
            'missing_feature_values_preserved_as_nan':True,'missing_feature_rows_deleted':False,
            'same_training_rows_labels_split_required':True,'additional_15m_features_guaranteed_selected':False,
            'coverage_candidate_features':['f_return_15m','f_timing_market_return_15m','f_timing_excess_return_15m'],
            'predicted_edge_is_expected_portfolio_net_pnl':False,'t_recheck_exit_side_at_execution':True,
            'source_execution_report_dir':str(self.execution_source),'source_learning_report_dir':str(self.learning_source)}
        for key,value in identities.items(): require(meta[key]==value,'报告身份/解释错误:'+key)
        selected=[]; guards=[]; exits=[]; horizons=[]
        from scripts.audit_inventory_learning_t import SESSION
        for year,rule in final_rules.items():
            selected.append(report_selection(year,rule))
            folder=self.report/f'year_{year}'
            events=self.table(folder/'test_trade_events.csv')
            first=events.loc[events.event_type.eq('t')&events.fragment_id.eq(0)]
            guard=self.table(folder/'first_leg_exit_side_audit.csv'); cancelled=guard.cancelled.eq(True)
            dates=parse_time(guard.datetime); stockdays=pd.DataFrame({'date':dates.dt.normalize(),'symbol':guard.symbol})
            actual_keys=set(zip(first.datetime,first.symbol,first.side))
            passed_keys=set(zip(dates.loc[~cancelled],guard.loc[~cancelled,'symbol'],guard.loc[~cancelled,'side']))
            guards.append(dict(test_year=year,guard_audit_requests=len(guard),guard_checked_requests=int(guard.opposite_checked.eq(True).sum()),
                guard_passed_requests=int((~cancelled).sum()),guard_cancelled_requests=int(cancelled.sum()),
                guard_cancelled_days=int(dates.loc[cancelled].dt.normalize().nunique()),guard_cancelled_stock_days=len(stockdays.loc[cancelled].drop_duplicates()),
                guard_cancelled_own_fillable_shares=float(guard.loc[cancelled,'own_fillable_shares'].sum()),guard_actual_first_leg_groups=len(first),
                guard_passed_without_actual_first_leg=len(passed_keys-actual_keys),guard_cancelled_price_limit=int((cancelled&guard.opposite_quote_reason.eq('price_limit')).sum()),
                guard_cancelled_limit_proxy=int((cancelled&guard.opposite_quote_reason.eq('conservative_price_limit_proxy')).sum())))
            frame=self.table(folder/'exit_decisions.csv')
            start=parse_time(frame.first_execution_time); deadline=parse_time(frame.original_deadline)
            positions={v:i for i,v in enumerate(SESSION)}
            delta=(deadline.dt.hour*60+deadline.dt.minute).map(positions)-(start.dt.hour*60+start.dt.minute).map(positions)
            require(delta.isin([20,40,60]).all() and start.dt.normalize().eq(deadline.dt.normalize()).all(),'报告pair原期限不合法')
            horizons.append(dict(test_year=year,latched_pairs_with_checked_deadline=len(frame),**{f'planned_h{h}_pairs':int(delta.eq(h).sum()) for h in (20,40,60)}))
            exits.append(dict(test_year=year,selected_exit_mode=rule['exit_mode'],observed_exit_modes=','.join(sorted(frame.exit_mode.unique())),
                started_pairs=len(first),latched_exit_pairs=len(frame),started_pairs_without_exit_signal=len(set(first.pair_id)-set(frame.pair_id)),
                **{f'{reason}_trigger_pairs':int(frame.exit_reason.eq(reason).sum()) for reason in ('profit_buffer','stop_loss','hard_deadline')},
                early_trigger_pairs=int(parse_time(frame.exit_earliest_time).lt(deadline).sum())))
        for name,expected,key in [('selected_coverage_rules.csv',selected,'selected_rules'),('guard_summary.csv',guards,'annual_guard_summaries'),
            ('dynamic_exit_summary.csv',exits,'annual_exit_summaries'),('planned_horizon_summary.csv',horizons,'annual_planned_horizon_summaries')]:
            saved=self.csv(self.report/name)
            if 'observed_exit_modes' in saved: saved['observed_exit_modes']=saved.observed_exit_modes.fillna('')
            if 'source_l2_profile' in saved: saved['source_l2_profile']=saved.source_l2_profile.astype(object).where(saved.source_l2_profile.notna(),None)
            exact(saved,pd.DataFrame(expected),'独立报告派生统计:'+name)
            require(meta[key]==expected,'报告JSON与实际逐笔计数不同:'+key)


def audit_helper_closure():
    """静态递归归档本地独立审计helpers；与56个策略运行源分开列示。"""
    import ast
    pending=[Path(__file__).resolve(), ROOT/'scripts/test_audit_inventory_coverage_t.py']
    found={}
    while pending:
        path=pending.pop().resolve()
        if path in found: continue
        found[path]=digest(path)
        tree=ast.parse(path.read_text())
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node,ast.Import): modules.extend(alias.name for alias in node.names)
        for module in modules:
            if module.startswith('scripts.'):
                other=ROOT/Path(*module.split('.')).with_suffix('.py')
                if other.exists(): pending.append(other)
    return {str(path):value for path,value in sorted(found.items())}


def main():
    """每次独审单独目录保留成功/失败脚本与日志，不覆盖既有任何产物。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    from scripts.test_audit_inventory_coverage_t import run_self_tests
    if args.self_test:
        print(json.dumps({'coverage':run_self_tests(),'guard':guard_self_tests()},ensure_ascii=False));return
    require(args.report_dir,'需要--report-dir')
    report=(ROOT/args.report_dir).resolve()
    require((report/'runtime_integrity_audit.json').exists() and (report/'final_report.md').exists(),'只允许主运行成功终态后独审')
    out=report/'independent_coverage_audit'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True,exist_ok=False)
    hashes=audit_helper_closure()
    for name,value in hashes.items():
        path=Path(name);target=out/'audit_dependencies'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        require(digest(target)==value,'审计依赖副本SHA不符')
    shutil.copy2(__file__,out/Path(__file__).name)
    (out/'command.txt').write_text(' '.join([sys.executable,*sys.argv])+'\n')
    oldout,olderr=sys.stdout,sys.stderr
    class Tee:
        def __init__(self,terminal,stream):self.terminal,self.stream=terminal,stream
        def write(self,text):self.terminal.write(text);self.stream.write(text);self.stream.flush()
        def flush(self):self.terminal.flush();self.stream.flush()
    start,auditor=time.time(),None
    with (out/'audit.log').open('w') as logfile:
        sys.stdout,sys.stderr=Tee(oldout,logfile),Tee(olderr,logfile)
        try:
            print(f'独立v18审计目录:{out}',flush=True)
            tests={'coverage':run_self_tests(),'guard':guard_self_tests()}
            auditor=CoverageAuditor(report,out)
            result=auditor.run()
            require(all(digest(path)==value for path,value in hashes.items()),'独审依赖在运行期间改变')
            # 记录实际加载的本地脚本模块，不能漏掉通过继承引入的close辅助核。
            loaded={str(Path(module.__file__).resolve()) for module in tuple(sys.modules.values())
                if getattr(module,'__file__',None) and Path(module.__file__).resolve().parent==ROOT/'scripts'}
            require(loaded.issubset(hashes),'静态审计依赖闭合漏掉实际加载的本地脚本:'+str(loaded-set(hashes)))
            result.update(self_tests=tests,elapsed_seconds=time.time()-start,script_sha256=digest(__file__),
                audit_dependencies_sha256=hashes,audit_dependencies_unchanged=True,loaded_audit_helpers=sorted(loaded))
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            result=dict(passed=False,complete=False,research_version=VERSION,error=error,
                elapsed_seconds=time.time()-start,script_sha256=digest(__file__),audit_dependencies_sha256=hashes,
                inputs_sha256=auditor.inputs if auditor else {})
        finally:sys.stdout,sys.stderr=oldout,olderr
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
    rows=['# v18 L2覆盖率原始短期限预测与实际执行独立审计','',
        '结果：'+('通过。' if result['passed'] else '失败，保留本次原错误与脚本。'),'',
        '只读保存产物及原始行情，不训练、不重跑账户、不改原报告或执行源。','']
    if result['passed']:
        rows += [f"核{len(result['models'])}个模型、{len(result['predictions'])}张完整当前预测、{result['candidate_accounts']}内部候选和{result['outer_accounts']}外层；含基准共{result['checked_accounts']}账户。",
            '',f"{result['source']['runtime_sources']}运行源、{len(hashes)}独立审计脚本依赖、{len(result['inputs_sha256'])}项读取输入始末一致。",
            '',f"逐笔核{result['raw_event_checks']}次成交、{result['raw_exit_intent_checks']}个退出意图；guard {result['guard_rows_verified']}次复查/{result['cancellations_verified']}次取消。跨候选数量不代表一个账户。",
            '', '完整原始标签、来源purge、0.70新矩阵与0.80重建控制、共同列NaN/dtype、实际L2参数/全部树及37规则独立复算。选择数量不预定；全部当前键不按未来端点可用性删减。',
            '',f"goal_achieved={result['goal_achieved']}。审计通过仅确认来源、训练、决策及账目可信；不等于盈利或交易频率改善。",'',
            *['- '+line for line in result['limitations']],'','详情见[verification.json](verification.json)，原错误/命令见audit.log/command.txt。']
    else:rows += ['```text',result['error'],'```']
    (out/'acceptance.md').write_text('\n'.join(rows)+'\n')
    manifest={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file()}
    (out/'artifact_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(out,flush=True)
    if not result['passed']:raise AssertionError(result['error'])


if __name__=='__main__':main()
