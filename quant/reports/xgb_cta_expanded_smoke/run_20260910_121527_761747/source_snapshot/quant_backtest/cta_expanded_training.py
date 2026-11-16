"""扩展XGBoost研究：内部向前比较三类目标，冻结后才读取2026预测。

输入为无标签特征PKL和只读报价；输出原生模型、训练行审计、全部候选、年度报告。
2023—2025最终模型的同期回放明确标为样本内，不冒充向前测试。
"""

from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from .annual_panel import write_json
from .cta_research_data import calculate_sha256, calculate_identity, save_pickle
from .cta_research_engine import run_portfolio
from .cta_research_reporting import audit_result, save_result, LIMITATIONS
from .cta_research_training import calculate_entry_mask, load_labels, select_features
from .strategy import BaseStrategy


def create_internal_folds() -> list[dict]:
    """返回预先固定的内部时间折，不包含2026；标签须在各训练截止之前完成。"""
    return [dict(year=year, train_start=pd.Timestamp('2023-01-01'), train_end=pd.Timestamp(start),
                 validation_start=pd.Timestamp(start), validation_end=pd.Timestamp(end))
            for year, start, end in [(2023, '2023-07-01', '2023-10-01'),
                                      (2024, '2024-01-01', '2025-01-01'),
                                      (2025, '2025-01-01', '2026-01-01')]]


def truncate_market(market, cutoff: pd.Timestamp):
    """截断market到cutoff之前，阻止训练路径标签读取2026价格，而非事后只删目标列。"""
    n = market.times.searchsorted(cutoff)
    return SimpleNamespace(times=market.times[:n], quotes=market.quotes[:n], flags=market.flags[:n],
                           boards=market.boards, symbols=market.symbols, symbol_index=market.symbol_index)


def fit_candidate(panel: pd.DataFrame, labels: pd.DataFrame, fold: dict, exit_id: int,
                  profile: dict, objective: str, cfg: dict, directory: Path) -> tuple[object, list[str]]:
    """只使用fold过去特征和结束标签，按objective拟合；保存实际行、特征、原生权重及参数。"""
    rows = labels.join(panel[['datetime', 'symbol']])
    target_key, end_key = f'net_return_{exit_id}', f'label_end_{exit_id}'
    rows = rows.loc[(rows.datetime >= fold['train_start']) & (rows.datetime < fold['train_end']) &
                    (rows[end_key] < fold['train_end']) & rows[target_key].notna()].copy()
    if len(rows) < 100:
        raise ValueError('核心训练标签不足100行')
    data = panel.loc[rows.index]
    net = rows[target_key].clip(-0.5, 0.5)
    target = (net > 0).astype(float) if objective == 'probability' else net * 100
    features, audit = select_features(data, target, cfg)
    params = {k: v for k, v in profile.items() if k != 'name'}
    params.update(tree_method='hist', learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                  reg_alpha=0.2, random_state=cfg['random_seed'], n_jobs=cfg['workers'])
    params['objective'] = {'return': 'reg:squarederror', 'probability': 'binary:logistic', 'downside': 'reg:quantileerror'}[objective]
    if objective == 'downside':
        params['quantile_alpha'] = 0.25
    model = xgb.XGBRegressor(**params)
    weights = 1 / data.groupby(data.datetime.dt.normalize()).symbol.transform('size')
    weights /= weights.mean()
    model.fit(data[features], target, sample_weight=weights)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(directory / 'xgboost_model.json')
    restored = xgb.XGBRegressor()
    restored.load_model(directory / 'xgboost_model.json')
    np.testing.assert_array_equal(model.predict(data[features].iloc[:1000]), restored.predict(data[features].iloc[:1000]))
    audit.to_csv(directory / 'model_feature_manifest.csv', index=False, encoding='utf-8-sig')
    ledger = rows[['datetime', 'symbol']].copy()
    ledger['label_end'] = rows[end_key]
    ledger['target_net_return'] = rows[target_key]
    ledger['actual_target'] = target
    ledger['sample_weight'] = weights
    ledger.to_csv(directory / 'training_rows.csv', index_label='row_id', encoding='utf-8-sig')
    write_json(directory / 'model_metadata.json', {
        'fold': fold, 'features': features, 'objective': objective, 'exit_profile': exit_id,
        'model_profile': profile, 'actual_parameters': model.get_params(), 'fit_rows': len(rows),
        'last_training_feature': data.datetime.max(), 'last_training_label': rows[end_key].max(),
        'training_input_sha256': calculate_sha256(directory / 'training_rows.csv'),
        'feature_values_sha256': calculate_identity(pd.util.hash_pandas_object(data[features], index=True).tolist()),
        'model_sha256': calculate_sha256(directory / 'xgboost_model.json'), 'native_reload_verified': True,
        'label': '独立批次T+1/ATR/时间退出后的扣费净收益；盈利概率用净收益>0，分位数为25%下行分位数',
        'xgboost_version': xgb.__version__, 'uses_2026_training': False,
        'calibration': '不使用2026校准；绝对门槛跨模型重训存在预测分布漂移风险'})
    return model, features


class ExpandedCtaStrategy(BaseStrategy):
    """冻结模型驱动的只做多开仓；独立批次退出交由已审计引擎，不因目标池变化平仓。"""

    name = 'expanded_xgb_cta'

    def __init__(self, model, features: list[str], rule: dict):
        """model和features为固定原生模型/列序，rule指定目标类型、门槛、频率及退出编号。"""
        self.model, self.features, self.rule = model, list(features), dict(rule)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """对data只传固定特征列；保持索引/长度，零开仓信号不代表卖出已有批次。"""
        score = self.model.predict(data[self.features])
        if self.rule['objective'] != 'probability':
            score = score / 100
        output = data[['datetime', 'symbol']].copy()
        output['prediction'] = score
        output['open_signal'] = (calculate_entry_mask(data, self.rule['style'], self.rule['entry_minutes']) &
                                  np.isfinite(score) & (score > self.rule['gate'])).astype(int)
        output['close_signal'] = 0
        output['exit_profile'] = self.rule['exit_profile']
        return output


def generate_signals(panel: pd.DataFrame, model, features: list[str], rule: dict,
                     start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """用固定model/features/rule对panel的[start,end)推断，不接收标签或未来收益。"""
    data = panel.loc[(panel.datetime >= start) & (panel.datetime < end)]
    output = ExpandedCtaStrategy(model, features, rule).generate_signals(data)
    return output.loc[output.open_signal.eq(1), ['datetime', 'symbol', 'prediction', 'exit_profile']]


def select_robust_rule(table: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """仅从传入的历史内部table按均值减0.25倍年际标准差选规则，保留不足交易候选。"""
    keys = ['rule_id', 'objective', 'exit_profile', 'profile_name', 'style', 'entry_minutes', 'gate']
    aggregate = table.groupby(keys, dropna=False).agg(mean_sharpe=('sharpe', 'mean'),
        std_sharpe=('sharpe', 'std'), total_closed=('closed_lot_count', 'sum'),
        mean_return=('total_return', 'mean'), worst_return=('total_return', 'min')).reset_index()
    aggregate['score'] = aggregate.mean_sharpe - 0.25 * aggregate.std_sharpe.fillna(0)
    aggregate['sufficient'] = aggregate.total_closed >= cfg['validation_min_closed_lots']
    valid = aggregate.loc[aggregate.sufficient]
    if valid.empty:
        valid = aggregate.sort_values(['total_closed', 'rule_id'], ascending=[False, True]).head(1)
    chosen = valid.sort_values(['score', 'worst_return', 'rule_id'], ascending=[False, False, True]).iloc[0]
    return chosen.to_dict(), aggregate


def run_expanded_research(panel: pd.DataFrame, market, manifest: dict, cfg: dict) -> Path:
    """执行cfg预注册的内部候选→全量重训→全部冻结→年度回放/2026单次测试，返回新报告目录。"""
    root = Path(__file__).resolve().parents[1]
    directory = Path(cfg['output_dir']) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    sources = [Path(__file__), Path(__file__).with_name('cta_expanded_features.py'),
               Path(__file__).with_name('cta_research_training.py'), Path(__file__).with_name('cta_research_engine.py'),
               Path(__file__).with_name('cta_research_data.py'), Path(__file__).with_name('cta_research_reporting.py'),
               root / 'scripts/run_xgb_cta_expanded.py', Path(cfg['config_source']), Path(cfg['base_config'])]
    snapshots = {}
    for path in sources:
        destination = directory / 'source_snapshot' / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        snapshots[str(path)] = calculate_sha256(path)
    cutoff = pd.Timestamp('2026-01-01')
    folds = create_internal_folds()
    write_json(directory / 'protocol.json', {'internal_folds': folds, 'final_fit_end_exclusive': cutoff,
        'test_end': cfg['end_date'], 'sharpe_target': cfg['sharpe_target'], 'test_historically_exposed': True,
        'stop': '只遍历预注册内部候选；冻结后2026只评一次，不按测试夏普反向迭代',
        'annual_in_sample': [2023, 2024, 2025], 'annual_cash_reset': True,
        'learned_exit_implemented': False, 'sources': snapshots})
    write_json(directory / 'data_manifest.json', manifest)
    training_panel = panel.loc[panel.datetime < cutoff]
    labels, label_manifest = load_labels(training_panel, truncate_market(market, cutoff), manifest, cfg)
    write_json(directory / 'label_manifest.json', label_manifest)
    records, forward_rules = [], []
    for fold in folds:
        print(f"内部向前比较：{fold['year']}；禁止2026参与", flush=True)
        validation = panel.loc[(panel.datetime >= fold['validation_start']) & (panel.datetime < fold['validation_end'])]
        for exit_id, profile, objective in itertools.product(range(len(cfg['exit_profiles'])), cfg['model_profiles'], cfg['objectives']):
            model_name = f"exit{exit_id}_{profile['name']}_{objective}"
            model, features = fit_candidate(panel, labels, fold, exit_id, profile, objective, cfg,
                                            directory / 'internal' / str(fold['year']) / model_name)
            scores = model.predict(validation[features]) / (1 if objective == 'probability' else 100)
            gates = cfg['probability_gates'] if objective == 'probability' else cfg['net_return_gates']
            for style, frequency, gate in itertools.product(cfg['entry_styles'], cfg['entry_minutes_candidates'], gates):
                rule = dict(rule_id=f'{model_name}_{style}_{frequency}_{gate}', objective=objective,
                            exit_profile=exit_id, profile_name=profile['name'], style=style, entry_minutes=frequency, gate=gate)
                eligible = calculate_entry_mask(validation, style, frequency) & np.isfinite(scores) & (scores > gate)
                signals = validation.loc[eligible, ['datetime', 'symbol']].copy()
                signals['prediction'], signals['exit_profile'] = scores[eligible], exit_id
                result = run_portfolio(signals, market, fold['validation_start'], fold['validation_end'], cfg)
                records.append(dict(year=fold['year'], **rule, **result['metrics']))
            pd.DataFrame(records).to_csv(directory / 'validation_rule_metrics.csv', index=False, encoding='utf-8-sig')
            print(f"{fold['year']}完成{model_name}；累计{len(records)}个内部账户", flush=True)
        winner, aggregate = select_robust_rule(pd.DataFrame(records), cfg)
        aggregate.to_csv(directory / f"internal_summary_through_{fold['year']}.csv", index=False, encoding='utf-8-sig')
        # 年度向前报告的规则只允许使用该年之前已经结束的内部折。
        if fold['year'] == 2023:
            forward_rules.extend([(2023, '2023-10-01', '2024-01-01', winner.copy()),
                                  (2024, '2024-01-01', '2025-01-01', winner.copy())])
        elif fold['year'] == 2024:
            forward_rules.append((2025, '2025-01-01', '2026-01-01', winner.copy()))
    selected, aggregate = select_robust_rule(pd.DataFrame(records), cfg)
    aggregate.to_csv(directory / 'internal_candidate_summary.csv', index=False, encoding='utf-8-sig')
    write_json(directory / 'selected_rule.json', selected)
    final_fold = dict(year='final_2023_2025', train_start=pd.Timestamp('2023-01-01'), train_end=cutoff)
    profile = next(p for p in cfg['model_profiles'] if p['name'] == selected['profile_name'])
    final_model, final_features = fit_candidate(panel, labels, final_fold, int(selected['exit_profile']), profile,
                                               selected['objective'], cfg, directory / 'final_model')
    fitted_forward = []
    for year, start, end, rule in forward_rules:
        profile = next(p for p in cfg['model_profiles'] if p['name'] == rule['profile_name'])
        fold = dict(year=year, train_start=pd.Timestamp('2023-01-01'), train_end=pd.Timestamp(start))
        model, features = fit_candidate(panel, labels, fold, int(rule['exit_profile']), profile, rule['objective'], cfg,
                                        directory / 'forward_models' / str(year))
        write_json(directory / 'forward_models' / str(year) / 'selected_rule.json', rule)
        fitted_forward.append((year, pd.Timestamp(start), pd.Timestamp(end), rule, model, features))
    frozen = {str(p.relative_to(directory)): calculate_sha256(p) for p in directory.rglob('*') if p.is_file()}
    write_json(directory / 'frozen_before_test.json', {'time': datetime.now().isoformat(), 'files': frozen})
    print('全部模型和规则已冻结；开始年度报告及单次2026测试', flush=True)
    annual, continuous_signals = [], []
    for year, start, end, rule, model, features in fitted_forward:
        signals = generate_signals(panel, model, features, rule, start, end)
        continuous_signals.append(signals)
        result = run_portfolio(signals, market, start, end, cfg)
        result['audit'] = audit_result(result, market, cfg)
        save_result(directory / 'forward' / str(year), result, rule, cfg, f'扩展XGB CTA：{year}向前历史验证')
        annual.append(dict(year=year, evaluation='向前历史验证', **result['metrics']))
    for year in [2023, 2024, 2025, 2026]:
        start = pd.Timestamp(f'{year}-01-01')
        end = min(pd.Timestamp(f'{year+1}-01-01'), pd.Timestamp(cfg['end_date']) + pd.Timedelta(days=1))
        signals = generate_signals(panel, final_model, final_features, selected, start, end)
        result = run_portfolio(signals, market, start, end, cfg)
        result['audit'] = audit_result(result, market, cfg)
        description = '样本内回放，不能证明泛化' if year < 2026 else '样本外历史复测（历史已曝光）'
        out = directory / 'in_sample' / str(year) if year < 2026 else directory
        save_result(out, result, selected, cfg, f'扩展XGB CTA：{year}{description}')
        annual.append(dict(year=year, evaluation=description, **result['metrics']))
        print(f"{year}{description}：收益{result['metrics']['total_return']:.2%}，夏普{result['metrics']['sharpe']:.3f}", flush=True)
        if year == 2026:
            test = result
            continuous_signals.append(signals)
            save_pickle(directory / 'frozen_test_signals.pkl', signals)
    continuous = run_portfolio(pd.concat(continuous_signals, ignore_index=True), market, pd.Timestamp('2023-10-01'),
                               pd.Timestamp(cfg['end_date']) + pd.Timedelta(days=1), cfg)
    continuous['audit'] = audit_result(continuous, market, cfg)
    save_result(directory / 'continuous', continuous, selected, cfg, '扩展XGB CTA：按年度过去规则切换的连续账户')
    table = pd.DataFrame(annual)
    table.to_csv(directory / 'annual_metrics.csv', index=False, encoding='utf-8-sig')
    achieved = bool(not cfg['is_smoke'] and test['metrics']['annual_volatility'] > 0 and test['metrics']['sharpe'] > cfg['sharpe_target'])
    write_json(directory / 'summary.json', {'target_achieved': achieved, 'test_2026': test['metrics'],
        'continuous': continuous['metrics'], 'internal_best': selected, 'live_approved': False,
        'test_end': cfg['end_date'], 'candidate_accounts': len(records), 'feature_count': manifest['feature_count']})
    base_report = (directory / 'final_report.md').read_text()
    display = table[['year', 'evaluation', 'total_return', 'annual_return', 'sharpe', 'max_drawdown', 'buy_count', 'closed_lot_count']].rename(columns={
        'year': '年度', 'evaluation': '评价性质', 'total_return': '收益率', 'annual_return': '年化收益',
        'sharpe': '夏普', 'max_drawdown': '最大回撤', 'buy_count': '买入次数', 'closed_lot_count': '完整平仓批次'})
    text = f"# 扩展特征XGBoost CTA研究总报告\n\n2026净日收益夏普>2：{'达到数值目标' if achieved else '未达到'}。不是未来盈利承诺。\n\n"
    text += f"特征候选{manifest['feature_count']}列，内部真实账户{len(records)}组；全量模型仅拟合2023—2025。2026截至4月30日，历史已曝光。\n\n"
    text += '## 年度结果\n\n' + display.to_markdown(index=False) + '\n\n'
    text += '收益/回撤为小数，例如0.10表示10%。2023向前验证仅第四季度。向前行的权重/规则只来自过去，但历史设计已被反复研究；样本内行不能用于判断泛化。\n\n'
    text += '## 策略与研究边界\n\n上周波动前300周池、模型门槛及CTA趋势/突破/回调条件共同触发；每次最多前5，每批预算为总资产10%，单股20%、总仓80%。信号后下一分钟执行，只做多，T+1。每批独立初始ATR止损、只上移的吊顶保护线、可选跌破过去EMA20退出和最大交易分钟期限，不因离开目标池平仓。\n\n'
    text += '本轮实现净收益、盈利概率、25%分位数三类目标比较；未实现持仓状态学习退出模型，也未做概率校准。全量重训后绝对分数门槛可能漂移；不能用2026调整。全部预注册候选及失败结果保留，结果揭示后停止本轮，不以反复看测试集制造达标。\n\n'
    text += '本轮仅755只有本地历史状态的股票中曾入池的722只（主板为主，创业板/科创板各3只）；不是全A研究。\n\n' + base_report
    (directory / 'final_report.md').write_text(text, encoding='utf-8')
    with (directory / '输出文件说明.md').open('a', encoding='utf-8') as handle:
        handle.write('\n## 研究级产物\n\n根目录test文件仅表示2026。annual_metrics是年度分类结果；in_sample是最终模型训练期回放；forward是年度过去训练的历史验证；continuous为跨年持仓连续账户。internal是全部候选原生模型、训练行、特征及参数。final_model仅用2023—2025训练；forward_models保留年度权重。validation_rule_metrics保留所有内部比较；internal_summary及internal_candidate_summary是只按已完成内部折聚合的评分。selected_rule是冻结方案，protocol是边界与研究合同；frozen_before_test和source_snapshot用于复现，data_manifest/label_manifest用于追溯缓存，summary是机器可读指标。模型不准直接用于实盘。\n')
    for name, digest in frozen.items():
        # 根目录报告/文件指南由结果生成，冻结时尚不存在；已有配置、候选及模型绝不能变化。
        if calculate_sha256(directory / name) != digest:
            raise AssertionError(f'冻结文件变更：{name}')
    if any(calculate_sha256(Path(p)) != digest for p, digest in snapshots.items()):
        raise AssertionError('运行期间源码发生变更')
    write_json(directory / 'freeze_audit.json', {'passed': True, 'file_count': len(frozen)})
    return directory
