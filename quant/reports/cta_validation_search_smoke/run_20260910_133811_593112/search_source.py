"""只优化固定内部验证集的CTA规则，保留每次试验，不读取2026预测或收益。

输入为已审计的过去训练模型/特征及YAML；输出可复用验证预测PKL和逐候选账务指标。
复用同一个Numba撮合内核，快速评价只省去重复构造中文账本，不简化费用或成交约束。
"""

from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from .annual_panel import write_json
from .cta_research_data import calculate_identity, calculate_sha256, save_pickle, MinuteMarket
from .cta_research_engine import create_parameters, simulate_account, run_portfolio
from .cta_research_training import calculate_entry_mask
from .cta_research_reporting import audit_result, save_result
from .cta_expanded_training import truncate_market
from .strategy import BaseStrategy


CONTEXT_COLUMNS = ['ctx_momentum5', 'ctx_momentum20', 'ctx_ema5_gap', 'ctx_ema20_gap',
    'ctx_breakout10', 'ctx_atr_ratio', 'ctx_daily_return', 'ctx_pool_breadth', 'ctx_pool_return', 'ctx_pool_momentum20']


def read_protocol(source: Path) -> list[dict]:
    """从source旧协议读取固定三折并检查时间隔离；不按验证表现移动或删减区间。"""
    folds = json.loads((source / 'protocol.json').read_text())['internal_folds']
    if [int(f['year']) for f in folds] != [2023, 2024, 2025]:
        raise ValueError('必须保留2023/2024/2025三段原验证区间')
    for f in folds:
        for key in ['train_start', 'train_end', 'validation_start', 'validation_end']:
            f[key] = pd.Timestamp(f[key])
        if not (f['train_start'] < f['train_end'] <= f['validation_start'] < f['validation_end'] <= pd.Timestamp('2026-01-01')):
            raise ValueError('验证模型时间边界非法')
    return folds


def create_model_names(cfg: dict) -> list[str]:
    """按cfg显式枚举模型来源，不接收最终全量模型或2026模型。"""
    return [f'exit{e}_{p}_{o}' for e, p, o in itertools.product(cfg['exit_ids'], cfg['profiles'], cfg['objectives'])]


def load_validation_predictions(cfg: dict) -> tuple[dict, dict, list[dict]]:
    """仅加载cfg源报告各折过去训练模型，构建/复用验证预测PKL；不读取旧测试预测。"""
    source = Path(cfg['source_report'])
    folds = read_protocol(source)
    manifest = json.loads((source / 'data_manifest.json').read_text())
    names = create_model_names(cfg)
    model_paths = [source/'internal'/str(f['year'])/name for f in folds for name in names]
    identity = {'schema': 1, 'panel_sha256': manifest['panel_sha256'], 'folds': folds,
        'context_columns': CONTEXT_COLUMNS, 'xgboost': xgb.__version__, 'models': {str(p): {
            'model': calculate_sha256(p/'xgboost_model.json'), 'metadata': calculate_sha256(p/'model_metadata.json')}
            for p in model_paths}}
    cache = Path(cfg['prediction_cache'])/calculate_identity(identity)
    complete = cache/'complete.json'
    if complete.exists():
        meta = json.loads(complete.read_text())
        for name, digest in meta['files'].items():
            if calculate_sha256(cache/name) != digest:
                raise AssertionError('验证预测缓存指纹错误')
        print(f'复用验证预测缓存：{cache}', flush=True)
        return {f['year']: pd.read_pickle(cache/f"validation_{f['year']}.pkl") for f in folds}, meta, folds
    panel_path = Path(manifest['cache_path'])/'panel.pkl'
    if calculate_sha256(panel_path) != manifest['panel_sha256']:
        raise AssertionError('来源特征面板被更改')
    panel = pd.read_pickle(panel_path)
    panel = panel.loc[panel.datetime < pd.Timestamp('2026-01-01')]
    data, files = {}, {}
    for fold in folds:
        current = panel.loc[(panel.datetime >= fold['validation_start']) & (panel.datetime < fold['validation_end'])]
        output = current[['datetime', 'symbol'] + CONTEXT_COLUMNS].copy().reset_index(drop=True)
        for name in names:
            path = source/'internal'/str(fold['year'])/name
            meta = json.loads((path/'model_metadata.json').read_text())
            if (pd.Timestamp(meta['last_training_feature']) >= fold['validation_start'] or
                    pd.Timestamp(meta['last_training_label']) >= fold['validation_start'] or
                    pd.Timestamp(meta['fold']['train_end']) != fold['train_end']):
                raise AssertionError('错误引用了验证期间或未来拟合模型')
            if calculate_sha256(path/'xgboost_model.json') != meta['model_sha256']:
                raise AssertionError('原生模型指纹改变')
            model = xgb.XGBRegressor(n_jobs=cfg['workers'])
            model.load_model(path/'xgboost_model.json')
            values = model.predict(current[meta['features']])
            output[f'score_{name}'] = values / (1 if meta['objective']=='probability' else 100)
            print(f"验证预测 {fold['year']}：{name}，{len(current):,}行", flush=True)
        filename = f"validation_{fold['year']}.pkl"
        save_pickle(cache/filename, output)
        files[filename] = calculate_sha256(cache/filename)
        data[fold['year']] = output
    meta = {'identity': identity, 'files': files, 'cache_path': str(cache), 'data_manifest': manifest}
    write_json(complete, meta)
    return data, meta, folds


def calculate_rule_mask(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    """对frame当时可知上下文施加rule趋势/恢复/市场风险条件；不使用未来收益。"""
    style, frequency = rule['style'], int(rule['frequency'])
    if style in ['trend', 'breakout', 'pullback']:
        mask = calculate_entry_mask(frame, style, frequency)
    else:
        minute = frame.datetime.dt.hour.to_numpy()*60 + frame.datetime.dt.minute.to_numpy()
        session = np.where(minute <= 690, minute-571, minute-781+120)
        mask = (session % frequency == 0) & (session < 236)
        if style == 'recovery':
            mask &= (frame.ctx_momentum20 > 0) & frame.ctx_ema20_gap.between(-0.05, 0.03)
            mask &= (frame.ctx_momentum5 < 0) & (frame.ctx_daily_return > 0)
        elif style != 'model_only':
            raise ValueError(f'未知开仓类型：{style}')
    context = rule['context']
    if context == 'breadth':
        mask &= (frame.ctx_pool_breadth > 0.55) & (frame.ctx_pool_return > 0)
    elif context == 'market_trend':
        mask &= (frame.ctx_pool_momentum20 > 0) & (frame.ctx_pool_return > 0)
    elif context == 'quiet':
        mask &= frame.ctx_atr_ratio.between(0.005, 0.035)
    elif context == 'avoid_chase':
        mask &= frame.ctx_daily_return.between(-0.025, 0.025) & (frame.ctx_ema5_gap < 0.03)
    elif context != 'none':
        raise ValueError(f'未知环境条件：{context}')
    score = frame[f"score_{rule['model_name']}"].to_numpy()
    return np.asarray(mask) & np.isfinite(score) & (score > rule['gate'])


class ValidationCtaStrategy(BaseStrategy):
    """沪深非ST候选池只做多开仓；rule固定前5排序前的条件，批次退出仍由原引擎负责。"""
    name = 'validation_cta_search'

    def __init__(self, rule: dict):
        """rule指定过去模型分数列、门槛、检查频率、环境过滤和独立退出编号。"""
        self.rule = dict(rule)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """从data已缓存的过去模型预测生成对齐信号，不改变数据、不读标签、不因空信号强制平仓。"""
        out = data[['datetime', 'symbol']].copy()
        out['prediction'] = data[f"score_{self.rule['model_name']}"]
        out['open_signal'] = calculate_rule_mask(data, self.rule).astype(int)
        out['close_signal'] = 0
        out['exit_profile'] = self.rule['exit_id']
        return out


def build_signals(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """把frame及rule转换为原执行入口四列合同，保留真实分数排序。"""
    out = ValidationCtaStrategy(rule).generate_signals(frame)
    return out.loc[out.open_signal.eq(1), ['datetime', 'symbol', 'prediction', 'exit_profile']]


def run_fast_metrics(frame: pd.DataFrame, rule: dict, market, fold: dict, cfg: dict) -> dict:
    """复用原simulate_account评价frame/rule；fold限定区间、cfg限定仓位费用，日夏普不删现金日。"""
    mask = calculate_rule_mask(frame, rule)
    time_ids = market.times.get_indexer(frame.loc[mask, 'datetime'])
    symbols = frame.loc[mask, 'symbol']
    scores = frame.loc[mask, f"score_{rule['model_name']}"].to_numpy()
    assets = symbols.map(market.symbol_index).to_numpy(dtype=np.int64)
    if (time_ids < 0).any():
        raise ValueError('信号日期不在报价时钟内')
    order = np.lexsort((symbols.to_numpy(dtype=str), -scores, time_ids))
    time_ids, assets, scores = time_ids[order], assets[order], scores[order]
    positions = np.arange(len(order))
    starts = np.r_[True, time_ids[1:] != time_ids[:-1]] if len(order) else np.array([], dtype=bool)
    ranks = positions - np.maximum.accumulate(np.where(starts, positions, 0))
    keep = ranks < cfg['entry_top_n']
    shape = (len(market.times), cfg['entry_top_n'])
    candidates = np.full(shape, -1, dtype=np.int64)
    predictions = np.zeros(shape)
    profile_ids = np.zeros(shape, dtype=np.int64)
    candidates[time_ids[keep], ranks[keep]] = assets[keep]
    predictions[time_ids[keep], ranks[keep]] = scores[keep]
    profile_ids[time_ids[keep], ranks[keep]] = rule['exit_id']
    parameters, profiles, stamps = create_parameters(cfg, market.times)
    start, end = market.times.searchsorted(fold['validation_start']), market.times.searchsorted(fold['validation_end'])
    curve, events, trips, blocks, lots, marks = simulate_account(market.quotes, market.flags, market.boards,
        candidates, predictions, profile_ids, profiles, parameters, stamps, start, end)
    day_ends = np.r_[np.flatnonzero((curve[:-1, 0].astype(int)+1) % 240 == 0), len(curve)-1]
    equity = curve[day_ends, 1]
    returns = np.r_[equity[0]/cfg['initial_cash']-1, equity[1:]/equity[:-1]-1]
    std = float(returns.std(ddof=1)) if len(returns)>1 else 0.0
    open_ids = lots[lots[:, 1]>0, 13]
    closed_ids = np.setdiff1d(np.unique(trips[:, 0]), open_ids)
    return {'sharpe': float(returns.mean()/std*np.sqrt(252)) if std else 0.0,
        'total_return': float(equity[-1]/cfg['initial_cash']-1), 'annual_volatility': std*np.sqrt(252),
        'max_drawdown': float(np.min(equity/np.maximum.accumulate(np.r_[cfg['initial_cash'], equity])[1:]-1)),
        'closed_lot_count': len(closed_ids), 'buy_count': int((events[:, 4]==1).sum()),
        'sell_count': int((events[:, 4]==-1).sum()), 'open_lots': len(open_ids), 'signal_count': int(keep.sum()),
        'mean_exposure': float(curve[:, 3].mean()), 'fees': float(events[:, 8].sum()), 'trading_days': len(equity)}


def create_rules(cfg: dict) -> list[dict]:
    """展开cfg预先固定的共同规则；规则不含年份或按事后挑选的股票名单。"""
    rules = []
    for name in create_model_names(cfg):
        gates = cfg['probability_gates'] if name.endswith('probability') else cfg['return_gates']
        for style, frequency, context, gate in itertools.product(cfg['styles'], cfg['frequencies'], cfg['contexts'], gates):
            rule = dict(model_name=name, exit_id=int(name.split('_')[0][4:]), style=style,
                        frequency=frequency, context=context, gate=gate)
            rule['rule_id'] = calculate_identity(rule)
            rules.append(rule)
    return rules


def rank_common_rules(rows: pd.DataFrame, folds: list[dict], cfg: dict) -> pd.DataFrame:
    """按rows的最差折夏普排序；folds必须齐全，cfg目标必须每折严格超过，不能用均值冒充。"""
    required = {f['year'] for f in folds}
    summary = []
    for key, group in rows.groupby('rule_id'):
        if set(group.year) != required or len(group) != len(required):
            continue
        valid = group.annual_volatility.gt(0).all() and group.closed_lot_count.ge(cfg['minimum_closed_lots_per_fold']).all()
        item = dict(rule_id=key, minimum_sharpe=group.sharpe.min(), mean_sharpe=group.sharpe.mean(),
                    minimum_closed_lots=int(group.closed_lot_count.min()), sufficient=bool(valid),
                    all_folds_pass=bool(valid and group.sharpe.gt(cfg['sharpe_target']).all()))
        for row in group.itertuples():
            item[f'sharpe_{row.year}'], item[f'return_{row.year}'] = row.sharpe, row.total_return
        summary.append(item)
    if not summary:
        raise ValueError('尚无完成全部验证区间的候选')
    return pd.DataFrame(summary).sort_values(['all_folds_pass', 'sufficient', 'minimum_sharpe', 'mean_sharpe', 'rule_id'],
        ascending=[False, False, False, False, True]).reset_index(drop=True)


def run_search(data: dict, cache: dict, folds: list[dict], cfg: dict) -> Path:
    """只在固定folds优化cfg规则；data/cache限定验证预测，结果与原测试报告完全隔离。"""
    source = Path(cfg['source_report'])
    execution = yaml.safe_load((source/'resolved_config.yaml').read_text())
    execution['is_smoke'] = cfg['is_smoke']
    market = truncate_market(MinuteMarket(Path(execution['source_cache'])/'minute_market'), pd.Timestamp('2026-01-01'))
    rules = create_rules(cfg)
    directory = Path(cfg['output_dir'])/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    identity = {'config': cfg, 'rules': rules, 'folds': folds, 'cache': cache['identity'],
        'engine_sha256': calculate_sha256(Path(__file__).with_name('cta_research_engine.py')),
        'search_sha256': calculate_sha256(Path(__file__)), 'evaluation': '反复优化过的内部验证，不是独立测试',
        'target': '所有固定验证集净日收益夏普严格>2，不能删年份或用均值替代', 'test_2026_evaluated': False}
    write_json(directory/'protocol.json', identity)
    write_json(directory/'prediction_cache_manifest.json', cache)
    (directory/'resolved_config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    shutil.copy2(Path(__file__), directory/'search_source.py')
    rows, parity = [], []
    # 正式评价前逐折对照一次原公开回测入口，确保加速没有改变结果口径。
    reference = dict(model_name='exit0_regularized_probability', exit_id=0, style='trend', frequency=120, context='none', gate=0.5)
    if reference['model_name'] not in create_model_names(cfg):
        reference = rules[0]
    for fold in folds:
        frame = data[fold['year']]
        fast = run_fast_metrics(frame, reference, market, fold, execution)
        full = run_portfolio(build_signals(frame, reference), market, fold['validation_start'], fold['validation_end'], execution)
        for key, value in fast.items():
            if not np.isclose(value, full['metrics'][key], rtol=1e-10, atol=1e-10):
                raise AssertionError(f'加速入口改变{key}：{value} != {full["metrics"][key]}')
        parity.append({'year': fold['year'], 'passed': True, 'metrics': fast})
    write_json(directory/'fast_engine_parity.json', parity)
    for i, rule in enumerate(rules):
        for fold in folds:
            metrics = run_fast_metrics(data[fold['year']], rule, market, fold, execution)
            rows.append(dict(year=fold['year'], **rule, **metrics))
        if (i+1) % cfg['save_every']==0 or i+1==len(rules):
            table = pd.DataFrame(rows)
            table.to_csv(directory/'validation_rule_metrics.csv', index=False, encoding='utf-8-sig')
            ranked = rank_common_rules(table, folds, cfg)
            ranked.to_csv(directory/'candidate_summary.csv', index=False, encoding='utf-8-sig')
            print(f"共同规则 {i+1}/{len(rules)}，当前最佳最差折夏普{ranked.iloc[0].minimum_sharpe:.4f}，全部达标{int(ranked.all_folds_pass.sum())}套", flush=True)
    ranked = rank_common_rules(pd.DataFrame(rows), folds, cfg)
    winner = next(r for r in rules if r['rule_id']==ranked.iloc[0].rule_id)
    write_json(directory/'selected_rule.json', winner)
    accounts = []
    for fold in folds:
        result = run_portfolio(build_signals(data[fold['year']], winner), market, fold['validation_start'], fold['validation_end'], execution)
        result['audit'] = audit_result(result, market, execution)
        expected = next(row for row in rows if row['rule_id']==winner['rule_id'] and row['year']==fold['year'])
        for key in ['sharpe', 'total_return', 'max_drawdown', 'closed_lot_count', 'buy_count', 'fees']:
            if not np.isclose(expected[key], result['metrics'][key], rtol=1e-10, atol=1e-10):
                raise AssertionError('最优候选标准回放与搜索指标不一致')
        save_result(directory/f"validation_{fold['year']}", result, winner, execution, f"CTA共同规则：{fold['year']}内部选参验证，非测试")
        accounts.append(dict(year=fold['year'], **result['metrics']))
    table = pd.DataFrame(accounts)
    table.to_csv(directory/'selected_validation_metrics.csv', index=False, encoding='utf-8-sig')
    achieved = bool(not cfg['is_smoke'] and ranked.iloc[0].all_folds_pass)
    write_json(directory/'summary.json', {'all_validation_folds_pass': achieved, 'selected': winner,
        'minimum_sharpe': ranked.iloc[0].minimum_sharpe, 'candidate_count': len(rules), 'account_count': len(rows),
        'test_2026_evaluated': False, 'independent_generalization_proven': False, 'goal_remains_active': not achieved})
    display = table[['year', 'total_return', 'sharpe', 'max_drawdown', 'closed_lot_count', 'buy_count']].rename(columns={
        'year':'验证年份', 'total_return':'净收益小数', 'sharpe':'净日夏普', 'max_drawdown':'最大回撤小数', 'closed_lot_count':'完整平仓批次', 'buy_count':'买入次数'})
    text = '# CTA逐验证集迭代报告\n\n' + ('全部验证集达到数值门槛，仍非独立测试证明。' if achieved else '尚未使各个验证集夏普均大于2，目标未完成，后续继续迭代。')
    text += f'\n\n本阶段共同规则{len(rules)}套，真实账户{len(rows)}个。禁止按年份选择不同的事后赢家，最低夏普为{ranked.iloc[0].minimum_sharpe:.4f}。\n\n'
    text += display.to_markdown(index=False)+'\n\n2023验证固定为第三季度，2024/2025为全年，没有移动原区间；夏普包含现金日、成本、T+1与实际成交限制。\n\n'
    text += '## 输入与局限\n\n复用原755只有历史状态的股票中722只曾入周池股票，创业板/科创板各3只；没有把未知ST当作非ST，不声称覆盖完整市场。\n'
    text += '所有模型只在对应验证期之前训练；这几个验证区间正在被反复用于研究选择，所以即使全部超过2，也不能宣称未来会有相同收益。2026没有重新预测或回测。原退出、T+1、费用及每批总资产10%保留。\n\n'
    for fold in folds:
        text += f"## {fold['year']}验证曲线\n\n![验证曲线](validation_{fold['year']}/测试集收益曲线.png)\n\n"
    (directory/'final_report.md').write_text(text, encoding='utf-8')
    (directory/'输出文件说明.md').write_text('# 输出文件说明\n\n架构：同一规则用于三段固定内部验证，每段从各自之前训练的模型推断。\n\n'
        '- final_report.md：生成的中文总报告与三个验证收益图。\n- validation_rule_metrics.csv：所有候选逐折真实账户指标。\n'
        '- candidate_summary.csv：最低折夏普、各年夏普、全部达标标记，不删除失败候选。\n- selected_validation_metrics.csv：所选共同规则的标准入口回放结果。\n'
        '- validation_年份/：该验证区间曲线、逐笔成交、部分平仓、期末持仓、拒单及独立账户审计；兼容test_文件名仅代表内部验证，不是2026。\n'
        '- protocol.json/resolved_config.yaml：本阶段研究配置、候选全集、固定区间与模型指纹。\n- prediction_cache_manifest.json：复用验证预测的内容指纹与数据来源。\n'
        '- selected_rule.json：同一共同策略，不能按年份事后替换。\n- fast_engine_parity.json：加速与原完整入口的逐折一致性证据。\n'
        '- summary.json：目标是否全部达到；未达到时目标保持活动。\n- search_source.py：本轮研究源码快照，不是独立可执行入口。\n', encoding='utf-8')
    if calculate_sha256(Path(__file__)) != identity['search_sha256'] or calculate_sha256(Path(__file__).with_name('cta_research_engine.py')) != identity['engine_sha256']:
        raise AssertionError('研究运行期间代码发生改变')
    return directory
