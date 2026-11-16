"""只读复核 v11 保存汇总、交易质量说明、链接与冻结身份，不重跑账户或模型。"""
from pathlib import Path
import hashlib
import json
import re
import urllib.parse

import numpy as np
import pandas as pd

ROOT = Path('/data/lxl/seed-emulator/topology/quant')
REPORT = ROOT/'reports/weekly_t_cross_validation/run_20260909_182945_388303'
DEST = REPORT/'engineering_validation/document_review'


def digest(path):
    """流式计算既有证据身份。"""
    signature = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            signature.update(chunk)
    return signature.hexdigest()


def close(actual, expected):
    """只容纳聚合求和的浮点误差，不放宽交易或模型门槛。"""
    assert np.isclose(actual, expected, rtol=0, atol=1e-7), (actual, expected)


def main():
    """由小型既有账本复核数字，保留文档快照及待修文字问题。"""
    result = {'scope': '只读既有账本/文档/哈希；无原始行情重放、模型预测或重训',
              'script_sha256': digest(__file__), 'document_sources': {}, 'issues': []}
    quality = REPORT/'timing_quality_diagnosis'
    diagnosis = json.loads((quality/'diagnosis.json').read_text())
    annual = pd.read_csv(REPORT/'annual_summary.csv', float_precision='round_trip').set_index('test_year')
    pairs = pd.read_csv(quality/'actual_pairs.csv', float_precision='round_trip')
    totals = {}
    annual_checks = []
    for dataset, directory in [('v11', REPORT), ('v10', Path(diagnosis['protocol']['reference_frequency_dir']))]:
        blocks = []
        for year in (2023, 2024, 2025, 2026):
            original = pd.read_csv(directory/f'year_{year}/test_round_trips.csv', float_precision='round_trip')
            original['fees'] = original.buy_fee+original.sell_fee
            original['reference_gross_pnl'] = original.gross_pnl+original.slippage_cost
            grouped = original.groupby('pair_id').agg(net_pnl=('net_pnl','sum'),
                reference_gross_pnl=('reference_gross_pnl','sum'), fees=('fees','sum'),
                slippage_cost=('slippage_cost','sum'), fragments=('pair_id','size'),
                complete=('pair_fully_recovered','any'))
            assert grouped.complete.all()
            group = pairs.loc[pairs.dataset.eq(dataset)&pairs.test_year.eq(year)].set_index('pair_id')
            assert set(group.index) == set(grouped.index)
            for key in ['net_pnl','reference_gross_pnl','fees','slippage_cost']:
                np.testing.assert_allclose(grouped[key].sort_index(), group[key].sort_index(), rtol=0, atol=1e-7)
            close(grouped.fragments.sum(), group.fragment_count.sum())
            record = {'dataset': dataset, 'test_year': year, 'groups': len(grouped), 'fragments': len(original),
                'wins': int(grouped.net_pnl.gt(0).sum()), 'net_win_rate': float(grouped.net_pnl.gt(0).mean()),
                'net_pnl': float(grouped.net_pnl.sum()), 'mean_net_per_pair': float(grouped.net_pnl.mean()),
                'reference_gross_pnl': float(grouped.reference_gross_pnl.sum()),
                'fees': float(grouped.fees.sum()), 'slippage_cost': float(grouped.slippage_cost.sum())}
            if dataset == 'v11':
                close(record['net_pnl'], annual.loc[year,'t_added_equity'])
                close(record['groups'], annual.loc[year,'t_pair_count'])
                close(record['fragments'], annual.loc[year,'fragment_count'])
                close(record['fees'], annual.loc[year,'t_fee'])
                close(record['slippage_cost'], annual.loc[year,'t_slippage_cost'])
            annual_checks.append(record)
            blocks.append(grouped)
        combined = pd.concat(blocks)
        summary = {'complete_pairs': len(combined), 'recovery_fragments': int(combined.fragments.sum()),
            'net_wins': int(combined.net_pnl.gt(0).sum()), 'net_win_rate': float(combined.net_pnl.gt(0).mean()),
            'net_pnl': float(combined.net_pnl.sum()), 'mean_net_per_pair': float(combined.net_pnl.mean()),
            'reference_gross_pnl': float(combined.reference_gross_pnl.sum()),
            'fees': float(combined.fees.sum()), 'slippage_cost': float(combined.slippage_cost.sum())}
        for key, value in summary.items():
            close(value, diagnosis['totals'][dataset][key])
        close(summary['reference_gross_pnl']-summary['slippage_cost']-summary['fees'], summary['net_pnl'])
        totals[dataset] = summary
    assert totals['v11']['complete_pairs'] == 86 and totals['v11']['recovery_fragments'] == 87
    result.update(numerical_tables_passed=True, totals=totals, annual=annual_checks,
                  loss_per_pair_worse_despite_smaller_total_loss=totals['v11']['mean_net_per_pair'] < totals['v10']['mean_net_per_pair'])
    current = pairs.loc[pairs.dataset.eq('v11')]
    result['descriptive_groups'] = {
        'horizon': current.groupby('horizon_minutes').net_pnl.agg(['size','sum']).reset_index().to_dict('records'),
        'direction': current.groupby('direction').net_pnl.agg(['size','sum']).reset_index().to_dict('records'),
        'mean_predicted_net_edge_bps': float(current.predicted_net_edge_bps.mean()),
        'mean_actual_net_bps': float(current.realized_net_bps.mean()),
        'probability_06_07': {'pairs': int(current.profit_event_probability.between(.6,.7,inclusive='left').sum()),
            'wins': int((current.profit_event_probability.between(.6,.7,inclusive='left')&current.net_pnl.gt(0)).sum())},
    }
    comparisons = []
    reference = Path(diagnosis['protocol']['reference_frequency_dir'])
    for year in (2023, 2024, 2025, 2026):
        a = pd.read_csv(REPORT/f'year_{year}/test_equity_curve.csv', usecols=['datetime','baseline_equity'], float_precision='round_trip')
        b = pd.read_csv(reference/f'year_{year}/test_equity_curve.csv', usecols=['datetime','baseline_equity'], float_precision='round_trip')
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        comparisons.append({'test_year': year, 'identical_saved_baseline_dates_values': True})
    result['baseline_comparisons'] = comparisons
    for path, expected in diagnosis['inputs_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in diagnosis['outputs_sha256'].items():
        assert digest(quality/name) == expected, name
    result['quality_diagnosis_hashes_unchanged'] = True
    sources = json.loads((REPORT/'source_hashes.json').read_text())
    for name, expected in sources.items():
        assert digest(ROOT/name) == digest(REPORT/'source_snapshot'/Path(name).name) == expected
    model_audit = json.loads((REPORT/'independent_timing_model_audit.json').read_text())
    model_metadata = {}
    for path in REPORT.glob('year_*/inner_t_*/model_metadata.json'):
        expected = model_audit['inputs'][str(path.resolve())]['sha256']
        assert digest(path) == expected
        model_metadata[str(path.relative_to(REPORT))] = expected
    assert len(model_metadata) == 24
    started = json.loads((REPORT/'independent_timing_execution_audit_archive/started_manifest.json').read_text())
    final_sha = digest(REPORT/'final_report.md')
    assert final_sha == started['original_artifacts_sha256']['final_report.md']
    result['frozen_identity'] = {'runtime_sources_unchanged': len(sources), 'model_metadata_unchanged': model_metadata,
        'final_report_unchanged': True, 'final_report_sha256': final_sha,
        'model_audit_sha256': digest(REPORT/'independent_timing_model_audit.json')}
    files = [ROOT/'README.md', ROOT/'5method_weekly_t/README.md', ROOT/'5method_weekly_t/做T择时能力改进说明.md',
             REPORT/'final_report.md', quality/'report.md']
    links = []
    for index, path in enumerate(files):
        text = path.read_text()
        result['document_sources'][str(path.relative_to(ROOT))] = digest(path)
        (DEST/f'document_{index:02d}.md').write_text(text)
        # 本任务核文件存在；网页和纯页内锚点不属于本地文件链接。
        scanned = re.sub(r'```.*?```','',text,flags=re.S)
        for match in re.finditer(r'\[[^\]]*\]\(([^\n)]+)\)',scanned):
            target = match.group(1).strip().strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):
                continue
            target = urllib.parse.unquote(target.split('#',1)[0])
            resolved = Path(target) if target.startswith('/') else path.parent/target
            links.append({'document': str(path.relative_to(ROOT)), 'target': target, 'exists': resolved.exists()})
    result['local_links'] = links
    result['all_local_links_exist'] = all(row['exists'] for row in links)
    assert result['all_local_links_exist']
    method = files[2].read_text()
    if '完整模型及原始行情独立审计正在进行' in method or '正式模型及原始行情独立审计尚待最终归档' in method:
        result['issues'].append({'kind':'stale_model_audit_status', 'file':'5method_weekly_t/做T择时能力改进说明.md',
            'detail':'模型审计已完成，应与仍在进行的执行审计分开表述。'})
    readme = files[0].read_text()
    last = readme.split('新增[做T择时能力研究（v11）]')[-1]
    if '独立执行审计' in last and '本轮T库存' in last and 'v10本轮' not in last:
        result['issues'].append({'kind':'ambiguous_readme_audit_version', 'file':'README.md',
            'detail':'v11段后本轮独立执行审计链接为v10，应明确标注v10，避免误示v11已通过。'})
    result['wording_avoids_false_timing_improvement'] = all(
        phrase in document for phrase, document in [
            ('择时改善未通过验证', readme), ('本轮未验证择时改善', files[1].read_text()),
            ('本轮没有通过择时能力改善验证', method), ('不是对同一交易机会随机分组后的因果证明', method)])
    assert result['wording_avoids_false_timing_improvement']
    result['awaited_independent_evidence'] = {
        'execution_audit':'只在独立执行审计 complete/passed/default全范围后引用通过；当前未要求完成。',
        'endpoint_diagnosis':'在端点诊断完成后另引其固定标签与实际恢复差异；不可将其等同盈利证据。'}
    result['passed_except_pending_wording_updates'] = True
    result['passed'] = not result['issues']
    (DEST/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'numerical_tables_passed':True,'local_links':len(links),'frozen_sources':len(sources),
        'metadata_files':len(model_metadata),'final_report_sha256':final_sha,'issues':result['issues']},ensure_ascii=False))


if __name__ == '__main__':
    main()
