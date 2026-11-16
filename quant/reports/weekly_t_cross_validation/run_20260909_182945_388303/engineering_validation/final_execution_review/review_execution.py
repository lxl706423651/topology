"""对完成的独立执行审计作只读验收；不重跑模型、策略、账户或原始行情。"""
from pathlib import Path
import hashlib
import json
import re
import time
import urllib.parse

import numpy as np
import pandas as pd

ROOT = Path('/data/lxl/seed-emulator/topology/quant')
REPORT = ROOT/'reports/weekly_t_cross_validation/run_20260909_182945_388303'
OUTPUT = REPORT/'engineering_validation/final_execution_review'


def digest(path):
    """流式核对既有文件字节身份。"""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    """核完成范围、年度结果、费用含义、五项执行要求与冻结身份。"""
    start = time.perf_counter()
    archive = REPORT/'independent_timing_execution_audit_archive'
    audit = json.loads((REPORT/'independent_execution_audit.json').read_text())
    verification = json.loads((archive/'verification.json').read_text())
    started = json.loads((archive/'started_manifest.json').read_text())
    assert audit['passed'] and audit['completed_all_four_years']
    assert audit['completed_years'] == [2023, 2024, 2025, 2026]
    assert audit['current_decisions_independently_rebuilt']
    assert verification['passed'] and verification['terminal_exit_code'] == 0
    assert verification['full_default_scope'] and not verification['accounting_only']
    assert not verification['reader_revision_changes_strategy']
    assert verification['original_artifacts_changes'] == {}
    expected_sources = json.loads((REPORT/'source_hashes.json').read_text())
    assert len(expected_sources) == verification['current_runtime_source_count'] == 32
    assert audit['snapshot_hashes_verified'] == expected_sources
    for name, expected in expected_sources.items():
        assert digest(ROOT/name) == digest(REPORT/'source_snapshot'/Path(name).name) == expected, name
    original = started['original_artifacts_sha256']
    assert len(original) == verification['original_artifacts_checked'] == 2269
    administrative_updates = []
    for name, expected in original.items():
        actual = digest(REPORT/name)
        if name == 'engineering_validation/verification.json' and actual != expected:
            backup = REPORT/'engineering_validation/verification_before_final_audit_acceptance.json'
            assert digest(backup) == expected, '工程汇总更新的原始备份不匹配'
            administrative_updates.append({'path':name,'original_sha256':expected,'current_sha256':actual,
                'preserved_original':str(backup.relative_to(REPORT)),
                'reason':'根代理明确授权在执行审计完成后补充通过状态；非策略/模型/交易结果修改'})
        else:
            assert actual == expected, name
    assert len(administrative_updates) == 1
    verified_outputs = {}
    for name, expected in verification['outputs_sha256'].items():
        assert digest(REPORT/name) == expected, name
        verified_outputs[name] = expected
    auditor_sha = audit['auditor_source_sha256']
    assert auditor_sha == verification['auditor_sha256']
    assert digest(ROOT/'scripts/audit_weekly_t_timing_execution.py') == auditor_sha
    assert digest(archive/'audit_weekly_t_timing_execution.py') == auditor_sha
    total = audit['totals']
    assert total == verification['totals']
    assert total['completed_pairs'] == total['t_first_leg_events'] == 86
    assert total['fragments'] == total['t_recovery_events'] == 87
    for field in ['pending_pairs','inventory_mismatch_days','inventory_mismatch_stock_days',
                  'recovery_failure_days','recovery_failure_stock_days','t1_violation_count']:
        assert total[field] == 0, field
    assert total['weekly_exit_failure_days'] == 171
    assert total['weekly_exit_failure_stock_days'] == 307
    assert total['weekly_unclosed_positions'] == total['baseline_weekly_unclosed_positions'] == 7
    assert total['final_stale_position_count'] == 1 and total['final_stale_market_value_CNY'] == 9750
    model_check = audit['model_isolation_audit_source']
    assert model_check['passed'] and model_check['current_decisions_independently_rebuilt']
    assert model_check['candidate_accounts_independently_checked'] == verification['candidate_accounts'] == 100
    assert model_check['models_checked'] == 24
    candidates = []
    for year, fold in model_check['folds'].items():
        assert fold['passed']
        assert len(fold['models']) == 6 and len(fold['candidates']) == 25
        assert sorted(row['candidate_index'] for row in fold['candidates']) == list(range(25))
        assert all(row['passed'] and row['requests']['passed'] for row in fold['candidates'])
        chosen = json.loads((REPORT/f'year_{year}/selected_rule.json').read_text())
        assert chosen['candidate_index'] == fold['selected_candidate_index']
        assert chosen['enabled'] == fold['selected_enabled']
        candidates.extend(fold['candidates'])
    assert len(candidates) == 100
    annual = pd.read_csv(REPORT/'annual_summary.csv',float_precision='round_trip').set_index('test_year')
    assert len(audit['annual']) == verification['outer_accounts'] == 4
    annual_checks = []
    for row in audit['annual']:
        year = row['year']
        assert row['audit_pass'] and row['timing_request_audit']['passed']
        assert row['strict_daily_inventory_contract_met'] and row['t1_legal']
        assert row['base_plan_mismatches'] == row['pending_pairs'] == 0
        assert row['inventory_mismatch_days'] == row['recovery_failure_days'] == 0
        assert row['frozen_v4_baseline_max_abs_error_CNY'] == 0
        assert row['pending_notional_cap_and_partial_release_verified']
        assert row['per_decision_horizon_planned_recovery_and_no_early_restore_verified']
        assert row['signal_priority_and_daily_one_pair_and_middle_week_days_verified']
        assert row['completed_pairs'] == annual.loc[year,'t_pair_count']
        assert row['fragments'] == annual.loc[year,'fragment_count']
        np.testing.assert_allclose(row['T_increment_CNY'],annual.loc[year,'t_added_equity'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(row['T_fees_CNY'],annual.loc[year,'t_fee'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(row['T_slippage_CNY'],annual.loc[year,'t_slippage_cost'],rtol=0,atol=1e-8)
        decomposition = row['increment_decomposition']
        for key in ['weekly_cashflow_difference_CNY','holding_market_value_difference_CNY',
                    'unmatched_T_first_leg_cashflow_CNY','pending_T_inventory_market_value_CNY','weekly_plan_effect_CNY']:
            np.testing.assert_allclose(decomposition[key],0,rtol=0,atol=1e-8)
        np.testing.assert_allclose(decomposition['paired_T_net_pnl_CNY'],row['T_increment_CNY'],rtol=0,atol=1e-8)
        annual_checks.append({'year':year,'net_CNY':row['T_increment_CNY'],'groups':row['completed_pairs'],
            'fragments':row['fragments'],'T_fees_CNY':row['T_fees_CNY'],'T_slippage_CNY':row['T_slippage_CNY'],
            'five_T_execution_failures_zero':True,'weekly_exit_failure_days':row['weekly_exit_failure_days'],
            'weekly_unclosed_positions':row['weekly_unclosed_positions']})
    total_net = sum(row['net_CNY'] for row in annual_checks)
    np.testing.assert_allclose(total_net, audit['sum_independent_annual_T_increment_CNY'],rtol=0,atol=1e-8)
    np.testing.assert_allclose(total_net,verification['net_CNY'],rtol=0,atol=1e-8)
    np.testing.assert_allclose(total_net,-573.7200510555413,rtol=0,atol=1e-8)
    assert audit['fees_include_slippage'] is False
    assert audit['slippage_is_already_in_execution_price_and_must_not_be_subtracted_twice']
    assert verification['all_baseline_daily_v4_errors_zero']
    assert verification['max_cash_error_CNY'] <= 1e-8 and verification['max_equity_error_CNY'] <= 1e-8
    assert len(audit['raw_execution_sources']) == verification['raw_files'] == 7039
    assert sum(row['executed_minutes_checked'] for row in audit['raw_execution_sources']) == verification['raw_executed_minutes'] == 32004
    text = (REPORT/'independent_execution_audit.md').read_text()
    for fragment in ['4/4','59/59','18/19','5/5','9,750.00','未修改']:
        assert fragment in text, fragment
    assert digest(REPORT/'final_report.md') == original['final_report.md'] == '67f969ea6e2fdc8835e06c3d39655102c83b664fd918c5bf65013d1c02d56b2b'
    administration = json.loads((REPORT/'engineering_validation/verification.json').read_text())
    assert administration['independent_model_audit']['sha256'] == verified_outputs['independent_timing_model_audit.json']
    assert administration['independent_execution_audit']['sha256'] == digest(archive/'verification.json')
    assert administration['independent_execution_audit']['report_sha256'] == verified_outputs['independent_execution_audit.json']
    documents, links = {}, []
    doc_paths = [ROOT/'README.md',ROOT/'5method_weekly_t/README.md',ROOT/'5method_weekly_t/做T择时能力改进说明.md']
    for index,path in enumerate(doc_paths):
        content = path.read_text()
        documents[str(path.relative_to(ROOT))] = digest(path)
        (OUTPUT/f'final_document_{index:02d}.md').write_text(content)
        scanned = re.sub(r'```.*?```','',content,flags=re.S)
        for match in re.finditer(r'\[[^\]]*\]\(([^\n)]+)\)',scanned):
            target = match.group(1).strip().strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):
                continue
            target = urllib.parse.unquote(target.split('#',1)[0])
            destination = Path(target) if target.startswith('/') else path.parent/target
            assert destination.exists(), (path,target)
            links.append({'document':str(path.relative_to(ROOT)),'target':target,'exists':True})
        scope = (content.split('## 做 T 择时能力研究（v11）')[-1] if path == ROOT/'README.md' else
                 content.split('主动版本见')[0] if path == ROOT/'5method_weekly_t/README.md' else content)
        assert not re.search(r'审计[^。\n]{0,25}(?:正在进行|完成中|尚待|待完成|另行补充|待最终)',scope), path
        assert 'independent_execution_audit.md' in scope and 'independent_timing_model_audit.md' in scope
        assert '通过' in scope
    result = {'passed':True,'scope':'只读验收完成的执行审计与保存报告；无模型或账户重跑',
        'script_sha256':digest(__file__),'verification_sha256':digest(archive/'verification.json'),
        'auditor_sha256':auditor_sha,'verified_outputs_sha256':verified_outputs,
        'candidate_accounts':100,'outer_accounts':4,'completed_pairs':86,'fragments':87,
        'net_CNY':total_net,'annual':annual_checks,'five_outer_T_execution_failures_zero':True,
        'source_and_snapshot_files_unchanged':32,'original_artifact_hashes_rechecked':2269,
        'original_artifacts_current_bytes_unchanged':2268,'administrative_summary_updates':administrative_updates,
        'raw_trade_model_and_result_artifacts_unchanged':True,
        'period_semantics':'执行审计起止期间2269份原产物均未变；本次后验验收时仅1份工程状态汇总获授权更新，其原始字节已备份。',
        'latest_documents_sha256':documents,'latest_document_local_links':links,
        'latest_v11_document_sections_have_no_pending_audit_status':True,
        'final_report_sha256':original['final_report.md'],
        'raw_audit_coverage_verified_from_completed_evidence':{'files':7039,'executed_minutes':32004},
        'max_cash_error_CNY':verification['max_cash_error_CNY'],'max_equity_error_CNY':verification['max_equity_error_CNY'],
        'weekly_exit_failures_remain':{'days':171,'stock_days':307,'final_positions':7,'stale_value_CNY':9750},
        'limitations':'审计证明实现、来源和账目一致，不代表盈利或全新未接触测试；原周退出失败仍保留。',
        'elapsed_seconds':time.perf_counter()-start}
    (OUTPUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUTPUT/'review.md').write_text('# v11 完成执行审计只读验收\n\n通过：100个内部候选、4个外层账户、86完整组/87恢复片段，净增量−573.7200510555413元。\n\n五项T执行失败均为0；32份源及快照保持不变。2269份启动清单中2268份仍原字节，另1份工程状态汇总在审计完成后获授权更新，旧备份SHA精确匹配；原交易、模型与结果未改。原周退出失败171日/307股日、期末7笔持仓及9750元陈旧估值仍保留。\n\n最新三份文档的本地链接有效，v11段落不再残留审计待完成状态。只核已完成证据和文件身份，没有重跑模型或账户；通过审计不等于通过盈利目标。\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
