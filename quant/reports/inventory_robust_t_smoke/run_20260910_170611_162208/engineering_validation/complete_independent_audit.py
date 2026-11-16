"""仅更新辅助工程归档；冻结运行源、模型、账户和独审原始证据均只读。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ROOT = Path('/data/lxl/seed-emulator/topology/quant')
SUCCESS = REPORT/'independent_robust_audit/run_20260910_172441_003843'
EXPECTED = '74f78f6012caa5765cb03a1525222c29caa94da41f3080dd9ef14001b7297f26'

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    proof=json.loads((SUCCESS/'verification.json').read_text())
    assert sha(SUCCESS/'verification.json')==EXPECTED
    assert all(proof[k] is True for k in ('passed','complete','all_read_inputs_unchanged','audit_dependencies_unchanged'))
    original=json.loads((HERE/'manifest.json').read_text())
    assert all(sha(HERE/name)==value for name,value in original.items())
    runtime=json.loads((REPORT/'source_hashes.json').read_text())
    frozen={str(ROOT/name):value for name,value in runtime.items()}
    frozen.update(proof['audit_dependencies_sha256'])
    assert len(runtime)==56 and len(proof['audit_dependencies_sha256'])==11
    assert all(sha(name)==value for name,value in frozen.items())
    out=HERE/f'audit_completion_{datetime.now():%Y%m%d_%H%M%S_%f}'
    previous=out/'previous';previous.mkdir(parents=True,exist_ok=False)
    for name in ('validation_summary.json','README.md','manifest.json'):shutil.copy2(HERE/name,previous/name)
    attempts=[]
    descriptions=[
        ('run_20260910_171807_816821',143,'peer_review_terminated',
         ['未完成组的marked账户增量不能充当完整组成本降低率分子。',
          'hold的来源L2档位须为JSON null/CSV空值，不能改成字符串hold。',
          '同轮只读复核另发现旧account未返回T滑点；新子类须从逐腿价格差独立补值。'],
         ['test_pending_marked_pnl_is_not_complete_profit_ratio','test_hold_report_source_profile_is_null',
          'test_account_adds_independent_t_slippage_not_base_slippage']),
        ('run_20260910_172125_408966',1,'auditor_exception',
         ['外层账户scope与分流统计scope同名，dict打包发生TypeError；已改为funnel_definition。'],
         ['test_zero_request_funnel_retains_late_profitable_score']),
        ('run_20260910_172441_003843',0,'complete_passed',[],[])]
    for name,exit_code,status,issues,tests in descriptions:
        folder=REPORT/'independent_robust_audit'/name
        value=json.loads((folder/'verification.json').read_text())
        manifest=json.loads((folder/'artifact_sha256.json').read_text())
        assert all(sha(folder/key)==digest for key,digest in manifest.items())
        attempts.append(dict(path=str(folder),exit_code=exit_code,status=status,passed=value['passed'],complete=value['complete'],
            verification_sha256=sha(folder/'verification.json'),artifact_manifest_sha256=sha(folder/'artifact_sha256.json'),
            script_sha256=value['script_sha256'],issues=issues,regression_tests=tests,
            original_attempt_artifacts_unchanged=True))
    summary=json.loads((HERE/'validation_summary.json').read_text())
    assert summary['independent_audit_status']=='pending'
    summary.update(independent_audit_status='passed',independent_audit_path=str(SUCCESS/'verification.json'),
        independent_audit_sha256=EXPECTED,independent_audit_exit_code=0,
        independent_audit_scope=dict(new_huber_models=6,new_model_artifacts=48,internal_candidates=37,
            engineering_outer_accounts=1,accounts_including_baselines=42,current_prediction_tables=len(proof['predictions']),
            actual_trade_events=3800,first_exit_intents=1267,guard_requests=1291,guard_cancellations=24,
            read_input_files=1250,frozen_runtime_sources=56,audit_script_dependencies=11),
        independent_audit_tests=proof['self_tests'],failed_audit_attempts_preserved=2,
        production_accounts_rerun_for_auditor_repairs=False)
    write(HERE/'validation_summary.json',summary)
    text=(HERE/'README.md').read_text().replace('独立复核尚待完成。',
        '独立复核已完整通过（最终进程exit0）；验收证据见下方。')
    text += ('\n独立审计补充：\n\n'
        '- [完整验收报告](../independent_robust_audit/run_20260910_172441_003843/acceptance.md)：'
        '6个Huber模型、48个模型产物、37个内部候选及1个工程外层，含基准共42个账户。'
        '核3,800条成交、1,267个首次退出意图与1,291次guard复查；24次取消是跨候选计数，不能当单一组合交易。\n'
        '- 56运行源、392原登记输入、1,250项独审读取文件及11个审计脚本依赖始末不变。'
        '18项新审计反例及8项原guard反例通过；不与原42项实现测试重复混算。\n'
        '- 两次未完成审计分别因审计合同错误主动终止、分流字段重名异常结束。'
        '它们的原目录、日志、脚本及SHA完整保留；只修独立审计器，没有重跑或改写生产模型和账户。\n'
        f'- [尝试与修复记录]({out.name}/attempt_history.md)；成功verification SHA：`{EXPECTED}`。\n'
        '- 原archive_smoke.py记录的是审计完成前的历史命令，本次状态更新由complete_independent_audit.py记录。'
        '工程0请求已按完整当前预测独立复算，收益/频率目标仍未实现。\n')
    (HERE/'README.md').write_text(text)
    write(out/'attempt_history.json',dict(production_runtime_or_models_or_accounts_modified=False,
        attempts=attempts,successful_test_evidence=str(SUCCESS/'audit.log'),
        successful_test_script=str(SUCCESS/'audit_dependencies/scripts/test_audit_inventory_robust_t.py'),
        final_current_decision_funnel=[v for v in proof['current_decision_funnels'] if v['scope']=='test']))
    lines=['# v17烟测独立审计尝试记录','',
        '三次均为只读审计尝试；失败不计验收通过，原错误不覆盖。生产模型、账户、策略配置和行情未因审计修复而改变。','']
    for item in attempts:
        lines += [f"## {Path(item['path']).name}",'',f"状态：{item['status']}；exit={item['exit_code']}。",
            f"脚本SHA：`{item['script_sha256']}`。",'',*['- '+v for v in item['issues']],
            *['- 回归用例：`'+v+'`。' for v in item['regression_tests']],
            '',f"原证据：[verification.json]({Path(item['path']).relative_to(REPORT).as_posix()}/verification.json)。",'']
    # 本文在engineering子目录内，指向不变的原尝试绝对位置，避免错误相对层级。
    content='\n'.join(lines)
    for item in attempts:
        content=content.replace(Path(item['path']).relative_to(REPORT).as_posix()+'/verification.json',item['path']+'/verification.json')
    (out/'attempt_history.md').write_text(content+'\n')
    shutil.copy2(__file__,out/'complete_independent_audit.py')
    (out/'command.txt').write_text('python '+str(Path(__file__).resolve())+'\n')
    assert all(sha(name)==value for name,value in frozen.items())
    assert sha(SUCCESS/'verification.json')==EXPECTED
    write(out/'completion_identity.json',dict(passed=True,updated_only_engineering_validation=True,
        immutable_runtime_sources_verified=len(runtime),immutable_audit_helpers_verified=len(proof['audit_dependencies_sha256']),
        verification_sha256=EXPECTED,updated_summary_sha256=sha(HERE/'validation_summary.json'),
        previous_engineering_manifest_sha256=sha(previous/'manifest.json'),attempts_unchanged=True))
    write(HERE/'manifest.json',{str(path.relative_to(HERE)):sha(path) for path in sorted(HERE.rglob('*'))
        if path.is_file() and path!=HERE/'manifest.json'})
    print(out)
    print('summary SHA',sha(HERE/'validation_summary.json'))
    print('engineering manifest SHA',sha(HERE/'manifest.json'))

if __name__=='__main__':main()
