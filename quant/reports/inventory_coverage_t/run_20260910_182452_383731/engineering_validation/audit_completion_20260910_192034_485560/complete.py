"""主独审真实结束后更新辅助工程终态；原研究产物不改，旧辅助状态先备份。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--verified-audit-sha256',required=True)
parser.add_argument('--observed-exit-code',type=int,required=True)
a=parser.parse_args()
assert a.observed_exit_code==0
root=Path('/data/lxl/seed-emulator/topology/quant')
report=root/'reports/inventory_coverage_t/run_20260910_182452_383731'
aux=report/'engineering_validation'
audit=report/'independent_coverage_audit/run_20260910_184346_798266'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((audit/'verification.json').read_text())
assert sha(audit/'verification.json')==a.verified_audit_sha256
assert all(proof[k] is True for k in ['passed','complete','all_read_inputs_unchanged','audit_dependencies_unchanged','no_outer_products_when_disabled_verified'])
assert proof['research_version']=='inventory_coverage_t_v18'
assert proof['outer_accounts']==0 and proof['candidate_accounts']==148 and proof['checked_accounts']==156
assert len(proof['models'])==24 and len(proof['predictions'])==24 and len(proof['audit_dependencies_sha256'])==12
assert proof['goal_achieved'] is False
summary=json.loads((aux/'validation_summary.json').read_text())
assert summary['independent_full_audit_status']=='running'
for name,value in summary['original_report_files_sha256'].items():assert sha(report/name)==value
completion=aux/f'audit_completion_{datetime.now():%Y%m%d_%H%M%S_%f}'
previous=completion/'previous';previous.mkdir(parents=True,exist_ok=False)
for name in ('validation_summary.json','README.md','manifest.json'):shutil.copy2(aux/name,previous/name)
counts={'models':len(proof['models']),'model_artifacts':240,'prediction_tables':len(proof['predictions']),'internal_accounts':proof['candidate_accounts'],'formal_outer_accounts':proof['outer_accounts'],'accounts_including_baselines':proof['checked_accounts'],'events':proof['raw_event_checks'],'exit_intents':proof['raw_exit_intent_checks'],'guard_checks':proof['guard_rows_verified'],'guard_cancellations':proof['cancellations_verified'],'read_inputs':len(proof['inputs_sha256']),'runtime_sources':56,'audit_dependencies':len(proof['audit_dependencies_sha256'])}
summary.update(independent_full_audit_status='passed',independent_full_audit_exit_code=0,independent_full_audit_sha256=a.verified_audit_sha256,independent_audit_counts=counts,independent_audit_tests=32,independent_audit_elapsed_seconds=proof['elapsed_seconds'])
(aux/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
s=(aux/'README.md').read_text();s=s.replace('完整独立审计进行中，不以主运行完整性替代独立核验。',f"完整独立审计已一次exit0通过：24模型、24完整预测表、148内部加8基准共156账户，{counts['read_inputs']}实际读取文件、56运行源及12审计依赖始末一致。")
s+=f"\n\n独审verification SHA `{a.verified_audit_sha256}`，耗时{proof['elapsed_seconds']:.3f}秒；{counts['events']}条成交、{counts['exit_intents']}首次退出意图、{counts['guard_checks']}首腿复查及{counts['guard_cancellations']}次取消（跨候选重复计数）已逐条复核。32项独立反例通过，核验通过不表示恢复或收益目标达成。\n"
(aux/'README.md').write_text(s)
shutil.copy2(__file__,completion/'complete.py')
shutil.copy2('/tmp/inventory_coverage_full_independent_audit.log',completion/'audit.log')
(completion/'audit_identity.json').write_text(json.dumps({'audit_verification':str(audit/'verification.json'),'audit_sha256':a.verified_audit_sha256,'exit_code':a.observed_exit_code,'original_research_files_unchanged':True},ensure_ascii=False,indent=2)+'\n')
assert sha(audit/'verification.json')==a.verified_audit_sha256
for name,value in summary['original_report_files_sha256'].items():assert sha(report/name)==value
(aux/'manifest.json').write_text(json.dumps({str(p.relative_to(aux)):sha(p) for p in sorted(aux.rglob('*')) if p.is_file() and p!=aux/'manifest.json'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completion_directory':str(completion),'summary_sha256':sha(aux/'validation_summary.json'),'manifest_sha256':sha(aux/'manifest.json'),'audit_counts':counts},ensure_ascii=False,indent=2))
