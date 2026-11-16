"""更新已完成v18烟测的辅助独审状态；先备份，不改原研究产物。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/lxl/seed-emulator/topology/quant')
report=root/'reports/inventory_coverage_t_smoke/run_20260910_181434_954042'
aux=report/'engineering_validation'
completion=aux/f'audit_completion_{datetime.now():%Y%m%d_%H%M%S_%f}'
previous=completion/'previous';previous.mkdir(parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ('validation_summary.json','README.md','manifest.json'):
 shutil.copy2(aux/name,previous/name)
audit=report/'independent_coverage_audit/run_20260910_182018_637252/verification.json'
proof=json.loads(audit.read_text()); expected='be2ab62e770c88f68525cf3109f9be87284f158be0f5ab8d4253581edab7c3f2'
assert sha(audit)==expected and all(proof[k] is True for k in ('passed','complete','all_read_inputs_unchanged'))
summary=json.loads((aux/'validation_summary.json').read_text())
for name,value in summary['original_report_files_sha256'].items():assert sha(report/name)==value
summary.update(independent_smoke_audit_status='passed',independent_smoke_audit=str(audit),independent_smoke_audit_sha256=expected,
 independent_smoke_audit_exit_code=0,independent_audit_tests=32,
 independent_audit_counts={'models':6,'model_artifacts':60,'prediction_tables':9,'internal_accounts':37,'engineering_outer_accounts':1,'accounts_including_baselines':42,'events':8172,'exit_intents':3413,'guard_checks':3446,'guard_cancellations':33,'read_inputs':1262,'runtime_sources':56,'audit_dependencies':12})
(aux/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
s=(aux/'README.md').read_text().replace('独立审核待完成，不以主运行完整性替代独立验收。','独立审核已一次exit0通过：6模型、9预测表、42账户，1262实际读取文件/56源/12审计依赖始末一致。')
s+='\n独审verification SHA：`'+expected+'`。32项反例及双Python编译通过，审核218.508秒；8172成交事件、3413首次退出意图、3446guard/33取消为跨候选数量，不是一个组合。审核通过不改变5%目标未达结论。\n'
(aux/'README.md').write_text(s)
shutil.copy2(__file__,completion/'complete.py')
(completion/'audit_identity.json').write_text(json.dumps({'audit':str(audit),'audit_sha256':expected,'exit_code':0,'original_research_files_unchanged':True},ensure_ascii=False,indent=2)+'\n')
assert sha(audit)==expected
manifest={str(p.relative_to(aux)):sha(p) for p in sorted(aux.rglob('*')) if p.is_file() and p!=aux/'manifest.json'}
(aux/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(completion)
print('summary',sha(aux/'validation_summary.json'),'manifest',sha(aux/'manifest.json'))
