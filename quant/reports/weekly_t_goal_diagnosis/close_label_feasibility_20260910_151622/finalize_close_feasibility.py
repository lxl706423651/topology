from pathlib import Path
import hashlib, json, shutil
from datetime import datetime
out=Path('/tmp/close_label_feasibility_path.txt').read_text().strip(); out=Path(out)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proposal=Path('reports/weekly_t_goal_diagnosis/inventory_close_proposal_20260910/proposal.md')
review={'scope':'read_only_design_review_no_account_loading','proposal_path':str(proposal),'proposal_sha256_at_review':sha(proposal),'compatible_with_existing_data_and_engine':True,'implementation_started':False,'clarifications':['recompute close target from preserved endpoints without old trade_week_complete qualification','14:00 signal to 14:55 is54 trading minutes after14:01 so not uniformly longer than h60','reference label can be finite at zero-volume next bar; no executable fill guarantee','close-only removes cross-horizon per-row choice but does not remove directional/model/candidate-selection error']}
(out/'proposal_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
for source in ['/tmp/archive_close_label_feasibility.py','/tmp/finalize_close_feasibility.py']:
 shutil.copy2(source,out/Path(source).name)
manifest={'archive_status':'complete_bounded_read_only_feasibility','created_at':datetime.now().isoformat(),'no_model_fit':True,'no_outer_account_read':True,'no_runtime_source_changed':True,'output_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
assert all(sha(out/p)==h for p,h in manifest['output_sha256'].items())
print(json.dumps({'directory':str(out),'files':len(manifest['output_sha256']),'memo_sha256':sha(out/'feasibility_memo.md'),'checks_sha256':sha(out/'checks.json'),'manifest_sha256':sha(out/'manifest.json')},ensure_ascii=False,indent=2))
