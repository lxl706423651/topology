"""Wait for finished v9 outer reports, then read-only audit and archive our artifacts."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,time
ROOT=Path('/data/lxl/seed-emulator/topology/quant')
REPORT=ROOT/'reports/weekly_t_cross_validation/run_20260909_124918_833635'
SCRIPT=ROOT/'scripts/audit_weekly_t_active_execution.py'
EXPECTED='ee055d8a7c1e4401026c5550eb2a34850663055862f7810a9fb759b8f3f3b6f1'
PYTHON='/home/lxl/anaconda3/envs/quant/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
print('Waiting for final_report and all four completed annual results',flush=True)
while not ((REPORT/'final_report.md').is_file() and all((REPORT/f'year_{year}/test_metrics.csv').is_file() for year in [2023,2024,2025,2026])):
    time.sleep(10)
assert sha(SCRIPT)==EXPECTED,'Auditor source changed after smoke'
archive=REPORT/'independent_active_audit_archive';archive.mkdir(exist_ok=False)
shutil.copy2(SCRIPT,archive/SCRIPT.name)
command=[PYTHON,'-u',str(SCRIPT),'--report-dir',str(REPORT)]
originals=[REPORT/'final_report.md',REPORT/'resolved_config.json',REPORT/'source_hashes.json',REPORT/'all_rules_frozen_before_outer_test.json']
originals.extend(p for year in [2023,2024,2025,2026] for p in (REPORT/f'year_{year}').glob('*.csv'))
originals.extend(REPORT/f'year_{year}/selected_rule.json' for year in [2023,2024,2025,2026])
before={str(p.relative_to(REPORT)):sha(p) for p in originals}
(archive/'read_only_input_sha256_before.json').write_text(json.dumps(before,indent=2))
started=time.monotonic();print('Starting independent audit',datetime.datetime.now().isoformat(),flush=True)
with (archive/'audit.log').open('w') as log:
    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
print('Independent audit exited',result.returncode,'elapsed',time.monotonic()-started,flush=True)
assert result.returncode==0,(archive/'audit.log').read_text()[-6000:]
audit=json.loads((REPORT/'independent_execution_audit.json').read_text())
assert audit['passed'] and audit['completed_all_four_years']
assert sha(SCRIPT)==EXPECTED
changed=[name for name,value in before.items() if sha(REPORT/name)!=value]
assert not changed,('Original report changed during read-only audit',changed)
summary={'passed':True,'created_at':datetime.datetime.now().isoformat(),'command':command,'elapsed_seconds':time.monotonic()-started,'auditor_sha256':EXPECTED,'original_outputs_unchanged':True,'original_file_count':len(before),'totals':audit['totals'],'net_increment_by_year':{str(row['year']):row['T_increment_CNY'] for row in audit['annual']},'archive_sha256':{p.name:sha(p) for p in archive.iterdir() if p.is_file()},'audit_output_sha256':{p.name:sha(p) for p in [REPORT/'independent_execution_audit.json',REPORT/'independent_execution_audit.md']}}
(archive/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
