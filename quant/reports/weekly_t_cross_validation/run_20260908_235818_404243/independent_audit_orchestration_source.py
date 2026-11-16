"""只等待指定v8正式报告写完，冻结审计器后并行执行两种只读核验并归档，不控制正式进程。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

ROOT=Path('/data/lxl/seed-emulator/topology/quant')
REPORT=ROOT/'reports/weekly_t_cross_validation/run_20260908_235818_404243'
LOG=Path('/tmp/weekly_t_full_cv_v8.log')
PYTHON='/home/lxl/anaconda3/envs/quant/bin/python'
EXPECTED={
 'audit_weekly_t_residual.py':'efef40eaf92e367c44386ccdad1b5f15445a667dae7ef52cdff5dc06d69f4c45',
 'audit_weekly_t_residual_execution.py':'89a2a8eb32521cb4f0ed3e70d9748370a0215c4d9c9059f553ece04a07aa1b5a'}

def digest(path):
    """摘要真实文件内容。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

print('等待指定v8正式四年报告完成，不读取尚未完成的外层结果。',flush=True)
while True:
    if REPORT.is_dir() and (REPORT/'final_report.md').exists() and (REPORT/'all_rules_frozen_before_outer_test.json').exists():
        cfg=json.loads((REPORT/'resolved_config.json').read_text())
        ready=all((REPORT/f'year_{year}/test_metrics.csv').exists() for year in cfg['test_years'])
        finished=LOG.exists() and f'报告完成：{REPORT}/final_report.md' in LOG.read_text()
        if ready and finished:break
    time.sleep(30)

processes=[];files={}
for script,sha in EXPECTED.items():
    path=ROOT/'scripts'/script;assert digest(path)==sha
    copy=REPORT/('independent_'+script);shutil.copyfile(path,copy);files[copy.name]=sha
    log_name='independent_residual_audit.log' if script=='audit_weekly_t_residual.py' else 'independent_execution_audit.log'
    stream=(REPORT/log_name).open('w')
    process=subprocess.Popen([PYTHON,'-u',str(path),'--report-dir',str(REPORT)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    processes.append((script,process,stream,log_name))
    print(f'已启动{script} PID={process.pid}，正式报告完成标志已确认。',flush=True)
codes={}
for script,process,stream,log_name in processes:
    code=process.wait();stream.close();codes[script]=code
    files[log_name]=digest(REPORT/log_name)
    print(f'{script} exit={code}',flush=True)

for script,sha in EXPECTED.items():assert digest(ROOT/'scripts'/script)==sha
for name in ['independent_residual_audit.json','independent_residual_audit.md','independent_execution_audit.json','independent_execution_audit.md']:
    if (REPORT/name).exists():files[name]=digest(REPORT/name)
passed=all(code==0 for code in codes.values())
summary={'classification':'v8_full_independent_model_and_execution_audit_archive','audit_exit_codes':codes,
    'auditors_unchanged':True,'artifact_sha256':files,'passed':passed,
    'frozen_smoke_auditor_evidence':str(ROOT/'reports/weekly_t_cross_validation_smoke/run_20260908_235631_167038/independent_v8_audit_archive.json'),
    'parallel_after_formal_report_finished':True}
if passed:
    model=json.loads((REPORT/'independent_residual_audit.json').read_text())
    execution=json.loads((REPORT/'independent_execution_audit.json').read_text())
    assert model['passed'] and execution['completed_all_four_years'] and all(item['audit_pass'] for item in execution['annual'])
    summary['models']=len(model['models']);summary['selectors']=len(model['selectors']);summary['execution_totals']=execution['totals']
    summary['annual_increment_CNY']={str(item['year']):item['T_increment_CNY'] for item in execution['annual']}
    summary['model_audit_seconds']=model['elapsed_seconds']
    print(json.dumps({key:value for key,value in summary.items() if key!='artifact_sha256'},ensure_ascii=False,indent=2),flush=True)
(REPORT/'independent_v8_audit_archive.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(0 if passed else 1)
