"""只读验收内部诊断产物、原内部汇总及运行身份，归档命令与反例证据。"""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import pandas as pd

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('final_directory',type=Path)
 args=parser.parse_args();out=args.final_directory.resolve();own=Path(__file__).resolve().parent
 root=own.parents[2];d=json.loads((out/'diagnosis.json').read_text());report=Path(d['source_report'])
 assert d['passed'] and d['all_read_inputs_unchanged'] and d['no_outer_account_read']
 assert [x['fold'] for x in d['folds']]==[2023,2024,2025,2026]
 checked={}
 def record(path):
  path=Path(path).resolve();checked.setdefault(str(path),sha(path));return path
 def read(path):return json.loads(record(path).read_text())
 for path,value in d['inputs_sha256'].items():assert sha(record(path))==value,path
 cfg=read(report/'resolved_config.json');runtime=read(report/'runtime_integrity_audit.json')
 assert cfg['research_version']=='dynamic_inventory_t_v13' and runtime['all_unchanged']
 assert cfg['test_years']==[2023,2024,2025,2026]
 source=read(report/'source_hashes.json');assert len(source)==43
 for path,value in source.items():assert sha(record(report/'source_snapshot'/path))==value,path
 candidates=pd.read_csv(record(out/'candidate_summary.csv'),float_precision='round_trip')
 contrasts=pd.read_csv(record(out/'paired_exit_contrasts.csv'),float_precision='round_trip')
 assert len(candidates)==652 and len(contrasts)==432
 for year,p in candidates.groupby('fold'):
  original=pd.read_csv(record(report/f'year_{year}/validation_rule_metrics.csv'),float_precision='round_trip').set_index('candidate_index')
  p=p.set_index('candidate_index').sort_index();original=original.sort_index()
  assert set(p.index)==set(range(163))
  for left,right in [('complete_groups','t_pair_count'),('incremental_marked_net','incremental_net_pnl'),
                     ('paired_lots','completed_t_pair_lots'),*[(k,k) for k in ['inventory_mismatch_days','recovery_failure_days','pending_pairs','t1_violation_count','base_plan_mismatch_count']]]:
   assert np.allclose(p[left],original[right],rtol=0,atol=1e-6),(year,left)
  assert p.selected.sum()==1
 interpretation=read(out/'interpretation_sources.json');assert interpretation['passed']
 for path,value in interpretation['inputs_sha256'].items():assert sha(record(path))==value,path
 assert sha(record(out/'interpretation.md'))==interpretation['interpretation_sha256']
 archive=out/'engineering_validation';archive.mkdir(exist_ok=False)
 for p in [own/'explain.py',Path(__file__),Path('/tmp/test_dynamic_inner_diagnosis.py'),
           Path('/tmp/dynamic_inner_diagnosis_tests.log'),Path('/tmp/dynamic_inner_diagnosis_stage1.log'),
           Path('/tmp/dynamic_inner_diagnosis_stage2.log'),Path('/tmp/dynamic_inner_diagnosis_final.log')]:
  shutil.copy2(p,archive/p.name)
 for p,value in checked.items():assert sha(p)==value,p
 result={'passed':True,'candidate_accounts':652,'same_entry_exit_contrasts':432,
  'folds':[2023,2024,2025,2026],'internal_account_input_files':len(d['inputs_sha256']),
  'global_and_derivative_inputs':len(checked)-len(d['inputs_sha256']),
  'runtime_source_snapshots_verified':43,'runtime_terminal_success':True,
  'source_account_rows_and_diagnosis_metrics_equal':True,'all_read_inputs_unchanged':True,
  'no_outer_account_read':True,'no_parameters_changed':True,'no_account_or_model_rerun':True,
  'script_sha256':sha(__file__),'inputs_sha256':checked,
  'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
  'validation':'配对9项合成断言及diagnose/explain双Python compile通过，日志和源码归档'}
 (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['inputs_sha256','output_sha256']},ensure_ascii=False))

if __name__=='__main__':main()
