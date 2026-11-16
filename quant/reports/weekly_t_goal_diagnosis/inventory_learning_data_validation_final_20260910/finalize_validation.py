"""冻结新训练数据预检证据，比较全部账户/来源CSV；不拟合或读取外层账户。"""
from pathlib import Path
import hashlib,json,shutil
import pandas as pd
old=Path('reports/weekly_t_goal_diagnosis/inventory_learning_data_validation_20260910')
out=Path('reports/weekly_t_goal_diagnosis/inventory_learning_data_validation_final_20260910')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
checks={}
for path in sorted((old/'inventory_learning_cohort').rglob('*.csv')):
 relative=path.relative_to(old)
 assert path.read_bytes()==(out/relative).read_bytes(),relative
 checks[str(relative)]={'byte_equal':True,'sha256':sha(path)}
books=[]
for year in [2024,2025]:
 base=out/f'inventory_learning_cohort/core_{year}'
 e=pd.read_csv(base/'base_entries.csv',float_precision='round_trip')
 a=pd.read_csv(base/'daily_inventory_audit.csv',float_precision='round_trip')
 last=1e6;maxerror=0.
 for event in e.itertuples(index=False):
  assert event.side=='buy' and event.shares%100==0 and 10000-1e-7<=event.notional<=20000+1e-7
  expected=last-event.notional-event.fee
  maxerror=max(maxerror,abs(expected-event.cash_after));last=event.cash_after
  assert event.cash_after>=150000-1e-7
 assert maxerror<1e-7 and e.symbol.nunique()==len(e)
 assert a.day_sold.eq(0).all() and a.inventory_difference.eq(0).all()
 assert a.t1_violation_count.eq(0).all() and a.recovery_failure.eq(False).all()
 books.append({'year':year,'base_entries':len(e),'cash_arithmetic_max_error':maxerror,'final_base_cash':last,
               'single_automatic_entry_per_symbol':True,'entry_notional_band_passed':True,
               'base_inventory_and_t1_passed':True})
verification=json.loads((out/'verification.json').read_text())
assert verification['passed'] and verification['source_sha256']['quant_backtest/inventory_t_learning_data.py']==sha('quant_backtest/inventory_t_learning_data.py')
assert verification['source_sha256']['quant_backtest/dynamic_t_engine.py']==sha('quant_backtest/dynamic_t_engine.py')
result={'passed':True,'compared_csv_count':len(checks),'all_account_and_lineage_csv_byte_equal':True,
        'only_current_frame_contract_change':'trade_week_complete omitted; old weekly label eligibility cannot suppress permanent holding opportunities',
        'source_frozen_for_entire_final_run':True,'checks':checks,'independent_base_cash_checks':books}
(out/'identity_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
shutil.copy2('/tmp/check_inventory_learning_real_final.log',out/'real_first_fold.log')
shutil.copy2('/tmp/finalize_inventory_learning_validation.py',out/'finalize_validation.py')
(out/'worklog.md').write_text('''# 验证命令与结果\n\n- 12项合成测试：`/home/lxl/anaconda3/envs/quant/bin/python scripts/test_inventory_t_learning_data.py`，exit 0。\n- 双编译：系统 python3 与 quant Python 的 `-m py_compile`，两文件均通过。\n- 最终真实首折：`/home/lxl/anaconda3/envs/quant/bin/python /tmp/check_inventory_learning_real_final.py > /tmp/check_inventory_learning_real_final.log 2>&1`，exit 0，165.96秒。数据模块在运行始末 SHA 一致；旧动态引擎未改。\n- `python /tmp/finalize_inventory_learning_validation.py` 核全部18份CSV逐字节相同，并独立复核自动底仓每腿现金/费用/整手/金额区间和无T库存审计。\n\n首次真实预检的 source_sha256 是完成时文件身份，加载后曾仅删旧trade_week_complete字段；最终轮已保存准确启动源码并再次完整构建，作为最终验证依据。两轮的账户和lineage均相同，没有拟合、标签连接或外层收益读取。\n''')
manifest={'status':'complete','files_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'comparison_count':len(checks),'books':books,'manifest_sha256':sha(out/'manifest.json'),
                  'verification_sha256':sha(out/'verification.json'),'identity_comparison_sha256':sha(out/'identity_comparison.json')},indent=2))
