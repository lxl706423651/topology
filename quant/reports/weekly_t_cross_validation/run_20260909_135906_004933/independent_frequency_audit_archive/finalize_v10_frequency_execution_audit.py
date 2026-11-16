from pathlib import Path
from decimal import Decimal
import csv,datetime,hashlib,json,shutil
ROOT=Path('/data/lxl/seed-emulator/topology/quant');RUN=ROOT/'reports/weekly_t_cross_validation/run_20260909_135906_004933';A=RUN/'independent_frequency_audit_archive'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((RUN/'independent_execution_audit.json').read_text());verification=json.loads((A/'verification.json').read_text())
assert audit['passed'] and audit['completed_all_four_years']
expected=json.loads((RUN/'independent_frequency_audit_attempt_01_csv_datetime/candidate_inputs_sha256.json').read_text())
assert all(sha(RUN/name)==digest for name,digest in expected.items())
source=json.loads((RUN/'source_hashes.json').read_text());assert len(source)==27 and all(sha(ROOT/name)==digest for name,digest in source.items())
assert all(sha(RUN/name)==digest for name,digest in json.loads((A/'read_only_input_sha256_before.json').read_text()).items())
costs=[]
for annual in audit['annual']:
 y=annual['year'];gross=Decimal(0);fees=Decimal(0);slippage=Decimal(0)
 with (RUN/f'year_{y}/test_trade_events.csv').open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   if row['event_type']!='t':continue
   gross+=(1 if row['side']=='sell' else -1)*Decimal(row['reference_price'])*Decimal(row['shares'])
   fees+=Decimal(row['fee']);slippage+=Decimal(row['slippage_cost'])
 net=gross-fees-slippage
 assert abs(float(net)-annual['T_increment_CNY'])<1e-5
 costs.append({'year':y,'reference_price_gross_CNY':str(gross),'fees_CNY':str(fees),'slippage_CNY':str(slippage),'net_CNY':str(net)})
summary={'passed':True,'completed_at':datetime.datetime.now().isoformat(),'auditor_sha256':audit['auditor_source_sha256'],'candidate_accounts':100,'runtime_sources_verified':27,'raw_price_csv_count':len(audit['raw_execution_sources']),'candidate_original_files_unchanged':len(expected),'root_and_annual_original_files_unchanged':verification['original_file_count'],'runtime_sources_changed':False,'reran_backtest_or_training':False,'account_results_modified':False,'totals':audit['totals'],'annual_cost_attribution':costs,'reference_price_gross_CNY':str(sum(Decimal(x['reference_price_gross_CNY']) for x in costs)),'fees_CNY':str(sum(Decimal(x['fees_CNY']) for x in costs)),'slippage_CNY':str(sum(Decimal(x['slippage_CNY']) for x in costs)),'net_CNY':str(sum(Decimal(x['net_CNY']) for x in costs)),'maximum_account_error_CNY':max(max(x['cash_max_abs_error_CNY'],x['equity_max_abs_error_CNY']) for x in audit['annual']),'frozen_v4_baseline_max_abs_error_CNY':max(x['frozen_v4_baseline_max_abs_error_CNY'] for x in audit['annual']),'test_counts':{'frequency_helper_tests':15,'candidate_denominator_and_reader_boundary_tests':19,'raw_replay_cases':10},'auditor_failed_attempts_preserved':['independent_frequency_audit_attempt_01_csv_datetime','independent_frequency_audit_attempt_02_order_intent'],'auditor_fixes':['CSV timestamps use format=mixed; identical datetime values with alternate string representations preserve results; future signal timestamp still rejected.','No-T original budget request differs from actual copied filled request on legal capacity reduction. Original planned quantity independently recomputed from prior price/budget; copied request must equal baseline fill. All economic fields still match.'],'audit_artifact_sha256':{name:sha(RUN/name) for name in ['independent_execution_audit.json','independent_execution_audit.md']}}
(A/'final_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# v10 独立执行审核归档','', '完整审核已通过。核对100个内层候选、27个冻结源文件、7039个原始行情CSV及四年实际/基准逐笔账户。没有重新训练或重跑回测，原1600个候选文件与112个年度/报告输入文件SHA均未变。','', '|年度|完成组|恢复片段|扣费后T增量(元)|','|---|---:|---:|---:|']
for r in audit['annual']:lines.append(f"|{r['year']}|{r['completed_pairs']}|{r['fragments']}|{r['T_increment_CNY']:.9f}|")
lines+=['',f"共1760组、1829个恢复片段。参考价格毛差约352.00元 − 滑点3337.408元 − 费用7305.97015769元 = 净增量−10291.37815769元。费用与滑点独立核对；滑点已在成交价中，不重复扣除。",'', 'T+1违规、每日T股数偏差、恢复失败、期末未恢复T以及基础计划偏差均为0。冻结v4基准逐日权益误差为0；最大逐笔现金/逐日权益复算误差约1.40e−9元。','', '原底仓仍有171个年度内周退出失败日、307个失败股票日，四年末合计7筆未退出周持仓；其中2026年SH.603718的2500股按4月29日15:00最后真实价格3.90元估值9750元，4月30日期末报价陈旧。两账户持仓完全相同，此风险没有被T增量隐藏。', '', '两次未完成的审计尝试完整保留：首次为pandas日期字符串混排解析边界；第二次为对计划请求意图的比较过严（SH.605336：基准请求500股因容量成交300股，实际账户复制300股）。修复仅修改独立审计器：日期采用mixed解析；分别验证原预算请求和复制成交请求。真实成交股数、现金、费用等经济字段继续严格核对。运行源、原账簿和策略结果均未修改。', '', '15项候选执行helper测试、19项独立候选/分母/边界测试、10项原始逐笔反例全部通过；两次失败的脚本/日志/输入SHA以及逐次diff均已归档。最终审核脚本SHA：`'+summary['auditor_sha256']+'`。','', '正式结果：[独立审核说明](../independent_execution_audit.md)、[完整JSON](../independent_execution_audit.json)、[本次归档摘要](final_review.json)。']
(A/'README.md').write_text('\n'.join(lines)+'\n')
shutil.copy2('/tmp/v10_frequency_execution_audit_orchestration.log',A/'orchestration.log')
shutil.copy2('/tmp/finalize_v10_frequency_execution_audit.py',A/'finalize_v10_frequency_execution_audit.py')
(A/'archive_sha256.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file() and p.name!='archive_sha256.json'},indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
