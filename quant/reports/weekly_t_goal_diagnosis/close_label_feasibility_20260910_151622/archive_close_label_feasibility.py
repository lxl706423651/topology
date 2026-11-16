from pathlib import Path
import hashlib, json, shutil
from datetime import datetime
out = Path('/tmp/close_label_feasibility_path.txt').read_text().strip()
out = Path(out)
checks = json.loads((out/'checks.json').read_text())
report = Path('reports/inventory_learning_t/run_20260910_151232_841321')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen = json.loads((report/'source_hashes.json').read_text())
observed = {name:sha(name) for name in frozen}
snapshot = {name:sha(report/'source_snapshot'/name) for name in frozen}
assert observed == frozen == snapshot
runtime = {'scope':'read_only_identity_check_no_accounts_read','runtime_files':len(frozen), 'current_matches_frozen':True,'snapshot_matches_frozen':True,'runtime_source_hashes_sha256':sha(report/'source_hashes.json'),'sources_sha256':observed}
(out/'runtime_identity.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
shutil.copy2('/tmp/check_close_label_feasibility.py', out/'check_close_label_feasibility.py')
shutil.copy2('/tmp/check_close_label_feasibility.log', out/'check_close_label_feasibility.log')
# 有界审查后的来源合同，不重新读取数GB缓存或运行任何模型。
zero = sum(row['finite_target_despite_zero_execution_volume'] for row in checks['sample'])
summary = {'status':'bounded_feasibility_passed_not_a_new_strategy_or_profit_test',
 'old_close_endpoint_available_in_existing_cache':True,'target_clock':'14:55','is_market_close_1500':False,
 'sample_signal_rows':checks['sample_rows'], 'raw_endpoint_field_comparisons':sum(row['endpoint_field_comparisons'] for row in checks['raw_checks']),
 'raw_endpoint_max_abs_error':max(row['max_abs_endpoint_error'] for row in checks['raw_checks']),
 'sample_finite_target_zero_execution_volume':zero,'full_missing_label_census':False,
 'aggregate_cache_identity_matches_previously_audited_input':True,
 'close_only_models_per_fold':2,'fourth_endpoint_models_per_fold':8,'unchanged_candidate_grid_size_if_only_endpoint_interface_changes':37,
 'current_runtime_frozen_files':len(frozen), 'read_outer_returns':False,'fit_or_select_parameters':False,'implemented_new_strategy':False,
 'minimum_join_order':['extract_separate_close_labels_from_loaded_original_intraday','build_frozen_causal_permanent_inventory_core','left_join_labels_preserving_all_current_keys','strict_core_excluded_embargo_purge_for_training_only'],
 'close_identity':{'recovery_kind':'fixed_clock','recovery_clock':'14:55','recovery_key':'close'},
 'cost_target':'reference_endpoint_return_then_same_day_rate_cost_conversion_not_actual_portfolio_pnl',
 'source_contracts':{'cache_builder':'quant_backtest/weekly_t_data.py:203','future_scrub':'quant_backtest/inventory_t_learning_data.py:186','training_join':'quant_backtest/inventory_t_learning_models.py:28','model_endpoint_validation':'quant_backtest/inventory_t_learning_models.py:87','policy_endpoint_choice':'quant_backtest/inventory_t_learning_policy.py:89','dynamic_deadline':'quant_backtest/dynamic_t_engine.py:301'}}
(out/'feasibility_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(out/'worklog.md').write_text('''# 只读检查记录

- 原任务：评估次分钟至14:55参考标签复用，不实现新策略、不拟合、不读取外层收益。
- 已执行 `/home/lxl/anaconda3/envs/quant/bin/python /tmp/check_close_label_feasibility.py > /tmp/check_close_label_feasibility.log 2>&1`，exit 0；脚本与完整输出已复制至本目录。
- 检查器对固定首8代码的2024/2025核心年缓存样本核端点，并对SH.600000两个核心年原始CSV独立核对，未以涨跌/收益选择样本；流式SHA核对全体intraday.pkl身份，未重新加载全市场或回测。
- 静态读缓存生成、未来字段剥离、标签左联、训练purge、政策时钟及动态执行期限；根据读取源码提出两个独立方案，未选择盈利参数。
- 归档时核48份当前运行源码与本次冻结清单及快照逐字节一致。没有修改原源码、配置、缓存、账户或历史报告。
- 抽样无缺价不代表全市场无缺价；零量仍可能有参考标签。本记录未检查原始分钟全市场价格完整性、未审计运行成功终态。
''',encoding='utf-8')
print(json.dumps({'out':str(out),'zero_volume_label_rows':zero,'runtime_files':len(frozen)},ensure_ascii=False))
