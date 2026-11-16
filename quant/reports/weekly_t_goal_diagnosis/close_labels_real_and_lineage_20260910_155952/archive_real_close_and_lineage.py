"""保存一次性真实标签接口和完整核心lineage复用说明；无重跑规划或模型。"""
from pathlib import Path
import json,hashlib,shutil
out=Path(Path('/tmp/close_labels_real_and_lineage_path.txt').read_text());result=json.loads((out/'verification.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['/tmp/validate_real_close_labels_interface.py','/tmp/archive_real_close_and_lineage.py']:
 shutil.copy2(name,out/Path(name).name)
shutil.copy2('/tmp/real_close_labels_interface.log',out/'verification.log')
smoke=Path('reports/inventory_guarded_t_smoke/run_20260910_154550_255425')
frozen=json.loads((smoke/'source_hashes.json').read_text())
assert len(frozen)==51 and all(sha(p)==h and sha(smoke/'source_snapshot'/p)==h for p,h in frozen.items())
(out/'source_identity.json').write_text(json.dumps({'runtime_files_unchanged':51,'current_and_frozen_snapshots_match':True,'new_interface_sha256':sha('quant_backtest/inventory_t_close_labels.py'),'new_test_sha256':sha('scripts/test_inventory_t_close_labels.py'),'independent_read_only_review_passed':True},ensure_ascii=False,indent=2)+'\n')
lines=['# 固定尾盘标签真实接口核验及完整核心队列复用','',
'本次只做一次固定小样本数据接口验证和已保存lineage身份核对；不重读原CSV、不重建市场、不重新规划底仓、不拟合，也没有读取外层收益。13个合成反例的既存通过证据在前一独立目录，本次没有重复运行它们。','',
'固定使用此前核过的首8股票、2024/2025两年31,040个当前键。标签价格比与原精确next-minute/14:55端点误差0；21条next-minute成交量为0的记录仍保留参考标签，不能称可成交。全部原f32当前列、输入行序和当前键经attach后不变；人为移除1条标签后仍保留原机会，价格/target为NaN，确定时钟仍正确。','',
'另用首折实际206,488条完整未联标签lineage验证13列真实来源/账本状态的逐值和dtype保持。这8个小样本股票不在该首折实际core底仓，所以该项206,488行均是缺整条label键的测试；不能把小样本标签充当完整core训练标签。完整core模型仍须使用完整标签覆盖并在最后严格purge。','',
'真正覆盖全部未联标签机会的是：','',
'`reports/inventory_learning_t/run_20260910_151232_841321/year_<fold>/training_cohort/inventory_learning_cohort/core_<core>/cohort_lineage.csv`','',
'不是`models/h20_*/training_inputs.pkl`或任何horizon训练有效行。8份lineage共811,552行，逐份内容hash与metadata一致；各折分别为206,488、192,744、206,160、206,160行。同一年在不同折的选股/现金来源可能不同，不能仅按core年份复用或合并。','',
'|留出折|core年|完整机会数|','|---|---|---:|']
for row in result['lineage_files']:lines.append(f"|{row['fold']}|{row['core_year']}|{row['rows']}|")
lines += ['', '完整路径、CSV字节SHA及结构化frameSHA分别列于[lineage_files.csv](lineage_files.csv)。本次读取时完整v14独立审计已成功落盘，8份CSV字节SHA全部匹配其`verification.inputs_sha256[absolute_path]`；此前还未完成审计时的暂定状态不再用于自动复用许可。原每core的`metadata.cohort_lineage_sha256`是按真实dtype/列序的frame hash，不是CSV文件字节SHA，二者必须分别核对。','',
'13个保存字段为：datetime、symbol、feature_source_start、intraday_feature_source_start、entry_feature_source_start、entry_rank_source_start、portfolio_cash_source_start、base_entry_datetime、baseline_opening_shares、frozen_base_shares、source_model_training_years、source_model_sha256。原总metadata明确labels_attached=False、future_fill_or_return_filter=False、core_sampling_performed=False。','',
'后续最小复用路径（这里只说明，未实现）：','',
'1. 要求来源完整独立审计passed/complete/inputs unchanged；锚定配置、拆分、8份lineage及metadata、原基准账本、48来源源码和selector模型身份。配置仍采用相同的初始现金、15%现金保留、1万—2万元底仓、原核心/排除年份、7日隔离和训练禁用人工加仓。','2. 复用原full intraday的只读加载及完整截面session/provenance/timing增强；必须在裁剪股票之前完成所有全市场同刻特征。不能仅凭PKL中旧f_*就认为具备后续全部新增因子，也不能在小持仓截面重算市场统计。','3. 每fold/core按lineage全部datetime/symbol键从完整当前表一对一取行；要求key唯一且全部找到，缺当前行必须报错，不按标签有效性取交集。再以原基准daily_inventory_audit中baseline_opening_shares>0的股日和原当前表做键集合核对，可验证完整性而无需再次撮合基础请求。','4. 剥离所有execution_/recovery_/target_/label_、旧4个weekly context因素（f_weekly_pred_return_rank、f_weekly_pred_amplitude_rank、f_weekly_score、f_weekly_selection_rank）以及trade_week_complete。保留原full current因子的实际dtype和列顺序。原日期列按当前datetime标准化。','5. 回取的原intraday feature_source_start应先与保存的intraday_feature_source_start逐值相同（NaT也须一致）；然后附全套保存lineage，尤其把最终feature_source_start设为保存的intraday/entry全排名供体/所有过去现金竞争请求来源的合并最小值。不能只用本股历史起点，也不能只从成功买单重算现金来源而漏失败请求。','6. 两core拼接后按原datetime/symbol稳定排序，核全部lineage frame SHA和总机会数。此时才左联独立14:55标签；只有后续训练purge可删除缺未来价或不满足来源区间的行。旧h20/40/60是否能恢复、是否有有效标签，不得作为当前全集或重建过滤条件。','',
'这种路径复用的是已独立证明的原无T底仓历史及来源，不改变T执行器，也不把v15取消记录带入训练队列。原v14独立auditor本身已在cohort()中从完整排名供体/干净请求/原真实基础事件重建日初旧股和cash source，再精确核lineage；新调用者可用这条已完成身份链省去重复规划，而不是跳过其数据证明。','',
'可复用的范围限于相同原配置和已审计核心队列。若改变选股、底仓金额、资金、人工训练假设、入场约束、原始行情或来源隔离，旧lineage不再代表新训练场景，需要重新构建/审计。两个交互OOF核心年仍只允许共同拟合，不能再拿其中一年当独立校准。','',
'51份运行源码与对应v15烟测快照保持相同；新close数据接口仍未接入任何模型。全部命令/检查结果、输入SHA和脚本见verification.json、verification.log及本目录脚本。']
(out/'reuse_memo.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'manifest.json').write_text(json.dumps({'files_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'out':str(out),'verification_sha256':sha(out/'verification.json'),'manifest_sha256':sha(out/'manifest.json')},ensure_ascii=False,indent=2))
