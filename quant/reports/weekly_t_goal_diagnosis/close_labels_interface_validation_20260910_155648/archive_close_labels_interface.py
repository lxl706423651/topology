"""仅归档固定14:55标签接口及合成验证，不接入任何模型或回测。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path.cwd();out=root/'reports/weekly_t_goal_diagnosis'/f'close_labels_interface_validation_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir()
Path('/tmp/close_labels_interface_validation_path.txt').write_text(str(out))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen_report=root/'reports/inventory_guarded_t_smoke/run_20260910_154550_255425'
frozen=json.loads((frozen_report/'source_hashes.json').read_text())
assert len(frozen)==51
assert all(sha(root/name)==value and sha(frozen_report/'source_snapshot'/name)==value for name,value in frozen.items())
newfiles=['quant_backtest/inventory_t_close_labels.py','scripts/test_inventory_t_close_labels.py']
for name in newfiles:
 target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
shutil.copy2('/tmp/inventory_t_close_labels_tests.log',out/'unit_tests.log')
shutil.copy2('/tmp/archive_close_labels_interface.py',out/'archive_close_labels_interface.py')
summary={'passed':True,'tests':13,'test_elapsed_seconds':.216,'quant_compile_passed':True,'system_compile_passed':True,
 'scope':'pure_data_interface_synthetic_tests_only','market_reloaded':False,'models_fitted':0,'accounts_run':0,
 'runtime_files_unchanged':51,'source_smoke':str(frozen_report),'source_hashes_manifest_sha256':sha(frozen_report/'source_hashes.json'),
 'new_source_sha256':{name:sha(root/name) for name in newfiles},
 'apis':['extract_fixed_close_labels(intraday,cfg)->(labels,audit)','attach_fixed_close_labels(core,labels,cfg)->DataFrame'],
 'identity':{'recovery_key':'close','recovery_kind':'fixed_clock','recovery_clock':'14:55'},
 'no_integer_horizon':True,'no_legacy_complete_week_eligibility':True,
 'missing_future_prices_preserved':True,'missing_label_rows_preserved_in_core':True,
 'current_features_source_dtype_row_order_and_index_preserved':True,'target_is_actual_execution_pnl':False}
(out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(out/'README.md').write_text('''# 固定14:55参考标签数据接口

仅准备纯数据接口，没有新配置、模型拟合或账户执行；不修改v15烟测51份运行源码。当前尚未决定用该目标启动新模型。

```python
labels, audit = extract_fixed_close_labels(full_original_intraday, cfg)
# core必须先按原严格来源/真实无T旧股账本构造并冻结。
training_frame = attach_fixed_close_labels(core, labels, cfg)
```

cfg只消费原8个signal_times及fixed_recovery_time=14:55；旧horizons=[20,40,60]不改变本标签身份。返回标签明确recovery_key=close、recovery_kind=fixed_clock、recovery_clock=14:55，没有horizon_minutes。

extract要求唯一datetime/symbol、精确execution_time/price与recovery_time/price；原label_start/end存在时一起验证。用价格重新计算target_return，原完整周资格及旧target不参与样本保留。缺未来价格保留NaN和label_available=False；零量仍可有参考目标，audit明确不能当成交或恢复保证。非正/无限价格、错误端点、非法信号或重复身份拒绝。

attach拒绝core已有execution_/recovery_/target_/label_及旧trade_week_complete/horizon_minutes；只附本schema的标签价格和由当前键确定的时钟。缺整条label键不删core，仍有确定的下一合法分钟和当日14:55，但price/target为NaN。保留core原索引（包括重复位置索引）、行序、f32因子及NaT来源；来源资格仍由后续模型训练purge负责。该接口不把未来端点可用性引入当前推断。

输入文件SHA由未来调用者原数据加载层记录，本纯函数不读取缓存、raw文件或模型。本次只跑13个合成反例：时钟与固定身份、旧周条件忽略、两侧缺价、零量参考、错误时间与重复键、配置与非法价格、缺行左联/NaT来源、未来改写独立、不原地修改、core未来列拒绝、标签篡改和空表。

验证命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_inventory_t_close_labels.py
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile quant_backtest/inventory_t_close_labels.py scripts/test_inventory_t_close_labels.py
python -m py_compile quant_backtest/inventory_t_close_labels.py scripts/test_inventory_t_close_labels.py
```

13项通过（0.216秒），两套Python编译通过。源副本、测试日志、原51源身份和接口SHA见validation.json及manifest.json。此前原缓存/原始端点核对独立保存在 ../close_label_feasibility_20260910_151622/，本次没有重复加载全市场。
''',encoding='utf-8')
(out/'manifest.json').write_text(json.dumps({'files_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'out':str(out),'manifest_sha256':sha(out/'manifest.json'),**summary['new_source_sha256']},ensure_ascii=False,indent=2))
