"""保存新首腿限价复核的测试/源码身份，拒绝旧48源变化。"""
from pathlib import Path
import hashlib,json,shutil,subprocess
from datetime import datetime
root=Path.cwd();report=root/'reports/inventory_learning_t/run_20260910_151232_841321'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen=json.loads((report/'source_hashes.json').read_text());assert all(sha(root/name)==value and sha(report/'source_snapshot'/name)==value for name,value in frozen.items())
out=root/'reports/weekly_t_goal_diagnosis'/f'guarded_engine_validation_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir()
Path('/tmp/guarded_engine_validation_path.txt').write_text(str(out))
for source in ['quant_backtest/guarded_t_engine.py','scripts/test_guarded_t_engine.py']:
 target=out/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/source,target)
shutil.copy2('/tmp/guarded_t_engine_tests.log',out/'unit_tests.log')
for src in ['/tmp/create_guarded_engine.py','/tmp/archive_guarded_engine_validation.py']:
 shutil.copy2(src,out/Path(src).name)
diff=subprocess.run(['diff','-u','quant_backtest/dynamic_t_engine.py','quant_backtest/guarded_t_engine.py'],capture_output=True,text=True)
assert diff.returncode==1;(out/'engine.diff').write_text(diff.stdout)
validation={'passed':True,'tests':15,'quant_python':'/home/lxl/anaconda3/envs/quant/bin/python','quant_compile_passed':True,'system_python_compile_passed':True,
 'source_sha256':{name:sha(root/name) for name in ['quant_backtest/guarded_t_engine.py','scripts/test_guarded_t_engine.py']},
 'old_runtime_files':len(frozen),'all_old_runtime_and_snapshots_unchanged':True,
 'off_control':'all original result tables, indices and metrics exact under 3 exit modes with partial recovery and min commission',
 'actual_market_case_scope':'only two known raw quotes embedded as a synthetic fixture; no full market replay or return claim',
 'read_outer_returns':False,'new_model_fit':False,
 'api':'run_guarded_portfolio(selections,daily,intraday,decisions,cfg,recovery_bars=None,manual_entries=None)',
 'new_switch':'t_recheck_exit_side_at_execution default False; explicit bool required',
 'new_result_table':'first_leg_exit_side_audit','planner':'reuse unchanged dynamic_t_engine.plan_dynamic_entries',
 'audit_population':'own quote fill quantity > 0, before cash/pending sizing; includes disabled control rows; not actual pair count'}
(out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
(out/'worklog.md').write_text('''# 新执行条件验证

仅新增 guarded_t_engine.py 和 test_guarded_t_engine.py，旧48源/旧引擎/缓存/模型保持原样。

执行命令：

```bash
/home/lxl/anaconda3/envs/quant/bin/python scripts/test_guarded_t_engine.py > /tmp/guarded_t_engine_tests.log 2>&1
/home/lxl/anaconda3/envs/quant/bin/python -m py_compile quant_backtest/guarded_t_engine.py scripts/test_guarded_t_engine.py
python -m py_compile quant_backtest/guarded_t_engine.py scripts/test_guarded_t_engine.py
```

15项最终测试通过，耗时6.719秒。初轮14项通过、1项manual fixture缺少原API必填signal_datetime而报错；只补新测试fixture后全套通过，未改变生产人工接口。真实案例fixture的信号high核成19.78。

关闭控制对三种退出模式、最低佣金和部分恢复的全部原result字段逐值精确相等。开关开启后验证买先跌停/卖先涨停取消、真实限价优先、6%普通平价允许、5%候选平价拒绝、零数量检查不要求虚拟反向容量、取消不消耗现金/费用/旧股/次数/预留、同刻现金留给下一股票、后续请求继续、未来报价独立、已有pending不被取消/隐藏、原次分钟检查、原planner与人工当日不可卖，以及布尔配置拒绝错误字符串。

本条件沿用现有“下一分钟close为到达报价”的成交模型。它不是上一信号分钟已知因子，也不依赖后来是否恢复。完整OHLCV只是该分钟的模拟报价证据，不证明真实盘口队列和零延迟撤单一定可实现。新报告应称新增执行报价条件，不能回写旧结果或当作恢复保证。
''',encoding='utf-8')
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
(out/'manifest.json').write_text(json.dumps({'files_sha256':manifest},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'out':str(out),'validation_sha256':sha(out/'validation.json'),'manifest_sha256':sha(out/'manifest.json'),**validation['source_sha256']},ensure_ascii=False,indent=2))
