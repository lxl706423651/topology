"""归档v16新报告器验证；不重复测试/账户，不改历史源及正式产物。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


sources = {}
for name in ['quant_backtest/inventory_t_close_reporting.py', 'scripts/test_inventory_t_close_reporting.py']:
    target = OUT / 'source_snapshot' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
    sources[name] = sha(target)
shutil.copy2('/tmp/inventory_t_close_reporting_tests.log', OUT / 'unit_tests.log')
old = ROOT / 'reports/inventory_guarded_t_smoke/run_20260910_154550_255425'
frozen = json.loads((old / 'source_hashes.json').read_text())
for name, digest in frozen.items():
    assert sha(ROOT / name) == digest and sha(old / 'source_snapshot' / name) == digest
result = {'passed': True, 'tests': 8, 'unit_elapsed_seconds': 3.967,
          'quant_python_compile': True, 'system_python_compile': True,
          'new_source_sha256': sources, 'old_runtime_files_unchanged': len(frozen),
          'commands': ['/home/lxl/anaconda3/envs/quant/bin/python scripts/test_inventory_t_close_reporting.py',
                       '/home/lxl/anaconda3/envs/quant/bin/python -m py_compile quant_backtest/inventory_t_close_reporting.py scripts/test_inventory_t_close_reporting.py',
                       'python -m py_compile quant_backtest/inventory_t_close_reporting.py scripts/test_inventory_t_close_reporting.py'],
          'fitted_models': 0, 'real_market_backtests': 0, 'synthetic_engine_accounts': 1,
          'scope': 'new_report_identity_and_shared_accounting_contract_only'}
(OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
manifest = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'verification_sha256': sha(OUT / 'verification.json'), 'manifest_sha256': sha(OUT / 'manifest.json')}, indent=2))
