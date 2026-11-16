"""归档本轮纯复用模块/测试及一次真实验证，不执行测试或修改冻结运行源。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


names = ['quant_backtest/inventory_t_frozen_cohort.py', 'scripts/test_inventory_t_frozen_cohort.py']
sources = {}
for name in names:
    destination = OUT / 'source_snapshot' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, destination)
    sources[name] = sha(destination)
runtime = ROOT / 'reports/inventory_guarded_t_smoke/run_20260910_154550_255425'
frozen = json.loads((runtime / 'source_hashes.json').read_text())
for name, expected in frozen.items():
    assert sha(ROOT / name) == expected
    assert sha(runtime / 'source_snapshot' / name) == expected
summary = {
    'new_module_and_test_sha256': sources, 'frozen_runtime_sources_unchanged': len(frozen),
    'frozen_runtime_reference': str(runtime), 'independent_read_only_review': 'inspect_entry: no substantive issue',
    'unit_tests': {'passed': True, 'count': 12, 'elapsed_seconds': .466,
                   'evidence': 'recorded exec tool completion before real validation; not rerun during archive',
                   'command': '/home/lxl/anaconda3/envs/quant/bin/python scripts/test_inventory_t_frozen_cohort.py'},
    'compile_checks': {'quant_python': True, 'system_python': True,
                       'commands': ['/home/lxl/anaconda3/envs/quant/bin/python -m py_compile '+' '.join(names),
                                    'python -m py_compile '+' '.join(names)]},
    'one_real_validation_command': '/home/lxl/anaconda3/envs/quant/bin/python '+str(OUT.relative_to(ROOT) / 'validate_real_reuse.py'),
    'implemented_new_model_or_strategy': False,
}
(OUT / 'source_identity.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
manifest = {str(path.relative_to(OUT)): sha(path) for path in sorted(OUT.rglob('*'))
            if path.is_file() and path.name != 'manifest.json' and '__pycache__' not in path.parts}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'source_sha256': sources, 'verification_sha256': sha(OUT / 'verification.json'),
                  'manifest_sha256': sha(OUT / 'manifest.json'), 'frozen_sources_unchanged': len(frozen)}, indent=2))
