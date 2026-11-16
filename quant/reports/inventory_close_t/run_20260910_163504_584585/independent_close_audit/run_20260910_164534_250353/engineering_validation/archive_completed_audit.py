"""Archive the completed read-only v16 full audit; never edit original results."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/data/lxl/seed-emulator/topology/quant')
OUT = ROOT / 'reports/inventory_close_t/run_20260910_163504_584585/independent_close_audit/run_20260910_164534_250353'
LOG = Path('/tmp/inventory_close_full_independent_audit.log')
EXPECTED = '5d258be69f6dbad541f81635c86d0489d5162d6aaa8b6a69d6126f4cd8d54c3b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification = OUT / 'verification.json'
    before = sha(verification)
    result = json.loads(verification.read_text())
    for key in ['passed', 'complete', 'all_read_inputs_unchanged', 'audit_dependencies_unchanged']:
        assert result[key] is True, (key, result.get(key))
    assert result['script_sha256'] == EXPECTED
    assert sha(ROOT / 'scripts/audit_inventory_close_t.py') == EXPECTED
    assert result['candidate_accounts'] == 148 and result['outer_accounts'] == 0
    assert len(result['models']) == 8
    # Compile/test provenance is the already accepted identical audit program,
    # while this full invocation also records its own self-test results.
    smoke = ROOT / 'reports/inventory_close_t_smoke/run_20260910_162815_034893/independent_close_audit/run_20260910_163129_614458/engineering_validation/validation_summary.json'
    validation = json.loads(smoke.read_text())
    assert validation['audit_script_sha256'] == EXPECTED
    assert validation['dual_compile_passed'] and validation['audit_exit_code'] == 0
    tests = result['self_tests']
    assert all(item['passed'] for item in tests.values())
    assert tests['close']['tests'] + tests['guard']['tests'] + tests['models']['cases'] == 26
    engineering = OUT / 'engineering_validation'
    engineering.mkdir(exist_ok=False)
    shutil.copy2(LOG, OUT / 'audit.log')
    raw_log = LOG.read_text()
    (engineering / 'self_tests.log').write_text(raw_log.split('57源与双来源')[0])
    validation['verification_sha256'] = before
    validation['audit_exit_code'] = 0
    validation['compile_evidence_source'] = str(smoke)
    validation['compile_evidence_source_sha256'] = sha(smoke)
    (engineering / 'validation_summary.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(Path(__file__), engineering / 'archive_completed_audit.py')
    manifest = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
                if p.is_file() and p.name != 'artifact_sha256.json'}
    (OUT / 'artifact_sha256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    assert sha(verification) == before
    print(json.dumps({'directory': str(OUT), 'verification_sha256': before,
                      'artifact_files': len(manifest),
                      'artifact_manifest_sha256': sha(OUT / 'artifact_sha256.json')}, indent=2))


if __name__ == '__main__':
    main()
