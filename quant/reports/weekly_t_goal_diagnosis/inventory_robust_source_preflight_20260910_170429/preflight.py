"""v17真实运行前只核配置、既有审计来源及固定候选，不拟合或执行账户。"""
from datetime import datetime
from pathlib import Path
import json
import sys

ROOT = Path('/data/lxl/seed-emulator/topology/quant')
sys.path.insert(0, str(ROOT))
from scripts.run_inventory_robust_t import (load_configuration, validate_execution_source,
    snapshot_sources, build_inventory_robust_candidates)
from quant_backtest.balanced_t_data import validate_balanced_sources, read_balanced_fold
from quant_backtest.weekly_t_quote_store import calculate_file_sha256
from quant_backtest.annual_panel import write_json

out = ROOT / 'reports/weekly_t_goal_diagnosis' / f'inventory_robust_source_preflight_{datetime.now():%Y%m%d_%H%M%S}'
out.mkdir(parents=True)
(out/'preflight.py').write_text(Path(__file__).read_text())
full = load_configuration([])
smoke = load_configuration(['--source-report', 'reports/inventory_guarded_t_smoke/run_20260910_154550_255425',
    '--candidate-workers', '8', '--output-dir', 'reports/inventory_robust_t_smoke'])
for name, cfg in [('full', full), ('smoke', smoke)]:
    write_json(out/f'{name}_resolved_config.json', cfg)
    assert cfg['learning_model_profiles'] == ['huber_depth4', 'huber_depth6']
    assert len(build_inventory_robust_candidates(cfg)) == 37
    assert cfg['t_huber_alpha'] == .01 and cfg['training_loss'] == 'huber'
    print(name, cfg['test_years'], cfg['report_scope'], flush=True)
inputs = {}
snapshot_sources(out, smoke)
assert len(json.loads((out/'source_hashes.json').read_text())) == 56
validate_execution_source(smoke, out, inputs)
market_audit = validate_balanced_sources(smoke, inputs, ROOT)
folds = [read_balanced_fold(smoke, year, inputs) for year in smoke['test_years']]
for name, identity in inputs.items():
    assert calculate_file_sha256(Path(name)) == identity['sha256'], name
write_json(out/'verification.json', {'passed': True, 'inputs_unchanged': True,
    'input_count': len(inputs), 'inputs': inputs, 'runtime_source_count': 56,
    'no_models_fitted': True, 'no_accounts_executed': True, 'market_audit': market_audit, 'folds': folds})
print('SOURCE PREFLIGHT PASSED:', out, 'inputs=', len(inputs), flush=True)
