"""保存独立模型审计辅助的源码/自检证据；不执行账户、拟合或重复市场核查。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
source = ROOT / 'scripts/inventory_close_audit_models.py'
shutil.copy2(source, OUT / source.name)
digest = hashlib.sha256(source.read_bytes()).hexdigest()
result = {'passed': True, 'scope': 'independent_model_helper_engineering_only',
          'self_test_cases': 6, 'self_test_command': '/home/lxl/anaconda3/envs/quant/bin/python scripts/inventory_close_audit_models.py --self-test',
          'quant_python_compile_passed': True, 'system_python_compile_passed': True,
          'script_sha256': digest, 'models_fit': 0, 'market_read': False,
          'full_real_smoke_audit_completed_in_this_record': False,
          'consumer': 'scripts/audit_inventory_close_t.py',
          'note': '完整真实烟测应随后使用主审计器的来源/队列/当前特征与本辅助联合核对，此处不预先宣称其通过。'}
(OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
            if p.is_file() and p.name != 'manifest.json'}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'script_sha256': digest, 'verification_sha256': hashlib.sha256((OUT / 'verification.json').read_bytes()).hexdigest()}, indent=2))
