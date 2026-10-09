"""One CPU epoch to verify new validation-only entry; not efficacy data."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
OUT = ROOT.parent/'phase4_runs_20261009'
sys.path.insert(0, str(MICRO))
from run_motil_windows import TASKS

command = list(TASKS['esol'])
command[0] = str(ROOT/'downstream_validation_only.py')
def replace(key, value):
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command.extend([key, str(value)])
for key, value in [('--epochs', 1), ('--num_runs', 1), ('--exp_id', 'cpu_safety'),
                   ('--exp_name', 'validation_only_preflight'), ('--dump_path', str(OUT/'preflight_models'))]:
    replace(key, value)
command += ['--no_cuda', '--save_smiles_splits']
report = OUT/'validation_entry_preflight.json'
if report.exists():
    raise SystemExit('Preflight report exists; preserve it')
environment = dict(os.environ, MOTIL_RUN_SEED='2024', PYTHONHASHSEED='2024',
                   MOTIL_VALIDATION_REPORT=str(report), PYTHONIOENCODING='utf-8',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
with (OUT/'validation_entry_preflight.log').open('w', encoding='utf-8') as logfile:
    subprocess.run([sys.executable, *command], cwd=MICRO, env=environment,
                   stdout=logfile, stderr=subprocess.STDOUT, check=True)
result = json.loads(report.read_text(encoding='utf-8'))
assert result['epochs']==1 and result['test_evaluated'] is False
assert result['prediction_guard']=={'test_prediction_attempts':0,'allowed_predictions':3}
result['scope']='CPU one-epoch validation-entry safety check only; excluded from phase-4 efficacy comparison'
(ROOT/'validation_only_entry_safety.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Validation-only CPU entry passed; zero test predictions and three allowed predictions.')
