"""ESOL one-epoch connection check; explicitly not an efficacy benchmark."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
OUT = ROOT.parent/'phase3_runs_20261009'
sys.path.insert(0, str(MICRO))
from run_motil_windows import TASKS, CHECKPOINT


def replace(command, key, value):
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command.extend([key, str(value)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = OUT/'downstream_wiring_v2'
    if output.exists():
        raise SystemExit('Downstream output exists; preserve previous results')
    output.mkdir()
    checkpoints = {'author': CHECKPOINT,
        **{v: OUT/f'short_{v}'/'encoder.pkl' for v in ['B0', 'B1', 'B2b']}}
    environment = dict(os.environ, MOTIL_RUN_SEED='2024', PYTHONHASHSEED='2024',
                       PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    plans = []
    for variant, checkpoint in checkpoints.items():
        assert checkpoint.exists()
        command = list(TASKS['esol'])
        command[0] = str(ROOT/'downstream_phase3_entry.py')
        for key, value in [('--epochs', 1), ('--num_runs', 1), ('--seed', 2024),
                           ('--exp_id', variant), ('--exp_name', 'esol_one_epoch_wiring'),
                           ('--dump_path', str(output/'models')), ('--checkpoint_path', checkpoint)]:
            replace(command, key, value)
        command += ['--gpu', '0', '--save_smiles_splits']
        plans.append({'variant': variant, 'command': [sys.executable, *command],
                      'checkpoint_sha256': sha(checkpoint), 'scope': 'one epoch only; not author 100-epoch ESOL protocol'})
    (output/'manifest.json').write_text(json.dumps(plans, indent=2), encoding='utf-8')
    results = []
    for plan in plans:
        print('START downstream wiring', plan['variant'], flush=True)
        logfile = output/(plan['variant']+'.log')
        with logfile.open('w', encoding='utf-8') as log:
            subprocess.run(plan['command'], cwd=MICRO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = logfile.read_text(encoding='utf-8')
        assert 'Effective PyTorch initialization seed: 2024' in text
        assert 'Loaded all' in text
        score = re.search(r'Overall test rmse = ([\d.]+)', text)
        assert score
        results.append({'variant': plan['variant'], 'one_epoch_wiring_test_rmse': float(score[1]),
                        'warning': 'Do not rank methods by this connection test.'})
        print('DONE downstream wiring', plan['variant'], flush=True)
    # Exact saved split contents must match across all four conditions.
    split_files = {}
    for path in (output/'models').rglob('*smiles.csv'):
        variant = next((name for name in checkpoints if name in path.parts), None)
        if variant is not None:
            split_files.setdefault(variant, {})[path.name] = sha(path)
    assert len(split_files) == 4 and all(v == next(iter(split_files.values())) for v in split_files.values())
    report = {'scope': 'one-epoch ESOL downstream wiring only; no performance inference',
              'same_saved_split_hashes': split_files, 'results': results}
    (output/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT/'phase3_downstream_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
