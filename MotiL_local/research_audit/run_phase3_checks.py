"""Run bounded 2-step wiring and 20-step paired checks; stop on any failure."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'phase3_runs_20261009'
plans = [('smoke', variant, 2) for variant in ['B0', 'B1', 'B2a-1', 'B2a-2', 'B2b']]
plans += [('short', variant, 20) for variant in ['B0', 'B1', 'B2b']]
results = []
environment = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
for stage, variant, steps in plans:
    directory = OUT/f'{stage}_{variant}'
    completion = directory/'complete.json'
    if not completion.exists():
        command = [sys.executable, str(ROOT/'continue_motif_pretraining.py'),
                   '--variant', variant, '--steps', str(steps), '--seed', '2024', '--output', str(directory)]
        print('START', stage, variant, steps, flush=True)
        with (OUT/f'{stage}_{variant}.log').open('w', encoding='utf-8') as logfile:
            subprocess.run(command, env=environment, stdout=logfile, stderr=subprocess.STDOUT, check=True)
    summary = json.loads(completion.read_text(encoding='utf-8'))
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['steps'] == steps and manifest['variant'] == variant and manifest['seed'] == 2024
    assert summary['counts']['diffusion'] == steps
    assert summary['counts']['molecule'] == steps//2
    assert summary['counts']['motif']+summary['counts']['motif_skipped'] == steps//2
    assert summary['initial_encoder_tensor_sha256'] != summary['final_encoder_tensor_sha256']
    results.append({'stage': stage, 'variant': variant, 'manifest': manifest, 'summary': summary})
    print('DONE', stage, variant, round(summary['seconds'], 2), 'seconds', flush=True)
    (OUT/'wiring_progress.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
for stage in ['smoke', 'short']:
    group = [r['manifest'] for r in results if r['stage'] == stage]
    for key in ['batch_sequence_sha256', 'initial_encoder_tensor_sha256',
                'initial_projection_tensor_sha256', 'initial_ddpm_mlp_tensor_sha256']:
        assert len(set(r[key] for r in group)) == 1, (stage, key)
report = {'scope': '2-minibatch five-group wiring + 20-minibatch three-group stability; not 200-step pilot',
          'paired_initialization_and_batch_checks': 'passed',
          'runs': [{'stage': r['stage'], 'variant': r['variant'], **r['summary']} for r in results]}
(ROOT/'phase3_update_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('All bounded update and paired-start checks passed.', flush=True)
