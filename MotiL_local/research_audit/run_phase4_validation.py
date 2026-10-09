"""Fixed 9 continuation + 12 full ESOL validation-only runs; fail closed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
OUT = ROOT.parent/'phase4_runs_20261009'
sys.path.insert(0, str(MICRO))
from run_motil_windows import TASKS, CHECKPOINT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(command, key, value):
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command.extend([key, str(value)])


def main():
    if OUT.exists():
        raise SystemExit('Phase 4 output already exists; preserve it, do not auto-restart')
    OUT.mkdir()
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    pretrain, downstream = [], []
    for seed in [2024, 2025, 2026]:
        for variant in ['B0', 'B1', 'B2b']:
            directory = OUT/f'{variant}_seed{seed}'
            command = [sys.executable, str(ROOT/'continue_motif_pretraining.py'), '--variant', variant,
                       '--steps', '200', '--seed', str(seed), '--output', str(directory)]
            pretrain.append({'variant': variant, 'seed': seed, 'directory': str(directory), 'command': command})
        for variant in ['author', 'B0', 'B1', 'B2b']:
            checkpoint = CHECKPOINT if variant=='author' else OUT/f'{variant}_seed{seed}'/'encoder.pkl'
            report = OUT/f'esol_{variant}_seed{seed}_validation.json'
            command = list(TASKS['esol'])
            command[0] = str(ROOT/'downstream_validation_only.py')
            for key, value in [('--num_runs', 1), ('--seed', 2024), ('--epochs', 100),
                               ('--exp_id', f'{variant}_seed{seed}'), ('--exp_name', 'phase4_esol'),
                               ('--dump_path', str(OUT/'downstream_models')), ('--checkpoint_path', checkpoint)]:
                replace(command, key, value)
            command += ['--gpu', '0', '--save_smiles_splits']
            downstream.append({'variant': variant, 'seed': seed, 'checkpoint': str(checkpoint),
                               'report': str(report), 'command': [sys.executable, *command]})
    manifest = {'scope': 'exploratory matched author-encoder continuation; validation only',
        'pretraining_steps': 200, 'pretraining_batch_size': 16, 'pretraining_subset_size': 4096,
        'downstream_epochs': 100, 'split_seed': 2024, 'seeds': [2024,2025,2026],
        'test_policy': 'no test prediction or scoring in any run; no test-driven tuning',
        'comparison': 'paired continuation+head seeds; not independent from-scratch encoder seeds',
        'data_manifest': json.loads((ROOT/'phase3_data_manifest.json').read_text(encoding='utf-8')),
        'author_checkpoint_sha256': sha(CHECKPOINT),
        'sources': {str(p.relative_to(ROOT.parent)): sha(p) for p in
            [Path(__file__), ROOT/'downstream_validation_only.py', ROOT/'continue_motif_pretraining.py',
             ROOT/'motif_ablation.py', MICRO/'run_motil_windows.py']},
        'protocol': json.loads((ROOT/'pilot_protocol.json').read_text(encoding='utf-8')),
        'pretrain_plans': pretrain, 'downstream_plans': downstream,
        'decision': 'execute all prespecified runs; do not extend B2a or alter parameters based on interim results'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    state = {'status': 'running', 'completed_pretraining': [], 'completed_downstream': [], 'active': None}
    def save_state():
        (OUT/'progress.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    save_state()
    try:
        for plan in pretrain:
            name = f"{plan['variant']}_seed{plan['seed']}"
            state['active'] = 'pretrain '+name; save_state()
            print('START pretrain', name, flush=True)
            with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run(plan['command'], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            directory = Path(plan['directory'])
            summary = json.loads((directory/'complete.json').read_text(encoding='utf-8'))
            counts = summary['counts']
            assert counts['diffusion']==200 and counts['molecule']==100 and counts['motif']+counts['motif_skipped']==100
            state['completed_pretraining'].append({'variant': plan['variant'], 'seed': plan['seed'], **summary})
            save_state(); print('DONE pretrain', name, round(summary['seconds'],1), flush=True)
        for seed in manifest['seeds']:
            records = [json.loads((OUT/f'{v}_seed{seed}'/'manifest.json').read_text(encoding='utf-8')) for v in ['B0','B1','B2b']]
            for key in ['batch_sequence_sha256','initial_encoder_tensor_sha256','initial_projection_tensor_sha256','initial_ddpm_mlp_tensor_sha256']:
                assert len({r[key] for r in records})==1, (seed,key)
        for plan in downstream:
            name = f"esol_{plan['variant']}_seed{plan['seed']}"
            state['active'] = name; save_state()
            print('START downstream', name, flush=True)
            start = time.perf_counter()
            child_env = dict(environment, MOTIL_RUN_SEED=str(plan['seed']), PYTHONHASHSEED=str(plan['seed']),
                             MOTIL_VALIDATION_REPORT=plan['report'])
            with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run(plan['command'], cwd=MICRO, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(Path(plan['report']).read_text(encoding='utf-8'))
            assert result['epochs']==100 and result['test_evaluated'] is False
            assert result['prediction_guard']['test_prediction_attempts']==0
            assert result['initialization_seed']==plan['seed'] and result['split_seed']==2024
            logtext = (OUT/f'{name}.log').read_text(encoding='utf-8')
            assert 'Loaded all' in logtext and 'Overall test rmse' not in logtext
            split_hashes = {p.name: sha(p) for p in Path(result['split_directory']).glob('*smiles.csv')}
            assert len(split_hashes)==3
            if state['completed_downstream']:
                assert split_hashes == state['completed_downstream'][0]['split_sha256']
            state['completed_downstream'].append({'variant': plan['variant'], 'seed': plan['seed'],
                **result, 'seconds': time.perf_counter()-start, 'encoder_sha256': sha(plan['checkpoint']),
                'selected_checkpoint_sha256': sha(result['checkpoint_path']), 'split_sha256': split_hashes})
            save_state(); print('DONE downstream', name, 'validation', round(result['best_validation_rmse'],6), flush=True)
        aggregates = {}
        for variant in ['author','B0','B1','B2b']:
            values = [r['best_validation_rmse'] for r in state['completed_downstream'] if r['variant']==variant]
            aggregates[variant] = {'validation_mean': float(np.mean(values)), 'validation_sample_std': float(np.std(values,ddof=1)), 'per_seed': values}
        paired = {}
        for left, right in [('author','B0'),('B0','B1'),('B1','B2b')]:
            differences = np.array(aggregates[right]['per_seed'])-np.array(aggregates[left]['per_seed'])
            paired[f'{right}_minus_{left}'] = {'per_seed': differences.tolist(), 'mean': float(differences.mean()),
                'sample_std': float(differences.std(ddof=1)), 'meaning': 'negative means lower validation RMSE; no test/generalization claim'}
        state['status']='complete'; state['active']=None; save_state()
        report = {'scope':manifest['scope'], 'test_evaluated':False, 'aggregates':aggregates,
                  'paired_differences':paired, 'pretraining':state['completed_pretraining'],
                  'downstream':state['completed_downstream']}
        (OUT/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (ROOT/'phase4_validation_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('All phase-4 fixed-budget runs complete; test set untouched.',flush=True)
    except BaseException as error:
        state['status']='failed'; state['error']=repr(error); save_state()
        raise


if __name__=='__main__':
    main()
