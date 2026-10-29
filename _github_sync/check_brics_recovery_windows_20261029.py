"""Exercise checkpoint commit windows using actual deterministic pretraining."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from motif.brics_atom_diffusion import pretrain, training

BASE = ROOT / 'motif/brics_atom_diffusion/runs/diagnostics_20261029'
CORPUS = ROOT / 'motif/brics_atom_diffusion/runs/smoke_20261029_inputs/esol_training_smiles.csv'
if not CORPUS.exists():
    import pandas as pd
    task_root = ROOT / 'data/ogbg_molesol'
    frame = pd.read_csv(task_root / 'mapping/mol.csv.gz')
    indices = pd.read_csv(task_root / 'split/scaffold/train.csv.gz', header=None).iloc[:96, 0].tolist()
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[indices][['smiles']].to_csv(CORPUS, index=False)
BASE.mkdir(parents=True, exist_ok=False)


def arguments(name, resume=False):
    return ['pretrain', '--smiles-csv', str(CORPUS), '--valid-fraction', '0.25', '--limit', '16',
            '--seed', '7', '--epochs', '2', '--patience', '2', '--batch-size', '8', '--hidden-dim', '16',
            '--layers', '2', '--dropout', '0', '--output-dir', str(BASE / name)] + (['--resume'] if resume else [])


def run(name, resume=False):
    sys.argv = arguments(name, resume)
    pretrain.main()


run('continuous')
reference = training.load_checkpoint(BASE / 'continuous/latest.pt')
write = training.atomic_write
rows = []
for name in ('missing_first_export_and_rogue_cache', 'before_latest_commit', 'after_latest_commit'):
    def injected(path, value, binary=False):
        epoch = value.get('epoch') if isinstance(value, dict) else None
        if name == 'missing_first_export_and_rogue_cache' and path.name == 'best.pt' and epoch == 1:
            raise RuntimeError('Injected checkpoint window')
        if name == 'before_latest_commit' and path.name == 'latest.pt' and epoch == 2:
            raise RuntimeError('Injected checkpoint window')
        write(path, value, binary)
        if name == 'after_latest_commit' and path.name == 'latest.pt' and epoch == 2:
            raise RuntimeError('Injected checkpoint window')
    with patch.object(training, 'atomic_write', side_effect=injected):
        try:
            run(name)
        except RuntimeError as error:
            assert str(error) == 'Injected checkpoint window', str(error)
        else:
            raise AssertionError('No injected interruption')
    folder = BASE / name
    latest = training.load_checkpoint(folder / 'latest.pt')
    if name == 'missing_first_export_and_rogue_cache':
        assert latest['epoch'] == 1 and not (folder / 'best.pt').exists()
        rogue = deepcopy(latest['best_checkpoint'])
        rogue.update(epoch=999, best_epoch=999, best_score=-999.)
        next(iter(rogue['model'].values())).fill_(123.)
        write(folder / 'best.pt', rogue, binary=True)
        write(folder / 'history.json', [{'epoch': 999}])
    elif name == 'before_latest_commit':
        assert latest['epoch'] == 1
        assert training.load_checkpoint(folder / 'best.pt')['epoch'] == latest['best_epoch']
    else:
        assert latest['epoch'] == 2
    run(name, resume=True)
    restored = training.load_checkpoint(folder / 'latest.pt')
    exported = training.load_checkpoint(folder / 'best.pt')
    authority = training.authoritative_best(restored)
    assert restored['history'] == reference['history']
    for key, value in reference['model'].items():
        assert torch.equal(value, restored['model'][key]), (name, key)
    for key in ('torch_rng', 'noise_rng', 'loader_rng'):
        assert torch.equal(reference[key], restored[key]), (name, key)
    assert exported['epoch'] == authority['epoch']
    for key in authority['model']:
        assert torch.equal(exported['model'][key], authority['model'][key])
    rows.append({'case': name, 'committed_epoch_at_interruption': latest['epoch'],
                 'weights_history_rng_match_continuous': True, 'best_export_matches_authority': True})
record = reference['history'][-1]
diagnostics = record['validation']['recovery']
assert set(diagnostics['all']['all']) == {'all', 'element', 'charge', 'aromatic'}
assert reference['noise_schedule']['alpha_bar_T'] < 0.0001
report = {'passed': True, 'cases': rows, 'noise_schedule': reference['noise_schedule'],
          'recovery_diagnostics_present': True, 'test_evaluated': False, 'formal_performance_claims': False}
write(BASE / 'acceptance.json', report)
print(json.dumps(report))
