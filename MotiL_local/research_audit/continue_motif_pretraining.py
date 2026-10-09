"""Bounded shared-encoder continuation, retaining public DDPM mathematics.

Fresh projection FFN, DDPM MLP and optimizer states: not author-state resume.
Logs each diffusion + alternating graph/motif update. Preserves source files.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
os.chdir(MICRO)
sys.path.insert(0, str(MICRO))
from chemprop.models import build_pretrain_model
from chemprop.models.loss.loss import PointwiseLoss
from motif_instances import load_patterns
from motif_ablation import VARIANTS, make_batch, install_variant


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def official_ddpm():
    tree = ast.parse((ROOT/'upstream/run_training.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.ClassDef))
             and (isinstance(n, ast.ClassDef) and n.name in ('MLP', 'DDPM')
                  or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                      ('ATOM_LIST', 'BOND_TYPES', 'HYBRIDIZATION_TYPES') for t in n.targets))]
    # Sole DDPM interface correction: remove obsolete False positional argument.
    fixed_calls = 0
    for node in nodes:
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'gnn':
                assert len(call.args) == 7 and isinstance(call.args[1], ast.Constant) and call.args[1].value is False
                del call.args[1]; fixed_calls += 1
    assert fixed_calls == 1
    environment = {'torch': torch, 'nn': nn, 'F': F, 'Chem': Chem, 'Data': Data}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<official-ddpm-interface-fixed>', 'exec'), environment)
    return environment['MLP'], environment['DDPM']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checked_update(loss, optimizer, shared_parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss; stop instead of adjusting parameters')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Missing or nonfinite gradients')
    norm = float(torch.sqrt(sum(p.grad.square().sum() for p in shared_parameters if p.grad is not None)))
    before = [p.detach().clone() for p in shared_parameters]
    optimizer.step()
    delta = float(torch.sqrt(sum((p.detach()-old).square().sum() for p, old in zip(shared_parameters, before))))
    if not all(torch.isfinite(p).all() for group in optimizer.param_groups for p in group['params']):
        raise FloatingPointError('Nonfinite updated parameters')
    if norm <= 0 or delta <= 0:
        raise AssertionError('Shared encoder did not update')
    return norm, delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--steps', type=int, required=True, help='Explicit bounded minibatch budget')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--data', type=Path, default=ROOT.parent/'phase3_runs_20261009/zinc_subset.csv')
    cli = parser.parse_args()
    if not 2 <= cli.steps <= 200:
        parser.error('Budget must be 2..200 minibatches; two needed for both contrastive branches')
    output = cli.output.resolve()
    if output.exists():
        parser.error('Output exists; choose a fresh directory to preserve prior evidence')
    output.mkdir(parents=True)
    RDLogger.DisableLog('rdApp.*')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0' if cli.device == 'cuda' else 'cpu')
    protocol = json.loads((ROOT/'pilot_protocol.json').read_text(encoding='utf-8'))
    data_path = cli.data.resolve()
    data_manifest = json.loads((data_path.parent/'data_manifest.json').read_text(encoding='utf-8'))
    assert sha(data_path) == data_manifest['subset_sha256'] and data_manifest['subset_exact_downstream_overlap'] == 0
    checkpoint = MICRO/'dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl'
    assert sha(checkpoint) == protocol['checkpoint_sha256']
    with data_path.open(encoding='utf-8', newline='') as f:
        smiles = [row['smiles'] for row in csv.DictReader(f)]
    seed_all(cli.seed)  # after legacy imports
    args = argparse.Namespace(hidden_size=300, depth=3, bias=False, undirected=False,
        atom_messages=False, features_only=False, use_input_features=False,
        activation='ReLU', cuda=False, device=device, dataset_type='regression',
        dropout=0.0, ffn_num_layers=2)
    model = build_pretrain_model(args, 'CMPNN')
    model.encoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    install_variant(model.encoder.encoder, cli.variant)
    initial_encoder_hash = tensor_sha(model.encoder.state_dict())
    initial_projection_hash = tensor_sha(model.ffn.state_dict())
    MLP, DDPM = official_ddpm()
    mlp = MLP(300, 300, 50)
    initial_mlp_hash = tensor_sha(mlp.state_dict())
    model.to(device); mlp.to(device); model.train()
    ddpm = DDPM(model, mlp, 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    optimizer_diffusion = torch.optim.Adam(list(model.parameters())+list(mlp.parameters()), lr=3e-5)
    criterion = PointwiseLoss(args)
    patterns = load_patterns(MICRO/'chemprop/data/funcgroup.txt')
    shared = list(model.encoder.parameters())
    rng = np.random.default_rng(cli.seed)
    batch_indices = []
    while len(batch_indices) < cli.steps:
        indices = rng.permutation(len(smiles)).tolist()
        batch_indices.extend(indices[i:i+16] for i in range(0, len(indices)-15, 16))
    batch_indices = batch_indices[:cli.steps]
    manifest = {'variant': cli.variant, 'seed': cli.seed, 'steps': cli.steps, 'batch_size': 16,
        'data_sha256': sha(data_path), 'checkpoint_sha256': sha(checkpoint), 'batch_indices': batch_indices,
        'batch_sequence_sha256': hashlib.sha256(json.dumps(batch_indices).encode()).hexdigest(),
        'initial_encoder_tensor_sha256': initial_encoder_hash,
        'initial_projection_tensor_sha256': initial_projection_hash,
        'initial_ddpm_mlp_tensor_sha256': initial_mlp_hash,
        'optimizer_state': 'new Adam for each run; not author resume',
        'randomness': 'separate per-minibatch diffusion, molecule-view, motif-view seeds; same across variants. Per-instance dropout draws cannot be identical when counts differ.',
        'sources': {p.name: sha(p) for p in [Path(__file__), ROOT/'motif_ablation.py', ROOT/'upstream/run_training.py', ROOT/'upstream/cmpn.py']},
        'protocol': protocol, 'pretrain_args': {k: str(v) if k == 'device' else v for k, v in vars(args).items()}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    counts = {'diffusion': 0, 'molecule': 0, 'motif': 0, 'motif_skipped': 0}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    with (output/'updates.jsonl').open('w', encoding='utf-8') as log:
        for index, ids in enumerate(batch_indices):
            start = time.perf_counter()
            batch_smiles = [smiles[i] for i in ids]
            batch = make_batch(batch_smiles, args, patterns, cli.variant)
            seed_all(cli.seed+100000+index)
            diff_loss = ddpm.compute_loss(batch_smiles, 'pretrain', 'contrast_mol', 3, 0.3)
            diff_norm, diff_delta = checked_update(diff_loss, optimizer_diffusion, shared)
            counts['diffusion'] += 1
            task = 'contrast_mol' if index % 2 == 0 else 'contrast_fgs'
            skipped = task == 'contrast_fgs' and len(batch.all_fgs) < 2
            contrast_loss, norm, delta = None, None, None
            if skipped:
                counts['motif_skipped'] += 1
            else:
                seed_all(cli.seed+(200000 if task == 'contrast_mol' else 300000)+index)
                encoder = model.encoder.encoder
                first = model.ffn(encoder('pretrain', batch, task, 3, 0.3))
                second = model.ffn(encoder('pretrain', batch, task, 2, 0.3))
                loss = criterion(first, second)
                norm, delta = checked_update(loss, optimizer, shared)
                contrast_loss = float(loss.detach())
                counts['molecule' if task == 'contrast_mol' else 'motif'] += 1
                del first, second, loss
            if device.type == 'cuda':
                torch.cuda.synchronize()
            record = {'minibatch': index, 'task': task, 'types': len(batch.all_fgs),
                'instances': sum(map(len, batch.all_fgs.values())), 'diffusion_loss': float(diff_loss.detach()),
                'diffusion_shared_grad_norm': diff_norm, 'diffusion_encoder_delta_norm': diff_delta,
                'contrastive_loss': contrast_loss, 'contrastive_shared_grad_norm': norm,
                'contrastive_encoder_delta_norm': delta, 'motif_skipped': skipped,
                'seconds': time.perf_counter()-start}
            log.write(json.dumps(record)+'\n'); log.flush()
            print(cli.variant, index+1, '/', cli.steps, 'diff', round(record['diffusion_loss'],3),
                  task, contrast_loss, flush=True)
            del diff_loss
    encoder_state = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
    torch.save(encoder_state, output/'encoder.pkl')
    torch.save({'model': model.state_dict(), 'ddpm_mlp': mlp.state_dict(),
                'optimizer_contrastive': optimizer.state_dict(), 'optimizer_diffusion': optimizer_diffusion.state_dict(),
                'seed': cli.seed, 'steps': cli.steps}, output/'training_state.pt')
    final_hash = tensor_sha(encoder_state)
    assert initial_encoder_hash != final_hash
    assert counts['molecule'] > 0 and counts['motif'] > 0
    summary = {'counts': counts, 'seconds': time.perf_counter()-total_start,
        'initial_encoder_tensor_sha256': initial_encoder_hash, 'final_encoder_tensor_sha256': final_hash,
        'final_ddpm_mlp_tensor_sha256': tensor_sha(mlp.state_dict()),
        'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
        'checkpoint_file_sha256': sha(output/'encoder.pkl'), 'all_updates_finite': True,
        'scope': 'bounded continuation update verification; no property-performance claim'}
    (output/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
