"""Bounded CUDA motif-branch timing/memory; no optimizer or parameter update."""
import csv
import json
import os
from pathlib import Path
import sys
import time
from argparse import Namespace
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
os.chdir(MICRO)
sys.path.insert(0, str(MICRO))
from chemprop.models.cmpn import CMPN
from chemprop.models.loss.loss import PointwiseLoss
from motif_instances import load_patterns
from motif_ablation import VARIANTS, make_batch, install_variant, motif_views

if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; CPU timings are in ablation_correctness.json')
torch.set_num_threads(2)
args = Namespace(hidden_size=300, depth=3, bias=False, undirected=False,
                 atom_messages=False, features_only=False, use_input_features=False,
                 activation='ReLU', cuda=False, device=torch.device('cuda:0'))
model = CMPN(args)
checkpoint = MICRO/'dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl'
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
model.to(args.device)
patterns = load_patterns(MICRO/'chemprop/data/funcgroup.txt')
with (MICRO/'data/esol.csv').open(encoding='utf-8-sig', newline='') as f:
    smiles = [row['smiles'] for row in csv.DictReader(f)][:16]
report = {'scope': 'two fixed ESOL batches; two views plus backward, no optimizer update',
          'device': torch.cuda.get_device_name(0), 'dtype': 'float32',
          'dropout': [0.3, 0.3], 'measurements': [],
          'limitations': 'Single measurement per condition, not full diffusion pretraining memory or throughput.'}
loss_fn = PointwiseLoss(args)
for size in [8, 16]:
    for variant in VARIANTS:
        install_variant(model.encoder, variant)
        batch = make_batch(smiles[:size], args, patterns, variant)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with torch.no_grad():
            motif_views(model.encoder, batch, 3)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_bytes = torch.cuda.memory_allocated()
        torch.manual_seed(2024)
        start = time.perf_counter()
        views = motif_views(model.encoder, batch, 3)
        loss = loss_fn(*views)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        report['measurements'].append({'batch_size': size, 'variant': variant,
            'atoms': sum(Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles[:size]),
            'types': len(batch.all_fgs), 'instances': sum(map(len, batch.all_fgs.values())),
            'two_view_backward_seconds': elapsed,
            'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
            'incremental_peak_MiB': (torch.cuda.max_memory_allocated()-initial_bytes)/2**20,
            'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20})
        print(size, variant, round(elapsed, 3), flush=True)
        del views, loss
        model.zero_grad(set_to_none=True)
(ROOT/'ablation_cuda_budget.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Finished bounded benchmark; no optimizer update.')
