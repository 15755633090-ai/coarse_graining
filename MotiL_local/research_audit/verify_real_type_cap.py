"""Verify actual uncapped vs capped encoder output on a real ZINC batch."""
import json
import os
from pathlib import Path
import sys
from argparse import Namespace
import torch

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
os.chdir(MICRO)
sys.path.insert(0, str(MICRO))
from chemprop.models.cmpn import CMPN
from motif_ablation import make_batch, install_variant
from motif_instances import load_patterns

torch.set_num_threads(2)
args = Namespace(hidden_size=300, depth=3, bias=False, undirected=False,
                 atom_messages=False, features_only=False, use_input_features=False,
                 activation='ReLU', cuda=False)
model = CMPN(args)
model.load_state_dict(torch.load(MICRO/'dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl',
                                map_location='cpu', weights_only=True), strict=True)
manifest = json.loads((ROOT.parent/'phase3_runs_20261009/data_manifest.json').read_text(encoding='utf-8'))
example = manifest['real_over_ten_example']
patterns = load_patterns(MICRO/'chemprop/data/funcgroup.txt')
rows = {}
with torch.no_grad():
    for variant in ['B2a-2', 'B2b']:
        install_variant(model.encoder, variant)
        batch = make_batch(example['smiles'], args, patterns, variant)
        vectors = model.encoder('pretrain', batch, 'contrast_fgs', 3, 0.0)
        assert torch.isfinite(vectors).all() and vectors.shape[0] == len(batch.all_fgs)
        rows[variant] = {'types': list(batch.all_fgs), 'instances': sum(map(len, batch.all_fgs.values())),
                         'vector_shape': list(vectors.shape)}
assert len(rows['B2a-2']['types']) == 10 and len(rows['B2b']['types']) > 10
report = {'scope': 'real ZINC batch inference only, author weights',
          'batches_over_ten': manifest['batches_over_ten'], 'batches_total': manifest['batches_total'],
          'subset_sha256': manifest['subset_sha256'], 'real_example': example, 'outputs': rows}
(ROOT/'real_type_cap_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print({variant: row['vector_shape'] for variant, row in rows.items()})
