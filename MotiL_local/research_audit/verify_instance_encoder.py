"""CPU inference-only integration check with author checkpoint; no optimizer."""
import hashlib
import json
import os
from pathlib import Path
import sys
from argparse import Namespace
import torch

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent / 'MotiL_micromolecule'
os.chdir(MICRO)  # Legacy code reads SMARTS relative to cwd.
sys.path.insert(0, str(MICRO))
from chemprop.models.cmpn import CMPN
from chemprop.features import mol2graph
from motif_instances import load_patterns, build_instance_graph, encode_instances

torch.set_num_threads(2)
checkpoint = MICRO / 'dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
args = Namespace(hidden_size=300, depth=3, bias=False, undirected=False,
                 atom_messages=False, features_only=False, use_input_features=False,
                 activation='ReLU', cuda=False)
model = CMPN(args)
model.load_state_dict(state, strict=True)
model.eval()
patterns = load_patterns(MICRO / 'chemprop/data/funcgroup.txt')
smiles = ['CC(=O)OCCOC(=O)C', 'CCC(=O)OCC']
graphs = [build_instance_graph(s, patterns) for s in smiles]
batch = mol2graph(smiles, args)
with torch.no_grad():
    baseline = model.encoder('pretrain', batch, 'contrast_fgs', args.depth, 0.0)
    vectors, records = encode_instances(model.encoder, batch, graphs, args.depth)
    reversed_vectors, reversed_records = encode_instances(
        model.encoder, mol2graph(smiles[::-1], args), graphs[::-1], args.depth)
    # Confirm retaining instances reproduces the old selected instance vectors.
    errors, batch_errors = [], []
    class SingleMotif:
        def __init__(self, kind, selected):
            self.kind, self.selected = kind, selected
        def get_components(self):
            return (*batch.get_components()[:7], {self.kind: self.selected})
    for row, (kind, selected) in enumerate(batch.all_fgs.items()):
        for i, record in enumerate(records):
            offset = batch.a_scope[record['molecule_id']][0]
            if record['type'] == kind and [offset + a for a in record['atoms']] == selected[0]:
                reference = model.encoder('pretrain', SingleMotif(kind, selected),
                                          'contrast_fgs', args.depth, 0.0)[0]
                errors.append(float((reference - vectors[i]).abs().max()))
                batch_errors.append(float((baseline[row] - vectors[i]).abs().max()))
                break
        else:
            raise AssertionError('Missing legacy instance in prototype')
    reverse_errors = []
    for i, record in enumerate(records):
        match = next(j for j, other in enumerate(reversed_records)
                     if other['molecule_id'] == 1 - record['molecule_id']
                     and other['type_id'] == record['type_id'] and other['atoms'] == record['atoms'])
        reverse_errors.append(float((vectors[i] - reversed_vectors[match]).abs().max()))
assert len(records) == sum(len(g['instances']) for g in graphs)
assert torch.isfinite(vectors).all()
assert max(errors, default=0) < 1e-5
assert max(reverse_errors, default=0) < 1e-5
report = {'mode': 'author checkpoint CPU inference only, dropout=0, no training',
          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'args': vars(args), 'legacy_shape': list(baseline.shape),
          'instance_shape': list(vectors.shape), 'legacy_selected_max_error': max(errors, default=0),
          'original_multi_type_batch_max_error': max(batch_errors, default=0),
          'matched_instance_batch_reversal_max_error': max(reverse_errors, default=0),
          'all_finite': True, 'strict_checkpoint_load': True,
          'limitation': 'Legacy single-instance path preserved; original loop overwrites depth across motif types. Prototype resets depth per instance. Connection graph not trained.'}
(ROOT / 'instance_encoder_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
