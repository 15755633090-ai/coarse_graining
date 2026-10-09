"""Selection, gradients, atom renumbering, timing: no optimizer updates."""
import json
import os
from pathlib import Path
import sys
import time
from argparse import Namespace
from collections import Counter
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent / 'MotiL_micromolecule'
os.chdir(MICRO)
sys.path.insert(0, str(MICRO))
from chemprop.models.cmpn import CMPN
from chemprop.models.loss.loss import PointwiseLoss
from motif_instances import load_patterns
from motif_ablation import VARIANTS, choose_motifs, make_batch, install_variant, motif_views
from diagnose_motif_selection import upstream_operations, batch_select

torch.set_num_threads(2)
args = Namespace(hidden_size=300, depth=3, bias=False, undirected=False,
                 atom_messages=False, features_only=False, use_input_features=False,
                 activation='ReLU', cuda=False, device=torch.device('cpu'))
checkpoint = MICRO / 'dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl'
model = CMPN(args)
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
patterns = load_patterns(MICRO / 'chemprop/data/funcgroup.txt')
smiles = ['CC(=O)OCCOC(=O)C', 'CCC(=O)OCC']
matcher, selection = upstream_operations(patterns)
baseline_batch = make_batch(smiles, args, patterns, 'B0')
assert baseline_batch.all_fgs == batch_select(smiles, matcher, selection)['global_atoms']

# Independent fixtures: first/whole matches, later molecule, type cap.
pair = [('pair', Chem.MolFromSmarts('[C][C]'))]
chain = [Chem.MolFromSmiles('CCCC'), Chem.MolFromSmiles('CCC')]
scopes = [(1, 4), (5, 3)]
counts = {v: sum(map(len, choose_motifs(chain, scopes, pair, v).values())) for v in VARIANTS}
assert counts == {'B0': 1, 'B1': 1, 'B2a-1': 3, 'B2a-2': 5, 'B2b': 5}
many_types = [('type' + str(i), pair[0][1]) for i in range(12)]
assert len(choose_motifs(chain, scopes, many_types, 'B2a-2')) == 10
assert all(len(v) == 5 for v in choose_motifs(chain, scopes, many_types, 'B2a-2').values())
assert len(choose_motifs(chain, scopes, many_types, 'B2b')) == 12

with torch.no_grad():
    original_mol = model.encoder('pretrain', baseline_batch, 'contrast_mol', 3, 0.0)
    original_fg = model.encoder('pretrain', baseline_batch, 'contrast_fgs', 3, 0.0)

result = {'scope': 'CPU correctness and backward checks only; no parameter update',
          'fixture_instance_counts': counts, 'variants': {}}
loss_fn = PointwiseLoss(args)
for variant in VARIANTS:
    install_variant(model.encoder, variant)
    batch = make_batch(smiles, args, patterns, variant)
    with torch.no_grad():
        fg = model.encoder('pretrain', batch, 'contrast_fgs', 3, 0.0)
        mol = model.encoder('pretrain', batch, 'contrast_mol', 3, 0.0)
    assert torch.equal(mol, original_mol)
    if variant == 'B0':
        assert torch.equal(fg, original_fg)
    model.zero_grad(set_to_none=True)
    # No dropout for deterministic derivative checks; production default kept
    # in motif_views. Forward/backward verifies shared encoder receives signal.
    views = motif_views(model.encoder, batch, 3, 0.0, 0.0)
    loss = loss_fn(*views)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
    grad_norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
    assert grad_norm > 0
    timings = []
    with torch.no_grad():
        for _ in range(3):
            start = time.perf_counter()
            model.encoder('pretrain', batch, 'contrast_fgs', 3, 0.0)
            timings.append(time.perf_counter() - start)
    result['variants'][variant] = {'types': len(batch.all_fgs),
        'instances': sum(map(len, batch.all_fgs.values())), 'loss': float(loss.detach()),
        'shared_encoder_gradient_norm': grad_norm, 'forward_seconds': timings,
        'molecular_branch_max_error': float((mol-original_mol).abs().max())}

# Direct RDKit atom-order fixture: relabel tensors, not SMILES serialization.
# This retains chirality/graph semantics and covers every bond and atom.
from chemprop.features.featurization import MolGraph, BatchMolGraph, atom_features, bond_features
def renumber_batch(molecule, order):
    relabeled = Chem.RenumberAtoms(molecule, order)
    graph = MolGraph(Chem.MolToSmiles(molecule), args)
    graph.n_atoms = relabeled.GetNumAtoms()
    graph.f_atoms = [atom_features(a) for a in relabeled.GetAtoms()]
    graph.f_bonds, graph.a2b, graph.b2a, graph.b2revb = [], [[] for _ in order], [], []
    graph.n_bonds = 0
    for a in range(len(order)):
        for b in range(a+1, len(order)):
            bond = relabeled.GetBondBetweenAtoms(a, b)
            if bond is None:
                continue
            index = graph.n_bonds
            features = bond_features(bond)
            graph.f_bonds.extend([graph.f_atoms[a]+features, graph.f_atoms[b]+features])
            graph.a2b[b].append(index); graph.a2b[a].append(index+1)
            graph.b2a.extend([a,b]); graph.b2revb.extend([index+1,index])
            graph.n_bonds += 2
    graph.all_fgs = {}
    batch = BatchMolGraph([graph], args)
    return batch, relabeled

molecule = Chem.MolFromSmiles(smiles[0])
orders = [list(range(molecule.GetNumAtoms())), list(reversed(range(molecule.GetNumAtoms()))),
          torch.randperm(molecule.GetNumAtoms(), generator=torch.Generator().manual_seed(2024)).tolist()]
rows, selections, graph_vectors = [], [], []
install_variant(model.encoder, 'B2b')
with torch.no_grad():
    for order in orders:
        batch, mol = renumber_batch(molecule, order)
        chosen = choose_motifs([mol], batch.a_scope, patterns, 'B2b')
        # Verify corresponding atom sets, avoiding ambiguity of SMARTS automorphisms.
        selections.append(Counter((kind, tuple(sorted(order[a-1] for a in instance)))
                                  for kind, instances in chosen.items() for instance in instances))
        view = __import__('motif_ablation').MotifBatch(batch, chosen)
        rows.append(model.encoder('pretrain', view, 'contrast_fgs', 3, 0.0))
        graph_vectors.append(model.encoder('pretrain', view, 'contrast_mol', 3, 0.0))
assert all(s == selections[0] for s in selections)
errors = [float((r-rows[0]).abs().max()) for r in rows[1:]]
assert max(errors) < 1e-5
result['atom_renumbering'] = {'tested_orders': orders, 'motif_type_mean_max_errors': errors,
    'molecular_gru_max_errors': [float((r-graph_vectors[0]).abs().max()) for r in graph_vectors[1:]],
    'limitation': 'Three orderings of one molecule; not a universal invariance proof.'}
assert motif_views(model.encoder, make_batch(['CC'], args, patterns, 'B2b'), 3) is None
result['low_type_batch_policy'] = 'fewer than two motif types: skip motif update, log count'
(ROOT/'ablation_correctness.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
