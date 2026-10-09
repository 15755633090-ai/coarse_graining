"""Execute upstream matcher and batch selection AST; quantify truncation.

No property labels are read; no optimizer, training, or hyperparameter search.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from rdkit import Chem
from motif_instances import load_patterns, build_instance_graph

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent / 'MotiL_micromolecule'


def upstream_operations(patterns):
    tree = ast.parse((ROOT / 'upstream/featurization.py').read_text(encoding='utf-8'))
    matcher = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'match_fg_all')
    batch_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchMolGraph')
    init = next(n for n in batch_class.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    selection = next(n for n in ast.walk(init) if isinstance(n, ast.For)
                     and ast.unparse(n.target) == 'key'
                     and ast.unparse(n.iter) == 'mol_graph.all_fgs')
    function = ast.parse('def select(self, mol_graph):\n    pass').body[0]
    function.body = [selection]
    module = ast.fix_missing_locations(ast.Module(body=[matcher, function], type_ignores=[]))
    env = {'smart': [p for _, p in patterns], 'smart2name': {p: n for n, p in patterns}}
    exec(compile(module, '<official-selection-ast>', 'exec'), env)
    return env['match_fg_all'], env['select']


def batch_select(smiles_list, matcher, selection):
    state = SimpleNamespace(all_fgs={}, n_atoms=1)
    ownership = {}
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        motifs = matcher(mol)
        previous = set(state.all_fgs)
        selection(state, SimpleNamespace(all_fgs=motifs))
        for key in set(state.all_fgs) - previous:
            ownership[key] = index
        state.n_atoms += mol.GetNumAtoms()
    return {'types': list(state.all_fgs), 'global_atoms': state.all_fgs,
            'owner_smiles': {key: smiles_list[i] for key, i in ownership.items()}}


def main():
    patterns = load_patterns(MICRO / 'chemprop/data/funcgroup.txt')
    matcher, selection = upstream_operations(patterns)
    summary = {'kind': 'label-free structural audit, not prediction benchmark',
               'matching': 'official vocabulary, >1 atoms, RDKit default matching cap',
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT / 'upstream').glob('*.py')},
               'datasets': {}}
    for task in ['esol', 'bbbp', 'clintox', 'bace', 'lipo']:
        path = MICRO / 'data' / (task + '.csv')
        if not path.exists():
            continue
        stats = {'molecules': 0, 'invalid': 0, 'without_eligible_motif': 0,
                 'all_enumerated_instances': 0, 'single_molecule_retained': 0,
                 'molecules_with_repeated_type': 0, 'examples': []}
        smiles_list = []
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                smiles = row.get('smiles') or row.get('SMILES')
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    stats['invalid'] += 1
                    continue
                smiles_list.append(smiles)
                stats['molecules'] += 1
                all_count = sum(len(matches) for _, pattern in patterns
                                if (matches := mol.GetSubstructMatches(pattern)) and len(matches[0]) > 1)
                retained = sum(len(v) for v in matcher(mol).values())
                stats['all_enumerated_instances'] += all_count
                stats['single_molecule_retained'] += retained
                stats['without_eligible_motif'] += retained == 0
                stats['molecules_with_repeated_type'] += all_count > retained
                if all_count > retained and len(stats['examples']) < 3:
                    stats['examples'].append({'smiles': smiles, 'enumerated': all_count, 'retained': retained})
        # Fixed dataset order, audit grouping only; not author training batch sizes.
        stats['audit_batches'] = {}
        for size in [16, 32, 64]:
            kept = sum(len(batch_select(smiles_list[i:i+size], matcher, selection)['types'])
                       for i in range(0, len(smiles_list), size))
            stats['audit_batches'][str(size)] = {'retained_instances': kept,
                'fraction_of_enumerated': kept / max(1, stats['all_enumerated_instances'])}
        summary['datasets'][task] = stats
        print(task, stats['molecules'], stats['all_enumerated_instances'], stats['single_molecule_retained'], flush=True)
    examples = ['CC(=O)OCCOC(=O)C', 'CCC(=O)OCC']
    forward = batch_select(examples, matcher, selection)
    reverse = batch_select(examples[::-1], matcher, selection)
    summary['batch_order_demo'] = {'forward': forward, 'reverse': reverse,
                                  'owner_changes': forward['owner_smiles'] != reverse['owner_smiles']}
    prototype = [build_instance_graph(s, patterns) for s in examples]
    for graph in prototype:
        assert len(graph['instances']) == sum(len(matches) for _, p in patterns
            if (matches := Chem.MolFromSmiles(graph['smiles']).GetSubstructMatches(p)) and len(matches[0]) > 1)
        assert all(0 <= a < graph['num_atoms'] for n in graph['instances'] for a in n['atoms'])
        assert all(n['atoms'] for n in graph['instances'])
    # Simple independent chain fixture verifies real attachment endpoints.
    fixture = build_instance_graph('CCCC', [('pair', Chem.MolFromSmarts('[C][C]'))])
    assert len(fixture['instances']) == 3
    assert any(e['bridge_bonds'] == [{'left_atom': 1, 'right_atom': 2, 'bond_type': 'SINGLE'}]
               for e in fixture['edges'])
    assert any(e['overlap_atoms'] for e in fixture['edges'])
    summary['prototype_checks'] = 'full counts, valid local atom indices, bridge endpoints, overlap: passed'
    (ROOT / 'motif_selection_diagnostics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'instance_graph_examples.json').write_text(json.dumps(prototype, ensure_ascii=False, indent=2), encoding='utf-8')
    print('diagnostic complete; no training performed', flush=True)


if __name__ == '__main__':
    main()
