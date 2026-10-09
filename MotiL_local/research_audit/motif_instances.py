"""Instance-preserving motif graph prototype; no training or baseline changes."""
from pathlib import Path
from rdkit import Chem


def load_patterns(path):
    patterns = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields:
            continue
        pattern = Chem.MolFromSmarts(fields[1])
        if pattern is None:
            raise ValueError('Invalid SMARTS: ' + line)
        patterns.append((fields[0], pattern))
    return patterns


def build_instance_graph(smiles, patterns):
    """Keep official >1-atom motif eligibility; preserve all enumerated matches.

    SMARTS matches may overlap. Overlap is represented separately from chemical
    bridge bonds. Atom indices are molecule-local, zero-based, never padding.
    RDKit's default match enumeration cap applies, as in the original matcher.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES: ' + smiles)
    nodes = []
    for type_id, (name, pattern) in enumerate(patterns):
        for match in mol.GetSubstructMatches(pattern):
            if len(match) <= 1:
                continue
            atoms = set(match)
            internal, boundary = [], []
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge = {'atoms': [a, b], 'bond_type': str(bond.GetBondType())}
                if a in atoms and b in atoms:
                    internal.append(edge)
                elif (a in atoms) != (b in atoms):
                    boundary.append({**edge, 'inside_atom': a if a in atoms else b,
                                     'outside_atom': b if a in atoms else a})
            nodes.append({'id': len(nodes), 'type_id': type_id, 'type': name,
                          'atoms': list(match), 'internal_bonds': internal,
                          'boundary_bonds': boundary})
    edges = []
    for i, left in enumerate(nodes):
        a_set = set(left['atoms'])
        for right in nodes[i + 1:]:
            b_set = set(right['atoms'])
            overlap = sorted(a_set & b_set)
            bridges = []
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a in a_set - b_set and b in b_set - a_set:
                    bridges.append({'left_atom': a, 'right_atom': b,
                                    'bond_type': str(bond.GetBondType())})
                elif b in a_set - b_set and a in b_set - a_set:
                    bridges.append({'left_atom': b, 'right_atom': a,
                                    'bond_type': str(bond.GetBondType())})
            if overlap or bridges:
                edges.append({'left': left['id'], 'right': right['id'],
                              'overlap_atoms': overlap, 'bridge_bonds': bridges})
    covered = set(a for n in nodes for a in n['atoms'])
    return {'smiles': smiles, 'num_atoms': mol.GetNumAtoms(), 'instances': nodes,
            'edges': edges, 'uncovered_atoms': sorted(set(range(mol.GetNumAtoms())) - covered)}


def encode_instances(encoder, mol_graph, instance_graphs, depth, dropout=0.0):
    """Reuse existing CMPNN motif branch weights, exposing one row per instance.

    This deliberately retains legacy per-instance message passing and padding
    mean, so the first ablation changes selection/aggregation only. Does not
    pool by motif type or add trainable parameters. Graph path stays untouched.
    Returned records retain molecule identity and real boundary metadata.
    """
    import torch
    components = mol_graph.get_components()
    scopes = components[5]
    if len(scopes) != len(instance_graphs):
        raise ValueError('Batch molecule count mismatch')
    vectors, records = [], []

    class View:
        def __init__(self, selected):
            self.selected = selected

        def get_components(self):
            return (*components[:7], self.selected)

    for molecule_id, (graph, (offset, size)) in enumerate(zip(instance_graphs, scopes)):
        if graph['num_atoms'] != size:
            raise ValueError('Molecule atom count mismatch')
        for instance in graph['instances']:
            indices = [offset + a for a in instance['atoms']]
            selected = {instance['type']: [indices]}
            vector = encoder('pretrain', View(selected), 'contrast_fgs', depth, dropout)
            vectors.append(vector[0])
            records.append({'molecule_id': molecule_id, **instance})
    if not vectors:
        return next(encoder.parameters()).new_empty((0, encoder.hidden_size)), records
    return torch.stack(vectors), records
