"""Clean topology/BRICS are disclosed conditions; atom attributes are diffusion targets."""
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import BRICS

ATOM_DIMS = (119, 12, 2)  # element (0=other), charge -5..5/other, aromaticity
BOND_DIMS = (5, 7, 2, 2)  # type, stereo, conjugated, in-ring
ATOM_DIM, BOND_DIM = sum(ATOM_DIMS), sum(BOND_DIMS)
SCHEMA = "brics_atom_v1_element_charge_aromatic_clean_bond_topology"


def one_hot(values, dimensions):
    return torch.cat([F.one_hot(values[:, k], n).float() for k, n in enumerate(dimensions)], dim=-1)


class AtomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "fragment_id":
            return int(self.fragment_count.item())
        return super().__inc__(key, value, *args, **kwargs)


def graph_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid/empty SMILES: {smiles}")
    cuts = {tuple(sorted(endpoints)) for endpoints, _ in BRICS.FindBRICSBonds(mol)}
    adjacent = [[] for _ in mol.GetAtoms()]
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if tuple(sorted((u, v))) not in cuts:
            adjacent[u].append(v)
            adjacent[v].append(u)
    assignment = [-1] * mol.GetNumAtoms()
    count = 0
    for start in range(mol.GetNumAtoms()):
        if assignment[start] >= 0:
            continue
        assignment[start] = count
        stack = [start]
        while stack:
            for v in adjacent[stack.pop()]:
                if assignment[v] < 0:
                    assignment[v] = count
                    stack.append(v)
        count += 1
    atoms = []
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        atoms.append([atom.GetAtomicNum() if atom.GetAtomicNum() <= 118 else 0,
                      charge + 5 if -5 <= charge <= 5 else 11, int(atom.GetIsAromatic())])
    types = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
             Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    stereo_types = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY,
                    Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE,
                    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS]
    edges, attributes, cross = [], [], []
    boundary = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = [types.get(bond.GetBondType(), 4),
                   stereo_types.index(bond.GetStereo()) if bond.GetStereo() in stereo_types else 6,
                   int(bond.GetIsConjugated()), int(bond.IsInRing())]
        is_cross = assignment[u] != assignment[v]
        if is_cross:
            boundary[u] = boundary[v] = True
        for a, b in ((u, v), (v, u)):
            edges.append([a, b])
            attributes.append(feature)
            cross.append(is_cross)
    return AtomData(
        x=one_hot(torch.tensor(atoms, dtype=torch.long), ATOM_DIMS),
        edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        edge_attr=one_hot(torch.tensor(attributes, dtype=torch.long).reshape(-1, 4), BOND_DIMS),
        is_cross=torch.tensor(cross, dtype=torch.bool), boundary=boundary,
        fragment_id=torch.tensor(assignment, dtype=torch.long), fragment_count=torch.tensor([count]),
    )


class AtomDataset(Dataset):
    def __init__(self, smiles, targets=None, masks=None):
        self.smiles, self.targets, self.masks = list(smiles), targets, masks
        self.cache = {}

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        if index not in self.cache:
            graph = graph_from_smiles(self.smiles[index])
            graph.sample_id = torch.tensor([index])
            if self.targets is not None:
                graph.y = self.targets[index].reshape(1, -1).float()
                graph.target_mask = self.masks[index].reshape(1, -1).bool()
            self.cache[index] = graph
        return self.cache[index]
