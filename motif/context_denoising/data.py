"""Non-overlapping BRICS components; the original atom graph stays intact."""
import torch
from ogb.utils.mol import smiles2graph
from rdkit import Chem
from rdkit.Chem import BRICS
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MotifData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("atom_to_motif", "motif_edge_index"):
            return int(self.motif_count.item())
        if key == "attachment_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def graph_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty molecule: {smiles}")
    cuts = {tuple(sorted(pair)) for pair, _ in BRICS.FindBRICSBonds(mol)}
    adjacency = [[] for _ in mol.GetAtoms()]
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if tuple(sorted((u, v))) not in cuts:
            adjacency[u].append(v)
            adjacency[v].append(u)
    assignment = [-1] * mol.GetNumAtoms()
    count = 0
    for start in range(len(assignment)):
        if assignment[start] >= 0:
            continue
        assignment[start] = count
        stack = [start]
        while stack:
            for v in adjacency[stack.pop()]:
                if assignment[v] < 0:
                    assignment[v] = count
                    stack.append(v)
        count += 1
    graph = smiles2graph(smiles)
    edges, attachments, features = [], [], []
    # OGB and RDKit use the same unrenumbered SMILES atom order.
    for k, (u, v) in enumerate(graph["edge_index"].T.tolist()):
        if assignment[u] != assignment[v]:
            edges.append([assignment[u], assignment[v]])
            attachments.append([u, v])
            features.append(graph["edge_feat"][k].tolist())
    return MotifData(
        x=torch.from_numpy(graph["node_feat"]).long(),
        edge_index=torch.from_numpy(graph["edge_index"]).long(),
        edge_attr=torch.from_numpy(graph["edge_feat"]).long(),
        atom_to_motif=torch.tensor(assignment, dtype=torch.long),
        motif_count=torch.tensor([count]),
        motif_edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        attachment_index=torch.tensor(attachments, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        motif_edge_attr=torch.tensor(features, dtype=torch.long).reshape(-1, 3),
    )


class MotifDataset(Dataset):
    def __init__(self, source):
        self.source = source
        self.cache = {}

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if index not in self.cache:
            data = graph_from_smiles(self.source.smiles[index])
            data.y = self.source.targets[index].view(1, -1).float()
            data.target_mask = self.source.target_mask[index].view(1, -1).bool()
            data.sample_id = torch.tensor([index])
            self.cache[index] = data
        return self.cache[index]
