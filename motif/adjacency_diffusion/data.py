"""Clean X is a condition; clean adjacency is exclusively a target during denoising."""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from rdkit import Chem

from motif.brics_atom_diffusion.data import graph_from_smiles as partition_graph

FEATURE_SCHEMA = "motil_eight_atom_attributes_133_binary_undirected_adjacency_v1"
ATOM_DIM = 133


def unknown_one_hot(value, choices):
    result = [0.] * (len(choices) + 1)
    result[choices.index(value) if value in choices else -1] = 1.
    return result


def atom_features(atom):
    """Eight attribute definitions verified against the pinned 2a8f006 featurizer.

    Total degree/H/charge/aromaticity are CLEAN conditions and can imply topology.
    No claim of unconditional structure recovery is made.
    """
    hybrid = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
              Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    parts = [(atom.GetAtomicNum() - 1, list(range(100))), (atom.GetTotalDegree(), list(range(6))),
             (atom.GetFormalCharge(), [-1, -2, 1, 2, 0]), (int(atom.GetChiralTag()), list(range(4))),
             (int(atom.GetTotalNumHs()), list(range(5))), (int(atom.GetHybridization()), [int(v) for v in hybrid])]
    return sum([unknown_one_hot(value, choices) for value, choices in parts], []) + [float(atom.GetIsAromatic()), atom.GetMass() * .01]


def graph_from_smiles(smiles):
    graph = partition_graph(smiles)
    mol = Chem.MolFromSmiles(smiles)
    graph.x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    # Do not retain clean bond attributes or true boundary flags in this dataset.
    del graph.edge_attr
    del graph.boundary
    del graph.is_cross
    return graph


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, targets=None, masks=None):
        self.smiles, self.targets, self.masks, self.cache = list(smiles), targets, masks, {}

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        if index not in self.cache:
            graph = graph_from_smiles(self.smiles[index])
            graph.sample_id = torch.tensor([index])
            if self.targets is not None:
                graph.y = self.targets[index].view(1, -1).float()
                graph.target_mask = self.masks[index].view(1, -1).bool()
            self.cache[index] = graph
        return self.cache[index]


@dataclass(frozen=True)
class Condition:
    """Encoder-facing interface deliberately has NO clean adjacency/bond/endpoints."""
    x: torch.Tensor
    nodes: torch.Tensor
    pairs: torch.Tensor
    same_fragment: torch.Tensor | None


def condition_from_batch(batch, motif_prior=False):
    x, nodes = to_dense_batch(batch.x, batch.batch)
    n = x.shape[1]
    pairs = nodes[:, :, None] & nodes[:, None, :] & ~torch.eye(n, device=x.device, dtype=torch.bool)[None]
    same = None
    if motif_prior:
        fragments, _ = to_dense_batch(batch.fragment_id, batch.batch, fill_value=-1)
        same = (fragments[:, :, None] == fragments[:, None, :]) & pairs
    return Condition(x=x, nodes=nodes, pairs=pairs, same_fragment=same)


def adjacency_target(batch, condition):
    # Only preparation/loss and clean downstream prediction call this function.
    target = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=condition.x.shape[1]).clamp_max(1)
    target = target * condition.pairs
    if not torch.equal(target, target.transpose(1, 2)):
        raise ValueError("Expected an undirected molecule graph")
    return target
