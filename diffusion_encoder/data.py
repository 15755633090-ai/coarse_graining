"""Dataset-neutral molecular property inputs for the frozen diffusion encoder."""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC, MASK_BOND = range(6)


@dataclass
class MoleculeGraph:
    node_features: Tensor
    bonds: Tensor
    smiles: str

    def __post_init__(self):
        nodes = self.node_features.size(0)
        if self.node_features.shape != (nodes, 5):
            raise ValueError("node_features must have shape [nodes, 5]")
        if self.bonds.shape != (nodes, nodes) or not torch.equal(self.bonds, self.bonds.T):
            raise ValueError("bonds must be a symmetric [nodes, nodes] matrix")


@dataclass
class MoleculeBatch:
    node_features: Tensor
    bonds: Tensor
    node_mask: Tensor
    smiles: list[str]

    def to(self, device: str | torch.device) -> "MoleculeBatch":
        return type(self)(
            self.node_features.to(device, non_blocking=True),
            self.bonds.to(device, non_blocking=True),
            self.node_mask.to(device, non_blocking=True),
            self.smiles,
        )


@dataclass
class PropertyRecord:
    graph: MoleculeGraph
    targets: Tensor
    target_mask: Tensor
    sample_id: int
    split: str


@dataclass
class PropertyBatch:
    graph: MoleculeBatch
    targets: Tensor
    target_mask: Tensor
    sample_ids: Tensor

    def to(self, device: str | torch.device) -> "PropertyBatch":
        return type(self)(
            self.graph.to(device), self.targets.to(device),
            self.target_mask.to(device), self.sample_ids.to(device),
        )


def graph_from_smiles(smiles: str) -> MoleculeGraph:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("SMILES conversion requires RDKit") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    hybridization = {
        Chem.HybridizationType.S: 0, Chem.HybridizationType.SP: 1,
        Chem.HybridizationType.SP2: 2, Chem.HybridizationType.SP3: 3,
        Chem.HybridizationType.SP3D: 4, Chem.HybridizationType.SP3D2: 5,
    }
    features = [[
        atom.GetAtomicNum(), max(0, min(10, atom.GetFormalCharge() + 5)),
        int(atom.GetIsAromatic()), hybridization.get(atom.GetHybridization(), 6),
        min(atom.GetDegree(), 6),
    ] for atom in molecule.GetAtoms()]
    nodes = molecule.GetNumAtoms()
    bonds = torch.zeros((nodes, nodes), dtype=torch.long)
    bond_types = {
        Chem.BondType.SINGLE: SINGLE, Chem.BondType.DOUBLE: DOUBLE,
        Chem.BondType.TRIPLE: TRIPLE, Chem.BondType.AROMATIC: AROMATIC,
    }
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[left, right] = bonds[right, left] = bond_types.get(bond.GetBondType(), SINGLE)
    return MoleculeGraph(torch.tensor(features, dtype=torch.long), bonds, smiles)


def collate_graphs(graphs: Sequence[MoleculeGraph]) -> MoleculeBatch:
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    maximum = max(graph.node_features.size(0) for graph in graphs)
    node_features = torch.zeros((len(graphs), maximum, 5), dtype=torch.long)
    bonds = torch.zeros((len(graphs), maximum, maximum), dtype=torch.long)
    node_mask = torch.zeros((len(graphs), maximum), dtype=torch.bool)
    for index, graph in enumerate(graphs):
        nodes = graph.node_features.size(0)
        node_features[index, :nodes] = graph.node_features
        bonds[index, :nodes, :nodes] = graph.bonds
        node_mask[index, :nodes] = True
    return MoleculeBatch(node_features, bonds, node_mask, [graph.smiles for graph in graphs])


def collate_property_records(records: Sequence[PropertyRecord]) -> PropertyBatch:
    return PropertyBatch(
        collate_graphs([record.graph for record in records]),
        torch.stack([record.targets for record in records]),
        torch.stack([record.target_mask for record in records]),
        torch.tensor([record.sample_id for record in records], dtype=torch.long),
    )


class OGBMoleculePropertyDataset(Dataset[PropertyRecord]):
    """Read any copied OGB molecular property task with its published scaffold split."""

    def __init__(self, root: str | Path, split: str | None = None):
        self.root = Path(root)
        if split not in (None, "train", "valid", "test"):
            raise ValueError("split must be train, valid, test or None")
        with gzip.open(self.root / "mapping/mol.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
            smiles = [row["smiles"] for row in csv.DictReader(handle)]
        assignments = [None] * len(smiles)
        split_indices = {}
        for name in ("train", "valid", "test"):
            with gzip.open(self.root / f"split/scaffold/{name}.csv.gz", "rt", encoding="utf-8") as handle:
                indices = [int(line.strip()) for line in handle if line.strip()]
            split_indices[name] = indices
            for index in indices:
                if assignments[index] is not None:
                    raise ValueError("Published scaffold splits overlap")
                assignments[index] = name
        if any(value is None for value in assignments):
            raise ValueError("Published scaffold splits do not cover the dataset")
        indices = list(range(len(smiles))) if split is None else split_indices[split]
        # Pandas' C parser avoids millions of Python float objects for large
        # multi-task datasets such as PCBA (437,929 x 128 labels).
        label_values = pd.read_csv(
            self.root / "raw/graph-label.csv.gz", header=None, dtype="float32",
        ).to_numpy(copy=True)
        if len(smiles) != label_values.shape[0]:
            raise ValueError("SMILES and graph-label row counts differ")
        self.sample_ids = indices
        self.smiles = [smiles[index] for index in indices]
        self.labels = torch.from_numpy(label_values[indices].copy())
        self.assignments = [assignments[index] for index in indices]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, item: int) -> PropertyRecord:
        index = self.sample_ids[item]
        targets = self.labels[item]
        return PropertyRecord(
            graph_from_smiles(self.smiles[item]), targets, torch.isfinite(targets),
            index, self.assignments[item],
        )


def discover_datasets(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        path.name: path for path in sorted(root.iterdir())
        if path.is_dir() and (path / "mapping/mol.csv.gz").is_file()
    }
