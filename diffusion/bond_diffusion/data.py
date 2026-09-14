from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


# Edge labels. Every graph stores a full symmetric NxN edge matrix.
NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC, MASK_BOND = range(6)
NODE_FEATURE_NAMES = ("atom_type", "formal_charge", "aromatic", "hybridization", "degree")


@dataclass
class MoleculeGraph:
    """One unpadded molecule.

    node_features has shape [N, 5], bonds has shape [N, N], and
    substructure_mask marks pairs belonging to a recognized functional group.
    """

    node_features: Tensor
    bonds: Tensor
    substructure_mask: Tensor
    smiles: str = ""

    def __post_init__(self) -> None:
        n = self.node_features.size(0)
        if self.node_features.shape != (n, 5):
            raise ValueError("node_features must have shape [N, 5]")
        if self.bonds.shape != (n, n):
            raise ValueError("bonds must have shape [N, N]")
        if self.substructure_mask.shape != (n, n):
            raise ValueError("substructure_mask must have shape [N, N]")
        if not torch.equal(self.bonds, self.bonds.T):
            raise ValueError("bonds must be symmetric")


@dataclass
class MoleculeBatch:
    node_features: Tensor  # [B, N, 5]
    bonds: Tensor  # [B, N, N]
    node_mask: Tensor  # [B, N]
    pair_mask: Tensor  # [B, N, N], upper triangle only
    substructure_mask: Tensor  # [B, N, N], upper triangle only
    smiles: List[str]

    def to(self, device: torch.device | str) -> "MoleculeBatch":
        return MoleculeBatch(
            self.node_features.to(device, non_blocking=True),
            self.bonds.to(device, non_blocking=True),
            self.node_mask.to(device, non_blocking=True),
            self.pair_mask.to(device, non_blocking=True),
            self.substructure_mask.to(device, non_blocking=True),
            self.smiles,
        )


def collate_graphs(graphs: Sequence[MoleculeGraph]) -> MoleculeBatch:
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(graphs)
    max_nodes = max(g.node_features.size(0) for g in graphs)
    x = torch.zeros(batch_size, max_nodes, 5, dtype=torch.long)
    e = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    substructure = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.bool)
    smiles: List[str] = []
    for batch_idx, graph in enumerate(graphs):
        n = graph.node_features.size(0)
        x[batch_idx, :n] = graph.node_features.long()
        e[batch_idx, :n, :n] = graph.bonds.long()
        node_mask[batch_idx, :n] = True
        substructure[batch_idx, :n, :n] = graph.substructure_mask.bool()
        smiles.append(graph.smiles)
    all_pairs = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    upper = torch.triu(torch.ones(max_nodes, max_nodes, dtype=torch.bool), diagonal=1)
    pair_mask = all_pairs & upper.unsqueeze(0)
    substructure = substructure & pair_mask
    return MoleculeBatch(x, e, node_mask, pair_mask, substructure, smiles)


def infer_substructure_mask(node_features: Tensor, bonds: Tensor) -> Tensor:
    """Recognize common motifs without RDKit: carbonyl/carboxyl, amine, aromatic bonds.

    This mask supplies additional recovery supervision; it is intentionally based
    only on graph labels so JSONL datasets and RDKit datasets behave identically.
    """

    atom = node_features[:, 0]
    n = atom.numel()
    mask = torch.zeros(n, n, dtype=torch.bool)
    # Aromatic systems.
    mask |= bonds == AROMATIC
    # Carbonyl C=O and carboxyl C(=O)-O.
    for carbon in torch.where(atom == 6)[0].tolist():
        oxygen_neighbors = torch.where((atom == 8) & (bonds[carbon] > 0))[0]
        has_carbonyl = any(bonds[carbon, o].item() == DOUBLE for o in oxygen_neighbors)
        if has_carbonyl:
            for oxygen in oxygen_neighbors.tolist():
                mask[carbon, oxygen] = mask[oxygen, carbon] = True
    # C-N edges and the immediate environment of an amine nitrogen.
    for nitrogen in torch.where(atom == 7)[0].tolist():
        neighbors = torch.where(bonds[nitrogen] > 0)[0]
        if any(atom[j].item() == 6 for j in neighbors):
            mask[nitrogen, neighbors] = True
            mask[neighbors, nitrogen] = True
    return mask


class JsonlGraphDataset(Dataset[MoleculeGraph]):
    """Preprocessed graph dataset, one JSON object per line.

    Required keys: ``node_features`` and ``bonds``. ``substructure_mask`` is
    optional and inferred when absent.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets: List[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> MoleculeGraph:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            item = json.loads(handle.readline())
        x = torch.tensor(item["node_features"], dtype=torch.long)
        e = torch.tensor(item["bonds"], dtype=torch.long)
        raw_mask = item.get("substructure_mask")
        sub = (
            torch.tensor(raw_mask, dtype=torch.bool)
            if raw_mask is not None
            else infer_substructure_mask(x, e)
        )
        return MoleculeGraph(x, e, sub, item.get("smiles", ""))


class SmilesDataset(Dataset[MoleculeGraph]):
    """SMILES dataset. RDKit is imported lazily and is an optional dependency."""

    def __init__(self, smiles: Sequence[str]):
        self.smiles = [s.strip() for s in smiles if s.strip()]

    @classmethod
    def from_file(
        cls, path: str | Path, smiles_column: str = "smiles"
    ) -> "SmilesDataset":
        path = Path(path)
        if path.suffix.lower() in {".smi", ".txt"}:
            smiles = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
        else:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if smiles_column not in (reader.fieldnames or []):
                    raise ValueError(f"CSV has no column named {smiles_column!r}")
                smiles = [row[smiles_column] for row in reader]
        return cls(smiles)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> MoleculeGraph:
        return graph_from_smiles(self.smiles[index])


def graph_from_smiles(smiles: str) -> MoleculeGraph:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "SMILES input requires RDKit. Install it with `conda install -c conda-forge rdkit`, "
            "or train from preprocessed JSONL."
        ) from exc
    mol = Chem.MolFromSmiles(smiles)
    # A few official OGB HIV/Tox21 entries contain hypervalent organometallic
    # structures rejected by newer RDKit sanitization rules. Preserve their
    # published connectivity instead of silently dropping benchmark samples.
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    hybrid_map: Dict[object, int] = {
        Chem.HybridizationType.S: 0,
        Chem.HybridizationType.SP: 1,
        Chem.HybridizationType.SP2: 2,
        Chem.HybridizationType.SP3: 3,
        Chem.HybridizationType.SP3D: 4,
        Chem.HybridizationType.SP3D2: 5,
    }
    features = []
    for atom in mol.GetAtoms():
        features.append(
            [
                atom.GetAtomicNum(),
                max(0, min(10, atom.GetFormalCharge() + 5)),
                int(atom.GetIsAromatic()),
                hybrid_map.get(atom.GetHybridization(), 6),
                min(atom.GetDegree(), 6),
            ]
        )
    n = mol.GetNumAtoms()
    bonds = torch.zeros(n, n, dtype=torch.long)
    bond_map = {
        Chem.BondType.SINGLE: SINGLE,
        Chem.BondType.DOUBLE: DOUBLE,
        Chem.BondType.TRIPLE: TRIPLE,
        Chem.BondType.AROMATIC: AROMATIC,
    }
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[i, j] = bonds[j, i] = bond_map.get(bond.GetBondType(), SINGLE)
    x = torch.tensor(features, dtype=torch.long)
    return MoleculeGraph(x, bonds, infer_substructure_mask(x, bonds), smiles)


def write_jsonl(graphs: Iterable[MoleculeGraph], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for graph in graphs:
            item = {
                "smiles": graph.smiles,
                "node_features": graph.node_features.tolist(),
                "bonds": graph.bonds.tolist(),
                "substructure_mask": graph.substructure_mask.int().tolist(),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
