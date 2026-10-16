from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .config import ModelConfig
from .data import MoleculeBatch, MoleculeGraph, collate_graphs, graph_from_smiles
from .model import BondAwareDiffusionModel


@dataclass(frozen=True)
class PropertyTaskSpec:
    name: str
    directory: str
    label_columns: tuple[str, ...]
    task_type: str
    primary_metric: str

    @property
    def num_tasks(self) -> int:
        return len(self.label_columns)


TASK_SPECS: Dict[str, PropertyTaskSpec] = {
    "esol": PropertyTaskSpec(
        "esol",
        "ogbg_molesol",
        ("measured log solubility in mols per litre",),
        "regression",
        "rmse",
    ),
    "freesolv": PropertyTaskSpec(
        "freesolv", "ogbg_molfreesolv", ("expt",), "regression", "rmse"
    ),
    "lipo": PropertyTaskSpec(
        "lipo", "ogbg_mollipo", ("exp",), "regression", "rmse"
    ),
    "bbbp": PropertyTaskSpec(
        "bbbp", "ogbg_molbbbp", ("p_np",), "classification", "roc_auc"
    ),
    "bace": PropertyTaskSpec(
        "bace", "ogbg_molbace", ("Class",), "classification", "roc_auc"
    ),
    "hiv": PropertyTaskSpec(
        "hiv", "ogbg_molhiv", ("HIV_active",), "classification", "roc_auc"
    ),
    "tox21": PropertyTaskSpec(
        "tox21",
        "ogbg_moltox21",
        (
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ),
        "classification",
        "roc_auc",
    ),
}


class PropertyDataset(Dataset[tuple[MoleculeGraph, Tensor, Tensor]]):
    def __init__(self, csv_path: str | Path, spec: PropertyTaskSpec):
        frame = pd.read_csv(csv_path)
        self.smiles = frame["smiles"].astype(str).tolist()
        values = frame.loc[:, list(spec.label_columns)].to_numpy(dtype=np.float32)
        self.targets = torch.from_numpy(np.nan_to_num(values, nan=0.0))
        self.target_mask = torch.from_numpy(~np.isnan(values))
        self.spec = spec

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> tuple[MoleculeGraph, Tensor, Tensor]:
        return (
            graph_from_smiles(self.smiles[index]),
            self.targets[index],
            self.target_mask[index],
        )


@dataclass
class PropertyBatch:
    graph: MoleculeBatch
    targets: Tensor
    target_mask: Tensor

    def to(self, device: torch.device | str) -> "PropertyBatch":
        return PropertyBatch(
            self.graph.to(device),
            self.targets.to(device, non_blocking=True),
            self.target_mask.to(device, non_blocking=True),
        )


def collate_property_batch(
    rows: Sequence[tuple[MoleculeGraph, Tensor, Tensor]]
) -> PropertyBatch:
    graphs, targets, masks = zip(*rows)
    return PropertyBatch(
        collate_graphs(graphs),
        torch.stack(targets),
        torch.stack(masks),
    )


def load_official_splits(
    data_root: str | Path, spec: PropertyTaskSpec
) -> Dict[str, List[int]]:
    split_root = Path(data_root) / spec.directory / "split" / "scaffold"
    result = {}
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(split_root / f"{split}.csv.gz", header=None)
        result[split] = frame.iloc[:, 0].astype(int).tolist()
    return result


def build_property_data(
    data_root: str | Path, task_name: str
) -> tuple[PropertyDataset, Dict[str, List[int]], PropertyTaskSpec]:
    spec = TASK_SPECS[task_name]
    mapping_path = Path(data_root) / spec.directory / "mapping" / "mol.csv.gz"
    dataset = PropertyDataset(mapping_path, spec)
    splits = load_official_splits(data_root, spec)
    all_indices = splits["train"] + splits["valid"] + splits["test"]
    if len(all_indices) != len(dataset) or len(set(all_indices)) != len(dataset):
        raise ValueError(f"Official split for {task_name} is incomplete or duplicated")
    return dataset, splits, spec


class PropertyPredictor(nn.Module):
    """Shared encoder + Sum/Mean readout + property MLP."""

    def __init__(
        self,
        encoder: BondAwareDiffusionModel,
        num_tasks: int,
        dropout: float = 0.2,
        encoder_frozen: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_frozen = encoder_frozen
        hidden = encoder.config.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_tasks),
        )
        self.set_encoder_frozen(encoder_frozen)

    def set_encoder_frozen(self, frozen: bool) -> None:
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        # Denoising decoders and the pretraining-specific graph projection are
        # never part of the downstream architecture in either comparison arm.
        self.encoder.node_heads.requires_grad_(False)
        self.encoder.edge_head.requires_grad_(False)
        self.encoder.graph_projection.requires_grad_(False)
        if frozen:
            self.encoder.eval()

    def train(self, mode: bool = True) -> "PropertyPredictor":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def readout(self, node_states: Tensor, node_mask: Tensor) -> Tensor:
        weights = node_mask.unsqueeze(-1).to(node_states.dtype)
        summed = (node_states * weights).sum(dim=1)
        averaged = summed / weights.sum(dim=1).clamp_min(1.0)
        return torch.cat([summed, averaged], dim=-1)

    def forward(self, batch: MoleculeBatch) -> Tensor:
        if self.encoder_frozen:
            with torch.no_grad():
                node_states = self.encoder.encode_nodes(
                    batch.node_features, batch.bonds, batch.node_mask
                )
        else:
            node_states = self.encoder.encode_nodes(
                batch.node_features, batch.bonds, batch.node_mask
            )
        return self.head(self.readout(node_states, batch.node_mask))


def create_property_model(
    mode: str,
    num_tasks: int,
    pretrained_checkpoint: str | Path,
    device: torch.device,
    dropout: float,
) -> PropertyPredictor:
    checkpoint = torch.load(
        pretrained_checkpoint, map_location="cpu", weights_only=False
    )
    config = ModelConfig(**checkpoint["model_config"])
    encoder = BondAwareDiffusionModel(config)
    if mode.startswith("pretrained"):
        encoder.load_state_dict(checkpoint["model_state"])
    frozen = mode.endswith("frozen")
    return PropertyPredictor(encoder, num_tasks, dropout, frozen).to(device)


@dataclass
class TargetScaler:
    center: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, targets: Tensor, mask: Tensor) -> "TargetScaler":
        centers, scales = [], []
        for task in range(targets.size(1)):
            values = targets[:, task][mask[:, task]]
            centers.append(values.mean())
            scales.append(values.std(unbiased=False).clamp_min(1e-8))
        return cls(torch.stack(centers), torch.stack(scales))

    def transform(self, values: Tensor) -> Tensor:
        return (values - self.center.to(values.device)) / self.scale.to(values.device)

    def inverse(self, values: Tensor) -> Tensor:
        return values * self.scale.to(values.device) + self.center.to(values.device)


def classification_positive_weight(targets: Tensor, mask: Tensor) -> Tensor:
    positives = ((targets > 0.5) & mask).sum(dim=0).float()
    negatives = ((targets <= 0.5) & mask).sum(dim=0).float()
    return negatives / positives.clamp_min(1.0)


def property_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    spec: PropertyTaskSpec,
    scaler: Optional[TargetScaler],
    positive_weight: Optional[Tensor],
) -> Tensor:
    if spec.task_type == "regression":
        assert scaler is not None
        element_loss = F.mse_loss(
            logits, scaler.transform(targets), reduction="none"
        )
    else:
        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=positive_weight.to(logits.device) if positive_weight is not None else None,
        )
    return (element_loss * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def predict_property(
    model: PropertyPredictor,
    loader: Iterable[PropertyBatch],
    device: torch.device,
    scaler: Optional[TargetScaler],
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    predictions, targets, masks = [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.graph)
        if scaler is not None:
            output = scaler.inverse(output)
        predictions.append(output.cpu())
        targets.append(batch.targets.cpu())
        masks.append(batch.target_mask.cpu())
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def compute_property_metrics(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    spec: PropertyTaskSpec,
) -> Dict[str, float]:
    per_task: Dict[str, Dict[str, float]] = {}
    aggregate: Dict[str, List[float]] = {}
    for task, name in enumerate(spec.label_columns):
        valid = mask[:, task].bool()
        y_true = targets[:, task][valid].numpy()
        y_pred = predictions[:, task][valid].numpy()
        task_metrics: Dict[str, float] = {}
        if spec.task_type == "regression":
            task_metrics["rmse"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))
            task_metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        elif len(np.unique(y_true)) >= 2:
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -30, 30)))
            task_metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
            task_metrics["average_precision"] = float(
                average_precision_score(y_true, probabilities)
            )
        per_task[name] = task_metrics
        for metric, value in task_metrics.items():
            aggregate.setdefault(metric, []).append(value)
    result = {
        metric: float(np.mean(values))
        for metric, values in aggregate.items()
        if values
    }
    result["evaluated_tasks"] = float(
        sum(bool(metrics) for metrics in per_task.values())
    )
    return result


def primary_score(metrics: Mapping[str, float], spec: PropertyTaskSpec) -> float:
    return float(metrics[spec.primary_metric])


def better_score(candidate: float, incumbent: float, spec: PropertyTaskSpec) -> bool:
    if spec.task_type == "regression":
        return candidate < incumbent
    return candidate > incumbent


def pretraining_overlap_indices(
    pretraining_csv: str | Path,
    dataset: PropertyDataset,
    splits: Mapping[str, Sequence[int]],
) -> Dict[str, List[int]]:
    pretraining = set(pd.read_csv(pretraining_csv, usecols=["smiles"])["smiles"].astype(str))
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("Overlap audit requires RDKit") from exc
    canonical_downstream = []
    for smiles in dataset.smiles:
        molecule = Chem.MolFromSmiles(smiles)
        canonical_downstream.append(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            if molecule is not None
            else smiles
        )
    return {
        split: [
            index for index in indices if canonical_downstream[index] in pretraining
        ]
        for split, indices in splits.items()
    }


def summarize_pretraining_overlap(
    overlap_indices: Mapping[str, Sequence[int]],
    splits: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, float | int]]:
    report: Dict[str, Dict[str, float | int]] = {}
    for split, indices in splits.items():
        count = len(overlap_indices[split])
        report[split] = {
            "overlap": count,
            "total": len(indices),
            "fraction": count / len(indices) if indices else 0.0,
        }
    return report


def filter_overlapping_splits(
    splits: Mapping[str, Sequence[int]],
    overlap_indices: Mapping[str, Sequence[int]],
) -> Dict[str, List[int]]:
    filtered = {}
    for split, indices in splits.items():
        excluded = set(overlap_indices[split])
        filtered[split] = [index for index in indices if index not in excluded]
        if not filtered[split]:
            raise ValueError(f"Overlap filtering removed the entire {split} split")
    return filtered


def summarize_runs(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["mode"])), []).append(row)
    summaries = []
    for (dataset, mode), group in sorted(groups.items()):
        metric_names = sorted(group[0]["test_metrics"])
        summary: Dict[str, object] = {
            "dataset": dataset,
            "mode": mode,
            "seeds": len(group),
        }
        for metric in metric_names:
            values = [float(row["test_metrics"][metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def save_benchmark_results(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, object]],
    overlap: Mapping[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_runs(rows)
    (output_dir / "downstream_results.json").write_text(
        json.dumps(
            {"summary": summaries, "runs": list(rows), "overlap_audit": overlap},
            indent=2,
        ),
        encoding="utf-8",
    )
    if summaries:
        fieldnames = sorted({key for row in summaries for key in row})
        with (output_dir / "downstream_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
