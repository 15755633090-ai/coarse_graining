from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .config import DiffusionConfig, ModelConfig
from .data import (
    AROMATIC,
    DOUBLE,
    SINGLE,
    MoleculeBatch,
    MoleculeGraph,
    infer_substructure_mask,
)
from .evaluation import (
    SUBSTRUCTURE_NAMES,
    _class_statistics,
    substructure_labels,
    timestep_for_noise,
    valence_violation_counts,
)
from .model import BondAwareDiffusionModel


NODE_FIELD_NAMES = (
    "atom_type",
    "formal_charge",
    "aromaticity",
    "hybridization",
    "degree",
)


def _macro_f1_multiclass(predicted: Tensor, target: Tensor) -> float:
    predicted = predicted.long().cpu()
    target = target.long().cpu()
    values = []
    for label in torch.unique(target).tolist():
        truth = target == label
        pred = predicted == label
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return mean(values) if values else float("nan")


def _multilabel_metrics(predicted: Tensor, target: Tensor) -> Dict[str, object]:
    predicted = predicted.bool().cpu()
    target = target.bool().cpu()
    per_label: Dict[str, Dict[str, float | int]] = {}
    f1_values, accuracies = [], []
    for index, name in enumerate(SUBSTRUCTURE_NAMES):
        pred = predicted[:, index]
        truth = target[:, index]
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        accuracy = float((pred == truth).float().mean())
        per_label[name] = {
            "accuracy": accuracy,
            "f1": f1,
            "positive_support": int(truth.sum()),
        }
        f1_values.append(f1)
        accuracies.append(accuracy)
    return {
        "substructure_macro_f1": mean(f1_values),
        "substructure_macro_accuracy": mean(accuracies),
        "per_substructure": per_label,
    }


def _substructure_component_masks(
    node_features: Tensor, bonds: Tensor
) -> tuple[Tensor, Tensor]:
    """Return node/edge membership for carbonyl, aromatic-ring and amine labels."""

    atoms = node_features[:, 0]
    count = atoms.numel()
    node_membership = torch.zeros(3, count, dtype=torch.bool)
    edge_membership = torch.zeros(3, count, count, dtype=torch.bool)

    carbonyl_edges = (
        ((atoms[:, None] == 6) & (atoms[None, :] == 8))
        | ((atoms[:, None] == 8) & (atoms[None, :] == 6))
    ) & (bonds == DOUBLE)
    edge_membership[0] = carbonyl_edges
    node_membership[0] = carbonyl_edges.any(dim=0) | carbonyl_edges.any(dim=1)

    graph = MoleculeGraph(
        node_features, bonds, infer_substructure_mask(node_features, bonds)
    )
    labels = substructure_labels(graph).bool()
    if labels[1]:
        aromatic_edges = bonds == AROMATIC
        edge_membership[1] = aromatic_edges
        node_membership[1] = aromatic_edges.any(dim=0) | aromatic_edges.any(dim=1)

    for nitrogen in torch.where(atoms == 7)[0].tolist():
        for carbon in torch.where(
            (atoms == 6) & (bonds[nitrogen] == SINGLE)
        )[0].tolist():
            if not ((atoms == 8) & (bonds[carbon] == DOUBLE)).any():
                node_membership[2, nitrogen] = True
                node_membership[2, carbon] = True
                edge_membership[2, nitrogen, carbon] = True
                edge_membership[2, carbon, nitrogen] = True
    return node_membership, edge_membership


def _affected_substructure_mask(
    batch: MoleculeBatch, corruption: "ControlledCorruption"
) -> Tensor:
    affected = torch.zeros(batch.node_features.size(0), 3, dtype=torch.bool)
    for batch_index in range(batch.node_features.size(0)):
        count = int(batch.node_mask[batch_index].sum())
        x = batch.node_features[batch_index, :count]
        e = batch.bonds[batch_index, :count, :count]
        node_membership, edge_membership = _substructure_component_masks(x, e)
        corrupted_nodes = corruption.node_mask[batch_index, :count]
        corrupted_edges = corruption.pair_mask[batch_index, :count, :count]
        for label_index in range(3):
            affected[batch_index, label_index] = bool(
                (node_membership[label_index] & corrupted_nodes).any()
                or (edge_membership[label_index] & corrupted_edges).any()
            )
    return affected


def _affected_multilabel_metrics(
    predicted: Tensor, target: Tensor, affected: Tensor
) -> Dict[str, object]:
    predicted = predicted.bool().cpu()
    target = target.bool().cpu()
    affected = affected.bool().cpu()
    per_label: Dict[str, Dict[str, float | int]] = {}
    f1_values, accuracies = [], []
    for index, name in enumerate(SUBSTRUCTURE_NAMES):
        selected = affected[:, index]
        support = int(selected.sum())
        if support:
            pred = predicted[selected, index]
            truth = target[selected, index]
            tp = int((pred & truth).sum())
            fp = int((pred & ~truth).sum())
            fn = int((~pred & truth).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            accuracy = float((pred == truth).float().mean())
            f1_values.append(f1)
            accuracies.append(accuracy)
        else:
            f1 = float("nan")
            accuracy = float("nan")
        per_label[name] = {
            "accuracy": accuracy,
            "f1": f1,
            "affected_support": support,
        }
    return {
        "affected_positive_substructure_macro_f1": (
            mean(f1_values) if f1_values else float("nan")
        ),
        "affected_positive_substructure_macro_accuracy": (
            mean(accuracies) if accuracies else float("nan")
        ),
        "affected_positive_substructure_count": int(affected.sum()),
        "per_affected_positive_substructure": per_label,
    }


@dataclass
class ControlledCorruption:
    node_features: Tensor
    bonds: Tensor
    node_mask: Tensor
    pair_mask: Tensor
    masked_positive_bonds: int
    total_positive_bonds: int


def controlled_corruption(
    batch: MoleculeBatch,
    fraction: float,
    model_config: ModelConfig,
    generator: torch.Generator,
) -> ControlledCorruption:
    """Mask nodes and a balanced set of positive/negative undirected pairs."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("noise fraction must be in (0, 1]")
    noisy_nodes = batch.node_features.clone()
    noisy_bonds = batch.bonds.clone()

    node_candidates = torch.nonzero(batch.node_mask, as_tuple=False)
    node_count = max(1, round(node_candidates.size(0) * fraction))
    selected_nodes = node_candidates[
        torch.randperm(node_candidates.size(0), generator=generator)[:node_count]
    ]
    selected_node_mask = torch.zeros_like(batch.node_mask)
    selected_node_mask[selected_nodes[:, 0], selected_nodes[:, 1]] = True
    vocab_sizes = (
        model_config.atom_vocab_size,
        model_config.charge_vocab_size,
        model_config.aromatic_vocab_size,
        model_config.hybrid_vocab_size,
        model_config.degree_vocab_size,
    )
    for field_index, vocab_size in enumerate(vocab_sizes):
        noisy_nodes[:, :, field_index][selected_node_mask] = vocab_size - 1

    positive = torch.nonzero(batch.pair_mask & (batch.bonds > 0), as_tuple=False)
    negative = torch.nonzero(batch.pair_mask & (batch.bonds == 0), as_tuple=False)
    positive_count = max(1, round(positive.size(0) * fraction))
    positive_count = min(positive_count, positive.size(0))
    negative_count = min(positive_count, negative.size(0))
    selected_positive = positive[
        torch.randperm(positive.size(0), generator=generator)[:positive_count]
    ]
    selected_negative = negative[
        torch.randperm(negative.size(0), generator=generator)[:negative_count]
    ]
    selected_pairs = torch.cat((selected_positive, selected_negative), dim=0)
    selected_pair_upper = torch.zeros_like(batch.pair_mask)
    selected_pair_upper[
        selected_pairs[:, 0], selected_pairs[:, 1], selected_pairs[:, 2]
    ] = True
    selected_pair_both = selected_pair_upper | selected_pair_upper.transpose(1, 2)
    noisy_bonds[selected_pair_both] = model_config.bond_vocab_size - 1
    return ControlledCorruption(
        noisy_nodes,
        noisy_bonds,
        selected_node_mask,
        selected_pair_upper,
        positive_count,
        positive.size(0),
    )


def compute_frequency_modes(
    loader: Iterable[MoleculeBatch], model_config: ModelConfig
) -> tuple[Tensor, int]:
    node_counts = [
        torch.zeros(size - 1, dtype=torch.long)
        for size in (
            model_config.atom_vocab_size,
            model_config.charge_vocab_size,
            model_config.aromatic_vocab_size,
            model_config.hybrid_vocab_size,
            model_config.degree_vocab_size,
        )
    ]
    bond_counts = torch.zeros(model_config.bond_vocab_size - 1, dtype=torch.long)
    for batch in loader:
        for field_index, counts in enumerate(node_counts):
            counts += torch.bincount(
                batch.node_features[:, :, field_index][batch.node_mask],
                minlength=counts.numel(),
            )[: counts.numel()]
        bond_counts += torch.bincount(
            batch.bonds[batch.pair_mask], minlength=bond_counts.numel()
        )[: bond_counts.numel()]
    return torch.tensor([int(counts.argmax()) for counts in node_counts]), int(
        bond_counts.argmax()
    )


@dataclass
class FormalAccumulator:
    node_predicted: list[list[Tensor]] = field(
        default_factory=lambda: [[] for _ in NODE_FIELD_NAMES]
    )
    node_targets: list[list[Tensor]] = field(
        default_factory=lambda: [[] for _ in NODE_FIELD_NAMES]
    )
    bond_predicted: list[Tensor] = field(default_factory=list)
    bond_targets: list[Tensor] = field(default_factory=list)
    substructure_predicted: list[Tensor] = field(default_factory=list)
    substructure_targets: list[Tensor] = field(default_factory=list)
    substructure_affected: list[Tensor] = field(default_factory=list)
    invalid_atoms: int = 0
    clean_invalid_atoms: int = 0
    total_atoms: int = 0
    masked_positive_bonds: int = 0
    total_positive_bonds: int = 0


def _substructure_batch(node_features: Tensor, bonds: Tensor, node_mask: Tensor) -> Tensor:
    labels = []
    for batch_index in range(node_features.size(0)):
        count = int(node_mask[batch_index].sum())
        x = node_features[batch_index, :count].cpu()
        e = bonds[batch_index, :count, :count].cpu()
        graph = MoleculeGraph(x, e, infer_substructure_mask(x, e))
        labels.append(substructure_labels(graph))
    return torch.stack(labels)


@torch.no_grad()
def evaluate_formal_recovery_once(
    model: BondAwareDiffusionModel | None,
    loader: Iterable[MoleculeBatch],
    fraction: float,
    diffusion_config: DiffusionConfig,
    model_config: ModelConfig,
    device: torch.device,
    seed: int,
    frequency_modes: tuple[Tensor, int] | None = None,
) -> Dict[str, object]:
    if model is None and frequency_modes is None:
        raise ValueError("frequency_modes are required when model is None")
    if model is not None:
        model.eval()
    generator = torch.Generator().manual_seed(seed)
    accumulator = FormalAccumulator()
    timestep = timestep_for_noise(fraction, diffusion_config, device)
    for cpu_batch in loader:
        corrupted = controlled_corruption(
            cpu_batch, fraction, model_config, generator
        )
        batch = cpu_batch.to(device)
        if model is not None:
            output = model(
                corrupted.node_features.to(device),
                corrupted.bonds.to(device),
                batch.node_mask,
                timestep.expand(batch.bonds.size(0)),
            )
            node_predictions = [logits.argmax(dim=-1) for logits in output.node_logits]
            bond_predictions = output.bond_logits.argmax(dim=-1)
        else:
            node_modes, bond_mode = frequency_modes
            node_predictions = [
                torch.full_like(batch.node_features[:, :, field_index], int(mode))
                for field_index, mode in enumerate(node_modes.tolist())
            ]
            bond_predictions = torch.full_like(batch.bonds, bond_mode)

        node_eval_mask = corrupted.node_mask.to(device)
        pair_eval_mask = corrupted.pair_mask.to(device)
        for field_index in range(len(NODE_FIELD_NAMES)):
            accumulator.node_predicted[field_index].append(
                node_predictions[field_index][node_eval_mask].cpu()
            )
            accumulator.node_targets[field_index].append(
                batch.node_features[:, :, field_index][node_eval_mask].cpu()
            )
        accumulator.bond_predicted.append(bond_predictions[pair_eval_mask].cpu())
        accumulator.bond_targets.append(batch.bonds[pair_eval_mask].cpu())

        recovered_nodes = batch.node_features.clone()
        for field_index, predicted in enumerate(node_predictions):
            recovered_nodes[:, :, field_index][node_eval_mask] = predicted[node_eval_mask]
        recovered_bonds = batch.bonds.clone()
        pair_both = pair_eval_mask | pair_eval_mask.transpose(1, 2)
        recovered_bonds[pair_both] = bond_predictions[pair_both]
        invalid, total = valence_violation_counts(
            recovered_nodes, recovered_bonds, batch.node_mask
        )
        clean_invalid, _ = valence_violation_counts(
            batch.node_features, batch.bonds, batch.node_mask
        )
        accumulator.invalid_atoms += invalid
        accumulator.clean_invalid_atoms += clean_invalid
        accumulator.total_atoms += total
        accumulator.masked_positive_bonds += corrupted.masked_positive_bonds
        accumulator.total_positive_bonds += corrupted.total_positive_bonds
        accumulator.substructure_predicted.append(
            _substructure_batch(recovered_nodes, recovered_bonds, batch.node_mask)
        )
        accumulator.substructure_targets.append(
            _substructure_batch(batch.node_features, batch.bonds, batch.node_mask)
        )
        accumulator.substructure_affected.append(
            _affected_substructure_mask(cpu_batch, corrupted)
        )

    node_metrics: Dict[str, Dict[str, float | int]] = {}
    node_accuracies, node_f1s = [], []
    for field_index, field_name in enumerate(NODE_FIELD_NAMES):
        predicted = torch.cat(accumulator.node_predicted[field_index])
        target = torch.cat(accumulator.node_targets[field_index])
        accuracy = float((predicted == target).float().mean())
        macro_f1 = _macro_f1_multiclass(predicted, target)
        node_metrics[field_name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "support": target.numel(),
        }
        node_accuracies.append(accuracy)
        node_f1s.append(macro_f1)
    bond_metrics = _class_statistics(
        torch.cat(accumulator.bond_predicted),
        torch.cat(accumulator.bond_targets),
    )
    all_bond_predicted = torch.cat(accumulator.bond_predicted)
    all_bond_targets = torch.cat(accumulator.bond_targets)
    positive_mask = all_bond_targets > 0
    positive_type_metrics = _class_statistics(
        all_bond_predicted[positive_mask],
        all_bond_targets[positive_mask],
    )
    # Existence is evaluated on the balanced positive/negative candidate set.
    # Type classification is evaluated only on true bonds that were corrupted.
    bond_metrics["bond_type_macro_f1"] = positive_type_metrics[
        "bond_type_macro_f1"
    ]
    bond_metrics["bond_positive_accuracy"] = positive_type_metrics["bond_accuracy"]
    bond_metrics["per_bond_type"] = positive_type_metrics["per_bond_type"]
    substructure_metrics = _multilabel_metrics(
        torch.cat(accumulator.substructure_predicted),
        torch.cat(accumulator.substructure_targets),
    )
    affected_substructure_metrics = _affected_multilabel_metrics(
        torch.cat(accumulator.substructure_predicted),
        torch.cat(accumulator.substructure_targets),
        torch.cat(accumulator.substructure_affected),
    )
    clean_valence_rate = accumulator.clean_invalid_atoms / accumulator.total_atoms
    recovered_valence_rate = accumulator.invalid_atoms / accumulator.total_atoms
    return {
        "requested_noise": fraction,
        "actual_positive_bond_mask_fraction": (
            accumulator.masked_positive_bonds / accumulator.total_positive_bonds
        ),
        "node_recovery_accuracy": mean(node_accuracies),
        "node_recovery_macro_f1": mean(node_f1s),
        "per_node_field": node_metrics,
        **bond_metrics,
        "valence_violation_rate": recovered_valence_rate,
        "clean_valence_violation_rate": clean_valence_rate,
        "valence_violation_delta": recovered_valence_rate - clean_valence_rate,
        "invalid_atoms": accumulator.invalid_atoms,
        "clean_invalid_atoms": accumulator.clean_invalid_atoms,
        "total_atoms": accumulator.total_atoms,
        **substructure_metrics,
        **affected_substructure_metrics,
        "seed": seed,
    }


FORMAL_SCALARS = (
    "node_recovery_accuracy",
    "node_recovery_macro_f1",
    "bond_accuracy",
    "bond_existence_f1",
    "bond_type_macro_f1",
    "bond_positive_accuracy",
    "valence_violation_rate",
    "clean_valence_violation_rate",
    "valence_violation_delta",
    "substructure_macro_f1",
    "substructure_macro_accuracy",
    "affected_positive_substructure_macro_f1",
    "affected_positive_substructure_macro_accuracy",
)


def summarize_formal_runs(
    method: str, noise: float, runs: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method,
        "noise": noise,
        "repeats": len(runs),
        "evaluated_nodes": sum(
            int(value["support"])
            for value in runs[0]["per_node_field"].values()
        ),
        "evaluated_pairs": runs[0]["evaluated_bonds"],
        "total_atoms": runs[0]["total_atoms"],
    }
    for key in FORMAL_SCALARS:
        values = [float(run[key]) for run in runs]
        row[key] = mean(values)
        row[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return row


def save_formal_recovery(
    output_dir: str | Path,
    summaries: Sequence[Mapping[str, object]],
    runs: Mapping[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "formal_recovery.json").write_text(
        json.dumps({"summary": list(summaries), "runs": runs}, indent=2),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in summaries for key in row})
    with (output_dir / "formal_recovery.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
