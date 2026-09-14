from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .config import DiffusionConfig, ModelConfig
from .data import (
    AROMATIC,
    DOUBLE,
    MASK_BOND,
    SINGLE,
    JsonlGraphDataset,
    MoleculeBatch,
    MoleculeGraph,
    SmilesDataset,
    collate_graphs,
)
from .model import BondAwareDiffusionModel


BOND_NAMES = ("no_bond", "single", "double", "triple", "aromatic")
SUBSTRUCTURE_NAMES = ("carbonyl", "aromatic_ring", "amine")
BOND_ORDERS = torch.tensor((0.0, 1.0, 2.0, 3.0, 1.5))


def require_non_leaky_checkpoint(checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    marker = checkpoint_path.parent / "LEAKY_DEVELOPMENT_ONLY.md"
    if marker.exists():
        raise ValueError(
            f"Refusing formal evaluation with development/leaky checkpoint "
            f"{checkpoint_path}. Use outputs/ogb_clean/best.pt."
        )


def load_graph_dataset(
    path: str | Path, smiles_column: str = "smiles"
) -> Dataset[MoleculeGraph]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return JsonlGraphDataset(path)
    return SmilesDataset.from_file(path, smiles_column)


def load_diffusion_checkpoint(
    checkpoint_path: str | Path, device: str | torch.device
) -> tuple[BondAwareDiffusionModel, DiffusionConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BondAwareDiffusionModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    diffusion_config = DiffusionConfig(**checkpoint.get("diffusion_config", {}))
    return model.to(device).eval(), diffusion_config


def timestep_for_noise(
    noise_fraction: float, config: DiffusionConfig, device: torch.device
) -> Tensor:
    steps = torch.arange(config.timesteps, device=device)
    phase = steps.float() / max(1, config.timesteps - 1)
    schedule = config.min_noise + (config.max_noise - config.min_noise) * (
        1.0 - torch.cos(phase * torch.pi / 2.0)
    )
    return torch.argmin((schedule - noise_fraction).abs())


def mask_existing_bonds(
    batch: MoleculeBatch, fraction: float, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Mask a fixed fraction of existing undirected bonds across the batch."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("noise fraction must be in (0, 1]")
    noisy = batch.bonds.clone()
    masked_upper = torch.zeros_like(batch.bonds, dtype=torch.bool)
    candidates = torch.nonzero(
        batch.pair_mask & (batch.bonds > 0), as_tuple=False
    )
    if candidates.numel():
        count = max(1, int(round(candidates.size(0) * fraction)))
        chosen = candidates[
            torch.randperm(candidates.size(0), generator=generator)[:count]
        ]
        masked_upper[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    masked = masked_upper | masked_upper.transpose(1, 2)
    noisy[masked] = MASK_BOND
    return noisy, masked_upper


def _class_statistics(predicted: Tensor, target: Tensor) -> Dict[str, object]:
    predicted = predicted.long().cpu()
    target = target.long().cpu()
    correct = int((predicted == target).sum())
    total = int(target.numel())
    per_type: Dict[str, Dict[str, float | int]] = {}
    f1_values = []
    # Masked targets are real bonds, so bond-type F1 concerns classes 1..4.
    for class_idx in range(1, 5):
        tp = int(((predicted == class_idx) & (target == class_idx)).sum())
        fp = int(((predicted == class_idx) & (target != class_idx)).sum())
        fn = int(((predicted != class_idx) & (target == class_idx)).sum())
        support = int((target == class_idx).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_type[BOND_NAMES[class_idx]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_values.append(f1)
    predicted_exists = predicted > 0
    target_exists = target > 0
    existence_tp = int((predicted_exists & target_exists).sum())
    existence_fp = int((predicted_exists & ~target_exists).sum())
    existence_fn = int((~predicted_exists & target_exists).sum())
    existence_f1 = (
        2 * existence_tp / (2 * existence_tp + existence_fp + existence_fn)
        if 2 * existence_tp + existence_fp + existence_fn
        else 0.0
    )
    return {
        "correct": correct,
        "evaluated_bonds": total,
        "bond_accuracy": correct / total if total else float("nan"),
        "bond_type_macro_f1": mean(f1_values) if f1_values else float("nan"),
        "bond_existence_f1": existence_f1,
        "per_bond_type": per_type,
    }


def atom_valence_limits(node_features: Tensor) -> Tensor:
    atomic_numbers = node_features[:, :, 0]
    formal_charge = node_features[:, :, 1] - 5
    limits = torch.full_like(atomic_numbers, 8, dtype=torch.float)
    base = {
        1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1,
        14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1,
    }
    for atomic_number, maximum in base.items():
        limits[atomic_numbers == atomic_number] = float(maximum)
    limits[(atomic_numbers == 7) & (formal_charge > 0)] = 4.0
    limits[(atomic_numbers == 8) & (formal_charge > 0)] = 3.0
    return limits


def valence_violation_counts(
    node_features: Tensor, bonds: Tensor, node_mask: Tensor
) -> tuple[int, int]:
    orders = BOND_ORDERS.to(bonds.device)[bonds.clamp(0, 4)]
    valence = orders.sum(dim=-1)
    limits = atom_valence_limits(node_features).to(bonds.device)
    invalid = (valence > limits + 1e-6) & node_mask
    return int(invalid.sum()), int(node_mask.sum())


@dataclass
class RecoveryAccumulator:
    predicted: List[Tensor]
    targets: List[Tensor]
    invalid_atoms: int = 0
    clean_invalid_atoms: int = 0
    total_atoms: int = 0
    masked_bonds: int = 0
    total_bonds: int = 0


@torch.no_grad()
def evaluate_recovery_once(
    model: BondAwareDiffusionModel,
    loader: Iterable[MoleculeBatch],
    fraction: float,
    diffusion_config: DiffusionConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    accumulator = RecoveryAccumulator([], [])
    timestep = timestep_for_noise(fraction, diffusion_config, device)
    for cpu_batch in loader:
        noisy_bonds, masked_upper = mask_existing_bonds(cpu_batch, fraction, generator)
        accumulator.masked_bonds += int(masked_upper.sum())
        accumulator.total_bonds += int(
            (cpu_batch.pair_mask & (cpu_batch.bonds > 0)).sum()
        )
        batch = cpu_batch.to(device)
        noisy_bonds = noisy_bonds.to(device)
        masked_upper = masked_upper.to(device)
        timesteps = timestep.expand(batch.bonds.size(0))
        output = model(batch.node_features, noisy_bonds, batch.node_mask, timesteps)
        predicted = output.bond_logits.argmax(dim=-1)
        accumulator.predicted.append(predicted[masked_upper].cpu())
        accumulator.targets.append(batch.bonds[masked_upper].cpu())
        recovered = batch.bonds.clone()
        masked_both = masked_upper | masked_upper.transpose(1, 2)
        recovered[masked_both] = predicted[masked_both]
        invalid, total = valence_violation_counts(
            batch.node_features, recovered, batch.node_mask
        )
        clean_invalid, _ = valence_violation_counts(
            batch.node_features, batch.bonds, batch.node_mask
        )
        accumulator.invalid_atoms += invalid
        accumulator.clean_invalid_atoms += clean_invalid
        accumulator.total_atoms += total
    if not accumulator.predicted:
        raise ValueError("Evaluation dataset contains no bonds")
    metrics = _class_statistics(
        torch.cat(accumulator.predicted), torch.cat(accumulator.targets)
    )
    metrics.update(
        {
            "requested_noise": fraction,
            "actual_masked_fraction": (
                accumulator.masked_bonds / accumulator.total_bonds
                if accumulator.total_bonds
                else float("nan")
            ),
            "invalid_atoms": accumulator.invalid_atoms,
            "clean_invalid_atoms": accumulator.clean_invalid_atoms,
            "total_atoms": accumulator.total_atoms,
            "valence_violation_rate": (
                accumulator.invalid_atoms / accumulator.total_atoms
                if accumulator.total_atoms
                else float("nan")
            ),
            "clean_valence_violation_rate": (
                accumulator.clean_invalid_atoms / accumulator.total_atoms
                if accumulator.total_atoms
                else float("nan")
            ),
            "seed": seed,
        }
    )
    return metrics


def summarize_recovery_runs(
    runs: Sequence[Dict[str, object]], noise: float
) -> Dict[str, object]:
    scalar_keys = (
        "actual_masked_fraction",
        "bond_accuracy",
        "bond_type_macro_f1",
        "bond_existence_f1",
        "valence_violation_rate",
    )
    summary: Dict[str, object] = {
        "noise": noise,
        "repeats": len(runs),
        "evaluated_bonds_per_repeat": runs[0]["evaluated_bonds"],
        "total_atoms_per_repeat": runs[0]["total_atoms"],
        "invalid_atoms_mean": mean(float(run["invalid_atoms"]) for run in runs),
        "invalid_atoms_std": (
            pstdev(float(run["invalid_atoms"]) for run in runs)
            if len(runs) > 1
            else 0.0
        ),
        "clean_invalid_atoms_per_repeat": runs[0]["clean_invalid_atoms"],
        "clean_valence_violation_rate": runs[0]["clean_valence_violation_rate"],
    }
    for key in scalar_keys:
        values = [float(run[key]) for run in runs]
        summary[key] = mean(values)
        summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def save_recovery_results(
    output_dir: str | Path,
    summaries: Sequence[Dict[str, object]],
    all_runs: Dict[str, Sequence[Dict[str, object]]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bond_recovery.json").write_text(
        json.dumps({"summary": summaries, "runs": all_runs}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "bond_recovery.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def has_aromatic_cycle(bonds: Tensor) -> bool:
    adjacency = bonds == AROMATIC
    n = bonds.size(0)
    visited = [False] * n

    def dfs(node: int, parent: int) -> bool:
        visited[node] = True
        for neighbor in torch.where(adjacency[node])[0].tolist():
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    return any(not visited[node] and dfs(node, -1) for node in range(n))


def substructure_labels(graph: MoleculeGraph) -> Tensor:
    atom = graph.node_features[:, 0]
    bonds = graph.bonds
    carbonyl = bool(
        (((atom[:, None] == 6) & (atom[None, :] == 8)) & (bonds == DOUBLE)).any()
    )
    aromatic_ring = has_aromatic_cycle(bonds)
    amine = False
    for nitrogen in torch.where(atom == 7)[0].tolist():
        for carbon in torch.where((atom == 6) & (bonds[nitrogen] == SINGLE))[0].tolist():
            # Exclude amide N-C(=O), retaining a chemically cleaner amine label.
            carbonyl_oxygen = (atom == 8) & (bonds[carbon] == DOUBLE)
            if not carbonyl_oxygen.any():
                amine = True
                break
        if amine:
            break
    return torch.tensor((carbonyl, aromatic_ring, amine), dtype=torch.float)


@torch.no_grad()
def extract_embeddings(
    model: BondAwareDiffusionModel,
    dataset: Dataset[MoleculeGraph],
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    embeddings, labels = [], []
    model.eval()
    offset = 0
    for batch in loader:
        batch_device = batch.to(device)
        embeddings.append(
            model.encode(
                batch_device.node_features,
                batch_device.bonds,
                batch_device.node_mask,
            ).cpu()
        )
        count = batch.node_features.size(0)
        subset_indices = indices[offset : offset + count]
        labels.append(torch.stack([substructure_labels(dataset[i]) for i in subset_indices]))
        offset += count
    return torch.cat(embeddings), torch.cat(labels)


def multilabel_split(
    labels: Tensor, seed: int, train_fraction: float = 0.7, val_fraction: float = 0.15
) -> tuple[List[int], List[int], List[int]]:
    dataset_size = labels.size(0)
    if dataset_size < 10:
        raise ValueError("Latent probe requires at least 10 molecules")
    # Stratify by the 3-bit substructure signature. This preserves combinations
    # better than a plain random split while remaining dependency-free.
    groups: Dict[int, List[int]] = {}
    codes = (
        labels[:, 0].long()
        + 2 * labels[:, 1].long()
        + 4 * labels[:, 2].long()
    )
    for index, code in enumerate(codes.tolist()):
        groups.setdefault(code, []).append(index)
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for group in groups.values():
        rng.shuffle(group)
        count = len(group)
        if count >= 3:
            val_count = max(1, round(count * val_fraction))
            test_count = max(1, round(count * (1.0 - train_fraction - val_fraction)))
            if val_count + test_count >= count:
                test_count = 1
                val_count = 1
            train_count = count - val_count - test_count
        else:
            train_count, val_count = count, 0
        train.extend(group[:train_count])
        validation.extend(group[train_count : train_count + val_count])
        test.extend(group[train_count + val_count :])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    # Extremely small/imbalanced sets can leave a held-out split empty.
    for destination in (validation, test):
        if not destination and len(train) > 1:
            destination.append(train.pop())
    return train, validation, test


def binary_auc(scores: Tensor, labels: Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks_sorted = torch.empty(scores.numel(), dtype=torch.float)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Average 1-based ranks give the standard 0.5 credit for ties.
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def probe_metrics(logits: Tensor, labels: Tensor) -> Dict[str, object]:
    probabilities = logits.sigmoid()
    predicted = probabilities >= 0.5
    target = labels.bool()
    per_label: Dict[str, Dict[str, float | int]] = {}
    f1s, accuracies, aucs = [], [], []
    for index, name in enumerate(SUBSTRUCTURE_NAMES):
        pred = predicted[:, index]
        truth = target[:, index]
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        accuracy = float((pred == truth).float().mean())
        auc = binary_auc(probabilities[:, index], truth)
        per_label[name] = {
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": auc,
            "positive_support": int(truth.sum()),
        }
        f1s.append(f1)
        accuracies.append(accuracy)
        if not math.isnan(auc):
            aucs.append(auc)
    majority = torch.maximum(target.float().mean(dim=0), 1 - target.float().mean(dim=0))
    return {
        "macro_accuracy": mean(accuracies),
        "macro_f1": mean(f1s),
        "macro_roc_auc": mean(aucs) if aucs else float("nan"),
        "majority_baseline_accuracy": float(majority.mean()),
        "per_substructure": per_label,
    }


def train_linear_probe(
    train_z: Tensor,
    train_y: Tensor,
    val_z: Tensor,
    val_y: Tensor,
    test_z: Tensor,
    test_y: Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[nn.Linear, Dict[str, object]]:
    torch.manual_seed(seed)
    probe = nn.Linear(train_z.size(1), train_y.size(1))
    positives = train_y.sum(dim=0)
    negatives = train_y.size(0) - positives
    positive_weight = negatives / positives.clamp_min(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = None
    best_val = float("inf")
    patience, stale = 30, 0
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(probe(train_z), train_y)
        loss.backward()
        optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_loss = float(criterion(probe(val_z), val_y))
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        metrics = probe_metrics(probe(test_z), test_y)
    metrics.update(
        {
            "best_validation_bce": best_val,
            "train_size": train_z.size(0),
            "validation_size": val_z.size(0),
            "test_size": test_z.size(0),
            "encoder_frozen": True,
            "probe": "single linear layer",
        }
    )
    return probe, metrics
