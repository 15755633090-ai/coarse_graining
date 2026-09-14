from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from bond_diffusion.config import ModelConfig
from bond_diffusion.data import MoleculeGraph, collate_graphs
from bond_diffusion.evaluation import (
    load_diffusion_checkpoint,
    load_graph_dataset,
    multilabel_split,
    probe_metrics,
    require_non_leaky_checkpoint,
    substructure_labels,
    train_linear_probe,
)
from bond_diffusion.model import BondAwareDiffusionModel


def file_identity(path: Path) -> Dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def structural_probe_run_config(
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "checkpoint": file_identity(args.checkpoint),
        "input": file_identity(args.input),
        "smiles_column": args.smiles_column,
        "seed": args.seed,
        "methods": list(args.methods),
        "probe_epochs": args.probe_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_samples_per_split": args.max_samples_per_split,
        "edge_mask_fraction": args.edge_mask_fraction,
        "device": args.device,
    }


def initialize_or_validate_probe_config(
    output_dir: Path, config: Dict[str, object]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_config.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != config:
            differing = sorted(
                key
                for key in set(existing) | set(config)
                if existing.get(key) != config.get(key)
            )
            raise ValueError(
                "Output directory belongs to a different structural-probe run. "
                f"Mismatched settings: {', '.join(differing)}. "
                "Use a new --output-dir."
            )
        return

    legacy_paths = (
        output_dir / "structural_probe_progress.json",
        output_dir / "structural_probe.json",
        output_dir / "feature_cache",
    )
    if any(path.exists() for path in legacy_paths):
        raise ValueError(
            f"{output_dir} contains legacy structural-probe results or caches "
            "without run_config.json. Refusing to mix them with a new run; "
            "use a new --output-dir."
        )

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def save_structural_probe_results(
    output_dir: Path, payload: Dict[str, object]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "structural_probe.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows: list[Dict[str, object]] = []
    methods = payload.get("methods", {})
    if isinstance(methods, dict):
        for method, tasks in methods.items():
            if not isinstance(tasks, dict):
                continue
            for task, metrics in tasks.items():
                if not isinstance(metrics, dict):
                    continue
                summary_row: Dict[str, object] = {
                    "row_type": "task_summary",
                    "method": method,
                    "task": task,
                    "substructure": "",
                }
                for metric, value in metrics.items():
                    if metric == "per_substructure" and isinstance(value, dict):
                        for label, label_metrics in value.items():
                            detail_row: Dict[str, object] = {
                                "row_type": "substructure_detail",
                                "method": method,
                                "task": task,
                                "substructure": label,
                            }
                            if isinstance(label_metrics, dict):
                                detail_row.update(label_metrics)
                            rows.append(detail_row)
                    elif isinstance(value, (dict, list)):
                        summary_row[metric] = json.dumps(
                            value, ensure_ascii=False
                        )
                    else:
                        summary_row[metric] = value
                rows.append(summary_row)

    comparison = payload.get("comparison", {})
    if isinstance(comparison, dict):
        for task, metrics in comparison.items():
            comparison_row: Dict[str, object] = {
                "row_type": "comparison",
                "method": "pretrained_minus_baseline",
                "task": task,
                "substructure": "",
            }
            if isinstance(metrics, dict):
                comparison_row.update(metrics)
            rows.append(comparison_row)

    preferred = ["row_type", "method", "task", "substructure"]
    extra_fields = sorted(
        {key for row in rows for key in row if key not in preferred}
    )
    csv_path = output_dir / "structural_probe.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen node/edge/graph linear probes comparing pretrained and "
            "random encoders"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples-per-split", type=int, default=100_000)
    parser.add_argument(
        "--edge-mask-fraction",
        type=float,
        default=0.3,
        help="fraction of true bonds hidden for edge probes (default: 0.3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("pretrained", "random", "raw"),
        default=("pretrained", "random", "raw"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/structural_probe")
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume cached encoder/split features and completed method results",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _limit(features: Tensor, labels: Tensor, maximum: int, seed: int) -> tuple[Tensor, Tensor]:
    if maximum <= 0 or features.size(0) <= maximum:
        return features, labels
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(features.size(0), generator=generator)[:maximum]
    return features[indices], labels[indices]


@torch.no_grad()
def extract_split_features(
    model: BondAwareDiffusionModel,
    dataset: Dataset[MoleculeGraph],
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    maximum: int,
    edge_mask_fraction: float,
) -> Dict[str, tuple[Tensor, Tensor]]:
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_graphs,
    )
    storage: Dict[str, list[Tensor]] = {
        "atom_type_features": [],
        "aromaticity_features": [],
        "valence_degree_features": [],
        "atom_type": [],
        "aromaticity": [],
        "valence_degree": [],
        "bond_existence_features": [],
        "bond_existence": [],
        "bond_type_features": [],
        "bond_type": [],
        "graph_features": [],
        "substructure": [],
    }
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    field_masks = {
        "atom_type_features": (0, model.config.atom_vocab_size - 1),
        "aromaticity_features": (2, model.config.aromatic_vocab_size - 1),
        "valence_degree_features": (4, model.config.degree_vocab_size - 1),
    }
    offset = 0
    for batch in loader:
        batch_device = batch.to(device)
        clean_states = model.encode_nodes(
            batch_device.node_features,
            batch_device.bonds,
            batch_device.node_mask,
        ).cpu()
        masked_field_states = {}
        for feature_name, (field_index, mask_token) in field_masks.items():
            masked_nodes = batch_device.node_features.clone()
            field_values = masked_nodes[:, :, field_index]
            field_values[batch_device.node_mask] = mask_token
            masked_field_states[feature_name] = model.encode_nodes(
                masked_nodes,
                batch_device.bonds,
                batch_device.node_mask,
            ).cpu()

        selected_edges = []
        edge_masked_bonds = batch.bonds.clone()
        for batch_index in range(clean_states.size(0)):
            count = int(batch.node_mask[batch_index].sum())
            e = batch.bonds[batch_index, :count, :count]
            upper = torch.triu(
                torch.ones(count, count, dtype=torch.bool), diagonal=1
            )
            all_positive = torch.nonzero(upper & (e > 0), as_tuple=False)
            positive_count = min(
                all_positive.size(0),
                max(1, round(all_positive.size(0) * edge_mask_fraction)),
            )
            positive = all_positive[
                torch.randperm(all_positive.size(0), generator=generator)[
                    :positive_count
                ]
            ]
            adjacency = (e > 0).float()
            hard_negative_mask = (
                (adjacency @ adjacency > 0) & (e == 0) & upper
            )
            hard_negative = torch.nonzero(hard_negative_mask, as_tuple=False)
            negative_target = min(positive_count, int((upper & (e == 0)).sum()))
            selected_hard = hard_negative[
                torch.randperm(hard_negative.size(0), generator=generator)[
                    : min(negative_target, hard_negative.size(0))
                ]
            ]
            remaining = negative_target - selected_hard.size(0)
            if remaining:
                selected_hard_mask = torch.zeros_like(upper)
                if selected_hard.numel():
                    selected_hard_mask[
                        selected_hard[:, 0], selected_hard[:, 1]
                    ] = True
                easy_pool = torch.nonzero(
                    upper & (e == 0) & ~selected_hard_mask, as_tuple=False
                )
                selected_easy = easy_pool[
                    torch.randperm(easy_pool.size(0), generator=generator)[
                        :remaining
                    ]
                ]
                negative = torch.cat((selected_hard, selected_easy), dim=0)
            else:
                negative = selected_hard
            pairs = torch.cat((positive, negative), dim=0)
            selected_edges.append((positive, pairs))
            if pairs.numel():
                edge_masked_bonds[
                    batch_index, pairs[:, 0], pairs[:, 1]
                ] = model.config.bond_vocab_size - 1
                edge_masked_bonds[
                    batch_index, pairs[:, 1], pairs[:, 0]
                ] = model.config.bond_vocab_size - 1

        edge_states = model.encode_nodes(
            batch_device.node_features,
            edge_masked_bonds.to(device),
            batch_device.node_mask,
        ).cpu()

        for batch_index in range(clean_states.size(0)):
            count = int(batch.node_mask[batch_index].sum())
            clean_h = clean_states[batch_index, :count]
            edge_h = edge_states[batch_index, :count]
            x = batch.node_features[batch_index, :count]
            e = batch.bonds[batch_index, :count, :count]
            for feature_name in field_masks:
                storage[feature_name].append(
                    masked_field_states[feature_name][batch_index, :count]
                )
            storage["atom_type"].append(x[:, 0])
            storage["aromaticity"].append(x[:, 2])
            storage["valence_degree"].append(x[:, 4])

            positive, pairs = selected_edges[batch_index]
            if positive.numel():
                pair_features = torch.cat(
                    (
                        edge_h[pairs[:, 0]] + edge_h[pairs[:, 1]],
                        edge_h[pairs[:, 0]] * edge_h[pairs[:, 1]],
                    ),
                    dim=-1,
                )
                storage["bond_existence_features"].append(pair_features)
                storage["bond_existence"].append(
                    (e[pairs[:, 0], pairs[:, 1]] > 0).long()
                )
                positive_features = torch.cat(
                    (
                        edge_h[positive[:, 0]] + edge_h[positive[:, 1]],
                        edge_h[positive[:, 0]] * edge_h[positive[:, 1]],
                    ),
                    dim=-1,
                )
                storage["bond_type_features"].append(positive_features)
                storage["bond_type"].append(
                    e[positive[:, 0], positive[:, 1]].long()
                )

            summed = clean_h.sum(dim=0)
            averaged = clean_h.mean(dim=0)
            storage["graph_features"].append(torch.cat((summed, averaged))[None])
            original_index = indices[offset + batch_index]
            storage["substructure"].append(
                substructure_labels(dataset[original_index])[None]
            )
        offset += clean_states.size(0)

    tasks = {
        "atom_type": (
            torch.cat(storage["atom_type_features"]),
            torch.cat(storage["atom_type"]),
        ),
        "aromaticity": (
            torch.cat(storage["aromaticity_features"]),
            torch.cat(storage["aromaticity"]),
        ),
        "valence_degree": (
            torch.cat(storage["valence_degree_features"]),
            torch.cat(storage["valence_degree"]),
        ),
        "bond_existence": (
            torch.cat(storage["bond_existence_features"]),
            torch.cat(storage["bond_existence"]),
        ),
        "bond_type": (
            torch.cat(storage["bond_type_features"]),
            torch.cat(storage["bond_type"]),
        ),
        "substructure": (
            torch.cat(storage["graph_features"]),
            torch.cat(storage["substructure"]),
        ),
    }
    return {
        name: _limit(features, labels, maximum, seed + task_index)
        if name != "substructure"
        else (features, labels)
        for task_index, (name, (features, labels)) in enumerate(tasks.items())
    }


def fit_classification_probe(
    train: tuple[Tensor, Tensor],
    validation: tuple[Tensor, Tensor],
    test: tuple[Tensor, Tensor],
    seed: int,
) -> Dict[str, object]:
    train_x, train_y = train
    # Keep validation molecules out of fitting. They are reserved for future
    # probe hyperparameter selection; the held-out test remains untouched.
    _val_x, _val_y = validation
    test_x, test_y = test
    fit_x = train_x.numpy().astype(np.float32, copy=False)
    fit_y = train_y.numpy()
    test_x_np = test_x.numpy().astype(np.float32, copy=False)
    test_y_np = test_y.numpy()
    classifier = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1000,
            tol=1e-4,
            early_stopping=False,
            random_state=seed,
        ),
    )
    classifier.fit(fit_x, fit_y)
    predicted = classifier.predict(test_x_np)
    majority_class = np.bincount(fit_y.astype(np.int64)).argmax()
    majority = np.full_like(test_y_np, majority_class)
    result: Dict[str, object] = {
        "accuracy": float(accuracy_score(test_y_np, predicted)),
        "macro_f1": float(
            f1_score(test_y_np, predicted, average="macro", zero_division=0)
        ),
        "majority_accuracy": float(accuracy_score(test_y_np, majority)),
        "majority_macro_f1": float(
            f1_score(test_y_np, majority, average="macro", zero_division=0)
        ),
        "train_samples": int(fit_y.size),
        "test_samples": int(test_y_np.size),
        "classes": int(np.unique(fit_y).size),
    }
    if np.unique(test_y_np).size == 2 and hasattr(classifier, "predict_proba"):
        result["roc_auc"] = float(
            roc_auc_score(test_y_np, classifier.predict_proba(test_x_np)[:, 1])
        )
    return result


def raw_graph_feature(graph: MoleculeGraph) -> Tensor:
    """Fixed raw-feature histograms with no learned encoder."""

    x = graph.node_features
    upper = torch.triu(
        torch.ones(graph.bonds.shape, dtype=torch.bool), diagonal=1
    )
    groups = (
        torch.bincount(x[:, 0], minlength=119)[:119].float(),
        torch.bincount(x[:, 1], minlength=11)[:11].float(),
        torch.bincount(x[:, 2], minlength=2)[:2].float(),
        torch.bincount(x[:, 3], minlength=7)[:7].float(),
        torch.bincount(x[:, 4], minlength=7)[:7].float(),
        torch.bincount(graph.bonds[upper], minlength=5)[:5].float(),
    )
    normalized = tuple(group / group.sum().clamp_min(1.0) for group in groups)
    return torch.cat(groups + normalized)


def evaluate_raw_graph_probe(
    dataset: Dataset[MoleculeGraph],
    splits: Dict[str, Sequence[int]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    extracted = {}
    for split, indices in splits.items():
        features = torch.stack([raw_graph_feature(dataset[index]) for index in indices])
        labels = torch.stack([substructure_labels(dataset[index]) for index in indices])
        extracted[split] = (features, labels)
    _, metrics = train_linear_probe(
        *extracted["train"],
        *extracted["validation"],
        *extracted["test"],
        args.probe_epochs,
        args.learning_rate,
        args.weight_decay,
        args.seed,
    )
    return {"substructure": metrics}


def evaluate_encoder(
    model: BondAwareDiffusionModel,
    dataset: Dataset[MoleculeGraph],
    splits: Dict[str, Sequence[int]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache_dir = args.output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    method_name = getattr(args, "_active_method")
    extracted = {}
    for split_index, (split, indices) in enumerate(splits.items()):
        cache_path = cache_dir / f"{method_name}_{split}.pt"
        if args.resume and cache_path.exists():
            extracted[split] = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
            print(f"loaded cached features {cache_path.name}", flush=True)
        else:
            extracted[split] = extract_split_features(
                model,
                dataset,
                indices,
                args.batch_size,
                args.num_workers,
                device,
                args.seed + split_index * 1000,
                args.max_samples_per_split,
                args.edge_mask_fraction,
            )
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(extracted[split], temporary)
            temporary.replace(cache_path)
            print(f"cached features {cache_path.name}", flush=True)
    results = {}
    for task_index, task in enumerate(
        ("atom_type", "aromaticity", "valence_degree", "bond_existence", "bond_type")
    ):
        results[task] = fit_classification_probe(
            extracted["train"][task],
            extracted["validation"][task],
            extracted["test"][task],
            args.seed + task_index,
        )

    train_z, train_y = extracted["train"]["substructure"]
    val_z, val_y = extracted["validation"]["substructure"]
    test_z, test_y = extracted["test"]["substructure"]
    _, graph_metrics = train_linear_probe(
        train_z,
        train_y,
        val_z,
        val_y,
        test_z,
        test_y,
        args.probe_epochs,
        args.learning_rate,
        args.weight_decay,
        args.seed,
    )
    results["substructure"] = graph_metrics
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    require_non_leaky_checkpoint(args.checkpoint)
    run_config = structural_probe_run_config(args)
    initialize_or_validate_probe_config(args.output_dir, run_config)

    pretrained, _ = load_diffusion_checkpoint(args.checkpoint, device)
    dataset = load_graph_dataset(args.input, args.smiles_column)
    all_labels = torch.stack(
        [substructure_labels(dataset[index]) for index in range(len(dataset))]
    )
    train_indices, validation_indices, test_indices = multilabel_split(
        all_labels, args.seed
    )
    splits = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    models = {"pretrained": pretrained}
    if "random" in args.methods:
        torch.manual_seed(args.seed)
        models["random"] = BondAwareDiffusionModel(
            ModelConfig(**pretrained.config.to_dict())
        ).to(device)

    progress_path = args.output_dir / "structural_probe_progress.json"
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("config") != run_config:
            raise ValueError(
                "Existing structural-probe progress was created with different "
                "settings. Use a new --output-dir."
            )
        results = progress.get("methods", {})
    else:
        results = {}

    def save_progress() -> None:
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"config": run_config, "methods": results},
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(progress_path)

    for method in args.methods:
        if args.resume and method in results:
            print(f"skip completed {method} encoder probes", flush=True)
            continue
        if method == "raw":
            print("fitting raw graph feature baseline", flush=True)
            results[method] = evaluate_raw_graph_probe(dataset, splits, args)
            save_progress()
            metrics = results[method]["substructure"]
            print(
                f"raw        substructure     "
                f"accuracy={metrics['macro_accuracy']:.4f} "
                f"F1={metrics['macro_f1']:.4f}",
                flush=True,
            )
            continue
        print(f"extracting and probing {method} encoder", flush=True)
        args._active_method = method
        results[method] = evaluate_encoder(
            models[method], dataset, splits, args, device
        )
        save_progress()
        cache_dir = args.output_dir / "feature_cache"
        for split in splits:
            cache_path = cache_dir / f"{method}_{split}.pt"
            if cache_path.exists():
                cache_path.unlink()
        for task, metrics in results[method].items():
            print(
                f"{method:10s} {task:16s} "
                f"accuracy={metrics.get('accuracy', metrics.get('macro_accuracy')):.4f} "
                f"F1={metrics.get('macro_f1', float('nan')):.4f}",
                flush=True,
            )

    comparison = {}
    if "pretrained" in results and "random" in results:
        for task in results["pretrained"]:
            comparison[task] = {}
            for metric in ("accuracy", "macro_accuracy", "macro_f1", "roc_auc", "macro_roc_auc"):
                if (
                    metric in results["pretrained"][task]
                    and metric in results["random"][task]
                ):
                    comparison[task][f"{metric}_delta_pretrained_minus_random"] = (
                        float(results["pretrained"][task][metric])
                        - float(results["random"][task][metric])
                    )
    if "pretrained" in results and "raw" in results:
        comparison.setdefault("substructure", {})
        for metric in ("macro_accuracy", "macro_f1", "macro_roc_auc"):
            if (
                metric in results["pretrained"]["substructure"]
                and metric in results["raw"]["substructure"]
            ):
                comparison["substructure"][
                    f"{metric}_delta_pretrained_minus_raw"
                ] = (
                    float(results["pretrained"]["substructure"][metric])
                    - float(results["raw"]["substructure"][metric])
                )
    payload = {
        "checkpoint": str(args.checkpoint),
        "split_seed": args.seed,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "methods": results,
        "comparison": comparison,
    }
    save_structural_probe_results(args.output_dir, payload)
    print(f"saved structural probe results to {args.output_dir}")


if __name__ == "__main__":
    main()
