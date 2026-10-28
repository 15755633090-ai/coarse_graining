"""Unified OGB-split benchmark for local PyG and Morgan baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import platform
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

# Required by deterministic CUDA matrix multiplication.  This must be set
# before torch initializes CUDA, so keep it above the torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from ogb.utils.mol import smiles2graph
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
DIFFUSION_ROOT = PROJECT_ROOT / "扩散GNN" / "01_扩散预训练"
for path in (MODEL_ROOT, DIFFUSION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attentivefp import AttentiveFPModel  # noqa: E402
from gine import GINEModel  # noqa: E402
from morgan import MorganConfig, MorganFingerprintModel  # noqa: E402
from pna import PNAModel  # noqa: E402
from bond_diffusion.property_prediction import (  # noqa: E402
    TASK_SPECS,
    TargetScaler,
    better_score,
    build_property_data,
    classification_positive_weight,
    compute_property_metrics,
    primary_score,
    save_benchmark_results,
)


PYG_MODELS = ("gine", "attentivefp", "pna")
MORGAN_MODELS = ("morgan_rf", "morgan_xgboost")
MODELS = (*MORGAN_MODELS, *PYG_MODELS)


class OGBGraphDataset(Dataset):
    def __init__(self, source) -> None:
        self.source = source
        self.cache: dict[int, Data] = {}

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Data:
        if index not in self.cache:
            graph = smiles2graph(self.source.smiles[index])
            self.cache[index] = Data(
                x=torch.from_numpy(graph["node_feat"]).long(),
                edge_index=torch.from_numpy(graph["edge_index"]).long(),
                edge_attr=torch.from_numpy(graph["edge_feat"]).long(),
                y=self.source.targets[index].view(1, -1).float(),
                target_mask=self.source.target_mask[index].view(1, -1).bool(),
                sample_id=torch.tensor([index], dtype=torch.long),
            )
        return self.cache[index]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def degree_histogram(dataset: OGBGraphDataset, indices: list[int]) -> torch.Tensor:
    histogram = torch.zeros(1, dtype=torch.long)
    for index in indices:
        graph = dataset[index]
        degree = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
        counts = torch.bincount(degree)
        if counts.numel() > histogram.numel():
            histogram = F.pad(histogram, (0, counts.numel() - histogram.numel()))
        histogram[: counts.numel()] += counts
    return histogram


def build_model(name: str, num_tasks: int, config, dataset, train_indices):
    common = dict(
        num_tasks=num_tasks,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
    if name == "gine":
        return GINEModel(**common, num_layers=config["layers"])
    if name == "attentivefp":
        return AttentiveFPModel(
            **common,
            num_layers=config["layers"],
            num_timesteps=config["attention_timesteps"],
        )
    if name == "pna":
        return PNAModel(
            **common,
            num_layers=config["layers"],
            degree_histogram=degree_histogram(dataset, train_indices),
        )
    raise ValueError(f"Unknown PyG model: {name}")


def make_loader(dataset, indices, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        shuffle=False,
        num_workers=0,
        generator=generator,
    ) if indices is None else DataLoader(
        torch.utils.data.Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def masked_loss(logits, targets, mask, spec, scaler, positive_weight):
    mask = mask.bool()
    if not mask.any():
        return None
    if spec.task_type == "regression":
        normalized = scaler.transform(targets)
        elements = F.mse_loss(logits, normalized, reduction="none")
    else:
        elements = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=positive_weight.to(logits.device) if positive_weight is not None else None,
        )
    return elements[mask].mean()


@torch.no_grad()
def predict(model, loader, device, scaler, spec):
    model.eval()
    outputs, targets, masks, indices = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        if spec.task_type == "regression":
            output = scaler.inverse(output)
        outputs.append(output.cpu())
        targets.append(batch.y.cpu())
        masks.append(batch.target_mask.cpu())
        indices.append(batch.sample_id.cpu())
    return tuple(map(torch.cat, (outputs, targets, masks, indices)))


def save_predictions(path, sample_indices, predictions, targets, masks, task_names, smiles):
    integer_indices = sample_indices.numpy()
    rows: dict[str, object] = {
        "sample_index": integer_indices,
        "smiles": [smiles[int(index)] for index in integer_indices],
    }
    for task, name in enumerate(task_names):
        rows[f"target__{name}"] = targets[:, task].numpy()
        rows[f"prediction__{name}"] = predictions[:, task].numpy()
        rows[f"observed__{name}"] = masks[:, task].numpy()
    pd.DataFrame(rows).to_csv(path, index=False)


def fit_pyg_candidate(name, graph_dataset, source, splits, spec, seed, args, config):
    seed_everything(seed, args.deterministic)
    device = torch.device(args.device)
    model = build_model(name, spec.num_tasks, config, graph_dataset, splits["train"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=args.weight_decay
    )
    train_targets = source.targets[splits["train"]]
    train_mask = source.target_mask[splits["train"]]
    scaler = TargetScaler.fit(train_targets, train_mask) if spec.task_type == "regression" else None
    positive_weight = (
        classification_positive_weight(train_targets, train_mask)
        if spec.task_type == "classification"
        else None
    )
    train_loader = make_loader(
        graph_dataset, splits["train"], config["batch_size"], True, seed
    )
    valid_loader = make_loader(
        graph_dataset, splits["valid"], config["batch_size"], False, seed
    )
    best_state, best_score, best_epoch, stale = None, None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = masked_loss(
                logits, batch.y, batch.target_mask, spec, scaler, positive_weight
            )
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {name} at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_pred, valid_y, valid_mask, _ = predict(model, valid_loader, device, scaler, spec)
        valid_metrics = compute_property_metrics(valid_pred, valid_y, valid_mask, spec)
        score = primary_score(valid_metrics, spec)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **valid_metrics})
        if best_score is None or better_score(score, best_score, spec):
            best_state = deepcopy(model.state_dict())
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    valid_pred, valid_y, valid_mask, _ = predict(model, valid_loader, device, scaler, spec)
    valid_metrics = compute_property_metrics(valid_pred, valid_y, valid_mask, spec)
    return model, best_state, best_epoch, history, valid_metrics, scaler


def train_pyg(
    name, source, splits, spec, seed, args, run_dir, locked_config=None,
    evaluate_test=True,
):
    graph_dataset = OGBGraphDataset(source)
    candidates = (
        [locked_config]
        if locked_config is not None
        else args.search_spaces[name][: args.search_trials]
    )
    if not candidates:
        raise ValueError(f"No hyperparameter candidates configured for {name}")
    trials = []
    selected = None
    for trial, config in enumerate(candidates):
        model, state, epoch, history, valid_metrics, scaler = fit_pyg_candidate(
            name, graph_dataset, source, splits, spec, seed, args, config
        )
        score = primary_score(valid_metrics, spec)
        trials.append(
            {
                "trial": trial,
                "hyperparameters": config,
                "best_epoch": epoch,
                "validation_metrics": valid_metrics,
            }
        )
        if selected is None or better_score(score, selected[0], spec):
            selected = (score, model, state, epoch, history, valid_metrics, scaler, config)
    _, model, best_state, best_epoch, history, valid_metrics, scaler, config = selected
    device = torch.device(args.device)
    metrics = None
    if evaluate_test:
        test_loader = make_loader(
            graph_dataset, splits["test"], config["batch_size"], False, seed
        )
        predictions, targets, masks, sample_indices = predict(
            model, test_loader, device, scaler, spec
        )
        metrics = compute_property_metrics(predictions, targets, masks, spec)
    torch.save(
        {
            "model": name,
            "seed": seed,
            "epoch": best_epoch,
            "hyperparameters": config,
            "state_dict": best_state,
        },
        run_dir / "best.pt",
    )
    if evaluate_test:
        save_predictions(
            run_dir / "test_predictions.csv",
            sample_indices,
            predictions,
            targets,
            masks,
            spec.label_columns,
            source.smiles,
        )
    (run_dir / "tuning_results.json").write_text(
        json.dumps(trials, indent=2), encoding="utf-8"
    )
    return (
        metrics,
        best_epoch,
        history,
        sum(p.numel() for p in model.parameters()),
        {"validation_metrics": valid_metrics, "selected_hyperparameters": config},
    )


def train_morgan(
    name, source, splits, spec, seed, args, run_dir, locked_config=None,
    evaluate_test=True,
):
    estimator = "rf" if name == "morgan_rf" else "xgboost"
    train = list(splits["train"])
    valid = list(splits["valid"])
    test = list(splits["test"])
    candidates = (
        [locked_config]
        if locked_config is not None
        else args.search_spaces[name][: args.search_trials]
    )
    trials, selected = [], None
    for trial, candidate in enumerate(candidates):
        config = MorganConfig(
            estimator=estimator,
            radius=args.morgan_radius,
            n_bits=args.morgan_bits,
            n_estimators=args.tree_estimators,
            early_stopping_rounds=args.xgb_early_stopping,
            seed=seed,
            n_jobs=args.tree_jobs,
            **candidate,
        )
        model = MorganFingerprintModel(spec.task_type, spec.num_tasks, config)
        model.fit(
            [source.smiles[i] for i in train],
            source.targets[train].numpy(),
            source.target_mask[train].numpy(),
            [source.smiles[i] for i in valid],
            source.targets[valid].numpy(),
            source.target_mask[valid].numpy(),
        )
        valid_predictions = torch.from_numpy(
            model.predict([source.smiles[i] for i in valid])
        )
        valid_targets = source.targets[valid]
        valid_mask = source.target_mask[valid] & torch.isfinite(valid_predictions)
        valid_metrics = compute_property_metrics(
            valid_predictions, valid_targets, valid_mask, spec
        )
        score = primary_score(valid_metrics, spec)
        trial_row = {
            "trial": trial,
            "hyperparameters": candidate,
            "validation_metrics": valid_metrics,
            "best_iterations": [
                getattr(estimator, "best_iteration", None)
                if estimator is not None
                else None
                for estimator in model.estimators
            ],
        }
        trials.append(trial_row)
        if selected is None or better_score(score, selected[0], spec):
            selected = (
                score,
                model,
                candidate,
                valid_metrics,
                trial_row["best_iterations"],
            )
    if selected is None:
        raise RuntimeError(f"No valid Morgan candidate for {name}")
    _, model, candidate, valid_metrics, best_iterations = selected
    metrics = None
    if evaluate_test:
        predictions = torch.from_numpy(model.predict([source.smiles[i] for i in test]))
        targets = source.targets[test]
        masks = source.target_mask[test] & torch.isfinite(predictions)
        metrics = compute_property_metrics(predictions, targets, masks, spec)
        save_predictions(
            run_dir / "test_predictions.csv",
            torch.tensor(test),
            predictions,
            targets,
            masks,
            spec.label_columns,
            source.smiles,
        )
    (run_dir / "tuning_results.json").write_text(
        json.dumps(trials, indent=2), encoding="utf-8"
    )
    fitted_iterations = [value for value in best_iterations if value is not None]
    selected_rounds = max(fitted_iterations) + 1 if fitted_iterations else None
    return (
        metrics,
        selected_rounds,
        [],
        0,
        {
            "validation_metrics": valid_metrics,
            "selected_hyperparameters": candidate,
            "best_iterations": best_iterations,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Unified full-graph molecular baselines")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=sorted(TASK_SPECS), default=["esol"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--search-space", type=Path, default=MODEL_ROOT / "search_spaces.json"
    )
    parser.add_argument("--search-trials", type=int, default=4)
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument("--tree-estimators", type=int, default=500)
    parser.add_argument("--tree-jobs", type=int, default=-1)
    parser.add_argument("--xgb-early-stopping", type=int, default=30)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def environment_manifest() -> dict[str, object]:
    packages = {}
    for name in ("torch", "torch-geometric", "ogb", "rdkit", "scikit-learn", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tuning_protocol(args, dataset_name, model_name, source, splits) -> dict[str, object]:
    model_source = MODEL_ROOT / {
        "morgan_rf": "morgan.py",
        "morgan_xgboost": "morgan.py",
        "gine": "gine.py",
        "attentivefp": "attentivefp.py",
        "pna": "pna.py",
    }[model_name]
    payload = {
        "schema_version": 1,
        "dataset": dataset_name,
        "model": model_name,
        "split_sha256": split_sha256(source, splits),
        "search_candidates": args.search_spaces[model_name][: args.search_trials],
        "tuning_seed": args.tuning_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "optimizer": {
            "pyg": "AdamW",
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "xgb_early_stopping": args.xgb_early_stopping,
            "tree_estimators": args.tree_estimators,
            "tree_jobs": args.tree_jobs,
        },
        "features": {
            "morgan_radius": args.morgan_radius,
            "morgan_bits": args.morgan_bits,
        },
        "execution": {
            "device": args.device,
            "deterministic": args.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "software": environment_manifest(),
        "source_sha256": {
            "benchmark.py": file_sha256(Path(__file__).resolve()),
            model_source.name: file_sha256(model_source),
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"sha256": digest, "payload": payload}


def split_sha256(source, splits) -> str:
    payload = {
        split: [
            {"index": int(index), "smiles": source.smiles[int(index)]}
            for index in indices
        ]
        for split, indices in splits.items()
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_config(
    args, dataset_name: str, model_name: str, seed: int, source, splits,
    role: str = "report", protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "role": role,
        "data_root": str(args.data_root.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "morgan_radius": args.morgan_radius,
        "morgan_bits": args.morgan_bits,
        "tree_estimators": args.tree_estimators,
        "xgb_early_stopping": args.xgb_early_stopping,
        "deterministic": args.deterministic,
        "split_sha256": split_sha256(source, splits),
        "search_trials": args.search_trials,
        "tuning_seed": args.tuning_seed,
        "search_candidates": args.search_spaces[model_name][: args.search_trials],
        "environment": environment_manifest(),
        "tuning_protocol_sha256": protocol["sha256"] if protocol else None,
        "tuning_protocol": protocol["payload"] if protocol else None,
    }


def initialize_run(run_dir: Path, config: dict[str, object]) -> None:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            changed = sorted(
                key for key in set(existing) | set(config) if existing.get(key) != config.get(key)
            )
            raise ValueError(
                f"Existing run has different settings ({', '.join(changed)}): {run_dir}"
            )
        return
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def collect_results(output_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(output_dir.glob("*/*/seed_*/result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.search_space = args.search_space.resolve()
    if min(args.epochs, args.patience, args.search_trials) < 1:
        raise ValueError("epochs, patience and search trials must be positive")
    args.search_spaces = json.loads(args.search_space.read_text(encoding="utf-8"))
    missing = sorted(set(args.models) - set(args.search_spaces))
    if missing:
        raise ValueError(f"Search space is missing models: {', '.join(missing)}")
    if args.tuning_seed in args.seeds:
        raise ValueError("--tuning-seed must be independent of all --seeds")
    rows = []
    for dataset_name in args.datasets:
        source, splits, spec = build_property_data(args.data_root, dataset_name)
        for name in args.models:
            model_dir = args.output_dir / dataset_name / name
            selection_path = model_dir / "selected_config.json"
            split_identity = split_sha256(source, splits)
            search_candidates = args.search_spaces[name][: args.search_trials]
            protocol = tuning_protocol(args, dataset_name, name, source, splits)
            locked_config = None
            if selection_path.exists():
                selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
                expected = {
                    "tuning_protocol_sha256": protocol["sha256"],
                }
                changed = sorted(
                    key for key, value in expected.items()
                    if selection_manifest.get(key) != value
                )
                if changed:
                    raise ValueError(
                        f"Locked configuration is stale ({', '.join(changed)}): {selection_path}"
                    )
                locked_config = selection_manifest["selected_hyperparameters"]
            if locked_config is None:
                tuning_dir = model_dir / "tuning" / f"seed_{args.tuning_seed}"
                tuning_dir.mkdir(parents=True, exist_ok=True)
                initialize_run(
                    tuning_dir,
                    run_config(
                        args, dataset_name, name, args.tuning_seed, source, splits,
                        role="tuning", protocol=protocol,
                    ),
                )
                tuning_result_path = tuning_dir / "result.json"
                if tuning_result_path.exists():
                    tuning_result = json.loads(
                        tuning_result_path.read_text(encoding="utf-8")
                    )
                    selection = {
                        "selected_hyperparameters": tuning_result["selected_hyperparameters"],
                        "validation_metrics": tuning_result["validation_metrics"],
                    }
                elif name in MORGAN_MODELS:
                    _, best_epoch, history, parameters, selection = train_morgan(
                        name, source, splits, spec, args.tuning_seed, args,
                        tuning_dir, evaluate_test=False,
                    )
                else:
                    _, best_epoch, history, parameters, selection = train_pyg(
                        name, source, splits, spec, args.tuning_seed, args,
                        tuning_dir, evaluate_test=False,
                    )
                locked_config = selection["selected_hyperparameters"]
                if not tuning_result_path.exists():
                    tuning_result = {
                        "dataset": dataset_name,
                        "mode": name,
                        "seed": args.tuning_seed,
                        "role": "tuning",
                        "test_metrics": None,
                        "best_epoch": best_epoch,
                        "parameters": parameters,
                        **selection,
                    }
                    tuning_result_path.write_text(
                        json.dumps(tuning_result, indent=2), encoding="utf-8"
                    )
                    (tuning_dir / "history.json").write_text(
                        json.dumps(history, indent=2), encoding="utf-8"
                    )
                selection_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "dataset": dataset_name,
                            "model": name,
                            "split_sha256": split_identity,
                            "tuning_seed": args.tuning_seed,
                            "search_candidates": search_candidates,
                            "selected_hyperparameters": locked_config,
                            "tuning_validation_metrics": selection["validation_metrics"],
                            "test_set_evaluated": False,
                            "tuning_protocol_sha256": protocol["sha256"],
                            "tuning_protocol": protocol["payload"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            for seed in args.seeds:
                run_dir = args.output_dir / dataset_name / name / f"seed_{seed}"
                result_path = run_dir / "result.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                initialize_run(
                    run_dir,
                    run_config(
                        args, dataset_name, name, seed, source, splits,
                        protocol=protocol,
                    ),
                )
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    rows.append(result)
                    print(f"skip completed {dataset_name}/{name}/seed_{seed}")
                    continue
                if name in MORGAN_MODELS:
                    metrics, best_epoch, history, parameters, selection = train_morgan(
                        name, source, splits, spec, seed, args, run_dir, locked_config
                    )
                else:
                    metrics, best_epoch, history, parameters, selection = train_pyg(
                        name, source, splits, spec, seed, args, run_dir, locked_config
                    )
                result = {
                    "dataset": dataset_name,
                    "mode": name,
                    "seed": seed,
                    "test_metrics": metrics,
                    "best_epoch": best_epoch,
                    "parameters": parameters,
                    **selection,
                }
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                (run_dir / "history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
                rows.append(result)
                print(f"done {dataset_name}/{name}/seed_{seed}: {metrics}")
                save_benchmark_results(args.output_dir, collect_results(args.output_dir), {})
    save_benchmark_results(args.output_dir, collect_results(args.output_dir), {})


if __name__ == "__main__":
    main()
