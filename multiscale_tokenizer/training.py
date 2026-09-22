"""Seed-separated training loop for property prediction."""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from diffusion_encoder.data import (
    OGBMoleculePropertyDataset,
    PropertyBatch,
    collate_property_records,
)
from diffusion_encoder.encoder import load_frozen_encoder

from .model import MultiscaleModelConfig, MultiscaleMolecularModel
from .partition import TokenizationConfig, derive_partition_seed


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    num_tasks: int

    def __post_init__(self) -> None:
        if self.task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be regression or classification")
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be positive")

    @property
    def primary_metric(self) -> str:
        if self.task_type == "regression":
            return "rmse"
        return "average_precision" if self.name == "pcba" else "roc_auc"


TASK_SPECS: dict[str, TaskSpec] = {
    "bace": TaskSpec("bace", "classification", 1),
    "bbbp": TaskSpec("bbbp", "classification", 1),
    "clintox": TaskSpec("clintox", "classification", 2),
    "esol": TaskSpec("esol", "regression", 1),
    "freesolv": TaskSpec("freesolv", "regression", 1),
    "hiv": TaskSpec("hiv", "classification", 1),
    "lipo": TaskSpec("lipo", "regression", 1),
    "muv": TaskSpec("muv", "classification", 17),
    "pcba": TaskSpec("pcba", "classification", 128),
    "sider": TaskSpec("sider", "classification", 27),
    "tox21": TaskSpec("tox21", "classification", 12),
    "toxcast": TaskSpec("toxcast", "classification", 617),
}


@dataclass
class TargetScaler:
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, dataset: OGBMoleculePropertyDataset) -> "TargetScaler":
        count = torch.zeros(dataset.labels.size(1), dtype=torch.float64)
        total = torch.zeros_like(count)
        square_total = torch.zeros_like(count)
        for target in dataset.labels:
            valid = torch.isfinite(target)
            values = target[valid].double()
            count[valid] += 1
            total[valid] += values
            square_total[valid] += values.square()
        mean = total / count.clamp_min(1)
        variance = square_total / count.clamp_min(1) - mean.square()
        std = variance.clamp_min(0).sqrt().clamp_min(1e-6)
        return cls(mean.float(), std.float())

    def transform(self, targets: Tensor) -> Tensor:
        return (targets - self.mean.to(targets.device)) / self.std.to(targets.device)

    def inverse(self, targets: Tensor) -> Tensor:
        return targets * self.std.to(targets.device) + self.mean.to(targets.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _partition_seeds(
    sample_ids: Tensor,
    *,
    base_seed: int,
    epoch: int,
    training: bool,
) -> list[int]:
    partition_epoch = epoch if training else 0
    return [
        derive_partition_seed(base_seed, int(sample_id), partition_epoch)
        for sample_id in sample_ids
    ]


def _loss(
    prediction: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    spec: TaskSpec,
    scaler: TargetScaler | None,
) -> Tensor:
    if prediction.shape != targets.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(targets.shape)}"
        )
    valid = target_mask.to(torch.bool)
    if not valid.any():
        return prediction.sum() * 0.0
    prediction = prediction[valid]
    targets = targets[valid]
    if scaler is not None:
        targets = scaler.transform(targets)
    if spec.task_type == "regression":
        return nn.functional.mse_loss(prediction, targets)
    return nn.functional.binary_cross_entropy_with_logits(prediction, targets)


def _regression_metrics(
    prediction: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    scaler: TargetScaler,
) -> dict[str, float]:
    valid = target_mask.to(torch.bool)
    if not valid.any():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    prediction = scaler.inverse(prediction)[valid].detach().cpu().numpy()
    targets = targets[valid].detach().cpu().numpy()
    residual = prediction - targets
    mse = float(np.mean(residual * residual))
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float("nan") if denominator == 0 else 1.0 - float(np.sum(residual ** 2)) / denominator
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
    }


def _classification_metrics(
    prediction: Tensor,
    targets: Tensor,
    target_mask: Tensor,
) -> dict[str, float]:
    probabilities = prediction.sigmoid().detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    mask = target_mask.detach().cpu().numpy().astype(bool)
    roc_auc: list[float] = []
    average_precision: list[float] = []
    for task in range(targets_np.shape[1]):
        valid = mask[:, task] & np.isfinite(targets_np[:, task])
        if not valid.any():
            continue
        labels = targets_np[valid, task]
        scores = probabilities[valid, task]
        if np.unique(labels).size < 2:
            continue
        roc_auc.append(float(roc_auc_score(labels, scores)))
        average_precision.append(float(average_precision_score(labels, scores)))
    return {
        "roc_auc": float(np.mean(roc_auc)) if roc_auc else float("nan"),
        "average_precision": (
            float(np.mean(average_precision))
            if average_precision else float("nan")
        ),
    }


def _metrics(
    prediction: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    spec: TaskSpec,
    scaler: TargetScaler | None,
) -> dict[str, float]:
    if spec.task_type == "regression":
        if scaler is None:
            raise ValueError("regression metrics require a target scaler")
        return _regression_metrics(prediction, targets, target_mask, scaler)
    return _classification_metrics(prediction, targets, target_mask)


def _run_epoch(
    model: MultiscaleMolecularModel,
    loader: DataLoader[PropertyBatch],
    *,
    device: torch.device,
    spec: TaskSpec,
    scaler: TargetScaler | None,
    base_partition_seed: int,
    epoch: int,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(training)
    totals = {"loss": 0.0, "samples": 0.0}
    all_predictions: list[Tensor] = []
    all_targets: list[Tensor] = []
    all_masks: list[Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        seeds = _partition_seeds(
            batch.sample_ids,
            base_seed=base_partition_seed,
            epoch=epoch,
            training=training,
        )
        output = model(
            batch.graph.node_features,
            batch.graph.bonds,
            batch.graph.node_mask,
            seeds,
        )
        loss = _loss(output.prediction, batch.targets, batch.target_mask, spec, scaler)
        if training:
            if optimizer is None:
                raise ValueError("optimizer is required in training mode")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            all_predictions.append(output.prediction.detach().cpu())
            all_targets.append(batch.targets.detach().cpu())
            all_masks.append(batch.target_mask.detach().cpu())
        totals["loss"] += float(loss.detach()) * batch.targets.size(0)
        totals["samples"] += batch.targets.size(0)
    metrics = {"loss": totals["loss"] / max(1.0, totals["samples"])}
    if not training:
        metrics.update(_metrics(
            torch.cat(all_predictions),
            torch.cat(all_targets),
            torch.cat(all_masks),
            spec,
            scaler,
        ))
    return metrics


def train_property_model(
    dataset_root: str | Path,
    *,
    task: str,
    output_dir: str | Path,
    device: str = "cuda",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    model_seed: int = 0,
    partition_seed: int = 0,
    data_seed: int = 0,
    dropout: float = 0.1,
    num_workers: int = 0,
) -> dict[str, object]:
    """Train only the token readout while the diffusion encoder stays frozen."""

    if task not in TASK_SPECS:
        raise ValueError(f"unknown task {task!r}")
    spec = TASK_SPECS[task]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        device_object = torch.device("cpu")

    train_dataset = OGBMoleculePropertyDataset(dataset_root, split="train")
    valid_dataset = OGBMoleculePropertyDataset(dataset_root, split="valid")
    test_dataset = OGBMoleculePropertyDataset(dataset_root, split="test")
    if train_dataset.labels.size(1) != spec.num_tasks:
        raise ValueError(
            f"task {task!r} expects {spec.num_tasks} targets, "
            f"but the dataset has {train_dataset.labels.size(1)}"
        )
    scaler = TargetScaler.fit(train_dataset) if spec.task_type == "regression" else None

    generator = torch.Generator().manual_seed(data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_property_records,
        num_workers=num_workers,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_property_records,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_property_records,
        num_workers=num_workers,
    )

    seed_everything(model_seed)
    encoder = load_frozen_encoder(device=device_object)
    model = MultiscaleMolecularModel(
        encoder,
        config=MultiscaleModelConfig(
            hidden_dim=encoder.config.hidden_dim,
            dropout=dropout,
            output_dim=spec.num_tasks,
            default_partition_seed=partition_seed,
        ),
    ).to(device_object)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    for epoch in range(epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device_object,
            spec=spec,
            scaler=scaler,
            base_partition_seed=partition_seed,
            epoch=epoch,
            training=True,
            optimizer=optimizer,
        )
        valid_metrics = _run_epoch(
            model,
            valid_loader,
            device=device_object,
            spec=spec,
            scaler=scaler,
            base_partition_seed=partition_seed,
            epoch=epoch,
            training=False,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            **{
                f"valid_{name}": value
                for name, value in valid_metrics.items()
                if name != "loss"
            },
        }
        history.append(row)
        if valid_metrics["loss"] < best_validation:
            best_validation = valid_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = _run_epoch(
        model,
        test_loader,
        device=device_object,
        spec=spec,
        scaler=scaler,
        base_partition_seed=partition_seed,
        epoch=0,
        training=False,
    )
    checkpoint = {
        "schema_version": 1,
        "task_spec": asdict(spec),
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "target_scaler": None if scaler is None else {
            "mean": scaler.mean,
            "std": scaler.std,
        },
        "model_seed": model_seed,
        "partition_seed": partition_seed,
        "data_seed": data_seed,
        "history": history,
        "selection_metric": "valid_loss",
        "primary_test_metric": spec.primary_metric,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, output / "best.pt")
    return checkpoint


def load_property_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MultiscaleMolecularModel, TargetScaler | None, TaskSpec]:
    """Reload the frozen encoder plus a trained tokenization head."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported multiscale checkpoint schema")
    config_payload = dict(checkpoint["model_config"])
    config_payload["tokenization"] = TokenizationConfig(
        **config_payload["tokenization"],
    )
    config = MultiscaleModelConfig(**config_payload)
    encoder = load_frozen_encoder(device=device)
    model = MultiscaleMolecularModel(encoder, config=config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    spec = TaskSpec(**checkpoint["task_spec"])
    scaler_payload = checkpoint.get("target_scaler")
    scaler = None if scaler_payload is None else TargetScaler(
        scaler_payload["mean"], scaler_payload["std"],
    )
    return model, scaler, spec
