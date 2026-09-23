"""Seed-separated training loop for property prediction."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from diffusion_encoder.data import (
    OGBMoleculePropertyDataset,
    PropertyBatch,
    PropertyRecord,
    collate_property_records,
)
from diffusion_encoder.encoder import DEFAULT_CHECKPOINT, load_encoder

from .model import (
    EXPERIMENT_MODES,
    MultiscaleModelConfig,
    MultiscaleMolecularModel,
)
from .partition import (
    TokenizationConfig,
    derive_partition_seed,
    partition_molecule,
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit_sha": None, "dirty": None}
    return {"commit_sha": commit, "dirty": dirty}


def _dataset_identity(dataset_root: str | Path) -> dict[str, object]:
    root = Path(dataset_root)
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if len(parts) == 1 or parts[0] in {"raw", "split", "mapping"}:
            files.append(path)
    manifest = {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(files)
    }
    combined = hashlib.sha256()
    for name, identity in manifest.items():
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(identity["size"]).encode("ascii"))
        combined.update(b"\0")
        combined.update(str(identity["sha256"]).encode("ascii"))
        combined.update(b"\n")
    return {
        "dataset_name": root.name,
        "combined_sha256": combined.hexdigest(),
        "files": manifest,
    }


GLOBAL_BEST_FIX_COMMIT = "416c297"


def _legacy_saved_model_epoch(payload: dict[str, object]) -> int | None:
    history = list(payload.get("history", []))
    if not history:
        return None
    metric = str(payload.get("selection_metric", "valid_loss"))
    minimize = metric == "valid_loss"
    local_improvements = [
        row for previous, row in zip(history, history[1:])
        if (
            row[metric] < previous[metric]
            if minimize else row[metric] > previous[metric]
        )
    ]
    return int(
        local_improvements[-1]["epoch"] if local_improvements
        else history[0]["epoch"]
    )


def _commit_contains_global_best_fix(commit: object) -> bool | None:
    if not isinstance(commit, str) or not commit:
        return None
    try:
        result = subprocess.run(
            [
                "git", "merge-base", "--is-ancestor",
                GLOBAL_BEST_FIX_COMMIT, commit,
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _saved_model_epoch(
    payload: dict[str, object],
) -> tuple[int | None, str]:
    explicit = payload.get("model_state_epoch")
    if explicit is not None:
        return int(explicit), "explicit_checkpoint_field"
    if payload.get("schema_version") == 2:
        return _legacy_saved_model_epoch(payload), "legacy_schema2_inference"
    provenance = payload.get("provenance")
    git_identity = (
        provenance.get("git") if isinstance(provenance, dict) else None
    )
    producer_commit = (
        git_identity.get("commit") if isinstance(git_identity, dict) else None
    )
    contains_fix = _commit_contains_global_best_fix(producer_commit)
    if contains_fix is True:
        best_epoch = payload.get("best_epoch")
        return (
            None if best_epoch is None else int(best_epoch),
            "producer_contains_global_best_fix",
        )
    if contains_fix is False:
        return (
            _legacy_saved_model_epoch(payload),
            "pre_fix_local_improvement_inference",
        )
    return None, "ambiguous_legacy_checkpoint"


def _stage1_source_identity(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    actual_epoch, epoch_basis = _saved_model_epoch(payload)
    return {
        "name": source.name,
        "size": source.stat().st_size,
        "sha256": _file_sha256(source),
        "schema_version": payload.get("schema_version"),
        "experiment_mode": payload.get("experiment_mode"),
        "task": payload.get("task_spec", {}).get("name"),
        "reported_best_epoch_zero_based": payload.get("best_epoch"),
        "actual_model_state_epoch_zero_based": actual_epoch,
        "actual_model_state_epoch_basis": epoch_basis,
    }


def _experiment_provenance(
    dataset_root: str | Path,
    encoder_init_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    checkpoint = Path(DEFAULT_CHECKPOINT)
    result = {
        "git": _git_identity(),
        "encoder_checkpoint": {
            "name": checkpoint.name,
            "size": checkpoint.stat().st_size,
            "sha256": _file_sha256(checkpoint),
        },
        "dataset": _dataset_identity(dataset_root),
    }
    if encoder_init_checkpoint is not None:
        result["stage1_encoder_source"] = _stage1_source_identity(
            encoder_init_checkpoint,
        )
    return result


def _load_stage1_encoder(
    checkpoint_path: str | Path,
    *,
    spec: TaskSpec,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False,
    )
    if checkpoint.get("experiment_mode") != "baseline_finetune":
        raise ValueError(
            "Stage 2 requires a baseline_finetune property checkpoint"
        )
    if checkpoint.get("task_spec") != asdict(spec):
        raise ValueError("Stage 1 checkpoint task does not match Stage 2")
    actual_epoch, epoch_basis = _saved_model_epoch(checkpoint)
    best_epoch = checkpoint.get("best_epoch")
    if actual_epoch is None:
        raise ValueError(
            "cannot verify which epoch the Stage 1 model_state represents "
            f"({epoch_basis}); use a newly generated checkpoint"
        )
    if best_epoch is None or actual_epoch != int(best_epoch):
        raise ValueError(
            "Stage 2 requires the validation-best Stage 1 encoder, but the "
            f"checkpoint model_state is epoch {actual_epoch + 1} and its "
            f"reported best is epoch {None if best_epoch is None else int(best_epoch) + 1}"
        )
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("Stage 1 checkpoint has no model_state")
    prefix = "encoder."
    encoder_state = {
        name[len(prefix):]: value
        for name, value in model_state.items()
        if name.startswith(prefix)
    }
    encoder = load_encoder(device="cpu", frozen=True)
    if set(encoder_state) != set(encoder.state_dict()):
        raise ValueError("Stage 1 checkpoint encoder state is incomplete")
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.requires_grad_(False).eval()
    return encoder.to(device)


def _validate_resume_protocol(
    checkpoint: dict[str, object],
    *,
    task_spec: dict[str, object],
    experiment_mode: str,
    seeds: dict[str, int],
    training_protocol: dict[str, object],
    provenance: dict[str, object],
) -> None:
    schema = checkpoint.get("schema_version")
    if schema not in {2, 3}:
        raise ValueError("resume requires checkpoint schema 2 or 3")
    expected_top_level = {
        "task_spec": task_spec,
        "experiment_mode": experiment_mode,
        **seeds,
    }
    mismatches = {
        name: {"checkpoint": checkpoint.get(name), "current": expected}
        for name, expected in expected_top_level.items()
        if checkpoint.get(name) != expected
    }
    saved_protocol = checkpoint.get("training_protocol", {})
    if schema == 3:
        if saved_protocol != training_protocol:
            mismatches["training_protocol"] = {
                "checkpoint": saved_protocol,
                "current": training_protocol,
            }
        if checkpoint.get("provenance") != provenance:
            mismatches["provenance"] = {
                "checkpoint": checkpoint.get("provenance"),
                "current": provenance,
            }
    else:
        # Schema 2 is retained only so the already-running development seed 0
        # can finish. Validate every field it recorded, then upgrade on save.
        protocol_overlap = {
            key: training_protocol.get(key) for key in saved_protocol
        }
        if saved_protocol != protocol_overlap:
            mismatches["training_protocol"] = {
                "checkpoint": saved_protocol,
                "current_overlap": protocol_overlap,
            }
        warnings.warn(
            "resuming legacy schema-2 checkpoint without complete provenance; "
            "the next save upgrades it to schema 3",
            RuntimeWarning,
        )
    if mismatches:
        raise ValueError(
            "resume checkpoint does not match the current experiment:\n"
            + json.dumps(mismatches, indent=2, default=str)
        )


def _rng_state(generator: torch.Generator) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "data_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: dict[str, object], generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["data_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def build_property_model(
    *,
    spec: TaskSpec,
    experiment_mode: str,
    model_seed: int,
    partition_seed: int,
    dropout: float,
    device: torch.device,
    encoder_init_checkpoint: str | Path | None = None,
) -> MultiscaleMolecularModel:
    """Construct a mode with explicitly reproducible shared initialization."""

    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"unknown experiment mode {experiment_mode!r}")
    seed_everything(model_seed)
    stage2_modes = {"baseline_stage2_frozen", "multiscale_stage2_frozen"}
    if experiment_mode in stage2_modes:
        if encoder_init_checkpoint is None:
            raise ValueError(
                f"{experiment_mode} requires encoder_init_checkpoint"
            )
        encoder = _load_stage1_encoder(
            encoder_init_checkpoint, spec=spec, device=device,
        )
    else:
        if encoder_init_checkpoint is not None:
            raise ValueError(
                "encoder_init_checkpoint is only valid for "
                "a stage2_frozen mode"
            )
        encoder = load_encoder(
            device=device, frozen=experiment_mode.endswith("_frozen"),
        )
    return MultiscaleMolecularModel(
        encoder,
        config=MultiscaleModelConfig(
            hidden_dim=encoder.config.hidden_dim,
            dropout=dropout,
            output_dim=spec.num_tasks,
            default_partition_seed=partition_seed,
            experiment_mode=experiment_mode,
        ),
    ).to(device)


def build_optimizer(
    model: MultiscaleMolecularModel,
    *,
    encoder_learning_rate: float,
    downstream_learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build named differential-LR groups with complete, unique coverage."""

    groups: list[dict[str, object]] = []
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    downstream = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if encoder:
        groups.append({
            "name": "encoder", "params": encoder,
            "lr": encoder_learning_rate,
        })
    if downstream:
        groups.append({
            "name": "downstream", "params": downstream,
            "lr": downstream_learning_rate,
        })
    if not groups:
        raise RuntimeError("model has no trainable parameters")
    covered = [id(p) for group in groups for p in group["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    if len(covered) != len(set(covered)) or set(covered) != set(expected):
        raise RuntimeError("optimizer parameter groups do not uniquely cover the model")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values = {str(group.get("name", index)): float(group["lr"])
              for index, group in enumerate(optimizer.param_groups)}
    return {
        "encoder_lr": values.get("encoder", 0.0),
        "downstream_lr": values["downstream"],
    }


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


def _collate_with_partitions(
    records: Sequence[PropertyRecord],
    *,
    base_partition_seed: int,
    epoch: int,
    training: bool,
    tokenization: TokenizationConfig,
    use_partitions: bool,
) -> PropertyBatch:
    batch = collate_property_records(records)
    if not use_partitions:
        return batch
    seeds = _partition_seeds(
        batch.sample_ids,
        base_seed=base_partition_seed,
        epoch=epoch,
        training=training,
    )
    batch.partitions = [
        partition_molecule(
            record.graph.bonds,
            node_mask=None,
            seed=seed,
            config=tokenization,
            include_stats=False,
        )
        for record, seed in zip(records, seeds)
    ]
    return batch


class _EpochAwareCollator:
    """Picklable collator whose epoch remains visible to persistent workers."""

    def __init__(
        self,
        *,
        base_partition_seed: int,
        epoch: int,
        training: bool,
        tokenization: TokenizationConfig,
        use_partitions: bool,
    ) -> None:
        self.base_partition_seed = base_partition_seed
        self.training = training
        self.tokenization = tokenization
        self.use_partitions = use_partitions
        # Tensor shared memory works with Windows spawn and does not require a
        # multiprocessing Manager process. Epoch changes happen only between
        # fully exhausted DataLoader iterations.
        self._epoch = torch.tensor(epoch, dtype=torch.long).share_memory_()

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self._epoch.fill_(epoch)

    def __call__(self, records: Sequence[PropertyRecord]) -> PropertyBatch:
        return _collate_with_partitions(
            records,
            base_partition_seed=self.base_partition_seed,
            epoch=self.epoch,
            training=self.training,
            tokenization=self.tokenization,
            use_partitions=self.use_partitions,
        )


def _make_loader(
    dataset: OGBMoleculePropertyDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    base_partition_seed: int,
    epoch: int,
    training: bool,
    generator: torch.Generator | None = None,
    tokenization: TokenizationConfig | None = None,
    pin_memory: bool = False,
    use_partitions: bool = True,
) -> DataLoader[PropertyBatch]:
    collator = _EpochAwareCollator(
        base_partition_seed=base_partition_seed,
        epoch=epoch,
        training=training,
        tokenization=tokenization or TokenizationConfig(),
        use_partitions=use_partitions,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )


def _set_loader_epoch(loader: DataLoader[PropertyBatch], epoch: int) -> None:
    collator = loader.collate_fn
    if not isinstance(collator, _EpochAwareCollator):
        raise TypeError("loader does not use the epoch-aware collator")
    collator.set_epoch(epoch)


def _prepare_loader_iteration(loader: DataLoader[PropertyBatch]) -> None:
    """Preserve the legacy per-iterator RNG trajectory with persistent workers.

    A fresh multiprocessing DataLoader draws one base seed whenever iteration
    starts. PyTorch's persistent-worker reset omits that draw after its first
    iteration. Consuming the same value here keeps sampler and model RNG states
    aligned with the former create-per-epoch implementation.
    """

    if not isinstance(loader, DataLoader):
        return
    iterations = int(getattr(loader, "_protocol_iterations", 0))
    if iterations and loader.num_workers > 0 and loader.persistent_workers:
        torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
    loader._protocol_iterations = iterations + 1


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
    _prepare_loader_iteration(loader)
    totals = {"loss_sum": 0.0, "valid_labels": 0.0, "samples": 0.0}
    level_norm_sum = torch.zeros(4)
    token_sum = torch.zeros(4)
    residual_sum = torch.zeros(4)
    present_count = torch.zeros(4)
    all_predictions: list[Tensor] = []
    all_targets: list[Tensor] = []
    all_masks: list[Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        seeds = (
            None if batch.partitions is not None
            else _partition_seeds(
                batch.sample_ids,
                base_seed=base_partition_seed,
                epoch=epoch,
                training=training,
            )
        )
        output = model(
            batch.graph.node_features,
            batch.graph.bonds,
            batch.graph.node_mask,
            seeds,
            batch.partitions,
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
        level_norm_sum += output.level_norms.detach().cpu().sum(dim=0)
        token_sum += output.token_counts.detach().cpu().sum(dim=0)
        residual_sum += output.residual_counts.detach().cpu().sum(dim=0)
        present_count += output.token_counts.detach().cpu().gt(0).sum(dim=0)
        valid_labels = float(batch.target_mask.sum())
        totals["loss_sum"] += float(loss.detach()) * valid_labels
        totals["valid_labels"] += valid_labels
        totals["samples"] += batch.targets.size(0)
    metrics = {
        "loss": totals["loss_sum"] / max(1.0, totals["valid_labels"]),
    }
    samples = max(1.0, totals["samples"])
    names = ("q", "h2", "h3", "h4")
    for index, name in enumerate(names):
        metrics[f"norm_{name}"] = float(level_norm_sum[index] / samples)
        metrics[f"tokens_{name}"] = float(token_sum[index] / samples)
        metrics[f"residual_{name}"] = float(residual_sum[index] / samples)
        if index:
            metrics[f"p_{name}_present"] = float(
                present_count[index] / samples
            )
    if not training:
        metrics.update(_metrics(
            torch.cat(all_predictions),
            torch.cat(all_targets),
            torch.cat(all_masks),
            spec,
            scaler,
        ))
    return metrics


def _write_history(output: Path, history: list[dict[str, float | int]]) -> None:
    (output / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    if not history:
        return
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _is_better_validation(
    candidate: dict[str, float],
    best_value: float,
    spec: TaskSpec,
) -> tuple[bool, float]:
    if spec.task_type == "regression":
        value = candidate["loss"]
        return (True, value) if value < best_value else (False, best_value)
    value = candidate[spec.primary_metric]
    if not np.isfinite(value):
        return False, best_value
    return (True, value) if value > best_value else (False, best_value)


def train_property_model(
    dataset_root: str | Path,
    *,
    task: str,
    output_dir: str | Path,
    device: str = "cuda",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float | None = None,
    encoder_learning_rate: float = 1e-5,
    downstream_learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.3,
    model_seed: int = 0,
    partition_seed: int = 0,
    data_seed: int = 0,
    dropout: float = 0.1,
    num_workers: int = 4,
    resume: bool = False,
    patience: int = 25,
    experiment_mode: str = "multiscale_frozen",
    encoder_init_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Train one arm of the locked baseline/multiscale protocol."""

    if task not in TASK_SPECS:
        raise ValueError(f"unknown task {task!r}")
    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"unknown experiment mode {experiment_mode!r}")
    stage2_modes = {"baseline_stage2_frozen", "multiscale_stage2_frozen"}
    if experiment_mode in stage2_modes:
        if encoder_init_checkpoint is None:
            raise ValueError(
                f"{experiment_mode} requires encoder_init_checkpoint"
            )
    elif encoder_init_checkpoint is not None:
        raise ValueError(
            "encoder_init_checkpoint is only valid for a stage2_frozen mode"
        )
    if patience < 0:
        raise ValueError("patience must be nonnegative")
    if scheduler_patience < 0:
        raise ValueError("scheduler_patience must be nonnegative")
    if not 0.0 < scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be in (0, 1)")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate is not None:
        downstream_learning_rate = learning_rate
    spec = TASK_SPECS[task]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        device_object = torch.device("cpu")

    training_protocol: dict[str, object] = {
        "optimizer": "AdamW",
        "batch_size": batch_size,
        "max_epochs": epochs,
        "encoder_learning_rate": encoder_learning_rate,
        "downstream_learning_rate": downstream_learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "early_stopping_patience": patience,
        "dropout": dropout,
        "num_workers": num_workers,
        "device_type": device_object.type,
    }
    provenance = _experiment_provenance(
        dataset_root, encoder_init_checkpoint,
    )

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
    valid_loader = _make_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        base_partition_seed=partition_seed,
        epoch=0,
        training=False,
        tokenization=TokenizationConfig(),
        pin_memory=device_object.type == "cuda",
        use_partitions=experiment_mode.startswith("multiscale_"),
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        base_partition_seed=partition_seed,
        epoch=0,
        training=False,
        tokenization=TokenizationConfig(),
        pin_memory=device_object.type == "cuda",
        use_partitions=experiment_mode.startswith("multiscale_"),
    )

    model = build_property_model(
        spec=spec,
        experiment_mode=experiment_mode,
        model_seed=model_seed,
        partition_seed=partition_seed,
        dropout=dropout,
        device=device_object,
        encoder_init_checkpoint=encoder_init_checkpoint,
    )
    optimizer = build_optimizer(
        model,
        encoder_learning_rate=encoder_learning_rate,
        downstream_learning_rate=downstream_learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if spec.task_type == "regression" else "max",
        patience=scheduler_patience,
        factor=scheduler_factor,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        base_partition_seed=partition_seed,
        epoch=0,
        training=True,
        generator=generator,
        tokenization=model.config.tokenization,
        pin_memory=device_object.type == "cuda",
        use_partitions=model.config.uses_multiscale,
    )

    history: list[dict[str, float | int]] = []
    start_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf") if spec.task_type == "regression" else -float("inf")
    epochs_without_improvement = 0
    if resume:
        last_path = output / "last.pt"
        if not last_path.is_file():
            raise FileNotFoundError(f"cannot resume without {last_path}")
        resume_checkpoint = torch.load(
            last_path, map_location="cpu", weights_only=False,
        )
        _validate_resume_protocol(
            resume_checkpoint,
            task_spec=asdict(spec),
            experiment_mode=experiment_mode,
            seeds={
                "model_seed": model_seed,
                "partition_seed": partition_seed,
                "data_seed": data_seed,
            },
            training_protocol=training_protocol,
            provenance=provenance,
        )
        best_path = output / "best.pt"
        if best_path.is_file():
            completed = torch.load(
                best_path, map_location="cpu", weights_only=False,
            )
            if "test_metrics" in completed:
                print("training already complete; returning saved result", flush=True)
                return completed
        model.load_state_dict(resume_checkpoint["last_model_state"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        history = list(resume_checkpoint["history"])
        start_epoch = int(resume_checkpoint["current_epoch"]) + 1
        best_state = resume_checkpoint["best_model_state"]
        best_validation = float(resume_checkpoint["best_selection"])
        epochs_without_improvement = int(
            resume_checkpoint["epochs_without_improvement"]
        )
        _restore_rng_state(resume_checkpoint["rng_state"], generator)
        print(f"resumed from epoch {start_epoch}", flush=True)

        if start_epoch >= epochs:
            print("training epochs already complete; finalizing best model", flush=True)

    def checkpoint_payload(current_epoch: int) -> dict[str, object]:
        selection_metric = (
            "valid_loss" if spec.task_type == "regression"
            else f"valid_{spec.primary_metric}"
        )
        best_epoch = int(min(
            history,
            key=lambda row: row["valid_loss"],
        )["epoch"]) if spec.task_type == "regression" else int(max(
            history,
            key=lambda row: row[f"valid_{spec.primary_metric}"],
        )["epoch"])
        return {
            "schema_version": 3,
            "task_spec": asdict(spec),
            "model_config": asdict(model.config),
            "experiment_mode": experiment_mode,
            "target_scaler": None if scaler is None else {
                "mean": scaler.mean,
                "std": scaler.std,
            },
            "model_seed": model_seed,
            "partition_seed": partition_seed,
            "data_seed": data_seed,
            "history": history,
            "selection_metric": selection_metric,
            "best_selection": best_validation,
            "best_model_state": best_state,
            "best_model_state_epoch": best_epoch,
            "primary_test_metric": spec.primary_metric,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "current_epoch": current_epoch,
            "encoder_lr": _learning_rates(optimizer)["encoder_lr"],
            "downstream_lr": _learning_rates(optimizer)["downstream_lr"],
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": _rng_state(generator),
            "training_protocol": training_protocol,
            "provenance": provenance,
        }

    for epoch in range(start_epoch, epochs):
        _set_loader_epoch(train_loader, epoch)
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
        epoch_lrs = _learning_rates(optimizer)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            **epoch_lrs,
            **{
                f"train_{name}": value
                for name, value in train_metrics.items()
                if name != "loss"
            },
            **{
                f"valid_{name}": value
                for name, value in valid_metrics.items()
                if name != "loss"
            },
        }
        history.append(row)
        _write_history(output, history)
        improved, best_validation = _is_better_validation(
            valid_metrics,
            best_validation,
            spec,
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler_value = (
            valid_metrics["loss"]
            if spec.task_type == "regression"
            else valid_metrics[spec.primary_metric]
        )
        scheduler.step(scheduler_value)
        checkpoint_common = checkpoint_payload(epoch)
        torch.save(
            {
                **checkpoint_common,
                "model_state": best_state,
                "model_state_epoch": checkpoint_common["best_epoch"],
            },
            output / "best.pt",
        )
        torch.save(
            {
                **checkpoint_common,
                "model_state": model.state_dict(),
                "model_state_epoch": epoch,
                "last_model_state": model.state_dict(),
                "last_model_state_epoch": epoch,
            },
            output / "last.pt",
        )
        fields = [
            f"epoch {epoch + 1}/{epochs}",
            f"train_loss={train_metrics['loss']:.6f}",
            f"valid_loss={valid_metrics['loss']:.6f}",
        ]
        if spec.task_type == "regression":
            fields.extend([
                f"valid_rmse={valid_metrics['rmse']:.6f}",
                f"valid_mae={valid_metrics['mae']:.6f}",
                f"valid_r2={valid_metrics['r2']:.6f}",
            ])
        else:
            fields.extend([
                f"valid_roc_auc={valid_metrics['roc_auc']:.6f}",
                f"valid_average_precision={valid_metrics['average_precision']:.6f}",
            ])
        fields.extend([
            f"encoder_lr={epoch_lrs['encoder_lr']:.3g}",
            f"downstream_lr={epoch_lrs['downstream_lr']:.3g}",
        ])
        print(" ".join(fields), flush=True)
        if patience and epochs_without_improvement >= patience:
            print(
                f"early stopping after {patience} epochs without improvement",
                flush=True,
            )
            break

    checkpoint_common = checkpoint_payload(int(history[-1]["epoch"]))
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
        **checkpoint_common,
        "model_state": best_state,
        "model_state_epoch": checkpoint_common["best_epoch"],
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, output / "best.pt")
    return checkpoint


def evaluate_property_model(
    checkpoint_path: str | Path,
    *,
    dataset_root: str | Path,
    task: str,
    split: str = "test",
    device: str | torch.device = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, float]:
    """Evaluate the best checkpoint without changing or retraining the model."""

    if split not in {"valid", "test"}:
        raise ValueError("split must be valid or test")
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False,
    )
    if checkpoint.get("task_spec", {}).get("name") != task:
        raise ValueError("checkpoint task does not match the requested task")
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        device_object = torch.device("cpu")
    model, scaler, spec = load_property_model(
        checkpoint_path,
        device=device_object,
    )
    model.eval()
    dataset = OGBMoleculePropertyDataset(dataset_root, split=split)
    if dataset.labels.size(1) != spec.num_tasks:
        raise ValueError("dataset target count does not match the checkpoint")
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        base_partition_seed=int(checkpoint["partition_seed"]),
        epoch=0,
        training=False,
        tokenization=model.config.tokenization,
        pin_memory=device_object.type == "cuda",
        use_partitions=model.config.uses_multiscale,
    )
    return _run_epoch(
        model,
        loader,
        device=device_object,
        spec=spec,
        scaler=scaler,
        base_partition_seed=int(checkpoint["partition_seed"]),
        epoch=0,
        training=False,
    )


def load_property_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MultiscaleMolecularModel, TargetScaler | None, TaskSpec]:
    """Reload the frozen encoder plus a trained tokenization head."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") not in {1, 2, 3}:
        raise ValueError("unsupported multiscale checkpoint schema")
    config_payload = dict(checkpoint["model_config"])
    config_payload["tokenization"] = TokenizationConfig(
        **config_payload["tokenization"],
    )
    config = MultiscaleModelConfig(**config_payload)
    encoder = load_encoder(
        device=device,
        frozen=not config.finetunes_encoder,
    )
    model = MultiscaleMolecularModel(encoder, config=config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    spec = TaskSpec(**checkpoint["task_spec"])
    scaler_payload = checkpoint.get("target_scaler")
    scaler = None if scaler_payload is None else TargetScaler(
        scaler_payload["mean"], scaler_payload["std"],
    )
    return model, scaler, spec
