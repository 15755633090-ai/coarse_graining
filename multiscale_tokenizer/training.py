"""Seed-separated training loop for property prediction."""
from __future__ import annotations

import csv
import hashlib
import json
import os
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
    MoleculeBatch,
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


@dataclass(frozen=True)
class FormalProtocol:
    """Locked benchmark protocol shared with the earlier formal experiments."""

    batch_size: int
    micro_batch_size: int
    encoder_learning_rate: float
    downstream_learning_rate: float
    weight_decay: float
    dropout: float
    grad_clip: float
    max_epochs: int = 200
    early_stopping_patience: int = 30
    amp_dtype: str = "bf16"


# Lipo selected configurations from model/results_formal/04_diffusion:
#   pretrained_finetune: dropout 0.3, batch 64, encoder 1e-4, head 5e-4
#   pretrained_frozen:   dropout 0.1, batch 32, head 1e-3
# Both used AdamW with weight_decay 1e-5, grad_clip 5.0, BF16 AMP,
# 200 epochs, early-stopping patience 30, micro_batch_size 8 and no LR schedule.
STAGE1_MODES = frozenset({"baseline_finetune", "multiscale_finetune"})
# Execution semantics copied from the earlier formal benchmark training loop.
# The old loop recorded no per-batch level/token diagnostics, ran validation
# and test under torch.no_grad(), and kept the selected state on the host.
LEGACY_EXECUTION = {
    "train_diagnostics": False,
    "eval_no_grad": True,
    "best_state_on_cpu": True,
    "micro_padding": "tighten_to_micro_batch",
    "amp_dtype": "bf16",
    "micro_batch_size": 8,
}
FORMAL_PROTOCOLS: dict[str, FormalProtocol] = {
    "stage1": FormalProtocol(
        batch_size=64,
        micro_batch_size=8,
        encoder_learning_rate=1e-4,
        downstream_learning_rate=5e-4,
        weight_decay=1e-5,
        dropout=0.3,
        grad_clip=5.0,
    ),
    "downstream": FormalProtocol(
        batch_size=32,
        micro_batch_size=8,
        encoder_learning_rate=0.0,
        downstream_learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=0.1,
        grad_clip=5.0,
    ),
}


def formal_protocol(experiment_mode: str) -> FormalProtocol:
    """Return the locked protocol preset for an experiment mode."""

    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"unknown experiment mode {experiment_mode!r}")
    group = "stage1" if experiment_mode in STAGE1_MODES else "downstream"
    return FORMAL_PROTOCOLS[group]


def _selection_metric(spec: TaskSpec) -> str:
    if spec.task_type == "regression":
        return "valid_rmse"
    return f"valid_{spec.primary_metric}"


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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    minimize = metric in {"valid_loss", "valid_rmse", "valid_mae"}
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
        (
            git_identity.get("commit_sha")
            or git_identity.get("commit")
        )
        if isinstance(git_identity, dict) else None
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
    if schema not in {2, 3, 4}:
        raise ValueError("resume requires checkpoint schema 2, 3 or 4")
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
    if training_protocol.get("execution_revision") != saved_protocol.get("execution_revision"):
        mismatches["execution_revision"] = {
            "checkpoint": saved_protocol.get("execution_revision"),
            "current": training_protocol.get("execution_revision"),
        }
    if schema in {3, 4}:
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


def _advance_legacy_decoder_initialization(config) -> None:
    """Replay discarded pretraining modules' CPU RNG draws before the head.

    The original Base constructed these modules before its property head.
    They are temporary here: no decoder weights enter the downstream model.
    Keep dimensions and construction order aligned with BondAwareDiffusionModel.
    """
    for size in (config.atom_vocab_size, config.charge_vocab_size,
                 config.aromatic_vocab_size, config.hybrid_vocab_size,
                 config.degree_vocab_size):
        nn.Linear(config.hidden_dim, size - 1, device="cpu")
    nn.Linear(config.hidden_dim * 6, config.hidden_dim * 2, device="cpu")
    nn.Linear(config.hidden_dim * 2, config.hidden_dim, device="cpu")
    nn.Linear(config.hidden_dim, config.bond_vocab_size - 1, device="cpu")
    nn.Linear(config.hidden_dim * 2, config.graph_dim, device="cpu")


def build_property_model(
    *,
    spec: TaskSpec,
    experiment_mode: str,
    model_seed: int,
    partition_seed: int,
    dropout: float,
    device: torch.device,
    encoder_init_checkpoint: str | Path | None = None,
    base_init_checkpoint: str | Path | None = None,
    region_radius: int = 2,
    atoms_per_center: int = 8,
    max_centers: int = 8,
    eval_views: int = 5,
) -> MultiscaleMolecularModel:
    """Construct a mode with explicitly reproducible shared initialization."""

    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"unknown experiment mode {experiment_mode!r}")
    seed_everything(model_seed)
    if experiment_mode.endswith("_residual_frozen"):
        if base_init_checkpoint is None or encoder_init_checkpoint is not None:
            raise ValueError("residual modes require base_init_checkpoint only")
        saved = _validated_residual_source(base_init_checkpoint, spec)
        base, _, _ = load_property_model(base_init_checkpoint, device=device)
        model = MultiscaleMolecularModel(base.encoder, config=MultiscaleModelConfig(
            hidden_dim=base.config.hidden_dim, output_dim=spec.num_tasks,
            dropout=dropout, default_partition_seed=partition_seed,
            experiment_mode=experiment_mode,
        )).to(device)
        model.base_head.load_state_dict(base.prediction_head.state_dict(), strict=True)
        return model
    if base_init_checkpoint is not None:
        raise ValueError("base_init_checkpoint is only valid for residual modes")
    stage2_modes = {"baseline_stage2_frozen", "multiscale_stage2_frozen", "random_region_stage2_frozen"}
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
    _advance_legacy_decoder_initialization(encoder.config)
    return MultiscaleMolecularModel(
        encoder,
        config=MultiscaleModelConfig(
            hidden_dim=encoder.config.hidden_dim,
            dropout=dropout,
            output_dim=spec.num_tasks,
            default_partition_seed=partition_seed,
            experiment_mode=experiment_mode,
            region_radius=region_radius, atoms_per_center=atoms_per_center,
            max_centers=max_centers, eval_views=eval_views,
        ),
    ).to(device)


def _validated_residual_source(path, spec):
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    if saved.get("experiment_mode") != "baseline_stage2_frozen" or saved.get("task_spec") != asdict(spec):
        raise ValueError("residual source must be a matching Stage-2 Base checkpoint")
    actual, _ = _saved_model_epoch(saved)
    history = saved.get("history", [])
    if (actual is None or actual != saved.get("best_epoch") or not history
            or actual != min(history, key=lambda r: r["valid_rmse"])["epoch"]):
        raise ValueError("residual source must contain the validation-best Base state")
    if spec.task_type != "regression":
        raise ValueError("residual experiments currently support regression only")
    return saved


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
    if downstream:
        groups.append({
            "name": "downstream", "params": downstream,
            "lr": downstream_learning_rate,
        })
    if encoder:
        groups.append({
            "name": "encoder", "params": encoder,
            "lr": encoder_learning_rate,
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


def _slice_property_batch(
    batch: PropertyBatch,
    start: int,
    end: int,
) -> PropertyBatch:
    """Slice one collated batch the way the old benchmark loop did.

    The old ``_slice_property_batch`` re-tightened padding to the largest
    molecule inside the micro-batch, so dense [N, N] message passing never ran
    on the wider padding of the full batch.
    """

    node_mask = batch.graph.node_mask[start:end]
    max_nodes = int(node_mask.sum(dim=1).max().item())
    graph = MoleculeBatch(
        batch.graph.node_features[start:end, :max_nodes],
        batch.graph.bonds[start:end, :max_nodes, :max_nodes],
        node_mask[:, :max_nodes],
        list(batch.graph.smiles[start:end]),
    )
    partitions = (
        None if batch.partitions is None
        else list(batch.partitions[start:end])
    )
    return PropertyBatch(
        graph,
        batch.targets[start:end],
        batch.target_mask[start:end],
        batch.sample_ids[start:end],
        partitions,
    )


def _autocast_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError("amp_dtype must be one of 'none', 'bf16'")


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
    micro_batch_size: int = 0,
    grad_clip: float = 0.0,
    amp_dtype: str = "none",
    collect_diagnostics: bool = True,
) -> dict[str, float]:
    model.train(training)
    _prepare_loader_iteration(loader)
    if amp_dtype != "none":
        _autocast_dtype(amp_dtype)
    # Original benchmark used AMP for training only; validation selects in FP32.
    amp_enabled = training and amp_dtype != "none" and device.type == "cuda"
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
        if training:
            if optimizer is None:
                raise ValueError("optimizer is required in training mode")
            optimizer.zero_grad(set_to_none=True)
            total_weight = max(1.0, float(batch.target_mask.sum()))
            micro_size = micro_batch_size or batch.targets.size(0)
            micro_size = min(micro_size, batch.targets.size(0))
            batch_loss = 0.0
            for start in range(0, batch.targets.size(0), micro_size):
                micro = _slice_property_batch(
                    batch, start, min(batch.targets.size(0), start + micro_size),
                )
                seeds = (
                    None if micro.partitions is not None
                    else _partition_seeds(
                        micro.sample_ids,
                        base_seed=base_partition_seed,
                        epoch=epoch,
                        training=training,
                    )
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=_autocast_dtype(amp_dtype) if amp_enabled else None,
                    enabled=amp_enabled,
                ):
                    output = model(
                        micro.graph.node_features,
                        micro.graph.bonds,
                        micro.graph.node_mask,
                        seeds,
                        micro.partitions,
                    )
                    loss = _loss(
                        output.prediction,
                        micro.targets,
                        micro.target_mask,
                        spec,
                        scaler,
                    )
                    micro_weight = float(micro.target_mask.sum()) / total_weight
                    weighted_loss = loss * micro_weight
                weighted_loss.backward()
                batch_loss += float(weighted_loss.detach())
                if collect_diagnostics:
                    level_norm_sum += (
                        output.level_norms.detach().cpu().sum(dim=0)
                    )
                    token_sum += (
                        output.token_counts.detach().cpu().sum(dim=0)
                    )
                    residual_sum += (
                        output.residual_counts.detach().cpu().sum(dim=0)
                    )
                    present_count += (
                        output.token_counts.detach().cpu().gt(0).sum(dim=0)
                    )
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [
                        parameter for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    grad_clip,
                )
            optimizer.step()
            loss_value = batch_loss
        else:
            seeds = (
                None if batch.partitions is not None
                else _partition_seeds(
                    batch.sample_ids,
                    base_seed=base_partition_seed,
                    epoch=epoch,
                    training=training,
                )
            )
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=_autocast_dtype(amp_dtype) if amp_enabled else None,
                    enabled=amp_enabled,
                ):
                    output = model(
                        batch.graph.node_features,
                        batch.graph.bonds,
                        batch.graph.node_mask,
                        seeds,
                        batch.partitions,
                    )
                    loss = _loss(
                        output.prediction,
                        batch.targets,
                        batch.target_mask,
                        spec,
                        scaler,
                    )
            all_predictions.append(output.prediction.detach().cpu())
            all_targets.append(batch.targets.detach().cpu())
            all_masks.append(batch.target_mask.detach().cpu())
            level_norm_sum += output.level_norms.detach().cpu().sum(dim=0)
            token_sum += output.token_counts.detach().cpu().sum(dim=0)
            residual_sum += output.residual_counts.detach().cpu().sum(dim=0)
            present_count += output.token_counts.detach().cpu().gt(0).sum(dim=0)
            loss_value = float(loss.detach())
        valid_labels = float(batch.target_mask.sum())
        totals["loss_sum"] += loss_value * valid_labels
        totals["valid_labels"] += valid_labels
        totals["samples"] += batch.targets.size(0)
    metrics = {
        "loss": totals["loss_sum"] / max(1.0, totals["valid_labels"]),
    }
    if collect_diagnostics:
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
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in history for k in row)))
        writer.writeheader()
        writer.writerows(history)


def _is_better_validation(
    candidate: dict[str, float],
    best_value: float,
    spec: TaskSpec,
) -> tuple[bool, float]:
    if spec.task_type == "regression":
        value = candidate["rmse"]
        if not np.isfinite(value):
            return False, best_value
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
    epochs: int | None = None,
    batch_size: int | None = None,
    micro_batch_size: int | None = None,
    learning_rate: float | None = None,
    encoder_learning_rate: float | None = None,
    downstream_learning_rate: float | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    grad_clip: float | None = None,
    amp_dtype: str | None = None,
    scheduler: str | None = None,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.3,
    model_seed: int = 0,
    partition_seed: int = 0,
    data_seed: int = 0,
    num_workers: int = 0,
    resume: bool = False,
    patience: int | None = None,
    train_diagnostics: bool = False,
    experiment_mode: str = "multiscale_frozen",
    encoder_init_checkpoint: str | Path | None = None,
    base_init_checkpoint: str | Path | None = None,
    region_radius: int = 2,
    atoms_per_center: int = 8,
    max_centers: int = 8,
    eval_views: int = 5,
) -> dict[str, object]:
    """Train one arm of the locked formal benchmark protocol."""

    if task not in TASK_SPECS:
        raise ValueError(f"unknown task {task!r}")
    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"unknown experiment mode {experiment_mode!r}")
    preset = formal_protocol(experiment_mode)
    epochs = preset.max_epochs if epochs is None else epochs
    batch_size = preset.batch_size if batch_size is None else batch_size
    micro_batch_size = (
        preset.micro_batch_size if micro_batch_size is None
        else micro_batch_size
    )
    encoder_learning_rate = (
        preset.encoder_learning_rate if encoder_learning_rate is None
        else encoder_learning_rate
    )
    downstream_learning_rate = (
        preset.downstream_learning_rate
        if downstream_learning_rate is None else downstream_learning_rate
    )
    weight_decay = preset.weight_decay if weight_decay is None else weight_decay
    dropout = preset.dropout if dropout is None else dropout
    grad_clip = preset.grad_clip if grad_clip is None else grad_clip
    amp_dtype = preset.amp_dtype if amp_dtype is None else amp_dtype
    patience = (
        preset.early_stopping_patience if patience is None else patience
    )
    scheduler = "none" if scheduler is None else scheduler
    stage2_modes = {"baseline_stage2_frozen", "multiscale_stage2_frozen", "random_region_stage2_frozen"}
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
    if micro_batch_size < 0:
        raise ValueError("micro_batch_size must be nonnegative")
    if grad_clip < 0:
        raise ValueError("grad_clip must be nonnegative")
    if amp_dtype not in {"none", "bf16"}:
        raise ValueError("amp_dtype must be one of 'none', 'bf16'")
    if scheduler not in {"none", "plateau"}:
        raise ValueError("scheduler must be one of 'none', 'plateau'")
    if scheduler_patience < 0:
        raise ValueError("scheduler_patience must be nonnegative")
    if not 0.0 < scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be in (0, 1)")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate is not None:
        downstream_learning_rate = learning_rate
    spec = TASK_SPECS[task]
    if experiment_mode.startswith("random_region_") and spec.task_type != "regression":
        raise ValueError("random region v1 supports regression prediction averaging only")
    residual = experiment_mode.endswith("_residual_frozen")
    source = None
    if residual:
        if base_init_checkpoint is None or encoder_learning_rate != 0:
            raise ValueError("residual mode requires Base checkpoint and encoder LR zero")
        source = _validated_residual_source(base_init_checkpoint, spec)
    elif base_init_checkpoint is not None:
        raise ValueError("base_init_checkpoint is only valid for residual modes")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        device_object = torch.device("cpu")

    training_protocol: dict[str, object] = {
        "optimizer": "AdamW",
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "max_epochs": epochs,
        "encoder_learning_rate": encoder_learning_rate,
        "downstream_learning_rate": downstream_learning_rate,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "early_stopping_patience": patience,
        "early_stopping_metric": _selection_metric(spec),
        "grad_clip": grad_clip,
        "amp_dtype": amp_dtype,
        "train_diagnostics": bool(train_diagnostics),
        "eval_no_grad": True,
        "execution_revision": "legacy_base_replay_v1",
        "head_initialization": "legacy_decoder_rng_replay",
        "eval_amp_dtype": "none",
        "deterministic": True,
        "cublas_workspace_config": ":4096:8",
        "evaluation_generator_offsets": [10000, 20000],
        "dropout": dropout,
        "num_workers": num_workers,
        "device_type": device_object.type,
    }
    if experiment_mode.startswith("random_region_"):
        training_protocol["random_regions"] = {
            "version": 1, "radius": region_radius, "atoms_per_center": atoms_per_center,
            "max_centers": max_centers, "eval_views": eval_views,
            "heads": 4, "distance_buckets": "0..8,9..12,13+,disconnected",
            "training_views": 1, "fusion": "concat",
        }
    provenance = _experiment_provenance(
        dataset_root, encoder_init_checkpoint,
    )
    if residual:
        training_protocol.update(residual_version=1, initial_candidate_epoch=-1,
                                 base_precision="fp32", correction_init="zero_last_linear")
        provenance["base_source"] = {"sha256": _file_sha256(Path(base_init_checkpoint)),
                                     "model_state_epoch": source["model_state_epoch"]}
        if source.get("provenance", {}).get("dataset") != provenance["dataset"]:
            raise ValueError("residual source dataset/split provenance mismatch")

    train_dataset = OGBMoleculePropertyDataset(dataset_root, split="train")
    valid_dataset = OGBMoleculePropertyDataset(dataset_root, split="valid")
    test_dataset = OGBMoleculePropertyDataset(dataset_root, split="test")
    if train_dataset.labels.size(1) != spec.num_tasks:
        raise ValueError(
            f"task {task!r} expects {spec.num_tasks} targets, "
            f"but the dataset has {train_dataset.labels.size(1)}"
        )
    scaler = TargetScaler.fit(train_dataset) if spec.task_type == "regression" else None
    if residual:
        if not all(torch.equal(value, source["target_scaler"][key]) for key, value in (("mean", scaler.mean), ("std", scaler.std))):
            raise ValueError("residual source target scaler mismatch")

    generator = torch.Generator().manual_seed(data_seed)
    valid_loader = _make_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        base_partition_seed=partition_seed,
        epoch=0,
        training=False,
        generator=torch.Generator().manual_seed(data_seed + 10000),
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
        generator=torch.Generator().manual_seed(data_seed + 20000),
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
        base_init_checkpoint=base_init_checkpoint,
        region_radius=region_radius, atoms_per_center=atoms_per_center,
        max_centers=max_centers, eval_views=eval_views,
    )
    optimizer = build_optimizer(
        model,
        encoder_learning_rate=encoder_learning_rate,
        downstream_learning_rate=downstream_learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if spec.task_type == "regression" else "max",
            patience=scheduler_patience,
            factor=scheduler_factor,
        )
        if training_protocol["scheduler"] == "plateau" else None
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
    # The old benchmark keeps the selected state on the host, not in VRAM.
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    best_validation = float("inf") if spec.task_type == "regression" else -float("inf")
    epochs_without_improvement = 0
    if residual and not resume:
        initial = _run_epoch(model, valid_loader, device=device_object, spec=spec,
                             scaler=scaler, base_partition_seed=partition_seed,
                             epoch=0, training=False, amp_dtype="none")
        _, best_validation = _is_better_validation(initial, best_validation, spec)
        if not np.isfinite(best_validation):
            raise ValueError("initial residual validation metric is nonfinite")
        history.append({"epoch": -1, "train_loss": None,
                        **{f"valid_{k}": v for k, v in initial.items()},
                        **_learning_rates(optimizer)})
        _write_history(output, history)
        print(f"initial Base candidate (epoch -1): valid_rmse={best_validation:.8f}", flush=True)
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
        if scheduler is not None:
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
        selection_metric = _selection_metric(spec)
        minimize = selection_metric in {"valid_loss", "valid_rmse", "valid_mae"}
        selector = min if minimize else max
        best_epoch = int(selector(
            history,
            key=lambda row: row[selection_metric],
        )["epoch"])
        return {
            "schema_version": 4,
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
            "scheduler_state": (
                None if scheduler is None else scheduler.state_dict()
            ),
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
            micro_batch_size=micro_batch_size,
            grad_clip=grad_clip,
            amp_dtype=amp_dtype,
            collect_diagnostics=train_diagnostics,
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
            micro_batch_size=micro_batch_size,
            grad_clip=grad_clip,
            amp_dtype=amp_dtype,
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
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step(valid_metrics[spec.primary_metric])
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
        micro_batch_size=micro_batch_size,
        grad_clip=grad_clip,
        amp_dtype=amp_dtype,
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
    if checkpoint.get("schema_version") not in {1, 2, 3, 4}:
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
