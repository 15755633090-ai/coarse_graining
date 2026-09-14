from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler

from .config import DiffusionConfig, ModelConfig, TrainConfig
from .data import MoleculeBatch
from .diffusion import DiscreteGraphDiffusion
from .losses import ChemicalDiffusionLoss
from .model import BondAwareDiffusionModel


class ResumableBatchSampler(Sampler[List[int]]):
    """Deterministic epoch permutation that can start at an exact batch offset."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        epoch: int,
        start_batch: int = 0,
    ):
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.start_batch = start_batch
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(dataset_size, generator=generator).tolist()
        if start_batch > self.total_batches:
            raise ValueError(
                f"start_batch={start_batch} exceeds total_batches={self.total_batches}"
            )

    @property
    def total_batches(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        offset = self.start_batch * self.batch_size
        for start in range(offset, self.dataset_size, self.batch_size):
            yield self.indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return self.total_batches - self.start_batch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_epoch(
    model: BondAwareDiffusionModel,
    diffusion: DiscreteGraphDiffusion,
    criterion: ChemicalDiffusionLoss,
    loader: Iterable[MoleculeBatch],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 5.0,
    initial_totals: Optional[Dict[str, float]] = None,
    initial_steps: int = 0,
    batch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = dict(initial_totals or {})
    steps = initial_steps
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            noisy = diffusion.corrupt(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp and device.type == "cuda",
            ):
                output = model(
                    noisy.node_features, noisy.bonds, batch.node_mask, noisy.timesteps
                )
                losses = criterion(output, batch, noisy)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key, value in losses.scalars().items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
            if batch_callback is not None:
                batch_callback(steps, totals)
    if steps == 0:
        raise ValueError("Data loader produced no batches")
    return {key: value / steps for key, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    model: BondAwareDiffusionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    train_config: TrainConfig,
    metrics: Dict[str, float],
    extra_state: Optional[Dict[str, object]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "diffusion_config": asdict(diffusion_config),
            "train_config": asdict(train_config),
            "metrics": metrics,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
    if extra_state:
        checkpoint.update(extra_state)
    # A killed process must never leave the only resumable checkpoint truncated.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_rng_state(checkpoint: Dict[str, object]) -> None:
    """Restore RNG state when available; old checkpoints remain loadable."""

    state = checkpoint.get("rng_state")
    if not isinstance(state, dict):
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"].detach().to(
        device="cpu", dtype=torch.uint8
    )
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = [
            item.detach().to(device="cpu", dtype=torch.uint8)
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def export_encoder(
    checkpoint_path: str | Path, output_path: str | Path, device: str = "cpu"
) -> None:
    """Create a compact inference checkpoint exposing z=f_theta(G)."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = BondAwareDiffusionModel(model_config)
    model.load_state_dict(checkpoint["model_state"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config.to_dict(),
            "model_state": model.state_dict(),
            "interface": "BondAwareDiffusionModel.encode(node_features, bonds, node_mask)",
        },
        output_path,
    )


def load_encoder(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> BondAwareDiffusionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BondAwareDiffusionModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()
