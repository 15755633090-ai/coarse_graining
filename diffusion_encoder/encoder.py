"""Load the untouched pretrained diffusion encoder for downstream experiments."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import DiffusionEncoder, ModelConfig


DEFAULT_CHECKPOINT = Path(__file__).with_name("encoder.pt")


def load_frozen_encoder(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> DiffusionEncoder:
    """Load h1/h2/h3/h4 backbone weights and prohibit accidental fine-tuning."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported clean encoder checkpoint schema")
    config = ModelConfig(**payload["model_config"])
    encoder = DiffusionEncoder(config)
    encoder.load_state_dict(payload["encoder_state"], strict=True)
    encoder.requires_grad_(False).eval()
    return encoder.to(device)
