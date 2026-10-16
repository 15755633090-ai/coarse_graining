from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Model dimensions and categorical vocabulary sizes."""

    hidden_dim: int = 192
    num_layers: int = 6
    dropout: float = 0.1
    time_dim: int = 64
    atom_vocab_size: int = 120  # 0=padding, 1..118=atomic number, 119=mask
    charge_vocab_size: int = 12  # encoded formal charge 0..10, 11=mask
    aromatic_vocab_size: int = 3  # false, true, mask
    hybrid_vocab_size: int = 8  # S, SP, SP2, SP3, SP3D, SP3D2, other, mask
    degree_vocab_size: int = 8  # degree 0..6, mask
    bond_vocab_size: int = 6  # no/single/double/triple/aromatic/mask
    graph_dim: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiffusionConfig:
    timesteps: int = 200
    min_noise: float = 0.02
    max_noise: float = 0.85
    random_replace_prob: float = 0.15
    substructure_mask_prob: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    num_workers: int = 0
    use_amp: bool = True
    seed: int = 42
    val_fraction: float = 0.1
    lambda_bond: float = 2.0
    lambda_valence: float = 0.2
    lambda_substructure: float = 1.0
    nonbond_weight: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
