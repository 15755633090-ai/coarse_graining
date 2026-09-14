"""Configuration for topology construction and the downstream network."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CoarseningConfig:
    radius: int = 4
    center_fraction: float = 0.1
    max_residual_size: int = 4

    def __post_init__(self):
        if not isinstance(self.radius, int) or self.radius < 0:
            raise ValueError("radius must be a nonnegative integer")
        if not 0 < self.center_fraction <= 1:
            raise ValueError("center_fraction must be in (0, 1]")
        if not isinstance(self.max_residual_size, int) or self.max_residual_size < 1:
            raise ValueError("max_residual_size must be a positive integer")


@dataclass(frozen=True)
class NetworkConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    edge_dim: int = 0
    region_layers: int = 2
    coarse_layers: int = 3
    output_dim: int = 1
    dropout: float = 0.1
    region_pool: str = "mean"
    graph_pool: str = "size_weighted_mean"
    edge_reduce: str = "sum"
    use_size_feature: bool = True

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "output_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("edge_dim", "region_layers", "coarse_layers"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.region_pool not in {"mean", "sum"}:
            raise ValueError("region_pool must be mean or sum")
        if self.graph_pool not in {"mean", "sum", "size_weighted_mean"}:
            raise ValueError("invalid graph_pool")
        if self.edge_reduce not in {"sum", "mean"}:
            raise ValueError("edge_reduce must be sum or mean")
