"""Small edge-aware GIN layers implemented with torch.index_add."""
import torch
from torch import Tensor, nn


class EdgeGINLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.edge_projection = nn.Linear(edge_dim, hidden_dim) if edge_dim else None
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edges: Tensor, edge_attr: Tensor) -> Tensor:
        # Each undirected edge is stored once and sent in both directions here.
        source = torch.cat((edges[0], edges[1]))
        target = torch.cat((edges[1], edges[0]))
        messages = x[source]
        if self.edge_projection is not None:
            messages = messages + self.edge_projection(torch.cat((edge_attr, edge_attr)))
        messages = torch.relu(messages)
        aggregated = torch.zeros_like(x).index_add(0, target, messages)
        return self.norm(x + self.mlp((1 + self.epsilon) * x + aggregated))


class EdgeGIN(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            EdgeGINLayer(hidden_dim, edge_dim, dropout) for _ in range(layers)
        ])

    def forward(self, x: Tensor, edges: Tensor, edge_attr: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, edges, edge_attr)
        return x
