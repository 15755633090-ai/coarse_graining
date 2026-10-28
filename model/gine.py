"""GINE baseline built from PyG's official GINEConv operator."""

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GINEModel(nn.Module):
    def __init__(self, num_tasks, hidden_dim=128, num_layers=5, dropout=0.2):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GINE requires at least two message-passing layers")
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)
        for conv in self.convs:
            x = x + self.dropout(torch.relu(conv(x, data.edge_index, edge_attr)))
        return self.head(global_add_pool(x, data.batch))
