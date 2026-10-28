"""PNA baseline with complete aggregators and degree scalers."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import PNAConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class PNAModel(nn.Module):
    def __init__(self, num_tasks, degree_histogram, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        if degree_histogram.numel() == 0 or degree_histogram.sum() == 0:
            raise ValueError("PNA requires a training-set degree histogram")
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.convs = nn.ModuleList([
            PNAConv(
                hidden_dim, hidden_dim,
                aggregators=["mean", "min", "max", "std"],
                scalers=["identity", "amplification", "attenuation"],
                deg=degree_histogram, edge_dim=hidden_dim,
                towers=1, pre_layers=1, post_layers=1, divide_input=False,
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    @staticmethod
    def _safe_batch_norm(norm: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
        """Use running statistics when a training batch contains one node.

        BatchNorm cannot estimate a variance from a single value per channel.
        Molecular mini-batches normally contain many nodes, but a final batch
        can consist of one single-atom molecule.  Only that degenerate case
        falls back to the running statistics accumulated by preceding batches.
        """
        if norm.training and values.size(0) == 1:
            return F.batch_norm(
                values,
                norm.running_mean,
                norm.running_var,
                norm.weight,
                norm.bias,
                training=False,
                momentum=0.0,
                eps=norm.eps,
            )
        return norm(values)

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            messages = conv(x, data.edge_index, edge_attr)
            messages = self._safe_batch_norm(norm, messages)
            x = x + self.dropout(torch.relu(messages))
        return self.head(global_mean_pool(x, data.batch))
