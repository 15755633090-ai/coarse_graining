"""AttentiveFP baseline using PyG's official implementation."""

from torch import nn
from torch_geometric.nn import AttentiveFP
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class AttentiveFPModel(nn.Module):
    def __init__(self, num_tasks, hidden_dim=128, num_layers=3, num_timesteps=2, dropout=0.2):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.network = AttentiveFP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=num_tasks,
            edge_dim=hidden_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, data):
        return self.network(
            self.atom_encoder(data.x), data.edge_index,
            self.bond_encoder(data.edge_attr), data.batch,
        )
