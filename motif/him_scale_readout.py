"""Three node-type readouts from an unchanged HimNet dual-path encoder.

No upstream files are edited. This interface does not remove any graph nodes.
"""
import copy
import hashlib
import inspect
import textwrap

import torch
from torch import nn

ARMS = ('A', 'S', 'G', 'AS', 'AG', 'SG', 'ASG')


def _expose_nodes(module):
    """Expose the exact pre-mean states, failing if upstream layout changes."""
    source = textwrap.dedent(inspect.getsource(type(module).forward))
    changes = (
        ('mol_vecs = []', 'mol_vecs = []\n    node_vecs = []'),
        ('mol_vec = mol_vec.sum(dim=0) / a_size',
         'node_vecs.append(mol_vec)\n            mol_vec = mol_vec.sum(dim=0) / a_size'),
        ('return mol_vecs', 'return mol_vecs, node_vecs'),
    )
    modified = source
    for old, new in changes:
        if modified.count(old) != 1:
            raise ValueError(f'Unsupported HGNN forward layout: {old}')
        modified = modified.replace(old, new)
    env = dict(type(module).forward.__globals__)
    exec(compile(modified, '<him_scale_node_forward>', 'exec'), env)
    derived = type('NodeExposingHGNN', (type(module),), {'forward': env['forward']})
    module.__class__ = derived
    return hashlib.sha256(source.encode()).hexdigest()


class HimScaleReadout(nn.Module):
    """Copies the encoder only; state_dict keys retain the original structure."""
    def __init__(self, encoder, frozen=True):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.forward_sha256 = _expose_nodes(self.encoder.mpnn_encoder)
        self.frozen = frozen
        if frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, batch):
        scopes = batch.a_scope
        if not scopes or len(scopes) != len(batch.mol_atom_num):
            raise ValueError('Missing molecule scopes/counts')
        counts = []
        for (_, size), atoms in zip(scopes, batch.mol_atom_num):
            atoms = int(atoms)
            motifs = size - atoms - 1
            if atoms < 1 or motifs < 0:
                raise ValueError('Invalid atom/motif/global node counts')
            counts.append((atoms, motifs, 1))
        enc = self.encoder
        base, p_nodes = enc.mpnn_encoder(batch)
        if len(p_nodes) != len(scopes):
            raise ValueError('Empty molecule ranges are unsupported')
        device = next(enc.parameters()).device
        initial = enc.atom_proj(batch.get_components()[0].to(device))
        q_nodes, _ = enc.cross_attention(initial, scopes, batch.mol_atom_num)
        p, q, masks = [], [], []
        for (start, size), ns, pn in zip(scopes, counts, p_nodes):
            qn = q_nodes[start:start + size]
            offset = 0
            pv, qv = [], []
            for n in ns:
                pv.append(pn[offset:offset+n].mean(0) if n else pn.new_zeros(pn.shape[-1]))
                qv.append(qn[offset:offset+n].mean(0) if n else qn.new_zeros(qn.shape[-1]))
                offset += n
            p.append(torch.stack(pv)); q.append(torch.stack(qv))
            masks.append([True, ns[1] > 0, True])
        p, q = torch.stack(p), torch.stack(q)
        present = torch.tensor(masks, device=device)
        z = enc.dropout(enc.act_func(enc.output_layer(p + q)))
        z = z.masked_fill(~present.unsqueeze(-1), 0)
        # The original complete graph readout remains available for verification.
        q_all = torch.stack([q_nodes[s:s+n].mean(0) for s, n in scopes])
        full = enc.dropout(enc.act_func(enc.output_layer(base + q_all)))
        return {'z': z, 'p': p, 'q': q, 'full': full,
                'counts': torch.tensor(counts, device=device), 'present': present}


class ScaleProbe(nn.Module):
    """Predict from cached [batch, A/S/G, hidden] vectors."""
    def __init__(self, hidden, arm, width=128, dropout=.1):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f'Unknown scale arm: {arm}')
        self.arm = arm
        self.indices = tuple('ASG'.index(x) for x in arm)
        self.head = nn.Sequential(nn.Linear(hidden * len(arm), width), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, z):
        return self.head(z[:, self.indices, :].flatten(1)).flatten()


def matched_width(hidden, arm, target_parameters):
    """Closest integer width for this exact two-linear-layer probe."""
    denominator = hidden * len(arm) + 2
    width = max(1, round((target_parameters - 1) / denominator))
    actual = denominator * width + 1
    return {'width': width, 'parameters': actual,
            'relative_error': abs(actual-target_parameters) / target_parameters}
