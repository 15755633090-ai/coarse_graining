"""Motif-independent absorbing attribute diffusion on fixed molecular topology.

Only element and formal charge are node inputs. Aromaticity, hybridization,
valence and H counts are omitted to avoid revealing masked bond attributes.
Degree/connectivity remain observable by design, not reconstruction targets.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem

SCHEMA = 'fixed-topology-attribute-diffusion-v1'
NODE_DIMS = (119, 17)
BOND_DIM = 5  # single, double, triple, aromatic, other; no nonedge class


@dataclass
class Config:
    hidden: int = 128
    layers: int = 3
    steps: int = 100
    min_noise: float = .05
    max_noise: float = .85
    region_probability: float = .2
    region_fraction: float = .2
    bond_weight: float = 1.

    def __post_init__(self):
        if min(self.hidden, self.layers, self.steps) < 1:
            raise ValueError('Positive dimensions and steps required')
        if not 0 <= self.min_noise < self.max_noise <= 1 or not 0 <= self.region_probability <= 1:
            raise ValueError('Invalid noise probabilities')
        if not 0 < self.region_fraction <= 1 or self.bond_weight < 0:
            raise ValueError('Invalid region fraction or loss weight')


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f'Invalid molecule: {smiles}')
    x = torch.tensor([[a.GetAtomicNum(), a.GetFormalCharge()+8] for a in mol.GetAtoms()])
    for k, size in enumerate(NODE_DIMS):
        if not ((x[:, k] >= 0) & (x[:, k] < size)).all():
            raise ValueError('Atom outside feature vocabulary')
    types = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
             Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    bonds = list(mol.GetBonds())
    uv = torch.tensor([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in bonds], dtype=torch.long).reshape(-1, 2).T
    e = torch.tensor([types.get(b.GetBondType(), 4) for b in bonds], dtype=torch.long)
    return {'x': x, 'uv': uv, 'e': e}


def records(data, rows, method=None, aligned=None):
    result = []
    for row in rows:
        item = {'graph': graph(row['smiles']), 'row': row['row']}
        if method is not None:
            groups = aligned[row['row']] if aligned is not None else [data['motifs'][i] for i in row['motifs'][method]]
            item['members'] = [g['members'] for g in groups]
        result.append(item)
    return result


def noise_probability(cfg, t):
    if not 0 <= t <= cfg.steps: raise ValueError('Time outside schedule')
    # Clean state at t=0, followed by an explicit first-step jump to the floor.
    if t == 0: return 0.
    return cfg.min_noise + (cfg.max_noise-cfg.min_noise)*(1-math.cos(math.pi*t/cfg.steps))/2


def corrupt(g, cfg, generator, timestep=None):
    """Nested absorbing masks; correlated neighborhood component has same t schedule.

    Region is selected independently of t; activation uses q(t). Every incident
    bond of a masked atom is also masked. All node fields are masked together.
    Each physical bond is stored once and only duplicated during propagation.
    """
    n, m = len(g['x']), len(g['e'])
    t = int(torch.randint(1, cfg.steps+1, (), generator=generator)) if timestep is None else timestep
    if not 1 <= t <= cfg.steps: raise ValueError('Corruption time must be in 1..T')
    q = noise_probability(cfg, t)
    nodes = torch.rand(n, generator=generator) < q
    edges = torch.rand(m, generator=generator) < q
    if float(torch.rand((), generator=generator)) < cfg.region_probability:
        seed = int(torch.randint(n, (), generator=generator))
        seen, queue = {seed}, [seed]
        adjacency = [[] for _ in range(n)]
        for u, v in g['uv'].T.tolist():
            adjacency[u].append(v); adjacency[v].append(u)
        target = max(1, math.ceil(n*cfg.region_fraction))
        for u in queue:
            for v in adjacency[u]:
                if len(seen) >= target: break
                if v not in seen: seen.add(v); queue.append(v)
            if len(seen) >= target: break
        if float(torch.rand((), generator=generator)) < q:
            nodes[list(seen)] = True
    if m: edges |= nodes[g['uv'][0]] | nodes[g['uv'][1]]
    x, e = g['x'].clone(), g['e'].clone()
    x[nodes] = torch.tensor(NODE_DIMS)
    e[edges] = BOND_DIM
    return x, e, nodes, edges, t


class Encoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = Config(**(config or {})); d = self.cfg.hidden
        self.config = {'hidden': d}
        self.atoms = nn.ModuleList(nn.Embedding(s+1, d) for s in NODE_DIMS)
        self.bonds = nn.Embedding(BOND_DIM+1, d)
        self.time = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.messages = nn.ModuleList(nn.Sequential(nn.Linear(2*d, d), nn.SiLU(), nn.Linear(d, d)) for _ in range(self.cfg.layers))
        self.updates = nn.ModuleList(nn.Sequential(nn.Linear(2*d, d), nn.SiLU(), nn.LayerNorm(d)) for _ in range(self.cfg.layers))
        self.atom_heads = nn.ModuleList(nn.Linear(d, s) for s in NODE_DIMS)
        self.bond_head = nn.Sequential(nn.Linear(2*d, d), nn.SiLU(), nn.Linear(d, BOND_DIM))

    def forward(self, g, x=None, e=None, t=1):
        device = next(self.parameters()).device
        x = (g['x'] if x is None else x).to(device)
        e = (g['e'] if e is None else e).to(device)
        uv = g['uv'].to(device)
        h = sum(emb(x[:, k]) for k, emb in enumerate(self.atoms))
        h = h + self.time(h.new_tensor([[t/self.cfg.steps]]))
        src, dst = torch.cat((uv, uv.flip(0)), dim=1)
        b = self.bonds(torch.cat((e, e)))
        degree = torch.bincount(dst, minlength=len(h)).clamp_min(1)[:, None]
        for message, update in zip(self.messages, self.updates):
            agg = torch.zeros_like(h).index_add(0, dst, message(torch.cat((h[src], b), dim=-1)))
            h = h + update(torch.cat((h, agg/degree), dim=-1))
        # Symmetric physical-bond decoder, independent of edge orientation.
        pair = torch.cat((h[uv[0]]+h[uv[1]], (h[uv[0]]-h[uv[1]]).abs()), dim=-1)
        return h, [head(h) for head in self.atom_heads], self.bond_head(pair)

    def objective(self, item, generator):
        g = item['graph']; x, e, nm, em, t = corrupt(g, self.cfg, generator)
        h, atoms, bonds = self(g, x, e, t)
        nm, em = nm.to(h.device), em.to(h.device)
        zero = h.sum()*0
        node = torch.stack([F.cross_entropy(logits[nm], g['x'][:, k].to(h.device)[nm]) for k, logits in enumerate(atoms)]).mean() if nm.any() else zero
        edge = F.cross_entropy(bonds[em], g['e'].to(h.device)[em]) if em.any() else zero
        return node+self.cfg.bond_weight*edge, {'masked_atoms': int(nm.sum()), 'masked_bonds': int(em.sum()),
                'atoms': len(g['x']), 'bonds': len(g['e']), 't': t}

    def encode_records(self, items, batch_size=8, pair_budget=0):
        # Per-molecule sparse propagation: no padding, no cross-molecule edges.
        vectors = []
        for item in items:
            h, _, _ = self(item['graph'], t=1)
            for members in item['members']:
                if not members or len(set(members)) != len(members) or min(members) < 0 or max(members) >= len(h):
                    raise ValueError('Invalid motif membership')
                vectors.append(h[members].mean(0))
        return torch.stack(vectors) if vectors else next(self.parameters()).new_empty((0, self.cfg.hidden))


def load_encoder(path, device='cpu', random=False):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['schema'] != SCHEMA: raise ValueError('Fixed topology schema mismatch')
    config = dict(ck['model_config'])
    # Pre-floor checkpoints retain their original training noise semantics.
    config.setdefault('min_noise', 0.)
    model = Encoder(config)
    if not random: model.load_state_dict(ck['model'], strict=True)
    return model.to(device), ck
