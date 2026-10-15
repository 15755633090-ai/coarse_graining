from pathlib import Path
import hashlib
import json
import torch
from rdkit import Chem

# Deliberately exclude whole-molecule degree, H count, chirality, hybridization,
# ring and conjugation flags. Degree is recomputed from induced connectivity.
ATOM_DIMS = (119, 17, 2, 12)  # element, charge+8, aromatic annotation, internal degree
BOND_CLASSES = 6  # none, single, double, triple, aromatic, other
SCHEMA = 'element-charge-aromatic-induced_degree-v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def code_identity():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def graph_features(mol, members):
    members = list(members)
    if not members or len(set(members)) != len(members): raise ValueError('Invalid motif membership')
    mapping = {a:i for i,a in enumerate(members)}
    edges = torch.zeros(len(members), len(members), dtype=torch.long)
    types = {Chem.BondType.SINGLE:1, Chem.BondType.DOUBLE:2,
             Chem.BondType.TRIPLE:3, Chem.BondType.AROMATIC:4}
    for b in mol.GetBonds():
        u,v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in mapping and v in mapping:
            i,j = mapping[u], mapping[v]
            edges[i,j] = edges[j,i] = types.get(b.GetBondType(),5)
    degree = (edges != 0).sum(-1)
    x = torch.tensor([[mol.GetAtomWithIdx(a).GetAtomicNum(),
                       mol.GetAtomWithIdx(a).GetFormalCharge()+8,
                       int(mol.GetAtomWithIdx(a).GetIsAromatic()), int(degree[i])]
                      for i,a in enumerate(members)], dtype=torch.long)
    for k,d in enumerate(ATOM_DIMS):
        if not bool(((x[:,k]>=0)&(x[:,k]<d)).all()): raise ValueError(f'Attribute outside schema: {k}')
    return {'x':x, 'edges':edges}

def collate_graphs(items):
    if not items: raise ValueError('Cannot collate an empty batch')
    n = max(len(g['x']) for g in items)
    x = torch.zeros(len(items), n, len(ATOM_DIMS), dtype=torch.long)
    e = torch.zeros(len(items), n, n, dtype=torch.long)
    mask = torch.zeros(len(items), n, dtype=torch.bool)
    for k,g in enumerate(items):
        s = len(g['x']); x[k,:s] = g['x']; e[k,:s,:s] = g['edges']; mask[k,:s] = True
    return {'x':x,'edges':e,'mask':mask}

def to_device(batch, device): return {k:v.to(device) for k,v in batch.items()}

def pair_mask(mask):
    n = mask.size(1)
    return mask[:,:,None] & mask[:,None,:] & ~torch.eye(n,dtype=torch.bool,device=mask.device)[None]

def load_dataset(path):
    # This project's own tensor/dict cache. Do not load arbitrary external pickles.
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['schema'] != SCHEMA: raise ValueError('Feature schema mismatch')
    return data
