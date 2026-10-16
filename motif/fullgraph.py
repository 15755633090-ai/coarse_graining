"""Full molecular graph conditioning; imports the actual legacy discrete GNN."""
import sys
from pathlib import Path
from collections import Counter
from contextvars import ContextVar
import torch
from torch import nn
from rdkit import Chem

LEGACY = Path(__file__).resolve().parents[1]/'扩散GNN/01_扩散预训练'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
from bond_diffusion.config import ModelConfig, DiffusionConfig
from bond_diffusion.data import graph_from_smiles, collate_graphs
from bond_diffusion.model import BondAwareDiffusionModel
from bond_diffusion.diffusion import DiscreteGraphDiffusion
from bond_diffusion.losses import ChemicalDiffusionLoss

SCHEMA = 'fullgraph-motif-conditioned-discrete-v1'

def signature(mol, members):
    """Exact induced chemical structure, shared across molecules and methods.

    Full-molecule annotations (charge/aromaticity) are retained. No SMARTS
    whitelist: rings, scaffolds and functional fragments are all represented.
    """
    members=list(members)
    if not members or len(set(members)) != len(members):
        raise ValueError('Empty or duplicate membership')
    if min(members)<0 or max(members)>=mol.GetNumAtoms():
        raise ValueError('Membership outside molecule')
    return Chem.MolFragmentToSmiles(mol, atomsToUse=members, canonical=True,
                                    isomericSmiles=True)

def vocabulary(data, method):
    counts=Counter()
    for row in data['molecules']:
        if row['split']!='train': continue
        mol=Chem.MolFromSmiles(row['smiles'])
        for idx in row['motifs'][method]:
            counts[signature(mol,data['motifs'][idx]['members'])]+=1
    return {'<no-motif>':0,'<unknown>':1,**{s:i+2 for i,s in enumerate(sorted(counts))}},dict(counts)

def records(data, rows, method, vocab, aligned=None):
    result=[]
    for row in rows:
        mol=Chem.MolFromSmiles(row['smiles'])
        groups=aligned[row['row']] if aligned is not None else [data['motifs'][i] for i in row['motifs'][method]]
        members=[g['members'] for g in groups]
        result.append({'graph':graph_from_smiles(row['smiles']), 'members':members,
                       'types':[vocab.get(signature(mol,m),1) for m in members], 'row':row['row']})
    return result

class FullGraphEncoder(nn.Module):
    def __init__(self, model_config, vocab):
        super().__init__()
        self.vocab=vocab
        self.legacy=BondAwareDiffusionModel(ModelConfig(**model_config))
        d=self.legacy.config.hidden_dim
        self.config={'hidden':d}
        self.type_embedding=nn.Embedding(len(vocab),d)
        self.pool_projection=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.LayerNorm(d))
        self._condition=ContextVar('fullgraph_condition',default=None)
        # Add condition once, before the first message passing layer. No copy of
        # the old backbone, noise process or decoding implementation is needed.
        self.legacy.node_embeddings[0].register_forward_hook(self._inject)

    def _inject(self,module,args,output):
        condition=self._condition.get()
        if condition is None: raise RuntimeError('Missing clean motif condition')
        return output+condition

    def batch(self, items, device):
        batch=collate_graphs([r['graph'] for r in items]).to(device)
        condition=self.type_embedding.weight.new_zeros((*batch.node_mask.shape,self.config['hidden']))
        for b,r in enumerate(items):
            n=r['graph'].node_features.size(0)
            sums=condition.new_zeros((n,self.config['hidden']));counts=condition.new_zeros((n,1))
            for members,t in zip(r['members'],r['types']):
                idx=torch.tensor(members,device=device)
                sums=sums.index_add(0,idx,self.type_embedding.weight[t].expand(len(idx),-1))
                counts.index_add_(0,idx,counts.new_ones((len(idx),1)))
            condition[b,:n]=torch.where(counts>0,sums/counts.clamp_min(1),self.type_embedding.weight[0])
        return batch,condition

    def forward(self,items,noisy=None):
        batch,condition=self.batch(items,next(self.parameters()).device)
        token=self._condition.set(condition)
        try:
            if noisy is not None:
                return self.legacy(noisy.node_features,noisy.bonds,batch.node_mask,noisy.timesteps)
            return self.legacy.encode_nodes(batch.node_features,batch.bonds,batch.node_mask)
        finally:
            self._condition.reset(token)

    def encode_records(self,items,batch_size=8,pair_budget=0):
        vectors=[]
        for start in range(0,len(items),batch_size):
            chunk=items[start:start+batch_size]
            n=max(r['graph'].node_features.size(0) for r in chunk)
            if pair_budget and len(chunk)*n*n>pair_budget:
                if len(chunk)==1: raise ValueError('Full molecule exceeds pair budget')
                vectors.append(self.encode_records(chunk,1,pair_budget));continue
            h=self(chunk)
            for b,r in enumerate(chunk):
                for members in r['members']:
                    vectors.append(self.pool_projection(h[b,members].mean(0)).unsqueeze(0))
        return torch.cat(vectors) if vectors else self.type_embedding.weight.new_empty((0,self.config['hidden']))

def load_encoder(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['schema']!=SCHEMA: raise ValueError('Fullgraph schema mismatch')
    model=FullGraphEncoder(ck['model_config'],ck['vocab'])
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device),ck
