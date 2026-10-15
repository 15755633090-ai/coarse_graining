import torch
from torch import nn
from torch.nn import functional as F
from .data import ATOM_DIMS, BOND_CLASSES, collate_graphs, to_device
from .diffusion import mlp
from .batching import encode_motifs

class ContextLayer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.chemical=mlp(2*d,d); self.membership=mlp(2*d,d)
        self.update=mlp(2*d,d); self.norm=nn.LayerNorm(d)

    def forward(self,h,chemical_index,chemical_attr,membership_index,membership_weight):
        agg=torch.zeros_like(h)
        if chemical_index.numel():
            src,dst=chemical_index
            agg.index_add_(0,dst,self.chemical(torch.cat([h[src],chemical_attr],-1)))
        if membership_index.numel():
            src,dst=membership_index
            msg=self.membership(torch.cat([h[src],h[dst]],-1))*membership_weight[:,None]
            agg.index_add_(0,dst,msg)
        return self.norm(h+self.update(torch.cat([h,agg],-1)))

class MotifContextEncoder(nn.Module):
    """Keep every atom/bond; add overlapping motifs and typed incidence edges."""
    def __init__(self,motif_dim=128,hidden=128,layers=3):
        super().__init__()
        self.hidden=hidden
        self.atom=nn.ModuleList(nn.Embedding(k,hidden) for k in ATOM_DIMS)
        self.bond=nn.Embedding(BOND_CLASSES,hidden)
        self.motif=nn.Linear(motif_dim,hidden)
        self.typed=nn.Linear(hidden+1,hidden)
        self.layers=nn.ModuleList(ContextLayer(hidden) for _ in range(layers))
        self.pool=mlp(2*hidden,hidden)

    def forward(self,molecules,motif_vectors,method,use_motifs=True):
        if not molecules or len(molecules)!=len(motif_vectors): raise ValueError('Invalid context batch')
        device=self.bond.weight.device
        hs=[]; cs=[]; ce=[]; inc=[]; iw=[]; scopes=[]; offset=0
        for mol,vectors in zip(molecules,motif_vectors):
            x=mol['graph']['x'].to(device); edges=mol['graph']['edges'].to(device)
            n=len(x); m=len(vectors) if use_motifs else 0
            if m!=len(mol['motifs'][method]) and use_motifs: raise ValueError('Motif ordering mismatch')
            a=sum(emb(x[:,i]) for i,emb in enumerate(self.atom))
            h=torch.cat([a,self.motif(vectors)] if m else [a],0)
            flag=torch.cat([torch.zeros(n,1,device=device),torch.ones(m,1,device=device)])
            hs.append(self.typed(torch.cat([h,flag],-1)))
            pairs=(edges!=0).nonzero().T
            cs.append(pairs+offset); ce.append(self.bond(edges[pairs[0],pairs[1]]))
            if m:
                groups=mol['member_atoms'][method]
                counts=torch.zeros(n,device=device)
                for group in groups: counts[group]+=1
                for j,group in enumerate(groups):
                    for i in group:
                        inc.extend([(offset+i,offset+n+j),(offset+n+j,offset+i)])
                        # Normalize overlap contributions in both directions.
                        iw.extend([float(1/counts[i]),float(1/counts[i])])
            scopes.append((offset,n,m)); offset+=n+m
        h=torch.cat(hs)
        chemical_index=torch.cat(cs,1); chemical_attr=torch.cat(ce)
        incidence=torch.tensor(inc,device=device,dtype=torch.long).reshape(-1,2).T
        weights=torch.tensor(iw,device=device)
        for layer in self.layers: h=layer(h,chemical_index,chemical_attr,incidence,weights)
        pooled=[]
        for start,n,m in scopes:
            a=h[start:start+n].mean(0)
            motif=h[start+n:start+n+m].mean(0) if m else torch.zeros_like(a)
            pooled.append(self.pool(torch.cat([a,motif])))
        return torch.stack(pooled)

class MotifPropertyModel(nn.Module):
    def __init__(self,denoiser,hidden=128,context_layers=3,freeze_pretrained=False,use_motifs=True,motif_batch_size=32):
        super().__init__()
        self.denoiser=denoiser; self.freeze_pretrained=freeze_pretrained; self.use_motifs=use_motifs
        if motif_batch_size<1: raise ValueError('motif_batch_size must be positive')
        self.motif_batch_size=motif_batch_size
        self.context=MotifContextEncoder(denoiser.config['hidden'],hidden,context_layers)
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        if freeze_pretrained or not use_motifs: self.denoiser.requires_grad_(False)

    def forward(self,molecules,all_motifs,method):
        device=next(self.parameters()).device
        ids=[i for mol in molecules for i in mol['motifs'][method]] if self.use_motifs else []
        if ids:
            # Bucketing avoids padding hundreds of tiny overlapping motifs to
            # one large scaffold. Restore exact original membership order.
            # This is direct deterministic t=0 encoding, not reverse sampling.
            with torch.set_grad_enabled(self.training and not self.freeze_pretrained):
                s=encode_motifs(self.denoiser,[all_motifs[i] for i in ids],device,self.motif_batch_size)
        else: s=torch.empty(0,self.denoiser.config['hidden'],device=device)
        vectors=[]; start=0
        for mol in molecules:
            k=len(mol['motifs'][method]) if self.use_motifs else 0
            vectors.append(s[start:start+k]);start+=k
        return self.head(self.context(molecules,vectors,method,self.use_motifs)).squeeze(-1)

def attach_members(data):
    for mol in data['molecules']:
        mol['member_atoms']={method:[data['motifs'][i]['members'] for i in ids]
                             for method,ids in mol['motifs'].items()}
    return data
