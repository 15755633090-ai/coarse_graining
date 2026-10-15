"""Adapters for existing paper model objects; no upstream files are modified.

Ordering is explicit: bindings MUST be built from the caller's token/node map.
No attempt is made to infer correspondence from matching feature vectors.
"""
import copy
from contextvars import ContextVar
import torch
from torch import nn
from torch.nn import functional as F
from .data import graph_features,collate_graphs,to_device

def hmg_motif_inputs(mols,fragment_maps,device='cpu'):
    """Use ORIGINAL RDKit atom order/maps, before upstream degree sorting.

    Matches create_fragment_features_with_dims: whole_mol at 0, then frag_N
    sorted numerically. Returns (None, []) when the batch has no real motifs.
    """
    if len(mols)!=len(fragment_maps): raise ValueError('Batch lengths differ')
    graphs=[];positions=[]
    for batch,(mol,mapping) in enumerate(zip(mols,fragment_maps)):
        if 'whole_mol' not in mapping: raise ValueError('Missing whole_mol token')
        keys=sorted((k for k in mapping if k!='whole_mol'),key=lambda k:int(k.split('_')[1]))
        for token,key in enumerate(keys,1):
            members,bonds=mapping[key]
            graphs.append(graph_features(mol,members));positions.append([batch,token])
    return (to_device(collate_graphs(graphs),device) if graphs else None),positions

def himnet_motif_inputs(mols,cliques,batch_graph,device='cpu'):
    """Bind BRICS clique order to original BatchMolGraph motif rows."""
    if not (len(mols)==len(cliques)==len(batch_graph.a_scope)): raise ValueError('Batch lengths differ')
    graphs=[];positions=[]
    for k,(mol,groups) in enumerate(zip(mols,cliques)):
        start,size=batch_graph.a_scope[k];n=int(batch_graph.mol_atom_num[k])
        if n!=mol.GetNumAtoms() or size!=n+len(groups)+1: raise ValueError('Original node order mismatch')
        for j,members in enumerate(groups):
            graphs.append(graph_features(mol,members));positions.append(start+n+j)
    return (to_device(collate_graphs(graphs),device) if graphs else None),positions

def validate_positions(positions,count,shape,device):
    positions=torch.as_tensor(positions,dtype=torch.long,device=device)
    if positions.shape!=(count,len(shape)): raise ValueError('Binding shape mismatch')
    if count:
        if bool((positions<0).any()): raise ValueError('Negative position')
        for k,n in enumerate(shape):
            if bool((positions[:,k]>=n).any()): raise ValueError('Position out of range')
        if len(torch.unique(positions,dim=0))!=count: raise ValueError('Duplicate positions')
    return positions

class _BoundProjection(nn.Module):
    def __init__(self,base,motif_dim):
        super().__init__();self.base=base
        self.motif_projection=nn.Linear(motif_dim,base.out_features)
        self.override=ContextVar('motif_projection_override',default=None)

    def forward(self,x):
        output=self.base(x)
        bound=self.override.get()
        if bound is None: return output
        vectors,positions=bound
        positions=validate_positions(positions,len(vectors),output.shape[:2],output.device)
        output=output.clone()
        output[positions[:,0],positions[:,1]]=self.motif_projection(vectors)
        return output

class HMGMRLAdapter(nn.Module):
    """Replace selected motif initializations, retaining CLS/whole-mol/padding.

    Pass an independently instantiated original backbone, not a shared baseline.
    backbone_inputs are exactly the original model's keyword arguments.
    positions[k] = [batch_index, original_token_index] for dense motif k.
    """
    def __init__(self,backbone,denoiser,freeze_pretrained=False):
        super().__init__();self.backbone=backbone;self.denoiser=denoiser
        self.freeze_pretrained=freeze_pretrained
        base=backbone.frag_input_projection
        if isinstance(base,_BoundProjection): raise ValueError('Backbone already adapted')
        backbone.frag_input_projection=_BoundProjection(base,denoiser.config['hidden'])
        if freeze_pretrained: denoiser.requires_grad_(False)

    def forward(self,motif_batch,positions,**backbone_inputs):
        if motif_batch is None:
            if len(positions): raise ValueError('Positions without motifs')
            return self.backbone(**backbone_inputs)
        projection=self.backbone.frag_input_projection
        with torch.set_grad_enabled(self.training and not self.freeze_pretrained):
            s=self.denoiser.encode(motif_batch)['motif']
        token=projection.override.set((s,positions))
        try: return self.backbone(**backbone_inputs)
        finally: projection.override.reset(token)

class HimNetAdapter(nn.Module):
    """Use pretrained features for actual motif node rows in a BatchMolGraph.

    Supply the original BatchMolGraph, including fp_x and all original indices.
    positions[k] is the GLOBAL f_atoms row of motif k, including padding offset.
    The caller must not bind atom, global, or padding nodes.
    """
    def __init__(self,backbone,denoiser,freeze_pretrained=False):
        super().__init__();self.backbone=backbone;self.denoiser=denoiser
        self.freeze_pretrained=freeze_pretrained
        self.projection=nn.Linear(denoiser.config['hidden'],backbone.atom_fdim)
        if freeze_pretrained: denoiser.requires_grad_(False)

    def forward(self,mol_graph,motif_batch,positions,training=False):
        if motif_batch is None:
            if len(positions): raise ValueError('Positions without motifs')
            vectors=self.projection.weight.new_empty((0,self.denoiser.config['hidden']))
        else:
            with torch.set_grad_enabled(self.training and not self.freeze_pretrained):
                vectors=self.denoiser.encode(motif_batch)['motif']
        graph=copy.copy(mol_graph)
        x=mol_graph.f_atoms.to(vectors.device).clone()
        indices=validate_positions(torch.as_tensor(positions).reshape(-1,1),len(vectors),(len(x),),x.device)[:,0]
        allowed=set()
        for k,(start,size) in enumerate(mol_graph.a_scope):
            n=int(mol_graph.mol_atom_num[k])
            allowed.update(range(start+n,start+size-1))
        if any(int(i) not in allowed for i in indices): raise ValueError('Binding is not a motif node')
        x[indices]=self.projection(vectors);graph.f_atoms=x
        ligand=self.backbone.encoder(graph)
        if training:
            fp,fp_proj,views=self.backbone.fp_encoder(graph.fp_x.to(vectors.device).float(),training=True)
            self.backbone.contrastive_data={'graph_proj':F.normalize(self.backbone.graph_proj_head(ligand),dim=1),
                                             'fp_proj':fp_proj,'fp_view_projs':views}
        else: fp=self.backbone.fp_encoder(graph.fp_x.to(vectors.device).float())
        fused=self.backbone.feature_fusion(torch.stack([ligand,fp],dim=0))
        return self.backbone.mlp(fused)

    def compute_contrastive_loss(self,*args,**kwargs):
        return self.backbone.compute_contrastive_loss(*args,**kwargs)
