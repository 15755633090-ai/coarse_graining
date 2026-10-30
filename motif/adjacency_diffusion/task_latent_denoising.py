"""Task-aligned, motif-grouped latent denoising on the unchanged strong backbone.

No atom/bond/coordinate corruption. The clean path is exactly MaskedBondModel.
After the first intra-motif update, corrupt its latent state and all parallel
target-content paths before context propagation. Reuse the existing time MLP.
"""
import math
import torch
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from .masked_bonds import MaskedBondModel
from .model import Config,pair_nodes,reduce_messages
from .typed_bonds import edge_target,FEATURE_SCHEMA
from .data import condition_from_batch
from .path_gates import controls,scaled

SCHEMA='motif_group_task_aligned_latent_denoising_v1'

def select_fragments(batch,condition,generator,ratio=.3):
    ids,_=to_dense_batch(batch.fragment_id,batch.batch,max_num_nodes=condition.x.shape[1],fill_value=-1)
    selected=torch.zeros_like(condition.nodes);eligible=torch.zeros(len(ids),device=ids.device,dtype=torch.bool)
    for i in range(len(ids)):
        fragments=ids[i][condition.nodes[i]].unique()
        if len(fragments)<2:continue
        eligible[i]=True;number=min(len(fragments)-1,max(1,math.ceil(ratio*len(fragments))))
        chosen=fragments[torch.randperm(len(fragments),device=ids.device,generator=generator)[:number]]
        selected[i]=torch.isin(ids[i],chosen)&condition.nodes[i]
    return selected,eligible

def normalized_feature_loss(student,target,selected):
    loss=(F.layer_norm(student,(student.shape[-1],))-F.layer_norm(target.detach(),(target.shape[-1],))).square().mean(-1)
    count=selected.sum(1);eligible=count>0
    per_graph=(loss*selected).sum(1)/count.clamp_min(1)
    return per_graph[eligible].mean() if bool(eligible.any()) else student.sum()*0

class TaskLatentModel(MaskedBondModel):
    def __init__(self,config,priors,mask_ratio=.3,num_tasks=1):
        if not config.dual_state or not config.motif_prior:raise ValueError('Requires the BRICS dual-state backbone')
        super().__init__(config,priors,mask_ratio,num_tasks)

    def clean_states(self,batch):
        c=condition_from_batch(batch,True);x=edge_target(batch,c)
        observed=torch.cat((x,torch.zeros_like(x[...,:1])),-1)
        states=self.encoder(c,observed,torch.zeros(len(x),device=x.device,dtype=torch.long))
        return states,c,observed

    def predict_states(self,states,c):
        return self.property_head(states['atoms'].sum(1)/c.nodes.sum(1).clamp_min(1)[:,None])

    def forward(self,batch):return super().forward(batch)

    def latent_states(self,batch,selected,sigma,generator=None,epsilon=None):
        # sigma=0 must match the original inference route bit for bit.
        if not bool(selected.any()) or not bool((sigma!=0).any()):return self.clean_states(batch)[:2]
        c=condition_from_batch(batch,True);e=self.encoder;x=edge_target(batch,c)
        observed=torch.cat((x,torch.zeros_like(x[...,:1])),-1)*c.pairs[...,None]
        zero=observed.new_zeros(len(x));base=e.time(torch.stack((zero,zero,torch.ones_like(zero)),-1))[:,None]
        marker=e.time(torch.stack((sigma,torch.sin(sigma*torch.pi),torch.cos(sigma*torch.pi)),-1))[:,None]-base
        h=(e.h_input(c.x)+base)*c.nodes[...,None]
        s=(e.s_input(c.x)+base)*c.nodes[...,None]
        edge=e.edge(observed);weight=(1-observed[...,0])*c.pairs
        first=e.layers[0];paths=controls(first)
        s=scaled(s,paths.internal_branch);a,b=pair_nodes(s)
        inside=c.same_fragment
        internal=reduce_messages(first.s_message(torch.cat((a,b,edge),-1)),inside,weight,first.message_normalization)
        next_s=scaled(first.s_norm(s+first.drop(first.s_update(torch.cat((s,internal),-1)))),paths.internal_branch)
        sources=[next_s,s,h]
        if epsilon is None:epsilon=[torch.randn(q.shape,device=q.device,dtype=q.dtype,generator=generator) for q in sources]
        if len(epsilon)!=3:raise ValueError('Three latent noise fields required')
        mask=selected[...,None];strength=sigma[:,None,None]
        def perturb(q,noise):
            if noise.shape!=q.shape:raise ValueError('Latent noise shape mismatch')
            scale=q.detach().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
            return torch.where(mask,q+strength*scale*noise+marker,q)*c.nodes[...,None]
        next_s,gate_s,h=[perturb(q,noise) for q,noise in zip(sources,epsilon)]
        # Recompute every parallel path: no clean s0/h0 survives in boundary gates.
        a,b=pair_nodes(gate_s);hi,hj=pair_nodes(h)
        gate=weight.new_ones(*weight.shape,1)
        if first.gate is not None and paths.boundary_mode=='learned':
            learned=torch.sigmoid(first.gate(torch.cat((a,b,hi,hj,edge),-1)))
            gate=torch.where((c.pairs&~inside)[...,None],learned,gate)
        message=first.h_message(torch.cat((hi,hj,edge),-1))*gate
        if paths.context_within!=1 or paths.context_cross!=1:
            route=torch.where(inside,weight.new_tensor(paths.context_within),weight.new_tensor(paths.context_cross))
            message=message*route[...,None]
        incoming=reduce_messages(message,c.pairs,weight,first.message_normalization)
        h=first.h_norm(h+first.drop(first.h_update(torch.cat((h,next_s,incoming),-1))))*c.nodes[...,None]
        s=next_s;gates=[gate.squeeze(-1)*c.pairs]
        for layer in e.layers[1:]:
            s,h,g=layer(s,h,edge,weight,c);gates.append(g)
        paths=controls(e)
        z=e.fuse(torch.cat((scaled(s,paths.internal_readout),scaled(h,paths.context_readout)),-1))*c.nodes[...,None]
        return {'atoms':z,'internal':s,'context':h,'gates':gates},c

    def checkpoint(self,stage,**extra):
        saved=super().checkpoint(stage,**extra);saved.update(schema=SCHEMA,
            training_protocol='joint property supervision of clean/noisy enhanced states; optional selected-node contextual-feature denoising')
        return saved

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved.get('feature_schema')!=FEATURE_SCHEMA:raise ValueError('Wrong task-latent checkpoint')
    model=TaskLatentModel(Config(**saved['config']),saved['priors'],saved['mask_ratio'],saved['num_tasks'])
    model.training_max_t=saved['training_max_t'];model.load_state_dict(saved['model'],strict=True)
    return model,saved
