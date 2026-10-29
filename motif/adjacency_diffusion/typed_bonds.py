"""Typed chemical-edge denoising; keeps binary experiments explicitly separate."""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj
from rdkit import Chem

from motif.brics_atom_diffusion.data import graph_from_smiles as partition_graph
from .data import GraphDataset as BinaryDataset,atom_features,condition_from_batch
from .model import Config,Encoder,mlp,pair_nodes
from .diffusion import Schedule

SCHEMA='typed_chemical_edge_denoising_v1'
FEATURE_SCHEMA='atom133_edge17_type6_stereo7_conjugation2_ring2_v1'
BLOCKS={'type':(0,6),'stereo':(6,13),'conjugated':(13,15),'ring':(15,17)}
EDGE_DIM=17
TYPE_NAMES=('no_bond','single','double','triple','aromatic','other')


class GraphDataset(BinaryDataset):
    def __getitem__(self,index):
        if index not in self.cache:
            graph=partition_graph(self.smiles[index])
            mol=Chem.MolFromSmiles(self.smiles[index])
            graph.x=torch.tensor([atom_features(a) for a in mol.GetAtoms()],dtype=torch.float32)
            # Preserve all 16 original bond channels, including stereo and ring flags.
            del graph.boundary
            del graph.is_cross
            graph.sample_id=torch.tensor([index])
            if self.targets is not None:
                graph.y=self.targets[index].view(1,-1).float()
                graph.target_mask=self.masks[index].view(1,-1).bool()
            self.cache[index]=graph
        return self.cache[index]


def edge_target(batch,condition):
    """Preparation/loss or clean task forward only; never attached to Condition."""
    original=to_dense_adj(batch.edge_index,batch.batch,edge_attr=batch.edge_attr,
                          max_num_nodes=condition.x.shape[1])
    present=original[...,:5].sum(-1)>0
    result=torch.cat(((~present)[...,None].to(original.dtype),original),-1)
    return result*condition.pairs[...,None]


def corrupt(target,t,pairs,schedule,generator=None):
    a=schedule.coefficient(target[...,0],t)[...,None]
    raw=torch.randn(target.shape,device=target.device,dtype=target.dtype,generator=generator)
    upper=torch.triu(pairs,diagonal=1)[...,None]
    noise=raw*upper
    noise=noise+noise.transpose(1,2)
    return (a.sqrt()*target+(1-a).sqrt()*noise)*pairs[...,None]


class TypedEncoder(Encoder):
    def __init__(self,config):
        super().__init__(config)
        self.edge=mlp(EDGE_DIM,config.hidden_dim,config.hidden_dim)

    def observation_weight(self,noisy,t):
        return 1-noisy[...,:6].softmax(-1)[...,0]

    def forward(self,condition,noisy,t):
        if noisy.shape!=(*condition.pairs.shape,EDGE_DIM):
            raise ValueError('Expected one 17-channel observation per candidate pair')
        if (condition.same_fragment is not None)!=self.config.motif_prior:
            raise ValueError('Condition/prior mismatch')
        noisy=noisy*condition.pairs[...,None]
        fraction=t.to(noisy.dtype)/self.config.noise_steps
        embedding=self.time(torch.stack((fraction,torch.sin(fraction*torch.pi),torch.cos(fraction*torch.pi)),-1))[:,None]
        h=(self.h_input(condition.x)+embedding)*condition.nodes[...,None]
        s=(self.s_input(condition.x)+embedding)*condition.nodes[...,None] if self.s_input is not None else h
        edge=self.edge(noisy)
        # A probability from the RECEIVED observation, never the clean target support.
        weight=self.observation_weight(noisy,t)
        clean=t.eq(0)[:,None,None]
        weight=torch.where(clean,1-noisy[...,0],weight)*condition.pairs
        gates=[]
        for layer in self.layers:
            s,h,gate=layer(s,h,edge,weight,condition)
            gates.append(gate)
        z=self.fuse(torch.cat((s,h),-1) if self.config.dual_state else h)*condition.nodes[...,None]
        return {'atoms':z,'internal':s,'context':h,'gates':gates}


def masked_graph_mean(values,mask):
    count=mask.sum((1,2))
    total=(values*mask).sum((1,2))/count.clamp_min(1)
    return total[count>0].mean() if bool((count>0).any()) else values.sum()*0


def reconstruction_objective(logits,target,pairs):
    upper=torch.triu(pairs,diagonal=1)
    bonded=upper & target[...,0].eq(0)
    losses={}
    for name,(start,end) in BLOCKS.items():
        labels=target[...,start:end].argmax(-1)
        ce=F.cross_entropy(logits[...,start:end].movedim(-1,1),labels,reduction='none')
        losses[name]=masked_graph_mean(ce,upper if name=='type' else bonded)
    return sum(losses.values()),losses


def recovery_counts(logits,target,pairs):
    upper=torch.triu(pairs,diagonal=1)
    bonded=upper & target[...,0].eq(0)
    truth=target[...,:6].argmax(-1)
    predicted=logits[...,:6].argmax(-1)
    exists=predicted.ne(0)&upper
    counts={'pairs':int(upper.sum()),'bonds':int(bonded.sum()),
            'true_positive':int((exists&bonded).sum()),'predicted_bonds':int(exists.sum()),
            'type_all_correct':int(((predicted==truth)&upper).sum()),
            'type_bond_correct':int(((predicted==truth)&bonded).sum())}
    joint=(predicted==truth)
    for name,(start,end) in list(BLOCKS.items())[1:]:
        correct=logits[...,start:end].argmax(-1)==target[...,start:end].argmax(-1)
        counts[name+'_correct']=int((correct&bonded).sum())
        joint &= correct
    counts['joint_bond_correct']=int((joint&bonded).sum())
    for index,name in enumerate(TYPE_NAMES[1:],1):
        selected=bonded&truth.eq(index)
        counts[name+'_count']=int(selected.sum())
        counts[name+'_correct']=int(((predicted==truth)&selected).sum())
    stereo=target[...,6:13].argmax(-1)
    selected=bonded&stereo.ne(0)
    counts['nondefault_stereo_count']=int(selected.sum())
    counts['nondefault_stereo_correct']=int(((logits[...,6:13].argmax(-1)==stereo)&selected).sum())
    return counts


class TypedBondModel(nn.Module):
    def __init__(self,config=None,num_tasks=1):
        super().__init__()
        self.config=config or Config(time_mode='random')
        self.num_tasks=num_tasks
        c=self.config
        self.schedule=Schedule(c.noise_steps,c.beta_min,c.beta_max)
        self.encoder=TypedEncoder(c)
        self.bond_head=mlp(3*c.hidden_dim+EDGE_DIM,c.hidden_dim,EDGE_DIM)
        self.property_head=mlp(c.hidden_dim,c.hidden_dim,num_tasks)

    def set_stage(self,stage):
        if stage not in ('pretrain','finetune'):
            raise ValueError(stage)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.bond_head.parameters():
            parameter.requires_grad_(stage=='pretrain')
        for parameter in self.property_head.parameters():
            parameter.requires_grad_(stage=='finetune')

    def predict_bonds(self,condition,noisy,t):
        states=self.encoder(condition,noisy,t)
        a,b=pair_nodes(states['atoms'])
        logits=self.bond_head(torch.cat((a+b,(a-b).abs(),a*b,noisy),-1))
        return .5*(logits+logits.transpose(1,2))*condition.pairs[...,None]

    def loss(self,batch,generator=None):
        condition=condition_from_batch(batch,self.config.motif_prior)
        target=edge_target(batch,condition)
        t=torch.randint(1,self.config.noise_steps+1,(len(target),),device=target.device,generator=generator)
        noisy=corrupt(target,t,condition.pairs,self.schedule,generator)
        logits=self.predict_bonds(condition,noisy,t)
        loss,blocks=reconstruction_objective(logits,target,condition.pairs)
        return loss,blocks

    def forward(self,batch):
        condition=condition_from_batch(batch,self.config.motif_prior)
        clean=edge_target(batch,condition)
        t=torch.zeros(len(clean),dtype=torch.long,device=clean.device)
        states=self.encoder(condition,clean,t)
        graph=states['atoms'].sum(1)/condition.nodes.sum(1).clamp_min(1)[:,None]
        return self.property_head(graph)

    def checkpoint(self,stage,**extra):
        return {'schema':SCHEMA,'feature_schema':FEATURE_SCHEMA,'stage':stage,
                'config':asdict(self.config),'num_tasks':self.num_tasks,
                'model':self.state_dict(),'encoder':self.encoder.state_dict(),**extra}


def load_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved.get('feature_schema')!=FEATURE_SCHEMA:
        raise ValueError('Not a typed chemical-edge checkpoint; binary adjacency checkpoints are incompatible')
    return saved
