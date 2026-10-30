"""Covalent directed edge states + optional BRICS internal path + attentive readout.

This is a new supervised combination, not a D-MPNN/AttentiveFP reproduction.
No diffusion, pseudo-labels, pretrained weights, motif nodes, or ID embeddings.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch_geometric.utils import softmax

SCHEMA='covalent_directed_brics_attention_supervised_v1'

@dataclass
class Config:
    hidden_dim:int=64
    layers:int=3
    dropout:float=.1
    readout_steps:int=2
    boundary:bool=True
    def __post_init__(self):
        if self.hidden_dim<4 or self.layers<1 or self.readout_steps<1 or not 0<=self.dropout<1:raise ValueError('Invalid configuration')

def mlp(a,d,b):return nn.Sequential(nn.Linear(a,d),nn.ReLU(),nn.Linear(d,b))

def reverse_edges(edge_index,n):
    """Order independent reverse indices for a simple, bidirectional graph."""
    if edge_index.shape[1]==0:return edge_index.new_empty(0)
    u,v=edge_index;keys=u*n+v;ordered,permutation=keys.sort()
    if len(ordered)>1 and bool((ordered[1:]==ordered[:-1]).any()):raise ValueError('Duplicate directed edges')
    position=torch.searchsorted(ordered,v*n+u)
    if bool((position>=len(ordered)).any()) or not torch.equal(ordered[position],v*n+u):raise ValueError('Missing reverse edge')
    return permutation[position]

def incoming(values,destination,n):
    out=values.new_zeros(n,values.shape[-1]);out.index_add_(0,destination,values);return out

class BoundaryPredictor(nn.Module):
    def __init__(self,config=None,num_tasks=1):
        super().__init__();self.config=config or Config();self.num_tasks=num_tasks
        c=self.config;d=c.hidden_dim
        self.edge_input=nn.Linear(133+16,d)
        self.edge_updates=nn.ModuleList([nn.Linear(d,d) for _ in range(c.layers)])
        if c.boundary:
            self.internal_input=nn.Linear(133,d)
            self.internal_messages=nn.ModuleList([mlp(2*d+16,d,d) for _ in range(c.layers)])
            self.internal_updates=nn.ModuleList([nn.GRUCell(d,d) for _ in range(c.layers)])
            self.internal_to_edge=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(c.layers)])
            self.boundary_gates=nn.ModuleList([mlp(3*d+16,d,1) for _ in range(c.layers)])
        self.atom_output=mlp(133+d+(d if c.boundary else 0),d,d)
        self.attention_node=nn.Linear(d,d)
        self.attention_query=nn.Linear(d,d,bias=False)
        self.attention_score=nn.Linear(d,1,bias=False)
        self.readout_update=nn.GRUCell(d,d)
        self.property_head=mlp(2*d,d,num_tasks)
        self.drop=nn.Dropout(c.dropout)

    def set_stage(self,stage):
        if stage!='finetune':raise ValueError('This predictor supports supervised training only')
        for p in self.parameters():p.requires_grad_(True)

    def encode(self,batch):
        x=batch.x;edge=batch.edge_attr
        if x.shape[1]!=133 or edge.shape[1]!=16:raise ValueError('Full atom133/bond16 features required')
        u,v=batch.edge_index;n=len(x);reverse=reverse_edges(batch.edge_index,n)
        if bool((batch.batch[u]!=batch.batch[v]).any()):raise ValueError('Cross-molecule chemical edge')
        inside=batch.fragment_id[u]==batch.fragment_id[v]
        h0=torch.relu(self.edge_input(torch.cat((x[u],edge),-1)));h=h0
        s=torch.relu(self.internal_input(x)) if self.config.boundary else None
        gates=[]
        for l in range(self.config.layers):
            if s is not None:
                m=self.internal_messages[l](torch.cat((s[u],s[v],edge),-1))*inside[:,None]
                s=self.internal_updates[l](incoming(m,v,n),s)
                learned=torch.sigmoid(self.boundary_gates[l](torch.cat((s[u],s[v],h,edge),-1)))
                gate=torch.where(inside[:,None],torch.ones_like(learned),learned)
            else:gate=h.new_ones(len(h),1)
            weighted=h*gate
            # Messages arriving at source, excluding the immediate reverse edge.
            cavity=incoming(weighted,v,n)[u]-weighted[reverse]
            update=h0+self.edge_updates[l](cavity)
            if s is not None:update=update+self.internal_to_edge[l](s[u])
            h=self.drop(torch.relu(update));gates.append(gate)
        parts=[x,incoming(h*gates[-1],v,n)]
        if s is not None:parts.append(s)
        z=self.drop(self.atom_output(torch.cat(parts,-1)))
        return {'atoms':z,'internal':s,'gates':gates}

    def graph_representation(self,batch,states=None):
        z=(states if states is not None else self.encode(batch))['atoms']
        groups=batch.batch;count=torch.bincount(groups,minlength=batch.num_graphs).clamp_min(1).to(z.dtype)
        mean=incoming(z,groups,batch.num_graphs)/count[:,None];q=mean
        projected=self.attention_node(z)
        for _ in range(self.config.readout_steps):
            score=self.attention_score(torch.tanh(projected+self.attention_query(q)[groups]))
            alpha=softmax(score,groups,num_nodes=batch.num_graphs)
            q=self.readout_update(incoming(alpha*z,groups,batch.num_graphs),q)
        return torch.cat((mean,q),-1)

    def forward(self,batch):return self.property_head(self.graph_representation(batch))

    def checkpoint(self,stage,**extra):
        if stage!='finetune':raise ValueError(stage)
        return {'schema':SCHEMA,'stage':stage,'config':asdict(self.config),'num_tasks':self.num_tasks,'model':self.state_dict(),**extra}

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA:raise ValueError('Wrong supervised predictor schema')
    model=BoundaryPredictor(Config(**saved['config']),saved['num_tasks']);model.load_state_dict(saved['model'],strict=True)
    return model,saved
