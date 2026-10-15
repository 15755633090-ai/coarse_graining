import math
import torch
from torch import nn
from torch.nn import functional as F
from .data import ATOM_DIMS, BOND_CLASSES, pair_mask

class DiffusionSchedule(nn.Module):
    def __init__(self, steps=100, clean_probability=.1):
        super().__init__()
        if steps<2 or not 0<=clean_probability<1: raise ValueError('Invalid schedule')
        self.steps, self.clean_probability = steps, clean_probability
        # Cosine signal schedule, including a genuinely clean t=0.
        t = torch.linspace(0,1,steps+1)
        alpha = torch.cos((t+.008)/1.008*math.pi/2).square()
        self.register_buffer('alpha_bar', (alpha/alpha[0]).clamp(1e-6,1))

    def sample_t(self, count, device, generator=None):
        t = torch.randint(1,self.steps+1,(count,),device=device,generator=generator)
        return t.masked_fill(torch.rand(count,device=device,generator=generator)<self.clean_probability,0)

    def forward(self, batch, t, generator=None):
        x,e,mask = batch['x'],batch['edges'],batch['mask']
        x0 = torch.cat([F.one_hot(x[...,i],d) for i,d in enumerate(ATOM_DIMS)],-1).float()
        e0 = F.one_hot(e,BOND_CLASSES).float()
        ex = torch.randn(x0.shape,device=x.device,generator=generator)
        # One noise draw per unordered pair; copying preserves variance (no averaging).
        z = torch.randn(e0.shape,device=x.device,generator=generator)
        upper = torch.triu(torch.ones(e.shape[-2:],device=x.device,dtype=torch.bool),diagonal=1)
        z = z*upper[None,:,:,None]; ee = z+z.transpose(1,2)
        a = self.alpha_bar[t]
        xt = a[:,None,None].sqrt()*x0 + (1-a[:,None,None]).sqrt()*ex
        et = a[:,None,None,None].sqrt()*e0 + (1-a[:,None,None,None]).sqrt()*ee
        return xt*mask[...,None], et*pair_mask(mask)[...,None]

def mlp(a,b): return nn.Sequential(nn.Linear(a,b),nn.SiLU(),nn.Linear(b,b))

class JointLayer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.edge=mlp(3*d,d); self.message=mlp(2*d,d); self.node=mlp(2*d,d)
        self.en=nn.LayerNorm(d); self.xn=nn.LayerNorm(d)

    def forward(self,h,e,mask):
        pm=pair_mask(mask)
        hi,hj=h[:,:,None,:],h[:,None,:,:]
        # Symmetric endpoint functions enforce undirected edge equivariance.
        e=self.en(e+self.edge(torch.cat([hi+hj,(hi-hj).square(),e],-1)))*pm[...,None]
        message=self.message(torch.cat([hj.expand_as(e),e],-1))*pm[...,None]
        agg=message.sum(2)/pm.sum(2).clamp_min(1)[...,None]
        h=self.xn(h+self.node(torch.cat([h,agg],-1)))*mask[...,None]
        return h,e

class MotifDenoiser(nn.Module):
    def __init__(self, hidden=128, layers=3, steps=100):
        super().__init__()
        if hidden<4: raise ValueError('hidden must be >=4')
        self.config={'hidden':hidden,'layers':layers,'steps':steps}
        self.x_in=nn.Linear(sum(ATOM_DIMS),hidden); self.e_in=nn.Linear(BOND_CLASSES,hidden)
        self.time=mlp(hidden,hidden)
        self.layers=nn.ModuleList(JointLayer(hidden) for _ in range(layers))
        self.atom_heads=nn.ModuleList(nn.Linear(hidden,d) for d in ATOM_DIMS)
        self.edge_head=nn.Linear(hidden,BOND_CLASSES)
        self.readout=mlp(2*hidden+2,hidden)
        self.node_decode=mlp(2*hidden,hidden); self.edge_decode=mlp(2*hidden,hidden)

    def time_embedding(self,t):
        d=self.config['hidden']; half=(d+1)//2
        freq=torch.exp(torch.arange(half,device=t.device)*(-math.log(10000)/max(half-1,1)))
        phase=t.float()[:,None]*freq[None]
        return torch.cat([phase.sin(),phase.cos()],-1)[:,:d]

    def forward(self,xt,et,t,mask):
        temb=self.time(self.time_embedding(t))
        h=(self.x_in(xt)+temb[:,None])*mask[...,None]
        e=(self.e_in(et)+temb[:,None,None])*pair_mask(mask)[...,None]
        for layer in self.layers: h,e=layer(h,e,mask)
        pm=pair_mask(mask); n=mask.sum(1).clamp_min(1); pairs=pm.sum((1,2))
        xn=h.sum(1)/n[:,None]
        en=e.sum((1,2))/pairs.clamp_min(1)[:,None]
        s=self.readout(torch.cat([xn,en,n.float().log1p()[:,None],pairs.float().log1p()[:,None]],-1))
        # Make the graph readout part of the supervised denoising path, rather
        # than exporting an untrained projection at downstream time.
        hd=self.node_decode(torch.cat([h,s[:,None].expand_as(h)],-1))
        ed=self.edge_decode(torch.cat([e,s[:,None,None].expand_as(e)],-1))
        return {'atom_logits':[head(hd) for head in self.atom_heads],
                'edge_logits':self.edge_head(ed),'nodes':h,'pairs':e,'motif':s}

    def encode(self,batch):
        x=torch.cat([F.one_hot(batch['x'][...,i],d) for i,d in enumerate(ATOM_DIMS)],-1).float()
        e=F.one_hot(batch['edges'],BOND_CLASSES).float()
        t=torch.zeros(x.size(0),device=x.device,dtype=torch.long)
        return self(x*batch['mask'][...,None],e*pair_mask(batch['mask'])[...,None],t,batch['mask'])

def denoising_loss(output,batch,atom_weight=1.,edge_weight=1.):
    mask=batch['mask']; pm=torch.triu(pair_mask(mask),diagonal=1)
    atom=torch.stack([F.cross_entropy(logit[mask],batch['x'][...,i][mask])
                      for i,logit in enumerate(output['atom_logits'])]).mean()
    logits=output['edge_logits'][pm]; target=batch['edges'][pm]
    # Balance existing bonds and no-bonds, without sampling away observed pairs.
    terms=[]
    for selected in (target!=0,target==0):
        if bool(selected.any()): terms.append(F.cross_entropy(logits[selected],target[selected]))
    edge=torch.stack(terms).mean() if terms else output['edge_logits'].sum()*0
    total=atom_weight*atom+edge_weight*edge
    return total, {'atom':float(atom.detach()),'edge':float(edge.detach())}
