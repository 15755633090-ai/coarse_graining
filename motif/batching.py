"""Shared size buckets, bounded clean encoding, item-stable validation noise."""
import hashlib
import torch
from torch.utils.data import Sampler
from torch.utils.checkpoint import checkpoint
from .data import collate_graphs,to_device,ATOM_DIMS,BOND_CLASSES

class SizeBucketSampler(Sampler):
    def __init__(self,sizes,batch_size=32,shuffle=False,seed=0,pair_budget=0):
        if batch_size<1 or pair_budget<0: raise ValueError('Invalid batch limits')
        self.sizes=list(sizes);self.batch_size=batch_size;self.shuffle=shuffle
        self.seed=seed;self.epoch=0;self.pair_budget=pair_budget
        self.batches=[];batch=[]
        for i in sorted(range(len(sizes)),key=lambda i:sizes[i]):
            if sizes[i]<1: raise ValueError('Empty motif')
            if pair_budget and sizes[i]**2>pair_budget:
                raise ValueError(f'Motif {i} exceeds pair budget even alone; increase budget explicitly')
            if batch and (len(batch)>=batch_size or (pair_budget and (len(batch)+1)*sizes[i]**2>pair_budget)):
                self.batches.append(batch);batch=[]
            batch.append(i)
        if batch:self.batches.append(batch)

    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.batches)
    def __iter__(self):
        rng=torch.Generator().manual_seed(self.seed+self.epoch)
        order=torch.randperm(len(self.batches),generator=rng).tolist() if self.shuffle else range(len(self.batches))
        for k in order:
            batch=self.batches[k]
            yield [batch[i] for i in torch.randperm(len(batch),generator=rng).tolist()] if self.shuffle else list(batch)

def as_graphs(batch):
    if not isinstance(batch,dict):return list(batch)
    return [{'x':x[m], 'edges':e[m][:,m]} for x,e,m in zip(batch['x'],batch['edges'],batch['mask'])]

def encode_motifs(model,graphs,device,batch_size=32,pair_budget=0,checkpoint_grad=True):
    graphs=as_graphs(graphs)
    if not graphs:return next(model.parameters()).new_empty((0,model.config['hidden']))
    sampler=SizeBucketSampler([len(g['x']) for g in graphs],batch_size,pair_budget=pair_budget)
    vectors=[];order=[]
    for indices in sampler:
        b=to_device(collate_graphs([graphs[i] for i in indices]),device)
        def encode(x,e,mask):return model.encode({'x':x,'edges':e,'mask':mask})['motif']
        if checkpoint_grad and torch.is_grad_enabled() and any(p.requires_grad for p in model.parameters()):
            s=checkpoint(encode,b['x'],b['edges'],b['mask'],use_reentrant=False)
        else:s=encode(b['x'],b['edges'],b['mask'])
        vectors.append(s);order.extend(indices)
    inverse=torch.argsort(torch.tensor(order,device=device))
    return torch.cat(vectors)[inverse]

def validation_noise(schedule,graphs,seed,device):
    """Per-item RNG means changing batch boundaries does not change corruptions."""
    b=to_device(collate_graphs(graphs),device)
    xt=torch.zeros(*b['x'].shape[:2],sum(ATOM_DIMS),device=device)
    et=torch.zeros(*b['edges'].shape,BOND_CLASSES,device=device);ts=[]
    for k,g in enumerate(graphs):
        key=f"{seed}:{g['row']}:{g['method']}:{g['index']}".encode()
        item_seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1)
        rng=torch.Generator(device=device).manual_seed(item_seed)
        one=to_device(collate_graphs([g]),device);t=schedule.sample_t(1,device,rng)
        x,e=schedule(one,t,rng);n=len(g['x']);xt[k,:n]=x[0];et[k,:n,:n]=e[0];ts.append(t[0])
    return b,xt,et,torch.stack(ts)
