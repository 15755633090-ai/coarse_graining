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
    if hasattr(model,'encode_records'):
        return model.encode_records(graphs,batch_size,pair_budget)
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

class CachedValidation:
    """Cache the SAME item-seeded GPU corruptions on bounded pinned CPU memory.

    No CPU re-sampling and no changed noise distribution. Over-budget batches
    transparently use the existing generator path. Never reads test samples.
    """
    def __init__(self,loader,schedule,seed,device,max_mb=512,max_batches=0):
        self.schedule=schedule;self.seed=seed;self.device=device;self.items=[]
        self.bytes=0;self.cached=0;budget=int(max_mb*1024*1024)
        if max_mb<0:raise ValueError('Cache size must be nonnegative')
        for i,graphs in enumerate(loader):
            if max_batches and i>=max_batches:break
            n=max(len(g['x']) for g in graphs);size=len(graphs)
            estimate=size*(n*(len(ATOM_DIMS)*8+1+sum(ATOM_DIMS)*4)+n*n*(8+BOND_CLASSES*4)+8)
            entry=None
            if self.bytes+estimate<=budget:
                b,x,e,t=validation_noise(schedule,graphs,seed,device)
                entry={**b,'xt':x,'et':e,'t':t}
                entry={k:v.cpu() for k,v in entry.items()}
                actual=sum(v.numel()*v.element_size() for v in entry.values())
                if self.bytes+actual<=budget:
                    if str(device).startswith('cuda'):entry={k:v.pin_memory() for k,v in entry.items()}
                    self.bytes+=actual;self.cached+=1
                else:entry=None
            self.items.append((graphs,entry))

    def __iter__(self):
        for graphs,entry in self.items:
            if entry is None:b,x,e,t=validation_noise(self.schedule,graphs,self.seed,self.device)
            else:
                d={k:v.to(self.device,non_blocking=True) for k,v in entry.items()}
                b={k:d[k] for k in ('x','edges','mask')};x,e,t=d['xt'],d['et'],d['t']
            yield graphs,b,x,e,t

    def metadata(self):return {'batches':len(self.items),'cached_batches':self.cached,'cpu_bytes':self.bytes,'noise_unchanged':True}
