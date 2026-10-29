"""Train-only categorical priors, hierarchical bond supervision and noise evidence."""
import torch
from torch.nn import functional as F
from .typed_bonds import (TypedBondModel,TypedEncoder,BLOCKS,FEATURE_SCHEMA,edge_target,
                          masked_graph_mean,reconstruction_objective,corrupt)
from .data import condition_from_batch

SCHEMA='typed_hierarchical_evidence_v2'


@torch.no_grad()
def training_priors(dataset):
    from torch_geometric.data import Batch
    counts={name:torch.ones(end-start,dtype=torch.float64) for name,(start,end) in BLOCKS.items()}
    for graph in dataset:
        batch=Batch.from_data_list([graph])
        condition=condition_from_batch(batch)
        target=edge_target(batch,condition)
        upper=torch.triu(condition.pairs,diagonal=1)
        bonded=upper&target[...,0].eq(0)
        for name,(start,end) in BLOCKS.items():
            counts[name]+=target[...,start:end][upper if name=='type' else bonded].sum(0).double()
    return {name:(count/count.sum()).float() for name,count in counts.items()}


def evidence_scale(schedule,noisy,t):
    alpha=schedule.alpha_bar.to(noisy.device)[t].to(noisy.dtype)[:,None,None,None]
    return alpha.sqrt()/(1-alpha).clamp_min(1e-8)


class CalibratedEncoder(TypedEncoder):
    def __init__(self,config,prior):
        super().__init__(config)
        self.register_buffer('type_prior',prior)
        from .diffusion import Schedule
        self.evidence_schedule=Schedule(config.noise_steps,config.beta_min,config.beta_max)

    def observation_weight(self,noisy,t):
        logit=noisy[...,:6]*evidence_scale(self.evidence_schedule,noisy,t)+self.type_prior.log()
        return 1-logit.softmax(-1)[...,0]


def factor_logits(raw):
    presence=raw[...,:1]
    return torch.cat((F.logsigmoid(-presence),F.logsigmoid(presence)+raw[...,1:6].log_softmax(-1),raw[...,6:]),-1)


def factorized_objective(logits,target,pairs):
    upper=torch.triu(pairs,diagonal=1)
    bonded=upper&target[...,0].eq(0)
    log_presence=torch.logsumexp(logits[...,1:6],-1)
    # Normalize joint log probabilities in case evidence was added to the classes.
    normalizer=torch.logsumexp(logits[...,:6],-1)
    binary=-(1-target[...,0])*(log_presence-normalizer)-target[...,0]*(logits[...,0]-normalizer)
    conditional=F.cross_entropy(logits[...,1:6].movedim(-1,1),target[...,1:6].argmax(-1),reduction='none')
    losses={'presence':masked_graph_mean(binary,upper),'bond_type':masked_graph_mean(conditional,bonded)}
    for name,(start,end) in list(BLOCKS.items())[1:]:
        ce=F.cross_entropy(logits[...,start:end].movedim(-1,1),target[...,start:end].argmax(-1),reduction='none')
        losses[name]=masked_graph_mean(ce,bonded)
    return sum(losses.values()),losses


class RefinedBondModel(TypedBondModel):
    def __init__(self,config,priors,variant,num_tasks=1):
        super().__init__(config,num_tasks)
        self.variant=variant
        self.training_max_t=config.noise_steps
        self.factorized=variant in ('factorized','factorized_evidence')
        self.evidence=variant=='factorized_evidence'
        for name,prior in priors.items():
            self.register_buffer('prior_'+name,prior.clone())
        if self.evidence:
            self.encoder=CalibratedEncoder(config,priors['type'].clone())

    @torch.no_grad()
    def initialize_priors(self):
        if self.variant=='trained_reference':
            return
        self.bond_head[-1].weight.zero_()
        bias=self.bond_head[-1].bias
        for name,(start,end) in BLOCKS.items():
            bias[start:end].copy_(getattr(self,'prior_'+name).log())
        if self.factorized:
            p=self.prior_type
            bias[0]=torch.logit(1-p[0])
            bias[1:6]=(p[1:]/p[1:].sum()).log()

    def predict_bonds(self,condition,noisy,t):
        raw=super().predict_bonds(condition,noisy,t)
        logits=factor_logits(raw) if self.factorized else raw
        if self.evidence:
            # Within each target block, one-hot class means have equal squared norm.
            logits=logits+noisy*evidence_scale(self.schedule,noisy,t)
        return logits*condition.pairs[...,None]

    def loss(self,batch,generator=None):
        condition=condition_from_batch(batch,self.config.motif_prior)
        target=edge_target(batch,condition)
        t=torch.randint(1,self.training_max_t+1,(len(target),),device=target.device,generator=generator)
        logits=self.predict_bonds(condition,corrupt(target,t,condition.pairs,self.schedule,generator),t)
        return (factorized_objective if self.factorized else reconstruction_objective)(logits,target,condition.pairs)

    def checkpoint(self,stage,**extra):
        saved=super().checkpoint(stage,**extra)
        saved.update(schema=SCHEMA,variant=self.variant,training_max_t=self.training_max_t,
                     priors={name:getattr(self,'prior_'+name).detach().cpu() for name in BLOCKS})
        return saved


def load_model(path):
    from .model import Config
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved.get('feature_schema')!=FEATURE_SCHEMA:
        raise ValueError('Not a hierarchical typed-edge checkpoint')
    model=RefinedBondModel(Config(**saved['config']),saved['priors'],saved['variant'],saved['num_tasks'])
    model.training_max_t=saved.get('training_max_t',model.config.noise_steps)
    model.load_state_dict(saved['model'],strict=True)
    return model,saved
