"""Joint Gaussian corruption and target-independent complete pair masking."""
import torch
from .model import Config,mlp,pair_nodes
from .data import condition_from_batch
from .typed_bonds import (TypedEncoder,EDGE_DIM,FEATURE_SCHEMA,edge_target,corrupt,reconstruction_objective)
from .typed_refined import RefinedBondModel,evidence_scale
from .diffusion import Schedule

SCHEMA='typed_gaussian_pair_mask_v1'


def mask_observation(noisy,pairs,ratio,generator=None):
    if not 0<=ratio<=1:
        raise ValueError('Mask ratio must be in [0,1]')
    upper=torch.triu(pairs,diagonal=1)
    sampled=(torch.rand(pairs.shape,device=noisy.device,generator=generator)<ratio)&upper
    # Keep a nonempty masked objective for each graph that has candidate pairs.
    # Selection depends ONLY on the candidate mask/RNG, never the true bond labels.
    if ratio>0:
        for i in range(len(pairs)):
            candidates=upper[i].nonzero()
            if len(candidates) and not bool(sampled[i].any()):
                index=torch.randint(len(candidates),(1,),device=noisy.device,generator=generator).item()
                u,v=candidates[index]
                sampled[i,u,v]=True
    selected=sampled|sampled.transpose(1,2)
    observed=noisy.masked_fill(selected[...,None],0)
    return torch.cat((observed,selected[...,None].to(noisy.dtype)),-1)*pairs[...,None],selected


class MaskedEncoder(TypedEncoder):
    input_dim=EDGE_DIM+1

    def __init__(self,config,type_prior):
        super().__init__(config)
        self.edge=mlp(self.input_dim,config.hidden_dim,config.hidden_dim)
        self.register_buffer('unknown_presence_prior',1-type_prior[0])
        if config.observation_weighting=='gaussian_posterior':
            self.register_buffer('type_prior',type_prior.clone())
            self.evidence_schedule=Schedule(config.noise_steps,config.beta_min,config.beta_max)

    def observation_weight(self,noisy,t):
        if self.config.observation_weighting=='gaussian_posterior':
            # Local categorical posterior for the actually received Gaussian data.
            # Equal-norm one-hot means make the class-independent quadratic cancel.
            logits=noisy[...,:6]*evidence_scale(self.evidence_schedule,noisy,t)+self.type_prior.log()
            visible=1-logits.softmax(-1)[...,0]
        else:
            visible=super().observation_weight(noisy,t)
        return torch.where(noisy[...,-1]>.5,self.unknown_presence_prior,visible)


class MaskedBondModel(RefinedBondModel):
    def __init__(self,config,priors,mask_ratio=.3,num_tasks=1):
        if not 0<=mask_ratio<=1:
            raise ValueError('Mask ratio must be in [0,1]')
        super().__init__(config,priors,'frequency_init',num_tasks)
        self.mask_ratio=mask_ratio
        self.encoder=MaskedEncoder(config,priors['type'])
        self.bond_head=mlp(3*config.hidden_dim+EDGE_DIM+1,config.hidden_dim,EDGE_DIM)

    def predict_bonds(self,condition,observed,t):
        states=self.encoder(condition,observed,t)
        a,b=pair_nodes(states['atoms'])
        raw=self.bond_head(torch.cat((a+b,(a-b).abs(),a*b,observed),-1))
        return .5*(raw+raw.transpose(1,2))*condition.pairs[...,None]

    def prepare_observation(self,target,t,pairs,generator=None):
        noisy=corrupt(target,t,pairs,self.schedule,generator)
        return mask_observation(noisy,pairs,self.mask_ratio,generator)

    def loss(self,batch,generator=None):
        condition=condition_from_batch(batch,self.config.motif_prior)
        target=edge_target(batch,condition)
        t=torch.randint(1,self.training_max_t+1,(len(target),),device=target.device,generator=generator)
        observed,selected=self.prepare_observation(target,t,condition.pairs,generator)
        logits=self.predict_bonds(condition,observed,t)
        return reconstruction_objective(logits,target,selected if self.mask_ratio>0 else condition.pairs)

    def forward(self,batch):
        condition=condition_from_batch(batch,self.config.motif_prior)
        clean=edge_target(batch,condition)
        clean=torch.cat((clean,torch.zeros_like(clean[...,:1])),-1)
        t=torch.zeros(len(clean),dtype=torch.long,device=clean.device)
        states=self.encoder(condition,clean,t)['atoms']
        return self.property_head(states.sum(1)/condition.nodes.sum(1).clamp_min(1)[:,None])

    def checkpoint(self,stage,**extra):
        saved=super().checkpoint(stage,**extra)
        saved.update(schema=SCHEMA,mask_ratio=self.mask_ratio,
                     masking='symmetric label-independent pair mask; all 17 channels removed; masked-only objective')
        return saved


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved.get('feature_schema')!=FEATURE_SCHEMA:
        raise ValueError('Not a joint Gaussian/pair-mask checkpoint')
    model=MaskedBondModel(Config(**saved['config']),saved['priors'],saved['mask_ratio'],saved['num_tasks'])
    model.training_max_t=saved['training_max_t']
    model.load_state_dict(saved['model'],strict=True)
    return model,saved
