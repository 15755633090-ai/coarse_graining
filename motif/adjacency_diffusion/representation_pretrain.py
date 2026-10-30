"""Clean/corrupt graph representation consistency with single-encoder SIGReg.

Graph adaptation inspired by LeJEPA (arXiv:2511.08544), not its vision recipe.
Both views receive gradients; no teacher, stop-gradient, or recovery objective.
"""
import torch
from .masked_bonds import MaskedBondModel
from .model import Config
from .typed_bonds import edge_target,FEATURE_SCHEMA
from .data import condition_from_batch

SCHEMA='brics_graph_representation_consistency_sigreg_v1'

def sigreg(x,generator=None,directions=256):
    """Mean projected Epps-Pulley statistic, frequencies[-5,5],17 points.

    Number of samples is molecules, not correlated atoms. Estimates at batch8
    have higher sampling noise than the source paper's recommended vision setup.
    """
    if x.ndim!=2 or len(x)<2 or directions<1:
        raise ValueError('SIGReg requires at least two independent graph samples')
    vectors=torch.randn(x.shape[1],directions,device=x.device,dtype=x.dtype,generator=generator)
    vectors=vectors/vectors.norm(dim=0,keepdim=True).clamp_min(1e-12)
    grid=torch.linspace(-5.,5.,17,device=x.device,dtype=x.dtype)
    angles=torch.einsum('bd,ds,t->bst',x,vectors,grid)
    gaussian=torch.exp(-grid.square()/2)
    error=((angles.cos().mean(0)-gaussian).square()+angles.sin().mean(0).square())*gaussian
    return torch.trapz(error,grid,dim=-1).mean()*len(x)

def representation_objective(clean,corrupted,generator=None,weight=.05,directions=256):
    if not 0<weight<1:raise ValueError('Regularizer weight must be in (0,1)')
    # Two global views; each predicts their mean. Equals one quarter of pair MSE.
    consistency=(clean-corrupted).square().mean()/4
    # Same directions across views; randomness does not affect corruption streams.
    state=generator.get_state() if generator is not None else None
    regularizer_clean=sigreg(clean,generator,directions)
    if generator is not None:generator.set_state(state)
    regularizer_corrupt=sigreg(corrupted,generator,directions)
    regularizer=(regularizer_clean+regularizer_corrupt)/2
    total=(1-weight)*consistency+weight*regularizer
    return total,{'consistency':consistency,'sigreg':regularizer}

class RepresentationModel(MaskedBondModel):
    def __init__(self,config,priors,mask_ratio=.3,num_tasks=1,regularizer_weight=.05,directions=256):
        super().__init__(config,priors,mask_ratio,num_tasks)
        if not 0<regularizer_weight<1 or directions<1:raise ValueError('Invalid representation objective')
        self.regularizer_weight=regularizer_weight;self.directions=directions

    def set_stage(self,stage):
        super().set_stage(stage)
        for p in self.bond_head.parameters():p.requires_grad_(False)

    def graph_representation(self,condition,observed,t):
        atoms=self.encoder(condition,observed,t)['atoms']
        return atoms.sum(1)/condition.nodes.sum(1).clamp_min(1)[:,None]

    def paired_representations(self,batch,t=None,noise_generator=None):
        c=condition_from_batch(batch,self.config.motif_prior);target=edge_target(batch,c)
        if t is None:t=torch.randint(1,self.training_max_t+1,(len(target),),device=target.device,generator=noise_generator)
        observed,selected=self.prepare_observation(target,t,c.pairs,noise_generator)
        return self.representations_from_observation(c,target,observed,t),selected

    def representations_from_observation(self,condition,target,observed,t):
        clean=torch.cat((target,torch.zeros_like(target[...,:1])),-1)
        zero=torch.zeros_like(t)
        return (self.graph_representation(condition,clean,zero),self.graph_representation(condition,observed,t))

    def loss(self,batch,generator=None,regularizer_generator=None):
        (a,b),_=self.paired_representations(batch,noise_generator=generator)
        return representation_objective(a,b,regularizer_generator,self.regularizer_weight,self.directions)

    def checkpoint(self,stage,**extra):
        saved=super().checkpoint(stage,**extra)
        saved.update(schema=SCHEMA,regularizer_weight=self.regularizer_weight,directions=self.directions,
            representation_target='two global graph readouts; gradients through both; no EMA/stopgrad',
            pretraining_objective='(1-weight)*pair_MSE/4 + weight*mean_view_SIGReg')
        return saved

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved.get('feature_schema')!=FEATURE_SCHEMA:
        raise ValueError('Not a graph-representation pretraining checkpoint')
    model=RepresentationModel(Config(**saved['config']),saved['priors'],saved['mask_ratio'],saved['num_tasks'],saved['regularizer_weight'],saved['directions'])
    model.training_max_t=saved['training_max_t'];model.load_state_dict(saved['model'],strict=True)
    return model,saved
