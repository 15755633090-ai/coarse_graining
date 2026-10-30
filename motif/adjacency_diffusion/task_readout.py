"""Task-side scale adaptation and an explicit molecular-size feature."""
import torch
from torch import nn
from .masked_bonds import MaskedBondModel
from .model import Config,mlp
from .data import condition_from_batch
from .typed_bonds import edge_target,FEATURE_SCHEMA

SCHEMA='typed_masked_normalized_size_readout_v1'

class TaskReadoutModel(MaskedBondModel):
    def __init__(self,config,priors,mask_ratio=.3,num_tasks=1):
        super().__init__(config,priors,mask_ratio,num_tasks)
        self.readout_norm=nn.LayerNorm(config.hidden_dim)
        self.property_head=mlp(config.hidden_dim+1,config.hidden_dim,num_tasks)

    def set_stage(self,stage):
        super().set_stage(stage)
        for p in self.readout_norm.parameters():p.requires_grad_(stage=='finetune')

    def forward(self,batch):
        c=condition_from_batch(batch,self.config.motif_prior);clean=edge_target(batch,c)
        observed=torch.cat((clean,torch.zeros_like(clean[...,:1])),-1)
        t=torch.zeros(len(clean),dtype=torch.long,device=clean.device)
        atoms=self.encoder(c,observed,t)['atoms'];count=c.nodes.sum(1).clamp_min(1).to(atoms.dtype)
        pooled=atoms.sum(1)/count[:,None]
        return self.property_head(torch.cat((self.readout_norm(pooled),count.log1p()[:,None]),-1))

    def checkpoint(self,stage,**extra):
        saved=super().checkpoint(stage,**extra)
        saved.update(schema=SCHEMA,readout='LayerNorm(mean_atoms) + log1p(atom_count)')
        return saved

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved.get('feature_schema')!=FEATURE_SCHEMA:
        raise ValueError('Not a normalized molecular-size readout checkpoint')
    model=TaskReadoutModel(Config(**saved['config']),saved['priors'],saved['mask_ratio'],saved['num_tasks'])
    model.training_max_t=saved['training_max_t'];model.load_state_dict(saved['model'],strict=True)
    return model,saved
