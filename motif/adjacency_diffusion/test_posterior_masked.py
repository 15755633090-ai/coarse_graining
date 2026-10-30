import copy,unittest
import torch
from torch_geometric.data import Batch
from .model import Config
from .masked_bonds import MaskedBondModel,mask_observation
from .typed_bonds import GraphDataset,edge_target
from .typed_refined import training_priors
from .data import condition_from_batch
from . import test_masked_bonds as base

class PosteriorMaskedTests(base.MaskedTests):
    def setUp(self):
        super().setUp();self.config.observation_weighting='gaussian_posterior'
        self.model=MaskedBondModel(self.config,training_priors(self.dataset),.3)

class PosteriorWeightTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.ds=GraphDataset(['CCOCN','[He]']);self.priors=training_priors(self.ds)
        self.cfg=Config(hidden_dim=12,layers=1,dropout=.1,motif_prior=True,conditional_gate=True,time_mode='random',observation_weighting='gaussian_posterior')
        self.model=MaskedBondModel(self.cfg,self.priors,.3)

    def test_low_noise_limit_and_mask_prior(self):
        batch=Batch.from_data_list([self.ds[0]]);c=condition_from_batch(batch,True)
        clean=edge_target(batch,c);observed,selected=mask_observation(clean,c.pairs,0.)
        t=torch.tensor([1]);weight=self.model.encoder.observation_weight(observed,t)
        upper=torch.triu(c.pairs,diagonal=1)
        torch.testing.assert_close(weight[upper],1-clean[...,0][upper],atol=1e-6,rtol=0)
        masked,selected=mask_observation(clean,c.pairs,.3,torch.Generator().manual_seed(9))
        masked.requires_grad_();weight=self.model.encoder.observation_weight(masked,t)
        torch.testing.assert_close(weight[selected],torch.full_like(weight[selected],1-self.priors['type'][0]))
        weight.sum().backward();self.assertEqual(float(masked.grad[selected].abs().sum()),0.)
        noisy=torch.zeros_like(observed)
        # No claim for nonfinite external inputs; extreme finite Gaussian input must stay finite.
        noisy[...,:6]=1e6
        self.assertTrue(torch.isfinite(self.model.encoder.observation_weight(noisy,torch.tensor([1000]))).all())

    def test_clean_predictions_and_gradients_unchanged(self):
        cfg=copy.deepcopy(self.cfg);cfg.observation_weighting='softmax'
        baseline=MaskedBondModel(cfg,self.priors,.3)
        common={k:v for k,v in self.model.state_dict().items() if k in baseline.state_dict()}
        baseline.load_state_dict(common,strict=True)
        for model in (baseline,self.model):model.train();model.set_stage('finetune')
        batch=Batch.from_data_list([self.ds[0],self.ds[1]])
        torch.manual_seed(19);a=baseline(batch);a.square().sum().backward()
        torch.manual_seed(19);b=self.model(batch);b.square().sum().backward()
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for (name,p),(other,q) in zip(baseline.named_parameters(),self.model.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
