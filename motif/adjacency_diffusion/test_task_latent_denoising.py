import unittest,io
import torch
from torch_geometric.data import Batch
from .task_latent_denoising import TaskLatentModel,select_fragments,normalized_feature_loss,load_model
from .masked_bonds import MaskedBondModel
from .typed_bonds import GraphDataset
from .typed_refined import training_priors
from .model import Config

class TaskLatentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        self.data=GraphDataset(['CCOC(=O)NCC','c1ccccc1','CC(=O)NCCC']);self.batch=Batch.from_data_list(list(self.data))
        self.cfg=Config(hidden_dim=12,layers=2,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random')
        self.priors=training_priors(self.data);self.model=TaskLatentModel(self.cfg,self.priors).eval()

    def test_clean_identity_and_capacity_and_checkpoint(self):
        baseline=MaskedBondModel(self.cfg,self.priors).eval();baseline.load_state_dict(self.model.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()),sum(p.numel() for p in self.model.parameters()))
        torch.testing.assert_close(self.model(self.batch),baseline(self.batch),rtol=0,atol=0)
        states,c,_=self.model.clean_states(self.batch);selected=c.nodes.clone()
        zero,_=self.model.latent_states(self.batch,selected,torch.zeros(3))
        torch.testing.assert_close(states['atoms'],zero['atoms'],rtol=0,atol=0)
        buffer=io.BytesIO();torch.save(self.model.checkpoint('finetune'),buffer);buffer.seek(0);replay,_=load_model(buffer);replay.eval()
        torch.testing.assert_close(self.model(self.batch),replay(self.batch),rtol=0,atol=0)

    def test_fragment_masks_and_no_context_samples(self):
        _,c,_=self.model.clean_states(self.batch)
        selected,eligible=select_fragments(self.batch,c,torch.Generator().manual_seed(7))
        self.assertFalse(bool(eligible[1]));self.assertFalse(bool(selected[1].any()))
        for i,graph in enumerate(self.data):
            for fid in graph.fragment_id.unique():
                part=selected[i,:len(graph.x)][graph.fragment_id==fid]
                self.assertTrue(bool(part.all()) or not bool(part.any()))
            if eligible[i]:self.assertTrue(bool((c.nodes[i]&~selected[i]).any()))

    def test_noise_and_property_gradients_reach_context_no_head_bypass(self):
        self.model.set_stage('finetune');clean,c,_=self.model.clean_states(self.batch)
        selected,_=select_fragments(self.batch,c,torch.Generator().manual_seed(3));sigma=torch.full((3,),.3)
        a,_=self.model.latent_states(self.batch,selected,sigma,torch.Generator().manual_seed(11))
        b,_=self.model.latent_states(self.batch,selected,sigma,torch.Generator().manual_seed(12))
        self.assertGreater(float((a['atoms']-b['atoms']).abs().sum()),0)
        prediction=self.model.predict_states(a,c);prediction.square().sum().backward(retain_graph=True)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.model.encoder.layers[0].h_message.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.model.encoder.fuse.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.bond_head.parameters()))
        changed=dict(a);changed['atoms']=torch.zeros_like(a['atoms'])
        self.assertGreater(float((prediction-self.model.predict_states(changed,c)).abs().sum()),0)
        x=a['atoms'].detach().requires_grad_();target=clean['atoms'].detach().requires_grad_()
        normalized_feature_loss(x,target,selected).backward();self.assertIsNotNone(x.grad);self.assertIsNone(target.grad)

    def test_batch_independence_with_fixed_noise(self):
        batch=self.batch;clean,c,_=self.model.clean_states(batch)
        selected,_=select_fragments(batch,c,torch.Generator().manual_seed(5));sigma=torch.full((3,),.3)
        epsilon=[torch.randn_like(clean['atoms']) for _ in range(3)]
        full,_=self.model.latent_states(batch,selected,sigma,epsilon=epsilon)
        one=Batch.from_data_list([self.data[0]]);n=len(self.data[0].x)
        actual,_=self.model.latent_states(one,selected[:1,:n],sigma[:1],epsilon=[v[:1,:n] for v in epsilon])
        torch.testing.assert_close(actual['atoms'][0,:n],full['atoms'][0,:n],atol=1e-6,rtol=1e-6)

if __name__=='__main__':unittest.main()
