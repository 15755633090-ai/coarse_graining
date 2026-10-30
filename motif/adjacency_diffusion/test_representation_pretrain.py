import tempfile,unittest
from pathlib import Path
import torch
from torch_geometric.data import Batch
from .model import Config
from .masked_bonds import MaskedBondModel
from .typed_bonds import GraphDataset
from .typed_refined import training_priors
from .representation_pretrain import RepresentationModel,load_model,sigreg,representation_objective

class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.ds=GraphDataset(['CCOCN','CCO','c1ccccc1','CCC(=O)O'])
        self.cfg=Config(hidden_dim=12,layers=1,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random')
        self.model=RepresentationModel(self.cfg,training_priors(self.ds),directions=32)

    def test_sigreg_matches_characteristic_function_and_detects_collapse(self):
        x=torch.randn(32,12,dtype=torch.float64,requires_grad=True);seed=17
        result=sigreg(x,torch.Generator().manual_seed(seed),32)
        g=torch.Generator().manual_seed(seed);a=torch.randn(12,32,dtype=x.dtype,generator=g);a/=a.norm(dim=0)
        t=torch.linspace(-5,5,17,dtype=x.dtype)
        cf=torch.exp(1j*((x@a)[...,None]*t)).mean(0);normal=torch.exp(-t*t/2)
        reference=torch.trapz((cf-normal).abs().square()*normal,t,dim=-1).mean()*len(x)
        torch.testing.assert_close(result,reference,atol=1e-12,rtol=1e-12)
        result.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
        normal_samples=torch.randn(512,12)
        good=sigreg(normal_samples,torch.Generator().manual_seed(seed),32)
        collapsed=sigreg(torch.zeros_like(normal_samples),torch.Generator().manual_seed(seed),32)
        self.assertGreater(float(collapsed),float(good)*5)
        with self.assertRaises(ValueError):sigreg(torch.zeros(1,12))

    def test_both_views_receive_gradients(self):
        a=torch.randn(8,12,requires_grad=True);b=torch.randn(8,12,requires_grad=True)
        loss,parts=representation_objective(a,b,torch.Generator().manual_seed(3),directions=32)
        torch.testing.assert_close(parts['consistency'],(a-b).square().mean()/4)
        loss.backward()
        for x in [a,b]:self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_clean_prediction_unchanged_and_heads_not_trained(self):
        baseline=MaskedBondModel(self.cfg,training_priors(self.ds));baseline.load_state_dict(self.model.state_dict())
        batch=Batch.from_data_list([self.ds[i] for i in range(4)])
        self.model.eval();baseline.eval();torch.testing.assert_close(self.model(batch),baseline(batch),atol=0,rtol=0)
        self.model.set_stage('pretrain');loss,_=self.model.loss(batch,torch.Generator().manual_seed(5),torch.Generator().manual_seed(6));loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.encoder.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in self.model.bond_head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.property_head.parameters()))

    def test_checkpoint_and_replay(self):
        batch=Batch.from_data_list([self.ds[i] for i in range(4)]);expected=self.model(batch)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as folder:
            p=Path(folder)/'repr.pt';torch.save(self.model.checkpoint('pretrain'),p)
            restored,_=load_model(p);torch.testing.assert_close(restored(batch),expected,atol=0,rtol=0)
            from .typed_train import training_checkpoint
            self.assertEqual(training_checkpoint(p,self.model)['directions'],32)
            from .masked_bonds import load_model as old_loader
            with self.assertRaises(ValueError):old_loader(p)
        from types import SimpleNamespace
        from .typed_train import fit
        with self.assertRaisesRegex(ValueError,'dedicated runner'):
            fit(self.model,None,None,SimpleNamespace(stage='pretrain'),{})

if __name__=='__main__':unittest.main()
