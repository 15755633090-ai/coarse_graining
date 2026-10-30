import unittest
import torch
from torch_geometric.data import Batch
from .chemical_targets import chemical_targets,target_statistics,knowledge_loss,graph_readout,DESCRIPTORS
from .masked_bonds import MaskedBondModel,load_model
from .typed_bonds import GraphDataset
from .typed_refined import training_priors
from .model import Config

class ChemicalTargetTests(unittest.TestCase):
    def test_fixed_targets_and_train_only_scaler(self):
        d,f=chemical_targets(['CCO','c1ccccc1','CC(=O)O','CCN'])
        d2,f2=chemical_targets(['OCC','c1ccccc1','CC(=O)O','NCC'])
        torch.testing.assert_close(d,d2);torch.testing.assert_close(f,f2)
        stats=target_statistics(d,f)
        z=(d-stats['center'])/stats['scale']
        torch.testing.assert_close(z.mean(0),torch.zeros(len(DESCRIPTORS)),atol=1e-6,rtol=0)
        pred=torch.zeros(4,len(DESCRIPTORS)+512,requires_grad=True)
        loss,_=knowledge_loss(pred,d,f,stats);loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad.abs().sum()),0)

    def test_exact_prediction_readout_and_gradients(self):
        dataset=GraphDataset(['CCO','c1ccccc1','CC(=O)O','CCN'])
        model=MaskedBondModel(Config(hidden_dim=12,layers=1,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random'),training_priors(dataset),.3)
        model.set_stage('pretrain')
        for p in model.bond_head.parameters():p.requires_grad_(False)
        batch=Batch.from_data_list(list(dataset));model.eval()
        z=graph_readout(model,batch)
        torch.testing.assert_close(model.property_head(z),model(batch),rtol=0,atol=0)
        head=torch.nn.Linear(12,len(DESCRIPTORS)+512)
        d,f=chemical_targets(['CCO','c1ccccc1','CC(=O)O','CCN'])
        knowledge_loss(head(z),d,f,target_statistics(d,f))[0].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.property_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.bond_head.parameters()))

if __name__=='__main__':unittest.main()
