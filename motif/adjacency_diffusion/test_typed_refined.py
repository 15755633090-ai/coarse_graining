import tempfile
from pathlib import Path
import unittest
import torch
from torch_geometric.data import Batch
from .typed_bonds import GraphDataset,BLOCKS,edge_target,reconstruction_objective
from .typed_refined import (RefinedBondModel,training_priors,evidence_scale,
                            factorized_objective,load_model)
from .data import condition_from_batch
from .model import Config


class RefinedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.data=GraphDataset(['CC','C=C','C#N','c1ccccc1','F/C=C/F'])
        self.priors=training_priors(self.data)
        self.config=Config(hidden_dim=12,layers=1,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random')

    def test_evidence_and_training_only_priors(self):
        for prior in self.priors.values():
            self.assertTrue(bool((prior>0).all()))
            torch.testing.assert_close(prior.sum(),torch.tensor(1.))
        model=RefinedBondModel(self.config,self.priors,'factorized_evidence').eval()
        model.initialize_priors()
        batch=Batch.from_data_list([self.data[1]])
        condition=condition_from_batch(batch,True)
        target=edge_target(batch,condition)
        t=torch.tensor([50])
        logits=model.predict_bonds(condition,target,t)
        scale=evidence_scale(model.schedule,target,t)
        prior=model.prior_type.log()
        expected=(prior+target[...,:6]*scale).softmax(-1)
        torch.testing.assert_close(logits[...,:6].softmax(-1)[condition.pairs],expected[condition.pairs])
        for name,(start,end) in BLOCKS.items():
            selected=condition.pairs & target[...,0].eq(0)
            self.assertTrue(torch.equal(logits[...,start:end].argmax(-1)[selected],target[...,start:end].argmax(-1)[selected]))

    def test_rare_type_gradient_and_clean_bond_use(self):
        batch=Batch.from_data_list([self.data[1],self.data[3]])
        condition=condition_from_batch(batch,True)
        target=edge_target(batch,condition)
        model=RefinedBondModel(self.config,self.priors,'factorized').eval()
        model.initialize_priors()
        logits=model.predict_bonds(condition,torch.randn_like(target),torch.tensor([500,500]))
        loss,blocks=factorized_objective(logits,target,condition.pairs)
        self.assertEqual(set(blocks),{'presence','bond_type','stereo','conjugated','ring'})
        loss.backward()
        for channel in (0,2,4):
            self.assertGreater(float(model.bond_head[-1].bias.grad[channel].abs()),0)
        first=model(batch)
        batch.edge_attr[:,0]=0
        batch.edge_attr[:,2]=1
        self.assertFalse(torch.allclose(first,model(batch),atol=1e-9,rtol=1e-9))

    def test_checkpoint_roundtrip(self):
        model=RefinedBondModel(self.config,self.priors,'factorized_evidence')
        model.training_max_t=300
        root=Path(__file__).resolve().parent/'runs'
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path=Path(folder)/'checkpoint.pt'
            torch.save(model.checkpoint('pretrain'),path)
            restored,_=load_model(path)
            self.assertEqual(restored.training_max_t,300)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key],value,atol=0,rtol=0)
            from .typed_train import training_checkpoint
            self.assertEqual(training_checkpoint(path,model)['variant'],'factorized_evidence')
            other=RefinedBondModel(self.config,self.priors,'frequency_init')
            with self.assertRaises(ValueError):
                training_checkpoint(path,other)

    def test_optimizer_groups(self):
        from types import SimpleNamespace
        from .typed_train import make_optimizer
        model=RefinedBondModel(self.config,self.priors,'frequency_init')
        model.set_stage('finetune')
        args=SimpleNamespace(stage='finetune',learning_rate=.001,weight_decay=0.,encoder_learning_rate=.0001)
        optimizer=make_optimizer(model,args)
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[.0001,.001])
        identities=[id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(identities),len(set(identities)))
        self.assertEqual(set(identities),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual({id(p) for p in optimizer.param_groups[0]['params']},
                         {id(p) for p in model.encoder.parameters()})
        args.encoder_learning_rate=None
        self.assertEqual(len(make_optimizer(model,args).param_groups),1)


if __name__=='__main__':
    unittest.main()
