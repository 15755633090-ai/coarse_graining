import unittest
import torch
from .posterior import likelihood_log_odds,PosteriorModel
from .model import Config
from .data import GraphDataset,condition_from_batch,adjacency_target
from torch_geometric.data import Batch


class PosteriorTests(unittest.TestCase):
    def test_gaussian_likelihood(self):
        y=torch.tensor([[[0.,.3],[.3,0.]]],requires_grad=True)
        a=torch.tensor([[[.7]]])
        mask=torch.tensor([[[False,True],[True,False]]])
        actual=likelihood_log_odds(y,a,mask)
        expected=(-(y-a.sqrt()).square()+y.square())/(2*(1-a))*mask
        torch.testing.assert_close(actual,expected)
        actual.sum().backward()
        self.assertGreater(float(y.grad.abs().sum()),0)

    def test_observation_control_and_alignment(self):
        batch=Batch.from_data_list([GraphDataset(['CCO'])[0]])
        condition=condition_from_batch(batch,True)
        target=adjacency_target(batch,condition)
        config=Config(hidden_dim=8,layers=1,dropout=0.,motif_prior=True,conditional_gate=True)
        t=torch.tensor([500])
        model=PosteriorModel(config,evidence=False).eval()
        with self.assertRaises(RuntimeError):
            model.reconstruction_loss(batch)
        first=model.predict_noise(condition,target,t)
        second=model.predict_noise(condition,torch.randn_like(target),t)
        torch.testing.assert_close(first,second)
        self.assertEqual(first.shape,target.shape)
        model.evidence=True
        output=model.predict_noise(condition,target,t)
        self.assertFalse(torch.allclose(output,first))
        torch.testing.assert_close(output,output.transpose(1,2))
        self.assertEqual(float(output.diagonal(dim1=1,dim2=2).abs().sum().detach()),0.)


if __name__=='__main__':
    unittest.main()
