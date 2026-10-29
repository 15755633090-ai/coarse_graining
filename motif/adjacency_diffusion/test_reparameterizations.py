"""Analytic tests for experimental transforms, objectives and gradient equivalence."""
import unittest
import torch

from .diffusion import Schedule
from .reparameterizations import decode, reconstruction_objective, degree_prior_baseline


class TransformTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.schedule=Schedule()
        self.pairs=(~torch.eye(4,dtype=torch.bool))[None].expand(3,-1,-1)
        target=torch.zeros(3,4,4,dtype=torch.float64)
        target[:,0,1]=target[:,1,0]=1
        self.target=target
        self.t=torch.tensor([1,500,1000])
        self.noisy,self.noise=self.schedule.sample(target,self.t,self.pairs)

    def test_preconditioned_epsilon_is_direct_A_with_matching_gradients(self):
        raw=torch.randn_like(self.target,requires_grad=True)
        clean,epsilon=decode(raw,self.noisy,self.t,self.pairs,self.schedule,'preconditioned_epsilon')
        restored=self.schedule.reconstruct(self.noisy,epsilon,self.t,self.pairs)
        torch.testing.assert_close(restored,clean,atol=1e-12,rtol=1e-12)
        first=torch.autograd.grad((clean-self.target).square().sum(),raw,retain_graph=True)[0]
        second=torch.autograd.grad((restored-self.target).square().sum(),raw)[0]
        torch.testing.assert_close(first,second,atol=1e-11,rtol=1e-11)

    def test_velocity_known_target_recovers_both_clean_and_noise(self):
        a=self.schedule.coefficient(self.noisy,self.t)
        velocity=a.sqrt()*self.noise-(1-a).sqrt()*self.target
        clean,epsilon=decode(velocity,self.noisy,self.t,self.pairs,self.schedule,'velocity')
        torch.testing.assert_close(clean,self.target,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(epsilon,self.noise,atol=1e-12,rtol=1e-12)

    def test_bounded_output_and_no_padding_or_diagonal(self):
        clean,epsilon=decode(torch.randn_like(self.target),self.noisy,self.t,self.pairs,self.schedule,'bounded_A')
        self.assertTrue(((clean[self.pairs]>0)&(clean[self.pairs]<1)).all())
        self.assertTrue((clean[~self.pairs]==0).all())
        torch.testing.assert_close(self.schedule.reconstruct(self.noisy,epsilon,self.t,self.pairs),clean,atol=1e-12,rtol=1e-12)

    def test_unmodified_loss_and_optional_degree_loss_are_explicit(self):
        clean=torch.zeros_like(self.target)
        actual=reconstruction_objective(clean,self.target,self.pairs,.2)
        torch.testing.assert_close(actual,((clean-self.target).square()*self.pairs).sum((1,2)).mean())
        changed=reconstruction_objective(clean,self.target,self.pairs,.2,degree_weight=.1)
        self.assertGreater(float(changed),float(actual))
        balanced=reconstruction_objective(clean,self.target,self.pairs,.2,balanced=True)
        self.assertTrue(torch.isfinite(balanced))

    def test_degree_baseline_only_needs_clean_node_conditions(self):
        from torch_geometric.data import Batch
        from .data import graph_from_smiles,condition_from_batch
        batch=Batch.from_data_list([graph_from_smiles(s) for s in ('CCO','c1ccccc1','C','[Na+].[Cl-]')])
        condition=condition_from_batch(batch)
        predicted=degree_prior_baseline(condition,.1)
        self.assertTrue(torch.isfinite(predicted).all())
        self.assertTrue(((predicted>=0)&(predicted<=1)).all())
        self.assertTrue((predicted[~condition.pairs]==0).all())
        torch.testing.assert_close(predicted,predicted.transpose(1,2),atol=1e-6,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
