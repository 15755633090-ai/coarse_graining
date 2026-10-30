import unittest
import torch
from torch_geometric.data import Batch
from .model import Config
from .masked_bonds import MaskedBondModel
from .typed_bonds import GraphDataset
from .typed_refined import training_priors
from .path_gates import gated_paths,controls,DEFAULT

class PathGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.ds=GraphDataset(['CCOC(=O)NCC','CCO'])
        self.model=MaskedBondModel(Config(hidden_dim=12,layers=2,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random'),training_priors(self.ds),.3).eval()
        self.batch=Batch.from_data_list([self.ds[0],self.ds[1]])

    def test_identity_state_and_exception_restore(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()};expected=self.model(self.batch)
        with gated_paths(self.model):torch.testing.assert_close(self.model(self.batch),expected,atol=0,rtol=0)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with gated_paths(self.model,context_cross=0):
                with self.assertRaisesRegex(ValueError,'Transient diagnostic'):
                    self.model.checkpoint('finetune')
                with gated_paths(self.model,internal_readout=0):
                    self.assertEqual(controls(self.model.encoder).context_cross,0)
                self.assertEqual(controls(self.model.encoder).internal_readout,1)
                raise RuntimeError('injected')
        self.assertEqual(controls(self.model.encoder),DEFAULT)
        self.assertEqual(set(before),set(self.model.state_dict()))
        for k,v in self.model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        torch.testing.assert_close(self.model(self.batch),expected,atol=0,rtol=0)

    def test_internal_off_cuts_all_s_routes(self):
        with gated_paths(self.model,internal_branch=0):
            expected=self.model(self.batch)
            with torch.no_grad():
                for name,p in self.model.encoder.named_parameters():
                    if name.startswith('s_input') or '.s_' in name:p.add_(3.)
            torch.testing.assert_close(self.model(self.batch),expected,atol=0,rtol=0)

    def test_context_readout_off_cuts_h_dependence(self):
        with gated_paths(self.model,context_readout=0):
            expected=self.model(self.batch)
            with torch.no_grad():
                for name,p in self.model.encoder.named_parameters():
                    if name.startswith('h_input') or '.h_' in name or '.gate.' in name:p.add_(3.)
            torch.testing.assert_close(self.model(self.batch),expected,atol=0,rtol=0)

    def test_cross_gate_noop_without_boundary_and_s_skip_is_not_branch(self):
        lone=Batch.from_data_list([self.ds[1]]);expected=self.model(lone)
        self.assertEqual(self.ds[1].fragment_id.unique().numel(),1)
        with gated_paths(self.model,context_cross=0):torch.testing.assert_close(self.model(lone),expected,atol=0,rtol=0)
        self.model.zero_grad()
        with gated_paths(self.model,internal_readout=0):self.model(self.batch).square().sum().backward()
        self.assertGreater(float(self.model.encoder.s_input.weight.grad.abs().sum()),0.)

if __name__=='__main__':unittest.main()
