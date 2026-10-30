import tempfile,unittest
from pathlib import Path
import torch
from torch_geometric.data import Batch
from .task_readout import TaskReadoutModel,load_model
from .model import Config
from .typed_bonds import GraphDataset
from .typed_refined import training_priors

class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.ds=GraphDataset(['CC','CCCC'])
        self.model=TaskReadoutModel(Config(hidden_dim=12,layers=1,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random'),training_priors(self.ds))

    def test_identical_mean_can_distinguish_size(self):
        batch=Batch.from_data_list([self.ds[0],self.ds[1]])
        received=[];hook=self.model.property_head.register_forward_pre_hook(lambda m,a:received.append(a[0]))
        original=self.model.encoder.forward
        self.model.encoder.forward=lambda c,o,t:{'atoms':torch.ones(*c.nodes.shape,12)*c.nodes[...,None]}
        self.model(batch);self.model.encoder.forward=original;hook.remove()
        torch.testing.assert_close(received[0][0,:12],received[0][1,:12])
        self.assertNotEqual(float(received[0][0,-1].detach()),float(received[0][1,-1].detach()))

    def test_gradients_checkpoint_and_permutation(self):
        self.model.set_stage('finetune');batch=Batch.from_data_list([self.ds[1]])
        expected=self.model(batch);expected.square().sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.encoder.parameters() if p.grad is not None),0.)
        self.assertGreater(float(self.model.readout_norm.weight.grad.abs().sum()),0.)
        g=self.ds[1].clone();p=torch.tensor([3,1,0,2]);inv=torch.argsort(p);g.x=g.x[p];g.fragment_id=g.fragment_id[p];g.edge_index=inv[g.edge_index]
        torch.testing.assert_close(self.model(Batch.from_data_list([g])),expected,atol=1e-6,rtol=1e-6)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as folder:
            path=Path(folder)/'model.pt';torch.save(self.model.checkpoint('finetune'),path)
            replay,_=load_model(path);torch.testing.assert_close(replay(batch),expected)
            from .typed_train import training_checkpoint
            self.assertEqual(training_checkpoint(path,self.model)['schema'],'typed_masked_normalized_size_readout_v1')

if __name__=='__main__':unittest.main()
