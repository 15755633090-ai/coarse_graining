import unittest
import torch
from torch_geometric.data import Batch
from .model import Config,reduce_messages
from .masked_bonds import MaskedBondModel,load_model
from .typed_refined import training_priors
from .typed_bonds import GraphDataset
import tempfile
from pathlib import Path
from . import test_masked_bonds as base

class NormalizedMaskedTests(base.MaskedTests):
    """Recheck masking, clean inputs, alignment and resume under the new reducer."""
    def setUp(self):
        super().setUp()
        self.config.message_normalization='observation_mass'
        self.model=MaskedBondModel(self.config,training_priors(self.dataset),.3)

class NormalizationTests(unittest.TestCase):
    def test_weighted_mean_and_zero_support(self):
        m=torch.tensor([[[[2.],[6.],[100.]]]])
        mask=torch.ones(1,1,3,dtype=torch.bool);w=torch.tensor([[[1.,1.,0.]]],requires_grad=True)
        result=reduce_messages(m,mask,w,'observation_mass')
        torch.testing.assert_close(result,torch.tensor([[[4.]]]))
        torch.testing.assert_close(reduce_messages(m,mask,w),torch.tensor([[[8/3]]]))
        result.sum().backward();self.assertTrue(torch.isfinite(w.grad).all())
        torch.testing.assert_close(reduce_messages(m,mask,w.detach()*0,'observation_mass'),torch.zeros(1,1,1))

    def test_checkpoint_and_permutation(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        ds=GraphDataset(['CCOCN']);cfg=Config(hidden_dim=12,layers=1,dropout=0.,motif_prior=True,
            conditional_gate=True,time_mode='random',message_normalization='observation_mass')
        model=MaskedBondModel(cfg,training_priors(ds),.3).eval();batch=Batch.from_data_list([ds[0]])
        expected=model(batch);graph=ds[0].clone();p=torch.tensor([4,2,0,1,3]);inverse=torch.argsort(p)
        graph.x=graph.x[p];graph.fragment_id=graph.fragment_id[p]
        graph.edge_index=inverse[graph.edge_index]
        torch.testing.assert_close(model(Batch.from_data_list([graph])),expected,atol=1e-6,rtol=1e-6)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'model.pt';torch.save(model.checkpoint('finetune'),path)
            replay,_=load_model(path);replay.eval();torch.testing.assert_close(replay(batch),expected)

if __name__=='__main__': unittest.main()
