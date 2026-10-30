import unittest
import torch
from .model import Config
from .masked_bonds import MaskedBondModel
from .typed_bonds import GraphDataset
from .typed_refined import training_priors
from .typed_train import initialize_encoder_transfer

class TransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);cfg=Config(hidden_dim=12,layers=1,time_mode='random')
        priors=training_priors(GraphDataset(['CCOCN']))
        torch.manual_seed(7);self.source=MaskedBondModel(cfg,priors,.3)
        torch.manual_seed(19);self.target=MaskedBondModel(cfg,priors,.3)

    def test_only_projection_is_kept_fresh(self):
        fresh={k:v.clone() for k,v in self.target.encoder.state_dict().items()}
        before=sum(p.numel() for p in self.target.parameters())
        initialize_encoder_transfer(self.target,self.source.encoder.state_dict(),True)
        for k,v in self.target.encoder.state_dict().items():
            torch.testing.assert_close(v,fresh[k] if k.startswith('fuse.') else self.source.encoder.state_dict()[k],atol=0,rtol=0)
        self.assertEqual(before,sum(p.numel() for p in self.target.parameters()))
        self.assertTrue(any(not torch.equal(v,self.source.encoder.state_dict()[k]) for k,v in self.target.encoder.state_dict().items() if k.startswith('fuse.')))

    def test_full_transfer_default_is_unchanged(self):
        initialize_encoder_transfer(self.target,self.source.encoder.state_dict())
        for k,v in self.target.encoder.state_dict().items():torch.testing.assert_close(v,self.source.encoder.state_dict()[k],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
