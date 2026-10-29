import tempfile
from pathlib import Path
import unittest
import torch
from torch_geometric.data import Batch
from .data import condition_from_batch,Condition
from .model import Config
from .typed_bonds import GraphDataset,edge_target,reconstruction_objective
from .typed_refined import training_priors
from .masked_bonds import MaskedBondModel,mask_observation,load_model


class MaskedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.dataset=GraphDataset(['CCOCN','C=C','c1ccccc1','[He]'])
        self.config=Config(hidden_dim=12,layers=1,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random')
        self.model=MaskedBondModel(self.config,training_priors(self.dataset),.3)

    def test_mask_symmetry_and_no_residual_path(self):
        batch=Batch.from_data_list([self.dataset[0],self.dataset[3]])
        condition=condition_from_batch(batch,True)
        noisy=torch.randn(*condition.pairs.shape,17,requires_grad=True)
        observed,selected=mask_observation(noisy,condition.pairs,.3,torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(selected,selected.transpose(1,2)))
        self.assertEqual(float(observed[...,:17][selected].abs().sum().detach()),0.)
        self.assertEqual(float(observed[~condition.pairs].abs().sum().detach()),0.)
        self.assertTrue(torch.equal(observed[...,-1].bool(),selected))
        observed[...,:17].sum().backward()
        self.assertEqual(float(noisy.grad[selected].abs().sum()),0.)
        self.assertGreater(float(noisy.grad[condition.pairs&~selected].abs().sum()),0.)

    def test_selection_is_label_independent_and_complete_mask_has_no_target(self):
        batch=Batch.from_data_list([self.dataset[0]])
        condition=condition_from_batch(batch,True)
        target=edge_target(batch,condition)
        t=torch.tensor([200])
        self.model.mask_ratio=1.
        first,mask1=self.model.prepare_observation(target,t,condition.pairs,torch.Generator().manual_seed(3))
        second,mask2=self.model.prepare_observation(target.roll(1,-1),t,condition.pairs,torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(mask1,mask2))
        self.model.eval()
        torch.testing.assert_close(self.model.predict_bonds(condition,first,t),self.model.predict_bonds(condition,second,t))
        _,a=mask_observation(target,condition.pairs,.3,torch.Generator().manual_seed(11))
        _,b=mask_observation(target.roll(1,-1),condition.pairs,.3,torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(a,b))

    def test_masked_only_objective(self):
        batch=Batch.from_data_list([self.dataset[0]])
        condition=condition_from_batch(batch,True)
        target=edge_target(batch,condition)
        _,selected=mask_observation(target,condition.pairs,.3,torch.Generator().manual_seed(4))
        logits=torch.randn_like(target,requires_grad=True)
        loss,_=reconstruction_objective(logits,target,selected)
        changed=logits.detach().clone()
        changed[~selected]=100
        other,_=reconstruction_objective(changed,target,selected)
        torch.testing.assert_close(loss,other)
        loss.backward()
        self.assertEqual(float(logits.grad[~selected].abs().sum()),0.)

    def test_clean_prediction_uses_features_and_flag_zero(self):
        batch=Batch.from_data_list([self.dataset[0]])
        received=[]
        handle=self.model.encoder.edge.register_forward_pre_hook(lambda module,inputs:received.append(inputs[0].detach().clone()))
        self.model.eval()
        first=self.model(batch)
        handle.remove()
        condition=condition_from_batch(batch,True)
        torch.testing.assert_close(received[0][...,:17],edge_target(batch,condition))
        self.assertEqual(float(received[0][...,-1].abs().sum()),0.)
        batch.edge_attr[:,0]=0
        batch.edge_attr[:,1]=1
        self.assertFalse(torch.equal(first,self.model(batch)))

    def test_checkpoint_permutation_and_empty_graph(self):
        batch=Batch.from_data_list([self.dataset[0],self.dataset[3]])
        condition=condition_from_batch(batch,True)
        t=torch.tensor([50,500])
        observed,_=self.model.prepare_observation(edge_target(batch,condition),t,condition.pairs)
        self.model.eval()
        expected=self.model.predict_bonds(condition,observed,t)
        order=torch.arange(observed.shape[1]-1,-1,-1)
        permuted=Condition(condition.x[:,order],condition.nodes[:,order],condition.pairs[:,order][:,:,order],condition.same_fragment[:,order][:,:,order])
        actual=self.model.predict_bonds(permuted,observed[:,order][:,:,order],t)
        torch.testing.assert_close(actual,expected[:,order][:,:,order],atol=1e-6,rtol=1e-5)
        lone=Batch.from_data_list([self.dataset[3]])
        loss,_=self.model.loss(lone)
        self.assertTrue(torch.isfinite(loss))
        root=Path(__file__).resolve().parent/'runs'
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path=Path(folder)/'masked.pt'
            torch.save(self.model.checkpoint('pretrain'),path)
            restored,_=load_model(path)
            for name,value in self.model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name],value,atol=0,rtol=0)
            from .typed_train import training_checkpoint
            self.assertEqual(training_checkpoint(path,self.model)['mask_ratio'],.3)

    def test_mask_rng_resume_after_latest_commit(self):
        import json
        from types import SimpleNamespace
        from . import typed_train as runner
        root=Path(__file__).resolve().parent/'runs'
        with tempfile.TemporaryDirectory(dir=root) as folder:
            def setup(name):
                out=Path(folder)/name
                out.mkdir()
                (out/'config.json').write_text(json.dumps({'test':'masked_resume'}),encoding='utf-8')
                torch.manual_seed(19)
                model=MaskedBondModel(self.config,training_priors(self.dataset),.3)
                args=SimpleNamespace(stage='pretrain',output_dir=out,seed=3,device='cpu',
                    learning_rate=.001,weight_decay=0.,grad_clip=5.,epochs=2,patience=3,resume=False,batch_size=2)
                return model,args,runner.io.loader(self.dataset,[0,1,2],args,True),runner.io.loader(self.dataset,[0,3],args,False)
            model,args,train,valid=setup('continuous')
            runner.fit(model,train,valid,args,{'test':'masked_resume'})
            _,expected=load_model(args.output_dir/'latest.pt')
            model,args,train,valid=setup('interrupted')
            original=runner.io.atomic_write
            def interrupt(path,value,**kwargs):
                original(path,value,**kwargs)
                if path.name=='latest.pt' and value['epoch']==1:
                    raise RuntimeError('injected masked interruption')
            runner.io.atomic_write=interrupt
            try:
                with self.assertRaisesRegex(RuntimeError,'injected masked'):
                    runner.fit(model,train,valid,args,{'test':'masked_resume'})
            finally:
                runner.io.atomic_write=original
            args.resume=True
            runner.fit(model,train,valid,args,{'test':'masked_resume'})
            _,actual=load_model(args.output_dir/'latest.pt')
            for name,value in expected['model'].items():
                torch.testing.assert_close(actual['model'][name],value,atol=0,rtol=0)
            self.assertEqual(actual['history'],expected['history'])


if __name__=='__main__':
    unittest.main()
