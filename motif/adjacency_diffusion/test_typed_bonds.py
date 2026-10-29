import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch_geometric.data import Batch
from .data import condition_from_batch
from .model import Config
from .typed_bonds import (GraphDataset,edge_target,corrupt,TypedBondModel,
                          reconstruction_objective,recovery_counts)


class TypedBondsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model=TypedBondModel(Config(hidden_dim=12,layers=2,dropout=0.,
                                 motif_prior=True,conditional_gate=True,time_mode='random'))

    def test_feature_retention(self):
        data=GraphDataset(['CC','C=C','C#N','c1ccccc1','F/C=C/F','C=CC=C'])
        expected=[1,2,3,4]
        for i,label in enumerate(expected):
            batch=Batch.from_data_list([data[i]])
            condition=condition_from_batch(batch,True)
            target=edge_target(batch,condition)
            self.assertEqual(target.shape[-1],17)
            self.assertEqual(int(target[0,0,1,:6].argmax()),label)
            self.assertEqual(batch.x.shape[-1],133)
            self.assertEqual(batch.edge_attr.shape[-1],16)
        batch=Batch.from_data_list([data[3],data[4],data[5]])
        condition=condition_from_batch(batch,True)
        target=edge_target(batch,condition)
        self.assertGreater(float(target[...,16].sum()),0)  # ring true
        self.assertGreater(float(target[...,14].sum()),0)  # conjugated true
        self.assertGreater(float(target[...,7:13].sum()),0)  # nondefault stereo
        torch.testing.assert_close(target,target.transpose(1,2))

    def test_no_clean_feature_bypass_and_gradients(self):
        batch=Batch.from_data_list([GraphDataset(['CCOC'])[0]])
        condition=condition_from_batch(batch,True)
        target=edge_target(batch,condition)
        t=torch.tensor([500])
        noisy=corrupt(target,t,condition.pairs,self.model.schedule).requires_grad_()
        self.model.eval()
        logits=self.model.predict_bonds(condition,noisy,t)
        batch.edge_attr.zero_()
        batch.edge_index=torch.empty(2,0,dtype=torch.long)
        after=self.model.predict_bonds(condition_from_batch(batch,True),noisy,t)
        torch.testing.assert_close(logits,after)
        loss,blocks=reconstruction_objective(logits,target,condition.pairs)
        self.assertEqual(set(blocks),{'type','stereo','conjugated','ring'})
        loss.backward()
        for start,end in ((0,6),(6,13),(13,15),(15,17)):
            self.assertGreater(float(noisy.grad[...,start:end].abs().sum()),0)
            self.assertGreater(float(self.model.bond_head[-1].weight.grad[start:end].abs().sum()),0)

    def test_clean_task_uses_bond_features(self):
        batch=Batch.from_data_list([GraphDataset(['CCO'])[0]])
        self.model.eval()
        first=self.model(batch)
        batch.edge_attr[:,0]=0
        batch.edge_attr[:,1]=1  # keep topology and atoms fixed, change only bond type
        second=self.model(batch)
        self.assertFalse(torch.allclose(first,second,atol=1e-9,rtol=1e-9))

    def test_alignment_padding_and_degenerate_graphs(self):
        batch=Batch.from_data_list([GraphDataset(['CCO'])[0],GraphDataset(['[He]'])[0]])
        condition=condition_from_batch(batch,True)
        target=edge_target(batch,condition)
        t=torch.tensor([200,500])
        noisy=corrupt(target,t,condition.pairs,self.model.schedule)
        torch.testing.assert_close(noisy,noisy.transpose(1,2))
        self.assertEqual(float(noisy[~condition.pairs].abs().sum()),0.)
        self.model.eval()
        logits=self.model.predict_bonds(condition,noisy,t)
        permutation=torch.tensor([2,0,1])
        from .data import Condition
        permuted=Condition(condition.x[:,permutation],condition.nodes[:,permutation],
                           condition.pairs[:,permutation][:,:,permutation],
                           condition.same_fragment[:,permutation][:,:,permutation])
        actual=self.model.predict_bonds(permuted,noisy[:,permutation][:,:,permutation],t)
        torch.testing.assert_close(actual,logits[:,permutation][:,:,permutation],atol=1e-6,rtol=1e-5)
        loss,_=reconstruction_objective(logits,target,condition.pairs)
        self.assertTrue(torch.isfinite(loss))
        counts=recovery_counts(target*20,target,condition.pairs)
        self.assertEqual(counts['joint_bond_correct'],counts['bonds'])

    def test_typed_resume_and_checkpoint_schema(self):
        from . import typed_train as runner
        from .typed_bonds import load_checkpoint
        # tempfile uses an existing workspace folder and removes only this test's files.
        root=Path(__file__).resolve().parent/'runs'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            data=GraphDataset(['CC','C=C','C#N','[He]'])
            def setup(name):
                folder=Path(temporary)/name
                folder.mkdir(exist_ok=True)
                (folder/'config.json').write_text(json.dumps({'test':'resume'}),encoding='utf-8')
                args=SimpleNamespace(stage='pretrain',output_dir=folder,seed=3,device='cpu',
                    learning_rate=.001,weight_decay=0.,grad_clip=5.,epochs=2,patience=3,
                    resume=False,batch_size=2)
                torch.manual_seed(11)
                model=TypedBondModel(self.model.config)
                train=runner.io.loader(data,[0,1,2],args,True)
                valid=runner.io.loader(data,[1,3],args,False)
                return model,train,valid,args
            model,train,valid,args=setup('continuous')
            runner.fit(model,train,valid,args,{'test':'resume'})
            expected=load_checkpoint(args.output_dir/'latest.pt')
            model,train,valid,args=setup('interrupted')
            original=runner.io.atomic_write
            def interrupt(path,value,**kwargs):
                original(path,value,**kwargs)
                if path.name=='latest.pt' and value['epoch']==1:
                    raise RuntimeError('injected after latest commit')
            runner.io.atomic_write=interrupt
            try:
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    runner.fit(model,train,valid,args,{'test':'resume'})
            finally:
                runner.io.atomic_write=original
            original(args.output_dir/'best.pt',{'wrong':'stale export'},binary=True)
            args.resume=True
            runner.fit(model,train,valid,args,{'test':'resume'})
            actual=load_checkpoint(args.output_dir/'latest.pt')
            for key,tensor in expected['model'].items():
                torch.testing.assert_close(actual['model'][key],tensor,atol=0,rtol=0)
            self.assertEqual(actual['history'],expected['history'])
            wrong=Path(temporary)/'binary.pt'
            torch.save({'schema':'paper_mechanism_binary_adjacency_reconstruction_v1'},wrong)
            with self.assertRaises(ValueError):
                load_checkpoint(wrong)


if __name__=='__main__':
    unittest.main()
