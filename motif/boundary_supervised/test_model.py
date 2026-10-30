import unittest,io
from pathlib import Path
import torch
from torch_geometric.data import Batch,Data
from motif.adjacency_diffusion.typed_bonds import GraphDataset
from .model import BoundaryPredictor,Config,reverse_edges,load_model

class PredictorTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(31);torch.set_num_threads(2)
    def test_reverse_and_nonbacktracking(self):
        e=torch.tensor([[0,1,1,2],[1,0,2,1]])
        r=reverse_edges(e,3);torch.testing.assert_close(r,torch.tensor([1,0,3,2]))
        h=torch.arange(1,5).float()[:,None];agg=torch.zeros(3,1).index_add_(0,e[1],h)
        torch.testing.assert_close(agg[e[0]]-h[r],torch.tensor([[0.],[4.],[1.],[0.]]))
        with self.assertRaises(ValueError):reverse_edges(e[:,:1],3)

    def test_batch_permutation_fragment_ids_and_isolated_nodes(self):
        ds=GraphDataset(['CCOC(=O)NCC','c1ccccc1','[Na+]']);model=BoundaryPredictor(Config(hidden_dim=12,layers=2,dropout=0.)).eval()
        batch=Batch.from_data_list(list(ds));original=model(batch)
        for i in range(3):torch.testing.assert_close(model(Batch.from_data_list([ds[i]])),original[i:i+1],atol=1e-6,rtol=1e-6)
        graph=ds[0].clone();n=len(graph.x);perm=torch.randperm(n);inv=torch.empty_like(perm);inv[perm]=torch.arange(n)
        graph.x=graph.x[perm];graph.fragment_id=graph.fragment_id[perm]*7+19;graph.edge_index=inv[graph.edge_index]
        order=torch.randperm(graph.edge_index.shape[1]);graph.edge_index=graph.edge_index[:,order];graph.edge_attr=graph.edge_attr[order]
        torch.testing.assert_close(model(Batch.from_data_list([graph])),original[:1],atol=1e-6,rtol=1e-6)

    def test_internal_isolation_cross_gate_gradients_and_checkpoint(self):
        ds=GraphDataset(['CCOC(=O)NCC']);batch=Batch.from_data_list(list(ds));model=BoundaryPredictor(Config(hidden_dim=12,layers=2,dropout=0.)).eval()
        unique=batch.fragment_id.unique();self.assertGreater(len(unique),1)
        changed=batch.clone();mask=changed.fragment_id!=unique[0];changed.x[mask]+=3
        before=model.encode(batch)['internal'];after=model.encode(changed)['internal']
        torch.testing.assert_close(before[~mask],after[~mask],rtol=0,atol=0)
        model(batch).square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.boundary_gates.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.internal_messages.parameters()))
        buffer=io.BytesIO();torch.save(model.checkpoint('finetune'),buffer);buffer.seek(0)
        loaded,_=load_model(buffer);loaded.eval()
        torch.testing.assert_close(loaded(batch),model(batch),rtol=0,atol=0)
        with torch.no_grad():
            for gate in loaded.boundary_gates:
                gate[-1].weight.zero_();gate[-1].bias.fill_(-100.)
        torch.testing.assert_close(loaded.encode(batch)['atoms'][~mask],loaded.encode(changed)['atoms'][~mask],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
