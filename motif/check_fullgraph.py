"""Semantic checks for overlap, uncovered atoms and fullgraph gradients."""
import torch
from rdkit import Chem
from .fullgraph import FullGraphEncoder,ModelConfig,graph_from_smiles,signature

def main():
    torch.set_num_threads(2);torch.manual_seed(7)
    cfg=ModelConfig(hidden_dim=16,num_layers=2,dropout=0).to_dict()
    model=FullGraphEncoder(cfg,{'<no-motif>':0,'<unknown>':1,'a':2,'b':3})
    r={'graph':graph_from_smiles('CCCO'),'members':[[0,1],[1,2]],'types':[2,3]}
    batch,c=model.batch([r],'cpu');w=model.type_embedding.weight
    torch.testing.assert_close(c[0,1],(w[2]+w[3])/2)
    torch.testing.assert_close(c[0,3],w[0])
    h=model([r]);s=model.encode_records([r]);assert s.shape==(2,16)
    s.square().sum().backward()
    assert model.type_embedding.weight.grad.abs().sum()>0
    assert model.legacy.layers[0].message[0].weight.grad.abs().sum()>0
    # External atom 3 is outside motif [1,2] but affects its encoded states.
    q={**r,'graph':graph_from_smiles('CCCN')}
    assert not torch.allclose(model([q])[0,2],h[0,2])
    assert signature(Chem.MolFromSmiles('CCO'),[0,1])==signature(Chem.MolFromSmiles('OCC'),[1,2])
    empty={**r,'members':[],'types':[]}
    assert model.encode_records([empty]).shape==(0,16)
    print('PASS overlap mean, no-motif, shared structural identity, external context, pooling gradients, empty motifs')

if __name__=='__main__':main()
