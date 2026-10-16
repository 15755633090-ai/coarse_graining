"""Focused checks for internal-pattern labels and s_M-only supervision."""
import tempfile
from pathlib import Path
import torch
from rdkit import Chem
from .data import graph_features,collate_graphs,SCHEMA
from .diffusion import MotifDenoiser
from .structure import PATTERNS,structure_target,structure_loss
from .pretrain import load_encoder

def main():
    torch.set_num_threads(2)
    mol=Chem.MolFromSmiles('CC(=O)NCC#N')
    graph=graph_features(mol,range(mol.GetNumAtoms()))
    target=structure_target(graph);names=[x[0] for x in PATTERNS]
    assert target[names.index('amide_connectivity')]==1
    assert target[names.index('nitrile')]==1
    assert target[names.index('carbon_halogen')]==0
    # Truncation excludes the carbonyl: external atoms cannot supply its label.
    local=graph_features(mol,[3,4]);assert structure_target(local)[0]==0
    permutation=torch.arange(len(graph['x'])-1,-1,-1)
    assert torch.equal(target,structure_target({'x':graph['x'][permutation],'edges':graph['edges'][permutation][:,permutation]}))
    singleton=graph_features(Chem.MolFromSmiles('C'),[0]);assert not structure_target(singleton).any()
    model=MotifDenoiser(16,1,10,structure_dim=len(PATTERNS))
    batch=collate_graphs([graph]);out=model.encode(batch)
    # Only provide s_M, proving loss cannot consume local decoder states.
    loss=structure_loss(model,{'motif':out['motif']},target[None]);loss.backward()
    assert model.readout[0].weight.grad.abs().sum()>0
    assert model.x_in.weight.grad.abs().sum()>0
    assert model.edge_head.weight.grad is None
    assert model.node_decode[0].weight.grad is None
    with tempfile.TemporaryDirectory() as directory:
        for candidate in [model,MotifDenoiser(16,1,10)]:
            path=Path(directory)/'checkpoint.pt'
            torch.save({'schema':SCHEMA,'config':candidate.config,'model':candidate.state_dict()},path)
            loaded,_=load_encoder(path)
            assert torch.equal(candidate.encode(batch)['motif'],loaded.encode(batch)['motif'])
    print('PASS: labels, boundary isolation, permutation, singleton, s_M gradients, old/new checkpoints')

if __name__=='__main__':main()
