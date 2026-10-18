"""Synthetic forward/backward through actual local paper model classes.

Use one process per paper to avoid their conflicting top-level module names.
This does not train on a molecular dataset or claim paper reproduction.
"""
import argparse
import sys
from pathlib import Path
import torch
from rdkit import Chem
from .data import graph_features,collate_graphs,to_device,save_json,sha
from .diffusion import MotifDenoiser
from .adapters import HMGMRLAdapter,HimNetAdapter

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--paper',choices=['hmg','him'],required=True)
    p.add_argument('--fixed-topology',action='store_true')
    args=p.parse_args();torch.manual_seed(18);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    denoiser=MotifDenoiser(32,2,20).to(device)
    g=graph_features(Chem.MolFromSmiles('CC(=O)O'),[1,2,3])
    mb=to_device(collate_graphs([g,g]),device)
    adapter_options={}
    if args.fixed_topology:
        from .fixed_topology import Encoder,graph
        denoiser=Encoder({'hidden':32,'layers':2,'steps':20}).to(device)
        mb=[{'graph':graph('CC(=O)O'),'members':[[1,2,3]]} for _ in range(2)]
        adapter_options={'fusion':'residual','gate_init':.1}
    if args.paper=='hmg':
        if device!='cuda': raise RuntimeError('Upstream HMG mask utilities require CUDA')
        source=ROOT/'research_code_audit/HMG-MRL'
        sys.path[:0]=[str(source),str(source/'model')]
        from config import cfg
        from model.bipartite_transformer import GNN_atom_bond
        cfg.atom_feature_dim=cfg.bond_feature_dim=32;cfg.frg_nhead=4;cfg.dropout=0.
        cfg.device=torch.device(device);cfg.radius=1;cfg.frag_num_encoder_layers=2
        backbone=GNN_atom_bond(cfg.atom_dim_in,cfg.bond_dim_in,1).to(device)
        model=HMGMRLAdapter(backbone,denoiser,**adapter_options).to(device)
        ax=torch.randn(2,4,cfg.atom_dim_in,device=device);be=torch.randn(2,3,cfg.bond_dim_in,device=device)
        aa=torch.full((2,4,6),3,device=device,dtype=torch.long)
        ab=torch.full((2,4,6),2,device=device,dtype=torch.long)
        aa[:,0,0]=1;aa[:,1,0]=0;aa[:,1,1]=2;aa[:,2,0]=1
        ab[:,0,0]=0;ab[:,1,0]=0;ab[:,1,1]=1;ab[:,2,0]=1
        bb=torch.full((2,3,13),2,device=device,dtype=torch.long)
        ba=torch.full((2,3,13),3,device=device,dtype=torch.long)
        ba[:,0,0]=0;ba[:,0,7]=1;ba[:,1,0]=1;ba[:,1,7]=2
        inputs={'x_atoms':ax,'x_bonds':be,'frag_src_features':torch.randn(2,2,cfg.atom_dim_in+cfg.bond_dim_in,device=device),
                'atom_neighbors_atom_index':aa,'atom_neighbors_bond_index':ab,
                'bond_neighbors_bond_index':bb,'bond_neighbors_atom_index':ba,
                'atom_mask':torch.tensor([[1,1,1,0]]*2,device=device,dtype=torch.float),
                'bond_mask':torch.tensor([[1,1,0]]*2,device=device,dtype=torch.float),
                'frag_src_valid_mask':torch.ones(2,2,device=device),'molecule_descriptor':torch.randn(2,200,device=device)}
        model.train();out=model(mb,[[0,1],[1,1]],**inputs)
        loss=out[1].square().mean()+out[2];loss.backward()
        output_shape=list(out[1].shape)
        source_file=source/'model/bipartite_transformer.py'
    else:
        # Reuse the already existing Windows import-compatible reproduction copy.
        source=ROOT/'coarse_graining1/HimNet_reproduction_20261013/windows'
        sys.path.insert(0,str(source))
        from model.HimNet import HimNet
        backbone=HimNet('synthetic',89,98,6338,hidden_size=32,depth=3,device=device,out_dim=1,num_heads=4).to(device)
        model=HimNetAdapter(backbone,denoiser,**adapter_options).to(device)
        class Graph:
            def __init__(self):
                self.f_atoms=torch.randn(11,89,device=device);self.f_atoms[0]=0
                self.fp_x=torch.randn(2,6338,device=device)
                self.mol_atom_num=[3,3];self.a_scope=[(1,5),(6,5)];self.b_scope=[(1,4),(5,4)]
                self.f_bonds=torch.randn(9,98,device=device);self.f_bonds[0]=0
                self.a2b=torch.tensor([[0,0],[2,0],[1,4],[3,0],[0,0],[0,0],
                                       [6,0],[5,8],[7,0],[0,0],[0,0]],device=device)
                self.b2a=torch.tensor([0,1,2,2,3,6,7,7,8],device=device)
                self.b2revb=torch.tensor([0,2,1,4,3,6,5,8,7],device=device)
            def get_components(self):
                return self.f_atoms,self.f_bonds,self.a2b,self.b2a,self.b2revb,self.a_scope,self.b_scope
        model.train();out=model(Graph(),mb,[4,9],training=True)
        loss=out.square().mean()+model.compute_contrastive_loss();loss.backward()
        output_shape=list(out.shape)
        source_file=source/'model/HimNet.py'
    assert torch.isfinite(loss)
    grad=denoiser.atoms[0].weight.grad if args.fixed_topology else denoiser.readout[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum())>0
    result={'paper':args.paper,'passed':True,'device':device,'source':str(source_file),
            'source_sha256':sha(source_file),'output_shape':output_shape,
            'pretrained_readout_grad_l1':float(grad.abs().sum()),'synthetic_only':True,
            'formal_dataset_training':False}
    suffix='_fixed' if args.fixed_topology else ''
    save_json(Path(__file__).parent/f'runs/{args.paper}{suffix}_adapter_check.json',result)
    print(result)

if __name__=='__main__':main()
