"""Bridge real molecular records to the original paper model input formats."""
import ast
from collections import Counter
import importlib
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .data import sha

def source_inventory(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(Path(root).rglob('*.py'))}

def load_paper(root,paper,device,hidden=None):
    root=Path(root).resolve()
    if not root.exists():raise FileNotFoundError(root)
    if 'model' in sys.modules or 'dataset' in sys.modules:
        raise RuntimeError('Paper packages collide: run each paper in a fresh process')
    sys.path[:0]=[str(root),str(root/'model')]
    if paper=='HimNet':
        cls=importlib.import_module('model.HimNet').HimNet
        loader=importlib.import_module('dataset.loader')
        return cls,loader,None
    if not str(device).startswith('cuda'):
        raise ValueError('Original HMG-MRL mask utilities require CUDA; not silently patched')
    cfg=importlib.import_module('config').cfg
    cfg.device=torch.device(device)
    if hidden is not None:cfg.atom_feature_dim=cfg.bond_feature_dim=hidden
    cls=importlib.import_module('model.bipartite_transformer').GNN_atom_bond
    return cls,None,cfg

def prepare_him_graphs(molecules,loader):
    graphs={}
    for row in molecules:
        graph=loader.HiMolGraph(row['smiles'])
        graphs[row['row']]={name:getattr(graph,name) for name in
                ('n_atoms','n_bonds','f_atoms','f_bonds','a2b','b2a','b2revb','num_part','fp_x','cliques')}
    return graphs

class HierarchicalBatch:
    """Same feature/index concatenation as upstream BatchMolGraph, in memory.

    Avoids the hardcoded cwd/data/{name}/raw/process_all.pkl lookup and unused
    auxiliary relation code. Does not repair/change original molecular indices.
    """
    def __init__(self,graphs):
        ad=len(graphs[0]['f_atoms'][0]);bd=len(graphs[0]['f_bonds'][0])
        atoms=[[0.]*ad];bonds=[[0.]*bd];a2b=[[]];b2a=[0];rev=[0]
        self.a_scope=[];self.b_scope=[];self.mol_atom_num=[]
        na=nb=1
        for g in graphs:
            atoms.extend(g['f_atoms']);bonds.extend(g['f_bonds'])
            a2b.extend([[i+nb for i in incoming] for incoming in g['a2b']])
            b2a.extend([a+na for a in g['b2a']]);rev.extend([b+nb for b in g['b2revb']])
            self.a_scope.append((na,g['n_atoms']));self.b_scope.append((nb,g['n_bonds']))
            self.mol_atom_num.append(int(g['num_part'][0][0]));na+=g['n_atoms'];nb+=g['n_bonds']
        width=max(1,max(map(len,a2b)))
        self.f_atoms=torch.tensor(atoms,dtype=torch.float);self.f_bonds=torch.tensor(bonds,dtype=torch.float)
        self.a2b=torch.tensor([r+[0]*(width-len(r)) for r in a2b],dtype=torch.long)
        self.b2a=torch.tensor(b2a,dtype=torch.long);self.b2revb=torch.tensor(rev,dtype=torch.long)
        self.fp_x=torch.cat([g['fp_x'] for g in graphs])

    def get_components(self):
        return self.f_atoms,self.f_bonds,self.a2b,self.b2a,self.b2revb,self.a_scope,self.b_scope

HMG_NAMES=('x_atoms','x_bonds','frag_src_features','atom_neighbors_atom_index','atom_neighbors_bond_index',
           'bond_neighbors_bond_index','bond_neighbors_atom_index','atom_mask','bond_mask','frag_src_valid_mask')

def load_hmg_features(path,root):
    # Explicit trusted original feature artifact, never downloaded or fabricated.
    with Path(path).open('rb') as f:features=pickle.load(f)
    source=Path(root)/'preprocess/get_atom_bond_frag_info.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_smiles_array')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    # Running the exact upstream cache assembly function avoids importing
    # DeepChem when features were already produced in the author's environment.
    return features,env['get_smiles_array']

def hmg_inputs(rows,features,assemble,device):
    keys=[];mols=[]
    for row in rows:
        raw=row['smiles'];canonical=Chem.MolToSmiles(Chem.MolFromSmiles(raw),isomericSmiles=True)
        key=raw if raw in features['smiles_to_atom_info'] else canonical
        if key not in features['smiles_to_atom_info']:raise KeyError(f"Missing original HMG features for row {row['row']}")
        keys.append(key);mols.append(Chem.MolFromSmiles(key))
    arrays=assemble(keys,features)
    inputs={name:torch.as_tensor(arrays[k],device=device,dtype=torch.long if 3<=k<=6 else torch.float)
            for k,name in enumerate(HMG_NAMES)}
    inputs['molecule_descriptor']=torch.as_tensor(arrays[11],device=device,dtype=torch.float)
    if inputs['molecule_descriptor'].shape[-1]!=200:raise ValueError('Original HMG model expects 200 descriptor features')
    maps=[features['smiles_to_frag_atom_bond_incices'][key] for key in keys]
    for i,(mol,mapping) in enumerate(zip(mols,maps)):
        atoms=sorted(mapping['whole_mol'][0])
        if atoms!=list(range(mol.GetNumAtoms())):raise ValueError('HMG source atom mapping mismatch')
        tokens=1+len([k for k in mapping if k!='whole_mol'])
        if tokens>inputs['frag_src_features'].shape[1]:raise ValueError('Original cache truncated motifs')
        if not bool((inputs['frag_src_valid_mask'][i,:tokens]>0).all()):raise ValueError('Invalid original motif token mask')
    return inputs,mols,maps

def align_motif_graphs(data,rows,paper,him_graphs=None,hmg_features=None):
    """Verify original-paper membership against pretraining, preserving paper order.

    Returned graphs are the exact clean pretraining inputs, not fixed vectors;
    the encoder remains trainable. HMG canonical SMILES may reorder atom indices.
    """
    aligned={};positions={};total=0;canonicalized=0
    for row in rows:
        original=Chem.MolFromSmiles(row['smiles']);n=original.GetNumAtoms()
        raw=[data['motifs'][i] for i in row['motifs'][paper]]
        lookup={tuple(sorted(g['members'])):g for g in raw}
        if len(lookup)!=len(raw):raise ValueError('Duplicate pretraining membership signatures')
        if paper=='HimNet':
            graph=him_graphs[row['row']]
            if int(graph['num_part'][0][0])!=n:raise ValueError('Original atom count mismatch')
            groups=graph['cliques'];token_indices=list(range(len(groups)))
        else:
            canonical=Chem.MolToSmiles(original,isomericSmiles=True)
            order=ast.literal_eval(original.GetProp('_smilesAtomOutputOrder'))
            key=row['smiles'] if row['smiles'] in hmg_features['smiles_to_atom_info'] else canonical
            mapping=hmg_features['smiles_to_frag_atom_bond_incices'][key]
            if key==row['smiles']:order=list(range(n))
            else:canonicalized+=1
            keyed=Chem.MolFromSmiles(key)
            if sorted(order)!=list(range(n)):raise ValueError('Canonical index mapping is not a permutation')
            for j,i in enumerate(order):
                if keyed.GetAtomWithIdx(j).GetAtomicNum()!=original.GetAtomWithIdx(i).GetAtomicNum():
                    raise ValueError('Canonical atom mapping mismatch')
            original_bonds={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):str(b.GetBondType()) for b in original.GetBonds()}
            keyed_bonds={tuple(sorted((order[b.GetBeginAtomIdx()],order[b.GetEndAtomIdx()]))):str(b.GetBondType()) for b in keyed.GetBonds()}
            if keyed_bonds!=original_bonds:raise ValueError('Canonical bond mapping mismatch')
            keys=sorted((k for k in mapping if k!='whole_mol'),key=lambda k:int(k.split('_')[1]))
            groups=[[order[i] for i in mapping[k][0]] for k in keys];token_indices=list(range(1,len(keys)+1))
        signatures=[tuple(sorted(group)) for group in groups]
        if Counter(signatures)!=Counter(lookup.keys()):
            raise ValueError(f"Motif membership differs from pretraining: {paper} row {row['row']}")
        aligned[row['row']]=[lookup[s] for s in signatures];positions[row['row']]=token_indices
        total+=len(groups)
    report={'passed':True,'paper':paper,'molecules':len(rows),'motifs':total,'canonicalized_rows':canonicalized,
            'member_sets_match_pretraining':True,'original_token_order_preserved':True,
            'input':'exact clean pretraining graphs, t=0; no cached frozen vectors'}
    return aligned,positions,report
