"""Repair missing original 3D features without dropping rows or changing motifs.

Only failed molecules retry RDKit embedding with random coordinates. Existing
records retain their values; padding/sentinel indices are expanded consistently.
The embedding change is explicitly recorded and shared by both downstream arms.
"""
import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .data import load_dataset,save_json,sha
from .paper_data import load_paper,load_hmg_features,hmg_inputs,align_motif_graphs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','paper-root','feature-cache','dependency-path','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    for key,value in vars(args).items():setattr(args,key,value.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.dependency_path))
    data=load_dataset(args.data);features,assemble=load_hmg_features(args.feature_cache,args.paper_root)
    missing=[]
    for row in data['molecules']:
        key=Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']),isomericSmiles=True)
        if row['smiles'] not in features['smiles_to_atom_info'] and key not in features['smiles_to_atom_info']:
            missing.append(row)
    if not missing:raise ValueError('No missing records; use existing cache')
    save_json(args.output/'repair_plan.json',{'missing_rows':[r['row'] for r in missing],
              'source_sha256':sha(args.feature_cache),'retry':'useRandomCoords=True, randomSeed=0; MMFF unchanged'})
    print({'missing_rows':[r['row'] for r in missing]},flush=True)
    _,_,cfg=load_paper(args.paper_root,'HMG-MRL','cuda:0')
    cfg.use_brics_fragments=cfg.use_murcko_fragments=cfg.use_smarts_fragments=True
    original=AllChem.EmbedMultipleConfs
    def retry(mol,*a,**kw):
        kw.update(useRandomCoords=True,randomSeed=0)
        ids=original(mol,*a,**kw)
        if not len(ids):raise RuntimeError('Random-coordinate retry also failed; no synthetic conformer substituted')
        return ids
    previous=Path.cwd()
    try:
        AllChem.EmbedMultipleConfs=retry;os.chdir(args.output)
        with (args.output/'repair.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
            module=importlib.import_module('preprocess.get_atom_bond_frag_info')
            extra=module.save_smiles_dicts([r['smiles'] for r in missing],str(args.output/'retried'))
    finally:
        AllChem.EmbedMultipleConfs=original;os.chdir(previous)
    if len(extra['smiles_to_atom_info'])!=len({Chem.MolToSmiles(Chem.MolFromSmiles(r['smiles']),isomericSmiles=True) for r in missing}):
        raise RuntimeError('Retry did not produce every missing molecule')
    sample=next(iter(features['smiles_to_atom_info']))
    added=next(iter(extra['smiles_to_atom_info']))
    a0=len(features['smiles_to_atom_info'][sample]);b0=len(features['smiles_to_bond_info'][sample])
    a1=len(extra['smiles_to_atom_info'][added]);b1=len(extra['smiles_to_bond_info'][added])
    atoms=max(a0,a1);bonds=max(b0,b1)
    for name in features:
        left=features[name];right=extra[name]
        if any(key in left for key in right):raise ValueError('Repair would overwrite an existing molecule')
        if isinstance(next(iter(left.values())),np.ndarray) and name!='smiles_to_descriptor':
            shape=tuple(max(x,y) for x,y in zip(next(iter(left.values())).shape,next(iter(right.values())).shape))
            indexed='neighbors' in name
            target_dummy=(atoms if name.endswith('_atom') else bonds)-1
            for records,old_a,old_b in [(left,a0,b0),(right,a1,b1)]:
                old_dummy=(old_a if name.endswith('_atom') else old_b)-1
                for key,value in records.items():
                    out=np.full(shape,target_dummy if indexed else 0,dtype=value.dtype)
                    value=np.where(value==old_dummy,target_dummy,value) if indexed else value
                    out[tuple(slice(0,n) for n in value.shape)]=value;records[key]=out
        left.update(right)
    path=args.output/'features.pickle'
    with path.open('wb') as stream:pickle.dump(features,stream)
    alignment=align_motif_graphs(data,data['molecules'],'HMG-MRL',hmg_features=features)[2]
    for start in range(0,len(data['molecules']),16):
        hmg_inputs(data['molecules'][start:start+16],features,assemble,'cpu')
    save_json(args.output/'manifest.json',{'repair':True,'missing_rows':[r['row'] for r in missing],
              'data_sha256':sha(args.data),'feature_cache_sha256':sha(path),
              'retry_embedding':{'useRandomCoords':True,'randomSeed':0},'original_MMFF_retained':True,
              'motif_alignment':alignment,'all_requested_rows_validated':True,
              'must_use_same_cache_for_both_arms':True})
    print('All molecules and motif mappings validated',flush=True)

if __name__=='__main__':main()
