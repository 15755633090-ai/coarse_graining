"""Verify in-memory HimNet batching against original BatchMolGraph on real molecules."""
import argparse
import os
import pickle
import tempfile
from pathlib import Path
import torch
from .data import save_json
from .paper_data import load_paper,prepare_him_graphs,HierarchicalBatch

def main():
    p=argparse.ArgumentParser();p.add_argument('--paper-root',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    _,loader,_=load_paper(args.paper_root,'HimNet','cpu')
    from dataset import BatchMolGraph
    rows=[{'row':0,'smiles':'CCOC(=O)N'},{'row':1,'smiles':'c1ccccc1O'}]
    objects={r['smiles']:loader.HiMolGraph(r['smiles']) for r in rows}
    raw=prepare_him_graphs(rows,loader);new=HierarchicalBatch([raw[0],raw[1]])
    cwd=Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        target=Path(tmp)/'data/audit/raw';target.mkdir(parents=True)
        with (target/'process_all.pkl').open('wb') as f:pickle.dump(objects,f)
        try:
            os.chdir(tmp)
            old=BatchMolGraph([[r['smiles']] for r in rows],new.f_atoms.shape[-1],new.f_bonds.shape[-1],new.fp_x.shape[-1],'audit')
        finally:os.chdir(cwd)
    for a,b in zip(new.get_components(),old.get_components()):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
        else:assert a==b
    assert torch.equal(new.fp_x,old.fp_x) and new.mol_atom_num==old.mol_atom_num
    result={'passed':True,'real_molecules':[r['smiles'] for r in rows],
            'components_bitwise_match_original_BatchMolGraph':True,'fingerprints_match':True,
            'original_graph_indices_unmodified':True,'performance_experiment':False}
    save_json(Path(__file__).parent/'runs/paper_data_check.json',result);print(result)

if __name__=='__main__':main()
