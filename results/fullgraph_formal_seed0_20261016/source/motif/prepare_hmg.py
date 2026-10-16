"""Generate an original HMG-MRL feature cache from our real molecular rows."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import importlib
import os
from pathlib import Path
import sys
import numpy as np
from rdkit import Chem,rdBase
from .data import load_dataset,save_json,sha,code_identity
from .paper_data import load_paper,source_inventory,load_hmg_features,hmg_inputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--paper-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New output directory; contains features.pickle')
    p.add_argument('--dependency-path',type=Path,help='Optional isolated DeepChem installation directory')
    p.add_argument('--smoke-per-split',type=int,default=0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.data=args.data.resolve();args.paper_root=args.paper_root.resolve();args.output=args.output.resolve()
    if args.dependency_path:sys.path.insert(0,str(args.dependency_path.resolve()))
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True);source=source_inventory(args.paper_root);code=code_identity()
    data=load_dataset(args.data);splits={s:[m for m in data['molecules'] if m['split']==s] for s in ('train','valid','test')}
    if args.smoke_per_split:splits={s:rows[:args.smoke_per_split] for s,rows in splits.items()}
    selected=[m for rows in splits.values() for m in rows]
    _,_,cfg=load_paper(args.paper_root,'HMG-MRL',args.device)
    # Explicitly select the audited original three extraction strategies.
    cfg.use_brics_fragments=cfg.use_murcko_fragments=cfg.use_smarts_fragments=True
    np.random.seed(args.seed);rdBase.SeedRandomNumberGenerator(args.seed)
    cwd=Path.cwd()
    try:
        os.chdir(args.output)
        with (args.output/'preprocess.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
            dc=importlib.import_module('deepchem')
            preprocess=importlib.import_module('preprocess.get_atom_bond_frag_info')
            # save_smiles_dicts reads bond_features from each arrayrep. The
            # similarly named get_smiles_dicts has a stale-variable bug; do not
            # silently patch it or use that alternate path here.
            features=preprocess.save_smiles_dicts([m['smiles'] for m in selected],str(args.output/'features'))
    finally:os.chdir(cwd)
    path=args.output/'features.pickle'
    features,assemble=load_hmg_features(path,args.paper_root)
    for start in range(0,len(selected),16):
        hmg_inputs(selected[start:start+16],features,assemble,args.device)
    if source_inventory(args.paper_root)!=source or code_identity()!=code:raise RuntimeError('Source changed during preprocessing')
    summary={'generated_with':'original save_smiles_dicts, no upstream source patches',
             'data_sha256':sha(args.data),'feature_cache_sha256':sha(path),'paper_source_sha256':source,
             'module_source_sha256':code,'molecules':len(selected),'rows':{s:[m['row'] for m in rows] for s,rows in splits.items()},
             'deepchem_version':dc.__version__,'rdkit_version':rdBase.rdkitVersion,
             'original_3d_and_normalized_descriptors':True,'descriptor_dim':200,
             'seed':args.seed,'exact_conformer_replay_guaranteed':False,
             'conformers':cfg.numConfs,'max_mmff_iterations':cfg.MMFF_maxIters,
             'extraction_flags':{'BRICS':True,'Murcko':True,'SMARTS':True},
             'smoke':bool(args.smoke_per_split or data['smoke']),'all_requested_rows_validated':True}
    save_json(args.output/'manifest.json',summary);print({k:v for k,v in summary.items() if 'sha256' not in k and k!='rows'})

if __name__=='__main__':main()
