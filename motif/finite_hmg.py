"""Train-only median imputation of undefined original HMG descriptors."""
import argparse
import pickle
from pathlib import Path
import numpy as np
from rdkit import Chem
from .data import load_dataset,save_json,sha
from .paper_data import load_hmg_features

def impute_descriptors(records,train_keys):
    matrix=np.stack([records[k] for k in sorted(set(train_keys))]).astype(np.float64)
    matrix[~np.isfinite(matrix)]=np.nan
    if np.isnan(matrix).all(0).any():raise ValueError('Descriptor column has no finite training values')
    medians=np.nanmedian(matrix,axis=0);changes=[]
    for key,value in records.items():
        bad=~np.isfinite(value)
        if bad.any():
            clean=value.copy();clean[bad]=medians[bad];records[key]=clean
            changes.append({'smiles':key,'columns':np.flatnonzero(bad).tolist(),
                            'replacement_values':clean[bad].tolist()})
    return medians,changes

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','paper-root','feature-cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    data=load_dataset(args.data);features,_=load_hmg_features(args.feature_cache,args.paper_root)
    def key(row):
        return row['smiles'] if row['smiles'] in features['smiles_to_descriptor'] else Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']),isomericSmiles=True)
    train=[key(row) for row in data['molecules'] if row['split']=='train']
    medians,changes=impute_descriptors(features['smiles_to_descriptor'],train)
    for name,records in features.items():
        for value in records.values():
            if isinstance(value,np.ndarray) and not np.isfinite(value).all():raise ValueError(f'Non-finite values remain in {name}')
    path=args.output/'features.pickle'
    with path.open('wb') as stream:pickle.dump(features,stream)
    affected=[{'row':row['row'],'split':row['split']} for row in data['molecules'] if key(row) in {c['smiles'] for c in changes}]
    report={'policy':'nonfinite descriptors only; per-column median of unique training molecules',
            'data_sha256':sha(args.data),'input_cache_sha256':sha(args.feature_cache),'feature_cache_sha256':sha(path),
            'training_molecule_count':len(set(train)),'training_medians':medians.tolist(),
            'changes':changes,'affected_rows':affected,'no_molecules_dropped':True,
            'other_features_unchanged':True,'same_cache_required_for_both_arms':True}
    save_json(args.output/'manifest.json',report);print({'affected_rows':affected,'changes':changes},flush=True)

if __name__=='__main__':main()
