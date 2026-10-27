"""Prepare new raw descriptors and five strict scaffold splits; no model training."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from .data import load_dataset,save_json
from .hmg_protocol_v2 import (VERSION,PAPER_DEFAULTS,DescriptorScaler,descriptor_names,
                             digest,raw_descriptors,scaffold_split,protocol_template)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('motif/data/lipo.pt'))
    p.add_argument('--normalization-source',type=Path,default=Path('motif/vendor/deepchem_runtime/deepchem/utils/rdkit_utils.py'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=load_dataset(a.data);molecules=data['molecules'];names=descriptor_names(a.normalization_source)
    save_json(out/'progress.json',dict(phase='raw_descriptors',completed=0,total=len(molecules)))
    save_json(out/'descriptor_requests.json',dict(names=names,molecules=[dict(row=r['row'],smiles=r['smiles']) for r in molecules]))
    # RDKit descriptor extensions and Torch load different OpenMP libraries on
    # this Windows runtime. Keep raw descriptor calculation in a Torch-free child.
    subprocess.run([sys.executable,str(Path(__file__).with_name('raw_hmg_descriptors_v2.py')),
                    '--requests',str(out/'descriptor_requests.json'),'--output',str(out/'raw_descriptors.npz')],check=True)
    with np.load(out/'raw_descriptors.npz',allow_pickle=False) as result:raw=result['raw'].copy()
    torch.save(dict(protocol_version=VERSION,raw=torch.from_numpy(raw),names=names,
                    molecules=[dict(row=r['row'],smiles=r['smiles'],label=r['label']) for r in molecules]),out/'raw_descriptors.pt')
    splits=[]
    for seed in PAPER_DEFAULTS['seeds']:
        indices=scaffold_split(molecules,seed)
        scaler=DescriptorScaler().fit(raw,indices['train'])
        standardized=scaler.transform(raw)
        train=standardized[indices['train']]
        variable=~scaler.constant
        mean_error=float(np.abs(train[:,variable].mean(0,dtype=np.float64)).max())
        std_error=float(np.abs(train[:,variable].std(0,dtype=np.float64)-1).max())
        assert mean_error<1e-5 and std_error<1e-5
        torch.save(scaler.state_dict(),out/f'seed_{seed}_descriptor_scaler.pt')
        state=dict(seed=seed,positions=indices,rows={s:[molecules[i]['row'] for i in v] for s,v in indices.items()},
            counts={s:len(v) for s,v in indices.items()},scaffold_overlap=0,
            descriptor_train_mean_max_error=mean_error,descriptor_train_std_max_error=std_error,
            constant_descriptor_columns=int(scaler.constant.sum()),
            definition='strict Murcko includeChirality=False; seeded grouped 80/10/10 allocation; not author exact rows')
        save_json(out/f'seed_{seed}_split.json',state);splits.append(state)
    manifest=dict(protocol_version=VERSION,data_sha256=digest(a.data),descriptor_schema=names,
        schema_source=str(a.normalization_source.resolve()),schema_source_sha256=digest(a.normalization_source),
        rdkit_version=rdBase.rdkitVersion,raw_descriptors_sha256=digest(out/'raw_descriptors.pt'),
        ipc_avg=True,cdf_applied=False,train_only_imputation_and_standardization=True,
        nonfinite_raw_entries=int((~np.isfinite(raw)).sum()),splits=splits,
        preprocessing_code_sha256={name:digest(Path(__file__).with_name(name)) for name in ('prepare_hmg_v2.py','hmg_protocol_v2.py','raw_hmg_descriptors_v2.py')},
        paper_exact_splits=False,model_training=False)
    save_json(out/'manifest.json',manifest);save_json(out/'protocol_template.json',protocol_template())
    save_json(out/'progress.json',dict(phase='complete',molecules=len(molecules),seeds=PAPER_DEFAULTS['seeds'],model_training=False))
    print(dict(phase='complete',molecules=len(molecules),split_counts=[s['counts'] for s in splits]),flush=True)


if __name__=='__main__':main()
