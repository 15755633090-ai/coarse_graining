import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from rdkit import Chem
import rdkit,chemprop
from chemprop.featurizers import V1RDKit2DNormalizedFeaturizer
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();out=a.output_dir.resolve();f=V1RDKit2DNormalizedFeaturizer()
for name in ['esol','freesolv']:
    values={}
    for split in ['train','valid']:
        with (out/'data'/f'{name}_{split}.csv').open(encoding='utf-8') as stream:records=list(csv.DictReader(stream))
        x=np.array([f(Chem.MolFromSmiles(r['smiles'])) for r in records],dtype=np.float64);x[np.isnan(x)]=0
        assert x.shape==(len(records),200) and np.isfinite(x).all()
        values[split]=x;values[split+'_y']=np.array([float(r['target']) for r in records])
        values[split+'_id']=np.array([int(r['sample_id']) for r in records])
        values[split+'_groups']=np.array([Chem.MolToSmiles(Chem.MolFromSmiles(r['smiles']),canonical=True) for r in records])
    path=out/f'{name}_normalized_descriptors.npz';np.savez_compressed(path,**values)
    meta={'chemprop':chemprop.__version__,'rdkit':rdkit.__version__,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'descriptor':'official V1RDKit2DNormalizedFeaturizer, same NaN-zero token; no fitted scaling'}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
