"""Read-only audit of actual split, descriptor tensors and saved initial states."""
import ast
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
data=torch.load(ROOT/'motif/data/lipo.pt',map_location='cpu',weights_only=True)
rows={s:[r for r in data['molecules'] if r['split']==s] for s in ('train','valid','test')}
scaffolds={};duplicates={}
for split,rs in rows.items():
    values=[]; canonical=[]
    for r in rs:
        mol=Chem.MolFromSmiles(r['smiles'])
        values.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False))
        canonical.append(Chem.MolToSmiles(mol,isomericSmiles=True))
    scaffolds[split]=values;duplicates[split]=set(canonical)
overlap=[]
for a,b in (('train','valid'),('train','test'),('valid','test')):
    common=set(scaffolds[a])&set(scaffolds[b])
    overlap.append(dict(splits=[a,b],shared_scaffolds=len(common),shared_nonempty_scaffolds=len(common-{''}),
        examples=sorted(common-{''})[:5],identical_smiles_shared=len(duplicates[a]&duplicates[b])))
print('Scaffold comparisons:',overlap,flush=True)

path=ROOT/'motif/runs/hmg_features_finite_20261016/features.pickle'
with path.open('rb') as stream:features=pickle.load(stream)
def key(r):
    raw=r['smiles']
    return raw if raw in features['smiles_to_descriptor'] else Chem.MolToSmiles(Chem.MolFromSmiles(raw),isomericSmiles=True)
matrix=np.stack([features['smiles_to_descriptor'][key(r)] for r in rows['train']]).astype(float)
mu=matrix.mean(0);sd=matrix.std(0)
nonconstant=sd>1e-12
descriptors=dict(shape=list(matrix.shape),all_finite=bool(np.isfinite(matrix).all()),
    train_mean_min=float(mu.min()),train_mean_max=float(mu.max()),train_mean_abs_median=float(np.median(np.abs(mu))),
    train_sd_median_nonconstant=float(np.median(sd[nonconstant])),nonconstant_columns=int(nonconstant.sum()),
    approximately_zero_mean_columns=int((np.abs(mu)<1e-5).sum()),
    approximately_unit_sd_columns=int((np.abs(sd-1)<1e-5).sum()),
    cache_transform='official DeepChem RDKitDescriptors(is_normalized=True), plus one documented validation-row median imputation',
    train_fitted_zscore_in_current_runner=False)
# Verify the actual adapter forwards cache descriptors without a hidden z-score.
source=ROOT/'motif/runs/scale_pairs_3seeds_20261018/upstream/hmg/preprocess/get_atom_bond_frag_info.py'
tree=ast.parse(source.read_text(encoding='utf-8-sig'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_smiles_array')
env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
sample=rows['train'][:2];arrays=env['get_smiles_array']([key(r) for r in sample],features)
descriptors['adapter_descriptor_array_matches_cache_exact']=bool(np.array_equal(arrays[11],np.stack([features['smiles_to_descriptor'][key(r)] for r in sample])))

reference=ROOT/'motif/runs/scale_pairs_3seeds_20261018/hmg_seed0/initial_states.pt'
initial=torch.load(reference,map_location='cpu',weights_only=True)['full']
norm_keys=[k for k in initial if 'motif_encoder' in k and 'norm' in k and (k.endswith('weight') or k.endswith('bias'))]
norms=[dict(name=k,unique_values=initial[k].unique().tolist()[:12],mean=float(initial[k].float().mean())) for k in norm_keys]
linears=[(k,v) for k,v in initial.items() if v.ndim==2 and k.endswith('weight')]
report=dict(split_counts={s:len(rs) for s,rs in rows.items()},scaffold_definition='RDKit Bemis-Murcko, includeChirality=False',
    scaffold_overlap=overlap,descriptors=descriptors,initialization=dict(
        source=str(reference),motif_transformer_norm_tensors=norms,
        all_2d_weights_are_default_uniform_bounded=all(float(v.abs().max())<=v.shape[1]**-.5+1e-6 for _,v in linears),
        runner_calls_official_initialize_weights=False),
    limitations='Descriptor statistics establish a protocol discrepancy, not its causal performance impact. Scaffold overlap does not establish the exact historical split generator.')
(OUT/'input_initialization_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(descriptors=descriptors,norms=norms),ensure_ascii=False),flush=True)
