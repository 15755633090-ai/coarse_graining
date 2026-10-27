"""Explicit v2 data/optimizer contract; no historical checkpoint reuse."""
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

VERSION = 'hmg-paper-informed-v2'
PAPER_DEFAULTS = dict(optimizer='Adam', lr=1e-3, batch_size=64, patience=50,
                      seeds=[2020, 2021, 2022, 2023, 2024],aux_weight=.15,contrastive_temperature=.30)
UNRESOLVED = ['exact author scaffold row IDs/split implementation',
              'regression target scaling (not stated in inspected article/supplement)',
              'regression LR scheduler (public classification reference uses Noam)',
              'regression maximum epochs (public classification reference has 100)',
              'author single-granularity ablation/auxiliary details']
SETTING_EVIDENCE = {
    'optimizer_lr_batch_patience': {'status':'paper_explicit','value':['Adam',.001,64,50]},
    'regression_prediction_loss': {'status':'paper_explicit','value':'MSE'},
    'regression_alignment': {'status':'supplement_section_5_explicit','value':{'alpha':.15,'tau':.30}},
    'target_scaling': {'status':'not_specified_in_inspected_public_materials','formal_default':None},
    'scheduler': {'status':'classification_reference_only','reference':'Noam; warmup=2; init/peak/final=1e-4/1e-3/1e-4','formal_default':None},
    'max_epochs': {'status':'classification_reference_only','reference':100,'formal_default':None},
    'split_rows': {'status':'author_exact_rows_unavailable','generated_v2_splits_are_author_splits':False},
    'source_commit': '6ad69261eb1af36c8c0b96594d79a7639b969223',
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def descriptor_names(normalization_source):
    """Recover the same ordered 200-name schema, not the CDF transformation."""
    tree=ast.parse(Path(normalization_source).read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DescriptorsNormalizationParameters')
    assignment=next(n for n in cls.body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='desc_norm_params' for t in n.targets))
    available=dict(Descriptors.descList)
    names=sorted(ast.literal_eval(k) for k in assignment.value.keys if ast.literal_eval(k) in available)
    if len(names)!=200:raise ValueError(f'Expected verified 200 descriptor names, got {len(names)}')
    return names


def raw_descriptors(smiles, names):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid SMILES')
    functions=dict(Descriptors.descList)
    values=[]
    for name in names:
        try:
            value=functions[name](mol,avg=True) if name=='Ipc' else functions[name](mol)
            values.append(float(value))
        except Exception:
            values.append(float('nan'))
    return np.asarray(values,dtype=np.float64)


class DescriptorScaler:
    """Train-only median imputation followed by train-only population z-score."""
    def fit(self, raw, train_positions):
        train=np.asarray(raw,dtype=np.float64)[np.asarray(train_positions,dtype=np.int64)].copy()
        train[~np.isfinite(train)]=np.nan
        if np.isnan(train).all(axis=0).any():raise ValueError('Descriptor has no finite training observations')
        self.median=np.nanmedian(train,axis=0)
        train=np.where(np.isfinite(train),train,self.median)
        self.mean=train.mean(axis=0);std=train.std(axis=0)
        self.constant=std<=1e-12
        self.scale=np.where(self.constant,1.,std)
        self.fit_positions=list(map(int,train_positions))
        return self

    def transform(self, raw):
        raw=np.asarray(raw,dtype=np.float64)
        values=(np.where(np.isfinite(raw),raw,self.median)-self.mean)/self.scale
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite standardized descriptors')
        result=values.astype(np.float32)
        if not np.isfinite(result).all():raise FloatingPointError('Descriptor float32 overflow')
        return result

    def state_dict(self):
        import torch
        return dict(version=VERSION,median=torch.from_numpy(self.median.copy()),
                    mean=torch.from_numpy(self.mean.copy()),scale=torch.from_numpy(self.scale.copy()),
                    constant=torch.from_numpy(self.constant.copy()),fit_positions=self.fit_positions)

    @classmethod
    def from_state_dict(cls,state):
        if state.get('version')!=VERSION:raise ValueError('Legacy descriptor statistics rejected')
        obj=cls()
        for key in ('median','mean','scale','constant'):setattr(obj,key,state[key].cpu().numpy().copy())
        obj.fit_positions=state['fit_positions']
        return obj


def scaffold_split(molecules, seed):
    """Documented strict scaffold grouping, NOT claimed as author's exact split."""
    groups=defaultdict(list)
    for i,row in enumerate(molecules):
        mol=Chem.MolFromSmiles(row['smiles'])
        if mol is None:raise ValueError(f'Invalid molecule row {row["row"]}')
        groups[MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)].append(i)
    items=sorted(groups.items())
    rng=np.random.RandomState(seed);rng.shuffle(items)
    # Oversize groups go first; remaining groups keep the seeded order.
    threshold=len(molecules)*.05
    items=[x for x in items if len(x[1])>threshold]+[x for x in items if len(x[1])<=threshold]
    splits={'train':[],'valid':[],'test':[]};scaffolds={s:set() for s in splits}
    limits={'train':int(.8*len(molecules)),'valid':int(.1*len(molecules))}
    for scaffold,positions in items:
        split=next((s for s in ('train','valid') if len(splits[s])+len(positions)<=limits[s]),'test')
        splits[split].extend(positions);scaffolds[split].add(scaffold)
    for positions in splits.values():positions.sort()
    if any(not v for v in splits.values()):raise ValueError('Empty split')
    if sorted(i for values in splits.values() for i in values)!=list(range(len(molecules))):
        raise AssertionError('Split coverage/uniqueness failure')
    for a,b in (('train','valid'),('train','test'),('valid','test')):
        if scaffolds[a]&scaffolds[b]:raise AssertionError('Cross-split scaffold overlap')
    return splits


def build_optimizer(model):
    import torch
    return torch.optim.Adam(model.parameters(),lr=PAPER_DEFAULTS['lr'],weight_decay=1e-5)


def apply_regression_settings(cfg):
    """SI section 5: alpha=.15, tau=.30 for ALL regression datasets."""
    cfg.ct_loss_coff=PAPER_DEFAULTS['aux_weight']
    cfg.temperature=PAPER_DEFAULTS['contrastive_temperature']
    return cfg


def require_v2_checkpoint(checkpoint, data_fingerprint, split_seed):
    if checkpoint.get('protocol_version')!=VERSION:
        raise ValueError('Old protocol/model weights cannot initialize or resume v2')
    if checkpoint.get('prepared_fingerprint')!=data_fingerprint or checkpoint.get('split_seed')!=split_seed:
        raise ValueError('Checkpoint descriptor/split provenance mismatch')


def protocol_template():
    return dict(protocol_version=VERSION,**PAPER_DEFAULTS,
                descriptor_transform='raw RDKit 200 -> train-median impute -> train-z-score',
                initialization='official initialize_weights, then copy shared initial tensors',
                max_epochs=None,target_scaling=None,scheduler=None,gradient_clip=None,
                paper_exact_reproduction=False,unresolved=UNRESOLVED,setting_evidence=SETTING_EVIDENCE,
                historical_checkpoint_reuse=False,formal_training_paused=True)
