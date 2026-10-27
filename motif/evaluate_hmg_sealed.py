"""Evaluate only previously sealed checkpoints; never train or select models."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .data import save_json
from .hmg_protocol_v2 import DescriptorScaler,digest,apply_regression_settings,require_v2_checkpoint
from .hmg_design_models import IsolatedMother,make_model
from .hmg_study_controls import DESIGN_VERSION
from .paper_data import load_paper,load_hmg_features,hmg_inputs,source_inventory
from .train_hmg_design import read,batches,table,verify_official

def verify_seal(seal):
    payload=dict(seal);signature=payload.pop('seal_sha256')
    if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=signature:
        raise ValueError('Seal changed')
    if not seal.get('sealed') or seal['design_version']!=DESIGN_VERSION:raise ValueError('Unsealed panel')
    if not seal['cases']:raise ValueError('Empty panel')

def restore_model(cls,cfg,protocol,saved,folder,device):
    seed=saved['split_seed']
    utils=importlib.import_module('other_utils');utils.seed_everything(seed)
    full=cls(cfg.atom_dim_in,cfg.bond_dim_in,1);utils.initialize_weights(full)
    mother=IsolatedMother(full) if protocol['study']=='isolated' else None
    width=protocol['head_width']
    if protocol['capacity_target']:width=read(folder/'capacity_match.json')['width']
    model=make_model(full,saved['arm'],protocol['study'],width,mother,seed+1234).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    return model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seal',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paper-root',type=Path,default=Path('research_code_audit/HMG-MRL'))
    p.add_argument('--feature-cache',type=Path,default=Path('motif/runs/hmg_features_finite_20261016/features.pickle'))
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(2)
    seal=read(a.seal);verify_seal(seal)
    if a.output.exists():raise ValueError('Refusing to overwrite an evaluation directory')
    fingerprint=digest(a.prepared/'manifest.json')
    if fingerprint!=seal['prepared_fingerprint']:raise ValueError('Prepared data changed')
    manifest=read(a.prepared/'manifest.json')
    if digest(a.prepared/'raw_descriptors.pt')!=manifest['raw_descriptors_sha256']:raise ValueError('Descriptor matrix changed')
    # Validate every artifact BEFORE any test forward or output creation.
    records=[]
    for case in seal['cases']:
        run=Path(case['run_directory']);protocol=read(run/'protocol.json')
        if digest(run/'protocol.json')!=case['protocol_sha256'] or protocol.get('smoke_only'):
            raise ValueError('Changed protocol or smoke result')
        if protocol['paper_source_sha256']!=source_inventory(a.paper_root.resolve()):raise ValueError('Paper source changed')
        if protocol['feature_cache_sha256']!=digest(a.feature_cache):raise ValueError('Feature cache changed')
        for name,expected in protocol['implementation_sha256'].items():
            if digest(Path(__file__).with_name(name))!=expected:raise ValueError('Implementation changed: '+name)
        checkpoint=run/f"seed_{case['seed']}"/case['case_id'].split('/',1)[1]/'best.pt'
        if digest(checkpoint)!=case['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require_v2_checkpoint(saved,fingerprint,case['seed'])
        if saved['case_id']!=case['case_id'] or saved['epoch']!=case['selected_epoch']:raise ValueError('Wrong selected checkpoint')
        records.append((case,protocol,checkpoint))
    verify_official(a.paper_root.resolve(),Path('motif/audits/hmg_reliability_20261026/official_tree.json'))
    cls,_,cfg=load_paper(a.paper_root.resolve(),'HMG-MRL',a.device);apply_regression_settings(cfg)
    features,assemble=load_hmg_features(a.feature_cache,a.paper_root.resolve())
    data=torch.load(a.prepared/'raw_descriptors.pt',weights_only=True);molecules=data['molecules']
    sample=hmg_inputs(molecules[:2],features,assemble,a.device)[0]
    cfg.atom_dim_in=sample['x_atoms'].shape[-1];cfg.bond_dim_in=sample['x_bonds'].shape[-1]
    utils=importlib.import_module('other_utils');a.output.mkdir(parents=True)
    save_json(a.output/'seal.json',seal);results=[]
    for case,protocol,path in records:
        seed=case['seed'];saved=torch.load(path,map_location='cpu',weights_only=True)
        split=read(a.prepared/f'seed_{seed}_split.json')
        if split!=next(s for s in manifest['splits'] if s['seed']==seed):raise ValueError('Split changed')
        scaler=DescriptorScaler().fit(data['raw'].numpy(),split['positions']['train'])
        standardized=scaler.transform(data['raw'].numpy())
        model=restore_model(cls,cfg,protocol,saved,path.parent,a.device)
        positions=split['positions']['test'];rows=[molecules[i] for i in positions];predictions=[]
        offset=0
        with torch.no_grad():
            for group in batches(rows,64):
                inputs=hmg_inputs(group,features,assemble,a.device)[0]
                inputs['molecule_descriptor']=torch.as_tensor(standardized[positions[offset:offset+len(group)]],device=a.device)
                predictions.extend((model(**inputs)[1].flatten()*saved['std']+saved['mean']).cpu().tolist());offset+=len(group)
        values=[dict(row=r['row'],target=r['label'],prediction=v) for r,v in zip(rows,predictions)]
        rmse=math.sqrt(float(np.mean([(r['target']-r['prediction'])**2 for r in values])))
        if not math.isfinite(rmse):raise FloatingPointError('Nonfinite test prediction')
        folder=a.output/f'seed_{seed}'/case['case_id'];folder.mkdir(parents=True)
        table(folder/'test_predictions.csv',values)
        results.append(dict(seed=seed,case_id=case['case_id'],test_rmse=rmse,checkpoint_sha256=case['checkpoint_sha256']))
        save_json(a.output/'results_partial.json',results);del model;torch.cuda.empty_cache()
    table(a.output/'test_runs.csv',results)
    save_json(a.output/'completion.json',dict(passed=True,seal_sha256=seal['seal_sha256'],cases=len(results),training_performed=False))

if __name__=='__main__':main()
