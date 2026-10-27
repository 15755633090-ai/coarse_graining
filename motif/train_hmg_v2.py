"""Paper-informed v2; smoke is default, formal runs require explicit open choices."""
import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from .data import save_json
from .hmg_fusion_ablation import ARMS,variant
from .hmg_protocol_v2 import (VERSION,PAPER_DEFAULTS,DescriptorScaler,build_optimizer,
                             digest,protocol_template,require_v2_checkpoint,apply_regression_settings)
from .paper_data import load_paper,load_hmg_features,hmg_inputs,source_inventory


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_save(value,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def batches(rows,size,seed=None):
    order=list(range(len(rows))) if seed is None else torch.randperm(len(rows),generator=torch.Generator().manual_seed(seed)).tolist()
    groups=[order[i:i+size] for i in range(0,len(rows),size)]
    if len(groups)>1 and len(groups[-1])==1:groups[-2].extend(groups.pop())
    return [[rows[i] for i in group] for group in groups]


def table(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_official(root,metadata):
    official=read(metadata)
    if official['commit']!='6ad69261eb1af36c8c0b96594d79a7639b969223':raise ValueError('Unexpected official version')
    for entry in official['tree']['tree']:
        if entry['type']!='blob' or not entry['path'].endswith('.py'):continue
        raw=(root/entry['path']).read_bytes()
        candidates=(raw,raw.replace(b'\r\n',b'\n'))
        actual=[hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest() for value in candidates]
        if entry['sha'] not in actual:raise ValueError('Official source mismatch: '+entry['path'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paper-root',type=Path,default=Path('research_code_audit/HMG-MRL'))
    p.add_argument('--official-tree',type=Path,default=Path('motif/audits/hmg_reliability_20261026/official_tree.json'))
    p.add_argument('--feature-cache',type=Path,default=Path('motif/runs/hmg_features_finite_20261016/features.pickle'))
    p.add_argument('--stage',choices=['smoke','baseline','fusion'],default='smoke')
    p.add_argument('--baseline',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--max-epochs',type=int)
    p.add_argument('--target-scaling',choices=['raw','train-zscore'])
    p.add_argument('--scheduler',choices=['constant'])
    p.add_argument('--gradient-clip',type=float)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(2)
    smoke=a.stage=='smoke'
    if not smoke and (a.max_epochs is None or a.target_scaling is None or a.scheduler is None):
        p.error('Explicit --max-epochs, --target-scaling and --scheduler are required: these regression details remain unverified')
    if a.max_epochs is not None and a.max_epochs<1:p.error('max epochs must be positive')
    if a.stage=='fusion' and a.baseline is None:p.error('Run and inspect a new v2 Full baseline before fusion ablations')
    a.prepared=a.prepared.resolve();out=a.output.resolve();paper=a.paper_root.resolve()
    prepared=read(a.prepared/'manifest.json')
    if prepared['protocol_version']!=VERSION:raise ValueError('v1 prepared inputs rejected')
    if digest(a.prepared/'raw_descriptors.pt')!=prepared['raw_descriptors_sha256']:raise ValueError('Descriptor hash mismatch')
    verify_official(paper,a.official_tree)
    data=torch.load(a.prepared/'raw_descriptors.pt',map_location='cpu',weights_only=True)
    molecules=data['molecules'];raw=data['raw'].numpy();row_position={r['row']:i for i,r in enumerate(molecules)}
    if len(row_position)!=len(molecules):raise ValueError('Duplicate row IDs')
    fingerprint=digest(a.prepared/'manifest.json')
    try:
        require_v2_checkpoint({'model':{}},fingerprint,2020)
    except ValueError:
        legacy_rejected=True
    else:
        raise AssertionError('Legacy checkpoint guard failed')
    cls,_,cfg=load_paper(paper,'HMG-MRL',a.device)
    apply_regression_settings(cfg)
    official_utils=importlib.import_module('other_utils')
    features,assemble=load_hmg_features(a.feature_cache,paper)
    sample,_,_=hmg_inputs(molecules[:2],features,assemble,a.device)
    cfg.atom_dim_in=sample['x_atoms'].shape[-1];cfg.bond_dim_in=sample['x_bonds'].shape[-1]
    seeds=[2020] if smoke else PAPER_DEFAULTS['seeds']
    arms=ARMS if a.stage in ('smoke','fusion') else ('ASD',)
    epochs=3 if smoke else a.max_epochs
    target_scaling='raw' if smoke else a.target_scaling
    protocol=protocol_template()
    protocol.update(stage=a.stage,smoke_only=smoke,seeds=seeds,arms=arms,max_epochs=epochs,
        target_scaling=target_scaling,scheduler='constant',gradient_clip=a.gradient_clip,
        prepared_fingerprint=fingerprint,raw_descriptors_sha256=prepared['raw_descriptors_sha256'],
        feature_cache_sha256=digest(a.feature_cache),paper_source_sha256=source_inventory(paper),
        official_commit='6ad69261eb1af36c8c0b96594d79a7639b969223',
        aux_weight=cfg.ct_loss_coff,contrastive_temperature=cfg.temperature,descriptor_schema=prepared['descriptor_schema'],
        formal_training_paused=smoke,device=a.device,
        effective_capacity_matched=False,ablation='fusion-subset only; all auxiliary paths kept',
        historical_test_seen=True,author_exact_split_rows=False,
        implementation_sha256={name:digest(Path(__file__).with_name(name)) for name in
            ('train_hmg_v2.py','hmg_protocol_v2.py','hmg_fusion_ablation.py','paper_data.py')})
    baseline_protocol=None
    if a.stage=='fusion':
        baseline_protocol=read(a.baseline/'protocol.json');completion=read(a.baseline/'completion.json')
        if not completion['passed'] or baseline_protocol['stage']!='baseline':raise ValueError('V2 baseline not complete')
        for key in ('protocol_version','prepared_fingerprint','feature_cache_sha256','paper_source_sha256',
                    'lr','batch_size','patience','max_epochs','target_scaling','scheduler','gradient_clip','seeds',
                    'implementation_sha256','aux_weight','contrastive_temperature'):
            if baseline_protocol[key]!=protocol[key]:raise ValueError('New Full baseline protocol mismatch: '+key)
    out.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        if read(out/'protocol.json')!=json.loads(json.dumps(protocol)):raise ValueError('Resume protocol mismatch')
    else:save_json(out/'protocol.json',protocol)
    for name in protocol['implementation_sha256']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    results=read(out/'results_partial.json') if a.resume and (out/'results_partial.json').exists() else []
    acceptance=[]
    def status(**state):save_json(out/'status.json',state)
    for seed in seeds:
        split=read(a.prepared/f'seed_{seed}_split.json')
        expected_split=next(s for s in prepared['splits'] if s['seed']==seed)
        if split!=expected_split:raise ValueError('Split file differs from prepared manifest')
        rows={s:[molecules[i] for i in positions] for s,positions in split['positions'].items()}
        scaler_state=torch.load(a.prepared/f'seed_{seed}_descriptor_scaler.pt',map_location='cpu',weights_only=True)
        scaler=DescriptorScaler.from_state_dict(scaler_state)
        if scaler.fit_positions!=split['positions']['train']:raise ValueError('Scaler fit split mismatch')
        refitted=DescriptorScaler().fit(raw,split['positions']['train'])
        for name in ('median','mean','scale','constant'):
            if not np.array_equal(getattr(scaler,name),getattr(refitted,name)):
                raise ValueError('Stored descriptor statistics differ from train-only refit: '+name)
        standardized=scaler.transform(raw)
        labels=np.array([r['label'] for r in rows['train']])
        mean,std=(float(labels.mean()),max(float(labels.std()),1e-6)) if target_scaling=='train-zscore' else (0.,1.)
        if smoke:rows={'train':rows['train'][:64],'valid':rows['valid'][:16]}
        def inputs(group):
            values=hmg_inputs(group,features,assemble,a.device)[0]
            values['molecule_descriptor']=torch.as_tensor(standardized[[row_position[r['row']] for r in group]],device=a.device)
            assert values['molecule_descriptor'].shape==(len(group),200)
            return values
        def evaluate(model,group_rows):
            model.eval();preds=[]
            with torch.no_grad():
                for group in batches(group_rows,64):preds.extend((model(**inputs(group))[1].flatten()*std+mean).cpu().tolist())
            records=[dict(row=r['row'],target=r['label'],prediction=v) for r,v in zip(group_rows,preds)]
            score=math.sqrt(float(np.mean([(r['prediction']-r['target'])**2 for r in records])))
            if not math.isfinite(score):raise FloatingPointError('Nonfinite validation/test score')
            return score,records
        official_utils.seed_everything(seed)
        mother=cls(cfg.atom_dim_in,cfg.bond_dim_in,1)
        official_utils.initialize_weights(mother)
        initial={k:v.detach().clone() for k,v in mother.state_dict().items()}
        seed_dir=out/f'seed_{seed}';seed_dir.mkdir(exist_ok=a.resume)
        atomic_save(dict(protocol_version=VERSION,prepared_fingerprint=fingerprint,split_seed=seed,
                         model=initial,official_initializer=True),seed_dir/'initial.pt')
        for arm in (('ASD',)+tuple(x for x in arms if x!='ASD')):
            if any(r['seed']==seed and r['arm']==arm for r in results):continue
            folder=seed_dir/arm;folder.mkdir(exist_ok=a.resume)
            model=variant(mother,arm,folder/'forward.py').to(a.device)
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
            optimizer=build_optimizer(model)
            if not isinstance(optimizer,torch.optim.Adam) or isinstance(optimizer,torch.optim.AdamW):raise AssertionError('Wrong optimizer')
            if smoke and arm=='ASD':
                mother.to(a.device).eval();model.eval()
                with torch.no_grad():
                    kw=inputs(rows['train'][:4]);reference=mother(**kw);actual=model(**kw)
                    torch.testing.assert_close(actual[1],reference[1],rtol=0,atol=0)
                    torch.testing.assert_close(actual[2],reference[2],rtol=0,atol=0)
                mother.cpu()
            if a.stage=='fusion' and arm=='ASD':
                source=a.baseline/f'seed_{seed}'/'ASD'
                checkpoint=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
                require_v2_checkpoint(checkpoint,fingerprint,seed)
                model.load_state_dict(checkpoint['model'],strict=True)
                valid,preds=evaluate(model,rows['valid'])
                if abs(valid-checkpoint['valid_rmse'])>1e-6:raise ValueError('New baseline reload mismatch')
                import shutil
                for name in ('best.pt','history.json','valid_predictions.csv','test_predictions.csv','result.json'):
                    shutil.copyfile(source/name,folder/name)
                result=read(folder/'result.json');result['reused_v2_baseline']=True
                results.append(result);save_json(out/'results_partial.json',results)
                del model,optimizer;continue
            history=[];best=float('inf');best_epoch=0;start_epoch=1
            if a.resume and (folder/'latest.pt').exists():
                latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
                require_v2_checkpoint(latest,fingerprint,seed)
                model.load_state_dict(latest['model'],strict=True);optimizer.load_state_dict(latest['optimizer'])
                history=latest['history'];best=latest['best'];best_epoch=latest['best_epoch'];start_epoch=latest['epoch']+1
            for epoch in range(start_epoch,epochs+1):
                if epoch-1-best_epoch>=50:break
                official_utils.seed_everything(seed+4000+epoch);model.train()
                for step,group in enumerate(batches(rows['train'],64,seed+epoch)):
                    prediction=model(**inputs(group))
                    targets=torch.tensor([(r['label']-mean)/std for r in group],device=a.device)
                    loss=(prediction[1].flatten()-targets).square().mean()+cfg.ct_loss_coff*prediction[2]
                    if not torch.isfinite(loss):raise FloatingPointError((seed,arm,epoch,step))
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),a.gradient_clip if a.gradient_clip else float('inf'),error_if_nonfinite=True)
                    optimizer.step()
                    status(phase='smoke_training' if smoke else 'training',seed=seed,arm=arm,epoch=epoch,
                           batch=step,completed=len(results),total=len(seeds)*len(arms))
                valid,preds=evaluate(model,rows['valid'])
                history.append(dict(epoch=epoch,valid_rmse=valid,gradient_norm=float(norm)))
                metadata=dict(protocol_version=VERSION,prepared_fingerprint=fingerprint,split_seed=seed,
                              arm=arm,epoch=epoch,mean=mean,std=std,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                if valid<best:
                    best,best_epoch=valid,epoch
                    atomic_save(dict(**metadata,valid_rmse=valid),folder/'best.pt')
                atomic_save(dict(**metadata,optimizer=optimizer.state_dict(),history=history,
                                 best=best,best_epoch=best_epoch),folder/'latest.pt')
                save_json(folder/'history.json',history)
                print(dict(stage=a.stage,seed=seed,arm=arm,epoch=epoch,valid_rmse=valid),flush=True)
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            require_v2_checkpoint(saved,fingerprint,seed);model.load_state_dict(saved['model'],strict=True)
            valid,preds=evaluate(model,rows['valid']);assert abs(valid-best)<1e-6
            table(folder/'valid_predictions.csv',preds)
            result=dict(seed=seed,arm=arm,best_epoch=best_epoch,epochs_run=history[-1]['epoch'],valid_rmse=valid,
                        protocol_version=VERSION,reused_v2_baseline=False)
            if not smoke:
                test,testpred=evaluate(model,rows['test']);table(folder/'test_predictions.csv',testpred);result['test_rmse']=test
            else:
                with torch.no_grad():
                    for parameter in model.parameters():parameter.add_(1)
                model.load_state_dict(saved['model'],strict=True)
                score,reloaded=evaluate(model,rows['valid']);assert reloaded==preds and score==valid
                acceptance.append(dict(arm=arm,official_initialized=True,shared_initial_exact=True,
                    optimizer='Adam',lr=optimizer.param_groups[0]['lr'],batch_size=64,strict_reload=True,
                    aux_weight=cfg.ct_loss_coff,contrastive_temperature=cfg.temperature))
            save_json(folder/'result.json',result);results.append(result);save_json(out/'results_partial.json',results)
            del model,optimizer;torch.cuda.empty_cache()
    table(out/'runs.csv',results)
    if not smoke:
        summary=[]
        for arm in arms:
            values=[r['test_rmse'] for r in results if r['arm']==arm]
            summary.append(dict(arm=arm,n=len(values),test_rmse_mean=statistics.mean(values),
                                test_rmse_sd=statistics.stdev(values) if len(values)>1 else 0.))
        table(out/'summary.csv',summary)
        if a.stage=='fusion':
            lookup={(r['seed'],r['arm']):r['test_rmse'] for r in results};paired=[]
            for combo in ARMS[3:]:
                for control in list(combo)+(list(ARMS[3:6]) if combo=='ASD' else []):
                    for seed in seeds:
                        paired.append(dict(combo=combo,control=control,seed=seed,
                                           delta_rmse=lookup[seed,combo]-lookup[seed,control]))
            table(out/'paired_deltas.csv',paired)
    if smoke:save_json(out/'acceptance.json',dict(passed=True,protocol_version=VERSION,test_forward=False,
        formal_training_resumed=False,arms=acceptance,prepared_fingerprint=fingerprint,
        descriptor_train_only=True,legacy_checkpoint_rejected=legacy_rejected,ASD_original_forward_exact=True,
        regression_temperature=cfg.temperature,alignment_weight=cfg.ct_loss_coff))
    save_json(out/'completion.json',dict(passed=True,stage=a.stage,results=len(results),test_forward=not smoke))
    status(phase='smoke_complete' if smoke else 'complete',completed=len(results),total=len(results),formal_training_resumed=False if smoke else True)


if __name__=='__main__':main()
