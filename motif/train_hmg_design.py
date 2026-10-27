"""Controlled HMG studies; validation first, no test evaluation in this runner."""
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
from .hmg_design_models import (STUDIES,IsolatedMother,make_model,auxiliary_weight,
                                objective,capacity_audit,match_capacity)
from .hmg_study_controls import (DESIGN_VERSION,standardized_output_initialization,convergence_record)


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
    p.add_argument('--stage',choices=['smoke','baseline','fusion','controls'],default='smoke')
    p.add_argument('--study',choices=STUDIES,default='native')
    p.add_argument('--capacity-target',choices=['AS','AD','SD','ASD'])
    p.add_argument('--head-width',type=int,default=128)
    p.add_argument('--budget-ladder',type=int,nargs='+')
    p.add_argument('--active-cap',type=int)
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
    if not smoke:
        if not a.budget_ladder or a.budget_ladder!=sorted(set(a.budget_ladder)) or a.budget_ladder[-1]!=a.max_epochs:
            p.error('Predeclare increasing --budget-ladder ending at --max-epochs')
        if a.active_cap is not None and a.active_cap not in a.budget_ladder:p.error('Active cap not in registered ladder')
    if a.stage in ('fusion','controls') and a.baseline is None:p.error('Run and inspect a new same-study Full baseline first')
    if a.stage=='controls' and (not a.capacity_target or a.study not in ('common-head','common-head-noaux','isolated')):
        p.error('Capacity controls need a common-head/isolated study and --capacity-target')
    if a.capacity_target and a.stage not in ('controls','smoke'):p.error('Capacity target only valid for controls/smoke')
    if a.head_width<1:p.error('head width must be positive')
    if a.gradient_clip is not None and (not math.isfinite(a.gradient_clip) or a.gradient_clip<=0):
        p.error('gradient clip must be finite and positive')
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
    arms=ARMS if a.stage in ('smoke','fusion') else tuple(a.capacity_target) if a.stage=='controls' else ('ASD',)
    if not smoke and a.study.startswith('single-direct') and a.stage=='fusion':arms=('A','S','D')
    if smoke and a.capacity_target:arms=tuple(a.capacity_target)
    epochs=3 if smoke else (a.active_cap or a.budget_ladder[0])
    ladder=[3] if smoke else a.budget_ladder
    target_scaling='raw' if smoke else a.target_scaling
    alpha=auxiliary_weight(a.study)
    protocol=protocol_template()
    protocol.update(stage=a.stage,smoke_only=smoke,seeds=seeds,arms=arms,max_epochs=ladder[-1],
        design_version=DESIGN_VERSION,study=a.study,capacity_target=a.capacity_target,head_width=a.head_width,
        budget_ladder=ladder,loss_units='raw target MSE',test_policy='deferred until complete panel sealed',
        target_scaling=target_scaling,scheduler='constant',gradient_clip=a.gradient_clip,
        prepared_fingerprint=fingerprint,raw_descriptors_sha256=prepared['raw_descriptors_sha256'],
        feature_cache_sha256=digest(a.feature_cache),paper_source_sha256=source_inventory(paper),
        official_commit='6ad69261eb1af36c8c0b96594d79a7639b969223',
        aux_weight=alpha,contrastive_temperature=cfg.temperature,descriptor_schema=prepared['descriptor_schema'],
        formal_training_paused=smoke,device=a.device,
        effective_capacity_matched=bool(a.capacity_target),ablation='explicit study paths; see study definition',
        historical_test_seen=True,author_exact_split_rows=False,
        implementation_sha256={name:digest(Path(__file__).with_name(name)) for name in
            ('train_hmg_design.py','hmg_protocol_v2.py','hmg_fusion_ablation.py','paper_data.py',
             'hmg_design_models.py','hmg_study_controls.py')})
    baseline_protocol=None
    if a.stage in ('fusion','controls'):
        baseline_protocol=read(a.baseline/'protocol.json');completion=read(a.baseline/'completion.json')
        if not completion['passed'] or baseline_protocol['stage']!='baseline':raise ValueError('V2 baseline not complete')
        baseline_study={'single-direct':'native','single-direct-noaux':'noaux'}.get(a.study,a.study)
        if baseline_protocol.get('study')!=baseline_study:raise ValueError('Wrong study baseline')
        for key in ('protocol_version','prepared_fingerprint','feature_cache_sha256','paper_source_sha256',
                    'lr','batch_size','patience','max_epochs','target_scaling','scheduler','gradient_clip','seeds',
                    'implementation_sha256','aux_weight','contrastive_temperature','design_version','budget_ladder','head_width'):
            if baseline_protocol[key]!=protocol[key]:raise ValueError('New Full baseline protocol mismatch: '+key)
    out.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        if read(out/'protocol.json')!=json.loads(json.dumps(protocol)):raise ValueError('Resume protocol mismatch')
    else:save_json(out/'protocol.json',protocol)
    for name in protocol['implementation_sha256']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    results=read(out/'results_partial.json') if a.resume and (out/'results_partial.json').exists() else []
    acceptance=[]
    def status(**state):save_json(out/'status.json',state)
    def require_checkpoint(checkpoint,seed):
        require_v2_checkpoint(checkpoint,fingerprint,seed)
        if checkpoint.get('design_version')!=DESIGN_VERSION or checkpoint.get('study')!=a.study:
            raise ValueError('Study/objective checkpoint mismatch')
        if checkpoint.get('case_id')!=case_id:raise ValueError('Wrong arm/capacity checkpoint')
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
        isolated_mother=IsolatedMother(mother) if a.study=='isolated' else None
        initial={k:v.detach().clone() for k,v in mother.state_dict().items()}
        seed_dir=out/f'seed_{seed}';seed_dir.mkdir(exist_ok=a.resume)
        atomic_save(dict(protocol_version=VERSION,prepared_fingerprint=fingerprint,split_seed=seed,
                         model=initial,official_initializer=True),seed_dir/'initial.pt')
        for arm in [x for x in ('ASD',*ARMS[:-1]) if x in arms]:
            name=arm if not a.capacity_target else f'{arm}_matched_to_{a.capacity_target}'
            case_id=f'{a.study}/{name}'
            previous=next((r for r in results if r['seed']==seed and r['case_id']==case_id),None)
            if previous:
                if previous['convergence']['budget']>epochs:raise ValueError('Cannot reduce a completed active cap')
                if not previous['convergence']['needs_budget_review'] or previous['convergence']['last_epoch']>=epochs:
                    previous['convergence']=convergence_record(read(seed_dir/name/'history.json'),epochs)
                    save_json(seed_dir/name/'result.json',previous)
                    continue
                results.remove(previous)
            folder=seed_dir/name;folder.mkdir(exist_ok=a.resume)
            model=make_model(mother,arm,a.study,a.head_width,isolated_mother,head_seed=seed+1234).to(a.device)
            probe_rows=rows['train'][:4];probe_inputs=inputs(probe_rows)
            probe_targets=torch.tensor([r['label'] for r in probe_rows],device=a.device)
            match=None
            if a.capacity_target:
                reference_model=make_model(mother,a.capacity_target,a.study,a.head_width,isolated_mother,head_seed=seed+1234).to(a.device)
                budget=capacity_audit(reference_model,probe_inputs,probe_targets,alpha)['objective_connected_parameters']
                del reference_model
                match=match_capacity(model,budget,probe_inputs,probe_targets,alpha,head_seed=seed+1234)
                save_json(folder/'capacity_match.json',match)
            save_json(folder/'capacity_audit.json',capacity_audit(model,probe_inputs,probe_targets,alpha))
            if target_scaling=='train-zscore':standardized_output_initialization(model,mean,std)
            optimizer=build_optimizer(model)
            if not isinstance(optimizer,torch.optim.Adam) or isinstance(optimizer,torch.optim.AdamW):raise AssertionError('Wrong optimizer')
            if smoke and arm=='ASD' and a.study in ('native','noaux','single-direct','single-direct-noaux'):
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
                if checkpoint.get('design_version')!=DESIGN_VERSION:raise ValueError('Legacy baseline objective')
                model.load_state_dict(checkpoint['model'],strict=True)
                valid,preds=evaluate(model,rows['valid'])
                if abs(valid-checkpoint['valid_rmse'])>1e-6:raise ValueError('New baseline reload mismatch')
                import shutil
                for file_name in ('best.pt','history.json','valid_predictions.csv','result.json'):
                    shutil.copyfile(source/file_name,folder/file_name)
                result=read(folder/'result.json');result['reused_v2_baseline']=True
                result['convergence']=convergence_record(read(folder/'history.json'),epochs)
                if result['convergence']['stop_reason']=='incomplete':
                    raise ValueError('Extend the paired Full baseline to the active cap before reuse')
                save_json(folder/'result.json',result)
                results.append(result);save_json(out/'results_partial.json',results)
                del model,optimizer;continue
            history=[];best=float('inf');best_epoch=0;start_epoch=1
            if a.resume and (folder/'latest.pt').exists():
                latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
                require_checkpoint(latest,seed)
                model.load_state_dict(latest['model'],strict=True);optimizer.load_state_dict(latest['optimizer'])
                history=latest['history'];best=latest['best'];best_epoch=latest['best_epoch'];start_epoch=latest['epoch']+1
            for epoch in range(start_epoch,epochs+1):
                if epoch-1-best_epoch>=50:break
                official_utils.seed_everything(seed+4000+epoch);model.train()
                for step,group in enumerate(batches(rows['train'],64,seed+epoch)):
                    prediction=model(**inputs(group))
                    targets=torch.tensor([r['label'] for r in group],device=a.device)
                    loss,_,_=objective(prediction,targets,mean,std,alpha)
                    if not torch.isfinite(loss):raise FloatingPointError((seed,arm,epoch,step))
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),a.gradient_clip if a.gradient_clip else float('inf'),error_if_nonfinite=True)
                    optimizer.step()
                    status(phase='smoke_training' if smoke else 'training',seed=seed,arm=arm,epoch=epoch,
                           batch=step,completed=len(results),total=len(seeds)*len(arms))
                valid,preds=evaluate(model,rows['valid'])
                history.append(dict(epoch=epoch,valid_rmse=valid,gradient_norm=float(norm)))
                metadata=dict(protocol_version=VERSION,prepared_fingerprint=fingerprint,split_seed=seed,
                              design_version=DESIGN_VERSION,study=a.study,case_id=case_id,
                              arm=arm,epoch=epoch,mean=mean,std=std,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                if valid<best:
                    best,best_epoch=valid,epoch
                    atomic_save(dict(**metadata,valid_rmse=valid),folder/'best.pt')
                atomic_save(dict(**metadata,optimizer=optimizer.state_dict(),history=history,
                                 best=best,best_epoch=best_epoch),folder/'latest.pt')
                save_json(folder/'history.json',history)
                print(dict(stage=a.stage,seed=seed,arm=arm,epoch=epoch,valid_rmse=valid),flush=True)
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            require_checkpoint(saved,seed);model.load_state_dict(saved['model'],strict=True)
            valid,preds=evaluate(model,rows['valid']);assert abs(valid-best)<1e-6
            table(folder/'valid_predictions.csv',preds)
            result=dict(seed=seed,arm=arm,best_epoch=best_epoch,epochs_run=history[-1]['epoch'],valid_rmse=valid,
                        protocol_version=VERSION,design_version=DESIGN_VERSION,case_id=case_id,study=a.study,
                        prepared_fingerprint=fingerprint,reused_v2_baseline=False,test_evaluated=False,
                        checkpoint_sha256=digest(folder/'best.pt'),capacity_matched=match is not None,
                        convergence=convergence_record(history,epochs),
                        shared_training_choices={k:protocol[k] for k in ('target_scaling','scheduler','loss_units','lr','batch_size','patience','budget_ladder',
                            'gradient_clip','contrastive_temperature','head_width','feature_cache_sha256','paper_source_sha256','implementation_sha256')})
            if smoke:
                with torch.no_grad():
                    for parameter in model.parameters():parameter.add_(1)
                model.load_state_dict(saved['model'],strict=True)
                score,reloaded=evaluate(model,rows['valid']);assert reloaded==preds and score==valid
                acceptance.append(dict(arm=arm,official_initialized=True,shared_initial_exact=True,
                    optimizer='Adam',lr=optimizer.param_groups[0]['lr'],batch_size=64,strict_reload=True,
                    aux_weight=alpha,contrastive_temperature=cfg.temperature))
            save_json(folder/'result.json',result);results.append(result);save_json(out/'results_partial.json',results)
            del model,optimizer;torch.cuda.empty_cache()
    table(out/'runs.csv',results)
    if not smoke:
        summary=[]
        for arm in arms:
            values=[r['valid_rmse'] for r in results if r['arm']==arm]
            summary.append(dict(arm=arm,n=len(values),valid_rmse_mean=statistics.mean(values),
                                valid_rmse_sd=statistics.stdev(values) if len(values)>1 else 0.))
        table(out/'validation_summary.csv',summary)
    if smoke:save_json(out/'acceptance.json',dict(passed=True,protocol_version=VERSION,test_forward=False,
        formal_training_resumed=False,arms=acceptance,prepared_fingerprint=fingerprint,
        design_version=DESIGN_VERSION,study=a.study,descriptor_train_only=True,legacy_checkpoint_rejected=legacy_rejected,
        ASD_original_forward_exact=a.study in ('native','noaux','single-direct','single-direct-noaux'),
        regression_temperature=cfg.temperature,alignment_weight=alpha))
    save_json(out/'completion.json',dict(passed=True,stage=a.stage,results=len(results),test_forward=False,
        design_version=DESIGN_VERSION,validation_only=True,active_cap=epochs))
    status(phase='smoke_complete' if smoke else 'validation_complete_test_deferred',completed=len(results),total=len(results),formal_training_resumed=False if smoke else True)


if __name__=='__main__':main()
