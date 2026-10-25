"""Original HMG architecture/protocol with seven prediction-fusion subsets."""
import argparse
import csv
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_dataset, save_json, sha
from .hmg_fusion_ablation import ARMS, variant
from .paper_data import load_paper, load_hmg_features, hmg_inputs, source_inventory


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def batches(rows, size=16, seed=None):
    indices = list(range(len(rows))) if seed is None else torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    groups = [indices[i:i+size] for i in range(0,len(indices),size)]
    if len(groups)>1 and len(groups[-1])==1: groups[-2].extend(groups.pop())
    return [[rows[i] for i in group] for group in groups]


def table(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('motif/runs/scale_pairs_3seeds_20261018'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-result',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); torch.set_num_threads(2); seed_all(25)
    reference=a.reference.resolve(); out=a.output.resolve()
    manifest0=read(reference/'hmg_seed0/manifest.json')
    old=manifest0['args']; paper=Path(old['paper_root']); data_path=Path(old['data'])
    feature_path=Path(old['feature_cache']); preflight=read(Path(old['preflight_dir'])/'preflight.json')
    assert preflight['passed'] and not manifest0.get('loss_ablation',False)
    assert sha(data_path)==preflight['data_sha256']==manifest0['data_sha256']
    assert sha(feature_path)==preflight['feature_cache_sha256']
    assert source_inventory(paper)==preflight['paper_source_sha256']
    implementation={name:sha(Path(__file__).parent/name) for name in
                    ('train_hmg_fusion_ablation.py','hmg_fusion_ablation.py','paper_data.py','data.py',
                     'report_hmg_fusion_ablation.py')}
    if not a.smoke:
        assert a.smoke_result is not None
        acceptance=read(a.smoke_result/'acceptance.json')
        assert acceptance['passed'] and not acceptance['test_forward']
        assert acceptance['implementation_sha256']==implementation
        assert acceptance['feature_cache_sha256']==sha(feature_path)
    data=load_dataset(data_path)
    all_rows={s:[r for r in data['molecules'] if r['split']==s] for s in ('train','valid','test')}
    for s,rs in all_rows.items():
        assert [r['row'] for r in rs]==preflight['rows'][s]==manifest0['rows'][s]
    labels=torch.tensor([r['label'] for r in all_rows['train']])
    mean,std=labels.mean().item(),labels.std(unbiased=False).clamp_min(1e-6).item()
    assert abs(mean-manifest0['target_mean'])<1e-7 and abs(std-manifest0['target_std'])<1e-7
    rows=({'train':all_rows['train'][:32],'valid':all_rows['valid'][:16]} if a.smoke else all_rows)
    cls,_,cfg=load_paper(paper,'HMG-MRL',a.device)
    features,assemble=load_hmg_features(feature_path,paper)
    sample,_,_=hmg_inputs(rows['train'][:2],features,assemble,a.device)
    cfg.atom_dim_in=sample['x_atoms'].shape[-1]; cfg.bond_dim_in=sample['x_bonds'].shape[-1]
    mother=cls(cfg.atom_dim_in,cfg.bond_dim_in,1)
    epochs,patience=(3,3) if a.smoke else (100,20)
    seeds=[0] if a.smoke else [0,1,2]
    out.mkdir(parents=True,exist_ok=a.resume)
    protocol=dict(paper='HMG-MRL',scope='end-to-end prediction-fusion view ablation, original auxiliary retained',
        arms=ARMS,seeds=seeds,epochs=epochs,patience=patience,batch_size=16,lr=.0001,
        optimizer='AdamW',weight_decay=1e-5,gradient_clip=5,aux_weight=cfg.ct_loss_coff,
        radius=cfg.radius,preGNN_num_layers=cfg.preGNN_num_layers,
        frag_num_encoder_layers=cfg.frag_num_encoder_layers,dropout=cfg.dropout,
        hidden=cfg.atom_feature_dim,smoke_only=a.smoke,device=a.device,
        data_sha256=sha(data_path),feature_cache_sha256=sha(feature_path),
        paper_source_sha256=source_inventory(paper),implementation_sha256=implementation,
        rows={s:[r['row'] for r in rs] for s,rs in rows.items()},mean=mean,std=std,
        D_definition='200 descriptors, descriptor MLP and original atom-context update; not HimNet graph node',
        motif_and_auxiliary_paths='computed for every arm, even if omitted from prediction fusion',
        original_fusion='subset stack, recomputed sum and attention, same classifier; no zero placeholders',
        ASD_policy='reuse original Full only after exact source/split/features/initialization/protocol checks and validation reload',
        historical_test_seen=True,registered_parameters_equal=True,effective_capacity_matched=False)
    if a.resume:
        previous_protocol=read(out/'protocol.json')
        assert previous_protocol==json.loads(json.dumps(protocol))
    else:
        save_json(out/'protocol.json',protocol)
    for name in implementation: (out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    def status(**kwargs): save_json(out/'status.json',kwargs)
    def inputs(rs): return hmg_inputs(rs,features,assemble,a.device)[0]
    def evaluate(model,rs):
        model.eval(); predictions=[]
        with torch.no_grad():
            for group in batches(rs):
                value=model(**inputs(group))[1].flatten()*std+mean
                assert torch.isfinite(value).all()
                predictions.extend(value.cpu().tolist())
        targets=[r['label'] for r in rs]
        rmse=float(np.sqrt(np.mean((np.array(predictions)-targets)**2)))
        return rmse,[dict(row=r['row'],target=r['label'],prediction=pred) for r,pred in zip(rs,predictions)]
    results=read(out/'results_partial.json') if a.resume and (out/'results_partial.json').exists() else []
    acceptance_arms=[]
    for seed in seeds:
        old_dir=reference/f'hmg_seed{seed}'; previous=read(old_dir/'manifest.json')
        assert previous['source']==manifest0['source']
        assert previous['rows']==manifest0['rows'] and previous['data_sha256']==protocol['data_sha256']
        for key,expected in [('epochs',100),('patience',20),('batch_size',16),('lr',.0001)]:
            assert previous['args'][key]==expected
        assert not previous.get('loss_ablation',False) and previous['aux_weight_full']==cfg.ct_loss_coff
        initial=torch.load(old_dir/'initial_states.pt',map_location='cpu',weights_only=True)['full']
        mother.load_state_dict(initial,strict=True)
        seed_dir=out/f'seed_{seed}';seed_dir.mkdir(exist_ok=a.resume)
        save_json(seed_dir/'initialization.json',dict(source=str(old_dir/'initial_states.pt'),
            sha256=sha(old_dir/'initial_states.pt'),all_variant_tensors_exact=True))
        order=ARMS if a.smoke else ('ASD',*ARMS[:-1])
        for arm in order:
            if any(r['seed']==seed and r['arm']==arm for r in results):continue
            folder=seed_dir/arm;folder.mkdir(exist_ok=a.resume)
            model=variant(mother,arm,folder/'forward.py').to(a.device)
            if a.smoke:
                model.eval(); original=mother.to(a.device).eval()
                with torch.no_grad():
                    kw=inputs(rows['train'][:4]); expected=original(**kw); actual=model(**kw)
                    torch.testing.assert_close(actual[2],expected[2])
                    if arm=='ASD':
                        torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
                        torch.testing.assert_close(actual[6],expected[6],rtol=0,atol=0)
                mother.cpu()
            if not a.smoke and arm=='ASD':
                old_results=read(old_dir/'results.json')['arms']['full']
                saved=torch.load(old_dir/'full_best.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(saved['model'],strict=True)
                valid,preds=evaluate(model,rows['valid'])
                assert abs(valid-old_results['valid_rmse'])<1e-6
                stored=read(old_dir/'full_test_predictions.json')
                assert stored['row_ids']==protocol['rows']['test']
                assert stored['target']==[r['label'] for r in rows['test']]
                test=float(np.sqrt(np.mean((np.array(stored['prediction'])-stored['target'])**2)))
                assert abs(test-old_results['test_rmse'])<1e-7
                shutil.copyfile(old_dir/'full_best.pt',folder/'best.pt')
                shutil.copyfile(old_dir/'full_history.json',folder/'history.json')
                table(folder/'valid_predictions.csv',preds)
                table(folder/'test_predictions.csv',[dict(row=r,target=y,prediction=v) for r,y,v in
                    zip(stored['row_ids'],stored['target'],stored['prediction'])])
                results.append(dict(seed=seed,arm=arm,reused=True,valid_rmse=valid,test_rmse=test,
                    best_epoch=old_results['best_epoch'],epochs_run=old_results['epochs_run']))
                save_json(out/'results_partial.json',results);del model;continue
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=1e-5)
            best,best_epoch,history=float('inf'),0,[]
            for epoch in range(1,epochs+1):
                seed_all(seed+4000+epoch);model.train();started=time.perf_counter();losses=[]
                for step,group in enumerate(batches(rows['train'],seed=seed+epoch)):
                    prediction=model(**inputs(group)); target=torch.tensor([r['label'] for r in group],device=a.device)
                    loss=(prediction[1].flatten()-(target-mean)/std).square().mean()+cfg.ct_loss_coff*prediction[2]
                    if not torch.isfinite(loss):raise FloatingPointError((seed,arm,epoch,step))
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if epoch==1 and step==0:
                        gradients={n:('none' if v.grad is None else 'finite' if torch.isfinite(v.grad).all() else 'nonfinite')
                                   for n,v in model.named_parameters()}
                        assert 'nonfinite' not in gradients.values()
                        save_json(folder/'gradient_audit.json',dict(parameters=sum(v.numel() for v in model.parameters()),
                            effective_parameters=sum(v.numel() for v in model.parameters() if v.grad is not None),
                            gradients=gradients))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                    optimizer.step();losses.append(float(loss.detach()))
                    if step%25==0:status(phase='training',seed=seed,arm=arm,epoch=epoch,batch=step,
                                         completed=len(results),total=len(seeds)*len(ARMS))
                valid,preds=evaluate(model,rows['valid'])
                history.append(dict(epoch=epoch,valid_rmse=valid,mean_batch_objective=float(np.mean(losses)),
                                    seconds=time.perf_counter()-started))
                if valid<best:
                    best,best_epoch=valid,epoch
                    torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        epoch=epoch,valid_rmse=valid,mean=mean,std=std),folder/'best.pt')
                save_json(folder/'history.json',history)
                print(dict(seed=seed,arm=arm,epoch=epoch,valid_rmse=valid),flush=True)
                if epoch-best_epoch>=patience:break
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model'],strict=True)
            valid,preds=evaluate(model,rows['valid']);assert abs(valid-best)<1e-6
            table(folder/'valid_predictions.csv',preds)
            if a.smoke:
                with torch.no_grad():
                    for v in model.parameters():v.add_(1)
                model.load_state_dict(saved['model'],strict=True)
                reloaded,again=evaluate(model,rows['valid'])
                assert preds==again and valid==reloaded
                acceptance_arms.append(dict(arm=arm,strict_disk_reload=True,
                    registered_parameters=sum(v.numel() for v in model.parameters())))
                result=dict(seed=seed,arm=arm,reused=False,valid_rmse=valid,best_epoch=best_epoch,epochs_run=epoch)
            else:
                test,testpred=evaluate(model,rows['test']);table(folder/'test_predictions.csv',testpred)
                result=dict(seed=seed,arm=arm,reused=False,valid_rmse=valid,test_rmse=test,
                            best_epoch=best_epoch,epochs_run=epoch)
            results.append(result);save_json(out/'results_partial.json',results)
            del model,optimizer;torch.cuda.empty_cache()
    if a.smoke:
        save_json(out/'acceptance.json',dict(passed=True,test_forward=False,arms=acceptance_arms,
            ASD_forward_original_exact=True,auxiliary_unchanged_all_arms=True,
            implementation_sha256=implementation,feature_cache_sha256=sha(feature_path)))
    else:
        table(out/'runs.csv',results)
        summary=[]
        for arm in ARMS:
            rr=[r for r in results if r['arm']==arm];values=[r['test_rmse'] for r in rr]
            summary.append(dict(arm=arm,n=len(values),test_rmse_mean=statistics.mean(values),test_rmse_sd=statistics.stdev(values)))
        table(out/'summary.csv',summary)
        lookup={(r['seed'],r['arm']):r['test_rmse'] for r in results}
        paired=[]
        for combo in ARMS[3:]:
            controls=list(combo)+(list(ARMS[3:6]) if combo=='ASD' else [])
            for control in controls:
                for seed in seeds:
                    paired.append(dict(combo=combo,control=control,seed=seed,
                                       delta_rmse=lookup[seed,combo]-lookup[seed,control]))
        table(out/'paired_deltas.csv',paired)
        subprocess.run([sys.executable,str(Path(__file__).with_name('report_hmg_fusion_ablation.py')),
                        '--results',str(out)],check=True)
    save_json(out/'completion.json',dict(passed=True,smoke=a.smoke,runs=len(results),
        new_training_runs=sum(not r['reused'] for r in results),test_forward=not a.smoke))
    status(phase='complete',completed=len(results),total=len(results))
    print(dict(phase='complete',smoke=a.smoke,runs=len(results)),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if '--output' in sys.argv:
            target=Path(sys.argv[sys.argv.index('--output')+1])
            if target.is_dir():save_json(target/'status.json',dict(phase='failed',error=repr(exc)))
        raise
