"""Read-only checkpoint audit: no training, model selection, or test access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, hashlib, statistics
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(xs):return statistics.mean(xs)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    old=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029'
    new=ROOT/'motif/adjacency_diffusion/runs/representation_matrix_20261030'
    out=ROOT/'motif/adjacency_diffusion/runs/transfer_gap_audit_20261030'
    out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Completed audit already exists')
    source,_,spec=runner.io.build_property_data(ROOT/'data','esol')
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    folders=[('scratch',s,old/f'scratch_task_seed_{s}') for s in range(3)]
    folders += [('small_edge',s,old/f'masked_task_seed_{s}') for s in range(3)]
    folders += [(f'{c}_{o}',s,new/'properties'/f'{c}_{o}_{p}') for c,o in [('small','representation'),('expanded','edge'),('expanded','representation')] for s,p in [(0,7),(1,1),(2,2)]]
    rows=[];baseline_splits=None
    for arm,seed,folder in folders:
        config=read(folder/'config.json');splits=config['splits']
        active={k:splits[k] for k in ('train','valid')}
        if baseline_splits is None:baseline_splits=active
        assert active==baseline_splits
        assert len(set(active['train']) & set(active['valid']))==0
        scaler=runner.io.TargetScaler.fit(source.targets[active['train']],source.target_mask[active['train']])
        history=read(folder/'history.json');assert len(history)==60
        bestrow=min(history,key=lambda r:r['validation']['rmse'])
        row={'arm':arm,'task_seed':seed,'folder':str(folder.relative_to(ROOT)),
             'history_sha256':sha(folder/'history.json'),'best_epoch':bestrow['epoch'],
             'curve':{str(e):{'train_loss_batch_mean':history[e-1]['train_loss_batch_mean'],
                                'valid_rmse':history[e-1]['validation']['rmse']} for e in (1,5,10,20,40,60)},
             'train_loss_at_best_epoch':bestrow['train_loss_batch_mean'],
             'last10_train_loss_mean':mean([r['train_loss_batch_mean'] for r in history[-10:]]),
             'mean_clip_fraction':mean([r['clip_fraction'] for r in history])}
        for tag in ('best','latest'):
            path=folder/f'{tag}.pt';model,saved=load_model(path);model.to('cuda');model.eval()
            assert saved['stage']=='finetune'
            scores={}
            for split in ('train','valid'):
                args=SimpleNamespace(device='cuda',batch_size=16,seed=seed)
                batches=runner.io.loader(data,active[split],args,False)
                p,y,m,_=runner.io.predict(model,batches,'cuda',scaler,spec)
                scores[split]=runner.io.compute_property_metrics(p,y,m,spec)
            expected=bestrow['validation']['rmse'] if tag=='best' else history[-1]['validation']['rmse']
            assert abs(scores['valid']['rmse']-expected)<1e-6
            row[tag]={'epoch':saved['epoch'],'checkpoint_sha256':sha(path),**scores,
                      'valid_minus_train_rmse':scores['valid']['rmse']-scores['train']['rmse']}
            del model
        row['late_train_rmse_change']=row['latest']['train']['rmse']-row['best']['train']['rmse']
        row['late_valid_rmse_change']=row['latest']['valid']['rmse']-row['best']['valid']['rmse']
        rows.append(row)
        print(arm,seed,'best',row['best']['train']['rmse'],row['best']['valid']['rmse'],'latest',row['latest']['train']['rmse'],row['latest']['valid']['rmse'],flush=True)
    aggregate={}
    for arm in dict.fromkeys(r['arm'] for r in rows):
        group=[r for r in rows if r['arm']==arm]
        aggregate[arm]={f'{tag}_{split}_rmse':mean([r[tag][split]['rmse'] for r in group]) for tag in ('best','latest') for split in ('train','valid')}
        aggregate[arm].update(mean_best_epoch=mean([r['best_epoch'] for r in group]),
            late_overfit_count=sum(r['late_train_rmse_change']<0 and r['late_valid_rmse_change']>0 for r in group))
    pairs=[]
    for r in rows:
        if r['arm']=='scratch':continue
        scratch=next(x for x in rows if x['arm']=='scratch' and x['task_seed']==r['task_seed'])
        pairs.append({'arm':r['arm'],'task_seed':r['task_seed'],**{f'{tag}_{split}_minus_scratch':r[tag][split]['rmse']-scratch[tag][split]['rmse'] for tag in ('best','latest') for split in ('train','valid')}})
    result={'scope':'read-only audit of all existing 15 runs, best and epoch60 checkpoints; no training or test access',
        'train_n':len(baseline_splits['train']),'valid_n':len(baseline_splits['valid']),
        'source_script_sha256':sha(Path(__file__)),'rows':rows,'aggregate':aggregate,'paired_deltas':pairs,
        'limits':['Best epochs selected by reused validation; gaps at different epochs do not isolate causal mechanisms.',
                  'Training logs contain dropout-enabled batch-averaged standardized MSE; not directly comparable to evaluation RMSE.',
                  'No intermediate checkpoints exist; full per-epoch evaluation-mode training curves cannot be reconstructed.',
                  'Lower training error plus higher validation error supports a fit/generalization mismatch, not its unique cause.'],
        'validation_replay_passed':True,'test_evaluated':False}
    runner.io.atomic_write(out/'results.json',result)
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':main()
