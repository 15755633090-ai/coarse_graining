"""Frozen encoder head fitting: raw versus train-only standardized features."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'motif/adjacency_diffusion/runs/typed_property_comparison_20261029'
sys.path.insert(0,str(EXPERIMENT/'frozen_source'))
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.typed_refined import RefinedBondModel,load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset,edge_target
from motif.adjacency_diffusion.data import condition_from_batch
from motif.adjacency_diffusion.model import Config
from _github_sync.search_adjacency_rescue_20261029 import named_initialize


@torch.no_grad()
def encode(model,batches):
    model.eval()
    result=[]
    for batch in batches:
        batch=batch.to('cuda:0')
        condition=condition_from_batch(batch,True)
        clean=edge_target(batch,condition)
        t=torch.zeros(len(clean),dtype=torch.long,device=clean.device)
        states=model.encoder(condition,clean,t)['atoms']
        result.append((states.sum(1)/condition.nodes.sum(1).clamp_min(1)[:,None]).cpu())
    return torch.cat(result)


def main():
    out=ROOT/'motif/adjacency_diffusion/runs/typed_frozen_head_diagnostics_20261029'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.45,0)
    plan=json.loads((EXPERIMENT/'plan.json').read_text(encoding='utf-8'))
    sources=json.loads((EXPERIMENT/'resolved_pretrain_sources.json').read_text(encoding='utf-8'))
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    indices=splits['train']+splits['valid']
    n=len(splits['train'])
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    loader=runner.io.loader(data,indices,SimpleNamespace(seed=0,batch_size=32),False)
    target=source.targets[indices].cpu()
    center,scale=target[:n].mean(0),target[:n].std(0,unbiased=False)
    # Match the original task scaler exactly, including its variance convention.
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    center,scale=scaler.center,scaler.scale
    y=(target-center)/scale
    rows=[]
    for seed in (0,1,2):
        for arm in ('scratch','full','bounded'):
            item=sources[str(plan['task_to_pretrain_seed'][str(seed)])]['full' if arm=='scratch' else arm]
            _,saved=load_model(Path(item['path']))
            config=Config(**saved['config'])
            config.dropout=.1
            model=RefinedBondModel(config,saved['priors'],'frequency_init')
            named_initialize(model,seed)
            model.initialize_priors()
            head=copy.deepcopy(model.property_head).cpu()
            if arm!='scratch':
                model.encoder.load_state_dict(saved['encoder'],strict=True)
            features=encode(model.to('cuda:0'),loader)
            torch.save({'features':features,'target':target,'train_count':n},out/f'{arm}_seed_{seed}_features.pt')
            for standardize in (False,True):
                predictor=copy.deepcopy(head)
                mean,std=features[:n].mean(0),features[:n].std(0,unbiased=False).clamp_min(1e-6)
                x=(features-mean)/std if standardize else features
                batches=DataLoader(TensorDataset(x[:n],y[:n]),batch_size=16,shuffle=True,
                                   generator=torch.Generator().manual_seed(seed))
                optimizer=torch.optim.Adam(predictor.parameters(),lr=.001)
                best,best_state,best_epoch,stale=None,None,0,0
                history=[]
                for epoch in range(1,61):
                    predictor.train()
                    for values,labels in batches:
                        optimizer.zero_grad(set_to_none=True)
                        loss=(predictor(values)-labels).square().mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(predictor.parameters(),5.,error_if_nonfinite=True)
                        optimizer.step()
                    predictor.eval()
                    with torch.no_grad():
                        predicted=predictor(x)*scale+center
                        train_rmse=float(((predicted[:n]-target[:n]).square().mean()).sqrt())
                        valid_rmse=float(((predicted[n:]-target[n:]).square().mean()).sqrt())
                    history.append({'epoch':epoch,'train_rmse':train_rmse,'validation_rmse':valid_rmse})
                    if best is None or valid_rmse<best:
                        best,best_epoch,stale=valid_rmse,epoch,0
                        best_state=copy.deepcopy(predictor.state_dict())
                    else:
                        stale+=1
                    if stale>=15:
                        break
                label='standardized' if standardize else 'raw'
                runner.io.atomic_write(out/f'{arm}_seed_{seed}_{label}_history.json',history)
                torch.save({'head':best_state,'feature_mean':mean,'feature_scale':std,
                            'standardized':standardize,'target_center':center,'target_scale':scale},
                           out/f'{arm}_seed_{seed}_{label}_head.pt')
                row={'seed':seed,'arm':arm,'input':label,'best_epoch':best_epoch,'epochs':len(history),
                     'validation_rmse':best,'training_rmse_at_best':history[best_epoch-1]['train_rmse']}
                rows.append(row)
                runner.io.atomic_write(out/'results.json',{'runs':rows,'test_evaluated':False})
                print(seed,arm,label,best,flush=True)
    runner.io.atomic_write(out/'results.json',{'runs':rows,'test_evaluated':False,
        'protocol':'frozen initial encoders; matched MLP initialization/data order; train-only feature statistics; Adam .001, batch16, 60 epochs, patience15',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    runner.io.atomic_write(out/'status.json',{'state':'complete','completed_head_probes':18,'test_evaluated':False})


if __name__=='__main__':
    main()
