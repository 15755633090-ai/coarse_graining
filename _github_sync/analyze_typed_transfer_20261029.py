"""Read-only diagnostics of the locked three-arm experiment; train-only probe CV."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'motif/adjacency_diffusion/runs/typed_property_comparison_20261029'
CODE=EXPERIMENT/'frozen_source'
sys.path.insert(0,str(CODE))
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.typed_refined import RefinedBondModel,load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset,edge_target
from motif.adjacency_diffusion.data import condition_from_batch
from motif.adjacency_diffusion.model import Config
from _github_sync.search_adjacency_rescue_20261029 import named_initialize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(x):
    x=x.astype(np.float64)
    centered=x-x.mean(0)
    eigen=np.linalg.eigvalsh(centered.T@centered/len(x)).clip(0)
    probabilities=eigen/eigen.sum() if eigen.sum() else eigen
    positive=probabilities>0
    norm=np.linalg.norm(x,axis=1)
    unit=x/np.maximum(norm[:,None],1e-12)
    return {'variance_trace':float(eigen.sum()),'mean_norm':float(norm.mean()),
            'effective_rank':float(np.exp(-np.sum(probabilities[positive]*np.log(probabilities[positive])))),
            'near_zero_variance_dimensions':int((x.var(0)<1e-8).sum()),
            'mean_pair_cosine':float(((unit.sum(0)**2).sum()-(unit**2).sum())/(len(x)*(len(x)-1)))}


def cka(x,y):
    x=x.astype(np.float64)-x.mean(0)
    y=y.astype(np.float64)-y.mean(0)
    return float(np.square(x.T@y).sum()/max(np.sqrt(np.square(x.T@x).sum()*np.square(y.T@y).sum()),1e-15))


def ridge_probe(x,y,train_count):
    cv=KFold(5,shuffle=True,random_state=71)
    pipeline=make_pipeline(StandardScaler(),Ridge(solver='cholesky'))
    probe=GridSearchCV(pipeline,{'ridge__alpha':[.001,.01,.1,1.,10.,100.,1000.]},
                       scoring='neg_mean_squared_error',cv=cv,n_jobs=1)
    probe.fit(x[:train_count].astype(np.float64),y[:train_count])
    predictions=probe.predict(x.astype(np.float64))
    return {'alpha_selected_on_training_only':probe.best_params_['ridge__alpha'],
            'training_cv_rmse':float(np.sqrt(-probe.best_score_)),
            'training_rmse':float(np.sqrt(np.mean((predictions[:train_count]-y[:train_count])**2))),
            'validation_rmse':float(np.sqrt(np.mean((predictions[train_count:]-y[train_count:])**2)))}


@torch.no_grad()
def encode(model,loader):
    model.eval()
    rows=[]
    for batch in loader:
        batch=batch.to('cuda:0')
        condition=condition_from_batch(batch,model.config.motif_prior)
        clean=edge_target(batch,condition)
        t=torch.zeros(len(clean),dtype=torch.long,device=clean.device)
        states=model.encoder(condition,clean,t)['atoms']
        pooled=states.sum(1)/condition.nodes.sum(1).clamp_min(1)[:,None]
        rows.append(pooled.cpu().numpy())
    return np.concatenate(rows)


def changes(initial,final):
    result={}
    for prefix in ('h_input','s_input','edge','time','layers','fuse'):
        numerator,denominator=0.,0.
        for name,x in initial.items():
            if name.startswith(prefix):
                y=final[name]
                numerator+=float((y.double()-x.double()).square().sum())
                denominator+=float(x.double().square().sum())
        result[prefix]=float(np.sqrt(numerator/max(denominator,1e-30)))
    return result


def main():
    out=ROOT/'motif/adjacency_diffusion/runs/typed_transfer_diagnostics_20261029'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.45,0)
    plan=json.loads((EXPERIMENT/'plan.json').read_text(encoding='utf-8'))
    for name,value in plan['source_sha256'].items():
        if digest(CODE/name)!=value:
            raise ValueError('Changed original frozen code '+name)
    sources=json.loads((EXPERIMENT/'resolved_pretrain_sources.json').read_text(encoding='utf-8'))
    results=json.loads((EXPERIMENT/'results.json').read_text(encoding='utf-8'))['runs']
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    indices=splits['train']+splits['valid']
    train_count=len(splits['train'])
    y=source.targets[indices,0].numpy().astype(np.float64)
    dataset=GraphDataset(source.smiles,source.targets,source.target_mask)
    args=SimpleNamespace(batch_size=32,seed=0)
    loader=runner.io.loader(dataset,indices,args,False)
    raw=[]
    for batch in loader:
        condition=condition_from_batch(batch,False)
        raw.append((condition.x.sum(1)/condition.nodes.sum(1)[:,None]).numpy())
    rows=[]
    features={}
    baseline=ridge_probe(np.concatenate(raw),y,train_count)
    for row in results:
        seed,arm=row['task_seed'],row['arm']
        folder=EXPERIMENT/f'{arm}_task_seed_{seed}'
        item=sources[str(plan['task_to_pretrain_seed'][str(seed)])]['full' if arm=='scratch' else arm]
        _,saved=load_model(Path(item['path']))
        config=Config(**saved['config'])
        config.dropout=.1
        model=RefinedBondModel(config,saved['priors'],'frequency_init')
        named_initialize(model,seed)
        model.initialize_priors()
        random_weights={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
        if arm!='scratch':
            model.encoder.load_state_dict(saved['encoder'])
        initial_weights={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
        initial_features=encode(model.to('cuda:0'),loader)
        initial_probe=ridge_probe(initial_features,y,train_count)
        features[(seed,arm)]=initial_features[:train_count]
        history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
        diagnostic={'seed':seed,'arm':arm,'initial_geometry':geometry(initial_features[:train_count]),
            'initial_frozen_ridge':initial_probe,'pretraining_weight_change':changes(random_weights,initial_weights),
            'epochs':len(history),'best_epoch':row['best_epoch'],
            'mean_clip_fraction':statistics.mean(r['clip_fraction'] for r in history),
            'mean_pre_clip_norm':statistics.mean(r['pre_clip_norm'] for r in history),
            'first5_clip_fraction':statistics.mean(r['clip_fraction'] for r in history[:5])}
        for stage in ('best','latest'):
            path=folder/(stage+'.pt')
            trained,checkpoint=load_model(path)
            trained.to('cuda:0')
            z=encode(trained,loader)
            predicted=[]
            trained.eval()
            with torch.no_grad():
                for batch in loader:
                    output=trained(batch.to('cuda:0')).cpu()
                    scaler=checkpoint['target_scaler']
                    predicted.append((output*scaler['scale']+scaler['center']).numpy())
            prediction=np.concatenate(predicted)[:,0]
            diagnostic[stage]={'training_rmse_eval':float(np.sqrt(np.mean((prediction[:train_count]-y[:train_count])**2))),
                'validation_rmse':float(np.sqrt(np.mean((prediction[train_count:]-y[train_count:])**2))),
                'geometry':geometry(z[:train_count]),'frozen_ridge':ridge_probe(z,y,train_count),
                'cka_initial_to_finetuned':cka(initial_features[:train_count],z[:train_count]),
                'finetuning_weight_change':changes(initial_weights,checkpoint['encoder']),
                'checkpoint_sha256':digest(path)}
            del trained
        rows.append(diagnostic)
        runner.io.atomic_write(out/'results.json',{'raw_atom_mean_ridge':baseline,'runs':rows,'test_evaluated':False})
        print(seed,arm,'frozen ridge',initial_probe['validation_rmse'],'train/val',
              diagnostic['best']['training_rmse_eval'],diagnostic['best']['validation_rmse'],flush=True)
    comparisons=[]
    for seed in (0,1,2):
        for arm in ('full','bounded'):
            comparisons.append({'seed':seed,'arm':arm,'cka_scratch_to_pretrained':cka(features[(seed,'scratch')],features[(seed,arm)])})
    runner.io.atomic_write(out/'results.json',{'raw_atom_mean_ridge':baseline,'runs':rows,
        'initial_cka_comparisons':comparisons,'test_evaluated':False,
        'probe_protocol':'train-only 5-fold CV over fixed 7 ridge alphas; validation not used for alpha selection',
        'script_sha256':digest(Path(__file__)),'original_run_plan_sha256':digest(EXPERIMENT/'plan.json')})
    runner.io.atomic_write(out/'status.json',{'state':'complete','analyzed_runs':9,'test_evaluated':False})


if __name__=='__main__':
    main()
