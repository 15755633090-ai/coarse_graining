"""Bounded paired tuning/design screen, followed by held-out-seed replication."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,shutil,sys,time,statistics
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch_geometric.data import Batch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import MaskedBondModel,mask_observation,load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset,edge_target,corrupt,reconstruction_objective
from motif.adjacency_diffusion.data import condition_from_batch
from motif.adjacency_diffusion.model import Config,reduce_messages
from _github_sync.search_adjacency_rescue_20261029 import named_initialize

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): runner.io.atomic_write(p,x)
def thash(state):
    h=hashlib.sha256()
    for k,v in state.items(): h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare(a):
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029'
    prior=json.loads((old/'plan.json').read_text(encoding='utf-8'))
    files=[ROOT/p for p in prior['source_sha256']]+[Path(__file__).resolve()]
    files=sorted(set(files))
    for p in files:
        q=out/'frozen_source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    plan={'old_run':str(old),'data_root':str(ROOT/'data'),'device':a.device,'smoke':a.smoke,
          'epochs':2 if a.smoke else 60,'pretrain_steps':2 if a.smoke else 1200,
          'screen_seed':0,'replication_seeds':[1,2],
          'candidates':{
              'slow_regularized':{'lr':.0003,'encoder_lr':None,'wd':.0001,'dropout':.2,'norm':'candidate'},
              'strong_regularized':{'lr':.001,'encoder_lr':None,'wd':.001,'dropout':.3,'norm':'candidate'},
              'observation_mass':{'lr':.001,'encoder_lr':None,'wd':0.,'dropout':.1,'norm':'observation_mass'}},
          'selection':'lowest masked RMSE on screening seed0; no changes after selection; replicate selected pair on seeds1,2',
          'previous_validation_reused':True,'test_evaluated':False,
          'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    write(out/'plan.json',plan);print(str(out),flush=True)

def checks():
    # Weighted mean is invariant to adding zero-weight candidates; no hidden degree.
    m=torch.tensor([[[[2.],[6.],[100.]]]])
    mask=torch.ones(1,1,3,dtype=torch.bool);w=torch.tensor([[[1.,1.,0.]]])
    torch.testing.assert_close(reduce_messages(m,mask,w,'observation_mass'),torch.tensor([[[4.]]]))
    torch.testing.assert_close(reduce_messages(m,mask,w),torch.tensor([[[8/3]]]))
    torch.testing.assert_close(reduce_messages(m,mask,w*0,'observation_mass'),torch.zeros(1,1,1))
    w.requires_grad_();reduce_messages(m,mask,w,'observation_mass').sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad.abs().sum()>0

def execute(a):
    out=a.output_dir.resolve();plan=json.loads((out/'plan.json').read_text(encoding='utf-8'));old=Path(plan['old_run'])
    assert ROOT==out/'frozen_source'
    for p,h in plan['source_sha256'].items(): assert sha(ROOT/p)==h,p
    checks();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.45,0)
    oldplan=json.loads((old/'plan.json').read_text(encoding='utf-8'));sources=json.loads((old/'pretrain_sources.json').read_text(encoding='utf-8'))
    device=plan['device'];rows=[];pretrain=[];start=time.perf_counter()
    # Immutable old curves expose train/validation divergence without new test access.
    curves=[]
    for arm in ['scratch','unmasked','masked']:
        for seed in range(3):
            h=json.loads((old/f'{arm}_task_seed_{seed}'/'history.json').read_text(encoding='utf-8'))
            b=min(h,key=lambda x:x['validation']['rmse'])
            curves.append({'arm':arm,'seed':seed,'best':b,'last':h[-1]})
    write(out/'prior_curve_diagnosis.json',curves)
    data_source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    splits={k:v[:16] if plan['smoke'] else v for k,v in splits.items() if k in ('train','valid')}
    dataset=GraphDataset(data_source.smiles,data_source.targets,data_source.target_mask)
    scaler=runner.io.TargetScaler.fit(data_source.targets[splits['train']],data_source.target_mask[splits['train']])
    mass_sources={}

    def get_pretrained(seed,norm):
        preseed=oldplan['task_to_pretrain_seed'][str(seed)]
        item=sources[str(preseed)]['masked']
        assert sha(Path(item['path']))==item['sha256']
        if norm=='candidate': return load_model(Path(item['path']))[1],item
        if seed in mass_sources: return mass_sources[seed]
        _,saved=load_model(Path(item['path']));cfg=Config(**saved['config']);cfg.message_normalization=norm
        model=MaskedBondModel(cfg,saved['priors'],.3);named_initialize(model,preseed);model.initialize_priors()
        initial=thash(model.state_dict());model.to(device).set_stage('pretrain')
        params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=.001)
        rngs=[torch.Generator(device='cpu' if j==0 else device).manual_seed(preseed+k) for j,k in enumerate([333,444,555,666])]
        train=GraphDataset(oldplan['train_smiles'][:8] if plan['smoke'] else oldplan['train_smiles'])
        folder=out/'pretraining'/f'mass_seed_{preseed}';folder.mkdir(parents=True,exist_ok=False)
        history=[]
        for step in range(1,plan['pretrain_steps']+1):
            model.train();ids=torch.randperm(len(train),generator=rngs[0])[:8].tolist()
            batch=Batch.from_data_list([train[i] for i in ids]).to(device);c=condition_from_batch(batch,True)
            target=edge_target(batch,c);t=torch.randint(1,1001,(len(target),),device=device,generator=rngs[1])
            noisy=corrupt(target,t,c.pairs,model.schedule,rngs[2]);observed,selected=mask_observation(noisy,c.pairs,.3,rngs[3])
            opt.zero_grad(set_to_none=True);loss,blocks=reconstruction_objective(model.predict_bonds(c,observed,t),target,selected)
            loss.backward();norm_value=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True));opt.step()
            if step%100==0 or step==plan['pretrain_steps']:
                history.append({'step':step,'loss':float(loss.detach()),'gradient_norm':norm_value});write(folder/'history.json',history)
                write(out/'status.json',{'state':'running','stage':'mass_pretraining','seed':seed,'step':step,'completed_property_jobs':len(rows)})
        cp=folder/'encoder.pt';runner.io.atomic_write(cp,model.checkpoint('pretrain',steps=plan['pretrain_steps']),binary=True)
        replay,saved=load_model(cp)
        assert thash(replay.state_dict())==thash(model.cpu().state_dict())
        item={'path':str(cp),'sha256':sha(cp),'initial_state_hash':initial,'checkpoint_replay_passed':True}
        mass_sources[seed]=(saved,item);pretrain.append(item);write(out/'pretraining_sources.json',pretrain)
        return saved,item

    def task(name,seed,arm,phase):
        settings=plan['candidates'][name];saved,item=get_pretrained(seed,settings['norm'])
        cfg=Config(**saved['config']);cfg.dropout=settings['dropout'];cfg.message_normalization=settings['norm']
        model=MaskedBondModel(cfg,saved['priors'],.3,spec.num_tasks);named_initialize(model,seed);model.initialize_priors()
        initial={'encoder':thash(model.encoder.state_dict()),'head':thash(model.property_head.state_dict())}
        if arm=='masked':model.encoder.load_state_dict(saved['encoder'],strict=True)
        model.to(device);torch.manual_seed(seed+10000)
        folder=out/f'{phase}_{name}_{arm}_seed_{seed}'
        fitargs=SimpleNamespace(stage='finetune',output_dir=folder,device=device,seed=seed,epochs=plan['epochs'],patience=plan['epochs'],
            batch_size=16,learning_rate=settings['lr'],encoder_learning_rate=settings['encoder_lr'],weight_decay=settings['wd'],grad_clip=5.,resume=False)
        meta={'candidate':name,'settings':settings,'seed':seed,'arm':arm,'phase':phase,'config':asdict(cfg),
              'initial':initial,'source':item if arm=='masked' else None,'splits':splits,'test_evaluated':False,'plan_sha256':sha(out/'plan.json')}
        write(out/'status.json',{'state':'running','stage':'property','candidate':name,'arm':arm,'seed':seed,'completed_property_jobs':len(rows)})
        with runner.io.run_directory(fitargs,meta):
            train=runner.io.loader(dataset,splits['train'],fitargs,True);valid=runner.io.loader(dataset,splits['valid'],fitargs,False)
            result=runner.fit(model,train,valid,fitargs,meta,spec,scaler,None)
            pred,y,mask,_=runner.io.predict(model,valid,device,scaler,spec)
            result.update(candidate=name,seed=seed,arm=arm,phase=phase,validation=runner.io.compute_property_metrics(pred,y,mask,spec),initial=initial)
            replay,_=load_model(folder/'best.pt');replay.to(device)
            actual,*_=runner.io.predict(replay,valid,device,scaler,spec);torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
            result['checkpoint_replay_passed']=True;write(folder/'completion.json',result)
        rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print(name,seed,arm,result['validation']['rmse'],flush=True)

    for name in plan['candidates']:
        for arm in ['scratch','masked']:task(name,0,arm,'screen')
    selected=min((r for r in rows if r['arm']=='masked'),key=lambda r:r['validation']['rmse'])['candidate']
    write(out/'selection.json',{'candidate':selected,'rule':plan['selection'],'screen_results':rows.copy(),'test_evaluated':False})
    for seed in ([] if plan['smoke'] else plan['replication_seeds']):
        for arm in ['scratch','masked']:task(selected,seed,arm,'replication')
    chosen=[r for r in rows if r['candidate']==selected];aggregate={}
    for arm in ['scratch','masked']:
        values=[r['validation']['rmse'] for r in chosen if r['arm']==arm]
        aggregate[arm]={'n':len(values),'mean_rmse':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else None}
    paired=[]
    for seed in sorted({r['seed'] for r in chosen}):
        pair={r['arm']:r for r in chosen if r['seed']==seed};assert pair['scratch']['initial']==pair['masked']['initial']
        paired.append({'seed':seed,'masked_minus_scratch':pair['masked']['validation']['rmse']-pair['scratch']['validation']['rmse']})
    write(out/'results.json',{'runs':rows,'selected':selected,'aggregate_selected':aggregate,'paired':paired,'test_evaluated':False,
        'selection_bias':'seed0 used for selection; seeds1,2 only seed replication on reused validation split'})
    write(out/'status.json',{'state':'complete','completed_property_jobs':len(rows),'selected':selected,'all_replays_passed':True,
        'paired_initializations_passed':True,'elapsed_seconds':time.perf_counter()-start,'test_evaluated':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.execute:
        try:execute(a)
        except Exception as e:write(a.output_dir/'status.json',{'state':'failed','error':repr(e)});raise
    else:prepare(a)
