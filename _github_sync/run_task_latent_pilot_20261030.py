"""Original backbone; task-aligned motif latent denoising, bounded go/no-go pilot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,shutil,sys,time,statistics
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.task_latent_denoising import TaskLatentModel,load_model,select_fragments,normalized_feature_loss
from motif.adjacency_diffusion.masked_bonds import load_model as load_baseline
from motif.adjacency_diffusion.typed_bonds import GraphDataset
from _github_sync.search_adjacency_rescue_20261029 import named_initialize
from _github_sync.check_masked_property_20261029 import tensor_hash

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):runner.io.atomic_write(p,x)

def prepare(out,smoke):
    out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029'
    previous=read(old/'plan.json')
    files=set(previous['source_sha256'])|{p.relative_to(ROOT).as_posix() for p in (ROOT/'motif/adjacency_diffusion').glob('*.py')}|{Path(__file__).relative_to(ROOT).as_posix()}
    for rel in files:
        dest=out/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    plan={'scope':'return to intrinsic motif states -> conditional context denoising -> property prediction; same original parameterization',
        'old_run':str(old),'data_root':str(ROOT/'data'),'arms':['task_augmentation','task_denoising'],
        'pilot_seed':0,'replication_seeds':[1,2],'epochs':2 if smoke else 60,'limit':16 if smoke else 0,'smoke':smoke,
        'latent_sigma_levels':[.1,.3,.5],'fragment_ratio':.3,'feature_denoise_weight':.1,
        'corruption':'after first intra-motif structure update; selected entire fragments; corrupt s1, parallel gate-s0 and h0; additive per-node RMS-scaled Gaussian plus shared noise-level embedding',
        'structure':'full clean atom/bond features and BRICS membership; no chemical feature, bond or coordinate diffusion; single-fragment molecules remain clean',
        'loss':'mean(clean_property, noisy_property) + weight*selected-atom normalized contextual-feature MSE; weight0 in augmentation control',
        'target':'stop-gradient clean final fused states trained by property labels, no EMA or unsupervised random teacher',
        'inference':'exact original clean forward through the shared context enhancer; no artificial noise, no reverse sampling',
        'learning_rate':.001,'batch_size':16,'dropout':.1,'weight_decay':0.,'grad_clip':5.,
        'gate':'replicate seeds1,2 only if denoising seed0 beats both original scratch seed0 and augmentation by at least .01 validation RMSE',
        'limits':'reused ESOL validation; seed0-selected go/no-go is exploratory; two forward branches cost more than original one; not original Noisy Nodes reproduction',
        'test_evaluated':False,'source_sha256':{rel:sha(ROOT/rel) for rel in sorted(files)}}
    write(out/'plan.json',plan)

def fit(model,data,splits,args,scaler,spec,plan,weight,metadata):
    model.set_stage('finetune');params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=.001)
    train=runner.io.loader(data,splits['train'],args,True);valid=runner.io.loader(data,splits['valid'],args,False)
    gs={key:torch.Generator(device='cuda').manual_seed(args.seed+k) for key,k in [('mask',333),('sigma',444),('epsilon',555)]}
    levels=torch.tensor(plan['latent_sigma_levels'],device='cuda');history=[];best=None;best_state=None;started=time.perf_counter()
    for epoch in range(1,plan['epochs']+1):
        model.train();values=[];clips=0;norms=[];eligible_count=selected_count=0
        for batch in train:
            batch=batch.cuda();optimizer.zero_grad(set_to_none=True)
            clean,c,_=model.clean_states(batch);p_clean=model.predict_states(clean,c)
            selected,eligible=select_fragments(batch,c,gs['mask'],plan['fragment_ratio'])
            sigma=levels[torch.randint(len(levels),(len(c.nodes),),device='cuda',generator=gs['sigma'])]
            noisy,_=model.latent_states(batch,selected,sigma,gs['epsilon'])
            p_noisy=model.predict_states(noisy,c)
            # Preserve the exact supervised weight and dropout draw for single-fragment cases.
            p_noisy=torch.where(eligible[:,None],p_noisy,p_clean)
            l_clean=runner.io.shared.masked_loss(p_clean,batch.y,batch.target_mask,spec,scaler,None)
            l_noisy=runner.io.shared.masked_loss(p_noisy,batch.y,batch.target_mask,spec,scaler,None)
            denoise=normalized_feature_loss(noisy['atoms'],clean['atoms'],selected)
            loss=.5*(l_clean+l_noisy)+weight*denoise
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite task-latent objective')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True));optimizer.step()
            values.append([float(v.detach()) for v in [loss,l_clean,l_noisy,denoise]])
            norms.append(norm);clips+=norm>5.;eligible_count+=int(eligible.sum());selected_count+=int(selected.sum())
        pred,y,m,_=runner.io.predict(model,valid,'cuda',scaler,spec);metrics=runner.io.compute_property_metrics(pred,y,m,spec);score=metrics['rmse']
        improved=best is None or score<best
        best_epoch=epoch if improved else best_state['best_epoch'];best=score if improved else best
        history.append({'epoch':epoch,'objective':statistics.mean(v[0] for v in values),
            'clean_property':statistics.mean(v[1] for v in values),'noisy_property':statistics.mean(v[2] for v in values),
            'feature_mse':statistics.mean(v[3] for v in values),'pre_clip_norm':statistics.mean(norms),'clip_fraction':clips/len(values),
            'eligible_molecules':eligible_count,'selected_atoms':selected_count,'validation':metrics})
        state=model.checkpoint('finetune',epoch=epoch,best_epoch=best_epoch,best_score=best,run_metadata=metadata,
            target_scaler={'center':scaler.center,'scale':scaler.scale})
        if improved:best_state=runner.io.shared.cpu_snapshot(state)
        latest={**state,'best_checkpoint':best_state,'optimizer':optimizer.state_dict(),'history':history,
            'noise_rng':{k:g.get_state() for k,g in gs.items()},'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':train.generator.get_state()}
        runner.io.atomic_write(args.output_dir/'latest.pt',latest,binary=True)
        runner.io.atomic_write(args.output_dir/'best.pt',best_state,binary=True)
        write(args.output_dir/'history.json',history);write(args.output_dir/'status.json',{'state':'running','epoch':epoch,'best_epoch':best_epoch})
        if epoch%10==0 or epoch==1:print(metadata['arm'],args.seed,epoch,score,best,flush=True)
    model.load_state_dict(best_state['model']);model.eval()
    pred,y,m,_=runner.io.predict(model,valid,'cuda',scaler,spec)
    replay,_=load_model(args.output_dir/'best.pt');replay.cuda().eval();actual,*_=runner.io.predict(replay,valid,'cuda',scaler,spec)
    torch.testing.assert_close(pred,actual,rtol=1e-6,atol=1e-6)
    train_eval=runner.io.loader(data,splits['train'],args,False);pp,yy,mm,_=runner.io.predict(replay,train_eval,'cuda',scaler,spec)
    return {'best_epoch':best_state['best_epoch'],'epochs_run':len(history),'validation':runner.io.compute_property_metrics(pred,y,m,spec),
        'train':runner.io.compute_property_metrics(pp,yy,mm,spec),'elapsed_seconds':time.perf_counter()-started,'replay_passed':True,
        'noise_rng_sha256':{k:hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest() for k,g in gs.items()}}

def execute(out):
    plan=read(out/'plan.json');assert ROOT==out/'frozen_source'
    for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.45)
    old=Path(plan['old_run']);reference=read(old/'results.json')
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol');splits={k:(splits[k][:plan['limit']] if plan['limit'] else splits[k]) for k in ['train','valid']}
    data=GraphDataset(source.smiles,source.targets,source.target_mask);scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    rows=[];seeds=[0];started=time.perf_counter();gate=False
    for seed in seeds:
        baseline,saved=load_baseline(old/f'scratch_task_seed_{seed}/best.pt')
        ref=next(r for r in reference['runs'] if r['arm']=='scratch' and r['task_seed']==seed)
        common=None
        for arm in plan['arms']:
            model=TaskLatentModel(baseline.config,saved['priors'],saved['mask_ratio'],spec.num_tasks)
            named_initialize(model,seed);model.initialize_priors()
            assert tensor_hash(model.encoder.state_dict())==ref['encoder_initial_hash']
            assert tensor_hash(model.property_head.state_dict())==ref['head_initial_hash']
            model.cuda();torch.manual_seed(seed+10000)
            folder=out/f'{arm}_seed_{seed}';args=SimpleNamespace(output_dir=folder,device='cuda',seed=seed,batch_size=16,resume=False)
            meta={'arm':arm,'seed':seed,'plan_sha256':sha(out/'plan.json'),'splits':splits,
                'reference':ref['validation'],'same_original_initial_encoder_and_head':True,'test_evaluated':False}
            weight=plan['feature_denoise_weight'] if arm=='task_denoising' else 0.
            write(out/'status.json',{'state':'running','arm':arm,'seed':seed,'completed':len(rows)})
            with runner.io.run_directory(args,meta):
                result=fit(model,data,splits,args,scaler,spec,plan,weight,meta)
                result.update(arm=arm,seed=seed,minus_original=result['validation']['rmse']-ref['validation']['rmse'])
                write(folder/'completion.json',result);write(folder/'status.json',{'state':'complete'})
            if common is None:common=result['noise_rng_sha256']
            else:assert common==result['noise_rng_sha256']
            rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print('completed',arm,seed,result,flush=True)
        if seed==0:
            a,b=[next(r for r in rows if r['seed']==0 and r['arm']==arm)['validation']['rmse'] for arm in plan['arms']]
            gate=not plan['smoke'] and b<=a-.01 and b<=ref['validation']['rmse']-.01
            write(out/'decision.json',{'pilot_augmentation':a,'pilot_denoising':b,'original_seed0':ref['validation']['rmse'],
                'replicate':gate,'rule':plan['gate'],'smoke':plan['smoke']})
            if gate:seeds.extend(plan['replication_seeds'])
    aggregate={}
    for arm in plan['arms']:
        v=[r['validation']['rmse'] for r in rows if r['arm']==arm];aggregate[arm]={'n':len(v),'mean_rmse':statistics.mean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None}
    write(out/'results.json',{'runs':rows,'aggregate':aggregate,'pilot_gate_passed':gate,'test_evaluated':False,'smoke':plan['smoke']})
    write(out/'status.json',{'state':'complete','runs':len(rows),'replication_triggered':gate,'wall_seconds':time.perf_counter()-started,'test_evaluated':False,'smoke':plan['smoke']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    execute(a.output_dir.resolve()) if a.execute else prepare(a.output_dir.resolve(),a.smoke)
