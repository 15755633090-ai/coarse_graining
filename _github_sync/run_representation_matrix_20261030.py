"""Predeclared objective x corpus experiment; no hyperparameter/seed search."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,shutil,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch_geometric.data import Batch
from rdkit import Chem
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import MaskedBondModel,mask_observation,load_model as load_edge
from motif.adjacency_diffusion.representation_pretrain import RepresentationModel,load_model as load_repr,representation_objective,sigreg
from motif.adjacency_diffusion.typed_bonds import GraphDataset,edge_target,corrupt,reconstruction_objective
from motif.adjacency_diffusion.typed_refined import training_priors
from motif.adjacency_diffusion.data import condition_from_batch
from motif.adjacency_diffusion.model import Config
from _github_sync.search_adjacency_rescue_20261029 import named_initialize
from _github_sync.check_masked_property_20261029 import tensor_hash

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):runner.io.atomic_write(p,x)
def canonical(s):
    mol=Chem.MolFromSmiles(s)
    if mol is None:raise ValueError('Invalid corpus molecule')
    return Chem.MolToSmiles(mol,canonical=True)

def prepare(a):
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029';prior=read(old/'plan.json')
    source,_,_=runner.io.build_property_data(ROOT/'data','esol');excluded={canonical(s) for s in source.smiles}
    heldout=set(prior['development_smiles'])|set(prior['confirmation_smiles'])|set(prior['challenge_smiles'])
    pool=ROOT/'motif/brics_atom_diffusion/runs/pilot_zinc4096_esol3_20261029/corpus/train.csv'
    with pool.open(encoding='utf-8-sig',newline='') as f:raw=[r['smiles'] for r in csv.DictReader(f)]
    expanded=sorted({canonical(s) for s in raw}-excluded-heldout)
    small=prior['train_smiles'];assert not set(small)&excluded and set(small)<=set(expanded)
    if len(expanded)<=len(small):raise ValueError('No genuine corpus expansion')
    corpora={'small':small,'expanded':expanded}
    if a.smoke:corpora={k:v[:8] for k,v in corpora.items()}
    files=sorted(set([ROOT/p for p in prior['source_sha256']]+list((ROOT/'motif/adjacency_diffusion').glob('*.py'))+[Path(__file__).resolve()]))
    for p in files:
        q=out/'frozen_source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    plan={'scope':'objective family x unique corpus coverage, reused ESOL validation; no test evaluation',
        'old_run':str(old),'data_root':str(ROOT/'data'),'device':a.device,'smoke':a.smoke,
        'corpora':corpora,'corpus_sizes':{k:len(v) for k,v in corpora.items()},'expanded_source_sha256':sha(pool),
        'excluded_esol_molecules':len(excluded),'excluded_prior_holdouts':len(heldout),
        'confirm':prior['confirmation_smiles'][:8] if a.smoke else prior['confirmation_smiles'],
        'seeds':[7] if a.smoke else [7,1,2],'task_seeds':{'7':0,'1':1,'2':2},
        'objectives':['edge','representation'],'steps':2 if a.smoke else 1200,'batch_size':8,
        'epochs':2 if a.smoke else 60,'task_limit':16 if a.smoke else 0,'hidden_dim':12 if a.smoke else 64,'layers':1 if a.smoke else 3,
        'mask_ratio':.3,'time_range':[1,1000],'learning_rate':.001,'weight_decay':0.,'grad_clip':5.,
        'regularizer_weight':.05,'directions':256,'representation_level':'same mean atom readout as prediction',
        'views':'two global views, both differentiated; no EMA or stopgrad; no additional trained projector',
        'recipe_status':'LeJEPA-inspired graph adaptation; batch8 is below recommended vision settings; not a reproduction',
        'budget':'same graph draws and optimizer updates within corpus; representation uses two differentiable encoder passes versus one; not compute matched',
        'rng_streams':['graph','time','Gaussian','mask','SIGReg'],
        'selection':'final pretrain step; full downstream epochs; all seeds; no loss-based source selection',
        'decision':'compare data and objective patterns; if a method improves, equal-compute edge control and independent task are required before a gain claim',
        'test_evaluated':False,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    write(out/'plan.json',plan);print(json.dumps({'prepared':str(out),'corpora':plan['corpus_sizes']}),flush=True)

@torch.no_grad()
def diagnostics(model,smiles,device):
    model.eval();data=GraphDataset(smiles);clean_values=[];noisy_values=[]
    g=torch.Generator(device=device).manual_seed(88713)
    for i in range(0,len(data),8):
        batch=Batch.from_data_list([data[j] for j in range(i,min(i+8,len(data)))]).to(device)
        c=condition_from_batch(batch,True);x=edge_target(batch,c);t=torch.full((len(x),),200,device=device,dtype=torch.long)
        observed,_=model.prepare_observation(x,t,c.pairs,g)
        clean=torch.cat((x,torch.zeros_like(x[...,:1])),-1)
        for obs,ts,collector in [(clean,torch.zeros_like(t),clean_values),(observed,t,noisy_values)]:
            z=model.encoder(c,obs,ts)['atoms'];collector.append((z.sum(1)/c.nodes.sum(1)[:,None]).cpu())
    a,b=torch.cat(clean_values),torch.cat(noisy_values)
    def stats(x):
        centered=x-x.mean(0);s=torch.linalg.svdvals(centered).square();total=float(s.sum())
        p=s/s.sum().clamp_min(1e-12);rank=float((-(p*p.clamp_min(1e-12).log()).sum()).exp()) if total>1e-12 else 0.
        return {'mean_dimension_std':float(x.std(0).mean()),'effective_rank':rank,'mean_norm':float(x.norm(dim=1).mean()),
                'sigreg':float(sigreg(x,torch.Generator().manual_seed(37))),
                'near_constant_flag':float(x.std(0).mean())<1e-3,'low_effective_rank_flag':rank<2}
    return {'n':len(a),'clean':stats(a),'corrupted_t200':stats(b),'view_mse':float((a-b).square().mean()),
            'scope':'unlabeled reused heldout molecules; collapse thresholds are diagnostics, not guarantees or selection criteria'}

def execute(a):
    out=a.output_dir.resolve();plan=read(out/'plan.json');assert ROOT==out/'frozen_source'
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=plan['device']
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.45,0)
    old=Path(plan['old_run']);old_sources=read(old/'pretrain_sources.json');old_results=read(old/'results.json')
    sources={};rows=[];started=time.perf_counter()
    for corpus,smiles in plan['corpora'].items():
        dataset=GraphDataset(smiles);priors=training_priors(dataset)
        counts=[]
        for graph in dataset:
            u,v=graph.edge_index;cross=(graph.fragment_id[u]!=graph.fragment_id[v])&(u<v)
            counts.append({'atoms':len(graph.x),'fragments':int(graph.fragment_count.item()),'cross_bonds':int(cross.sum())})
        write(out/f'{corpus}_coverage.json',{'unique_molecules':len(smiles),
            'mean_atoms':statistics.mean(r['atoms'] for r in counts),'mean_fragments':statistics.mean(r['fragments'] for r in counts),
            'multi_fragment_fraction':statistics.mean(r['fragments']>1 for r in counts),
            'mean_cross_bonds':statistics.mean(r['cross_bonds'] for r in counts),
            'scope':'coarse structural coverage only; not evidence of covering every motif/environment type'})
        write(out/f'{corpus}_priors.json',{k:v.tolist() for k,v in priors.items()})
        for seed in plan['seeds']:
            for objective in plan['objectives']:
                key=f'{corpus}_{objective}_{seed}';folder=out/'pretraining'/key;folder.mkdir(parents=True,exist_ok=False)
                if corpus=='small' and objective=='edge' and not plan['smoke']:
                    item=old_sources[str(seed)]['masked'];assert sha(Path(item['path']))==item['sha256']
                    model,saved=load_edge(Path(item['path']));model.to(device)
                    record={'reused_original':True,'path':item['path'],'checkpoint_sha256':item['sha256'],
                        'initial_state_hash':item['initial_state_hash'],'diagnostics':diagnostics(model,plan['confirm'],device)}
                    write(folder/'completion.json',record);sources[key]=record;continue
                cfg=Config(hidden_dim=plan['hidden_dim'],layers=plan['layers'],dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random')
                model=RepresentationModel(cfg,priors,.3,regularizer_weight=.05,directions=256) if objective=='representation' else MaskedBondModel(cfg,priors,.3)
                named_initialize(model,seed);model.initialize_priors();initial=tensor_hash(model.state_dict())
                model.to(device).set_stage('pretrain');params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=.001)
                streams=[torch.Generator(device='cpu' if i==0 else device).manual_seed(seed+k) for i,k in enumerate([333,444,555,666,777])]
                history=[];seen=set();candidate_count=bond_count=0;wall=time.perf_counter();gpu_ms=0.;events=[]
                for step in range(1,plan['steps']+1):
                    model.train();ids=torch.randperm(len(dataset),generator=streams[0])[:8].tolist();seen.update(ids)
                    batch=Batch.from_data_list([dataset[i] for i in ids]).to(device);c=condition_from_batch(batch,True);target=edge_target(batch,c)
                    t=torch.randint(1,1001,(len(target),),device=device,generator=streams[1]);noisy=corrupt(target,t,c.pairs,model.schedule,streams[2])
                    observed,selected=mask_observation(noisy,c.pairs,.3,streams[3]);opt.zero_grad(set_to_none=True)
                    if device.startswith('cuda'):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                    if objective=='representation':
                        v1,v2=model.representations_from_observation(c,target,observed,t)
                        loss,parts=representation_objective(v1,v2,streams[4],.05,256)
                    else:loss,parts=reconstruction_objective(model.predict_bonds(c,observed,t),target,selected)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pretraining objective')
                    loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True));opt.step()
                    if device.startswith('cuda'):end.record();events.append((begin,end))
                    upper=torch.triu(selected,diagonal=1);candidate_count+=int(upper.sum());bond_count+=int((upper&target[...,0].eq(0)).sum())
                    if step%100==0 or step==plan['steps']:
                        if device.startswith('cuda'):
                            torch.cuda.synchronize();gpu_ms+=sum(x.elapsed_time(y) for x,y in events);events=[]
                        history.append({'step':step,'loss':float(loss.detach()),'parts':{k:float(v.detach()) for k,v in parts.items()},'pre_clip_norm':norm,'unique_molecules_seen':len(seen)})
                        write(folder/'history.json',history);write(out/'status.json',{'state':'running','stage':'pretraining','cell':key,'step':step,'completed_sources':len(sources)})
                        state=model.checkpoint('pretrain',steps=step,optimizer=opt.state_dict(),rng_states=[g.get_state() for g in streams],
                            history=history,run_plan_sha256=sha(out/'plan.json'),run_metadata={'inputs':{'train_smiles':smiles,'valid_smiles':plan['confirm']},'plan':plan})
                        runner.io.atomic_write(folder/'latest.pt',state,binary=True)
                checkpoint=folder/'encoder.pt';runner.io.atomic_write(checkpoint,state,binary=True)
                restored,saved=(load_repr if objective=='representation' else load_edge)(checkpoint)
                assert tensor_hash(restored.state_dict())==tensor_hash(model.cpu().state_dict())
                model.to(device);diag=diagnostics(model,plan['confirm'],device)
                record={'path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'initial_state_hash':initial,'diagnostics':diag,
                    'checkpoint_replay_passed':True,'unique_molecules_seen':len(seen),'graph_draws':plan['steps']*8,
                    'encoder_graph_passes':plan['steps']*8*(2 if objective=='representation' else 1),'gpu_compute_ms':gpu_ms,
                    'wall_seconds':time.perf_counter()-wall,'masked_candidate_positions':candidate_count,'masked_true_bonds':bond_count}
                write(folder/'completion.json',record);sources[key]=record;write(out/'sources.json',sources)
                print('pretraining',key,'complete',diag['clean'],flush=True)
            x,y=[sources[f'{corpus}_{obj}_{seed}'] for obj in plan['objectives']]
            assert x['initial_state_hash']==y['initial_state_hash']
    write(out/'sources.json',sources)
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    splits={k:v[:plan['task_limit']] if plan['task_limit'] else v for k,v in splits.items() if k in ('train','valid')}
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    for corpus in plan['corpora']:
        for seed in plan['seeds']:
            task_seed=plan['task_seeds'][str(seed)]
            for objective in plan['objectives']:
                key=f'{corpus}_{objective}_{seed}';item=sources[key]
                if corpus=='small' and objective=='edge' and not plan['smoke']:
                    row=next(r for r in old_results['runs'] if r['arm']=='masked' and r['task_seed']==task_seed).copy()
                    baseline,_=load_edge(old/f'masked_task_seed_{task_seed}/best.pt');baseline.to(device)
                    args_ref=SimpleNamespace(device=device,batch_size=16,seed=task_seed)
                    valid_ref=runner.io.loader(data,splits['valid'],args_ref,False)
                    p,y,m,_=runner.io.predict(baseline,valid_ref,device,scaler,spec)
                    assert abs(runner.io.compute_property_metrics(p,y,m,spec)['rmse']-row['validation']['rmse'])<1e-6
                    row.update(corpus=corpus,objective=objective,seed=seed,task_seed=task_seed,reused_original=True);rows.append(row);continue
                _,saved=(load_repr if objective=='representation' else load_edge)(Path(item['path']))
                cfg=Config(**saved['config']);cfg.dropout=.1
                # All downstream models are the SAME old predictor class and readout.
                model=MaskedBondModel(cfg,saved['priors'],.3,spec.num_tasks);named_initialize(model,task_seed);model.initialize_priors()
                head_hash=tensor_hash(model.property_head.state_dict())
                if not plan['smoke']:
                    scratch=next(r for r in old_results['runs'] if r['arm']=='scratch' and r['task_seed']==task_seed)
                    assert head_hash==scratch['head_initial_hash']
                model.encoder.load_state_dict(saved['encoder'],strict=True);model.to(device);torch.manual_seed(task_seed+10000)
                folder=out/'properties'/key
                args=SimpleNamespace(stage='finetune',output_dir=folder,device=device,seed=task_seed,epochs=plan['epochs'],patience=plan['epochs'],batch_size=16,
                    learning_rate=.001,encoder_learning_rate=None,weight_decay=0.,grad_clip=5.,resume=False)
                metadata={'corpus':corpus,'objective':objective,'seed':seed,'task_seed':task_seed,'source':item,'splits':splits,'head_initial_hash':head_hash,
                    'plan_sha256':sha(out/'plan.json'),'test_evaluated':False}
                write(out/'status.json',{'state':'running','stage':'property','cell':key,'completed_properties':len(rows)})
                with runner.io.run_directory(args,metadata):
                    train=runner.io.loader(data,splits['train'],args,True);valid=runner.io.loader(data,splits['valid'],args,False)
                    result=runner.fit(model,train,valid,args,metadata,spec,scaler,None);pred,y,mask,_=runner.io.predict(model,valid,device,scaler,spec)
                    replay,_=load_edge(folder/'best.pt');replay.to(device);actual,*_=runner.io.predict(replay,valid,device,scaler,spec)
                    torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
                    result.update(corpus=corpus,objective=objective,seed=seed,task_seed=task_seed,validation=runner.io.compute_property_metrics(pred,y,mask,spec),checkpoint_replay_passed=True)
                    write(folder/'completion.json',result);write(folder/'status.json',{'state':'complete',**result})
                rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print('property',key,result['validation']['rmse'],flush=True)
    aggregate={}
    for corpus in plan['corpora']:
        for objective in plan['objectives']:
            values=[r['validation']['rmse'] for r in rows if r['corpus']==corpus and r['objective']==objective]
            aggregate[f'{corpus}_{objective}']={'n':len(values),'mean_rmse':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None}
    write(out/'results.json',{'runs':rows,'aggregate':aggregate,'scratch_reference':None if plan['smoke'] else old_results['aggregate']['scratch'],
        'budget_note':plan['budget'],'test_evaluated':False})
    write(out/'status.json',{'state':'complete','sources':len(sources),'properties':len(rows),'new_property_runs':len(rows)-(0 if plan['smoke'] else 3),
        'all_replays_passed':True,'initialization_pairing_passed':True,'test_evaluated':False,'wall_seconds':time.perf_counter()-started})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.execute:
        try:execute(a)
        except Exception as e:write(a.output_dir/'status.json',{'state':'failed','error':repr(e)});raise
    else:prepare(a)
