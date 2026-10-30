"""Single preregistered local Gaussian-posterior weighting correction; three seeds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,hashlib,json,shutil,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch_geometric.data import Batch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import MaskedBondModel,mask_observation,load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset,edge_target,corrupt,reconstruction_objective
from motif.adjacency_diffusion.data import condition_from_batch
from motif.adjacency_diffusion.model import Config
from _github_sync.search_adjacency_rescue_20261029 import named_initialize
from _github_sync.check_masked_property_20261029 import tensor_hash
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):runner.io.atomic_write(p,x)

def prepare(a):
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029';prior=read(old/'plan.json')
    files=sorted(set([ROOT/p for p in prior['source_sha256']]+list((ROOT/'motif/adjacency_diffusion').glob('*.py'))+[Path(__file__).resolve()]))
    for p in files:
        q=out/'frozen_source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    write(out/'plan.json',{'old_run':str(old),'data_root':str(ROOT/'data'),'device':a.device,'seeds':[0,1,2],
        'pretrain_steps':1200,'epochs':60,'learning_rate':.001,'dropout':.1,'weight_decay':0.,
        'mask_ratio':.3,'normalization':'candidate','weighting':'gaussian_posterior','training_max_t':a.max_t,
        'scope':'single weighting correction; do not tune against validation outcomes',
        'reuse_scratch':'same clean computation, trainable initialization, loader/dropout RNG and optimizer; check exact predictions and gradients',
        'test_evaluated':False,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    print(str(out),flush=True)

def execute(a):
    out=a.output_dir.resolve();plan=read(out/'plan.json');assert ROOT==out/'frozen_source'
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=plan['device']
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.45,0)
    old=Path(plan['old_run']);oldplan=read(old/'plan.json');sources=read(old/'pretrain_sources.json');prior_results=read(old/'results.json')
    train=GraphDataset(oldplan['train_smiles']);source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    splits={k:v for k,v in splits.items() if k in ('train','valid')};data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']]);rows=[]
    for seed in plan['seeds']:
        preseed=oldplan['task_to_pretrain_seed'][str(seed)];item=sources[str(preseed)]['masked'];assert sha(Path(item['path']))==item['sha256']
        _,saved=load_model(Path(item['path']));cfg=Config(**saved['config']);cfg.observation_weighting='gaussian_posterior'
        model=MaskedBondModel(cfg,saved['priors'],.3);model.training_max_t=plan['training_max_t'];named_initialize(model,preseed);model.initialize_priors()
        softcfg=copy.deepcopy(cfg);softcfg.observation_weighting='softmax'
        reference=MaskedBondModel(softcfg,saved['priors'],.3);named_initialize(reference,preseed);reference.initialize_priors()
        for (n,p),(m,q) in zip(model.named_parameters(),reference.named_parameters()):
            assert n==m;torch.testing.assert_close(p,q,atol=0,rtol=0)
        del reference
        model.to(device).set_stage('pretrain');params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=.001)
        rngs=[torch.Generator(device='cpu' if j==0 else device).manual_seed(preseed+k) for j,k in enumerate([333,444,555,666])]
        folder=out/'pretraining'/f'posterior_seed_{preseed}';folder.mkdir(parents=True,exist_ok=False);history=[];pairs=0;bonds=0
        for step in range(1,1201):
            model.train();ids=torch.randperm(len(train),generator=rngs[0])[:8].tolist()
            batch=Batch.from_data_list([train[i] for i in ids]).to(device);c=condition_from_batch(batch,True);target=edge_target(batch,c)
            t=torch.randint(1,plan['training_max_t']+1,(len(target),),device=device,generator=rngs[1]);noisy=corrupt(target,t,c.pairs,model.schedule,rngs[2])
            observed,selected=mask_observation(noisy,c.pairs,.3,rngs[3]);opt.zero_grad(set_to_none=True)
            loss,_=reconstruction_objective(model.predict_bonds(c,observed,t),target,selected);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True));opt.step()
            upper=torch.triu(selected,diagonal=1);pairs+=int(upper.sum());bonds+=int((upper&target[...,0].eq(0)).sum())
            if step%100==0:
                history.append({'step':step,'loss':float(loss.detach()),'gradient_norm':norm});write(folder/'history.json',history)
                write(out/'status.json',{'state':'running','stage':'pretraining','seed':seed,'step':step,'completed_jobs':len(rows)})
        previous=read(old/'pretraining'/f'masked_seed_{preseed}'/'completion.json')
        assert pairs==previous['supervised_candidate_pairs_accumulated'] and bonds==previous['supervised_true_bonds_accumulated']
        cp=folder/'encoder.pt';runner.io.atomic_write(cp,model.checkpoint('pretrain',steps=1200),binary=True)
        replay,_=load_model(cp);assert tensor_hash(replay.state_dict())==tensor_hash(model.cpu().state_dict())
        evalargs=SimpleNamespace(device=device,seed=preseed,batch_size=8)
        valid=runner.io.loader(GraphDataset(oldplan['confirmation_smiles']),list(range(128)),evalargs,False)
        replay.to(device);recovery=runner.evaluate(replay,valid,evalargs)
        write(folder/'completion.json',{'checkpoint_sha256':sha(cp),'checkpoint_replay_passed':True,'recovery_confirmation':recovery,
            'supervised_candidate_pairs_accumulated':pairs,'supervised_true_bonds_accumulated':bonds,'target_counts_match_original':True,
            'trainable_initialization_matches_original':True})
        cfg.dropout=.1;task=MaskedBondModel(cfg,saved['priors'],.3,spec.num_tasks);task.training_max_t=plan['training_max_t'];named_initialize(task,seed);task.initialize_priors()
        # Show scratch forward and gradients are identical, despite the extra nonlearned prior/schedule buffers.
        softcfg=copy.deepcopy(cfg);softcfg.observation_weighting='softmax'
        baseline=MaskedBondModel(softcfg,saved['priors'],.3,spec.num_tasks);named_initialize(baseline,seed);baseline.initialize_priors()
        baseline.set_stage('finetune');task.set_stage('finetune')
        check_batch=Batch.from_data_list([data[i] for i in splits['train'][:16]])
        torch.manual_seed(seed+10000);pa=baseline(check_batch);pa.square().mean().backward()
        torch.manual_seed(seed+10000);pb=task(check_batch);pb.square().mean().backward();torch.testing.assert_close(pa,pb,atol=0,rtol=0)
        for (n,p),(m,q) in zip(baseline.named_parameters(),task.named_parameters()):
            assert n==m
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        head_hash=tensor_hash(task.property_head.state_dict())
        scratch=next(r for r in prior_results['runs'] if r['arm']=='scratch' and r['task_seed']==seed)
        assert head_hash==scratch['head_initial_hash']
        _,new_saved=load_model(cp);task.encoder.load_state_dict(new_saved['encoder'],strict=True);task.to(device);task.zero_grad(set_to_none=True)
        del model,replay,baseline,opt
        torch.manual_seed(seed+10000);task_folder=out/f'posterior_task_seed_{seed}'
        fitargs=SimpleNamespace(stage='finetune',output_dir=task_folder,device=device,seed=seed,epochs=60,patience=60,batch_size=16,
            learning_rate=.001,encoder_learning_rate=None,weight_decay=0.,grad_clip=5.,resume=False)
        meta={'seed':seed,'weighting':'gaussian_posterior','pretraining_checkpoint_sha256':sha(cp),'plan_sha256':sha(out/'plan.json'),
            'splits':splits,'head_initial_hash':head_hash,'clean_forward_and_gradients_identical_to_scratch':True,'training_max_t':plan['training_max_t'],'test_evaluated':False}
        write(out/'status.json',{'state':'running','stage':'property','seed':seed,'completed_jobs':len(rows)})
        with runner.io.run_directory(fitargs,meta):
            tr=runner.io.loader(data,splits['train'],fitargs,True);va=runner.io.loader(data,splits['valid'],fitargs,False)
            result=runner.fit(task,tr,va,fitargs,meta,spec,scaler,None);pred,y,mask,_=runner.io.predict(task,va,device,scaler,spec)
            replay,_=load_model(task_folder/'best.pt');replay.to(device);actual,*_=runner.io.predict(replay,va,device,scaler,spec)
            torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
            result.update(seed=seed,validation=runner.io.compute_property_metrics(pred,y,mask,spec),scratch_rmse=scratch['validation']['rmse'],
                original_masked_rmse=next(r for r in prior_results['runs'] if r['arm']=='masked' and r['task_seed']==seed)['validation']['rmse'],
                clean_scratch_equivalence_passed=True,checkpoint_replay_passed=True)
            write(task_folder/'completion.json',result)
        rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print(seed,result['validation']['rmse'],flush=True)
    aggregate={}
    for arm,key in [('posterior',None),('scratch','scratch_rmse'),('original_masked','original_masked_rmse')]:
        values=[r[key] if key else r['validation']['rmse'] for r in rows]
        aggregate[arm]={'mean_rmse':statistics.mean(values),'sd':statistics.stdev(values),'n':3}
    write(out/'results.json',{'runs':rows,'aggregate':aggregate,'test_evaluated':False,'scratch_reference':'reused original paired runs; exact clean computation/gradient equivalence checked'})
    write(out/'status.json',{'state':'complete','completed_pretrain_jobs':3,'completed_property_jobs':3,'all_replays_passed':True,'all_target_counts_matched':True,'test_evaluated':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--max-t',type=int,default=1000);a=p.parse_args()
    if not 1<=a.max_t<=1000:p.error('--max-t must be in [1,1000]')
    if a.execute:
        try:execute(a)
        except Exception as e:write(a.output_dir/'status.json',{'state':'failed','error':repr(e)});raise
    else:prepare(a)
