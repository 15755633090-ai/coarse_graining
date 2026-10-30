"""One frozen chemical-target recipe, three seeds; no validation-driven search."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib,shutil,time,statistics,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch_geometric.data import Batch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import MaskedBondModel,load_model
from motif.adjacency_diffusion.chemical_targets import chemical_targets,target_statistics,knowledge_loss,graph_readout,DESCRIPTORS
from motif.adjacency_diffusion.typed_bonds import GraphDataset
from motif.adjacency_diffusion.typed_refined import training_priors
from motif.adjacency_diffusion.model import Config
from _github_sync.search_adjacency_rescue_20261029 import named_initialize
from _github_sync.check_masked_property_20261029 import tensor_hash

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):runner.io.atomic_write(p,x)

def prepare(out):
    out.mkdir(exist_ok=False,parents=True)
    original=ROOT/'motif/adjacency_diffusion/runs/representation_matrix_20261030'
    prior=read(original/'plan.json')
    paths=set(prior['source_sha256'])|{'motif/adjacency_diffusion/chemical_targets.py','motif/adjacency_diffusion/test_chemical_targets.py','_github_sync/run_chemical_anchor_20261030.py'}
    for rel in paths:
        dest=out/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    plan={'purpose':'test fixed chemical targets through the exact downstream clean graph readout',
        'recipe':'clean graph -> same BRICS encoder/mean readout -> linear 12 descriptors +512 RDKFP logits',
        'corpus':prior['corpora']['expanded'],'confirm':prior['confirm'],'steps':1200,'batch_size':8,
        'seeds':[7,1,2],'task_seeds':[0,1,2],'epochs':60,'learning_rate':.001,'grad_clip':5.,
        'descriptor_names':[n for n,_ in DESCRIPTORS],'fingerprint':'RDKit RDKFingerprint fpSize512 defaults',
        'objective':'train-corpus standardized descriptor mean MSE + active-bit mean BCE with train-only pos_weight capped20',
        'transfer':'encoder only, original fresh property head; original full fine-tune recipe',
        'changes':'joint change of target and clean-input training; not a single-factor diffusion ablation or KPGT reproduction',
        'selection':'final pretrain step for all3 seeds; best validation epoch within60 downstream; no recipe selection',
        'decision':'compare original scratch and original expanded-edge references; retain all failed seeds; no automatic follow-up tuning',
        'knowledge_disclosure':'calculated 2D chemical targets, including MolLogP related to solubility; no measured labels upstream; not label-free of chemical priors',
        'heldout_policy':'same 4096 corpus excludes canonical ESOL molecules; descriptor stats fitted upstream only; confirmation not used for stopping',
        'data_root':str(ROOT/'data'),'old_run':str(ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029'),
        'original_matrix':str(original),'test_evaluated':False,
        'source_sha256':{rel:sha(ROOT/rel) for rel in sorted(paths)}}
    write(out/'plan.json',plan)

@torch.no_grad()
def diagnose(model,head,dataset,d,f,stats):
    model.eval();head.eval();zs=[];ps=[]
    for i in range(0,len(dataset),8):
        batch=Batch.from_data_list([dataset[j] for j in range(i,min(i+8,len(dataset)))]).to('cuda')
        z=graph_readout(model,batch);zs.append(z.cpu());ps.append(head(z))
    pred=torch.cat(ps);loss,parts=knowledge_loss(pred,d,f,stats)
    constant=torch.zeros_like(pred);freq=stats['positive_weight']
    # Exact optimum of weighted constant BCE on upstream prevalence.
    constant[:,len(DESCRIPTORS):]=stats['constant_logits']
    base,baseparts=knowledge_loss(constant,d,f,stats)
    z=torch.cat(zs);s=torch.linalg.svdvals(z-z.mean(0)).square();p=s/s.sum().clamp_min(1e-12)
    return {'n':len(dataset),'objective':float(loss),'constant_objective':float(base),
        'descriptor_mse':float(parts['descriptor_mse']),'constant_descriptor_mse':float(baseparts['descriptor_mse']),
        'fingerprint_bce':float(parts['fingerprint_bce']),'constant_fingerprint_bce':float(baseparts['fingerprint_bce']),
        'effective_rank':float((-(p*p.clamp_min(1e-12).log()).sum()).exp()),'dimension_std':float(z.std(0).mean())}

def execute(out):
    plan=read(out/'plan.json');assert ROOT==out/'frozen_source'
    for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.45)
    dataset=GraphDataset(plan['corpus']);d,f=chemical_targets(plan['corpus']);stats=target_statistics(d,f)
    freq=f.mean(0).clamp(1e-6,1-1e-6);stats['constant_logits']=(freq*stats['positive_weight']/(1-freq)).log()
    write(out/'target_statistics.json',{k:v.tolist() for k,v in stats.items()})
    stats={k:v.cuda() for k,v in stats.items()};d=d.cuda();f=f.cuda()
    hold=GraphDataset(plan['confirm']);hd,hf=chemical_targets(plan['confirm']);hd=hd.cuda();hf=hf.cuda()
    priors=training_priors(dataset);old=Path(plan['old_run']);old_results=read(old/'results.json');rows=[];sources={}
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    splits={k:splits[k] for k in ('train','valid')};property_data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    started=time.perf_counter()
    for seed,task_seed in zip(plan['seeds'],plan['task_seeds']):
        folder=out/f'pretraining/seed_{seed}';folder.mkdir(parents=True)
        model=MaskedBondModel(Config(hidden_dim=64,layers=3,dropout=0.,motif_prior=True,conditional_gate=True,time_mode='random'),priors,.3)
        named_initialize(model,seed);model.initialize_priors();initial=tensor_hash(model.encoder.state_dict())
        # Auxiliary head RNG is independent of named backbone/task initialization.
        torch.manual_seed(seed+90000);head=torch.nn.Linear(64,len(DESCRIPTORS)+512).cuda()
        model.cuda().set_stage('pretrain')
        for p in model.bond_head.parameters():p.requires_grad_(False)
        params=[p for p in model.parameters() if p.requires_grad]+list(head.parameters())
        opt=torch.optim.Adam(params,lr=.001);g=torch.Generator().manual_seed(seed+333)
        before=diagnose(model,head,hold,hd,hf,stats);history=[];seen=set();wall=time.perf_counter()
        for step in range(1,plan['steps']+1):
            model.train();head.train();ids=torch.randperm(len(dataset),generator=g)[:8].tolist();seen.update(ids)
            batch=Batch.from_data_list([dataset[i] for i in ids]).to('cuda');opt.zero_grad(set_to_none=True)
            loss,parts=knowledge_loss(head(graph_readout(model,batch)),d[ids],f[ids],stats)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite chemical objective')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True));opt.step()
            if step%100==0:
                history.append({'step':step,'loss':float(loss),'parts':{k:float(v) for k,v in parts.items()},'pre_clip_norm':norm})
                write(folder/'history.json',history);write(out/'status.json',{'state':'running','stage':'pretrain','seed':seed,'step':step})
        after=diagnose(model,head,hold,hd,hf,stats)
        checkpoint=folder/'encoder.pt'
        state=model.checkpoint('pretrain',steps=1200,chemical_target_head=head.state_dict(),target_statistics=stats,
            optimizer=opt.state_dict(),sampling_rng=g.get_state(),history=history,run_metadata={'plan_sha256':sha(out/'plan.json')})
        runner.io.atomic_write(checkpoint,state,binary=True)
        replay,saved=load_model(checkpoint);assert tensor_hash(replay.encoder.state_dict())==tensor_hash(model.encoder.state_dict())
        item={'path':str(checkpoint),'sha256':sha(checkpoint),'initial_encoder_hash':initial,'before':before,'after':after,
              'unique_seen':len(seen),'graph_draws':9600,'encoder_graph_passes':9600,'wall_seconds':time.perf_counter()-wall}
        sources[str(seed)]=item;write(out/'sources.json',sources);print('pretrain',seed,after,flush=True)
        cfg=Config(**saved['config']);cfg.dropout=.1;task=MaskedBondModel(cfg,priors,.3,spec.num_tasks)
        named_initialize(task,task_seed);task.initialize_priors()
        reference=next(r for r in old_results['runs'] if r['arm']=='scratch' and r['task_seed']==task_seed)
        headhash=tensor_hash(task.property_head.state_dict());assert headhash==reference['head_initial_hash']
        task.encoder.load_state_dict(saved['encoder'],strict=True);task.cuda();torch.manual_seed(task_seed+10000)
        dest=out/f'properties/seed_{task_seed}'
        args=SimpleNamespace(stage='finetune',output_dir=dest,device='cuda',seed=task_seed,epochs=60,patience=60,batch_size=16,learning_rate=.001,encoder_learning_rate=None,weight_decay=0.,grad_clip=5.,resume=False)
        meta={'source':item,'splits':splits,'task_seed':task_seed,'head_initial_hash':headhash,'plan_sha256':sha(out/'plan.json'),'test_evaluated':False}
        with runner.io.run_directory(args,meta):
            train=runner.io.loader(property_data,splits['train'],args,True);valid=runner.io.loader(property_data,splits['valid'],args,False)
            write(out/'status.json',{'state':'running','stage':'finetune','seed':task_seed})
            result=runner.fit(task,train,valid,args,meta,spec,scaler,None)
            pred,y,m,_=runner.io.predict(task,valid,'cuda',scaler,spec)
            replay,_=load_model(dest/'best.pt');replay.cuda();actual,*_=runner.io.predict(replay,valid,'cuda',scaler,spec)
            torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
            result.update(task_seed=task_seed,seed=seed,validation=runner.io.compute_property_metrics(pred,y,m,spec),
                minus_scratch=runner.io.compute_property_metrics(pred,y,m,spec)['rmse']-reference['validation']['rmse'],replay_passed=True)
            write(dest/'completion.json',result)
        rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print('property',task_seed,result,flush=True)
    values=[r['validation']['rmse'] for r in rows]
    write(out/'results.json',{'runs':rows,'mean_rmse':statistics.mean(values),'sd_rmse':statistics.stdev(values),
        'scratch_reference':old_results['aggregate']['scratch'],'expanded_edge_reference':read(Path(plan['original_matrix'])/'results.json')['aggregate']['expanded_edge'],
        'test_evaluated':False,'limits':'reused validation, n3; joint target/input recipe; no independent task replication'})
    write(out/'status.json',{'state':'complete','new_pretrain_runs':3,'new_property_runs':3,'all_replays_passed':True,'wall_seconds':time.perf_counter()-started,'test_evaluated':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else prepare)(a.output_dir.resolve())
