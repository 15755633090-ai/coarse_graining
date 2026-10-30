"""Reset only the shared encoder's output projection during transfer."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,shutil,statistics,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import MaskedBondModel,load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset
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
    write(out/'plan.json',{'old_run':str(old),'data_root':str(ROOT/'data'),'device':a.device,'seeds':[0,1,2],'epochs':60,
        'strategy':'load original masked encoder except fuse.*; fuse and property head use original task-seed initialization',
        'learning_rate':.001,'weight_decay':0.,'dropout':.1,'batch_size':16,'clip':5.,'test_evaluated':False,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    print(str(out),flush=True)
def execute(a):
    out=a.output_dir.resolve();plan=read(out/'plan.json');assert ROOT==out/'frozen_source'
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=plan['device']
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.45,0)
    old=Path(plan['old_run']);oldplan=read(old/'plan.json');sources=read(old/'pretrain_sources.json');oldrows=read(old/'results.json')['runs']
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol');splits={k:v for k,v in splits.items() if k in ('train','valid')}
    data=GraphDataset(source.smiles,source.targets,source.target_mask);scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']]);rows=[]
    for seed in [0,1,2]:
        preseed=oldplan['task_to_pretrain_seed'][str(seed)];item=sources[str(preseed)]['masked'];assert sha(Path(item['path']))==item['sha256']
        _,saved=load_model(Path(item['path']));cfg=Config(**saved['config']);cfg.dropout=.1
        model=MaskedBondModel(cfg,saved['priors'],.3,spec.num_tasks);named_initialize(model,seed);model.initialize_priors()
        scratch=next(r for r in oldrows if r['arm']=='scratch' and r['task_seed']==seed)
        initial=model.encoder.state_dict();head_hash=tensor_hash(model.property_head.state_dict())
        assert head_hash==scratch['head_initial_hash'] and tensor_hash(initial)==scratch['encoder_initial_hash']
        mixed={k:initial[k].clone() if k.startswith('fuse.') else v for k,v in saved['encoder'].items()}
        model.encoder.load_state_dict(mixed,strict=True)
        for k,v in model.encoder.state_dict().items():torch.testing.assert_close(v,initial[k] if k.startswith('fuse.') else saved['encoder'][k],atol=0,rtol=0)
        fuse_hash=tensor_hash({k:v for k,v in mixed.items() if k.startswith('fuse.')})
        model.to(device);torch.manual_seed(seed+10000);folder=out/f'reset_fuse_seed_{seed}'
        args=SimpleNamespace(stage='finetune',output_dir=folder,device=device,seed=seed,epochs=60,patience=60,batch_size=16,
            learning_rate=.001,encoder_learning_rate=None,weight_decay=0.,grad_clip=5.,resume=False)
        meta={'seed':seed,'strategy':plan['strategy'],'source':item,'splits':splits,'head_initial_hash':head_hash,'fresh_fuse_hash':fuse_hash,
            'plan_sha256':sha(out/'plan.json'),'test_evaluated':False,'software':runner.io.shared.provenance()['software']}
        write(out/'status.json',{'state':'running','seed':seed,'completed_jobs':len(rows)})
        with runner.io.run_directory(args,meta):
            train=runner.io.loader(data,splits['train'],args,True);valid=runner.io.loader(data,splits['valid'],args,False)
            result=runner.fit(model,train,valid,args,meta,spec,scaler,None);pred,y,mask,_=runner.io.predict(model,valid,device,scaler,spec)
            replay,_=load_model(folder/'best.pt');replay.to(device);actual,*_=runner.io.predict(replay,valid,device,scaler,spec)
            torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
            result.update(seed=seed,validation=runner.io.compute_property_metrics(pred,y,mask,spec),scratch_rmse=scratch['validation']['rmse'],
                original_masked_rmse=next(r for r in oldrows if r['arm']=='masked' and r['task_seed']==seed)['validation']['rmse'],
                initialization_and_partial_load_verified=True,checkpoint_replay_passed=True)
            write(folder/'completion.json',result);write(folder/'status.json',{'state':'complete',**result})
        rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print(seed,result['validation']['rmse'],flush=True)
    values=[r['validation']['rmse'] for r in rows]
    write(out/'results.json',{'runs':rows,'aggregate':{'n':3,'mean_rmse':statistics.mean(values),'sample_sd':statistics.stdev(values)},
        'paired':[{'seed':r['seed'],'reset_minus_scratch':r['validation']['rmse']-r['scratch_rmse'],'reset_minus_full_transfer':r['validation']['rmse']-r['original_masked_rmse']} for r in rows],
        'scratch_reference':'original unchanged architecture, initial hashes and dropout/loader/optimizer settings matched','test_evaluated':False})
    write(out/'status.json',{'state':'complete','completed_jobs':3,'all_replays_passed':True,'partial_load_verified':True,'test_evaluated':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.execute:
        try:execute(a)
        except Exception as e:write(a.output_dir/'status.json',{'state':'failed','error':repr(e)});raise
    else:prepare(a)
