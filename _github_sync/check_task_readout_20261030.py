"""Task-only adapter, matched scratch/pretrained pairs, no upstream retraining."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,shutil,statistics,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import load_model as load_upstream
from motif.adjacency_diffusion.task_readout import TaskReadoutModel,load_model
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
    write(out/'plan.json',{'old_run':str(old),'data_root':str(ROOT/'data'),'device':a.device,'seeds':[0] if a.smoke else [0,1,2],
        'epochs':2 if a.smoke else 60,'limit':16 if a.smoke else 0,'learning_rate':.001,'dropout':.1,'weight_decay':0.,
        'readout':'LayerNorm(mean_atoms) plus log1p(atom_count)','upstream':'original full-range masked checkpoints, unchanged',
        'scope':'matched scratch versus masked; no tuning or selecting seeds','test_evaluated':False,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    print(str(out),flush=True)
def execute(a):
    out=a.output_dir.resolve();plan=read(out/'plan.json');assert ROOT==out/'frozen_source'
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=plan['device']
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.45,0)
    old=Path(plan['old_run']);oldplan=read(old/'plan.json');sources=read(old/'pretrain_sources.json')
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    splits={k:v[:plan['limit']] if plan['limit'] else v for k,v in splits.items() if k in ('train','valid')}
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']]);rows=[]
    for seed in plan['seeds']:
        preseed=oldplan['task_to_pretrain_seed'][str(seed)];item=sources[str(preseed)]['masked'];assert sha(Path(item['path']))==item['sha256']
        _,saved=load_upstream(Path(item['path']))
        for arm in ['scratch','masked']:
            cfg=Config(**saved['config']);cfg.dropout=.1
            model=TaskReadoutModel(cfg,saved['priors'],.3,spec.num_tasks);named_initialize(model,seed);model.initialize_priors()
            initial={'encoder':tensor_hash(model.encoder.state_dict()),'head':tensor_hash(model.property_head.state_dict()),'norm':tensor_hash(model.readout_norm.state_dict())}
            if arm=='masked':model.encoder.load_state_dict(saved['encoder'],strict=True)
            model.to(device);torch.manual_seed(seed+10000);folder=out/f'{arm}_seed_{seed}'
            args=SimpleNamespace(stage='finetune',output_dir=folder,device=device,seed=seed,epochs=plan['epochs'],patience=plan['epochs'],batch_size=16,
                learning_rate=.001,encoder_learning_rate=None,weight_decay=0.,grad_clip=5.,resume=False)
            meta={'arm':arm,'seed':seed,'readout':plan['readout'],'initial':initial,'splits':splits,'source':item if arm=='masked' else None,
                  'plan_sha256':sha(out/'plan.json'),'test_evaluated':False,'software':runner.io.shared.provenance()['software']}
            write(out/'status.json',{'state':'running','arm':arm,'seed':seed,'completed_jobs':len(rows)})
            with runner.io.run_directory(args,meta):
                train=runner.io.loader(data,splits['train'],args,True);valid=runner.io.loader(data,splits['valid'],args,False)
                result=runner.fit(model,train,valid,args,meta,spec,scaler,None);pred,y,mask,_=runner.io.predict(model,valid,device,scaler,spec)
                replay,_=load_model(folder/'best.pt');replay.to(device);actual,*_=runner.io.predict(replay,valid,device,scaler,spec)
                torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
                result.update(arm=arm,seed=seed,validation=runner.io.compute_property_metrics(pred,y,mask,spec),initial=initial,checkpoint_replay_passed=True)
                write(folder/'completion.json',result);write(folder/'status.json',{'state':'complete',**result})
            rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print(seed,arm,result['validation']['rmse'],flush=True)
        assert rows[-1]['initial']==rows[-2]['initial']
    aggregate={};paired=[]
    for arm in ['scratch','masked']:
        v=[r['validation']['rmse'] for r in rows if r['arm']==arm]
        aggregate[arm]={'mean_rmse':statistics.mean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None,'n':len(v)}
    for seed in plan['seeds']:
        p={r['arm']:r['validation']['rmse'] for r in rows if r['seed']==seed};paired.append({'seed':seed,'masked_minus_scratch':p['masked']-p['scratch']})
    write(out/'results.json',{'runs':rows,'aggregate':aggregate,'paired':paired,'test_evaluated':False})
    write(out/'status.json',{'state':'complete','completed_jobs':len(rows),'all_replays_passed':True,'paired_initialization_verified':True,'test_evaluated':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.execute:
        try:execute(a)
        except Exception as e:write(a.output_dir/'status.json',{'state':'failed','error':repr(e)});raise
    else:prepare(a)
