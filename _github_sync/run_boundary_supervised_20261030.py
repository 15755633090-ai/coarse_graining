"""Frozen direct-prediction comparison: two architectures, three seeds, no pretraining."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib,shutil,math,time,argparse,statistics
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.boundary_supervised.model import BoundaryPredictor,Config,load_model
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.typed_bonds import GraphDataset

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):runner.io.atomic_write(p,x)
def parameter_hash(state):
    h=hashlib.sha256()
    for n,v in state.items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def initialize(model,seed):
    for name,p in model.named_parameters():
        generator=torch.Generator().manual_seed(int(hashlib.sha256((str(seed)+':'+name).encode()).hexdigest()[:15],16))
        bound=1/math.sqrt(p.shape[-1] if p.ndim>1 else model.config.hidden_dim)
        torch.nn.init.uniform_(p,-bound,bound,generator=generator)

def prepare(out,smoke):
    out.mkdir(parents=True,exist_ok=False)
    previous=read(ROOT/'motif/adjacency_diffusion/runs/representation_matrix_20261030/plan.json')
    paths=set(previous['source_sha256'])|{p.relative_to(ROOT).as_posix() for p in (ROOT/'motif/boundary_supervised').glob('*.py')}|{Path(__file__).relative_to(ROOT).as_posix()}
    for rel in paths:
        dest=out/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    plan={'scope':'direct supervised covalent directed message passing with BRICS/attention combination',
        'arms':['directed','boundary'],'seeds':[0] if smoke else [0,1,2],'epochs':2 if smoke else 60,
        'limit':16 if smoke else 0,'smoke':smoke,'data_root':str(ROOT/'data'),
        'hidden_dim':12 if smoke else 64,'layers':1 if smoke else 3,'readout_steps':2,
        'learning_rate':.001,'batch_size':16,'dropout':.1,'grad_clip':5.,'weight_decay':0.,
        'initialization':'independent seed per named parameter, common shape/name tensors paired including GRU and readout',
        'selection':'all three seeds, full60 epochs, best reused validation checkpoint per run; no recipe search',
        'architecture_control':'directed has same attention readout and actual covalent graph, but no BRICS internal branch/gates; different capacity disclosed',
        'reference':'original supervised scratch .787109; new directed arm must also be compared, not only old model',
        'interpretation':'combined architecture test; not independent proof of gates, BRICS, attention or nonbacktracking contribution',
        'inputs':'same atom133/bond16; fragment equality only in boundary arm; no descriptors, fingerprints, y-based prior or pretrained weights',
        'no_pretraining':True,'test_evaluated':False,'source_sha256':{rel:sha(ROOT/rel) for rel in sorted(paths)}}
    write(out/'plan.json',plan)

def execute(out):
    plan=read(out/'plan.json');assert ROOT==out/'frozen_source'
    for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.45)
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    splits={k:(splits[k][:plan['limit']] if plan['limit'] else splits[k]) for k in ('train','valid')}
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    rows=[];paired={};started=time.perf_counter()
    for seed in plan['seeds']:
        for arm in plan['arms']:
            cfg=Config(hidden_dim=plan['hidden_dim'],layers=plan['layers'],dropout=.1,readout_steps=2,boundary=arm=='boundary')
            model=BoundaryPredictor(cfg,spec.num_tasks);initialize(model,seed)
            shared={n:p for n,p in model.state_dict().items() if not n.startswith(('internal','boundary','atom_output'))}
            h=parameter_hash(shared)
            if seed in paired:assert paired[seed]==h
            else:paired[seed]=h
            count=sum(p.numel() for p in model.parameters());model.cuda();torch.manual_seed(seed+10000)
            folder=out/f'{arm}_seed_{seed}';args=SimpleNamespace(stage='finetune',output_dir=folder,device='cuda',seed=seed,epochs=plan['epochs'],patience=plan['epochs'],batch_size=16,learning_rate=.001,encoder_learning_rate=None,weight_decay=0.,grad_clip=5.,resume=False)
            meta={'arm':arm,'seed':seed,'config':cfg.__dict__,'splits':splits,'active_parameters':count,'shared_initial_hash':h,'plan_sha256':sha(out/'plan.json'),'test_evaluated':False}
            write(out/'status.json',{'state':'running','arm':arm,'seed':seed,'completed':len(rows)})
            with runner.io.run_directory(args,meta):
                train=runner.io.loader(data,splits['train'],args,True);valid=runner.io.loader(data,splits['valid'],args,False)
                result=runner.fit(model,train,valid,args,meta,spec,scaler,None)
                pred,y,m,_=runner.io.predict(model,valid,'cuda',scaler,spec)
                replay,_=load_model(folder/'best.pt');replay.cuda().eval();actual,*_=runner.io.predict(replay,valid,'cuda',scaler,spec)
                torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
                train_eval=runner.io.loader(data,splits['train'],args,False);pp,yy,mm,_=runner.io.predict(replay,train_eval,'cuda',scaler,spec)
                result.update(arm=arm,seed=seed,active_parameters=count,validation=runner.io.compute_property_metrics(pred,y,m,spec),train=runner.io.compute_property_metrics(pp,yy,mm,spec),checkpoint_replay_passed=True)
                write(folder/'completion.json',result)
            rows.append(result);write(out/'results.json',{'runs':rows,'test_evaluated':False});print('completed',arm,seed,result['validation'],flush=True)
    aggregate={}
    for arm in plan['arms']:
        values=[r['validation']['rmse'] for r in rows if r['arm']==arm]
        aggregate[arm]={'mean_rmse':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,'n':len(values)}
    write(out/'results.json',{'runs':rows,'aggregate':aggregate,'test_evaluated':False,'smoke':plan['smoke']})
    write(out/'status.json',{'state':'complete','new_supervised_runs':len(rows),'all_replays_passed':True,'paired_shared_initialization':True,'wall_seconds':time.perf_counter()-started,'test_evaluated':False,'smoke':plan['smoke']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    execute(a.output_dir.resolve()) if a.execute else prepare(a.output_dir.resolve(),a.smoke)
