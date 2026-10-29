"""Matched downstream validation of the locked rescue candidate and readout controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion.model import AdjacencyModel,Config,mlp
from motif.adjacency_diffusion.data import GraphDataset,condition_from_batch,adjacency_target
from motif.adjacency_diffusion import training
from _github_sync.search_adjacency_rescue_20261029 import named_initialize


class ReadoutModel(AdjacencyModel):
    def __init__(self,config,readout):
        super().__init__(config,1)
        self.readout=readout
        d=config.hidden_dim
        inputs=2*d+1 if readout=='mean_sum' else d
        width=round(d*(2*d+3)/(d+2)) if readout=='mean_capacity' else d
        self.property_head=mlp(inputs,width,1)

    def forward(self,batch):
        condition=condition_from_batch(batch,self.config.motif_prior)
        adjacency=adjacency_target(batch,condition)
        t=torch.zeros(len(adjacency),dtype=torch.long,device=adjacency.device)
        states=self.encoder(condition,adjacency,t)['atoms']
        total=states.sum(1)
        count=condition.nodes.sum(1).clamp_min(1).to(total.dtype)[:,None]
        mean=total/count
        if self.readout=='mean_sum':
            representation=torch.cat((mean,total,count.log1p()),-1)
        elif self.readout=='sum':
            representation=total
        else:
            representation=mean
        return self.property_head(representation)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(state):
    value=hashlib.sha256()
    for name,tensor in state.items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def prepare(args):
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rescue=ROOT/'motif/adjacency_diffusion/runs/rescue_screen_20261029'
    selection=json.loads((rescue/'selection.json').read_text(encoding='utf-8'))
    if not selection['selected']:
        raise ValueError('No locked rescue candidate')
    candidate=selection['selected'][0]
    sources={0:rescue/'screen'/(candidate+'_seed_7')/'encoder_experiment.pt',
             1:rescue/'confirm_training'/(candidate+'_seed_1')/'encoder_experiment.pt',
             2:rescue/'confirm_training'/(candidate+'_seed_2')/'encoder_experiment.pt'}
    if not all(path.exists() for path in sources.values()):
        raise ValueError('Incomplete confirmation checkpoints')
    files=list((ROOT/'motif').glob('*.py'))
    for module in ('adjacency_diffusion','brics_atom_diffusion'):
        files+=list((ROOT/'motif'/module).glob('*.py'))
    files+=[ROOT/'model'/name for name in ('benchmark.py','gine.py','pna.py','attentivefp.py','morgan.py')]
    files+=list((ROOT/'扩散GNN/01_扩散预训练/bond_diffusion').glob('*.py'))
    files+=[Path(__file__).resolve(),ROOT/'_github_sync/search_adjacency_rescue_20261029.py']
    for path in files:
        dest=out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    jobs=[]
    for seed in (0,1,2):
        modes=('mean','sum','mean_sum','mean_capacity') if seed==0 else ('mean','mean_sum')
        for readout in modes:
            for arm in ('scratch','pretrained'):
                jobs.append({'seed':seed,'readout':readout,'arm':arm})
    if args.smoke:
        jobs=[{'seed':0,'readout':mode,'arm':arm} for mode in ('mean','mean_sum','mean_capacity') for arm in ('scratch','pretrained')]
    plan={'scope':'exploratory validation only; no paper benchmark/test claim',
          'candidate_locked_by_reconstruction_development':candidate,'selection_sha256':sha(rescue/'selection.json'),
          'pretrained_sources':{str(seed):{'path':str(path),'sha256':sha(path),'pretraining_seed':7 if seed==0 else seed} for seed,path in sources.items()},
          'same_encoder_configuration_for_scratch_and_pretrained':True,
          'same_named_initialization_and_task_head_within_pair':True,
          'task_dropout':.1,'optimizer':'Adam','learning_rate':.001,'weight_decay':0.,'grad_clip':5.,
          'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,'device':args.device,
          'readout_changes_capacity':True,'mean_capacity_hidden':255,'mean_sum_hidden':128,
          'capacity_control_only_seed0':True,'primary_readouts_predeclared':['mean','mean_sum'],
          'jobs':jobs,'data_root':str(ROOT/'data'),'limit':args.limit,'smoke_only':args.smoke,
          'gpu_fraction':.45,'test_evaluated':False,'final_method_not_selected_on_test':True,
          'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    training.atomic_write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'planned_jobs':len(jobs),'candidate':candidate}))


def report(out,rows):
    training.atomic_write(out/'results.json',{'runs':rows,'test_evaluated':False,'validation_only':True})
    lines=['# Matched encoder transfer and readout controls (validation only)','',
           '| seed | readout | arm | best epoch | RMSE | head parameters |',
           '|---:|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["seed"]} | {row["readout"]} | {row["arm"]} | {row["best_epoch"]} | '
                     f'{row["validation"]["rmse"]:.5f} | {row["task_head_parameters"]} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(args):
    out=args.output_dir.resolve()
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if ROOT.resolve()!=(out/'frozen_source').resolve():
        raise ValueError('Use the frozen source')
    for name,value in plan['source_sha256'].items():
        if sha(ROOT/name)!=value:
            raise ValueError('Changed frozen source '+name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(plan['gpu_fraction'],0)
    source,splits,spec=training.build_property_data(Path(plan['data_root']),'esol')
    if plan['limit']:
        splits={k:v[:plan['limit']] for k,v in splits.items()}
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    y,mask=source.targets[splits['train']],source.target_mask[splits['train']]
    scaler=training.TargetScaler.fit(y,mask)
    rows=[]
    started=time.perf_counter()
    for job in plan['jobs']:
        seed,readout,arm=job['seed'],job['readout'],job['arm']
        name=f'{readout}_{arm}_seed_{seed}'
        pretrain=plan['pretrained_sources'][str(seed)]
        if sha(Path(pretrain['path']))!=pretrain['sha256']:
            raise ValueError('Pretrain checkpoint changed')
        saved=torch.load(pretrain['path'],map_location='cpu',weights_only=True)
        if saved.get('experimental_schema')!='adjacency_rescue_encoder_v1':
            raise ValueError('Wrong rescue encoder schema')
        config=Config(**saved['config'])
        config.dropout=plan['task_dropout']
        torch.manual_seed(seed)
        model=ReadoutModel(config,readout)
        named_initialize(model,seed)
        initial_encoder=tensor_hash(model.encoder.state_dict())
        initial_head=tensor_hash(model.property_head.state_dict())
        if arm=='pretrained':
            model.encoder.load_state_dict(saved['encoder'],strict=True)
        model.to(plan['device'])
        # Reset after construction/transfer so paired arms use identical dropout streams.
        torch.manual_seed(10000+seed)
        settings=SimpleNamespace(output_dir=out/name,seed=seed,device=plan['device'],threads=2,
            epochs=plan['epochs'],patience=plan['patience'],batch_size=plan['batch_size'],
            learning_rate=plan['learning_rate'],weight_decay=0.,grad_clip=5.,resume=False,limit=plan['limit'])
        inputs={'dataset':'esol','splits':splits,'readout':readout,'arm':arm,
                'pretraining_variant':saved['variant'],'pretrain_sha256':pretrain['sha256'],
                'pretraining_seed':pretrain['pretraining_seed'],'task_seed':seed,
                'encoder_initial_hash_before_transfer':initial_encoder,'task_head_initial_hash':initial_head,
                'pretrain_dropout':saved['config']['dropout'],'task_dropout':plan['task_dropout'],
                'adapter_source_sha256':sha(Path(__file__).resolve())}
        metadata=training.metadata(settings,config,inputs,'finetune')
        metadata['experiment']='readout_adapter; use experimental readout loader, not standard head reconstruction'
        training.atomic_write(out/'status.json',{'state':'running','current_job':name,'completed_jobs':len(rows),
                                               'planned_jobs':len(plan['jobs']),'elapsed_seconds':time.perf_counter()-started})
        with training.run_directory(settings,metadata):
            train_loader=training.loader(data,splits['train'],settings,True)
            valid_loader=training.loader(data,splits['valid'],settings,False)
            result=training.fit(model,train_loader,valid_loader,settings,'finetune',spec,scaler,None)
            pred,target,observed,_=training.predict(model,valid_loader,plan['device'],scaler,spec)
            metrics=training.compute_property_metrics(pred,target,observed,spec)
            result.update(job,validation=metrics,task_head_parameters=sum(p.numel() for p in model.property_head.parameters()),
                          test_evaluated=False,encoder_initial_hash=initial_encoder,task_head_initial_hash=initial_head)
            training.atomic_write(settings.output_dir/'completion.json',result)
            exported=training.load_checkpoint(settings.output_dir/'best.pt')
            exported['schema']='experimental_readout_task_v1'
            exported['readout']=readout
            training.atomic_write(settings.output_dir/'readout_best.pt',exported,binary=True)
            # Replay via the actual readout architecture; the noise head remains unused.
            replay=ReadoutModel(config,readout).to(plan['device'])
            replay.load_state_dict(exported['model'],strict=True)
            actual,*_=training.predict(replay,valid_loader,plan['device'],scaler,spec)
            torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
            training.atomic_write(settings.output_dir/'status.json',{'state':'complete',**result})
        rows.append(result)
        report(out,rows)
        print(f'{name} validation_rmse={metrics["rmse"]:.5f}',flush=True)
        del model,replay
    # Initialization hashes establish matched encoder/head starts within each pair.
    for row in rows:
        counterpart=next(r for r in rows if r['seed']==row['seed'] and r['readout']==row['readout'] and r['arm']!=row['arm'])
        assert row['encoder_initial_hash']==counterpart['encoder_initial_hash']
        assert row['task_head_initial_hash']==counterpart['task_head_initial_hash']
    training.atomic_write(out/'status.json',{'state':'complete','completed_jobs':len(rows),'planned_jobs':len(plan['jobs']),
        'paired_initialization_verified':True,'validation_replays_passed':True,'test_evaluated':False,
        'elapsed_seconds':time.perf_counter()-started})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--patience',type=int,default=15)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.execute:
        try:
            execute(args)
        except Exception as error:
            training.atomic_write(args.output_dir.resolve()/'status.json',{'state':'failed','error':repr(error)})
            raise
    else:
        prepare(args)


if __name__=='__main__':
    main()
