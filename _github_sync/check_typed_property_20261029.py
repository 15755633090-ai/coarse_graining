"""Locked three-arm ESOL comparison with three independent pretraining seeds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
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
from torch_geometric.data import Batch
from rdkit import Chem
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.model import Config
from motif.adjacency_diffusion.typed_bonds import GraphDataset
from motif.adjacency_diffusion.typed_refined import RefinedBondModel,load_model
from _github_sync.search_adjacency_rescue_20261029 import named_initialize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare(args):
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    module=ROOT/'motif/adjacency_diffusion'
    runs=module/'runs'
    full=runs/'typed_recovery_comparison_20261029'
    bounded=runs/'typed_bounded_noise_20261029'
    pretrain_plan=json.loads((full/'plan.json').read_text(encoding='utf-8'))
    other=json.loads((bounded/'plan.json').read_text(encoding='utf-8'))
    for key in ('train_smiles','development_smiles','confirmation_smiles','steps','hidden_dim','layers','dropout','batch_size'):
        if pretrain_plan[key]!=other[key]:
            raise ValueError('Unmatched pretraining inputs/budget: '+key)
    sources={}
    for seed in (7,1):
        sources[str(seed)]={}
        for arm,folder,maximum in [('full',full,1000),('bounded',bounded,300)]:
            path=folder/f'frequency_init_seed_{seed}'/'encoder.pt'
            model,saved=load_model(path)
            if saved['variant']!='frequency_init' or model.training_max_t!=maximum:
                raise ValueError('Wrong source variant/horizon')
            sources[str(seed)][arm]={'path':str(path),'sha256':sha(path),'training_max_t':maximum}
    files=[ROOT/name for name in pretrain_plan['source_sha256'] if (ROOT/name).exists()]
    files+=list(module.glob('*.py'))+[Path(__file__).resolve()]
    files=sorted(set(files))
    for path in files:
        dest=out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    plan={'scope':'locked exploratory validation; no test evaluation',
          'arms':['scratch','full','bounded'],'variant':'frequency_init','bounded_training_max_t':300,
          'task_to_pretrain_seed':{'0':7,'1':1,'2':2},'pretrain_sources':sources,
          'additional_pretraining_seed':None if args.smoke else 2,'pretraining_protocol':pretrain_plan,
          'task_seeds':[0] if args.smoke else [0,1,2],
          'readout':'mean; no readout search','task_dropout':.1,'pretrain_dropout':0.,
          'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
          'learning_rate':.001,'weight_decay':0.,'grad_clip':5.,'device':args.device,
          'data_root':str(ROOT/'data'),'limit':16 if args.smoke else 0,'smoke_only':args.smoke,
          'gpu_fraction':.45,'test_evaluated':False,'selection':'none; report all three arms',
          'stereo_disclosure':'retain source typed recovery results and new seed2 diagnostics; no rare-stereo optimization',
          'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    runner.io.atomic_write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'property_jobs':len(plan['task_seeds'])*3,'additional_pretrain_jobs':0 if args.smoke else 2}))


def add_seed_two(out,plan,sources):
    protocol=plan['pretraining_protocol']
    train=GraphDataset(protocol['train_smiles'])
    _,reference=load_model(Path(sources['7']['full']['path']))
    settings=SimpleNamespace(device=plan['device'],seed=2,batch_size=8)
    folds={name:runner.io.loader(GraphDataset(protocol[key]),list(range(len(protocol[key]))),settings,False)
           for name,key in [('development','development_smiles'),('confirmation','confirmation_smiles'),('challenge','challenge_smiles')]}
    sources['2']={}
    for arm,maximum in [('full',1000),('bounded',300)]:
        folder=out/'additional_pretraining'/f'{arm}_seed_2'
        folder.mkdir(parents=True,exist_ok=False)
        model=RefinedBondModel(Config(**reference['config']),reference['priors'],'frequency_init')
        named_initialize(model,2)
        model.initialize_priors()
        model.training_max_t=maximum
        model.to(plan['device']).set_stage('pretrain')
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters,lr=.001)
        graph_rng=torch.Generator().manual_seed(335)
        noise_rng=torch.Generator(device=plan['device']).manual_seed(7921)
        history=[]
        for step in range(1,protocol['steps']+1):
            model.train()
            indices=torch.randperm(len(train),generator=graph_rng)[:8].tolist()
            batch=Batch.from_data_list([train[i] for i in indices]).to(plan['device'])
            optimizer.zero_grad(set_to_none=True)
            loss,blocks=model.loss(batch,noise_rng)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite additional pretraining')
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True))
            optimizer.step()
            if step%100==0 or step==protocol['steps']:
                history.append({'step':step,'loss':float(loss.detach()),'pre_clip_norm':norm,
                                'blocks':{k:float(v.detach()) for k,v in blocks.items()}})
                runner.io.atomic_write(folder/'history.json',history)
                runner.io.atomic_write(out/'status.json',{'state':'running','stage':'additional_pretraining','arm':arm,
                    'seed':2,'step':step,'steps':protocol['steps'],'completed_property_jobs':0,'planned_property_jobs':9})
        checkpoint=folder/'encoder.pt'
        runner.io.atomic_write(checkpoint,model.checkpoint('pretrain',steps=protocol['steps'],run_metadata=protocol),binary=True)
        metrics={name:runner.evaluate(model,loader,settings) for name,loader in folds.items()}
        replay,_=load_model(checkpoint)
        replay.to(plan['device'])
        if runner.evaluate(replay,folds['confirmation'],settings)!=metrics['confirmation']:
            raise ValueError('Additional pretraining replay mismatch')
        runner.io.atomic_write(folder/'completion.json',{'arm':arm,'seed':2,'training_max_t':maximum,
            'recovery':metrics,'checkpoint_replay_passed':True,'checkpoint_sha256':sha(checkpoint),'test_evaluated':False})
        sources['2'][arm]={'path':str(checkpoint),'sha256':sha(checkpoint),'training_max_t':maximum}
        print(f'additional pretraining {arm} seed=2 complete',flush=True)
        del model,replay
    runner.io.atomic_write(out/'resolved_pretrain_sources.json',sources)


def report(out,rows):
    import statistics
    groups={arm:[r['validation']['rmse'] for r in rows if r['arm']==arm] for arm in ('scratch','full','bounded')}
    aggregate={arm:{'completed':len(values),'mean_rmse':statistics.mean(values),
                   'sample_sd':statistics.stdev(values) if len(values)>1 else None} for arm,values in groups.items() if values}
    pairs=[]
    for seed in sorted({r['task_seed'] for r in rows}):
        group={r['arm']:r['validation']['rmse'] for r in rows if r['task_seed']==seed}
        if len(group)==3:
            pairs.append({'task_seed':seed,'full_minus_scratch':group['full']-group['scratch'],
                          'bounded_minus_scratch':group['bounded']-group['scratch'],'bounded_minus_full':group['bounded']-group['full']})
    runner.io.atomic_write(out/'results.json',{'runs':rows,'aggregate':aggregate,'paired_deltas':pairs,'test_evaluated':False})
    lines=['# Locked three-arm ESOL validation','',
           '| task seed | pretraining seed | arm | best epoch | validation RMSE |',
           '|---:|---:|---|---:|---:|']
    for row in rows:
        lines.append(f'| {row["task_seed"]} | {row["pretrain_seed"]} | {row["arm"]} | {row["best_epoch"]} | {row["validation"]["rmse"]:.5f} |')
    lines+=['','Aggregates are provisional until all nine runs complete.','']
    for arm,row in aggregate.items():
        lines.append(f'{arm}: n={row["completed"]}, mean RMSE={row["mean_rmse"]:.5f}, sample SD={row["sample_sd"]}')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(args):
    out=args.output_dir.resolve()
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if ROOT!=out/'frozen_source':
        raise ValueError('Use frozen source')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Changed frozen source: '+name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(plan['gpu_fraction'],0)
    sources=plan['pretrain_sources']
    for by_arm in sources.values():
        for item in by_arm.values():
            if sha(Path(item['path']))!=item['sha256']:
                raise ValueError('Source checkpoint changed')
    if plan['additional_pretraining_seed'] is not None:
        add_seed_two(out,plan,sources)
    source,full_splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    protocol=plan['pretraining_protocol']
    seen=set(protocol['train_smiles'])|set(protocol['development_smiles'])|set(protocol['confirmation_smiles'])
    overlap=[i for i,smiles in enumerate(source.smiles) if Chem.MolToSmiles(Chem.MolFromSmiles(smiles),canonical=True) in seen]
    if overlap:
        raise ValueError('Upstream/downstream molecule overlap')
    splits={k:v[:plan['limit']] if plan['limit'] else v for k,v in full_splits.items() if k in ('train','valid')}
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    rows=[]
    started=time.perf_counter()
    for task_seed in plan['task_seeds']:
        pretrain_seed=plan['task_to_pretrain_seed'][str(task_seed)]
        for arm in plan['arms']:
            item=sources[str(pretrain_seed)]['full' if arm=='scratch' else arm]
            _,saved=load_model(Path(item['path']))
            config=Config(**saved['config'])
            config.dropout=plan['task_dropout']
            model=RefinedBondModel(config,saved['priors'],'frequency_init',spec.num_tasks)
            named_initialize(model,task_seed)
            model.initialize_priors()
            encoder_start=tensor_hash(model.encoder.state_dict())
            head_start=tensor_hash(model.property_head.state_dict())
            if arm!='scratch':
                model.encoder.load_state_dict(saved['encoder'],strict=True)
            model.training_max_t=item['training_max_t']
            model.to(plan['device'])
            torch.manual_seed(task_seed+10000)
            folder=out/f'{arm}_task_seed_{task_seed}'
            settings=SimpleNamespace(stage='finetune',output_dir=folder,device=plan['device'],seed=task_seed,
                epochs=plan['epochs'],patience=plan['patience'],batch_size=plan['batch_size'],threads=2,
                learning_rate=.001,weight_decay=0.,grad_clip=5.,resume=False,limit=plan['limit'])
            metadata={'schema':model.checkpoint('finetune')['schema'],'variant':'frequency_init','arm':arm,
                'config':asdict(config),'task_seed':task_seed,'pretrain_seed':pretrain_seed if arm!='scratch' else None,
                'checkpoint_source':item if arm!='scratch' else None,'task_dropout':.1,'pretrain_dropout':0.,
                'splits':splits,'readout':'mean','upstream_downstream_overlap':overlap,
                'encoder_initial_hash':encoder_start,'head_initial_hash':head_start,
                'run_plan_sha256':sha(out/'plan.json'),'software':runner.io.shared.provenance()['software'],
                'mapping_sha256':sha(Path(plan['data_root'])/spec.directory/'mapping/mol.csv.gz'),
                'test_evaluated':False}
            runner.io.atomic_write(out/'status.json',{'state':'running','stage':'property','current_arm':arm,
                'task_seed':task_seed,'completed_property_jobs':len(rows),'planned_property_jobs':len(plan['task_seeds'])*3})
            with runner.io.run_directory(settings,metadata):
                train=runner.io.loader(data,splits['train'],settings,True)
                valid=runner.io.loader(data,splits['valid'],settings,False)
                result=runner.fit(model,train,valid,settings,metadata,spec,scaler,None)
                pred,target,observed,_=runner.io.predict(model,valid,plan['device'],scaler,spec)
                result.update(arm=arm,task_seed=task_seed,pretrain_seed=pretrain_seed if arm!='scratch' else None,
                    validation=runner.io.compute_property_metrics(pred,target,observed,spec),
                    encoder_initial_hash=encoder_start,head_initial_hash=head_start,test_evaluated=False)
                replay,_=load_model(folder/'best.pt')
                replay.to(plan['device'])
                actual,*_=runner.io.predict(replay,valid,plan['device'],scaler,spec)
                torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
                result['checkpoint_replay_passed']=True
                runner.io.atomic_write(folder/'completion.json',result)
                runner.io.atomic_write(folder/'status.json',{'state':'complete',**result})
            rows.append(result)
            report(out,rows)
            print(f'{arm} task_seed={task_seed} validation_rmse={result["validation"]["rmse"]:.5f}',flush=True)
            del model,replay
    for seed in plan['task_seeds']:
        group=[r for r in rows if r['task_seed']==seed]
        if len({r['encoder_initial_hash'] for r in group})!=1 or len({r['head_initial_hash'] for r in group})!=1:
            raise ValueError('Three-arm initialization mismatch')
    runner.io.atomic_write(out/'status.json',{'state':'complete','completed_property_jobs':len(rows),
        'paired_initialization_verified':True,'checkpoint_replays_passed':True,'test_evaluated':False,
        'property_elapsed_seconds':time.perf_counter()-started})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.execute:
        try:
            execute(args)
        except Exception as exc:
            runner.io.atomic_write(args.output_dir/'status.json',{'state':'failed','error':repr(exc)})
            raise
    else:
        prepare(args)


if __name__=='__main__':
    main()
