"""Fixed 18-channel unmasked/masked/scratch comparison; no model changes/search."""
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
from motif.adjacency_diffusion.typed_bonds import GraphDataset,edge_target,corrupt,reconstruction_objective
from motif.adjacency_diffusion.masked_bonds import MaskedBondModel,mask_observation,load_model
from motif.adjacency_diffusion.typed_refined import training_priors
from motif.adjacency_diffusion.data import condition_from_batch
from motif.adjacency_diffusion.model import Config
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
    old=json.loads((module/'runs/typed_recovery_comparison_20261029/plan.json').read_text(encoding='utf-8'))
    files=[ROOT/name for name in old['source_sha256'] if (ROOT/name).exists()]
    files+=list(module.glob('*.py'))+[Path(__file__).resolve()]
    files=sorted(set(files))
    for path in files:
        dest=out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    plan={'scope':'fixed masked-edge recovery under Gaussian-corrupted context; validation only',
          'train_smiles':old['train_smiles'],'development_smiles':old['development_smiles'],
          'confirmation_smiles':old['confirmation_smiles'],'challenge_smiles':old['challenge_smiles'],
          'seeds':[7] if args.smoke else [7,1,2],'task_seeds':[0] if args.smoke else [0,1,2],
          'task_to_pretrain_seed':{'0':7,'1':1,'2':2},'pretraining_arms':{'unmasked':0.,'masked':.3},
          'property_arms':['scratch','unmasked','masked'],'pretrain_steps':2 if args.smoke else 1200,
          'hidden_dim':12 if args.smoke else 64,'layers':1 if args.smoke else 3,
          'pretrain_batch_size':8,'pretrain_dropout':0.,'task_dropout':.1,
          'noise_steps':1000,'training_max_t':1000,'learning_rate':.001,'weight_decay':0.,'grad_clip':5.,
          'epochs':2 if args.smoke else 60,'patience':2 if args.smoke else 60,'property_batch_size':16,
          'device':args.device,'data_root':str(ROOT/'data'),'property_limit':16 if args.smoke else 0,
          'input_channels':18,'target_channels':17,'readout':'mean','smoke_only':args.smoke,
          'independent_rng_streams':['graph order','time','Gaussian noise','mask'],
          'same_gaussian_noise_and_timesteps_between_pretrain_arms':True,
          'supervision_count_changes_with_masking':True,'do_not_compare_pretrain_loss_between_arms':True,
          'context_dependence_claim':False,'prior_validation_corpus_used_before':True,
          'test_evaluated':False,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    if args.smoke:
        plan['train_smiles']=old['train_smiles'][16:24]
        plan['development_smiles']=old['development_smiles'][16:24]
        plan['confirmation_smiles']=old['confirmation_smiles'][16:24]
    runner.io.atomic_write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'pretrain_jobs':len(plan['seeds'])*2,'property_jobs':len(plan['task_seeds'])*3}))


def report(out,rows,planned):
    import statistics
    groups={}
    for arm in ('scratch','unmasked','masked'):
        values=[r['validation']['rmse'] for r in rows if r['arm']==arm]
        if values:
            groups[arm]={'n':len(values),'mean_rmse':statistics.mean(values),
                         'sample_sd':statistics.stdev(values) if len(values)>1 else None}
    pairs=[]
    for seed in sorted({r['task_seed'] for r in rows}):
        m={r['arm']:r['validation']['rmse'] for r in rows if r['task_seed']==seed}
        if len(m)==3:
            pairs.append({'task_seed':seed,'unmasked_minus_scratch':m['unmasked']-m['scratch'],
                          'masked_minus_scratch':m['masked']-m['scratch'],'masked_minus_unmasked':m['masked']-m['unmasked']})
    runner.io.atomic_write(out/'results.json',{'runs':rows,'aggregate':groups,'paired_deltas':pairs,
                                             'planned_runs':planned,'test_evaluated':False})
    lines=['# Fixed 18-channel three-arm ESOL validation','',
           '| task seed | arm | best epoch | validation RMSE |','|---:|---|---:|---:|']
    for r in rows:
        lines.append(f'| {r["task_seed"]} | {r["arm"]} | {r["best_epoch"]} | {r["validation"]["rmse"]:.5f} |')
    lines+=['',f'Completed {len(rows)}/{planned}; aggregates are final only when all runs complete.','']
    for arm,m in groups.items():
        lines.append(f'{arm}: n={m["n"]}, mean={m["mean_rmse"]:.5f}, sample SD={m["sample_sd"]}')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(args):
    out=args.output_dir.resolve()
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if ROOT!=out/'frozen_source':
        raise ValueError('Use frozen source')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Changed frozen source '+name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.45,0)
    dataset=GraphDataset(plan['train_smiles'])
    priors=training_priors(dataset)
    runner.io.atomic_write(out/'training_priors.json',{k:v.tolist() for k,v in priors.items()})
    config=Config(hidden_dim=plan['hidden_dim'],layers=plan['layers'],dropout=0.,
                  motif_prior=True,conditional_gate=True,time_mode='random')
    settings=SimpleNamespace(seed=7,device=plan['device'],batch_size=8)
    folds={name:runner.io.loader(GraphDataset(plan[key]),list(range(len(plan[key]))),settings,False)
           for name,key in [('development','development_smiles'),('confirmation','confirmation_smiles'),('challenge','challenge_smiles')]}
    sources={}
    started=time.perf_counter()
    for seed in plan['seeds']:
        sources[str(seed)]={}
        settings.seed=seed
        for arm,ratio in plan['pretraining_arms'].items():
            folder=out/'pretraining'/f'{arm}_seed_{seed}'
            folder.mkdir(parents=True,exist_ok=False)
            model=MaskedBondModel(config,priors,ratio)
            named_initialize(model,seed)
            model.initialize_priors()
            initial_hash=tensor_hash(model.state_dict())
            model.to(plan['device']).set_stage('pretrain')
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(parameters,lr=.001)
            graph_rng=torch.Generator().manual_seed(seed+333)
            time_rng=torch.Generator(device=plan['device']).manual_seed(seed+444)
            noise_rng=torch.Generator(device=plan['device']).manual_seed(seed+555)
            mask_rng=torch.Generator(device=plan['device']).manual_seed(seed+666)
            history=[]
            target_count,bond_count=0,0
            for step in range(1,plan['pretrain_steps']+1):
                model.train()
                ids=torch.randperm(len(dataset),generator=graph_rng)[:8].tolist()
                batch=Batch.from_data_list([dataset[i] for i in ids]).to(plan['device'])
                condition=condition_from_batch(batch,True)
                target=edge_target(batch,condition)
                t=torch.randint(1,1001,(len(target),),device=plan['device'],generator=time_rng)
                noisy=corrupt(target,t,condition.pairs,model.schedule,noise_rng)
                observed,selected=mask_observation(noisy,condition.pairs,ratio,mask_rng)
                supervised=selected if ratio>0 else condition.pairs
                optimizer.zero_grad(set_to_none=True)
                logits=model.predict_bonds(condition,observed,t)
                loss,blocks=reconstruction_objective(logits,target,supervised)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite pretraining objective')
                loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True))
                optimizer.step()
                upper=torch.triu(supervised,diagonal=1)
                target_count+=int(upper.sum())
                bond_count+=int((upper&target[...,0].eq(0)).sum())
                if step%100==0 or step==plan['pretrain_steps']:
                    history.append({'step':step,'objective':float(loss.detach()),'pre_clip_norm':norm,
                                    'blocks':{k:float(v.detach()) for k,v in blocks.items()}})
                    runner.io.atomic_write(folder/'history.json',history)
                    runner.io.atomic_write(out/'status.json',{'state':'running','stage':'pretraining','arm':arm,'seed':seed,
                        'step':step,'steps_per_job':plan['pretrain_steps'],'completed_pretrain_jobs':sum(len(x) for x in sources.values()),
                        'planned_pretrain_jobs':len(plan['seeds'])*2,'completed_property_jobs':0})
            path=folder/'encoder.pt'
            metadata={'inputs':{'train_smiles':plan['train_smiles'],'valid_smiles':plan['development_smiles']+plan['confirmation_smiles']},'plan':plan}
            runner.io.atomic_write(path,model.checkpoint('pretrain',steps=plan['pretrain_steps'],run_metadata=metadata),binary=True)
            recovery={name:runner.evaluate(model,loader,settings) for name,loader in folds.items()}
            replay,_=load_model(path)
            replay.to(plan['device'])
            if runner.evaluate(replay,folds['confirmation'],settings)!=recovery['confirmation']:
                raise ValueError('Pretraining replay mismatch')
            completion={'arm':arm,'seed':seed,'ratio':ratio,'initial_state_hash':initial_hash,'recovery':recovery,
                        'supervised_candidate_pairs_accumulated':target_count,'supervised_true_bonds_accumulated':bond_count,
                        'checkpoint_sha256':sha(path),'checkpoint_replay_passed':True,'test_evaluated':False}
            runner.io.atomic_write(folder/'completion.json',completion)
            sources[str(seed)][arm]={'path':str(path),'sha256':sha(path),'ratio':ratio,'initial_state_hash':initial_hash}
            print('pretraining',arm,seed,'complete',flush=True)
            del model,replay
        assert sources[str(seed)]['unmasked']['initial_state_hash']==sources[str(seed)]['masked']['initial_state_hash']
    runner.io.atomic_write(out/'pretrain_sources.json',sources)
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    upstream=set(plan['train_smiles'])|set(plan['development_smiles'])|set(plan['confirmation_smiles'])
    if any(Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True) in upstream for s in source.smiles):
        raise ValueError('Pretraining/downstream overlap')
    splits={k:v[:plan['property_limit']] if plan['property_limit'] else v for k,v in splits.items() if k in ('train','valid')}
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    rows=[]
    for task_seed in plan['task_seeds']:
        pretrain_seed=plan['task_to_pretrain_seed'][str(task_seed)]
        for arm in plan['property_arms']:
            item=sources[str(pretrain_seed)]['unmasked' if arm=='scratch' else arm]
            _,saved=load_model(Path(item['path']))
            cfg=Config(**saved['config'])
            cfg.dropout=.1
            model=MaskedBondModel(cfg,saved['priors'],item['ratio'],spec.num_tasks)
            named_initialize(model,task_seed)
            model.initialize_priors()
            encoder_hash=tensor_hash(model.encoder.state_dict())
            head_hash=tensor_hash(model.property_head.state_dict())
            if arm!='scratch':
                model.encoder.load_state_dict(saved['encoder'],strict=True)
            model.to(plan['device'])
            torch.manual_seed(task_seed+10000)
            folder=out/f'{arm}_task_seed_{task_seed}'
            args_fit=SimpleNamespace(stage='finetune',output_dir=folder,device=plan['device'],seed=task_seed,
                epochs=plan['epochs'],patience=plan['patience'],batch_size=plan['property_batch_size'],
                learning_rate=.001,weight_decay=0.,grad_clip=5.,resume=False,encoder_learning_rate=None)
            metadata={'arm':arm,'task_seed':task_seed,'pretrain_seed':pretrain_seed if arm!='scratch' else None,
                'config':asdict(cfg),'checkpoint_source':item if arm!='scratch' else None,'splits':splits,
                'head_initial_hash':head_hash,'encoder_initial_hash':encoder_hash,'run_plan_sha256':sha(out/'plan.json'),
                'software':runner.io.shared.provenance()['software'],'test_evaluated':False}
            runner.io.atomic_write(out/'status.json',{'state':'running','stage':'property','arm':arm,'task_seed':task_seed,
                'completed_property_jobs':len(rows),'planned_property_jobs':len(plan['task_seeds'])*3})
            with runner.io.run_directory(args_fit,metadata):
                train=runner.io.loader(data,splits['train'],args_fit,True)
                valid=runner.io.loader(data,splits['valid'],args_fit,False)
                result=runner.fit(model,train,valid,args_fit,metadata,spec,scaler,None)
                pred,target,mask,_=runner.io.predict(model,valid,plan['device'],scaler,spec)
                result.update(arm=arm,task_seed=task_seed,pretrain_seed=pretrain_seed if arm!='scratch' else None,
                    validation=runner.io.compute_property_metrics(pred,target,mask,spec),head_initial_hash=head_hash,
                    encoder_initial_hash=encoder_hash,test_evaluated=False)
                replay,_=load_model(folder/'best.pt')
                replay.to(plan['device'])
                actual,*_=runner.io.predict(replay,valid,plan['device'],scaler,spec)
                torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
                result['checkpoint_replay_passed']=True
                runner.io.atomic_write(folder/'completion.json',result)
                runner.io.atomic_write(folder/'status.json',{'state':'complete',**result})
            rows.append(result)
            report(out,rows,len(plan['task_seeds'])*3)
            print('property',arm,task_seed,result['validation']['rmse'],flush=True)
            del model,replay
    for seed in plan['task_seeds']:
        grouped=[r for r in rows if r['task_seed']==seed]
        assert len({r['head_initial_hash'] for r in grouped})==1
        assert len({r['encoder_initial_hash'] for r in grouped})==1
    runner.io.atomic_write(out/'status.json',{'state':'complete','completed_pretrain_jobs':len(plan['seeds'])*2,
        'completed_property_jobs':len(rows),'initialization_verified':True,'all_checkpoint_replays_passed':True,
        'test_evaluated':False,'elapsed_seconds':time.perf_counter()-started})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.execute:
        try:
            execute(args)
        except Exception as error:
            runner.io.atomic_write(args.output_dir/'status.json',{'state':'failed','error':repr(error)})
            raise
    else:
        prepare(args)


if __name__=='__main__':
    main()
