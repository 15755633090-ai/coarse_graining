"""Frozen matched-budget chemical-edge recovery comparisons; validation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from torch_geometric.data import Batch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.typed_bonds import GraphDataset,BLOCKS
from motif.adjacency_diffusion.typed_refined import (RefinedBondModel,training_priors,
                                                   load_model,evidence_scale)
from motif.adjacency_diffusion.model import Config
from motif.adjacency_diffusion.diffusion import Schedule
from _github_sync.search_adjacency_rescue_20261029 import named_initialize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ObservationOnly(torch.nn.Module):
    def __init__(self,config,priors):
        super().__init__()
        self.config=config
        self.priors=priors
        self.schedule=Schedule(config.noise_steps,config.beta_min,config.beta_max)

    def predict_bonds(self,condition,noisy,t):
        prior=torch.cat([self.priors[name] for name in BLOCKS]).to(noisy.device)
        return (prior.log()+noisy*evidence_scale(self.schedule,noisy,t))*condition.pairs[...,None]


def prepare(args):
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    corpus=ROOT/'motif/brics_atom_diffusion/runs/pilot_zinc4096_esol3_20261029/corpus'
    train=pd.read_csv(corpus/'train.csv').smiles.tolist()
    valid=pd.read_csv(corpus/'valid.csv').smiles.tolist()
    plan={'train_smiles':train[:512],'development_smiles':valid[:128],
          'confirmation_smiles':valid[256:384],
          'challenge_smiles':['F/C=C/F','F/C=C\\F','Cl/C=C/Br','Cl/C=C\\Br','C/C=C/C','C/C=C\\C'],
          'challenge_scope':'explicit stereochemical diagnostic; not training or model selection',
          'prior_corpus_has_been_used_in_earlier_pilots':True,'selection':'none; report every cell',
          'steps':args.steps,'device':args.device,'hidden_dim':64,'layers':3,'dropout':0.,
          'training_max_t':args.max_training_t,
          'batch_size':8,'learning_rate':.001,'grad_clip':5.,'seeds':[7,1],
          'variants':['trained_reference','frequency_init','factorized','factorized_evidence'],
          'test_evaluated':False,'corpus_sha256':{'train':sha(corpus/'train.csv'),'valid':sha(corpus/'valid.csv')}}
    if args.frequency_only:
        plan['variants']=['frequency_init']
    if args.smoke:
        plan.update(train_smiles=train[600:608],development_smiles=valid[400:408],
                    confirmation_smiles=valid[408:416],hidden_dim=12,layers=1,seeds=[7])
    files=[]
    old=json.loads((ROOT/'motif/adjacency_diffusion/runs/posterior_comparison_20261029/plan.json').read_text(encoding='utf-8'))
    files += [ROOT/name for name in old['source_sha256'] if (ROOT/name).exists()]
    files += list((ROOT/'motif/adjacency_diffusion').glob('*.py'))
    files += [Path(__file__).resolve()]
    files=sorted(set(files))
    for path in files:
        dest=out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    plan['source_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    runner.io.atomic_write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'cells':len(plan['seeds'])*len(plan['variants'])}))


def report(out,rows):
    runner.io.atomic_write(out/'results.json',{'runs':rows,'test_evaluated':False,'selection':None})
    lines=['# Typed chemical-edge recovery: all matched-budget cells','',
           '| variant | seed | confirm AP200 | type200 | joint200 | AP500 | type500 | joint500 | AP1000 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['confirmation']['by_timestep']
        lines.append(f'| {row["variant"]} | {row["seed"]} | '+ ' | '.join(f'{value:.4f}' for value in (
            m['200']['presence_AP'],m['200']['type_accuracy_true_bonds'],m['200']['joint_accuracy_true_bonds'],
            m['500']['presence_AP'],m['500']['type_accuracy_true_bonds'],m['500']['joint_accuracy_true_bonds'],m['1000']['presence_AP']))+' |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(args):
    out=args.output_dir.resolve()
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if ROOT!=out/'frozen_source':
        raise ValueError('Use frozen source')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen source changed: '+name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.45,0)
    train=GraphDataset(plan['train_smiles'])
    priors=training_priors(train)
    runner.io.atomic_write(out/'training_priors.json',{name:value.tolist() for name,value in priors.items()})
    config=Config(hidden_dim=plan['hidden_dim'],layers=plan['layers'],dropout=0.,
                  motif_prior=True,conditional_gate=True,time_mode='random')
    settings=SimpleNamespace(device=plan['device'],seed=7,batch_size=8)
    folds={name:runner.io.loader(GraphDataset(plan[key]),list(range(len(plan[key]))),settings,False)
           for name,key in [('development','development_smiles'),('confirmation','confirmation_smiles'),('challenge','challenge_smiles')]}
    baseline=ObservationOnly(config,priors).to(plan['device'])
    runner.io.atomic_write(out/'observation_only.json',{name:runner.evaluate(baseline,batches,settings) for name,batches in folds.items()})
    rows=[]
    started=time.perf_counter()
    for seed in plan['seeds']:
        settings.seed=seed
        for variant in plan['variants']:
            folder=out/(variant+'_seed_'+str(seed))
            folder.mkdir(exist_ok=False)
            model=RefinedBondModel(config,priors,variant)
            model.training_max_t=plan['training_max_t']
            named_initialize(model,seed)
            model.initialize_priors()
            model.to(plan['device']).set_stage('pretrain')
            initial=runner.evaluate(model,folds['development'],settings)
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(parameters,lr=.001)
            graph_rng=torch.Generator().manual_seed(seed+333)
            noise_rng=torch.Generator(device=plan['device']).manual_seed(seed+7919)
            history=[]
            for step in range(1,plan['steps']+1):
                model.train()
                indices=torch.randperm(len(train),generator=graph_rng)[:8].tolist()
                batch=Batch.from_data_list([train[i] for i in indices]).to(plan['device'])
                optimizer.zero_grad(set_to_none=True)
                loss,blocks=model.loss(batch,noise_rng)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite typed loss')
                loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True))
                optimizer.step()
                if step%100==0 or step==plan['steps']:
                    history.append({'step':step,'loss':float(loss.detach()),'pre_clip_norm':norm,
                                    'blocks':{k:float(v.detach()) for k,v in blocks.items()}})
                    runner.io.atomic_write(folder/'history.json',history)
                    runner.io.atomic_write(out/'status.json',{'state':'running','variant':variant,'seed':seed,
                        'step':step,'steps_per_cell':plan['steps'],'completed_cells':len(rows),'planned_cells':len(plan['seeds'])*len(plan['variants'])})
            path=folder/'encoder.pt'
            runner.io.atomic_write(path,model.checkpoint('pretrain',steps=plan['steps'],
                run_metadata=plan,optimizer=optimizer.state_dict(),noise_rng=noise_rng.get_state(),
                graph_rng=graph_rng.get_state()),binary=True)
            measured={name:runner.evaluate(model,batches,settings) for name,batches in folds.items()}
            restored,_=load_model(path)
            restored.to(plan['device'])
            if runner.evaluate(restored,folds['confirmation'],settings)!=measured['confirmation']:
                raise ValueError('Checkpoint replay mismatch')
            row={'variant':variant,'seed':seed,'initial_development':initial,**measured,
                 'checkpoint_replay_passed':True,'instantiated_parameters':sum(p.numel() for p in model.parameters()),
                 'optimized_parameters':sum(p.numel() for p in parameters),'checkpoint_sha256':sha(path)}
            runner.io.atomic_write(folder/'result.json',row)
            rows.append(row)
            report(out,rows)
            print(f'{variant} seed={seed} AP500={measured["confirmation"]["by_timestep"]["500"]["presence_AP"]:.4f}',flush=True)
    runner.io.atomic_write(out/'status.json',{'state':'complete','completed_cells':len(rows),
                          'elapsed_seconds':time.perf_counter()-started,'test_evaluated':False})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-training-t',type=int,default=1000)
    p.add_argument('--frequency-only',action='store_true')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not 1<=args.max_training_t<=1000:
        raise ValueError('Training horizon must be in [1,1000]')
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
