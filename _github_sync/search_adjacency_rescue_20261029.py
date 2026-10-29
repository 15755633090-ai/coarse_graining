"""Predeclared rescue screening with fresh noise and a separate confirmation fold."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score,roc_auc_score
from motif.adjacency_diffusion.data import GraphDataset,condition_from_batch,adjacency_target
from motif.adjacency_diffusion.model import AdjacencyModel,Config
from motif.adjacency_diffusion.metrics import recovery_statistics,summarize
from motif.adjacency_diffusion.reparameterizations import decode,reconstruction_objective,degree_prior_baseline
from motif.adjacency_diffusion.training import atomic_write,shared


def variants():
    common=dict(lr=.001,prior=False,gate=False,initialization='named',balanced=False,degree_weight=0.)
    rows=[]
    def add(name,mode,policy,**changes):
        rows.append(dict(common,id=name,mode=mode,policy=policy,**changes))
    add('epsilon_terminal_original_lr','epsilon','terminal',lr=3e-5)
    add('epsilon_terminal','epsilon','terminal')
    add('epsilon_random','epsilon','random')
    add('preconditioned_terminal','preconditioned_epsilon','terminal')
    add('preconditioned_random','preconditioned_epsilon','random')
    add('preconditioned_random_zero','preconditioned_epsilon','random',initialization='zero_head')
    add('preconditioned_t200','preconditioned_epsilon','t200')
    add('preconditioned_t500','preconditioned_epsilon','t500')
    add('preconditioned_curriculum','preconditioned_epsilon','curriculum')
    add('velocity_random','velocity','random')
    add('bounded_random','bounded_A','random')
    add('bounded_density','bounded_A','random',initialization='density')
    add('bounded_density_brics','bounded_A','random',initialization='density',prior=True)
    add('bounded_density_gate','bounded_A','random',initialization='density',prior=True,gate=True)
    add('bounded_density_gate_balanced','bounded_A','random',initialization='density',prior=True,gate=True,balanced=True)
    add('bounded_density_gate_degree','bounded_A','random',initialization='density',prior=True,gate=True,degree_weight=.1)
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    corpus=ROOT/'motif/brics_atom_diffusion/runs/pilot_zinc4096_esol3_20261029/corpus'
    train_all=pd.read_csv(corpus/'train.csv')['smiles'].tolist()
    held_all=pd.read_csv(corpus/'valid.csv')['smiles'].tolist()
    # Tool smoke uses disjoint rows; it never previews the main confirmation fold.
    train_start,dev_start,confirm_start=(128,128,160) if args.smoke else (0,32,64)
    train=train_all[train_start:train_start+args.train_molecules]
    dev=held_all[dev_start:dev_start+args.eval_molecules]
    confirm=held_all[confirm_start:confirm_start+args.eval_molecules]
    if len(train)!=args.train_molecules or min(len(dev),len(confirm))!=args.eval_molecules:
        raise ValueError('Insufficient corpus')
    if set(train)&set(dev) or set(train)&set(confirm) or set(dev)&set(confirm):
        raise ValueError('Corpus overlap')
    files=list((ROOT/'motif').glob('*.py'))
    for module in ('adjacency_diffusion','brics_atom_diffusion'):
        files+=list((ROOT/'motif'/module).glob('*.py'))
    files+=[ROOT/'model'/n for n in ('benchmark.py','gine.py','pna.py','attentivefp.py','morgan.py')]
    files+=list((ROOT/'扩散GNN/01_扩散预训练/bond_diffusion').glob('*.py'))
    files+=[Path(__file__).resolve()]
    for source in files:
        destination=out/'frozen_source'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    plan={'scope':'experimental rescue screen; core published defaults unchanged; no test labels',
          'train_smiles':train,'development_smiles':dev,'confirmation_smiles':confirm,
          'corpus_train_sha256':sha(corpus/'train.csv'),'corpus_valid_sha256':sha(corpus/'valid.csv'),
          'row_ranges':{'train_start':train_start,'development_start':dev_start,'confirmation_start':confirm_start},
          'variants':variants(),'steps':args.steps,'batch_size':8,'hidden_dim':args.hidden_dim,'layers':args.layers,
          'probe_draws':args.probe_draws,'evaluation_t':[50,200,500,1000],
          'dropout':0.,'model_seed':7,'confirmation_seeds':[1,2],'device':args.device,'gpu_fraction':.4,
          'optimizer':'Adam','grad_clip':5.,'weight_decay':0.,'fresh_noise_every_step':True,
          'initialization':'per-module-name seeded linear weights; shared named weights match across gate controls',
          'reference_ids':['epsilon_terminal','epsilon_random'],
          'selection_rule':'candidate must beat epsilon_random mean development density-normalized MSE over t200/t500; t500 ratios to density and degree baselines both <1 and AP>prevalence; select minimum mean MSE and max t500 AP, distinct, at most two',
          'confirmation_rule':'only after selection file is committed; selected plus both references, screen seed7 replay plus new seeds1/2; no retuning on confirmation',
          'density_baseline':'one positive rate from training graphs only, shared over every graph/pair',
          'degree_baseline':'clean-X total degree minus H count; symmetric probability scaling 20 iterations; no target adjacency read',
          'preconditioned_equals_direct_A':True,'noisy_pair_input_support':'all candidate pairs',
          'extra_prior':'BRICS membership variants explicitly get clean structural side information',
          'balanced_and_degree_are_objective_changes':True,'test_evaluated':False,
          'smoke_only':args.smoke,'software':shared.provenance()['software'],
          'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    atomic_write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'screen_cells':len(plan['variants']),
                      'maximum_new_training_cells':len(plan['variants'])+8,'train':len(train),'development':len(dev),'confirmation':len(confirm)}))


def named_initialize(model,seed):
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):
            code=int(hashlib.sha256((str(seed)+':'+name).encode()).hexdigest()[:15],16)
            generator=torch.Generator().manual_seed(code)
            torch.nn.init.kaiming_uniform_(module.weight,a=math.sqrt(5),generator=generator)
            if module.bias is not None:
                bound=1/math.sqrt(module.weight.shape[1])
                torch.nn.init.uniform_(module.bias,-bound,bound,generator=generator)


def pack(dataset,indices,device,prior):
    batch=Batch.from_data_list([dataset[i] for i in indices]).to(device)
    condition=condition_from_batch(batch,prior)
    return condition,adjacency_target(batch,condition)


def positive_rate(dataset):
    positive,pairs=0.,0
    for i in range(len(dataset)):
        graph=dataset[i]
        positive+=graph.edge_index.shape[1]
        pairs+=graph.num_nodes*(graph.num_nodes-1)
    return positive/pairs


@torch.no_grad()
def evaluate(model,dataset,spec,plan,density):
    model.eval()
    result={}
    for timestep in plan['evaluation_t']:
        generator=torch.Generator(device=plan['device']).manual_seed(80000+timestep)
        statistics,truth,scores,density_sum,degree_sum={},[],[],0.,0.
        for draw in range(plan['probe_draws']):
            for offset in range(0,len(dataset),8):
                condition,target=pack(dataset,list(range(offset,min(offset+8,len(dataset)))),plan['device'],spec['prior'])
                t=torch.full((len(target),),timestep,dtype=torch.long,device=target.device)
                noisy,noise=model.schedule.sample(target,t,condition.pairs,generator)
                raw=model.predict_noise(condition,noisy,t)
                clean,epsilon=decode(raw,noisy,t,condition.pairs,model.schedule,spec['mode'])
                row=recovery_statistics(target,clean,noise,epsilon,condition,t,1000)
                for key,value in row.items():
                    statistics[key]=statistics.get(key,0)+value
                upper=torch.triu(condition.pairs,diagonal=1)
                truth.append(target[upper].cpu())
                scores.append(clean[upper].cpu())
                density_sum+=float((((density-target).square())*condition.pairs).sum())
                degree_prob=degree_prior_baseline(condition,density)
                degree_sum+=float(((degree_prob-target).square()*condition.pairs).sum())
        metrics=summarize(statistics)['all']
        y,z=torch.cat(truth).numpy(),torch.cat(scores).numpy()
        metrics['average_precision']=float(average_precision_score(y,z))
        metrics['roc_auc']=float(roc_auc_score(y,z))
        metrics['prevalence']=metrics['positive_pairs']/metrics['undirected_pairs']
        metrics['density_frobenius']=density_sum/metrics['graphs']
        metrics['density_ratio']=metrics['reconstruction_frobenius']/metrics['density_frobenius']
        metrics['degree_frobenius']=degree_sum/metrics['graphs']
        metrics['degree_ratio']=metrics['reconstruction_frobenius']/metrics['degree_frobenius'] if metrics['degree_frobenius']>0 else None
        result[str(timestep)]=metrics
    return result


def mid_score(metrics):
    return .5*(metrics['200']['density_ratio']+metrics['500']['density_ratio'])


def make_model(spec,plan,seed,density):
    config=Config(hidden_dim=plan['hidden_dim'],layers=plan['layers'],dropout=0.,
                  motif_prior=spec['prior'],conditional_gate=spec['gate'],
                  time_mode='random' if spec['policy'] in ('random','curriculum') else 'terminal')
    model=AdjacencyModel(config)
    named_initialize(model,seed)
    if spec['initialization'] in ('zero_head','density'):
        with torch.no_grad():
            model.noise_head[-1].weight.zero_()
            model.noise_head[-1].bias.fill_(math.log(density/(1-density)) if spec['initialization']=='density' else 0.)
    model.to(plan['device']).set_stage('pretrain')
    return model


def train_cell(spec,seed,stage,out,plan,train,dev,density,completed):
    folder=out/stage/(spec['id']+'_seed_'+str(seed))
    folder.mkdir(parents=True,exist_ok=False)
    model=make_model(spec,plan,seed,density)
    initial=evaluate(model,dev,spec,plan,density) if stage=='screen' else None
    parameters=tuple(p for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.Adam(parameters,lr=spec['lr'],weight_decay=0.)
    batch_generator=torch.Generator().manual_seed(seed+333)
    time_generator=torch.Generator(device=plan['device']).manual_seed(seed+444)
    noise_generator=torch.Generator(device=plan['device']).manual_seed(seed+555)
    history=[]
    norm_sum,clips,loss_sum=0.,0,0.
    for step in range(1,plan['steps']+1):
        model.train()
        ids=torch.randperm(len(train),generator=batch_generator)[:min(8,len(train))].tolist()
        condition,target=pack(train,ids,plan['device'],spec['prior'])
        if spec['policy']=='terminal':
            t=torch.full((len(target),),1000,device=target.device,dtype=torch.long)
        elif spec['policy'] in ('t200','t500'):
            t=torch.full((len(target),),int(spec['policy'][1:]),device=target.device,dtype=torch.long)
        else:
            cap=1000
            if spec['policy']=='curriculum':
                cap=(100,300,600,1000)[min(3,(step-1)*4//plan['steps'])]
            t=torch.randint(1,cap+1,(len(target),),generator=time_generator,device=target.device)
        noisy,_=model.schedule.sample(target,t,condition.pairs,noise_generator)
        optimizer.zero_grad(set_to_none=True)
        raw=model.predict_noise(condition,noisy,t)
        clean,_=decode(raw,noisy,t,condition.pairs,model.schedule,spec['mode'])
        loss=reconstruction_objective(clean,target,condition.pairs,density,spec['balanced'],spec['degree_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite candidate '+spec['id'])
        loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True))
        optimizer.step()
        norm_sum+=norm
        clips+=norm+1e-6>5.
        loss_sum+=float(loss.detach())
        if step%50==0 or step==plan['steps']:
            history.append({'step':step,'train_objective_batch_mean':loss_sum/step,
                            'pre_clip_norm_mean':norm_sum/step,'clip_fraction':clips/step})
            atomic_write(folder/'history.json',history)
            atomic_write(out/'status.json',{'state':'running','stage':stage,'current_variant':spec['id'],
                'seed':seed,'step':step,'steps_per_cell':plan['steps'],'completed_training_cells':completed,
                'maximum_training_cells':len(plan['variants'])+8})
    development=evaluate(model,dev,spec,plan,density)
    row={'variant':spec,'seed':seed,'stage':stage,'initial_development':initial,
         'development':development,'development_mid_score':mid_score(development),
         'clip_fraction':clips/plan['steps'],'parameter_counts':{'instantiated':sum(p.numel() for p in model.parameters()),
                 'optimized':sum(p.numel() for p in parameters)}}
    atomic_write(folder/'encoder_experiment.pt',{'experimental_schema':'adjacency_rescue_encoder_v1',
        'config':asdict(model.config),'variant':spec,'model':model.state_dict(),'encoder':model.encoder.state_dict(),
        'seed':seed,'training_positive_rate':density,'method_deviations':spec},binary=True)
    atomic_write(folder/'result.json',row)
    print(f'{stage} {spec["id"]} seed={seed} dev_mid_ratio={row["development_mid_score"]:.4f} '
          f'AP500={development["500"]["average_precision"]:.4f}',flush=True)
    return model,row


def report(out,screen,confirmation):
    atomic_write(out/'results.json',{'screen':screen,'confirmation':confirmation,'test_evaluated':False,
        'experimental_variants_not_paper_equivalent':True})
    lines=['# Rescue screening (development only until selection is frozen)','',
        '| variant | dev mean ratio t200/500 | dev t500 ratio | dev t500 AP | dev t1000 ratio |',
        '|---|---:|---:|---:|---:|']
    for r in screen:
        m=r['development']
        lines.append(f'| {r["variant"]["id"]} | {r["development_mid_score"]:.4f} | {m["500"]["density_ratio"]:.4f} | '
                     f'{m["500"]["average_precision"]:.4f} | {m["1000"]["density_ratio"]:.4f} |')
    if confirmation:
        lines+=['','## Confirmation (not used to pick or retune candidates)','',
                '| variant | seed | confirm mean ratio t200/500 | confirm t500 AP | confirm t1000 ratio |',
                '|---|---:|---:|---:|---:|']
        for r in confirmation:
            m=r['confirmation']
            lines.append(f'| {r["variant"]["id"]} | {r["seed"]} | {mid_score(m):.4f} | '
                         f'{m["500"]["average_precision"]:.4f} | {m["1000"]["density_ratio"]:.4f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(args):
    out=args.output_dir.resolve()
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if ROOT.resolve()!=(out/'frozen_source').resolve():
        raise ValueError('Run through frozen source')
    for name,value in plan['source_sha256'].items():
        if sha(ROOT/name)!=value:
            raise ValueError('Source changed '+name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(plan['gpu_fraction'],0)
    train=GraphDataset(plan['train_smiles'])
    dev=GraphDataset(plan['development_smiles'])
    confirm=GraphDataset(plan['confirmation_smiles'])
    density=positive_rate(train)
    atomic_write(out/'training_density.json',{'positive_rate':density,'source':'training graphs only'})
    screen,confirmation=[],[]
    completed=0
    started=time.perf_counter()
    for spec in plan['variants']:
        model,row=train_cell(spec,7,'screen',out,plan,train,dev,density,completed)
        screen.append(row)
        completed+=1
        report(out,screen,confirmation)
        del model
    reference=next(r for r in screen if r['variant']['id']=='epsilon_random')
    eligible=[r for r in screen if r['variant']['id'] not in plan['reference_ids'] and
        r['development_mid_score']<reference['development_mid_score'] and
        r['development']['500']['density_ratio']<1 and
        r['development']['500']['degree_ratio'] is not None and r['development']['500']['degree_ratio']<1 and
        r['development']['500']['average_precision']>r['development']['500']['prevalence']]
    selected=[]
    if eligible:
        selected.append(min(eligible,key=lambda r:r['development_mid_score'])['variant']['id'])
        best_ap=max(eligible,key=lambda r:r['development']['500']['average_precision'])['variant']['id']
        if best_ap not in selected:
            selected.append(best_ap)
    selection={'selected':selected,'references':plan['reference_ids'],'development_eligible':[r['variant']['id'] for r in eligible],
               'selection_locked_before_confirmation':True,'no_retuning_on_confirmation':True}
    atomic_write(out/'selection.json',selection)
    for name in plan['reference_ids']+selected:
        spec=next(r for r in plan['variants'] if r['id']==name)
        saved=torch.load(out/'screen'/(name+'_seed_7')/'encoder_experiment.pt',map_location='cpu',weights_only=True)
        model=make_model(spec,plan,7,density)
        model.load_state_dict(saved['model'],strict=True)
        row={'variant':spec,'seed':7,'confirmation':evaluate(model,confirm,spec,plan,density)}
        confirmation.append(row)
        del model
        report(out,screen,confirmation)
        for seed in plan['confirmation_seeds']:
            model,row=train_cell(spec,seed,'confirm_training',out,plan,train,dev,density,completed)
            completed+=1
            row['confirmation']=evaluate(model,confirm,spec,plan,density)
            atomic_write(out/'confirm_training'/(name+'_seed_'+str(seed))/'result.json',row)
            confirmation.append(row)
            del model
            report(out,screen,confirmation)
    atomic_write(out/'status.json',{'state':'complete','screen_cells':len(screen),
        'completed_training_cells':completed,'confirmation_rows':len(confirmation),'selected':selected,
        'elapsed_seconds':time.perf_counter()-started,'test_evaluated':False,'formal_performance_claims':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--train-molecules',type=int,default=64)
    p.add_argument('--eval-molecules',type=int,default=32)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--hidden-dim',type=int,default=128)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--probe-draws',type=int,default=2)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.execute:
        try:
            execute(args)
        except Exception as error:
            atomic_write(args.output_dir.resolve()/'status.json',{'state':'failed','error':repr(error)})
            raise
    else:
        if min(args.train_molecules,args.eval_molecules,args.steps,args.hidden_dim,args.layers,args.probe_draws)<1:
            raise ValueError('Invalid screening budget')
        prepare(args)


if __name__=='__main__':
    main()
