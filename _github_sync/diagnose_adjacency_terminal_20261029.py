"""Frozen, small-batch diagnostics. Does not change the adjacency model defaults."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score, roc_auc_score
from motif.adjacency_diffusion.data import GraphDataset, condition_from_batch, adjacency_target
from motif.adjacency_diffusion.model import AdjacencyModel, Config
from motif.adjacency_diffusion.metrics import recovery_statistics, summarize
from motif.adjacency_diffusion.training import atomic_write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_config = ROOT/'motif/adjacency_diffusion/runs/smoke_20261029/baseline_pretrain/config.json'
    corpus = json.loads(source_config.read_text(encoding='utf-8'))['inputs']
    train, valid = corpus['train_smiles'][:args.molecules], corpus['valid_smiles'][:args.molecules]
    if len(train) != args.molecules or len(valid) != args.molecules or set(train) & set(valid):
        raise ValueError('Insufficient/disjoint fixed molecules')
    paths = list((ROOT/'motif').glob('*.py'))
    for module in ('adjacency_diffusion','brics_atom_diffusion'):
        paths += list((ROOT/'motif'/module).glob('*.py'))
    paths += [ROOT/'model'/n for n in ('benchmark.py','gine.py','pna.py','attentivefp.py','morgan.py')]
    paths += list((ROOT/'扩散GNN/01_扩散预训练/bond_diffusion').glob('*.py'))
    paths += [Path(__file__).resolve()]
    for path in paths:
        dest = out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,dest)
    plan = {'purpose':'diagnostic only; no implementation switch or enlarged corpus',
            'source_commit':'7a76e5cf4fa59ab4a881a19817b88a0f092fec2c',
            'train_smiles':train,'heldout_smiles':valid,'source_config_sha256':sha(source_config),
            'steps':args.steps,'check_interval':args.check_interval,'probe_draws':args.probe_draws,
            'learning_rates':args.learning_rates,'timesteps':[50,500,1000],
            'parameterizations':['epsilon','direct_A'],'model_seed':7,'train_noise_seed':909,
            'hidden_dim':args.hidden_dim,'layers':args.layers,'dropout':0.,'noise_steps':1000,
            'motif_prior':False,'conditional_gate':False,'optimizer':'Adam','weight_decay':0.,'grad_clip':5.,
            'device':args.device,'threads':2,'gpu_memory_fraction':.4,
            'fixed_noise_training':True,'fixed_noise_is_not_generalization':True,
            'metrics':['frobenius','zero-baseline ratio','positive recall','F1','AP','ROC-AUC'],
            'evaluation':['same fixed noisy batch','fresh noises on same graphs','heldout graphs'],
            'same_tensor_initialization_not_same_Ahat_initialization':True,
            'overfit_criterion':'fixed-batch loss < 0.1*zero baseline AND F1 >= 0.95',
            'planned_cells':len(args.learning_rates)*3*2,
            'test_evaluated':False,'formal_performance_claims':False,
            'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    atomic_write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'planned_cells':plan['planned_cells'],'fixed_train_molecules':len(train)}))


def pack(smiles, device):
    dataset = GraphDataset(smiles)
    batch = Batch.from_data_list([dataset[i] for i in range(len(dataset))]).to(device)
    condition = condition_from_batch(batch,False)
    return condition,adjacency_target(batch,condition)


def raw_to_adjacency(model, condition, noisy, t, mode):
    raw = model.predict_noise(condition,noisy,t)
    if mode == 'epsilon':
        return model.schedule.reconstruct(noisy,raw,t,condition.pairs),raw
    # Same architecture and raw pair head; only its target interpretation differs.
    a = model.schedule.coefficient(noisy,t)
    implied_epsilon = (noisy-a.sqrt()*raw)/(1-a).sqrt()
    return raw,implied_epsilon*condition.pairs


def sample(model, condition, target, timestep, seed):
    generator = torch.Generator(device=target.device).manual_seed(seed)
    t = torch.full((len(target),),timestep,dtype=torch.long,device=target.device)
    noisy,noise = model.schedule.sample(target,t,condition.pairs,generator)
    return condition,target,t,noisy,noise


@torch.no_grad()
def evaluate(model, probes, mode):
    model.eval()
    stats,truth,scores = {},[],[]
    for condition,target,t,noisy,noise in probes:
        restored,epsilon = raw_to_adjacency(model,condition,noisy,t,mode)
        row = recovery_statistics(target,restored,noise,epsilon,condition,t,1000)
        for key,value in row.items():
            stats[key] = stats.get(key,0)+value
        upper = torch.triu(condition.pairs,diagonal=1)
        truth.append(target[upper].cpu())
        scores.append(restored[upper].cpu())
    result = summarize(stats)['all']
    y,z = torch.cat(truth).numpy(),torch.cat(scores).numpy()
    result['average_precision'] = float(average_precision_score(y,z)) if len(set(y.tolist()))==2 else None
    result['roc_auc'] = float(roc_auc_score(y,z)) if len(set(y.tolist()))==2 else None
    baseline = result['zero_adjacency_frobenius']
    result['loss_to_zero_ratio'] = result['reconstruction_frobenius']/baseline if baseline else None
    result['beats_zero'] = result['reconstruction_frobenius'] < baseline if baseline else False
    result['fixed_batch_overfit_criterion'] = bool(result['loss_to_zero_ratio'] is not None
        and result['loss_to_zero_ratio'] < .1 and result['f1'] is not None and result['f1'] >= .95)
    return result


def make_report(out,rows):
    atomic_write(out/'results.json',{'cells':rows,'diagnostic_only':True,'test_evaluated':False})
    lines=['# Fixed-batch adjacency diagnostics', '',
           'Same tensor initialization is not matched A_hat initialization. No generalization or method superiority claim.', '',
           '| lr | t | head | initial ratio | final fixed ratio | fixed F1 | fixed AP | fresh ratio | heldout ratio | overfit |',
           '|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x):
        return 'NA' if x is None else f'{x:.4g}'
    for r in rows:
        first,last=r['initial']['fixed'],r['final']['fixed']
        lines.append('| '+' | '.join((str(r['learning_rate']),str(r['timestep']),r['parameterization'],
            fmt(first['loss_to_zero_ratio']),fmt(last['loss_to_zero_ratio']),fmt(last['f1']),fmt(last['average_precision']),
            fmt(r['final']['fresh_same_graphs']['loss_to_zero_ratio']),fmt(r['final']['heldout_graphs']['loss_to_zero_ratio']),
            str(last['fixed_batch_overfit_criterion'])))+' |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(args):
    out = args.output_dir.resolve()
    plan = json.loads((out/'plan.json').read_text(encoding='utf-8'))
    # Prevent accidental execution of this frozen experiment through live sources.
    if ROOT.resolve() != (out/'frozen_source').resolve():
        raise ValueError('Execute from the frozen_source copy')
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    torch.set_num_threads(2)
    torch.manual_seed(plan['model_seed'])
    torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'],0)
        torch.cuda.manual_seed_all(plan['model_seed'])
    config=Config(hidden_dim=plan['hidden_dim'],layers=plan['layers'],dropout=0.)
    mother=AdjacencyModel(config)
    initial={k:v.detach().clone() for k,v in mother.state_dict().items()}
    train_condition,train_target=pack(plan['train_smiles'],plan['device'])
    held_condition,held_target=pack(plan['heldout_smiles'],plan['device'])
    rows=[]
    started=time.perf_counter()
    for lr in plan['learning_rates']:
        for timestep in plan['timesteps']:
            for mode in plan['parameterizations']:
                name=f'lr_{lr}_t_{timestep}_{mode}'
                folder=out/name
                folder.mkdir(exist_ok=False)
                model=AdjacencyModel(config).to(plan['device'])
                model.load_state_dict(initial,strict=True)
                model.set_stage('pretrain')
                parameters=tuple(p for p in model.parameters() if p.requires_grad)
                optimizer=torch.optim.Adam(parameters,lr=lr,weight_decay=0.)
                fixed=sample(model,train_condition,train_target,timestep,plan['train_noise_seed'])
                fresh=[sample(model,train_condition,train_target,timestep,10001+i) for i in range(plan['probe_draws'])]
                held=[sample(model,held_condition,held_target,timestep,20001+i) for i in range(plan['probe_draws'])]
                probes={'fixed':[fixed],'fresh_same_graphs':fresh,'heldout_graphs':held}
                initial_metrics={key:evaluate(model,value,mode) for key,value in probes.items()}
                history=[]
                norm_sum,clipped=0.,0
                for step in range(1,plan['steps']+1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    condition,target,t,noisy,noise=fixed
                    restored,_=raw_to_adjacency(model,condition,noisy,t,mode)
                    loss=((restored-target).square()*condition.pairs).sum((1,2)).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite diagnostic loss: '+name)
                    loss.backward()
                    norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True))
                    optimizer.step()
                    norm_sum+=norm
                    clipped+=norm+1e-6>5.
                    if step%plan['check_interval']==0 or step==plan['steps']:
                        metrics={key:evaluate(model,value,mode) for key,value in probes.items()}
                        history.append({'step':step,'metrics':metrics,'mean_pre_clip_norm':norm_sum/step,
                                        'clip_fraction':clipped/step})
                        atomic_write(folder/'history.json',history)
                        atomic_write(out/'status.json',{'state':'running','current_cell':name,'step':step,
                            'steps_per_cell':plan['steps'],'completed_cells':len(rows),'planned_cells':plan['planned_cells'],
                            'elapsed_seconds':time.perf_counter()-started})
                        print(f'{name} step={step} ratio={metrics["fixed"]["loss_to_zero_ratio"]:.5f} '
                              f'F1={metrics["fixed"]["f1"]}',flush=True)
                a=float(model.schedule.alpha_bar[timestep])
                row={'learning_rate':lr,'timestep':timestep,'parameterization':mode,'alpha_bar':a,
                     'epsilon_to_A_squared_error_factor':(1-a)/a,'initial':initial_metrics,'final':history[-1]['metrics'],
                     'mean_pre_clip_norm':norm_sum/plan['steps'],'clip_fraction':clipped/plan['steps']}
                atomic_write(folder/'result.json',row)
                # Different schema: these diagnostic heads cannot be mistaken for pretrained encoders.
                atomic_write(folder/'diagnostic_weights.pt',{'diagnostic_schema':'fixed_batch_parameterization_probe_v1',
                    'config':asdict(config),'head_interpretation':mode,'model':model.state_dict()},binary=True)
                rows.append(row)
                make_report(out,rows)
                del model,optimizer
    atomic_write(out/'status.json',{'state':'complete','completed_cells':len(rows),'planned_cells':plan['planned_cells'],
        'elapsed_seconds':time.perf_counter()-started,'test_evaluated':False,'diagnostic_only':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--check-interval',type=int,default=50)
    parser.add_argument('--probe-draws',type=int,default=4)
    parser.add_argument('--molecules',type=int,default=8)
    parser.add_argument('--learning-rates',type=float,nargs='+',default=[3e-5,1e-3])
    parser.add_argument('--hidden-dim',type=int,default=128)
    parser.add_argument('--layers',type=int,default=3)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.execute:
        try:
            execute(args)
        except Exception as error:
            atomic_write(args.output_dir.resolve()/'status.json',{'state':'failed','error':repr(error)})
            raise
    else:
        if min(args.steps,args.check_interval,args.probe_draws,args.molecules)<1 or any(x<=0 for x in args.learning_rates):
            raise ValueError('Invalid frozen diagnostic budget')
        prepare(args)


if __name__=='__main__':
    main()
