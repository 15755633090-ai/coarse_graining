"""Fixed-weight path interventions: dependency screening, not pruning proof."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,statistics,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset
from motif.adjacency_diffusion.path_gates import gated_paths,controls,DEFAULT
from _github_sync.check_masked_property_20261029 import tensor_hash

VARIANTS={'full':{},'no_internal_branch':{'internal_branch':0},
    'no_context_within':{'context_within':0},'no_context_cross':{'context_cross':0},
    'no_internal_readout':{'internal_readout':0},'no_context_readout':{'context_readout':0},
    'uniform_boundary':{'boundary_mode':'uniform'},
    'no_internal_and_context_within':{'internal_branch':0,'context_within':0}}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):runner.io.atomic_write(p,x)
def execute(a):
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029'
    source,splits,spec=runner.io.build_property_data(ROOT/'data','esol')
    ds=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    ids=splits['valid'];multi=torch.tensor([ds[i].fragment_id.unique().numel()>1 for i in ids])
    args=SimpleNamespace(device=a.device,batch_size=16,seed=0)
    loader=runner.io.loader(ds,ids,args,False)
    files=list((ROOT/'motif/adjacency_diffusion').glob('*.py'))+[Path(__file__).resolve(),ROOT/'model/benchmark.py']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest() for p in files}
    plan={'question':'which trained prediction paths are currently used; not whether retraining can remove them',
        'variants':VARIANTS,'arms':['scratch','masked'],'seeds':[0,1,2],'old_run':str(old),'valid_ids':ids,
        'groups':{'all':len(ids),'multi_fragment':int(multi.sum()),'single_fragment':int((~multi).sum())},
        'screen_thresholds':{'max_absolute_rmse_change':.01,'max_prediction_rms_shift':.05},
        'threshold_scope':'descriptive ESOL original-label units; not equivalence statistics or deletion approval',
        'source_normalized_utf8_sha256':hashes,'test_evaluated':False,'new_training_runs':0,
        'software':runner.io.shared.provenance()['software']}
    write(out/'plan.json',plan);rows=[];predictions={}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.45,0)
    for arm in ['scratch','masked']:
        for seed in [0,1,2]:
            folder=old/f'{arm}_task_seed_{seed}';model,saved=load_model(folder/'best.pt');model.to(a.device).eval()
            config=read(folder/'config.json');assert config['splits']['valid']==ids and config['splits']['train']==splits['train']
            for key in ['center','scale']:torch.testing.assert_close(saved['target_scaler'][key],getattr(scaler,key),atol=0,rtol=0)
            initial=tensor_hash(model.state_dict());base=None
            for name,changes in VARIANTS.items():
                with gated_paths(model,**changes):pred,y,mask,_=runner.io.predict(model,loader,a.device,scaler,spec)
                assert controls(model.encoder)==DEFAULT
                if name=='full':
                    base=pred.clone();reference=read(folder/'completion.json')['validation']['rmse']
                    actual=runner.io.compute_property_metrics(pred,y,mask,spec)['rmse']
                    assert abs(actual-reference)<1e-6,(actual,reference)
                metrics={}
                for group,selected in [('all',torch.ones(len(ids),dtype=torch.bool)),('multi_fragment',multi),('single_fragment',~multi)]:
                    metrics[group]={'n':int(selected.sum())}
                    if bool(selected.any()):
                        current=runner.io.compute_property_metrics(pred[selected],y[selected],mask[selected],spec)
                        baseline=runner.io.compute_property_metrics(base[selected],y[selected],mask[selected],spec)
                        current.update(delta_rmse=current['rmse']-baseline['rmse'],
                            prediction_rms_shift=float((pred[selected]-base[selected]).square().mean().sqrt()))
                        metrics[group].update(current)
                if name=='no_context_cross' and bool((~multi).any()):torch.testing.assert_close(pred[~multi],base[~multi],atol=0,rtol=0)
                row={'arm':arm,'seed':seed,'variant':name,'metrics':metrics,'checkpoint_sha256':sha(folder/'best.pt')}
                rows.append(row);predictions[f'{arm}_{seed}_{name}']=pred
                write(out/'status.json',{'state':'running','completed_evaluations':len(rows),'planned_evaluations':48,'arm':arm,'seed':seed,'variant':name})
            assert tensor_hash(model.state_dict())==initial
            write(out/'results.json',{'rows':rows,'test_evaluated':False})
    summaries=[]
    for arm in ['scratch','masked']:
        for name in VARIANTS:
            for group in ['all','multi_fragment','single_fragment']:
                records=[r['metrics'][group] for r in rows if r['arm']==arm and r['variant']==name and r['metrics'][group]['n']]
                if not records:continue
                deltas=[r['delta_rmse'] for r in records];shifts=[r['prediction_rms_shift'] for r in records]
                summaries.append({'arm':arm,'variant':name,'group':group,'n_molecules':records[0]['n'],
                    'mean_rmse':statistics.mean(r['rmse'] for r in records),'mean_delta_rmse':statistics.mean(deltas),
                    'per_seed_delta_rmse':deltas,'max_prediction_rms_shift':max(shifts),
                    'low_dependency_candidate':name!='full' and max(abs(d) for d in deltas)<=.01 and max(shifts)<=.05})
    write(out/'results.json',{'rows':rows,'summary':summaries,'test_evaluated':False})
    runner.io.atomic_write(out/'predictions.pt',{'predictions':predictions,'targets':y,'mask':mask,'valid_ids':ids,'multi_fragment':multi},binary=True)
    for p,h in hashes.items():assert hashlib.sha256((ROOT/p).read_text(encoding='utf-8-sig').encode()).hexdigest()==h,p
    lines=['# Fixed-weight path dependency screen','','No training; no claim of structural redundancy. Positive RMSE change means removing/bypassing the path hurts.','',
        '| arm | intervention | all: mean ΔRMSE | multi-fragment: mean ΔRMSE | low-dependency candidate on multi-fragment |',
        '|---|---|---:|---:|---|']
    for arm in ['scratch','masked']:
        for name in VARIANTS:
            all_row=next(r for r in summaries if r['arm']==arm and r['variant']==name and r['group']=='all')
            m=next((r for r in summaries if r['arm']==arm and r['variant']==name and r['group']=='multi_fragment'),None)
            lines.append(f"| {arm} | {name} | {all_row['mean_delta_rmse']:+.5f} | {m['mean_delta_rmse'] if m else None} | {m['low_dependency_candidate'] if m else None} |")
    lines+=['','Joint intervention tests compensation between independent internal and context-within propagation. Direct internal-readout suppression retains internal-to-context/gate routes. Uniform boundary changes both conditionality and message amplitude.',
        'A low-dependency candidate requires per-seed |ΔRMSE|<=0.01 and prediction RMS shift<=0.05. These are descriptive thresholds, not equivalence tests. Fixed-weight suppression is out-of-distribution and is not a retrained removal comparison.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'status.json',{'state':'complete','evaluations':48,'baseline_replay_passed':True,'weights_unchanged':True,
        'single_fragment_cross_invariance_passed':True,'source_hashes_unchanged':True,'test_evaluated':False,'new_training_runs':0})
    print(json.dumps({'groups':plan['groups'],'summary':summaries}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    execute(a)
