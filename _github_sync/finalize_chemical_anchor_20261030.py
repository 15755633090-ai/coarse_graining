"""Validate the fixed completed chemical-target recipe and package evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib,tarfile,io,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score
from rdkit import Chem
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.masked_bonds import load_model
from motif.adjacency_diffusion.chemical_targets import chemical_targets,graph_readout,DESCRIPTORS
from motif.adjacency_diffusion.typed_bonds import GraphDataset

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):runner.io.atomic_write(p,x)

@torch.no_grad()
def main(out):
    torch.set_num_threads(2)
    assert read(out/'status.json')['state']=='complete'
    plan=read(out/'plan.json');result=read(out/'results.json');assert len(result['runs'])==3
    for rel,h in plan['source_sha256'].items():assert sha(out/'frozen_source'/rel)==h,rel
    source,splits,spec=runner.io.build_property_data(Path(plan['data_root']),'esol')
    canonical=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True)
    in_domain='domain_correction' in plan
    if in_domain:
        assert plan['corpus']==[source.smiles[i] for i in splits['train']]
        assert not set(map(canonical,plan['corpus'])) & set(map(canonical,[source.smiles[i] for k in ('valid','test') for i in splits[k]]))
    else:
        assert not set(map(canonical,plan['corpus'])) & set(map(canonical,source.smiles))
        assert len(set(map(canonical,plan['corpus'])))==4096
    common_stats=read(ROOT/'motif/adjacency_diffusion/runs/chemical_anchor_20261030/target_statistics.json')
    common_scale=torch.tensor(common_stats['scale'])
    d,f=chemical_targets(plan['corpus']);held=GraphDataset(plan['confirm']);hd,hf=chemical_targets(plan['confirm'])
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    td,tf=chemical_targets([source.smiles[i] for i in splits['train']])
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    audited=[]
    for seed,task_seed in zip(plan['seeds'],plan['task_seeds']):
        path=out/f'pretraining/seed_{seed}/encoder.pt';model,saved=load_model(path);model.cuda().eval()
        stats={k:v.cpu() for k,v in saved['target_statistics'].items()}
        torch.testing.assert_close(stats['center'],d.mean(0));torch.testing.assert_close(stats['scale'],d.std(0,unbiased=False).clamp_min(1e-6))
        head=torch.nn.Linear(64,len(DESCRIPTORS)+512).cuda();head.load_state_dict(saved['chemical_target_head']);head.eval()
        preds=[]
        for i in range(0,len(held),8):
            b=Batch.from_data_list([held[j] for j in range(i,min(i+8,len(held)))]).cuda()
            preds.append(head(graph_readout(model,b)).cpu())
        pred=torch.cat(preds);target=(hd-stats['center'])/stats['scale']
        per_descriptor={name:{'mse':float((pred[:,i]-target[:,i]).square().mean()),
                              'constant_mse':float(target[:,i].square().mean())} for i,(name,_) in enumerate(DESCRIPTORS)}
        per_descriptor={k:v for i,(k,v) in enumerate(per_descriptor.items()) if stats['descriptor_active'][i]}
        target_domain_predictions=[]
        for i in range(0,len(splits['train']),8):
            ids=splits['train'][i:i+8]
            b=Batch.from_data_list([data[j] for j in ids]).cuda()
            target_domain_predictions.append(head(graph_readout(model,b)).cpu())
        tp=torch.cat(target_domain_predictions);tz=(td-stats['center'])/stats['scale']
        target_domain_mse={name:float((tp[:,i]-tz[:,i]).square().mean()) for i,(name,_) in enumerate(DESCRIPTORS) if stats['descriptor_active'][i]}
        raw_estimate=tp[:,:len(DESCRIPTORS)]*stats['scale']+stats['center']
        common_domain_error=((raw_estimate-td)/common_scale).square().mean(0)
        active=stats['fingerprint_active'];y=hf[:,active].numpy();p=pred[:,len(DESCRIPTORS):][:,active].sigmoid().numpy()
        valid=y.sum(0)>0
        aps=[average_precision_score(y[:,i],p[:,i]) for i in range(y.shape[1]) if valid[i]]
        prevalence=y.mean(0)[valid]
        task,saved_task=load_model(out/f'properties/seed_{task_seed}/best.pt');task.cuda().eval();metrics={}
        for split in ('train','valid'):
            batches=runner.io.loader(data,splits[split],SimpleNamespace(seed=task_seed,batch_size=16,device='cuda'),False)
            pp,yy,mm,_=runner.io.predict(task,batches,'cuda',scaler,spec)
            metrics[split]=runner.io.compute_property_metrics(pp,yy,mm,spec)
        expected=next(r for r in result['runs'] if r['task_seed']==task_seed)
        assert abs(metrics['valid']['rmse']-expected['validation']['rmse'])<1e-6
        hist=read(out/f'properties/seed_{task_seed}/history.json');assert len(hist)==60
        audited.append({'seed':seed,'task_seed':task_seed,'source_sha256':sha(path),
            'property_best_sha256':sha(out/f'properties/seed_{task_seed}/best.pt'),
            'confirmation_per_descriptor':per_descriptor,'confirmation_macro_fp_AP':float(sum(aps)/len(aps)),
            'esol_train_structures_descriptor_mse':target_domain_mse,
            'common_source_scaled_esol_train_descriptor_mse':dict(zip([n for n,_ in DESCRIPTORS],map(float,common_domain_error))),
            'confirmation_macro_fp_prevalence':float(prevalence.mean()),'fp_AP_valid_bits':len(aps),
            'best_property_metrics':metrics,'epochs':len(hist)})
    write(out/'acceptance.json',{'passed':True,'audit_script_sha256':sha(Path(__file__)),'source_hashes_verified':len(plan['source_sha256']),
        'upstream_excludes_all_esol':not in_domain,'upstream_esol_training_structures_only':in_domain,
        'upstream_records':len(plan['corpus']),'upstream_canonical_unique':len(set(map(canonical,plan['corpus']))),
        'train_only_target_stats_replayed':True,'property_replays_passed':True,
        'fixed_three_seeds_complete':True,'test_evaluated':False,'runs':audited})
    archive=out/'frozen_source.tar.xz'
    with tarfile.open(archive,'w:xz',preset=6) as z:
        for rel,h in sorted(plan['source_sha256'].items()):
            raw=(out/'frozen_source'/rel).read_bytes();entry=tarfile.TarInfo('frozen_source/'+rel);entry.size=len(raw);entry.mtime=0;entry.mode=0o644
            z.addfile(entry,io.BytesIO(raw))
    write(out/'frozen_source_archive.json',{'path':'frozen_source.tar.xz','sha256':sha(archive),'source_sha256':plan['source_sha256']})
    lines=['# Fixed chemical-target pretraining result','',
        '| task seed | candidate validation RMSE | minus original scratch | best epoch |','|---:|---:|---:|---:|']
    for r in result['runs']:lines.append(f"| {r['task_seed']} | {r['validation']['rmse']:.6f} | {r['minus_scratch']:+.6f} | {r['best_epoch']} |")
    lines+=['',f"Candidate: {result['mean_rmse']:.6f} +/- {result['sd_rmse']:.6f} (n=3).",f"Original scratch: {result['scratch_reference']}",f"Original expanded edge: {result['expanded_edge_reference']}",'',
        'Full 60 epochs, same original downstream model and fresh paired task head. All best checkpoints replayed.',
        'Joint change of input and target; not a single-factor ablation, not KPGT reproduction, and not compute matched.',
        'Calculated MolLogP and other chemical pseudo-labels are additional prior knowledge, not measured target labels.',
        'Reused ESOL validation, no test evaluation. No independent-task confirmation. No subsequent recipe search.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'acceptance':'passed','mean_rmse':result['mean_rmse'],'sd_rmse':result['sd_rmse']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'motif/adjacency_diffusion/runs/chemical_anchor_20261030');a=p.parse_args();main(a.output_dir.resolve())
