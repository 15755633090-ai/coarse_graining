"""Read-only checkpoint diagnostics, no optimization or validation selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,statistics,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.task_latent_denoising import load_model,select_fragments,normalized_feature_loss
from motif.adjacency_diffusion.typed_bonds import GraphDataset
from motif.adjacency_diffusion.masked_bonds import load_model as load_original

def mean(v):return statistics.mean(v)
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45)
    torch.use_deterministic_algorithms(True)
    folder=ROOT/'motif/adjacency_diffusion/runs/task_latent_pilot_20261030'
    source,splits,spec=runner.io.build_property_data(ROOT/'data','esol')
    data=GraphDataset(source.smiles,source.targets,source.target_mask)
    scaler=runner.io.TargetScaler.fit(source.targets[splits['train']],source.target_mask[splits['train']])
    args=SimpleNamespace(batch_size=16,seed=0)
    batches=[]
    for batch in runner.io.loader(data,splits['train'][:128],args,False):batches.append(batch.cuda())
    output={'scope':'Gradient and stochastic-state diagnostics: 8 fixed training batches, first 128 train molecules. Grouped clean evaluation: original train/validation splits, saved best checkpoints. No updates, new model selection, or test evaluation.',
            'limitations':'Endpoint, batch-local gradients do not reconstruct training trajectories or prove validation causality. Stochastic losses are not additively decomposable.', 'arms':{}}
    for arm in ['task_augmentation','task_denoising']:
        checkpoint=folder/f'{arm}_seed_0/best.pt';model,saved=load_model(checkpoint);model.cuda();model.set_stage('finetune')
        params=[p for p in model.encoder.parameters() if p.requires_grad]
        rows=[]
        for index,batch in enumerate(batches):
            gen=torch.Generator(device='cuda').manual_seed(7300+index)
            model.eval()
            with torch.no_grad():
                clean,c,_=model.clean_states(batch);selected,eligible=select_fragments(batch,c,gen)
                sigmas=torch.tensor([.1,.3,.5],device='cuda')[torch.arange(len(c.nodes),device='cuda')%3]
                noisy,_=model.latent_states(batch,selected,sigmas,gen)
                eval_mse=float(normalized_feature_loss(noisy['atoms'],clean['atoms'],selected))
                zero=torch.zeros_like(sigmas);same,_=model.latent_states(batch,selected,zero,gen)
                eval_zero=float(normalized_feature_loss(same['atoms'],clean['atoms'],selected))
            model.train();torch.manual_seed(9100+index)
            clean,c,_=model.clean_states(batch)
            other,_,_=model.clean_states(batch)
            dropout_mse=float(normalized_feature_loss(other['atoms'],clean['atoms'],selected).detach())
            noisy,_=model.latent_states(batch,selected,sigmas,gen)
            pc=model.predict_states(clean,c);pn=model.predict_states(noisy,c);pn=torch.where(eligible[:,None],pn,pc)
            lc=runner.io.shared.masked_loss(pc,batch.y,batch.target_mask,spec,scaler,None)
            ln=runner.io.shared.masked_loss(pn,batch.y,batch.target_mask,spec,scaler,None)
            aux=normalized_feature_loss(noisy['atoms'],clean['atoms'],selected)
            losses=[lc,.5*(lc+ln),aux]
            grads=[]
            for loss in losses:
                g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                grads.append(torch.cat([(torch.zeros_like(p) if q is None else q).flatten() for p,q in zip(params,g)]))
            gc,gp,ga=grads
            def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
            rows.append({'batch':index,'graphs':len(c.nodes),'eligible':int(eligible.sum()),'selected_atoms':int(selected.sum()),
                'clean_loss':float(lc.detach()),'noisy_loss':float(ln.detach()),'train_aux':float(aux.detach()),
                'eval_corruption_mse':eval_mse,'eval_zero_mse':eval_zero,'train_clean_clean_dropout_mse':dropout_mse,
                'cos_aux_vs_clean_property':cosine(ga,gc),'cos_aux_vs_joint_property':cosine(ga,gp),
                'cos_noisy_joint_vs_clean_property':cosine(gp,gc),'weighted_aux_norm_over_joint_property':float(.1*ga.norm()/gp.norm().clamp_min(1e-12))})
        keys=[k for k in rows[0] if k not in ['batch','graphs','eligible','selected_atoms']]
        output['arms'][arm]={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'best_epoch':saved['best_epoch'],'means':{k:mean([r[k] for r in rows]) for k in keys},
            'negative_aux_joint_batches':sum(r['cos_aux_vs_joint_property']<0 for r in rows),'batches':rows}
    grouped={}
    original=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029/scratch_task_seed_0/best.pt'
    candidates=[('original',original),*[(arm,folder/f'{arm}_seed_0/best.pt') for arm in output['arms']]]
    for name,path in candidates:
        model,saved=(load_original if name=='original' else load_model)(path);model.cuda().eval();grouped[name]={}
        with torch.no_grad():
            for split in ['train','valid']:
                errors={'single':[],'multiple':[]}
                for batch in runner.io.loader(data,splits[split],args,False):
                    flags=[data[i].fragment_id.unique().numel()>1 for i in batch.sample_id.tolist()]
                    batch=batch.cuda();prediction=model(batch).cpu()
                    prediction=prediction*scaler.scale+scaler.center
                    truth=batch.y.cpu()
                    for j,flag in enumerate(flags):errors['multiple' if flag else 'single'].append(float((prediction[j,0]-truth[j,0]).square()))
                all_errors=errors['single']+errors['multiple']
                grouped[name][split]={'all':{'n':len(all_errors),'rmse':mean(all_errors)**.5},
                    **{group:{'n':len(v),'rmse':mean(v)**.5} for group,v in errors.items()}}
    output['clean_checkpoint_group_evaluation']=grouped
    for name in ['task_augmentation','task_denoising']:
        expected=json.loads((folder/f'{name}_seed_0/completion.json').read_text(encoding='utf-8'))['validation']['rmse']
        assert abs(grouped[name]['valid']['all']['rmse']-expected)<1e-6
    output['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'motif/adjacency_diffusion/task_latent_denoising.py',ROOT/'motif/adjacency_diffusion/model.py',ROOT/'motif/adjacency_diffusion/masked_bonds.py',ROOT/'motif/adjacency_diffusion/typed_bonds.py',ROOT/'motif/adjacency_diffusion/data.py']}
    out=ROOT/'motif/adjacency_diffusion/TASK_LATENT_DIAGNOSTICS_20261030.json'
    out.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'gradient_summary':{arm:{k:v for k,v in row.items() if k!='batches'} for arm,row in output['arms'].items()},'groups':grouped},indent=2))
if __name__=='__main__':main()
