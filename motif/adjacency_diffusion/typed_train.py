"""Train typed chemical-edge denoising or clean bond-aware property prediction."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from sklearn.metrics import average_precision_score
from rdkit import Chem
from . import training as io
from .typed_bonds import (TypedBondModel,GraphDataset,SCHEMA,FEATURE_SCHEMA,BLOCKS,
                         edge_target,corrupt,reconstruction_objective,recovery_counts,load_checkpoint)
from .data import condition_from_batch
from motif.brics_atom_diffusion.pretrain import read_smiles


def training_checkpoint(path,model):
    if hasattr(model,'variant'):
        from .typed_refined import load_model
        _,saved=load_model(path)
        if saved['variant']!=model.variant:
            raise ValueError('Typed recovery variant mismatch')
        return saved
    return load_checkpoint(path)


@torch.no_grad()
def evaluate(model,batches,args):
    model.eval()
    result={}
    for step in (50,200,500,1000):
        timestep=max(1,round(step*model.config.noise_steps/1000))
        generator=torch.Generator(device=args.device).manual_seed(args.seed+100003+step)
        counts,loss_sum,graphs,truth,scores={},0.,0,[],[]
        for batch in batches:
            batch=batch.to(args.device)
            condition=condition_from_batch(batch,model.config.motif_prior)
            target=edge_target(batch,condition)
            t=torch.full((len(target),),timestep,dtype=torch.long,device=args.device)
            noisy=corrupt(target,t,condition.pairs,model.schedule,generator)
            logits=model.predict_bonds(condition,noisy,t)
            loss,_=reconstruction_objective(logits,target,condition.pairs)
            loss_sum+=float(loss)*len(target)
            graphs+=len(target)
            for key,value in recovery_counts(logits,target,condition.pairs).items():
                counts[key]=counts.get(key,0)+value
            upper=torch.triu(condition.pairs,diagonal=1)
            truth.append(target[...,0].eq(0)[upper].cpu())
            scores.append((1-logits[...,:6].softmax(-1)[...,0])[upper].cpu())
        y,z=torch.cat(truth).numpy(),torch.cat(scores).numpy()
        def ratio(key,denominator):
            return counts[key]/counts[denominator] if counts[denominator] else None
        row={'objective':loss_sum/graphs,'counts':counts,
             'presence_AP':float(average_precision_score(y,z)) if y.any() else None,
             'presence_precision':ratio('true_positive','predicted_bonds'),
             'presence_recall':ratio('true_positive','bonds'),
             'type_accuracy_all_pairs':ratio('type_all_correct','pairs'),
             'type_accuracy_true_bonds':ratio('type_bond_correct','bonds'),
             'joint_accuracy_true_bonds':ratio('joint_bond_correct','bonds')}
        for name in ('stereo','conjugated','ring'):
            row[name+'_accuracy_true_bonds']=ratio(name+'_correct','bonds')
        for name in ('single','double','triple','aromatic','other','nondefault_stereo'):
            row[name+'_accuracy']=ratio(name+'_correct',name+'_count')
        result[str(timestep)]=row
    return {'objective':sum(row['objective'] for row in result.values())/len(result),'by_timestep':result}


def fit(model,train,valid,args,metadata,spec=None,scaler=None,weight=None):
    model.set_stage(args.stage)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=args.learning_rate,weight_decay=args.weight_decay)
    generator=torch.Generator(device=args.device).manual_seed(args.seed+7919)
    history,best,stale,best_state,first=[],None,0,None,1
    identity=io.digest(args.output_dir/'config.json')
    if args.resume:
        saved=training_checkpoint(args.output_dir/'latest.pt',model)
        if saved['run_config_sha256']!=identity or saved['stage']!=args.stage:
            raise ValueError('Resume identity mismatch')
        best_state=io.shared.authoritative_best(saved)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history,best,stale,first=saved['history'],saved['best_score'],saved['stale'],saved['epoch']+1
        torch.set_rng_state(saved['torch_rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        generator.set_state(saved['noise_rng'])
        train.generator.set_state(saved['loader_rng'])
        io.atomic_write(args.output_dir/'best.pt',best_state,binary=True)
        io.atomic_write(args.output_dir/'history.json',history)
    started=time.perf_counter()
    for epoch in range(first,args.epochs+1):
        if stale>=args.patience:
            break
        model.train()
        losses,norms,clips=[],[],0
        for batch in train:
            batch=batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss=model.loss(batch,generator)[0] if args.stage=='pretrain' else io.shared.masked_loss(model(batch),batch.y,batch.target_mask,spec,scaler,weight)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite typed objective')
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,args.grad_clip,error_if_nonfinite=True))
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(norm)
            clips+=norm>args.grad_clip
        if not losses:
            raise ValueError('No observed training batches')
        if args.stage=='pretrain':
            validation=evaluate(model,valid,args)
            score=validation['objective']
            improved=best is None or score<best
        else:
            pred,target,mask,_=io.predict(model,valid,args.device,scaler,spec)
            validation=io.compute_property_metrics(pred,target,mask,spec)
            score=io.shared.primary_score(validation,spec)
            improved=best is None or io.shared.better_score(score,best,spec)
        if score is None or not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError('Undefined validation objective')
        best_epoch=epoch if improved else best_state['best_epoch']
        best,stale=(score,0) if improved else (best,stale+1)
        history.append({'epoch':epoch,'train_loss_batch_mean':sum(losses)/len(losses),
                        'pre_clip_norm':sum(norms)/len(norms),'clip_fraction':clips/len(norms),'validation':validation})
        state=model.checkpoint(args.stage,epoch=epoch,best_epoch=best_epoch,best_score=best,stale=stale,
            checkpoint_protocol=2,run_config_sha256=identity,run_metadata=metadata,
            target_scaler=None if scaler is None else {'center':scaler.center,'scale':scaler.scale})
        if improved:
            best_state=io.shared.cpu_snapshot(state)
        latest={**state,'best_checkpoint':best_state,'optimizer':optimizer.state_dict(),'history':history,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                'noise_rng':generator.get_state(),'loader_rng':train.generator.get_state()}
        io.atomic_write(args.output_dir/'latest.pt',latest,binary=True)
        io.atomic_write(args.output_dir/'best.pt',best_state,binary=True)
        io.atomic_write(args.output_dir/'history.json',history)
        io.atomic_write(args.output_dir/'status.json',{'state':'running','epoch':epoch,'best_epoch':best_epoch})
        print(f'{args.stage} epoch={epoch} validation={score:.6f} best={best:.6f}',flush=True)
    if best_state is None:
        raise ValueError('No checkpoint committed')
    model.load_state_dict(best_state['model'])
    return {'best_epoch':best_state['best_epoch'],'best_score':best_state['best_score'],
            'epochs_run':history[-1]['epoch'],'elapsed_seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    io.arguments(p)
    p.set_defaults(time_mode='random',learning_rate=.001)
    p.add_argument('--stage',choices=('pretrain','finetune'),required=True)
    p.add_argument('--smiles-csv',type=Path)
    p.add_argument('--valid-csv',type=Path)
    p.add_argument('--smiles-column',default='smiles')
    p.add_argument('--pretrained',type=Path)
    p.add_argument('--dataset',choices=sorted(io.TASK_SPECS),default='esol')
    p.add_argument('--data-root',type=Path,default=io.ROOT/'data')
    p.add_argument('--variant',choices=('v1','frequency_init','factorized','factorized_evidence'),default='v1')
    p.add_argument('--training-max-t',type=int)
    args=p.parse_args()
    config=io.initialize(args)
    if args.training_max_t is not None and (args.variant=='v1' or not 1<=args.training_max_t<=config.noise_steps):
        raise ValueError('A bounded training horizon requires a refined variant and 1 <= max_t <= noise_steps')
    if args.time_mode!='random':
        raise ValueError('Typed v1 uses randomly sampled timesteps; choose --time-mode random')
    spec=scaler=weight=None
    if args.stage=='pretrain':
        if args.pretrained or not args.smiles_csv or not args.valid_csv:
            raise ValueError('Pretrain requires explicit disjoint train/validation CSVs, without --pretrained')
        train_smiles=read_smiles(args.smiles_csv,args.smiles_column)
        valid_smiles=read_smiles(args.valid_csv,args.smiles_column)
        if set(train_smiles)&set(valid_smiles):
            raise ValueError('Overlapping corpus')
        if args.limit:
            train_smiles,valid_smiles=train_smiles[:args.limit],valid_smiles[:args.limit]
        data=GraphDataset(train_smiles+valid_smiles)
        splits={'train':list(range(len(train_smiles))),'valid':list(range(len(train_smiles),len(data)))}
        inputs={'train_smiles':train_smiles,'valid_smiles':valid_smiles,
                'train_csv_sha256':io.digest(args.smiles_csv),'valid_csv_sha256':io.digest(args.valid_csv)}
    else:
        source,full_splits,spec=io.build_property_data(args.data_root,args.dataset)
        splits={k:v[:args.limit] if args.limit else v for k,v in full_splits.items() if k!='test'}
        data=GraphDataset(source.smiles,source.targets,source.target_mask)
        inputs={'dataset':args.dataset,'splits':splits,'pretrained_sha256':io.digest(args.pretrained) if args.pretrained else None}
        if args.pretrained:
            if args.variant=='v1':
                saved=load_checkpoint(args.pretrained)
            else:
                from .typed_refined import load_model
                _,saved=load_model(args.pretrained)
                if saved['variant']!=args.variant:
                    raise ValueError('Pretrained recovery variant mismatch')
            if saved['stage']!='pretrain' or saved['config']!=asdict(config):
                raise ValueError('Typed encoder pretraining configuration mismatch')
            corpus=saved['run_metadata']['inputs']
            seen=set(corpus['train_smiles'])|set(corpus['valid_smiles'])
            overlap=[i for key in ('valid','test') for i in full_splits[key]
                     if Chem.MolToSmiles(Chem.MolFromSmiles(source.smiles[i]),canonical=True) in seen]
            if overlap:
                raise ValueError('Pretraining corpus overlaps downstream validation/test molecules')
        y,mask=source.targets[splits['train']],source.target_mask[splits['train']]
        scaler=io.TargetScaler.fit(y,mask) if spec.task_type=='regression' else None
        weight=io.classification_positive_weight(y,mask) if spec.task_type=='classification' else None
    if args.variant=='v1':
        model=TypedBondModel(config,spec.num_tasks if spec else 1)
    else:
        from .typed_refined import RefinedBondModel,training_priors
        priors=saved['priors'] if args.stage=='finetune' and args.pretrained else training_priors([data[i] for i in splits['train']])
        model=RefinedBondModel(config,priors,args.variant,spec.num_tasks if spec else 1)
        model.training_max_t=args.training_max_t or config.noise_steps
        if args.stage=='finetune' and args.pretrained:
            expected=saved.get('training_max_t',config.noise_steps)
            if args.training_max_t is not None and args.training_max_t!=expected:
                raise ValueError('Pretraining noise-horizon mismatch')
            model.training_max_t=expected
        model.initialize_priors()
    if args.stage=='finetune' and args.pretrained:
        model.encoder.load_state_dict(saved['encoder'],strict=True)
    model.to(args.device)
    metadata={'schema':model.checkpoint(args.stage)['schema'],'feature_schema':FEATURE_SCHEMA,'config':asdict(config),'inputs':inputs,
              'args':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'},
              'objective':('presence BCE plus independently normalized conditional type CE plus three auxiliary block CE'
                           if getattr(model,'factorized',False) else 'sum of four block CE; type on all pairs; remaining attributes on true bonds; per-graph normalized'),
              'recovery_variant':args.variant,
              'class_priors':{name:getattr(model,'prior_'+name).tolist() for name in BLOCKS} if hasattr(model,'variant') else None,
              'noise':'independent Gaussian channels, symmetric across endpoints, all 17 channels corrupted',
              'extra_prior':'clean atom133 and optional clean BRICS membership; no clean edge feature condition',
              'noise_schedule':model.schedule.description(),'test_evaluated':False,
              'source_sha256':{p.relative_to(io.ROOT).as_posix():io.digest(p) for p in Path(__file__).parent.glob('*.py')},
              'dependency_sha256':{name:io.digest(io.ROOT/name) for name in ('motif/brics_atom_diffusion/data.py','motif/brics_atom_diffusion/training.py','model/benchmark.py')},
              'software':io.shared.provenance()['software']}
    with io.run_directory(args,metadata):
        train=io.loader(data,splits['train'],args,True)
        valid=io.loader(data,splits['valid'],args,False)
        result=fit(model,train,valid,args,metadata,spec,scaler,weight)
        result['validation']=evaluate(model,valid,args) if args.stage=='pretrain' else io.compute_property_metrics(*io.predict(model,valid,args.device,scaler,spec)[:3],spec)
        if args.variant=='v1':
            replay=TypedBondModel(config,model.num_tasks).to(args.device)
            replay.load_state_dict(load_checkpoint(args.output_dir/'best.pt')['model'],strict=True)
        else:
            from .typed_refined import load_model
            replay,_=load_model(args.output_dir/'best.pt')
            replay.to(args.device)
        if args.stage=='finetune':
            first=io.predict(model,valid,args.device,scaler,spec)[0]
            second=io.predict(replay,valid,args.device,scaler,spec)[0]
            torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)
        else:
            assert evaluate(replay,valid,args)==result['validation']
        result.update(schema=model.checkpoint(args.stage)['schema'],variant=args.variant,test_evaluated=False,checkpoint_replay_passed=True,smoke_only=bool(args.limit))
        io.atomic_write(args.output_dir/'completion.json',result)
        io.atomic_write(args.output_dir/'status.json',{'state':'complete',**result})


if __name__=='__main__':
    main()
