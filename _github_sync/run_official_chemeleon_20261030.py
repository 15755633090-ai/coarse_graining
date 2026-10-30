"""Official Chemprop 2.2.3 API + published CheMeleon weights, no custom layers."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,gzip,hashlib,json,shutil,sys,time,statistics,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');temp.replace(p)

def prepare(out,smoke=False):
    out.mkdir(parents=True,exist_ok=False)
    datasets={};source_hashes={}
    for name in ['esol','freesolv']:
        base=ROOT/'data'/('ogbg_molesol' if name=='esol' else 'ogbg_molfreesolv')
        with gzip.open(base/'mapping/mol.csv.gz','rt',encoding='utf-8') as f:smiles=[r['smiles'] for r in csv.DictReader(f)]
        with gzip.open(base/'raw/graph-label.csv.gz','rt',encoding='utf-8') as f:labels=[float(r[0]) for r in csv.reader(f)]
        splits={}
        for split in ['train','valid']:
            with gzip.open(base/f'split/scaffold/{split}.csv.gz','rt') as f:ids=[int(r[0]) for r in csv.reader(f)]
            if smoke:ids=ids[:16]
            splits[split]=ids
            dest=out/'data'/f'{name}_{split}.csv';dest.parent.mkdir(exist_ok=True)
            with dest.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['sample_id','smiles','target']);writer.writeheader()
                writer.writerows({'sample_id':i,'smiles':smiles[i],'target':labels[i]} for i in ids)
            source_hashes[str(dest.relative_to(out))]=sha(dest)
        assert not set(splits['train']) & set(splits['valid'])
        if name=='esol' and not smoke:
            old=read(ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029/scratch_task_seed_0/config.json')
            assert splits['train']==old['splits']['train'] and splits['valid']==old['splits']['valid']
        datasets[name]={'split':splits,'train_n':len(splits['train']),'valid_n':len(splits['valid']),
            'raw_mapping_sha256':sha(base/'mapping/mol.csv.gz'),'raw_labels_sha256':sha(base/'raw/graph-label.csv.gz')}
    script=Path(__file__);dest=out/'frozen_source'/'_github_sync'/script.name;dest.parent.mkdir(parents=True);shutil.copy2(script,dest)
    ridge=script.with_name('run_frozen_chemeleon_ridge_20261030.py');ridge_dest=dest.with_name(ridge.name);shutil.copy2(ridge,ridge_dest)
    plan={'method':'published CheMeleon checkpoint + unmodified Chemprop 2.2.3 components and Lightning trainer',
        'datasets':datasets,'data_sha256':source_hashes,'seeds':[0] if smoke else [0,1,2],
        'epochs':2 if smoke else 50,'smoke':smoke,'batch_size':64,'ffn_hidden_dim':300,
        'optimizer':'official MPNN Adam + NoamLikeLR defaults: warmup2, init1e-4 max1e-3 final1e-4',
        'model':'checkpoint-defined BondMessagePassing; official MeanAggregation; fresh RegressionFFN; batch_normFalse',
        'weight_path':str(ROOT/'model/pretrained/chemeleon_mp.pt'),
        'weight_url':'https://zenodo.org/records/15460715/files/chemeleon_mp.pt','expected_md5':'6a80b54fdb7de37ef0374d302f01e8ce',
        'reference':'Chemprop official CheMeleon foundation finetuning notebook; no custom motif/attention/readout/recovery layer',
        'model_selection':'all seeds, fixed epochs, best val_loss checkpoint; no tuning or validation-dependent recipe change',
        'pretrain_overlap':'unknown exact overlap of published PubChem pretraining corpus with these task structures; no claim of disjoint pretraining',
        'capacity_and_budget':'external million-molecule pretrained encoder2048 vs original custom64; method comparison, not attribution to pretraining or our motifs',
        'heldout':'only existing scaffold train/valid data exported; no test predictions, labels or metrics exported/evaluated',
        'adaptation_methods':['official_full_finetune','frozen_official_encoder_training_only_CV_ridge'],
        'ridge_protocol':'StandardScaler+Ridge; canonical-molecule GroupKFold5 within training only; alpha .001,.01,.1,1,10,100,1000,10000; one deterministic fit per task',
        'source_sha256':{str(p.relative_to(out/'frozen_source')):sha(p) for p in [dest,ridge_dest]},'test_evaluated':False}
    write(out/'plan.json',plan)

def execute(out):
    import numpy as np
    import torch,chemprop,rdkit,lightning
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint,Callback
    from chemprop import data,featurizers,nn,models
    assert ROOT==out/'frozen_source'
    plan=read(out/'plan.json');project=Path(plan['weight_path']).parents[2]
    assert chemprop.__version__=='2.2.3'
    for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h
    for rel,h in plan['data_sha256'].items():assert sha(out/rel)==h
    weight=Path(plan['weight_path']);raw=weight.read_bytes();assert hashlib.md5(raw).hexdigest()==plan['expected_md5']
    official_dir=Path(chemprop.__file__).parent;pinned=project/'model/official/chemprop/chemprop'
    identities={}
    for rel in ['models/model.py','nn/message_passing/base.py','nn/agg.py','nn/predictors.py','nn/transforms.py','data/datasets.py','data/dataloader.py','featurizers/atom.py','featurizers/bond.py','featurizers/molgraph/molecule.py','schedulers.py']:
        assert sha(official_dir/rel)==sha(pinned/rel),rel
        identities[rel]=sha(official_dir/rel)
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    write(out/'provenance.json',{'chemprop':chemprop.__version__,'pinned_source_commit':'c83f4838cfc05a072f6741ae5b9700a1453ed571',
        'core_sha256':identities,'weight_sha256':sha(weight),'weight_md5':hashlib.md5(raw).hexdigest(),
        'hyper_parameters':{k:str(v) for k,v in checkpoint['hyper_parameters'].items()},
        'python':sys.version,'torch':torch.__version__,'rdkit':rdkit.__version__,'lightning':lightning.__version__,
        'device':torch.cuda.get_device_name(0),'pretraining_overlap':'unknown'})
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.45)
    rows=[];started=time.perf_counter()
    class History(Callback):
        def __init__(self,folder):self.folder=folder;self.rows=[]
        def on_validation_epoch_end(self,trainer,module):
            if trainer.sanity_checking:return
            metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if v.numel()==1}
            self.rows.append({'epoch':trainer.current_epoch+1,**metrics})
            write(self.folder/'history.json',self.rows)
            print('epoch',trainer.current_epoch+1,metrics,flush=True)
    for name in plan['datasets']:
        for seed in plan['seeds']:
            pl.seed_everything(seed,workers=True)
            folder=out/name/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=False)
            sets={};raw_targets={};records={};feat=featurizers.SimpleMoleculeMolGraphFeaturizer()
            for split in ['train','valid']:
                with (out/'data'/f'{name}_{split}.csv').open(encoding='utf-8') as f:records[split]=list(csv.DictReader(f))
                ys=np.array([[float(r['target'])] for r in records[split]],dtype=np.float64)
                raw_targets[split]=ys
                points=[data.MoleculeDatapoint.from_smi(r['smiles'],y) for r,y in zip(records[split],ys)]
                sets[split]=data.MoleculeDataset(points,feat)
            scaler=sets['train'].normalize_targets();sets['valid'].normalize_targets(scaler)
            train=data.build_dataloader(sets['train'],batch_size=64,num_workers=0,seed=seed,shuffle=True,drop_last=False)
            valid=data.build_dataloader(sets['valid'],batch_size=64,num_workers=0,shuffle=False,drop_last=False)
            mp=nn.BondMessagePassing(**checkpoint['hyper_parameters']);mp.load_state_dict(checkpoint['state_dict'],strict=True)
            assert all(torch.equal(mp.state_dict()[k],v) for k,v in checkpoint['state_dict'].items())
            ffn=nn.RegressionFFN(input_dim=mp.output_dim,hidden_dim=300,output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
            model=models.MPNN(mp,nn.MeanAggregation(),ffn,batch_norm=False,metrics=[nn.metrics.RMSE(),nn.metrics.MAE()])
            count=sum(p.numel() for p in model.parameters() if p.requires_grad)
            config={'dataset':name,'seed':seed,'active_parameters':count,'encoder_output_dim':mp.output_dim,
                'target_center':scaler.mean_.tolist(),'target_scale':scaler.scale_.tolist(),
                'initial_encoder_matches_published_weights':True,'plan_sha256':sha(out/'plan.json'),'test_evaluated':False}
            write(folder/'config.json',config);write(out/'status.json',{'state':'running','dataset':name,'seed':seed,'completed':len(rows)})
            cb=ModelCheckpoint(dirpath=folder/'checkpoints',filename='best',monitor='val_loss',mode='min',save_top_k=1,save_last=True,enable_version_counter=False)
            history=History(folder);wall=time.perf_counter()
            trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=plan['epochs'],logger=False,enable_progress_bar=False,
                enable_model_summary=False,deterministic=True,num_sanity_val_steps=0,callbacks=[history,cb],default_root_dir=folder)
            trainer.fit(model,train,valid)
            replay=models.MPNN.load_from_checkpoint(cb.best_model_path,map_location='cpu');replay.cuda().eval()
            expected_epoch=int(torch.load(cb.best_model_path,map_location='cpu',weights_only=False)['epoch'])+1
            metrics={}
            for split in ['train','valid']:
                loader=data.build_dataloader(sets[split],batch_size=64,num_workers=0,shuffle=False,drop_last=False)
                pred=torch.cat(trainer.predict(replay,loader)).cpu().numpy()
                pred2=torch.cat(trainer.predict(replay,loader)).cpu().numpy();np.testing.assert_allclose(pred,pred2,rtol=1e-6,atol=1e-6)
                truth=raw_targets[split];assert pred.shape==truth.shape and np.isfinite(pred).all()
                metrics[split]={'rmse':float(np.sqrt(np.mean((pred-truth)**2))),'mae':float(np.mean(np.abs(pred-truth)))}
                with (folder/f'{split}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f);writer.writerow(['sample_id','prediction','target']);writer.writerows((r['sample_id'],float(p[0]),float(y[0])) for r,p,y in zip(records[split],pred,truth))
            # Official val_loss is standardized MSE; verify selection and inverse scale.
            mse=metrics['valid']['rmse']**2/scaler.scale_[0]**2
            assert abs(mse-float(cb.best_model_score))<1e-4,(mse,float(cb.best_model_score))
            row={'dataset':name,'seed':seed,**metrics,'best_epoch':expected_epoch,'epochs_run':len(history.rows),
                'elapsed_seconds':time.perf_counter()-wall,'best_checkpoint_sha256':sha(Path(cb.best_model_path)),
                'active_parameters':count,'inverse_scale_and_val_selection_verified':True,'prediction_replay_verified':True,'test_evaluated':False}
            rows.append(row);write(folder/'completion.json',row);write(out/'results.json',{'runs':rows,'test_evaluated':False})
            print('completed',row,flush=True)
            del trainer,model,replay,mp,ffn;torch.cuda.empty_cache()
    aggregate={}
    for name in plan['datasets']:
        values=[r['valid']['rmse'] for r in rows if r['dataset']==name]
        aggregate[name]={'n':len(values),'mean_rmse':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None}
    write(out/'results.json',{'runs':rows,'aggregate':aggregate,'test_evaluated':False,'smoke':plan['smoke']})
    write(out/'status.json',{'state':'complete','new_finetune_runs':len(rows),'all_replays_passed':True,'test_evaluated':False,'wall_seconds':time.perf_counter()-started,'smoke':plan['smoke']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    execute(a.output_dir.resolve()) if a.execute else prepare(a.output_dir.resolve(),a.smoke)
