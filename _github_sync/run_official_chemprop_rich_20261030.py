"""Official Chemprop v1-style rich features path; no invented neural layers."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,argparse,csv,json,hashlib,shutil,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from _github_sync.run_official_chemeleon_20261030 import prepare as export,read,write,sha

def prepare(out):
    export(out,False);plan=read(out/'plan.json')
    plan.update(method='official Chemprop D-MPNN + v1_rdkit_2d_normalized molecular features; random initialization',
        model='official V1 atom133/bond14, BondMessagePassing300, MeanAggregation, RegressionFFN500->300->1; no extra feature scaling',
        adaptations=['official_rich_chemprop','standard_rbf_svr_on_same_200_features'],
        svr_protocol='C[1,10,100], gamma[scale,.01,.1], epsilon.1, GroupKFold5 canonical molecules within training only; no task validation selection')
    for k in ['weight_path','weight_url','expected_md5','pretrain_overlap','adaptation_methods','ridge_protocol']:plan.pop(k,None)
    plan['capacity_and_budget']='standard300 random encoder and external fixed normalized 200 features; different public feature protocol, not module ablation'
    plan['pretraining']='none; feature CDF parameters are fixed descriptastorus priors, no target labels'
    script=Path(__file__);dest=out/'frozen_source'/'_github_sync'/script.name;shutil.copy2(script,dest)
    plan['source_sha256'][str(dest.relative_to(out/'frozen_source'))]=sha(dest)
    write(out/'plan.json',plan)

def execute(out):
    import numpy as np,torch,chemprop,rdkit,lightning,joblib
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint,Callback
    from chemprop import data,featurizers,nn,models
    from rdkit import Chem
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV,GroupKFold
    from threadpoolctl import threadpool_limits
    assert ROOT==out/'frozen_source';plan=read(out/'plan.json')
    for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h
    for rel,h in plan['data_sha256'].items():assert sha(out/rel)==h
    assert chemprop.__version__=='2.2.3'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.45)
    write(out/'provenance.json',{'chemprop':chemprop.__version__,'torch':torch.__version__,'rdkit':rdkit.__version__,'lightning':lightning.__version__,
        'official_core_sha256':{str(p.relative_to(Path(chemprop.__file__).parent)):sha(p) for p in Path(chemprop.__file__).parent.rglob('*.py')},
        'device':torch.cuda.get_device_name(0),'python':sys.version,'features':'v1_rdkit_2d_normalized; no extra standard scaling'})
    atom=featurizers.MultiHotAtomFeaturizer.v1();feat=featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=atom)
    molecular=featurizers.V1RDKit2DNormalizedFeaturizer();rows=[];svrs=[];started=time.perf_counter()
    class History(Callback):
        def __init__(self,folder):self.folder=folder;self.rows=[]
        def on_validation_epoch_end(self,trainer,module):
            if trainer.sanity_checking:return
            row={'epoch':trainer.current_epoch+1,**{k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if v.numel()==1}}
            self.rows.append(row);write(self.folder/'history.json',self.rows)
    for name in plan['datasets']:
        records={};targets={};xs={}
        for split in ['train','valid']:
            with (out/'data'/f'{name}_{split}.csv').open(encoding='utf-8') as f:records[split]=list(csv.DictReader(f))
            targets[split]=np.array([[float(r['target'])] for r in records[split]])
            raw=np.array([molecular(Chem.MolFromSmiles(r['smiles'])) for r in records[split]])
            nan_count=int(np.isnan(raw).sum());raw[np.isnan(raw)]=0 # same official datapoint NaN token
            assert raw.shape==(len(records[split]),200) and np.isfinite(raw).all()
            xs[split]=raw
            write(out/f'{name}_{split}_features.json',{'nan_replaced_with_official_zero_token':nan_count,'min':float(raw.min()),'max':float(raw.max())})
        # Standard non-neural adaptation, fixed train-only CV. No new neural component.
        groups=[Chem.MolToSmiles(Chem.MolFromSmiles(r['smiles']),canonical=True) for r in records['train']]
        search=GridSearchCV(SVR(epsilon=.1),{'C':[1.,10.,100.],'gamma':['scale',.01,.1]},cv=GroupKFold(5),scoring='neg_mean_squared_error',n_jobs=1)
        with threadpool_limits(limits=2):search.fit(xs['train'],targets['train'].ravel(),groups=groups)
        sm={}
        for split in xs:
            pp=search.predict(xs[split]);yy=targets[split].ravel();sm[split]={'rmse':float(np.sqrt(np.mean((pp-yy)**2))),'mae':float(np.mean(abs(pp-yy)))}
        dest=out/name/'svr';dest.mkdir(parents=True);joblib.dump(search.best_estimator_,dest/'model.joblib')
        svr={'dataset':name,**sm,'selected':search.best_params_,'training_cv_mse':(-search.cv_results_['mean_test_score']).tolist(),'selection':'training-only canonical GroupKFold5','test_evaluated':False}
        svrs.append(svr);write(dest/'completion.json',svr);write(out/'svr_results.json',{'runs':svrs,'test_evaluated':False});print('svr',svr,flush=True)
        for seed in plan['seeds']:
            pl.seed_everything(seed,workers=True);folder=out/name/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=False)
            sets={}
            for split in xs:
                points=[data.MoleculeDatapoint.from_smi(r['smiles'],y,x_d=x.copy()) for r,y,x in zip(records[split],targets[split],xs[split])]
                sets[split]=data.MoleculeDataset(points,feat)
            scaler=sets['train'].normalize_targets();sets['valid'].normalize_targets(scaler)
            mp=nn.BondMessagePassing(d_v=len(atom),d_h=300)
            ffn=nn.RegressionFFN(input_dim=500,hidden_dim=300,output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
            model=models.MPNN(mp,nn.MeanAggregation(),ffn,batch_norm=False,metrics=[nn.metrics.RMSE(),nn.metrics.MAE()])
            config={'dataset':name,'seed':seed,'atom_dim':len(atom),'bond_dim':feat.bond_fdim,'descriptor_dim':200,
                'active_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'target_center':scaler.mean_.tolist(),'target_scale':scaler.scale_.tolist(),'plan_sha256':sha(out/'plan.json'),'pretraining':False}
            write(folder/'config.json',config);write(out/'status.json',{'state':'running','dataset':name,'seed':seed,'completed':len(rows)})
            train=data.build_dataloader(sets['train'],batch_size=64,num_workers=0,seed=seed,shuffle=True,drop_last=False)
            valid=data.build_dataloader(sets['valid'],batch_size=64,num_workers=0,shuffle=False,drop_last=False)
            cb=ModelCheckpoint(dirpath=folder/'checkpoints',filename='best',monitor='val_loss',mode='min',save_top_k=1,save_last=True,enable_version_counter=False)
            hist=History(folder);wall=time.perf_counter();trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=50,logger=False,enable_progress_bar=False,
                enable_model_summary=False,deterministic=True,num_sanity_val_steps=0,callbacks=[hist,cb],default_root_dir=folder)
            trainer.fit(model,train,valid);replay=models.MPNN.load_from_checkpoint(cb.best_model_path,map_location='cpu');replay.cuda().eval();metrics={}
            for split in xs:
                loader=data.build_dataloader(sets[split],batch_size=64,num_workers=0,shuffle=False,drop_last=False)
                pp=torch.cat(trainer.predict(replay,loader)).cpu().numpy();again=torch.cat(trainer.predict(replay,loader)).cpu().numpy();np.testing.assert_allclose(pp,again,rtol=1e-6,atol=1e-6)
                yy=targets[split];metrics[split]={'rmse':float(np.sqrt(np.mean((pp-yy)**2))),'mae':float(np.mean(abs(pp-yy)))}
                with (folder/f'{split}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.writer(f);w.writerow(['sample_id','prediction','target']);w.writerows((r['sample_id'],float(p[0]),float(y[0])) for r,p,y in zip(records[split],pp,yy))
            assert abs(metrics['valid']['rmse']**2/scaler.scale_[0]**2-float(cb.best_model_score))<1e-4
            row={'dataset':name,'seed':seed,**metrics,'best_epoch':int(torch.load(cb.best_model_path,map_location='cpu',weights_only=False)['epoch'])+1,
                'epochs_run':len(hist.rows),'active_parameters':config['active_parameters'],'elapsed_seconds':time.perf_counter()-wall,
                'best_checkpoint_sha256':sha(Path(cb.best_model_path)),'replay_and_inverse_scale_passed':True,'test_evaluated':False}
            rows.append(row);write(folder/'completion.json',row);write(out/'results.json',{'runs':rows,'test_evaluated':False});print('completed',row,flush=True)
            del trainer,model,replay,mp,ffn;torch.cuda.empty_cache()
    aggregate={}
    for name in plan['datasets']:
        v=[r['valid']['rmse'] for r in rows if r['dataset']==name];aggregate[name]={'n':len(v),'mean_rmse':statistics.mean(v),'sample_sd':statistics.stdev(v)}
    write(out/'results.json',{'runs':rows,'aggregate':aggregate,'test_evaluated':False})
    write(out/'status.json',{'state':'complete','runs':len(rows),'all_replays_passed':True,'test_evaluated':False,'wall_seconds':time.perf_counter()-started})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else prepare)(a.output_dir.resolve())
