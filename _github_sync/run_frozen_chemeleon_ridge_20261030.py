"""Standard frozen public representation + train-only CV ridge adaptation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch,chemprop
from chemprop import data,featurizers,nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,GroupKFold
from rdkit import Chem
from threadpoolctl import threadpool_limits
import joblib

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def main(out):
    plan=read(out/'plan.json');dest=out/'frozen_ridge';dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.25)
    weight=Path(plan['weight_path']);assert hashlib.md5(weight.read_bytes()).hexdigest()==plan['expected_md5']
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    mp=nn.BondMessagePassing(**checkpoint['hyper_parameters']);mp.load_state_dict(checkpoint['state_dict'],strict=True);mp.cuda().eval()
    agg=nn.MeanAggregation().cuda();rows=[];started=time.perf_counter()
    for name in plan['datasets']:
        folder=dest/name;folder.mkdir();features={};targets={};records={}
        for split in ['train','valid']:
            path=out/'data'/f'{name}_{split}.csv';assert sha(path)==plan['data_sha256'][str(path.relative_to(out))]
            with path.open(encoding='utf-8') as f:records[split]=list(csv.DictReader(f))
            points=[data.MoleculeDatapoint.from_smi(r['smiles']) for r in records[split]]
            dataset=data.MoleculeDataset(points,featurizers.SimpleMoleculeMolGraphFeaturizer())
            loader=data.build_dataloader(dataset,batch_size=64,num_workers=0,shuffle=False,drop_last=False)
            values=[]
            with torch.inference_mode():
                for bmg,*_ in loader:
                    bmg.to('cuda');values.append(agg(mp(bmg),bmg.batch).cpu().numpy())
            features[split]=np.vstack(values);assert features[split].shape==(len(points),2048)
            targets[split]=np.array([float(r['target']) for r in records[split]])
        np.savez_compressed(folder/'features.npz',**features)
        pipeline=Pipeline([('scale',StandardScaler()),('ridge',Ridge(solver='cholesky'))])
        search=GridSearchCV(pipeline,{'ridge__alpha':[.001,.01,.1,1.,10.,100.,1000.,10000.]},
            cv=GroupKFold(n_splits=5),scoring='neg_mean_squared_error',refit=True,n_jobs=1)
        groups=[Chem.MolToSmiles(Chem.MolFromSmiles(r['smiles']),canonical=True) for r in records['train']]
        with threadpool_limits(limits=2):search.fit(features['train'],targets['train'],groups=groups)
        metrics={}
        for split in ['train','valid']:
            pred=search.predict(features[split]);truth=targets[split]
            metrics[split]={'rmse':float(np.sqrt(np.mean((pred-truth)**2))),'mae':float(np.mean(np.abs(pred-truth)))}
            with (folder/f'{split}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(['sample_id','prediction','target']);writer.writerows((r['sample_id'],float(p),float(y)) for r,p,y in zip(records[split],pred,truth))
        joblib.dump(search.best_estimator_,folder/'ridge.joblib');replay=joblib.load(folder/'ridge.joblib')
        np.testing.assert_allclose(replay.predict(features['valid']),search.predict(features['valid']),atol=1e-10,rtol=1e-10)
        row={'dataset':name,**metrics,'selected_alpha':search.best_params_['ridge__alpha'],
            'training_cv_mean_mse':(-search.cv_results_['mean_test_score']).tolist(),'alphas':[.001,.01,.1,1.,10.,100.,1000.,10000.],
            'selection':'5-fold training-only canonical-molecule GroupKFold CV, StandardScaler inside each fold; no task validation used',
            'model_sha256':sha(folder/'ridge.joblib'),'features_sha256':sha(folder/'features.npz'),'replay_passed':True,'test_evaluated':False}
        write(folder/'completion.json',row);rows.append(row);print('frozen_ridge',row,flush=True)
    write(dest/'results.json',{'runs':rows,'method':'frozen official CheMeleon encoder and standard ridge adaptation; not a new combined neural architecture',
        'independent_models_per_task':1,'deterministic_not_three_seed_replicates':True,'source_script_sha256':sha(Path(__file__)),
        'weight_sha256':sha(weight),'chemprop':chemprop.__version__,'wall_seconds':time.perf_counter()-started,'test_evaluated':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();main(a.output_dir.resolve())
