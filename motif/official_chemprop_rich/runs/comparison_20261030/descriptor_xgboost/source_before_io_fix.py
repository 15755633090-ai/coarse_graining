"""A mature tree model, existing candidate space, train-only CV, no neural fusion."""
import argparse,hashlib,json,csv,time,statistics
from pathlib import Path
import numpy as np,xgboost,sklearn,joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();out=a.output_dir.resolve();dest=out/'descriptor_xgboost';dest.mkdir(exist_ok=False)
candidates=read(ROOT/'model/search_spaces.json')['morgan_xgboost']
write(dest/'plan.json',{'method':'XGBoost on official normalized200 descriptors, no graph model or feature concatenation',
    'candidates_from':'model/search_spaces.json/morgan_xgboost','candidate_space_sha256':sha(ROOT/'model/search_spaces.json'),'candidates':candidates,
    'trees':500,'reg_lambda':1.,'cv':'canonical-molecule GroupKFold5 within training only; no task validation or early stopping',
    'selection_seed':42,'refit_seeds':[0,1,2],'xgboost':xgboost.__version__,'sklearn':sklearn.__version__,
    'source_script_sha256':sha(Path(__file__)),'test_evaluated':False})
rows=[];started=time.perf_counter()
for name in ['esol','freesolv']:
    path=out/f'{name}_normalized_descriptors.npz';assert sha(path)==read(path.with_suffix('.json'))['sha256'];cache=np.load(path)
    x=cache['train'];y=cache['train_y'];groups=cache['train_groups'];folds=list(GroupKFold(5).split(x,y,groups));scores=[]
    for config in candidates:
        mse=[]
        for train,valid in folds:
            assert not set(groups[train]) & set(groups[valid])
            model=XGBRegressor(**config,n_estimators=500,reg_lambda=1.,random_state=42,n_jobs=2,tree_method='hist',objective='reg:squarederror')
            model.fit(x[train],y[train]);pred=model.predict(x[valid]);mse.append(float(np.mean((pred-y[valid])**2)))
        scores.append({'config':config,'fold_mse':mse,'mean_mse':statistics.mean(mse)})
    selected=min(scores,key=lambda r:r['mean_mse'])['config'];folder=dest/name;folder.mkdir()
    write(folder/'selection.json',{'scores':scores,'selected':selected,'features_sha256':sha(path),'selection':'training-only CV'})
    for seed in [0,1,2]:
        model=XGBRegressor(**selected,n_estimators=500,reg_lambda=1.,random_state=seed,n_jobs=2,tree_method='hist',objective='reg:squarederror')
        model.fit(x,y);metrics={}
        for split in ['train','valid']:
            pred=model.predict(cache[split]);truth=cache[split+'_y'];metrics[split]={'rmse':float(np.sqrt(np.mean((pred-truth)**2))),'mae':float(np.mean(abs(pred-truth)))}
            with (folder/f'{split}_seed_{seed}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['sample_id','prediction','target']);w.writerows((int(i),float(p),float(y)) for i,p,y in zip(cache[split+'_id'],pred,truth))
        saved=folder/f'model_seed_{seed}.json';model.save_model(saved);replay=XGBRegressor();replay.load_model(saved)
        np.testing.assert_allclose(replay.predict(cache['valid']),model.predict(cache['valid']),atol=0,rtol=0)
        row={'dataset':name,'seed':seed,**metrics,'model_sha256':sha(saved),'replay_passed':True,'test_evaluated':False}
        rows.append(row);print('xgboost',row,flush=True)
    write(dest/'results.json',{'runs':rows,'test_evaluated':False})
aggregate={}
for name in ['esol','freesolv']:
    v=[r['valid']['rmse'] for r in rows if r['dataset']==name];aggregate[name]={'n':len(v),'mean_rmse':statistics.mean(v),'sample_sd':statistics.stdev(v)}
write(dest/'results.json',{'runs':rows,'aggregate':aggregate,'wall_seconds':time.perf_counter()-started,'test_evaluated':False})
