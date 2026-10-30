"""Numerical audit of an ill-conditioned float32 ridge fit, same folds/grid."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold,GridSearchCV
from rdkit import Chem
from threadpoolctl import threadpool_limits
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();out=a.output_dir.resolve();dest=out/'frozen_ridge_float64';dest.mkdir(exist_ok=False);rows=[]
for name in ['esol','freesolv']:
    feature_path=out/'frozen_ridge'/name/'features.npz';cache=np.load(feature_path)
    x={k:cache[k].astype(np.float64) for k in ['train','valid']};records={};y={}
    for split in x:
        with (out/'data'/f'{name}_{split}.csv').open(encoding='utf-8') as f:records[split]=list(csv.DictReader(f))
        y[split]=np.array([float(r['target']) for r in records[split]])
    groups=[Chem.MolToSmiles(Chem.MolFromSmiles(r['smiles']),canonical=True) for r in records['train']]
    search=GridSearchCV(Pipeline([('scale',StandardScaler()),('ridge',Ridge(solver='cholesky'))]),
        {'ridge__alpha':[.001,.01,.1,1.,10.,100.,1000.,10000.]},cv=GroupKFold(5),scoring='neg_mean_squared_error',refit=True,n_jobs=1)
    with threadpool_limits(limits=2):search.fit(x['train'],y['train'],groups=groups)
    folder=dest/name;folder.mkdir();metrics={}
    for split in x:
        pred=search.predict(x[split]);error=pred-y[split]
        metrics[split]={'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(abs(error)))}
        with (folder/f'{split}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['sample_id','prediction','target']);w.writerows((r['sample_id'],float(pp),float(yy)) for r,pp,yy in zip(records[split],pred,y[split]))
    scaled=search.best_estimator_['scale'].transform(x['valid'])
    joblib.dump(search.best_estimator_,folder/'ridge.joblib')
    row={'dataset':name,**metrics,'selected_alpha':search.best_params_['ridge__alpha'],'max_abs_standardized_validation_feature':float(abs(scaled).max()),
        'training_cv_mean_mse':(-search.cv_results_['mean_test_score']).tolist(),'cached_features_sha256':hashlib.sha256(feature_path.read_bytes()).hexdigest()}
    rows.append(row);print(json.dumps(row),flush=True)
result={'runs':rows,'revision':'float64 for numerical stability after float32 warnings; identical features, folds, regularization grid; original results retained',
    'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_evaluated':False}
(dest/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
