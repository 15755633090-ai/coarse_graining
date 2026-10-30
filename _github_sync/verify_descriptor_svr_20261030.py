import csv,json,hashlib,sys
from pathlib import Path
import numpy as np,joblib,chemprop,sklearn,rdkit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
out=ROOT/'motif/official_chemprop_rich/runs/comparison_20261030'
expected=json.loads((out/'svr_results.json').read_text(encoding='utf-8'))['runs']
for name in ['esol','freesolv']:
    folder=out/name/'svr';model=joblib.load(folder/'model.joblib');cache=np.load(out/f'{name}_normalized_descriptors.npz')
    metrics={}
    for split in ['train','valid']:
        with (out/'data'/f'{name}_{split}.csv').open(encoding='utf-8') as f:records=list(csv.DictReader(f))
        pred=model.predict(cache[split]);truth=cache[split+'_y']
        metrics[split]={'rmse':float(np.sqrt(np.mean((pred-truth)**2))),'mae':float(np.mean(abs(pred-truth)))}
        row=next(r for r in expected if r['dataset']==name)
        assert abs(row[split]['rmse']-metrics[split]['rmse'])<1e-8
        with (folder/f'{split}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['sample_id','prediction','target']);w.writerows((r['sample_id'],float(p),float(y)) for r,p,y in zip(records,pred,truth))
    receipt={'model_sha256':hashlib.sha256((folder/'model.joblib').read_bytes()).hexdigest(),'metrics':metrics,'replay_passed':True,
        'chemprop':chemprop.__version__,'sklearn':sklearn.__version__,'rdkit':rdkit.__version__,'test_evaluated':False}
    (folder/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
# Verify the standalone inference path against all held-out ESOL predictions.
from motif.official_method_search.predict_svr import predict
with (out/'data/esol_valid.csv').open(encoding='utf-8') as f:records=list(csv.DictReader(f))
actual=predict([r['smiles'] for r in records]);saved=np.load(out/'esol_normalized_descriptors.npz')
expected_prediction=joblib.load(out/'esol/svr/model.joblib').predict(saved['valid'])
np.testing.assert_allclose(actual,expected_prediction,atol=1e-8,rtol=1e-8)
print(json.dumps({'svr_replays':2,'standalone_inference_rows':len(actual),'passed':True}))
