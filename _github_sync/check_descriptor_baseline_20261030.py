"""Fixed alpha=1 descriptor ridge baseline; train split only, no search/test."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.chemical_targets import chemical_targets,DESCRIPTORS

source,splits,spec=runner.io.build_property_data(ROOT/'data','esol')
d,_=chemical_targets(source.smiles);x=d.numpy();y=source.targets[:,0].numpy()
assert source.target_mask[splits['train'],0].all() and source.target_mask[splits['valid'],0].all()
model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
model.fit(x[splits['train']],y[splits['train']])
result={'recipe':'12 RDKit descriptors -> train-only StandardScaler -> Ridge(alpha1), intercept; no search',
    'purpose':'test direct task relevance of the fixed chemical targets; not a GNN or pretraining gain',
    'n_train':len(splits['train']),'n_valid':len(splits['valid']),'test_evaluated':False,
    'coefficients_standardized_features':dict(zip([n for n,_ in DESCRIPTORS],map(float,model[-1].coef_))),
    'intercept':float(model[-1].intercept_),'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'descriptor_names':[n for n,_ in DESCRIPTORS],
    'limits':'single deterministic fit on reused ESOL validation; calculated descriptors supplied directly, richer inference features than graph-only model'}
for split in ('train','valid'):
    pred=model.predict(x[splits[split]]);truth=y[splits[split]]
    result[split]={'rmse':float(((pred-truth)**2).mean()**.5),'mae':float(abs(pred-truth).mean())}
path=ROOT/'motif/adjacency_diffusion/runs/chemical_anchor_20261030/descriptor_baseline.json'
if path.exists():raise FileExistsError(path)
runner.io.atomic_write(path,result);print(json.dumps(result,indent=2))
