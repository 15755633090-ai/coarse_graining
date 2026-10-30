import csv,json,sys,hashlib,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motif.adjacency_diffusion import typed_train as runner
out=ROOT/'motif/official_chemprop_rich/runs/comparison_20261030';result={}
for name in ['esol','freesolv']:
    source,splits,_=runner.io.build_property_data(ROOT/'data',name);result[name]={}
    for split in ['train','valid']:
        with (out/'data'/f'{name}_{split}.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        assert [int(r['sample_id']) for r in rows]==splits[split]
        assert all(r['smiles']==source.smiles[int(r['sample_id'])] and abs(float(r['target'])-float(source.targets[int(r['sample_id']),0]))<1e-5 for r in rows)
        result[name][split]={'rows':len(rows),'identical_smiles_labels_and_indices':True}
dest=ROOT/'motif/official_method_search/data_verification.json'
dest.write_text(json.dumps({'results':result,'test_evaluated':False,'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
