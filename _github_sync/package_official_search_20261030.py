import argparse,json,hashlib,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];central=ROOT/'motif/official_method_search'
runs=[ROOT/'motif/official_chemeleon/runs/smoke_20261030',ROOT/'motif/official_chemeleon/runs/comparison_20261030',ROOT/'motif/official_chemprop_rich/runs/comparison_20261030']
scripts=['run_official_chemeleon_20261030.py','run_frozen_chemeleon_ridge_20261030.py','recheck_chemeleon_ridge_float64_20261030.py','run_official_chemprop_rich_20261030.py',
         'export_normalized_descriptors_20261030.py','run_descriptor_xgboost_20261030.py','verify_descriptor_svr_20261030.py','check_official_search_data_20261030.py',
         'finalize_official_search_20261030.py','package_official_search_20261030.py']
files=[ROOT/'_github_sync'/n for n in scripts]
for module in [central,ROOT/'motif/official_chemeleon',ROOT/'motif/official_chemprop_rich']:
    files += [p for p in module.iterdir() if p.is_file() and p.suffix in ['.md','.py','.json'] and not p.name.startswith('SYNC_MANIFEST')]
weights=[];features=[]
for run in runs:
    for q in run.rglob('*'):
        if not q.is_file() or 'frozen_source' in q.parts:continue
        if q.suffix in ['.ckpt','.pt','.joblib'] or q.name.startswith('model_seed_') and q.suffix=='.json':weights.append(q)
        elif q.suffix=='.npz':features.append(q)
        elif q.suffix in ['.json','.csv','.md','.py']:files.append(q)
files=sorted(set(files));binaries=[central/'frozen_sources_20261030.tar.xz',central/'esol_svr_20261030.tar.xz']
p=argparse.ArgumentParser();p.add_argument('--manifest',action='store_true');p.add_argument('--index',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--chars',type=int,default=14000);p.add_argument('--binary-index',type=int);a=p.parse_args()
if a.manifest:
    def binary(q):return {'path':q.relative_to(ROOT).as_posix(),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
    m={'files':[{'path':q.relative_to(ROOT).as_posix(),'chars':len(q.read_text(encoding='utf-8-sig')),'sha256':hashlib.sha256(q.read_text(encoding='utf-8-sig').encode()).hexdigest(),
        'git_blob_sha1':hashlib.sha1(b'blob '+str(len(q.read_text(encoding='utf-8-sig').encode())).encode()+b'\0'+q.read_text(encoding='utf-8-sig').encode()).hexdigest()} for q in files],
       'binary_files':[{**binary(q),'base64_chars':4*((q.stat().st_size+2)//3),'git_blob_sha1':hashlib.sha1(b'blob '+str(q.stat().st_size).encode()+b'\0'+q.read_bytes()).hexdigest()} for q in binaries],
       'model_files':[binary(q) for q in weights],'local_only_features':[binary(q) for q in features],
       'published_model':'ESOL descriptor-SVR included in esol_svr_20261030.tar.xz; other models local',
       'external_foundation':{'path':'model/pretrained/chemeleon_mp.pt','sha256':hashlib.sha256((ROOT/'model/pretrained/chemeleon_mp.pt').read_bytes()).hexdigest(),'source':'https://zenodo.org/records/15460715'}}
    print(json.dumps(m))
elif a.binary_index is not None:print(json.dumps(base64.b64encode(binaries[a.binary_index].read_bytes()).decode()[a.offset:a.offset+a.chars]))
else:
    q=files[a.index];print(json.dumps({'path':q.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob','content':q.read_text(encoding='utf-8-sig')[a.offset:a.offset+a.chars]}))
