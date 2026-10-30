import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];module=ROOT/'motif/adjacency_diffusion'
p=argparse.ArgumentParser();p.add_argument('--manifest',action='store_true');p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int,default=1);p.add_argument('--offset',type=int,default=0);p.add_argument('--chars',type=int,default=20000);a=p.parse_args()
files=[module/n for n in ['model.py','typed_bonds.py','masked_bonds.py','path_gates.py','test_path_gates.py','PATH_DEPENDENCIES_20261030.md','README.md']]
files += [ROOT/'_github_sync/check_path_dependencies_20261030.py',Path(__file__).resolve()]
out=module/'runs/path_dependencies_verified_20261030'
files+=list(out.glob('*.json'))+list(out.glob('REPORT.md'));files=sorted(set(files))
if a.manifest:
    print(json.dumps({'files':[{'path':f.relative_to(ROOT).as_posix(),'chars':len(f.read_text(encoding='utf-8-sig')),
        'sha256':hashlib.sha256(f.read_text(encoding='utf-8-sig').encode()).hexdigest()} for f in files],
        'local_only_predictions':{'path':(out/'predictions.pt').relative_to(ROOT).as_posix(),'sha256':hashlib.sha256((out/'predictions.pt').read_bytes()).hexdigest()}}))
else:
    print(json.dumps([{'path':f.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob',
        'content':f.read_text(encoding='utf-8-sig')[a.offset:a.offset+a.chars]} for f in files[a.start:a.start+a.limit]],ensure_ascii=True))
