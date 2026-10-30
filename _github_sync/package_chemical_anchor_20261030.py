import argparse,json,hashlib,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];module=ROOT/'motif/adjacency_diffusion'
runs=[module/'runs'/n for n in ['chemical_anchor_20261030','in_domain_chemical_anchor_20261030']]
p=argparse.ArgumentParser();p.add_argument('--manifest',action='store_true');p.add_argument('--index',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--chars',type=int,default=12000);p.add_argument('--binary',action='store_true');p.add_argument('--binary-index',type=int,default=0);a=p.parse_args()
files=[module/n for n in ['chemical_targets.py','test_chemical_targets.py','CHEMICAL_ANCHOR_20261030.md','CHEMICAL_ANCHOR_TESTS_20261030.json']]
files += [ROOT/'_github_sync'/n for n in ['run_chemical_anchor_20261030.py','run_in_domain_chemical_anchor_20261030.py','finalize_chemical_anchor_20261030.py','package_chemical_anchor_20261030.py','check_descriptor_baseline_20261030.py']]
for run in runs:files += [q for q in run.rglob('*.json') if 'frozen_source' not in q.parts]+[run/'REPORT.md']
files=sorted(set(files));archives=[run/'frozen_source.tar.xz' for run in runs];archive=archives[a.binary_index]
if a.manifest:
    result={'files':[{'path':q.relative_to(ROOT).as_posix(),'chars':len(q.read_text(encoding='utf-8-sig')),'sha256':hashlib.sha256(q.read_text(encoding='utf-8-sig').encode()).hexdigest()} for q in files],
        'binary_files':[{'path':v.relative_to(ROOT).as_posix(),'bytes':v.stat().st_size,'sha256':hashlib.sha256(v.read_bytes()).hexdigest(),'base64_chars':len(base64.b64encode(v.read_bytes()))} for v in archives],
        'local_only_weights':[{'path':q.relative_to(ROOT).as_posix(),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for run in runs for q in sorted(run.rglob('*.pt'))]}
    print(json.dumps(result))
elif a.binary:print(json.dumps(base64.b64encode(archive.read_bytes()).decode()[a.offset:a.offset+a.chars]))
else:
    q=files[a.index];print(json.dumps({'path':q.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob','content':q.read_text(encoding='utf-8-sig')[a.offset:a.offset+a.chars]}))
