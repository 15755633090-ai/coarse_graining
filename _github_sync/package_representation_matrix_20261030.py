import argparse,base64,hashlib,io,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];module=ROOT/'motif/adjacency_diffusion'
p=argparse.ArgumentParser();p.add_argument('--manifest',action='store_true');p.add_argument('--manifest-file',type=Path)
p.add_argument('--archive',action='store_true');p.add_argument('--paths-json');p.add_argument('--binary-path')
p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int,default=1);p.add_argument('--offset',type=int,default=0);p.add_argument('--chars',type=int,default=18000);a=p.parse_args()
names=['representation_matrix_smoke_20261030','representation_matrix_20261030'];archive=module/'runs/representation_sources_20261030.tar.xz'
if a.binary_path:
    value=base64.b64encode((ROOT/a.binary_path).read_bytes()).decode();print(json.dumps(value[a.offset:a.offset+a.chars]));sys.exit()
if a.archive:
    inventories=[]
    with tarfile.open(archive,'w:xz',preset=6) as z:
        for name in names:
            folder=module/'runs'/name;plan=json.loads((folder/'plan.json').read_text(encoding='utf-8'));entries=[]
            for rel,h in sorted(plan['source_sha256'].items()):
                raw=(folder/'frozen_source'/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,rel
                info=tarfile.TarInfo(name+'/frozen_source/'+rel);info.size=len(raw);info.mtime=0;info.mode=0o644
                z.addfile(info,io.BytesIO(raw));entries.append({'path':rel,'sha256':h})
            inventories.append((folder,entries))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    for folder,entries in inventories:
        (folder/'source_archive.json').write_text(json.dumps({'archive':'../representation_sources_20261030.tar.xz','sha256':digest,
            'extract_into':'parent runs directory','entries':entries},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'bytes':archive.stat().st_size,'runs':len(names)}));sys.exit()
files=[module/n for n in ['representation_pretrain.py','test_representation_pretrain.py','typed_train.py','REPRESENTATION_MATRIX_20261030.md','REPRESENTATION_MATRIX_TESTS_20261030.json','README.md']]
files += [ROOT/'_github_sync'/n for n in ['run_representation_matrix_20261030.py','finalize_representation_matrix_20261030.py','summarize_representation_matrix_20261030.py','package_representation_matrix_20261030.py']]
weights=[]
for name in names:
    folder=module/'runs'/name
    files += [f for f in folder.rglob('*.json') if 'frozen_source' not in f.parts]+list(folder.glob('REPORT.md'))
    weights += list(folder.rglob('*.pt'))
files=sorted(set(f for f in files if f.exists()))
if a.manifest:
    m={'files':[{'path':f.relative_to(ROOT).as_posix(),'chars':len(f.read_text(encoding='utf-8-sig')),
        'sha256':hashlib.sha256(f.read_text(encoding='utf-8-sig').encode()).hexdigest()} for f in files],
        'local_only_weights':[{'path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in weights],
        'binary_files':[{'path':archive.relative_to(ROOT).as_posix(),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'base64_chars':4*((archive.stat().st_size+2)//3)}]}
    if a.manifest_file:a.manifest_file.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8');print(json.dumps({'files':len(files),'weights':len(weights)}))
    else:print(json.dumps(m))
else:
    if a.paths_json:files=[ROOT/rel for rel in json.loads(a.paths_json)]
    print(json.dumps([{'path':f.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob',
        'content':f.read_text(encoding='utf-8-sig')[a.offset:a.offset+a.chars]} for f in files[a.start:a.start+a.limit]],ensure_ascii=True))
