"""Publish source, frozen source and validation evidence; keep tensor checkpoints local."""
import argparse,base64,hashlib,io,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--manifest',action='store_true');p.add_argument('--start',type=int,default=0)
p.add_argument('--limit',type=int,default=1);p.add_argument('--offset',type=int,default=0);p.add_argument('--chars',type=int)
p.add_argument('--manifest-file',type=Path);p.add_argument('--path',type=str);p.add_argument('--paths-json')
p.add_argument('--archive-sources',action='store_true');p.add_argument('--binary-path');a=p.parse_args()
module=ROOT/'motif/adjacency_diffusion'
if a.binary_path:
    encoded=base64.b64encode((ROOT/a.binary_path).read_bytes()).decode('ascii')
    print(json.dumps(encoded[a.offset:a.offset+a.chars if a.chars is not None else None]));sys.exit()
if a.archive_sources:
    archived=[];archive=module/'runs/frozen_sources_20261030.tar.xz';inventories=[]
    with tarfile.open(archive,'w:xz',preset=6) as z:
        for folder in sorted((module/'runs').glob('*20261030')):
            if not (folder/'frozen_source').is_dir() or not (folder/'plan.json').exists():continue
            plan=json.loads((folder/'plan.json').read_text(encoding='utf-8'));entries=[]
            for name,h in sorted(plan['source_sha256'].items()):
                raw=(folder/'frozen_source'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,name
                entry=tarfile.TarInfo(folder.name+'/frozen_source/'+name);entry.size=len(raw);entry.mtime=0;entry.mode=0o644
                z.addfile(entry,io.BytesIO(raw))
                entries.append({'path':name,'sha256':h})
            inventories.append((folder,entries))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    for folder,entries in inventories:
        inventory={'archive':'../frozen_sources_20261030.tar.xz','sha256':digest,'bytes':archive.stat().st_size,
            'extract_into':'parent runs directory; restores every run/frozen_source path with exact CRLF/BOM bytes','entries':entries}
        (folder/'frozen_source_archive.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':archive.relative_to(ROOT).as_posix(),'bytes':archive.stat().st_size,'runs':len(inventories)}));sys.exit()
def content(path):
    # Frozen experiment hashes refer to exact source bytes, including CRLF/BOM.
    return path.read_bytes().decode('utf-8') if 'frozen_source' in path.parts else path.read_text(encoding='utf-8-sig')
files=[module/n for n in ('model.py','masked_bonds.py','typed_train.py','test_message_normalization.py','test_posterior_masked.py',
                          'task_readout.py','test_task_readout.py','TASK_READOUT_20261030.md',
                          'REVISION_RESULTS_20261030.md','REVISION_RESULTS_20261030.json',
                          'PARTIAL_TRANSFER_20261030.md','test_partial_transfer.py',
                          'METHOD_AUDIT_20261030.md',
                          'TRANSFER_REVISION_20261030.md','POSTERIOR_WEIGHTING_20261030.md','REVISION_TESTS_20261030.json','README.md')]
files += [ROOT/'_github_sync'/n for n in ('search_transfer_revision_20261030.py','check_posterior_weighting_20261030.py',
          'run_posterior_after_revision_20261030.py','finalize_transfer_revision_20261030.py','finalize_posterior_weighting_20261030.py',
          'run_bounded_after_posterior_20261030.py',
          'check_task_readout_20261030.py','finalize_task_readout_20261030.py',
          'summarize_transfer_revision_20261030.py','verify_transfer_revision_20261030.py',
          'check_partial_transfer_20261030.py','finalize_partial_transfer_20261030.py',
          'package_transfer_revision_20261030.py')]
weights=[]
binary_files=[module/'runs/frozen_sources_20261030.tar.xz'] if (module/'runs/frozen_sources_20261030.tar.xz').exists() else []
for name in ('transfer_revision_smoke_v2_20261030','transfer_revision_cli_smoke_20261030','transfer_revision_20261030','posterior_weighting_20261030','posterior_bounded_20261030','task_readout_smoke_20261030','task_readout_20261030','partial_transfer_20261030'):
    folder=module/'runs'/name
    files += [f for f in folder.rglob('*.json') if 'frozen_source' not in f.parts]+list(folder.glob('REPORT.md'))
    weights += list(folder.rglob('*.pt'))
files=sorted(set(f for f in files if f.exists()))
if a.manifest:
    manifest={'files':[{'path':f.relative_to(ROOT).as_posix(),'chars':len(content(f)),
        'sha256':hashlib.sha256(content(f).encode()).hexdigest()} for f in files],
        'local_only_checkpoints':[{'path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in weights],
        'binary_files':[{'path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size,'base64_chars':4*((f.stat().st_size+2)//3),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in binary_files]}
    if a.manifest_file:
        a.manifest_file.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'files':len(files),'checkpoints':len(weights),'manifest':str(a.manifest_file)}))
    else:print(json.dumps(manifest))
else:
    if a.path:files=[ROOT/a.path]
    if a.paths_json:files=[ROOT/name for name in json.loads(a.paths_json)]
    print(json.dumps([{'path':f.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob',
        'content':content(f)[a.offset:a.offset+a.chars if a.chars is not None else None]}
        for f in files[a.start:a.start+a.limit]],ensure_ascii=True))
