"""Verify local/archive blobs against an independently fetched official Git tree."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
official=json.loads((OUT/'official_tree.json').read_text(encoding='utf-8-sig'))
local=ROOT/'research_code_audit/HMG-MRL'
archive=ROOT/'motif/runs/scale_pairs_3seeds_20261018/upstream/hmg'
worker=ROOT/'motif/runs/hmg_fusion_7arms_3seeds_20261025/source/motif'
protocol=json.loads((ROOT/'motif/runs/hmg_fusion_7arms_3seeds_20261025/results/protocol.json').read_text())

def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

records=[]
for entry in official['tree']['tree']:
    if entry['type']!='blob':continue
    path=entry['path']; source=local/path
    raw=source.read_bytes(); canonical=raw.replace(b'\r\n',b'\n')
    verified=blob(raw)==entry['sha'] or blob(canonical)==entry['sha']
    if not verified:raise ValueError('Official local checkout mismatch: '+path)
    record=dict(path=path,official_git_blob=entry['sha'],local_matches_official_blob=True,archive_present=(archive/path).exists())
    if (archive/path).exists():
        if source.suffix=='.pkl':
            record['archive_bytes_equal']=(archive/path).read_bytes()==raw
            records.append(record)
            continue
        old=(archive/path).read_text(encoding='utf-8-sig');new=raw.decode('utf-8-sig')
        record['archive_text_equal']=old.replace('\r\n','\n')==new.replace('\r\n','\n')
        if path.endswith('.py'):
            record['archive_ast_equal']=ast.dump(ast.parse(old))==ast.dump(ast.parse(new))
        if not record['archive_text_equal']:
            patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='running_archive/'+path,tofile='official/'+path))
            target=OUT/'diffs'/(path.replace('/','__')+'.diff');target.parent.mkdir(exist_ok=True)
            target.write_text(patch,encoding='utf-8')
    records.append(record)
report=dict(official_repository=official['repository'],official_commit=official['commit'],files=records,
    local_checkout_verified_against_remote_git_blobs=all(r['local_matches_official_blob'] for r in records),
    runtime_source_files=[dict(path=p,sha256_matches_protocol=hashlib.sha256((archive/p).read_bytes()).hexdigest()==digest)
                          for p,digest in protocol['paper_source_sha256'].items()],
    worker_implementation=[dict(path=p,sha256_matches_protocol=hashlib.sha256((worker/p).read_bytes()).hexdigest()==digest)
                           for p,digest in protocol['implementation_sha256'].items()])
(OUT/'source_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(official_commit=official['commit'],verified_files=len(records),
    changed_archive=[r['path'] for r in records if r.get('archive_present') and not r.get('archive_text_equal',r.get('archive_bytes_equal',False))],
    ast_changed_archive=[r['path'] for r in records if r.get('archive_ast_equal') is False],
    missing_from_runtime=[r['path'] for r in records if not r['archive_present']]),ensure_ascii=False))
