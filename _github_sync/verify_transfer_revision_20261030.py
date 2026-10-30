"""Verify every published blob using one git batch process."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
commit=sys.argv[1];manifest=json.loads((ROOT/'_github_sync/transfer_revision_manifest.json').read_text(encoding='utf-8'))
manifest_path='motif/adjacency_diffusion/SYNC_MANIFEST_TRANSFER_REVISION_20261030.json'
files=manifest['files']+manifest.get('binary_files',[])+[{'path':manifest_path,'manifest':True}]
queries=''.join(commit+':'+f['path']+'\n' for f in files).encode('utf-8')
raw=subprocess.check_output(['git','-C',str(ROOT/'coarse_graining1'),'cat-file','--batch'],input=queries)
cursor=0
for f in files:
    end=raw.index(b'\n',cursor);header=raw[cursor:end].split();assert len(header)==3 and header[1]==b'blob',f['path']
    size=int(header[2]);data=raw[end+1:end+1+size];cursor=end+1+size+1
    if f.get('manifest'):assert json.loads(data)==manifest
    else:assert hashlib.sha256(data).hexdigest()==f['sha256'],f['path']
assert cursor==len(raw)
receipt={'commit':commit,'verified_files':len(files),'all_remote_hashes_match':True,'frozen_source_bytes_preserved':True,'test_evaluated':False}
(ROOT/'_github_sync/transfer_revision_20261030_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
